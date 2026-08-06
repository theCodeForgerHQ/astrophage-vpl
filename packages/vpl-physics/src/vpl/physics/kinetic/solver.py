"""The assembled PIC-MCC loop — doc 03 §4.2, and the ``ForwardSolver`` of doc 08 §4.

doc 03 §4.2 states the algorithm in six lines and this module is those six lines::

    1. Deposit charge to grid          -> fields.deposit_cic
    2. Solve Poisson on grid           -> fields.solve_poisson_dirichlet
    3. Interpolate field to particles  -> fields.gather_cic      (the same weighting)
    4. Push particles                  -> push.push_leapfrog
    5. Monte-Carlo collisions          -> collisions.NullCollisionSampler
    6. Apply boundary conditions       -> boundary.apply_boundaries

Nothing in steps 1-6 is re-implemented here. That is not tidiness: each of those modules
carries its own verification — the Poisson stencil against a manufactured solution, the
gather against the transpose of the deposition, the collision rate against
``nu = n sigma v``, the inflow against its printed density — and a loop that quietly
reproduced any of them would be a second, unverified copy. What this module owns is the
*assembly*, and the failures that belong to an assembly are bookkeeping failures:
:class:`PicDiagnostics` is therefore a closed ledger, and a test asserts that it balances.

## What comes out, and why it is more than a state

doc 08 §4's contract returns a :class:`~vpl.core.state.PlasmaState`. doc 01 §1.1's
deliverable is the **ion energy flux at the surface** and doc 03 §5.2 names the **IEDF at
the wall**, and neither survives a round trip through a gridded state: ``Gamma_E`` is
accumulated from every macroparticle that crosses ``z = 0``, and re-deriving it from a
binned ``f_i(z, v_z)`` would discard the transverse energy and the arrival statistics. So
:meth:`Pic1D3VSolver.run` returns a :class:`PicRun` carrying all three and
:meth:`Pic1D3VSolver.solve` returns the state out of it. :meth:`Pic1D3VSolver.flux` hands
back the *recorded* flux for the state it produced and refuses any other state, rather
than reconstructing a plausible wrong number from a histogram.

## Why the loop is eager rather than one ``lax.scan``

Putting the step under ``jax.jit`` and ``jax.lax.scan`` would cost a control.
:func:`~vpl.physics.kinetic.fields.deposit_cic` and
:func:`~vpl.physics.kinetic.fields.gather_cic` assert that every particle is inside the
domain, and that assertion reads a concrete value — exactly what tracing forbids. The
guard exists because a particle silently clamped onto the wall node deposits its charge at
the one place this project measures its deliverable, so the trade would buy a faster wrong
answer. The loop runs eagerly and the guard stays.

The two steps that *can* be traced are, and they are wrapped rather than rewritten:
:data:`_jit_poisson` and :data:`_jit_push` are ``jax.jit`` applied to the very functions
``test_kinetic_fields.py`` and ``test_kinetic_push.py`` verify, so the fast path and the
verified path are the same code. Measured on the reference machine, that takes the Poisson
solve from 0.72 ms to 0.008 ms and the push from 0.26 ms to 0.06 ms per call — about a
quarter of the step.

The other half of the same problem is where the cost actually is, and it *is* handled:
**absorbed particles are parked, not deleted** (see :mod:`~vpl.physics.kinetic.boundary`).
Shrinking the arrays as particles are absorbed changes their shape almost every step, which
retraces every jitted kernel underneath — over twenty times the cost of the physics.
Fixed-capacity arrays with an alive mask cost one multiply per step and never retrace.

## Simplifications, and what each one costs

Stated here rather than discovered, per doc 00 C4. The boundary module states its own four.

1. **The domain is the sheath and the bulk boundary is a reservoir.** doc 03 §3.3's bulk
   condition — density ``n_s``, ions entering at ``u = c_s`` — is *applied* at ``z = L``
   rather than produced by modelling the presheath. So the ion flux reaching the wall is
   imposed, not predicted, and the Bohm-flux comparison against L0 is a conservation check
   on this loop rather than a physics prediction; the sheath thickness, the potential
   profile and the IEDF *are* predictions. Bound: the imposed flux is exact to the
   injection arithmetic, and the residual error is whatever ``h_l`` is wrong by — a
   ``+/- 10 %`` PUBLISHED uncertainty the registry already carries into the error budget.
2. **Both species carry the same macroparticle weight.** A heavier electron marker would
   pay for the ``sqrt(m_i / m_e)`` higher reservoir flux, but then the two deposited
   charge densities differ in *granularity* and quasineutral cancellation leaves a residue
   of the larger weight. Bound: the electron population follows from the reservoir flux
   rather than being chosen, so its ``N^(-1/2)`` noise is reported in
   :class:`PicDiagnostics` rather than assumed adequate.
3. **The electron energy equation is not solved; ``T_e(z)`` is measured.** The emitted
   field is the second moment of the electron markers per node — what a kinetic code has
   instead of a closure. Where a node holds no electrons, which inside a high-bias sheath
   is most of them (``n_e / n_0 ~ exp(-83)`` at RP-1), the reported value falls back to
   the bulk ``T_e``: zero would be a claim that the electrons there are cold, and there
   are none there to be cold.
4. **Time-resolved output is snapshots, not phase-averaged accumulation.** A ``TimeGrid``
   selects instants at which the fields are recorded. doc 02 §10.3's cyclo-stationary
   phase binning, which the RF regimes D and E need, is not implemented; a DC solve is
   unaffected and an RF one would report single-cycle snapshots.
5. **The run length is derived from an ion transit time, not from a convergence test.**
   :attr:`Pic1D3VSolver.transits` sets it and :attr:`equilibration_fraction` discards the
   start-up transient. The residual is transient leakage into the window average, bounded
   by re-running with more transits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.params import default_registry
from vpl.core.protocols import (
    Citation,
    CostEstimate,
    Device,
    IonEnergyFlux,
    SolverConfig,
    SolverMetadata,
)
from vpl.core.random import Stream, generator
from vpl.core.state import (
    Fidelity,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    TimeGrid,
    VelocityDistribution,
)
from vpl.core.units import Q_, magnitude_in
from vpl.physics.analytic.sheath import (
    DEFAULT_EDGE_TO_CENTRE_RATIO,
    GAMMA_I_COLD_ION,
    bohm_speed,
    child_langmuir_thickness,
)
from vpl.physics.kinetic.boundary import (
    DEFAULT_ION_REFLECTION,
    PARKED_POSITION_M,
    InjectionSchedule,
    InjectionSource,
    ParticlePopulation,
    SurfaceModel,
    WallModel,
    apply_boundaries,
    sample_inflow_velocities,
    sample_secondary_velocities,
)
from vpl.physics.kinetic.collisions import NullCollisionSampler, collision_generator
from vpl.physics.kinetic.constraints import (
    TIMESTEP_SAFETY,
    check_stability,
    debye_length_m,
    electron_plasma_frequency_rad_per_s,
)
from vpl.physics.kinetic.fields import deposit_cic, gather_cic, solve_poisson_dirichlet
from vpl.physics.kinetic.grid import UniformGrid
from vpl.physics.kinetic.precision import enable_double_precision, is_double_precision
from vpl.physics.kinetic.push import Species as PushSpecies
from vpl.physics.kinetic.push import push_leapfrog

enable_double_precision()

#: ``jax.jit`` applied to the verified kernels themselves, with the grid and the species
#: static because they fix the array shapes and nothing else.
#:
#: Wrapping rather than rewriting is the whole point: the compiled path *is*
#: :func:`~vpl.physics.kinetic.fields.solve_poisson_dirichlet` and
#: :func:`~vpl.physics.kinetic.push.push_leapfrog`, so it cannot drift from the code their
#: verification tests exercise. The boundary voltages and the timestep stay **traced** and
#: not static: making them static would recompile once per distinct bias, and doc 11 WBS
#: 3.1's ensemble sweeps thousands of them.
_jit_poisson = jax.jit(solve_poisson_dirichlet, static_argnums=(0,))
_jit_push = jax.jit(push_leapfrog, static_argnums=(0,))


@jax.jit
def _electric_field(potential: Array, dz_m: Array) -> Array:
    """``E = -dPhi/dz`` — second order inside, one-sided at the two boundary nodes.

    ``asarray`` because ``jnp.gradient``'s return type widens to a list when it is given
    several axes, and this call gives it one.
    """
    return -jnp.asarray(jnp.gradient(potential, dz_m))


__all__ = [
    "AXIS_MARGIN",
    "CFL_SPEED_SIGMAS",
    "DEFAULT_CAPACITY_FACTOR",
    "DEFAULT_CELLS_PER_DEBYE",
    "DEFAULT_DOMAIN_SHEATHS",
    "DEFAULT_ENERGY_BINS",
    "DEFAULT_EQUILIBRATION_FRACTION",
    "DEFAULT_N_PPC",
    "DEFAULT_SEED",
    "DEFAULT_TRANSITS",
    "DEFAULT_VELOCITY_BINS",
    "SHEATH_EDGE_TOLERANCE",
    "IonEnergyDistribution",
    "Pic1D3VSolver",
    "PicDiagnostics",
    "PicRun",
]

type FloatArray = NDArray[np.float64]

_E: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))
_EPS0: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

_REGISTRY = default_registry()

#: doc 03 §4.3's default macroparticles per cell per species.
DEFAULT_N_PPC: Final[float] = float(_REGISTRY.value_in("pic.n_ppc", "dimensionless"))

#: Cells per Debye length — doc 05 §7.1's truth mesh, ``dz = lambda_D / 2``.
DEFAULT_CELLS_PER_DEBYE: Final[float] = float(
    _REGISTRY.value_in("mesh.A.cells_per_debye", "dimensionless")
)

#: doc 03 §2.1's sheath-edge definition, ``(n_i - n_e) / n_i = delta``. A registered DESIGN
#: parameter precisely because the sheath edge is a definition and not a surface.
SHEATH_EDGE_TOLERANCE: Final[float] = float(
    _REGISTRY.value_in("sheath.edge_tolerance", "dimensionless")
)

#: Domain length in Child-Langmuir sheath thicknesses. Beyond 1 so the quasineutral
#: presheath and doc 03 §3.3's ``Phi = 0`` boundary sit outside the space-charge region;
#: not far beyond, because every extra sheath of quasineutral plasma adds an ion transit at
#: ``c_s`` — the slowest timescale in the problem — to the run for no new physics.
DEFAULT_DOMAIN_SHEATHS: Final[float] = 1.5

#: Ion transit times simulated: one to clear the loaded sheath, the rest to average over.
DEFAULT_TRANSITS: Final[float] = 3.0

#: Fraction of the run discarded as start-up transient before anything is recorded.
DEFAULT_EQUILIBRATION_FRACTION: Final[float] = 0.4

#: Slots per species as a multiple of the loaded population. The population is bounded by
#: the reservoir flux and the transit time rather than chosen, so the headroom is real;
#: running out is an error rather than a silent truncation of the injected flux.
DEFAULT_CAPACITY_FACTOR: Final[float] = 4.0

#: Electron thermal speeds at which doc 03 §4.3's CFL bound is evaluated. Five sigma
#: excludes 3e-7 of a Maxwellian and — not by coincidence — makes ``dz / v_max`` equal
#: ``0.2 / omega_pe`` exactly at ``dz = lambda_D``, so the two doc 03 §4.3 timestep
#: constraints meet rather than one silently dominating the other.
CFL_SPEED_SIGMAS: Final[float] = 5.0

#: Bins in the emitted wall IEDF and on the ``f_i(z, v_z)`` velocity axis.
DEFAULT_ENERGY_BINS: Final[int] = 120
DEFAULT_VELOCITY_BINS: Final[int] = 160

#: Headroom on the energy and velocity axes over the free-fall value ``sqrt(2 e V_w / m_i)``.
#: Ions arrive with the sheath drop *plus* their entry energy, so an axis stopping at the
#: closed form would pile the fastest arrivals — the slowest-converging part of the IEDF,
#: and the part doc 03 §4.3 warns about — into its last bin.
AXIS_MARGIN: Final[float] = 1.25

#: Thermal energies the IEDF axis spans even at zero bias, so an unbiased or closed-box run
#: still produces a resolved distribution rather than one occupied bin.
AXIS_TEMPERATURE_FLOOR: Final[float] = 10.0

#: doc 03 §4.4's throughput figure, particle-steps per second per core. doc 03 §4.4 calls
#: it "an estimate to be measured" and gate G-1.4 requires the measurement to replace it,
#: which is why :meth:`Pic1D3VSolver.cost_estimate` reports ``ESTIMATED``.
PARTICLE_STEPS_PER_SECOND: Final[float] = 5.0e7

#: Root seed for a solver constructed without one, so that "no seed" is still reproducible.
DEFAULT_SEED: Final[int] = 20260804

_INT_KEYS: Final[tuple[str, ...]] = ("n_cells", "n_steps", "seed")
_FLOAT_KEYS: Final[tuple[str, ...]] = (
    "capacity_factor",
    "cells_per_debye",
    "domain_length_m",
    "domain_sheaths",
    "equilibration_fraction",
    "gamma_i",
    "h_l",
    "ion_reflection",
    "n_ppc",
    "timestep_refinement",
    "transits",
)
_CONFIG_KEYS: Final[frozenset[str]] = frozenset(_INT_KEYS + _FLOAT_KEYS + ("wall",))

#: Field names, and the order the snapshot tuple packs them in.
_FIELD_ORDER: Final[tuple[str, ...]] = ("n_i", "n_e", "Phi", "u_i", "T_e")
_FIELD_UNITS: Final[dict[str, str]] = {
    "n_i": "m**-3",
    "n_e": "m**-3",
    "Phi": "V",
    "u_i": "m/s",
    "T_e": "eV",
}


# ── what a run produces ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class IonEnergyDistribution:
    """The IEDF at the wall — doc 01 §1.1's deliverable, doc 03 §5.2's emulated quantity.

    Attributes:
        energy_ev: Histogram bin centres, in eV.
        flux_per_m2_s_per_ev: Differential arrival flux, normalised so that
            ``sum(flux * d_energy)`` recovers ``Gamma_i``.
        sample_energies_ev: Every absorbed macroparticle's impact energy, unbinned. doc 03
            §4.3 requires the ``N_ppc`` study to be run on the IEDF *shape* by KS distance,
            and a KS statistic taken from a histogram measures the binning as much as it
            measures the distribution.
        weight_per_sample: Physical ions each sample stands for.
        duration_s: The averaging window the flux is per-second over.
    """

    energy_ev: FloatArray
    flux_per_m2_s_per_ev: FloatArray
    sample_energies_ev: FloatArray
    weight_per_sample: float
    duration_s: float

    @property
    def n_samples(self) -> int:
        """Macroparticles that reached the wall during the window."""
        return int(self.sample_energies_ev.size)

    @property
    def mean_energy_ev(self) -> float:
        """``<E_i>`` — doc 01 §1.2's second factor. NaN, not zero, when nothing arrived."""
        return math.nan if self.n_samples == 0 else float(np.mean(self.sample_energies_ev))


@dataclass(frozen=True, slots=True)
class PicDiagnostics:
    """The run's account of itself, and the ledger doc 03 §7's gates are read from.

    The particle counts close: every macroparticle either sits in the domain or left
    through a named boundary, and every one present came from the load, the reservoir, the
    surface, or an ionising collision. A loop that leaked would move ``Gamma_E`` without
    moving anything else a reader would think to check, which is the shape of failure
    ADR-011 records.
    """

    n_steps: int
    n_equilibration_steps: int
    dt_s: float
    cell_width_m: float
    v_max_m_per_s: float
    window_duration_s: float
    weight: float
    total_energy_j: FloatArray
    kinetic_energy_j: FloatArray
    field_energy_j: FloatArray
    ions_loaded: int
    ions_injected: int
    ions_absorbed: int
    ions_left_bulk: int
    ions_alive: int
    electrons_loaded: int
    electrons_injected: int
    electrons_absorbed: int
    electrons_left_bulk: int
    electrons_alive: int
    secondaries_emitted: int
    collisions_by_process: tuple[int, ...]

    @property
    def energy_drift(self) -> float:
        """``(E_end - E_start) / E_start`` — verification gate V-07's statistic.

        The endpoints rather than a fitted slope: V-07 asks for drift *over the run*, and a
        symplectic integrator's energy error oscillates about zero, so a fit would report
        whichever phase of the oscillation the run happened to stop on.
        """
        if self.total_energy_j.size < 2 or self.total_energy_j[0] == 0.0:
            return 0.0
        start, end = float(self.total_energy_j[0]), float(self.total_energy_j[-1])
        return (end - start) / abs(start)


@dataclass(frozen=True, slots=True)
class PicRun:
    """Everything one L2 solve produced.

    Attributes:
        state: The doc 08 §4 return value, and all a caller of :meth:`Pic1D3VSolver.solve`
            sees.
        flux: ``Gamma_E`` and ``Gamma_i`` at the wall, accumulated from arrivals.
        iedf: The wall IEDF.
        diagnostics: The ledger.
        sheath_thickness_m: Where doc 03 §2.1's ``(n_i - n_e) / n_i = delta`` puts the
            sheath edge in the *solved* densities — the quantity verification gate V-03
            compares against
            :func:`~vpl.physics.analytic.sheath.child_langmuir_thickness`.
    """

    state: PlasmaState
    flux: IonEnergyFlux
    iedf: IonEnergyDistribution
    diagnostics: PicDiagnostics
    sheath_thickness_m: float


# ── internal machinery ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Plan:
    """Everything the loop needs that is derived rather than configured."""

    grid: UniformGrid
    dt_s: float
    n_steps: int
    n_equilibration: int
    v_max_m_per_s: float
    weight: float
    capacity: int
    sheath_thickness_m: float
    bulk_temperature_ev: float
    ion_source: InjectionSource
    electron_source: InjectionSource
    ion_species: PushSpecies
    electron_species: PushSpecies
    surface: SurfaceModel
    electron_surface: SurfaceModel
    energy_edges_ev: FloatArray
    velocity_edges_m_per_s: FloatArray
    snapshot_steps: tuple[int, ...]


@dataclass(slots=True)
class _Moments:
    """Per-node sums of weight, momentum and ``|v|^2`` — the raw material of every field."""

    weight: FloatArray
    momentum: FloatArray
    energy: FloatArray

    def add(self, other: _Moments) -> None:
        self.weight += other.weight
        self.momentum += other.momentum
        self.energy += other.energy


@dataclass(slots=True)
class _Accumulator:
    """Time *sums* over the diagnostic window. Sums until the very end, then one divide."""

    ion_density: FloatArray
    electron_density: FloatArray
    potential: FloatArray
    ion_moments: _Moments
    electron_moments: _Moments
    distribution: FloatArray
    steps: int = 0
    absorbed: list[FloatArray] = dataclass_field(default_factory=list)
    energies: list[tuple[float, float]] = dataclass_field(default_factory=list)


def _zero_moments(n_nodes: int) -> _Moments:
    return _Moments(
        weight=np.zeros(n_nodes, dtype=np.float64),
        momentum=np.zeros((n_nodes, 3), dtype=np.float64),
        energy=np.zeros(n_nodes, dtype=np.float64),
    )


def _bin_sum(index: NDArray[np.intp], weights: FloatArray, *, size: int) -> FloatArray:
    """``bincount`` with float weights — the per-node sums, without ``np.add.at``.

    ``np.add.at`` is the obvious spelling and is roughly an order of magnitude slower;
    inside a loop that runs tens of thousands of times that is the whole diagnostic budget.
    """
    return np.asarray(np.bincount(index, weights=weights, minlength=size), dtype=np.float64)


def _safe_ratio(numerator: FloatArray, denominator: FloatArray, *, fallback: float) -> FloatArray:
    """``numerator / denominator``, with a *stated* value where the denominator vanishes.

    An empty node is physical — a high-bias sheath holds no electrons at all — so the
    fallback is an argument. Zero for a temperature would be a claim that the electrons
    there are cold, and there are none there to be cold.
    """
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, fallback, dtype=np.float64),
        where=denominator > 0.0,
    )


def _nearest_node(positions_m: FloatArray, *, dz: float, n_nodes: int) -> NDArray[np.intp]:
    return np.asarray(np.clip(np.round(positions_m / dz).astype(np.intp), 0, n_nodes - 1))


def _node_moments(population: ParticlePopulation, *, dz: float, n_nodes: int) -> _Moments:
    """Weight, momentum and ``|v|^2`` sums per node, over the live particles only.

    Nearest-node binning, and only for the *moments*: the densities come from
    :func:`~vpl.physics.kinetic.fields.deposit_cic` and therefore carry the same weighting
    as the gather, which is what makes the self-force cancel (doc 03 §4.2 step 3). A
    velocity moment carries no such constraint, and nearest-node keeps
    ``sum(w v) / sum(w)`` an honest per-node average rather than a ratio of two
    differently smeared quantities.
    """
    moments = _zero_moments(n_nodes)
    live = np.flatnonzero(population.alive)
    if live.size == 0:
        return moments
    index = _nearest_node(population.positions_m[live], dz=dz, n_nodes=n_nodes)
    velocities = population.velocities_m_per_s[live]
    weights = np.full(live.size, population.weight, dtype=np.float64)
    moments.weight = _bin_sum(index, weights, size=n_nodes)
    for axis in range(3):
        moments.momentum[:, axis] = _bin_sum(index, weights * velocities[:, axis], size=n_nodes)
    moments.energy = _bin_sum(index, weights * np.sum(velocities**2, axis=1), size=n_nodes)
    return moments


def _temperature_ev(moments: _Moments, *, mass_kg: float, fallback_ev: float) -> FloatArray:
    """``T = m (<|v|^2> - |<v>|^2) / (3 e)`` per node, in eV — the measured second moment."""
    mean_square = _safe_ratio(moments.energy, moments.weight, fallback=0.0)
    drift = np.stack(
        [_safe_ratio(moments.momentum[:, k], moments.weight, fallback=0.0) for k in range(3)],
        axis=1,
    )
    spread = np.maximum(mean_square - np.sum(drift**2, axis=1), 0.0)
    return np.asarray(
        np.where(moments.weight > 0.0, mass_kg * spread / (3.0 * _E), fallback_ev),
        dtype=np.float64,
    )


def _sheath_edge_m(
    ion_density: FloatArray, electron_density: FloatArray, *, nodes_m: FloatArray
) -> float:
    """doc 03 §2.1's ``z_s``: the outermost node where ``(n_i - n_e) / n_i`` exceeds ``delta``.

    Outermost rather than first. The departure is not monotonic in a particle code — shot
    noise crosses ``delta`` repeatedly out in the quasineutral plasma — so the first
    crossing measures the noise floor and the last one measures the sheath.
    """
    departure = _safe_ratio(ion_density - electron_density, ion_density, fallback=0.0)
    inside = np.flatnonzero(departure > SHEATH_EDGE_TOLERANCE)
    return 0.0 if inside.size == 0 else float(nodes_m[int(inside[-1])])


# ── the solver ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Pic1D3VSolver:
    """L2 of the doc 03 §1 hierarchy — 1D3V electrostatic PIC-MCC, doc 03 §4.

    Implements the doc 08 §4 ``ForwardSolver`` protocol. Every attribute has a default, so
    ``Pic1D3VSolver()`` is runnable; a manifest overrides through :meth:`configure`.

    Attributes:
        n_ppc: Macroparticles per cell per species at load — doc 03 §4.3.
        cells_per_debye: ``lambda_D / dz``. doc 03 §4.3 bounds it below by 1.
        domain_sheaths: Domain length in Child-Langmuir sheath thicknesses, used when
            ``domain_length_m`` is unset.
        domain_length_m: Explicit domain length. Required at zero bias, where the sheath
            thickness vanishes and cannot size a domain.
        n_cells: Explicit cell count, overriding ``cells_per_debye``.
        n_steps: Explicit step count, overriding ``transits``.
        transits: Ion transit times to simulate.
        equilibration_fraction: Fraction of the run discarded before recording begins.
        timestep_refinement: Divides the derived ``dt``, and the knob verification gate
            V-05 turns. ``dt`` is otherwise *derived* from doc 03 §4.3's two bounds, so a
            solver cannot be handed an illegal timestep by accident.
        wall: doc 03 §3.3's absorbing wall, or the closed reflecting box V-07 needs.
        ion_reflection: ``R_i`` — doc 03 §8 A10.
        h_l: Planar edge-to-centre ratio setting the reservoir density ``n_s = h_l n_0``.
        gamma_i: Ion adiabatic index in ``c_s`` — see
            :func:`~vpl.physics.analytic.sheath.bohm_speed`.
        capacity_factor: Particle slots as a multiple of the loaded population.
        n_energy_bins: Bins in the emitted wall IEDF.
        n_velocity_bins: Bins on the ``f_i(z, v_z)`` velocity axis.
        seed: Root seed; streams derive from it by name (doc 10 §5).
        check_constraints: Whether to enforce doc 03 §4.3. **Only** a verification harness
            should set this false, and only to demonstrate that what the constraint forbids
            is what breaks: doc 03 §4.3 is explicit that a violation "is a runtime error
            rather than a warning".
        ion_collisions: doc 03 §4.2 step 5 for the ions, or ``None`` for a collisionless
            run.
        electron_collisions: The same for electrons; ionisation products from this sampler
            are admitted to both populations.
    """

    n_ppc: float = DEFAULT_N_PPC
    cells_per_debye: float = DEFAULT_CELLS_PER_DEBYE
    domain_sheaths: float = DEFAULT_DOMAIN_SHEATHS
    domain_length_m: float | None = None
    n_cells: int | None = None
    n_steps: int | None = None
    transits: float = DEFAULT_TRANSITS
    equilibration_fraction: float = DEFAULT_EQUILIBRATION_FRACTION
    timestep_refinement: float = 1.0
    wall: WallModel = WallModel.ABSORBING
    ion_reflection: float = DEFAULT_ION_REFLECTION
    h_l: float = DEFAULT_EDGE_TO_CENTRE_RATIO
    gamma_i: float = GAMMA_I_COLD_ION
    capacity_factor: float = DEFAULT_CAPACITY_FACTOR
    n_energy_bins: int = DEFAULT_ENERGY_BINS
    n_velocity_bins: int = DEFAULT_VELOCITY_BINS
    seed: int = DEFAULT_SEED
    check_constraints: bool = True
    ion_collisions: NullCollisionSampler | None = None
    electron_collisions: NullCollisionSampler | None = None
    _last_run: PicRun | None = dataclass_field(default=None, repr=False, compare=False)

    # ── the doc 08 §4 contract ──────────────────────────────────────────────────

    def configure(self, cfg: SolverConfig) -> None:
        """Apply the ``forward:`` block of a manifest — doc 08 §6.

        Unknown keys are refused. A silently ignored key is a run that finishes, is
        reproducible, and does not do what the file said — the failure
        :mod:`vpl.core.protocols.config` exists to prevent.
        """
        unknown = sorted(set(cfg.values) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"unknown forward-block key(s) {', '.join(unknown)}. This solver reads: "
                f"{', '.join(sorted(_CONFIG_KEYS))}."
            )
        for key in _INT_KEYS:
            if key in cfg:
                setattr(self, key, cfg.require_int(key))
        for key in _FLOAT_KEYS:
            if key in cfg:
                setattr(self, key, cfg.require_float(key))
        if "wall" in cfg:
            self.wall = WallModel(cfg.require_str("wall"))

    def fidelity(self) -> Fidelity:
        return Fidelity.L2

    def metadata(self) -> SolverMetadata:
        """doc 00 C2: every algorithm has a citation, and none of this one is new."""
        return SolverMetadata(
            name="vpl.physics.kinetic.pic1d3v",
            version="0.1.0",
            citations=(
                Citation(
                    key="birdsall-langdon-1991",
                    reference=(
                        "C. K. Birdsall and A. B. Langdon, Plasma Physics via Computer "
                        "Simulation, Adam Hilger, 1991 — CIC weighting, the leapfrog "
                        "advance, the 1-D electrostatic field solve and the stability "
                        "constraints of doc 03 §4.3."
                    ),
                ),
                Citation(
                    key="vahedi-surendra-1995",
                    reference=(
                        "V. Vahedi and M. Surendra, Computer Physics Communications 87 "
                        "(1995) 179 — the null-collision Monte-Carlo model for PIC and the "
                        "bounded-plasma treatment of a biased electrode."
                    ),
                ),
            ),
            description="1D3V electrostatic particle-in-cell with Monte-Carlo collisions.",
        )

    def cost_estimate(self, cfg: SolverConfig) -> CostEstimate:
        """What one solve costs — doc 10 §3, for the doc 10 §6 work queue.

        ``ESTIMATED``, and it stays so until gate G-1.4 (doc 11 §2) records a measured
        throughput: doc 03 §4.4 flags its own figure as an estimate in the table that
        states it.
        """
        cells = (cfg.require_int("n_cells") if "n_cells" in cfg else self.n_cells) or math.ceil(
            DEFAULT_DOMAIN_SHEATHS * self.cells_per_debye * DEFAULT_TRANSITS
        )
        steps = (cfg.require_int("n_steps") if "n_steps" in cfg else self.n_steps) or math.ceil(
            DEFAULT_TRANSITS * cells * CFL_SPEED_SIGMAS
        )
        return CostEstimate.estimated(
            wall_clock=Q_(2.0 * self.n_ppc * cells * steps / PARTICLE_STEPS_PER_SECOND, "s"),
            device=Device.CPU,
            source="doc 03 §4.4 throughput estimate; superseded at gate G-1.4",
        )

    def solve(self, params: PlasmaParams, t: TimeGrid | None) -> PlasmaState:
        """Advance the plasma and return the doc 08 §4 state.

        ``t`` selects the instants the fields are recorded at; ``None`` gives a steady state
        averaged over the diagnostic window. The wall flux and the IEDF live on
        :class:`PicRun` — use :meth:`run` when they are wanted, which for this project is
        always, since they are the deliverable (doc 01 §1.1).
        """
        return self.run(params, t).state

    def flux(self, state: PlasmaState, z: float) -> IonEnergyFlux:
        """``Gamma_E`` at ``z``, in metres — doc 03 §6's functional at L2.

        Only the state this solver last produced is accepted, and only at the wall.
        ``Gamma_E`` is accumulated from every macroparticle that crossed ``z = 0`` during
        the run; it is not recoverable from the gridded state, and reconstructing an
        approximation from the binned ``f_i`` would return a plausible wrong number, which
        is precisely the failure ADR-011 records. So it refuses instead.
        """
        run = self._last_run
        if run is None or run.state is not state:
            raise ValueError(
                "this solver did not produce that state, so it cannot supply its wall "
                "flux: Gamma_E is accumulated from arrivals during the run and is not "
                "recoverable from a gridded state. Call run() and read PicRun.flux."
            )
        if abs(z) > 0.5 * run.diagnostics.cell_width_m:
            raise NotImplementedError(
                f"the L2 flux functional is recorded at the wall, not at z = {z} m. "
                "doc 01 §1.1's quantity of interest is the flux at the surface."
            )
        return run.flux

    def surface_model(self, params: PlasmaParams) -> SurfaceModel:
        """The wall material's response for these parameters — doc 03 §3.3.

        ``gamma_se`` comes from :attr:`~vpl.core.state.PlasmaParams.gamma_se`, not from the
        registry: doc 05 §2.1 makes it a *control parameter*, and reading the registry here
        would make a swept yield silently inert inside the inversion.
        """
        return SurfaceModel(secondary_yield=params.gamma_se, ion_reflection=self.ion_reflection)

    # ── the run ─────────────────────────────────────────────────────────────────

    def run(self, params: PlasmaParams, t: TimeGrid | None = None) -> PicRun:
        """Execute doc 03 §4.2's six steps and return everything they produced."""
        if not is_double_precision():
            raise RuntimeError(
                "JAX x64 is off. This kernel is not correct in float32 — see "
                "vpl.physics.kinetic.precision for why that is a correctness statement."
            )
        plan = self._plan(params, t)
        if self.check_constraints:
            check_stability(
                plan.grid,
                dt_s=plan.dt_s,
                density_m3=params.n_0_per_m3,
                electron_temperature_ev=plan.bulk_temperature_ev,
                n_ppc=self.n_ppc,
                v_max_m_per_s=plan.v_max_m_per_s,
            )

        rng = generator(self.seed, Stream.PLASMA_INIT)
        mcc = collision_generator(self.seed)
        ions = self._load(rng, plan, plan.ion_species, plan.ion_source)
        electrons = self._load(rng, plan, plan.electron_species, plan.electron_source)
        schedules = (
            InjectionSchedule(plan.ion_source.flux_per_m2_s * plan.dt_s / plan.weight),
            InjectionSchedule(plan.electron_source.flux_per_m2_s * plan.dt_s / plan.weight),
        )
        acc = _Accumulator(
            ion_density=np.zeros(plan.grid.n_nodes),
            electron_density=np.zeros(plan.grid.n_nodes),
            potential=np.zeros(plan.grid.n_nodes),
            ion_moments=_zero_moments(plan.grid.n_nodes),
            electron_moments=_zero_moments(plan.grid.n_nodes),
            distribution=np.zeros((plan.grid.n_nodes, self.n_velocity_bins)),
        )
        wanted = set(plan.snapshot_steps)
        snapshots: dict[int, FloatArray] = {}
        counts: tuple[int, ...] = ()
        secondaries = 0

        for step in range(plan.n_steps):
            densities, potential, e_field = self._fields(params, plan, ions, electrons)
            acc.energies.append(self._energies(plan, ions, electrons, e_field))
            self._push(plan, ions, plan.ion_species, e_field)
            self._push(plan, electrons, plan.electron_species, e_field)
            counts = self._collide(mcc, plan, ions, electrons, counts)
            arrivals, emitted = self._boundaries(rng, plan, ions, electrons)
            secondaries += emitted
            self._inject(rng, plan, ions, electrons, schedules)
            if step >= plan.n_equilibration:
                self._record(acc, plan, ions, densities, potential, arrivals, electrons)
            if step in wanted:
                snapshots[step] = np.stack(
                    (*densities, potential, *self._instant(plan, ions, electrons))
                )

        return self._assemble(params, plan, t, acc, ions, electrons, snapshots, counts, secondaries)

    # ── planning ────────────────────────────────────────────────────────────────

    def _plan(self, params: PlasmaParams, t: TimeGrid | None) -> _Plan:
        """Derive the grid, the timestep and the run length from doc 03 §4.3's bounds."""
        temperature_ev = float(magnitude_in(params.T_e_eV, "eV"))
        ion_temperature_ev = float(magnitude_in(params.T_i_eV, "eV"))
        mass_i = params.species.mass_kg
        charge_i = params.species.charge_number * _E
        drop_v = float(magnitude_in(params.bias_magnitude, "V"))
        c_s = float(magnitude_in(bohm_speed(params, gamma_i=self.gamma_i), "m/s"))
        lambda_d = debye_length_m(
            density_m3=params.n_0_per_m3, electron_temperature_ev=temperature_ev
        )
        sheath = (
            float(magnitude_in(child_langmuir_thickness(params, h_l=self.h_l), "m"))
            if drop_v > 0.0
            else 0.0
        )
        length = self.domain_length_m or self.domain_sheaths * sheath
        if not length > 0.0:
            raise ValueError(
                "cannot size a domain: the Child-Langmuir thickness vanishes at zero bias. "
                "Set domain_length_m explicitly for an unbiased or closed configuration."
            )
        grid = UniformGrid(
            length_m=length,
            n_cells=self.n_cells or max(1, math.ceil(length * self.cells_per_debye / lambda_d)),
        )

        thermal_e = math.sqrt(_E * temperature_ev / _M_E)
        thermal_i = math.sqrt(_E * ion_temperature_ev / mass_i)
        free_fall_i = math.sqrt(2.0 * abs(charge_i) * drop_v / mass_i)
        v_max = max(
            CFL_SPEED_SIGMAS * thermal_e,
            free_fall_i + c_s + CFL_SPEED_SIGMAS * thermal_i,
            # A secondary emitted at the wall recrosses the whole drop, so whenever that
            # channel is on it is the fastest thing in the run — doc 03 §3.3.
            math.sqrt(2.0 * _E * drop_v / _M_E) if params.gamma_se > 0.0 else 0.0,
        )
        omega_pe = electron_plasma_frequency_rad_per_s(density_m3=params.n_0_per_m3)
        dt = min(TIMESTEP_SAFETY / omega_pe, grid.dz_m / v_max) / self.timestep_refinement

        # The ion transit, and the run length that follows from it: the quasineutral
        # presheath is crossed at c_s and the sheath in free fall, and the two differ by
        # sqrt(2 V_w / T_e) — a factor of 13 at RP-1. Using the slow branch for the whole
        # domain, which is the safe-looking thing to write, multiplies the run by that
        # factor for no extra physics.
        presheath = max(length - sheath, 0.0) / c_s
        transit = presheath + sheath / free_fall_i if sheath > 0.0 else length / c_s
        n_steps = self.n_steps or max(1, math.ceil(self.transits * transit / dt))
        loaded = max(1, int(self.n_ppc * grid.n_cells))
        density_s = self.h_l * params.n_0_per_m3
        weight = density_s * length / loaded
        energy_max = AXIS_MARGIN * max(
            abs(params.species.charge_number) * drop_v, AXIS_TEMPERATURE_FLOOR * temperature_ev
        )
        speed_max = AXIS_MARGIN * max(
            free_fall_i + c_s, CFL_SPEED_SIGMAS * thermal_i, math.sqrt(_E * energy_max / mass_i)
        )
        return _Plan(
            grid=grid,
            dt_s=dt,
            n_steps=n_steps,
            n_equilibration=min(int(self.equilibration_fraction * n_steps), n_steps - 1),
            v_max_m_per_s=v_max,
            weight=weight,
            capacity=math.ceil(self.capacity_factor * loaded),
            sheath_thickness_m=sheath,
            bulk_temperature_ev=temperature_ev,
            # doc 03 §3.3's bulk row: ions enter at c_s, electrons with no net drift.
            ion_source=InjectionSource(
                density_s, thermal_i, c_s if self.wall is WallModel.ABSORBING else 0.0
            ),
            electron_source=InjectionSource(density_s, thermal_e, 0.0),
            ion_species=PushSpecies("ion", charge_i, mass_i, weight),
            electron_species=PushSpecies("electron", -_E, _M_E, weight),
            surface=self.surface_model(params),
            # doc 03 §3.3's wall row absorbs the electron flux outright. R_i is doc 03 §8
            # A10's *ion* reflection coefficient — Ar+ on tungsten — and the boundary step
            # applies whatever surface it is handed to whichever species it is handed, so
            # reusing the ion surface for the electrons reflects them at R_i as well. The
            # default R_i = 0 hides that completely; it activates over the registry's own
            # [0, 0.2] sweep, where it suppresses the electron wall flux and shifts
            # Gamma_E with nothing in the ledger looking wrong.
            electron_surface=SurfaceModel(secondary_yield=0.0, ion_reflection=0.0),
            energy_edges_ev=np.linspace(0.0, energy_max, self.n_energy_bins + 1),
            velocity_edges_m_per_s=np.linspace(-speed_max, speed_max, self.n_velocity_bins + 1),
            snapshot_steps=tuple(
                int(np.clip(round(float(instant) / dt), 0, n_steps - 1))
                for instant in (t.t_s if t is not None else ())
            ),
        )

    def _load(
        self,
        rng: np.random.Generator,
        plan: _Plan,
        species: PushSpecies,
        source: InjectionSource,
    ) -> ParticlePopulation:
        """A quiet start: evenly spaced markers at the reservoir density and drift.

        Evenly spaced rather than randomly placed. Random loading begins the run with an
        ``N^(-1/2)`` density fluctuation that radiates away as a plasma oscillation, and at
        ``dz = lambda_D`` that oscillation is exactly the mode the grid resolves worst. A
        quiet start begins where the run is trying to get to, which is also why the
        equilibration window can be an ion transit rather than a damping time.
        """
        loaded = max(1, int(self.n_ppc * plan.grid.n_cells))
        positions = np.full(plan.capacity, PARKED_POSITION_M, dtype=np.float64)
        velocities = np.zeros((plan.capacity, 3), dtype=np.float64)
        alive = np.zeros(plan.capacity, dtype=bool)

        positions[:loaded] = (np.arange(loaded, dtype=np.float64) + 0.5) * (
            plan.grid.length_m / loaded
        )
        velocities[:loaded] = rng.normal(scale=source.thermal_speed_m_per_s, size=(loaded, 3))
        velocities[:loaded, 2] -= source.drift_speed_m_per_s
        alive[:loaded] = True
        return ParticlePopulation(
            positions_m=positions,
            velocities_m_per_s=velocities,
            alive=alive,
            weight=species.weight,
            name=species.name,
            loaded=loaded,
        )

    # ── the six steps ───────────────────────────────────────────────────────────

    def _fields(
        self,
        params: PlasmaParams,
        plan: _Plan,
        ions: ParticlePopulation,
        electrons: ParticlePopulation,
    ) -> tuple[tuple[FloatArray, FloatArray], FloatArray, Array]:
        """Steps 1-2: deposit both species, then Poisson with doc 03 §3.3's two ends."""
        ion_density = deposit_cic(
            plan.grid,
            positions_m=jnp.asarray(ions.positions_m),
            weights=jnp.asarray(ions.deposition_weights()),
        )
        electron_density = deposit_cic(
            plan.grid,
            positions_m=jnp.asarray(electrons.positions_m),
            weights=jnp.asarray(electrons.deposition_weights()),
        )
        charge = (
            plan.ion_species.charge_c * ion_density
            + plan.electron_species.charge_c * electron_density
        )
        potential = _jit_poisson(plan.grid, charge, left_v=params.bias_volts, right_v=0.0)
        return (
            (np.asarray(ion_density), np.asarray(electron_density)),
            np.asarray(potential),
            _electric_field(potential, jnp.asarray(plan.grid.dz_m)),
        )

    def _push(
        self,
        plan: _Plan,
        population: ParticlePopulation,
        species: PushSpecies,
        e_field: Array,
    ) -> None:
        """Steps 3-4: gather with the deposition's own weighting, then leapfrog.

        The gathered field is masked by ``alive`` rather than the dead slots being sliced
        out. A parked particle is already at rest, so a zero force leaves it exactly where
        it is — and the array keeps its shape, which is the whole point of parking it.
        """
        positions = jnp.asarray(population.positions_m)
        gathered = gather_cic(plan.grid, e_field, positions_m=positions) * jnp.asarray(
            population.alive
        )
        moved, accelerated = _jit_push(
            species,
            positions_m=positions,
            velocities_m_per_s=jnp.asarray(population.velocities_m_per_s),
            e_field_v_per_m=gathered,
            dt_s=plan.dt_s,
        )
        population.positions_m = np.asarray(moved, dtype=np.float64)
        population.velocities_m_per_s = np.asarray(accelerated, dtype=np.float64)

    def _collide(
        self,
        rng: np.random.Generator,
        plan: _Plan,
        ions: ParticlePopulation,
        electrons: ParticlePopulation,
        counts: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Step 5, for whichever species has a sampler. Parked slots are never offered.

        Passing them through would be *nearly* harmless — they are at rest, so
        ``nu = n sigma v`` vanishes — but their relative speed against a thermally drawn
        target does not, and a charge-exchange draw would resurrect one. They are excluded
        rather than relied on to be inert.
        """
        samplers = ((ions, self.ion_collisions), (electrons, self.electron_collisions))
        for population, sampler in samplers:
            if sampler is None:
                continue
            live = np.flatnonzero(population.alive)
            if live.size == 0:
                continue
            outcome = sampler.collide(
                rng,
                positions_m=population.positions_m[live],
                velocities_m_per_s=population.velocities_m_per_s[live],
                dt_s=plan.dt_s,
            )
            population.velocities_m_per_s[live] = np.asarray(outcome.velocities_m_per_s)
            counts = (
                outcome.counts_by_process
                if not counts
                else tuple(a + b for a, b in zip(counts, outcome.counts_by_process, strict=True))
            )
            born = np.asarray(outcome.new_electron_positions_m)
            if born.size:
                # Ionisation makes one electron-ion pair per event, so both populations
                # grow together and the plasma stays neutral to the macroparticle.
                electrons.admit(born, np.asarray(outcome.new_electron_velocities_m_per_s))
                ions.admit(
                    np.asarray(outcome.new_ion_positions_m),
                    np.asarray(outcome.new_ion_velocities_m_per_s),
                )
                electrons.injected += int(born.size)
                ions.injected += int(born.size)
        return counts

    def _boundaries(
        self,
        rng: np.random.Generator,
        plan: _Plan,
        ions: ParticlePopulation,
        electrons: ParticlePopulation,
    ) -> tuple[FloatArray, int]:
        """Step 6, plus the secondary electrons the ion half of it produces."""
        arrivals = np.empty(0, dtype=np.float64)
        absorbed_ions = 0
        for population, mass_kg, surface in (
            (ions, plan.ion_species.mass_kg, plan.surface),
            (electrons, plan.electron_species.mass_kg, plan.electron_surface),
        ):
            outcome = apply_boundaries(
                rng,
                positions_m=population.positions_m,
                velocities_m_per_s=population.velocities_m_per_s,
                alive=population.alive,
                domain_length_m=plan.grid.length_m,
                mass_kg=mass_kg,
                wall=self.wall,
                surface=None if self.wall is WallModel.REFLECTING else surface,
            )
            population.positions_m = outcome.positions_m
            population.velocities_m_per_s = outcome.velocities_m_per_s
            population.alive = outcome.alive
            population.absorbed += outcome.n_absorbed_at_wall
            population.left_bulk += outcome.n_left_bulk
            if population is ions:
                arrivals = outcome.absorbed_energies_ev
                absorbed_ions = outcome.n_absorbed_at_wall

        emitted = sample_secondary_velocities(rng, n_absorbed=absorbed_ions, surface=plan.surface)
        if emitted.shape[0]:
            electrons.admit(np.zeros(emitted.shape[0], dtype=np.float64), emitted)
        return arrivals, int(emitted.shape[0])

    def _inject(
        self,
        rng: np.random.Generator,
        plan: _Plan,
        ions: ParticlePopulation,
        electrons: ParticlePopulation,
        schedules: tuple[InjectionSchedule, InjectionSchedule],
    ) -> None:
        """Refill from doc 03 §3.3's reservoir at ``z = L``.

        Nothing is injected into a closed box: :attr:`WallModel.REFLECTING` loses no
        particles, so a source there would grow the plasma without bound and destroy the
        conservation statement V-07 is stated on.
        """
        if self.wall is WallModel.REFLECTING:
            return
        for population, schedule, source in (
            (ions, schedules[0], plan.ion_source),
            (electrons, schedules[1], plan.electron_source),
        ):
            count = schedule.take()
            if count == 0:
                continue
            population.admit(
                np.full(count, plan.grid.length_m),
                sample_inflow_velocities(rng, count, source=source),
            )
            population.injected += count

    # ── diagnostics ─────────────────────────────────────────────────────────────

    def _energies(
        self,
        plan: _Plan,
        ions: ParticlePopulation,
        electrons: ParticlePopulation,
        e_field: Array,
    ) -> tuple[float, float]:
        """Kinetic and electrostatic energy per unit wall area — verification gate V-07's.

        Both are evaluated *before* the push, against the field the same positions
        produced, so the pair is synchronised. A kinetic energy taken after the push would
        be half a step ahead of the field and would report an ``O(dt)`` offset as drift.
        """
        kinetic = 0.0
        for population, species in ((ions, plan.ion_species), (electrons, plan.electron_species)):
            speeds = np.sum(population.velocities_m_per_s**2, axis=1)
            kinetic += (
                0.5
                * species.mass_kg
                * population.weight
                * float(np.sum(np.where(population.alive, speeds, 0.0)))
            )
        return kinetic, 0.5 * _EPS0 * float(jnp.sum(e_field**2)) * plan.grid.dz_m

    def _instant(
        self, plan: _Plan, ions: ParticlePopulation, electrons: ParticlePopulation
    ) -> tuple[FloatArray, FloatArray]:
        """``u_i`` and ``T_e`` at one instant, for a snapshot."""
        nodes = plan.grid.n_nodes
        ion_moments = _node_moments(ions, dz=plan.grid.dz_m, n_nodes=nodes)
        electron_moments = _node_moments(electrons, dz=plan.grid.dz_m, n_nodes=nodes)
        return (
            _safe_ratio(ion_moments.momentum[:, 2], ion_moments.weight, fallback=0.0),
            _temperature_ev(electron_moments, mass_kg=_M_E, fallback_ev=plan.bulk_temperature_ev),
        )

    def _record(
        self,
        acc: _Accumulator,
        plan: _Plan,
        ions: ParticlePopulation,
        densities: tuple[FloatArray, FloatArray],
        potential: FloatArray,
        arrivals: FloatArray,
        electrons: ParticlePopulation,
    ) -> None:
        """Add one step to the window sums."""
        nodes = plan.grid.n_nodes
        acc.steps += 1
        acc.ion_density += densities[0]
        acc.electron_density += densities[1]
        acc.potential += potential
        acc.ion_moments.add(_node_moments(ions, dz=plan.grid.dz_m, n_nodes=nodes))
        acc.electron_moments.add(_node_moments(electrons, dz=plan.grid.dz_m, n_nodes=nodes))
        if arrivals.size:
            acc.absorbed.append(arrivals)

        live = np.flatnonzero(ions.alive)
        if live.size == 0:
            return
        edges = plan.velocity_edges_m_per_s
        speeds = ions.velocities_m_per_s[live, 2]
        # Masked rather than clipped: clipping would pile every out-of-range ion into an
        # end bin, and f_i's tail is the part doc 03 §4.3 says converges last.
        inside = (speeds >= edges[0]) & (speeds < edges[-1])
        if not inside.any():
            return
        node = _nearest_node(ions.positions_m[live][inside], dz=plan.grid.dz_m, n_nodes=nodes)
        bins = np.clip(np.digitize(speeds[inside], edges) - 1, 0, self.n_velocity_bins - 1)
        flat = np.asarray(node * self.n_velocity_bins + bins, dtype=np.intp)
        acc.distribution += _bin_sum(
            flat, np.full(flat.size, ions.weight), size=nodes * self.n_velocity_bins
        ).reshape(nodes, self.n_velocity_bins)

    # ── assembly ────────────────────────────────────────────────────────────────

    def _assemble(
        self,
        params: PlasmaParams,
        plan: _Plan,
        t: TimeGrid | None,
        acc: _Accumulator,
        ions: ParticlePopulation,
        electrons: ParticlePopulation,
        snapshots: dict[int, FloatArray],
        counts: tuple[int, ...],
        secondaries: int,
    ) -> PicRun:
        """Turn the window sums into a state, a flux and an IEDF."""
        steps = max(acc.steps, 1)
        grid = SpatialGrid(z_m=np.asarray(plan.grid.nodes_m, dtype=np.float64))
        ion_density = acc.ion_density / steps
        electron_density = acc.electron_density / steps
        averaged = np.stack(
            (
                ion_density,
                electron_density,
                acc.potential / steps,
                _safe_ratio(acc.ion_moments.momentum[:, 2], acc.ion_moments.weight, fallback=0.0),
                _temperature_ev(
                    acc.electron_moments, mass_kg=_M_E, fallback_ev=plan.bulk_temperature_ev
                ),
            )
        )
        values = (
            averaged
            if t is None
            else np.stack([snapshots[step] for step in plan.snapshot_steps], axis=0)
        )
        fields = {
            name: ScalarField(
                name=name,
                values=values[index] if t is None else values[:, index, :],
                units=_FIELD_UNITS[name],
                grid=grid,
                time=t,
            )
            for index, name in enumerate(_FIELD_ORDER)
        }

        window = steps * plan.dt_s
        arrivals = np.concatenate(acc.absorbed) if acc.absorbed else np.empty(0, dtype=np.float64)
        edges = plan.velocity_edges_m_per_s
        state = PlasmaState(
            params=params,
            grid=grid,
            time=t,
            fields=fields,
            ion_distribution=VelocityDistribution(
                grid=grid,
                v_m_per_s=0.5 * (edges[:-1] + edges[1:]),
                # int f dv_z = n: the sums are per node over `steps` steps, so dividing by
                # the window length, the cell volume and the bin width leaves a density.
                values=acc.distribution / (steps * plan.grid.dz_m * float(np.diff(edges)[0])),
                species=params.species,
            ),
            fidelity=Fidelity.L2,
        )
        energies = np.asarray(acc.energies, dtype=np.float64)
        run = PicRun(
            state=state,
            flux=IonEnergyFlux(
                position=Q_(0.0, "m"),
                species=params.species,
                energy_flux_toward_wall_watt_per_m2=np.asarray(
                    float(np.sum(arrivals)) * _E * plan.weight / window
                ),
                particle_flux_toward_wall_per_m2_s=np.asarray(arrivals.size * plan.weight / window),
                fidelity=Fidelity.L2,
                time=None,
            ),
            iedf=self._iedf(plan, arrivals, window),
            diagnostics=PicDiagnostics(
                n_steps=plan.n_steps,
                n_equilibration_steps=plan.n_equilibration,
                dt_s=plan.dt_s,
                cell_width_m=plan.grid.dz_m,
                v_max_m_per_s=plan.v_max_m_per_s,
                window_duration_s=window,
                weight=plan.weight,
                total_energy_j=energies.sum(axis=1),
                kinetic_energy_j=energies[:, 0],
                field_energy_j=energies[:, 1],
                ions_loaded=ions.loaded,
                ions_injected=ions.injected,
                ions_absorbed=ions.absorbed,
                ions_left_bulk=ions.left_bulk,
                ions_alive=ions.n_alive,
                electrons_loaded=electrons.loaded,
                electrons_injected=electrons.injected,
                electrons_absorbed=electrons.absorbed,
                electrons_left_bulk=electrons.left_bulk,
                electrons_alive=electrons.n_alive,
                secondaries_emitted=secondaries,
                collisions_by_process=counts,
            ),
            sheath_thickness_m=_sheath_edge_m(ion_density, electron_density, nodes_m=grid.z_m),
        )
        self._last_run = run
        return run

    def _iedf(self, plan: _Plan, arrivals: FloatArray, window_s: float) -> IonEnergyDistribution:
        """Bin the arrivals into doc 01 §1.1's deliverable."""
        edges = plan.energy_edges_ev
        counts, _ = np.histogram(arrivals, bins=edges)
        return IonEnergyDistribution(
            energy_ev=0.5 * (edges[:-1] + edges[1:]),
            flux_per_m2_s_per_ev=counts * plan.weight / (window_s * np.diff(edges)),
            sample_energies_ev=arrivals,
            weight_per_sample=plan.weight,
            duration_s=window_s,
        )
