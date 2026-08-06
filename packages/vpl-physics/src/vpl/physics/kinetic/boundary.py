"""Boundary conditions — doc 03 §4.2 step 6, doc 03 §3.3.

doc 03 §4.2 gives step 6 in five words — "absorption, SEE, reflection, injection" — and
doc 03 §3.3 states the two boundaries it means:

    | Wall (``z = 0``)  | ``Phi = V_w``; ion flux absorbed; secondary electron emission
    |                   | with yield ``gamma_se(E_i, material)``; ion reflection ``R_i`` |
    | Bulk (``z = L``)  | ``Phi = 0``; ``n_i = n_e = n_0``; ion flux entering at ``u = c_s`` |

The Dirichlet halves of both rows belong to the Poisson solve
(:func:`~vpl.physics.kinetic.fields.solve_poisson_dirichlet`). This module owns the
particle halves, and they are where the project's deliverable is made: doc 01 §1.1 asks
for the ion energy flux **at the surface**, so every ion that crosses ``z = 0`` has to be
counted exactly once, with the energy it actually carried.

## Why dead particles are parked rather than removed

JAX recompiles per array shape. Deleting absorbed particles shrinks the arrays almost
every step, which retraces the push and the field gather each time and costs more than the
physics does. So an absorbed particle keeps its slot and is marked not-alive, moved to a
position inside the domain, and set to **rest**:

- at rest it deposits nothing, because the solver zeroes the weight of a dead slot;
- at rest ``nu = n_gas sigma(E) v = 0``, so it is inert to the null-collision test of
  :mod:`~vpl.physics.kinetic.collisions` even if a caller passes it through;
- inside the domain it does not trip the bounds guard in
  :func:`~vpl.physics.kinetic.fields.deposit_cic`, which exists precisely so a lost
  particle cannot masquerade as wall flux.

The slot is then available for the next injected particle, which is what keeps the
population bounded without ever changing a shape.

## The wall is at z = 0 and only at z = 0

doc 02 §2 places the wall at ``z = 0`` with ``z`` increasing into the plasma, and calls
sign errors in flux quantities "common, silent and catastrophic". A particle leaving at
``z = L`` has returned to the bulk reservoir; it is removed, and it is **not** wall flux.
:class:`BoundaryOutcome` counts the two separately for that reason, and the solver's
particle balance is checked against both.

## Simplifications, and what each one costs

Stated here rather than discovered, per doc 00 C4.

1. **Secondary electrons are emitted at rest** (:data:`SECONDARY_EMISSION_ENERGY_EV` is
   zero by default, and is an argument so it can be swept). Measured ion-induced emission
   spectra peak at a few eV. doc 03 §3.3's point is that the secondary "is accelerated
   back through the full sheath potential, gaining 250 eV"; a 2 eV emission energy is
   therefore 0.8 % of the energy the electron carries into the plasma at RP-1, and the
   error scales as ``E_emit / (e V_w)``.
2. **Emission is isotropic into the half-space, and the yield is constant.** doc 03 §8 A7
   registers constant ``gamma_se`` as an assumption with an energy-dependent form
   available; ``SurfaceModel`` takes the yield as a number so the caller can supply
   ``gamma_se(E_i)`` per run. At the emission energy of simplification 1 the direction is
   irrelevant anyway — the particle starts at rest.
3. **Ion reflection is specular and elastic.** doc 03 §8 A10 sets ``R_i = 0`` by default
   and calls it negligible for Ar+ on W at these energies; the sweep range in the registry
   is ``[0, 0.2]``. A real reflected ion loses energy to the surface, so at ``R_i = 0.2``
   this over-returns energy to the plasma by at most the reflected fraction times the
   inelastic loss — bounded by ``R_i``, i.e. 20 % of a channel that is 0 % by default.
4. **The bulk boundary is a reservoir, not a source region.** Injection reproduces doc 03
   §3.3's stated condition — a Maxwellian at ``n_0`` whose ions drift at ``c_s`` — rather
   than solving the presheath that produces it. The consequence is that the ion flux
   arriving at the wall is *imposed* by the injection rate rather than predicted, so the
   Bohm flux is a bookkeeping check on this layer and not a physics prediction. The sheath
   thickness, the potential profile and the IEDF *are* predictions, and those are what the
   L0 cross-check tests.

## Sampling the inflow

The distribution of the wall-normal velocity of particles *crossing* a plane is not the
Maxwellian; it is the Maxwellian weighted by ``|v_z|``:

    p(w) ~ w exp(-(w - u)^2 / (2 sigma^2)),   w = |v_z| > 0

Injecting from a number-weighted half-Maxwellian instead is the classic inflow-boundary
error. It over-populates slow particles, so the reservoir it represents is denser and
colder than the one the manifest asked for, and every ion in the run enters the sheath
with the wrong energy. :func:`sample_inflow_velocities` samples the flux-weighted form
**exactly**, by rejection against the envelope ``(u + |w - u|) exp(...)`` which bounds it
everywhere and is itself a two-component mixture with closed-form inverses. At ``u = 0``
the envelope equals the target, acceptance is 1, and the algorithm degenerates to the
Rayleigh sampler that is the textbook answer there (Garcia & Wagner 2006).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from vpl.core.params import default_registry
from vpl.core.units import magnitude_in

__all__ = [
    "DEFAULT_ION_REFLECTION",
    "DEFAULT_SECONDARY_YIELD",
    "PARKED_POSITION_M",
    "SECONDARY_EMISSION_ENERGY_EV",
    "BoundaryOutcome",
    "CapacityExceededError",
    "InjectionSchedule",
    "InjectionSource",
    "ParticlePopulation",
    "SurfaceModel",
    "WallModel",
    "apply_boundaries",
    "sample_inflow_velocities",
    "sample_secondary_velocities",
]

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]

_E: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))

_REGISTRY = default_registry()

#: doc 03 §3.3's yield for Ar+ on tungsten at ~250 eV, from the registry. doc 03 §3.3:
#: "Secondary electron emission is not optional." The registry owns the number so the
#: uncertainty on it (+/- 0.03, PUBLISHED) propagates rather than being retyped here.
DEFAULT_SECONDARY_YIELD: Final[float] = float(
    _REGISTRY.value_in("sheath.gamma_se_W", "dimensionless")
)

#: doc 03 §8 A10's ion reflection coefficient. Zero by default and swept over ``[0, 0.2]``,
#: which is the difference between an assumption and a stated one.
DEFAULT_ION_REFLECTION: Final[float] = float(
    _REGISTRY.value_in("sheath.ion_reflection", "dimensionless")
)

#: Emission energy of a secondary electron. Zero — emitted at rest, then accelerated
#: through the sheath, which is where doc 03 §3.3 says its energy comes from. See
#: simplification 1 of the module docstring for the bound this carries.
SECONDARY_EMISSION_ENERGY_EV: Final[float] = 0.0

#: Where an absorbed particle is parked. The wall node itself: it is inside the domain, so
#: the deposition guard is satisfied, and it is where the particle physically went.
PARKED_POSITION_M: Final[float] = 0.0

#: Rejection loops are bounded so a mis-specified envelope cannot hang a 5 000-case
#: ensemble. The envelope's acceptance rate exceeds 1/2 for every ``u/sigma``, so thirty
#: passes leaves a failure probability below 1e-9 per particle.
_MAX_REJECTION_PASSES: Final[int] = 30


class WallModel(StrEnum):
    """What the surface at ``z = 0`` and the boundary at ``z = L`` do to a particle.

    A ``StrEnum`` because manifests are data (doc 08 §1 principle 4): ``wall: absorbing``
    round-trips into a checked member and back out for the provenance record.
    """

    #: doc 03 §3.3's physical wall: absorb, emit secondaries, reflect a fraction ``R_i``.
    #: Particles reaching ``z = L`` leave to the bulk reservoir.
    ABSORBING = "absorbing"

    #: Specular reflection at **both** ends: nothing enters and nothing leaves. Not a
    #: physical sheath boundary — it exists because verification gate V-07 asks for energy
    #: conservation, and a wall that absorbs particles removes energy from the system by
    #: design. The closed configuration is the only one in which "total energy drift" is a
    #: statement about the integrator rather than about the boundary.
    REFLECTING = "reflecting"


@dataclass(frozen=True, slots=True)
class SurfaceModel:
    """The wall material's response to an arriving ion — doc 03 §3.3.

    Attributes:
        secondary_yield: ``gamma_se``, electrons emitted per absorbed ion. Defaults to the
            registry's tungsten value; pass ``params.gamma_se`` to let a manifest set it.
        ion_reflection: ``R_i``, the probability that an arriving ion returns instead of
            being absorbed. doc 03 §8 A10.
        secondary_energy_ev: Kinetic energy a secondary electron leaves with. See
            :data:`SECONDARY_EMISSION_ENERGY_EV`.
    """

    secondary_yield: float = DEFAULT_SECONDARY_YIELD
    ion_reflection: float = DEFAULT_ION_REFLECTION
    secondary_energy_ev: float = SECONDARY_EMISSION_ENERGY_EV

    def __post_init__(self) -> None:
        if self.secondary_yield < 0.0:
            raise ValueError(
                f"secondary yield is a number of electrons per ion and cannot be "
                f"negative, got {self.secondary_yield}"
            )
        if not 0.0 <= self.ion_reflection <= 1.0:
            raise ValueError(
                f"ion reflection coefficient must lie in [0, 1], got {self.ion_reflection}. "
                "Above 1 the wall would return more ions than arrived and manufacture flux."
            )
        if self.secondary_energy_ev < 0.0:
            raise ValueError(
                f"secondary emission energy cannot be negative, got {self.secondary_energy_ev}"
            )


@dataclass(frozen=True, slots=True)
class InjectionSource:
    """The bulk reservoir at ``z = L`` — doc 03 §3.3's second row.

    Attributes:
        density_m3: The reservoir density. doc 03 §3.3 writes ``n_i = n_e = n_0``; a
            sheath-only domain whose outer boundary is the sheath edge uses ``n_s = h_l n_0``
            instead, and the caller decides which because only it knows where ``z = L`` is.
        thermal_speed_m_per_s: ``sqrt(k T / m)`` — the width of *one* Cartesian velocity
            component, matching ``BackgroundGas.thermal_speed_m_per_s`` in
            :mod:`~vpl.physics.kinetic.collisions`. Not the mean speed, and not the most
            probable speed: quoting it as either is how a Maxwellian acquires a stray
            ``sqrt(2)``.
        drift_speed_m_per_s: Bulk drift **toward the wall**, as a magnitude. doc 03 §3.3
            sets it to ``c_s`` for ions and it is zero for electrons. A magnitude rather
            than a signed component because doc 02 §2 records that sign errors in flux
            quantities are common, silent and catastrophic, and this is the one place a
            sign would flip the whole simulation inside out.
    """

    density_m3: float
    thermal_speed_m_per_s: float
    drift_speed_m_per_s: float

    def __post_init__(self) -> None:
        if not self.density_m3 > 0.0:
            raise ValueError(f"reservoir density must be positive, got {self.density_m3}")
        if self.thermal_speed_m_per_s < 0.0:
            raise ValueError(f"thermal speed cannot be negative, got {self.thermal_speed_m_per_s}")
        if self.drift_speed_m_per_s < 0.0:
            raise ValueError(
                f"drift speed is a magnitude toward the wall and cannot be negative, got "
                f"{self.drift_speed_m_per_s}. See doc 02 §2 for the sign convention."
            )

    @property
    def flux_per_m2_s(self) -> float:
        """The one-way particle flux across ``z = L`` toward the wall.

            Gamma / n = sigma exp(-u^2 / 2 sigma^2) / sqrt(2 pi)
                        + (u / 2) (1 + erf(u / (sigma sqrt2)))

        the first moment of the inward half of a drifting Maxwellian. At ``u = 0`` it
        reduces to the textbook ``n <v> / 4``; at ``u >> sigma`` it reduces to ``n u``,
        which is the Bohm flux doc 03 §2.3 calls the number every model must reproduce.
        """
        u, sigma = self.drift_speed_m_per_s, self.thermal_speed_m_per_s
        if sigma == 0.0:
            return self.density_m3 * u
        thermal = sigma * math.exp(-(u**2) / (2.0 * sigma**2)) / math.sqrt(2.0 * math.pi)
        directed = 0.5 * u * (1.0 + math.erf(u / (sigma * math.sqrt(2.0))))
        return self.density_m3 * (thermal + directed)


@dataclass(frozen=True, slots=True)
class BoundaryOutcome:
    """What doc 03 §4.2 step 6 did to one species this step.

    Attributes:
        positions_m: Positions after the boundary, shape ``(n,)``. A new array; the input
            is not mutated.
        velocities_m_per_s: Velocities after the boundary, shape ``(n, 3)``.
        alive: Which slots still hold a particle, shape ``(n,)``.
        absorbed_energies_ev: ``m |v|^2 / 2`` for each macroparticle absorbed **at the
            wall**, in eV. This is the raw material of the IEDF, and it is the velocity the
            particle arrived with rather than any recomputed quantity.
        absorbed_velocities_m_per_s: The same particles' velocities, shape ``(k, 3)``.
            Carried so that a caller — or a test — can check the energy against the
            kinematics instead of against a number this module also produced.
        n_absorbed_at_wall: How many were absorbed at ``z = 0``.
        n_reflected_at_wall: How many arrived at ``z = 0`` and were returned.
        n_left_bulk: How many left at ``z = L``. Counted separately from the wall because
            only the wall is the deliverable (doc 01 §1.1).
    """

    positions_m: FloatArray
    velocities_m_per_s: FloatArray
    alive: BoolArray
    absorbed_energies_ev: FloatArray
    absorbed_velocities_m_per_s: FloatArray
    n_absorbed_at_wall: int
    n_reflected_at_wall: int
    n_left_bulk: int


# ── the step ────────────────────────────────────────────────────────────────────


def apply_boundaries(
    rng: np.random.Generator,
    *,
    positions_m: ArrayLike,
    velocities_m_per_s: ArrayLike,
    alive: ArrayLike,
    domain_length_m: float,
    mass_kg: float,
    wall: WallModel,
    surface: SurfaceModel | None = None,
) -> BoundaryOutcome:
    """Apply doc 03 §4.2 step 6 to one species.

    Args:
        rng: The plasma stream's generator (doc 10 §5). Passed in rather than owned so the
            run's random sequence does not depend on how many objects were constructed.
        positions_m: Particle positions after the push, shape ``(n,)``. May lie outside
            ``[0, L]``; that is what this function is for.
        velocities_m_per_s: Shape ``(n, 3)``.
        alive: Which slots hold a particle. Dead slots are left exactly as they are — a
            dead particle re-entering the wall record would inflate ``Gamma_E`` every step
            it sat there, and the run would still finish.
        domain_length_m: ``L``. The bulk boundary.
        mass_kg: Mass of a *physical* particle of this species, for the recorded energy.
        wall: Which boundary model — see :class:`WallModel`.
        surface: The material response. Required for :attr:`WallModel.ABSORBING`, and
            never defaulted there: silently taking ``gamma_se = 0`` is the omission doc 03
            §3.3 calls "a common and significant modelling error".

    Returns:
        A :class:`BoundaryOutcome` holding new arrays. Nothing passed in is modified.
    """
    positions = np.array(positions_m, dtype=np.float64, copy=True)
    velocities = np.array(velocities_m_per_s, dtype=np.float64, copy=True)
    living = np.array(alive, dtype=bool, copy=True)
    _check_shapes(positions, velocities, living)

    if not domain_length_m > 0.0:
        raise ValueError(f"domain length must be positive, got {domain_length_m}")

    if wall is WallModel.REFLECTING:
        _reflect_into_domain(positions, velocities, living, length=domain_length_m)
        return BoundaryOutcome(
            positions_m=positions,
            velocities_m_per_s=velocities,
            alive=living,
            absorbed_energies_ev=np.empty(0, dtype=np.float64),
            absorbed_velocities_m_per_s=np.empty((0, 3), dtype=np.float64),
            n_absorbed_at_wall=0,
            n_reflected_at_wall=0,
            n_left_bulk=0,
        )

    if surface is None:
        raise ValueError(
            "an absorbing wall needs a surface model. Defaulting it would switch "
            "secondary emission off silently, which doc 03 §3.3 calls a common and "
            "significant modelling error; pass SurfaceModel(...) explicitly."
        )

    left_bulk = np.flatnonzero(living & (positions > domain_length_m))
    living[left_bulk] = False

    arrived = np.flatnonzero(living & (positions < 0.0))
    reflected = arrived[rng.random(arrived.size) < surface.ion_reflection]
    absorbed = np.setdiff1d(arrived, reflected, assume_unique=True)

    # Specular: mirror the position back across the wall and reverse the normal velocity.
    # The tangential components are untouched, which is what makes the reflection an
    # isometry and therefore energy-neutral.
    positions[reflected] = -positions[reflected]
    velocities[reflected, 2] = -velocities[reflected, 2]

    absorbed_velocities = velocities[absorbed].copy()
    absorbed_energies = 0.5 * mass_kg * np.sum(absorbed_velocities**2, axis=1) / _E

    # Park, do not delete. See the module docstring: at rest and inside the domain the
    # slot is inert to deposition, to the gather and to the collision test, and is
    # available for the next injected particle without any array ever changing shape.
    parked = np.concatenate([absorbed, left_bulk])
    living[absorbed] = False
    positions[parked] = PARKED_POSITION_M
    velocities[parked] = 0.0

    return BoundaryOutcome(
        positions_m=positions,
        velocities_m_per_s=velocities,
        alive=living,
        absorbed_energies_ev=np.asarray(absorbed_energies, dtype=np.float64),
        absorbed_velocities_m_per_s=absorbed_velocities,
        n_absorbed_at_wall=int(absorbed.size),
        n_reflected_at_wall=int(reflected.size),
        n_left_bulk=int(left_bulk.size),
    )


def _check_shapes(positions: FloatArray, velocities: FloatArray, alive: BoolArray) -> None:
    if positions.ndim != 1:
        raise ValueError(f"positions must have shape (n,), got {positions.shape}")
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError(
            f"velocities must have shape (n, 3) for a 1D3V model, got {velocities.shape}"
        )
    if velocities.shape[0] != positions.shape[0]:
        raise ValueError(
            f"velocities shape {velocities.shape} does not match positions shape {positions.shape}"
        )
    if alive.shape != positions.shape:
        raise ValueError(
            f"alive shape {alive.shape} does not match positions shape {positions.shape}"
        )


def _reflect_into_domain(
    positions: FloatArray, velocities: FloatArray, alive: BoolArray, *, length: float
) -> None:
    """Fold positions into ``[0, L]``, flipping ``v_z`` once per reflection. In place.

    The exact triangle-wave fold rather than a clamp or a single conditional bounce. A
    clamp would pile particles onto the wall node — which is exactly where this project
    measures its deliverable — and a single bounce would leave a particle outside the
    domain if it ever overshot by more than ``L``. The doc 03 §4.3 CFL constraint makes
    that unreachable, and the fold costs nothing, so the code does not depend on the
    constraint holding.
    """
    outside = alive & ((positions < 0.0) | (positions > length))
    if not outside.any():
        return

    folds = np.floor(positions[outside] / length)
    flipped = np.mod(folds, 2.0) != 0.0
    folded = positions[outside] - folds * length
    positions[outside] = np.where(flipped, length - folded, folded)

    indices = np.flatnonzero(outside)[flipped]
    velocities[indices, 2] = -velocities[indices, 2]


# ── secondary electron emission ─────────────────────────────────────────────────


def sample_secondary_velocities(
    rng: np.random.Generator, *, n_absorbed: int, surface: SurfaceModel
) -> FloatArray:
    """Velocities of the electrons doc 03 §3.3 says the wall emits — shape ``(k, 3)``.

    One Bernoulli trial per absorbed ion at yield ``gamma_se``, so ``k`` is binomial. That
    is the right sampler for a yield below one; for a yield above one the count is the
    integer part plus a Bernoulli on the remainder, which the same expression gives via
    :meth:`~numpy.random.Generator.binomial` on the integer part.

    The caller places the electrons at the wall (``z = 0``) and gives them the electron
    species' weight. This function owns only the kinematics, because the position and the
    weight are the solver's bookkeeping and mixing the two is how a secondary ends up
    carrying an ion's weight.

    Args:
        rng: The plasma stream's generator.
        n_absorbed: How many ions were absorbed this step.
        surface: Supplies the yield and the emission energy.

    Returns:
        Velocities with ``v_z >= 0`` — emitted **into** the plasma (doc 02 §2).
    """
    if n_absorbed < 0:
        raise ValueError(f"absorbed count cannot be negative, got {n_absorbed}")
    if n_absorbed == 0 or surface.secondary_yield == 0.0:
        return np.empty((0, 3), dtype=np.float64)

    whole, remainder = divmod(surface.secondary_yield, 1.0)
    emitted = int(whole) * n_absorbed + int(rng.binomial(n_absorbed, remainder))
    if emitted == 0:
        return np.empty((0, 3), dtype=np.float64)

    speed = math.sqrt(2.0 * surface.secondary_energy_ev * _E / _M_E)
    if speed == 0.0:
        # Emitted at rest — simplification 1. The direction is then meaningless and
        # inventing one would suggest the model resolves something it does not.
        return np.zeros((emitted, 3), dtype=np.float64)

    direction = _isotropic_into_plasma(rng, emitted)
    return np.asarray(speed * direction, dtype=np.float64)


def _isotropic_into_plasma(rng: np.random.Generator, n: int) -> FloatArray:
    """``n`` unit vectors uniform on the hemisphere ``v_z >= 0``.

    ``cos theta`` uniform on ``[0, 1]`` and ``phi`` uniform on ``[0, 2 pi)`` — the same
    construction as :func:`~vpl.physics.kinetic.collisions._isotropic`, halved. Sampling
    ``theta`` uniformly instead is the classic error and it clusters directions along
    ``z``, which in a 1D3V code is the one axis every observable is projected onto.
    """
    cos_theta = rng.random(n)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = 2.0 * np.pi * rng.random(n)
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=1)


# ── injection at the bulk boundary ──────────────────────────────────────────────


def sample_inflow_velocities(
    rng: np.random.Generator, n: int, *, source: InjectionSource
) -> FloatArray:
    """``n`` velocities entering at ``z = L`` and moving toward the wall — shape ``(n, 3)``.

    The wall-normal component is drawn from the **flux-weighted** drifting Maxwellian
    ``p(w) ~ w exp(-(w - u)^2 / (2 sigma^2))`` on ``w = |v_z| > 0``, exactly; see the
    module docstring for why the number-weighted form is wrong and what it costs. The two
    transverse components are plain Gaussians of width ``sigma``: the flux weighting acts
    on the normal component alone, and weighting the others too would narrow the
    projected distribution LIF measures (doc 04 §3.2).

    Args:
        rng: The plasma stream's generator.
        n: How many to draw.
        source: The reservoir.

    Returns:
        Velocities with ``v_z <= 0``. The sign is doc 02 §2's: ``z`` increases into the
        plasma, so a particle entering at ``z = L`` and heading for the wall has
        ``v_z < 0``.
    """
    if n < 0:
        raise ValueError(f"injection count cannot be negative, got {n}")
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    sigma = source.thermal_speed_m_per_s
    u = source.drift_speed_m_per_s
    if sigma == 0.0:
        # The cold limit. Every particle enters at the drift speed, which is the
        # zero-temperature limit of the flux-weighted density above.
        velocities = np.zeros((n, 3), dtype=np.float64)
        velocities[:, 2] = -u
        return velocities

    transverse = rng.normal(scale=sigma, size=(n, 2))
    normal = _sample_flux_weighted_speed(rng, n, drift=u, sigma=sigma)
    return np.concatenate([transverse, -normal[:, None]], axis=1)


def _sample_flux_weighted_speed(
    rng: np.random.Generator, n: int, *, drift: float, sigma: float
) -> FloatArray:
    """Sample ``p(w) ~ w exp(-(w - u)^2 / (2 sigma^2))`` on ``w > 0``, exactly.

    Rejection against ``q(w) ~ (u + |w - u|) exp(-(w - u)^2 / (2 sigma^2))``, which bounds
    the target everywhere because ``w = u + (w - u) <= u + |w - u|`` for ``w > 0``, and
    which is a two-component mixture with closed-form inverses:

    - ``u exp(-s^2 / 2 sigma^2)`` on ``s > -u`` — a normal truncated to the half line;
    - ``|s| exp(-s^2 / 2 sigma^2)`` on ``s > -u`` — a two-sided Rayleigh, each side of
      which inverts in one logarithm.

    Acceptance is ``(u + s) / (u + |s|)``, i.e. one for every ``s >= 0``, so the rate
    exceeds one half for every ``u / sigma`` and is exactly one at ``u = 0`` — where the
    envelope collapses onto the target and the algorithm becomes the Rayleigh sampler that
    is the textbook answer there.
    """
    accepted = np.empty(n, dtype=np.float64)
    filled = 0
    for _ in range(_MAX_REJECTION_PASSES):
        wanted = n - filled
        shift = _sample_envelope(rng, wanted, drift=drift, sigma=sigma)
        keep = rng.random(wanted) * (drift + np.abs(shift)) < drift + shift
        taken = shift[keep]
        accepted[filled : filled + taken.size] = drift + taken
        filled += taken.size
        if filled == n:
            return accepted
    raise RuntimeError(
        f"the inflow rejection sampler failed to fill {n} draws in "
        f"{_MAX_REJECTION_PASSES} passes at u/sigma = {drift / sigma:.3g}. The envelope "
        "no longer bounds the target, which would bias every injected particle."
    )


def _sample_envelope(rng: np.random.Generator, n: int, *, drift: float, sigma: float) -> FloatArray:
    """Draw ``s = w - u`` from the envelope of :func:`_sample_flux_weighted_speed`."""
    cutoff = math.exp(-(drift**2) / (2.0 * sigma**2))
    # Component masses, as integrals over s > -u. The normal component vanishes at u = 0,
    # which is what makes the algorithm exact there rather than merely accurate.
    normal_mass = drift * sigma * math.sqrt(2.0 * math.pi) * _standard_normal_cdf(drift / sigma)
    rayleigh_mass = sigma**2 * (2.0 - cutoff)

    from_normal = rng.random(n) * (normal_mass + rayleigh_mass) < normal_mass
    shift = np.empty(n, dtype=np.float64)

    count = int(np.count_nonzero(from_normal))
    if count:
        shift[from_normal] = _truncated_normal(rng, count, sigma=sigma, lower=-drift)

    remaining = int(n - count)
    if remaining:
        # The two-sided Rayleigh. The inward side is truncated at s = -u, so its inverse
        # carries the truncation through the same logarithm.
        negative_mass = 1.0 - cutoff
        inward = rng.random(remaining) * (negative_mass + 1.0) < negative_mass
        draw = rng.random(remaining)
        scaled = np.where(inward, 1.0 - draw * negative_mass, draw)
        radius = sigma * np.sqrt(-2.0 * np.log(np.maximum(scaled, np.finfo(np.float64).tiny)))
        shift[~from_normal] = np.where(inward, -radius, radius)

    return shift


def _truncated_normal(
    rng: np.random.Generator, n: int, *, sigma: float, lower: float
) -> FloatArray:
    """``N(0, sigma)`` conditioned on ``s > lower``, by resampling.

    Resampling rather than an inverse CDF because ``lower = -u <= 0`` always here, so the
    acceptance rate is at least one half and never approaches the regime where naive
    resampling of a truncated normal becomes pathological.
    """
    accepted = np.empty(n, dtype=np.float64)
    filled = 0
    for _ in range(_MAX_REJECTION_PASSES):
        draw = rng.normal(scale=sigma, size=n - filled)
        taken = draw[draw > lower]
        accepted[filled : filled + taken.size] = taken
        filled += taken.size
        if filled == n:
            return accepted
    raise RuntimeError(  # pragma: no cover - unreachable while lower <= 0
        f"truncated normal sampling failed to fill {n} draws in {_MAX_REJECTION_PASSES} passes"
    )


def _standard_normal_cdf(x: float) -> float:
    """``Phi(x)``, from ``erf``. One place, so the factor of ``sqrt2`` cannot drift."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── the particle store the parking scheme implies ───────────────────────────────


class CapacityExceededError(RuntimeError):
    """A fixed-capacity particle array ran out of free slots.

    A distinct type, and a ``RuntimeError``, for the same reason
    :class:`~vpl.physics.kinetic.constraints.StabilityViolationError` is one: an ensemble
    driver needs to tell "this configuration outgrew its allocation" apart from "this
    configuration is malformed". Raised rather than dropping the excess, because a dropped
    injection is a quietly reduced flux and the flux is the deliverable.
    """


@dataclass(slots=True)
class ParticlePopulation:
    """One species' fixed-capacity store, with the alive mask the parking scheme needs.

    Mutable, and deliberately: this is the array a 65 000-step loop rewrites twice per
    step, and returning a new copy each time would allocate more memory than the run uses.
    Every function in this module that *transforms* particles returns new arrays; this
    type is the buffer they are written back into.

    Attributes:
        positions_m: ``(capacity,)``. Dead slots hold :data:`PARKED_POSITION_M`.
        velocities_m_per_s: ``(capacity, 3)``. Dead slots hold zero.
        alive: ``(capacity,)``. The only thing that distinguishes a particle from a slot.
        weight: Physical particles one macroparticle stands for.
        name: Species label, for the error a full array raises.
        loaded: How many were placed at ``t = 0``.
        injected: How many have been admitted since — from the reservoir, from the
            surface as secondaries, or born in an ionising collision.
        absorbed: How many were absorbed at the wall.
        left_bulk: How many left at ``z = L``.
    """

    positions_m: FloatArray
    velocities_m_per_s: FloatArray
    alive: BoolArray
    weight: float
    name: str
    loaded: int = 0
    injected: int = 0
    absorbed: int = 0
    left_bulk: int = 0

    @property
    def n_alive(self) -> int:
        return int(np.count_nonzero(self.alive))

    def deposition_weights(self) -> FloatArray:
        """Per-slot weight, zero where the slot is dead.

        This is what makes a parked particle inert rather than merely stationary: it
        deposits no charge, so it cannot perturb the Poisson solve from the wall node it
        is parked on — which is the node where this project measures its deliverable.
        """
        return np.where(self.alive, self.weight, 0.0)

    def admit(self, positions_m: FloatArray, velocities_m_per_s: FloatArray) -> None:
        """Place new particles into free slots, or refuse loudly."""
        if positions_m.size == 0:
            return
        free = np.flatnonzero(~self.alive)
        if free.size < positions_m.size:
            raise CapacityExceededError(
                f"{self.name}: {positions_m.size} particles to admit but only {free.size} "
                f"free slots of {self.alive.size}. Dropping them would quietly reduce the "
                f"injected flux, and the flux is the deliverable. Raise capacity_factor."
            )
        slots = free[: positions_m.size]
        self.positions_m[slots] = positions_m
        self.velocities_m_per_s[slots] = velocities_m_per_s
        self.alive[slots] = True


@dataclass(slots=True)
class InjectionSchedule:
    """How many macroparticles the reservoir sends in each step, from a fractional rate.

    Deterministic rather than Poisson. The reservoir of doc 03 §3.3 is a boundary
    *condition*, not a physical shot-noise source, and adding sampling noise to it would
    put a second ``N^(-1/2)`` term into the IEDF that doc 03 §4.3's convergence study would
    then have to converge away. The residual carries the fractional part forward, so the
    injected flux is exact in the mean over any window longer than ``1 / rate``, and the
    accumulated error is bounded by one macroparticle at every instant.
    """

    rate_per_step: float
    residual: float = 0.0

    def __post_init__(self) -> None:
        if self.rate_per_step < 0.0:
            raise ValueError(f"injection rate cannot be negative, got {self.rate_per_step}")

    def take(self) -> int:
        """The count for this step."""
        self.residual += self.rate_per_step
        whole = math.floor(self.residual)
        self.residual -= whole
        return int(whole)
