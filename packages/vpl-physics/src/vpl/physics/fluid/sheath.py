"""L1 — the coupled fluid sheath. Doc 03 §3, implemented as written.

The three-field system doc 03 §3.4 names — ``(n_i, u_i, Phi)``, solved by Newton, with
Scharfetter-Gummel exponential fitting on the ion advection, on a mesh graded toward the
wall at ``dz <= lambda_D / 4``. In the scaled variables of
:mod:`vpl.physics.fluid.scaling`:

    ``(nu w)'           = sigma``                                    ion continuity
    ``(nu w^2)'         = -Theta nu phi' - c_p nu' - nu w |w| / lambda``   ion momentum
    ``-lambda_D^2 phi'' = nu - exp(phi) - nu_se(phi)``               Poisson

## Why inertia is retained and drift-diffusion is not enough

doc 03 §1 calls L1 "drift-diffusion ions", and doc 03 §3.1 writes its momentum equation
under the heading "drift-diffusion **with inertia**". The distinction is the whole of
verification gate V-03. A constant-mobility drift-diffusion ion flux gives the
Mott-Gurney law ``J ~ V^2 / s^3``; the collisionless Child-Langmuir law doc 03 §2.3 makes
"the framework's primary analytic verification target" is ``J ~ V^{3/2} / s^2``, and it
follows from the *inertial* momentum equation and from nothing else. An L1 built on the
mobility closure alone would fail V-03 by construction, and the failure would look like a
numerical defect.

## Where this departs from doc 03 §3.3, and why

doc 03 §3.3's bulk boundary condition reads ``n_i = n_e = n_0``. This module applies
``n_i = n_e = n_s = h_l n_0`` instead, for the reason
:func:`vpl.physics.analytic.sheath.AnalyticSheathSolver.solve` gives at length: ``h_l``
already encodes the presheath drop, so referring the Boltzmann electrons to ``n_0`` at a
boundary where ``Phi = 0`` counts that drop twice and inflates every flux downstream by
``1 / h_l = 1.64``. Applied literally, doc 03 §3.3 would put L1's ion flux 64 % above the
L0 Bohm flux that doc 03 §7 V-03 checks it against — the same class of internal
disagreement ADR-007 records, resolved the same way.

The second departure is the injection speed; see :data:`DEFAULT_BOHM_MARGIN`.

## What "Newton on the fully coupled system" actually costs

doc 03 §3.4 chooses Newton over Gummel iteration and gives a good reason. It does not say
that Newton **does not converge from the L0 seed at a sheath bias**, and it does not: at
RP-1's ``V_w / T_e = 83`` the residual grows by four orders in four iterations and then
overflows, on every mesh, domain length, Bohm margin and damping factor tried. The wall
potential is therefore walked up from a quarter of an electron temperature in geometric
steps, each converged solution seeding the next, with any step that fails halved and
retried — :func:`vpl.physics.fluid.system.solve_system`. At RP-1 that is 19 steps and 108
linear solves for one answer.

It is still fast: 86 ms per solve at the default resolution, against doc 03 §1's "~1 s"
estimate. But the *shape* of the cost is not what doc 03 §3.4 implies, and it is the thing
about this implementation a reviewer should look at first.

## What L1 cannot do, stated here rather than discovered later

doc 03 §6 is explicit, and the honesty is load-bearing:

    **The L1 reconstruction is a genuine weakness and is treated as one.** A fluid model
    cannot represent the bimodal RF IEDF or the collisional low-energy tail; forcing a
    drifting Maxwellian onto them produces a systematically wrong ``Gamma_E``.

So :meth:`FluidSheathSolver.flux` reconstructs a drifting Maxwellian from the fluid
moments and the returned :class:`~vpl.core.state.PlasmaState` carries **no** ion
distribution — ``Fidelity.L1.resolves_distribution`` is ``False``. Benchmark B-03
quantifies the error; nothing here is entitled to pretend it is absent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from vpl.core.params import default_registry
from vpl.core.protocols.config import SolverConfig
from vpl.core.protocols.forward import CostEstimate, Device, IonEnergyFlux
from vpl.core.protocols.metadata import Citation, SolverMetadata
from vpl.core.state import (
    Fidelity,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    TimeGrid,
)
from vpl.core.units import Q_, Quantity, magnitude_in
from vpl.physics.analytic.sheath import (
    DEFAULT_DOMAIN_SHEATHS,
    DEFAULT_EDGE_TO_CENTRE_RATIO,
    GAMMA_I_COLD_ION,
    SheathModel,
    child_langmuir_thickness,
    matrix_sheath_thickness,
)
from vpl.physics.fluid.forms import (
    DEFAULT_NEWTON_RTOL,
    NewtonReport,
    SourceTerm,
)
from vpl.physics.fluid.mesh import (
    DEFAULT_DOMAIN_LENGTH,
    DEFAULT_MESH_STRETCH,
    MIN_CELLS_PER_DEBYE,
    graded_sheath_grid,
)
from vpl.physics.fluid.scaling import SheathScaling
from vpl.physics.fluid.system import (
    CONTINUATION_START,
    MAX_BOLTZMANN_EXPONENT,
    MIN_CONTINUATION_STEP,
    solve_system,
)

__all__ = [
    "CONTINUATION_START",
    "DEFAULT_BOHM_MARGIN",
    "DEFAULT_CONTINUATION_FACTOR",
    "MAX_BOLTZMANN_EXPONENT",
    "MIN_CONTINUATION_STEP",
    "SECONDARY_EMISSION_ENERGY",
    "SHEATH_EDGE_TOLERANCE",
    "FluidSheathSolver",
    "SheathSolution",
]

#: A field sampled on the spatial grid, in SI magnitudes.
type FloatArray = NDArray[np.float64]

_REGISTRY: Final = default_registry()

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E_KG: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))

#: ``delta`` in doc 03 §2.1's sheath-edge definition ``(n_i - n_e) / n_i = delta``.
#: Registered as ``sheath.edge_tolerance``, and doc 03 §2.1 names it as "precisely the
#: sort of hidden convention that doc 00 C4 exists to forbid".
SHEATH_EDGE_TOLERANCE: Final[float] = float(
    _REGISTRY.value_in("sheath.edge_tolerance", "dimensionless")
)

#: Mean initial energy of an ion-induced secondary electron leaving the wall.
#:
#: **This value is not in the parameter registry and should be.** doc 03 §3.3 requires
#: secondary emission and registers the *yield* (``sheath.gamma_se_W``) but not the
#: emission energy, and the emission energy is needed as soon as the secondaries are
#: modelled as a beam rather than as a boundary current: born exactly at rest at the wall
#: they would have zero velocity and infinite density there. Ion-induced (potential)
#: emission from a clean metal produces electrons with a roughly Maxwellian distribution
#: peaking near 2 eV and almost independent of the impact energy — Hagstrum (1954),
#: Phys. Rev. 96, 336. Named here so it is greppable and reviewable, and flagged so that
#: it is promoted to ``sheath.E_se`` when ``vpl-core`` is next touched.
SECONDARY_EMISSION_ENERGY: Final[Quantity] = Q_(2.0, "eV")

#: How far above the Bohm speed ions are injected at the bulk boundary.
#:
#: doc 03 §2.1 states the Bohm criterion as ``u_s >= c_s`` and doc 03 §3.3 applies it at
#: equality. Exactly at equality the quasineutral presheath is *sonic*: the reduced
#: system is ``w'(w^2 - 1) = 0`` (see :mod:`vpl.physics.fluid.scaling`), so the inflow
#: characteristic is stationary and the linearised problem is singular there. Injecting a
#: tenth of a percent supersonically restores strict hyperbolicity and is the standard
#: regularisation of the classical sheath-matching problem.
#:
#: It is not free, and the cost is stated rather than hidden: the ion flux, and therefore
#: ``Gamma_E``, is high by exactly this fraction against L0. At the default that is
#: 0.1 %, against a doc 03 §7 V-03 tolerance of 5 % and a doc 01 R-ACC-5 budget of 20 %.
DEFAULT_BOHM_MARGIN: Final[float] = 1e-3

#: Newton iteration cap per continuation step.
DEFAULT_MAX_ITERATIONS: Final[int] = 60

#: Newton relaxation (damping) factor.
DEFAULT_RELAXATION: Final[float] = 1.0

#: Ratio between successive continuation steps. 1.4 was the largest that reached RP-1's
#: ``V_w / T_e = 83`` without the adaptive halving of :func:`_continue_to` having to
#: intervene; 2.0 does not.
DEFAULT_CONTINUATION_FACTOR: Final[float] = 1.4

#: Component indices in the mixed space.
_NU, _W, _PHI = 0, 1, 2


@dataclass(frozen=True, slots=True)
class SheathSolution:
    """One L1 solve, with the quantities the verification gates ask for.

    Attributes:
        state: The :class:`~vpl.core.state.PlasmaState` at :attr:`Fidelity.L1`.
        scaling: The reference values the system was solved in.
        newton: How the nonlinear solve went.
        sheath_thickness: ``s``, located by doc 03 §2.1's ``(n_i - n_e) / n_i = delta``
            criterion. The V-03 comparison target.
        wall_current_density: ``J_i = e n_i u_i`` at ``z = 0``. The other half of V-03.
        continuation_steps: How many bias steps :func:`_continue_to` needed. One means
            Newton reached the answer directly; more is the honest cost of doc 03 §3.4's
            coupled Newton, and it is what the G-1.4 throughput measurement is measuring.
    """

    state: PlasmaState
    scaling: SheathScaling
    newton: NewtonReport
    sheath_thickness: Quantity
    wall_current_density: Quantity
    continuation_steps: int


@dataclass(frozen=True, slots=True)
class FluidSheathSolver:
    """L1 of the doc 03 §1 hierarchy.

    Shaped to the ``ForwardSolver`` contract of doc 08 §4 — ``solve``, ``flux``,
    ``fidelity``, ``cost_estimate``, ``metadata`` — but, like
    :class:`~vpl.physics.analytic.sheath.AnalyticSheathSolver`, not declared to implement
    it. Two of the protocol's signatures do not fit and forcing them would cost more than
    the nominal conformance is worth:

    - ``configure(cfg) -> None`` mutates, and this codebase's immutability rule (doc 08
      §5, and every state type in ``vpl-core``) says otherwise. :meth:`with_config`
      returns a new solver instead.
    - ``solve(params, t: TimeGrid)`` requires a time grid, and P1 is steady. Passing one
      would imply a time dependence this implementation does not model; doc 03 §3.4
      specifies BDF2 for when it does.

    Structural typing means the annotation can be added when the manifest engine (WBS
    1.3) fixes the remaining shape.

    Attributes:
        h_l: Planar edge-to-centre density ratio. The bulk boundary sits at ``h_l n_0``;
            see the module docstring.
        gamma_i: Ion adiabatic index, in the Bohm speed *and* the pressure closure. See
            :mod:`vpl.physics.fluid.scaling`.
        cells_per_debye: Cells per Debye length inside the sheath. doc 03 §3.4 requires
            at least 4; V-04 halves ``dz`` by doubling this.
        stretch: Geometric cell growth outside the sheath — ``mesh.A.stretch``.
        domain_sheaths: Domain length in sheath thicknesses, floored at
            ``mesh.domain_length``. doc 01 R-SPAT-3 wants at least ten.
        mean_free_path: Ion-neutral mean free path for the doc 03 §3.1 drag term, or
            ``None`` for the collisionless limit that doc 03 §7 V-03 is stated in.
            Never derived silently: :func:`vpl.physics.analytic.sheath.cx_mean_free_path`
            takes a cross section explicitly for the same reason.
        ion_source: ``S_iz - L_rec`` in m^-3 s^-1, or ``None`` for none.
        bohm_margin: See :data:`DEFAULT_BOHM_MARGIN`.
        secondary_emission_energy: See :data:`SECONDARY_EMISSION_ENERGY`.
        initial_model: Which doc 03 §2 profile seeds Newton. doc 03 §2.2 offers the
            matrix sheath "as a numerical initialisation"; Child-Langmuir is closer at
            high bias. The converged answer must not depend on it, and a test checks that.
        edge_tolerance: ``delta`` in the sheath-edge definition.
        continuation_factor: Ratio between successive bias steps. See :func:`_continue_to`
            for why a continuation is needed at all.
        relaxation: Newton damping. One is undamped; :func:`_continue_to`'s adaptive step
            halving does the job damping would, and does it only where it is needed.
        rtol: Relative tolerance on the Newton increment.
        max_iterations: Newton iteration cap, per continuation step.
    """

    h_l: float = DEFAULT_EDGE_TO_CENTRE_RATIO
    gamma_i: float = GAMMA_I_COLD_ION
    cells_per_debye: float = MIN_CELLS_PER_DEBYE
    stretch: float = DEFAULT_MESH_STRETCH
    domain_sheaths: int = DEFAULT_DOMAIN_SHEATHS
    mean_free_path: Quantity | None = None
    ion_source: SourceTerm | None = None
    bohm_margin: float = DEFAULT_BOHM_MARGIN
    secondary_emission_energy: Quantity = SECONDARY_EMISSION_ENERGY
    initial_model: SheathModel = SheathModel.CHILD_LANGMUIR
    edge_tolerance: float = SHEATH_EDGE_TOLERANCE
    continuation_factor: float = DEFAULT_CONTINUATION_FACTOR
    relaxation: float = DEFAULT_RELAXATION
    rtol: float = DEFAULT_NEWTON_RTOL
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    def __post_init__(self) -> None:
        if not 0.0 < self.h_l <= 1.0:
            raise ValueError(
                f"h_l is an edge-to-centre density ratio and must lie in (0, 1], got {self.h_l}"
            )
        if self.gamma_i < 0.0:
            raise ValueError(f"gamma_i cannot be negative, got {self.gamma_i}")
        if self.bohm_margin < 0.0:
            raise ValueError(
                f"bohm_margin cannot be negative, got {self.bohm_margin}. A negative "
                "margin injects ions subsonically, which over-determines the inflow "
                "boundary; see DEFAULT_BOHM_MARGIN."
            )
        if not 0.0 < self.edge_tolerance < 1.0:
            raise ValueError(
                f"edge_tolerance is a quasineutrality departure and must lie in (0, 1), "
                f"got {self.edge_tolerance}"
            )

    # ── configuration ───────────────────────────────────────────────────────────

    def fidelity(self) -> Fidelity:
        return Fidelity.L1

    def with_config(self, cfg: SolverConfig) -> FluidSheathSolver:
        """A new solver with the manifest's ``forward:`` block applied — doc 08 §6.

        Immutable rather than the protocol's mutating ``configure``; see the class
        docstring. Only keys the manifest actually sets are read, so a block written for
        another solver cannot silently reset this one's defaults.
        """
        changes: dict[str, object] = {}
        if "cells_per_debye" in cfg:
            changes["cells_per_debye"] = cfg.require_float("cells_per_debye")
        if "stretch" in cfg:
            changes["stretch"] = cfg.require_float("stretch")
        if "domain_sheaths" in cfg:
            changes["domain_sheaths"] = cfg.require_int("domain_sheaths")
        if "h_l" in cfg:
            changes["h_l"] = cfg.require_float("h_l")
        if "gamma_i" in cfg:
            changes["gamma_i"] = cfg.require_float("gamma_i")
        if "bohm_margin" in cfg:
            changes["bohm_margin"] = cfg.require_float("bohm_margin")
        return replace(self, **changes)  # type: ignore[arg-type]

    def metadata(self) -> SolverMetadata:
        """Doc 00 C2: every algorithm has a citation, and there is no permissive path."""
        return SolverMetadata(
            name="vpl.physics.fluid.sheath",
            version="0.1.0",
            description=(
                "One-dimensional fluid sheath: inertial drift-diffusion ions, Boltzmann "
                "electrons, Poisson, solved by Newton on the coupled system with "
                "Scharfetter-Gummel exponential fitting on a wall-graded mesh."
            ),
            citations=(
                Citation(
                    key="lieberman2005",
                    reference=(
                        "M. A. Lieberman and A. J. Lichtenberg, Principles of Plasma "
                        "Discharges and Materials Processing, 2nd ed., Wiley 2005, §6."
                    ),
                ),
                Citation(
                    key="scharfetter1969",
                    reference=(
                        "D. L. Scharfetter and H. K. Gummel, Large-signal analysis of a "
                        "silicon Read diode oscillator, IEEE Trans. Electron Devices 16, "
                        "64 (1969)."
                    ),
                    doi="10.1109/T-ED.1969.16566",
                ),
                Citation(
                    key="brooks1982",
                    reference=(
                        "A. N. Brooks and T. J. R. Hughes, Streamline "
                        "upwind/Petrov-Galerkin formulations, Comput. Methods Appl. "
                        "Mech. Eng. 32, 199 (1982) — §3.3, the equivalence of the "
                        "optimal upwind parameter with exponential fitting."
                    ),
                    doi="10.1016/0045-7825(82)90071-8",
                ),
                Citation(
                    key="hagstrum1954",
                    reference=(
                        "H. D. Hagstrum, Auger ejection of electrons from tungsten by "
                        "noble gas ions, Phys. Rev. 96, 325 (1954) — the secondary "
                        "emission energy of SECONDARY_EMISSION_ENERGY."
                    ),
                    doi="10.1103/PhysRev.96.325",
                ),
                Citation(
                    key="baratta2023dolfinx",
                    reference=(
                        "I. A. Baratta et al., DOLFINx: the next generation FEniCS "
                        "problem solving environment, Zenodo (2023)."
                    ),
                    doi="10.5281/zenodo.10447666",
                ),
            ),
        )

    def cost_estimate(self, cfg: SolverConfig) -> CostEstimate:
        """The doc 03 §1 estimate, labelled as one — doc 10 §3.2, gate G-1.4.

        Returns an ``ESTIMATED`` cost deliberately. doc 11 G-1.4 requires the measured
        figure to replace it, and :meth:`measure_cost` is how that measurement is taken;
        a schedule built from estimates after G-1.4 has passed is a schedule nobody
        re-derived, which is exactly what :class:`~vpl.core.protocols.forward.CostBasis`
        exists to make visible.
        """
        del cfg  # The estimate does not yet depend on the configuration.
        return CostEstimate.estimated(
            wall_clock=Q_(1.0, "s"),
            device=Device.CPU,
            source="doc 03 §1 — L1 tabled at ~1 s per solve",
        )

    def measure_cost(self, params: PlasmaParams, *, repeats: int = 5) -> CostEstimate:
        """Time one solve — the measurement doc 11 G-1.4 requires.

        The first solve is discarded. FEniCSx compiles each variational form to C on
        first use, so including it would measure a one-off compilation that a campaign of
        5 500 cases (doc 10 §3.1) pays exactly once. The median of the rest is reported
        rather than the mean, because a single scheduler preemption should not become the
        number in the table.
        """
        if repeats < 1:
            raise ValueError(f"repeats must be at least one, got {repeats}")

        grid = self.grid(params)
        self.solve(params, grid=grid)

        durations: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            self.solve(params, grid=grid)
            durations.append(time.perf_counter() - started)

        return CostEstimate.measured(
            wall_clock=Q_(float(np.median(durations)), "s"),
            device=Device.CPU,
            source=(
                f"median of {repeats} solves after one warm-up, "
                f"{grid.n_cells} cells, {self.cells_per_debye:g} cells per Debye length"
            ),
        )

    # ── the mesh ────────────────────────────────────────────────────────────────

    def scaling(self, params: PlasmaParams) -> SheathScaling:
        return SheathScaling.from_params(params, h_l=self.h_l, gamma_i=self.gamma_i)

    def sheath_estimate(self, params: PlasmaParams) -> Quantity:
        """The thickness the mesh is sized on, from L0.

        The ADR-007 self-consistent Child-Langmuir value, or the matrix sheath when that
        is what seeds Newton. Only the mesh and the initial guess depend on it; the
        converged solution does not.
        """
        if self.initial_model is SheathModel.MATRIX:
            return matrix_sheath_thickness(params)
        return child_langmuir_thickness(params, h_l=self.h_l)

    def grid(self, params: PlasmaParams) -> SpatialGrid:
        """The doc 03 §3.4 graded mesh for this operating point."""
        estimate = float(magnitude_in(self.sheath_estimate(params), "m"))
        if estimate <= 0.0:
            raise ValueError(
                "cannot size a mesh at zero bias: the sheath thickness vanishes. Pass an "
                "explicit grid= if you want the (flat, quasineutral) zero-bias state."
            )
        span = max(float(magnitude_in(DEFAULT_DOMAIN_LENGTH, "m")), self.domain_sheaths * estimate)
        return graded_sheath_grid(
            debye_length=self.scaling(params).debye_length,
            sheath_thickness=Q_(estimate, "m"),
            domain_length=Q_(span, "m"),
            cells_per_debye=self.cells_per_debye,
            stretch=self.stretch,
        )

    # ── the solve ───────────────────────────────────────────────────────────────

    def solve(
        self,
        params: PlasmaParams,
        *,
        grid: SpatialGrid | None = None,
        time_grid: TimeGrid | None = None,
    ) -> PlasmaState:
        """The L1 state. See :meth:`solve_detailed` for the verification quantities."""
        return self.solve_detailed(params, grid=grid, time_grid=time_grid).state

    def solve_detailed(
        self,
        params: PlasmaParams,
        *,
        grid: SpatialGrid | None = None,
        time_grid: TimeGrid | None = None,
    ) -> SheathSolution:
        """Solve the coupled system and assemble everything the gates need.

        Args:
            params: Control parameters.
            grid: Where to solve. Defaults to :meth:`grid`.
            time_grid: Carried through to the state. L1 as implemented is steady; a time
                grid would imply a time dependence this level does not model, so the
                default is ``None`` and a caller supplying one is responsible for what it
                means.

        Returns:
            The state, the scaling, the Newton report, the measured sheath thickness and
            the wall current density.

        Raises:
            RuntimeError: If Newton does not converge.
        """
        mesh_grid = self.grid(params) if grid is None else grid
        scaling = self.scaling(params)
        fields = solve_system(self, params, mesh_grid, scaling)

        nu, w, phi = fields.density, fields.velocity, fields.potential
        n_i = nu.nodal * scaling.density
        u_i = np.abs(w.nodal) * scaling.speed
        potential = phi.nodal * scaling.potential
        n_e = scaling.density * np.exp(phi.nodal)
        n_se = _secondary_density(self, params, scaling, phi.nodal, injected=self._injected_flux())

        def _field(name: str, values: FloatArray, units: str) -> ScalarField:
            return ScalarField(
                name=name, values=values, units=units, grid=mesh_grid, time=time_grid
            )

        state = PlasmaState(
            params=params,
            grid=mesh_grid,
            time=time_grid,
            fields={
                "n_e": _field("n_e", n_e, "m**-3"),
                "n_i": _field("n_i", n_i, "m**-3"),
                "n_se": _field("n_se", n_se, "m**-3"),
                "Phi": _field("Phi", potential, "V"),
                # Isothermal, both species: doc 03 §3.1 makes the electron energy
                # equation optional and this implementation does not solve it. Stated as
                # a field rather than left implicit.
                "T_e": _field("T_e", np.full(mesh_grid.n_points, scaling.potential), "eV"),
                "T_i": _field(
                    "T_i",
                    np.full(mesh_grid.n_points, params.T_i_J / _E_C),
                    "eV",
                ),
                "u_i": _field("u_i", u_i, "m/s"),
            },
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        return SheathSolution(
            state=state,
            scaling=scaling,
            newton=fields.newton,
            sheath_thickness=_sheath_edge(mesh_grid, n_i, n_e, self.edge_tolerance),
            wall_current_density=Q_(_E_C * float(n_i[0] * u_i[0]), "A/m**2"),
            continuation_steps=fields.continuation_steps,
        )

    # ── the quantity of interest ────────────────────────────────────────────────

    def flux(self, state: PlasmaState, z: float = 0.0) -> IonEnergyFlux:
        """``Gamma_E`` at ``z`` — doc 03 §6, from the fluid moments.

        doc 03 §6 obtains ``f_i`` at L1 by reconstructing a drifting Maxwellian from
        ``(n_i, u_i, T_i)``, so the energy flux is the enthalpy flux of that distribution:

            ``Gamma_E = n u (1/2 m u^2 + 5/2 k T_i)``

        The ``5/2 k T_i`` is the thermal enthalpy the drifting Maxwellian carries. At RP-1
        it is 0.125 eV against 250 eV of directed energy, and it is kept rather than
        dropped because doc 03 §6 requires the same functional at every level and L2 will
        contain it whether or not L1 does.

        **This is the approximation doc 03 §6 calls "a genuine weakness".** It cannot
        represent the bimodal RF IEDF or the collisional low-energy tail. Benchmark B-03
        measures how wrong it is; this docstring exists so that nobody has to find out.

        Args:
            state: A state this solver produced.
            z: Where to evaluate, in metres. Defaults to the wall. A bare float rather
                than a ``Quantity`` because doc 08 §4 declares it that way and doc 08 §5
                puts raw SI magnitudes in the sampler's inner loop.
        """
        grid = state.grid
        n_i = float(np.interp(z, grid.z_m, state.field("n_i").values))
        u_i = float(np.interp(z, grid.z_m, state.field("u_i").values))
        mass = state.params.species.mass_kg
        thermal = 2.5 * state.params.T_i_J

        particle_flux = n_i * u_i
        energy_flux = particle_flux * (0.5 * mass * u_i**2 + thermal)

        return IonEnergyFlux(
            position=Q_(z, "m"),
            species=state.params.species,
            energy_flux_toward_wall_watt_per_m2=np.asarray(energy_flux, dtype=np.float64),
            particle_flux_toward_wall_per_m2_s=np.asarray(particle_flux, dtype=np.float64),
            fidelity=Fidelity.L1,
            time=None,
        )

    def _injected_flux(self) -> float:
        """``|nu w|`` at the bulk boundary — see :data:`DEFAULT_BOHM_MARGIN`."""
        return 1.0 + self.bohm_margin


# ── the assembled state ─────────────────────────────────────────────────────────


def _secondary_density(
    solver: FluidSheathSolver,
    params: PlasmaParams,
    scaling: SheathScaling,
    phi: FloatArray,
    *,
    injected: float,
) -> FloatArray:
    """The NumPy twin of :func:`_ufl_secondary_density`, for the emitted state."""
    if params.gamma_se == 0.0:
        return np.zeros_like(phi)

    emission = float(magnitude_in(solver.secondary_emission_energy, "J"))
    drop = np.maximum(phi + scaling.normalised_bias(params), 0.0)
    velocity = np.sqrt(2.0 * (emission + _E_C * scaling.potential * drop) / _M_E_KG)
    return np.asarray(params.gamma_se * injected * scaling.flux / velocity, dtype=np.float64)


def _sheath_edge(grid: SpatialGrid, n_i: FloatArray, n_e: FloatArray, tolerance: float) -> Quantity:
    """``s`` from doc 03 §2.1's ``(n_i - n_e) / n_i = delta``, interpolated between nodes.

    doc 03 §2.1: "**The sheath edge is a definition, not a physical surface.** Different
    definitions shift ``z_s`` by a fraction of a Debye length and therefore shift the
    flux." ``delta`` is :data:`SHEATH_EDGE_TOLERANCE`, registered and swept.

    Located from the bulk inward: the departure from quasineutrality rises monotonically
    toward the wall, so the *outermost* crossing is the edge. Searching from the wall
    would find the same point on a converged solution and a spurious one on any solution
    with a ripple in the presheath — which is precisely the case one wants to detect.
    """
    departure = (n_i - n_e) / n_i
    crossings = np.flatnonzero(departure >= tolerance)
    if crossings.size == 0:
        raise ValueError(
            f"no point in the domain departs from quasineutrality by delta = {tolerance}; "
            "there is no sheath to measure. At zero or very small bias that is the "
            "correct answer and the thickness is undefined rather than zero."
        )

    outermost = int(crossings[-1])
    if outermost >= grid.n_points - 1:
        raise ValueError(
            "the sheath reaches the bulk boundary: the domain is too short for the "
            "quasineutral boundary condition doc 03 §3.3 applies there. doc 01 R-SPAT-3 "
            "wants at least ten sheath thicknesses."
        )

    # Linear interpolation onto the crossing, so the answer is not quantised by the mesh
    # — which would make the V-04 mesh-independence study measure the mesh rather than
    # the solution.
    lower, upper = departure[outermost], departure[outermost + 1]
    span = grid.z_m[outermost + 1] - grid.z_m[outermost]
    fraction = 0.0 if lower == upper else (lower - tolerance) / (lower - upper)
    return Q_(float(grid.z_m[outermost] + fraction * span), "m")
