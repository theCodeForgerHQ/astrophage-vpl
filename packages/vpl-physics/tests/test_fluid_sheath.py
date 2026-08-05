"""L1 — the coupled fluid sheath. V-03, V-04, and the G-1.4 throughput measurement.

**V-03** (doc 03 §7, doc 11 G-1.2): ``s`` and ``J_i`` within 5 % of L0 in the
collisionless high-bias limit. ADR-007 decides *which* L0 ``s`` that is, and the decision
is load-bearing: comparing against doc 01 §2.2's bulk-density value instead of the
self-consistent sheath-edge one would make a correct solver look 28 % wrong.

**``J_i`` passes. ``s`` does not, and the reason is in doc 03 §2.3 rather than in the
solver.** The Child-Langmuir thickness is derived by neglecting the electron space charge
entirely, and the resulting error decays only as ``(T_e / V_w)^{1/2}``. Measured here:
``s / s_CL`` is 1.50 at RP-1's 250 V and 1.21 at doc 01 R-ENV-4's 1000 V, and an
independent quadrature of doc 03 §3.1's own equations — written below, sharing no code
with the solver — reproduces both to better than 1 %. The 5 % criterion is first met at
``V_w / T_e ~ 5000``, i.e. **``V_w`` of roughly 15 kV**, fifteen times the top of the
documented operating envelope.

So the finding is not that L1 is 50 % wrong. It is that **doc 03 §7 V-03's 5 % criterion
on ``s`` is unreachable anywhere in doc 01 R-ENV-4's bias range**, and that the L1
solution is nonetheless verified — against an independent solution of the equations
doc 03 §3.1 actually specifies, to 0.3 %. This is ADR-007's situation exactly: two
Baseline statements that cannot both hold, resolved on the record rather than by tuning.

**V-04** (doc 03 §7): ``Gamma_E`` changes by less than 1 % on halving ``dz``.

**G-1.4** (doc 11 §2): the *measured* L1 throughput, replacing an estimate. doc 03 §1
tables L1 at "~1 s" per solve. The number this test prints is the measurement, and a
number that disagrees with the document is a result, not a failure.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

pytest.importorskip("dolfinx")

from vpl.core.constants import ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.protocols.config import SolverConfig
from vpl.core.protocols.forward import CostBasis, Device
from vpl.core.state import (
    REQUIRED_FIELDS,
    Fidelity,
    PlasmaParams,
    Species,
)
from vpl.core.units import Q_, Quantity, magnitude_in
from vpl.physics.analytic.sheath import (
    DEFAULT_EDGE_TO_CENTRE_RATIO,
    SIGMA_CX_ARGON,
    AnalyticSheathSolver,
    SheathModel,
    bohm_speed,
    child_langmuir_current_density,
    child_langmuir_thickness,
    cx_mean_free_path,
    ion_energy_flux,
)
from vpl.physics.fluid import (
    MIN_CELLS_PER_DEBYE,
    FluidSheathSolver,
    SheathSolution,
)
from vpl.physics.fluid.sheath import DEFAULT_BOHM_MARGIN

_E_C = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_EPS0 = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

ARGON = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)


def _rp1(**overrides: object) -> PlasmaParams:
    """The reference operating point RP-1 of doc 01 §2.1."""
    defaults: dict[str, object] = {
        "species": ARGON,
        "n_0": Q_(1e17, "m**-3"),
        "T_e": Q_(3.0, "eV"),
        "T_i": Q_(0.05, "eV"),
        "T_g": Q_(300.0, "K"),
        "pressure": Q_(5.0, "mTorr"),
        "bias": Q_(-250.0, "V"),
        "gamma_se": 0.10,
        "kappa": 1.0,
    }
    return PlasmaParams(**{**defaults, **overrides})  # type: ignore[arg-type]


#: doc 01 R-ENV-4 tops the bias envelope at 1000 V, which is the highest bias at which
#: V-03 can be asked anything about a *documented* operating point.
ENVELOPE_TOP_BIAS = Q_(-1000.0, "V")


def _m(value: Quantity, units: str) -> float:
    return float(magnitude_in(value, units))


def _relative(actual: float, reference: float) -> float:
    return abs(actual - reference) / abs(reference)


# Module-scoped so the FEniCSx form compilation and the solves are paid once each. Plain
# functions rather than class-scoped methods: pytest 9 deprecates the latter, and the
# FEniCSx environment runs pytest 9 while the workspace runs pytest 8.


@pytest.fixture(scope="module")
def rp1_solution() -> SheathSolution:
    """RP-1, collisionless — the regime doc 03 §7 V-03 is stated in."""
    return FluidSheathSolver(mean_free_path=None).solve_detailed(_rp1())


@pytest.fixture(scope="module")
def envelope_top_solution() -> SheathSolution:
    """doc 01 R-ENV-4's 1000 V, collisionless."""
    return FluidSheathSolver(mean_free_path=None).solve_detailed(_rp1(bias=ENVELOPE_TOP_BIAS))


# ── an independent solution of the same equations ───────────────────────────────


def _exact_sheath_thickness(params: PlasmaParams, *, edge_tolerance: float) -> float:
    """Quadrature solution of doc 03 §3.1 in the collisionless, cold-ion limit.

    Shares no code with the solver, on purpose. With no source and no collisions the
    three-field system collapses: continuity gives ``nu |w| = w_0``, momentum integrates
    to ``w^2 = w_0^2 - 2 phi``, and Poisson has a first integral

        ``lambda_D^2 phi'^2 / 2 = w_0 (sqrt(w_0^2 - 2 phi) - w_0) + exp(phi) - 1``

    so the thickness is a single quadrature, ``s = int dphi / |phi'|`` from the wall
    potential out to the doc 03 §2.1 sheath edge. This is the "method of exact solutions"
    half of doc 07 §2.1's code verification, and it is what tells V-03's discrepancy apart
    from a solver defect.
    """
    density = DEFAULT_EDGE_TO_CENTRE_RATIO * params.n_0_per_m3
    potential = params.T_e_J / _E_C
    debye = float(np.sqrt(_EPS0 * potential / (density * _E_C)))
    injected = 1.0 + DEFAULT_BOHM_MARGIN

    def slope(phi: float) -> float:
        value = (
            2.0
            * (injected * (np.sqrt(injected**2 - 2.0 * phi) - injected) + np.exp(phi) - 1.0)
            / debye**2
        )
        return float(np.sqrt(max(value, 0.0)))

    def departure(x: float) -> float:
        """``(n_i - n_e) / n_i - delta`` at ``phi = -x``."""
        return 1.0 - np.exp(-x) * np.sqrt(injected**2 + 2.0 * x) / injected - edge_tolerance

    edge = -brentq(departure, 1e-9, 20.0)
    wall = -_m(params.bias_magnitude, "V") / potential
    return float(quad(lambda phi: 1.0 / slope(phi), wall, edge, limit=400)[0])


class TestAgreementWithAnIndependentSolution:
    """The verification V-03 was meant to be — against the equations, not a limit of them."""

    @pytest.mark.physics
    @pytest.mark.slow
    def test_rp1_thickness_matches_the_quadrature(self, rp1_solution: SheathSolution) -> None:
        exact = _exact_sheath_thickness(_rp1(), edge_tolerance=0.01)
        measured = _m(rp1_solution.sheath_thickness, "m")
        error = _relative(measured, exact)
        print(
            f"\nL1 vs quadrature at 250 V: {measured * 1e3:.5f} mm against "
            f"{exact * 1e3:.5f} mm -> {100.0 * error:.3f} %"
        )
        assert error < 0.02

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_agreement_improves_on_refinement(self) -> None:
        """First-order upwinding, so the gap should shrink when ``dz`` does.

        doc 03 §3.4's exponential fitting reduces to pure upwinding in the collisionless
        limit, which is first-order accurate. Confirming the direction here is what
        distinguishes "converging to the right answer slowly" from "converging to a
        slightly wrong answer".
        """
        params = _rp1()
        exact = _exact_sheath_thickness(params, edge_tolerance=0.01)
        errors = [
            _relative(
                _m(
                    FluidSheathSolver(mean_free_path=None, cells_per_debye=cells)
                    .solve_detailed(params)
                    .sheath_thickness,
                    "m",
                ),
                exact,
            )
            for cells in (MIN_CELLS_PER_DEBYE, 4.0 * MIN_CELLS_PER_DEBYE)
        ]
        print(f"\nL1 vs quadrature on refinement: {errors[0]:.5f} -> {errors[1]:.5f}")
        assert errors[1] < errors[0]


# ── V-03 ────────────────────────────────────────────────────────────────────────


class TestV03ChildLangmuirRecovery:
    """doc 03 §7 V-03 / doc 11 G-1.2, in the collisionless high-bias limit."""

    @pytest.mark.physics
    @pytest.mark.slow
    def test_wall_current_density_is_within_five_percent(
        self, envelope_top_solution: SheathSolution
    ) -> None:
        """The half of V-03 that passes.

        ADR-007 establishes that the Child-Langmuir current evaluated at the
        self-consistent thickness *is* the Bohm flux, to one part in ``1e10``. L1 injects
        that flux at the bulk boundary and conserves it, so the agreement checks that the
        discretisation does not leak flux across the sheath rather than discovering
        anything. It is still the stated criterion and still worth failing on.
        """
        params = _rp1(bias=ENVELOPE_TOP_BIAS)
        reference = child_langmuir_current_density(
            params, child_langmuir_thickness(params, h_l=DEFAULT_EDGE_TO_CENTRE_RATIO)
        )
        measured = envelope_top_solution.wall_current_density
        error = _relative(_m(measured, "A/m**2"), _m(reference, "A/m**2"))
        print(
            f"\nV-03 J_i at 1000 V: L1 {_m(measured, 'A/m**2'):.5g} A/m^2 vs L0 "
            f"{_m(reference, 'A/m**2'):.5g} A/m^2 -> {100.0 * error:.3f} %"
        )
        assert error < 0.05

    @pytest.mark.physics
    @pytest.mark.slow
    def test_sheath_thickness_exceeds_child_langmuir_by_a_quantified_amount(
        self, rp1_solution: SheathSolution, envelope_top_solution: SheathSolution
    ) -> None:
        """**The half of V-03 that does not pass**, recorded with its evidence.

        The gap is not a solver defect: the quadrature of
        :func:`_exact_sheath_thickness`, which solves the same equations by a completely
        different route, reproduces both the solver's answer and the gap. It is the
        Child-Langmuir *approximation* that is off, because it drops the electron space
        charge and the correction decays only as ``(T_e / V_w)^{1/2}``.
        """
        print("\nV-03 s against Child-Langmuir:")
        for params, solution in (
            (_rp1(), rp1_solution),
            (_rp1(bias=ENVELOPE_TOP_BIAS), envelope_top_solution),
        ):
            child = _m(child_langmuir_thickness(params, h_l=DEFAULT_EDGE_TO_CENTRE_RATIO), "m")
            exact = _exact_sheath_thickness(params, edge_tolerance=0.01)
            measured = _m(solution.sheath_thickness, "m")
            print(
                f"  V_w = {_m(params.bias_magnitude, 'V'):7.1f} V  "
                f"(V_w/T_e = {solution.scaling.normalised_bias(params):7.1f})  "
                f"s_L1/s_CL = {measured / child:.4f}   "
                f"s_quadrature/s_CL = {exact / child:.4f}"
            )
            # The solver agrees with the quadrature; both disagree with Child-Langmuir.
            assert _relative(measured, exact) < 0.02
            assert measured / child > 1.05

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_gap_closes_as_the_bias_rises(
        self, rp1_solution: SheathSolution, envelope_top_solution: SheathSolution
    ) -> None:
        """Child-Langmuir *is* recovered — far beyond the documented envelope.

        doc 03 §2.3 asks for "recovery of this scaling [...] in the collisionless,
        high-bias limit". It is recovered. The quadrature gives ``s / s_CL`` of 1.891,
        1.502, 1.214, 1.093, 1.041, 1.019, 1.008 and 1.004 at ``V_w / T_e`` of 33, 83,
        333, 1333, 5333, 21333, 85333 and 341333, with the local exponent of ``s(V_w)``
        running 0.499, 0.596, 0.674, 0.715, 0.734, 0.743, 0.747 toward Child-Langmuir's
        3/4. The 5 % criterion is first met near ``V_w / T_e = 5000``, which at
        ``T_e = 3 eV`` is 15 kV — fifteen times doc 01 R-ENV-4's ceiling.
        """
        rp1 = _m(rp1_solution.sheath_thickness, "m") / _m(
            child_langmuir_thickness(_rp1(), h_l=DEFAULT_EDGE_TO_CENTRE_RATIO), "m"
        )
        top_params = _rp1(bias=ENVELOPE_TOP_BIAS)
        top = _m(envelope_top_solution.sheath_thickness, "m") / _m(
            child_langmuir_thickness(top_params, h_l=DEFAULT_EDGE_TO_CENTRE_RATIO), "m"
        )
        print(f"\nV-03 gap closing: {rp1:.4f} at 250 V -> {top:.4f} at 1000 V")
        assert top < rp1
        assert top > 1.05, (
            "the 5 % criterion is not met at the top of doc 01 R-ENV-4; if this now "
            "passes, something has changed and the module docstring needs re-measuring"
        )

    @pytest.mark.physics
    def test_the_doc_01_convention_would_have_moved_the_target_by_28_percent(self) -> None:
        """ADR-007, kept on the record as an assertion rather than as prose.

        doc 01 §2.2 evaluates ``lambda_D`` at the bulk density, giving an ``s`` smaller by
        ``sqrt(h_l)``. A future reader who switches the V-03 comparison back to the doc 01
        convention should meet a failing test with ADR-007's name on it rather than an
        unexplained 28 % discrepancy.
        """
        params = _rp1(bias=ENVELOPE_TOP_BIAS)
        bulk = _m(child_langmuir_thickness(params), "m")
        edge = _m(child_langmuir_thickness(params, h_l=DEFAULT_EDGE_TO_CENTRE_RATIO), "m")
        assert edge / bulk == pytest.approx(1.0 / np.sqrt(DEFAULT_EDGE_TO_CENTRE_RATIO), rel=1e-9)
        assert _relative(edge, bulk) == pytest.approx(0.2804, abs=1e-3)


# ── V-04 ────────────────────────────────────────────────────────────────────────


class TestV04MeshIndependence:
    """doc 03 §7 V-04: ``Gamma_E`` changes < 1 % on halving ``dz``."""

    @pytest.mark.physics
    @pytest.mark.slow
    def test_energy_flux_is_mesh_independent_to_one_percent(self) -> None:
        params = _rp1()
        fluxes = []
        for cells in (MIN_CELLS_PER_DEBYE, 2.0 * MIN_CELLS_PER_DEBYE):
            solver = FluidSheathSolver(mean_free_path=None, cells_per_debye=cells)
            fluxes.append(
                float(solver.flux(solver.solve(params)).energy_flux_toward_wall_watt_per_m2)
            )
        change = _relative(fluxes[1], fluxes[0])
        print(
            f"\nV-04 Gamma_E: {fluxes[0]:.6g} -> {fluxes[1]:.6g} W/m^2 on halving dz "
            f"-> {100.0 * change:.3f} %"
        )
        assert change < 0.01

    @pytest.mark.physics
    @pytest.mark.slow
    def test_sheath_thickness_is_mesh_independent_too(self) -> None:
        """The quantity that is actually mesh-sensitive, reported alongside the gate.

        ``Gamma_E`` is dominated by the injected flux and the imposed wall potential, so
        it would pass V-04 even on a mesh too coarse to resolve the sheath at all. The
        edge position is where the discretisation error lives, so it is where a mesh study
        earns its keep.
        """
        params = _rp1()
        thicknesses = [
            _m(
                FluidSheathSolver(mean_free_path=None, cells_per_debye=cells)
                .solve_detailed(params)
                .sheath_thickness,
                "m",
            )
            for cells in (MIN_CELLS_PER_DEBYE, 2.0 * MIN_CELLS_PER_DEBYE)
        ]
        change = _relative(thicknesses[1], thicknesses[0])
        print(f"\nV-04 sheath thickness change on halving dz: {100.0 * change:.3f} %")
        assert change < 0.01


# ── the state contract ──────────────────────────────────────────────────────────


class TestPlasmaStateContract:
    def test_carries_every_required_field(self, rp1_solution: SheathSolution) -> None:
        assert set(REQUIRED_FIELDS) <= set(rp1_solution.state.field_names)

    def test_is_labelled_l1(self, rp1_solution: SheathSolution) -> None:
        assert rp1_solution.state.fidelity is Fidelity.L1

    def test_carries_no_ion_distribution(self, rp1_solution: SheathSolution) -> None:
        """doc 03 §6: L1 reconstructs ``f_i`` as a drifting Maxwellian, and that is "a
        genuine weakness [...] treated as one". Recording the absence is more honest than
        manufacturing a distribution the level never solved for."""
        assert rp1_solution.state.ion_distribution is None
        assert not Fidelity.L1.resolves_distribution

    def test_is_steady(self, rp1_solution: SheathSolution) -> None:
        assert rp1_solution.state.is_steady

    def test_solver_reports_l1(self) -> None:
        assert FluidSheathSolver().fidelity() is Fidelity.L1

    def test_metadata_carries_citations(self) -> None:
        """doc 00 C2: every algorithm must have one."""
        metadata = FluidSheathSolver().metadata()
        keys = {citation.key for citation in metadata.citations}
        assert any("scharfetter" in key.lower() for key in keys)

    def test_configuration_returns_a_new_solver(self) -> None:
        """doc 08 §4 declares a mutating ``configure``; this codebase does not mutate."""
        original = FluidSheathSolver()
        configured = original.with_config(SolverConfig(values={"cells_per_debye": 8.0}))
        assert configured.cells_per_debye == 8.0
        assert original.cells_per_debye == MIN_CELLS_PER_DEBYE


class TestSheathPhysics:
    @pytest.mark.physics
    def test_the_potential_falls_monotonically_to_the_wall(
        self, rp1_solution: SheathSolution
    ) -> None:
        phi = rp1_solution.state.field("Phi").values
        assert np.all(np.diff(phi) >= -1e-6 * np.abs(phi).max())
        assert phi[0] == pytest.approx(-250.0, rel=1e-9)
        assert phi[-1] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.physics
    def test_ions_accelerate_toward_the_wall(self, rp1_solution: SheathSolution) -> None:
        u_i = rp1_solution.state.field("u_i").values
        assert np.all(np.diff(u_i) <= 1e-6 * u_i.max())
        assert u_i.min() > 0.0

    @pytest.mark.physics
    def test_the_bohm_criterion_holds_at_the_bulk_boundary(
        self, rp1_solution: SheathSolution
    ) -> None:
        """doc 03 §7 V-09: ``u_s / c_s = 1.0 +/- 0.05``.

        Imposed rather than discovered — this checks the boundary condition was applied,
        and that the :data:`DEFAULT_BOHM_MARGIN` regularisation is the size it claims.
        """
        ratio = rp1_solution.state.field("u_i").values[-1] / _m(bohm_speed(_rp1()), "m/s")
        assert ratio == pytest.approx(1.0 + DEFAULT_BOHM_MARGIN, rel=1e-6)

    @pytest.mark.physics
    def test_ion_flux_reaches_the_wall_undiminished(self, rp1_solution: SheathSolution) -> None:
        """Continuity with no source. A scheme that leaked flux would fail V-03 quietly.

        The quantity the discretisation conserves exactly is the *fitted* flux
        ``n_i u_i - D n_i'``, not its advective part alone, so the advective flux dips by
        a few tenths of a percent through the steepest part of the sheath where ``n_i'``
        is largest. That diffusive term is ``O(h)`` and is exactly the artificial
        diffusion doc 03 §3.4 bought monotonicity with. What has to hold to a much tighter
        tolerance is the end-to-end statement: what is injected at the bulk boundary
        arrives at the wall.
        """
        state = rp1_solution.state
        flux = state.field("n_i").values * state.field("u_i").values
        assert _relative(float(flux[0]), float(flux[-1])) < 1e-3
        assert float(flux.std() / flux.mean()) < 1e-2

    @pytest.mark.physics
    @pytest.mark.slow
    def test_energy_flux_converges_upward_onto_l0(self) -> None:
        """``Gamma_E`` against doc 03 §2.3's 6.6 kW/m^2, and where the difference is from.

        Two small effects with opposite signs, both stated:

        - L1 keeps the entry energy that
          :func:`vpl.physics.analytic.sheath.ion_energy_flux` drops — ``T_e / 2`` plus
          ``5 T_i / 2`` per ion, +0.65 % at RP-1 — and the Bohm margin adds +0.1 %.
        - The collisionless limit of exponential fitting is first-order upwinding, which
          is dissipative, so a coarse mesh loses ion energy across the sheath.

        At the default resolution the second wins slightly and ``Gamma_E`` lands 0.7 %
        *below* L0; refining recovers it monotonically. The convergence is the test — a
        fixed offset would mean the two effects are not what is happening.
        """
        params = _rp1()
        reference = _m(ion_energy_flux(params), "W/m**2")
        fluxes = [
            float(
                FluidSheathSolver(mean_free_path=None, cells_per_debye=cells)
                .flux(FluidSheathSolver(mean_free_path=None, cells_per_debye=cells).solve(params))
                .energy_flux_toward_wall_watt_per_m2
            )
            for cells in (
                MIN_CELLS_PER_DEBYE,
                2.0 * MIN_CELLS_PER_DEBYE,
                4.0 * MIN_CELLS_PER_DEBYE,
            )
        ]
        print(
            f"\nL0 Gamma_E = {reference:.5g} W/m^2; L1 on refinement = "
            + ", ".join(f"{value:.5g}" for value in fluxes)
        )
        assert fluxes[0] < fluxes[1] < fluxes[2]
        assert _relative(fluxes[-1], reference) < 0.01


class TestSecondaryElectronEmission:
    """doc 03 §3.3: "Secondary electron emission is not optional"."""

    @pytest.mark.physics
    def test_the_emitted_beam_is_present_and_densest_at_the_wall(
        self, rp1_solution: SheathSolution
    ) -> None:
        n_se = rp1_solution.state.field("n_se").values
        assert np.all(n_se >= 0.0)
        assert n_se[0] == pytest.approx(n_se.max())

    @pytest.mark.physics
    @pytest.mark.slow
    def test_a_zero_yield_removes_the_beam_entirely(self) -> None:
        state = FluidSheathSolver(mean_free_path=None).solve(_rp1(gamma_se=0.0))
        assert np.all(state.field("n_se").values == 0.0)

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_effect_on_the_sheath_is_measured_rather_than_assumed(self) -> None:
        """How much does doc 03 §3.3's "common and significant modelling error" cost here?

        At RP-1 the answer is "almost nothing", and saying so with a number is the point.
        The secondary beam crosses the sheath at ~1e7 m/s against the ions' 1e4, so its
        density is four orders below theirs and its contribution to the space charge is
        correspondingly small. That is a property of *this* operating point and not a
        licence to drop the term: doc 03 §8 A7 notes ``gamma_se`` rises at high bias, and
        the beam density scales linearly with it.
        """
        solver = FluidSheathSolver(mean_free_path=None)
        with_emission = solver.solve_detailed(_rp1(gamma_se=0.10))
        without = solver.solve_detailed(_rp1(gamma_se=0.0))
        shift = _relative(
            _m(with_emission.sheath_thickness, "m"), _m(without.sheath_thickness, "m")
        )
        peak = with_emission.state.field("n_se").values.max() / _rp1().n_0_per_m3
        print(
            f"\nSEE at gamma_se = 0.1: sheath thickness shifts {100.0 * shift:.4f} %, "
            f"peak n_se / n_0 = {peak:.3g}"
        )
        assert peak > 0.0
        assert shift < 0.01


# ── G-1.4 ───────────────────────────────────────────────────────────────────────


class TestG14Throughput:
    """doc 11 G-1.4: **measured** L1 throughput recorded; doc 03 §4.4 estimates corrected."""

    @pytest.mark.slow
    def test_measured_wall_clock_per_solve_is_recorded(self, rp1_solution: SheathSolution) -> None:
        cost = FluidSheathSolver(mean_free_path=None).measure_cost(_rp1(), repeats=5)
        print(
            f"\nG-1.4 measured L1 cost: {cost.wall_clock_s * 1e3:.1f} ms per solve on "
            f"{cost.device.value}\n  {cost.source}\n  "
            f"{rp1_solution.continuation_steps} continuation steps, "
            f"{rp1_solution.newton.iterations} Newton iterations in total"
        )
        assert cost.basis is CostBasis.MEASURED
        assert cost.device is Device.CPU
        # A ceiling, not a claim. doc 03 §1 tables L1 at ~1 s; three orders above that
        # would be a defect rather than a slow machine.
        assert cost.wall_clock_s < 60.0

    def test_the_declared_estimate_is_labelled_as_one(self) -> None:
        """doc 10 §3.2's basis field exists so an uncorrected estimate stays visible."""
        cost = FluidSheathSolver().cost_estimate(SolverConfig(values={}))
        assert cost.basis is CostBasis.ESTIMATED
        assert "doc 03" in cost.source


class TestCollisionalMode:
    @pytest.mark.physics
    @pytest.mark.slow
    def test_charge_exchange_drag_lowers_the_ion_impact_energy(self) -> None:
        """doc 03 §4.5: a CX event "is what makes <E_i> differ substantially from e V_w".

        The fluid model cannot produce the low-energy IEDF structure that does it — that
        is the doc 03 §6 weakness — but the momentum drag it does carry must at least move
        the mean in the right direction.
        """
        params = _rp1()
        free = FluidSheathSolver(mean_free_path=None)
        drag = FluidSheathSolver(mean_free_path=cx_mean_free_path(params, SIGMA_CX_ARGON))
        free_energy = float(free.flux(free.solve(params)).mean_impact_energy_joules)
        drag_energy = float(drag.flux(drag.solve(params)).mean_impact_energy_joules)
        print(
            f"\ncollisionless <E_i> = {free_energy / _E_C:.2f} eV, collisional = "
            f"{drag_energy / _E_C:.2f} eV, lambda_CX = "
            f"{_m(cx_mean_free_path(params, SIGMA_CX_ARGON), 'mm'):.3g} mm"
        )
        assert drag_energy < free_energy


class TestNewtonRobustness:
    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_answer_does_not_depend_on_the_initial_guess(self) -> None:
        """L1 is seeded from L0, and L0 is what V-03 checks it against.

        That is fine for a seed and fatal if the converged answer inherits it, so the two
        L0 profiles doc 03 §2 offers are compared. doc 03 §2.2 nominates the matrix sheath
        "as a numerical initialisation"; Child-Langmuir is closer at high bias.
        """
        params = _rp1()
        child = FluidSheathSolver(mean_free_path=None, initial_model=SheathModel.CHILD_LANGMUIR)
        matrix = FluidSheathSolver(mean_free_path=None, initial_model=SheathModel.MATRIX)
        grid = child.grid(params)
        first = child.solve(params, grid=grid).field("Phi").values
        second = matrix.solve(params, grid=grid).field("Phi").values
        assert first == pytest.approx(second, rel=1e-6, abs=1e-6)

    def test_l0_and_l1_agree_on_refusing_a_zero_bias_grid(self) -> None:
        """Both refuse rather than dividing by a vanished sheath."""
        params = _rp1(bias=Q_(0.0, "V"))
        with pytest.raises(ValueError, match="zero bias"):
            FluidSheathSolver().grid(params)
        with pytest.raises(ValueError, match="zero bias"):
            AnalyticSheathSolver().default_grid(params)

    def test_a_negative_bohm_margin_is_refused(self) -> None:
        with pytest.raises(ValueError, match="bohm_margin"):
            FluidSheathSolver(bohm_margin=-0.1)
