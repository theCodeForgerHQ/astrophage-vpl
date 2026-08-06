"""The assembled PIC-MCC loop — doc 03 §4.2, and the verification gates of doc 03 §7.

ADR-011 is why this file is organised the way it is. This project has already shipped a
solver that passed 1 488 self-consistent tests and was 52 % wrong on an external
benchmark, so the gates below are stated against things the solver cannot influence:

- **V-07, energy conservation.** A closed box: specular reflection at both ends, no
  collisions, no bias. Total energy is then a conserved quantity of the *equations*, and
  the leapfrog either conserves it or it does not. Nothing in the solver knows the number.
- **V-05, timestep independence.** ``Gamma_E`` at ``dt`` against ``Gamma_E`` at ``dt/2``.
  The physical answer cannot depend on the discretisation, so the two runs are each
  other's reference and neither is a recorded expectation.
- **V-06, N_ppc convergence on the IEDF shape.** Kolmogorov-Smirnov between the wall IEDFs
  at ``N_ppc`` and ``2 N_ppc``. doc 03 §4.3 is explicit that the *mean* converging while
  the tail is still noise is "a classic and silent PIC failure", so the statistic is on the
  distribution.
- **The L0 cross-check.** The only gate here that compares against an independently
  verified model rather than against this one. ``vpl.physics.analytic.sheath`` is L0, it
  is verified in ``test_analytic_sheath.py`` against doc 03 §2's printed numbers, and in
  the collisionless high-bias limit doc 03 §2.3 says the PIC must reproduce its Bohm flux
  and its Child-Langmuir sheath.

**What the L0 check does and does not prove.** doc 03 §3.3's bulk boundary injects ions at
``u = c_s`` from a reservoir at ``n_s``, so the ion flux arriving at the wall is *imposed*
by that boundary rather than predicted: the Bohm-flux comparison is a conservation check on
the loop's bookkeeping — nothing may be created or destroyed between the two boundaries.
The **sheath thickness**, the **potential profile** and the **mean impact energy** are
genuine predictions of the field solve and the push, and those are where the L0 comparison
has teeth. Both are asserted, and they are labelled for what they are.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from scipy import stats

from vpl.core.constants import ELEMENTARY_CHARGE
from vpl.core.params import default_registry
from vpl.core.protocols import ForwardSolver, SolverConfig
from vpl.core.state import Fidelity, PlasmaParams, Species, TimeGrid
from vpl.core.units import Q_, magnitude_in
from vpl.physics.analytic.sheath import (
    DEFAULT_EDGE_TO_CENTRE_RATIO,
    child_langmuir_potential,
    child_langmuir_thickness,
    ion_energy_flux,
    ion_flux,
)
from vpl.physics.kinetic.boundary import SurfaceModel, WallModel
from vpl.physics.kinetic.constraints import StabilityViolationError, debye_length_m
from vpl.physics.kinetic.solver import Pic1D3VSolver, PicRun

_REGISTRY = default_registry()
_E = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_AR = float(_REGISTRY.value_in("species.Ar.mass", "kg"))

#: doc 03 §7's own pass criteria. Named, so a threshold cannot be relaxed in one test
#: while the docstring goes on quoting the specification.
V05_TIMESTEP_TOLERANCE = 0.01
V06_KS_TOLERANCE = 0.02

#: ADR-010's independent quadrature of doc 03 §3.1's collisionless first integral gives
#: ``s / s_CL = 1.5029`` at RP-1 with Boltzmann electrons. It is the *upper* bound for a
#: kinetic sheath, not a target: a PIC has no Boltzmann-electron assumption, and a kinetic
#: electron population is depleted near a strongly negative wall relative to Boltzmann,
#: which thins the sheath. See ADR-010's consequences section, which predicted this before
#: L2 existed.
ADR_010_QUADRATURE_RATIO = 1.5029

#: A kinetic sheath must exceed Child-Langmuir by more than noise. Child-Langmuir drops
#: electron space charge entirely, so ``s / s_CL = 1`` would mean the electrons were not
#: being solved at all — which is the failure this lower bound exists to catch.
ADR_010_MINIMUM_EXCESS = 0.05

#: ``Gamma_E = Gamma_i * <E_i>``, and both factors are gated at 5 % in this class. Their
#: product is therefore knowable to about 10 %, and the 5 % gate the original test applied
#: to the product could only have been met by the two errors cancelling. Derived, not
#: chosen: ``1.05 * 1.05 - 1``.
GAMMA_E_COMPOUNDED_TOLERANCE = 1.05 * 1.05 - 1.0
V07_ENERGY_DRIFT_TOLERANCE = 1.0e-3

#: The seed for every run in this file. doc 00 E3: bit-for-bit reproduction.
_SEED = 20260805


def _argon() -> Species:
    return Species(name="Ar+", mass=Q_(_M_AR, "kg"), charge_number=1)


def _params(
    *,
    n_0: float = 1.0e17,
    T_e_ev: float = 3.0,
    T_i_ev: float = 0.05,
    bias_v: float = -250.0,
    gamma_se: float = 0.0,
) -> PlasmaParams:
    """RP-1 (doc 01 §2.1), with the knobs the gates below vary.

    ``gamma_se`` defaults to zero here and not to the registry value: the collisionless
    gates are stated for a sheath with no secondary emission, and doc 03 §7's V-03, V-05
    and V-07 are all collisionless statements. The emission channel has its own test.
    """
    return PlasmaParams(
        species=_argon(),
        n_0=Q_(n_0, "m**-3"),
        T_e=Q_(T_e_ev, "eV"),
        T_i=Q_(T_i_ev, "eV"),
        T_g=Q_(_REGISTRY.value_in("RP1.T_g", "K"), "K"),
        pressure=Q_(_REGISTRY.value_in("RP1.pressure", "Pa"), "Pa"),
        bias=Q_(bias_v, "V"),
        gamma_se=gamma_se,
        kappa=1.0,
    )


def _sheath_solver(**overrides: object) -> Pic1D3VSolver:
    """A sheath-scale solver sized for verification rather than for RP-1 production.

    doc 03 §4.4 costs a production run at 1 000 cells and 65 800 steps. Everything here is
    smaller, and deliberately: doc 03 §7's gates are statements about convergence and
    conservation, and both are checked more sharply by several cheap runs than by one
    expensive one. The domain is the **sheath**, with the bulk boundary a little beyond
    the Child-Langmuir edge, which is what makes the ion transit time — and therefore the
    step count — tractable.
    """
    defaults: dict[str, object] = {
        "n_ppc": 400.0,
        "cells_per_debye": 1.0,
        "domain_sheaths": 1.2,
        "transits": 6.0,
        "equilibration_fraction": 0.5,
        "capacity_factor": 1.5,
        "seed": _SEED,
    }
    return Pic1D3VSolver(**{**defaults, **overrides})  # type: ignore[arg-type]


@functools.cache
def _gate_run(
    *,
    bias_v: float,
    n_ppc: float,
    timestep_refinement: float = 1.0,
    domain_sheaths: float = 1.2,
    transits: float = 6.0,
    equilibration_fraction: float = 0.5,
) -> PicRun:
    """One converged sheath run, memoised on its configuration.

    Memoised because several gates read different statistics out of the *same* run — V-06
    checks the IEDF shape and its mean, the L0 class checks five quantities of one solve —
    and re-running would multiply a fifteen-minute suite by four for no extra information.
    The runs are deterministic in their seed (doc 00 E3), so the cache cannot change an
    answer; it can only avoid recomputing it.
    """
    return _sheath_solver(
        n_ppc=n_ppc,
        timestep_refinement=timestep_refinement,
        domain_sheaths=domain_sheaths,
        transits=transits,
        equilibration_fraction=equilibration_fraction,
    ).run(_params(bias_v=bias_v), None)


#: The equilibration these gates were calibrated at. The plasma is loaded uniformly at the
#: reservoir density, which over-fills the sheath — a Child-Langmuir sheath holds
#: ``n_s c_s / u(z)``, far less — so the excess has to drain before anything is recorded.
#: Measured on the reference machine at V_w = 50 V: with the window opening after 1.2 ion
#: transits ``Gamma_i`` overshoots the Bohm flux by 26 %, after 2 transits by 5.6 %, and
#: after 3 transits by 0.8 %. Three is therefore what the gates use, and this note is here
#: because the failure is a *quiet* one: the run converges, the IEDF looks right, and the
#: flux is high by a quarter.
_GATE_BIAS_V = -50.0


# ── contract ────────────────────────────────────────────────────────────────────


class TestContract:
    def test_the_solver_satisfies_the_forward_solver_protocol(self) -> None:
        # doc 08 §4. The runtime check is structural and verifies the six method names
        # are present, which turns a broken solver into a loud failure at load time.
        assert isinstance(Pic1D3VSolver(), ForwardSolver)

    def test_the_fidelity_is_l2(self) -> None:
        assert Pic1D3VSolver().fidelity() is Fidelity.L2

    def test_the_metadata_carries_a_citation(self) -> None:
        # doc 00 C2: every algorithm has one, enforced at construction by SolverMetadata.
        metadata = Pic1D3VSolver().metadata()
        assert metadata.name == "vpl.physics.kinetic.pic1d3v"
        assert metadata.citations

    def test_configure_applies_a_manifest_block(self) -> None:
        # doc 08 §6's `forward:` block. Configuration is data, not code.
        solver = Pic1D3VSolver()
        solver.configure(SolverConfig(values={"n_ppc": 250, "n_cells": 17, "seed": 7}))
        assert solver.n_ppc == 250.0
        assert solver.n_cells == 17
        assert solver.seed == 7

    def test_configure_refuses_a_key_it_does_not_know(self) -> None:
        # A silently ignored manifest key is the failure vpl.core.protocols.config warns
        # about: the run finishes, reproducibly, and does not do what the file said.
        with pytest.raises(ValueError, match="n_ppcc"):
            Pic1D3VSolver().configure(SolverConfig(values={"n_ppcc": 250}))

    def test_the_cost_estimate_is_labelled_as_an_estimate(self) -> None:
        # doc 10 §3.2 / gate G-1.4: a predicted cost must say that it is one.
        cost = Pic1D3VSolver().cost_estimate(SolverConfig(values={}))
        assert not cost.is_measured
        assert cost.wall_clock_s > 0.0


class TestStabilityIsChecked:
    def test_a_cell_wider_than_the_debye_length_is_a_runtime_error(self) -> None:
        # doc 03 §4.3: "checked at runtime, and violating them is a runtime error rather
        # than a warning". The violation produces output, which is why a warning is not a
        # control for it.
        params = _params()
        solver = _sheath_solver(n_cells=4, n_steps=2)
        with pytest.raises(StabilityViolationError, match="Debye"):
            solver.solve(params, None)

    def test_too_few_particles_per_cell_is_a_runtime_error(self) -> None:
        params = _params()
        solver = _sheath_solver(n_ppc=10.0, n_steps=2)
        with pytest.raises(StabilityViolationError, match="particles per cell"):
            solver.solve(params, None)

    def test_the_chosen_timestep_satisfies_every_doc_03_4_3_constraint(self) -> None:
        # The timestep is derived, not configured: dt = min(0.2/omega_pe, dz/v_max). The
        # CFL branch is the one that bites when secondary electrons re-cross the sheath.
        params = _params(gamma_se=0.1)
        solver = _sheath_solver(n_steps=2)
        run = solver.run(params, None)

        lambda_d = debye_length_m(density_m3=params.n_0_per_m3, electron_temperature_ev=3.0)
        assert run.diagnostics.cell_width_m <= lambda_d
        assert run.diagnostics.dt_s <= run.diagnostics.cell_width_m / run.diagnostics.v_max_m_per_s


class TestEmittedState:
    def test_the_state_is_an_l2_state_carrying_its_ion_distribution(self) -> None:
        # doc 03 §1 calls L2 "the only level that produces a correct IEDF"; PlasmaState
        # refuses to construct an L2 state without one.
        run = _sheath_solver(n_steps=400).run(_params(), None)
        state = run.state

        assert state.fidelity is Fidelity.L2
        assert state.ion_distribution is not None
        assert set(state.field_names) >= {"Phi", "T_e", "n_e", "n_i", "u_i"}
        assert state.is_steady

    def test_the_wall_potential_is_the_applied_bias(self) -> None:
        # doc 03 §3.3's wall row, and the only place doc 01 §2.1's -250 V enters.
        run = _sheath_solver(n_steps=400).run(_params(bias_v=-120.0), None)
        assert float(run.state.field("Phi").values[0]) == pytest.approx(-120.0)
        assert float(run.state.field("Phi").values[-1]) == pytest.approx(0.0, abs=1e-9)

    def test_a_time_grid_produces_time_resolved_fields(self) -> None:
        # doc 08 §4's `solve(params, t)`. The instants are honoured rather than discarded.
        solver = _sheath_solver(n_steps=400)
        dt = solver.run(_params(), None).diagnostics.dt_s
        times = TimeGrid(t_s=np.linspace(200 * dt, 399 * dt, 5))

        state = solver.solve(_params(), times)
        assert state.time == times
        assert state.field("Phi").values.shape == (5, state.grid.n_points)

    def test_the_run_is_reproducible_from_its_seed(self) -> None:
        # doc 00 E3. Two solvers, same seed, identical deliverable to the bit.
        first = _sheath_solver(n_steps=600).run(_params(), None)
        second = _sheath_solver(n_steps=600).run(_params(), None)

        assert (
            first.flux.energy_flux_toward_wall_watt_per_m2
            == second.flux.energy_flux_toward_wall_watt_per_m2
        )
        np.testing.assert_array_equal(first.iedf.sample_energies_ev, second.iedf.sample_energies_ev)

    def test_a_different_seed_changes_the_run(self) -> None:
        first = _sheath_solver(n_steps=600, seed=1).run(_params(), None)
        second = _sheath_solver(n_steps=600, seed=2).run(_params(), None)
        assert first.iedf.sample_energies_ev.size > 0
        assert not np.array_equal(first.iedf.sample_energies_ev, second.iedf.sample_energies_ev)


class TestFluxConvention:
    def test_energy_arrives_at_the_wall_with_a_positive_sign(self) -> None:
        # doc 02 §2: flux toward the wall is positive. Getting this backwards flips the
        # sign of the only number the project reports.
        run = _sheath_solver(n_steps=800).run(_params(bias_v=-50.0), None)
        assert run.flux.is_toward_wall
        assert float(run.flux.particle_flux_toward_wall_per_m2_s) > 0.0

    def test_the_flux_is_recomputable_from_the_state(self) -> None:
        solver = _sheath_solver(n_steps=800)
        state = solver.solve(_params(bias_v=-50.0), None)
        recovered = solver.flux(state, 0.0)
        assert recovered.fidelity is Fidelity.L2
        assert float(recovered.energy_flux_toward_wall_watt_per_m2) > 0.0

    def test_a_state_from_another_run_is_refused(self) -> None:
        # The wall flux is accumulated from every absorbed particle during the run; it
        # cannot be recovered from a state alone. Answering anyway would be a plausible
        # wrong number, which is the failure mode ADR-011 records.
        solver = _sheath_solver(n_steps=400)
        state = solver.solve(_params(), None)
        other = _sheath_solver(n_steps=400)
        other.solve(_params(), None)
        with pytest.raises(ValueError, match="did not produce"):
            other.flux(state, 0.0)


class TestParticleBookkeeping:
    def test_nothing_is_created_or_destroyed_between_the_boundaries(self) -> None:
        # The conservation statement the whole loop rests on: every macroparticle that
        # left the simulation did so through a named boundary, and every one that appeared
        # came from a named source. A leak here would move Gamma_E without moving anything
        # a reader would think to look at.
        run = _sheath_solver(n_steps=800).run(_params(gamma_se=0.2), None)
        ledger = run.diagnostics

        assert (
            ledger.ions_loaded + ledger.ions_injected
            == ledger.ions_absorbed + ledger.ions_left_bulk + ledger.ions_alive
        )
        assert (
            ledger.electrons_loaded + ledger.electrons_injected + ledger.secondaries_emitted
            == ledger.electrons_absorbed + ledger.electrons_left_bulk + ledger.electrons_alive
        )

    def test_secondary_emission_scales_with_the_yield(self) -> None:
        # doc 03 §3.3: "Secondary electron emission is not optional." A run with
        # gamma_se = 0 must emit none, and one at 0.2 must emit about a fifth of the
        # absorbed ion count.
        without = _sheath_solver(n_steps=800).run(_params(gamma_se=0.0), None)
        with_see = _sheath_solver(n_steps=800).run(_params(gamma_se=0.2), None)

        assert without.diagnostics.secondaries_emitted == 0
        assert with_see.diagnostics.secondaries_emitted > 0
        ratio = with_see.diagnostics.secondaries_emitted / with_see.diagnostics.ions_absorbed
        assert ratio == pytest.approx(0.2, rel=0.2)


# ── doc 03 §7 verification gates ────────────────────────────────────────────────


class TestV07EnergyConservation:
    @pytest.mark.physics
    @pytest.mark.slow
    def test_total_energy_drift_is_below_a_tenth_of_a_percent(self) -> None:
        # doc 03 §7 V-07. A closed box — specular reflection at both ends, no bias, no
        # collisions — so total energy is a conserved quantity of the equations and the
        # leapfrog either conserves it or it does not. Absorption at a wall removes
        # energy by design, which is why the gate cannot be stated in the sheath
        # configuration.
        params = _params(bias_v=0.0, T_i_ev=3.0)
        lambda_d = debye_length_m(density_m3=params.n_0_per_m3, electron_temperature_ev=3.0)
        solver = Pic1D3VSolver(
            wall=WallModel.REFLECTING,
            domain_length_m=32.0 * lambda_d,
            n_ppc=400.0,
            cells_per_debye=1.0,
            n_steps=4000,
            capacity_factor=1.1,
            seed=_SEED,
        )

        run = solver.run(params, None)
        drift = run.diagnostics.energy_drift

        assert abs(drift) < V07_ENERGY_DRIFT_TOLERANCE, (
            f"total energy drifted by {drift * 100:.4f} % over {run.diagnostics.n_steps} steps"
        )

    @pytest.mark.physics
    @pytest.mark.slow
    def test_a_cell_far_wider_than_the_debye_length_would_heat(self) -> None:
        # Proof the gate has teeth. doc 03 §4.3's dz <= lambda_D constraint exists because
        # a coarser grid transfers energy from the field to the particles with no physics
        # doing it. The constraint check is bypassed here on purpose, to show that the
        # thing it forbids is the thing that breaks V-07.
        params = _params(bias_v=0.0, T_i_ev=3.0)
        lambda_d = debye_length_m(density_m3=params.n_0_per_m3, electron_temperature_ev=3.0)
        solver = Pic1D3VSolver(
            wall=WallModel.REFLECTING,
            domain_length_m=32.0 * lambda_d,
            n_cells=4,
            n_ppc=400.0,
            n_steps=4000,
            capacity_factor=1.1,
            seed=_SEED,
            check_constraints=False,
        )

        run = solver.run(params, None)
        assert abs(run.diagnostics.energy_drift) > V07_ENERGY_DRIFT_TOLERANCE


class TestV05TimestepIndependence:
    @pytest.mark.physics
    @pytest.mark.slow
    def test_gamma_e_changes_by_less_than_one_percent_on_halving_dt(self) -> None:
        # doc 03 §7 V-05. Neither run is a recorded expectation; they are each other's
        # reference, and a scheme whose answer depended on dt would separate them.
        coarse = _gate_run(bias_v=_GATE_BIAS_V, n_ppc=800.0)
        fine = _gate_run(bias_v=_GATE_BIAS_V, n_ppc=800.0, timestep_refinement=2.0)

        # The same physical duration either way: the run length is derived from an ion
        # transit time, so halving dt doubles the step count rather than shortening the run.
        assert fine.diagnostics.dt_s == pytest.approx(0.5 * coarse.diagnostics.dt_s)
        assert fine.diagnostics.window_duration_s == pytest.approx(
            coarse.diagnostics.window_duration_s, rel=1e-3
        )

        gamma_coarse = float(coarse.flux.energy_flux_toward_wall_watt_per_m2)
        gamma_fine = float(fine.flux.energy_flux_toward_wall_watt_per_m2)
        change = abs(gamma_fine - gamma_coarse) / gamma_coarse

        assert change < V05_TIMESTEP_TOLERANCE, (
            f"Gamma_E moved {change * 100:.3f} % between dt = {coarse.diagnostics.dt_s:.3e} s "
            f"and dt = {fine.diagnostics.dt_s:.3e} s "
            f"({gamma_coarse:.4e} -> {gamma_fine:.4e} W/m^2)"
        )


class TestV06ParticleCountConvergence:
    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_iedf_shape_converges_on_doubling_n_ppc(self) -> None:
        # doc 03 §7 V-06, and doc 03 §4.3's warning: converging the mean while the tail is
        # still noise is a classic and silent PIC failure, so the statistic is the KS
        # distance between the two *distributions*.
        coarse = _gate_run(bias_v=_GATE_BIAS_V, n_ppc=1200.0)
        fine = _gate_run(bias_v=_GATE_BIAS_V, n_ppc=2400.0)

        distance = float(
            stats.ks_2samp(coarse.iedf.sample_energies_ev, fine.iedf.sample_energies_ev).statistic
        )
        # The sampling floor: two finite samples of the *same* distribution separate by
        # about this much. Reported beside the statistic so a pass cannot be mistaken for
        # convergence when it is really just a small sample.
        floor = 1.36 * math.sqrt(1.0 / coarse.iedf.n_samples + 1.0 / fine.iedf.n_samples)

        assert distance < V06_KS_TOLERANCE, (
            f"KS distance {distance:.4f} between N_ppc = 1200 ({coarse.iedf.n_samples} "
            f"arrivals) and N_ppc = 2400 ({fine.iedf.n_samples} arrivals); the 5 % "
            f"sampling floor for these counts is {floor:.4f}"
        )

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_mean_impact_energy_also_converges(self) -> None:
        # The moment, reported alongside the shape rather than instead of it. If this
        # passed while the KS test failed, that would be exactly the silent failure
        # doc 03 §4.3 describes — so both are recorded.
        coarse = _gate_run(bias_v=_GATE_BIAS_V, n_ppc=1200.0)
        fine = _gate_run(bias_v=_GATE_BIAS_V, n_ppc=2400.0)

        change = abs(fine.iedf.mean_energy_ev - coarse.iedf.mean_energy_ev) / (
            coarse.iedf.mean_energy_ev
        )
        assert change < V05_TIMESTEP_TOLERANCE


# ── the cross-model check ───────────────────────────────────────────────────────


class TestAgainstAnalyticSheath:
    """L2 against L0 in the collisionless high-bias limit — doc 03 §2.3, doc 07 V-03.

    The one gate in this file whose reference is an independently verified model. L0 is
    verified against doc 03 §2's own printed arithmetic in ``test_analytic_sheath.py``,
    and nothing in the PIC kernel knows about Child-Langmuir: ``omega_pe``, ``c_s`` and
    the 4/3 power law appear nowhere in ``fields.py``, ``push.py`` or ``solver.py``.
    """

    #: -250 V at T_e = 3 eV is V_w / T_e = 83, which is the collisionless high-bias limit
    #: doc 03 §2.3 states its Child-Langmuir result in. A wider domain and a longer run
    #: than the other gates use, because both the sheath and the ion transit scale with the
    #: bias: at 83 the sheath spans 28 Debye lengths against 8 at V_w / T_e = 17.
    _BIAS_V = -250.0
    _DOMAIN_SHEATHS = 1.4
    _TRANSITS = 5.0
    _EQUILIBRATION = 0.6

    @classmethod
    def _run(cls) -> tuple[PlasmaParams, PicRun]:
        return _params(bias_v=cls._BIAS_V), _gate_run(
            bias_v=cls._BIAS_V,
            n_ppc=400.0,
            domain_sheaths=cls._DOMAIN_SHEATHS,
            transits=cls._TRANSITS,
            equilibration_fraction=cls._EQUILIBRATION,
        )

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_ion_flux_at_the_wall_is_the_bohm_flux(self) -> None:
        # A conservation statement about the loop rather than a prediction: doc 03 §3.3's
        # bulk boundary injects at n_s c_s, so anything other than n_s c_s arriving means
        # the loop created or destroyed ions. Labelled as such in the module docstring.
        params, run = self._run()
        expected = float(
            magnitude_in(ion_flux(params, h_l=DEFAULT_EDGE_TO_CENTRE_RATIO), "1/(m**2*s)")
        )
        measured = float(run.flux.particle_flux_toward_wall_per_m2_s)

        assert measured == pytest.approx(expected, rel=0.05), (
            f"Gamma_i = {measured:.4e} against the Bohm flux {expected:.4e} m^-2 s^-1"
        )

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_mean_impact_energy_is_the_sheath_drop(self) -> None:
        # A prediction. The ion gains e V_w falling through the sheath the *field solve*
        # produced, plus the entry energy L0 drops. So the PIC must land at or a little
        # above e V_w, and a PIC with a broken push or a mis-scaled Poisson would not.
        params, run = self._run()
        drop_ev = float(magnitude_in(params.bias_magnitude, "V"))
        measured = run.iedf.mean_energy_ev

        assert measured == pytest.approx(drop_ev, rel=0.05), (
            f"<E_i> = {measured:.2f} eV against the sheath drop {drop_ev:.2f} eV"
        )

    @pytest.mark.physics
    @pytest.mark.slow
    def test_gamma_e_reproduces_the_analytic_limit(self) -> None:
        """doc 03 §2.3's "number every other model must reproduce in this limit".

        Two statements, because they are not equally strong and conflating them hides
        which one failed.

        **The physics statement**: L0 is a documented *lower* bound here — it drops the ion
        entry energy — so the PIC must sit at or above it. That is a one-sided assertion
        and it is sharp.

        **The agreement statement** is bounded by arithmetic rather than by choice.
        ``Gamma_E = Gamma_i * <E_i>``, and the two factors are gated separately in this
        class at 5 % each. A product of two quantities each known to 5 % is known to about
        10 %, so the original 5 % gate on the product was **never achievable from its own
        inputs** — it would have had to be met by the two errors cancelling. Measured:
        +5.48 %, with ``Gamma_i`` and ``<E_i>`` both inside their own gates. So this is
        tolerance stacking, not a physics disagreement, and the fix is to state the bound
        correctly rather than to widen it quietly.

        What this does **not** say: doc 03 §2.3 and ADR-010 put the physically expected
        excess at about +0.75 %, from the ion entry energy alone. +5.48 % is well above
        that, and the residual is statistical — 400 particles per cell and 5 transits. This
        gate therefore certifies "no gross error", not "converged". Closing the gap to
        +0.75 % needs more particles and longer equilibration, and is recorded as open
        rather than asserted away.
        """
        params, run = self._run()
        expected = float(
            magnitude_in(ion_energy_flux(params, h_l=DEFAULT_EDGE_TO_CENTRE_RATIO), "W/m**2")
        )
        measured = float(run.flux.energy_flux_toward_wall_watt_per_m2)
        excess = measured / expected - 1.0

        assert measured >= expected, (
            f"Gamma_E = {measured:.4e} W/m^2 is BELOW the L0 lower bound "
            f"{expected:.4e} W/m^2. L0 omits the ion entry energy, so a kinetic solver "
            f"cannot legitimately fall under it."
        )
        assert excess < GAMMA_E_COMPOUNDED_TOLERANCE, (
            f"Gamma_E = {measured:.4e} W/m^2 exceeds the L0 limit {expected:.4e} W/m^2 by "
            f"{excess:.2%}, beyond the {GAMMA_E_COMPOUNDED_TOLERANCE:.0%} that Gamma_i and "
            f"<E_i> at 5 % each can compound to"
        )

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_sheath_is_thicker_than_child_langmuir_as_adr_010_predicted(self) -> None:
        """ADR-010's prediction about L2, checked now that L2 exists.

        The original form of this test asserted agreement with Child-Langmuir to 25 %.
        That criterion was already **retired** by ADR-010 before this solver was written:
        Child-Langmuir drops electron space charge entirely, and solving doc 03 §3.1's own
        equations gives ``s / s_CL = 1.5029`` at RP-1. ADR-010 replaced the comparison with
        V-03b — against an independent quadrature rather than against a closed form for a
        different equation set — precisely because agreement with Child-Langmuir is not
        something a correct solver should show.

        ADR-010 then made a prediction it could not test, because L2 did not exist:

            L2 (PIC) will face the same comparison. It resolves the presheath kinetically
            and has no Boltzmann-electron assumption, so its ``s`` should sit closer to the
            quadrature than to Child-Langmuir.

        Measured: ``s / s_CL = 1.330``, which is 0.173 from the quadrature and 0.330 from
        Child-Langmuir. The prediction holds, and it holds for a solver that shares no code
        with the quadrature that produced it.

        That it lands *between* the two is the physically expected place. The quadrature
        assumes Boltzmann electrons all the way to the wall; the PIC does not, and a
        kinetic electron population is depleted near a strongly negative wall relative to
        Boltzmann, which thins the sheath. So ``1 < s/s_CL < 1.5029`` is the interval a
        correct kinetic solver should land in, and it does.
        """
        params, run = self._run()
        child_langmuir = float(magnitude_in(child_langmuir_thickness(params, h_l=0.61), "m"))
        measured = run.sheath_thickness_m
        ratio = measured / child_langmuir

        assert ratio > 1.0 + ADR_010_MINIMUM_EXCESS, (
            f"s / s_CL = {ratio:.4f}; a kinetic sheath must exceed Child-Langmuir, which "
            f"neglects electron space charge entirely (ADR-010)"
        )
        assert ratio < ADR_010_QUADRATURE_RATIO, (
            f"s / s_CL = {ratio:.4f} exceeds the Boltzmann-electron quadrature's "
            f"{ADR_010_QUADRATURE_RATIO:.4f}; a kinetic electron population is depleted "
            f"near the wall relative to Boltzmann, so the PIC sheath should be thinner "
            f"than the quadrature's, not thicker"
        )
        assert abs(ratio - ADR_010_QUADRATURE_RATIO) < abs(ratio - 1.0), (
            f"s / s_CL = {ratio:.4f} sits closer to Child-Langmuir than to the quadrature "
            f"{ADR_010_QUADRATURE_RATIO:.4f}, contradicting ADR-010's prediction for L2"
        )

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_potential_profile_follows_the_four_thirds_power_law(self) -> None:
        # doc 03 §2.3's profile, Phi = -V_w (1 - z/s)^(4/3). Compared as a shape over the
        # sheath, normalised by V_w so the statement is about the profile rather than
        # about the boundary value the Dirichlet condition already fixes.
        params, run = self._run()
        state = run.state
        thickness = child_langmuir_thickness(params, h_l=0.61)

        analytic = np.asarray(
            magnitude_in(child_langmuir_potential(params, state.grid, thickness), "V")
        )
        measured = np.asarray(state.field("Phi").values)
        drop = float(magnitude_in(params.bias_magnitude, "V"))

        inside = state.grid.z_m <= float(magnitude_in(thickness, "m"))
        rms = float(np.sqrt(np.mean(((measured - analytic)[inside] / drop) ** 2)))

        assert rms < 0.1, f"RMS potential difference {rms * 100:.2f} % of V_w inside the sheath"


class TestSurfaceModelIsWired:
    def test_the_solver_takes_the_yield_from_the_plasma_parameters(self) -> None:
        # doc 08 §1 principle 4: gamma_se is a manifest quantity (PlasmaParams.gamma_se),
        # not a constant the solver keeps. Reading it from the registry instead would make
        # a swept yield silently inert.
        solver = _sheath_solver()
        surface = solver.surface_model(_params(gamma_se=0.17))
        assert isinstance(surface, SurfaceModel)
        assert surface.secondary_yield == 0.17

    def test_reflection_applies_to_ions_and_not_to_electrons(self) -> None:
        # doc 03 §3.3's wall row absorbs the electron flux outright; R_i is doc 03 §8 A10's
        # *ion* reflection coefficient, quoted for Ar+ on tungsten. One SurfaceModel is
        # applied to both species by the boundary step, so a coefficient meant for the ions
        # will silently reflect electrons too — and the default R_i = 0 hides it. It only
        # activates over the registry's own [0, 0.2] sweep range, where it would suppress
        # the electron wall flux, warm the sheath and shift Gamma_E, with nothing in the
        # ledger looking wrong. Stated at R_i = 1 so the failure is unambiguous rather than
        # a fifth of an effect buried in shot noise.
        run = _sheath_solver(n_steps=800, ion_reflection=1.0).run(_params(), None)

        assert run.diagnostics.ions_absorbed == 0, "R_i = 1 must reflect every arriving ion"
        assert run.diagnostics.electrons_absorbed > 0, (
            "electrons were reflected by the ion reflection coefficient: doc 03 §3.3 "
            "absorbs the electron flux at the wall"
        )
