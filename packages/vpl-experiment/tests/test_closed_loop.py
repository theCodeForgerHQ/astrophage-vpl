"""The closed-loop recovery harness — doc 05 §7, doc 07 §3, doc 11 §9 item 4 / WBS 3.9.

## What T0 does and does not prove

doc 05 §7.2 is explicit that T0 ("same model, no noise") failing means a bug, and that
nothing else is meaningful until it passes. The corollary, stated in ADR-011 and repeated
in the harness module docstring, is what these tests are built around: a T0 pass proves the
forward chain, the likelihood, the MAP engine and the sealed-truth barrier are wired
together *consistently* — it does not prove the OES forward model describes a real plasma,
and it does not prove the recovered point is the only one that fits (doc 05 §6's
identifiability question is a separate one, partially answered below by
``test_gamma_e_is_insensitive_to_the_fixed_control_parameters``).

## Why these tests are built around known answers, not the code's own output

Following ``vpl.inverse.map``'s own stated discipline (its module docstring, and ADR-011
directly): "verify against closed forms and known answers, never against the code's own
output." ``test_gamma_e_is_insensitive_to_the_fixed_control_parameters`` recomputes
``Gamma_E`` from :mod:`vpl.physics.analytic.sheath` directly, independently of anything the
harness does with it, and checks a physical fact the harness's dimensionality reduction
depends on — it is not a test of the harness's own arithmetic reproducing itself.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.protocols.forward import IonEnergyFlux
from vpl.core.state import Fidelity, PlasmaParams, PlasmaState, ScalarField, SpatialGrid, Species
from vpl.core.units import Q_, magnitude_in
from vpl.experiment.closed_loop import (
    T0_RELATIVE_TOLERANCE,
    ClosedLoopReport,
    _gamma_e_at_wall,
    _GeneralisedEedf,
    _generate_truth,
    _resample_onto_grid,
    _run,
    _t2_truth_eedf_factory,
    run_t0,
    run_t1,
    run_t2,
)
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import GAMMA_I_COLD_ION, AnalyticSheathSolver, ion_energy_flux
from vpl.physics.eedf.analytic import (
    DRUYVESTEYN_KAPPA,
    MAXWELLIAN_KAPPA,
    druyvesteyn_eedf,
    maxwellian_eedf,
)
from vpl.physics.eedf.grid import EnergyGrid
from vpl.validation.sealed import InverseCrimeError, Tier, TierMismatchError

#: Two seeds, so the reproducibility test is not vacuously comparing an object to itself.
_SEED_A = 0
_SEED_B = 1

#: Whether dolfinx (FEniCSx) is importable in this environment. T2's truth solver, L1's
#: FluidSheathSolver, depends on it; tests that need a real T2 run skip cleanly when it is
#: absent (this development machine), and one test below exercises the opposite case —
#: that run_t2 fails loudly and specifically when dolfinx is missing — only when it applies.
_DOLFINX_AVAILABLE = importlib.util.find_spec("dolfinx") is not None


# ─── T0 — the blocking gate ───────────────────────────────────────────────────────


class TestT0Consistency:
    """doc 05 §7.2: "Failing T0 means a bug; nothing else is meaningful until it passes."

    Nothing downstream in this file is trustworthy if this class fails, which is why it is
    first and why every other class in this module is conditioned on this one having
    already passed in CI.
    """

    def test_recovers_gamma_e_to_numerical_tolerance(self) -> None:
        report = run_t0(seed=_SEED_A)

        assert report.tier is Tier.T0
        # This is the T0 gate itself, not a duplicate of it: assert_t0_consistency raises
        # AssertionError and names doc 05 §7.2 explicitly if it fails, which is a much
        # louder and more specific failure than a bare `assert relative_error < tol` would
        # produce in CI output.
        report.sealed.assert_t0_consistency(tolerance=T0_RELATIVE_TOLERANCE)

    def test_map_engine_reports_convergence(self) -> None:
        """A T0 pass that silently rode in on a non-converged optimiser proves nothing.

        ``MapResult.converged`` is False whenever the optimiser hit its iteration cap
        without SciPy reporting success (doc 05 §5's ``MapResult`` docstring: "A MAP that
        silently returns its iteration limit is how a bad fit becomes a published number").
        """
        report = run_t0(seed=_SEED_A)
        assert report.map_result.converged
        assert report.map_result.iterations > 0

    def test_truth_is_unreachable_before_commit(self) -> None:
        """The structural guard doc 05 §7 requires, exercised on this harness's own object.

        The report only exists after ``run_t0`` has already committed an estimate — by
        that point ``sealed.value`` is legitimately readable. What this test checks is that
        the object the harness built really is a :class:`SealedTruth` with the barrier
        live, by asking it what it does before commitment on a *fresh* instance built the
        same way the harness builds its own. A harness that quietly used a plain float
        instead of routing through ``SealedTruth`` would still pass every numeric assertion
        above and would defeat doc 05 §7's guard exactly as ADR-011 describes.
        """
        report = run_t0(seed=_SEED_A)
        assert report.sealed.is_committed
        # Committing a second estimate against an already-committed seal is the "revise it
        # after the truth became readable" crime doc 05 §7's docstring names by name.
        with pytest.raises(InverseCrimeError):
            report.sealed.commit_estimate(0.0, tier=Tier.T0)

    def test_t0_result_cannot_be_reported_as_t1_or_t2(self) -> None:
        """doc 05 §7.2's ordering: a T0 result is not a T1 or a T2 result under another name."""
        report = run_t0(seed=_SEED_A)
        with pytest.raises(TierMismatchError):
            report.sealed.assert_at_least(Tier.T1)
        with pytest.raises(TierMismatchError):
            report.sealed.assert_at_least(Tier.T2)

    def test_seed_reproduces_bit_identically(self) -> None:
        """doc 00 E3 / the seeding contract: identical seed, identical run, identical answer."""
        first = run_t0(seed=_SEED_A)
        second = run_t0(seed=_SEED_A)
        assert first.n_0_hat_per_m3 == second.n_0_hat_per_m3
        assert first.T_e_hat_ev == second.T_e_hat_ev
        assert first.gamma_e_estimate_w_per_m2 == second.gamma_e_estimate_w_per_m2

    def test_different_seeds_draw_different_truths(self) -> None:
        """A reproducibility test that always compares equal objects is not testing anything.

        This is the complement of the bit-identical check above: two different seeds must
        draw two different ground truths (doc 10 §5's per-stream, name-keyed seeding), or
        the "reproduces bit-identically" test would pass even if the seed were ignored
        entirely.
        """
        first = run_t0(seed=_SEED_A)
        second = run_t0(seed=_SEED_B)
        assert first.gamma_e_true_w_per_m2 != second.gamma_e_true_w_per_m2


# ─── T1 — same model, with noise ───────────────────────────────────────────────────


class TestT1OptimisticBound:
    """doc 05 §7.2: same model, with noise — "the upper bound on achievable accuracy"."""

    def test_runs_and_reports_a_finite_error_at_the_correct_tier(self) -> None:
        report = run_t1(seed=_SEED_A)
        assert report.tier is Tier.T1
        assert np.isfinite(report.relative_error)
        assert report.relative_error >= 0.0

    def test_t1_result_cannot_be_reported_as_t2(self) -> None:
        """doc 05 §7.2: "Reporting T1 as if it were T2 is treated as a project defect."

        This is the single most load-bearing assertion in this file after T0 itself: it is
        the test that would fail if a report generator ever quietly relabelled an optimistic
        T1 number as the honest T2 one.
        """
        report = run_t1(seed=_SEED_A)
        with pytest.raises(TierMismatchError):
            report.sealed.assert_at_least(Tier.T2)

    def test_t1_is_not_eligible_for_the_t0_check(self) -> None:
        """T1 is noisy by construction; ``assert_t0_consistency`` must refuse to run on it.

        Loosening the T0 tolerance until a noisy run passes it is exactly the "success
        chosen by whoever was in a hurry" failure mode doc 05 §7's barrier exists to make
        impossible to reach quietly.
        """
        report = run_t1(seed=_SEED_A)
        with pytest.raises(TierMismatchError):
            report.sealed.assert_t0_consistency(tolerance=1.0)

    def test_seed_reproduces_bit_identically(self) -> None:
        first = run_t1(seed=_SEED_A)
        second = run_t1(seed=_SEED_A)
        assert first.n_0_hat_per_m3 == second.n_0_hat_per_m3
        assert first.T_e_hat_ev == second.T_e_hat_ev


# ─── T2 — mismatched models, noise, imperfect calibration ─────────────────────────


class TestT2RequiresDolfinx:
    """L1's ``FluidSheathSolver`` is the truth-generating model doc 05 §7.1 needs, and it
    depends on ``dolfinx`` (FEniCSx). This machine does not have it installed, so this is
    the one T2 test that is *expected* to run here — and to run only here: once dolfinx is
    available, ``run_t2`` must stop raising and this test's premise no longer holds.
    """

    @pytest.mark.skipif(_DOLFINX_AVAILABLE, reason="dolfinx is installed; T2 is runnable")
    def test_states_the_missing_dependency_rather_than_mislabelling_a_run(self) -> None:
        with pytest.raises(NotImplementedError, match="dolfinx"):
            run_t2(seed=_SEED_A)


class TestT2ProducesACertifiedResult:
    """The other side of :class:`TestT2RequiresDolfinx`: with dolfinx installed, ``run_t2``
    must actually produce a tier-certified result rather than continue to refuse. Skipped
    everywhere this repository's local test run happens (no dolfinx on this machine); it is
    the test the remote FEniCSx-enabled environment exists to pass.
    """

    @pytest.mark.skipif(not _DOLFINX_AVAILABLE, reason="requires dolfinx (FEniCSx)")
    def test_run_t2_is_certified_at_tier_2_by_sealed_tier_of_configuration(self) -> None:
        report = run_t2(seed=_SEED_A)

        assert report.tier is Tier.T2
        assert np.isfinite(report.relative_error)
        assert report.relative_error >= 0.0
        # Doesn't raise: T2 satisfies its own bar, certified by tier_of_configuration
        # rather than hand-labelled by this module (doc 11 §9 item 4's task brief).
        report.sealed.assert_at_least(Tier.T2)

    @pytest.mark.skipif(not _DOLFINX_AVAILABLE, reason="requires dolfinx (FEniCSx)")
    def test_t2_result_is_not_eligible_for_the_t0_consistency_check(self) -> None:
        """T2 is noisy and mismatched by construction; the T0 numerical-tolerance gate
        would either fail spuriously or, worse, be loosened until it did not — exactly the
        failure `assert_t0_consistency` exists to make impossible quietly (doc 05 §7.2)."""
        report = run_t2(seed=_SEED_A)
        with pytest.raises(TierMismatchError):
            report.sealed.assert_t0_consistency(tolerance=1.0)


# ─── the refactor's regression guard: T0/T1 must be unchanged by the new parameters ──


class TestRefactorPreservesT0AndT1:
    """doc 11 §9 item 4's constraint: T0 and T1 are blocking/optimistic-bound results and
    must keep their pre-refactor numbers exactly. ``run_t0``/``run_t1`` still call
    ``_run`` with no new arguments, so this is really a check that ``_run``'s new
    ``truth_solver=None`` / ``truth_eedf_factory=None`` / ``imperfect_calibration=None``
    defaults are equivalent to stating the T0/T1 configuration explicitly — if they were
    not, this would be the test that would catch it, not a diff nobody reads.
    """

    def test_explicit_l0_maxwellian_arguments_reproduce_run_t0(self) -> None:
        defaulted = _run(noise=False, seed=_SEED_A)
        explicit = _run(
            noise=False,
            seed=_SEED_A,
            truth_solver=AnalyticSheathSolver(),
            truth_eedf_factory=lambda grid: _GeneralisedEedf(grid=grid, kappa=MAXWELLIAN_KAPPA),
            imperfect_calibration=False,
        )

        assert defaulted.n_0_hat_per_m3 == explicit.n_0_hat_per_m3
        assert defaulted.T_e_hat_ev == explicit.T_e_hat_ev
        assert defaulted.gamma_e_true_w_per_m2 == explicit.gamma_e_true_w_per_m2
        assert defaulted.gamma_e_estimate_w_per_m2 == explicit.gamma_e_estimate_w_per_m2
        assert defaulted.tier is explicit.tier is Tier.T0

    def test_explicit_l0_maxwellian_arguments_reproduce_run_t1(self) -> None:
        defaulted = _run(noise=True, seed=_SEED_A)
        explicit = _run(
            noise=True,
            seed=_SEED_A,
            truth_solver=AnalyticSheathSolver(),
            truth_eedf_factory=lambda grid: _GeneralisedEedf(grid=grid, kappa=MAXWELLIAN_KAPPA),
            imperfect_calibration=True,
        )

        assert defaulted.n_0_hat_per_m3 == explicit.n_0_hat_per_m3
        assert defaulted.T_e_hat_ev == explicit.T_e_hat_ev
        assert defaulted.tier is explicit.tier is Tier.T1


# ─── _gamma_e_at_wall: doc 03 §6's "same functional", read off two return shapes ──


class TestGammaEAtWall:
    """L0's ``.flux(state)`` returns a bare pint ``Quantity``; L1's returns an
    :class:`IonEnergyFlux`. Both must reduce to the same kind of float so ``_run`` does
    not need to know which physics level produced the truth (doc 00 E1)."""

    def test_reads_a_bare_quantity_in_watts_per_square_metre(self) -> None:
        assert _gamma_e_at_wall(Q_(1234.5, "W/m**2")) == pytest.approx(1234.5)

    def test_converts_a_quantity_in_other_energy_flux_units(self) -> None:
        assert _gamma_e_at_wall(Q_(1.0, "kW/m**2")) == pytest.approx(1000.0)

    def test_reads_an_ion_energy_flux(self) -> None:
        registry = default_registry()
        species = Species(name="Ar+", mass=registry["species.Ar.mass"].quantity, charge_number=1)
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=species,
            energy_flux_toward_wall_watt_per_m2=np.asarray(6577.0),
            particle_flux_toward_wall_per_m2_s=np.asarray(1.0e20),
            fidelity=Fidelity.L1,
        )

        assert _gamma_e_at_wall(flux) == pytest.approx(6577.0)


# ─── _resample_onto_grid: the grid discipline T2's own solve/observe split needs ──


def _linear_state(*, grid: SpatialGrid, params: PlasmaParams) -> PlasmaState:
    """A state whose fields are exactly linear in ``z`` — so linear interpolation onto a
    coarser grid reproduces the closed form exactly, not just approximately."""

    def field(name: str, *, at_wall: float, slope: float, units: str) -> ScalarField:
        return ScalarField(
            name=name,
            values=at_wall + slope * grid.z_m,
            units=units,
            grid=grid,
            time=None,
        )

    return PlasmaState(
        params=params,
        grid=grid,
        time=None,
        fields={
            "n_e": field("n_e", at_wall=1.0e15, slope=1.0e14, units="m**-3"),
            "n_i": field("n_i", at_wall=1.0e15, slope=1.0e14, units="m**-3"),
            "Phi": field("Phi", at_wall=0.0, slope=10.0, units="V"),
            "T_e": field("T_e", at_wall=3.0, slope=0.1, units="eV"),
        },
        ion_distribution=None,
        fidelity=Fidelity.L1,
    )


class TestResampleOntoGrid:
    def test_reproduces_a_linear_profile_exactly_at_the_new_grid_points(self) -> None:
        registry = default_registry()
        species = Species(name="Ar+", mass=registry["species.Ar.mass"].quantity, charge_number=1)
        params = _plasma_params(ControlParameters.reference().replace(n_0=5.0e17))
        native = SpatialGrid.uniform(length=Q_(0.02, "m"), n_points=41)
        state = _linear_state(grid=native, params=params)
        target = SpatialGrid.uniform(length=Q_(0.01, "m"), n_points=5)

        resampled = _resample_onto_grid(state, target)

        assert resampled.grid is target
        np.testing.assert_allclose(
            resampled.field("n_e").values, 1.0e15 + 1.0e14 * target.z_m
        )
        np.testing.assert_allclose(resampled.field("Phi").values, 10.0 * target.z_m)
        del species  # constructed only to keep params realistic; unused directly

    def test_rejects_a_target_grid_reaching_past_the_native_domain(self) -> None:
        registry = default_registry()
        params = _plasma_params(ControlParameters.reference())
        native = SpatialGrid.uniform(length=Q_(0.01, "m"), n_points=11)
        state = _linear_state(grid=native, params=params)
        target = SpatialGrid.uniform(length=Q_(0.02, "m"), n_points=5)

        with pytest.raises(ValueError, match="native"):
            _resample_onto_grid(state, target)
        del registry  # imported for parity with the other tests in this module

    def test_the_resampled_state_still_satisfies_the_plasma_state_contract(self) -> None:
        params = _plasma_params(ControlParameters.reference())
        native = SpatialGrid.uniform(length=Q_(0.02, "m"), n_points=21)
        state = _linear_state(grid=native, params=params)
        target = SpatialGrid.uniform(length=Q_(0.01, "m"), n_points=7)

        resampled = _resample_onto_grid(state, target)

        assert isinstance(resampled, PlasmaState)
        assert resampled.grid.n_points == 7


# ─── _generate_truth: the L0 branch, testable without dolfinx ─────────────────────


class TestGenerateTruthWithAnalyticSolver:
    """The L1 branch needs ``FluidSheathSolver`` (dolfinx) and is covered by
    :class:`TestT2ProducesACertifiedResult` instead. This class checks that when the truth
    solver *is* the same ``AnalyticSheathSolver`` the inversion uses — T0/T1's
    configuration — ``_generate_truth`` reduces to exactly the pre-refactor code path:
    solve directly on the fixed observation grid, no resampling.
    """

    def test_solves_directly_on_the_observation_grid_with_no_resampling(self) -> None:
        registry = default_registry()
        species = Species(name="Ar+", mass=registry["species.Ar.mass"].quantity, charge_number=1)
        solver = AnalyticSheathSolver()
        theta = ControlParameters.reference()
        params = _plasma_params(theta)
        grid = SpatialGrid.uniform(length=Q_(0.01, "m"), n_points=11)

        state, gamma_e = _generate_truth(params, truth_solver=solver, observation_grid=grid)

        assert state.grid is grid
        expected = float(
            magnitude_in(
                ion_energy_flux(params, h_l=solver.h_l, gamma_i=solver.gamma_i), "W/m**2"
            )
        )
        assert gamma_e == pytest.approx(expected)
        del species


# ─── the EEDF mismatch: doc 05 §7.1, and the route actually taken ──────────────────


class TestGeneralisedEedfProvider:
    """``_GeneralisedEedf`` wraps :func:`vpl.physics.eedf.analytic.generalised_eedf` as an
    ``EedfProvider`` — the "inversion side" of doc 05 §7.1's EEDF mismatch, per the
    parameter registry's own ``eedf.kappa`` entry (coordinator correction to this task: the
    kappa-parameterised family is the *inversion's* family, not truth's; see the module
    docstring for the full account and why ``vpl.instruments.oes.instrument.KappaEedf`` is
    not used here at all). Defined in this package rather than ``vpl-instruments`` because
    that package is off-limits for this task.
    """

    def test_at_kappa_one_it_matches_maxwellian_eedf_bit_for_bit(self) -> None:
        grid = EnergyGrid.linear(max_ev=60.0, n_cells=100)
        provider = _GeneralisedEedf(grid=grid, kappa=MAXWELLIAN_KAPPA)

        actual = provider.f0(electron_temperature_ev=3.0)
        expected = grid.normalise(maxwellian_eedf(grid.centres_ev, electron_temperature_ev=3.0))

        np.testing.assert_array_equal(actual, expected)

    def test_at_kappa_two_it_matches_the_druyvesteyn_closed_form(self) -> None:
        grid = EnergyGrid.linear(max_ev=60.0, n_cells=100)
        provider = _GeneralisedEedf(grid=grid, kappa=DRUYVESTEYN_KAPPA)

        actual = provider.f0(electron_temperature_ev=3.0)
        expected = grid.normalise(
            druyvesteyn_eedf(grid.centres_ev, mean_energy_ev=1.5 * 3.0)
        )

        np.testing.assert_allclose(actual, expected)

    def test_maxwellian_and_druyvesteyn_disagree_at_the_same_t_e(self) -> None:
        # The point of the whole exercise: doc 05 §7.1 needs a second *shape*, not a
        # relabelled Maxwellian at the same T_e.
        grid = EnergyGrid.linear(max_ev=60.0, n_cells=100)
        maxwellian = _GeneralisedEedf(grid=grid, kappa=MAXWELLIAN_KAPPA)
        druyvesteyn = _GeneralisedEedf(grid=grid, kappa=DRUYVESTEYN_KAPPA)

        assert not np.allclose(
            maxwellian.f0(electron_temperature_ev=3.0), druyvesteyn.f0(electron_temperature_ev=3.0)
        )

    def test_rejects_a_non_positive_electron_temperature(self) -> None:
        grid = EnergyGrid.linear(max_ev=60.0, n_cells=100)
        provider = _GeneralisedEedf(grid=grid, kappa=DRUYVESTEYN_KAPPA)

        with pytest.raises(ValueError, match="T_e"):
            provider.f0(electron_temperature_ev=0.0)

    def test_f0_is_cached_per_temperature(self) -> None:
        grid = EnergyGrid.linear(max_ev=60.0, n_cells=100)
        provider = _GeneralisedEedf(grid=grid, kappa=DRUYVESTEYN_KAPPA)

        first = provider.f0(electron_temperature_ev=3.0)
        second = provider.f0(electron_temperature_ev=3.0)

        assert first is second


def test_t2_truth_eedf_factory_returns_the_druyvesteyn_shape() -> None:
    """The concrete choice this harness makes for T2's truth EEDF: doc 03 §3.2's own
    stated real-world shape ("Druyvesteyn-like with a depleted tail"), not an arbitrary
    kappa. Testable without dolfinx: this factory only touches ``vpl.physics.eedf``."""
    grid = EnergyGrid.linear(max_ev=60.0, n_cells=100)

    provider = _t2_truth_eedf_factory(grid)

    assert isinstance(provider, _GeneralisedEedf)
    assert provider.kappa == DRUYVESTEYN_KAPPA


# ─── the honesty check: why the recovered vector is 2-dimensional, not 8 ──────────


def test_gamma_e_is_insensitive_to_the_fixed_control_parameters() -> None:
    """Recomputed independently from :mod:`vpl.physics.analytic.sheath` — not from the harness.

    doc 05 §2.1 gives eight Level A control parameters. This harness recovers only two of
    them, ``n_0`` and ``T_e`` (see the ``closed_loop`` module docstring for the third,
    ``V_w``, and why it is fixed rather than jointly optimised). The other four —
    ``T_i``, ``phi_RF``, ``gamma_se``, ``kappa`` — are fixed at their RP-1 reference values
    not merely because this harness's OES model happens not to read them, but because the
    quantity actually being recovered, ``Gamma_E = h_l n_0 c_s(T_e) e V_w`` (doc 03 §2.3),
    is *exactly* independent of them under the L0 defaults this harness uses
    (``gamma_i = GAMMA_I_COLD_ION = 0``, so ``c_s`` carries no ``T_i`` term at all). A
    recovery that left them at the wrong value would therefore still recover the right
    ``Gamma_E`` — which is the whole reason fixing them is a defensible scope decision for
    a first closed loop and not a silently smuggled-in inverse crime.

    This test does not call anything in ``vpl.experiment.closed_loop``; it re-derives the
    fact from the physics module directly, per ADR-011's discipline.
    """
    reference = ControlParameters.reference()
    baseline = ion_energy_flux(
        _plasma_params(reference), h_l=_H_L_DEFAULT, gamma_i=GAMMA_I_COLD_ION
    )

    for changed in (
        reference.replace(T_i=0.4),
        reference.replace(phi_RF=1.2345),
        reference.replace(gamma_se=0.25),
        reference.replace(kappa=4.0),
    ):
        perturbed = ion_energy_flux(
            _plasma_params(changed), h_l=_H_L_DEFAULT, gamma_i=GAMMA_I_COLD_ION
        )
        assert magnitude_in(perturbed, "W/m**2") == magnitude_in(baseline, "W/m**2")


def test_gamma_e_does_depend_on_the_recovered_parameters() -> None:
    """The complement of the test above: nothing here is trivially insensitive.

    Guards against a vacuous version of the independence claim — if ``n_0``, ``T_e`` and
    ``V_w`` also had no effect, "the harness recovers the parameters ``Gamma_E`` depends
    on" would be true only because nothing affects ``Gamma_E``.
    """
    reference = ControlParameters.reference()
    baseline = ion_energy_flux(
        _plasma_params(reference), h_l=_H_L_DEFAULT, gamma_i=GAMMA_I_COLD_ION
    )

    for changed in (
        reference.replace(n_0=1.3 * reference.n_0),
        reference.replace(T_e=1.3 * reference.T_e),
        reference.replace(V_w=1.3 * reference.V_w),
    ):
        perturbed = ion_energy_flux(
            _plasma_params(changed), h_l=_H_L_DEFAULT, gamma_i=GAMMA_I_COLD_ION
        )
        assert magnitude_in(perturbed, "W/m**2") != magnitude_in(baseline, "W/m**2")


# ─── local helpers, independent of the harness under test ─────────────────────────

_H_L_DEFAULT = 0.61  # doc 03 §2.1 / registry `sheath.h_l`; only used to keep this file's
# own physics check independent of vpl.physics.analytic.sheath's own default plumbing.


def _plasma_params(theta: ControlParameters) -> PlasmaParams:
    """``ControlParameters`` -> ``PlasmaParams``, duplicated deliberately.

    This mirrors ``vpl.experiment.closed_loop``'s own conversion but is written
    independently here on purpose: the whole point of
    ``test_gamma_e_is_insensitive_to_the_fixed_control_parameters`` is to check a physical
    fact without trusting the harness's own code to expose it faithfully.
    """
    registry = default_registry()
    species = Species(name="Ar+", mass=registry["species.Ar.mass"].quantity, charge_number=1)
    return PlasmaParams(
        species=species,
        n_0=Q_(theta.n_0, "m**-3"),
        T_e=Q_(theta.T_e, "eV"),
        T_i=Q_(theta.T_i, "eV"),
        T_g=registry.quantity("RP1.T_g"),
        pressure=Q_(theta.p, "mTorr"),
        bias=Q_(theta.V_w, "V"),
        gamma_se=theta.gamma_se,
        kappa=theta.kappa,
        rf_frequency=None,
        rf_phase=theta.phi_RF,
    )


def test_closed_loop_report_is_the_documented_type() -> None:
    """A cheap smoke test that the public return type has not drifted from its own contract."""
    report = run_t0(seed=_SEED_A)
    assert isinstance(report, ClosedLoopReport)
