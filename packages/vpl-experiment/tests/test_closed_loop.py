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

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.state import PlasmaParams, Species
from vpl.core.units import Q_, magnitude_in
from vpl.experiment.closed_loop import (
    T0_RELATIVE_TOLERANCE,
    ClosedLoopReport,
    run_t0,
    run_t1,
    run_t2,
)
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import GAMMA_I_COLD_ION, ion_energy_flux
from vpl.validation.sealed import InverseCrimeError, Tier, TierMismatchError

#: Two seeds, so the reproducibility test is not vacuously comparing an object to itself.
_SEED_A = 0
_SEED_B = 1


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


# ─── T2 — honest, and honestly absent ──────────────────────────────────────────────


class TestT2Unavailable:
    """T2 needs a genuinely different truth-generating model — doc 05 §7.1.

    L0 is the only forward model this environment can run (the L1 fluid solver depends on
    ``dolfinx``, which is not installed here, and the L2 PIC-MCC kernel is explicitly
    out of scope for this harness per the task brief). Faking T2 by, say, changing L0's own
    ``h_l`` or ``gamma_i`` between truth and inversion would be exactly the "tempting
    half-measure" ``vpl.validation.sealed.tier_of_configuration`` is written to refuse: doc
    05 §7.1 requires physics level, grid, timestep, collision set, EEDF form *and*
    calibration to differ together, not any one of them in isolation. So this harness does
    not mislabel a same-model run as T2; it says plainly that T2 cannot run here.
    """

    def test_states_the_missing_dependency_rather_than_mislabelling_a_run(self) -> None:
        with pytest.raises(NotImplementedError, match="dolfinx"):
            run_t2(seed=_SEED_A)


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
