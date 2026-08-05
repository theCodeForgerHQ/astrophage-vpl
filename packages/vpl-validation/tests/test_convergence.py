"""Order-of-accuracy verification — doc 07 §2.3, doc 06 §3.

    Method of manufactured solutions is applied to every PDE solver... Reported as a
    log-log error-vs-h plot with fitted slope. **Design order ± 0.1 is required.** A
    solver converging at first order when second was designed has a bug, and MMS is the
    only reliable way to find it.

This module is what turns that sentence into a gate. It is deliberately in a package that
depends on no solver, so nothing being judged can influence the judgement.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.validation.convergence import (
    ConvergenceResult,
    RefinementLevel,
    assert_design_order,
    linf_error,
    observed_order,
    relative_l2_error,
    richardson_extrapolate,
    weighted_l2_error,
)


def _clean_sequence(
    order: float, *, n: int = 5, h0: float = 0.1, c: float = 3.0
) -> list[RefinementLevel]:
    """An ideal ``e = c h^p`` sequence, halving ``h`` each time."""
    return [RefinementLevel(h=h0 / 2**k, error=c * (h0 / 2**k) ** order) for k in range(n)]


class TestObservedOrder:
    @pytest.mark.physics
    @pytest.mark.parametrize("design", [1.0, 2.0, 3.0, 4.0])
    def test_recovers_the_order_of_an_ideal_sequence(self, design: float) -> None:
        result = observed_order(_clean_sequence(design))

        assert result.observed_order == pytest.approx(design, abs=1e-9)

    @pytest.mark.physics
    def test_recovers_the_order_from_non_halving_refinement(self) -> None:
        # Graded meshes (doc 03 §3.4) do not refine by clean factors of two, and a study
        # that only worked for halving would be unusable on the mesh the solver runs on.
        levels = [RefinementLevel(h=h, error=3.0 * h**2) for h in (0.1, 0.07, 0.041, 0.019)]

        assert observed_order(levels).observed_order == pytest.approx(2.0, abs=1e-9)

    def test_reports_the_pairwise_orders_as_well_as_the_fit(self) -> None:
        # The global fit can look healthy while the finest pair has already fallen off.
        # Reporting both is what distinguishes "second order" from "second order on
        # average, first order where it matters".
        result = observed_order(_clean_sequence(2.0, n=4))

        assert len(result.pairwise_orders) == 3
        np.testing.assert_allclose(result.pairwise_orders, 2.0, atol=1e-9)

    def test_a_clean_sequence_fits_perfectly(self) -> None:
        assert observed_order(_clean_sequence(2.0)).fit_quality == pytest.approx(1.0)

    def test_levels_may_be_supplied_in_any_order(self) -> None:
        levels = _clean_sequence(2.0)

        assert observed_order(list(reversed(levels))).observed_order == pytest.approx(
            observed_order(levels).observed_order
        )

    def test_rejects_fewer_than_two_levels(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            observed_order(_clean_sequence(2.0, n=1))

    def test_rejects_a_non_positive_mesh_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RefinementLevel(h=0.0, error=1.0)

    def test_rejects_a_negative_error(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            RefinementLevel(h=0.1, error=-1.0)

    def test_rejects_duplicate_mesh_sizes(self) -> None:
        # Two errors at the same h is not a refinement study; the log-log fit would be
        # vertical and the reported order meaningless rather than absent.
        with pytest.raises(ValueError, match="distinct"):
            observed_order([RefinementLevel(h=0.1, error=1.0), RefinementLevel(h=0.1, error=2.0)])


class TestTheRoundOffFloor:
    """The failure that makes a correct solver look broken.

    Below some mesh size the discretisation error drops beneath the accumulated
    floating-point error and the sequence stops converging. A study that includes those
    levels reports an order far below design and invites someone to go hunting for a bug
    in a solver that does not have one.
    """

    def test_detects_a_sequence_that_has_stalled(self) -> None:
        stalled = [
            RefinementLevel(h=0.1, error=3e-2),
            RefinementLevel(h=0.05, error=7.5e-3),
            RefinementLevel(h=0.025, error=1.9e-3),
            RefinementLevel(h=0.0125, error=1.8e-3),
            RefinementLevel(h=0.00625, error=1.85e-3),
        ]

        result = observed_order(stalled)

        assert result.has_stalled is True

    def test_a_clean_sequence_has_not_stalled(self) -> None:
        assert observed_order(_clean_sequence(2.0)).has_stalled is False

    def test_names_the_level_at_which_it_stalled(self) -> None:
        stalled = [
            RefinementLevel(h=0.1, error=1e-2),
            RefinementLevel(h=0.05, error=2.5e-3),
            RefinementLevel(h=0.025, error=2.4e-3),
        ]

        assert observed_order(stalled).stalled_from == pytest.approx(0.025)

    def test_the_asymptotic_range_excludes_stalled_levels(self) -> None:
        # The order that matters is the one over the levels that were still converging.
        # Fitting through the floor is how a second-order solver reports 1.2 and someone
        # spends a day looking for a bug that is not there.
        stalled = [
            RefinementLevel(h=0.1, error=1e-2),
            RefinementLevel(h=0.05, error=2.5e-3),
            RefinementLevel(h=0.025, error=6.25e-4),
            RefinementLevel(h=0.0125, error=6.2e-4),
        ]

        result = observed_order(stalled)

        assert result.asymptotic_order == pytest.approx(2.0, abs=0.05)
        assert result.asymptotic_order != pytest.approx(result.observed_order, abs=0.05)


class TestTheDesignOrderGate:
    def test_a_solver_at_its_design_order_passes(self) -> None:
        assert_design_order(observed_order(_clean_sequence(2.0)), design_order=2.0)

    def test_a_first_order_solver_designed_second_order_fails(self) -> None:
        # doc 07 §2.3, verbatim: "A solver converging at first order when second was
        # designed has a bug." This is the assertion that catches it.
        with pytest.raises(AssertionError, match=r"1\.0"):
            assert_design_order(observed_order(_clean_sequence(1.0)), design_order=2.0)

    def test_the_failure_message_names_both_orders_and_the_tolerance(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_design_order(observed_order(_clean_sequence(1.0)), design_order=2.0)

        message = str(excinfo.value)
        assert "2.0" in message
        assert "1.0" in message
        assert "0.1" in message

    def test_the_default_tolerance_is_the_doc_07_value(self) -> None:
        # doc 07 §2.3: "Design order ± 0.1 is required." Not a parameter with a
        # convenient default — the default *is* the specification.
        marginal = _clean_sequence(2.0)
        marginal[-1] = RefinementLevel(h=marginal[-1].h, error=marginal[-1].error * 1.5)

        with pytest.raises(AssertionError):
            assert_design_order(observed_order(marginal), design_order=2.0)

    def test_a_looser_tolerance_can_be_requested_explicitly(self) -> None:
        # doc 03 §7 gives V-02 (the coupled fluid system) ±0.15 rather than ±0.1.
        levels = _clean_sequence(2.0)
        levels[-1] = RefinementLevel(h=levels[-1].h, error=levels[-1].error * 1.2)

        assert_design_order(observed_order(levels), design_order=2.0, tolerance=0.15)

    def test_a_stalled_study_fails_loudly_rather_than_reporting_a_low_order(self) -> None:
        # Reporting "observed 1.1, expected 2.0" for a study that hit the round-off floor
        # is a true statement and a useless one. The gate says what actually happened.
        stalled = [
            RefinementLevel(h=0.1, error=1e-2),
            RefinementLevel(h=0.05, error=2.5e-3),
            RefinementLevel(h=0.025, error=2.4e-3),
            RefinementLevel(h=0.0125, error=2.45e-3),
        ]

        with pytest.raises(AssertionError, match=r"stalled|round-off"):
            assert_design_order(observed_order(stalled), design_order=2.0)


class TestErrorNorms:
    def test_weighted_l2_is_independent_of_grid_refinement(self) -> None:
        # The point of weighting by cell width. An unweighted discrete L2 grows with the
        # point count, so refining the mesh would appear to increase the error — and the
        # convergence study would report a negative order on a perfect solver.
        for n in (11, 101, 1001):
            z = np.linspace(0.0, 1.0, n)
            error = weighted_l2_error(np.sin(z), np.sin(z) + 0.1, weights=np.gradient(z))

            assert error == pytest.approx(0.1, rel=1e-6)

    def test_weighted_l2_of_an_exact_match_is_zero(self) -> None:
        z = np.linspace(0.0, 1.0, 21)

        assert weighted_l2_error(np.sin(z), np.sin(z), weights=np.gradient(z)) == 0.0

    def test_unit_weights_are_the_default(self) -> None:
        assert weighted_l2_error(np.zeros(4), np.ones(4)) == pytest.approx(1.0)

    def test_linf_is_the_worst_point_not_the_average(self) -> None:
        # A sheath solver can be excellent everywhere and wrong at the wall, which is the
        # only place the answer is wanted. An L2 norm hides that; Linf is what does not.
        numeric = np.zeros(100)
        numeric[0] = 5.0

        assert linf_error(numeric, np.zeros(100)) == pytest.approx(5.0)

    def test_relative_l2_normalises_by_the_exact_solution(self) -> None:
        exact = np.full(10, 4.0)

        assert relative_l2_error(np.full(10, 5.0), exact) == pytest.approx(0.25)

    def test_relative_l2_refuses_an_all_zero_reference(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            relative_l2_error(np.ones(4), np.zeros(4))

    def test_norms_reject_a_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            weighted_l2_error(np.zeros(4), np.zeros(5))


class TestRichardsonExtrapolation:
    @pytest.mark.physics
    def test_recovers_the_exact_value_from_two_resolutions(self) -> None:
        # doc 06 §3 uses Richardson to turn the convergence study into a discretisation
        # *bias* term with an uncertainty, rather than leaving it as an unquantified
        # "we refined until it stopped moving".
        exact = 6.577
        coarse = exact + 3.0 * 0.1**2
        fine = exact + 3.0 * 0.05**2

        assert richardson_extrapolate(coarse, fine, ratio=2.0, order=2.0) == pytest.approx(
            exact, abs=1e-12
        )

    def test_the_estimated_discretisation_error_is_the_correction_applied(self) -> None:
        coarse, fine = 1.04, 1.01
        extrapolated = richardson_extrapolate(coarse, fine, ratio=2.0, order=2.0)

        assert extrapolated - fine == pytest.approx((fine - coarse) / (2.0**2 - 1.0))

    def test_rejects_a_refinement_ratio_of_one(self) -> None:
        with pytest.raises(ValueError, match="ratio"):
            richardson_extrapolate(1.0, 1.0, ratio=1.0, order=2.0)

    def test_rejects_a_non_positive_order(self) -> None:
        with pytest.raises(ValueError, match="order"):
            richardson_extrapolate(1.0, 1.0, ratio=2.0, order=0.0)


class TestConvergenceResultReporting:
    def test_is_printable_as_a_table(self) -> None:
        # These go into the doc 07 §2.3 report and into CI output. A result nobody can
        # read at a glance gets skimmed, and a skimmed convergence table is no gate.
        rendered = str(observed_order(_clean_sequence(2.0, n=3)))

        assert "order" in rendered.lower()
        assert "0.1" in rendered

    def test_is_immutable(self) -> None:
        result = observed_order(_clean_sequence(2.0))

        with pytest.raises(AttributeError):
            result.observed_order = 1.0  # type: ignore[misc]

    def test_carries_every_level_it_was_given(self) -> None:
        result: ConvergenceResult = observed_order(_clean_sequence(2.0, n=4))

        assert len(result.levels) == 4
