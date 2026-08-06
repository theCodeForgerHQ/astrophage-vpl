"""Tests for the identifiability map — doc 00 S5, doc 05 §6.

Every test here is built against a Fisher/curvature matrix whose eigenstructure is known
independently of the module under test — a diagonal matrix, a rank-deficient matrix with an
exactly known null vector, and doc 05 §6.2's predicted `n_0`-`T_e` degeneracy. That is the
only kind of check worth having for an eigen-decomposition: comparing against a second
eigen-decomposition would just be comparing the module to itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.validation.identifiability import (
    STRONG_IDENTIFIABILITY_FLOOR,
    WEAK_IDENTIFIABILITY_FLOOR,
    IdentifiabilityVerdict,
    classify_identifiability,
    compare_to_cramer_rao_bound,
    cramer_rao_bound,
    fisher_information_from_posterior,
)


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


class TestADiagonalMatrixWithKnownEntries:
    """The trivial eigenstructure: eigenvalues are the diagonal, eigenvectors are the
    standard basis. Chosen so every one of doc 00 S5's three verdicts appears once."""

    def _classify(self):
        # Relative curvatures against the largest (1e6): 1e-8, 1e-3, 1.
        fisher = np.diag([1.0e-2, 1.0e3, 1.0e6])
        return classify_identifiability(fisher, ["a", "b", "c"])

    def test_eigenvalues_are_recovered_exactly_and_ascending(self) -> None:
        result = self._classify()

        assert result.eigenvalues == pytest.approx([1.0e-2, 1.0e3, 1.0e6])

    def test_the_weakest_direction_is_non_identifiable(self) -> None:
        result = self._classify()

        weakest = result.directions[0]
        assert weakest.eigenvalue == pytest.approx(1.0e-2)
        assert weakest.verdict is IdentifiabilityVerdict.NON_IDENTIFIABLE
        assert weakest.combination == "1.00·a"

    def test_the_middle_direction_is_weakly_identifiable(self) -> None:
        result = self._classify()

        middle = result.directions[1]
        assert middle.eigenvalue == pytest.approx(1.0e3)
        assert middle.verdict is IdentifiabilityVerdict.WEAKLY_IDENTIFIABLE
        assert middle.combination == "1.00·b"

    def test_the_strongest_direction_is_identifiable(self) -> None:
        result = self._classify()

        strongest = result.directions[2]
        assert strongest.eigenvalue == pytest.approx(1.0e6)
        assert strongest.verdict is IdentifiabilityVerdict.IDENTIFIABLE
        assert strongest.combination == "1.00·c"

    def test_condition_number_is_the_ratio_of_extreme_eigenvalues(self) -> None:
        result = self._classify()

        assert result.condition_number == pytest.approx(1.0e8)

    def test_partition_properties_group_by_verdict(self) -> None:
        result = self._classify()

        assert [d.eigenvalue for d in result.non_identifiable] == pytest.approx([1.0e-2])
        assert [d.eigenvalue for d in result.weakly_identifiable] == pytest.approx([1.0e3])
        assert [d.eigenvalue for d in result.identifiable] == pytest.approx([1.0e6])

    def test_cramer_rao_bound_matches_the_diagonal_by_hand(self) -> None:
        # 1e-2's relative curvature (1e-8) is below WEAK_IDENTIFIABILITY_FLOOR, so its
        # bound is reported as infinite rather than as the finite-but-meaningless
        # 1/sqrt(1e-2) numpy's inverse would silently produce — the same discipline
        # laplace.py applies to the posterior covariance itself.
        fisher = np.diag([1.0e-2, 1.0e3, 1.0e6])

        bound = cramer_rao_bound(fisher)

        assert math.isinf(bound[0])
        assert bound[1] == pytest.approx(1.0 / math.sqrt(1.0e3))
        assert bound[2] == pytest.approx(1.0 / math.sqrt(1.0e6))


class TestARankDeficientMatrixWithAnExactlyKnownNullVector:
    """`fisher = lam * outer(v, v)` is exactly rank 1 by construction: one eigenvalue is
    `lam`, along `v`; the other is exactly 0, along the vector orthogonal to `v`."""

    def _fisher_and_directions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measured = _unit([3.0, 4.0])
        null = _unit([-4.0, 3.0])
        fisher = 100.0 * np.outer(measured, measured)
        return fisher, measured, null

    def test_the_measured_direction_is_identifiable_with_the_right_eigenvalue(self) -> None:
        fisher, measured, _null = self._fisher_and_directions()

        result = classify_identifiability(fisher, ["x", "y"])

        strongest = result.directions[-1]
        assert strongest.eigenvalue == pytest.approx(100.0)
        assert strongest.verdict is IdentifiabilityVerdict.IDENTIFIABLE
        assert abs(float(np.dot(strongest.eigenvector, measured))) == pytest.approx(1.0)

    def test_the_null_direction_is_non_identifiable_with_zero_eigenvalue(self) -> None:
        fisher, _measured, null = self._fisher_and_directions()

        result = classify_identifiability(fisher, ["x", "y"])

        weakest = result.directions[0]
        assert weakest.eigenvalue == pytest.approx(0.0, abs=1e-9)
        assert weakest.verdict is IdentifiabilityVerdict.NON_IDENTIFIABLE
        assert abs(float(np.dot(weakest.eigenvector, null))) == pytest.approx(1.0, abs=1e-6)

    def test_the_cramer_rao_bound_is_infinite_in_both_coordinates(self) -> None:
        # The null direction has a nonzero component along *both* axes here (the null
        # vector is (-0.8, 0.6)), so the singular direction contaminates the bound for
        # every coordinate that touches it, not just one.
        fisher, _measured, _null = self._fisher_and_directions()

        bound = cramer_rao_bound(fisher)

        assert math.isinf(bound[0])
        assert math.isinf(bound[1])


class TestTheDoc05Section62AnchorDegeneracy:
    """`Gamma_i ~ n_0 sqrt(T_e)` means the data constrain `log n_0 + 0.5 log T_e` far
    better than either factor alone. This is doc 05 §6.2's own predicted example, and the
    module's central claim to usefulness rests on recovering it from a Fisher matrix built
    to have exactly that structure."""

    def _classify(self):
        well_constrained = _unit([1.0, 0.5])  # log n_0 + 0.5 log T_e
        orthogonal = _unit([0.5, -1.0])  # the complementary, poorly constrained combination
        fisher = 1.0e6 * np.outer(well_constrained, well_constrained) + 10.0 * np.outer(
            orthogonal, orthogonal
        )
        return classify_identifiability(fisher, ["log n_0", "log T_e"])

    def test_the_n0_te_product_direction_is_identifiable(self) -> None:
        result = self._classify()

        strongest = result.directions[-1]
        assert strongest.eigenvalue == pytest.approx(1.0e6)
        assert strongest.verdict is IdentifiabilityVerdict.IDENTIFIABLE

    def test_the_identifiable_combination_is_named_correctly(self) -> None:
        result = self._classify()

        strongest = result.directions[-1]
        assert strongest.combination == "0.89·log n_0 + 0.45·log T_e"

    def test_the_orthogonal_combination_is_weakly_identifiable(self) -> None:
        result = self._classify()

        weakest = result.directions[0]
        assert weakest.eigenvalue == pytest.approx(10.0)
        assert weakest.verdict is IdentifiabilityVerdict.WEAKLY_IDENTIFIABLE

    def test_the_weak_combination_is_named_correctly(self) -> None:
        result = self._classify()

        weakest = result.directions[0]
        assert weakest.combination == "-0.45·log n_0 + 0.89·log T_e"


class TestPriorCurvatureDoesNotMasqueradeAsIdentifiability:
    """The distinction the task exists to get right: classifying a posterior Hessian
    directly can report a direction as identifiable when only the *prior* constrains it.
    doc 05 §6.1's Fisher information is a statement about the data, and
    fisher_information_from_posterior recovers exactly that by subtracting the prior's own
    curvature."""

    def _matrices(self) -> tuple[np.ndarray, np.ndarray]:
        measured_combination = _unit([1.0, 1.0])
        unmeasured_combination = _unit([1.0, -1.0])
        # The data constrain only one combination; the orthogonal one has exactly zero
        # data curvature.
        fisher_data = 1.0e6 * np.outer(measured_combination, measured_combination)
        # The prior is informative *only* along the direction the data cannot see —
        # e.g. doc 05 §2.1's 1 % wall-bias prior sitting on an otherwise-flat likelihood
        # direction.
        prior_hessian = 1.0e4 * np.outer(unmeasured_combination, unmeasured_combination)
        return fisher_data, prior_hessian

    def test_classifying_the_raw_posterior_hessian_wrongly_looks_fully_identifiable(self) -> None:
        fisher_data, prior_hessian = self._matrices()
        posterior_hessian = fisher_data + prior_hessian

        # This is the mistake the task warns against, demonstrated: taken naively, both
        # directions clear the identifiable floor because the prior supplied the
        # missing curvature, not the data.
        result = classify_identifiability(posterior_hessian, ["x", "y"])

        assert all(d.verdict is IdentifiabilityVerdict.IDENTIFIABLE for d in result.directions)

    def test_subtracting_the_prior_reveals_the_true_data_only_identifiability(self) -> None:
        fisher_data, prior_hessian = self._matrices()
        posterior_hessian = fisher_data + prior_hessian

        recovered = fisher_information_from_posterior(posterior_hessian, prior_hessian)
        result = classify_identifiability(recovered, ["x", "y"])

        assert result.directions[0].verdict is IdentifiabilityVerdict.NON_IDENTIFIABLE
        assert result.directions[0].eigenvalue == pytest.approx(0.0, abs=1e-6)
        assert result.directions[-1].verdict is IdentifiabilityVerdict.IDENTIFIABLE
        assert result.directions[-1].eigenvalue == pytest.approx(1.0e6)

    def test_the_recovered_fisher_information_matches_the_data_matrix_exactly(self) -> None:
        fisher_data, prior_hessian = self._matrices()
        posterior_hessian = fisher_data + prior_hessian

        recovered = fisher_information_from_posterior(posterior_hessian, prior_hessian)

        assert recovered == pytest.approx(fisher_data)

    def test_mismatched_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            fisher_information_from_posterior(np.eye(2), np.eye(3))


class TestCramerRaoComparison:
    def test_an_efficient_estimator_has_efficiency_one(self) -> None:
        # achieved_std built to equal the bound exactly, by construction.
        fisher = np.diag([100.0, 400.0])
        achieved_std = 1.0 / np.sqrt(np.diag(fisher))

        comparison = compare_to_cramer_rao_bound(fisher, achieved_std, ["a", "b"])

        assert comparison.efficiency == pytest.approx([1.0, 1.0])

    def test_a_wider_than_necessary_posterior_has_efficiency_below_one(self) -> None:
        fisher = np.diag([100.0, 400.0])
        bound = 1.0 / np.sqrt(np.diag(fisher))
        achieved_std = 2.0 * bound  # twice as wide as the data alone would demand

        comparison = compare_to_cramer_rao_bound(fisher, achieved_std, ["a", "b"])

        assert comparison.efficiency == pytest.approx([0.5, 0.5])

    def test_mismatched_lengths_are_refused(self) -> None:
        fisher = np.diag([100.0, 400.0])

        with pytest.raises(ValueError, match="shape"):
            compare_to_cramer_rao_bound(fisher, np.array([0.1, 0.1, 0.1]), ["a", "b"])

    def test_a_non_positive_achieved_std_is_refused(self) -> None:
        fisher = np.diag([100.0, 400.0])

        with pytest.raises(ValueError, match="positive"):
            compare_to_cramer_rao_bound(fisher, np.array([0.1, 0.0]), ["a", "b"])


class TestInputValidation:
    def test_a_non_square_matrix_is_refused(self) -> None:
        with pytest.raises(ValueError, match="square"):
            classify_identifiability(np.zeros((2, 3)), ["a", "b"])

    def test_a_mismatched_name_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names"):
            classify_identifiability(np.eye(2), ["a", "b", "c"])

    def test_a_matrix_with_no_positive_curvature_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            classify_identifiability(-np.eye(2), ["a", "b"])


class TestTheThresholdsAreNamedAndOrdered:
    def test_the_strong_floor_is_stricter_than_the_weak_floor(self) -> None:
        # Structural: doc 00 S5's three-way partition requires this ordering to make
        # sense as a partition at all.
        assert WEAK_IDENTIFIABILITY_FLOOR < STRONG_IDENTIFIABILITY_FLOOR
