"""The energy grid the two-term solve lives on — doc 03 §3.2.

Every quantity doc 03 §3.2 tabulates is an integral of ``f0`` against a weight, so the
grid and the quadrature are not separable concerns: if ``moment`` and the finite-volume
discretisation disagree about what a cell contains, the normalisation drifts and every
rate coefficient drifts with it. These tests pin the quadrature to the same
piecewise-constant reconstruction the solver conserves against.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.physics.eedf.grid import EnergyGrid


class TestConstruction:
    def test_a_linear_grid_spans_zero_to_the_requested_maximum(self) -> None:
        grid = EnergyGrid.linear(max_ev=50.0, n_cells=100)

        assert grid.n_cells == 100
        assert grid.boundaries_ev[0] == 0.0
        assert grid.boundaries_ev[-1] == pytest.approx(50.0)
        assert grid.boundaries_ev.size == 101

    def test_linear_cells_are_all_the_same_width(self) -> None:
        grid = EnergyGrid.linear(max_ev=10.0, n_cells=20)

        np.testing.assert_allclose(grid.widths_ev, np.full(20, 0.5))

    def test_a_quadratic_grid_is_finer_at_low_energy(self) -> None:
        grid = EnergyGrid.quadratic(max_ev=50.0, n_cells=100)

        assert grid.widths_ev[0] < grid.widths_ev[-1]
        assert grid.boundaries_ev[0] == 0.0
        assert grid.boundaries_ev[-1] == pytest.approx(50.0)

    def test_boundaries_are_strictly_increasing(self) -> None:
        for grid in (
            EnergyGrid.linear(max_ev=30.0, n_cells=64),
            EnergyGrid.quadratic(max_ev=30.0, n_cells=64),
        ):
            assert np.all(np.diff(grid.boundaries_ev) > 0.0)

    def test_centres_lie_inside_their_cells(self) -> None:
        grid = EnergyGrid.quadratic(max_ev=30.0, n_cells=40)

        assert np.all(grid.centres_ev > grid.boundaries_ev[:-1])
        assert np.all(grid.centres_ev < grid.boundaries_ev[1:])

    def test_the_arrays_are_read_only(self) -> None:
        grid = EnergyGrid.linear(max_ev=10.0, n_cells=8)

        with pytest.raises(ValueError, match="read-only"):
            grid.boundaries_ev[0] = 1.0

    @pytest.mark.parametrize("n_cells", [0, -3])
    def test_a_grid_needs_at_least_one_cell(self, n_cells: int) -> None:
        with pytest.raises(ValueError, match="at least one cell"):
            EnergyGrid.linear(max_ev=10.0, n_cells=n_cells)

    @pytest.mark.parametrize("max_ev", [0.0, -1.0])
    def test_the_maximum_energy_must_be_positive(self, max_ev: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            EnergyGrid.linear(max_ev=max_ev, n_cells=8)

    def test_explicit_boundaries_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            EnergyGrid(boundaries_ev=np.array([0.5, 1.0, 2.0]))

    def test_explicit_boundaries_must_increase(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            EnergyGrid(boundaries_ev=np.array([0.0, 2.0, 1.0]))


class TestQuadrature:
    """``moment(f0, k) = integral eps**k f0 sqrt(eps) deps``, exact for cellwise-constant f0.

    Exactness matters more than accuracy here. The solver conserves particles against
    this same reconstruction, so a quadrature that were merely accurate would leave the
    normalisation ``integral f0 sqrt(eps) deps = 1`` true only to the quadrature error —
    and doc 03 §3.2's rate coefficients are all normalised by exactly that integral.
    """

    @pytest.mark.physics
    def test_the_zeroth_moment_of_a_constant_is_exact(self) -> None:
        grid = EnergyGrid.linear(max_ev=4.0, n_cells=7)
        f0 = np.full(grid.n_cells, 3.0)

        # integral_0^4 3 sqrt(eps) deps = 3 * (2/3) * 4**1.5 = 16
        assert grid.moment(f0, 0) == pytest.approx(16.0)

    @pytest.mark.physics
    def test_the_first_moment_of_a_constant_is_exact(self) -> None:
        grid = EnergyGrid.quadratic(max_ev=4.0, n_cells=9)
        f0 = np.full(grid.n_cells, 1.0)

        # integral_0^4 eps**1.5 deps = (2/5) * 4**2.5 = 12.8
        assert grid.moment(f0, 1) == pytest.approx(12.8)

    def test_the_cell_masses_sum_to_the_zeroth_moment_weight(self) -> None:
        grid = EnergyGrid.quadratic(max_ev=9.0, n_cells=32)

        assert float(grid.cell_masses.sum()) == pytest.approx((2.0 / 3.0) * 9.0**1.5)

    def test_normalise_makes_the_zeroth_moment_exactly_one(self) -> None:
        grid = EnergyGrid.linear(max_ev=20.0, n_cells=50)
        rough = np.exp(-grid.centres_ev / 3.0)

        normalised = grid.normalise(rough)

        assert grid.moment(normalised, 0) == pytest.approx(1.0, rel=1e-14)

    def test_normalising_a_vanishing_distribution_is_refused(self) -> None:
        grid = EnergyGrid.linear(max_ev=10.0, n_cells=8)

        with pytest.raises(ValueError, match="zero"):
            grid.normalise(np.zeros(grid.n_cells))

    def test_a_wrongly_shaped_distribution_is_refused(self) -> None:
        grid = EnergyGrid.linear(max_ev=10.0, n_cells=8)

        with pytest.raises(ValueError, match="one value per cell"):
            grid.moment(np.ones(7), 0)


class TestRefinement:
    def test_refine_halves_every_cell_width(self) -> None:
        grid = EnergyGrid.linear(max_ev=10.0, n_cells=8)

        finer = grid.refined(2)

        assert finer.n_cells == 16
        assert finer.max_ev == pytest.approx(grid.max_ev)

    def test_refine_preserves_the_grid_family(self) -> None:
        grid = EnergyGrid.quadratic(max_ev=10.0, n_cells=8)

        finer = grid.refined(2)

        # A quadratic grid refined must stay finer at the bottom than at the top.
        assert finer.widths_ev[0] < finer.widths_ev[-1]

    def test_refining_by_one_returns_the_same_boundaries(self) -> None:
        grid = EnergyGrid.quadratic(max_ev=10.0, n_cells=8)

        np.testing.assert_allclose(grid.refined(1).boundaries_ev, grid.boundaries_ev)

    def test_refining_by_a_non_positive_factor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            EnergyGrid.linear(max_ev=10.0, n_cells=8).refined(0)


class TestRejectedInputs:
    """The grid is read by every integral in a sweep, so it is validated once, hard."""

    def test_a_two_dimensional_boundary_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            EnergyGrid(boundaries_ev=np.zeros((2, 2)))

    def test_a_non_finite_boundary_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            EnergyGrid(boundaries_ev=np.array([0.0, np.inf]))

    def test_a_single_boundary_is_not_a_grid(self) -> None:
        with pytest.raises(ValueError, match="at least one cell"):
            EnergyGrid(boundaries_ev=np.array([0.0]))


class TestPresentation:
    def test_the_length_is_the_cell_count(self) -> None:
        assert len(EnergyGrid.linear(max_ev=10.0, n_cells=17)) == 17

    def test_the_repr_names_the_family_and_the_range(self) -> None:
        assert "linear" in repr(EnergyGrid.linear(max_ev=10.0, n_cells=8))
        assert "graded" in repr(EnergyGrid.quadratic(max_ev=10.0, n_cells=8))
        assert "10 eV" in repr(EnergyGrid.linear(max_ev=10.0, n_cells=8))
