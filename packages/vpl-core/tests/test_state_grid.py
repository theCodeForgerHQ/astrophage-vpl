"""Grid types — the coordinate frame of doc 02 §2 and the meshes of doc 03 §3.4.

The frame is normative: the wall sits at ``z = 0`` and ``z`` is positive *into* the
plasma. Doc 02 §2 records that registration errors between channels are among the most
damaging and least modelled systematics in multi-diagnostic fusion, so the grid refuses
coordinates that do not obey the frame rather than quietly accepting them.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.state import PhaseGrid, SpatialGrid, TimeGrid
from vpl.core.units import Q_, DimensionalityError, magnitude_in


class TestSpatialGridConstruction:
    def test_builds_from_a_length_quantity(self) -> None:
        grid = SpatialGrid.from_quantity(Q_(np.array([0.0, 1.0, 2.0]), "mm"))

        np.testing.assert_allclose(grid.z_m, [0.0, 1e-3, 2e-3])

    def test_exposes_coordinates_as_a_quantity(self) -> None:
        grid = SpatialGrid.from_quantity(Q_(np.array([0.0, 1.0]), "mm"))

        np.testing.assert_allclose(magnitude_in(grid.z, "mm"), [0.0, 1.0])

    def test_rejects_a_non_length_quantity(self) -> None:
        with pytest.raises(DimensionalityError):
            SpatialGrid.from_quantity(Q_(np.array([0.0, 1.0]), "s"))

    def test_uniform_spans_wall_to_far_boundary(self) -> None:
        grid = SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=5)

        np.testing.assert_allclose(magnitude_in(grid.z, "mm"), [0.0, 5.0, 10.0, 15.0, 20.0])

    def test_geometric_refines_toward_the_wall(self) -> None:
        # doc 03 §3.4: lambda_D at the wall is the controlling scale, so a uniform mesh
        # wastes ~90% of its cells. The first cell must be the smallest.
        grid = SpatialGrid.geometric(length=Q_(20.0, "mm"), n_points=64, stretch=1.05)

        dz = grid.dz_m
        assert dz[0] < dz[-1]
        np.testing.assert_allclose(dz[1:] / dz[:-1], 1.05, rtol=1e-9)

    def test_geometric_still_spans_the_full_domain(self) -> None:
        grid = SpatialGrid.geometric(length=Q_(20.0, "mm"), n_points=64, stretch=1.05)

        assert magnitude_in(grid.z, "mm")[0] == pytest.approx(0.0)
        assert magnitude_in(grid.z, "mm")[-1] == pytest.approx(20.0)

    def test_geometric_with_unit_stretch_is_uniform(self) -> None:
        # The degenerate case must not divide by zero.
        graded = SpatialGrid.geometric(length=Q_(10.0, "mm"), n_points=11, stretch=1.0)
        uniform = SpatialGrid.uniform(length=Q_(10.0, "mm"), n_points=11)

        np.testing.assert_allclose(graded.z_m, uniform.z_m)


class TestSpatialGridInvariants:
    def test_rejects_a_non_monotonic_grid(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            SpatialGrid.from_quantity(Q_(np.array([0.0, 2.0, 1.0]), "mm"))

    def test_rejects_duplicate_coordinates(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            SpatialGrid.from_quantity(Q_(np.array([0.0, 1.0, 1.0]), "mm"))

    def test_rejects_fewer_than_two_points(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            SpatialGrid.from_quantity(Q_(np.array([0.0]), "mm"))

    def test_rejects_coordinates_behind_the_wall(self) -> None:
        # doc 02 §2: the wall is at z = 0 and z is positive into the plasma. A negative
        # coordinate is inside the electrode.
        with pytest.raises(ValueError, match="behind the wall"):
            SpatialGrid.from_quantity(Q_(np.array([-1.0, 0.0, 1.0]), "mm"))

    def test_rejects_a_multidimensional_array(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            SpatialGrid.from_quantity(Q_(np.zeros((2, 2)), "mm"))

    def test_rejects_non_finite_coordinates(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            SpatialGrid.from_quantity(Q_(np.array([0.0, np.inf]), "mm"))


class TestSpatialGridImmutability:
    def test_the_underlying_array_cannot_be_written(self) -> None:
        # A frozen dataclass stops attribute rebinding, not array mutation. Without the
        # write lock a caller could silently move the mesh under a solver that had
        # already cached its spacings.
        grid = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=4)

        with pytest.raises(ValueError, match="read-only"):
            grid.z_m[0] = 5.0

    def test_the_field_cannot_be_rebound(self) -> None:
        grid = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=4)

        with pytest.raises(AttributeError):
            grid.z_m = np.zeros(4)  # type: ignore[misc]

    def test_construction_copies_the_caller_array(self) -> None:
        source = np.array([0.0, 1.0, 2.0])
        grid = SpatialGrid.from_quantity(Q_(source, "mm"))

        source[1] = 99.0

        np.testing.assert_allclose(grid.z_m, [0.0, 1e-3, 2e-3])


class TestSpatialGridDerivedQuantities:
    def test_reports_point_and_cell_counts(self) -> None:
        grid = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=11)

        assert grid.n_points == 11
        assert grid.n_cells == 10

    def test_cell_widths_are_the_coordinate_differences(self) -> None:
        grid = SpatialGrid.from_quantity(Q_(np.array([0.0, 1.0, 3.0]), "mm"))

        np.testing.assert_allclose(grid.dz_m, [1e-3, 2e-3])

    def test_reports_the_domain_length(self) -> None:
        grid = SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=5)

        assert magnitude_in(grid.length, "mm") == pytest.approx(20.0)

    def test_reports_the_smallest_cell(self) -> None:
        # The stability constraints of doc 03 §4.3 are checked against the *smallest*
        # cell, not the mean; a mean-based check passes a mesh that is locally illegal.
        grid = SpatialGrid.geometric(length=Q_(20.0, "mm"), n_points=64, stretch=1.05)

        assert magnitude_in(grid.min_dz, "m") == pytest.approx(grid.dz_m.min())

    def test_equality_compares_coordinates(self) -> None:
        a = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=5)
        b = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=5)
        c = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=6)

        assert a == b
        assert a != c
        assert a != "not a grid"


class TestTimeGrid:
    def test_builds_from_a_time_quantity(self) -> None:
        grid = TimeGrid.from_quantity(Q_(np.array([0.0, 1.0, 2.0]), "ns"))

        np.testing.assert_allclose(grid.t_s, [0.0, 1e-9, 2e-9])

    def test_uniform_spans_the_requested_duration(self) -> None:
        grid = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=3)

        np.testing.assert_allclose(magnitude_in(grid.t, "ns"), [0.0, 36.85, 73.7])

    def test_rejects_a_non_time_quantity(self) -> None:
        with pytest.raises(DimensionalityError):
            TimeGrid.from_quantity(Q_(np.array([0.0, 1.0]), "m"))

    def test_rejects_a_non_monotonic_grid(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            TimeGrid.from_quantity(Q_(np.array([0.0, 2.0, 1.0]), "ns"))

    def test_allows_negative_times(self) -> None:
        # Unlike z, t = 0 carries no privileged meaning: a pre-trigger baseline window
        # is a legitimate acquisition (doc 02 §10.2).
        grid = TimeGrid.from_quantity(Q_(np.array([-5.0, 0.0, 5.0]), "ns"))

        np.testing.assert_allclose(magnitude_in(grid.t, "ns"), [-5.0, 0.0, 5.0])

    def test_is_immutable(self) -> None:
        grid = TimeGrid.uniform(duration=Q_(1.0, "ns"), n_points=4)

        with pytest.raises(ValueError, match="read-only"):
            grid.t_s[0] = 1.0


class TestPhaseGrid:
    def test_divides_the_rf_period_into_bins(self) -> None:
        # doc 02 §10.3: N_phi = 16 bins of 4.6 ns each over the 73.7 ns RF period.
        grid = PhaseGrid(n_bins=16, period=Q_(73.7, "ns"))

        assert grid.n_bins == 16
        assert magnitude_in(grid.bin_width, "ns") == pytest.approx(73.7 / 16)

    def test_bin_centres_span_zero_to_two_pi(self) -> None:
        grid = PhaseGrid(n_bins=4, period=Q_(73.7, "ns"))

        np.testing.assert_allclose(grid.centres_rad, np.pi / 4 * np.array([1.0, 3.0, 5.0, 7.0]))

    def test_bin_index_wraps_a_phase_into_range(self) -> None:
        grid = PhaseGrid(n_bins=4, period=Q_(73.7, "ns"))

        assert grid.bin_index(0.0) == 0
        assert grid.bin_index(2 * np.pi - 1e-12) == 3
        # Cyclo-stationarity (doc 02 §10.3) means phase is periodic: a phase one full
        # cycle on lands in the same bin, and an accumulation that got this wrong would
        # smear the phase-resolved result it exists to produce.
        assert grid.bin_index(2 * np.pi + 0.1) == grid.bin_index(0.1)
        assert grid.bin_index(-0.1) == 3

    def test_rejects_a_non_positive_bin_count(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PhaseGrid(n_bins=0, period=Q_(73.7, "ns"))

    def test_rejects_a_non_positive_period(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PhaseGrid(n_bins=16, period=Q_(0.0, "ns"))

    def test_rejects_a_non_time_period(self) -> None:
        with pytest.raises(DimensionalityError):
            PhaseGrid(n_bins=16, period=Q_(1.0, "m"))
