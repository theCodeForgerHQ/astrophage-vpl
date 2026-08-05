"""Scalar fields and velocity distributions.

These are the payload of a :class:`PlasmaState`: what a solver produces and what an
instrument's forward model reads. doc 03 §5.2 names the set precisely — ``n_e(z)``,
``n_i(z)``, ``Phi(z)``, ``f_i(v_z; z=0)``, ``T_e(z)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.state import ScalarField, SpatialGrid, Species, TimeGrid, VelocityDistribution
from vpl.core.units import Q_, DimensionalityError, magnitude_in


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=5)


@pytest.fixture
def argon() -> Species:
    return Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)


class TestScalarFieldConstruction:
    def test_holds_values_on_a_grid(self, grid: SpatialGrid) -> None:
        field = ScalarField(
            name="n_e", values=np.full(5, 1e17), units="m**-3", grid=grid, time=None
        )

        assert field.name == "n_e"
        np.testing.assert_allclose(field.values, 1e17)

    def test_exposes_values_as_a_quantity(self, grid: SpatialGrid) -> None:
        field = ScalarField(name="Phi", values=np.zeros(5), units="V", grid=grid, time=None)

        assert magnitude_in(field.quantity, "V").shape == (5,)

    def test_rejects_a_length_mismatch_against_the_grid(self, grid: SpatialGrid) -> None:
        with pytest.raises(ValueError, match="shape"):
            ScalarField(name="n_e", values=np.zeros(4), units="m**-3", grid=grid, time=None)

    def test_rejects_units_it_cannot_parse(self, grid: SpatialGrid) -> None:
        with pytest.raises(DimensionalityError, match="units"):
            ScalarField(name="n_e", values=np.zeros(5), units="wombats", grid=grid, time=None)

    def test_rejects_non_finite_values(self, grid: SpatialGrid) -> None:
        values = np.zeros(5)
        values[2] = np.nan

        with pytest.raises(ValueError, match="finite"):
            ScalarField(name="n_e", values=values, units="m**-3", grid=grid, time=None)

    def test_rejects_an_empty_name(self, grid: SpatialGrid) -> None:
        with pytest.raises(ValueError, match="name"):
            ScalarField(name="", values=np.zeros(5), units="m**-3", grid=grid, time=None)


class TestScalarFieldTimeDependence:
    def test_a_field_without_a_time_grid_is_steady(self, grid: SpatialGrid) -> None:
        field = ScalarField(name="n_e", values=np.zeros(5), units="m**-3", grid=grid, time=None)

        assert field.is_steady is True

    def test_a_time_dependent_field_is_indexed_time_first(self, grid: SpatialGrid) -> None:
        # (n_t, n_z), not (n_z, n_t). Fixed once here because a transposed field is a
        # silent, shape-compatible disaster whenever n_t happens to equal n_z.
        time = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=3)
        field = ScalarField(
            name="n_e", values=np.zeros((3, 5)), units="m**-3", grid=grid, time=time
        )

        assert field.is_steady is False
        assert field.values.shape == (3, 5)

    def test_rejects_a_time_dependent_shape_mismatch(self, grid: SpatialGrid) -> None:
        time = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=3)

        with pytest.raises(ValueError, match="shape"):
            ScalarField(name="n_e", values=np.zeros((5, 3)), units="m**-3", grid=grid, time=time)

    def test_a_time_slice_is_a_steady_field(self, grid: SpatialGrid) -> None:
        time = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=3)
        values = np.arange(15, dtype=float).reshape(3, 5)
        field = ScalarField(name="n_e", values=values, units="m**-3", grid=grid, time=time)

        slice_ = field.at_time_index(1)

        assert slice_.is_steady is True
        np.testing.assert_allclose(slice_.values, [5.0, 6.0, 7.0, 8.0, 9.0])

    def test_slicing_a_steady_field_is_an_error(self, grid: SpatialGrid) -> None:
        field = ScalarField(name="n_e", values=np.zeros(5), units="m**-3", grid=grid, time=None)

        with pytest.raises(ValueError, match="steady"):
            field.at_time_index(0)


class TestScalarFieldImmutability:
    def test_values_cannot_be_written(self, grid: SpatialGrid) -> None:
        field = ScalarField(name="n_e", values=np.zeros(5), units="m**-3", grid=grid, time=None)

        with pytest.raises(ValueError, match="read-only"):
            field.values[0] = 1.0

    def test_construction_copies_the_caller_array(self, grid: SpatialGrid) -> None:
        source = np.zeros(5)
        field = ScalarField(name="n_e", values=source, units="m**-3", grid=grid, time=None)

        source[0] = 99.0

        assert field.values[0] == 0.0


def _drifting_beam(
    grid: SpatialGrid, argon: Species, *, density: float, drift: float, spread: float
) -> VelocityDistribution:
    """A Maxwellian drifting at ``drift`` m/s, identical at every z."""
    v = np.linspace(drift - 8.0 * spread, drift + 8.0 * spread, 601)
    profile = np.exp(-((v - drift) ** 2) / (2.0 * spread**2)) / (spread * np.sqrt(2.0 * np.pi))
    return VelocityDistribution(
        grid=grid,
        v_m_per_s=v,
        values=np.tile(density * profile, (grid.n_points, 1)),
        species=argon,
    )


class TestVelocityDistributionConstruction:
    def test_holds_a_distribution_over_position_and_velocity(
        self, grid: SpatialGrid, argon: Species
    ) -> None:
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=-3000.0, spread=200.0)

        assert ivdf.values.shape == (grid.n_points, ivdf.v_m_per_s.size)

    def test_rejects_a_shape_mismatch(self, grid: SpatialGrid, argon: Species) -> None:
        with pytest.raises(ValueError, match="shape"):
            VelocityDistribution(
                grid=grid,
                v_m_per_s=np.linspace(-1.0, 1.0, 7),
                values=np.zeros((grid.n_points, 5)),
                species=argon,
            )

    def test_rejects_a_non_monotonic_velocity_axis(self, grid: SpatialGrid, argon: Species) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            VelocityDistribution(
                grid=grid,
                v_m_per_s=np.array([0.0, 2.0, 1.0]),
                values=np.zeros((grid.n_points, 3)),
                species=argon,
            )

    def test_rejects_negative_values(self, grid: SpatialGrid, argon: Species) -> None:
        # A distribution function is a density in phase space. A negative entry is not a
        # small numerical wobble to tolerate; it means whatever produced it is wrong,
        # and every moment computed from it afterwards is meaningless.
        values = np.zeros((grid.n_points, 3))
        values[0, 1] = -1.0

        with pytest.raises(ValueError, match="negative"):
            VelocityDistribution(
                grid=grid,
                v_m_per_s=np.array([-1.0, 0.0, 1.0]),
                values=values,
                species=argon,
            )

    def test_is_immutable(self, grid: SpatialGrid, argon: Species) -> None:
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=-3000.0, spread=200.0)

        with pytest.raises(ValueError, match="read-only"):
            ivdf.values[0, 0] = 1.0


class TestVelocityDistributionMoments:
    def test_zeroth_moment_recovers_the_density(self, grid: SpatialGrid, argon: Species) -> None:
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=-3000.0, spread=200.0)

        np.testing.assert_allclose(ivdf.density_per_m3(), 1e17, rtol=1e-6)

    def test_first_moment_recovers_the_drift(self, grid: SpatialGrid, argon: Species) -> None:
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=-3000.0, spread=200.0)

        np.testing.assert_allclose(ivdf.mean_velocity_m_per_s(), -3000.0, rtol=1e-6)

    def test_particle_flux_toward_the_wall_is_positive_for_ions_moving_inward(
        self, grid: SpatialGrid, argon: Species
    ) -> None:
        # doc 02 §2 fixes the sign: flux toward the wall is positive, and the wall is at
        # z = 0 with z positive into the plasma — so ions arriving at the wall have
        # v_z < 0. Doc 02 §2 calls sign errors in flux quantities "common, silent and
        # catastrophic", which is why this convention gets its own named accessor
        # rather than an inline minus somewhere downstream.
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=-3000.0, spread=1.0)

        flux = ivdf.particle_flux_toward_wall_per_m2_s()

        np.testing.assert_allclose(flux, 1e17 * 3000.0, rtol=1e-4)
        assert np.all(flux > 0.0)

    def test_particle_flux_is_negative_for_ions_moving_away_from_the_wall(
        self, grid: SpatialGrid, argon: Species
    ) -> None:
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=+3000.0, spread=1.0)

        assert np.all(ivdf.particle_flux_toward_wall_per_m2_s() < 0.0)

    def test_moments_are_reported_per_grid_point(self, grid: SpatialGrid, argon: Species) -> None:
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=-3000.0, spread=200.0)

        assert ivdf.density_per_m3().shape == (grid.n_points,)
        assert ivdf.mean_velocity_m_per_s().shape == (grid.n_points,)

    def test_mean_velocity_of_an_empty_distribution_is_zero_not_nan(
        self, grid: SpatialGrid, argon: Species
    ) -> None:
        # A zero-density cell is physical (nothing has arrived yet in a transient), and
        # a NaN propagating out of 0/0 would poison every downstream aggregate silently.
        ivdf = VelocityDistribution(
            grid=grid,
            v_m_per_s=np.linspace(-1.0, 1.0, 3),
            values=np.zeros((grid.n_points, 3)),
            species=argon,
        )

        np.testing.assert_allclose(ivdf.mean_velocity_m_per_s(), 0.0)

    def test_exposes_the_velocity_axis_as_a_quantity(
        self, grid: SpatialGrid, argon: Species
    ) -> None:
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=-3000.0, spread=200.0)

        assert magnitude_in(ivdf.v, "m/s").shape == ivdf.v_m_per_s.shape

    def test_at_the_wall_returns_the_first_grid_point(
        self, grid: SpatialGrid, argon: Species
    ) -> None:
        # doc 03 §5.2 emulates f_i(v_z; z=0) specifically: the IEDF at the wall is the
        # deliverable, so reaching it must not require the caller to remember that the
        # wall is index 0 rather than index -1.
        ivdf = _drifting_beam(grid, argon, density=1e17, drift=-3000.0, spread=200.0)

        np.testing.assert_allclose(ivdf.at_wall(), ivdf.values[0, :])
