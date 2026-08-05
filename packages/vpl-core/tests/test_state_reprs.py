"""Every state type must be printable.

This is a regression guard, and it earned its place. ``PlasmaParams.__repr__`` shipped
with a malformed pint format specifier, and the failure mode was worse than a bad string:
pytest calls ``repr`` when it builds an assertion failure, so a raising ``repr`` replaced
the real diagnostic with ``[ValueError raised in repr()]``. A broken ``repr`` does not
just fail to help — it actively hides the error you were trying to read.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.state import (
    Fidelity,
    PhaseGrid,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    Species,
    TimeGrid,
    VelocityDistribution,
)
from vpl.core.units import Q_


@pytest.fixture
def argon() -> Species:
    return Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=5)


@pytest.fixture
def params(argon: Species) -> PlasmaParams:
    return PlasmaParams(
        species=argon,
        n_0=Q_(1e17, "m**-3"),
        T_e=Q_(3.0, "eV"),
        T_i=Q_(0.05, "eV"),
        T_g=Q_(300.0, "K"),
        pressure=Q_(5.0, "mTorr"),
        bias=Q_(-250.0, "V"),
        gamma_se=0.10,
        kappa=1.0,
    )


class TestEveryStateTypeIsPrintable:
    def test_spatial_grid(self, grid: SpatialGrid) -> None:
        assert "SpatialGrid" in repr(grid)

    def test_time_grid(self) -> None:
        assert "TimeGrid" in repr(TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=4))

    def test_phase_grid(self) -> None:
        assert "PhaseGrid" in repr(PhaseGrid(n_bins=16, period=Q_(73.7, "ns")))

    def test_species(self, argon: Species) -> None:
        assert "Ar+" in repr(argon)

    def test_plasma_params_with_dc_bias(self, params: PlasmaParams) -> None:
        assert "DC" in repr(params)

    def test_plasma_params_with_rf_bias(self, params: PlasmaParams) -> None:
        # The RF branch formats a quantity the DC branch never touches, so it needs its
        # own exercise — this is the branch that was broken.
        assert "MHz" in repr(params.replace(rf_frequency=Q_(13.56, "MHz")))

    def test_scalar_field(self, grid: SpatialGrid) -> None:
        field = ScalarField(name="n_e", values=np.ones(5), units="m**-3", grid=grid, time=None)

        assert "n_e" in repr(field)

    def test_velocity_distribution(self, grid: SpatialGrid, argon: Species) -> None:
        v = np.linspace(-1e4, 1e4, 11)
        ivdf = VelocityDistribution(grid=grid, v_m_per_s=v, values=np.ones((5, 11)), species=argon)

        assert "Ar+" in repr(ivdf)

    def test_plasma_state(self, grid: SpatialGrid, params: PlasmaParams) -> None:
        spec = {"n_e": "m**-3", "n_i": "m**-3", "Phi": "V", "T_e": "eV"}
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields={
                name: ScalarField(name=name, values=np.ones(5), units=units, grid=grid, time=None)
                for name, units in spec.items()
            },
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        assert "L1" in repr(state)
