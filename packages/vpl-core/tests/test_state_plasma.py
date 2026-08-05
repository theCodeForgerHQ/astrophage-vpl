"""PlasmaState — what ``ForwardSolver.solve`` returns, at every fidelity level.

doc 00 E1: the inverse solver never learns which level produced the state it is looking
at. That only holds if every level produces the *same shape of object*, which is what
this type is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.state import (
    Fidelity,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    Species,
    TimeGrid,
    VelocityDistribution,
)
from vpl.core.units import Q_, DimensionalityError


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


def _fields(grid: SpatialGrid, *, time: TimeGrid | None = None) -> dict[str, ScalarField]:
    shape = (grid.n_points,) if time is None else (time.n_points, grid.n_points)
    spec = {"n_e": "m**-3", "n_i": "m**-3", "Phi": "V", "T_e": "eV"}
    return {
        name: ScalarField(name=name, values=np.ones(shape), units=units, grid=grid, time=time)
        for name, units in spec.items()
    }


def _ivdf(grid: SpatialGrid, argon: Species) -> VelocityDistribution:
    v = np.linspace(-1e4, 1e4, 51)
    return VelocityDistribution(
        grid=grid,
        v_m_per_s=v,
        values=np.ones((grid.n_points, v.size)),
        species=argon,
    )


class TestPlasmaStateConstruction:
    def test_holds_the_fields_a_forward_model_reads(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=_fields(grid),
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        assert state.fidelity is Fidelity.L1
        assert state.field("n_e").units == "m**-3"

    def test_field_lookup_names_what_is_present_when_asked_for_something_absent(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        # A forward model asking for a field the solver did not emit is a wiring error,
        # and "KeyError: 'n_metastable'" does not say whether the solver is wrong or the
        # instrument is.
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=_fields(grid),
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        with pytest.raises(KeyError, match="n_e"):
            state.field("n_metastable")

    def test_requires_the_fields_doc_03_5_2_names(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        incomplete = _fields(grid)
        del incomplete["Phi"]

        with pytest.raises(ValueError, match="Phi"):
            PlasmaState(
                params=params,
                grid=grid,
                time=None,
                fields=incomplete,
                ion_distribution=None,
                fidelity=Fidelity.L1,
            )

    def test_rejects_a_field_whose_dimensionality_is_wrong(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        # A potential in metres type-checks perfectly and is nonsense. Catching it here
        # is the difference between a loud constructor and a silent 10^19 in a spectrum.
        wrong = _fields(grid)
        wrong["Phi"] = ScalarField(
            name="Phi", values=np.ones(grid.n_points), units="m", grid=grid, time=None
        )

        with pytest.raises(DimensionalityError, match="Phi"):
            PlasmaState(
                params=params,
                grid=grid,
                time=None,
                fields=wrong,
                ion_distribution=None,
                fidelity=Fidelity.L1,
            )

    def test_rejects_a_field_on_a_different_grid(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        other = SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=7)
        mismatched = _fields(grid)
        mismatched["T_e"] = ScalarField(
            name="T_e", values=np.ones(7), units="eV", grid=other, time=None
        )

        with pytest.raises(ValueError, match="grid"):
            PlasmaState(
                params=params,
                grid=grid,
                time=None,
                fields=mismatched,
                ion_distribution=None,
                fidelity=Fidelity.L1,
            )

    def test_rejects_a_field_keyed_under_a_different_name(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        # The mapping key and the field's own name must agree, or every error message
        # and every artifact group name downstream refers to the wrong quantity.
        mislabelled = _fields(grid)
        mislabelled["n_e"] = ScalarField(
            name="n_i", values=np.ones(grid.n_points), units="m**-3", grid=grid, time=None
        )

        with pytest.raises(ValueError, match="keyed"):
            PlasmaState(
                params=params,
                grid=grid,
                time=None,
                fields=mislabelled,
                ion_distribution=None,
                fidelity=Fidelity.L1,
            )


class TestPlasmaStateFidelityInvariants:
    def test_a_kinetic_state_must_carry_its_distribution(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        # doc 03 §1 calls L2 "the only level that produces a correct IEDF"; producing an
        # L2 state without one means the run cannot answer the question it was launched
        # to answer, and finding that out at the likelihood is far too late.
        with pytest.raises(ValueError, match="distribution"):
            PlasmaState(
                params=params,
                grid=grid,
                time=None,
                fields=_fields(grid),
                ion_distribution=None,
                fidelity=Fidelity.L2,
            )

    def test_a_fluid_state_may_omit_the_distribution(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        # doc 03 §6: L1 reconstructs a drifting Maxwellian from its moments, and doc 03
        # calls that a genuine weakness. The state records the absence honestly instead
        # of manufacturing a distribution the solver never computed.
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=_fields(grid),
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        assert state.ion_distribution is None

    def test_a_distribution_must_share_the_state_grid(
        self, grid: SpatialGrid, params: PlasmaParams, argon: Species
    ) -> None:
        other = SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=7)

        with pytest.raises(ValueError, match="grid"):
            PlasmaState(
                params=params,
                grid=grid,
                time=None,
                fields=_fields(grid),
                ion_distribution=_ivdf(other, argon),
                fidelity=Fidelity.L2,
            )

    def test_a_distribution_must_be_of_the_parameterised_species(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        xenon = Species(name="Xe+", mass=Q_(131.29, "u"), charge_number=1)

        with pytest.raises(ValueError, match="species"):
            PlasmaState(
                params=params,
                grid=grid,
                time=None,
                fields=_fields(grid),
                ion_distribution=_ivdf(grid, xenon),
                fidelity=Fidelity.L2,
            )


class TestPlasmaStateTime:
    def test_a_state_without_a_time_grid_is_steady(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=_fields(grid),
            ion_distribution=None,
            fidelity=Fidelity.L0,
        )

        assert state.is_steady is True

    def test_a_time_dependent_state_carries_time_dependent_fields(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        time = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=4)
        state = PlasmaState(
            params=params,
            grid=grid,
            time=time,
            fields=_fields(grid, time=time),
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        assert state.is_steady is False
        assert state.field("n_e").values.shape == (4, 5)

    def test_rejects_a_steady_field_in_a_time_dependent_state(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        time = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=4)
        mixed = _fields(grid, time=time)
        mixed["T_e"] = ScalarField(
            name="T_e", values=np.ones(grid.n_points), units="eV", grid=grid, time=None
        )

        with pytest.raises(ValueError, match="time"):
            PlasmaState(
                params=params,
                grid=grid,
                time=time,
                fields=mixed,
                ion_distribution=None,
                fidelity=Fidelity.L1,
            )


class TestPlasmaStateImmutability:
    def test_the_field_mapping_cannot_be_mutated(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=_fields(grid),
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        with pytest.raises(TypeError):
            state.fields["n_e"] = state.field("n_i")  # type: ignore[index]

    def test_mutating_the_caller_mapping_does_not_reach_the_state(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        source = _fields(grid)
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=source,
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        source["Phi"] = ScalarField(
            name="Phi", values=np.full(grid.n_points, 999.0), units="V", grid=grid, time=None
        )

        assert state.field("Phi").values[0] == 1.0

    def test_field_names_are_reported_in_a_deterministic_order(
        self, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        # doc 00 E3 promises bit-for-bit reproducibility. Anything that iterates fields
        # and reduces over them — a hash, an artifact write, a diff — must see the same
        # order in every process, not whatever order the solver happened to insert.
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=_fields(grid),
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        assert state.field_names == ("Phi", "T_e", "n_e", "n_i")
