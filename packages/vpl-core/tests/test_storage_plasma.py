"""Plasma state artifacts — doc 08 §7, "Plasma state fields | HDF5 (chunked, compressed)".

The round trip is tested for **exactness**, not for approximate agreement. Doc 00 E3
promises that ``(manifest, commit, environment lock, seed)`` reproduces a result
bit-for-bit, and a comparison written with a tolerance cannot tell a lossless writer from
one that quietly dropped the last three mantissa bits — which is the failure that makes
the whole promise unverifiable.

Doc 13 §5 treats raw fields as a 90-day cache and only the reduced artifacts as the
archive, so what these tests really pin is that the cache can be regenerated *and
compared against* what it replaced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray

from vpl.core.provenance import EnvironmentLockSource, Provenance, Tier
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
from vpl.core.storage import (
    ArtifactKind,
    MissingProvenanceError,
    read_plasma_state,
    write_plasma_state,
)
from vpl.core.units import Q_

_FIELD_UNITS = {"n_e": "m**-3", "n_i": "m**-3", "Phi": "V", "T_e": "eV"}


@pytest.fixture
def argon() -> Species:
    return Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid.geometric(length=Q_(20.0, "mm"), n_points=9, stretch=1.3)


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
        kappa=1.7,
    )


@pytest.fixture
def record() -> Provenance:
    return Provenance(
        manifest_sha256="4a7f2e91" + "0" * 56,
        git_commit="9c1d8b3" + "0" * 33,
        git_dirty=False,
        seed=20260804,
        environment_lock_hash="e81c" + "f" * 60,
        environment_lock_source=EnvironmentLockSource.UV_LOCK,
        created_utc=datetime(2026, 8, 4, 12, 30, 15, 123456, tzinfo=UTC),
        vpl_version="0.1.0",
        solver_versions={"dolfinx": "0.8.0", "petsc": "3.20", "numpyro": "0.15"},
        tier=Tier.T2,
    )


def _awkward(shape: tuple[int, ...], *, seed: int) -> NDArray[np.float64]:
    """Values whose mantissas are full, so a lossy write cannot hide in the noise."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) * np.pi


def _fields(grid: SpatialGrid, *, time: TimeGrid | None = None) -> dict[str, ScalarField]:
    shape = (grid.n_points,) if time is None else (time.n_points, grid.n_points)
    return {
        name: ScalarField(
            name=name,
            values=_awkward(shape, seed=index),
            units=units,
            grid=grid,
            time=time,
        )
        for index, (name, units) in enumerate(_FIELD_UNITS.items())
    }


def _ivdf(grid: SpatialGrid, argon: Species) -> VelocityDistribution:
    v = np.linspace(-2.0e4, 1.0e4, 33)
    rng = np.random.default_rng(7)
    return VelocityDistribution(
        grid=grid,
        v_m_per_s=v,
        values=rng.random((grid.n_points, v.size)),
        species=argon,
    )


def _steady(grid: SpatialGrid, params: PlasmaParams) -> PlasmaState:
    return PlasmaState(
        params=params,
        grid=grid,
        time=None,
        fields=_fields(grid),
        ion_distribution=None,
        fidelity=Fidelity.L1,
    )


class TestPlasmaStateRoundTrip:
    def test_a_written_state_reads_back_equal_to_the_one_written(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        state = _steady(grid, params)

        write_plasma_state(tmp_path / "state.h5", state, provenance=record)
        archived = read_plasma_state(tmp_path / "state.h5")

        assert archived.state == state

    def test_field_values_survive_bit_for_bit(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        # doc 00 E3 promises bit-for-bit reproducibility. A write that rounded even one
        # ulp would make that promise unverifiable rather than merely inaccurate.
        state = _steady(grid, params)

        write_plasma_state(tmp_path / "state.h5", state, provenance=record)
        archived = read_plasma_state(tmp_path / "state.h5")

        for name in state.field_names:
            assert archived.state.field(name).values.tobytes() == state.field(name).values.tobytes()

    def test_the_grid_survives_bit_for_bit(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        # A geometrically graded mesh has no closed-form node positions worth rounding to;
        # if the archived grid is not the grid that was solved on, every comparison
        # against the regenerated cache is against the wrong geometry.
        state = _steady(grid, params)

        write_plasma_state(tmp_path / "state.h5", state, provenance=record)
        archived = read_plasma_state(tmp_path / "state.h5")

        assert archived.state.grid.z_m.tobytes() == grid.z_m.tobytes()

    def test_every_field_keeps_its_units(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        state = _steady(grid, params)

        write_plasma_state(tmp_path / "state.h5", state, provenance=record)
        archived = read_plasma_state(tmp_path / "state.h5")

        assert {name: archived.state.field(name).units for name in _FIELD_UNITS} == _FIELD_UNITS

    def test_a_time_dependent_state_keeps_its_time_grid(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        time = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=5)
        state = PlasmaState(
            params=params,
            grid=grid,
            time=time,
            fields=_fields(grid, time=time),
            ion_distribution=None,
            fidelity=Fidelity.L1,
        )

        write_plasma_state(tmp_path / "state.h5", state, provenance=record)
        archived = read_plasma_state(tmp_path / "state.h5")

        assert archived.state.time == time
        assert archived.state.field("n_e").values.shape == (5, grid.n_points)

    def test_a_steady_state_reads_back_steady(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        write_plasma_state(tmp_path / "state.h5", _steady(grid, params), provenance=record)

        assert read_plasma_state(tmp_path / "state.h5").state.is_steady is True

    def test_the_ion_distribution_survives(
        self,
        tmp_path: Path,
        grid: SpatialGrid,
        params: PlasmaParams,
        argon: Species,
        record: Provenance,
    ) -> None:
        # doc 03 §1 makes the IEDF the only thing L2 produces that no other level can.
        # An artifact that dropped it would be an L2 run nobody could check.
        state = PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=_fields(grid),
            ion_distribution=_ivdf(grid, argon),
            fidelity=Fidelity.L2,
        )

        write_plasma_state(tmp_path / "state.h5", state, provenance=record)
        archived = read_plasma_state(tmp_path / "state.h5")

        assert archived.state.ion_distribution == state.ion_distribution
        assert archived.state.ion_distribution is not None
        assert archived.state.ion_distribution.species == argon

    def test_the_fidelity_survives(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        # doc 05 §7.2 makes reporting one fidelity as another a project defect; the same
        # argument applies to the level that produced a state.
        state = _steady(grid, params)

        write_plasma_state(tmp_path / "state.h5", state, provenance=record)

        assert read_plasma_state(tmp_path / "state.h5").state.fidelity is Fidelity.L1

    def test_the_control_parameters_survive(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        write_plasma_state(tmp_path / "state.h5", _steady(grid, params), provenance=record)

        recovered = read_plasma_state(tmp_path / "state.h5").state.params

        assert recovered == params
        assert recovered.bias_volts == params.bias_volts
        assert recovered.gamma_se == params.gamma_se
        assert recovered.kappa == params.kappa

    def test_an_rf_drive_survives_and_dc_stays_dc(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        # ``rf_frequency=None`` means DC, not "missing". Writing a zero here would turn a
        # DC run into an RF run with an impossible drive, which PlasmaParams rejects.
        driven = params.replace(rf_frequency=Q_(13.56, "MHz"), rf_phase=0.25)

        write_plasma_state(tmp_path / "dc.h5", _steady(grid, params), provenance=record)
        write_plasma_state(tmp_path / "rf.h5", _steady(grid, driven), provenance=record)

        assert read_plasma_state(tmp_path / "dc.h5").state.params.rf_frequency is None
        recovered = read_plasma_state(tmp_path / "rf.h5").state.params
        assert recovered.is_rf is True
        assert recovered.rf_frequency == Q_(13.56, "MHz")
        assert recovered.rf_phase == 0.25

    def test_the_provenance_survives(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        write_plasma_state(tmp_path / "state.h5", _steady(grid, params), provenance=record)

        assert read_plasma_state(tmp_path / "state.h5").provenance == record


class TestPlasmaStateOnDisk:
    def test_field_datasets_are_chunked_and_compressed(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        # doc 08 §7 names both, and doc 10 §7's 860 GB raw budget is what pays for them.
        write_plasma_state(tmp_path / "state.h5", _steady(grid, params), provenance=record)

        with h5py.File(tmp_path / "state.h5", "r") as handle:
            dataset = handle["fields/n_e"]
            assert dataset.chunks is not None
            assert dataset.compression == "gzip"

    def test_the_writer_returns_the_path_it_wrote(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        state = _steady(grid, params)

        written = write_plasma_state(tmp_path / "state.h5", state, provenance=record)

        assert written == tmp_path / "state.h5"
        assert written.is_file()


class TestPlasmaStateRefusesToLoseProvenance:
    def test_writing_without_provenance_raises_rather_than_warning(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        # doc 00 E4 requires every figure to carry embedded provenance. An artifact that
        # *can* be written without it will be, and the figure it feeds inherits nothing.
        with pytest.raises(MissingProvenanceError):
            write_plasma_state(
                tmp_path / "state.h5",
                _steady(grid, params),
                provenance=None,  # type: ignore[arg-type]
            )

    def test_nothing_is_left_on_disk_when_provenance_is_refused(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams
    ) -> None:
        # A half-written file that a later read treats as an artifact is worse than no
        # file: it is an artifact whose provenance is absent for a reason nobody recorded.
        with pytest.raises(MissingProvenanceError):
            write_plasma_state(
                tmp_path / "state.h5",
                _steady(grid, params),
                provenance=None,  # type: ignore[arg-type]
            )

        assert not (tmp_path / "state.h5").exists()

    def test_reading_a_file_whose_provenance_was_stripped_raises(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        write_plasma_state(tmp_path / "state.h5", _steady(grid, params), provenance=record)
        with h5py.File(tmp_path / "state.h5", "a") as handle:
            del handle.attrs["git_commit"]

        with pytest.raises(MissingProvenanceError, match="git_commit"):
            read_plasma_state(tmp_path / "state.h5")

    def test_reading_a_file_that_is_not_a_plasma_state_names_what_it_is(
        self, tmp_path: Path, grid: SpatialGrid, params: PlasmaParams, record: Provenance
    ) -> None:
        write_plasma_state(tmp_path / "state.h5", _steady(grid, params), provenance=record)
        with h5py.File(tmp_path / "state.h5", "a") as handle:
            handle.attrs["vpl_artifact_kind"] = ArtifactKind.MEASUREMENTS.value

        with pytest.raises(ValueError, match="measurements"):
            read_plasma_state(tmp_path / "state.h5")
