"""The embedded-provenance rule, across all four artifact formats — doc 08 §7.

doc 08 §7 lists nine fields "every artifact embeds" and does not exempt any format. The
rule is therefore tested once, over every writer, rather than four times in four places
where one of the four could quietly drift.

Two properties are easy to lose and hard to notice afterwards, so they are pinned
separately:

- **Types.** HDF5 attributes come back as NumPy scalars, and a ``git_dirty`` that reads as
  ``np.True_`` is not a ``bool``. :meth:`Provenance.from_dict` rejects it — which is the
  behaviour that makes the round trip either correct or loud, never silently degraded.
- **Completeness.** The field list is checked against ``Provenance.to_dict()`` itself, so
  adding a tenth field to the record cannot leave the storage layer writing nine.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from vpl.core.provenance import EnvironmentLockSource, Provenance, Tier
from vpl.core.state import (
    AcquisitionWindow,
    Fidelity,
    Measurement,
    MeasurementSet,
    ParameterLevel,
    PlasmaParams,
    PlasmaState,
    Posterior,
    SamplerDiagnostics,
    ScalarField,
    SpatialGrid,
    Species,
)
from vpl.core.storage import (
    MetricRecord,
    MissingProvenanceError,
    read_measurement_set,
    read_metrics,
    read_plasma_state,
    read_posterior,
    write_measurement_set,
    write_metrics,
    write_plasma_state,
    write_posterior,
)
from vpl.core.units import Q_

#: The nine fields doc 08 §7 names, plus the lock *source* the record adds so that a
#: reader can tell a resolved lock from a scan of what happened to be importable.
_DOC_08_7_FIELDS = frozenset(
    {
        "manifest_sha256",
        "git_commit",
        "git_dirty",
        "seed",
        "environment_lock_hash",
        "created_utc",
        "vpl_version",
        "solver_versions",
        "tier",
    }
)


@pytest.fixture
def record() -> Provenance:
    return Provenance(
        manifest_sha256="4a7f2e91" + "0" * 56,
        git_commit="9c1d8b3" + "0" * 33,
        git_dirty=True,
        seed=20260804,
        environment_lock_hash="e81c" + "f" * 60,
        environment_lock_source=EnvironmentLockSource.INSTALLED_DISTRIBUTIONS,
        created_utc=datetime(2026, 8, 4, 12, 30, 15, 123456, tzinfo=UTC),
        vpl_version="0.1.0",
        solver_versions={"dolfinx": "0.8.0", "petsc": "3.20"},
        tier=Tier.T0,
    )


def _state() -> PlasmaState:
    grid = SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=4)
    species = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)
    params = PlasmaParams(
        species=species,
        n_0=Q_(1e17, "m**-3"),
        T_e=Q_(3.0, "eV"),
        T_i=Q_(0.05, "eV"),
        T_g=Q_(300.0, "K"),
        pressure=Q_(5.0, "mTorr"),
        bias=Q_(-250.0, "V"),
        gamma_se=0.1,
        kappa=1.0,
    )
    spec = {"n_e": "m**-3", "n_i": "m**-3", "Phi": "V", "T_e": "eV"}
    fields = {
        name: ScalarField(
            name=name, values=np.ones(grid.n_points), units=units, grid=grid, time=None
        )
        for name, units in spec.items()
    }
    return PlasmaState(
        params=params,
        grid=grid,
        time=None,
        fields=fields,
        ion_distribution=None,
        fidelity=Fidelity.L0,
    )


def _measurements() -> MeasurementSet:
    return MeasurementSet.of(
        Measurement(
            instrument_id="interf",
            values=np.ones(3),
            uncertainty=np.full(3, 0.1),
            units="rad",
            window=AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(1.0, "s")),
        )
    )


def _posterior() -> Posterior:
    rng = np.random.default_rng(0)
    return Posterior(
        samples=rng.standard_normal((4, 500, 1)),
        names=("V_w",),
        levels={"V_w": ParameterLevel.CONTROL},
        tier=Tier.T0,
        diagnostics=SamplerDiagnostics(r_hat={"V_w": 1.001}, ess={"V_w": 1500.0}, divergences=0),
    )


def _metrics() -> tuple[MetricRecord, ...]:
    return (MetricRecord(name="rel_error", value=0.07, units="dimensionless"),)


class _Archived(Protocol):
    """What every reader returns, as far as this module is concerned."""

    @property
    def provenance(self) -> Provenance: ...


type _Write = Callable[[Path, Provenance], object]
type _Read = Callable[[Path], _Archived]

#: ``(label, write, read)`` for every artifact doc 08 §7 tabulates.
_WRITERS: tuple[tuple[str, _Write, _Read], ...] = (
    (
        "plasma_state.h5",
        lambda path, record: write_plasma_state(path, _state(), provenance=record),
        read_plasma_state,
    ),
    (
        "measurements.h5",
        lambda path, record: write_measurement_set(path, _measurements(), provenance=record),
        read_measurement_set,
    ),
    (
        "posterior.zarr",
        lambda path, record: write_posterior(path, _posterior(), provenance=record),
        read_posterior,
    ),
    (
        "metrics.parquet",
        lambda path, record: write_metrics(path, _metrics(), provenance=record),
        read_metrics,
    ),
)

_IDS = [name for name, _, _ in _WRITERS]


class TestTheFieldListCannotDrift:
    def test_the_record_still_carries_exactly_the_fields_doc_08_7_names(
        self, record: Provenance
    ) -> None:
        # If a tenth field is added to Provenance, this fails and whoever added it has to
        # decide, explicitly, whether artifacts must carry it too.
        assert set(record.to_dict()) >= _DOC_08_7_FIELDS
        assert set(record.to_dict()) - _DOC_08_7_FIELDS == {"environment_lock_source"}


@pytest.mark.parametrize(("filename", "write", "read"), _WRITERS, ids=_IDS)
class TestEveryArtifactEmbedsItsProvenance:
    def test_the_whole_record_reads_back_equal(
        self,
        tmp_path: Path,
        record: Provenance,
        filename: str,
        write: _Write,
        read: _Read,
    ) -> None:
        write(tmp_path / filename, record)

        recovered = read(tmp_path / filename)

        assert recovered.provenance == record

    def test_the_solver_versions_mapping_is_not_flattened_into_a_string(
        self,
        tmp_path: Path,
        record: Provenance,
        filename: str,
        write: _Write,
        read: _Read,
    ) -> None:
        # doc 13 §2 records solver_versions as a mapping because a reproducibility check
        # asks "which PETSc", not "does this string contain 3.20".
        write(tmp_path / filename, record)

        recovered = read(tmp_path / filename).provenance

        assert dict(recovered.solver_versions) == {"dolfinx": "0.8.0", "petsc": "3.20"}

    def test_the_tier_and_lock_source_read_back_as_their_enums(
        self,
        tmp_path: Path,
        record: Provenance,
        filename: str,
        write: _Write,
        read: _Read,
    ) -> None:
        # doc 05 §7.2 makes reporting one tier as another a project defect, so the tier
        # is never allowed to degrade into an unvalidated free-text label.
        write(tmp_path / filename, record)

        recovered = read(tmp_path / filename).provenance

        assert recovered.tier is Tier.T0
        assert recovered.environment_lock_source is EnvironmentLockSource.INSTALLED_DISTRIBUTIONS

    def test_the_scalar_fields_read_back_as_python_types_not_backend_scalars(
        self,
        tmp_path: Path,
        record: Provenance,
        filename: str,
        write: _Write,
        read: _Read,
    ) -> None:
        # HDF5 hands back ``np.bool_`` and ``np.int64``. Neither is an instance of the
        # Python type Provenance validates against, so an unnormalised read would either
        # raise on every artifact or — worse, if the validation were relaxed — put a
        # NumPy scalar into a YAML sidecar that no other tool can load.
        write(tmp_path / filename, record)

        recovered = read(tmp_path / filename).provenance

        assert type(recovered.git_dirty) is bool
        assert type(recovered.seed) is int
        assert recovered.created_utc == record.created_utc
        assert recovered.created_utc.microsecond == 123456

    def test_writing_without_provenance_raises_rather_than_warning(
        self,
        tmp_path: Path,
        filename: str,
        write: _Write,
        read: _Read,
    ) -> None:
        # doc 00 E4 requires embedded provenance on every figure. Every figure comes from
        # an artifact, so an artifact that may be written without provenance is a figure
        # that will be published without it.
        with pytest.raises(MissingProvenanceError):
            write(tmp_path / filename, None)  # type: ignore[arg-type]
