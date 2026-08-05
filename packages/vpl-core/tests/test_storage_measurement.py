"""Measurement artifacts — doc 08 §7, "HDF5, one group per instrument".

Doc 01 SYS-4 is the requirement these tests really enforce: the inversion is a joint fit
across asynchronous channels, and *without per-sample time and error it cannot be posed*.
So every one of the four things doc 08 §7 lists — per-sample timestamp, acquisition
window, phase bin, uncertainty — is asserted to survive the round trip individually,
because an artifact that carried three of the four would still read back as a
:class:`MeasurementSet` and would still be useless.
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
    AcquisitionWindow,
    CalibrationState,
    Measurement,
    MeasurementSet,
    PhaseGrid,
)
from vpl.core.storage import (
    MissingProvenanceError,
    read_measurement_set,
    write_measurement_set,
)
from vpl.core.units import Q_


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
        solver_versions={"numpyro": "0.15"},
        tier=Tier.T1,
    )


def _awkward(size: int, *, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size) * np.pi


def _interferometer() -> Measurement:
    return Measurement(
        instrument_id="interf",
        values=_awkward(4, seed=1),
        uncertainty=np.abs(_awkward(4, seed=2)),
        units="rad",
        window=AcquisitionWindow.absolute(start=Q_(1.5, "s"), duration=Q_(700.0, "s")),
    )


def _oes(bin_index: int) -> Measurement:
    phase = PhaseGrid(n_bins=8, period=Q_(73.7, "ns"))
    return Measurement(
        instrument_id="oes",
        values=_awkward(6, seed=10 + bin_index),
        uncertainty=np.abs(_awkward(6, seed=20 + bin_index)),
        units="count",
        window=AcquisitionWindow.phase_locked(
            grid=phase,
            bin_index=bin_index,
            gate=Q_(2.0, "ns"),
            accumulation=Q_(30.0, "s"),
        ),
        calibration=CalibrationState.ESTIMATED,
    )


@pytest.fixture
def measurements() -> MeasurementSet:
    return MeasurementSet.of(_interferometer(), _oes(0), _oes(3))


class TestMeasurementSetRoundTrip:
    def test_a_written_set_reads_back_equal_to_the_one_written(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)

        assert read_measurement_set(tmp_path / "y.h5").measurements == measurements

    def test_sample_values_survive_bit_for_bit(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)
        recovered = read_measurement_set(tmp_path / "y.h5").measurements

        for written, read_back in zip(measurements, recovered, strict=True):
            assert read_back.values.tobytes() == written.values.tobytes()

    def test_the_per_sample_uncertainty_survives_bit_for_bit(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        # doc 01 SYS-4: without per-sample error the inversion cannot be posed. An
        # uncertainty that came back broadcast or rounded produces a confident wrong
        # posterior, and nothing in the output says so.
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)
        recovered = read_measurement_set(tmp_path / "y.h5").measurements

        for written, read_back in zip(measurements, recovered, strict=True):
            assert read_back.uncertainty.tobytes() == written.uncertainty.tobytes()

    def test_the_acquisition_window_survives(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)
        recovered = read_measurement_set(tmp_path / "y.h5").measurements

        window = recovered.by_instrument("interf")[0].window
        assert window.is_absolute is True
        assert window.start == Q_(1.5, "s")
        assert window.duration == Q_(700.0, "s")

    def test_a_phase_locked_window_keeps_its_bin_grid_and_accumulation(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        # doc 02 §10.3: the accumulation span is what benchmark B-09 needs to quantify
        # drift-induced phase smearing. It is not recoverable from the gate width.
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)
        recovered = read_measurement_set(tmp_path / "y.h5").measurements

        window = recovered.by_instrument("oes")[1].window
        assert window.is_phase_locked is True
        assert window.phase_bin == 3
        assert window.phase_grid == PhaseGrid(n_bins=8, period=Q_(73.7, "ns"))
        assert window.accumulation == Q_(30.0, "s")
        assert window.duration == Q_(2.0, "ns")

    def test_the_units_of_every_channel_survive(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)
        recovered = read_measurement_set(tmp_path / "y.h5").measurements

        assert recovered.by_instrument("interf")[0].units == "rad"
        assert recovered.by_instrument("oes")[0].units == "count"

    def test_the_calibration_state_survives(self, tmp_path: Path, record: Provenance) -> None:
        # doc 04 §7.3 names using the true calibration an inverse crime. A round trip
        # that lost the flag would let a crime-committing run be archived as an honest one.
        crime = Measurement(
            instrument_id="lif",
            values=_awkward(3, seed=99),
            uncertainty=np.abs(_awkward(3, seed=98)),
            units="count",
            window=AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(1.0, "s")),
            calibration=CalibrationState.TRUE,
        )
        write_measurement_set(tmp_path / "y.h5", MeasurementSet.of(crime), provenance=record)

        recovered = read_measurement_set(tmp_path / "y.h5").measurements

        assert recovered.measurements[0].calibration is CalibrationState.TRUE
        assert recovered.measurements[0].is_inverse_crime is True

    def test_several_records_from_one_channel_are_all_kept(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        # A "measurement" is a set spread over minutes (doc 02 §10.1), so one instrument
        # contributes many records. Keying the group by instrument must not collapse them.
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)

        recovered = read_measurement_set(tmp_path / "y.h5").measurements

        assert len(recovered.by_instrument("oes")) == 2
        assert recovered.n_observations == measurements.n_observations

    def test_the_provenance_survives(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)

        assert read_measurement_set(tmp_path / "y.h5").provenance == record


class TestMeasurementSetOnDisk:
    def test_there_is_one_group_per_instrument(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        # doc 08 §7 states the layout, and it is the layout an analyst browsing the file
        # with h5ls needs: channels are the unit of ablation (doc 02 §13).
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)

        with h5py.File(tmp_path / "y.h5", "r") as handle:
            assert sorted(handle["measurements"].keys()) == ["interf", "oes"]

    def test_sample_datasets_are_chunked_and_compressed(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)

        with h5py.File(tmp_path / "y.h5", "r") as handle:
            group = handle["measurements/interf"]
            dataset = group[next(iter(sorted(group.keys())))]["values"]
            assert dataset.chunks is not None
            assert dataset.compression == "gzip"


class TestMeasurementSetRefusesToLoseProvenance:
    def test_writing_without_provenance_raises_rather_than_warning(
        self, tmp_path: Path, measurements: MeasurementSet
    ) -> None:
        with pytest.raises(MissingProvenanceError):
            write_measurement_set(
                tmp_path / "y.h5",
                measurements,
                provenance=None,  # type: ignore[arg-type]
            )

    def test_reading_a_file_whose_provenance_was_stripped_raises(
        self, tmp_path: Path, measurements: MeasurementSet, record: Provenance
    ) -> None:
        write_measurement_set(tmp_path / "y.h5", measurements, provenance=record)
        with h5py.File(tmp_path / "y.h5", "a") as handle:
            del handle.attrs["tier"]

        with pytest.raises(MissingProvenanceError, match="tier"):
            read_measurement_set(tmp_path / "y.h5")


class TestMeasurementSetRefusesAnUnwritableChannelName:
    def test_an_instrument_id_containing_a_path_separator_is_refused(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        # HDF5 reads "/" as a group separator, so an id containing one would silently
        # nest the channel somewhere nobody asked for and read back under a different
        # name. Instrument ids are plugin-supplied (doc 08 §10), so this is reachable.
        awkward = Measurement(
            instrument_id="oes/iccd",
            values=_awkward(2, seed=5),
            uncertainty=np.abs(_awkward(2, seed=6)),
            units="count",
            window=AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(1.0, "s")),
        )

        with pytest.raises(ValueError, match="oes/iccd"):
            write_measurement_set(tmp_path / "y.h5", MeasurementSet.of(awkward), provenance=record)
