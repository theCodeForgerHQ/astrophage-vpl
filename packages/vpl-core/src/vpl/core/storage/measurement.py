"""Measurement artifacts — doc 08 §7, "HDF5, one group per instrument".

doc 08 §7 states what a measurement file must carry: *per-sample timestamp, acquisition
window, phase bin, uncertainty* — and points at doc 01 SYS-4 for why. SYS-4 puts it
bluntly: the inversion is a joint fit across asynchronous channels, and *without per-sample
time and error it cannot be posed*. Every one of the four therefore has a place in the
layout below, and none of them is optional.

## The layout, and why records nest inside the channel group

::

    /measurements/<instrument_id>/record_0000/values        (+ units)
                                            /uncertainty   (+ units)
                                            /window/...

doc 02 §10.1 is explicit that a "measurement" is not a snapshot but a set of observations
spread over minutes. One instrument therefore contributes many records — an OES channel
contributes one per phase bin — so the per-instrument group holds a numbered record per
observation rather than a single array. Stacking them would require padding to a common
sample count, and a padded LIF frequency scan is indistinguishable from a short one.

The instrument group is keyed by the channel id because that is the unit doc 02 §13
ablates on: "what if one diagnostic fails" is answered by not reading one group.

## The acquisition window

Stored in the shape :class:`~vpl.core.state.measurement.AcquisitionWindow` has it, with
the phase grid present only for a cyclo-stationary window (doc 02 §10.3). The mode is
*derived on read* from whether that grid is there, rather than stored as its own
attribute: two fields that can disagree about the same fact are two chances to be wrong,
and only one of them can be checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

import h5py

from vpl.core.provenance import Provenance
from vpl.core.state.grid import PhaseGrid
from vpl.core.state.measurement import (
    AcquisitionWindow,
    CalibrationState,
    Measurement,
    MeasurementSet,
)
from vpl.core.storage.hdf5 import (
    checked_group_name,
    group_attributes,
    read_array,
    read_optional_quantity,
    read_quantity,
    read_units,
    write_array,
    write_quantity,
)
from vpl.core.storage.metadata import (
    ArtifactKind,
    artifact_attributes,
    check_artifact_kind,
    provenance_from_attributes,
    require_provenance,
)

__all__ = ["ArchivedMeasurements", "read_measurement_set", "write_measurement_set"]

_MEASUREMENTS_GROUP: Final[str] = "measurements"
_WINDOW_GROUP: Final[str] = "window"

_VALUES_DATASET: Final[str] = "values"
_UNCERTAINTY_DATASET: Final[str] = "uncertainty"

_CALIBRATION_ATTRIBUTE: Final[str] = "calibration"
_PHASE_BIN_ATTRIBUTE: Final[str] = "phase_bin"
_PHASE_BINS_ATTRIBUTE: Final[str] = "n_phase_bins"

_RECORD_PREFIX: Final[str] = "record_"

#: Digits in a record's group name.
#:
#: Six, so that the names of a channel's records sort lexicographically in the order they
#: were written for any acquisition this project can produce — doc 02 §10.1's longest is a
#: few thousand observations. Read order does not rely on it (see :func:`_record_index`),
#: but a file that reads sensibly in ``h5ls`` is worth four characters.
_RECORD_DIGITS: Final[int] = 6


class ArchivedMeasurements(NamedTuple):
    """A measurement set read back from disk, beside the provenance it was written with."""

    measurements: MeasurementSet
    provenance: Provenance


def _record_index(name: str) -> int:
    """Sort key for a record group, read as a number rather than as text."""
    return int(name.removeprefix(_RECORD_PREFIX))


def _write_window(parent: h5py.Group, window: AcquisitionWindow) -> None:
    group = parent.create_group(_WINDOW_GROUP)
    write_quantity(group, "start", window.start)
    write_quantity(group, "duration", window.duration)

    if window.accumulation is not None:
        # doc 02 §10.3: the wall-clock span of a phase-locked accumulation is what
        # benchmark B-09 needs to quantify drift-induced smearing, and it is not
        # recoverable from the gate width.
        write_quantity(group, "accumulation", window.accumulation)

    if window.phase_grid is not None and window.phase_bin is not None:
        write_quantity(group, "phase_period", window.phase_grid.period)
        group.attrs[_PHASE_BINS_ATTRIBUTE] = window.phase_grid.n_bins
        group.attrs[_PHASE_BIN_ATTRIBUTE] = window.phase_bin


def _read_window(parent: h5py.Group) -> AcquisitionWindow:
    group = parent[_WINDOW_GROUP]
    phase_grid = (
        PhaseGrid(
            n_bins=int(group.attrs[_PHASE_BINS_ATTRIBUTE]),
            period=read_quantity(group, "phase_period"),
        )
        if _PHASE_BIN_ATTRIBUTE in group.attrs
        else None
    )

    return AcquisitionWindow(
        start=read_quantity(group, "start"),
        duration=read_quantity(group, "duration"),
        phase_grid=phase_grid,
        phase_bin=int(group.attrs[_PHASE_BIN_ATTRIBUTE]) if phase_grid is not None else None,
        accumulation=read_optional_quantity(group, "accumulation"),
    )


def _write_record(parent: h5py.Group, index: int, measurement: Measurement) -> None:
    group = parent.create_group(f"{_RECORD_PREFIX}{index:0{_RECORD_DIGITS}d}")
    write_array(group, _VALUES_DATASET, measurement.values, units=measurement.units)
    write_array(group, _UNCERTAINTY_DATASET, measurement.uncertainty, units=measurement.units)
    # doc 04 §7.3 calls correcting with the true calibration an inverse crime. A file that
    # lost the flag would let a crime-committing run be archived as an honest one.
    group.attrs[_CALIBRATION_ATTRIBUTE] = measurement.calibration.value
    _write_window(group, measurement.window)


def _read_record(group: h5py.Group, instrument_id: str) -> Measurement:
    return Measurement(
        instrument_id=instrument_id,
        values=read_array(group, _VALUES_DATASET),
        uncertainty=read_array(group, _UNCERTAINTY_DATASET),
        units=read_units(group, _VALUES_DATASET),
        window=_read_window(group),
        calibration=CalibrationState(str(group.attrs[_CALIBRATION_ATTRIBUTE])),
    )


def write_measurement_set(
    path: Path, measurements: MeasurementSet, *, provenance: Provenance
) -> Path:
    """Write a measurement set to HDF5, one group per instrument — doc 08 §7.

    Records are written in the set's own deterministic order (doc 00 E3), so two runs that
    produced the same observations produce the same file.

    Args:
        path: Destination file. Overwritten if it exists.
        measurements: The observations to archive.
        provenance: The doc 08 §7 record. Required.

    Returns:
        The path written.

    Raises:
        MissingProvenanceError: If ``provenance`` is absent.
        ValueError: If an instrument id cannot be used as an HDF5 group name.
    """
    record = require_provenance(provenance)
    grouped = measurements.grouped_by_instrument()
    for instrument_id in grouped:
        checked_group_name(instrument_id, what="instrument id")

    with h5py.File(path, "w") as handle:
        handle.attrs.update(artifact_attributes(ArtifactKind.MEASUREMENTS, record))

        root = handle.create_group(_MEASUREMENTS_GROUP)
        for instrument_id, records in grouped.items():
            channel = root.create_group(instrument_id)
            for index, measurement in enumerate(records):
                _write_record(channel, index, measurement)

    return path


def read_measurement_set(path: Path) -> ArchivedMeasurements:
    """Read back a set written by :func:`write_measurement_set`.

    Every record passes back through :class:`~vpl.core.state.measurement.Measurement`, so
    an uncertainty that no longer matches its values, or a gate that no longer fits its
    phase bin, fails here rather than inside a likelihood.

    Raises:
        MissingProvenanceError: If the doc 08 §7 provenance block is incomplete.
        ValueError: If the file holds a different artifact, or observations the model
            rejects.
    """
    with h5py.File(path, "r") as handle:
        attributes = group_attributes(handle)
        check_artifact_kind(attributes, ArtifactKind.MEASUREMENTS)
        provenance = provenance_from_attributes(attributes)

        root = handle[_MEASUREMENTS_GROUP]
        records = [
            _read_record(root[instrument_id][name], instrument_id)
            for instrument_id in sorted(root)
            for name in sorted(root[instrument_id], key=_record_index)
        ]

    return ArchivedMeasurements(
        measurements=MeasurementSet.from_iterable(records),
        provenance=provenance,
    )
