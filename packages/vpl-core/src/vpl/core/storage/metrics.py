"""Metric artifacts — doc 08 §7 ("Parquet, queryable across runs") and doc 07 §7.

doc 07 §7 fixes the schema, not doc 08 §7: *"every metric is stored per commit; a change
beyond its historical noise band fails the build. Physics regressions are silent otherwise
— the code runs, the plots look plausible, and the answer is wrong."*

Two consequences follow, and both are in the layout rather than in a convention.

**Provenance is columnar, not just file metadata.** Every row repeats the commit, the
manifest digest, the tier and the rest of the doc 08 §7 block. The regression store's
question is "how has ``rel_error`` moved over the last fifty commits", and a store that
kept the commit only in a file header would have to open and parse fifty headers to answer
it. Repeating a dozen dictionary-encoded strings per row costs Parquet almost nothing; the
alternative costs the query.

**The file-level metadata carries the block as well.** A run that produced no metrics
still writes a file, and a zero-row table has no rows to carry columns. doc 08 §7 says
*every* artifact embeds its provenance, with no exception for an empty one.

## What is refused, and why it is refused here

A non-finite metric. ``nan`` compares false against every bound, so a ``nan`` never lands
outside its historical noise band and the doc 07 §7 gate passes it silently — which is
precisely the silent physics regression that gate exists to catch, arriving through the
store meant to catch it. And two values for one metric name in one run: a regression check
that picks one of them is a build that passes or fails on row order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

# pyarrow ships no stubs; the suppression lives in the root pyproject alongside the
# rest of the untyped scientific stack, so there is one place to look when one of them
# starts shipping types.
import pyarrow as pa
import pyarrow.parquet as pq

from vpl.core.provenance import Provenance
from vpl.core.storage.metadata import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_KIND_ATTRIBUTE,
    PROVENANCE_FIELDS,
    STORAGE_FORMAT_ATTRIBUTE,
    ArtifactKind,
    MissingProvenanceError,
    check_artifact_kind,
    provenance_attributes,
    provenance_from_attributes,
    require_provenance,
)
from vpl.core.units import UREG

__all__ = ["ArchivedMetrics", "MetricRecord", "read_metrics", "write_metrics"]

_METRIC_COLUMN: Final[str] = "metric"
_VALUE_COLUMN: Final[str] = "value"
_UNITS_COLUMN: Final[str] = "units"

#: Key under which the doc 08 §7 block lives in the file-level metadata.
_PROVENANCE_KEY: Final[bytes] = b"vpl_provenance"

#: Lossless codec. Nothing in doc 08 §7's table may round a stored number, and a metric
#: that drifted by a bit in storage would be indistinguishable from one that regressed.
_COMPRESSION: Final[str] = "zstd"

#: Columns carrying the doc 08 §7 provenance block. All non-nullable: a row that cannot
#: name the commit it was measured at is not a row the regression store can use.
_PROVENANCE_COLUMN_TYPES: Final[dict[str, pa.DataType]] = {
    "manifest_sha256": pa.string(),
    "git_commit": pa.string(),
    "git_dirty": pa.bool_(),
    "seed": pa.int64(),
    "environment_lock_hash": pa.string(),
    "environment_lock_source": pa.string(),
    # A string, not a Parquet timestamp. It is the exact ISO 8601 text the record round
    # trips through, offset included, and it is the same text the HDF5 and Zarr artifacts
    # carry — so one query can join a metric row to the state file it came from.
    "created_utc": pa.string(),
    "vpl_version": pa.string(),
    # JSON text, as everywhere else in this package. Parquet has a map type; using it here
    # would make this the one artifact whose provenance block needs its own decoder.
    "solver_versions": pa.string(),
    "tier": pa.string(),
}


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One scalar measured by a run — a row of the doc 07 §7 regression store.

    Validated at construction rather than at write. A metric that cannot be gated is a
    metric that should never have been produced, and the call site that computed it is the
    only place that still knows why it came out ``nan``.

    Attributes:
        name: The metric's identifier, e.g. ``"rel_error"``, ``"coverage"``. It is the key
            the regression store groups a history by, so it must be stable across runs.
        value: The measured number. Finite — see the module docstring.
        units: Registry-parseable units. ``"dimensionless"`` for a ratio or a coverage
            fraction; doc 06 §8 quotes several metrics dimensionally (``eV``, ``kW/m^2``)
            and a store that dropped the unit would compare two of them happily.
    """

    name: str
    value: float
    units: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a metric needs a name; it is the key its history is keyed by")

        if not math.isfinite(self.value):
            raise ValueError(
                f"metric {self.name!r} is {self.value}, which is not finite. doc 07 §7 gates "
                "every metric against its historical noise band, and a non-finite value "
                "compares false against every bound — so it would pass the gate that exists "
                "to catch it."
            )

        try:
            UREG.Unit(self.units)
        except Exception as exc:
            raise ValueError(
                f"metric {self.name!r} has units {self.units!r}, which this registry cannot "
                f"parse: {exc}"
            ) from exc


class ArchivedMetrics(NamedTuple):
    """Metrics read back from disk, beside the provenance they were written with."""

    metrics: tuple[MetricRecord, ...]
    provenance: Provenance


def _schema(record: Provenance) -> pa.Schema:
    """The regression-store schema, identical for every run so tables concatenate."""
    fields = [
        pa.field(_METRIC_COLUMN, pa.string(), nullable=False),
        pa.field(_VALUE_COLUMN, pa.float64(), nullable=False),
        pa.field(_UNITS_COLUMN, pa.string(), nullable=False),
        *(
            pa.field(name, _PROVENANCE_COLUMN_TYPES[name], nullable=False)
            for name in PROVENANCE_FIELDS
        ),
    ]
    metadata = {
        STORAGE_FORMAT_ATTRIBUTE.encode(): str(ARTIFACT_FORMAT_VERSION).encode(),
        ARTIFACT_KIND_ATTRIBUTE.encode(): ArtifactKind.METRICS.value.encode(),
        _PROVENANCE_KEY: json.dumps(provenance_attributes(record), sort_keys=True).encode(),
    }
    return pa.schema(fields, metadata=metadata)


def _checked_names(metrics: Sequence[MetricRecord]) -> None:
    seen: set[str] = set()
    for metric in metrics:
        if metric.name in seen:
            raise ValueError(
                f"metric {metric.name!r} appears twice in one run. doc 07 §7 gates one value "
                "per metric per commit, and a check that picked one of two is a build that "
                "passes or fails on row order."
            )
        seen.add(metric.name)


def write_metrics(path: Path, metrics: Sequence[MetricRecord], *, provenance: Provenance) -> Path:
    """Write a run's metrics as a Parquet table — doc 08 §7, doc 07 §7.

    Args:
        path: Destination file. Overwritten if it exists.
        metrics: The run's scalars. May be empty; the file is still written, and still
            carries its provenance.
        provenance: The doc 08 §7 record. Required.

    Returns:
        The path written.

    Raises:
        MissingProvenanceError: If ``provenance`` is absent.
        ValueError: If one metric name carries two values.
    """
    record = require_provenance(provenance)
    _checked_names(metrics)

    columns: dict[str, list[Any]] = {
        _METRIC_COLUMN: [metric.name for metric in metrics],
        _VALUE_COLUMN: [metric.value for metric in metrics],
        _UNITS_COLUMN: [metric.units for metric in metrics],
    }
    embedded = provenance_attributes(record)
    for name in PROVENANCE_FIELDS:
        columns[name] = [embedded[name]] * len(metrics)

    schema = _schema(record)
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), path, compression=_COMPRESSION)
    return path


def read_metrics(path: Path) -> ArchivedMetrics:
    """Read back a table written by :func:`write_metrics`.

    The provenance comes from the file-level metadata rather than from the columns,
    because that is the only place a run with no metrics can have kept it.

    Raises:
        MissingProvenanceError: If the file carries no provenance metadata.
        ValueError: If the file holds a different artifact.
    """
    table = pq.read_table(path)
    metadata: dict[bytes, bytes] = dict(table.schema.metadata or {})

    if _PROVENANCE_KEY not in metadata:
        raise MissingProvenanceError(
            f"{path.name} carries no provenance metadata. doc 08 §7 requires every artifact "
            "to embed the manifest digest, commit, seed, environment lock and tier that "
            "produced it, and a metric that cannot name its commit cannot be regressed "
            "against its own history (doc 07 §7)."
        )

    attributes: dict[str, object] = {
        ARTIFACT_KIND_ATTRIBUTE: metadata.get(ARTIFACT_KIND_ATTRIBUTE.encode(), b"").decode(),
        **json.loads(metadata[_PROVENANCE_KEY].decode()),
    }
    check_artifact_kind(attributes, ArtifactKind.METRICS)
    record = provenance_from_attributes(attributes)

    rows = table.select([_METRIC_COLUMN, _VALUE_COLUMN, _UNITS_COLUMN]).to_pydict()
    metrics = tuple(
        MetricRecord(name=str(name), value=float(value), units=str(units))
        for name, value, units in zip(
            rows[_METRIC_COLUMN], rows[_VALUE_COLUMN], rows[_UNITS_COLUMN], strict=True
        )
    )

    return ArchivedMetrics(metrics=metrics, provenance=record)
