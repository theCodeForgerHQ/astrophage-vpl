"""Artifact storage — doc 08 §7.

doc 08 §7 fixes one format per artifact, and gives the reason for each:

===================== ================================= ==================================
Artifact              Format                            Why
===================== ================================= ==================================
Plasma state fields   HDF5, chunked, compressed         Large, hierarchical, standard
Measurements          HDF5, one group per instrument    Carries the doc 01 SYS-4 metadata
Posterior samples     Zarr, chunked                     Large, appended incrementally
Metrics / scalars     Parquet                           Queryable across runs
===================== ================================= ==================================

and rules out the obvious alternative in one line: **"Never CSV for simulation output. No
dtype, no units, no metadata, no compression, no chunking — every one of which the project
needs."** Each writer here supplies all five.

Two rules cut across every format in this package.

**Nothing is written without provenance.** doc 08 §7 requires every artifact to embed
``manifest_sha256, git_commit, git_dirty, seed, environment_lock_hash, created_utc,
vpl_version, solver_versions, tier``, and doc 00 E4 requires the figures made from those
artifacts to carry the same. Every writer takes a
:class:`~vpl.core.provenance.Provenance` as a required keyword and raises
:class:`MissingProvenanceError` rather than warning — an artifact that *can* be written
without provenance eventually will be, and a warning in a batch of ten thousand runs is a
warning nobody reads.

**Nothing is rounded.** Every codec used here is lossless, and no magnitude is converted
between units on the way in or out. doc 00 E3 promises bit-for-bit reproducibility from
``(manifest, commit, environment lock, seed)``; doc 13 §5 then treats raw fields as a
90-day cache to be *regenerated* rather than kept, which is only affordable because a
regenerated field can be compared against the archived one. A writer that dropped three
mantissa bits would make that comparison unable to tell "the solver changed" from "the
writer rounded", and the whole retention policy with it.

## Per-stream seeds

doc 10 §5 requires per-stream seeds and doc 13 §2 shows the run record listing five of
them, while doc 08 §7 has artifacts embed a single ``seed``. The artifacts written here
store the root, and deliberately not the stream map. The streams are *derived* from the
root by a documented derivation that lives with the engines, so the root together with the
commit — both already in the record — determines every stream. A stored copy of the map
adds no information and can disagree with the derivation after any refactor, at which
point the artifact asserts two incompatible things and nothing can say which is true. The
expansion belongs in the run record of doc 13 §2, which is a human-readable sidecar and is
covered by ``manifest_sha256`` — not in an attribute block that ``vpl-core``, which owns no
RNG, would be in no position to verify.
"""

from vpl.core.storage.measurement import (
    ArchivedMeasurements,
    read_measurement_set,
    write_measurement_set,
)
from vpl.core.storage.metadata import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_KIND_ATTRIBUTE,
    PROVENANCE_FIELDS,
    STORAGE_FORMAT_ATTRIBUTE,
    ArtifactKind,
    MissingProvenanceError,
)
from vpl.core.storage.metrics import (
    ArchivedMetrics,
    MetricRecord,
    read_metrics,
    write_metrics,
)
from vpl.core.storage.plasma import ArchivedState, read_plasma_state, write_plasma_state
from vpl.core.storage.posterior import (
    DRAW_CHUNK,
    ArchivedPosterior,
    PosteriorSummary,
    read_posterior,
    write_posterior,
)

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ARTIFACT_KIND_ATTRIBUTE",
    "DRAW_CHUNK",
    "PROVENANCE_FIELDS",
    "STORAGE_FORMAT_ATTRIBUTE",
    "ArchivedMeasurements",
    "ArchivedMetrics",
    "ArchivedPosterior",
    "ArchivedState",
    "ArtifactKind",
    "MetricRecord",
    "MissingProvenanceError",
    "PosteriorSummary",
    "read_measurement_set",
    "read_metrics",
    "read_plasma_state",
    "read_posterior",
    "write_measurement_set",
    "write_metrics",
    "write_plasma_state",
    "write_posterior",
]
