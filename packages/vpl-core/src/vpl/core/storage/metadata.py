"""The provenance block every artifact carries — doc 08 §7.

doc 08 §7 lists nine fields "every artifact embeds" and exempts no format. This module is
the one encoder and the one decoder for that block, shared by the HDF5, Zarr and Parquet
writers, so the three cannot drift apart and a reader written against one of them can be
pointed at any of the others.

## How the awkward types are stored

HDF5 attributes hold only scalars and arrays of them. Zarr attributes are JSON. Parquet
key-value metadata is bytes. The intersection is *flat scalars plus strings*, and that is
what :func:`provenance_attributes` produces:

- **Enums** — :class:`~vpl.core.provenance.Tier` and
  :class:`~vpl.core.provenance.EnvironmentLockSource` are ``StrEnum``, so they are written
  as their label text and read back through the record's own ``from_dict``, which rejects
  anything not a member. The type is recovered by *validation*, not by the encoding: a
  hand-edited ``"T2-ish"`` fails at the read rather than becoming a plausible-looking
  string in a figure caption (doc 05 §7.2).
- **Mappings** — ``solver_versions`` becomes a JSON object in a single string attribute.
  No backend in doc 08 §7's table has a native mapping attribute; JSON is already the
  sidecar format doc 08 §7 names, and it round-trips ``str`` keys and values exactly. It
  is decoded back into a ``dict[str, str]``, so what a caller receives is a mapping and
  never a string that happens to contain braces.
- **pint units** are *not* handled here. They belong to the arrays they describe, not to
  the provenance block, and are stored as the registry-parseable strings the state model
  already keeps them as — recovered by :class:`~vpl.core.state.fields.ScalarField`'s own
  constructor, which parses them through the shared registry and raises on anything it
  does not know.

## Why NumPy scalars are normalised on read

h5py hands back ``np.True_`` and ``np.int64(20260804)``. Neither is an instance of the
Python type :meth:`Provenance.from_dict` validates against, so an unnormalised read would
fail on every artifact ever written. Relaxing the validation instead would be worse: a
``np.int64`` seed written into a YAML sidecar is not loadable by anything but Python,
which defeats the reason doc 08 §7 chose YAML in the first place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Final

import numpy as np

from vpl.core.provenance import Provenance

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ARTIFACT_KIND_ATTRIBUTE",
    "PROVENANCE_FIELDS",
    "STORAGE_FORMAT_ATTRIBUTE",
    "ArtifactKind",
    "MissingProvenanceError",
    "artifact_attributes",
    "check_artifact_kind",
    "provenance_attributes",
    "provenance_from_attributes",
    "require_provenance",
]

#: Version of the on-disk layout this package writes.
#:
#: doc 13 §5 retains manifests, provenance and reduced artifacts *forever*, which means a
#: file written today will be opened by code that does not exist yet. A version stamp is
#: the cheapest thing that lets that code tell "a layout I do not understand" from "a
#: corrupt file", and the difference matters when the alternative is a silent misread.
ARTIFACT_FORMAT_VERSION: Final[int] = 1

#: Attribute naming the layout version.
STORAGE_FORMAT_ATTRIBUTE: Final[str] = "vpl_storage_format"

#: Attribute naming which of doc 08 §7's artifacts a file holds.
ARTIFACT_KIND_ATTRIBUTE: Final[str] = "vpl_artifact_kind"

#: Fields of the embedded block, in the order doc 08 §7 lists them.
#:
#: Checked against :meth:`Provenance.to_dict` at import, so a field added to the record
#: cannot leave the storage layer silently writing the old set.
PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "manifest_sha256",
    "git_commit",
    "git_dirty",
    "seed",
    "environment_lock_hash",
    "environment_lock_source",
    "created_utc",
    "vpl_version",
    "solver_versions",
    "tier",
)

#: Provenance fields with no scalar spelling, stored as JSON in a single string attribute.
_JSON_FIELDS: Final[frozenset[str]] = frozenset({"solver_versions"})


class ArtifactKind(StrEnum):
    """Which row of doc 08 §7's format table a file belongs to.

    Recorded so that a reader can refuse a file of the wrong kind instead of failing
    later on a missing group, with an error that blames the file rather than the caller.
    An analyst who points ``read_plasma_state`` at a measurement archive has made an
    ordinary mistake, and the message should say so.
    """

    PLASMA_STATE = "plasma_state"
    MEASUREMENTS = "measurements"
    POSTERIOR = "posterior"
    METRICS = "metrics"


class MissingProvenanceError(ValueError):
    """An artifact was written, or read, without the doc 08 §7 provenance block.

    Raised, never warned. doc 00 E4 requires every figure to carry embedded provenance,
    and every figure comes from an artifact — so an artifact that *can* be produced
    without provenance eventually will be, and the figure it feeds inherits nothing. A
    warning in a batch of ten thousand runs is a warning nobody reads.

    A subclass of :class:`ValueError` because that is what the rest of the data model
    raises for a value it cannot accept; callers that already handle bad input keep
    working, and callers that care about this case specifically can name it.
    """


def require_provenance(provenance: Provenance | None) -> Provenance:
    """Return ``provenance``, or refuse.

    The type annotations already make provenance mandatory at every writer. This is the
    runtime half of the same rule, for the call sites the type checker never sees — a
    manifest-driven pipeline assembling arguments from YAML, or a plugin (doc 08 §10)
    written without ``--strict``.

    Raises:
        MissingProvenanceError: If ``provenance`` is ``None`` or is not a record.
    """
    if isinstance(provenance, Provenance):
        return provenance

    raise MissingProvenanceError(
        "an artifact cannot be written without provenance: doc 08 §7 requires every "
        "artifact to embed the manifest digest, commit, seed, environment lock and tier "
        f"that produced it, and doc 00 E4 requires the figures made from it to carry the "
        f"same. Got {provenance!r}. Build one with Provenance.capture()."
    )


def provenance_attributes(record: Provenance) -> dict[str, str | bool | int]:
    """Flatten a record into attributes every backend in doc 08 §7's table can hold."""
    flat = record.to_dict()
    attributes: dict[str, str | bool | int] = {}

    for key in PROVENANCE_FIELDS:
        value = flat[key]
        if isinstance(value, dict):
            # The only nested field. Sorted keys so that one record produces one byte
            # sequence, whichever order the mapping happened to be built in.
            attributes[key] = json.dumps(value, sort_keys=True)
        else:
            attributes[key] = value

    return attributes


def artifact_attributes(kind: ArtifactKind, record: Provenance) -> dict[str, str | bool | int]:
    """The full attribute block a writer stamps on the root of an artifact."""
    return {
        STORAGE_FORMAT_ATTRIBUTE: ARTIFACT_FORMAT_VERSION,
        ARTIFACT_KIND_ATTRIBUTE: kind.value,
        **provenance_attributes(record),
    }


def _native(value: object) -> object:
    """A backend scalar as the Python type :meth:`Provenance.from_dict` validates.

    ``np.bool_`` is not a ``bool`` and ``np.int64`` is not an ``int``, so without this
    every HDF5 artifact would fail its own read. Bytes appear where a backend stores
    strings as UTF-8 blobs, which Parquet metadata does.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def provenance_from_attributes(attributes: Mapping[str, object]) -> Provenance:
    """Rebuild a record from an artifact's attribute block.

    Args:
        attributes: The attributes read from the artifact root. Extra keys are ignored;
            an artifact is free to carry more than the doc 08 §7 block.

    Returns:
        The validated record.

    Raises:
        MissingProvenanceError: If any doc 08 §7 field is absent. A partially stripped
            block is treated exactly like an absent one — an artifact that can name its
            commit but not its seed is no more reproducible than one that names neither.
        ValueError: If a field is present but unrecognisable.
        TypeError: If a field is present but of the wrong type.
    """
    missing = [field for field in PROVENANCE_FIELDS if field not in attributes]
    if missing:
        raise MissingProvenanceError(
            f"artifact is missing provenance field(s) {', '.join(missing)}. doc 08 §7 "
            "requires the whole block, and a record that can name its commit but not its "
            "seed is no more reproducible than one that names neither."
        )

    flat: dict[str, object] = {}
    for field in PROVENANCE_FIELDS:
        value = _native(attributes[field])
        if field in _JSON_FIELDS and isinstance(value, str):
            flat[field] = json.loads(value)
        else:
            flat[field] = value

    return Provenance.from_dict(flat)


def check_artifact_kind(attributes: Mapping[str, object], expected: ArtifactKind) -> None:
    """Refuse a file that holds a different artifact.

    An absent kind attribute is refused too. Every writer in this package stamps one, so
    its absence means the file was produced by something else — and guessing at the
    layout of a foreign file is how a plausible but wrong array reaches a figure.
    """
    found = _native(attributes.get(ARTIFACT_KIND_ATTRIBUTE))
    if found == expected.value:
        return

    raise ValueError(
        f"expected a {expected.value} artifact but the file declares {found!r}. "
        f"doc 08 §7 gives each artifact its own format and layout; reading one as "
        f"another would produce arrays with the wrong meaning rather than an error."
    )


# The record and the storage layer must agree on the field set, and the only way to be
# sure of that is to ask the record. Checked at import so a mismatch fails on the first
# use of the package rather than on the first artifact nobody can read back.
if set(PROVENANCE_FIELDS) != set(
    Provenance.__dataclass_fields__
):  # pragma: no cover - a guard against future drift
    raise RuntimeError(
        "vpl.core.storage.metadata.PROVENANCE_FIELDS has drifted from Provenance: "
        f"{sorted(set(Provenance.__dataclass_fields__) ^ set(PROVENANCE_FIELDS))}"
    )
