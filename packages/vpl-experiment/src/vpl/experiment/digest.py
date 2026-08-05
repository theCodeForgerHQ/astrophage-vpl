"""What "bit-identical" means — gate G-1.3, doc 00 E3, doc 13 §6.

doc 00 E3 promises that ``(manifest, code version, seed)`` reproduces a result
bit-for-bit, and gate G-1.3 is that promise made checkable. Comparing the two files byte
for byte does not work, and the reason is not a defect: doc 08 §7 requires every artifact
to embed ``created_utc``, so two honest runs of the same manifest *must* differ in their
bytes. A comparison that ignored that would be comparing nothing; one that failed on it
would fail every time.

So the digest here covers **every byte of every stored value and every attribute except
the ones doc 13 §2 records under the execution rather than the result** — which is
``created_utc`` and only ``created_utc``. The commit, the seed, the environment lock, the
tier and the manifest digest are all *inside* the digest, because a reproduction that
changed any of them has not reproduced anything.

Two exclusion sets, because two questions are being asked:

- :data:`EXECUTION_ONLY_FIELDS` — for ``vpl reproduce``. Everything must match. This is the
  gate.
- :data:`PROVENANCE_ONLY_FIELDS` — for ``vpl compare``. Two *different* runs necessarily
  carry different provenance, and the interesting question is whether the science agrees,
  so the whole doc 08 §7 block is set aside and the arrays are compared alone.

## Why the digest walks the data rather than hashing the file

An HDF5 file's bytes depend on chunk layout, fill values, allocation order and the version
of the library that wrote it; a Parquet file's depend on the codec build. None of those is
a property of the result. Walking the stored values and hashing *those* asks the question
G-1.3 actually asks, and it survives an h5py upgrade — which doc 13 §5's "forever"
retention makes a certainty rather than a possibility.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import pyarrow.parquet as pq

from vpl.core.storage import PROVENANCE_FIELDS

__all__ = [
    "EXECUTION_ONLY_FIELDS",
    "PROVENANCE_ONLY_FIELDS",
    "UnknownArtifactFormatError",
    "artifact_digest",
    "artifact_digests",
    "artifact_paths",
    "run_content_digest",
]

#: Fields that record *when* a run happened rather than *what* it produced — doc 13 §2.
#:
#: Exactly one. Every other field of doc 08 §7's block is part of the result's identity:
#: a reproduction at a different commit, from a different manifest or with a different
#: seed has not reproduced the run, it has performed a different one.
EXECUTION_ONLY_FIELDS: Final[frozenset[str]] = frozenset({"created_utc"})

#: The whole doc 08 §7 provenance block, set aside when comparing two different runs.
PROVENANCE_ONLY_FIELDS: Final[frozenset[str]] = frozenset(PROVENANCE_FIELDS)

#: File suffix to the reader that understands it — doc 08 §7's format table.
_HDF5_SUFFIXES: Final[frozenset[str]] = frozenset({".h5", ".hdf5"})
_PARQUET_SUFFIXES: Final[frozenset[str]] = frozenset({".parquet"})

#: Separator between hashed items. A byte that cannot appear in an HDF5 name or a column
#: name, so that ``["ab", "c"]`` and ``["a", "bc"]`` cannot hash alike.
_SEPARATOR: Final[bytes] = b"\x00"


class UnknownArtifactFormatError(ValueError):
    """An artifact was written in a format this module cannot read.

    Raised rather than falling back to hashing the raw bytes. A raw-byte fallback would
    silently reintroduce exactly the timestamp sensitivity this module exists to remove,
    and gate G-1.3 would then fail for a reason that has nothing to do with the physics.
    """


class _Accumulator:
    """A SHA-256 over a sequence of length-delimited items."""

    __slots__ = ("_hash",)

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def add(self, item: bytes) -> None:
        self._hash.update(len(item).to_bytes(8, "big"))
        self._hash.update(item)
        self._hash.update(_SEPARATOR)

    def add_text(self, item: str) -> None:
        self.add(item.encode("utf-8"))

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _canonical_scalar(value: object) -> str:
    """A backend scalar as text that does not depend on which backend produced it.

    ``np.True_``, ``np.int64(3)`` and ``b"T2"`` all arrive here from one artifact or
    another; without normalisation two files holding the same value would hash apart.
    """
    if isinstance(value, np.generic):
        return _canonical_scalar(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return json.dumps([_canonical_scalar(item) for item in value.tolist()])
    return repr(value)


def _strip_json(payload: str, excluded: frozenset[str]) -> str:
    """Drop excluded keys from a JSON object, or return the text unchanged.

    The metrics artifact keeps its provenance block as a JSON string in Parquet's
    key-value metadata (doc 08 §7's "no backend has a native mapping attribute"), so the
    exclusion has to reach inside it. Text that is not a JSON object is hashed as it is.
    """
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    if not isinstance(decoded, dict):
        return payload
    kept = {key: value for key, value in decoded.items() if key not in excluded}
    return json.dumps(kept, sort_keys=True)


def _add_attributes(
    accumulator: _Accumulator, attributes: Mapping[str, object], excluded: frozenset[str]
) -> None:
    """Hash an attribute block in name order, minus the excluded names."""
    for name in sorted(attributes):
        if name in excluded:
            continue
        accumulator.add_text(name)
        value = attributes[name]
        text = _canonical_scalar(value)
        accumulator.add_text(_strip_json(text, excluded) if isinstance(value, str) else text)


def _add_hdf5_group(accumulator: _Accumulator, group: h5py.Group, excluded: frozenset[str]) -> None:
    """Walk a group in name order.

    Sorted rather than in file order: doc 00 E3's promise is about the *contents*, and
    HDF5 link order is a property of how the file was assembled.
    """
    _add_attributes(accumulator, dict(group.attrs), excluded)

    for name in sorted(group):
        accumulator.add_text(name)
        member = group[name]
        if isinstance(member, h5py.Group):
            _add_hdf5_group(accumulator, member, excluded)
            continue

        _add_attributes(accumulator, dict(member.attrs), excluded)
        values = np.asarray(member[()])
        accumulator.add_text(values.dtype.str)
        accumulator.add_text(str(values.shape))
        # ``ascontiguousarray`` so that a view's strides cannot change the bytes of an
        # otherwise identical array.
        accumulator.add(np.ascontiguousarray(values).tobytes())


def _digest_hdf5(path: Path, excluded: frozenset[str]) -> str:
    accumulator = _Accumulator()
    with h5py.File(path, "r") as handle:
        _add_hdf5_group(accumulator, handle, excluded)
    return accumulator.hexdigest()


def _digest_parquet(path: Path, excluded: frozenset[str]) -> str:
    table = pq.read_table(path)
    accumulator = _Accumulator()

    metadata = table.schema.metadata or {}
    for key in sorted(metadata):
        name = key.decode("utf-8")
        if name in excluded:
            continue
        accumulator.add_text(name)
        accumulator.add_text(_strip_json(metadata[key].decode("utf-8"), excluded))

    columns = [name for name in table.column_names if name not in excluded]
    for name in sorted(columns):
        accumulator.add_text(name)
        accumulator.add_text(str(table.schema.field(name).type))
        for value in table.column(name).to_pylist():
            accumulator.add_text(_canonical_scalar(value))

    return accumulator.hexdigest()


def artifact_digest(path: Path, *, excluded: frozenset[str] = EXECUTION_ONLY_FIELDS) -> str:
    """A content digest of one artifact.

    Args:
        path: The artifact. Its suffix selects the reader, per doc 08 §7's format table.
        excluded: Attribute, column and JSON keys to leave out. Defaults to
            :data:`EXECUTION_ONLY_FIELDS` — the gate G-1.3 comparison. Pass
            :data:`PROVENANCE_ONLY_FIELDS` to compare two different runs' science.

    Returns:
        The lowercase hexadecimal SHA-256 of the artifact's canonical content.

    Raises:
        UnknownArtifactFormatError: If the suffix is not one doc 08 §7 defines.
    """
    suffix = path.suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        return _digest_hdf5(path, excluded)
    if suffix in _PARQUET_SUFFIXES:
        return _digest_parquet(path, excluded)

    known = ", ".join(sorted(_HDF5_SUFFIXES | _PARQUET_SUFFIXES))
    raise UnknownArtifactFormatError(
        f"{path.name}: no artifact reader for {suffix!r}. doc 08 §7 fixes one format per "
        f"artifact kind ({known}); hashing an unrecognised file as raw bytes would make "
        "gate G-1.3 fail on a stored timestamp rather than on the physics."
    )


def artifact_paths(directory: Path) -> tuple[Path, ...]:
    """Every artifact in a run's artifact directory, in name order."""
    if not directory.is_dir():
        return ()
    return tuple(sorted(path for path in directory.iterdir() if path.is_file()))


def run_content_digest(directory: Path, *, excluded: frozenset[str] = EXECUTION_ONLY_FIELDS) -> str:
    """One digest over every artifact a run produced.

    The file names are hashed alongside their contents, so a run that produced an extra
    artifact — or produced the same arrays under a different name — does not claim to have
    reproduced one that did not.
    """
    accumulator = _Accumulator()
    for path in artifact_paths(directory):
        accumulator.add_text(path.name)
        accumulator.add_text(artifact_digest(path, excluded=excluded))
    return accumulator.hexdigest()


def artifact_digests(
    directory: Path, *, excluded: frozenset[str] = EXECUTION_ONLY_FIELDS
) -> Mapping[str, str]:
    """Per-artifact digests, keyed by file name.

    What ``vpl reproduce`` reports when the run digests disagree: "the run differs" is not
    actionable, "``plasma_state.h5`` differs and ``metrics.parquet`` does not" is.
    """
    return {
        path.name: artifact_digest(path, excluded=excluded) for path in artifact_paths(directory)
    }
