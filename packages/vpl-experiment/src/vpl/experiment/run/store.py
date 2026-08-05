"""The run directory layout and the run index — doc 13 §2, doc 13 §5, doc 08 §7.

    <store>/
      index.json                     # a cache; see below
      <run-id>/
        manifest.yaml                # the archived manifest — what `vpl reproduce` runs
        provenance.json              # the doc 08 §7 block, as a sidecar
        run.yaml                     # the doc 13 §2 record
        artifacts/                   # HDF5, Parquet, Zarr per doc 08 §7's table
      .reproductions/<run-id>/       # scratch; see `vpl.experiment.run.reproduce`

doc 08 §7's row for manifests and provenance reads "YAML + JSON sidecar", and that is
what the three files are: the manifest and the human-readable record in YAML, the machine
block in JSON.

## The run identity

``<date>-<name>-<manifest digest prefix>``, as doc 13 §2 writes it
(``20260804-b02-4a7f2e91``). It is deterministic, which has one consequence worth stating
plainly: **two runs of the same manifest on the same day are the same run.** That follows
from doc 00 E3 — the same manifest at the same commit produces the same result — so
``vpl run`` refuses to overwrite an existing directory and points at ``vpl reproduce``,
which is what someone re-running a manifest almost always meant.

## The index is a cache, not an asset

doc 13 §5 treats regenerable data as a cache, and the index is the smallest instance of
that: it is rebuilt from the directories whenever it is missing or a lookup misses. A run
that exists on disk is therefore never lost to a corrupt index, and the index never has to
be repaired by hand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import yaml

from vpl.core.provenance import Provenance
from vpl.experiment.manifest import Manifest, load_manifest
from vpl.experiment.run.record import RunRecord, RunStatus

__all__ = [
    "ARTIFACTS_DIRNAME",
    "INDEX_FILENAME",
    "MANIFEST_FILENAME",
    "PROVENANCE_FILENAME",
    "RECORD_FILENAME",
    "REPRODUCTIONS_DIRNAME",
    "IndexEntry",
    "RunDirectory",
    "RunNotFoundError",
    "RunStore",
    "run_id_for",
]

MANIFEST_FILENAME: Final[str] = "manifest.yaml"
PROVENANCE_FILENAME: Final[str] = "provenance.json"
RECORD_FILENAME: Final[str] = "run.yaml"
ARTIFACTS_DIRNAME: Final[str] = "artifacts"
INDEX_FILENAME: Final[str] = "index.json"

#: Where ``vpl reproduce`` re-executes. Dot-prefixed so it cannot be mistaken for a run id
#: and is skipped by the index scan.
REPRODUCTIONS_DIRNAME: Final[str] = ".reproductions"

#: Characters of the manifest digest in a run id — doc 13 §2's ``4a7f2e91``.
_ID_DIGEST_CHARS: Final[int] = 8

#: Longest experiment-name slug a run id carries. A path component, not a magnitude:
#: names are free text and some filesystems still stop at 255 bytes.
_ID_SLUG_CHARS: Final[int] = 48

_NON_SLUG = re.compile(r"[^a-z0-9]+")

#: Date format of the identity's first field — doc 13 §2's ``20260804``.
_ID_DATE_FORMAT: Final[str] = "%Y%m%d"


class RunNotFoundError(LookupError):
    """No run in the store answers to that identity."""


def _slug(name: str) -> str:
    """A path-safe form of an experiment name.

    Lossy on purpose: the identity's uniqueness comes from the manifest digest, and the
    name is there so a human scanning a directory listing can tell the runs apart.
    """
    reduced = _NON_SLUG.sub("-", name.strip().lower()).strip("-")
    return (reduced or "experiment")[:_ID_SLUG_CHARS].strip("-")


def run_id_for(manifest: Manifest, when: datetime) -> str:
    """The identity doc 13 §2 gives a run: date, name and manifest digest."""
    return (
        f"{when.strftime(_ID_DATE_FORMAT)}-{_slug(manifest.experiment.name)}-"
        f"{manifest.sha256[:_ID_DIGEST_CHARS]}"
    )


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One line of the run index — enough to find a run without opening it."""

    id: str
    manifest_sha256: str
    experiment: str
    tier: str
    status: RunStatus
    started_utc: str

    @classmethod
    def of(cls, record: RunRecord, experiment: str) -> IndexEntry:
        return cls(
            id=record.id,
            manifest_sha256=record.manifest_sha256,
            experiment=experiment,
            tier=record.tier.value,
            status=record.status,
            started_utc=record.started_utc.isoformat(),
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "id": self.id,
            "manifest_sha256": self.manifest_sha256,
            "experiment": self.experiment,
            "tier": self.tier,
            "status": self.status.value,
            "started_utc": self.started_utc,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> IndexEntry:
        return cls(
            id=str(data["id"]),
            manifest_sha256=str(data["manifest_sha256"]),
            experiment=str(data["experiment"]),
            tier=str(data["tier"]),
            status=RunStatus(str(data["status"])),
            started_utc=str(data["started_utc"]),
        )


@dataclass(frozen=True, slots=True)
class RunDirectory:
    """One run on disk.

    Every accessor reads from the files rather than caching, because a run directory is
    written by one process and read by others — a reproduction, a comparison, a report —
    and a cached record would be stale exactly when it mattered.
    """

    id: str
    path: Path

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_FILENAME

    @property
    def provenance_path(self) -> Path:
        return self.path / PROVENANCE_FILENAME

    @property
    def record_path(self) -> Path:
        return self.path / RECORD_FILENAME

    @property
    def artifacts_path(self) -> Path:
        return self.path / ARTIFACTS_DIRNAME

    def read_manifest(self) -> Manifest:
        """The archived manifest — what ``vpl reproduce`` re-executes."""
        return load_manifest(self.manifest_path)

    def read_provenance(self) -> Provenance:
        """The doc 08 §7 block from the JSON sidecar."""
        return Provenance.from_dict(json.loads(self.provenance_path.read_text(encoding="utf-8")))

    def read_record(self) -> RunRecord:
        """The doc 13 §2 record from ``run.yaml``."""
        loaded = yaml.safe_load(self.record_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{self.record_path}: a run record is a mapping")
        return RunRecord.from_mapping(loaded)

    def write_manifest(self, manifest: Manifest) -> None:
        self.manifest_path.write_text(manifest.to_yaml(), encoding="utf-8")

    def write_provenance(self, provenance: Provenance) -> None:
        self.provenance_path.write_text(
            json.dumps(provenance.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_record(self, record: RunRecord) -> None:
        self.record_path.write_text(
            yaml.safe_dump(record.to_mapping(), sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )

    def is_run(self) -> bool:
        """Whether this directory holds a run at all — used by the index rebuild."""
        return self.record_path.is_file() and self.manifest_path.is_file()

    def __repr__(self) -> str:
        return f"RunDirectory({self.id!r})"


class RunStore:
    """A directory of runs, with an index that can always be rebuilt from it."""

    __slots__ = ("root",)

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ── layout ──────────────────────────────────────────────────────────────────

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    @property
    def reproductions_path(self) -> Path:
        return self.root / REPRODUCTIONS_DIRNAME

    def directory(self, run_id: str) -> RunDirectory:
        """The directory for an identity, whether or not it exists yet."""
        return RunDirectory(id=run_id, path=self.root / run_id)

    def create(self, run_id: str, *, force: bool = False) -> RunDirectory:
        """Make a run directory, refusing to overwrite one.

        Raises:
            FileExistsError: If the directory exists and ``force`` is false. Overwriting
                would destroy the archived result that ``vpl reproduce`` compares against,
                and re-running an unchanged manifest is what ``vpl reproduce`` is for.
        """
        run = self.directory(run_id)
        if run.path.exists() and not force:
            raise FileExistsError(
                f"run {run_id} already exists at {run.path}. The identity is the manifest "
                "digest (doc 13 §2), so this is the same experiment: use `vpl reproduce "
                f"{run_id}` to re-execute and verify it, or --force to overwrite."
            )
        run.artifacts_path.mkdir(parents=True, exist_ok=True)
        return run

    # ── discovery ───────────────────────────────────────────────────────────────

    def _scan(self) -> Iterator[RunDirectory]:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            run = RunDirectory(id=path.name, path=path)
            if run.is_run():
                yield run

    def reindex(self) -> tuple[IndexEntry, ...]:
        """Rebuild the index from the directories and write it out."""
        entries: list[IndexEntry] = []
        for run in self._scan():
            record = run.read_record()
            manifest_name = run.read_manifest().experiment.name
            entries.append(IndexEntry.of(record, manifest_name))

        if entries or self.root.is_dir():
            self.root.mkdir(parents=True, exist_ok=True)
            self.index_path.write_text(
                json.dumps([entry.to_mapping() for entry in entries], indent=2) + "\n",
                encoding="utf-8",
            )
        return tuple(entries)

    def index(self) -> tuple[IndexEntry, ...]:
        """The run index, rebuilding it if it is absent or unreadable."""
        if not self.index_path.is_file():
            return self.reindex()
        try:
            loaded = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A truncated index is a cache miss, not a failure: doc 13 §5's rule is that
            # regenerable data is regenerated rather than repaired.
            return self.reindex()
        return tuple(IndexEntry.from_mapping(entry) for entry in loaded)

    def resolve(self, run_id: str) -> RunDirectory:
        """Find a run by its identity or by an unambiguous prefix of it.

        Raises:
            RunNotFoundError: If nothing matches, or if a prefix matches more than one
                run. An ambiguous prefix is refused rather than resolved to the first
                match, because the two runs it names are usually the two a reader is
                trying to tell apart.
        """
        exact = self.directory(run_id)
        if exact.is_run():
            return exact

        known = [entry.id for entry in self.index()]
        matches = sorted(identity for identity in known if identity.startswith(run_id))

        if len(matches) == 1:
            return self.directory(matches[0])
        if len(matches) > 1:
            raise RunNotFoundError(
                f"{run_id!r} is an ambiguous run identity: it matches "
                f"{', '.join(matches)}. Give enough of the identity to pick one."
            )
        if not known:
            raise RunNotFoundError(f"there are no runs in {self.root}")
        raise RunNotFoundError(
            f"no run {run_id!r} in {self.root}. This store holds: {', '.join(sorted(known))}"
        )

    def __repr__(self) -> str:
        return f"RunStore({str(self.root)!r})"
