"""The atomic-data access layer — doc 09 §2, §5, §6; doc 03 §4.5; doc 04 §2.

One object that knows where the cached data is, checks it against the version lock before
anybody reads it, parses it into the types the rest of the framework holds, and remembers
what a run actually touched so the report can cite exactly that and nothing else.

## Why it is a store and not a set of module-level loaders

Three reasons, all of them doc 09's:

1. **Verification has to happen on every read, not once at import.** Doc 09 §5 requires
   the framework to "fail loudly if the upstream data has changed", and a file can change
   between one read and the next. The store holds the lock and consults it every time it
   goes to disk.
2. **The touched set is per-run state.** Doc 09 §6 wants the bibliography of "the
   specific sources a given run actually touched". That is a property of a run, so it
   lives on an object a run owns, not in module state that would accumulate across a
   sweep.
3. **Doc 09 §5 says the data-access layer is designed to be swappable from the outset**,
   because OpenADAS and datasheet restrictions mean a product cannot ship these files.
   A store is swappable; a module of functions reading a global path is not.

## Running under all three electron databases

Doc 09 §2.1 keeps three independent electron sets and makes the spread between them a
budgeted error term (doc 06 §4 term 2). :meth:`AtomicDataStore.each_electron_set` is what
makes that a loop rather than three call sites — because a sweep that needs three
hand-written call sites gets run once, at the point somebody writes the paper, and then
never again.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final, Self

from vpl.physics.atomic.dataset import (
    DATASETS,
    LOCK_FILENAME,
    DatasetId,
    DatasetLock,
    DatasetSpec,
    ElectronDatabase,
    bibliography,
    checked_cache_root,
)
from vpl.physics.atomic.lxcat import CrossSectionSet, parse_lxcat
from vpl.physics.atomic.nist import LineList, parse_nist_asd

__all__ = ["AtomicDataStore"]

#: The ion database of doc 09 §2.1 — the Ar+/Ar elastic and charge-exchange set doc 03
#: §4.5 calls the most consequential atomic-data choice in the project.
_ION_DATASET: Final[DatasetId] = DatasetId.LXCAT_PHELPS_ION

#: The line data of doc 09 §2.2, feeding the CR model of doc 04 §2.
_LINE_DATASET: Final[DatasetId] = DatasetId.NIST_ASD_ARGON

#: Cached files are text in a documented encoding. Reading them as the locale's encoding
#: would make a load succeed or fail depending on ``LANG``, which doc 00 E3 forbids.
_ENCODING: Final[str] = "utf-8"


class AtomicDataStore:
    """Verified, parsed, citable access to the doc 09 §2 datasets.

    Not a frozen dataclass, unlike most types in this package: the touched set is
    per-run state that the store exists to accumulate. Everything the store *reads* is
    immutable, and the lock it holds cannot be changed once given.
    """

    __slots__ = ("_cross_sections", "_lines", "_lock", "_root", "_touched")

    def __init__(self, root: Path, lock: DatasetLock) -> None:
        """Bind a cache directory to the lock that describes it.

        Args:
            root: The cache directory. Refused if it lies inside the installed package
                (doc 09 §5; see :func:`~vpl.physics.atomic.dataset.checked_cache_root`).
            lock: The recorded versions. Normally read from ``root`` by :meth:`open`;
                taken separately so that a caller can verify a cache against a lock
                committed elsewhere, which is what reproducing an archived run needs.
        """
        self._root: Final[Path] = checked_cache_root(root)
        self._lock: Final[DatasetLock] = lock
        self._touched: Final[dict[DatasetId, None]] = {}
        self._cross_sections: Final[dict[DatasetId, CrossSectionSet]] = {}
        self._lines: Final[dict[DatasetId, LineList]] = {}

    @classmethod
    def open(cls, root: Path) -> Self:
        """Open a cache directory and the lock that sits in it."""
        checked = checked_cache_root(root)
        return cls(checked, DatasetLock.read(checked / LOCK_FILENAME))

    # ── where things are ────────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """The cache directory."""
        return self._root

    @property
    def lock(self) -> DatasetLock:
        """The recorded versions. Read-only; a store never rewrites its own lock."""
        return self._lock

    def path(self, dataset_id: DatasetId) -> Path:
        """Where a dataset would be cached. Does not read it, so does not touch it."""
        return self._root / DATASETS[dataset_id].filename

    # ── reading, with verification ──────────────────────────────────────────────

    def read_bytes(self, dataset_id: DatasetId) -> bytes:
        """The cached bytes, verified against the lock — doc 09 §5.

        Raises:
            FileNotFoundError: If the dataset has not been fetched.
            ~vpl.physics.atomic.dataset.UpstreamDataChangedError: If the bytes on disk
                are not the bytes the lock recorded.
            ~vpl.physics.atomic.dataset.DatasetNotRecordedError: If the lock has no entry.
        """
        path = self.path(dataset_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"{dataset_id} is not cached at {path}. Run the fetch step; doc 09 §5 "
                "keeps the data itself out of the repository, so a fresh checkout has "
                "loaders and no data."
            )

        payload = path.read_bytes()
        self._lock.verify(dataset_id, payload)
        self._touched[dataset_id] = None
        return payload

    def read_text(self, dataset_id: DatasetId) -> str:
        """The cached bytes as UTF-8 text, verified against the lock."""
        return self.read_bytes(dataset_id).decode(_ENCODING)

    # ── the typed loaders ───────────────────────────────────────────────────────

    def _cross_section_set(self, dataset_id: DatasetId) -> CrossSectionSet:
        """Parse once per store.

        Memoised because a rate-coefficient sweep asks for the same set on every point,
        and because a set that were re-parsed per call could differ between two callers
        within one run if the file changed underneath them — which the verification would
        catch, but only after half the run had used the old table.
        """
        cached = self._cross_sections.get(dataset_id)
        if cached is not None:
            self._touched[dataset_id] = None
            return cached

        parsed = parse_lxcat(
            self.read_text(dataset_id), database=DATASETS[dataset_id].database_name
        )
        self._cross_sections[dataset_id] = parsed
        return parsed

    def electron_cross_sections(self, database: ElectronDatabase) -> CrossSectionSet:
        """One of the three independent electron sets of doc 09 §2.1."""
        return self._cross_section_set(database.dataset_id)

    def each_electron_set(self) -> Iterator[tuple[ElectronDatabase, CrossSectionSet]]:
        """Every electron database in turn — doc 09 §2.1, doc 06 §4 term 2.

        Yields lazily, so a sweep that stops early has not paid to parse the sets it
        never reached, and so a run that iterates this cites exactly the databases it got
        to (doc 09 §6).
        """
        for database in ElectronDatabase:
            yield database, self.electron_cross_sections(database)

    def ion_cross_sections(self) -> CrossSectionSet:
        """Ar+ + Ar elastic and charge exchange — doc 03 §4.5.

        A method of its own rather than another member of
        :class:`~vpl.physics.atomic.dataset.ElectronDatabase`: the ion set is not an
        alternative to any electron set, and making it one would let a sweep over "all
        the databases" run a case with no electron data in it at all.
        """
        return self._cross_section_set(_ION_DATASET)

    def argon_lines(self) -> LineList:
        """Ar I and Ar II line data — doc 09 §2.2, feeding doc 04 §2's CR model."""
        cached = self._lines.get(_LINE_DATASET)
        if cached is not None:
            self._touched[_LINE_DATASET] = None
            return cached

        parsed = parse_nist_asd(self.read_text(_LINE_DATASET))
        self._lines[_LINE_DATASET] = parsed
        return parsed

    # ── the citation ledger ─────────────────────────────────────────────────────

    @property
    def touched(self) -> tuple[DatasetId, ...]:
        """Which datasets this store has been asked for, in first-use order."""
        return tuple(self._touched)

    def touched_specs(self) -> tuple[DatasetSpec, ...]:
        """The specifications of those datasets, for a report to render."""
        return tuple(DATASETS[dataset_id] for dataset_id in self._touched)

    def bibliography(self) -> str:
        """BibTeX for what this run touched, pinned to the fetched versions — doc 09 §6.

        Doc 09 §6: the report emits the bibliography "for the specific sources a given run
        actually touched — not a static list. A run that used only the Phelps set does not
        cite Biagi."
        """
        return bibliography(self.touched_specs(), versions=self._lock.versions)

    def reset_touched(self) -> None:
        """Forget what has been touched, without discarding what has been parsed.

        For a driver that reuses one store across several runs and needs each run's
        bibliography to describe that run. The parsed data is unaffected because it is
        the *citation* that is per-run, not the data.
        """
        self._touched.clear()

    def __repr__(self) -> str:
        return (
            f"AtomicDataStore({str(self._root)!r}, {len(self._lock)} locked, "
            f"{len(self._touched)} touched)"
        )
