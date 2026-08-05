"""The atomic-data registry, version lock and citation ledger — doc 09 §1, §2, §5, §6.

Doc 09 §5 states the architecture of this module in two sentences:

    The repository stores *references and loaders*, not bulk third-party data. A setup
    script fetches from the primary source at install time, records the version hash, and
    fails loudly if the upstream data has changed.

Everything here follows from that.

## What is stored, and what is not

Stored: a :class:`DatasetSpec` per dataset — where it comes from, on what licence terms,
and how it must be cited — and a :class:`DatasetLock`, which is the SHA-256 of the bytes
that were actually fetched, with the version label and the access date doc 09 §1 requires
of a ``MEASURED`` entry.

Not stored: the bytes. LXCat's raw tables and OpenADAS's ADF files may not be
redistributed at all (doc 09 §5), and NIST's may — but the rule is applied uniformly,
because "the repository stores references and loaders" is an architectural statement and
not a per-licence accommodation. :func:`checked_cache_root` makes it executable: no cache
may live inside the installed package, which is the one path by which third-party data
ends up inside a wheel.

## Why the fetch takes its downloader as an argument

There is no default. A loader with a hard-wired ``urlopen`` cannot be exercised without a
network connection, so it would be the one part of the data path CI never checks — and it
is the part that decides whether a run is reproducible. :func:`fetch_dataset` takes a
``download`` callable, the setup script supplies a real one, and every test supplies a
function that returns bytes.

## Why the lock hashes itself

``manifest_sha256`` from :mod:`vpl.core.provenance` is the same canonical hash the run
identity of doc 08 §7 is built from. Applying it to the lock's own contents means a
hand-edited lock is detected rather than believed. A lock that can be edited undetectably
is not a lock; it is a comment.

## The citation ledger

Doc 09 §6 wants ``refs/CITATIONS.bib`` to accumulate every source while the report cites
"the specific sources a given run actually touched — not a static list. A run that used
only the Phelps set does not cite Biagi." Both come out of :func:`bibliography`, from the
same specs, so the accumulated ledger and the per-run subset cannot disagree.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Self

from vpl.core.params import ProvenanceClass
from vpl.core.provenance import ManifestValue, manifest_sha256

__all__ = [
    "DATASETS",
    "LOCK_FILENAME",
    "LXCAT_LICENCE",
    "NIST_LICENCE",
    "OPENADAS_LICENCE",
    "BundledDataError",
    "DatasetId",
    "DatasetLock",
    "DatasetNotRecordedError",
    "DatasetSpec",
    "DatasetVersion",
    "ElectronDatabase",
    "Licence",
    "LockIntegrityError",
    "UpstreamDataChangedError",
    "bibliography",
    "checked_cache_root",
    "fetch_dataset",
]

#: The lock lives beside the data it describes, so that moving a cache moves both.
LOCK_FILENAME: Final[str] = "atomic-datasets.lock.json"

_LOCK_SCHEMA: Final[str] = "vpl.atomic.datasets/1"

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

#: How much of a digest a citation carries. Enough to identify the fetch in a
#: bibliography without turning the entry into a wall of hex; the full digest is in the
#: lock, which is what a reproduction actually reads.
_DIGEST_PREFIX: Final[int] = 12

#: The installed ``vpl`` package. Nothing third-party may be cached inside it.
_PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class BundledDataError(RuntimeError):
    """An attempt to put bulk third-party data inside the installed package."""


class UpstreamDataChangedError(RuntimeError):
    """The bytes on disk are not the bytes the lock recorded — doc 09 §5."""


class DatasetNotRecordedError(LookupError):
    """The lock has no entry for a dataset that was asked for."""


class LockIntegrityError(ValueError):
    """The lock file is not a lock file, or has been edited since it was written."""


class DatasetId(StrEnum):
    """Every dataset doc 09 §2 registers for the atomic-data layer.

    A ``StrEnum`` because manifests are data (doc 08 §1 principle 4): an experiment names
    ``cross_sections: lxcat.biagi`` and gets back a checked member.
    """

    #: doc 09 §2.1 — e + Ar elastic, excitation, ionisation.
    LXCAT_PHELPS = "lxcat.phelps"

    #: doc 09 §2.1 — e + Ar full set from Magboltz v8.9.
    LXCAT_BIAGI = "lxcat.biagi"

    #: doc 09 §2.1 — e + Ar, independently evaluated.
    LXCAT_IST_LISBON = "lxcat.ist-lisbon"

    #: doc 09 §2.1 — Ar+ + Ar elastic and charge exchange. Doc 03 §4.5 makes this the
    #: single most consequential dataset in the project.
    LXCAT_PHELPS_ION = "lxcat.phelps-ion"

    #: doc 09 §2.2 — Ar I / Ar II wavelengths, levels, ``A_ul`` and statistical weights.
    NIST_ASD_ARGON = "nist.asd.argon"

    #: doc 09 §2.2 — ionisation/recombination coefficients. Registered for provenance and
    #: citation; the ADF reader belongs with the CR model that consumes it (doc 08 §2
    #: "CR model assembly: Build").
    OPENADAS_ADF11_ARGON = "openadas.adf11.argon"

    #: doc 09 §2.2 — photon emissivity coefficients. Registered, as above.
    OPENADAS_ADF15_ARGON = "openadas.adf15.argon"


class ElectronDatabase(StrEnum):
    """The three independent electron sets doc 09 §2.1 insists on keeping.

    Doc 09 §2.1: cross-section databases "disagree, sometimes by tens of percent in the
    excitation channels that drive the OES inference. Running the inference under all
    three and reporting the spread is an honest measure of atomic-data uncertainty — and
    it is a term in the error budget (doc 06 §4, term 2) rather than an unstated risk."

    They are a separate enum from :class:`DatasetId` so that "run this under every
    electron set" is ``for database in ElectronDatabase`` and not a hand-maintained list
    that quietly falls out of date. The ion database is deliberately not a member: it is
    not an alternative to any of these, and iterating it as one would run the sweep with
    no electron data at all.
    """

    PHELPS = "phelps"
    BIAGI = "biagi"
    IST_LISBON = "ist-lisbon"

    @property
    def dataset_id(self) -> DatasetId:
        return _ELECTRON_DATASETS[self]


_ELECTRON_DATASETS: Final[Mapping[ElectronDatabase, DatasetId]] = MappingProxyType(
    {
        ElectronDatabase.PHELPS: DatasetId.LXCAT_PHELPS,
        ElectronDatabase.BIAGI: DatasetId.LXCAT_BIAGI,
        ElectronDatabase.IST_LISBON: DatasetId.LXCAT_IST_LISBON,
    }
)


# ── licences ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Licence:
    """The doc 09 §5 redistribution table, as data.

    Attributes:
        name: What to call it.
        terms: The wording doc 09 §2 and §5 use, so that the code and the document can be
            diffed against each other.
        redistribute_raw: Whether the source files themselves may be redistributed.
        redistribute_derived: Whether quantities computed from them may be — rate
            coefficients, fitted curves, the reduced artifacts of doc 09 §4.2.
    """

    name: str
    terms: str
    redistribute_raw: bool
    redistribute_derived: bool

    def __post_init__(self) -> None:
        if self.redistribute_raw and not self.redistribute_derived:
            raise ValueError(
                f"{self.name}: no licence in doc 09 §5 permits the raw tables but forbids "
                "derived quantities. This combination is a typo, and the code would "
                "enforce it."
            )


#: doc 09 §2.1 and §5. "Derived rate coefficients: yes with attribution. **Do not
#: redistribute raw tables**."
LXCAT_LICENCE: Final[Licence] = Licence(
    name="LXCat",
    terms="Free; citation of the specific database required",
    redistribute_raw=False,
    redistribute_derived=True,
)

#: doc 09 §2.2 and §5. US Government work.
NIST_LICENCE: Final[Licence] = Licence(
    name="NIST ASD",
    terms="US Government work — public domain; citation requested",
    redistribute_raw=True,
    redistribute_derived=True,
)

#: doc 09 §2.2 and §5. "**Do not redistribute raw ADF files**; derived quantities
#: acceptable." Doc 09 §5 draws the commercial consequence: a product cannot ship these,
#: which is why the data-access layer is swappable from the outset.
OPENADAS_LICENCE: Final[Licence] = Licence(
    name="OpenADAS",
    terms="Free registration; academic use",
    redistribute_raw=False,
    redistribute_derived=True,
)


# ── the register ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One registered dataset: where it comes from and how it must be cited.

    Attributes:
        dataset_id: Its identity in a manifest.
        database_name: The name this set carries in a report. Distinct from the name in
            the file's own header, which cannot tell the Phelps *electron* database from
            the Phelps *ion* database — both call themselves "Phelps", and confusing them
            would silently swap the doc 03 §4.5 charge-exchange curve for an electron
            excitation set.
        title: What the dataset is, for a bibliography.
        source_url: The primary source. doc 09 §1 requires a ``MEASURED`` entry to name
            one.
        filename: What the fetch step writes it as inside a cache directory.
        provenance_class: doc 09 §1 class. Every dataset here is ``MEASURED``.
        licence: doc 09 §5 terms.
        citation_key: BibTeX key. Unique across the register.
        citation_author: The attribution the source asks for.
        citation_note: The exact wording the source requires, where it requires one.
            LXCat's databases publish a "HOW TO REFERENCE" string and doc 09 §2.1 makes
            citing the *specific* database a condition of use.
    """

    dataset_id: DatasetId
    database_name: str
    title: str
    source_url: str
    filename: str
    provenance_class: ProvenanceClass
    licence: Licence
    citation_key: str
    citation_author: str
    citation_note: str

    def __post_init__(self) -> None:
        if not self.citation_key.strip():
            raise ValueError(
                f"{self.dataset_id}: a registered dataset needs a citation key. doc 09 §6 "
                "emits a bibliography for what a run touched, and an entry it cannot name "
                "is an entry the report drops."
            )
        if not self.filename.strip():
            raise ValueError(f"{self.dataset_id}: needs a filename to cache under")
        if not self.source_url.startswith("https://"):
            raise ValueError(
                f"{self.dataset_id}: source_url must be an https URL, got {self.source_url!r}"
            )

    def bibtex(self, version: DatasetVersion | None = None) -> str:
        """A BibTeX entry for this dataset, optionally pinned to a fetched version.

        With a version the entry states the retrieval date, the upstream version label
        and the leading digits of the digest — which is what makes the citation point at
        the *bytes* a result was computed from rather than at a moving web page. Without
        one it says nothing about a fetch, rather than inventing a date.
        """
        note = [self.citation_note] if self.citation_note else []
        if version is not None:
            note.extend(
                [
                    f"retrieved {version.accessed_utc.date().isoformat()}",
                    f"upstream version {version.upstream_version}",
                    f"SHA-256 {version.sha256[:_DIGEST_PREFIX]}",
                ]
            )
        return "\n".join(
            [
                f"@misc{{{self.citation_key},",
                f"  author       = {{{self.citation_author}}},",
                f"  title        = {{{self.title}}},",
                f"  howpublished = {{\\url{{{self.source_url}}}}},",
                f"  note         = {{{'; '.join(note)}}},",
                "}",
                "",
            ]
        )


def _lxcat(
    dataset_id: DatasetId, *, database_name: str, title: str, key: str, contents: str
) -> DatasetSpec:
    """One LXCat database. The permalink and the required wording follow the identifier.

    Derived from ``dataset_id`` rather than from ``database_name`` because the electron
    and ion Phelps databases share a name and are different downloads.
    """
    slug = dataset_id.value.split(".", 1)[1]
    return DatasetSpec(
        dataset_id=dataset_id,
        database_name=database_name,
        title=title,
        source_url=f"https://nl.lxcat.net/data/set_type.php?database={slug}",
        filename=f"{dataset_id.value.replace('.', '-')}.txt",
        provenance_class=ProvenanceClass.MEASURED,
        licence=LXCAT_LICENCE,
        citation_key=key,
        citation_author="LXCat",
        citation_note=(
            f"{database_name} database, www.lxcat.net, retrieved at the date below; {contents}"
        ),
    )


#: Every dataset the atomic layer knows how to name, cite and verify — doc 09 §2.
DATASETS: Final[Mapping[DatasetId, DatasetSpec]] = MappingProxyType(
    {
        DatasetId.LXCAT_PHELPS: _lxcat(
            DatasetId.LXCAT_PHELPS,
            database_name="Phelps (e/Ar)",
            title="LXCat Phelps database: electron-impact cross sections for argon",
            key="lxcat_phelps",
            contents="e + Ar elastic, excitation and ionisation",
        ),
        DatasetId.LXCAT_BIAGI: _lxcat(
            DatasetId.LXCAT_BIAGI,
            database_name="Biagi (e/Ar)",
            title="LXCat Biagi database (Magboltz v8.9): electron-argon cross sections",
            key="lxcat_biagi",
            contents="e + Ar complete set",
        ),
        DatasetId.LXCAT_IST_LISBON: _lxcat(
            DatasetId.LXCAT_IST_LISBON,
            database_name="IST-Lisbon (e/Ar)",
            title="LXCat IST-Lisbon database: electron-argon cross sections",
            key="lxcat_ist_lisbon",
            contents="e + Ar, independently evaluated",
        ),
        DatasetId.LXCAT_PHELPS_ION: _lxcat(
            DatasetId.LXCAT_PHELPS_ION,
            database_name="Phelps (Ar+/Ar)",
            title="LXCat Phelps ion database: Ar+ in Ar elastic and charge exchange",
            key="lxcat_phelps_ion",
            contents="Ar+ + Ar elastic scattering and symmetric charge exchange",
        ),
        DatasetId.NIST_ASD_ARGON: DatasetSpec(
            dataset_id=DatasetId.NIST_ASD_ARGON,
            database_name="NIST ASD",
            title="NIST Atomic Spectra Database: Ar I and Ar II lines and levels",
            source_url="https://physics.nist.gov/asd",
            filename="nist-asd-argon.tsv",
            provenance_class=ProvenanceClass.MEASURED,
            licence=NIST_LICENCE,
            citation_key="nist_asd",
            citation_author="A. Kramida and Yu. Ralchenko and J. Reader and NIST ASD Team",
            citation_note=(
                "National Institute of Standards and Technology, Gaithersburg MD; "
                "accuracy grades ingested and propagated per doc 09 §2.2"
            ),
        ),
        DatasetId.OPENADAS_ADF11_ARGON: DatasetSpec(
            dataset_id=DatasetId.OPENADAS_ADF11_ARGON,
            database_name="OpenADAS ADF11",
            title="OpenADAS ADF11: argon ionisation and recombination coefficients",
            source_url="https://open.adas.ac.uk/adf11",
            filename="openadas-adf11-argon.dat",
            provenance_class=ProvenanceClass.MEASURED,
            licence=OPENADAS_LICENCE,
            citation_key="openadas_adf11_argon",
            citation_author="H. P. Summers and the ADAS Project",
            citation_note="Atomic Data and Analysis Structure, University of Strathclyde",
        ),
        DatasetId.OPENADAS_ADF15_ARGON: DatasetSpec(
            dataset_id=DatasetId.OPENADAS_ADF15_ARGON,
            database_name="OpenADAS ADF15",
            title="OpenADAS ADF15: argon photon emissivity coefficients",
            source_url="https://open.adas.ac.uk/adf15",
            filename="openadas-adf15-argon.dat",
            provenance_class=ProvenanceClass.MEASURED,
            licence=OPENADAS_LICENCE,
            citation_key="openadas_adf15_argon",
            citation_author="H. P. Summers and the ADAS Project",
            citation_note="Atomic Data and Analysis Structure, University of Strathclyde",
        ),
    }
)


def bibliography(
    specs: Iterable[DatasetSpec],
    versions: Mapping[DatasetId, DatasetVersion] | None = None,
) -> str:
    """Render BibTeX entries, in citation-key order — doc 09 §6.

    Args:
        specs: The datasets to cite. Pass ``DATASETS.values()`` for the accumulated
            ledger; pass what a run touched for the report's bibliography.
        versions: The lock's records, so each entry can pin the bytes it refers to.

    Returns:
        The entries, concatenated. Deterministic, so a regenerated ``CITATIONS.bib``
        diffs cleanly against the committed one.
    """
    ordered = sorted(specs, key=lambda spec: spec.citation_key)
    return "".join(
        spec.bibtex(None if versions is None else versions.get(spec.dataset_id)) for spec in ordered
    )


# ── the recorded version ────────────────────────────────────────────────────────


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """What was fetched, when, and from where — doc 09 §1's ``MEASURED`` requirements.

    Attributes:
        dataset_id: Which dataset.
        sha256: Digest of the bytes as fetched. The thing doc 09 §5 compares against.
        source_url: Where they came from, recorded rather than looked up again, because
            a register entry can be edited after a fetch.
        upstream_version: The version label the source advertises — an ASD release, an
            LXCat retrieval stamp. Required: doc 09 §1 defines ``MEASURED`` as "with
            version and access date", so a blank label makes the class a claim the record
            cannot support.
        accessed_utc: When. Timezone-aware, always stored as UTC.
        n_bytes: Size, as a cheap independent check on a truncated transfer.
    """

    dataset_id: DatasetId
    sha256: str
    source_url: str
    upstream_version: str
    accessed_utc: datetime
    n_bytes: int

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError(
                f"{self.dataset_id}: sha256 must be 64 lowercase hexadecimal characters, "
                f"got {self.sha256!r}"
            )
        if not self.upstream_version.strip():
            raise ValueError(
                f"{self.dataset_id}: upstream_version is required. doc 09 §1 defines "
                "MEASURED as 'from a standard evaluated database, with version and access "
                "date'."
            )
        if self.n_bytes <= 0:
            raise ValueError(f"{self.dataset_id}: n_bytes must be positive, got {self.n_bytes}")
        if self.accessed_utc.utcoffset() is None:
            raise ValueError(
                f"{self.dataset_id}: accessed_utc must be timezone-aware; a naive access "
                "date cannot be ordered against one recorded on another machine, which is "
                "the only reason to keep it"
            )
        object.__setattr__(self, "accessed_utc", self.accessed_utc.astimezone(UTC))

    @classmethod
    def of(
        cls,
        spec: DatasetSpec,
        payload: bytes,
        *,
        upstream_version: str,
        accessed_utc: datetime,
    ) -> Self:
        """Record a fetched payload against its specification."""
        return cls(
            dataset_id=spec.dataset_id,
            sha256=hashlib.sha256(payload).hexdigest(),
            source_url=spec.source_url,
            upstream_version=upstream_version,
            accessed_utc=accessed_utc,
            n_bytes=len(payload),
        )

    def to_dict(self) -> dict[str, ManifestValue]:
        """Flatten to the plain types JSON holds.

        The same treatment :meth:`vpl.core.provenance.Provenance.to_dict` gives a run
        record, for the same reason: a reader with neither this package nor Python
        installed still has to be able to check what a result was computed against.
        """
        return {
            "dataset_id": self.dataset_id.value,
            "sha256": self.sha256,
            "source_url": self.source_url,
            "upstream_version": self.upstream_version,
            "accessed_utc": self.accessed_utc.isoformat(),
            "n_bytes": self.n_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Rebuild a record from the lock file, checking every field rather than trusting it."""
        raw_id = data.get("dataset_id")
        try:
            dataset_id = DatasetId(str(raw_id))
        except ValueError as exc:
            raise ValueError(
                f"lock entry has an unknown dataset_id {raw_id!r}; expected one of "
                f"{', '.join(d.value for d in DatasetId)}"
            ) from exc

        def text(key: str) -> str:
            value = data.get(key)
            if not isinstance(value, str):
                raise ValueError(f"{dataset_id}: lock entry field {key!r} must be a string")
            return value

        n_bytes = data.get("n_bytes")
        if not isinstance(n_bytes, int) or isinstance(n_bytes, bool):
            raise ValueError(f"{dataset_id}: lock entry field 'n_bytes' must be an integer")

        return cls(
            dataset_id=dataset_id,
            sha256=text("sha256"),
            source_url=text("source_url"),
            upstream_version=text("upstream_version"),
            accessed_utc=datetime.fromisoformat(text("accessed_utc")),
            n_bytes=n_bytes,
        )


# ── the lock ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, eq=False)
class DatasetLock:
    """The recorded version of every fetched dataset — doc 09 §5.

    Immutable: :meth:`with_version` returns a new lock rather than mutating this one, so
    a lock that has been handed to a store cannot change underneath it mid-run.
    """

    versions: Mapping[DatasetId, DatasetVersion]

    def __post_init__(self) -> None:
        object.__setattr__(self, "versions", MappingProxyType(dict(self.versions)))

    def __len__(self) -> int:
        return len(self.versions)

    def __contains__(self, dataset_id: object) -> bool:
        return dataset_id in self.versions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatasetLock):
            return NotImplemented
        return dict(self.versions) == dict(other.versions)

    def ids(self) -> tuple[DatasetId, ...]:
        """The recorded datasets, in a stable order."""
        return tuple(sorted(self.versions, key=lambda dataset_id: dataset_id.value))

    def with_version(self, version: DatasetVersion) -> DatasetLock:
        """This lock plus one record."""
        return DatasetLock(versions={**self.versions, version.dataset_id: version})

    def verify(self, dataset_id: DatasetId, payload: bytes) -> DatasetVersion:
        """Check bytes against the record, failing loudly on any difference — doc 09 §5.

        Loudly and not merely differently: a rerun of an archived manifest that quietly
        picked up a revised cross-section table would produce a result the archive claims
        is reproducible and is not, which is exactly the failure doc 00 E3 exists to
        prevent.

        Raises:
            DatasetNotRecordedError: If the lock has no entry. A dataset present on disk
                but absent from the lock has no access date and no version, so it cannot
                be a ``MEASURED`` source (doc 09 §1) whatever it contains.
            UpstreamDataChangedError: If the digest differs.
        """
        recorded = self.versions.get(dataset_id)
        if recorded is None:
            raise DatasetNotRecordedError(
                f"{dataset_id} is not in the version lock. Run the fetch step to record "
                f"it; doc 09 §1 requires a version and an access date before a number "
                "from it can be called MEASURED."
            )

        digest = hashlib.sha256(payload).hexdigest()
        if digest != recorded.sha256:
            raise UpstreamDataChangedError(
                f"{dataset_id} has changed since it was fetched on "
                f"{recorded.accessed_utc.date().isoformat()} "
                f"(upstream version {recorded.upstream_version}).\n"
                f"  recorded: {recorded.sha256} ({recorded.n_bytes} bytes)\n"
                f"  on disk:  {digest} ({len(payload)} bytes)\n"
                f"doc 09 §5 requires this to fail loudly. Either restore the recorded "
                f"bytes, or re-fetch and record the new version deliberately — a result "
                f"computed against one table and reproduced against another is not "
                f"reproducible."
            )
        return recorded

    def write(self, path: Path) -> None:
        """Write the lock, with a digest of its own contents."""
        datasets: dict[str, ManifestValue] = {
            dataset_id.value: self.versions[dataset_id].to_dict() for dataset_id in self.ids()
        }
        document = {
            "schema": _LOCK_SCHEMA,
            "datasets": datasets,
            "lock_sha256": manifest_sha256(datasets),
        }
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read a lock, rejecting one that has been edited since it was written.

        Raises:
            FileNotFoundError: If there is no lock. The message names the fetch step,
                because "no lock" almost always means "the setup script has not been run"
                rather than anything the caller did wrong.
            LockIntegrityError: If the file is not a lock, or its self-digest disagrees
                with its contents.
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"no dataset lock at {path}. Run the fetch step to download the atomic "
                f"data and record its version hashes; doc 09 §5 keeps the data itself out "
                "of the repository."
            )

        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "lock_sha256" not in document:
            raise LockIntegrityError(
                f"{path} has no 'lock_sha256' field, so it is not a dataset lock written "
                f"by this framework (schema {_LOCK_SCHEMA})."
            )

        datasets = document.get("datasets")
        if not isinstance(datasets, dict):
            raise LockIntegrityError(f"{path}: 'datasets' must be a mapping")

        if manifest_sha256(datasets) != document["lock_sha256"]:
            raise LockIntegrityError(
                f"{path} has been edited since it was written: its contents do not match "
                f"its own digest. The lock is what makes doc 09 §5's change detection "
                "meaningful, so an edited one is refused rather than trusted."
            )

        return cls(
            versions={
                DatasetId(key): DatasetVersion.from_dict(entry) for key, entry in datasets.items()
            }
        )

    def __repr__(self) -> str:
        return f"DatasetLock({len(self)} datasets: {', '.join(d.value for d in self.ids())})"


# ── fetching ────────────────────────────────────────────────────────────────────


def checked_cache_root(root: Path) -> Path:
    """Resolve a cache directory, refusing one inside the installed package.

    Doc 09 §5 as an executable rule. Caching inside the package directory is the one path
    by which bulk third-party data ends up inside a built wheel — which for LXCat and
    OpenADAS would be a licence breach, and for all of them contradicts "the repository
    stores references and loaders, not bulk third-party data".

    Applied uniformly, including to NIST, which is public domain and could legally be
    shipped. The rule is architectural; carving out an exception per licence is how the
    architecture stops holding.
    """
    resolved = root.resolve()
    if resolved.is_relative_to(_PACKAGE_ROOT):
        raise BundledDataError(
            f"{resolved} is inside the installed vpl package ({_PACKAGE_ROOT}). doc 09 §5: "
            "the repository stores references and loaders, not bulk third-party data. Use "
            "a cache directory outside the package — the fetch step's default is a "
            "user-level cache, not a package directory."
        )
    return resolved


def fetch_dataset(
    spec: DatasetSpec,
    *,
    download: Callable[[str], bytes],
    destination: Path,
    upstream_version: str,
    now: Callable[[], datetime] = _utc_now,
) -> DatasetVersion:
    """Fetch one dataset into a cache and record what was fetched — doc 09 §5.

    The only function in this package that can reach outside the process, and it cannot
    do so by itself: ``download`` is supplied by the caller. That is deliberate. A loader
    with a hard-wired HTTP client would be the one part of the data path CI never
    exercises, and it is the part that decides whether a run can be reproduced.

    Args:
        spec: Which dataset.
        download: Given a URL, returns bytes. The setup script supplies a real client;
            tests supply a function.
        destination: Cache directory. Created if absent; refused if it lies inside the
            installed package (see :func:`checked_cache_root`).
        upstream_version: The version label the source advertises, for the record doc 09
            §1 requires.
        now: Clock, injected so a test can pin the access date.

    Returns:
        The record to add to the lock with :meth:`DatasetLock.with_version`.

    Raises:
        BundledDataError: If ``destination`` is inside the package.
        ValueError: If the download is empty — a proxy error page or a truncated transfer
            hashes just as well as real data, and would be recorded as a version.
    """
    root = checked_cache_root(destination)
    payload = download(spec.source_url)
    if not payload:
        raise ValueError(
            f"{spec.dataset_id}: the download from {spec.source_url} was empty. Recording "
            "a hash of nothing would make the lock certify a file that is not the dataset."
        )

    root.mkdir(parents=True, exist_ok=True)
    (root / spec.filename).write_bytes(payload)
    return DatasetVersion.of(spec, payload, upstream_version=upstream_version, accessed_utc=now())
