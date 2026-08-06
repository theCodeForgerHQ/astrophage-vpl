"""Artifact provenance — doc 08 §7.

Doc 08 §7 requires every artifact to embed ``manifest_sha256, git_commit, git_dirty,
seed, environment_lock_hash, created_utc, vpl_version, solver_versions, tier``. This
module is that record and the environment probes that fill it in.

Why it is a typed object rather than a dict assembled at each write site: doc 00 E3
promises that ``(manifest, commit, environment lock, seed)`` reproduces a result
bit-for-bit, and doc 00 C4 forbids hidden assumptions. A record with a field silently
omitted, or a tier defaulted to the flattering value, is a promise the archive cannot
keep — and doc 05 §7.2 makes reporting T1 as if it were T2 a project defect, so
:class:`Tier` never defaults anywhere it appears.

The two probes that can fail — :func:`git_state` and :func:`environment_lock_hash` —
degrade *loudly*. An unverifiable tree is reported dirty and an unlocked environment
names itself as the weaker source, because the failure that matters is a run claiming a
reproducibility it does not have.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import distributions
from pathlib import Path
from types import MappingProxyType
from typing import Final, Self, TypeAlias, TypeVar

from vpl import core as _core

__all__ = [
    "UNKNOWN_COMMIT",
    "EnvironmentLockSource",
    "ManifestValue",
    "Provenance",
    "ProvenanceValue",
    "Tier",
    "environment_lock_hash",
    "git_state",
    "manifest_sha256",
]

#: Reported as the commit when the checkout cannot be interrogated at all.
#:
#: A sentinel rather than an exception because provenance capture must never be the
#: reason a run dies; the record simply states what it could not establish.
UNKNOWN_COMMIT: Final[str] = "unknown"

#: Anything a manifest may hold once loaded from YAML.
#:
#: Quoted because the alias is *recursive* — see the identical note on
#: :data:`vpl.core.protocols.config.ConfigValue`. A ``TypeAlias`` assignment evaluates its
#: right-hand side eagerly, so ``ManifestValue`` must be forward-referenced as a string.
ManifestValue: TypeAlias = (
    "str | int | float | bool | Sequence[ManifestValue] | Mapping[str, ManifestValue] | None"
)

#: Anything a serialised provenance record may hold: the types HDF5 attributes and YAML
#: sidecars carry without a custom representer (doc 08 §7).
ProvenanceValue: TypeAlias = str | bool | int | dict[str, str]

#: The enum coerced by :func:`_parse_enum`. The pre-PEP-695 spelling of
#: ``def _parse_enum[E: StrEnum]``.
E = TypeVar("E", bound=StrEnum)

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

_LOCK_FILENAME: Final[str] = "uv.lock"

# Long enough that a cold index refresh on a large checkout completes; short enough that
# a hung git cannot stall a batch of runs.
_GIT_TIMEOUT_S: Final[float] = 30.0


class Tier(StrEnum):
    """Validation tier of a result — doc 05 §7.2.

    The three tiers answer different questions and are not interchangeable:

    - ``T0`` **Consistency.** Same model, no noise. Tests that the code is
      self-consistent. Failing T0 means a bug, and nothing else is meaningful until it
      passes.
    - ``T1`` **Inverse crime.** Same model, with noise. Optimistic — the upper bound on
      achievable accuracy, not an achievable accuracy.
    - ``T2`` **Honest.** Mismatched forward and inverse models, with noise and imperfect
      calibration. This is the number quoted publicly.

    Doc 05 §7.2 treats reporting T1 as if it were T2 as a project defect, which is why
    the tier is a required, non-defaulting field of every record, and why the values are
    the exact label text doc 13 §3.2 renders onto every accuracy figure.
    """

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"


class EnvironmentLockSource(StrEnum):
    """Where :func:`environment_lock_hash` got its answer.

    Doc 13 §1 makes the environment lock one of the four inputs that must reproduce a
    result. The two sources are not equally strong, so the record names which one it
    used rather than presenting both as "the environment hash".

    - ``UV_LOCK`` — the resolved lock file. Pins the whole resolution, including
      transitive versions, and is what ``vpl reproduce`` can restore from.
    - ``INSTALLED_DISTRIBUTIONS`` — a scan of what happens to be importable. Pins the
      versions actually in use but not the resolution that produced them, so it records
      the environment without being able to rebuild it.
    """

    UV_LOCK = "uv.lock"
    INSTALLED_DISTRIBUTIONS = "importlib.metadata"


# ── canonical hashing ────────────────────────────────────────────────────────────────


def manifest_sha256(payload: ManifestValue) -> str:
    """Hash a manifest structure canonically.

    Canonical means the digest is invariant under things that do not change the
    experiment, and sensitive to everything that does:

    - Mapping key order does not change it — two YAML loaders that ordered the same
      manifest differently must produce one run identity, not two.
    - Sequence order *does* change it, because doc 08 §6 orders instruments and noise
      sources meaningfully.
    - Floats are serialised at ``repr`` precision, which round-trips exactly. Doc 00 E3
      promises bit-for-bit reproducibility, so a digest that truncated a 1e-17
      difference would let two genuinely different runs claim one identity — the failure
      this function exists to prevent.

    Keys are assumed to be strings, as they are in every manifest doc 08 §6 describes.

    Args:
        payload: A nested mapping/sequence/scalar structure, typically a loaded manifest.

    Returns:
        The lowercase hexadecimal SHA-256 of the canonical UTF-8 encoding.

    Raises:
        ValueError: If the payload contains ``nan`` or an infinity. JSON has no spelling
            for either, and the non-standard one Python emits by default cannot be read
            back by another tool — so the archive would hold a digest nobody else could
            reproduce.
        TypeError: If the payload contains a value JSON cannot represent. Convert it to
            a scalar at the boundary where its meaning is still known.
    """
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError(f"manifest contains a value JSON cannot represent: {exc}") from exc
    except TypeError as exc:
        raise TypeError(f"manifest is not JSON-serialisable: {exc}") from exc

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── environment probes ───────────────────────────────────────────────────────────────


def _default_repo_root() -> Path:
    """The checkout this module was imported from, or the working directory.

    Walking up from ``__file__`` rather than trusting the process's working directory: a
    run launched from ``experiments/`` must record the same commit as one launched from
    the repository root.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / _LOCK_FILENAME).is_file() or (candidate / ".git").exists():
            return candidate

    # An installed wheel with no checkout above it. The probes then degrade explicitly.
    return Path.cwd()


def _git_output(root: Path, *args: str) -> str | None:
    """Stripped stdout of a git invocation, or ``None`` if it could not be run."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        # git absent, the directory gone, or the call hung. All three mean one thing to
        # the caller: the checkout could not be interrogated.
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def git_state(repo_root: Path | None = None) -> tuple[str, bool]:
    """The commit a run is executing, and whether the tree differs from it.

    Doc 13 §2 makes ``git_dirty`` the gate that keeps unreproducible results out of the
    release archive, which fixes the direction this function fails in: an unverifiable
    tree is reported **dirty**. Claiming a clean tree that was never checked would let
    exactly the results nobody can regenerate through the gate that exists to stop them.

    Untracked files count as dirty. An untracked module can be imported by the run and
    will not exist for anybody else, so it is as unreproducible as an uncommitted edit.

    Args:
        repo_root: Directory to interrogate. Used as given; when ``None``, the checkout
            containing this module is discovered instead.

    Returns:
        ``(commit_sha, is_dirty)``. The sha is :data:`UNKNOWN_COMMIT` when git is
        absent, the directory is not a repository, or no commit exists yet — each of
        which also yields ``is_dirty=True``.
    """
    root = repo_root if repo_root is not None else _default_repo_root()

    commit = _git_output(root, "rev-parse", "HEAD")
    if commit is None:
        return UNKNOWN_COMMIT, True

    status = _git_output(root, "status", "--porcelain")
    if status is None:
        # The commit is known but its cleanliness is not, so assume the worst.
        return commit, True

    return commit, bool(status)


def environment_lock_hash(repo_root: Path | None = None) -> tuple[str, EnvironmentLockSource]:
    """Hash the resolved dependency set, and say where the answer came from.

    Doc 13 §1 lists the environment lock among the four inputs that must reproduce a
    result. The lock file is preferred because it pins the whole resolution; the
    installed-distribution scan is a real but weaker guarantee, and the returned source
    is what lets a reader tell the two apart instead of having to infer it.

    Args:
        repo_root: Directory to look for the lock file in. Used as given; when ``None``,
            the checkout containing this module is discovered instead.

    Returns:
        ``(sha256_hex, source)``.
    """
    root = repo_root if repo_root is not None else _default_repo_root()

    lock = root / _LOCK_FILENAME
    if lock.is_file():
        return hashlib.sha256(lock.read_bytes()).hexdigest(), EnvironmentLockSource.UV_LOCK

    # Distribution scan order follows the filesystem, so it is sorted before hashing;
    # unsorted, one unchanged environment would hash differently on consecutive runs.
    pins = sorted(f"{dist.name}=={dist.version}" for dist in distributions())
    digest = hashlib.sha256("\n".join(pins).encode("utf-8")).hexdigest()
    return digest, EnvironmentLockSource.INSTALLED_DISTRIBUTIONS


# ── field parsing ────────────────────────────────────────────────────────────────────


def _parse_enum(value: str | E, enum_cls: type[E], name: str) -> E:
    """Coerce to a member of ``enum_cls``, naming the field in the failure."""
    try:
        return enum_cls(value)
    except ValueError as exc:
        permitted = ", ".join(member.value for member in enum_cls)
        raise ValueError(f"{name} must be one of [{permitted}], got {value!r}") from exc


def _require(data: Mapping[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"provenance record is missing {key!r}")
    return data[key]


def _as_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__} {value!r}")
    return value


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__} {value!r}")
    return value


def _as_int(value: object, name: str) -> int:
    # ``bool`` is a subclass of ``int``; a flag arriving where a seed belongs is a
    # mistake worth naming rather than silently reading as 0 or 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__} {value!r}")
    return value


def _as_str_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__} {value!r}")

    return {
        _as_str(key, f"{name} key"): _as_str(entry, f"{name}[{key!r}]")
        for key, entry in value.items()
    }


def _as_utc_datetime(value: object, name: str) -> datetime:
    try:
        return datetime.fromisoformat(_as_str(value, name))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 timestamp, got {value!r}") from exc


# ── the record ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Provenance:
    """What every artifact embeds — doc 08 §7.

    Frozen and slotted because a record is written into an archive and then read by
    people who cannot re-derive it: a field mutated after capture, or a typo'd attribute
    that attaches itself silently, would be undetectable afterwards.

    Attributes:
        manifest_sha256: Canonical digest of the experiment manifest, from
            :func:`manifest_sha256`. With the commit, the lock and the seed this is the
            identity doc 00 E3 promises is sufficient to reproduce the run.
        git_commit: The commit the run executed, or :data:`UNKNOWN_COMMIT`.
        git_dirty: Whether the working tree differed from that commit. Doc 13 §2 fails
            the run on a dirty tree in release mode.
        seed: The root seed. Doc 10 §5 requires per-stream seeds, which are derived from
            this one — recording the root keeps the derivation reproducible while
            leaving the streams independently perturbable.
        environment_lock_hash: Digest of the resolved dependency set.
        environment_lock_source: Which source that digest came from, since the two are
            not equally strong.
        created_utc: When the record was captured. Always timezone-aware UTC.
        vpl_version: Version of the running core.
        solver_versions: Versions of the numerical libraries this run depended on
            (``dolfinx``, ``petsc``, ``numpyro``, …). Read-only.
        tier: Which of doc 05 §7.2's three claims this artifact supports. Never defaulted.
    """

    manifest_sha256: str
    git_commit: str
    git_dirty: bool
    seed: int
    environment_lock_hash: str
    environment_lock_source: EnvironmentLockSource
    created_utc: datetime
    vpl_version: str
    # Excluded from the hash, not from equality. ``MappingProxyType`` is unhashable, and
    # the alternative — dropping it from equality too — would make two records that
    # disagree about which PETSc built them compare equal, which is worse than a hash
    # that leans on the other eight fields.
    solver_versions: Mapping[str, str] = field(hash=False)
    tier: Tier

    def __post_init__(self) -> None:
        for name in ("manifest_sha256", "environment_lock_hash"):
            digest = getattr(self, name)
            if not _SHA256_PATTERN.fullmatch(digest):
                # Catches storing a path or a truncated display hash, either of which
                # would be indistinguishable from a real digest once in the archive.
                raise ValueError(
                    f"{name} must be 64 lowercase hexadecimal characters, got {digest!r}"
                )

        if self.seed < 0:
            raise ValueError(
                f"seed indexes a generator's state and cannot be negative, got {self.seed}"
            )

        if self.created_utc.utcoffset() is None:
            # A naive timestamp cannot be ordered against one written elsewhere, which
            # is the only reason to record it.
            raise ValueError(f"created_utc must be timezone-aware, got {self.created_utc!r}")

        object.__setattr__(self, "created_utc", self.created_utc.astimezone(UTC))
        object.__setattr__(self, "tier", _parse_enum(self.tier, Tier, "tier"))
        object.__setattr__(
            self,
            "environment_lock_source",
            _parse_enum(
                self.environment_lock_source, EnvironmentLockSource, "environment_lock_source"
            ),
        )
        # Copy before wrapping: a proxy over the caller's own dict would let a later
        # edit rewrite the history of an artifact that has already been written.
        object.__setattr__(self, "solver_versions", MappingProxyType(dict(self.solver_versions)))

    @classmethod
    def capture(
        cls,
        *,
        manifest_sha256: str,
        seed: int,
        tier: Tier,
        solver_versions: Mapping[str, str],
        repo_root: Path | None = None,
    ) -> Self:
        """Record the current environment around the caller's four facts.

        The split is deliberate. The commit, the tree state, the environment lock, the
        timestamp and the core version are properties of the machine and are read, never
        supplied. The manifest digest, the seed, the tier and the solver versions are
        properties of the experiment and are supplied, never guessed — doc 00 C4 leaves
        no third category, and ``tier`` in particular has no defensible default.

        Args:
            manifest_sha256: Digest from :func:`manifest_sha256`.
            seed: The run's root seed.
            tier: Which claim the resulting artifact supports (doc 05 §7.2).
            solver_versions: Versions of the numerical libraries in play. Copied.
            repo_root: Checkout to interrogate; discovered from this module when ``None``.

        Returns:
            A complete record, timestamped now.
        """
        commit, dirty = git_state(repo_root)
        lock_hash, lock_source = environment_lock_hash(repo_root)

        return cls(
            manifest_sha256=manifest_sha256,
            git_commit=commit,
            git_dirty=dirty,
            seed=seed,
            environment_lock_hash=lock_hash,
            environment_lock_source=lock_source,
            created_utc=datetime.now(UTC),
            vpl_version=_core.__version__,
            solver_versions=solver_versions,
            tier=tier,
        )

    def to_dict(self) -> dict[str, ProvenanceValue]:
        """Flatten to the plain types an HDF5 attribute set or YAML sidecar can hold.

        Enums become their label text and the timestamp becomes ISO 8601 with an
        explicit offset, so a reader with neither this package nor Python installed can
        still make sense of the record (doc 08 §7, doc 13 §1).
        """
        return {
            "manifest_sha256": self.manifest_sha256,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "seed": self.seed,
            "environment_lock_hash": self.environment_lock_hash,
            "environment_lock_source": self.environment_lock_source.value,
            "created_utc": self.created_utc.isoformat(),
            "vpl_version": self.vpl_version,
            "solver_versions": dict(self.solver_versions),
            "tier": self.tier.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Rebuild a record read back from an artifact.

        Every field is checked rather than trusted. The input has been through HDF5 or
        YAML and may have been hand-edited in between, and a record that quietly accepts
        a seed of ``"20260804"`` or a tier of ``"T2-ish"`` is a record that cannot be
        relied on to mean what it says.

        Raises:
            ValueError: If a field is missing, or a tier, source or timestamp is
                unrecognisable.
            TypeError: If a field is present but of the wrong type.
        """
        return cls(
            manifest_sha256=_as_str(_require(data, "manifest_sha256"), "manifest_sha256"),
            git_commit=_as_str(_require(data, "git_commit"), "git_commit"),
            git_dirty=_as_bool(_require(data, "git_dirty"), "git_dirty"),
            seed=_as_int(_require(data, "seed"), "seed"),
            environment_lock_hash=_as_str(
                _require(data, "environment_lock_hash"), "environment_lock_hash"
            ),
            environment_lock_source=_parse_enum(
                _as_str(_require(data, "environment_lock_source"), "environment_lock_source"),
                EnvironmentLockSource,
                "environment_lock_source",
            ),
            created_utc=_as_utc_datetime(_require(data, "created_utc"), "created_utc"),
            vpl_version=_as_str(_require(data, "vpl_version"), "vpl_version"),
            solver_versions=_as_str_mapping(_require(data, "solver_versions"), "solver_versions"),
            tier=_parse_enum(_as_str(_require(data, "tier"), "tier"), Tier, "tier"),
        )
