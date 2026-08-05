"""The run record — doc 13 §2, written verbatim.

doc 13 §2 prints the record a run captures, and this is that structure as a type. Three
of its fields are worth explaining because they are the ones that would otherwise be
quietly dropped.

**Per-stream seeds, not one global seed.** doc 13 §2 lists five and doc 10 §5 gives the
reason: with a single global RNG, adding one noise source shifts every subsequent draw and
two runs that should be comparable are not. The record writes out *every* stream
:class:`~vpl.core.random.Stream` defines, derived from the manifest's one root seed, so
that a reader can check any of them without re-deriving it — and so that a stream added
later is visibly absent from an older record rather than silently assumed.

**``quarantined_cases``, and the failure that is never dropped.** doc 10 §6 is explicit:
"a quarantined case is never silently dropped from statistics", because silently
discarding failed runs biases every aggregate. For a single run the rule reduces to: the
directory and this record survive the failure, with ``status: failed``, a count of one and
the diagnostic that caused it. The alternative — an exception and no trace — is precisely
the silent drop.

**``hardware`` records what the standard library knows and no more.** doc 13 §2 lists a
GPU model and a RAM figure; probing either needs a dependency this package will not take
for a metadata field, and a *guessed* hardware line is worse than an absent one (doc 00
C4). What determines the result is captured by the environment lock and the container
digest, both of which are here.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Self

from vpl.core.provenance import EnvironmentLockSource, Provenance, Tier
from vpl.core.random import Stream, stream_seed

__all__ = ["CONTAINER_DIGEST_VARIABLE", "RunRecord", "RunStatus", "hardware_description"]

#: Environment variable naming the container a run executed in — doc 13 §2.
#:
#: Read rather than probed. A container cannot reliably identify its own image from the
#: inside, and the orchestration that started it can; an absent variable is recorded as
#: absent rather than guessed.
CONTAINER_DIGEST_VARIABLE: Final[str] = "VPL_CONTAINER_DIGEST"


class RunStatus(StrEnum):
    """Where a run got to — doc 13 §2's ``status``.

    ``RUNNING`` is written *before* execution starts, so an interrupted run is
    discoverable as interrupted rather than absent. doc 10 §6 requires the campaign to be
    resumable, and a queue cannot resume what it cannot see.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def hardware_description() -> Mapping[str, str]:
    """What the standard library can say about this machine. See the module docstring."""
    return MappingProxyType(
        {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cpu_count": str(os.cpu_count() or 0),
            "python": platform.python_version(),
        }
    )


def stream_seeds(root_seed: int) -> Mapping[str, int]:
    """Every stream's seed, derived from the run's one root seed — doc 10 §5."""
    return MappingProxyType({stream.value: stream_seed(root_seed, stream) for stream in Stream})


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"run record: {name} must be a mapping, got {type(value).__name__}")
    return {str(key): entry for key, entry in value.items()}


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run's provenance sidecar in human-readable form — doc 13 §2.

    Frozen: it is written once and read by people who cannot re-derive it. The one
    transition it undergoes — ``running`` to ``completed`` or ``failed`` — is expressed by
    :meth:`finished` returning a new record, not by mutating this one.
    """

    id: str
    manifest_sha256: str
    git_commit: str
    git_dirty: bool
    environment_lock_sha256: str
    environment_lock_source: EnvironmentLockSource
    container_digest: str | None
    seeds: Mapping[str, int]
    hardware: Mapping[str, str]
    solver_versions: Mapping[str, str]
    data_versions: Mapping[str, str]
    parameter_sources: Mapping[str, str]
    artifacts: Mapping[str, str]
    tier: Tier
    started_utc: datetime
    duration_s: float
    status: RunStatus
    quarantined_cases: int
    failure: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "seeds",
            "hardware",
            "solver_versions",
            "data_versions",
            "parameter_sources",
            "artifacts",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.started_utc.utcoffset() is None:
            raise ValueError("run record: started_utc must be timezone-aware")
        object.__setattr__(self, "started_utc", self.started_utc.astimezone(UTC))

    # ── construction ────────────────────────────────────────────────────────────

    @classmethod
    def opened(
        cls,
        *,
        run_id: str,
        provenance: Provenance,
        parameter_sources: Mapping[str, str],
        data_versions: Mapping[str, str] = MappingProxyType({}),
    ) -> Self:
        """The record written before the run starts — ``status: running``."""
        return cls(
            id=run_id,
            manifest_sha256=provenance.manifest_sha256,
            git_commit=provenance.git_commit,
            git_dirty=provenance.git_dirty,
            environment_lock_sha256=provenance.environment_lock_hash,
            environment_lock_source=provenance.environment_lock_source,
            container_digest=os.environ.get(CONTAINER_DIGEST_VARIABLE),
            seeds=stream_seeds(provenance.seed),
            hardware=hardware_description(),
            solver_versions=provenance.solver_versions,
            data_versions=data_versions,
            parameter_sources=parameter_sources,
            artifacts={},
            tier=provenance.tier,
            started_utc=provenance.created_utc,
            duration_s=0.0,
            status=RunStatus.RUNNING,
            quarantined_cases=0,
        )

    def finished(self, *, duration_s: float, artifacts: Mapping[str, str]) -> RunRecord:
        """The completed form of this record."""
        return _replaced(
            self,
            duration_s=duration_s,
            artifacts=artifacts,
            status=RunStatus.COMPLETED,
            quarantined_cases=0,
        )

    def quarantined(self, *, duration_s: float, failure: str) -> RunRecord:
        """The failed form — doc 10 §6's "never silently dropped"."""
        return _replaced(
            self,
            duration_s=duration_s,
            status=RunStatus.FAILED,
            quarantined_cases=1,
            failure=failure,
        )

    # ── serialisation ───────────────────────────────────────────────────────────

    def to_mapping(self) -> dict[str, Any]:
        """The doc 13 §2 structure, in the plain types YAML carries."""
        return {
            "id": self.id,
            "manifest_sha256": self.manifest_sha256,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "environment_lock_sha256": self.environment_lock_sha256,
            "environment_lock_source": self.environment_lock_source.value,
            "container_digest": self.container_digest,
            "seeds": dict(self.seeds),
            "hardware": dict(self.hardware),
            "solver_versions": dict(self.solver_versions),
            "data_versions": dict(self.data_versions),
            "parameter_sources": dict(self.parameter_sources),
            "artifacts": dict(self.artifacts),
            "tier": self.tier.value,
            "started_utc": self.started_utc.isoformat(),
            "duration_s": self.duration_s,
            "status": self.status.value,
            "quarantined_cases": self.quarantined_cases,
            "failure": self.failure,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """Rebuild a record read back from ``run.yaml``.

        Every enum and timestamp is re-validated rather than trusted: the file is
        human-readable by design (doc 08 §7) and therefore human-editable, and a record
        that quietly accepted ``status: finshed`` would make the run index disagree with
        the archive it indexes.
        """
        missing = sorted({"id", "manifest_sha256", "status", "tier"} - set(data))
        if missing:
            raise ValueError(f"run record is missing {', '.join(missing)}")

        return cls(
            id=str(data["id"]),
            manifest_sha256=str(data["manifest_sha256"]),
            git_commit=str(data["git_commit"]),
            git_dirty=bool(data["git_dirty"]),
            environment_lock_sha256=str(data["environment_lock_sha256"]),
            environment_lock_source=EnvironmentLockSource(str(data["environment_lock_source"])),
            container_digest=(
                None if data.get("container_digest") is None else str(data["container_digest"])
            ),
            seeds={key: int(value) for key, value in _mapping(data["seeds"], name="seeds").items()},
            hardware={
                key: str(value)
                for key, value in _mapping(data["hardware"], name="hardware").items()
            },
            solver_versions={
                key: str(value)
                for key, value in _mapping(data["solver_versions"], name="solver_versions").items()
            },
            data_versions={
                key: str(value)
                for key, value in _mapping(data["data_versions"], name="data_versions").items()
            },
            parameter_sources={
                key: str(value)
                for key, value in _mapping(
                    data["parameter_sources"], name="parameter_sources"
                ).items()
            },
            artifacts={
                key: str(value)
                for key, value in _mapping(data["artifacts"], name="artifacts").items()
            },
            tier=Tier(str(data["tier"])),
            started_utc=datetime.fromisoformat(str(data["started_utc"])),
            duration_s=float(data["duration_s"]),
            status=RunStatus(str(data["status"])),
            quarantined_cases=int(data["quarantined_cases"]),
            failure=None if data.get("failure") is None else str(data["failure"]),
        )

    def __repr__(self) -> str:
        return f"RunRecord({self.id!r}, {self.status.value}, {self.tier.value})"


def _replaced(record: RunRecord, **changes: Any) -> RunRecord:
    """``dataclasses.replace`` with the mapping fields carried over intact.

    Written out rather than using ``dataclasses.replace`` directly because ``replace``
    on a slotted frozen dataclass re-runs ``__post_init__`` over already-frozen mapping
    proxies; naming every field here also makes a field added to the record and forgotten
    in the transition a visible omission rather than a silent one.
    """
    return RunRecord(
        id=record.id,
        manifest_sha256=record.manifest_sha256,
        git_commit=record.git_commit,
        git_dirty=record.git_dirty,
        environment_lock_sha256=record.environment_lock_sha256,
        environment_lock_source=record.environment_lock_source,
        container_digest=record.container_digest,
        seeds=record.seeds,
        hardware=record.hardware,
        solver_versions=record.solver_versions,
        data_versions=record.data_versions,
        parameter_sources=record.parameter_sources,
        artifacts=changes.get("artifacts", record.artifacts),
        tier=record.tier,
        started_utc=record.started_utc,
        duration_s=changes.get("duration_s", record.duration_s),
        status=changes.get("status", record.status),
        quarantined_cases=changes.get("quarantined_cases", record.quarantined_cases),
        failure=changes.get("failure", record.failure),
    )
