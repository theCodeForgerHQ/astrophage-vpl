"""`vpl reproduce` — gate G-1.3 (doc 11 §2, doc 00 E3, doc 13 §6).

    Every experiment is defined by a single declarative manifest and is bit-for-bit
    reproducible given the manifest, the code version and the seed. — doc 00 E3

This module is that sentence executed. It re-runs the **archived** manifest — not the one
on the command line, and not a cached result — and compares what comes out against what
was stored.

## Three deliberate choices

**The archived manifest is verified before it is trusted.** Its digest is recomputed and
checked against the one the run recorded. A hand-edited archived manifest would otherwise
reproduce a different experiment perfectly and report success, which is worse than any
failure this command can report.

**The reproduction runs into a scratch directory**, ``<store>/.reproductions/<run-id>``,
never over the original. doc 13 §5 keeps the archived artifacts forever, and a verification
that destroys the thing it is verifying against can only be run once.

**Failure is reported, not raised.** A reproduction that differs is a *result* — doc 13 §6
schedules exactly this comparison nightly, and the nightly job wants a diff rather than a
traceback. The command's exit code carries the verdict.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vpl.core.provenance import git_state
from vpl.experiment.digest import artifact_digests, run_content_digest
from vpl.experiment.run.engine import execute
from vpl.experiment.run.store import ARTIFACTS_DIRNAME, RunStore

__all__ = ["ReproductionResult", "reproduce"]

#: Characters of a digest shown in a summary line. Enough to distinguish two runs at a
#: glance, short enough to fit beside the artifact name.
_DIGEST_PREVIEW_CHARS: Final[int] = 12


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    """What ``vpl reproduce`` found — the evidence for gate G-1.3.

    Attributes:
        run_id: The run that was reproduced.
        reproduced_path: Where the re-execution wrote.
        original_digest: Content digest of the archived artifacts.
        reproduced_digest: Content digest of the re-executed artifacts.
        artifact_digests: Per artifact, ``(original, reproduced)``. What makes a failure
            actionable — "the run differs" is not, "``plasma_state.h5`` differs and
            ``metrics.parquet`` does not" is.
        commit_matches: Whether the reproduction ran at the commit the original recorded.
            Reported rather than enforced: doc 13 §6's weekly and release checks
            deliberately re-run at a *later* commit, and a mismatch there is the finding,
            not an error.
    """

    run_id: str
    reproduced_path: Path
    original_digest: str
    reproduced_digest: str
    artifact_digests: Mapping[str, tuple[str | None, str | None]]
    commit_matches: bool

    @property
    def is_identical(self) -> bool:
        """The gate. Every artifact, byte for byte, modulo the write timestamp."""
        return self.original_digest == self.reproduced_digest

    def differing_artifacts(self) -> tuple[str, ...]:
        return tuple(
            name for name, (left, right) in sorted(self.artifact_digests.items()) if left != right
        )

    def summary(self) -> str:
        """A verdict a nightly job can paste into an alert."""
        head = f"run {self.run_id}"
        if self.is_identical:
            body = f"reproduction is bit-identical ({self.original_digest[:_DIGEST_PREVIEW_CHARS]})"
        else:
            differing = ", ".join(self.differing_artifacts()) or "the artifact set itself"
            body = (
                f"reproduction differs in {differing} "
                f"({self.original_digest[:_DIGEST_PREVIEW_CHARS]} vs "
                f"{self.reproduced_digest[:_DIGEST_PREVIEW_CHARS]})"
            )
        commit = "" if self.commit_matches else " [at a different commit]"
        return f"{head}: {body}{commit}"


def _paired(
    original: Mapping[str, str], reproduced: Mapping[str, str]
) -> dict[str, tuple[str | None, str | None]]:
    """Line the two artifact sets up by name, keeping the ones that appear in only one."""
    return {
        name: (original.get(name), reproduced.get(name))
        for name in sorted(set(original) | set(reproduced))
    }


def reproduce(store: RunStore, run_id: str) -> ReproductionResult:
    """Re-execute an archived run and verify the result — gate G-1.3.

    Args:
        store: The store holding the run.
        run_id: The run's identity, or an unambiguous prefix of it.

    Returns:
        The comparison. Check :attr:`ReproductionResult.is_identical`.

    Raises:
        RunNotFoundError: If ``run_id`` names nothing, or names more than one run.
        ValueError: If the archived manifest's digest does not match the one the run
            recorded — the archive has been edited, and re-running it would verify a
            different experiment.
    """
    original = store.resolve(run_id)
    record = original.read_record()
    manifest = original.read_manifest()

    if manifest.sha256 != record.manifest_sha256:
        raise ValueError(
            f"the archived manifest of {record.id} does not match the digest the run "
            f"recorded: {manifest.sha256} against {record.manifest_sha256}. The archive "
            "has been edited since the run, so re-executing it would verify a different "
            "experiment against this run's artifacts (doc 00 E3)."
        )

    scratch = store.reproductions_path / record.id
    if scratch.exists():
        # Cleared rather than reused: a stale artifact from a previous reproduction would
        # be compared as though this run had produced it.
        shutil.rmtree(scratch)

    reproduced = execute(
        manifest,
        store=RunStore(store.reproductions_path),
        force=True,
        run_id=record.id,
    )

    commit, _ = git_state()

    return ReproductionResult(
        run_id=record.id,
        reproduced_path=reproduced.path,
        original_digest=run_content_digest(original.path / ARTIFACTS_DIRNAME),
        reproduced_digest=run_content_digest(reproduced.path / ARTIFACTS_DIRNAME),
        artifact_digests=_paired(
            artifact_digests(original.path / ARTIFACTS_DIRNAME),
            artifact_digests(reproduced.path / ARTIFACTS_DIRNAME),
        ),
        commit_matches=commit == record.git_commit,
    )
