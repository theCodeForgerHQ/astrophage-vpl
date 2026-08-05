"""Gate G-1.3 — "manifest reproduction bit-identical" (doc 11 §2, doc 00 E3).

This is the gate the whole package exists to pass. doc 13 §6 schedules the same check
nightly against a random archived manifest; these tests are that check, run per commit.
"""

from __future__ import annotations

from typing import Any

import pytest

from vpl.core.protocols.config import SolverConfig
from vpl.core.registry import PluginGroup, clear_registrations, register
from vpl.experiment import RunStore, execute, reproduce
from vpl.experiment.digest import artifact_digest, run_content_digest
from vpl.experiment.solvers import AnalyticSheathForwardSolver


class TestGateG13:
    def test_a_run_reproduced_from_its_archived_manifest_is_bit_identical(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        original = execute(runnable_manifest, store=store)
        result = reproduce(store, original.id)

        assert result.is_identical, result.summary()
        assert result.original_digest == result.reproduced_digest

    def test_the_two_runs_agree_on_the_digest_of_every_artifact_separately(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        original = execute(runnable_manifest, store=store)
        result = reproduce(store, original.id)

        assert set(result.artifact_digests) == {"plasma_state.h5", "metrics.parquet"}
        for name, (left, right) in result.artifact_digests.items():
            assert left == right, name

    def test_the_reproduction_re_executes_rather_than_copying(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        original = execute(runnable_manifest, store=store)
        result = reproduce(store, original.id)

        assert result.reproduced_path != original.path
        assert (result.reproduced_path / "artifacts" / "plasma_state.h5").is_file()

    def test_only_the_creation_timestamp_differs_between_the_two_artifact_files(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        original = execute(runnable_manifest, store=store)
        result = reproduce(store, original.id)

        left = original.artifacts_path / "plasma_state.h5"
        right = result.reproduced_path / "artifacts" / "plasma_state.h5"

        assert artifact_digest(left) == artifact_digest(right)
        assert left.read_bytes() != right.read_bytes()

    def test_the_reproduction_reports_whether_it_ran_at_the_same_commit(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        original = execute(runnable_manifest, store=store)
        result = reproduce(store, original.id)
        assert result.commit_matches is True


class TestWhenReproductionFails:
    def test_a_reproduction_that_differs_is_reported_as_differing(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.experiment import manifest_from_document

        runnable_document["forward"] = {"solver": "test.drifting"}
        manifest = manifest_from_document(runnable_document)

        register(PluginGroup.SOLVERS, "test.drifting", AnalyticSheathForwardSolver)
        try:
            original = execute(manifest, store=store)

            # The code changed underneath the archived manifest — which is exactly the
            # regression doc 13 §6's nightly re-run exists to catch.
            class _Drifted(AnalyticSheathForwardSolver):
                def configure(self, cfg: SolverConfig) -> None:
                    super().configure(SolverConfig(values={"model": "matrix"}))

            register(PluginGroup.SOLVERS, "test.drifting", _Drifted)
            result = reproduce(store, original.id)

            assert not result.is_identical
            assert "differ" in result.summary()
        finally:
            clear_registrations(PluginGroup.SOLVERS)

    def test_a_tampered_archived_manifest_is_detected_before_it_is_re_executed(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        original = execute(runnable_manifest, store=store)
        original.manifest_path.write_text(
            original.manifest_path.read_text(encoding="utf-8").replace("value: 3.0", "value: 4.0"),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="does not match"):
            reproduce(store, original.id)

    def test_reproducing_twice_replaces_the_previous_reproduction(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        original = execute(runnable_manifest, store=store)
        first = reproduce(store, original.id)
        second = reproduce(store, original.id)

        assert first.reproduced_path == second.reproduced_path
        assert second.is_identical


class TestTheContentDigest:
    def test_it_is_stable_across_two_reads_of_one_run(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        assert run_content_digest(run.artifacts_path) == run_content_digest(run.artifacts_path)

    def test_it_changes_when_any_stored_value_changes(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.experiment import manifest_from_document

        first = execute(runnable_manifest, store=store)
        runnable_document["plasma"]["Te"] = {"value": 4.0, "units": "eV"}
        second = execute(manifest_from_document(runnable_document), store=store)

        assert run_content_digest(first.artifacts_path) != run_content_digest(second.artifacts_path)
