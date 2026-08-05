"""The run directory layout and the run index — doc 13 §2, doc 13 §5."""

from __future__ import annotations

from typing import Any

import pytest

from vpl.experiment import RunNotFoundError, RunStatus, RunStore, execute


class TestTheRunIdentity:
    def test_it_carries_the_date_the_experiment_name_and_the_manifest_digest(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        date = run.id.split("-", 1)[0]

        assert len(date) == len("20260805")
        assert date.isdigit()
        assert "l0-child-langmuir-rp1" in run.id
        assert run.id.endswith(runnable_manifest.sha256[:8])

    def test_a_name_with_characters_a_path_cannot_hold_is_reduced_to_ones_that_can(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.experiment import manifest_from_document

        runnable_document["experiment"]["name"] = "B02 / reference (RP-1)"
        run = execute(manifest_from_document(runnable_document), store=store)
        assert "b02-reference-rp-1" in run.id

    def test_two_runs_of_the_same_manifest_share_an_identity(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        first = execute(runnable_manifest, store=store)
        second = execute(runnable_manifest, store=store, force=True)
        assert first.id == second.id


class TestTheRunDirectory:
    def test_it_holds_the_manifest_the_provenance_sidecar_and_the_artifacts(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)

        assert run.manifest_path.is_file()
        assert run.provenance_path.is_file()
        assert run.record_path.is_file()
        assert (run.artifacts_path / "plasma_state.h5").is_file()
        assert (run.artifacts_path / "metrics.parquet").is_file()

    def test_the_archived_manifest_reloads_to_the_digest_the_run_recorded(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        assert run.read_manifest().sha256 == runnable_manifest.sha256

    def test_the_provenance_sidecar_rebuilds_into_the_record_it_was_written_from(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        provenance = run.read_provenance()

        assert provenance.manifest_sha256 == runnable_manifest.sha256
        assert provenance.seed == runnable_manifest.experiment.seed
        assert provenance.tier is runnable_manifest.experiment.tier


class TestDiscovery:
    def test_a_run_is_found_by_its_full_identity(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        assert store.resolve(run.id).id == run.id

    def test_a_run_is_found_by_an_unambiguous_prefix(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        assert store.resolve(run.id[:12]).id == run.id

    def test_an_ambiguous_prefix_is_refused_rather_than_guessed(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.experiment import manifest_from_document

        first = execute(runnable_manifest, store=store)
        runnable_document["experiment"]["seed"] = 7
        second = execute(manifest_from_document(runnable_document), store=store)

        assert first.id != second.id
        with pytest.raises(RunNotFoundError, match="ambiguous"):
            store.resolve(first.id[:10])

    def test_an_unknown_identity_lists_what_the_store_does_hold(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        with pytest.raises(RunNotFoundError, match=run.id):
            store.resolve("nothing-like-this")

    def test_an_empty_store_says_so_rather_than_listing_nothing(self, store: RunStore) -> None:
        with pytest.raises(RunNotFoundError, match="no runs"):
            store.resolve("anything")

    def test_every_completed_run_appears_in_the_index(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        entries = store.index()

        assert [entry.id for entry in entries] == [run.id]
        assert entries[0].status is RunStatus.COMPLETED
        assert entries[0].manifest_sha256 == runnable_manifest.sha256

    def test_the_index_is_rebuilt_from_the_directories_when_it_is_lost(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        store.index_path.unlink()

        assert [entry.id for entry in store.index()] == [run.id]
        assert store.index_path.is_file()

    def test_a_stray_directory_that_is_not_a_run_is_ignored_by_the_index(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        execute(runnable_manifest, store=store)
        (store.root / "not-a-run").mkdir()
        store.index_path.unlink()

        assert len(store.index()) == 1
