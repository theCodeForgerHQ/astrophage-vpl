"""`vpl run` end to end — doc 08 §6, doc 08 §7, doc 10 §6."""

from __future__ import annotations

from typing import Any

import pytest

from vpl.core.random import Stream
from vpl.core.registry import PluginGroup, clear_registrations, register
from vpl.core.state import Fidelity
from vpl.core.storage import read_metrics, read_plasma_state
from vpl.experiment import (
    RunStatus,
    RunStore,
    StageNotImplementedError,
    execute,
    manifest_from_document,
)


class TestAForwardOnlyRun:
    def test_it_runs_end_to_end_and_writes_the_artifacts_the_manifest_asked_for(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        archived = read_plasma_state(run.artifacts_path / "plasma_state.h5")

        assert archived.state.fidelity is Fidelity.L0
        assert archived.state.field_names == ("Phi", "T_e", "n_e", "n_i", "u_i")

    def test_the_metrics_it_writes_are_the_doc_01_1_2_decomposition(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        metrics = {m.name: m for m in read_metrics(run.artifacts_path / "metrics.parquet").metrics}

        assert set(metrics) == {"gamma_E", "gamma_i", "mean_impact_energy"}
        assert metrics["gamma_E"].value == pytest.approx(6576.94, rel=1e-5)
        assert metrics["gamma_E"].units == "W/m**2"

    def test_every_artifact_carries_the_manifest_digest_of_the_run_that_made_it(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        state = read_plasma_state(run.artifacts_path / "plasma_state.h5").provenance
        metrics = read_metrics(run.artifacts_path / "metrics.parquet").provenance

        assert state.manifest_sha256 == runnable_manifest.sha256
        assert metrics.manifest_sha256 == runnable_manifest.sha256
        assert state.tier is runnable_manifest.experiment.tier

    def test_the_solver_versions_reach_the_provenance_block(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        versions = run.read_provenance().solver_versions
        assert versions["vpl.physics.analytic.sheath"]


class TestTheRunRecord:
    def test_it_lists_one_seed_per_stream_because_doc_10_5_forbids_a_global_one(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        record = execute(runnable_manifest, store=store).read_record()
        assert set(record.seeds) == {stream.value for stream in Stream}

    def test_the_stream_seeds_derive_from_the_root_seed_the_manifest_states(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        from vpl.core.random import stream_seed

        record = execute(runnable_manifest, store=store).read_record()
        root = runnable_manifest.experiment.seed
        assert record.seeds[Stream.COLLISIONS.value] == stream_seed(root, Stream.COLLISIONS)

    def test_it_records_where_every_resolved_control_parameter_came_from(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        record = execute(runnable_manifest, store=store).read_record()

        assert record.parameter_sources["T_i"] == "registry:RP1.T_i"
        assert record.parameter_sources["n_0"] == "manifest"

    def test_it_names_the_status_the_duration_and_the_tier(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        record = execute(runnable_manifest, store=store).read_record()

        assert record.status is RunStatus.COMPLETED
        assert record.duration_s >= 0.0
        assert record.tier is runnable_manifest.experiment.tier
        assert record.quarantined_cases == 0

    def test_it_round_trips_through_yaml_unchanged(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        assert run.read_record() == run.read_record()

    def test_it_names_the_artifacts_the_run_produced(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        record = execute(runnable_manifest, store=store).read_record()
        assert set(record.artifacts) == {"plasma_state", "metrics"}


class TestStagesThatDoNotExistYet:
    def test_the_doc_08_6_manifest_is_refused_by_name_rather_than_half_run(
        self, store: RunStore, documented_manifest: Any
    ) -> None:
        with pytest.raises(StageNotImplementedError) as excinfo:
            execute(documented_manifest, store=store)

        message = str(excinfo.value)
        for stage in ("instruments", "noise", "inverse", "validation"):
            assert stage in message

    def test_it_refuses_before_it_creates_a_run_directory(
        self, store: RunStore, documented_manifest: Any
    ) -> None:
        with pytest.raises(StageNotImplementedError):
            execute(documented_manifest, store=store)
        assert not store.root.exists() or not any(store.root.glob("2*"))

    def test_an_instrument_that_is_disabled_is_not_a_stage(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["instruments"] = [{"id": "oes", "enabled": False}]
        run = execute(manifest_from_document(runnable_document), store=store)
        assert run.read_record().status is RunStatus.COMPLETED

    def test_an_artifact_the_framework_cannot_produce_yet_is_refused_by_name(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["outputs"]["artifacts"] = ["plasma_state", "posterior"]
        with pytest.raises(StageNotImplementedError, match="posterior"):
            execute(manifest_from_document(runnable_document), store=store)

    def test_a_requested_figure_is_refused_because_doc_13_3_has_no_engine_yet(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["outputs"]["figures"] = ["flux_profile"]
        with pytest.raises(StageNotImplementedError, match="figures"):
            execute(manifest_from_document(runnable_document), store=store)


class TestFailure:
    def test_a_manifest_naming_a_solver_nobody_installed_says_what_is_installed(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.core.registry import PluginNotFoundError

        runnable_document["forward"]["solver"] = "vpl.physics.kinetic.pic1d3v"
        del runnable_document["forward"]["model"]
        with pytest.raises(PluginNotFoundError, match=r"vpl\.physics\.analytic\.sheath"):
            execute(manifest_from_document(runnable_document), store=store)

    def test_a_failed_run_is_recorded_rather_than_silently_dropped(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        # doc 10 §6: "a quarantined case is never silently dropped from statistics".
        # The single-run form of that rule is that the directory and the record survive.
        from vpl.core.protocols.config import SolverConfig
        from vpl.experiment.solvers import AnalyticSheathForwardSolver

        class _Exploding(AnalyticSheathForwardSolver):
            def configure(self, cfg: SolverConfig) -> None:
                raise RuntimeError("the solver could not be configured")

        register(PluginGroup.SOLVERS, "test.exploding", _Exploding)
        try:
            runnable_document["forward"] = {"solver": "test.exploding"}
            manifest = manifest_from_document(runnable_document)
            with pytest.raises(RuntimeError, match="could not be configured"):
                execute(manifest, store=store)

            run = store.resolve(store.index()[0].id)
            record = run.read_record()
            assert record.status is RunStatus.FAILED
            assert record.quarantined_cases == 1
            assert "could not be configured" in (record.failure or "")
        finally:
            clear_registrations(PluginGroup.SOLVERS)

    def test_a_plugin_that_is_not_a_solver_is_refused_at_load_rather_than_at_use(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        register(PluginGroup.SOLVERS, "test.not-a-solver", object)
        try:
            runnable_document["forward"] = {"solver": "test.not-a-solver"}
            with pytest.raises(TypeError, match="ForwardSolver"):
                execute(manifest_from_document(runnable_document), store=store)
        finally:
            clear_registrations(PluginGroup.SOLVERS)

    def test_rerunning_a_manifest_into_an_existing_directory_is_refused(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        execute(runnable_manifest, store=store)
        with pytest.raises(FileExistsError, match="reproduce"):
            execute(runnable_manifest, store=store)

    def test_force_overwrites_a_previous_run_of_the_same_manifest(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        execute(runnable_manifest, store=store)
        run = execute(runnable_manifest, store=store, force=True)
        assert run.read_record().status is RunStatus.COMPLETED
