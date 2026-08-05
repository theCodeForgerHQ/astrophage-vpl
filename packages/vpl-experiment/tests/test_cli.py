"""The three commands doc 08 §6 writes out verbatim."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from vpl.experiment import RunStore, execute, manifest_from_document
from vpl.experiment.cli import main


class TestVplRun:
    def test_it_writes_a_run_and_prints_its_identity(
        self, tmp_path: Path, runnable_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = tmp_path / "runs"
        assert main(["run", str(runnable_path), "--store", str(store)]) == 0

        printed = capsys.readouterr().out
        run_id = RunStore(store).index()[0].id
        assert run_id in printed

    def test_it_applies_a_dotted_override_from_the_command_line(
        self, tmp_path: Path, runnable_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = tmp_path / "runs"
        code = main(
            ["run", str(runnable_path), "--store", str(store), "--set", "experiment.seed=7"]
        )
        assert code == 0
        capsys.readouterr()
        assert RunStore(store).resolve(RunStore(store).index()[0].id).read_provenance().seed == 7

    def test_it_reports_a_stage_that_does_not_exist_without_a_traceback(
        self, tmp_path: Path, documented_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["run", str(documented_path), "--store", str(tmp_path / "runs")])
        assert code == 1
        assert "instruments" in capsys.readouterr().err

    def test_it_reports_a_manifest_that_will_not_parse_without_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("experiment: {name: x}\n", encoding="utf-8")
        assert main(["run", str(path), "--store", str(tmp_path / "runs")]) == 1
        printed = capsys.readouterr().err
        assert "vpl run:" in printed
        assert "missing required key" in printed
        assert "Traceback" not in printed


class TestVplReproduce:
    def test_it_exits_zero_when_the_reproduction_is_bit_identical(
        self, tmp_path: Path, runnable_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = tmp_path / "runs"
        main(["run", str(runnable_path), "--store", str(store)])
        run_id = RunStore(store).index()[0].id
        capsys.readouterr()

        assert main(["reproduce", run_id, "--store", str(store)]) == 0
        assert "identical" in capsys.readouterr().out

    def test_it_exits_nonzero_when_the_reproduction_differs(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        runnable_document: dict[str, Any],
    ) -> None:
        from vpl.core.protocols.config import SolverConfig
        from vpl.core.registry import PluginGroup, clear_registrations, register
        from vpl.experiment.solvers import AnalyticSheathForwardSolver

        store = RunStore(tmp_path / "runs")
        runnable_document["forward"] = {"solver": "test.drifting"}
        register(PluginGroup.SOLVERS, "test.drifting", AnalyticSheathForwardSolver)
        try:
            run = execute(manifest_from_document(runnable_document), store=store)

            class _Drifted(AnalyticSheathForwardSolver):
                def configure(self, cfg: SolverConfig) -> None:
                    super().configure(SolverConfig(values={"model": "matrix"}))

            register(PluginGroup.SOLVERS, "test.drifting", _Drifted)
            capsys.readouterr()
            assert main(["reproduce", run.id, "--store", str(store.root)]) == 1
            assert "differ" in capsys.readouterr().out
        finally:
            clear_registrations(PluginGroup.SOLVERS)

    def test_an_unknown_run_identity_is_reported_without_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["reproduce", "nope", "--store", str(tmp_path / "runs")]) == 1
        assert "no runs" in capsys.readouterr().err


class TestVplCompare:
    def test_it_prints_the_differences_between_two_runs(
        self,
        tmp_path: Path,
        runnable_path: Path,
        capsys: pytest.CaptureFixture[str],
        runnable_document: dict[str, Any],
        write_manifest: Callable[[Path, Mapping[str, Any]], Path],
    ) -> None:
        store = tmp_path / "runs"
        main(["run", str(runnable_path), "--store", str(store)])
        runnable_document["plasma"]["Te"] = {"value": 4.0, "units": "eV"}
        other = write_manifest(tmp_path / "other.yaml", runnable_document)
        main(["run", str(other), "--store", str(store)])
        capsys.readouterr()

        ids = sorted(entry.id for entry in RunStore(store).index())
        assert main(["compare", ids[0], ids[1], "--store", str(store)]) == 0
        printed = capsys.readouterr().out
        assert "plasma.Te.value" in printed
        assert "gamma_E" in printed

    def test_it_emits_json_when_asked(
        self,
        tmp_path: Path,
        runnable_path: Path,
        capsys: pytest.CaptureFixture[str],
        runnable_document: dict[str, Any],
        write_manifest: Callable[[Path, Mapping[str, Any]], Path],
    ) -> None:
        store = tmp_path / "runs"
        main(["run", str(runnable_path), "--store", str(store)])
        runnable_document["experiment"]["description"] = "another"
        other = write_manifest(tmp_path / "other.yaml", runnable_document)
        main(["run", str(other), "--store", str(store)])
        capsys.readouterr()

        ids = sorted(entry.id for entry in RunStore(store).index())
        assert main(["compare", ids[0], ids[1], "--store", str(store), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["left"] == ids[0]
        assert payload["content_identical"] is True


class TestTheCommandLineItself:
    def test_it_requires_a_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 2
        assert "run" in capsys.readouterr().err

    def test_an_unknown_subcommand_fails(self) -> None:
        with pytest.raises(SystemExit):
            main(["fly"])

    def test_the_store_defaults_to_the_environment_when_it_is_set(
        self, tmp_path: Path, runnable_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VPL_RUN_STORE", str(tmp_path / "elsewhere"))
        assert main(["run", str(runnable_path)]) == 0
        assert (tmp_path / "elsewhere").is_dir()
