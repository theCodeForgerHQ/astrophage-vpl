"""`vpl compare` — a structured diff of two runs (doc 08 §6)."""

from __future__ import annotations

from typing import Any

import pytest

from vpl.experiment import RunStore, compare_runs, execute, manifest_from_document


def _differences_by_path(comparison: Any) -> dict[str, Any]:
    return {difference.path: difference for difference in comparison.manifest_differences}


def _renamed(document: dict[str, Any], description: str) -> Any:
    """The same experiment described differently — a different manifest, same physics."""
    document["experiment"]["description"] = description
    return manifest_from_document(document)


class TestComparingTwoRunsOfTheSamePhysics:
    def test_two_manifests_differing_only_in_prose_produce_the_same_content(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        first = execute(runnable_manifest, store=store)
        second = execute(_renamed(runnable_document, "described differently"), store=store)
        comparison = compare_runs(store, first.id, second.id)

        assert [d.path for d in comparison.manifest_differences] == ["experiment.description"]
        assert comparison.metric_differences == ()
        assert comparison.content_identical is True


class TestComparingManifests:
    def test_a_changed_value_is_reported_with_both_sides(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        first = execute(runnable_manifest, store=store)
        runnable_document["plasma"]["Te"] = {"value": 4.0, "units": "eV"}
        second = execute(manifest_from_document(runnable_document), store=store)

        difference = _differences_by_path(compare_runs(store, first.id, second.id))
        assert difference["plasma.Te.value"].left == pytest.approx(3.0)
        assert difference["plasma.Te.value"].right == pytest.approx(4.0)

    def test_a_key_only_one_manifest_sets_is_reported_as_absent_on_the_other(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        first = execute(runnable_manifest, store=store)
        runnable_document["plasma"]["Ti"] = {"value": 0.2, "units": "eV"}
        second = execute(manifest_from_document(runnable_document), store=store)

        difference = _differences_by_path(compare_runs(store, first.id, second.id))
        assert difference["plasma.Ti.value"].left is None
        assert difference["plasma.Ti.value"].right == pytest.approx(0.2)

    def test_a_list_element_is_addressed_by_its_index(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        first = execute(runnable_manifest, store=store)
        runnable_document["outputs"]["artifacts"] = ["metrics", "plasma_state"]
        second = execute(manifest_from_document(runnable_document), store=store)

        difference = _differences_by_path(compare_runs(store, first.id, second.id))
        assert difference["outputs.artifacts[0]"].left == "plasma_state"
        assert difference["outputs.artifacts[0]"].right == "metrics"


class TestComparingMetrics:
    def test_a_changed_metric_reports_the_absolute_and_the_relative_difference(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        first = execute(runnable_manifest, store=store)
        runnable_document["plasma"]["bias"] = {"mode": "dc", "value": -500.0, "units": "V"}
        second = execute(manifest_from_document(runnable_document), store=store)

        comparison = compare_runs(store, first.id, second.id)
        by_name = {difference.name: difference for difference in comparison.metric_differences}

        gamma_e = by_name["gamma_E"]
        assert gamma_e.left is not None
        assert gamma_e.right is not None
        assert gamma_e.absolute == pytest.approx(gamma_e.right - gamma_e.left)
        assert gamma_e.relative == pytest.approx(1.0, rel=1e-6)
        assert comparison.content_identical is False

    def test_a_metric_present_in_one_run_only_is_reported_rather_than_dropped(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.core.storage import MetricRecord, write_metrics

        first = execute(runnable_manifest, store=store)
        second = execute(_renamed(runnable_document, "with a hand-edited metric"), store=store)
        write_metrics(
            second.artifacts_path / "metrics.parquet",
            [MetricRecord(name="extra", value=1.0, units="dimensionless")],
            provenance=second.read_provenance(),
        )

        comparison = compare_runs(store, first.id, second.id)
        by_name = {difference.name: difference for difference in comparison.metric_differences}
        assert by_name["extra"].left is None
        assert by_name["gamma_E"].right is None


class TestComparingProvenance:
    def test_a_changed_seed_is_reported_as_a_provenance_difference(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        first = execute(runnable_manifest, store=store)
        runnable_document["experiment"]["seed"] = 1
        second = execute(manifest_from_document(runnable_document), store=store)

        comparison = compare_runs(store, first.id, second.id)
        paths = {difference.path for difference in comparison.provenance_differences}
        assert "seed" in paths
        assert "manifest_sha256" in paths

    def test_the_creation_timestamp_is_not_reported_as_a_difference(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        first = execute(runnable_manifest, store=store)
        second = execute(_renamed(runnable_document, "later"), store=store)

        comparison = compare_runs(store, first.id, second.id)
        paths = {difference.path for difference in comparison.provenance_differences}
        assert "created_utc" not in paths


class TestRendering:
    def test_the_comparison_renders_as_text_and_as_a_mapping(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        first = execute(runnable_manifest, store=store)
        second = execute(_renamed(runnable_document, "the other one"), store=store)
        comparison = compare_runs(store, first.id, second.id)

        assert first.id in comparison.render()
        assert "experiment.description" in comparison.render()
        assert comparison.to_mapping()["left"] == first.id
