"""The refusals — every way a manifest, a record or an artifact is told it is wrong.

These are one module rather than scattered because they are one rule: nothing in this
package guesses. doc 08 §1 principle 4 makes configuration data, and data that is silently
coerced is data that silently means something else.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from vpl.core.provenance import Tier
from vpl.experiment import (
    ManifestConsistencyError,
    RunStore,
    execute,
    manifest_from_document,
)
from vpl.experiment.compare import MetricDifference, ValueDifference
from vpl.experiment.digest import artifact_digest
from vpl.experiment.manifest import parse
from vpl.experiment.run.record import RunRecord, RunStatus


class TestTheBlockReader:
    def test_a_block_that_is_not_a_mapping_names_what_it_found(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            parse.block([1, 2], where="plasma")

    def test_a_non_string_key_is_refused(self) -> None:
        with pytest.raises(TypeError, match="strings"):
            parse.block({1: "a"}, where="plasma")

    def test_a_required_key_that_is_absent_names_itself(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            parse.required({}, "seed", where="experiment")

    def test_a_string_field_given_a_number_is_refused(self) -> None:
        with pytest.raises(TypeError, match="string"):
            parse.text(3, where="experiment.name")

    def test_a_boolean_field_given_one_is_refused_rather_than_coerced(self) -> None:
        with pytest.raises(TypeError, match="true or false"):
            parse.flag(1, where="instruments[0].enabled")

    def test_an_integer_field_given_a_float_is_refused(self) -> None:
        with pytest.raises(TypeError, match="integer"):
            parse.integer(4000.5, where="inverse.draws")

    def test_a_number_field_given_a_string_is_refused(self) -> None:
        with pytest.raises(TypeError, match="number"):
            parse.number("3.0", where="plasma.kappa")

    def test_a_string_list_given_a_bare_string_is_refused(self) -> None:
        # A bare string is a sequence, and accepting one would silently read
        # `enabled_sources: N1` as thirteen single-character source names.
        with pytest.raises(TypeError, match="list"):
            parse.strings("N1", where="noise.enabled_sources")

    def test_a_value_yaml_could_not_have_produced_is_refused(self) -> None:
        with pytest.raises(TypeError, match="not a value a manifest can hold"):
            parse.frozen(object(), where="plasma")  # type: ignore[arg-type]

    def test_a_frozen_document_cannot_be_edited(self) -> None:
        frozen = parse.frozen({"experiment": {"seed": 1}}, where="")
        with pytest.raises(TypeError):
            frozen["experiment"] = {}  # type: ignore[index]


class TestThePlasmaBlockDefaultsThatAreStated:
    def test_a_stated_secondary_emission_yield_overrides_the_registry(
        self, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.experiment import resolve_plasma

        runnable_document["plasma"]["gamma_se"] = 0.2
        resolved = resolve_plasma(manifest_from_document(runnable_document).plasma)

        assert resolved.params.gamma_se == pytest.approx(0.2)
        assert resolved.sources["gamma_se"] == "manifest"

    def test_a_stated_eedf_shape_parameter_overrides_the_convention(
        self, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.experiment import resolve_plasma

        runnable_document["plasma"]["kappa"] = 2.5
        resolved = resolve_plasma(manifest_from_document(runnable_document).plasma)

        assert resolved.params.kappa == pytest.approx(2.5)
        assert resolved.sources["kappa"] == "manifest"


class TestTheRunRecord:
    def test_a_naive_timestamp_is_refused_because_it_cannot_be_ordered(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        record = execute(runnable_manifest, store=store).read_record()
        with pytest.raises(ValueError, match="timezone-aware"):
            RunRecord(
                **{
                    **{field: getattr(record, field) for field in RunRecord.__dataclass_fields__},
                    "started_utc": datetime(2026, 8, 5),
                }
            )

    def test_a_record_missing_a_field_names_it(self) -> None:
        with pytest.raises(ValueError, match="manifest_sha256"):
            RunRecord.from_mapping({"id": "x", "status": "completed", "tier": "T0"})

    def test_a_record_whose_seeds_are_not_a_mapping_is_refused(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        record = execute(runnable_manifest, store=store).read_record()
        broken = record.to_mapping()
        broken["seeds"] = ["not", "a", "mapping"]
        with pytest.raises(TypeError, match="seeds"):
            RunRecord.from_mapping(broken)

    def test_it_round_trips_through_its_mapping_form(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        record = execute(runnable_manifest, store=store).read_record()
        assert RunRecord.from_mapping(record.to_mapping()) == record

    def test_it_describes_itself_by_identity_status_and_tier(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        record = execute(runnable_manifest, store=store).read_record()
        assert record.status.value in repr(record)
        assert record.tier.value in repr(record)


class TestTheRunStore:
    def test_it_describes_itself_by_its_root(self, store: RunStore) -> None:
        assert str(store.root) in repr(store)

    def test_a_run_directory_describes_itself_by_its_identity(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        assert run.id in repr(run)

    def test_a_truncated_index_is_rebuilt_rather_than_repaired(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        store.index_path.write_text("{not json", encoding="utf-8")
        assert [entry.id for entry in store.index()] == [run.id]

    def test_a_record_that_is_not_a_mapping_is_refused(
        self, store: RunStore, runnable_manifest: Any
    ) -> None:
        run = execute(runnable_manifest, store=store)
        run.record_path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            run.read_record()


class TestTheEngine:
    def test_a_plugin_that_cannot_be_constructed_is_refused(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.core.registry import PluginGroup, clear_registrations, register

        register(PluginGroup.SOLVERS, "test.not-a-class", "just a string")
        try:
            runnable_document["forward"] = {"solver": "test.not-a-class"}
            with pytest.raises(TypeError, match="not constructible"):
                execute(manifest_from_document(runnable_document), store=store)
        finally:
            clear_registrations(PluginGroup.SOLVERS)

    def test_a_manifest_asking_for_no_artifacts_still_runs_and_records_none(
        self, store: RunStore, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["outputs"]["artifacts"] = []
        run = execute(manifest_from_document(runnable_document), store=store)

        assert run.read_record().status is RunStatus.COMPLETED
        assert run.read_record().artifacts == {}


class TestTheComparisonTypes:
    def test_a_metric_that_moved_from_zero_reports_no_relative_change(self) -> None:
        difference = MetricDifference(name="x", left=0.0, right=1.0, units="dimensionless")
        assert difference.absolute == pytest.approx(1.0)
        assert difference.relative is None

    def test_a_metric_present_on_one_side_has_neither_difference(self) -> None:
        difference = MetricDifference(name="x", left=None, right=1.0, units="dimensionless")
        assert difference.absolute is None
        assert difference.relative is None
        assert "<absent>" in difference.render()

    def test_an_absent_manifest_key_renders_as_absent(self) -> None:
        difference = ValueDifference(path="plasma.Ti", left=None, right=1.0, left_present=False)
        assert "<absent>" in difference.render()


class TestTheDigestCanonicalisation:
    def test_a_byte_string_attribute_and_its_text_form_digest_alike(self, tmp_path: Path) -> None:
        import h5py

        left = tmp_path / "left.h5"
        right = tmp_path / "right.h5"
        with h5py.File(left, "w") as handle:
            handle.attrs["tier"] = Tier.T0.value
            handle.create_dataset("x", data=[1.0])
        with h5py.File(right, "w") as handle:
            handle.attrs["tier"] = Tier.T0.value.encode()
            handle.create_dataset("x", data=[1.0])

        assert artifact_digest(left) == artifact_digest(right)

    def test_an_array_valued_attribute_is_covered(self, tmp_path: Path) -> None:
        import h5py

        path = tmp_path / "a.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["shape"] = [1, 2, 3]
            handle.create_dataset("x", data=[1.0])
        before = artifact_digest(path)

        with h5py.File(path, "r+") as handle:
            handle.attrs["shape"] = [1, 2, 4]
        assert artifact_digest(path) != before

    def test_an_excluded_parquet_metadata_key_is_left_out(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _write(path: Path, when: str) -> None:
            schema = pa.schema(
                [pa.field("metric", pa.string())],
                metadata={b"created_utc": when.encode(), b"kind": b"metrics"},
            )
            pq.write_table(pa.table({"metric": ["gamma_E"]}, schema), path)

        _write(tmp_path / "a.parquet", "2026-08-05T00:00:00+00:00")
        _write(tmp_path / "b.parquet", "2027-01-01T00:00:00+00:00")

        assert artifact_digest(tmp_path / "a.parquet") == artifact_digest(tmp_path / "b.parquet")

    def test_a_digest_of_an_absent_directory_is_the_empty_one(self, tmp_path: Path) -> None:
        from vpl.experiment.digest import run_content_digest

        assert run_content_digest(tmp_path / "nowhere") == run_content_digest(tmp_path / "empty")


class TestTheRemainingRefusals:
    def test_an_empty_experiment_name_is_refused(self, runnable_document: dict[str, Any]) -> None:
        runnable_document["experiment"]["name"] = "   "
        with pytest.raises(ValueError, match="run directory"):
            manifest_from_document(runnable_document)

    def test_an_empty_solver_name_is_refused(self, runnable_document: dict[str, Any]) -> None:
        runnable_document["forward"]["solver"] = ""
        with pytest.raises(ValueError, match="plugin to resolve"):
            manifest_from_document(runnable_document)

    def test_a_forward_block_with_no_solver_is_refused(
        self, runnable_document: dict[str, Any]
    ) -> None:
        del runnable_document["forward"]["solver"]
        with pytest.raises(ValueError, match="solver"):
            manifest_from_document(runnable_document)

    def test_a_draw_count_below_one_is_refused(self, documented_document: dict[str, Any]) -> None:
        documented_document["inverse"]["draws"] = 0
        with pytest.raises(ValueError, match="draws"):
            manifest_from_document(documented_document)

    def test_a_repeat_count_below_one_is_refused(self, documented_document: dict[str, Any]) -> None:
        documented_document["validation"]["n_repeats"] = 0
        with pytest.raises(ValueError, match="n_repeats"):
            manifest_from_document(documented_document)

    def test_a_t0_manifest_with_a_matching_model_but_a_different_mesh_is_refused(
        self, documented_document: dict[str, Any]
    ) -> None:
        documented_document["experiment"]["tier"] = "T0"
        documented_document["inverse"]["model"] = documented_document["forward"]["solver"]
        with pytest.raises(ManifestConsistencyError, match="neither"):
            manifest_from_document(documented_document)

    def test_a_t2_manifest_whose_forward_block_states_no_mesh_at_all_is_accepted(
        self, documented_document: dict[str, Any]
    ) -> None:
        # Two absent meshes are not "the same mesh": neither run has stated one, so there
        # is nothing yet for doc 05 §7.1 to require a mismatch between.
        del documented_document["forward"]["mesh"]
        del documented_document["inverse"]["mesh"]
        assert manifest_from_document(documented_document).experiment.tier is Tier.T2

    def test_a_manifest_describes_itself_by_name_tier_solver_and_digest(
        self, runnable_manifest: Any
    ) -> None:
        described = repr(runnable_manifest)
        assert runnable_manifest.experiment.name in described
        assert runnable_manifest.sha256[:8] in described

    def test_a_run_with_no_metric_artifact_contributes_no_metric_differences(
        self, store: RunStore, runnable_manifest: Any, runnable_document: dict[str, Any]
    ) -> None:
        from vpl.experiment import compare_runs

        first = execute(runnable_manifest, store=store)
        runnable_document["outputs"]["artifacts"] = ["plasma_state"]
        second = execute(manifest_from_document(runnable_document), store=store)

        comparison = compare_runs(store, first.id, second.id)
        by_name = {d.name: d for d in comparison.metric_differences}
        assert by_name["gamma_E"].right is None

    def test_a_metric_difference_renders_as_a_mapping(self) -> None:
        difference = MetricDifference(name="x", left=1.0, right=2.0, units="dimensionless")
        assert difference.to_mapping()["absolute"] == pytest.approx(1.0)

    def test_a_numeric_attribute_is_canonicalised_the_same_way_from_either_backend(
        self, tmp_path: Path
    ) -> None:
        import h5py
        import numpy as np

        left = tmp_path / "left.h5"
        right = tmp_path / "right.h5"
        with h5py.File(left, "w") as handle:
            handle.attrs["seed"] = np.int64(20260805)
            handle.create_dataset("x", data=[1.0])
        with h5py.File(right, "w") as handle:
            handle.attrs["seed"] = 20260805
            handle.create_dataset("x", data=[1.0])

        assert artifact_digest(left) == artifact_digest(right)

    def test_a_binary_parquet_column_is_read_as_text_rather_than_as_a_repr(
        self, tmp_path: Path
    ) -> None:
        # pyarrow hands back `bytes` for a binary column while h5py decodes its own byte
        # attributes to `str`. Without normalisation the same value stored in the two
        # backends would digest apart, which is what doc 08 §7's one-format-per-artifact
        # table would otherwise cost the comparison.
        import pyarrow as pa
        import pyarrow.parquet as pq

        left = tmp_path / "left.parquet"
        right = tmp_path / "right.parquet"
        pq.write_table(pa.table({"payload": [b"gamma_E"]}), left)
        pq.write_table(pa.table({"payload": [b"gamma_i"]}), right)

        assert artifact_digest(left) != artifact_digest(right)
