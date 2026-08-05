"""Metric artifacts — doc 08 §7 ("Parquet, queryable across runs") and doc 07 §7.

Doc 07 §7 is the consumer that fixes the schema: *"every metric is stored per commit; a
change beyond its historical noise band fails the build. Physics regressions are silent
otherwise — the code runs, the plots look plausible, and the answer is wrong."*

That sentence has two consequences these tests pin. The commit has to be a **column**, not
merely file metadata, or the regression store cannot ask "how has ``rel_error`` moved
across the last fifty commits" without opening fifty files and parsing fifty headers. And
a metric that is not a number has to be refused at the boundary, because ``nan`` compares
false against every noise band and would pass the gate silently — which is the exact
failure mode doc 07 §7 names.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from vpl.core.provenance import EnvironmentLockSource, Provenance, Tier
from vpl.core.storage import (
    MetricRecord,
    MissingProvenanceError,
    read_metrics,
    write_metrics,
)


def _record(**overrides: object) -> Provenance:
    fields: dict[str, object] = {
        "manifest_sha256": "4a7f2e91" + "0" * 56,
        "git_commit": "9c1d8b3" + "0" * 33,
        "git_dirty": False,
        "seed": 20260804,
        "environment_lock_hash": "e81c" + "f" * 60,
        "environment_lock_source": EnvironmentLockSource.UV_LOCK,
        "created_utc": datetime(2026, 8, 4, 12, 30, 15, 123456, tzinfo=UTC),
        "vpl_version": "0.1.0",
        "solver_versions": {"dolfinx": "0.8.0"},
        "tier": Tier.T2,
    }
    return Provenance(**(fields | overrides))  # type: ignore[arg-type]


@pytest.fixture
def provenance() -> Provenance:
    return _record()


@pytest.fixture
def metrics() -> tuple[MetricRecord, ...]:
    return (
        MetricRecord(name="rel_error", value=0.0731234567890123, units="dimensionless"),
        MetricRecord(name="coverage", value=0.9487654321098765, units="dimensionless"),
        MetricRecord(name="wasserstein_iedf", value=1.2345678901234567e-3, units="eV"),
    )


class TestMetricsRoundTrip:
    def test_written_metrics_read_back_equal(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...], provenance: Provenance
    ) -> None:
        write_metrics(tmp_path / "m.parquet", metrics, provenance=provenance)

        assert read_metrics(tmp_path / "m.parquet").metrics == metrics

    def test_values_survive_bit_for_bit(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...], provenance: Provenance
    ) -> None:
        # A regression gate compares this run's number against a historical band. A
        # storage layer that rounded would inject a spurious drift of its own.
        write_metrics(tmp_path / "m.parquet", metrics, provenance=provenance)
        recovered = read_metrics(tmp_path / "m.parquet").metrics

        for written, read_back in zip(metrics, recovered, strict=True):
            assert read_back.value.hex() == written.value.hex()

    def test_units_survive(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...], provenance: Provenance
    ) -> None:
        write_metrics(tmp_path / "m.parquet", metrics, provenance=provenance)

        recovered = read_metrics(tmp_path / "m.parquet").metrics

        assert [m.units for m in recovered] == ["dimensionless", "dimensionless", "eV"]

    def test_the_provenance_survives(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...], provenance: Provenance
    ) -> None:
        write_metrics(tmp_path / "m.parquet", metrics, provenance=provenance)

        assert read_metrics(tmp_path / "m.parquet").provenance == provenance

    def test_a_run_that_produced_no_metrics_still_carries_its_provenance(
        self, tmp_path: Path, provenance: Provenance
    ) -> None:
        # A zero-row table has no rows to carry the provenance columns, so the file-level
        # metadata is what makes doc 08 §7's "every artifact embeds ..." true of it too.
        write_metrics(tmp_path / "m.parquet", (), provenance=provenance)

        archived = read_metrics(tmp_path / "m.parquet")

        assert archived.metrics == ()
        assert archived.provenance == provenance


class TestTheRegressionStoreSchema:
    def test_every_row_names_the_commit_it_was_measured_at(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...], provenance: Provenance
    ) -> None:
        # doc 07 §7: "every metric is stored per commit". As a column, so that a query
        # across fifty runs does not have to open and parse fifty file headers.
        write_metrics(tmp_path / "m.parquet", metrics, provenance=provenance)

        table = pq.read_table(tmp_path / "m.parquet")

        assert table.column("git_commit").to_pylist() == [provenance.git_commit] * len(metrics)

    def test_every_row_carries_the_tier_and_the_manifest_digest(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...], provenance: Provenance
    ) -> None:
        # doc 05 §7.2 makes comparing a T1 number against a T2 band a project defect, and
        # a table that cannot distinguish them invites exactly that comparison.
        write_metrics(tmp_path / "m.parquet", metrics, provenance=provenance)

        table = pq.read_table(tmp_path / "m.parquet")

        assert set(table.column("tier").to_pylist()) == {"T2"}
        assert set(table.column("manifest_sha256").to_pylist()) == {provenance.manifest_sha256}

    def test_metrics_from_two_commits_concatenate_into_one_queryable_table(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...], provenance: Provenance
    ) -> None:
        # This is what "queryable across runs" means in practice, and it only works if
        # every run writes the identical schema.
        later = _record(git_commit="ff" + "0" * 38)
        write_metrics(tmp_path / "a.parquet", metrics, provenance=provenance)
        write_metrics(tmp_path / "b.parquet", metrics, provenance=later)

        history = pq.read_table(tmp_path)

        assert history.num_rows == 2 * len(metrics)
        assert set(history.column("git_commit").to_pylist()) == {
            provenance.git_commit,
            later.git_commit,
        }

    def test_a_dirty_tree_is_recorded_as_a_boolean_column(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...]
    ) -> None:
        # doc 13 §2 fails a release run on a dirty tree. Filtering the regression store to
        # clean runs has to be a predicate, not a string comparison.
        write_metrics(tmp_path / "m.parquet", metrics, provenance=_record(git_dirty=True))

        table = pq.read_table(tmp_path / "m.parquet")

        assert table.column("git_dirty").to_pylist() == [True] * len(metrics)


class TestMetricsRefuseWhatCannotBeGated:
    def test_a_non_finite_metric_is_refused_at_construction(self) -> None:
        # nan compares false against every noise band, so it passes the doc 07 §7 gate
        # without ever being inside it. That is the silent physics regression doc 07 §7
        # exists to catch, arriving through the store meant to catch it.
        with pytest.raises(ValueError, match="finite"):
            MetricRecord(name="rel_error", value=math.nan, units="dimensionless")

    def test_a_metric_with_no_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name"):
            MetricRecord(name="  ", value=1.0, units="dimensionless")

    def test_a_metric_whose_units_the_registry_cannot_parse_is_refused(self) -> None:
        with pytest.raises(ValueError, match="furlong"):
            MetricRecord(name="rel_error", value=1.0, units="furlongs_per_fortnight")

    def test_two_values_for_one_metric_name_are_refused(
        self, tmp_path: Path, provenance: Provenance
    ) -> None:
        # A regression check that silently picks one of two values for the same metric is
        # a build that passes or fails on row order.
        duplicated = (
            MetricRecord(name="rel_error", value=0.07, units="dimensionless"),
            MetricRecord(name="rel_error", value=0.09, units="dimensionless"),
        )

        with pytest.raises(ValueError, match="rel_error"):
            write_metrics(tmp_path / "m.parquet", duplicated, provenance=provenance)


class TestMetricsRefuseToLoseProvenance:
    def test_writing_without_provenance_raises_rather_than_warning(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...]
    ) -> None:
        with pytest.raises(MissingProvenanceError):
            write_metrics(
                tmp_path / "m.parquet",
                metrics,
                provenance=None,  # type: ignore[arg-type]
            )

    def test_reading_a_table_with_no_provenance_metadata_raises(
        self, tmp_path: Path, metrics: tuple[MetricRecord, ...], provenance: Provenance
    ) -> None:
        write_metrics(tmp_path / "m.parquet", metrics, provenance=provenance)
        table = pq.read_table(tmp_path / "m.parquet")
        pq.write_table(table.replace_schema_metadata({}), tmp_path / "stripped.parquet")

        with pytest.raises(MissingProvenanceError):
            read_metrics(tmp_path / "stripped.parquet")
