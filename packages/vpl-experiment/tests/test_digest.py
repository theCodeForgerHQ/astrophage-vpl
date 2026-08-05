"""What "bit-identical" means when every artifact also records when it was written."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from vpl.experiment.digest import EXECUTION_ONLY_FIELDS, UnknownArtifactFormatError, artifact_digest


class TestWhatTheDigestCovers:
    def test_it_ignores_only_the_field_that_records_when_the_run_happened(self) -> None:
        # doc 08 §7 puts `created_utc` in every artifact, and doc 13 §2 lists it under the
        # *execution*, not the result. Excluding anything else would let a run change its
        # commit, seed or tier and still claim to have reproduced.
        assert set(EXECUTION_ONLY_FIELDS) == {"created_utc"}

    def test_changing_one_stored_value_changes_the_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "a.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["created_utc"] = "2026-08-05T00:00:00+00:00"
            handle.create_dataset("x", data=[1.0, 2.0, 3.0])
        before = artifact_digest(path)

        with h5py.File(path, "r+") as handle:
            handle["x"][0] = 1.5
        assert artifact_digest(path) != before

    def test_changing_only_the_creation_timestamp_leaves_the_digest_alone(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "a.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["created_utc"] = "2026-08-05T00:00:00+00:00"
            handle.create_dataset("x", data=[1.0, 2.0])
        before = artifact_digest(path)

        with h5py.File(path, "r+") as handle:
            handle.attrs["created_utc"] = "2027-01-01T00:00:00+00:00"
        assert artifact_digest(path) == before

    def test_changing_the_commit_the_artifact_records_does_change_the_digest(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "a.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["git_commit"] = "aaaa"
            handle.create_dataset("x", data=[1.0])
        before = artifact_digest(path)

        with h5py.File(path, "r+") as handle:
            handle.attrs["git_commit"] = "bbbb"
        assert artifact_digest(path) != before

    def test_a_nested_group_is_covered(self, tmp_path: Path) -> None:
        path = tmp_path / "a.h5"
        with h5py.File(path, "w") as handle:
            handle.create_group("fields").create_dataset("n_e", data=[1.0, 2.0])
        before = artifact_digest(path)

        with h5py.File(path, "r+") as handle:
            handle["fields"]["n_e"][1] = 3.0
        assert artifact_digest(path) != before

    def test_renaming_a_dataset_changes_the_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "a.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("n_e", data=[1.0])
        before = artifact_digest(path)

        other = tmp_path / "b.h5"
        with h5py.File(other, "w") as handle:
            handle.create_dataset("n_i", data=[1.0])
        assert artifact_digest(other) != before


class TestFormats:
    def test_it_covers_parquet_as_well_as_hdf5(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        first = tmp_path / "a.parquet"
        second = tmp_path / "b.parquet"
        pq.write_table(pa.table({"metric": ["gamma_E"], "value": [1.0]}), first)
        pq.write_table(pa.table({"metric": ["gamma_E"], "value": [2.0]}), second)

        assert artifact_digest(first) != artifact_digest(second)

    def test_a_parquet_file_that_differs_only_in_its_timestamps_digests_the_same(
        self, tmp_path: Path
    ) -> None:
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        def _write(path: Path, when: str) -> None:
            schema = pa.schema(
                [pa.field("metric", pa.string()), pa.field("created_utc", pa.string())],
                metadata={b"vpl_provenance": json.dumps({"created_utc": when}).encode()},
            )
            pq.write_table(pa.table({"metric": ["gamma_E"], "created_utc": [when]}, schema), path)

        _write(tmp_path / "a.parquet", "2026-08-05T00:00:00+00:00")
        _write(tmp_path / "b.parquet", "2027-01-01T00:00:00+00:00")

        assert artifact_digest(tmp_path / "a.parquet") == artifact_digest(tmp_path / "b.parquet")

    def test_an_unrecognised_format_is_refused_rather_than_hashed_as_bytes(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(UnknownArtifactFormatError, match=r"notes\.txt"):
            artifact_digest(path)
