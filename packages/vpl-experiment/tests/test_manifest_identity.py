"""Manifest identity — doc 00 E3, and the digest every artifact embeds (doc 08 §7)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from vpl.core.provenance import manifest_sha256
from vpl.experiment import load_manifest, manifest_from_document


class TestManifestIdentity:
    def test_two_manifests_differing_only_in_key_order_hash_the_same(
        self,
        tmp_path: Path,
        runnable_document: dict[str, Any],
        write_manifest: Callable[[Path, Mapping[str, Any]], Path],
    ) -> None:
        reversed_document = dict(reversed(list(runnable_document.items())))
        first = write_manifest(tmp_path / "a.yaml", runnable_document)
        second = write_manifest(tmp_path / "b.yaml", reversed_document)

        assert load_manifest(first).sha256 == load_manifest(second).sha256

    def test_two_manifests_differing_only_in_comments_hash_the_same(
        self, tmp_path: Path, runnable_path: Path
    ) -> None:
        original = runnable_path.read_text(encoding="utf-8")
        commented = "# an extra comment nobody executes\n" + original.replace(
            "gas: argon", "gas: argon  # the only gas RP-1 uses"
        )
        path = tmp_path / "commented.yaml"
        path.write_text(commented, encoding="utf-8")

        assert load_manifest(path).sha256 == load_manifest(runnable_path).sha256

    def test_changing_any_value_changes_the_digest(self, runnable_document: dict[str, Any]) -> None:
        before = manifest_from_document(runnable_document).sha256
        runnable_document["plasma"]["Te"]["value"] = 3.0000001
        assert manifest_from_document(runnable_document).sha256 != before

    def test_reordering_a_sequence_changes_the_digest_because_order_is_meaningful(
        self, documented_document: dict[str, Any]
    ) -> None:
        before = manifest_from_document(documented_document).sha256
        documented_document["instruments"].reverse()
        assert manifest_from_document(documented_document).sha256 != before

    def test_the_digest_is_the_one_manifest_sha256_computes_over_the_document(
        self, runnable_manifest: Any
    ) -> None:
        assert runnable_manifest.sha256 == manifest_sha256(runnable_manifest.as_document())

    def test_the_digest_is_sixty_four_lowercase_hexadecimal_characters(
        self, runnable_manifest: Any
    ) -> None:
        digest = runnable_manifest.sha256
        assert len(digest) == 64
        assert digest == digest.lower()
        assert set(digest) <= set("0123456789abcdef")


class TestTheArchivedForm:
    """`vpl reproduce` re-executes the archived manifest, so the round trip must hold."""

    def test_a_manifest_dumped_and_reloaded_keeps_its_digest(
        self, tmp_path: Path, documented_manifest: Any
    ) -> None:
        path = tmp_path / "archived.yaml"
        path.write_text(documented_manifest.to_yaml(), encoding="utf-8")
        assert load_manifest(path).sha256 == documented_manifest.sha256

    def test_a_reloaded_manifest_carries_the_same_blocks(
        self, tmp_path: Path, documented_manifest: Any
    ) -> None:
        path = tmp_path / "archived.yaml"
        path.write_text(documented_manifest.to_yaml(), encoding="utf-8")
        reloaded = load_manifest(path)

        assert reloaded.experiment == documented_manifest.experiment
        assert reloaded.inverse == documented_manifest.inverse
        assert reloaded.outputs == documented_manifest.outputs

    def test_the_document_a_manifest_exposes_cannot_be_edited_in_place(
        self, runnable_manifest: Any
    ) -> None:
        document = runnable_manifest.as_document()
        document["experiment"]["seed"] = 1
        assert runnable_manifest.experiment.seed != 1


class TestOverrides:
    """The seam the doc 10 §6 sweep layer will drive — see ADR-008."""

    def test_a_dotted_override_changes_the_manifest_and_therefore_its_identity(
        self, runnable_path: Path
    ) -> None:
        baseline = load_manifest(runnable_path)
        overridden = load_manifest(runnable_path, overrides=("experiment.seed=99",))

        assert overridden.experiment.seed == 99
        assert overridden.sha256 != baseline.sha256

    def test_an_override_naming_a_key_the_schema_does_not_know_is_refused(
        self, runnable_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="wibble"):
            load_manifest(runnable_path, overrides=("experiment.wibble=1",))

    def test_an_override_that_is_not_a_key_value_pair_is_refused(self, runnable_path: Path) -> None:
        with pytest.raises(ValueError, match="override"):
            load_manifest(runnable_path, overrides=("experiment.seed",))
