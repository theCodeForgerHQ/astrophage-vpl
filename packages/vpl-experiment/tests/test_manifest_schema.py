"""The manifest schema — doc 08 §6, every block of it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from vpl.core.provenance import Tier
from vpl.experiment import (
    ArtifactRequest,
    BiasMode,
    CalibrationMode,
    ManifestConsistencyError,
    UnknownKeyError,
    load_manifest,
    manifest_from_document,
    resolve_plasma,
)


class TestTheExperimentBlock:
    def test_it_carries_the_name_description_tier_and_seed_doc_08_6_writes(
        self, documented_manifest: Any
    ) -> None:
        experiment = documented_manifest.experiment
        assert experiment.name == "b02-reference-operating-point"
        assert experiment.description == "Closed-loop validation at RP-1, honest tier"
        assert experiment.tier is Tier.T2
        assert experiment.seed == 20260804

    def test_it_rejects_a_tier_that_is_not_one_of_the_three_doc_05_7_2_defines(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["experiment"]["tier"] = "T3"
        with pytest.raises(ValueError, match="T0, T1, T2"):
            manifest_from_document(runnable_document)

    def test_it_rejects_a_negative_seed_because_a_seed_indexes_a_generator(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["experiment"]["seed"] = -1
        with pytest.raises(ValueError, match="seed"):
            manifest_from_document(runnable_document)

    def test_it_rejects_a_boolean_where_a_seed_belongs(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["experiment"]["seed"] = True
        with pytest.raises(TypeError, match="seed"):
            manifest_from_document(runnable_document)

    def test_a_missing_required_block_names_itself(self, runnable_document: dict[str, Any]) -> None:
        del runnable_document["plasma"]
        with pytest.raises(ValueError, match="plasma"):
            manifest_from_document(runnable_document)


class TestUnknownKeys:
    def test_a_misspelled_top_level_block_is_rejected_rather_than_ignored(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["output"] = runnable_document.pop("outputs")
        with pytest.raises(UnknownKeyError, match="output"):
            manifest_from_document(runnable_document)

    def test_a_misspelled_key_inside_a_block_suggests_the_one_that_was_meant(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["experiment"]["descrition"] = "typo"
        with pytest.raises(UnknownKeyError, match="description"):
            manifest_from_document(runnable_document)

    def test_extra_keys_in_the_forward_block_become_the_solver_configuration(
        self, documented_manifest: Any
    ) -> None:
        forward = documented_manifest.forward
        assert forward.solver == "vpl.physics.kinetic.pic1d3v"
        assert forward.config.require_int("n_ppc") == 1000
        assert forward.config.section("mesh").require_str("grading") == "wall_refined_A"
        assert "solver" not in forward.config

    def test_a_misspelled_instrument_key_is_rejected(
        self, documented_document: dict[str, Any]
    ) -> None:
        documented_document["instruments"][0]["enable"] = True
        with pytest.raises(UnknownKeyError, match="enable"):
            manifest_from_document(documented_document)

    def test_a_manifest_that_is_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_manifest(path)


class TestThePlasmaBlock:
    def test_a_quantity_is_written_as_a_value_and_its_units(self, documented_manifest: Any) -> None:
        plasma = documented_manifest.plasma
        assert plasma.gas == "argon"
        assert float(plasma.pressure.m_as("mTorr")) == pytest.approx(5.0)
        assert float(plasma.n_0.m_as("m**-3")) == pytest.approx(1.0e17)
        assert float(plasma.T_e.m_as("eV")) == pytest.approx(3.0)

    def test_a_bare_number_where_a_quantity_belongs_is_refused(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["plasma"]["pressure"] = 5.0
        with pytest.raises(TypeError, match="units"):
            manifest_from_document(runnable_document)

    def test_a_quantity_with_units_the_registry_cannot_parse_is_refused(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["plasma"]["pressure"] = {"value": 5.0, "units": "wibbleflop"}
        with pytest.raises(ValueError, match="wibbleflop"):
            manifest_from_document(runnable_document)

    def test_a_quantity_in_the_wrong_dimension_is_refused_when_it_is_used(
        self, runnable_document: dict[str, Any]
    ) -> None:
        # `metre` parses; a pressure in metres does not. The units layer catches it at the
        # boundary where the meaning is still known (doc 08 §5).
        runnable_document["plasma"]["pressure"] = {"value": 5.0, "units": "metre"}
        manifest = manifest_from_document(runnable_document)
        with pytest.raises(TypeError, match="Pa"):
            resolve_plasma(manifest.plasma)

    def test_the_dc_bias_of_doc_08_6_carries_no_frequency(self, documented_manifest: Any) -> None:
        bias = documented_manifest.plasma.bias
        assert bias.mode is BiasMode.DC
        assert float(bias.value.m_as("V")) == pytest.approx(-250.0)
        assert bias.frequency is None

    def test_an_rf_bias_may_state_its_own_frequency(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["plasma"]["bias"] = {
            "mode": "rf",
            "value": -250.0,
            "units": "V",
            "frequency": {"value": 60.0, "units": "MHz"},
        }
        manifest = manifest_from_document(runnable_document)
        bias = manifest.plasma.bias
        assert bias.mode is BiasMode.RF
        assert bias.frequency is not None
        assert float(bias.frequency.m_as("MHz")) == pytest.approx(60.0)

    def test_a_dc_bias_that_states_a_frequency_is_refused(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["plasma"]["bias"] = {
            "mode": "dc",
            "value": -250.0,
            "units": "V",
            "frequency": {"value": 60.0, "units": "MHz"},
        }
        with pytest.raises(ValueError, match="frequency"):
            manifest_from_document(runnable_document)


class TestTheNoiseBlock:
    def test_it_carries_the_enabled_sources_in_the_order_the_manifest_wrote_them(
        self, documented_manifest: Any
    ) -> None:
        noise = documented_manifest.noise
        assert noise is not None
        assert noise.enabled_sources[0] == "N1"
        assert noise.enabled_sources[-1] == "N18"
        assert noise.calibration is CalibrationMode.ESTIMATED

    def test_an_unquoted_yaml_true_calibration_is_refused_with_the_quoting_named(
        self, documented_document: dict[str, Any]
    ) -> None:
        # doc 08 §6 comments "NOT 'true' — doc 04 §7.3". YAML turns an unquoted `true`
        # into a boolean, so the mode name and the boolean are indistinguishable by the
        # time the loader sees them unless the failure says so.
        documented_document["noise"]["calibration"] = True
        with pytest.raises(ValueError, match="quote"):
            manifest_from_document(documented_document)

    def test_the_true_calibration_mode_is_expressible_when_it_is_quoted(
        self, documented_document: dict[str, Any]
    ) -> None:
        documented_document["noise"]["calibration"] = "true"
        manifest = manifest_from_document(documented_document)
        assert manifest.noise is not None
        assert manifest.noise.calibration is CalibrationMode.TRUE


class TestTheInverseAndValidationBlocks:
    def test_the_inverse_block_of_doc_08_6_parses_in_full(self, documented_manifest: Any) -> None:
        inverse = documented_manifest.inverse
        assert inverse is not None
        assert inverse.model == "vpl.physics.surrogate.gp"
        assert inverse.engine == "numpyro_nuts"
        assert inverse.draws == 4000
        assert inverse.chains == 4
        assert inverse.mesh["grading"] == "wall_refined_B"
        assert inverse.parameters["control"] == "all"

    def test_the_validation_block_of_doc_08_6_parses_in_full(
        self, documented_manifest: Any
    ) -> None:
        validation = documented_manifest.validation
        assert validation is not None
        assert validation.seal_truth is True
        assert validation.n_repeats == 200
        assert "wasserstein_iedf" in validation.metrics

    def test_a_chain_count_below_one_is_refused(self, documented_document: dict[str, Any]) -> None:
        documented_document["inverse"]["chains"] = 0
        with pytest.raises(ValueError, match="chains"):
            manifest_from_document(documented_document)


class TestTheOutputsBlock:
    def test_the_artifacts_of_doc_08_6_parse_into_known_requests(
        self, documented_manifest: Any
    ) -> None:
        outputs = documented_manifest.outputs
        assert ArtifactRequest.POSTERIOR in outputs.artifacts
        assert ArtifactRequest.ERROR_BUDGET in outputs.artifacts
        assert outputs.report is True
        assert "fim_spectrum" in outputs.figures

    def test_an_artifact_nothing_in_the_framework_produces_is_refused(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["outputs"]["artifacts"] = ["posteriorr"]
        with pytest.raises(ValueError, match="posteriorr"):
            manifest_from_document(runnable_document)


class TestTheMandatoryForwardInverseMismatch:
    """doc 05 §7.1 makes the mismatch structural, so the manifest must express it."""

    def test_a_t2_manifest_whose_inverse_model_equals_its_forward_solver_is_refused(
        self, documented_document: dict[str, Any]
    ) -> None:
        documented_document["inverse"]["model"] = documented_document["forward"]["solver"]
        with pytest.raises(ManifestConsistencyError, match="inverse crime"):
            manifest_from_document(documented_document)

    def test_a_t2_manifest_whose_two_meshes_match_is_refused(
        self, documented_document: dict[str, Any]
    ) -> None:
        documented_document["inverse"]["mesh"] = dict(documented_document["forward"]["mesh"])
        with pytest.raises(ManifestConsistencyError, match="mesh"):
            manifest_from_document(documented_document)

    def test_a_t1_manifest_must_invert_with_the_same_model_it_generated_with(
        self, documented_document: dict[str, Any]
    ) -> None:
        documented_document["experiment"]["tier"] = "T1"
        with pytest.raises(ManifestConsistencyError, match="T1"):
            manifest_from_document(documented_document)

    def test_a_t1_manifest_with_matching_model_and_mesh_is_accepted(
        self, documented_document: dict[str, Any]
    ) -> None:
        documented_document["experiment"]["tier"] = "T1"
        documented_document["inverse"]["model"] = documented_document["forward"]["solver"]
        documented_document["inverse"]["mesh"] = dict(documented_document["forward"]["mesh"])
        manifest = manifest_from_document(documented_document)
        assert manifest.experiment.tier is Tier.T1

    def test_a_manifest_with_no_inverse_block_is_not_checked_for_a_mismatch(
        self, runnable_manifest: Any
    ) -> None:
        assert runnable_manifest.inverse is None


class TestTheDocumentedManifestAsAConformanceTest:
    def test_every_block_of_the_doc_08_6_example_parses_and_validates(
        self, documented_path: Path
    ) -> None:
        manifest = load_manifest(documented_path)
        assert manifest.experiment.tier is Tier.T2
        assert len(manifest.instruments) == 4
        assert manifest.noise is not None
        assert manifest.inverse is not None
        assert manifest.validation is not None

    def test_the_instrument_list_keeps_the_order_the_manifest_wrote(
        self, documented_manifest: Any
    ) -> None:
        assert [entry.id for entry in documented_manifest.instruments] == [
            "oes",
            "lif",
            "thomson",
            "interf",
        ]

    def test_an_instrument_config_path_is_carried_as_written(
        self, documented_manifest: Any
    ) -> None:
        oes = documented_manifest.instruments[0]
        assert oes.enabled is True
        assert oes.config == Path("configs/instruments/oes_iccd.yaml")


class TestLoadingFromDisk:
    def test_a_manifest_loaded_from_a_path_equals_one_built_from_its_document(
        self,
        tmp_path: Path,
        runnable_document: Mapping[str, Any],
        write_manifest: Callable[[Path, Mapping[str, Any]], Path],
    ) -> None:
        path = write_manifest(tmp_path / "m.yaml", runnable_document)
        assert load_manifest(path).sha256 == manifest_from_document(runnable_document).sha256

    def test_a_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"absent\.yaml"):
            load_manifest(tmp_path / "absent.yaml")
