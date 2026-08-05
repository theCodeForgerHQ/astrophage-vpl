"""The parameter registry — doc 02 §12, doc 08 §5, doc 09 §1.

    The registry is the single source of truth. No number appears in code as a literal.
    The count of ASSUMED-class entries is a tracked project metric (doc 00 C1) and is
    reported in CI.

Everything here exists to make that enforceable rather than aspirational.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vpl.core.params import (
    Parameter,
    ParameterRegistry,
    ProvenanceClass,
    Uncertainty,
    default_registry,
)
from vpl.core.units import DimensionalityError, magnitude_in

#: The doc 02 §12 example entry, verbatim, plus ``category``.
#:
#: ``category`` is an extension of the published schema and it is not gratuitous: doc 09
#: §1 states its target as "zero ASSUMED entries in the physics-constants and atomic-data
#: categories", which cannot be expressed — let alone enforced in CI — unless entries say
#: which category they are in.
VALID_ENTRY = """
- id: TS-L1.energy
  description: Nd:YAG pulse energy at the measurement volume
  value: 0.5
  units: J
  class: SPEC
  source: "Continuum Powerlite DLS 8010 datasheet, rev 2019"
  uncertainty: {type: relative, value: 0.05}
  sweep_range: [0.1, 2.0]
  affects: [thomson.photon_budget, thomson.snr]
  category: instrument
"""


def _write(tmp_path: Path, text: str, name: str = "entries.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


class TestParameterSchema:
    def test_loads_every_field_of_the_doc_02_12_schema(self, tmp_path: Path) -> None:
        registry = ParameterRegistry.load([_write(tmp_path, VALID_ENTRY)])

        entry = registry["TS-L1.energy"]

        assert entry.description.startswith("Nd:YAG")
        assert entry.units == "J"
        assert entry.provenance_class is ProvenanceClass.SPEC
        assert "Continuum" in entry.source
        assert entry.uncertainty == Uncertainty(kind="relative", value=0.05)
        assert entry.sweep_range == (0.1, 2.0)
        assert entry.affects == ("thomson.photon_budget", "thomson.snr")
        assert entry.category == "instrument"

    def test_exposes_the_value_as_a_dimensional_quantity(self, tmp_path: Path) -> None:
        registry = ParameterRegistry.load([_write(tmp_path, VALID_ENTRY)])

        assert magnitude_in(registry["TS-L1.energy"].quantity, "mJ") == pytest.approx(500.0)

    def test_value_in_converts_and_strips(self, tmp_path: Path) -> None:
        # This is the accessor physics code actually calls, so it does the doc 08 §5
        # boundary assertion for the caller rather than trusting them to.
        registry = ParameterRegistry.load([_write(tmp_path, VALID_ENTRY)])

        assert registry.value_in("TS-L1.energy", "J") == pytest.approx(0.5)

    def test_value_in_refuses_the_wrong_dimensionality(self, tmp_path: Path) -> None:
        registry = ParameterRegistry.load([_write(tmp_path, VALID_ENTRY)])

        with pytest.raises(DimensionalityError):
            registry.value_in("TS-L1.energy", "m")

    def test_rejects_units_pint_cannot_parse(self, tmp_path: Path) -> None:
        text = VALID_ENTRY.replace("units: J", "units: wombats")

        with pytest.raises(DimensionalityError, match="wombats"):
            ParameterRegistry.load([_write(tmp_path, text)])

    def test_rejects_an_unknown_provenance_class(self, tmp_path: Path) -> None:
        text = VALID_ENTRY.replace("class: SPEC", "class: PROBABLY_FINE")

        with pytest.raises(ValueError, match="PROBABLY_FINE"):
            ParameterRegistry.load([_write(tmp_path, text)])

    def test_rejects_a_missing_required_key(self, tmp_path: Path) -> None:
        text = VALID_ENTRY.replace("  units: J\n", "")

        with pytest.raises(ValueError, match="units"):
            ParameterRegistry.load([_write(tmp_path, text)])

    def test_rejects_an_unrecognised_key(self, tmp_path: Path) -> None:
        # A typo'd key that is silently ignored is a parameter that silently does not
        # apply — which is the parameter-fog trap of doc 00 §6 wearing a YAML costume.
        text = VALID_ENTRY + "  uncertianty: {type: relative, value: 0.1}\n"

        with pytest.raises(ValueError, match="uncertianty"):
            ParameterRegistry.load([_write(tmp_path, text)])


class TestProvenanceDiscipline:
    def test_a_sourced_class_must_actually_carry_a_source(self, tmp_path: Path) -> None:
        # doc 09 §1: MEASURED, PUBLISHED and SPEC each mean "traceable to something".
        # An entry claiming SPEC with no datasheet is an ASSUMED wearing a better label.
        text = VALID_ENTRY.replace(
            '  source: "Continuum Powerlite DLS 8010 datasheet, rev 2019"\n', ""
        )

        with pytest.raises(ValueError, match="source"):
            ParameterRegistry.load([_write(tmp_path, text)])

    def test_a_design_choice_must_be_swept(self, tmp_path: Path) -> None:
        # doc 09 §1, verbatim: "an unswept design choice is a hidden assumption wearing
        # a different label". DESIGN entries are unlimited but must all appear in the
        # sensitivity study, and the only way to guarantee that is to require the range.
        text = VALID_ENTRY.replace("class: SPEC", "class: DESIGN").replace(
            "  sweep_range: [0.1, 2.0]\n", ""
        )

        with pytest.raises(ValueError, match="sweep"):
            ParameterRegistry.load([_write(tmp_path, text)])

    def test_an_assumed_entry_is_permitted_and_counted(self, tmp_path: Path) -> None:
        # doc 00 C1 calls every ASSUMED value a defect, but refusing to load one would
        # push it back into an unlabelled literal — which is strictly worse. It loads,
        # it is labelled, and it is counted.
        text = VALID_ENTRY.replace("class: SPEC", "class: ASSUMED").replace(
            '  source: "Continuum Powerlite DLS 8010 datasheet, rev 2019"\n', ""
        )

        registry = ParameterRegistry.load([_write(tmp_path, text)])

        assert registry.assumed_count == 1
        assert registry["TS-L1.energy"].is_defect is True

    def test_counts_by_class_for_the_ci_metric(self, tmp_path: Path) -> None:
        registry = ParameterRegistry.load([_write(tmp_path, VALID_ENTRY)])

        assert registry.count_by_class()[ProvenanceClass.SPEC] == 1
        assert registry.count_by_class()[ProvenanceClass.ASSUMED] == 0

    def test_a_sweep_range_must_bracket_the_nominal_value(self, tmp_path: Path) -> None:
        # A range that excludes its own nominal value means one of the two is wrong, and
        # the sensitivity study would silently explore somewhere the model never runs.
        text = VALID_ENTRY.replace("sweep_range: [0.1, 2.0]", "sweep_range: [0.6, 2.0]")

        with pytest.raises(ValueError, match="bracket"):
            ParameterRegistry.load([_write(tmp_path, text)])

    def test_a_sweep_range_must_be_ordered(self, tmp_path: Path) -> None:
        text = VALID_ENTRY.replace("sweep_range: [0.1, 2.0]", "sweep_range: [2.0, 0.1]")

        with pytest.raises(ValueError, match="ascending"):
            ParameterRegistry.load([_write(tmp_path, text)])

    def test_a_negative_uncertainty_is_rejected(self, tmp_path: Path) -> None:
        text = VALID_ENTRY.replace("value: 0.05}", "value: -0.05}")

        with pytest.raises(ValueError, match="uncertainty"):
            ParameterRegistry.load([_write(tmp_path, text)])

    def test_an_unknown_uncertainty_kind_is_rejected(self, tmp_path: Path) -> None:
        text = VALID_ENTRY.replace("type: relative", "type: vibes")

        with pytest.raises(ValueError, match="vibes"):
            ParameterRegistry.load([_write(tmp_path, text)])


class TestRegistryComposition:
    def test_loads_several_files_into_one_registry(self, tmp_path: Path) -> None:
        second = (
            VALID_ENTRY.replace("TS-L1.energy", "TS-L1.wavelength")
            .replace("units: J", "units: nm")
            .replace("value: 0.5", "value: 532.0")
            .replace("sweep_range: [0.1, 2.0]", "sweep_range: [500.0, 550.0]")
        )
        registry = ParameterRegistry.load(
            [_write(tmp_path, VALID_ENTRY, "a.yaml"), _write(tmp_path, second, "b.yaml")]
        )

        assert len(registry) == 2

    def test_a_duplicate_id_is_an_error_naming_both_files(self, tmp_path: Path) -> None:
        # Last-one-wins would mean the answer depends on file iteration order, which
        # doc 00 E3 forbids outright — and the losing entry would never announce itself.
        a = _write(tmp_path, VALID_ENTRY, "a.yaml")
        b = _write(tmp_path, VALID_ENTRY, "b.yaml")

        with pytest.raises(ValueError, match="duplicate parameter id") as excinfo:
            ParameterRegistry.load([a, b])

        assert "a.yaml" in str(excinfo.value)
        assert "b.yaml" in str(excinfo.value)

    def test_an_unknown_id_error_suggests_near_matches(self, tmp_path: Path) -> None:
        registry = ParameterRegistry.load([_write(tmp_path, VALID_ENTRY)])

        with pytest.raises(KeyError, match=r"TS-L1\.energy"):
            registry["TS-L1.enrgy"]

    def test_iterates_in_a_deterministic_order(self, tmp_path: Path) -> None:
        # doc 00 E3: anything that hashes or writes the registry must see one order.
        second = VALID_ENTRY.replace("TS-L1.energy", "AAA.first")
        registry = ParameterRegistry.load(
            [_write(tmp_path, VALID_ENTRY, "a.yaml"), _write(tmp_path, second, "b.yaml")]
        )

        assert registry.ids() == ("AAA.first", "TS-L1.energy")

    def test_records_which_file_each_entry_came_from(self, tmp_path: Path) -> None:
        # Provenance of the provenance. When a value looks wrong the first question is
        # always "where is this defined", and grepping a shipped wheel is not an answer.
        path = _write(tmp_path, VALID_ENTRY)
        registry = ParameterRegistry.load([path])

        assert registry["TS-L1.energy"].defined_in == path

    def test_is_immutable(self, tmp_path: Path) -> None:
        registry = ParameterRegistry.load([_write(tmp_path, VALID_ENTRY)])

        with pytest.raises(TypeError):
            registry.entries["TS-L1.energy"] = registry["TS-L1.energy"]  # type: ignore[index]

    def test_rejects_a_file_that_is_not_a_list(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="list"):
            ParameterRegistry.load([_write(tmp_path, "id: lonely\n")])

    def test_an_empty_file_contributes_nothing_rather_than_failing(self, tmp_path: Path) -> None:
        registry = ParameterRegistry.load(
            [_write(tmp_path, VALID_ENTRY, "a.yaml"), _write(tmp_path, "", "b.yaml")]
        )

        assert len(registry) == 1


class TestShippedRegistry:
    def test_the_default_registry_loads(self) -> None:
        assert len(default_registry()) > 0

    def test_the_default_registry_is_memoised(self) -> None:
        assert default_registry() is default_registry()

    def test_every_shipped_entry_satisfies_the_provenance_rules(self) -> None:
        # Loading enforces the rules, so this passing means the shipped data is clean.
        # It is here so the failure says "the registry data is wrong" rather than
        # surfacing inside whichever unrelated test happened to import first.
        for entry in default_registry():
            assert isinstance(entry, Parameter)

    def test_the_reference_operating_point_is_registered(self) -> None:
        # doc 01 §2.1 defines RP-1 and every other document derives from it. It must be
        # in the registry, not retyped into whichever module needs it next.
        registry = default_registry()

        assert registry.value_in("RP1.pressure", "mTorr") == pytest.approx(5.0)
        assert registry.value_in("RP1.n_0", "m**-3") == pytest.approx(1e17)
        assert registry.value_in("RP1.T_e", "eV") == pytest.approx(3.0)
        assert registry.value_in("RP1.bias", "V") == pytest.approx(-250.0)

    def test_the_assumed_count_is_zero_in_physics_categories(self) -> None:
        # doc 09 §1: "Target: zero ASSUMED entries in the physics-constants and
        # atomic-data categories before Phase 2 exit". Asserting it from the start means
        # the count can only be driven up deliberately, never drift up by accident.
        offenders = [
            entry.id
            for entry in default_registry()
            if entry.is_defect and entry.category in {"physics", "atomic"}
        ]

        assert offenders == []
