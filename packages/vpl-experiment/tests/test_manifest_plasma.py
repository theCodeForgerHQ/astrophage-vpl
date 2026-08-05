"""``manifest.plasma`` becomes ``PlasmaParams`` — doc 08 §6, doc 05 §2.1."""

from __future__ import annotations

from typing import Any

import pytest

from vpl.core.state import PlasmaParams
from vpl.experiment import manifest_from_document, resolve_plasma


class TestResolvingThePlasmaBlock:
    def test_the_doc_08_6_plasma_block_produces_the_rp1_control_parameters(
        self, documented_manifest: Any
    ) -> None:
        resolved = resolve_plasma(documented_manifest.plasma)
        params = resolved.params

        assert isinstance(params, PlasmaParams)
        assert params.species.name == "Ar+"
        assert params.n_0_per_m3 == pytest.approx(1.0e17)
        assert float(params.T_e_eV.magnitude) == pytest.approx(3.0)
        assert float(params.pressure.m_as("mTorr")) == pytest.approx(5.0)

    def test_the_signed_bias_convention_survives_the_manifest(
        self, documented_manifest: Any
    ) -> None:
        params = resolve_plasma(documented_manifest.plasma).params
        assert params.bias_volts == pytest.approx(-250.0)
        assert float(params.bias_magnitude.m_as("V")) == pytest.approx(250.0)

    def test_a_dc_manifest_produces_parameters_with_no_drive_frequency(
        self, documented_manifest: Any
    ) -> None:
        params = resolve_plasma(documented_manifest.plasma).params
        assert params.rf_frequency is None
        assert params.is_rf is False


class TestWhereUnstatedParametersComeFrom:
    """doc 08 §5: the registry is the sole source of numeric defaults, and doc 00 C4
    forbids the assumption being invisible — so every field names its own source."""

    def test_an_unstated_ion_temperature_comes_from_the_registry_and_says_so(
        self, documented_manifest: Any
    ) -> None:
        resolved = resolve_plasma(documented_manifest.plasma)
        assert resolved.sources["T_i"] == "registry:RP1.T_i"
        assert float(resolved.params.T_i_eV.magnitude) == pytest.approx(0.05)

    def test_an_unstated_gas_temperature_comes_from_the_registry_and_says_so(
        self, documented_manifest: Any
    ) -> None:
        resolved = resolve_plasma(documented_manifest.plasma)
        assert resolved.sources["T_g"] == "registry:RP1.T_g"

    def test_an_unstated_secondary_emission_yield_comes_from_the_registry(
        self, documented_manifest: Any
    ) -> None:
        resolved = resolve_plasma(documented_manifest.plasma)
        assert resolved.sources["gamma_se"] == "registry:sheath.gamma_se_W"
        assert resolved.params.gamma_se == pytest.approx(0.10)

    def test_a_stated_value_overrides_the_registry_and_says_so(
        self, documented_document: dict[str, Any]
    ) -> None:
        documented_document["plasma"]["Ti"] = {"value": 0.2, "units": "eV"}
        manifest = manifest_from_document(documented_document)
        resolved = resolve_plasma(manifest.plasma)

        assert resolved.sources["T_i"] == "manifest"
        assert float(resolved.params.T_i_eV.magnitude) == pytest.approx(0.2)

    def test_every_control_parameter_names_a_source(self, documented_manifest: Any) -> None:
        resolved = resolve_plasma(documented_manifest.plasma)
        expected = {
            "species",
            "n_0",
            "T_e",
            "T_i",
            "T_g",
            "pressure",
            "bias",
            "gamma_se",
            "kappa",
            "rf_frequency",
            "rf_phase",
        }
        assert set(resolved.sources) == expected
        assert all(source for source in resolved.sources.values())

    def test_the_eedf_shape_parameter_is_a_stated_convention_not_a_registry_entry(
        self, documented_manifest: Any
    ) -> None:
        # doc 05 §2.1 gives kappa a *prior*, not a nominal, so there is nothing in the
        # registry to read. The default is therefore a modelling statement and says so.
        resolved = resolve_plasma(documented_manifest.plasma)
        assert resolved.sources["kappa"].startswith("convention:")
        assert resolved.params.kappa == pytest.approx(1.0)


class TestGases:
    def test_xenon_resolves_to_its_registered_mass(self, runnable_document: dict[str, Any]) -> None:
        runnable_document["plasma"]["gas"] = "xenon"
        resolved = resolve_plasma(manifest_from_document(runnable_document).plasma)

        assert resolved.params.species.name == "Xe+"
        assert resolved.params.species.mass.m_as("u") == pytest.approx(131.293)
        assert resolved.sources["species"] == "registry:species.Xe.mass"

    def test_an_unknown_gas_is_refused_and_the_known_ones_are_listed(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["plasma"]["gas"] = "krypton"
        manifest = manifest_from_document(runnable_document)
        with pytest.raises(ValueError, match="argon"):
            resolve_plasma(manifest.plasma)


class TestRadioFrequencyBias:
    def test_an_rf_bias_with_no_stated_frequency_takes_the_registered_one(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["plasma"]["bias"] = {"mode": "rf", "value": -250.0, "units": "V"}
        resolved = resolve_plasma(manifest_from_document(runnable_document).plasma)

        assert resolved.params.is_rf is True
        assert resolved.params.rf_frequency is not None
        assert float(resolved.params.rf_frequency.m_as("MHz")) == pytest.approx(13.56)
        assert resolved.sources["rf_frequency"] == "registry:RP1.rf_frequency"

    def test_a_stated_rf_phase_reaches_the_control_parameters(
        self, runnable_document: dict[str, Any]
    ) -> None:
        runnable_document["plasma"]["bias"] = {
            "mode": "rf",
            "value": -250.0,
            "units": "V",
            "phase": 0.25,
        }
        resolved = resolve_plasma(manifest_from_document(runnable_document).plasma)
        assert resolved.params.rf_phase == pytest.approx(0.25)
        assert resolved.sources["rf_phase"] == "manifest"
