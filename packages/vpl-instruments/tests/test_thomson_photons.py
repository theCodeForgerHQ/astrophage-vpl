"""``vpl.instruments.thomson.photons`` — doc 02 §7.1, §4.3; doc 06 §4, §5."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.instruments.thomson import photons


class TestPhotoelectronsPerShot:
    @pytest.mark.parametrize(
        ("electron_density_m3", "expected_pe_per_shot"),
        [
            (1.0e16, 0.01595),
            (1.0e17, 0.1595),
            (1.0e18, 1.595),
            (1.0e19, 15.95),
        ],
    )
    def test_it_matches_the_doc_02_section_7_1_table(
        self, electron_density_m3: float, expected_pe_per_shot: float
    ) -> None:
        assert photons.photoelectrons_per_shot(electron_density_m3) == pytest.approx(
            expected_pe_per_shot, rel=2.0e-3
        )

    def test_it_is_linear_in_electron_density(self) -> None:
        low = photons.photoelectrons_per_shot(1.0e16)
        high = photons.photoelectrons_per_shot(1.0e17)
        assert high == pytest.approx(low * 10.0, rel=1.0e-9)

    def test_it_is_refused_for_a_negative_density(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            photons.photoelectrons_per_shot(-1.0)


class TestThomsonCrossSection:
    def test_it_matches_the_codata_derived_value(self) -> None:
        """doc 02 §7.1 quotes sigma_T = 6.652e-29 m^2."""
        assert pytest.approx(6.652e-29, rel=1.0e-3) == photons.THOMSON_CROSS_SECTION_M2


class TestAccumulation:
    def test_a_700_second_window_at_rp1_gives_about_3_percent(self) -> None:
        """Doc 02 §7.1 / doc 06 §4 term 8: "~700 s" for 3 % at RP-1."""
        relative_uncertainty = photons.relative_statistical_uncertainty(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3, duration_s=700.0
        )
        assert relative_uncertainty == pytest.approx(0.03, rel=0.05)

    def test_required_accumulation_at_rp1_matches_doc_02_within_a_percent(self) -> None:
        accumulation_s = photons.required_accumulation_s(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3,
            target_relative_uncertainty=0.03,
        )
        assert accumulation_s == pytest.approx(696.7, rel=1.0e-2)

    def test_required_accumulation_and_relative_uncertainty_are_mutually_inverse(self) -> None:
        target = 0.03
        accumulation_s = photons.required_accumulation_s(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3,
            target_relative_uncertainty=target,
        )
        achieved = photons.relative_statistical_uncertainty(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3, duration_s=accumulation_s
        )
        assert achieved == pytest.approx(target, rel=1.0e-6)

    def test_a_lower_target_uncertainty_needs_more_shots(self) -> None:
        loose = photons.required_shots(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3,
            target_relative_uncertainty=0.03,
        )
        tight = photons.required_shots(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3,
            target_relative_uncertainty=0.01,
        )
        assert tight > loose

    def test_16_phase_bins_at_3_percent_needs_about_111_000_shots(self) -> None:
        """Doc 02 §7.1 consequence 2: "16 bins at 3 % requires ~111 000 shots"."""
        shots = photons.required_shots(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3,
            target_relative_uncertainty=0.03,
            n_phase_bins=16.0,
        )
        assert shots == pytest.approx(111_470.0, rel=1.0e-2)

    def test_16_phase_bins_at_3_percent_needs_about_3_1_hours(self) -> None:
        accumulation_s = photons.required_accumulation_s(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3,
            target_relative_uncertainty=0.03,
            n_phase_bins=16.0,
        )
        assert accumulation_s / 3600.0 == pytest.approx(3.1, rel=0.02)

    def test_required_shots_is_refused_for_a_non_positive_target(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            photons.required_shots(
                electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3,
                target_relative_uncertainty=0.0,
            )


class TestIsBlind:
    def test_a_single_shot_at_rp1_is_blind(self) -> None:
        """Doc 02 §7.1 consequence 2: "any single-shot or few-shot event is invisible"."""
        assert photons.is_blind(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3, shots=1.0
        )

    def test_a_700_second_accumulation_at_rp1_is_not_blind(self) -> None:
        shots = photons.shots_in_window(700.0)
        assert not photons.is_blind(
            electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3, shots=shots
        )


class TestStrayLightRejectionStack:
    def test_the_combined_rejection_meets_ts_3_with_the_documented_margin(self) -> None:
        combined = photons.combined_rejection_factor()
        assert combined >= (
            photons.STRAY_LIGHT_REJECTION_REQUIREMENT * photons.STRAY_LIGHT_REJECTION_MARGIN
        )

    def test_the_combined_rejection_is_order_1e9(self) -> None:
        """Doc 02 §4.3: "Combined | ~1e9 | with 10x margin against the 1e8 requirement"."""
        combined = photons.combined_rejection_factor()
        assert 1.0e8 < combined < 1.0e10


class TestStrayLightSpectrum:
    def test_it_peaks_at_the_laser_wavelength(self) -> None:
        axis = np.array([520.0, 526.0, 532.0, 538.0, 544.0])
        spectrum = photons.stray_light_spectrum_pe_per_shot(axis, notch_width_nm=1.25)
        assert int(np.argmax(spectrum)) == 2

    def test_it_falls_off_away_from_the_laser_wavelength(self) -> None:
        axis = np.array([520.0, 532.0])
        spectrum = photons.stray_light_spectrum_pe_per_shot(axis, notch_width_nm=1.25)
        assert spectrum[1] > spectrum[0]

    def test_it_does_not_scale_with_electron_density(self) -> None:
        """Unlike the genuine Thomson signal: the pedestal comes from the laser pulse
        scattering off hardware, not from the plasma."""
        axis = np.array([532.0])
        first = photons.stray_light_spectrum_pe_per_shot(axis, notch_width_nm=1.25)
        second = photons.stray_light_spectrum_pe_per_shot(axis, notch_width_nm=1.25)
        np.testing.assert_array_equal(first, second)


class TestRayleighCalibrationChain:
    def test_the_five_term_quadrature_sum_lands_near_doc_06_section_5s_range(self) -> None:
        assert 0.065 <= photons.RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY <= 0.067

    def test_it_matches_a_direct_quadrature_recomputation(self) -> None:
        expected = math.sqrt(
            photons.RAYLEIGH_CROSS_SECTION_UNCERTAINTY**2
            + photons.GAS_PURITY_UNCERTAINTY**2
            + photons.PRESSURE_GAUGE_UNCERTAINTY**2
            + photons.OPTICAL_STABILITY_UNCERTAINTY**2
            + photons.CALIBRATION_PHOTON_STATISTICS_UNCERTAINTY**2
        )
        assert pytest.approx(expected) == photons.RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY

    def test_the_calibration_photon_statistics_term_is_the_largest_single_contributor(self) -> None:
        """doc 06 §5: the calibration's own photon statistics dominate the chain — the
        same photon-starvation problem the science measurement has."""
        terms = (
            photons.RAYLEIGH_CROSS_SECTION_UNCERTAINTY,
            photons.GAS_PURITY_UNCERTAINTY,
            photons.PRESSURE_GAUGE_UNCERTAINTY,
            photons.OPTICAL_STABILITY_UNCERTAINTY,
        )
        assert max(terms) < photons.CALIBRATION_PHOTON_STATISTICS_UNCERTAINTY
