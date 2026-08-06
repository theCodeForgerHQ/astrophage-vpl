"""``vpl.instruments.thomson.spectrum`` — doc 04 §4.1-§4.2, doc 02 §4.2 check 1.

Plain module-level helpers rather than a shared ``conftest.py`` — see the note at the top
of ``oes_system.py`` for why (this package is developed by more than one person at once,
and a shared ``conftest`` is the file they collide on).
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.instruments.thomson.spectrum import (
    INCOHERENT_REGIME_MAXIMUM_ALPHA,
    LASER_WAVELENGTH_NM,
    SCATTERING_ANGLE_DEG,
    CoherentScatteringRegimeError,
    check_incoherent_regime,
    debye_length_m,
    gaussian_half_width_nm,
    gaussian_spectrum,
    relativistic_spectrum,
    relativistic_weighting_factor,
    salpeter_alpha,
    scattering_wavenumber_per_m,
)

_RP1_N_E_M3 = 1.0e17
_RP1_T_E_EV = 3.0


class TestGeometry:
    def test_the_scattering_angle_is_90_degrees_per_ts_o1(self) -> None:
        assert pytest.approx(90.0) == SCATTERING_ANGLE_DEG

    def test_the_laser_wavelength_is_532_nm_per_ts_l1(self) -> None:
        assert pytest.approx(532.0) == LASER_WAVELENGTH_NM


class TestDebyeLength:
    def test_it_matches_the_doc_02_worked_example_at_rp1(self) -> None:
        """Doc 02 §4.2 check 1 quotes 40.7 um at RP-1."""
        lambda_d_m = debye_length_m(
            electron_density_m3=_RP1_N_E_M3, electron_temperature_ev=_RP1_T_E_EV
        )
        assert lambda_d_m * 1.0e6 == pytest.approx(40.72, rel=1.0e-3)

    def test_it_is_refused_for_a_non_positive_density(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            debye_length_m(electron_density_m3=0.0, electron_temperature_ev=_RP1_T_E_EV)


class TestSalpeterAlpha:
    def test_it_matches_the_doc_04_worked_example_at_rp1(self) -> None:
        """Doc 04 §4.1: alpha = 0.0015 at RP-1, "firmly incoherent"."""
        alpha = salpeter_alpha(electron_density_m3=_RP1_N_E_M3, electron_temperature_ev=_RP1_T_E_EV)
        assert alpha == pytest.approx(0.00147, rel=1.0e-2)

    def test_rp1_is_comfortably_inside_the_incoherent_regime(self) -> None:
        alpha = salpeter_alpha(electron_density_m3=_RP1_N_E_M3, electron_temperature_ev=_RP1_T_E_EV)
        assert alpha < INCOHERENT_REGIME_MAXIMUM_ALPHA


class TestScatteringWavenumber:
    def test_it_is_positive_and_finite(self) -> None:
        k = scattering_wavenumber_per_m()
        assert k > 0.0
        assert np.isfinite(k)


class TestCheckIncoherentRegime:
    def test_it_accepts_the_rp1_alpha(self) -> None:
        check_incoherent_regime(0.0015)  # must not raise

    def test_it_refuses_an_alpha_at_the_coherent_crossover(self) -> None:
        with pytest.raises(CoherentScatteringRegimeError, match="TS-2"):
            check_incoherent_regime(1.0)

    def test_it_refuses_an_alpha_right_at_its_own_threshold(self) -> None:
        with pytest.raises(CoherentScatteringRegimeError):
            check_incoherent_regime(INCOHERENT_REGIME_MAXIMUM_ALPHA)


class TestGaussianHalfWidth:
    def test_it_matches_doc_04_section_4_2_at_3_ev(self) -> None:
        half_width_nm = gaussian_half_width_nm(electron_temperature_ev=3.0)
        assert half_width_nm == pytest.approx(2.578, rel=1.0e-3)

    def test_it_fits_inside_the_ts_s2_window_at_10_ev(self) -> None:
        """Doc 04 §4.2: TS-S2's 520-545 nm coverage "accommodates T_e well beyond 10 eV"."""
        half_width_nm = gaussian_half_width_nm(electron_temperature_ev=10.0)
        assert half_width_nm == pytest.approx(4.707, rel=1.0e-3)
        assert half_width_nm < 25.0 / 2.0

    def test_it_grows_with_electron_temperature(self) -> None:
        cold = gaussian_half_width_nm(electron_temperature_ev=1.0)
        hot = gaussian_half_width_nm(electron_temperature_ev=9.0)
        assert hot > cold


class TestGaussianSpectrum:
    def test_it_peaks_at_one_on_the_laser_line(self) -> None:
        shape = gaussian_spectrum(np.array([0.0]), electron_temperature_ev=3.0)
        assert float(shape[0]) == pytest.approx(1.0)

    def test_it_is_symmetric_about_the_laser_line(self) -> None:
        offsets = np.array([-2.0, 2.0])
        shape = gaussian_spectrum(offsets, electron_temperature_ev=3.0)
        assert float(shape[0]) == pytest.approx(float(shape[1]))

    def test_it_falls_to_1_over_e_at_the_half_width(self) -> None:
        half_width_nm = gaussian_half_width_nm(electron_temperature_ev=3.0)
        shape = gaussian_spectrum(np.array([half_width_nm]), electron_temperature_ev=3.0)
        assert float(shape[0]) == pytest.approx(1.0 / np.e, rel=1.0e-6)


class TestRelativisticSpectrum:
    def test_it_reduces_to_the_gaussian_spectrum_as_t_e_goes_to_zero(self) -> None:
        """Doc 04 §4.2: the relativistic correction is a ~1 % effect at 10 eV, so at a
        colder temperature it must vanish into the plain Gaussian."""
        cold_ev = 0.05
        half_width_nm = gaussian_half_width_nm(electron_temperature_ev=cold_ev)
        offsets = np.linspace(-2.0 * half_width_nm, 2.0 * half_width_nm, 9)

        relativistic = relativistic_spectrum(offsets, electron_temperature_ev=cold_ev)
        gaussian = gaussian_spectrum(offsets, electron_temperature_ev=cold_ev)

        np.testing.assert_allclose(relativistic, gaussian, rtol=2.0e-3)

    def test_it_peaks_at_one_on_the_laser_line(self) -> None:
        shape = relativistic_spectrum(np.array([0.0]), electron_temperature_ev=10.0)
        assert float(shape[0]) == pytest.approx(1.0)

    def test_the_blue_shifted_side_is_enhanced_relative_to_the_gaussian(self) -> None:
        """Doc 04 §4.2: "a small blue-shifted asymmetry" — negative delta_lambda is blue."""
        half_width_nm = gaussian_half_width_nm(electron_temperature_ev=10.0)
        offsets = np.array([-half_width_nm, half_width_nm])

        relativistic = relativistic_spectrum(offsets, electron_temperature_ev=10.0)
        gaussian = gaussian_spectrum(offsets, electron_temperature_ev=10.0)

        blue_ratio = relativistic[0] / gaussian[0]
        red_ratio = relativistic[1] / gaussian[1]
        assert blue_ratio > red_ratio

    def test_the_asymmetry_grows_with_electron_temperature(self) -> None:
        def asymmetry(electron_temperature_ev: float) -> float:
            half_width_nm = gaussian_half_width_nm(electron_temperature_ev=electron_temperature_ev)
            offsets = np.array([-half_width_nm, half_width_nm])
            relativistic = relativistic_spectrum(
                offsets, electron_temperature_ev=electron_temperature_ev
            )
            gaussian = gaussian_spectrum(offsets, electron_temperature_ev=electron_temperature_ev)
            return float(relativistic[0] / gaussian[0] - relativistic[1] / gaussian[1])

        cooler = asymmetry(3.0)
        hotter = asymmetry(10.0)
        assert hotter > cooler > 0.0


class TestRelativisticWeightingFactor:
    def test_it_equals_selden_1980s_leading_factor_to_machine_precision(self) -> None:
        """Algebraic identity (peer-reviewed independently, see the module docstring):
        the kinematic Jacobian and the Doppler-boost weighting this module derives from
        first principles combine to *exactly* ``(1 + xi)^-3``, ``xi = delta_lambda /
        lambda_0`` — Selden's own leading relativistic factor ``k(epsilon, theta)`` at
        this geometry. Pinned here so a future "simplification" of either factor cannot
        silently break the identity without a test noticing."""
        offsets_nm = np.linspace(-10.0, 10.0, 41)
        xi = offsets_nm / LASER_WAVELENGTH_NM

        weighting = relativistic_weighting_factor(offsets_nm)
        identity = (1.0 + xi) ** (-3)

        np.testing.assert_allclose(weighting, identity, rtol=1.0e-12, atol=1.0e-15)

    def test_relativistic_spectrum_uses_this_same_weighting_factor(self) -> None:
        """Cross-checks that :func:`relativistic_spectrum` is built from this factor and
        not a second, independently-maintained copy of it."""
        offsets_nm = np.array([-3.0, 0.0, 4.0])
        electron_temperature_ev = 5.0

        spectrum_values = relativistic_spectrum(
            offsets_nm, electron_temperature_ev=electron_temperature_ev
        )
        weighting = relativistic_weighting_factor(offsets_nm)
        implied_exponential = spectrum_values / weighting

        # The Gaussian-only spectrum, isolated by dividing the weighting factor back out,
        # must itself be <= 1 everywhere and exactly 1 on the laser line.
        assert np.all(implied_exponential <= 1.0 + 1.0e-12)
        assert implied_exponential[1] == pytest.approx(1.0)
