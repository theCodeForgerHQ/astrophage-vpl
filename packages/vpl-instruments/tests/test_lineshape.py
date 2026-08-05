"""Line profiles — doc 04 §2.4.

The externally-anchored checks:

* :func:`test_voigt_fwhm_matches_olivero_longbothum` — Olivero & Longbothum (1977) publish
  a closed-form approximation to the Voigt FWHM and state it accurate to 0.02 %. The
  implementation does not use it; the test measures the FWHM of the profile numerically
  and compares. Two routes, one published.
* :func:`test_voigt_matches_a_direct_numerical_convolution` — the Faddeeva-function form
  against a brute-force convolution of a Gaussian with a Lorentzian, done here.
* :func:`test_doppler_width_matches_the_textbook_coefficient` — the ``7.16e-7`` prefactor
  that every plasma-spectroscopy text quotes for ``Delta lambda / lambda``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.instruments.oes.lineshape import (
    doppler_fwhm_nm,
    natural_fwhm_nm,
    voigt_fwhm_nm,
    voigt_profile,
)

# ── Doppler ─────────────────────────────────────────────────────────────────────


@pytest.mark.physics
def test_doppler_width_matches_the_textbook_coefficient() -> None:
    """``Delta lambda_FWHM / lambda = 7.1623e-7 sqrt(T[K] / M[u])``.

    The coefficient is ``sqrt(8 ln 2 k_B / (u c^2))`` and is quoted to five figures in
    every text on plasma spectroscopy. Reproducing it checks the Boltzmann constant, the
    atomic mass unit, the ``8 ln 2`` and the ``c^2`` all at once.
    """
    temperature_k = 300.0
    mass_u = 39.948
    wavelength_nm = 811.53
    expected = wavelength_nm * 7.1623e-7 * np.sqrt(temperature_k / mass_u)
    assert doppler_fwhm_nm(
        wavelength_nm=wavelength_nm, temperature_k=temperature_k, mass_u=mass_u
    ) == pytest.approx(expected, rel=1e-4)


@pytest.mark.physics
def test_doppler_width_scales_as_the_square_root_of_temperature() -> None:
    cold = doppler_fwhm_nm(wavelength_nm=750.39, temperature_k=300.0, mass_u=39.948)
    hot = doppler_fwhm_nm(wavelength_nm=750.39, temperature_k=1200.0, mass_u=39.948)
    assert hot == pytest.approx(2.0 * cold)


@pytest.mark.physics
def test_a_heavier_emitter_gives_a_narrower_line() -> None:
    argon = doppler_fwhm_nm(wavelength_nm=750.39, temperature_k=300.0, mass_u=39.948)
    xenon = doppler_fwhm_nm(wavelength_nm=750.39, temperature_k=300.0, mass_u=131.293)
    assert xenon < argon


def test_doppler_width_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        doppler_fwhm_nm(wavelength_nm=750.39, temperature_k=0.0, mass_u=39.948)


# ── natural ─────────────────────────────────────────────────────────────────────


@pytest.mark.physics
def test_natural_width_is_the_upper_level_lifetime_in_wavelength() -> None:
    """``Delta nu = sum(A) / 2 pi``, converted by ``Delta lambda = lambda^2 Delta nu / c``.

    The Ar I 811.53 nm upper level has a total transition probability of order 3.9e7 /s,
    giving a natural width of ~14 fm — four orders below the Doppler width, which is the
    statement doc 04 §3.3 makes for LIF and which holds here too.
    """
    width = natural_fwhm_nm(wavelength_nm=811.53, total_a_per_s=3.9e7)
    assert width == pytest.approx(1.36e-5, rel=0.05)
    doppler = doppler_fwhm_nm(wavelength_nm=811.53, temperature_k=300.0, mass_u=39.948)
    assert width < 1e-2 * doppler


@pytest.mark.physics
def test_natural_width_is_linear_in_the_decay_rate() -> None:
    base = natural_fwhm_nm(wavelength_nm=500.0, total_a_per_s=1e8)
    assert natural_fwhm_nm(wavelength_nm=500.0, total_a_per_s=3e8) == pytest.approx(3.0 * base)


# ── Voigt ───────────────────────────────────────────────────────────────────────


def _measured_fwhm(gaussian_fwhm: float, lorentz_fwhm: float) -> float:
    """FWHM of the profile, measured off a fine sampling of it."""
    span = 40.0 * max(gaussian_fwhm, lorentz_fwhm)
    x = np.linspace(-span, span, 2_000_001)
    y = voigt_profile(x, gaussian_fwhm=gaussian_fwhm, lorentz_fwhm=lorentz_fwhm)
    half = 0.5 * float(y.max())
    above = np.flatnonzero(y >= half)
    # Linear interpolation onto the half-maximum crossing, so the answer is not limited
    # to the sample spacing.
    left = np.interp(half, [y[above[0] - 1], y[above[0]]], [x[above[0] - 1], x[above[0]]])
    right = np.interp(half, [y[above[-1] + 1], y[above[-1]]], [x[above[-1] + 1], x[above[-1]]])
    return float(right - left)


@pytest.mark.physics
@pytest.mark.parametrize(
    ("gaussian_fwhm", "lorentz_fwhm"),
    [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.1), (0.1, 1.0), (1.0, 5.0)],
)
def test_voigt_fwhm_matches_olivero_longbothum(gaussian_fwhm: float, lorentz_fwhm: float) -> None:
    """Olivero & Longbothum (1977), J. Quant. Spectrosc. Radiat. Transfer 17, 233.

    ``f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2)``, stated accurate to 0.02 %. The
    left-hand side here is measured off the profile the module actually returns.
    """
    published = 0.5346 * lorentz_fwhm + np.sqrt(0.2166 * lorentz_fwhm**2 + gaussian_fwhm**2)
    assert voigt_fwhm_nm(
        gaussian_fwhm_nm=gaussian_fwhm, lorentz_fwhm_nm=lorentz_fwhm
    ) == pytest.approx(published, rel=1e-9)
    assert _measured_fwhm(gaussian_fwhm, lorentz_fwhm) == pytest.approx(published, rel=1e-3)


@pytest.mark.physics
def test_voigt_matches_a_direct_numerical_convolution() -> None:
    """Gaussian ⊗ Lorentzian, convolved here on a fine grid."""
    gaussian_fwhm, lorentz_fwhm = 1.0, 0.7
    step = 1e-3
    grid = np.arange(-60.0, 60.0 + step, step)

    sigma = gaussian_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gaussian = np.exp(-0.5 * (grid / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    half_width = 0.5 * lorentz_fwhm
    lorentzian = half_width / (np.pi * (grid**2 + half_width**2))
    convolved = np.convolve(gaussian, lorentzian, mode="same") * step

    sample = np.linspace(-4.0, 4.0, 41)
    expected = np.interp(sample, grid, convolved)
    assert voigt_profile(
        sample, gaussian_fwhm=gaussian_fwhm, lorentz_fwhm=lorentz_fwhm
    ) == pytest.approx(expected, rel=1e-6, abs=1e-12)


@pytest.mark.physics
def test_voigt_reduces_to_a_gaussian_with_no_lorentz_component() -> None:
    x = np.linspace(-4.0, 4.0, 81)
    fwhm = 1.3
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gaussian = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    assert voigt_profile(x, gaussian_fwhm=fwhm, lorentz_fwhm=0.0) == pytest.approx(gaussian)


@pytest.mark.physics
def test_voigt_reduces_to_a_lorentzian_with_no_doppler_component() -> None:
    x = np.linspace(-4.0, 4.0, 81)
    fwhm = 1.3
    half_width = 0.5 * fwhm
    lorentzian = half_width / (np.pi * (x**2 + half_width**2))
    assert voigt_profile(x, gaussian_fwhm=0.0, lorentz_fwhm=fwhm) == pytest.approx(lorentzian)


@pytest.mark.physics
def test_voigt_is_normalised() -> None:
    x = np.linspace(-500.0, 500.0, 2_000_001)
    for gaussian_fwhm, lorentz_fwhm in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.2, 3.0)):
        y = voigt_profile(x, gaussian_fwhm=gaussian_fwhm, lorentz_fwhm=lorentz_fwhm)
        assert float(np.trapezoid(y, x)) == pytest.approx(1.0, rel=2e-3)


def test_voigt_needs_at_least_one_width() -> None:
    with pytest.raises(ValueError, match="at least one"):
        voigt_profile(np.zeros(3), gaussian_fwhm=0.0, lorentz_fwhm=0.0)


def test_voigt_rejects_a_negative_width() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        voigt_profile(np.zeros(3), gaussian_fwhm=-1.0, lorentz_fwhm=1.0)
