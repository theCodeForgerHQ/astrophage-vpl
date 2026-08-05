"""The spectrograph — doc 04 §5, §6.1 and doc 02 §6.

The externally-anchored checks:

* :func:`test_reciprocal_dispersion_matches_the_oes_s3_specification` — doc 02 §6.2 OES-S3
  states 0.62 nm/mm as a *derived* value. Reproducing it from the grating equation and the
  OES-S1/OES-S2 rows is a check of this module against the specification, and it is the
  reason the mount angle is a registered parameter rather than a number in the code.
* :func:`test_angular_dispersion_matches_a_numerical_derivative` — the closed-form
  ``dbeta/dlambda = m / (d cos beta)`` at fixed incidence, against a finite difference of
  the diffraction angle taken here. This one caught a real bug: differentiating along the
  mount's *tuning* curve instead of across the focal plane gives an answer 2.1x too large,
  and the first implementation did exactly that.
* :func:`test_instrument_fwhm_matches_the_oes_s4_specification` — doc 02 §6.2 OES-S4's
  0.026 nm, reached by convolving the slit, the pixel and the registered optical residual.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.instruments.oes.spectrograph import Grating, Spectrograph

REGISTRY = default_registry()


@pytest.fixture(scope="module")
def spectrograph() -> Spectrograph:
    return Spectrograph.from_registry()


@pytest.fixture(scope="module")
def grating() -> Grating:
    return Grating(
        groove_density_per_mm=float(REGISTRY.value_in("OES-S2.groove_density", "1 / mm")),
        order=1,
    )


# ── dispersion ──────────────────────────────────────────────────────────────────


@pytest.mark.physics
def test_reciprocal_dispersion_matches_the_oes_s3_specification(
    spectrograph: Spectrograph,
) -> None:
    """doc 02 §6.2 OES-S3: 0.62 nm/mm, derived from OES-S1 and OES-S2."""
    blaze_nm = float(REGISTRY.value_in("OES-S2.blaze_wavelength", "nm"))
    specified = float(REGISTRY.value_in("OES-S3.reciprocal_dispersion", "nm / mm"))
    assert spectrograph.reciprocal_dispersion_nm_per_mm(blaze_nm) == pytest.approx(
        specified, rel=0.01
    )


@pytest.mark.physics
def test_angular_dispersion_matches_a_numerical_derivative(grating: Grating) -> None:
    """``dbeta/dlambda = m / (d cos beta)``, against a finite difference at fixed incidence.

    Fixed incidence is the point. The grating does not move while a frame is recorded, so
    the dispersion across the focal plane is the derivative at fixed ``alpha``; taking it
    along the mount's tuning curve instead gives an answer larger by
    ``(cos alpha + cos beta)/cos beta``, which at this mount is a factor of 2.1.
    """
    incidence = 0.36
    step = 1e-6
    # Near the grating setting, and necessarily so: at 20.6 deg incidence a 1800 gr/mm
    # grating cannot diffract beyond ~570 nm in first order, and the 1024-pixel detector
    # only spans 8 nm anyway. Each line of doc 02 §6.3 gets its own grating setting.
    for wavelength_nm in (470.0, 500.0, 530.0, 560.0):
        analytic = grating.angular_dispersion_per_nm(wavelength_nm, incidence_angle_rad=incidence)
        numerical = (
            grating.diffraction_angle_at_incidence_rad(
                wavelength_nm + step, incidence_angle_rad=incidence
            )
            - grating.diffraction_angle_at_incidence_rad(
                wavelength_nm - step, incidence_angle_rad=incidence
            )
        ) / (2.0 * step)
        assert analytic == pytest.approx(numerical, rel=1e-6)


@pytest.mark.physics
def test_the_grating_equation_is_satisfied_by_the_angles_it_returns(grating: Grating) -> None:
    """``sin(alpha) + sin(beta) = m lambda / d``, checked on the returned pair."""
    deviation = 0.11
    groove_spacing_nm = 1e6 / grating.groove_density_per_mm
    for wavelength_nm in (450.0, 700.0):
        incidence = grating.incidence_angle_rad(wavelength_nm, deviation_half_angle_rad=deviation)
        diffraction = grating.diffraction_angle_rad(
            wavelength_nm, deviation_half_angle_rad=deviation
        )
        assert np.sin(incidence) + np.sin(diffraction) == pytest.approx(
            grating.order * wavelength_nm / groove_spacing_nm
        )
        assert diffraction - incidence == pytest.approx(2.0 * deviation)


@pytest.mark.physics
def test_dispersion_tightens_towards_the_red(spectrograph: Spectrograph) -> None:
    """``cos beta`` falls as the grating is turned, so nm per mm falls with wavelength."""
    wavelengths = np.array([434.81, 500.0, 750.39, 811.53])
    dispersions = np.array(
        [spectrograph.reciprocal_dispersion_nm_per_mm(float(w)) for w in wavelengths]
    )
    assert np.all(np.diff(dispersions) < 0.0)


def test_a_wavelength_the_grating_cannot_reach_is_refused(grating: Grating) -> None:
    with pytest.raises(ValueError, match="cannot diffract"):
        grating.diffraction_angle_rad(2000.0, deviation_half_angle_rad=0.11)


def test_a_grating_needs_a_positive_order() -> None:
    with pytest.raises(ValueError, match="order"):
        Grating(groove_density_per_mm=1800.0, order=0)


# ── the instrument function ─────────────────────────────────────────────────────


def _measure_fwhm(spectrograph: Spectrograph, wavelength_nm: float) -> float:
    span = 20.0 * spectrograph.slit_bandpass_nm(wavelength_nm) + 20.0 * spectrograph.optical_fwhm_nm
    offsets = np.linspace(-span, span, 400_001)
    profile = spectrograph.instrument_function(offsets, wavelength_nm=wavelength_nm)
    half = 0.5 * float(profile.max())
    above = np.flatnonzero(profile >= half)
    left = np.interp(
        half, [profile[above[0] - 1], profile[above[0]]], [offsets[above[0] - 1], offsets[above[0]]]
    )
    right = np.interp(
        half,
        [profile[above[-1] + 1], profile[above[-1]]],
        [offsets[above[-1] + 1], offsets[above[-1]]],
    )
    return float(right - left)


@pytest.mark.physics
def test_instrument_fwhm_matches_the_oes_s4_specification(spectrograph: Spectrograph) -> None:
    """doc 02 §6.2 OES-S4: 0.026 nm FWHM for the 20 um slit.

    The 0.026 nm is 2.1x the slit-limited bandpass that OES-S3 and OES-S4 imply on their
    own, so doc 02 is carrying an unstated optical term; ``OES-S4.optical_fwhm`` is that
    term made explicit, and this test is what pins it. If the specification is ever
    decomposed properly this test becomes a genuine cross-check rather than a consistency
    condition, and the registry entry says so.
    """
    blaze_nm = float(REGISTRY.value_in("OES-S2.blaze_wavelength", "nm"))
    specified = float(REGISTRY.value_in("OES-S4.instrument_fwhm", "nm"))
    assert spectrograph.instrument_fwhm_nm(blaze_nm) == pytest.approx(specified, rel=0.02)
    assert _measure_fwhm(spectrograph, blaze_nm) == pytest.approx(specified, rel=0.02)


@pytest.mark.physics
def test_the_slit_limited_bandpass_is_dispersion_times_slit_width(
    spectrograph: Spectrograph,
) -> None:
    blaze_nm = float(REGISTRY.value_in("OES-S2.blaze_wavelength", "nm"))
    slit_mm = float(REGISTRY.value_in("OES-S4.slit_width", "mm"))
    assert spectrograph.slit_bandpass_nm(blaze_nm) == pytest.approx(
        spectrograph.reciprocal_dispersion_nm_per_mm(blaze_nm) * slit_mm
    )


@pytest.mark.physics
def test_the_instrument_function_is_normalised(spectrograph: Spectrograph) -> None:
    """Over a wide enough window. The Lorentzian component has ``1/x`` wings.

    Truncating at +-0.5 nm loses ``(2/pi) (gamma / x) = 2.2e-3`` of the area, and that is
    not a defect — it is what doc 04 §6.1's "measured-style Voigt" means, and it is why a
    line-ratio measurement on a crowded spectrum has to model the wings rather than fit a
    Gaussian and integrate a box.
    """
    offsets = np.linspace(-50.0, 50.0, 400_001)
    profile = spectrograph.instrument_function(offsets, wavelength_nm=500.0)
    assert float(np.trapezoid(profile, offsets)) == pytest.approx(1.0, rel=1e-4)

    narrow = np.linspace(-0.5, 0.5, 200_001)
    truncated = float(
        np.trapezoid(spectrograph.instrument_function(narrow, wavelength_nm=500.0), narrow)
    )
    predicted_tail = (2.0 / np.pi) * (0.5 * spectrograph.optical_lorentz_fwhm_nm / 0.5)
    assert 1.0 - truncated == pytest.approx(predicted_tail, rel=0.05)


@pytest.mark.physics
def test_a_wide_slit_makes_the_instrument_function_slit_limited(
    spectrograph: Spectrograph,
) -> None:
    """At 500 um the geometry dominates and the FWHM is the slit image, to a percent."""
    wide = spectrograph.with_slit_width_um(500.0)
    expected = wide.slit_bandpass_nm(500.0)
    assert _measure_fwhm(wide, 500.0) == pytest.approx(expected, rel=0.02)


@pytest.mark.physics
def test_closing_the_slit_leaves_only_the_optical_residual(
    spectrograph: Spectrograph,
) -> None:
    """An infinitely narrow slit and no pixel binning leaves the Voigt aberration term."""
    ideal = spectrograph.with_slit_width_um(0.0).with_pixel_pitch_um(0.0)
    from vpl.instruments.oes.lineshape import voigt_fwhm_nm

    expected = voigt_fwhm_nm(
        gaussian_fwhm_nm=ideal.optical_gaussian_fwhm_nm,
        lorentz_fwhm_nm=ideal.optical_lorentz_fwhm_nm,
    )
    assert _measure_fwhm(ideal, 500.0) == pytest.approx(expected, rel=2e-3)


@pytest.mark.physics
def test_the_instrument_function_broadens_towards_the_blue(
    spectrograph: Spectrograph,
) -> None:
    """The geometric terms scale with the dispersion, which is larger in the blue."""
    assert spectrograph.instrument_fwhm_nm(434.81) > spectrograph.instrument_fwhm_nm(811.53)


# ── the detector axis ───────────────────────────────────────────────────────────


@pytest.mark.physics
def test_the_wavelength_axis_is_uniform_and_centred(spectrograph: Spectrograph) -> None:
    axis = spectrograph.wavelength_axis(centre_nm=750.39)
    assert axis.size == spectrograph.n_pixels
    spacing = np.diff(axis)
    assert np.allclose(spacing, spacing[0])
    assert float(np.mean(axis)) == pytest.approx(750.39)
    expected_step = spectrograph.reciprocal_dispersion_nm_per_mm(750.39) * (
        float(REGISTRY.value_in("OES-D1.pixel_pitch", "mm"))
    )
    assert float(spacing[0]) == pytest.approx(expected_step)


@pytest.mark.physics
def test_a_synthesised_spectrum_conserves_line_flux(spectrograph: Spectrograph) -> None:
    """Convolving with a normalised instrument function moves photons, it does not make them."""
    centre = 750.39
    axis = spectrograph.wavelength_axis(centre_nm=centre)
    spectrum = spectrograph.synthesise(
        axis, wavelengths_nm=(centre,), amplitudes=(1.0,), intrinsic_fwhm_nm=(1.6e-3,)
    )
    step = float(axis[1] - axis[0])
    # 1e-3 and not 1e-6: the detector spans +-4.1 nm and the Lorentzian wings of the
    # instrument function put 2.6e-4 of the line outside it. That loss is physical — a real
    # frame does not see the far wings either — so the tolerance states it rather than
    # hiding it behind a wider axis.
    assert float(np.sum(spectrum) * step) == pytest.approx(1.0, rel=1e-3)


@pytest.mark.physics
def test_two_lines_inside_the_instrument_function_are_not_resolved(
    spectrograph: Spectrograph,
) -> None:
    centre = 750.39
    axis = spectrograph.wavelength_axis(centre_nm=centre)
    separation = 0.3 * spectrograph.instrument_fwhm_nm(centre)
    blended = spectrograph.synthesise(
        axis,
        wavelengths_nm=(centre - 0.5 * separation, centre + 0.5 * separation),
        amplitudes=(1.0, 1.0),
        intrinsic_fwhm_nm=(1.6e-3, 1.6e-3),
    )
    peaks = np.flatnonzero((blended[1:-1] > blended[:-2]) & (blended[1:-1] > blended[2:]))
    assert peaks.size == 1


@pytest.mark.physics
def test_two_well_separated_lines_are_resolved(spectrograph: Spectrograph) -> None:
    centre = 750.39
    axis = spectrograph.wavelength_axis(centre_nm=centre)
    separation = 6.0 * spectrograph.instrument_fwhm_nm(centre)
    resolved = spectrograph.synthesise(
        axis,
        wavelengths_nm=(centre - 0.5 * separation, centre + 0.5 * separation),
        amplitudes=(1.0, 1.0),
        intrinsic_fwhm_nm=(1.6e-3, 1.6e-3),
    )
    peaks = np.flatnonzero((resolved[1:-1] > resolved[:-2]) & (resolved[1:-1] > resolved[2:]))
    assert peaks.size == 2


def test_synthesise_rejects_mismatched_line_lists(spectrograph: Spectrograph) -> None:
    axis = spectrograph.wavelength_axis(centre_nm=750.39)
    with pytest.raises(ValueError, match="same length"):
        spectrograph.synthesise(
            axis, wavelengths_nm=(750.39, 751.47), amplitudes=(1.0,), intrinsic_fwhm_nm=(1e-3,)
        )


def test_a_negative_slit_width_is_refused(spectrograph: Spectrograph) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        spectrograph.with_slit_width_um(-1.0)
