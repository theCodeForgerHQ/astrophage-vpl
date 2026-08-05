"""The spectrograph — doc 04 §5, doc 04 §6.1 and doc 02 §6.

Doc 04 §6.3's build/buy table makes this a **build**, and says exactly how much of one:

    Spectrograph model | Build — thin wrapper computing dispersion, instrument function,
    imaging mapping

So that is what is here, and nothing else. The ray tracing of doc 04 §6.1 — vignetting,
aberration, the depth-of-field weighting of doc 04 §6.2 — is Raysect's job in
``vpl-optics``, and none of it is approximated here under another name.

## Dispersion

The grating equation ``d (sin alpha + sin beta) = m lambda`` with a Czerny-Turner's fixed
angular deviation ``2K`` between the arms. Writing ``alpha = theta - K`` and
``beta = theta + K``, with ``theta`` the grating rotation, it becomes
``2 sin(theta) cos(K) = m lambda / d``, so ``theta`` follows from the wavelength and the
diffraction angle from ``theta + K``. That is the *tuning* relation: it says where to set
the grating so a chosen wavelength lands on the detector centre.

**The dispersion is a different derivative and the distinction is worth a paragraph.**
Across one recorded frame the grating does not move, so ``alpha`` is fixed and only
``beta`` varies: ``dbeta/dlambda = m / (d cos beta)``, giving a reciprocal linear
dispersion ``dlambda/dx = d cos(beta) / (m f)`` at focal length ``f`` — the quantity doc 02
§6.2 tabulates as OES-S3. Differentiating the tuning relation instead gives
``m / (d (cos alpha + cos beta))``, larger by a factor of 2.1 at this mount. The first
implementation here did that, and it passed every internal consistency check it had,
because the wavelength axis and the instrument function were both wrong by the same factor
and the *ratio* between them — the resolving power — was right. The finite-difference test
in ``test_spectrograph.py`` is what found it.

**The arm assignment is not arbitrary.** Putting the *diffracted* arm at the larger angle
is what reproduces OES-S3's 0.62 nm/mm; the mirror-image assignment gives 0.69 nm/mm at the
same mount angle. Since doc 02 §6.2 pins the dispersion and not the geometry,
``OES-S1.deviation_half_angle`` is registered as a swept DESIGN parameter whose nominal
value is the one that closes the specification, and ``test_spectrograph.py`` asserts that
it does.

## The instrument function

Doc 04 §6.1 requires a "measured-style Voigt, not a delta", and doc 02 §11 fits one to an
Hg/Ar lamp. What is assembled here is the physical composition::

    ILS  =  rect(slit image)  ⊗  rect(pixel)  ⊗  Voigt(optical residual)

The two rectangles convolve exactly into a trapezoid, and the trapezoid is convolved with
the Voigt by panelled Gauss-Legendre quadrature split at its corners — where the integrand
is only piecewise smooth, so an unsplit rule would converge at first order.

**A discrepancy in doc 02 that this module makes visible.** OES-S3 and OES-S4 together give
a slit-limited bandpass of ``0.62 nm/mm x 20 um = 0.0124 nm``, and adding the 13 um pixel
does not change the FWHM of a trapezoid. But OES-S4 states the instrument function as
0.026 nm — 2.1 times larger. Doc 02 does not decompose its number, so something optical is
folded into it and left unstated. Rather than quietly using one figure or the other,
``OES-S4.optical_fwhm`` registers the residual explicitly as a DESIGN parameter, swept from
zero, and the test suite pins the assembled width to OES-S4's stated value. That makes the
inconsistency a visible, swept modelling choice rather than a silent factor of two in the
line blending — which for the doc 02 §6.3 line set is the difference between an
over-determined ratio measurement and a blended one.

## What is deliberately not modelled

- **No grating efficiency curve.** The blaze wavelength is registered but the efficiency is
  taken as flat. Over the 434-812 nm span of doc 02 §6.3 a 500 nm blaze varies by a factor
  of about two, so **absolute** line intensities across that span carry that error; the
  ratios within a species pair separated by tens of nanometres do not.
- **No spatial (imaging) axis.** Doc 02 §6.1 uses the spectrograph in imaging mode with the
  slit along the sheath normal, so a real frame is wavelength x position.
  ``OES-S5.magnification`` is registered for it, but only the dispersion axis is modelled;
  the spatial axis is handled by evaluating the emissivity per grid point and treating each
  as an independent spectrum, which omits the cross-talk a real PSF introduces between
  adjacent rows.
- **No wavelength-dependent instrument function beyond the dispersion.** The optical
  residual is taken as constant in nanometres; a real spectrograph's aberrations grow away
  from the blaze.
- **No stray light, no second-order overlap.** Doc 02 §6.1 operates in first order over
  434-812 nm, where second order from 217-406 nm would overlap; a real instrument uses an
  order-sorting filter and this assumes a perfect one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.core.params import ParameterRegistry, default_registry
from vpl.instruments.oes.lineshape import voigt_profile

__all__ = [
    "FWHM_SEARCH_SPAN",
    "PANEL_NODES",
    "Grating",
    "Spectrograph",
]

type FloatArray = NDArray[np.float64]

#: Gauss-Legendre nodes per trapezoid panel in the instrument-function convolution. Three
#: panels, split at the trapezoid corners; the integrand is smooth on each, so this
#: converges to machine precision and the tests assert normalisation to 1e-5 against an
#: independent trapezoid rule.
PANEL_NODES: Final[int] = 64

#: How far out, in multiples of the crude width estimate, :meth:`Spectrograph.instrument_fwhm_nm`
#: samples the profile when measuring its width. Generous: the Voigt component has wings.
FWHM_SEARCH_SPAN: Final[float] = 12.0

#: Samples used in that width measurement. The result is interpolated onto the
#: half-maximum crossing, so this sets the smoothness of the profile and not the precision
#: of the answer.
_FWHM_SAMPLES: Final[int] = 20001

#: Nanometres in a millimetre, for the groove spacing.
_NM_PER_MM: Final[float] = 1.0e6

#: Millimetres in a micrometre.
_MM_PER_UM: Final[float] = 1.0e-3


@dataclass(frozen=True, slots=True)
class Grating:
    """A plane diffraction grating in a fixed-deviation mount.

    Attributes:
        groove_density_per_mm: Grooves per millimetre — doc 02 §6.2 OES-S2.
        order: Diffraction order ``m``. Positive.
    """

    groove_density_per_mm: float
    order: int

    def __post_init__(self) -> None:
        if not self.groove_density_per_mm > 0.0:
            raise ValueError(
                f"groove density must be positive, got {self.groove_density_per_mm} /mm"
            )
        if self.order < 1:
            raise ValueError(
                f"the diffraction order must be at least 1, got {self.order}. Zero order "
                "is the specular reflection and carries no dispersion."
            )

    @property
    def groove_spacing_nm(self) -> float:
        """``d``, the distance between grooves."""
        return _NM_PER_MM / self.groove_density_per_mm

    def _rotation_rad(self, wavelength_nm: float, *, deviation_half_angle_rad: float) -> float:
        """``theta`` from ``2 sin(theta) cos(K) = m lambda / d``."""
        if not wavelength_nm > 0.0:
            raise ValueError(f"wavelength must be positive, got {wavelength_nm} nm")
        sine = (
            self.order
            * wavelength_nm
            / (2.0 * self.groove_spacing_nm * math.cos(deviation_half_angle_rad))
        )
        if abs(sine) > 1.0:
            reachable = (
                2.0 * self.groove_spacing_nm * math.cos(deviation_half_angle_rad) / self.order
            )
            raise ValueError(
                f"the grating cannot diffract {wavelength_nm} nm in order {self.order}: the "
                f"grating equation has no solution above {reachable:.1f} nm for this mount. "
                "A coarser grating or a lower order is needed."
            )
        return math.asin(sine)

    def incidence_angle_rad(
        self, wavelength_nm: float, *, deviation_half_angle_rad: float
    ) -> float:
        """``alpha = theta - K``."""
        return (
            self._rotation_rad(wavelength_nm, deviation_half_angle_rad=deviation_half_angle_rad)
            - deviation_half_angle_rad
        )

    def diffraction_angle_rad(
        self, wavelength_nm: float, *, deviation_half_angle_rad: float
    ) -> float:
        """``beta = theta + K``. See the module docstring on the arm assignment."""
        return (
            self._rotation_rad(wavelength_nm, deviation_half_angle_rad=deviation_half_angle_rad)
            + deviation_half_angle_rad
        )

    def diffraction_angle_at_incidence_rad(
        self, wavelength_nm: float, *, incidence_angle_rad: float
    ) -> float:
        """``beta`` for a wavelength at a **fixed grating setting**.

        Distinct from :meth:`diffraction_angle_rad`, which re-tunes the mount so that the
        wavelength lands on the detector centre. This is the one that describes what
        happens *across* a frame: the grating does not move while a spectrum is recorded,
        so the dispersion at the focal plane is this derivative and not that one. Confusing
        the two gives a dispersion too large by ``(cos alpha + cos beta) / cos beta`` —
        about a factor of two at the doc 02 §6.2 mount, which would put the instrument
        function and the wavelength axis both out by that much in the same direction, where
        they would partly cancel and be very hard to see.
        """
        if not wavelength_nm > 0.0:
            raise ValueError(f"wavelength must be positive, got {wavelength_nm} nm")
        sine = self.order * wavelength_nm / self.groove_spacing_nm - math.sin(incidence_angle_rad)
        if abs(sine) > 1.0:
            raise ValueError(
                f"the grating cannot diffract {wavelength_nm} nm in order {self.order} at an "
                f"incidence of {math.degrees(incidence_angle_rad):.2f} deg"
            )
        return math.asin(sine)

    def angular_dispersion_per_nm(
        self, wavelength_nm: float, *, incidence_angle_rad: float
    ) -> float:
        """``dbeta/dlambda = m / (d cos beta)`` at fixed incidence, in radians per nm."""
        beta = self.diffraction_angle_at_incidence_rad(
            wavelength_nm, incidence_angle_rad=incidence_angle_rad
        )
        return self.order / (self.groove_spacing_nm * math.cos(beta))

    def __repr__(self) -> str:
        return f"Grating({self.groove_density_per_mm:g} /mm, order {self.order})"


def _geometric_kernel(slit_nm: float, pixel_nm: float) -> tuple[FloatArray, FloatArray]:
    """Quadrature nodes and weights integrating against ``rect(slit) ⊗ rect(pixel)``.

    Three cases, and all three are reachable: a real instrument has both widths, a sweep
    down to an ideal slit has one, and the analytic limit has neither.

    The convolution of two rectangles is a **trapezoid** with corners at
    ``+-|slit - pixel|/2`` and support ``+-(slit + pixel)/2``. Splitting the panels at
    those corners is not a refinement: the integrand of the instrument-function
    convolution has a discontinuous derivative at each of them, and a single
    Gauss-Legendre panel spanning a corner converges at first order rather than
    spectrally, which shows up as a percent-level error in the instrument width — the one
    number doc 02 §6.2 checks requirement OES-2 against.
    """
    nodes, quadrature = np.polynomial.legendre.leggauss(PANEL_NODES)

    if slit_nm == 0.0 and pixel_nm == 0.0:
        # No geometric smearing: the kernel is a delta and the caller gets the Voigt back.
        return np.zeros(1, dtype=np.float64), np.ones(1, dtype=np.float64)

    if slit_nm == 0.0 or pixel_nm == 0.0:
        # One rectangle. Smooth on its own support, so one panel is exact enough.
        width = max(slit_nm, pixel_nm)
        return np.asarray(0.5 * width * nodes), np.asarray(0.5 * quadrature)

    wide, narrow = max(slit_nm, pixel_nm), min(slit_nm, pixel_nm)
    half_sum = 0.5 * (wide + narrow)
    half_difference = 0.5 * (wide - narrow)
    edges = sorted({-half_sum, -half_difference, half_difference, half_sum})

    lower = np.asarray(edges[:-1], dtype=np.float64)
    upper = np.asarray(edges[1:], dtype=np.float64)
    half = 0.5 * (upper - lower)
    positions = np.ravel(half[:, None] * nodes[None, :] + 0.5 * (lower + upper)[:, None])
    weights = np.ravel(half[:, None] * quadrature[None, :])

    # The trapezoid itself, written as the overlap length of the two rectangles so that the
    # plateau and the two ramps come out of one expression rather than three branches.
    overlap = np.minimum(0.5 * wide, positions + 0.5 * narrow) - np.maximum(
        -0.5 * wide, positions - 0.5 * narrow
    )
    return positions, np.asarray(weights * np.clip(overlap, 0.0, None) / (wide * narrow))


@dataclass(frozen=True, slots=True)
class Spectrograph:
    """Dispersion, finite slit and instrument function — doc 02 §6.2, doc 04 §6.1.

    Prefer :meth:`from_registry`: every field below is a registered parameter, and building
    one by hand is for the sweeps and the tests.

    Attributes:
        grating: The grating and the order it is used in.
        focal_length_mm: OES-S1.
        deviation_half_angle_rad: Half the fixed angular deviation of the mount.
        slit_width_um: OES-S4. Zero is legal and means an ideal slit.
        pixel_pitch_um: OES-D1. Zero is legal and means an ideal detector.
        n_pixels: OES-D1, along the dispersion axis.
        optical_fwhm_nm: The residual optical broadening — see the module docstring.
        optical_lorentz_fraction: How much of that residual is Lorentzian rather than
            Gaussian. Doc 04 §6.1's "measured-style Voigt".
    """

    grating: Grating
    focal_length_mm: float
    deviation_half_angle_rad: float
    slit_width_um: float
    pixel_pitch_um: float
    n_pixels: int
    optical_fwhm_nm: float
    optical_lorentz_fraction: float

    def __post_init__(self) -> None:
        if not self.focal_length_mm > 0.0:
            raise ValueError(f"the focal length must be positive, got {self.focal_length_mm} mm")
        for name, value in (
            ("slit_width_um", self.slit_width_um),
            ("pixel_pitch_um", self.pixel_pitch_um),
            ("optical_fwhm_nm", self.optical_fwhm_nm),
        ):
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative, got {value}")
        if self.n_pixels < 1:
            raise ValueError(f"a detector needs at least one pixel, got {self.n_pixels}")
        if not 0.0 <= self.optical_lorentz_fraction <= 1.0:
            raise ValueError(
                f"the Lorentz fraction is a fraction, got {self.optical_lorentz_fraction}"
            )
        if self.slit_width_um == 0.0 and self.pixel_pitch_um == 0.0 and self.optical_fwhm_nm == 0.0:
            raise ValueError(
                "a spectrograph with no slit, no pixel and no optical residual has a delta "
                "function for an instrument function. Doc 04 §6.1 requires a "
                "'measured-style Voigt, not a delta'."
            )

    # ── construction ────────────────────────────────────────────────────────────

    @classmethod
    def from_registry(cls, registry: ParameterRegistry | None = None) -> Self:
        """The doc 02 §6.2 instrument, entirely from the parameter registry — doc 08 §5."""
        entries = registry if registry is not None else default_registry()
        return cls(
            grating=Grating(
                groove_density_per_mm=float(entries.value_in("OES-S2.groove_density", "1 / mm")),
                order=int(entries.value_in("OES-S2.diffraction_order", "dimensionless")),
            ),
            focal_length_mm=float(entries.value_in("OES-S1.focal_length", "mm")),
            deviation_half_angle_rad=float(
                entries.value_in("OES-S1.deviation_half_angle", "radian")
            ),
            slit_width_um=float(entries.value_in("OES-S4.slit_width", "um")),
            pixel_pitch_um=float(entries.value_in("OES-D1.pixel_pitch", "um")),
            n_pixels=int(entries.value_in("OES-D1.n_pixels", "dimensionless")),
            optical_fwhm_nm=float(entries.value_in("OES-S4.optical_fwhm", "nm")),
            optical_lorentz_fraction=float(
                entries.value_in("OES-S4.instrument_lorentz_fraction", "dimensionless")
            ),
        )

    def with_slit_width_um(self, slit_width_um: float) -> Self:
        """The same instrument with a different slit — for the doc 07 §5 sweeps."""
        if slit_width_um < 0.0:
            raise ValueError(f"a slit width cannot be negative, got {slit_width_um} um")
        return replace(self, slit_width_um=slit_width_um)

    def with_pixel_pitch_um(self, pixel_pitch_um: float) -> Self:
        if pixel_pitch_um < 0.0:
            raise ValueError(f"a pixel pitch cannot be negative, got {pixel_pitch_um} um")
        return replace(self, pixel_pitch_um=pixel_pitch_um)

    # ── dispersion ──────────────────────────────────────────────────────────────

    def reciprocal_dispersion_nm_per_mm(self, wavelength_nm: float) -> float:
        """``dlambda/dx = d cos(beta) / (m f)`` — doc 02 §6.2 OES-S3.

        Evaluated at the grating setting that puts ``wavelength_nm`` on the detector
        centre, which is what a tabulated reciprocal linear dispersion means.
        """
        beta = self.grating.diffraction_angle_rad(
            wavelength_nm, deviation_half_angle_rad=self.deviation_half_angle_rad
        )
        angular = self.grating.order / (self.grating.groove_spacing_nm * math.cos(beta))
        return 1.0 / (angular * self.focal_length_mm)

    def slit_bandpass_nm(self, wavelength_nm: float) -> float:
        """The slit image, in wavelength. The geometric floor on the resolution."""
        return self.reciprocal_dispersion_nm_per_mm(wavelength_nm) * (
            self.slit_width_um * _MM_PER_UM
        )

    def pixel_bandpass_nm(self, wavelength_nm: float) -> float:
        """One pixel, in wavelength — also the sampling interval of the spectral axis."""
        return self.reciprocal_dispersion_nm_per_mm(wavelength_nm) * (
            self.pixel_pitch_um * _MM_PER_UM
        )

    # ── the instrument function ─────────────────────────────────────────────────

    @property
    def optical_gaussian_fwhm_nm(self) -> float:
        return (1.0 - self.optical_lorentz_fraction) * self.optical_fwhm_nm

    @property
    def optical_lorentz_fwhm_nm(self) -> float:
        return self.optical_lorentz_fraction * self.optical_fwhm_nm

    def instrument_function(
        self,
        offset_nm: ArrayLike,
        *,
        wavelength_nm: float,
        intrinsic_fwhm_nm: float = 0.0,
    ) -> FloatArray:
        """The normalised instrument line-shape function at an offset from a line.

        Args:
            offset_nm: Distance from the line centre, in nanometres.
            wavelength_nm: Where on the spectral axis this is being evaluated. The
                geometric terms scale with the dispersion, which is wavelength-dependent.
            intrinsic_fwhm_nm: Gaussian FWHM of the *line's own* profile, folded in here
                so the caller does not have to convolve twice. Doppler broadening of a
                doc 02 §6.3 line is ~1.6e-3 nm against a 0.026 nm instrument function, so
                it is a small correction — but it is the correction that makes the modelled
                spectrum depend on ``T_g``, and dropping it would remove that dependence
                silently.

        Returns:
            The profile, integrating to one over ``offset_nm``.
        """
        offsets = np.asarray(offset_nm, dtype=np.float64)
        gaussian = math.hypot(self.optical_gaussian_fwhm_nm, intrinsic_fwhm_nm)
        lorentz = self.optical_lorentz_fwhm_nm

        positions, weights = _geometric_kernel(
            self.slit_bandpass_nm(wavelength_nm), self.pixel_bandpass_nm(wavelength_nm)
        )
        kernel = voigt_profile(
            offsets[..., None] - positions, gaussian_fwhm=gaussian, lorentz_fwhm=lorentz
        )
        return np.asarray(kernel @ weights)

    def instrument_fwhm_nm(self, wavelength_nm: float, *, intrinsic_fwhm_nm: float = 0.0) -> float:
        """FWHM of the assembled instrument function, measured off the profile.

        Measured rather than estimated. There is no closed form for a trapezoid convolved
        with a Voigt, and an approximation here would be an unverifiable number reported
        as the instrument's resolution — the one quantity doc 02 §6.2 checks requirement
        OES-2 against.
        """
        scale = (
            self.slit_bandpass_nm(wavelength_nm)
            + self.pixel_bandpass_nm(wavelength_nm)
            + self.optical_fwhm_nm
            + intrinsic_fwhm_nm
        )
        offsets = np.linspace(-FWHM_SEARCH_SPAN * scale, FWHM_SEARCH_SPAN * scale, _FWHM_SAMPLES)
        profile = self.instrument_function(
            offsets, wavelength_nm=wavelength_nm, intrinsic_fwhm_nm=intrinsic_fwhm_nm
        )
        half = 0.5 * float(profile.max())
        above = np.flatnonzero(profile >= half)
        left = np.interp(
            half,
            [profile[above[0] - 1], profile[above[0]]],
            [offsets[above[0] - 1], offsets[above[0]]],
        )
        right = np.interp(
            half,
            [profile[above[-1] + 1], profile[above[-1]]],
            [offsets[above[-1] + 1], offsets[above[-1]]],
        )
        return float(right - left)

    # ── the detector axis ───────────────────────────────────────────────────────

    def wavelength_axis(self, *, centre_nm: float) -> FloatArray:
        """The wavelength of each detector pixel, ascending.

        Uniform in wavelength, which is an approximation: a real grating's dispersion
        varies across the focal plane, by about 2 % over a 1024-pixel span at 1800 gr/mm.
        The wavelength calibration of doc 02 §11 would absorb that in a real instrument,
        and modelling the curvature here without also modelling the calibration that
        removes it would be a needless asymmetry between the two.
        """
        step = self.pixel_bandpass_nm(centre_nm)
        if step == 0.0:
            raise ValueError("a spectral axis needs a finite pixel pitch; this instrument has none")
        offsets = (np.arange(self.n_pixels, dtype=np.float64) - 0.5 * (self.n_pixels - 1)) * step
        return np.asarray(centre_nm + offsets)

    def synthesise(
        self,
        axis_nm: ArrayLike,
        *,
        wavelengths_nm: Sequence[float],
        amplitudes: Sequence[float],
        intrinsic_fwhm_nm: Sequence[float],
    ) -> FloatArray:
        """A spectrum on the detector axis, from a list of lines.

        Args:
            axis_nm: The wavelength axis, from :meth:`wavelength_axis`.
            wavelengths_nm: Line centres.
            amplitudes: Integrated intensity of each line, in whatever units the caller is
                working in. The result carries those units per nanometre.
            intrinsic_fwhm_nm: Each line's own Gaussian FWHM — Doppler, from
                :func:`~vpl.instruments.oes.lineshape.doppler_fwhm_nm`.

        Returns:
            Spectral intensity at each axis point.

        Raises:
            ValueError: If the three line lists are not the same length.
        """
        axis = np.asarray(axis_nm, dtype=np.float64)
        if not len(wavelengths_nm) == len(amplitudes) == len(intrinsic_fwhm_nm):
            raise ValueError(
                f"the line lists must be the same length, got {len(wavelengths_nm)} "
                f"wavelengths, {len(amplitudes)} amplitudes and {len(intrinsic_fwhm_nm)} widths"
            )

        spectrum = np.zeros_like(axis)
        for wavelength, amplitude, width in zip(
            wavelengths_nm, amplitudes, intrinsic_fwhm_nm, strict=True
        ):
            spectrum = spectrum + amplitude * self.instrument_function(
                axis - wavelength, wavelength_nm=wavelength, intrinsic_fwhm_nm=width
            )
        return spectrum

    def __repr__(self) -> str:
        return (
            f"Spectrograph({self.grating!r}, f={self.focal_length_mm:g} mm, "
            f"slit={self.slit_width_um:g} um, {self.n_pixels} pixels)"
        )
