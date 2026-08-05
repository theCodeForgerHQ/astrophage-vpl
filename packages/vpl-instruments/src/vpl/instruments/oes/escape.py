"""Radiation trapping — doc 04 §2.3, the ``Lambda_ul`` of doc 04 §2.2.

Doc 04 §2.3 states why this module exists rather than a constant ``1.0``:

    The Ar I resonance lines (104.8, 106.7 nm) are strongly self-absorbed, which pumps the
    metastable population far above its optically-thin value. Since the 811.53 nm and
    763.51 nm lines used in the ``T_e`` diagnostic are *metastable-coupled*, **an
    optically-thin CR model would produce systematically wrong line ratios and therefore a
    systematically wrong** ``T_e``.

## The escape factor as implemented

The formulation is the mean escape probability with complete frequency redistribution —
Holstein (1947, 1951) as reviewed by Irons (*J. Quant. Spectrosc. Radiat. Transfer* **22**
(1979) 1). A photon emitted at line-centre optical depth ``tau_0`` into a homogeneous slab
escapes on its first flight with a probability that depends on where in the line profile
it was emitted, and the escape factor is that probability averaged over the profile::

    Lambda  =  integral dnu phi(nu) P( tau_0 phi(nu)/phi(nu_0) )

``P(tau)`` is the position- and angle-averaged single-flight escape probability of a
uniform slab of optical thickness ``tau``. It has a closed form. For an emitter at optical
depth ``t`` radiating isotropically, escape either way is
``[E_2(t) + E_2(tau - t)] / 2``; averaging over ``t`` and using
``integral_0^x E_2 = 1/2 - E_3(x)`` gives

    P(tau)  =  ( 1 - 2 E_3(tau) ) / (2 tau)

which is :func:`slab_escape_probability`. ``E_3`` comes from SciPy, and the closed form is
checked in the tests against a brute-force quadrature over emitter position and emission
angle written independently — the only external anchor available for a quantity nobody
tabulates.

## Which escape factor this is, and which it is not

There are two quantities in the literature under this name and they are **not** the same
number. Getting the wrong one is worth a factor of ``ln tau_0``, which at the resonance-line
opacity of the doc 01 §2.1 operating point is a factor of four.

*Holstein's* ``g`` is the decay eigenvalue of his integral equation: the rate at which the
*fundamental spatial mode* of a trapped resonance line dies away, ``g = 1.875 / (k_0 L
sqrt(pi ln k_0 L))`` for a Doppler line in a slab. That is the right quantity for a
transient — an afterglow, a pulsed source relaxing into its slowest mode.

What is implemented here is the *mean escape probability* of Irons' review: the fraction of
photons emitted by a **uniformly excited** volume that leave without re-absorption. That is
the right quantity for doc 04 §2.2, whose balance equation is a local steady-state rate
balance with ``n_e n_j K_ju`` as a uniform volumetric source. A re-absorbed photon
re-excites the level, so the net loss from the level is ``A_ul`` times the probability of
*not* being re-absorbed, which is this. The population is held in a uniform profile by the
source term and never relaxes into Holstein's mode.

Both are stated because the difference is real and in the unsafe direction: the mean escape
probability is the *larger* of the two, so using it where Holstein's ``g`` belongs would
under-estimate trapping.

## Verification anchors

Three, and all of them are analytic rather than internal:

- ``tau_0 -> 0`` gives ``Lambda -> 1``: doc 04 §8 **V-26**, "escape factor -> 1 as
  ``n_g`` -> 0".
- **Lorentz, optically thick:** ``Lambda sqrt(tau_0) -> 4 / (3 sqrt(pi)) = 0.7522528``.
  Derived by taking ``P(tau) -> 1/(2 tau)``, writing
  ``E_3(z) = integral_0^1 mu exp(-z/mu) dmu`` and using
  ``integral_0^inf (1 - exp(-a u^2)) / u^2 du = sqrt(pi a)``; the ``theta`` integral then
  collapses to ``(2/pi) sqrt(pi) integral_0^1 sqrt(mu) dmu``. The implementation reproduces
  all eight digits, which is what makes the quadrature panelling checkable rather than
  merely plausible.
- **Doppler, optically thick:** ``Lambda -> sqrt(ln tau_0) / (tau_0 sqrt(pi))``. Same
  substitution, with the profile integral reducing to
  ``integral_0^1 ds / sqrt(1 - s) = 2``. The approach is slow — ``1 + O(1/ln tau_0)`` — so
  the test asserts convergence towards the limit rather than agreement with it.

## What is deliberately not modelled

Named here rather than discovered later (doc 00 C4). Each bound is stated, because doc 06
§4 budgets the escape-factor uncertainty and a term with no bound cannot be budgeted.

- **Slab geometry only.** The doc 02 §3.1 chamber is a 400 mm x 300 mm cylinder and this
  treats it as a slab of the diameter. Holstein's own asymptotic constants for the two
  geometries differ by 1.875/1.60 = 1.17, so the geometry choice is a **~17 % systematic
  on ``Lambda``** in the optically thick limit and less than that at moderate opacity.
  That is the bound; it is not small, and it is the largest single approximation here.
- **Uniform absorber density.** ``n_l`` is taken constant along the path. In a sheath it
  is not, but the absorber for the resonance lines is the *ground state*, which is uniform
  to the accuracy of the neutral-density profile.
- **No stimulated emission correction.** The absorption coefficient omits the
  ``1 - g_l n_u / (g_u n_l)`` factor. At the doc 01 §2.1 operating point the excited
  fraction is below 1e-5, so the correction is below 1e-5.
- **No Voigt profile.** :class:`LineProfileShape` offers a Doppler and a Lorentz limit and
  not their convolution. For the resonance lines at 5 mTorr the Doppler width is ~5.5 GHz
  and the natural width 19 MHz, so the profile is Doppler to within 0.4 % in the core;
  the *wings* are Lorentzian and escape more easily, so a Doppler-only treatment
  **under**-estimates escape in the far wing and therefore over-estimates trapping. The
  sign of that error is stated because it is the direction that matters for the metastable
  population.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from vpl.core.constants import SPEED_OF_LIGHT
from vpl.core.units import Q_, magnitude_in

__all__ = [
    "PANEL_DECADES_BELOW_UNITY",
    "PANEL_NODES",
    "SMALL_OPTICAL_DEPTH",
    "LineProfileShape",
    "TrappedLine",
    "escape_factor",
    "line_centre_optical_depth",
    "slab_escape_probability",
]

type FloatArray = NDArray[np.float64]

# ── constants ───────────────────────────────────────────────────────────────────

#: Speed of light in m/s, from CODATA via :mod:`vpl.core.constants`. Unwrapped once.
_C_M_PER_S: Final[float] = float(magnitude_in(SPEED_OF_LIGHT, "m / s"))

#: Metres in a nanometre. A unit conversion rather than a measurement, but taken from the
#: shared registry rather than written: doc 08 §5's rule does not carve out exceptions for
#: numbers the author was sure about, and a stray 1e9 in an opacity is precisely the error
#: the rule exists to catch.
_METRES_PER_NM: Final[float] = float(magnitude_in(Q_(1.0, "nm"), "m"))

#: ``psi(3) = 1 + 1/2 - gamma``, the constant in the small-``tau`` expansion of
#: :func:`slab_escape_probability`. Computed rather than typed: it is a value of a
#: special function, and doc 09 §2.5's rule about hand-entered constants applies to any
#: number whose provenance is a formula.
_PSI_3: Final[float] = float(special.digamma(3))

#: Below this optical depth the closed form ``(1 - 2 E_3(tau)) / (2 tau)`` loses its
#: leading digits to cancellation — ``E_3`` approaches exactly 1/2 — so the series
#: expansion is used instead. At 1e-4 the two agree to eleven digits, which is where the
#: switch is placed: high enough that the series is still converged, low enough that the
#: closed form has not yet started to shed precision.
SMALL_OPTICAL_DEPTH: Final[float] = 1.0e-4

#: Gauss-Legendre nodes per panel in the profile integral. The panels are split so that
#: the integrand is smooth on each (see :func:`_panelled_rule`), which is what lets this
#: be small; doubling it moves the answer by less than 1e-10 at every opacity the tests
#: cover, and the convergence is asserted there rather than claimed here.
PANEL_NODES: Final[int] = 48

#: How many decades of optical depth below unity the panelling continues to. The
#: integrand is within 1e-6 of its thin-limit value below ``tau = 1e-6``, so panels
#: further down buy nothing.
PANEL_DECADES_BELOW_UNITY: Final[int] = 6

#: How far into the Gaussian tail the Doppler profile integral runs, as an exponent:
#: the upper limit is ``x`` such that ``exp(-x**2)`` is ``exp(-this)``. At 40 the
#: neglected tail is 4e-18 of the profile, below double-precision relevance.
_DOPPLER_TAIL_EXPONENT: Final[float] = 40.0

#: ``2 sqrt(ln 2 / pi)`` — the line-centre value of a Doppler profile normalised in
#: frequency and parameterised by its FWHM.
_DOPPLER_PEAK: Final[float] = 2.0 * math.sqrt(math.log(2.0) / math.pi)

#: ``2 / pi`` — the same for a Lorentzian.
_LORENTZ_PEAK: Final[float] = 2.0 / math.pi


class LineProfileShape(StrEnum):
    """Which line profile the frequency redistribution is taken over.

    Not a cosmetic choice: the profile decides how the escape factor falls off with
    opacity, because the photons that escape a thick line are the ones emitted in its
    wings. Doppler and Lorentz differ by ``ln tau`` versus ``sqrt(tau)`` there.
    """

    #: Gaussian in frequency — thermal motion of the emitters. The right choice for every
    #: line in doc 02 §6.3 at the doc 01 §2.1 operating point.
    DOPPLER = "doppler"

    #: Lorentzian — natural and collisional broadening. Carried because it is the other
    #: analytic limit and because its published ``tau^(-1/2)`` asymptote is a verification
    #: anchor.
    LORENTZ = "lorentz"

    #: A delta function: every photon at line centre, no wings, no redistribution. Not
    #: physical — no line is monochromatic — but it is the lower bound on escape at a
    #: given line-centre opacity, and :func:`slab_escape_probability` is exactly it.
    MONOCHROMATIC = "monochromatic"


# ── the single-flight escape probability ────────────────────────────────────────


def slab_escape_probability(optical_depth: ArrayLike) -> FloatArray:
    """``P(tau) = (1 - 2 E_3(tau)) / (2 tau)`` — see the module docstring.

    The position- and angle-averaged probability that a photon emitted isotropically in a
    uniform slab of optical thickness ``tau`` leaves it without absorption.

    Args:
        optical_depth: One or more optical thicknesses. Zero is allowed and gives one.

    Returns:
        The escape probability, same shape as the input.

    Raises:
        ValueError: If any optical depth is negative. A negative opacity is a sign error
            upstream, and returning a probability above one for it would propagate into a
            level population as a spurious source.
    """
    tau = np.asarray(optical_depth, dtype=np.float64)
    if np.any(tau < 0.0):
        raise ValueError(
            f"an optical depth cannot be negative, got a minimum of {float(np.min(tau))}"
        )

    small = tau < SMALL_OPTICAL_DEPTH
    # np.where evaluates both branches, so the arguments are made safe first rather than
    # left to produce a division by zero that is then discarded — under
    # `filterwarnings = ["error"]` (root pyproject) that discarded warning is a failure.
    safe_tau = np.where(small, 1.0, tau)
    safe_log = np.where(tau > 0.0, np.where(small, tau, 1.0), 1.0)

    closed_form = (1.0 - 2.0 * special.expn(3, safe_tau)) / (2.0 * safe_tau)
    series = 1.0 - 0.5 * tau * (_PSI_3 - np.log(safe_log)) - tau**2 / 6.0
    return np.asarray(np.where(small, series, closed_form))


# ── the profile average ─────────────────────────────────────────────────────────


def _panelled_rule(edges: Sequence[float]) -> tuple[FloatArray, FloatArray]:
    """Nodes and weights of a composite :data:`PANEL_NODES`-point rule over ``edges``.

    Panelling is not a refinement, it is the difference between a converged answer and a
    plausible one. Both integrands here are essentially flat except in one narrow band
    where the line stops being opaque, and that band is *narrower the thicker the line*:
    for a Lorentzian at ``tau_0 = 1e10`` it is 1e-5 rad wide inside a range of 1.57. A
    single Gauss-Legendre panel puts no node in it, returns a smooth wrong number, and
    nothing about the result says so. Splitting at every decade of ``tau`` puts a whole
    rule inside the band at any opacity.

    All panels are returned as one flat pair, so the caller makes a single vectorised
    evaluation of :func:`slab_escape_probability` rather than one per panel.
    """
    nodes, weights = np.polynomial.legendre.leggauss(PANEL_NODES)
    lower = np.asarray(edges[:-1], dtype=np.float64)
    upper = np.asarray(edges[1:], dtype=np.float64)
    half = 0.5 * (upper - lower)
    return (
        np.ravel(half[:, None] * nodes[None, :] + 0.5 * (lower + upper)[:, None]),
        np.ravel(half[:, None] * weights[None, :]),
    )


def _decade_edges(
    tau_0: float, *, coordinate: Callable[[float], float], stop: float
) -> list[float]:
    """Panel boundaries at ``tau = 10**k``, mapped into the integration variable.

    ``coordinate`` takes an optical depth to the position in the integration variable at
    which the profile reaches it; the profile is monotonically decreasing away from line
    centre, so descending ``tau`` gives ascending coordinates.
    """
    top = math.floor(math.log10(tau_0))
    levels = (10.0**k for k in range(top, -PANEL_DECADES_BELOW_UNITY - 1, -1))
    edges = [0.0, *(coordinate(level) for level in levels if 0.0 < level < tau_0), stop]
    # Two decades can map to the same coordinate once the profile has underflowed; a
    # zero-width panel contributes nothing but does make the weights degenerate.
    return sorted({edge for edge in edges if 0.0 <= edge <= stop})


def _doppler_escape(tau_0: float) -> float:
    """``(2/sqrt(pi)) integral_0^X exp(-x^2) P(tau_0 exp(-x^2)) dx``.

    The Doppler profile in the reduced frequency ``x = (nu - nu_0)/Delta nu_(1/e)`` is
    ``exp(-x^2)``, so the local optical depth is ``tau_0 exp(-x^2)`` and the normalising
    ``phi dnu`` becomes ``exp(-x^2) dx / sqrt(pi)``. Symmetric, hence the factor of two
    and the half-range.
    """
    x_max = math.sqrt(_DOPPLER_TAIL_EXPONENT + math.log(max(tau_0, 1.0)))
    edges = _decade_edges(
        tau_0, coordinate=lambda tau: math.sqrt(math.log(tau_0 / tau)), stop=x_max
    )
    x, w = _panelled_rule(edges)
    profile = np.exp(-(x**2))
    integral = float(np.dot(w, profile * slab_escape_probability(tau_0 * profile)))
    return 2.0 / math.sqrt(math.pi) * integral


def _lorentz_escape(tau_0: float) -> float:
    """``(2/pi) integral_0^(pi/2) P(tau_0 cos^2 theta) dtheta``.

    The substitution ``x = tan(theta)`` maps the infinite frequency axis of a Lorentzian
    onto a finite interval exactly: ``phi(x)/phi(0) = 1/(1 + x^2) = cos^2 theta``, and the
    normalising ``dx / (pi (1 + x^2))`` becomes ``dtheta / pi``. No tail is truncated,
    which matters for a profile whose wings are what escapes.
    """
    edges = _decade_edges(
        tau_0, coordinate=lambda tau: math.acos(math.sqrt(tau / tau_0)), stop=0.5 * math.pi
    )
    theta, w = _panelled_rule(edges)
    return _LORENTZ_PEAK * float(np.dot(w, slab_escape_probability(tau_0 * np.cos(theta) ** 2)))


def escape_factor(optical_depth: float, *, shape: LineProfileShape) -> float:
    """``Lambda_ul`` — the doc 04 §2.2 radiation-trapping factor.

    Args:
        optical_depth: ``tau_0 = k_0 L``, the optical depth at line centre over the
            trapping path. :func:`line_centre_optical_depth` computes it.
        shape: The line profile the frequency redistribution is taken over.

    Returns:
        The fraction of emitted photons that leave the plasma, between zero and one.
        Exactly one at zero opacity.

    Raises:
        ValueError: If the opacity is negative.
    """
    tau_0 = float(optical_depth)
    if tau_0 < 0.0:
        raise ValueError(f"a line-centre optical depth cannot be negative, got {tau_0}")
    if tau_0 == 0.0:
        # doc 04 §8 V-26 exactly rather than to quadrature accuracy: with no absorbers
        # there is no trapping, and a 1 - 1e-16 here would leak into every ratio.
        return 1.0

    if shape is LineProfileShape.MONOCHROMATIC:
        return float(slab_escape_probability(tau_0))
    if shape is LineProfileShape.DOPPLER:
        return _doppler_escape(tau_0)
    return _lorentz_escape(tau_0)


# ── the opacity ─────────────────────────────────────────────────────────────────


def line_centre_optical_depth(
    *,
    wavelength_nm: float,
    a_ul_per_s: float,
    g_upper: int,
    g_lower: int,
    profile_fwhm_nm: float,
    shape: LineProfileShape,
    lower_density_per_m3: float,
    path_length_m: float,
) -> float:
    """``tau_0 = n_l sigma_0 L`` for one transition.

    The frequency-integrated absorption cross section of a line is
    ``(lambda^2 / 8 pi) (g_u / g_l) A_ul``; at line centre it is that times the
    line-centre value of the normalised profile, which is where the width enters.

    Args:
        wavelength_nm: Transition wavelength.
        a_ul_per_s: Einstein A coefficient — doc 09 §2.2, NIST ASD.
        g_upper: Upper level statistical weight.
        g_lower: Lower level statistical weight — the absorbing level.
        profile_fwhm_nm: Full width at half maximum of the line profile, in wavelength.
        shape: Which profile that width describes.
        lower_density_per_m3: Density of the absorbing lower level. For a resonance line
            this is the ground state; for the doc 02 §6.3 near-infrared lines it is a
            1s metastable or resonant level, and it is a *state variable*.
        path_length_m: The trapping path — doc 02 §3.1's chamber dimension.

    Returns:
        The line-centre optical depth.

    Raises:
        ValueError: If any input is non-positive where it must be positive, or if the
            profile is monochromatic — a delta function has no line-centre value and its
            opacity is not a number.
    """
    if shape is LineProfileShape.MONOCHROMATIC:
        raise ValueError(
            "a monochromatic profile has infinite line-centre absorption, so it has no "
            "optical depth. It exists as a verification limit on escape_factor, which "
            "takes tau_0 directly."
        )
    for name, value in (
        ("wavelength_nm", wavelength_nm),
        ("a_ul_per_s", a_ul_per_s),
        ("profile_fwhm_nm", profile_fwhm_nm),
        ("path_length_m", path_length_m),
    ):
        if not value > 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    if g_upper < 1 or g_lower < 1:
        raise ValueError(f"statistical weights are at least 1, got {g_upper} and {g_lower}")
    if lower_density_per_m3 < 0.0:
        raise ValueError(f"an absorber density cannot be negative, got {lower_density_per_m3}")

    wavelength_m = wavelength_nm * _METRES_PER_NM
    # dnu = c dlambda / lambda^2, the differential of nu = c / lambda taken in magnitude.
    fwhm_hz = _C_M_PER_S * (profile_fwhm_nm * _METRES_PER_NM) / wavelength_m**2
    peak = _DOPPLER_PEAK if shape is LineProfileShape.DOPPLER else _LORENTZ_PEAK

    cross_section_m2 = (
        wavelength_m**2 / (8.0 * math.pi) * (g_upper / g_lower) * a_ul_per_s * (peak / fwhm_hz)
    )
    return lower_density_per_m3 * cross_section_m2 * path_length_m


@dataclass(frozen=True, slots=True)
class TrappedLine:
    """One transition, with everything the escape factor needs except the state.

    Splitting the state out is deliberate: doc 04 §2.3 says escape factors "are recomputed
    as the density profile changes", so the absorber density is an argument and not a
    field. A :class:`TrappedLine` is atomic data plus a width; it is reusable across a
    whole profile and across a whole sweep.

    Attributes:
        wavelength_nm: Transition wavelength.
        a_ul_per_s: Einstein A coefficient.
        g_upper: Upper level statistical weight.
        g_lower: Lower level statistical weight.
        shape: Line profile the width describes.
        profile_fwhm_nm: FWHM of that profile, in wavelength.
    """

    wavelength_nm: float
    a_ul_per_s: float
    g_upper: int
    g_lower: int
    shape: LineProfileShape
    profile_fwhm_nm: float

    def __post_init__(self) -> None:
        # Validate by doing: the opacity function already rejects everything invalid, and
        # a second copy of those checks here is a second copy to keep in step.
        self.optical_depth(lower_density_per_m3=0.0, path_length_m=1.0)

    def optical_depth(self, *, lower_density_per_m3: float, path_length_m: float) -> float:
        """``tau_0`` for this line at a stated absorber density and path."""
        return line_centre_optical_depth(
            wavelength_nm=self.wavelength_nm,
            a_ul_per_s=self.a_ul_per_s,
            g_upper=self.g_upper,
            g_lower=self.g_lower,
            profile_fwhm_nm=self.profile_fwhm_nm,
            shape=self.shape,
            lower_density_per_m3=lower_density_per_m3,
            path_length_m=path_length_m,
        )

    def escape_factor(self, *, lower_density_per_m3: float, path_length_m: float) -> float:
        """``Lambda_ul`` for this line at a stated absorber density and path."""
        return escape_factor(
            self.optical_depth(
                lower_density_per_m3=lower_density_per_m3, path_length_m=path_length_m
            ),
            shape=self.shape,
        )

    def __repr__(self) -> str:
        return (
            f"TrappedLine({self.wavelength_nm} nm, A_ul={self.a_ul_per_s:.3g} /s, "
            f"{self.shape}, FWHM={self.profile_fwhm_nm:.3g} nm)"
        )
