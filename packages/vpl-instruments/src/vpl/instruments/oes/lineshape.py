"""Line profiles — doc 04 §2.4.

Doc 04 §2.4 specifies the profile every line carries:

    natural (Lorentzian) ⊗ Doppler (Gaussian at ``T_g`` or ``T_i``) ⊗ Stark (negligible
    here, doc 01 §4.2) ⊗ Zeeman (if ``B != 0``, §4.5) — combined as a Voigt profile.

Two of those four are here. Stark is negligible by doc 01 §4.2's own assessment and is
omitted; Zeeman is modelled in the LIF channel (doc 04 §3.3), where the 15 deg geometry
and the polarising optics make the pattern an *observable* rather than a nuisance, and
adding a magnetic sublevel decomposition to the OES lines would buy nothing the CR model
can use.

## The Voigt profile

``V(x) = Re[w(z)] / (sigma sqrt(2 pi))``, ``z = (x + i gamma) / (sigma sqrt 2)``, with
``w`` the Faddeeva function from SciPy (doc 08 §2: buy). The widths are carried as FWHM
throughout rather than as ``sigma`` and ``gamma``, because FWHM is what doc 02 §6.2
tabulates for the instrument function and what doc 04 §3.3 tabulates for the broadening
budget, and one convention across the package is worth more than the two lines it saves.

:func:`voigt_fwhm_nm` is Olivero & Longbothum's published closed form. It is a genuine
approximation — 0.02 % — and it is used only where a width is *reported*; nothing computes
a profile from it. The tests measure the FWHM off the profile itself and check it against
the same published expression, which is what makes the citation load-bearing rather than
decorative.

## What is deliberately not modelled

- **No Stark broadening.** doc 01 §4.2 assesses it as negligible at the doc 01 §2.1
  operating point; at ``n_e = 1e17 m^-3`` the Ar I Stark width is of order 1e-7 nm,
  four orders below the Doppler width.
- **No Zeeman splitting.** See above. With the optional 50 G Helmholtz field of doc 02
  §3.2 the splitting is ~70 MHz, which is 10 % of a *LIF* linewidth but 1 % of an OES
  Doppler width at 811 nm and far below the 0.026 nm instrument function.
- **No self-absorption distortion of the emitted profile.** The escape factor of
  :mod:`vpl.instruments.oes.escape` removes the right *number* of photons but returns an
  undistorted profile; a strongly trapped line is in reality self-reversed. This is
  irrelevant for the doc 02 §6.3 lines, which are optically thin or nearly so, and would
  be badly wrong for the 104.8 / 106.7 nm resonance lines if anyone tried to synthesise
  a spectrum of them with this.
- **No hyperfine or isotope structure.** Argon's dominant isotope is even-even with zero
  nuclear spin, so there is none to model for 99.6 % of the gas.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from vpl.core.constants import ATOMIC_MASS, BOLTZMANN, SPEED_OF_LIGHT
from vpl.core.units import Q_, magnitude_in

__all__ = [
    "OLIVERO_GAUSSIAN_COEFFICIENT",
    "OLIVERO_LORENTZ_COEFFICIENT",
    "doppler_fwhm_nm",
    "fwhm_to_gaussian_sigma",
    "natural_fwhm_nm",
    "voigt_fwhm_nm",
    "voigt_profile",
]

type FloatArray = NDArray[np.float64]

# ── constants ───────────────────────────────────────────────────────────────────

_C_M_PER_S: Final[float] = float(magnitude_in(SPEED_OF_LIGHT, "m / s"))
_K_B_J_PER_K: Final[float] = float(magnitude_in(BOLTZMANN, "J / K"))
_ATOMIC_MASS_KG: Final[float] = float(magnitude_in(ATOMIC_MASS, "kg"))

#: Metres in a nanometre — see the note in :mod:`vpl.instruments.oes.escape`.
_METRES_PER_NM: Final[float] = float(magnitude_in(Q_(1.0, "nm"), "m"))

#: ``2 sqrt(2 ln 2)`` — FWHM of a Gaussian in units of its standard deviation.
_FWHM_PER_SIGMA: Final[float] = 2.0 * math.sqrt(2.0 * math.log(2.0))

#: ``sqrt(8 ln 2)`` — the coefficient in ``Delta lambda / lambda = sqrt(8 ln 2 kT / m c^2)``.
_DOPPLER_COEFFICIENT: Final[float] = math.sqrt(8.0 * math.log(2.0))

#: Olivero & Longbothum (1977), *J. Quant. Spectrosc. Radiat. Transfer* **17** 233:
#: ``f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2)``, stated accurate to 0.02 %.
OLIVERO_LORENTZ_COEFFICIENT: Final[float] = 0.5346

#: The other half of the same expression. Note ``0.5346^2 = 0.2858``, not 0.2166; the two
#: coefficients are an independent fit and not related by squaring, which is the mistake
#: this note exists to prevent.
OLIVERO_GAUSSIAN_COEFFICIENT: Final[float] = 0.2166


def _checked_width(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative, got {value}")
    return value


def fwhm_to_gaussian_sigma(fwhm: float) -> float:
    """Standard deviation of a Gaussian of this full width at half maximum."""
    return fwhm / _FWHM_PER_SIGMA


# ── the individual mechanisms ───────────────────────────────────────────────────


def doppler_fwhm_nm(*, wavelength_nm: float, temperature_k: float, mass_u: float) -> float:
    """Thermal Doppler FWHM of a line, in wavelength — doc 04 §2.4.

    ``Delta lambda / lambda = sqrt(8 ln 2 k_B T / (m c^2))``. For argon at 300 K this is
    the ``7.1623e-7 sqrt(T/M)`` coefficient every plasma-spectroscopy text quotes; the
    test asserts that agreement rather than the formula, so a slip in ``k_B`` or in the
    ``8 ln 2`` cannot pass.

    Args:
        wavelength_nm: Transition wavelength.
        temperature_k: Temperature of the *emitters*. For the neutral lines of doc 02
            §6.3 that is ``T_g``; for the Ar II lines it is ``T_i``, and doc 01 §2.1 puts
            those at 300 K and 580 K respectively — not the same line width.
        mass_u: Emitter mass in unified atomic mass units.

    Returns:
        Full width at half maximum, in nanometres.

    Raises:
        ValueError: If any argument is not strictly positive.
    """
    for name, value in (
        ("wavelength_nm", wavelength_nm),
        ("temperature_k", temperature_k),
        ("mass_u", mass_u),
    ):
        if not value > 0.0:
            raise ValueError(f"{name} must be positive, got {value}")

    mass_kg = mass_u * _ATOMIC_MASS_KG
    return (
        wavelength_nm
        * _DOPPLER_COEFFICIENT
        * math.sqrt(_K_B_J_PER_K * temperature_k / (mass_kg * _C_M_PER_S**2))
    )


def natural_fwhm_nm(*, wavelength_nm: float, total_a_per_s: float) -> float:
    """Natural (lifetime) FWHM, in wavelength.

    ``Delta nu = sum_l A_ul / (2 pi)``, converted by ``Delta lambda = lambda^2 Delta nu / c``.

    Args:
        wavelength_nm: Transition wavelength.
        total_a_per_s: **Total** transition probability out of the upper level, summed
            over every downward channel — not the ``A_ul`` of the line being modelled.
            The lifetime is set by all of them, and using one channel's ``A_ul`` here
            under-estimates the width by the branching ratio.

    Returns:
        Full width at half maximum, in nanometres.

    Raises:
        ValueError: If the wavelength is not positive or the decay rate is negative.
    """
    if not wavelength_nm > 0.0:
        raise ValueError(f"wavelength_nm must be positive, got {wavelength_nm}")
    _checked_width(total_a_per_s, name="total_a_per_s")

    wavelength_m = wavelength_nm * _METRES_PER_NM
    fwhm_hz = total_a_per_s / (2.0 * math.pi)
    return fwhm_hz * wavelength_m**2 / _C_M_PER_S / _METRES_PER_NM


# ── the combination ─────────────────────────────────────────────────────────────


def voigt_profile(offset: ArrayLike, *, gaussian_fwhm: float, lorentz_fwhm: float) -> FloatArray:
    """The area-normalised Voigt profile at an offset from line centre.

    Both limits are taken explicitly rather than left to the Faddeeva function: a zero
    Gaussian width divides by zero inside ``z``, and the result would arrive as a nan
    inside a synthetic spectrum rather than as an error here.

    Args:
        offset: Distance from line centre. Units are the caller's, and are whatever the
            two widths are in — the profile is a density in that variable, so the return
            carries its inverse.
        gaussian_fwhm: FWHM of the Gaussian component. Zero gives a pure Lorentzian.
        lorentz_fwhm: FWHM of the Lorentzian component. Zero gives a pure Gaussian.

    Returns:
        The profile, integrating to one over the offset variable.

    Raises:
        ValueError: If either width is negative or non-finite, or if both are zero — a
            line with no width is a delta function and has no sampled representation.
    """
    _checked_width(gaussian_fwhm, name="gaussian_fwhm")
    _checked_width(lorentz_fwhm, name="lorentz_fwhm")
    if gaussian_fwhm == 0.0 and lorentz_fwhm == 0.0:
        raise ValueError(
            "a line profile needs at least one non-zero width; with both zero it is a "
            "delta function, which has no value to return at line centre"
        )

    x = np.asarray(offset, dtype=np.float64)
    half_lorentz = 0.5 * lorentz_fwhm

    if gaussian_fwhm == 0.0:
        return np.asarray(half_lorentz / (math.pi * (x**2 + half_lorentz**2)))

    sigma = fwhm_to_gaussian_sigma(gaussian_fwhm)
    z = (x + 1j * half_lorentz) / (sigma * math.sqrt(2.0))
    return np.asarray(np.real(special.wofz(z)) / (sigma * math.sqrt(2.0 * math.pi)))


def voigt_fwhm_nm(*, gaussian_fwhm_nm: float, lorentz_fwhm_nm: float) -> float:
    """FWHM of the Voigt profile — Olivero & Longbothum (1977).

    ``f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2)``, accurate to 0.02 % over the whole
    range of the ratio. Reported, never used to build a profile: everything that needs
    the shape calls :func:`voigt_profile`, so this approximation can only ever appear in
    a summary and never inside a spectrum.

    Args:
        gaussian_fwhm_nm: FWHM of the Gaussian component.
        lorentz_fwhm_nm: FWHM of the Lorentzian component.

    Returns:
        The combined FWHM, in the same units.

    Raises:
        ValueError: If either width is negative or non-finite.
    """
    _checked_width(gaussian_fwhm_nm, name="gaussian_fwhm_nm")
    _checked_width(lorentz_fwhm_nm, name="lorentz_fwhm_nm")
    return OLIVERO_LORENTZ_COEFFICIENT * lorentz_fwhm_nm + math.sqrt(
        OLIVERO_GAUSSIAN_COEFFICIENT * lorentz_fwhm_nm**2 + gaussian_fwhm_nm**2
    )
