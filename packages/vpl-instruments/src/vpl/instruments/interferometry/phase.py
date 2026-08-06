"""Refractive index, line-integrated phase, and the doc 01 IF-6 detection floor.

Doc 04 §5.1 fixes the physics:

    n_refr  =  sqrt(1 - n_e/n_c)  ~=  1 - n_e/(2 n_c)  ,   n_c = eps0 m_e omega^2 / e^2
    Delta phi  =  r_e lambda INTEGRAL n_e dl              r_e = classical electron radius

and doc 04 §5.2 lists what is modelled beyond that ideal line; this module implements the
refractive index, the electron phase, the neutral-gas correction and the IF-6 floor
derivation. The state-to-chord binding (interpolating a solver's ``n_e(z)`` onto the eight
fixed chord positions of doc 02 §8.2 IF-G1) lives in :mod:`vpl.instruments.interferometry.
instrument`, not here — this module knows about densities and lengths, not about a
:class:`~vpl.core.state.PlasmaState`.

## The honesty point (doc 02 §8.3), stated once, loudly

A phase shift is a **line integral along the chord**, not a local density: this channel
has "no intrinsic z resolution — it cannot see the sheath structure at all" (doc 02 §8.3).
Recovering a density profile from a set of chord phases is itself an inverse problem
(Abel/tomographic inversion, doc 01 IF-5), and it is not attempted here.
:func:`electron_phase_shift_rad` and :func:`net_phase_shift_rad` therefore return a phase
in radians, never a density — the instrument module returns one phase per chord, and
nothing downstream is allowed to treat that as a point measurement of ``n_e``.

## The detection floor, worked twice, and the two documents disagree on the rounding

Doc 01 §5.4 (IF-6) computes the floor over the 0.1 m chord assumed before the chamber
diameter was fixed: 3.3e16 m^-3. Doc 02 §8.2 revises it once §3.1 sets the chamber diameter
to 400 mm: 8.4e15 m^-3, "a design decision made *because* of a requirements calculation".
Doc 02 is the later and binding statement, so :data:`CHAMBER_DIAMETER_M` (not the 0.1 m
figure) is the default this module's instrument uses. Both numbers are reproduced exactly
by :func:`detection_floor_n_e_per_m3` (see ``test_interferometry_phase.py``), and the
module is honest about a wrinkle worth flagging: ``4 * 8.4e15 = 3.36e16``, not doc 01's
quoted ``3.3e16`` — the two documents round their headline numbers differently, and this
module does not paper over the 2% discrepancy by picking whichever constant makes the
arithmetic look cleaner.

## The neutral-gas term, and a finding that overturns the intuitive guess

Doc 04 §5.2: "Ar has a non-zero polarisability; the neutral term partially cancels the
electron term and is included." The polarisability used is the **static** (zero-frequency)
value of Dalgarno & Kingston (1960) rather than an extrapolation of Peck & Fisher's (1964)
visible-light Sellmeier fit, which was measured only out to 2.06 micron — stretching it to
10.6 micron would be extrapolating a dispersion curve nearly 5x past its own validated
range. Using the static polarisability instead is the better-justified approximation:
argon's first strong resonance is in the vacuum ultraviolet (~11.8 eV), so the entire
visible-through-far-infrared window, including 10.6 micron, sits deep in the
non-resonant, near-static regime, and the two sources agree to a few percent where they
overlap (see ``test_matches_the_dilute_gas_refractivity_at_stp_within_the_literature_range``).

The intuitive guess is that this correction is negligible. **It is not**, at the doc's own
reference point: at RP-1 (n_e = 1e17 m^-3, 5 mTorr argon, 10.6 micron, 0.4 m chord) the
neutral term is roughly a third of the electron term (see
``test_the_neutral_contribution_at_5_mtorr_is_not_negligible``). The reason is that
electron-phase sensitivity grows with wavelength (``r_e lambda``) while neutral-gas phase
sensitivity falls with it (``1/lambda``, since (n_gas - 1) is essentially
wavelength-independent in this transparent band) — CO2's long wavelength, chosen
specifically to make the electron term detectable at all (doc 01 §5.4), simultaneously
makes the neutral term relatively *more* important than it would be for a visible-light
interferometer at the same fill pressure, not less. This is exactly why doc 04 §5.2 lists
the neutral term as included rather than assumed away, and this module makes the
finding a checked fact rather than a comment.

## Citations

- I. H. Hutchinson, *Principles of Plasma Diagnostics*, 2nd ed., Cambridge University
  Press (2002), ch. 6 — the refractive-index/interferometry derivation this module
  follows (no DOI: a book).
- A. Dalgarno and A. E. Kingston, "The refractive indices and Verdet constants of the
  inert gases", Proc. R. Soc. Lond. A 259 (1960) 424 — the argon static polarisability,
  11.080 in units of the Bohr radius cubed.
- E. R. Peck and D. J. Fisher, "Dispersion of argon", J. Opt. Soc. Am. 54 (1964) 1362 —
  corroborates the near-static magnitude of argon's refractivity through the visible and
  near infrared, motivating the extrapolation to 10.6 micron argued above.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.constants import (
    CLASSICAL_ELECTRON_RADIUS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
)
from vpl.core.units import magnitude_in

__all__ = [
    "ARGON_STATIC_POLARIZABILITY_VOLUME_M3",
    "CHAMBER_DIAMETER_M",
    "CHORD_SPACING_M",
    "CO2_WAVELENGTH_M",
    "HETERODYNE_OFFSET_HZ",
    "N_CHORDS",
    "PHASE_RESOLUTION_RAD",
    "critical_density_per_m3",
    "detection_floor_n_e_per_m3",
    "electron_phase_shift_rad",
    "net_phase_shift_rad",
    "neutral_phase_shift_rad",
    "refractive_index",
    "refractive_index_linearized",
]

type FloatArray = NDArray[np.float64]

# ── constants pulled from the shared registry once, as plain floats for hot loops ──
# (doc 08 §5: cross a module boundary once, then work in SI magnitudes.)

_CLASSICAL_ELECTRON_RADIUS_M: Final[float] = float(magnitude_in(CLASSICAL_ELECTRON_RADIUS, "m"))
_VACUUM_PERMITTIVITY_F_PER_M: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))
_ELECTRON_MASS_KG: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))
_ELEMENTARY_CHARGE_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_SPEED_OF_LIGHT_M_PER_S: Final[float] = float(magnitude_in(SPEED_OF_LIGHT, "m/s"))

# ── the doc 02 §8.2 hardware specification ──────────────────────────────────────

#: IF-L1: CO2 laser line. Doc 01 §5.4 rejects 633 nm HeNe on sensitivity grounds — a HeNe
#: interferometer is blind at RP-1 by a factor of about six.
CO2_WAVELENGTH_M: Final[float] = 10.6e-6

#: IF-P2 / doc 02 §3.1: the full 400 mm chamber diameter, used as the default chord
#: length. Widening the chamber from the 100 mm assumed in doc 01 §5.4 is "a design
#: decision made because of a requirements calculation" — see the module docstring.
CHAMBER_DIAMETER_M: Final[float] = 0.400

#: IF-P1: the heterodyne phase resolution, 0.1 mrad.
PHASE_RESOLUTION_RAD: Final[float] = 1.0e-4

#: IF-G1: eight parallel chords.
N_CHORDS: Final[int] = 8

#: IF-G1: 5 mm spacing between chords in z.
CHORD_SPACING_M: Final[float] = 5.0e-3

#: IF-L2: the AOM heterodyne offset. Not used numerically by this module — it is what
#: makes the detection heterodyne (phase-sign recovery, drift rejection); see
#: :mod:`vpl.instruments.interferometry.instrument`'s module docstring for where that
#: shows up in the model — but it is named here as part of the doc 02 §8.2 specification
#: table this module implements.
HETERODYNE_OFFSET_HZ: Final[float] = 40.0e6

# ── the neutral-gas term ────────────────────────────────────────────────────────

#: Static (zero-frequency) polarisability *volume* of argon: 11.080 in units of the Bohr
#: radius cubed (Dalgarno & Kingston 1960, Table 2), converted with
#: a0 = 5.29177210903e-11 m (CODATA), a0^3 = 1.481847e-31 m^3.
#: 11.080 * 1.481847e-31 = 1.6419e-30 m^3.
ARGON_STATIC_POLARIZABILITY_VOLUME_M3: Final[float] = 1.6419e-30


def _positive(value: float, *, name: str) -> float:
    if not value > 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def critical_density_per_m3(wavelength_m: float) -> float:
    """The cutoff density ``n_c = eps0 m_e omega^2 / e^2`` at ``wavelength_m`` — doc 04 §5.1.

    Above this density the probe beam does not propagate at all; it is the natural upper
    bound on everything else this module computes.
    """
    _positive(wavelength_m, name="wavelength")
    omega = 2.0 * math.pi * _SPEED_OF_LIGHT_M_PER_S / wavelength_m
    return _VACUUM_PERMITTIVITY_F_PER_M * _ELECTRON_MASS_KG * omega**2 / _ELEMENTARY_CHARGE_C**2


def refractive_index(n_e_per_m3: float | FloatArray, wavelength_m: float) -> FloatArray:
    """The exact ``n_refr = sqrt(1 - n_e/n_c)`` — doc 04 §5.1.

    Raises:
        ValueError: If any density is at or above the critical density. The plasma is
            opaque there and a real beam would not transmit; returning ``nan`` from a
            negative square root would be a silent wrong answer rather than a refusal.
    """
    n_c = critical_density_per_m3(wavelength_m)
    n_e = np.atleast_1d(np.asarray(n_e_per_m3, dtype=np.float64))
    if np.any(n_e >= n_c):
        raise ValueError(
            f"n_e reaches or exceeds the critical density {n_c:.3e} m^-3 at "
            f"{wavelength_m} m wavelength; the beam does not propagate there"
        )
    return np.sqrt(1.0 - n_e / n_c)


def refractive_index_linearized(n_e_per_m3: float | FloatArray, wavelength_m: float) -> FloatArray:
    """The doc 04 §5.1 small-parameter expansion ``n_refr ~= 1 - n_e/(2 n_c)``.

    Valid to ``O((n_e/n_c)^2)``; see ``test_interferometry_phase.py`` for the check that
    the two forms agree to better than 1e-12 at RP-1, where ``n_e/n_c ~ 1e-8``.
    """
    n_c = critical_density_per_m3(wavelength_m)
    n_e = np.atleast_1d(np.asarray(n_e_per_m3, dtype=np.float64))
    return 1.0 - n_e / (2.0 * n_c)


def electron_phase_shift_rad(
    n_e_per_m3: float | FloatArray, wavelength_m: float, chord_length_m: float
) -> FloatArray:
    """``Delta phi_e = r_e lambda n_e L`` — doc 04 §5.1, doc 01 §5.4.

    The line integral ``INTEGRAL n_e dl`` collapses to ``n_e L`` under the transverse
    homogeneity assumption documented in
    :mod:`vpl.instruments.interferometry.instrument`: the plasma solver this project
    carries is one-dimensional in the sheath-normal coordinate ``z`` and has no
    information about how density varies along the chord's own axis, so it is treated as
    uniform over that path.
    """
    _positive(chord_length_m, name="chord_length_m")
    n_e = np.atleast_1d(np.asarray(n_e_per_m3, dtype=np.float64))
    return _CLASSICAL_ELECTRON_RADIUS_M * wavelength_m * n_e * chord_length_m


def neutral_phase_shift_rad(
    n_neutral_per_m3: float, wavelength_m: float, chord_length_m: float
) -> float:
    """The doc 04 §5.2 neutral-gas correction, positive-valued.

    Dilute-gas (Clausius-Mossotti) limit: ``n_gas - 1 = 2 pi N alpha'``, with ``alpha'``
    the polarisability *volume*; then ``Delta phi = (2 pi / lambda)(n_gas - 1) L``. Ar's
    refractive index exceeds 1 (ordinary, phase-delaying dielectric behaviour) where the
    plasma's is below 1 (phase-advancing); see :func:`net_phase_shift_rad` for how the two
    combine, and the module docstring for why the resulting correction is not small at
    doc 01's own reference point.
    """
    _positive(chord_length_m, name="chord_length_m")
    if n_neutral_per_m3 < 0.0:
        raise ValueError(f"a neutral density cannot be negative, got {n_neutral_per_m3}")
    refractivity = 2.0 * math.pi * n_neutral_per_m3 * ARGON_STATIC_POLARIZABILITY_VOLUME_M3
    return (2.0 * math.pi / wavelength_m) * refractivity * chord_length_m


def net_phase_shift_rad(
    *,
    n_e_per_m3: float | FloatArray,
    n_neutral_per_m3: float,
    wavelength_m: float,
    chord_length_m: float,
) -> FloatArray:
    """``electron_phase_shift_rad - neutral_phase_shift_rad`` — the doc 04 §5.2 net phase.

    This is the quantity the instrument's ``forward``/``observe`` return: one value per
    chord, in radians, never a density. See the module docstring's "honesty point".
    """
    electron = electron_phase_shift_rad(n_e_per_m3, wavelength_m, chord_length_m)
    neutral = neutral_phase_shift_rad(n_neutral_per_m3, wavelength_m, chord_length_m)
    return electron - neutral


def detection_floor_n_e_per_m3(
    *, phase_resolution_rad: float, wavelength_m: float, chord_length_m: float
) -> float:
    """The doc 01 IF-6 gate: the smallest ``n_e`` this channel can distinguish from zero.

    ``floor = phase_resolution / (r_e lambda L)`` — the inverse of
    :func:`electron_phase_shift_rad`'s per-density-length coefficient. Reproduces doc 02
    §8.2's 8.4e15 m^-3 over the 400 mm chord and doc 01 §5.4's 3.3e16 m^-3 over the 0.1 m
    chord assumed before the chamber was widened; see the module docstring for the
    residual rounding discrepancy between the two.
    """
    _positive(phase_resolution_rad, name="phase_resolution_rad")
    _positive(chord_length_m, name="chord_length_m")
    return phase_resolution_rad / (_CLASSICAL_ELECTRON_RADIUS_M * wavelength_m * chord_length_m)
