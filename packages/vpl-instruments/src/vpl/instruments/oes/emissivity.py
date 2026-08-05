"""Line emissivity and the chord integral — doc 04 §2.1, §2.4.

Doc 04 §2.1, verbatim::

    eps_ul(r, t)  =  (1 / 4pi) . n_u(r, t) . A_ul . h nu_ul            [W m^-3 sr^-1]

    `A_ul` from NIST ASD (doc 09). The physics problem is `n_u`, the upper-state
    population.

The physics problem is solved in :mod:`vpl.instruments.oes.cr`. This module is the last
line of arithmetic and the one place where a wavelength in nanometres meets a Planck
constant in joule-seconds, which is why it exists as a named function with a test on it
rather than as an expression inside the instrument.

## Emitted against escaping

Doc 04 §2.1 writes the emissivity *without* an escape factor, and that is correct: it is
the rate at which the level radiates. What leaves the plasma is smaller by ``Lambda_ul``.
:class:`LineEmission` carries both, named differently, because for the Ar I resonance lines
they differ by three orders of magnitude and a model that used one where it meant the other
would be wrong by that much with nothing to show for it.

For the doc 02 §6.3 near-infrared lines the two are within a percent of each other, which
is exactly why that confusion could survive a review of the observed line set and then
detonate on the resonance lines.

## The chord

:func:`chord_radiance` multiplies by a path length. That is the whole optical transport
model in this package, and it is a placeholder for doc 04 §6.1's ray tracing, which doc 04
§6.3 buys Raysect for and which lives in ``vpl-optics``.

What it assumes, and what each assumption costs:

- **The emissivity is uniform along the chord.** Doc 02 §6.1 orients the slit along the
  sheath normal, so the line of sight runs perpendicular to the gradient and this is a good
  approximation *for the sheath structure*. It is a poor one for the radial profile of the
  discharge, which falls off towards the chamber wall; a Bessel-like radial profile over a
  400 mm chord would give roughly 60 % of the uniform answer, so **absolute** radiances
  from this are high by a factor of order 1.5 and line *ratios* are unaffected.
- **No optical depth along the chord.** The escape factor already removes trapped photons
  from the emission; applying an additional ``exp(-tau)`` along the sight line would
  double-count. For an optically thin line neither matters.
- **No collection optics.** No solid angle, no vignetting, no aberration, no depth of
  field. Doc 04 §6.2 is emphatic that the measurement volume is an integral rather than a
  point and that treating it otherwise "is a **systematic bias, not a resolution limit**";
  that integral is ``vpl-optics``'s and is not approximated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from vpl.core.constants import PLANCK, SPEED_OF_LIGHT
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.oes.cr import LevelPopulations

__all__ = [
    "LineEmission",
    "chord_radiance",
    "emission_spectrum",
    "line_emissivity",
]

#: ``h c`` in joule-metres. One product, unwrapped once, from CODATA via
#: :mod:`vpl.core.constants` — doc 09 §2.5 calls a hand-entered fundamental constant "a
#: plausible, undetectable and catastrophic error", and this one scales every radiance in
#: the package.
_HC_J_M: Final[float] = float(magnitude_in(PLANCK * SPEED_OF_LIGHT, "J * m"))

#: Metres in a nanometre — see the note in :mod:`vpl.instruments.oes.escape`.
_METRES_PER_NM: Final[float] = float(magnitude_in(Q_(1.0, "nm"), "m"))


def line_emissivity(
    *, upper_density_per_m3: float, a_ul_per_s: float, wavelength_nm: float
) -> float:
    """``eps_ul = n_u A_ul h nu_ul / 4 pi`` — doc 04 §2.1, in W m^-3 sr^-1.

    Args:
        upper_density_per_m3: ``n_u``, from the CR solve.
        a_ul_per_s: ``A_ul``, from NIST ASD (doc 09 §2.2).
        wavelength_nm: ``lambda_ul``, which fixes ``h nu_ul``.

    Returns:
        The volumetric emission coefficient, optically thin.

    Raises:
        ValueError: If the population is negative or either atomic quantity is not
            positive.
    """
    if upper_density_per_m3 < 0.0:
        raise ValueError(f"a population cannot be negative, got {upper_density_per_m3}")
    if not a_ul_per_s > 0.0:
        raise ValueError(f"A_ul must be positive, got {a_ul_per_s} /s")
    if not wavelength_nm > 0.0:
        raise ValueError(f"wavelength must be positive, got {wavelength_nm} nm")

    photon_energy_j = _HC_J_M / (wavelength_nm * _METRES_PER_NM)
    return upper_density_per_m3 * a_ul_per_s * photon_energy_j / (4.0 * math.pi)


@dataclass(frozen=True, slots=True)
class LineEmission:
    """One line's emission at one point in the plasma.

    Attributes:
        upper: Label of the emitting level.
        lower: Label of the terminating level.
        wavelength_nm: Transition wavelength.
        emissivity_w_per_m3_sr: Doc 04 §2.1's ``eps_ul``, optically thin.
        escape_factor: The ``Lambda_ul`` the CR solve used — doc 04 §2.3.
    """

    upper: str
    lower: str
    wavelength_nm: float
    emissivity_w_per_m3_sr: float
    escape_factor: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.escape_factor <= 1.0:
            raise ValueError(
                f"{self.upper} -> {self.lower}: an escape factor is a probability, got "
                f"{self.escape_factor}"
            )

    @property
    def escaping_emissivity_w_per_m3_sr(self) -> float:
        """What actually leaves the plasma: ``eps_ul Lambda_ul``."""
        return self.emissivity_w_per_m3_sr * self.escape_factor

    @property
    def photon_energy_j(self) -> float:
        return _HC_J_M / (self.wavelength_nm * _METRES_PER_NM)

    @property
    def photon_emissivity_per_m3_s_sr(self) -> float:
        """The same emission counted in photons — what a photon-counting detector sees."""
        return self.emissivity_w_per_m3_sr / self.photon_energy_j

    def __repr__(self) -> str:
        return (
            f"LineEmission({self.upper!r} -> {self.lower!r}, {self.wavelength_nm} nm, "
            f"{self.emissivity_w_per_m3_sr:.3g} W/m3/sr, Lambda={self.escape_factor:.3g})"
        )


def emission_spectrum(populations: LevelPopulations) -> tuple[LineEmission, ...]:
    """Every radiative channel's emission, in the level system's own channel order.

    Order is the level system's, which is fixed at construction, so two runs of the same
    manifest produce the same sequence — doc 00 E3.
    """
    return tuple(
        LineEmission(
            upper=channel.upper,
            lower=channel.lower,
            wavelength_nm=channel.wavelength_nm,
            emissivity_w_per_m3_sr=line_emissivity(
                upper_density_per_m3=populations[channel.upper],
                a_ul_per_s=channel.a_ul_per_s,
                wavelength_nm=channel.wavelength_nm,
            ),
            escape_factor=populations.escape_factors[channel.key],
        )
        for channel in populations.system.radiative
    )


def chord_radiance(emission: LineEmission, *, path_length_m: float) -> float:
    """Spectral radiance from a uniform chord, in W m^-2 sr^-1.

    ``L = eps_ul Lambda_ul . l``. See the module docstring for what this stands in for and
    what it costs.

    Args:
        emission: The line, from :func:`emission_spectrum`.
        path_length_m: Chord length — doc 02 §3.1's chamber dimension.

    Returns:
        Line-integrated radiance. Not per nanometre: the line's own profile is applied by
        :meth:`~vpl.instruments.oes.spectrograph.Spectrograph.synthesise`, and dividing by
        a width here would mean doing it twice.

    Raises:
        ValueError: If the path length is not positive.
    """
    if not path_length_m > 0.0:
        raise ValueError(f"a chord length must be positive, got {path_length_m} m")
    return emission.escaping_emissivity_w_per_m3_sr * path_length_m
