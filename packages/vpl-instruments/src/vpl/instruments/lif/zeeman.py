"""The Zeeman pattern of the probe transition — doc 04 §3.3.

Doc 04 §3.3 puts this in the "NOT negligible" row of the broadening budget and says why:

    With the optional Helmholtz field on (doc 02 §3.2), Zeeman splitting is 10 % of the
    thermal Doppler width. Ignoring it would bias the inferred ion temperature and distort
    the IVDF shape. The framework therefore models the full Zeeman pattern (pi and sigma
    components with their polarisation dependence) whenever ``B != 0``.

A single line therefore becomes a weighted comb::

    dnu(m, q)  =  (mu_B B / h) [ g_u (m + q) - g_l m ]        q in {-1, 0, +1}

with ``g`` from the Lande formula and the weights from the electric-dipole strengths. The
comb is what :mod:`vpl.instruments.lif.scan` convolves the velocity distribution against,
so an unsplit model does not merely lose a small effect: it attributes 62.5 MHz of RMS
instrumental structure to ion thermal motion, which is a ``T_i`` bias.

**A number in doc 04 §3.3 needs qualifying.** Its "~70 MHz at 50 G" is the *scale*
``mu_B B / h`` (69.98 MHz), which :func:`zeeman_scale` returns. The pattern is wider: the
extreme component of the doc 02 §5.3 assignment sits at 110 MHz, because
``|g_u m_u - g_l m_l|`` peaks at ``m = -3/2 -> -5/2`` where the two terms add rather than
at the largest ``|m|`` where they subtract. The document's *conclusion* is unaffected —
:func:`rms_spread` gives 62.5 MHz, which is the 10 % of the 734 MHz Doppler width doc 04
§3.3 quotes — but a scan window sized to +/-70 MHz would clip the pattern.

## The observable the polarisation buys

Doc 04 §3.3 finishes: "the LIF collection optics include a polariser so the components can
be separated — turning a nuisance into an additional observable." :class:`PumpPolarisation`
is the driving half of that. A pi-polarised pump (field along ``B``) drives only
``Delta m = 0``, whose shifts carry ``g_u - g_l`` and are therefore small; a sigma pump
drives ``Delta m = +/-1``, whose shifts carry ``g_u`` itself and are three times larger.
Two scans of the same distribution with different instrumental broadening constrain ``T_i``
and ``B`` separately, which one scan cannot.

## Stated simplifications, with their consequence

- **The lower sublevels are taken to be equally populated.** True in the absence of optical
  pumping. A strong pump depletes the sublevels it couples to most strongly, which
  redistributes the weights below but never the positions. Exact in the weak-pump limit;
  see :mod:`vpl.instruments.lif.rates` for the bound at large ``S``.
- **No coherences between sublevels**, so no Hanle effect and no level crossings. The rate
  equations of doc 04 §3.1 are themselves an incoherent model, so this is the same
  approximation the specification already makes, not an additional one.
- **LS coupling** for the Lande factors. Real Ar II is intermediate-coupled; measured
  ``g`` values differ from the LS ones by a few percent, which moves the 110 MHz extreme
  by a few MHz — under 0.5 % of the Doppler width. The level quantum numbers are
  registry entries, so a measured ``g`` replaces the formula by editing one file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from vpl.core.units import Q_, Quantity, ScalarQuantity, magnitude_in
from vpl.instruments.lif.transition import Level

__all__ = [
    "PumpPolarisation",
    "ZeemanComponent",
    "lande_g_factor",
    "rms_spread",
    "unsplit_pattern",
    "zeeman_pattern",
    "zeeman_scale",
]

#: ``mu_B / h`` in Hz/T — 13.996 GHz/T.
#:
#: Assembled from CODATA rather than looked up, because :mod:`vpl.core.constants` does not
#: export the Bohr magneton and doc 09 §2.5 forbids typing one in. ``mu_B = e hbar / 2 m_e``,
#: so ``mu_B / h = e / (4 pi m_e)``.
_BOHR_MAGNETON_OVER_H_HZ_PER_T: float = float(magnitude_in(ELEMENTARY_CHARGE, "C")) / (
    4.0 * math.pi * float(magnitude_in(ELECTRON_MASS, "kg"))
)

#: Guard for the ``J = 0`` case in the Lande formula, whose denominator vanishes there.
#: A ``J = 0`` level has no magnetic structure at all, so ``g`` is undefined rather than
#: infinite, and the sensible value to return is the one that produces no shift.
_LANDE_G_FOR_ZERO_J: float = 0.0


class PumpPolarisation(StrEnum):
    """Which ``Delta m`` the pump laser drives — doc 04 §3.3's polarisation dependence.

    The reference axis is the magnetic field. A linear polarisation along ``B`` drives
    ``Delta m = 0``; circular polarisations about ``B`` drive ``Delta m = +/-1``; an
    unpolarised or depolarised beam drives all three.
    """

    #: Linear along ``B``. Shifts scale with ``g_u - g_l``.
    PI = "pi"
    #: Circular, ``Delta m = +1``. Shifts scale with ``g_u`` about a ``g_u``-sized offset.
    SIGMA_PLUS = "sigma_plus"
    #: Circular, ``Delta m = -1``. The mirror image of :attr:`SIGMA_PLUS`.
    SIGMA_MINUS = "sigma_minus"
    #: All three, equally weighted — an unpolarised pump, and the ``B = 0`` default.
    ISOTROPIC = "isotropic"

    @property
    def driven_delta_m(self) -> tuple[int, ...]:
        """The ``Delta m`` values this polarisation couples to."""
        if self is PumpPolarisation.PI:
            return (0,)
        if self is PumpPolarisation.SIGMA_PLUS:
            return (1,)
        if self is PumpPolarisation.SIGMA_MINUS:
            return (-1,)
        return (-1, 0, 1)


@dataclass(frozen=True, slots=True)
class ZeemanComponent:
    """One ``m -> m + Delta m`` sub-transition of the split line.

    Attributes:
        m_lower: Magnetic quantum number of the lower sublevel the component starts from.
        delta_m: ``-1``, ``0`` or ``+1``.
        shift: Frequency offset from the unsplit line centre. Signed.
        weight: Fraction of the total driven absorption this component carries. The
            weights of a pattern sum to one, which is what makes the ``B -> 0`` limit
            reduce **exactly** to the unsplit line rather than to within a tolerance.
    """

    m_lower: float
    delta_m: int
    shift: Quantity
    weight: float

    @property
    def m_upper(self) -> float:
        return self.m_lower + self.delta_m

    @property
    def polarisation(self) -> PumpPolarisation:
        """Which pump polarisation drives this component."""
        if self.delta_m == 0:
            return PumpPolarisation.PI
        return PumpPolarisation.SIGMA_PLUS if self.delta_m > 0 else PumpPolarisation.SIGMA_MINUS

    @property
    def shift_hz(self) -> float:
        return float(magnitude_in(self.shift, "Hz"))

    def __repr__(self) -> str:
        return (
            f"ZeemanComponent({self.polarisation.value}, m={self.m_lower:+g}"
            f"->{self.m_upper:+g}, {self.shift:.4g~P}, w={self.weight:.4f})"
        )


def lande_g_factor(level: Level) -> float:
    """``g_J`` in LS coupling — the standard Lande formula.

    ``g = 1 + [J(J+1) + S(S+1) - L(L+1)] / [2 J(J+1)]``, which returns 2 for a spin-only
    level and 1 for an orbit-only one. Both limits are pinned in ``test_lif_zeeman.py``,
    because a transcription error in this expression is invisible: it still produces a
    plausible pattern of a plausible width.
    """
    if level.j == 0.0:
        return _LANDE_G_FOR_ZERO_J

    j_term = level.j * (level.j + 1.0)
    spin_term = level.spin * (level.spin + 1.0)
    orbital_term = level.orbital * (level.orbital + 1.0)
    return 1.0 + (j_term + spin_term - orbital_term) / (2.0 * j_term)


def zeeman_scale(magnetic_field: Quantity) -> ScalarQuantity:
    """``mu_B B / h`` — the frequency doc 04 §3.3 tabulates as "~70 MHz" at 50 G.

    The *scale* of the pattern, not its extent. Every component shift is this times a
    number of order one, and for the doc 02 §5.3 assignment those numbers run from 0.07
    to 1.57 — so quoting the scale as the splitting understates the widest component by
    57 %. Exposed separately so that the two are never confused for one another.
    """
    return Q_(_BOHR_MAGNETON_OVER_H_HZ_PER_T * float(magnitude_in(magnetic_field, "T")), "Hz")


def rms_spread(components: Sequence[ZeemanComponent]) -> ScalarQuantity:
    """Intensity-weighted RMS shift of a pattern about its own centroid.

    This is the quantity that adds in quadrature to the Doppler width, and therefore the
    one that biases a fitted ``T_i`` if the pattern is ignored — which is exactly the
    failure doc 04 §3.3 warns about. At 50 G it is 62.5 MHz against the 734 MHz Doppler
    width, reproducing the document's "10 %".
    """
    total_weight = sum(component.weight for component in components)
    if total_weight <= 0.0:
        raise ValueError("a Zeeman pattern with no weight has no spread")

    centroid = sum(c.weight * c.shift_hz for c in components) / total_weight
    variance = sum(c.weight * (c.shift_hz - centroid) ** 2 for c in components) / total_weight
    return Q_(math.sqrt(variance), "Hz")


def _relative_strength(*, j_lower: float, j_upper: float, m_lower: float, delta_m: int) -> float:
    """Relative electric-dipole strength of one ``m -> m + Delta m`` component.

    The standard Zeeman intensity table, in the three branches ``J_u = J_l + 1``,
    ``J_u = J_l`` and ``J_u = J_l - 1``. Every expression returns zero for a component the
    selection rules forbid, so no separate bounds check is needed — and the sum over the
    three ``Delta m`` is independent of ``m``, which is the theorem
    ``test_lif_zeeman.py`` uses to check the table rather than trusting the transcription.
    """
    m = m_lower
    j = j_lower

    if j_upper > j_lower:
        if delta_m == 1:
            return (j + m + 1.0) * (j + m + 2.0)
        if delta_m == 0:
            return 2.0 * ((j + 1.0) ** 2 - m**2)
        return (j - m + 1.0) * (j - m + 2.0)

    if j_upper == j_lower:
        if delta_m == 1:
            return (j - m) * (j + m + 1.0)
        if delta_m == 0:
            return 2.0 * m**2
        return (j + m) * (j - m + 1.0)

    if delta_m == 1:
        return (j - m) * (j - m - 1.0)
    if delta_m == 0:
        return 2.0 * (j**2 - m**2)
    return (j + m) * (j + m - 1.0)


def _check_dipole_allowed(lower: Level, upper: Level) -> None:
    """Reject a level pair no electric-dipole photon can connect."""
    if abs(upper.j - lower.j) > 1.0 or (lower.j == 0.0 and upper.j == 0.0):
        raise ValueError(
            f"J = {lower.j:g} -> J = {upper.j:g} is not an electric-dipole transition "
            "(Delta J in {-1, 0, +1}, and 0 -> 0 forbidden). The doc 02 §5.3 scheme is a "
            "dipole transition, so a pair reaching here means the registered level "
            "designations and the registered wavelengths describe different schemes."
        )


def zeeman_pattern(
    *,
    lower: Level,
    upper: Level,
    magnetic_field: Quantity,
    polarisation: PumpPolarisation = PumpPolarisation.ISOTROPIC,
) -> tuple[ZeemanComponent, ...]:
    """The split line as a weighted comb of components — doc 04 §3.3.

    Args:
        lower: The pumped level.
        upper: The laser-coupled level.
        magnetic_field: Field magnitude. ``0`` is legal and returns the unsplit line as a
            comb whose components all sit at zero shift, so callers never branch on
            whether the Helmholtz pair is energised.
        polarisation: Which ``Delta m`` the pump drives.

    Returns:
        Components in ascending shift order, weights summing to one over the driven set.

    Raises:
        ValueError: If the field is negative, or the level pair is not dipole-allowed.
    """
    field_t = float(magnitude_in(magnetic_field, "T"))
    if field_t < 0.0:
        # A signed field would flip sigma+ and sigma-, which is a real physical effect and
        # a real source of confusion. The direction belongs to the geometry, not here, so
        # this takes the magnitude and says so.
        raise ValueError(
            f"magnetic field must be given as a magnitude, got {magnetic_field}. The sense "
            "of the field relative to the beam belongs to the geometry, not to the pattern."
        )

    _check_dipole_allowed(lower, upper)

    g_lower = lande_g_factor(lower)
    g_upper = lande_g_factor(upper)
    scale_hz_per_t = _BOHR_MAGNETON_OVER_H_HZ_PER_T * field_t

    raw: list[tuple[float, int, float, float]] = []
    for m_lower in lower.magnetic_quantum_numbers:
        for delta_m in polarisation.driven_delta_m:
            m_upper = m_lower + delta_m
            if abs(m_upper) > upper.j:
                continue
            strength = _relative_strength(
                j_lower=lower.j, j_upper=upper.j, m_lower=m_lower, delta_m=delta_m
            )
            if strength <= 0.0:
                continue
            shift_hz = scale_hz_per_t * (g_upper * m_upper - g_lower * m_lower)
            raw.append((m_lower, delta_m, shift_hz, strength))

    # No emptiness check here, deliberately. For any dipole-allowed pair every
    # polarisation drives at least one component — the sum rule guarantees it — so a guard
    # would be untestable dead code claiming to protect against something that cannot
    # happen. The reachable failure is an empty *sequence* arriving at the velocity
    # integral, and vpl.instruments.lif.scan.fluorescence_response refuses that by name.
    total = sum(entry[3] for entry in raw)
    components = tuple(
        ZeemanComponent(
            m_lower=m_lower,
            delta_m=delta_m,
            shift=Q_(shift_hz, "Hz"),
            weight=strength / total,
        )
        for m_lower, delta_m, shift_hz, strength in raw
    )
    return tuple(sorted(components, key=lambda c: (c.shift_hz, c.m_lower)))


def unsplit_pattern() -> tuple[ZeemanComponent, ...]:
    """The single-component comb representing an unsplit line.

    Handed to the scan model when ``B = 0`` is known a priori, so that the zero-field
    verification case (doc 04 V-22's analytic Voigt) runs through the same code path as
    the split case rather than through a separate branch that could drift from it.
    """
    return (ZeemanComponent(m_lower=0.0, delta_m=0, shift=Q_(0.0, "Hz"), weight=1.0),)
