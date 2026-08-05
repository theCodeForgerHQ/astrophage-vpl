"""The Ar II probe transition — doc 02 §5.3, doc 04 §3.1 and §3.4.

Two levels and the numbers that couple them. Everything here is loaded from the
parameter registry (doc 08 §5); the module contains no physical magnitude of its own.

## What ``I_sat`` is built from, and what it is not

Doc 04 §3.4 fixes the saturation intensity as

    I_sat = 2 pi^2 h c A_21 / (3 lambda^3)

and requires it to be "computed and reported for every run". ``A_21`` is the Einstein
coefficient of the pump transition alone. What the project can source is the *total*
radiative width of the upper level — doc 04 §3.3's 5 MHz natural linewidth, which is
``A_total / 2 pi`` — so ``A_21 = b A_total`` with ``b`` the registered pump branching
ratio ``LIF.pump_branching_ratio``.

The nominal ``b = 1`` is not a claim that the upper level decays only down the pump
branch; it is the bound ``A_21 <= A_total``. It makes ``I_sat`` an **upper** bound and
therefore the modelled saturation parameter ``S = I / I_sat`` a **lower** bound, which is
the conservative direction: the framework's finding that the doc 02 §5.2 operating point
is deeply saturated survives any smaller ``b``, and only gets stronger. ``lif.yaml``
records what replaces the sweep with a measurement.

## The term symbols are the specification's, and the specification disagrees with itself

Doc 02 §5.3 pairs the term symbols of one standard Ar II scheme with the wavelengths of
another; ``lif.yaml`` documents which is which. The wavelengths are taken as written
because they fix hardware that exists (a 668.6 nm ECDL, a 442.7 nm filter). The term
symbols enter only through the Lande g factors of :mod:`vpl.instruments.lif.zeeman`, and
the two candidate assignments differ there by under 10 % — enough to move a fitted ``T_i``
at the percent level, not enough to change any conclusion drawn in this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from vpl.core.constants import PLANCK, SPEED_OF_LIGHT
from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.units import Q_, Quantity, ScalarQuantity, magnitude_in

__all__ = ["Level", "ProbeTransition"]

#: CODATA constants unwrapped once, at import (doc 08 §5: quantities at boundaries, bare
#: floats in the arithmetic). Doc 09 §2.5 forbids typing either of these by hand.
_H_J_S: float = float(magnitude_in(PLANCK, "J*s"))
_C_M_PER_S: float = float(magnitude_in(SPEED_OF_LIGHT, "m/s"))


def _is_integer(value: float) -> bool:
    """Whether ``value`` is a whole number, up to floating-point representation.

    Angular-momentum quantum numbers arrive as ``3.5`` and ``1.5`` from YAML, so the
    coupling checks below are differences of half-integers and are exact in binary.
    """
    return math.isclose(value, round(value), abs_tol=math.sqrt(math.ulp(1.0)))


@dataclass(frozen=True, slots=True)
class Level:
    """One atomic level, in the LS-coupling quantum numbers the Lande formula takes.

    Attributes:
        j: Total angular momentum ``J``.
        orbital: Total orbital angular momentum ``L``. Named in full because ``l`` is
            unreadable next to ``1`` and the project lints for it.
        spin: Total spin ``S``.
    """

    j: float
    orbital: float
    spin: float

    def __post_init__(self) -> None:
        for name, value in (("J", self.j), ("L", self.orbital), ("S", self.spin)):
            if value < 0.0 or not _is_integer(value * 2.0):
                raise ValueError(f"{name} must be a non-negative multiple of 1/2, got {value}")

        low = abs(self.orbital - self.spin)
        high = self.orbital + self.spin
        if not (low <= self.j <= high) or not _is_integer(high - self.j):
            # A term symbol mistyped in the registry produces a level that no vector
            # addition of L and S can reach. It would still yield a finite Lande g and a
            # finite Zeeman pattern, so nothing downstream would object.
            raise ValueError(
                f"L = {self.orbital} and S = {self.spin} cannot couple to J = {self.j}; "
                f"the reachable values are {low} to {high} in integer steps"
            )

    @property
    def degeneracy(self) -> int:
        """``2J + 1`` — the number of magnetic sublevels, and the statistical weight."""
        return round(2.0 * self.j + 1.0)

    @property
    def magnetic_quantum_numbers(self) -> tuple[float, ...]:
        """``m = -J ... +J``, ascending. The sublevels the Zeeman pattern runs over."""
        return tuple(-self.j + index for index in range(self.degeneracy))

    def __repr__(self) -> str:
        return f"Level(J={self.j:g}, L={self.orbital:g}, S={self.spin:g})"


@dataclass(frozen=True, slots=True)
class ProbeTransition:
    """The pump/fluorescence pair of doc 02 §5.3 and its rate-equation constants.

    Attributes:
        lower: The pumped level — level 1 of doc 04 §3.1's rate equation.
        upper: The laser-coupled level — level 2. Its decay to a third level is the
            observed fluorescence.
        pump_wavelength: Vacuum wavelength of the ``1 -> 2`` transition.
        fluorescence_wavelength: Vacuum wavelength of the observed ``2 -> 3`` branch.
        natural_linewidth: Radiative FWHM of the upper level, ``A_total / 2 pi``.
        pressure_broadening: Collisional contribution to the homogeneous FWHM.
        quench_rate: ``Q`` of doc 04 §3.1 — non-radiative depopulation of the upper level.
        pump_branching: ``A_21 / A_total``. See the module docstring.
    """

    lower: Level
    upper: Level
    pump_wavelength: Quantity
    fluorescence_wavelength: Quantity
    natural_linewidth: Quantity
    pressure_broadening: Quantity
    quench_rate: Quantity
    pump_branching: float

    def __post_init__(self) -> None:
        for name, value in (
            ("pump_wavelength", self.pump_wavelength),
            ("fluorescence_wavelength", self.fluorescence_wavelength),
        ):
            if float(magnitude_in(value, "m")) <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")

        for name, value in (
            ("natural_linewidth", self.natural_linewidth),
            ("pressure_broadening", self.pressure_broadening),
        ):
            if float(magnitude_in(value, "Hz")) < 0.0:
                raise ValueError(f"{name} is a width and cannot be negative, got {value}")

        if float(magnitude_in(self.quench_rate, "1/s")) < 0.0:
            raise ValueError(f"quench_rate cannot be negative, got {self.quench_rate}")

        if not 0.0 < self.pump_branching <= 1.0:
            raise ValueError(
                f"pump_branching is A_21 / A_total and must lie in (0, 1], got "
                f"{self.pump_branching}"
            )

    @classmethod
    def from_registry(cls, registry: ParameterRegistry | None = None) -> ProbeTransition:
        """Build from ``lif.yaml`` — the doc 08 §5 sole source of numeric defaults."""
        entries = registry if registry is not None else default_registry()
        return cls(
            lower=Level(
                j=float(entries.value_in("LIF.lower_level.J", "dimensionless")),
                orbital=float(entries.value_in("LIF.lower_level.L", "dimensionless")),
                spin=float(entries.value_in("LIF.lower_level.S", "dimensionless")),
            ),
            upper=Level(
                j=float(entries.value_in("LIF.upper_level.J", "dimensionless")),
                orbital=float(entries.value_in("LIF.upper_level.L", "dimensionless")),
                spin=float(entries.value_in("LIF.upper_level.S", "dimensionless")),
            ),
            pump_wavelength=entries.quantity("LIF.pump_wavelength"),
            fluorescence_wavelength=entries.quantity("LIF.fluorescence_wavelength"),
            natural_linewidth=entries.quantity("LIF.natural_linewidth"),
            pressure_broadening=entries.quantity("LIF.pressure_broadening"),
            quench_rate=entries.quantity("LIF.quench_rate"),
            pump_branching=float(entries.value_in("LIF.pump_branching_ratio", "dimensionless")),
        )

    # ── derived quantities ──────────────────────────────────────────────────────

    @property
    def pump_frequency(self) -> ScalarQuantity:
        """``nu_0 = c / lambda`` — the line centre the doc 04 §3.2 scan is detuned from."""
        return Q_(_C_M_PER_S / self.pump_wavelength_m, "Hz")

    @property
    def fluorescence_photon_energy(self) -> ScalarQuantity:
        """``h nu_23`` — the energy of one observed photon."""
        return Q_(_H_J_S * _C_M_PER_S / float(magnitude_in(self.fluorescence_wavelength, "m")), "J")

    @property
    def total_decay_rate(self) -> ScalarQuantity:
        """``A_total = 2 pi x`` the natural linewidth: all radiative loss from level 2."""
        return Q_(2.0 * math.pi * float(magnitude_in(self.natural_linewidth, "Hz")), "1/s")

    @property
    def pump_einstein_coefficient(self) -> ScalarQuantity:
        """``A_21``, reconstructed as ``b A_total``. See the module docstring."""
        return Q_(self.pump_branching * self.total_decay_rate_per_s, "1/s")

    @property
    def relaxation_rate(self) -> ScalarQuantity:
        """``A_21 + A_23 + Q`` of doc 04 §3.1 — the whole loss term for level 2.

        The two radiative branches are not separable here (only their sum is sourceable),
        so this is ``A_total + Q``, which is exactly what the rate equation's loss term
        needs. Only ``I_sat`` requires the split, and only through ``pump_branching``.
        """
        return Q_(self.total_decay_rate_per_s + float(magnitude_in(self.quench_rate, "1/s")), "1/s")

    @property
    def homogeneous_fwhm(self) -> ScalarQuantity:
        """Natural plus collisional FWHM of the transition itself.

        Lorentzian widths add linearly under convolution, which is why this is a sum and
        the Doppler contribution — Gaussian, and added in quadrature — is nowhere near it.
        The laser linewidth is a third Lorentzian and belongs to
        :class:`~vpl.instruments.lif.laser.Laser`, not to the atom.
        """
        return Q_(
            float(magnitude_in(self.natural_linewidth, "Hz"))
            + float(magnitude_in(self.pressure_broadening, "Hz")),
            "Hz",
        )

    @property
    def saturation_intensity(self) -> ScalarQuantity:
        """``I_sat = 2 pi^2 h c A_21 / (3 lambda^3)`` — doc 04 §3.4, verbatim."""
        return Q_(
            2.0
            * math.pi**2
            * _H_J_S
            * _C_M_PER_S
            * float(magnitude_in(self.pump_einstein_coefficient, "1/s"))
            / (3.0 * self.pump_wavelength_m**3),
            "W/m**2",
        )

    # ── SI magnitudes for hot loops (doc 08 §5) ─────────────────────────────────

    @property
    def pump_wavelength_m(self) -> float:
        return float(magnitude_in(self.pump_wavelength, "m"))

    @property
    def pump_frequency_hz(self) -> float:
        return float(magnitude_in(self.pump_frequency, "Hz"))

    @property
    def total_decay_rate_per_s(self) -> float:
        return float(magnitude_in(self.total_decay_rate, "1/s"))

    @property
    def relaxation_rate_per_s(self) -> float:
        return float(magnitude_in(self.relaxation_rate, "1/s"))

    @property
    def saturation_intensity_w_per_m2(self) -> float:
        return float(magnitude_in(self.saturation_intensity, "W/m**2"))

    # ── sweeps (doc 09 §1: a DESIGN parameter is a variable, not a constant) ─────

    def with_pump_wavelength(self, wavelength: Quantity) -> ProbeTransition:
        """The same transition at a different pump wavelength."""
        return replace(self, pump_wavelength=wavelength)

    def with_pump_branching(self, branching: float) -> ProbeTransition:
        """The same transition with a different ``A_21 / A_total``."""
        return replace(self, pump_branching=branching)

    def with_quench_rate(self, quench_rate: Quantity) -> ProbeTransition:
        """The same transition at a different collisional quenching rate."""
        return replace(self, quench_rate=quench_rate)

    def __repr__(self) -> str:
        return (
            f"ProbeTransition(pump={self.pump_wavelength:.4g~P}, "
            f"fluorescence={self.fluorescence_wavelength:.4g~P}, "
            f"I_sat={self.saturation_intensity:.4g~P})"
        )
