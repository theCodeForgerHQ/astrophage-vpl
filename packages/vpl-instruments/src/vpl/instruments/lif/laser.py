"""The pump laser, the grazing geometry, and the tuning-range limit — doc 02 §5.2.

## The limitation this module exists to represent

Doc 01 §5.1 states it as bluntly as anything in the specification:

    Full-energy IVDF measurement inside a high-bias sheath by single-photon LIF is
    therefore **not achievable with a conventional scanning setup**, and any proposal
    claiming otherwise is wrong.

The cause is :class:`TuningRange`: a single-frequency ECDL tunes 20 GHz before it mode-hops
(doc 02 LIF-L3), and 20 GHz of Doppler shift is a finite ion speed. Doc 14 RS-03 scores
this the fifth-highest risk in the project. It is therefore **not** a docstring caveat
here — it is a computed velocity ceiling, an energy ceiling, and a hard refusal in
:meth:`TuningRange.require_reachable`, so a scan that asks for a detuning the laser cannot
reach fails rather than returning a number.

## What the grazing geometry does to that limit, and a discrepancy worth stating

Doc 01 §5.1 works the limit out from the *unprojected* Doppler shift: a 250 eV Ar+ ion at
3.47e4 m/s shifts 51.9 GHz, far outside 20 GHz. Doc 02 §4.2 then fixes the geometry, and a
beam at 15 deg to the electrode sees only ``v_z sin(15 deg) = 0.259 v_z``. The two
statements pull in opposite directions and both are in the Baseline:

- the projection makes the **velocity resolution** 3.86x worse (doc 04 §3.2, doc 06 §4
  item 5 budgets 5.0 % for it);
- the same projection makes the **reach** 3.86x better in velocity and 14.9x better in
  energy, because a given laser detuning now corresponds to a 3.86x larger ``v_z``.

So the ceiling is ~138 eV rather than the ~9 eV doc 01's arithmetic implies. Doc 01's
conclusion is unaffected — 138 eV is still far below a 250 V sheath, and the mitigation it
prescribes (measure at the sheath edge, propagate through the reconstructed field) is
still the right one — but the margin is an order of magnitude larger than the document
says, and ``test_lif_laser.py`` pins both numbers so the difference stays visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.state import Species
from vpl.core.units import Q_, Quantity, ScalarQuantity, magnitude_in
from vpl.instruments.lif.transition import ProbeTransition

__all__ = ["Laser", "TuningRange"]

#: Below this projection factor the beam is treated as blind rather than merely inefficient.
#:
#: Doc 02 §4.2 check 2 calls a diagnostic geometry that cannot see the velocity component
#: of interest "a complete design failure". At ``sin(theta_L) -> 0`` the recovered ``v_z``
#: is a finite detuning divided by zero, so the failure mode is an infinity rather than a
#: large error; the guard turns it into a message that names the design check it violates.
_MINIMUM_PROJECTION: float = 1e-6


@dataclass(frozen=True, slots=True)
class Laser:
    """The doc 02 §5.2 ECDL and the beam it puts through the measurement volume.

    Attributes:
        power: Optical power delivered to the measurement volume.
        beam_waist: Gaussian beam waist, as a 1/e^2 **radius**. The convention is fixed by
            doc 02 LIF-O2's 26 mm Rayleigh range; see :meth:`rayleigh_range`.
        linewidth: Laser FWHM linewidth — a third homogeneous Lorentzian alongside the
            natural and collisional widths of the transition itself.
        grazing_angle: Angle between the beam axis and the electrode plane, doc 02 LIF-O3.
    """

    power: Quantity
    beam_waist: Quantity
    linewidth: Quantity
    grazing_angle: Quantity

    def __post_init__(self) -> None:
        if float(magnitude_in(self.power, "W")) <= 0.0:
            raise ValueError(f"laser power must be positive, got {self.power}")
        if float(magnitude_in(self.beam_waist, "m")) <= 0.0:
            raise ValueError(f"beam waist must be positive, got {self.beam_waist}")
        if float(magnitude_in(self.linewidth, "Hz")) < 0.0:
            raise ValueError(f"linewidth is a width and cannot be negative, got {self.linewidth}")

        if abs(math.sin(float(magnitude_in(self.grazing_angle, "rad")))) < _MINIMUM_PROJECTION:
            raise ValueError(
                f"a grazing angle of {self.grazing_angle} leaves the beam blind to v_z. "
                "doc 02 §4.2 check 2: a beam parallel to the electrode measures the "
                "x-component of velocity, not the z-component the project needs, and calls "
                "that a complete design failure rather than a resolution limit."
            )

    @classmethod
    def from_registry(cls, registry: ParameterRegistry | None = None) -> Laser:
        """Build from ``lif.yaml`` — doc 08 §5."""
        entries = registry if registry is not None else default_registry()
        return cls(
            power=entries.quantity("LIF.laser_power"),
            beam_waist=entries.quantity("LIF.beam_waist"),
            linewidth=entries.quantity("LIF.laser_linewidth"),
            grazing_angle=entries.quantity("LIF.grazing_angle"),
        )

    # ── the beam ────────────────────────────────────────────────────────────────

    @property
    def peak_intensity(self) -> ScalarQuantity:
        """``2 P / (pi w0^2)`` — the on-axis intensity of a Gaussian beam.

        The peak, not the average over the waist. Doc 04 §3.4's saturation parameter is a
        local quantity and an ion sits somewhere in the beam profile; using the peak makes
        the reported ``S`` the largest one anywhere in the measurement volume, which is
        the one the doc 07 F-13 sweep should be indexed by. The intensity *variation*
        across the volume is a doc 04 §6.2 weighting-function effect and belongs to F3.
        """
        return Q_(
            2.0 * float(magnitude_in(self.power, "W")) / (math.pi * self.beam_waist_m**2),
            "W/m**2",
        )

    def rayleigh_range(self, wavelength: Quantity) -> ScalarQuantity:
        """``z_R = pi w0^2 / lambda`` — doc 02 LIF-O2's consistency check.

        Included because it is what fixes the beam-waist convention. Doc 02 quotes 75 um
        and 26 mm together; only the radius reading of the first reproduces the second,
        and a factor of two in the waist is a factor of four in every saturation parameter
        the framework reports.
        """
        return Q_(
            math.pi * self.beam_waist_m**2 / float(magnitude_in(wavelength, "m")),
            "m",
        )

    # ── the grazing geometry (doc 02 §4.2 check 2) ──────────────────────────────

    @property
    def projection_factor(self) -> float:
        """``sin(theta_L)`` — the fraction of ``v_z`` the beam actually sees."""
        return math.sin(float(magnitude_in(self.grazing_angle, "rad")))

    @property
    def velocity_amplification(self) -> float:
        """``1 / sin(theta_L)`` — doc 04 §3.2's 3.86x error amplification.

        Named rather than written inline wherever a velocity is recovered, because it is a
        budgeted systematic (doc 06 §4 item 5) and a budget term that appears as a bare
        division somewhere is a budget term nobody can find.
        """
        return 1.0 / self.projection_factor

    # ── saturation (doc 04 §3.4) ────────────────────────────────────────────────

    def saturation_parameter(self, transition: ProbeTransition) -> float:
        """``S = I / I_sat`` — the number doc 04 §3.4 requires every run to report."""
        return float(magnitude_in(self.peak_intensity, "W/m**2")) / (
            transition.saturation_intensity_w_per_m2
        )

    def attenuated_to(self, *, saturation: float, transition: ProbeTransition) -> Laser:
        """The same beam at the power that gives a stated ``S`` — the doc 07 F-13 knob.

        F-13 sweeps ``S`` from 0.01 to 10. The nominal doc 02 LIF-L1 power sits three
        orders of magnitude above the top of that sweep, so the sweep is realised by
        attenuation and this is where the arithmetic lives.
        """
        if saturation <= 0.0:
            raise ValueError(f"saturation parameter must be positive, got {saturation}")
        factor = saturation / self.saturation_parameter(transition)
        return replace(self, power=self.power * factor)

    # ── sweeps ──────────────────────────────────────────────────────────────────

    def with_beam_waist(self, beam_waist: Quantity) -> Laser:
        """The same laser focused to a different waist."""
        return replace(self, beam_waist=beam_waist)

    def with_grazing_angle(self, grazing_angle: Quantity) -> Laser:
        """The same laser at a different angle to the electrode — doc 02 LIF-O3's sweep."""
        return replace(self, grazing_angle=grazing_angle)

    def with_power(self, power: Quantity) -> Laser:
        """The same laser at a different delivered power."""
        return replace(self, power=power)

    # ── SI magnitudes for hot loops (doc 08 §5) ─────────────────────────────────

    @property
    def beam_waist_m(self) -> float:
        return float(magnitude_in(self.beam_waist, "m"))

    @property
    def linewidth_hz(self) -> float:
        return float(magnitude_in(self.linewidth, "Hz"))

    def __repr__(self) -> str:
        return (
            f"Laser({self.power:.4g~P}, w0={self.beam_waist:.4g~P}, "
            f"theta_L={self.grazing_angle:.4g~P})"
        )


@dataclass(frozen=True, slots=True)
class TuningRange:
    """The mode-hop-free span the laser can scan — doc 02 LIF-L3, doc 01 §5.1, doc 14 RS-03.

    Attributes:
        half_span: Largest detuning from line centre the laser reaches, in either
            direction. Half of doc 02 LIF-L3's 20 GHz.
    """

    half_span: Quantity

    def __post_init__(self) -> None:
        if float(magnitude_in(self.half_span, "Hz")) <= 0.0:
            raise ValueError(f"tuning half-span must be positive, got {self.half_span}")

    @classmethod
    def from_registry(cls, registry: ParameterRegistry | None = None) -> TuningRange:
        """Build from ``lif.yaml``: half of the registered mode-hop-free range."""
        entries = registry if registry is not None else default_registry()
        return cls(half_span=entries.quantity("LIF.mode_hop_free_range") / 2.0)

    def contains(self, detuning: Quantity) -> bool:
        """Whether the laser can actually be tuned to ``detuning`` without mode-hopping."""
        return abs(float(magnitude_in(detuning, "Hz"))) <= self.half_span_hz

    def require_reachable(self, detuning: Quantity) -> None:
        """Refuse a detuning outside the mode-hop-free range.

        Deliberately an exception rather than a clip or a warning. Doc 01 §5.1 is a
        statement about what the hardware *cannot do*; a forward model that quietly
        returned a signal there would let an inversion draw information from a
        measurement no experiment could make, which is the inverse crime of doc 04 §7.3
        in a different costume.
        """
        if not self.contains(detuning):
            raise ValueError(
                f"detuning {detuning:.4g~P} is outside the {2.0 * self.half_span:.4g~P} "
                "mode-hop-free tuning range of doc 02 LIF-L3. doc 01 §5.1 records this as "
                "a hardware limit, not a modelling convenience: the mitigation is to "
                "measure at the sheath edge and propagate the entry distribution through "
                "the reconstructed field, not to scan further."
            )

    # ── what the limit means in velocity and energy ─────────────────────────────

    def velocity_ceiling(self, *, transition: ProbeTransition, laser: Laser) -> ScalarQuantity:
        """The largest ``|v_z|`` the scan can reach — doc 04 §3.2's mapping, inverted.

        ``dnu = nu_0 v_z sin(theta_L) / c``, so ``v_z_max = lambda dnu_max / sin(theta_L)``.
        The projection factor is in the denominator: the grazing geometry costs resolution
        and buys reach, and this is where the second half of that trade appears.
        """
        return Q_(
            transition.pump_wavelength_m * self.half_span_hz / laser.projection_factor,
            "m/s",
        )

    def energy_ceiling(
        self, *, species: Species, transition: ProbeTransition, laser: Laser
    ) -> ScalarQuantity:
        """Kinetic energy of an ion at the velocity ceiling — the doc 01 §5.1 number.

        Non-relativistic, which is exact to well under a part in ``1e8`` here: the ceiling
        is tens of km/s against a ``c`` of ``3e5`` km/s.
        """
        speed = float(
            magnitude_in(self.velocity_ceiling(transition=transition, laser=laser), "m/s")
        )
        return Q_(species.mass_kg * speed**2 / 2.0, "J").to("eV")

    def covers(self, *, speed: Quantity, transition: ProbeTransition, laser: Laser) -> bool:
        """Whether an ion of this speed along ``z`` would appear in the scan at all."""
        ceiling = self.velocity_ceiling(transition=transition, laser=laser)
        return abs(float(magnitude_in(speed, "m/s"))) <= float(magnitude_in(ceiling, "m/s"))

    @property
    def half_span_hz(self) -> float:
        return float(magnitude_in(self.half_span, "Hz"))

    def __repr__(self) -> str:
        return f"TuningRange(+/-{self.half_span:.4g~P} mode-hop-free)"
