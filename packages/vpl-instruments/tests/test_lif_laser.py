"""The LIF laser, the beam, and the tuning-range limit — doc 02 §5.2, doc 01 §5.1.

The tuning-range tests are the point of this module. Doc 01 §5.1 calls the mode-hop-free
range "an honest constraint that must be surfaced, not buried", and doc 14 RS-03 scores it
as the fifth-highest project risk. It is checked here as a computed velocity and energy
ceiling, not as a comment.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from vpl.core.params import default_registry
from vpl.core.state import Species
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.lif.laser import Laser, TuningRange
from vpl.instruments.lif.transition import ProbeTransition


def _argon() -> Species:
    return Species(name="Ar+", mass=default_registry().quantity("species.Ar.mass"), charge_number=1)


class TestTheBeam:
    def test_the_peak_intensity_is_twice_the_power_over_the_beam_area(self) -> None:
        laser = Laser.from_registry()

        expected = 2.0 * 0.020 / (math.pi * (75.0e-6) ** 2)

        assert magnitude_in(laser.peak_intensity, "W/m**2") == pytest.approx(expected, rel=1e-9)

    def test_the_beam_waist_convention_reproduces_the_doc_02_rayleigh_range(self) -> None:
        """doc 02 LIF-O2 gives 26 mm. ``z_R = pi w0^2 / lambda`` returns that only if the
        registered 75 um is a 1/e^2 *radius*; as a diameter it would give 6.6 mm."""
        laser = Laser.from_registry()
        transition = ProbeTransition.from_registry()

        rayleigh_mm = magnitude_in(laser.rayleigh_range(transition.pump_wavelength), "mm")

        assert rayleigh_mm == pytest.approx(26.0, rel=0.02)

    def test_the_intensity_falls_as_the_inverse_square_of_the_waist(self) -> None:
        laser = Laser.from_registry()
        wider = laser.with_beam_waist(laser.beam_waist * 2.0)

        ratio = float(magnitude_in(wider.peak_intensity, "W/m**2")) / float(
            magnitude_in(laser.peak_intensity, "W/m**2")
        )

        assert ratio == pytest.approx(0.25, rel=1e-12)


class TestTheGrazingGeometry:
    def test_the_projection_factor_is_sin_fifteen_degrees(self) -> None:
        # doc 02 §4.2 check 2 and doc 04 §3.2: v_par = v_z sin(15 deg) = 0.259 v_z.
        assert Laser.from_registry().projection_factor == pytest.approx(0.259, abs=5e-4)

    def test_the_velocity_error_amplification_is_the_documented_3_86(self) -> None:
        # doc 04 §3.2: "this projection factor must be inverted to recover v_z, which
        # amplifies velocity errors by 3.86x".
        assert Laser.from_registry().velocity_amplification == pytest.approx(3.86, rel=2e-3)

    def test_a_beam_parallel_to_the_electrode_is_blind_and_says_so(self) -> None:
        # doc 02 §4.2: "a beam propagating along x measures the x-component of velocity,
        # not the z-component we need". Zero projection is a design failure, not a limit.
        with pytest.raises(ValueError, match="blind"):
            Laser.from_registry().with_grazing_angle(Q_(0.0, "deg"))


class TestSaturation:
    def test_the_saturation_parameter_is_intensity_over_saturation_intensity(self) -> None:
        laser = Laser.from_registry()
        transition = ProbeTransition.from_registry()

        expected = float(magnitude_in(laser.peak_intensity, "W/m**2")) / float(
            magnitude_in(transition.saturation_intensity, "W/m**2")
        )

        assert laser.saturation_parameter(transition) == pytest.approx(expected, rel=1e-12)

    def test_the_nominal_operating_point_is_deeply_saturated(self) -> None:
        """20 mW into a 75 um waist is ~2.3e6 W/m^2 against an I_sat of ~137 W/m^2.

        Doc 07 F-13 sweeps S from 0.01 to 10, so the nominal beam has to be attenuated by
        three orders of magnitude to reach the top of that sweep. That is a real statement
        about the instrument and it is asserted here so it cannot quietly stop being true.
        """
        saturation = Laser.from_registry().saturation_parameter(ProbeTransition.from_registry())

        assert saturation > 1.0e3

    def test_attenuating_the_beam_scales_the_saturation_parameter_linearly(self) -> None:
        laser = Laser.from_registry()
        transition = ProbeTransition.from_registry()

        attenuated = laser.attenuated_to(saturation=1.0, transition=transition)

        assert attenuated.saturation_parameter(transition) == pytest.approx(1.0, rel=1e-12)


class TestTuningRange:
    def test_the_half_span_is_half_the_mode_hop_free_range(self) -> None:
        assert magnitude_in(TuningRange.from_registry().half_span, "GHz") == pytest.approx(10.0)

    def test_a_detuning_inside_the_range_is_reachable(self) -> None:
        assert TuningRange.from_registry().contains(Q_(5.0, "GHz"))

    def test_a_detuning_outside_the_range_is_not_reachable(self) -> None:
        tuning = TuningRange.from_registry()

        assert not tuning.contains(Q_(15.0, "GHz"))
        assert not tuning.contains(Q_(-15.0, "GHz"))

    def test_the_velocity_ceiling_is_the_projected_doppler_inverse(self) -> None:
        tuning = TuningRange.from_registry()
        laser = Laser.from_registry()
        transition = ProbeTransition.from_registry()

        # v_z = c dnu / (nu_0 sin theta_L) = lambda dnu / sin theta_L.
        expected = 668.614e-9 * 10.0e9 / math.sin(math.radians(15.0))

        ceiling = tuning.velocity_ceiling(transition=transition, laser=laser)

        assert magnitude_in(ceiling, "m/s") == pytest.approx(expected, rel=1e-9)
        assert magnitude_in(ceiling, "km/s") == pytest.approx(25.8, rel=1e-2)

    def test_the_energy_ceiling_for_argon_is_about_140_electronvolts(self) -> None:
        """The number doc 01 §5.1 is really about, recomputed with the doc 02 §4.2 geometry.

        Doc 01 §5.1 computes the limit *without* the grazing projection and concludes that
        a 250 eV ion is unreachable. With the projection the ceiling is 3.86x higher in
        velocity and 14.9x higher in energy — still short of a 250 V sheath, so doc 01's
        conclusion stands, but the margin is much larger than its arithmetic implies.
        """
        ceiling = TuningRange.from_registry().energy_ceiling(
            species=_argon(),
            transition=ProbeTransition.from_registry(),
            laser=Laser.from_registry(),
        )

        assert magnitude_in(ceiling, "eV") == pytest.approx(138.0, rel=0.02)

    def test_the_unprojected_energy_ceiling_reproduces_doc_01_arithmetic(self) -> None:
        """doc 01 §5.1: 250 eV Ar+ travels at 3.47e4 m/s and shifts by 51.9 GHz.

        Scaled to the 10 GHz half-span that gives 6.7 km/s and 9.3 eV — which is what
        doc 01's own numbers imply for a beam with no projection factor. Checked here so
        that the difference between the two ceilings is a tested fact rather than a claim.
        """
        laser = Laser.from_registry().with_grazing_angle(Q_(90.0, "deg"))

        ceiling = TuningRange.from_registry().energy_ceiling(
            species=_argon(), transition=ProbeTransition.from_registry(), laser=laser
        )

        assert magnitude_in(ceiling, "eV") == pytest.approx(9.3, rel=0.02)

    def test_the_sheath_edge_ion_speeds_of_doc_01_fit_inside_the_range(self) -> None:
        """doc 01 §5.1: "ion speeds are ~c_s-3c_s (2.7-8 km/s) and are comfortably within
        tuning range". Checked at the top of that band."""
        tuning = TuningRange.from_registry()
        laser = Laser.from_registry()
        transition = ProbeTransition.from_registry()

        ceiling = float(
            magnitude_in(tuning.velocity_ceiling(transition=transition, laser=laser), "m/s")
        )

        assert ceiling > 8.0e3

    def test_a_wider_mode_hop_free_range_reaches_faster_ions(self) -> None:
        laser = Laser.from_registry()
        transition = ProbeTransition.from_registry()
        narrow = TuningRange.from_registry()
        wide = TuningRange(half_span=narrow.half_span * 2.0)

        ratio = float(
            magnitude_in(wide.velocity_ceiling(transition=transition, laser=laser), "m/s")
        ) / float(magnitude_in(narrow.velocity_ceiling(transition=transition, laser=laser), "m/s"))

        assert ratio == pytest.approx(2.0, rel=1e-12)


class TestGuards:
    def test_a_non_positive_power_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="power must be positive"):
            Laser.from_registry().with_power(Q_(0.0, "mW"))

    def test_a_non_positive_beam_waist_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="beam waist must be positive"):
            Laser.from_registry().with_beam_waist(Q_(0.0, "um"))

    def test_a_negative_linewidth_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="linewidth"):
            replace(Laser.from_registry(), linewidth=Q_(-1.0, "MHz"))

    def test_a_non_positive_target_saturation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            Laser.from_registry().attenuated_to(
                saturation=0.0, transition=ProbeTransition.from_registry()
            )

    def test_a_non_positive_tuning_span_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="half-span must be positive"):
            TuningRange(half_span=Q_(0.0, "GHz"))

    def test_requiring_an_unreachable_detuning_names_the_mitigation(self) -> None:
        with pytest.raises(ValueError, match="sheath edge"):
            TuningRange.from_registry().require_reachable(Q_(30.0, "GHz"))

    def test_a_reachable_detuning_passes_silently(self) -> None:
        assert TuningRange.from_registry().require_reachable(Q_(1.0, "GHz")) is None

    def test_covers_answers_in_velocity(self) -> None:
        tuning = TuningRange.from_registry()
        transition = ProbeTransition.from_registry()
        laser = Laser.from_registry()

        assert tuning.covers(speed=Q_(8.0, "km/s"), transition=transition, laser=laser)
        assert not tuning.covers(speed=Q_(40.0, "km/s"), transition=transition, laser=laser)

    def test_the_reprs_name_what_they_carry(self) -> None:
        assert "theta_L" in repr(Laser.from_registry())
        assert "mode-hop-free" in repr(TuningRange.from_registry())
