"""The Ar II probe transition — doc 02 §5.3, doc 04 §3.1, §3.3, §3.4.

Every number checked here is either recomputed in the test from CODATA constants and a
registry entry, or is a value one of the Baseline documents prints. Nothing is compared
against a constant that the module under test also supplies, because a test that reads
its expectation from the code it is testing checks only that the code is deterministic.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from vpl.core.constants import PLANCK, SPEED_OF_LIGHT
from vpl.core.params import ProvenanceClass, default_registry
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.lif.transition import Level, ProbeTransition

_C_M_PER_S = float(magnitude_in(SPEED_OF_LIGHT, "m/s"))
_H_J_S = float(magnitude_in(PLANCK, "J*s"))


class TestTheRegistryEntries:
    def test_every_lif_entry_is_sourced(self) -> None:
        lif = [p for p in default_registry() if p.id.startswith("LIF.")]

        assert lif, "the LIF registry entries are missing entirely"
        assert [p.id for p in lif if p.provenance_class is ProvenanceClass.ASSUMED] == []

    def test_the_pump_wavelength_is_the_one_doc_02_specifies(self) -> None:
        assert default_registry().value_in("LIF.pump_wavelength", "nm") == pytest.approx(668.614)

    def test_the_fluorescence_wavelength_is_the_one_doc_02_specifies(self) -> None:
        assert default_registry().value_in("LIF.fluorescence_wavelength", "nm") == pytest.approx(
            442.72
        )

    def test_the_mode_hop_free_range_is_the_doc_02_lif_l3_spec(self) -> None:
        assert default_registry().value_in("LIF.mode_hop_free_range", "GHz") == pytest.approx(20.0)


class TestLevel:
    def test_degeneracy_is_two_j_plus_one(self) -> None:
        # 3d 4F_7/2 as doc 02 §5.3 designates it.
        assert Level(j=3.5, orbital=3.0, spin=1.5).degeneracy == 8

    def test_a_half_integer_j_with_integer_l_and_s_is_rejected(self) -> None:
        # J = L + S must be reachable by vector addition: J - (L + S) is an integer.
        with pytest.raises(ValueError, match="cannot couple"):
            Level(j=3.5, orbital=3.0, spin=1.0)

    def test_j_outside_the_triangle_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot couple"):
            Level(j=9.0, orbital=3.0, spin=1.5)


class TestProbeTransition:
    def test_it_loads_from_the_registry(self) -> None:
        transition = ProbeTransition.from_registry()

        assert magnitude_in(transition.pump_wavelength, "nm") == pytest.approx(668.614)
        assert magnitude_in(transition.fluorescence_wavelength, "nm") == pytest.approx(442.72)

    def test_the_pump_frequency_is_c_over_lambda(self) -> None:
        transition = ProbeTransition.from_registry()

        expected = _C_M_PER_S / 668.614e-9

        assert magnitude_in(transition.pump_frequency, "Hz") == pytest.approx(expected, rel=1e-12)

    def test_the_pump_frequency_is_448_terahertz(self) -> None:
        # A sanity anchor a reader can check by hand: 3e8 / 6.686e-7.
        assert magnitude_in(ProbeTransition.from_registry().pump_frequency, "THz") == pytest.approx(
            448.4, rel=1e-3
        )

    def test_the_degeneracies_follow_the_doc_02_term_symbols(self) -> None:
        transition = ProbeTransition.from_registry()

        # 3d 4F_7/2 -> g = 8; 4p 4D_5/2 -> g = 6.
        assert transition.lower.degeneracy == 8
        assert transition.upper.degeneracy == 6

    def test_the_total_decay_rate_is_two_pi_times_the_natural_linewidth(self) -> None:
        transition = ProbeTransition.from_registry()
        natural_hz = default_registry().value_in("LIF.natural_linewidth", "Hz")

        expected = 2.0 * math.pi * float(natural_hz)

        assert magnitude_in(transition.total_decay_rate, "1/s") == pytest.approx(expected)

    def test_the_natural_linewidth_is_the_five_megahertz_of_doc_04_section_3_3(self) -> None:
        assert magnitude_in(
            ProbeTransition.from_registry().natural_linewidth, "MHz"
        ) == pytest.approx(5.0)

    def test_the_relaxation_rate_is_the_decay_rate_plus_quenching(self) -> None:
        transition = ProbeTransition.from_registry()

        expected = float(magnitude_in(transition.total_decay_rate, "1/s")) + float(
            magnitude_in(transition.quench_rate, "1/s")
        )

        assert magnitude_in(transition.relaxation_rate, "1/s") == pytest.approx(expected)

    def test_homogeneous_widths_add_as_lorentzians(self) -> None:
        # Lorentzian FWHMs add linearly under convolution, unlike Gaussian ones.
        transition = ProbeTransition.from_registry()
        registry = default_registry()

        expected = float(registry.value_in("LIF.natural_linewidth", "MHz")) + float(
            registry.value_in("LIF.pressure_broadening", "MHz")
        )

        assert magnitude_in(transition.homogeneous_fwhm, "MHz") == pytest.approx(expected)


class TestSaturationIntensity:
    """Doc 04 §3.4: ``I_sat = 2 pi^2 h c A_21 / (3 lambda^3)``."""

    def test_it_matches_the_doc_04_closed_form(self) -> None:
        transition = ProbeTransition.from_registry()
        registry = default_registry()

        lambda_m = 668.614e-9
        a_21 = (
            2.0
            * math.pi
            * float(registry.value_in("LIF.natural_linewidth", "Hz"))
            * float(registry.value_in("LIF.pump_branching_ratio", "dimensionless"))
        )
        expected = 2.0 * math.pi**2 * _H_J_S * _C_M_PER_S * a_21 / (3.0 * lambda_m**3)

        assert magnitude_in(transition.saturation_intensity, "W/m**2") == pytest.approx(
            expected, rel=1e-10
        )

    def test_it_is_of_order_a_hundred_watts_per_square_metre(self) -> None:
        # 2 pi^2 h c A / (3 lambda^3) with A = 3.14e7 and lambda = 668.6 nm is ~137 W/m^2:
        # a number a reader can bound by hand, and the anchor for the doc 07 F-13 sweep.
        intensity = float(
            magnitude_in(ProbeTransition.from_registry().saturation_intensity, "W/m**2")
        )

        assert 100.0 < intensity < 200.0

    def test_it_scales_as_the_inverse_cube_of_the_wavelength(self) -> None:
        base = ProbeTransition.from_registry()
        doubled = base.with_pump_wavelength(base.pump_wavelength * 2.0)

        ratio = float(magnitude_in(doubled.saturation_intensity, "W/m**2")) / float(
            magnitude_in(base.saturation_intensity, "W/m**2")
        )

        assert ratio == pytest.approx(1.0 / 8.0, rel=1e-12)

    def test_it_is_proportional_to_the_pump_einstein_coefficient(self) -> None:
        base = ProbeTransition.from_registry()
        halved = base.with_pump_branching(base.pump_branching / 2.0)

        ratio = float(magnitude_in(halved.saturation_intensity, "W/m**2")) / float(
            magnitude_in(base.saturation_intensity, "W/m**2")
        )

        assert ratio == pytest.approx(0.5, rel=1e-12)


class TestGuards:
    """The validation paths. Each one turns a silent wrong answer into a message."""

    def test_a_negative_quantum_number_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="multiple of 1/2"):
            Level(j=-1.0, orbital=1.0, spin=0.0)

    def test_a_quantum_number_that_is_not_a_half_integer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="multiple of 1/2"):
            Level(j=1.3, orbital=1.0, spin=0.3)

    def test_a_non_positive_wavelength_is_rejected(self) -> None:
        base = ProbeTransition.from_registry()

        with pytest.raises(ValueError, match="pump_wavelength must be positive"):
            base.with_pump_wavelength(Q_(0.0, "nm"))

    def test_a_negative_quench_rate_is_rejected(self) -> None:
        base = ProbeTransition.from_registry()

        with pytest.raises(ValueError, match="quench_rate"):
            base.with_quench_rate(Q_(-1.0, "1/s"))

    def test_a_branching_ratio_above_one_is_rejected(self) -> None:
        base = ProbeTransition.from_registry()

        with pytest.raises(ValueError, match="pump_branching"):
            base.with_pump_branching(1.5)

    def test_a_negative_width_is_rejected(self) -> None:
        base = ProbeTransition.from_registry()

        with pytest.raises(ValueError, match="cannot be negative"):
            replace(base, pressure_broadening=Q_(-1.0, "MHz"))

    def test_the_magnetic_sublevels_run_from_minus_j_to_plus_j(self) -> None:
        assert Level(j=1.5, orbital=1.0, spin=0.5).magnetic_quantum_numbers == (
            -1.5,
            -0.5,
            0.5,
            1.5,
        )

    def test_the_repr_names_the_saturation_intensity(self) -> None:
        assert "I_sat" in repr(ProbeTransition.from_registry())
        assert "J=3.5" in repr(ProbeTransition.from_registry().lower)

    def test_the_fluorescence_photon_energy_is_h_c_over_lambda(self) -> None:
        transition = ProbeTransition.from_registry()

        expected = _H_J_S * _C_M_PER_S / 442.72e-9

        assert magnitude_in(transition.fluorescence_photon_energy, "J") == pytest.approx(
            expected, rel=1e-12
        )
