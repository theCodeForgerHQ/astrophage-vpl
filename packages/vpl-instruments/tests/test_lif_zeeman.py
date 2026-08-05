"""Zeeman structure of the probe transition — doc 04 §3.3.

Doc 04 §3.3 is unambiguous that this is not optional:

    Zeeman splitting at 50 G ... ~70 MHz ... 10 % of the Doppler width — NOT negligible.
    Ignoring it would bias the inferred ion temperature and distort the IVDF shape.

The checks here are analytic rather than self-referential. The Lande formula is pinned at
the two limits where it is exact (spin-only, orbit-only); the dipole strengths are checked
against the ``J = 0 -> 1`` case, where every component has equal strength for reasons of
symmetry alone, and against the sum rule that must hold for every ``m`` in any transition;
and the pattern's intensity-weighted centroid must sit exactly on the unshifted line.
"""

from __future__ import annotations

import math

import pytest

from vpl.core.units import Q_, magnitude_in
from vpl.instruments.lif.transition import Level, ProbeTransition
from vpl.instruments.lif.zeeman import (
    PumpPolarisation,
    ZeemanComponent,
    lande_g_factor,
    rms_spread,
    unsplit_pattern,
    zeeman_pattern,
    zeeman_scale,
)

_FIFTY_GAUSS = Q_(5.0, "mT")


def _expected_lande(*, j: float, orbital: float, spin: float) -> float:
    """The Lande formula written out longhand, independently of the module under test."""
    return 1.0 + (j * (j + 1) + spin * (spin + 1) - orbital * (orbital + 1)) / (2 * j * (j + 1))


def _shift_mhz(component: ZeemanComponent) -> float:
    return float(magnitude_in(component.shift, "MHz"))


class TestLandeGFactor:
    def test_a_spin_only_level_has_g_equal_to_two(self) -> None:
        assert lande_g_factor(Level(j=1.0, orbital=0.0, spin=1.0)) == pytest.approx(2.0, rel=1e-15)

    def test_an_orbit_only_level_has_g_equal_to_one(self) -> None:
        assert lande_g_factor(Level(j=1.0, orbital=1.0, spin=0.0)) == pytest.approx(1.0, rel=1e-15)

    def test_the_doc_02_lower_level_gives_the_quartet_f_value(self) -> None:
        # 3d 4F_7/2: 1 + 7.5/31.5.
        level = Level(j=3.5, orbital=3.0, spin=1.5)

        assert lande_g_factor(level) == pytest.approx(
            _expected_lande(j=3.5, orbital=3.0, spin=1.5), rel=1e-15
        )
        assert lande_g_factor(level) == pytest.approx(1.238095, rel=1e-6)

    def test_the_doc_02_upper_level_gives_the_quartet_d_value(self) -> None:
        # 4p 4D_5/2: 1 + 6.5/17.5.
        assert lande_g_factor(Level(j=2.5, orbital=2.0, spin=1.5)) == pytest.approx(
            1.371429, rel=1e-6
        )


class TestDipoleStrengths:
    """The relative strengths, checked where symmetry fixes the answer."""

    def test_a_j_zero_to_j_one_transition_has_three_equal_components(self) -> None:
        # With no lower-level structure the three polarisations cannot be distinguished,
        # so any correct strength table must return them equal.
        components = zeeman_pattern(
            lower=Level(j=0.0, orbital=0.0, spin=0.0),
            upper=Level(j=1.0, orbital=1.0, spin=0.0),
            magnetic_field=_FIFTY_GAUSS,
            polarisation=PumpPolarisation.ISOTROPIC,
        )

        assert len(components) == 3
        assert {round(c.weight, 12) for c in components} == {round(1.0 / 3.0, 12)}

    @pytest.mark.parametrize(
        ("lower_j", "upper_j"),
        [(3.5, 2.5), (3.5, 3.5), (2.5, 3.5), (0.5, 1.5), (1.0, 1.0)],
    )
    def test_the_sum_rule_holds_for_every_lower_sublevel(
        self, lower_j: float, upper_j: float
    ) -> None:
        """Summed over the three polarisations, every ``m`` couples equally strongly.

        This is a theorem about dipole matrix elements, not a property of this code, and it
        is violated by every plausible transcription error in the strength tables.
        """
        components = zeeman_pattern(
            lower=Level(j=lower_j, orbital=lower_j - 0.5, spin=0.5),
            upper=Level(j=upper_j, orbital=upper_j + 0.5, spin=0.5),
            magnetic_field=_FIFTY_GAUSS,
            polarisation=PumpPolarisation.ISOTROPIC,
        )

        per_sublevel: dict[float, float] = {}
        for component in components:
            per_sublevel[component.m_lower] = (
                per_sublevel.get(component.m_lower, 0.0) + component.weight
            )

        expected = 1.0 / (2.0 * lower_j + 1.0)
        for m_lower, total in per_sublevel.items():
            assert total == pytest.approx(expected, rel=1e-12), f"sublevel m = {m_lower}"

    def test_the_weights_are_normalised_for_every_polarisation(self) -> None:
        transition = ProbeTransition.from_registry()

        for polarisation in PumpPolarisation:
            components = zeeman_pattern(
                lower=transition.lower,
                upper=transition.upper,
                magnetic_field=_FIFTY_GAUSS,
                polarisation=polarisation,
            )
            total = sum(component.weight for component in components)

            assert total == pytest.approx(1.0, rel=1e-12), polarisation

    def test_forbidden_components_are_absent_rather_than_zero_weighted(self) -> None:
        transition = ProbeTransition.from_registry()

        components = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=_FIFTY_GAUSS,
            polarisation=PumpPolarisation.ISOTROPIC,
        )

        assert all(component.weight > 0.0 for component in components)
        # 6 pi + 6 sigma+ + 6 sigma- for J = 7/2 -> 5/2.
        assert len(components) == 18


class TestTheZeemanShifts:
    def test_a_zero_field_collapses_the_pattern_onto_the_line_centre(self) -> None:
        transition = ProbeTransition.from_registry()

        components = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=Q_(0.0, "mT"),
            polarisation=PumpPolarisation.ISOTROPIC,
        )

        assert all(_shift_mhz(component) == 0.0 for component in components)

    def test_the_shifts_are_linear_in_the_field(self) -> None:
        transition = ProbeTransition.from_registry()

        def spread(field_mt: float) -> float:
            components = zeeman_pattern(
                lower=transition.lower,
                upper=transition.upper,
                magnetic_field=Q_(field_mt, "mT"),
                polarisation=PumpPolarisation.ISOTROPIC,
            )
            return max(_shift_mhz(c) for c in components) - min(_shift_mhz(c) for c in components)

        assert spread(10.0) == pytest.approx(2.0 * spread(5.0), rel=1e-12)

    def test_the_intensity_weighted_centroid_is_unshifted(self) -> None:
        """A theorem: the Zeeman pattern's centre of gravity does not move with the field.

        If it did, the fitted line centre would move with ``B`` and the inferred drift
        velocity would acquire a field-dependent bias with nothing in the output to show it.
        """
        transition = ProbeTransition.from_registry()

        for polarisation in (PumpPolarisation.ISOTROPIC, PumpPolarisation.PI):
            components = zeeman_pattern(
                lower=transition.lower,
                upper=transition.upper,
                magnetic_field=_FIFTY_GAUSS,
                polarisation=polarisation,
            )
            centroid = sum(c.weight * _shift_mhz(c) for c in components)

            assert centroid == pytest.approx(0.0, abs=1e-9), polarisation

    def test_the_zeeman_scale_at_fifty_gauss_is_the_seventy_megahertz_of_doc_04(self) -> None:
        """``mu_B B / h`` at 50 G is 69.98 MHz — the number doc 04 §3.3 tabulates.

        Worth isolating, because the *pattern* is wider than its scale: see
        :meth:`test_the_widest_component_is_half_again_the_doc_04_figure`.
        """
        assert magnitude_in(zeeman_scale(_FIFTY_GAUSS), "MHz") == pytest.approx(69.98, rel=1e-3)

    def test_the_effective_broadening_is_ten_percent_of_the_doppler_width(self) -> None:
        """Doc 04 §3.3's conclusion, checked as a broadening rather than as a scale.

        The quantity that adds in quadrature to the Doppler width — and therefore the one
        that biases a fitted ``T_i`` — is the intensity-weighted RMS spread of the comb,
        62.5 MHz. Against the 734 MHz of doc 04 §3.3 that is 8.5 %, so the document's
        "10 % of the Doppler width — NOT negligible" is reproduced by the model.
        """
        transition = ProbeTransition.from_registry()
        components = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=_FIFTY_GAUSS,
            polarisation=PumpPolarisation.ISOTROPIC,
        )

        spread_mhz = float(magnitude_in(rms_spread(components), "MHz"))

        assert spread_mhz == pytest.approx(62.5, rel=0.02)
        assert spread_mhz / 734.0 == pytest.approx(0.10, abs=0.02)

    def test_the_widest_component_is_half_again_the_doc_04_figure(self) -> None:
        """The extreme component sits at 110 MHz, not at the tabulated ~70 MHz.

        ``|g_u m_u - g_l m_l|`` is largest not at the largest ``|m_l|`` but at
        ``m_l = -3/2 -> m_u = -5/2``, where the two terms *add*: 1.571 rather than 0.905.
        Doc 04 §3.3's ~70 MHz is the Zeeman scale ``mu_B B / h``; the comb extends 57 %
        beyond it. The distinction matters for the wings of the fitted lineshape and for
        anyone sizing a scan window around the splitting.
        """
        transition = ProbeTransition.from_registry()
        g_lower = _expected_lande(j=3.5, orbital=3.0, spin=1.5)
        g_upper = _expected_lande(j=2.5, orbital=2.0, spin=1.5)

        # mu_B / h = e / (4 pi m_e) = 13.996 GHz/T, at 5 mT.
        scale_mhz = 13.99624 * 1.0e3 * 5.0e-3
        expected = scale_mhz * abs(g_upper * -2.5 - g_lower * -1.5)

        components = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=_FIFTY_GAUSS,
            polarisation=PumpPolarisation.SIGMA_MINUS,
        )
        largest = max(abs(_shift_mhz(component)) for component in components)

        assert largest == pytest.approx(expected, rel=1e-5)
        assert largest == pytest.approx(110.0, rel=0.01)

    def test_the_rms_spread_vanishes_at_zero_field(self) -> None:
        transition = ProbeTransition.from_registry()

        components = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=Q_(0.0, "mT"),
            polarisation=PumpPolarisation.ISOTROPIC,
        )

        assert float(magnitude_in(rms_spread(components), "Hz")) == 0.0

    def test_pi_components_split_far_less_than_sigma_components(self) -> None:
        """The pi shifts carry ``g_u - g_l`` while the sigma shifts carry ``g_u`` itself.

        The consequence is the observable doc 04 §3.3 wants: pumping pi and sigma gives two
        differently broadened scans of the same distribution, so the polarisation turns the
        splitting from a nuisance into a second constraint.
        """
        transition = ProbeTransition.from_registry()

        def extreme(polarisation: PumpPolarisation) -> float:
            components = zeeman_pattern(
                lower=transition.lower,
                upper=transition.upper,
                magnetic_field=_FIFTY_GAUSS,
                polarisation=polarisation,
            )
            return max(abs(_shift_mhz(component)) for component in components)

        assert extreme(PumpPolarisation.PI) < 0.5 * extreme(PumpPolarisation.SIGMA_PLUS)

    def test_sigma_plus_and_sigma_minus_are_mirror_images(self) -> None:
        transition = ProbeTransition.from_registry()

        plus = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=_FIFTY_GAUSS,
            polarisation=PumpPolarisation.SIGMA_PLUS,
        )
        minus = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=_FIFTY_GAUSS,
            polarisation=PumpPolarisation.SIGMA_MINUS,
        )

        plus_pairs = sorted((round(_shift_mhz(c), 9), round(c.weight, 12)) for c in plus)
        minus_pairs = sorted((round(-_shift_mhz(c), 9), round(c.weight, 12)) for c in minus)

        assert plus_pairs == minus_pairs


class TestGuards:
    def test_a_negative_field_is_rejected(self) -> None:
        transition = ProbeTransition.from_registry()

        with pytest.raises(ValueError, match="magnitude"):
            zeeman_pattern(
                lower=transition.lower,
                upper=transition.upper,
                magnetic_field=Q_(-1.0, "mT"),
                polarisation=PumpPolarisation.ISOTROPIC,
            )

    def test_a_transition_violating_the_dipole_selection_rule_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="dipole"):
            zeeman_pattern(
                lower=Level(j=3.5, orbital=3.0, spin=1.5),
                upper=Level(j=0.5, orbital=1.0, spin=1.5),
                magnetic_field=_FIFTY_GAUSS,
                polarisation=PumpPolarisation.ISOTROPIC,
            )

    def test_a_zero_to_zero_transition_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="dipole"):
            zeeman_pattern(
                lower=Level(j=0.0, orbital=0.0, spin=0.0),
                upper=Level(j=0.0, orbital=1.0, spin=1.0),
                magnetic_field=_FIFTY_GAUSS,
                polarisation=PumpPolarisation.ISOTROPIC,
            )


class TestComponentRepr:
    def test_a_component_names_its_polarisation(self) -> None:
        transition = ProbeTransition.from_registry()

        components = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=_FIFTY_GAUSS,
            polarisation=PumpPolarisation.PI,
        )

        assert all(component.delta_m == 0 for component in components)
        assert "pi" in repr(components[0])
        assert not math.isnan(components[0].weight)


class TestMoreGuards:
    def test_an_empty_pattern_has_no_spread(self) -> None:
        with pytest.raises(ValueError, match="no weight"):
            rms_spread(())

    def test_the_unsplit_pattern_is_a_single_unshifted_component(self) -> None:
        components = unsplit_pattern()

        assert len(components) == 1
        assert components[0].weight == 1.0
        assert float(magnitude_in(components[0].shift, "Hz")) == 0.0

    def test_a_j_zero_level_has_no_magnetic_structure(self) -> None:
        assert lande_g_factor(Level(j=0.0, orbital=1.0, spin=1.0)) == 0.0

    def test_every_polarisation_drives_something_for_a_dipole_allowed_pair(self) -> None:
        """The sum rule again, in the form the code relies on for having no empty-set guard."""
        pairs = (
            (Level(j=3.5, orbital=3.0, spin=0.5), Level(j=2.5, orbital=3.0, spin=0.5)),
            (Level(j=1.0, orbital=1.0, spin=0.0), Level(j=1.0, orbital=2.0, spin=1.0)),
            (Level(j=0.5, orbital=0.0, spin=0.5), Level(j=1.5, orbital=1.0, spin=0.5)),
            (Level(j=0.0, orbital=0.0, spin=0.0), Level(j=1.0, orbital=1.0, spin=0.0)),
        )
        for lower, upper in pairs:
            for polarisation in PumpPolarisation:
                components = zeeman_pattern(
                    lower=lower,
                    upper=upper,
                    magnetic_field=_FIFTY_GAUSS,
                    polarisation=polarisation,
                )

                assert components, (lower, upper, polarisation)
