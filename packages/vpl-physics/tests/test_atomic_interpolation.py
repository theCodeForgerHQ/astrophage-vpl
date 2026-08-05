"""Putting a tabulated cross section on an arbitrary energy grid — doc 03 §3.2, §4.5.

Doc 03 §3.2 makes the two-term Boltzmann solver consume ``sigma(E)`` on whatever grid it
chose, and doc 03 §4.5 says getting charge exchange "right matters more than any other
atomic-data choice". The place both statements meet is the tail: a rate coefficient is an
integral of ``sigma(E) f(E) sqrt(E)``, so an extrapolation invented off the end of the
table is indistinguishable from data once it is inside the integral.

Hence the policy is explicit and the default off the top of the table is to **refuse**.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.physics.atomic.interpolation import (
    DEFAULT_ABOVE,
    DEFAULT_BELOW,
    ExtrapolationPolicy,
    OutsideTabulatedRangeError,
    interpolate_cross_section,
    interpolate_set,
)
from vpl.physics.atomic.lxcat import CrossSection, CrossSectionSet, ProcessType


def _threshold_section(**overrides: object) -> CrossSection:
    """An ionisation cross section: zero at threshold, a peak, then a falling tail."""
    defaults: dict[str, object] = {
        "process": ProcessType.IONIZATION,
        "database": "Phelps",
        "projectile": "e",
        "target": "Ar",
        "reactants": ("E", "Ar"),
        "products": ("E", "E", "Ar+"),
        "threshold_ev": 16.0,
        "mass_ratio": None,
        "energy_ev": np.array([16.0, 20.0, 100.0, 200.0]),
        "sigma_m2": np.array([0.0, 1e-21, 3e-20, 1.5e-20]),
        "parameters": {},
    }
    return CrossSection(**{**defaults, **overrides})  # type: ignore[arg-type]


def _cx_section(**overrides: object) -> CrossSection:
    """A charge-exchange cross section: no threshold, falling from low energy."""
    defaults: dict[str, object] = {
        "process": ProcessType.CHARGE_EXCHANGE,
        "database": "Phelps",
        "projectile": "Ar+",
        "target": "Ar",
        "reactants": ("Ar+", "Ar"),
        "products": ("Ar", "Ar+"),
        "threshold_ev": None,
        "mass_ratio": None,
        "energy_ev": np.array([1.0, 10.0, 100.0]),
        "sigma_m2": np.array([8e-19, 4e-19, 2e-19]),
        "parameters": {},
    }
    return CrossSection(**{**defaults, **overrides})  # type: ignore[arg-type]


# ── inside the table ────────────────────────────────────────────────────────────


class TestInsideTheTabulatedRange:
    def test_a_tabulated_point_returns_its_tabulated_value(self) -> None:
        section = _threshold_section()

        result = interpolate_cross_section(section, np.array([16.0, 20.0, 100.0, 200.0]))

        np.testing.assert_allclose(result, section.sigma_m2)

    @pytest.mark.physics
    def test_interpolation_is_linear_in_energy_and_cross_section(self) -> None:
        # Linear-linear rather than log-log, and the choice is not free. Cross-section
        # tables carry exact zeros at and below threshold, which log-log cannot
        # represent; and linear interpolation invents no structure between points, which
        # is the conservative failure mode for a quantity that goes on to be integrated.
        section = _cx_section()

        midpoint = interpolate_cross_section(section, np.array([5.5]))

        assert midpoint[0] == pytest.approx(6e-19)

    def test_the_result_has_the_shape_of_the_requested_grid(self) -> None:
        result = interpolate_cross_section(_cx_section(), np.linspace(1.0, 100.0, 37))

        assert result.shape == (37,)

    def test_a_scalar_grid_is_accepted(self) -> None:
        assert interpolate_cross_section(_cx_section(), 10.0).shape == ()

    def test_a_list_grid_is_accepted(self) -> None:
        result = interpolate_cross_section(_cx_section(), [1.0, 10.0])

        np.testing.assert_allclose(result, [8e-19, 4e-19])

    def test_the_result_is_never_negative(self) -> None:
        result = interpolate_cross_section(_threshold_section(), np.linspace(0.0, 200.0, 101))

        assert np.all(result >= 0.0)


# ── below the table ─────────────────────────────────────────────────────────────


class TestBelowTheThreshold:
    """Not a policy. An inelastic process cannot occur below its threshold."""

    @pytest.mark.physics
    def test_the_cross_section_is_exactly_zero_below_the_threshold(self) -> None:
        result = interpolate_cross_section(_threshold_section(), np.array([0.0, 1.0, 15.9]))

        np.testing.assert_array_equal(result, np.zeros(3))

    @pytest.mark.physics
    def test_the_sub_threshold_zero_is_not_an_extrapolation_and_never_raises(self) -> None:
        # The table starts at the threshold, so every sub-threshold point is also below
        # the tabulated range. Applying the physics first is what keeps a caller asking
        # for a grid from 0 eV from having to opt out of a policy that does not apply.
        result = interpolate_cross_section(
            _threshold_section(),
            np.array([0.0, 10.0]),
            below=ExtrapolationPolicy.RAISE,
        )

        np.testing.assert_array_equal(result, np.zeros(2))

    @pytest.mark.physics
    def test_the_zero_overrides_whatever_the_policy_would_have_produced(self) -> None:
        result = interpolate_cross_section(
            _threshold_section(),
            np.array([1.0]),
            below=ExtrapolationPolicy.CONSTANT,
        )

        assert result[0] == 0.0


class TestBelowTheTabulatedRange:
    """Where there is no threshold to appeal to, the caller chooses."""

    def test_the_default_holds_the_first_tabulated_value(self) -> None:
        # Charge exchange rises slowly and monotonically as E -> 0 and has no threshold.
        # Holding the first value is the mildest defensible statement; the alternatives
        # either invent a divergence or delete a channel that is real at low energy.
        assert DEFAULT_BELOW is ExtrapolationPolicy.CONSTANT

        result = interpolate_cross_section(_cx_section(), np.array([0.01, 0.5]))

        np.testing.assert_allclose(result, [8e-19, 8e-19])

    def test_zero_deletes_the_channel_below_the_table(self) -> None:
        result = interpolate_cross_section(
            _cx_section(), np.array([0.5]), below=ExtrapolationPolicy.ZERO
        )

        assert result[0] == 0.0

    @pytest.mark.physics
    def test_the_power_law_continues_the_first_two_points_in_log_log(self) -> None:
        # sigma = 8e-19 at 1 eV and 4e-19 at 10 eV is a slope of -log10(2) per decade,
        # so 0.1 eV must return 1.6e-18.
        result = interpolate_cross_section(
            _cx_section(), np.array([0.1]), below=ExtrapolationPolicy.POWER_LAW
        )

        assert result[0] == pytest.approx(1.6e-18)

    def test_raise_names_the_offending_energy_and_the_tabulated_range(self) -> None:
        with pytest.raises(OutsideTabulatedRangeError, match=r"0\.5"):
            interpolate_cross_section(
                _cx_section(), np.array([0.5]), below=ExtrapolationPolicy.RAISE
            )

    def test_a_table_starting_at_zero_energy_has_nothing_below_it(self) -> None:
        # LXCat elastic tables commonly start at exactly 0 eV. Energies are validated
        # non-negative, so the below-range branch is unreachable for such a table and no
        # logarithm of zero is ever taken.
        section = _cx_section(
            energy_ev=np.array([0.0, 10.0, 100.0]), sigma_m2=np.array([8e-19, 4e-19, 2e-19])
        )

        result = interpolate_cross_section(
            section, np.array([0.0]), below=ExtrapolationPolicy.RAISE
        )

        assert result[0] == pytest.approx(8e-19)

    def test_a_power_law_refuses_a_zero_valued_anchor(self) -> None:
        section = _cx_section(sigma_m2=np.array([0.0, 4e-19, 2e-19]))

        with pytest.raises(OutsideTabulatedRangeError, match="zero"):
            interpolate_cross_section(section, np.array([0.5]), below=ExtrapolationPolicy.POWER_LAW)


# ── above the table, which is the one that matters ──────────────────────────────


class TestAboveTheTabulatedRange:
    def test_the_default_is_to_refuse(self) -> None:
        # doc 03 §4.5 makes the energy dependence of the cross section the single most
        # consequential atomic-data choice in the project. Every silent option above the
        # table is wrong in a different direction, and none of them announce themselves
        # in the answer, so the default announces itself instead.
        assert DEFAULT_ABOVE is ExtrapolationPolicy.RAISE

        with pytest.raises(OutsideTabulatedRangeError):
            interpolate_cross_section(_threshold_section(), np.array([500.0]))

    def test_the_refusal_names_the_section_and_both_ends_of_the_table(self) -> None:
        with pytest.raises(OutsideTabulatedRangeError, match="200"):
            interpolate_cross_section(_threshold_section(), np.array([500.0]))

    def test_constant_holds_the_last_tabulated_value(self) -> None:
        result = interpolate_cross_section(
            _threshold_section(), np.array([500.0]), above=ExtrapolationPolicy.CONSTANT
        )

        assert result[0] == pytest.approx(1.5e-20)

    def test_zero_truncates_the_tail(self) -> None:
        result = interpolate_cross_section(
            _threshold_section(), np.array([500.0]), above=ExtrapolationPolicy.ZERO
        )

        assert result[0] == 0.0

    @pytest.mark.physics
    def test_the_power_law_continues_the_last_two_points_in_log_log(self) -> None:
        # 3e-20 at 100 eV and 1.5e-20 at 200 eV is sigma ~ 1/E, which is the correct
        # asymptotic form for electron-impact ionisation up to the Bethe log. 400 eV
        # must therefore return 7.5e-21.
        result = interpolate_cross_section(
            _threshold_section(), np.array([400.0]), above=ExtrapolationPolicy.POWER_LAW
        )

        assert result[0] == pytest.approx(7.5e-21)

    def test_the_power_law_refuses_a_zero_valued_final_point(self) -> None:
        section = _threshold_section(sigma_m2=np.array([0.0, 1e-21, 3e-20, 0.0]))

        with pytest.raises(OutsideTabulatedRangeError, match="zero"):
            interpolate_cross_section(
                section, np.array([500.0]), above=ExtrapolationPolicy.POWER_LAW
            )

    def test_a_grid_entirely_inside_the_table_never_consults_the_policy(self) -> None:
        result = interpolate_cross_section(
            _threshold_section(),
            np.array([20.0, 100.0]),
            above=ExtrapolationPolicy.RAISE,
            below=ExtrapolationPolicy.RAISE,
        )

        np.testing.assert_allclose(result, [1e-21, 3e-20])


# ── the grid itself ─────────────────────────────────────────────────────────────


class TestTheRequestedGrid:
    def test_rejects_a_negative_energy(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            interpolate_cross_section(_cx_section(), np.array([-1.0, 10.0]))

    def test_rejects_a_non_finite_energy(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            interpolate_cross_section(_cx_section(), np.array([np.nan]))

    def test_rejects_a_two_dimensional_grid(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            interpolate_cross_section(_cx_section(), np.zeros((2, 2)))

    def test_an_empty_grid_returns_an_empty_result(self) -> None:
        assert interpolate_cross_section(_cx_section(), np.array([])).size == 0

    def test_the_result_is_not_the_tabulated_array(self) -> None:
        # The section's arrays are write-locked; a result aliasing one of them would
        # hand the caller a read-only array with no explanation.
        section = _cx_section()

        result = interpolate_cross_section(section, section.energy_ev)

        assert result.flags.writeable


# ── whole sets, which is what the Boltzmann solver asks for ─────────────────────


class TestInterpolatingAWholeSet:
    def _set(self) -> CrossSectionSet:
        return CrossSectionSet(
            database="Phelps",
            sections=(_threshold_section(), _cx_section()),
            reference="Phelps database, www.lxcat.net",
        )

    def test_every_section_lands_on_the_same_grid_keyed_by_its_reaction(self) -> None:
        grid = np.array([20.0, 50.0])

        result = interpolate_set(self._set(), grid, above=ExtrapolationPolicy.CONSTANT)

        assert set(result) == {"E + Ar -> E + E + Ar+", "Ar+ + Ar -> Ar + Ar+"}
        assert all(values.shape == grid.shape for values in result.values())

    def test_the_mapping_is_read_only(self) -> None:
        result = interpolate_set(self._set(), np.array([20.0]), above=ExtrapolationPolicy.CONSTANT)

        with pytest.raises(TypeError):
            result["E + Ar -> E + E + Ar+"] = np.zeros(1)  # type: ignore[index]

    def test_two_sections_with_the_same_reaction_are_rejected(self) -> None:
        duplicated = CrossSectionSet(
            database="Phelps",
            sections=(_cx_section(), _cx_section()),
            reference="",
        )

        with pytest.raises(ValueError, match="Ar\\+ \\+ Ar"):
            interpolate_set(duplicated, np.array([10.0]))

    def test_the_policy_reaches_every_section(self) -> None:
        with pytest.raises(OutsideTabulatedRangeError):
            interpolate_set(self._set(), np.array([500.0]))
