"""The LIF rate equations and saturation — doc 04 §3.1 and §3.4, doc 04 V-23.

The verification strategy here is deliberately not internal. Four independent references
are used, and none of them is another function in this package:

1. **The weak-pump limit.** The signal must be exactly linear in laser power as
   ``S -> 0``. Checked as a ratio, so it cannot be satisfied by a wrong constant.
2. **The saturated limit.** ``n_2/n -> g_2/(g_1 + g_2)``, the statistical ceiling a
   two-level system cannot exceed. For the doc 02 §5.3 pair that is ``6/14``.
3. **Power broadening.** The half-width of the saturated profile must be
   ``Gamma sqrt(1 + S)``. Measured off the computed profile by bisection, not read from
   any formula in the module.
4. **Numerical integration of the rate equation itself.** ``scipy.integrate.solve_ivp``
   is run on doc 04 §3.1's ``dn_2/dt`` from an empty upper state; its asymptote must be
   the closed-form steady state and its approach must have the algebraic time constant.
   This is what catches an algebra slip in the steady-state solution, which no amount of
   comparing the module against itself would.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vpl.instruments.lif.rates import (
    lorentzian_response,
    power_broadened_fwhm,
    pump_rate_from_saturation,
    saturation_ceiling,
    steady_state_excited_fraction,
    upper_state_rate,
)
from vpl.instruments.lif.transition import ProbeTransition

_G_LOWER = 8
_G_UPPER = 6
_CEILING = _G_UPPER / (_G_LOWER + _G_UPPER)


class TestTheHomogeneousResponse:
    def test_the_peak_is_unity_on_resonance(self) -> None:
        assert lorentzian_response(np.zeros(1), fwhm_hz=5.0e6)[0] == pytest.approx(1.0)

    def test_it_is_half_at_half_the_full_width(self) -> None:
        response = lorentzian_response(np.array([2.5e6]), fwhm_hz=5.0e6)

        assert response[0] == pytest.approx(0.5, rel=1e-12)

    def test_it_is_symmetric(self) -> None:
        detuning = np.array([-3.0e6, 3.0e6])

        response = lorentzian_response(detuning, fwhm_hz=5.0e6)

        assert response[0] == pytest.approx(response[1], rel=1e-15)

    def test_the_far_wing_falls_as_the_inverse_square_of_detuning(self) -> None:
        near = lorentzian_response(np.array([1.0e9]), fwhm_hz=5.0e6)[0]
        far = lorentzian_response(np.array([2.0e9]), fwhm_hz=5.0e6)[0]

        assert near / far == pytest.approx(4.0, rel=1e-3)

    def test_a_non_positive_width_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            lorentzian_response(np.zeros(1), fwhm_hz=0.0)


class TestTheWeakPumpLimit:
    """Doc 04 §3.4: below saturation the signal is proportional to ``n_1``, and to power."""

    @pytest.mark.physics
    def test_the_excited_fraction_is_linear_in_laser_power(self) -> None:
        one = steady_state_excited_fraction(
            saturation=1.0e-8,
            response=np.ones(1),
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )[0]
        two = steady_state_excited_fraction(
            saturation=2.0e-8,
            response=np.ones(1),
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )[0]

        assert two / one == pytest.approx(2.0, rel=1e-7)

    @pytest.mark.physics
    def test_the_weak_pump_coefficient_is_the_degeneracy_ratio(self) -> None:
        saturation = 1.0e-9

        fraction = steady_state_excited_fraction(
            saturation=saturation,
            response=np.ones(1),
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )[0]

        assert fraction / saturation == pytest.approx(_CEILING, rel=1e-8)

    @pytest.mark.physics
    def test_the_weak_pump_lineshape_is_the_bare_lorentzian(self) -> None:
        """Doc 04 V-22 compares the low-saturation lineshape against an analytic profile.

        Below saturation the excited fraction is ``const x L(delta)``, so the *shape* must
        be the homogeneous Lorentzian with no power broadening at all.
        """
        detuning = np.linspace(-5.0e7, 5.0e7, 401)
        response = lorentzian_response(detuning, fwhm_hz=5.0e6)

        fraction = steady_state_excited_fraction(
            saturation=1.0e-8,
            response=response,
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )

        np.testing.assert_allclose(fraction / fraction.max(), response / response.max(), rtol=1e-7)


class TestTheSaturatedLimit:
    @pytest.mark.physics
    def test_the_excited_fraction_cannot_exceed_the_statistical_ceiling(self) -> None:
        fraction = steady_state_excited_fraction(
            saturation=1.0e12,
            response=np.ones(1),
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )[0]

        assert fraction == pytest.approx(_CEILING, rel=1e-10)
        assert fraction < _CEILING

    @pytest.mark.physics
    def test_equal_degeneracies_saturate_at_one_half(self) -> None:
        fraction = steady_state_excited_fraction(
            saturation=1.0e12, response=np.ones(1), lower_degeneracy=3, upper_degeneracy=3
        )[0]

        assert fraction == pytest.approx(0.5, rel=1e-10)

    @pytest.mark.physics
    @pytest.mark.parametrize("saturation", [0.01, 0.1, 1.0, 10.0])
    def test_the_saturation_curve_matches_the_analytic_two_level_form(
        self, saturation: float
    ) -> None:
        """Doc 04 V-23: "LIF saturation curve vs analytic two-level | within 2 %".

        The analytic form is ``[g_2/(g_1+g_2)] S/(1+S)``, written out here rather than
        imported. The doc 07 F-13 robustness sweep runs exactly this range.
        """
        expected = _CEILING * saturation / (1.0 + saturation)

        fraction = steady_state_excited_fraction(
            saturation=saturation,
            response=np.ones(1),
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )[0]

        assert fraction == pytest.approx(expected, rel=1e-12)


class TestPowerBroadening:
    """Doc 04 §3.4: "At high laser intensity the transition saturates, the measured
    lineshape broadens"."""

    @staticmethod
    def _measured_fwhm(*, saturation: float, fwhm_hz: float) -> float:
        """Half-maximum of the saturated profile, found by bisection on the profile itself."""

        def profile(detuning: float) -> float:
            response = lorentzian_response(np.array([detuning]), fwhm_hz=fwhm_hz)
            return float(
                steady_state_excited_fraction(
                    saturation=saturation,
                    response=response,
                    lower_degeneracy=_G_LOWER,
                    upper_degeneracy=_G_UPPER,
                )[0]
            )

        half = profile(0.0) / 2.0
        low, high = 0.0, fwhm_hz * 1.0e4
        for _ in range(200):
            middle = 0.5 * (low + high)
            if profile(middle) > half:
                low = middle
            else:
                high = middle
        return 2.0 * 0.5 * (low + high)

    @pytest.mark.physics
    @pytest.mark.parametrize("saturation", [0.01, 1.0, 10.0, 100.0])
    def test_the_measured_width_is_gamma_root_one_plus_s(self, saturation: float) -> None:
        fwhm_hz = 5.0e6

        measured = self._measured_fwhm(saturation=saturation, fwhm_hz=fwhm_hz)

        assert measured == pytest.approx(fwhm_hz * math.sqrt(1.0 + saturation), rel=1e-6)

    def test_the_reported_broadened_width_agrees_with_the_measured_one(self) -> None:
        fwhm_hz = 5.0e6

        reported = power_broadened_fwhm(fwhm_hz=fwhm_hz, saturation=9.0)

        assert reported == pytest.approx(
            self._measured_fwhm(saturation=9.0, fwhm_hz=fwhm_hz), rel=1e-6
        )

    def test_there_is_no_broadening_without_a_pump(self) -> None:
        assert power_broadened_fwhm(fwhm_hz=5.0e6, saturation=0.0) == pytest.approx(5.0e6)


class TestTheRateEquationItself:
    """Doc 04 §3.1, integrated numerically rather than solved on paper twice."""

    @pytest.mark.physics
    def test_the_steady_state_is_the_asymptote_of_the_rate_equation(self) -> None:
        transition = ProbeTransition.from_registry()
        gamma = transition.relaxation_rate_per_s
        saturation = 3.0
        pump = pump_rate_from_saturation(
            saturation=saturation,
            relaxation_rate=gamma,
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )

        def derivative(_t: float, state: np.ndarray) -> list[float]:
            return [
                upper_state_rate(
                    n_lower=1.0 - float(state[0]),
                    n_upper=float(state[0]),
                    pump_rate=pump,
                    relaxation_rate=gamma,
                    lower_degeneracy=_G_LOWER,
                    upper_degeneracy=_G_UPPER,
                )
            ]

        solution = solve_ivp(
            derivative, (0.0, 100.0 / gamma), [0.0], rtol=1e-10, atol=1e-14, dense_output=True
        )
        expected = steady_state_excited_fraction(
            saturation=saturation,
            response=np.ones(1),
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )[0]

        assert float(solution.y[0, -1]) == pytest.approx(float(expected), rel=1e-8)

    @pytest.mark.physics
    def test_the_approach_has_the_algebraic_time_constant(self) -> None:
        """``tau = 1 / (R_abs + R_stim + Gamma)`` — the whole bracket of doc 04 §3.1."""
        transition = ProbeTransition.from_registry()
        gamma = transition.relaxation_rate_per_s
        saturation = 3.0
        pump = pump_rate_from_saturation(
            saturation=saturation,
            relaxation_rate=gamma,
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )
        tau = 1.0 / (pump * (1.0 + _G_LOWER / _G_UPPER) + gamma)

        def derivative(_t: float, state: np.ndarray) -> list[float]:
            return [
                upper_state_rate(
                    n_lower=1.0 - float(state[0]),
                    n_upper=float(state[0]),
                    pump_rate=pump,
                    relaxation_rate=gamma,
                    lower_degeneracy=_G_LOWER,
                    upper_degeneracy=_G_UPPER,
                )
            ]

        solution = solve_ivp(derivative, (0.0, 20.0 * tau), [0.0], rtol=1e-11, atol=1e-15)
        steady = float(solution.y[0, -1])

        one_tau = solve_ivp(derivative, (0.0, tau), [0.0], rtol=1e-11, atol=1e-15)

        assert float(one_tau.y[0, -1]) / steady == pytest.approx(1.0 - math.exp(-1.0), rel=1e-5)

    def test_the_rate_vanishes_at_the_steady_state(self) -> None:
        transition = ProbeTransition.from_registry()
        gamma = transition.relaxation_rate_per_s
        pump = pump_rate_from_saturation(
            saturation=0.7,
            relaxation_rate=gamma,
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )
        steady = float(
            steady_state_excited_fraction(
                saturation=0.7,
                response=np.ones(1),
                lower_degeneracy=_G_LOWER,
                upper_degeneracy=_G_UPPER,
            )[0]
        )

        rate = upper_state_rate(
            n_lower=1.0 - steady,
            n_upper=steady,
            pump_rate=pump,
            relaxation_rate=gamma,
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )

        assert rate == pytest.approx(0.0, abs=1e-9 * gamma)

    def test_an_empty_upper_state_is_pumped_upward(self) -> None:
        rate = upper_state_rate(
            n_lower=1.0,
            n_upper=0.0,
            pump_rate=1.0e6,
            relaxation_rate=3.0e7,
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )

        assert rate > 0.0

    def test_an_unpumped_upper_state_decays_at_the_relaxation_rate(self) -> None:
        rate = upper_state_rate(
            n_lower=0.0,
            n_upper=0.25,
            pump_rate=0.0,
            relaxation_rate=3.0e7,
            lower_degeneracy=_G_LOWER,
            upper_degeneracy=_G_UPPER,
        )

        assert rate == pytest.approx(-0.25 * 3.0e7, rel=1e-12)


class TestGuards:
    def test_a_negative_saturation_parameter_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            steady_state_excited_fraction(
                saturation=-1.0,
                response=np.ones(1),
                lower_degeneracy=_G_LOWER,
                upper_degeneracy=_G_UPPER,
            )

    def test_a_negative_response_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            steady_state_excited_fraction(
                saturation=1.0,
                response=np.array([-0.1]),
                lower_degeneracy=_G_LOWER,
                upper_degeneracy=_G_UPPER,
            )

    def test_a_zero_degeneracy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="degenerac"):
            steady_state_excited_fraction(
                saturation=1.0, response=np.ones(1), lower_degeneracy=0, upper_degeneracy=6
            )


class TestMoreGuards:
    def test_a_non_positive_relaxation_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="relaxation rate must be positive"):
            pump_rate_from_saturation(
                saturation=1.0,
                relaxation_rate=0.0,
                lower_degeneracy=_G_LOWER,
                upper_degeneracy=_G_UPPER,
            )

    def test_a_non_positive_width_is_rejected_by_the_broadening_report(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            power_broadened_fwhm(fwhm_hz=0.0, saturation=1.0)

    def test_a_negative_saturation_is_rejected_by_the_broadening_report(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            power_broadened_fwhm(fwhm_hz=5.0e6, saturation=-1.0)

    def test_the_ceiling_is_the_degeneracy_ratio(self) -> None:
        assert saturation_ceiling(
            lower_degeneracy=_G_LOWER, upper_degeneracy=_G_UPPER
        ) == pytest.approx(_CEILING)
