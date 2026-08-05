"""Tests for the doc 05 §2.1 and §4 priors.

Three things are checked and they are checked against three independent authorities:

1. **Closed forms against `scipy.stats`.** The implementations here use
   `scipy.special` primitives only, so `scipy.stats` is a genuinely independent second
   opinion on the parameterisation — which is where the errors are. ``lognorm`` takes a
   shape and a *scale*, ``truncnorm`` takes bounds in *standardised* units, and getting
   either wrong produces a smooth, plausible, wrong density.
2. **Normalisation by quadrature.** A density that is off by a constant factor is
   invisible in a MAP fit and fatal in a model comparison (doc 05 §5 uses evidence for
   model selection). ``quad`` says whether the constant is one.
3. **The transform Jacobian, by quadrature in the unconstrained space.** This is the
   test the module exists for. If the log-Jacobian is dropped, every closed form above
   still matches ``scipy.stats``, every support check still passes, and the unconstrained
   prior integrates to something that is not one. Nothing else catches it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from vpl.inverse.parameters import (
    CONTROL_PARAMETERS,
    ControlParameters,
    log_abs_det_jacobian,
    unconstrained_bounds,
)
from vpl.inverse.priors import (
    ControlPrior,
    LogNormalPrior,
    LogUniformPrior,
    NormalPrior,
    TruncatedNormalPrior,
    UniformPrior,
    bohm_penalty,
    default_control_prior,
    quasineutrality_penalty,
    smoothness_log_prior,
)

#: How closely a closed form must reproduce `scipy.stats`. The brief says ~1e-12; these
#: are the same expression evaluated in a different order, so the agreement is at the
#: level of the last couple of bits and 1e-13 is comfortably met.
SCIPY_TOL = 1e-12

#: How closely a density must integrate to one. `quad` on a smooth 1-D integrand does far
#: better than this; the tolerance is set by the tails it truncates, not by the method.
QUADRATURE_TOL = 1e-9

INTERIOR = ControlParameters(
    n_0=3.7e16,
    T_e=4.25,
    T_i=0.11,
    p=7.5,
    V_w=-310.0,
    phi_RF=1.9,
    gamma_se=0.14,
    kappa=2.3,
)


def _integration_window(prior: object) -> tuple[float, float]:
    """A finite window holding all but ~1e-12 of a marginal's mass.

    QUADPACK's adaptive rule samples a semi-infinite interval on a fixed transformed grid
    and will report zero for a density that is narrow and far from the origin — which the
    2 %-wide pressure prior at 5 mTorr and the 1 %-wide bias prior at -250 V both are. The
    window is derived from the prior, not hard-coded, so it follows the prior if it moves.
    """
    low, high = prior.support  # type: ignore[attr-defined]
    if math.isfinite(low) and math.isfinite(high):
        return (low, high)
    median = prior.median  # type: ignore[attr-defined]
    if isinstance(prior, LogNormalPrior):
        span = math.exp(12.0 * prior.log_sigma)
        return (median / span, median * span)
    assert isinstance(prior, NormalPrior)
    return (median - 12.0 * prior.sigma, median + 12.0 * prior.sigma)


def _quad_split(density, low: float, high: float, transform, prior) -> float:  # type: ignore[no-untyped-def]
    """Integrate over `[low, high]`, splitting at the image of the prior median.

    Same reason as `_integration_window`: an infinite interval whose integrand is a narrow
    spike far from zero defeats the global adaptive rule. Splitting at the mass puts the
    spike on an endpoint of each half, where QUADPACK's mapping concentrates its nodes.
    """
    centre = transform.to_unconstrained(prior.median)
    left, _ = integrate.quad(density, low, centre, limit=400)
    right, _ = integrate.quad(density, centre, high, limit=400)
    return float(left + right)


class TestClosedFormsAgainstScipy:
    """Every log-density against `scipy.stats`, to 1e-12."""

    def test_normal(self) -> None:
        prior = NormalPrior(mean=-250.0, sigma=2.5)

        for x in (-260.0, -250.0, -247.3, -240.0):
            assert prior.log_prob(x) == pytest.approx(
                float(stats.norm.logpdf(x, loc=-250.0, scale=2.5)), rel=SCIPY_TOL
            )

    def test_uniform(self) -> None:
        prior = UniformPrior(low=1.0, high=5.0)

        for x in (1.0, 2.3, 5.0):
            assert prior.log_prob(x) == pytest.approx(
                float(stats.uniform.logpdf(x, loc=1.0, scale=4.0)), rel=SCIPY_TOL
            )

    def test_log_uniform(self) -> None:
        prior = LogUniformPrior(low=1.0e15, high=1.0e19)

        for x in (1.0e15, 3.7e16, 1.0e17, 1.0e19):
            assert prior.log_prob(x) == pytest.approx(
                float(stats.loguniform.logpdf(x, 1.0e15, 1.0e19)), rel=SCIPY_TOL
            )

    def test_log_normal(self) -> None:
        # `lognorm` takes the log-scale sigma as its shape and the *median* as `scale`.
        # Passing the mean of the log as `scale` instead is the classic error and would
        # shift the T_e prior by a factor of e^3.
        prior = LogNormalPrior(median=3.0, log_sigma=0.4)

        for x in (0.5, 1.0, 3.0, 8.0, 30.0):
            assert prior.log_prob(x) == pytest.approx(
                float(stats.lognorm.logpdf(x, 0.4, scale=3.0)), rel=SCIPY_TOL
            )

    def test_truncated_normal(self) -> None:
        # `truncnorm` bounds are standardised: a = (low - mean)/sigma. Passing the raw
        # bounds is the classic error and silently renormalises the density.
        prior = TruncatedNormalPrior(mean=0.10, sigma=0.03, low=0.0, high=0.3)
        a, b = (0.0 - 0.10) / 0.03, (0.3 - 0.10) / 0.03

        for x in (0.0, 0.02, 0.10, 0.19, 0.3):
            assert prior.log_prob(x) == pytest.approx(
                float(stats.truncnorm.logpdf(x, a, b, loc=0.10, scale=0.03)), rel=SCIPY_TOL
            )

    def test_truncated_normal_matches_when_the_truncation_is_severe(self) -> None:
        # The stable branch of log(Phi(b) - Phi(a)) is only exercised when both bounds sit
        # far into one tail, which is where the naive difference of two ~1.0 numbers loses
        # every significant digit.
        prior = TruncatedNormalPrior(mean=0.0, sigma=1.0, low=4.0, high=9.0)
        expected = float(stats.truncnorm.logpdf(5.0, 4.0, 9.0, loc=0.0, scale=1.0))

        assert prior.log_prob(5.0) == pytest.approx(expected, rel=SCIPY_TOL)


class TestSupports:
    def test_a_value_outside_the_support_has_zero_density(self) -> None:
        cases = [
            (UniformPrior(low=1.0, high=5.0), 5.5),
            (LogUniformPrior(low=1.0e15, high=1.0e19), 1.0e20),
            (LogNormalPrior(median=3.0, log_sigma=0.4), -1.0),
            (TruncatedNormalPrior(mean=0.10, sigma=0.03, low=0.0, high=0.3), 0.31),
        ]

        for prior, outside in cases:
            assert prior.log_prob(outside) == -math.inf

    def test_a_normal_prior_has_unbounded_support(self) -> None:
        assert NormalPrior(mean=0.0, sigma=1.0).support == (-math.inf, math.inf)

    def test_a_degenerate_scale_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            NormalPrior(mean=0.0, sigma=0.0)
        with pytest.raises(ValueError, match="positive"):
            LogNormalPrior(median=3.0, log_sigma=-0.1)

    def test_a_descending_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            UniformPrior(low=5.0, high=1.0)
        with pytest.raises(ValueError, match="ascending"):
            LogUniformPrior(low=1.0e19, high=1.0e15)

    def test_a_non_positive_log_uniform_bound_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LogUniformPrior(low=0.0, high=1.0)


class TestNormalisationByQuadrature:
    """Each prior integrates to one over its own support."""

    def test_every_marginal_of_the_default_prior_is_normalised(self) -> None:
        prior = default_control_prior()

        for spec in CONTROL_PARAMETERS:
            marginal = prior.marginal(spec.name)
            low, high = _integration_window(marginal)
            mass, _ = integrate.quad(
                lambda x, m=marginal: math.exp(m.log_prob(x)), low, high, limit=400
            )

            assert mass == pytest.approx(1.0, abs=QUADRATURE_TOL), spec.name

    def test_a_truncated_normal_is_renormalised_by_its_truncated_mass(self) -> None:
        # The point of the truncation: a normal restricted to [0, 0.3] must integrate to
        # one over [0, 0.3], not to the 0.99957 the untruncated density leaves there.
        prior = TruncatedNormalPrior(mean=0.10, sigma=0.03, low=0.0, high=0.3)
        mass, _ = integrate.quad(lambda x: math.exp(prior.log_prob(x)), 0.0, 0.3)

        assert mass == pytest.approx(1.0, abs=QUADRATURE_TOL)

    def test_the_truncation_correction_is_the_documented_size(self) -> None:
        # A cross-check on the direction of the correction, computed independently: the
        # normal N(0.10, 0.03) puts 4.2779e-4 of its mass below zero, so the truncated
        # density is larger than the untruncated one by exactly 1/(1 - 4.2779e-4).
        untruncated = NormalPrior(mean=0.10, sigma=0.03)
        truncated = TruncatedNormalPrior(mean=0.10, sigma=0.03, low=0.0, high=0.3)
        retained = float(stats.norm.cdf(0.3, 0.10, 0.03) - stats.norm.cdf(0.0, 0.10, 0.03))

        difference = truncated.log_prob(0.12) - untruncated.log_prob(0.12)

        # 1 - Phi(-10/3) - Phi(-20/3) = 1 - 4.29060e-4 - 1.3065e-11, from a normal table.
        assert retained == pytest.approx(1.0 - 4.29060e-4 - 1.3065e-11, abs=1e-9)
        assert difference == pytest.approx(-math.log(retained), rel=1e-13)


class TestTheJacobianInUnconstrainedSpace:
    """The single most important file in this package, per the brief.

    A prior stated in doc 05 §2.1's units is not a density in the space a sampler works
    in. The change of variables supplies the missing factor, and without it the posterior
    is wrong in a way that no other test in this repository would catch.
    """

    def test_each_transformed_marginal_integrates_to_one_over_the_reals(self) -> None:
        prior = default_control_prior()
        lower, upper = unconstrained_bounds()

        for index, spec in enumerate(CONTROL_PARAMETERS):
            marginal = prior.marginal(spec.name)
            transform = spec.transform

            def density(u: float, m=marginal, t=transform) -> float:  # type: ignore[no-untyped-def]
                return math.exp(m.log_prob(t.to_constrained(u)) + t.log_abs_det_jacobian(u))

            mass = _quad_split(
                density, float(lower[index]), float(upper[index]), spec.transform, marginal
            )

            assert mass == pytest.approx(1.0, abs=1e-8), spec.name

    def test_dropping_the_jacobian_would_not_integrate_to_one(self) -> None:
        # The negative control. If this test ever passes for a parameter, the Jacobian for
        # that parameter is a no-op and the corresponding assertion above proves nothing.
        prior = default_control_prior()
        lower, upper = unconstrained_bounds()

        for index, spec in enumerate(CONTROL_PARAMETERS):
            if spec.name == "V_w":
                continue  # identity transform; the Jacobian is genuinely one
            marginal = prior.marginal(spec.name)
            transform = spec.transform

            def wrong(u: float, m=marginal, t=transform) -> float:  # type: ignore[no-untyped-def]
                return math.exp(m.log_prob(t.to_constrained(u)))

            # A finite window that contains essentially all the mass of the *correct*
            # density. Restricting it is not a weakening of the control: the correct
            # integral over this same window is 1 to better than 1e-9, so any departure
            # here is the missing Jacobian and nothing else. The unrestricted integral
            # for several of these is genuinely divergent, which is the point.
            centre = transform.to_unconstrained(marginal.median)
            low = max(float(lower[index]), centre - 40.0)
            high = min(float(upper[index]), centre + 40.0)
            mass, _ = integrate.quad(wrong, low, high, limit=400)

            assert abs(mass - 1.0) > 1e-3, spec.name

    def test_the_joint_unconstrained_density_is_the_sum_plus_the_log_determinant(self) -> None:
        prior = default_control_prior()
        u = INTERIOR.to_unconstrained()

        expected = prior.log_prob(INTERIOR) + log_abs_det_jacobian(u)

        assert prior.log_prob_unconstrained(u) == pytest.approx(expected, rel=1e-14)

    def test_the_joint_constrained_density_is_the_sum_of_the_marginals(self) -> None:
        prior = default_control_prior()

        expected = sum(
            prior.marginal(spec.name).log_prob(float(getattr(INTERIOR, spec.name)))
            for spec in CONTROL_PARAMETERS
        )

        assert prior.log_prob(INTERIOR) == pytest.approx(expected, rel=1e-14)

    def test_a_point_outside_the_support_is_minus_infinity_not_an_exception(self) -> None:
        # An optimiser stepping out of a log-uniform box must be told "no" by the
        # objective, not by a traceback: L-BFGS-B backtracks on -inf and cannot on a raise.
        prior = default_control_prior()
        u = INTERIOR.to_unconstrained()
        u[0] = math.log(1.0e21)

        assert prior.log_prob_unconstrained(u) == -math.inf

    def test_the_joint_is_finite_at_the_reference_point_in_its_own_units(self) -> None:
        assert math.isfinite(default_control_prior().log_prob(ControlParameters.reference()))

    def test_the_reference_point_is_not_a_usable_unconstrained_start(self) -> None:
        # Surfaced by this test suite rather than by a debugging session downstream.
        # `eedf.kappa`'s registered nominal of 1.0 is a Maxwellian EEDF and is exactly the
        # lower edge of doc 05 §2.1's `uniform [1, 5]`; a logit sends an edge to -inf. The
        # behaviour is correct, the trap is real, and `ControlPrior.median()` is the answer.
        prior = default_control_prior()
        u = ControlParameters.reference().to_unconstrained()

        assert u[-1] == -math.inf
        assert prior.log_prob_unconstrained(u) == -math.inf

    def test_the_prior_median_is_a_usable_unconstrained_start(self) -> None:
        prior = default_control_prior()
        start = prior.median()

        assert math.isfinite(prior.log_prob(start))
        assert np.all(np.isfinite(start.to_unconstrained()))
        assert math.isfinite(prior.log_prob_unconstrained(start.to_unconstrained()))


class TestTheDocumentedLevelAPriors:
    """The default prior is doc 05 §2.1's table and not a paraphrase of it."""

    def test_the_density_prior_is_log_uniform_over_the_documented_decades(self) -> None:
        marginal = default_control_prior().marginal("n_0")

        assert isinstance(marginal, LogUniformPrior)
        assert marginal.support == pytest.approx((1.0e15, 1.0e19), rel=1e-12)

    def test_the_electron_temperature_prior_is_log_normal_at_3_ev_with_sigma_0p4(self) -> None:
        marginal = default_control_prior().marginal("T_e")

        assert isinstance(marginal, LogNormalPrior)
        assert marginal.median == pytest.approx(3.0)
        assert marginal.log_sigma == pytest.approx(0.4)

    def test_the_ion_temperature_prior_is_log_uniform_over_the_documented_range(self) -> None:
        marginal = default_control_prior().marginal("T_i")

        assert isinstance(marginal, LogUniformPrior)
        assert marginal.support == pytest.approx((0.02, 0.5), rel=1e-12)

    def test_the_pressure_prior_is_the_gauge_reading_to_two_percent(self) -> None:
        # doc 05 §2.1: "informative — gauge-measured, sigma = 2 %". Relative, so it is
        # applied on the log scale where 2 % is what it says.
        marginal = default_control_prior(gauge_pressure=8.0).marginal("p")

        assert isinstance(marginal, LogNormalPrior)
        assert marginal.median == pytest.approx(8.0)
        assert marginal.log_sigma == pytest.approx(0.02)

    def test_the_bias_prior_is_the_supply_reading_to_one_percent(self) -> None:
        # doc 05 §2.1: "informative — supply-measured, sigma = 1 %", on a quantity that is
        # negative, so the sigma is one percent of the *magnitude*.
        marginal = default_control_prior(supply_bias=-400.0).marginal("V_w")

        assert isinstance(marginal, NormalPrior)
        assert marginal.mean == pytest.approx(-400.0)
        assert marginal.sigma == pytest.approx(4.0)

    def test_the_phase_prior_is_uniform_over_the_full_turn(self) -> None:
        marginal = default_control_prior().marginal("phi_RF")

        assert isinstance(marginal, UniformPrior)
        assert marginal.support == pytest.approx((0.0, 2.0 * math.pi), rel=1e-14)

    def test_the_secondary_emission_prior_is_normal_at_0p10_plus_minus_0p03(self) -> None:
        marginal = default_control_prior().marginal("gamma_se")

        assert isinstance(marginal, TruncatedNormalPrior)
        assert marginal.mean == pytest.approx(0.10)
        assert marginal.sigma == pytest.approx(0.03)

    def test_the_eedf_shape_prior_is_uniform_from_maxwellian_to_beyond_druyvesteyn(self) -> None:
        marginal = default_control_prior().marginal("kappa")

        assert isinstance(marginal, UniformPrior)
        assert marginal.support == pytest.approx((1.0, 5.0), rel=1e-14)

    def test_every_marginal_support_matches_the_parameter_support(self) -> None:
        # If they disagree, the joint prior is not normalised over the space the parameter
        # vector can represent, and no single-parameter test would show it.
        prior = default_control_prior()

        for spec in CONTROL_PARAMETERS:
            if spec.support is None:
                continue
            assert prior.marginal(spec.name).support == pytest.approx(spec.support, rel=1e-14)

    def test_an_unknown_parameter_name_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="n_e"):
            default_control_prior().marginal("n_e")

    def test_a_wrong_length_marginal_tuple_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="8"):
            ControlPrior(marginals=(NormalPrior(mean=0.0, sigma=1.0),))


class TestTheExplicitPhysicsPriors:
    """doc 05 §4.2 — the soft penalties, and what each is anchoring."""

    def test_the_bohm_penalty_is_silent_when_the_criterion_holds(self) -> None:
        # doc 05 §4.2: "soft penalty on u_s/c_s < 1". A supersonic sheath edge satisfies
        # the criterion and must not be penalised for exceeding it.
        u_s = np.array([2400.0, 3000.0, 2701.0])
        c_s = np.full(3, 2700.0)

        assert bohm_penalty(u_s, c_s, weight=10.0) < 0.0
        assert bohm_penalty(c_s, c_s, weight=10.0) == 0.0
        assert bohm_penalty(u_s * 2.0, c_s, weight=10.0) == 0.0

    def test_the_bohm_penalty_is_a_quadratic_hinge(self) -> None:
        # -w/2 * sum(max(0, 1 - u_s/c_s))^2, computed here by hand.
        u_s = np.array([1350.0, 2700.0])
        c_s = np.array([2700.0, 2700.0])

        assert bohm_penalty(u_s, c_s, weight=8.0) == pytest.approx(-8.0 / 2.0 * 0.25, rel=1e-14)

    def test_the_bohm_penalty_rejects_a_non_positive_sound_speed(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bohm_penalty(np.ones(2), np.zeros(2), weight=1.0)

    def test_the_quasineutrality_penalty_is_zero_in_a_neutral_bulk(self) -> None:
        # doc 05 §4.2: "soft penalty on (n_i - n_e)/n_i for z > 5 z_s". The caller has
        # already selected the bulk region; this scores the residual there.
        n_i = np.array([1.0e17, 9.0e16])

        assert quasineutrality_penalty(n_i, n_i, weight=100.0) == 0.0

    def test_the_quasineutrality_penalty_is_quadratic_in_the_relative_residual(self) -> None:
        n_i = np.array([1.0e17, 1.0e17])
        n_e = np.array([0.9e17, 1.1e17])

        expected = -100.0 / 2.0 * (0.1**2 + 0.1**2)

        assert quasineutrality_penalty(n_i, n_e, weight=100.0) == pytest.approx(expected, rel=1e-12)

    def test_the_smoothness_prior_is_a_zero_mean_gaussian_on_the_coefficients(self) -> None:
        # doc 05 §2.3: alpha ~ N(0, tau^2), the hierarchical shrinkage prior; doc 05 §4.2
        # lists the same object as "profile smoothness | Tikhonov on discrepancy
        # coefficients". Checked against scipy.stats, summed over coefficients.
        alpha = np.array([0.12, -0.4, 0.03, 0.0])
        tau = 0.25

        expected = float(np.sum(stats.norm.logpdf(alpha, loc=0.0, scale=tau)))

        assert smoothness_log_prior(alpha, tau=tau) == pytest.approx(expected, rel=SCIPY_TOL)

    def test_the_smoothness_prior_rejects_a_non_positive_scale(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            smoothness_log_prior(np.zeros(3), tau=0.0)

    def test_a_negative_penalty_weight_is_rejected(self) -> None:
        # A negative weight turns a penalty into a reward for violating the physics.
        with pytest.raises(ValueError, match="negative"):
            bohm_penalty(np.ones(2), np.ones(2), weight=-1.0)
        with pytest.raises(ValueError, match="negative"):
            quasineutrality_penalty(np.ones(2), np.ones(2), weight=-1.0)

    def test_the_penalties_reject_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            bohm_penalty(np.ones(3), np.ones(2), weight=1.0)
        with pytest.raises(ValueError, match="shape"):
            quasineutrality_penalty(np.ones(3), np.ones(2), weight=1.0)


class TestMediansAgainstScipy:
    """`Prior.median` against `scipy.stats`'s own inverse CDF.

    The median is what an engine initialises from, so a wrong one is a wrong starting
    point rather than a wrong answer — but a log-uniform median computed as `(low+high)/2`
    would start an `n_0` fit at 5e18 instead of 1e17, a factor of 50, and the identifiability
    map (doc 05 §6.2) is not flat over that distance.
    """

    def test_normal(self) -> None:
        assert NormalPrior(mean=-250.0, sigma=2.5).median == pytest.approx(
            float(stats.norm.ppf(0.5, loc=-250.0, scale=2.5)), rel=1e-13
        )

    def test_uniform(self) -> None:
        assert UniformPrior(low=1.0, high=5.0).median == pytest.approx(
            float(stats.uniform.ppf(0.5, loc=1.0, scale=4.0)), rel=1e-13
        )

    def test_log_uniform_is_the_geometric_mean(self) -> None:
        prior = LogUniformPrior(low=1.0e15, high=1.0e19)

        assert prior.median == pytest.approx(
            float(stats.loguniform.ppf(0.5, 1.0e15, 1.0e19)), rel=1e-13
        )
        assert prior.median == pytest.approx(1.0e17, rel=1e-13)
        assert prior.median != pytest.approx(0.5 * (1.0e15 + 1.0e19), rel=1e-3)

    def test_log_normal(self) -> None:
        assert LogNormalPrior(median=3.0, log_sigma=0.4).median == pytest.approx(
            float(stats.lognorm.ppf(0.5, 0.4, scale=3.0)), rel=1e-13
        )

    def test_truncated_normal(self) -> None:
        prior = TruncatedNormalPrior(mean=0.10, sigma=0.03, low=0.0, high=0.3)
        a, b = (0.0 - 0.10) / 0.03, (0.3 - 0.10) / 0.03

        assert prior.median == pytest.approx(
            float(stats.truncnorm.ppf(0.5, a, b, loc=0.10, scale=0.03)), rel=1e-12
        )

    def test_a_severely_truncated_median_is_not_the_parent_mean(self) -> None:
        prior = TruncatedNormalPrior(mean=0.0, sigma=1.0, low=4.0, high=9.0)

        assert prior.median == pytest.approx(
            float(stats.truncnorm.ppf(0.5, 4.0, 9.0, loc=0.0, scale=1.0)), rel=1e-9
        )
        assert prior.median > 4.0

    def test_every_default_median_is_strictly_inside_its_support(self) -> None:
        prior = default_control_prior()

        for spec in CONTROL_PARAMETERS:
            marginal = prior.marginal(spec.name)
            low, high = marginal.support

            assert low < marginal.median < high, spec.name
