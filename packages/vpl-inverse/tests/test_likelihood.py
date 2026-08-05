"""Tests for the per-channel likelihoods — doc 05 §3.

Every closed form is checked against an independent authority:

* Poisson and Student-t against ``scipy.stats``;
* the heteroscedastic Gaussian against ``scipy.stats.norm.logpdf`` summed by hand;
* the correlated Gaussian against ``-0.5 rᵀΣ⁻¹r - 0.5 logdet(2πΣ)`` assembled from
  ``numpy.linalg.inv`` and ``numpy.linalg.slogdet``, *and* against
  ``scipy.stats.multivariate_normal``.

Two tests do something different and are the most interesting in the file:
:meth:`TestColouredNoise.test_a_diagonal_covariance_overstates_the_information` and
:meth:`TestSharedSystematics.test_a_shared_calibration_error_does_not_average_down`. Those
check the two *claims* doc 05 §3.1 and doc 06 §3 make about correlated noise, rather than
checking an implementation against a formula. A covariance builder that produced a matrix
satisfying neither claim would pass every other test here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from vpl.core.state import (
    AcquisitionWindow,
    Measurement,
    MeasurementSet,
    Observable,
)
from vpl.core.units import Q_
from vpl.inverse.likelihood import (
    OES_GAUSSIAN_SWITCH_COUNTS,
    ChannelMismatchError,
    CorrelatedGaussianChannel,
    GaussianChannel,
    MixtureChannel,
    PoissonChannel,
    StudentTChannel,
    SwitchedPoissonGaussianChannel,
    coloured_noise_covariance,
    correlated_gaussian_log_likelihood,
    detection_mask,
    diagonal_covariance,
    gaussian_log_likelihood,
    outlier_mixture_log_likelihood,
    poisson_log_likelihood,
    shared_systematic_covariance,
    student_t_log_likelihood,
    switched_poisson_gaussian_log_likelihood,
    total_log_likelihood,
)

SCIPY_TOL = 1e-12

WINDOW = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(2.0e-9, "s"))
LONG_WINDOW = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(700.0, "s"))


def _measurement(
    values: list[float],
    uncertainty: list[float],
    *,
    instrument_id: str = "oes",
    window: AcquisitionWindow = WINDOW,
    units: str = "count",
) -> Measurement:
    return Measurement(
        instrument_id=instrument_id,
        values=np.array(values),
        uncertainty=np.array(uncertainty),
        units=units,
        window=window,
    )


def _observable(
    values: list[float],
    *,
    instrument_id: str = "oes",
    window: AcquisitionWindow = WINDOW,
    units: str = "count",
) -> Observable:
    return Observable(
        instrument_id=instrument_id,
        values=np.array(values),
        units=units,
        window=window,
    )


class TestPoisson:
    """doc 05 §3.1 Thomson: "Poisson on photoelectron counts"."""

    def test_it_matches_scipy_poisson_logpmf(self) -> None:
        counts = np.array([0.0, 1.0, 2.0, 7.0, 40.0])
        expected = np.array([0.008, 0.5, 3.0, 4.2, 37.5])

        assert poisson_log_likelihood(counts, expected) == pytest.approx(
            float(np.sum(stats.poisson.logpmf(counts, expected))), rel=SCIPY_TOL
        )

    def test_it_matches_scipy_in_the_thomson_single_photoelectron_regime(self) -> None:
        # doc 02 §7.1: 0.008 pe/channel/shot. This is the regime the Gaussian is invalid
        # in, so it is the regime the Poisson has to be right in.
        counts = np.zeros(64)
        counts[3] = 1.0
        expected = np.full(64, 0.008)

        assert poisson_log_likelihood(counts, expected) == pytest.approx(
            float(np.sum(stats.poisson.logpmf(counts, expected))), rel=SCIPY_TOL
        )

    def test_a_zero_rate_with_zero_counts_is_certain(self) -> None:
        assert poisson_log_likelihood(np.zeros(3), np.zeros(3)) == 0.0

    def test_a_zero_rate_with_a_count_is_impossible(self) -> None:
        assert poisson_log_likelihood(np.array([1.0]), np.array([0.0])) == -math.inf

    def test_a_negative_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            poisson_log_likelihood(np.array([1.0]), np.array([-0.5]))

    def test_a_fractional_count_is_rejected(self) -> None:
        # scipy returns -inf here, which would silently annihilate the posterior. A
        # fractional "count" means a background was subtracted before the likelihood, and
        # that is a modelling error worth a traceback.
        with pytest.raises(ValueError, match="integer"):
            poisson_log_likelihood(np.array([1.5]), np.array([2.0]))

    def test_a_negative_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            poisson_log_likelihood(np.array([-1.0]), np.array([2.0]))


class TestHeteroscedasticGaussian:
    """doc 05 §3.1 LIF: "Gaussian with heteroscedastic variance"."""

    def test_it_matches_scipy_norm_logpdf(self) -> None:
        observed = np.array([1.4, -0.2, 9.8, 3.3])
        predicted = np.array([1.0, 0.0, 10.0, 3.0])
        sigma = np.array([0.2, 0.05, 1.5, 0.4])

        assert gaussian_log_likelihood(observed, predicted, sigma) == pytest.approx(
            float(np.sum(stats.norm.logpdf(observed, loc=predicted, scale=sigma))), rel=SCIPY_TOL
        )

    def test_the_variance_is_genuinely_per_sample(self) -> None:
        # The failure this guards: broadcasting one sigma across every sample. It tightens
        # the fit and nothing in the output says so (see `vpl.core.state.measurement`).
        observed = np.array([1.0, 1.0])
        predicted = np.array([0.0, 0.0])

        narrow = gaussian_log_likelihood(observed, predicted, np.array([0.1, 1.0]))
        uniform = gaussian_log_likelihood(observed, predicted, np.array([1.0, 1.0]))

        assert narrow < uniform

    def test_a_zero_sigma_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            gaussian_log_likelihood(np.zeros(2), np.zeros(2), np.array([1.0, 0.0]))

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            gaussian_log_likelihood(np.zeros(3), np.zeros(2), np.ones(3))


class TestCorrelatedGaussian:
    """doc 05 §3.1 interferometry: "Gaussian, with a correlated noise covariance"."""

    def test_it_matches_the_explicit_quadratic_form(self) -> None:
        # -0.5 r' Sigma^-1 r - 0.5 logdet(2 pi Sigma), assembled from `inv` and `slogdet`
        # rather than from the Cholesky the implementation uses.
        times = np.linspace(0.0, 1.0e-3, 12)
        covariance = coloured_noise_covariance(times, sigma=2.0e-4, correlation_time=2.5e-4)
        observed = np.linspace(-1.0, 1.0, 12) * 3.0e-4
        predicted = np.zeros(12)

        residual = observed - predicted
        quadratic = float(residual @ np.linalg.inv(covariance) @ residual)
        _, logdet = np.linalg.slogdet(covariance)
        expected = -0.5 * quadratic - 0.5 * (residual.size * math.log(2.0 * math.pi) + logdet)

        assert correlated_gaussian_log_likelihood(observed, predicted, covariance) == pytest.approx(
            expected, rel=SCIPY_TOL
        )

    def test_it_matches_scipy_multivariate_normal(self) -> None:
        times = np.linspace(0.0, 1.0e-3, 9)
        covariance = coloured_noise_covariance(times, sigma=2.0e-4, correlation_time=3.0e-4)
        observed = np.array([1.0, -2.0, 0.5, 0.1, -0.4, 2.2, -1.1, 0.0, 0.9]) * 1.0e-4
        predicted = np.full(9, 1.0e-5)

        expected = float(stats.multivariate_normal.logpdf(observed, mean=predicted, cov=covariance))

        assert correlated_gaussian_log_likelihood(observed, predicted, covariance) == pytest.approx(
            expected, rel=SCIPY_TOL
        )

    def test_a_diagonal_covariance_reduces_to_the_heteroscedastic_form(self) -> None:
        observed = np.array([1.4, -0.2, 9.8])
        predicted = np.array([1.0, 0.0, 10.0])
        sigma = np.array([0.2, 0.05, 1.5])

        assert correlated_gaussian_log_likelihood(
            observed, predicted, diagonal_covariance(sigma)
        ) == pytest.approx(gaussian_log_likelihood(observed, predicted, sigma), rel=1e-13)

    def test_a_non_positive_definite_covariance_is_rejected(self) -> None:
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="positive definite"):
            correlated_gaussian_log_likelihood(np.zeros(2), np.zeros(2), singular)

    def test_an_asymmetric_covariance_is_rejected(self) -> None:
        asymmetric = np.array([[1.0, 0.3], [0.1, 1.0]])

        with pytest.raises(ValueError, match="symmetric"):
            correlated_gaussian_log_likelihood(np.zeros(2), np.zeros(2), asymmetric)

    def test_a_non_square_covariance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            correlated_gaussian_log_likelihood(np.zeros(2), np.zeros(2), np.ones((2, 3)))


class TestColouredNoise:
    """doc 05 §3.1: "Vibration noise is coloured, not white"."""

    def test_the_kernel_is_exponential_in_the_lag(self) -> None:
        times = np.array([0.0, 1.0, 3.0])
        covariance = coloured_noise_covariance(times, sigma=2.0, correlation_time=2.0)

        assert covariance[0, 0] == pytest.approx(4.0, rel=1e-14)
        assert covariance[0, 1] == pytest.approx(4.0 * math.exp(-0.5), rel=1e-14)
        assert covariance[0, 2] == pytest.approx(4.0 * math.exp(-1.5), rel=1e-14)

    def test_a_short_correlation_time_recovers_the_diagonal(self) -> None:
        times = np.arange(6.0)
        covariance = coloured_noise_covariance(times, sigma=1.0, correlation_time=1.0e-6)

        assert covariance == pytest.approx(np.eye(6), abs=1e-9)

    def test_a_diagonal_covariance_overstates_the_information(self) -> None:
        # The doc 05 §3.1 claim, tested rather than asserted. Fisher information about a
        # constant offset is 1' Sigma^-1 1. With correlated noise it is strictly smaller
        # than with the same per-sample variances and no correlation — which is exactly
        # what "a diagonal covariance would overstate the information" means.
        times = np.arange(32.0)
        correlated = coloured_noise_covariance(times, sigma=1.0, correlation_time=5.0)
        diagonal = diagonal_covariance(np.ones(32))
        ones = np.ones(32)

        correlated_information = float(ones @ np.linalg.solve(correlated, ones))
        diagonal_information = float(ones @ np.linalg.solve(diagonal, ones))

        assert diagonal_information == pytest.approx(32.0, rel=1e-12)
        assert correlated_information < diagonal_information
        # Quantitatively, and against a closed form rather than an inequality. The inverse
        # of an AR(1) covariance is tridiagonal, and summing it gives exactly
        #     1' Sigma^-1 1  =  [N (1 - rho) + 2 rho] / (1 + rho)
        # with rho = exp(-dt/tau) = exp(-1/5) = 0.818731. For N = 32 that is 4.0897
        # effective samples instead of 32 — the diagonal covariance would claim 7.8x the
        # information that is there, i.e. an error bar 2.8x too small.
        rho = math.exp(-1.0 / 5.0)
        assert correlated_information == pytest.approx(
            (32.0 * (1.0 - rho) + 2.0 * rho) / (1.0 + rho), rel=1e-12
        )
        assert correlated_information == pytest.approx(4.0897078, rel=1e-7)

    def test_a_non_positive_correlation_time_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            coloured_noise_covariance(np.arange(3.0), sigma=1.0, correlation_time=0.0)

    def test_unsorted_times_are_accepted_and_symmetric(self) -> None:
        covariance = coloured_noise_covariance(
            np.array([3.0, 0.0, 1.0]), sigma=1.0, correlation_time=2.0
        )

        assert covariance == pytest.approx(covariance.T, abs=0.0)


class TestSharedSystematics:
    """doc 06 §3 and §4 terms 3 and 9 — a calibration error that does not average down."""

    def test_a_shared_calibration_error_does_not_average_down(self) -> None:
        # doc 06 §3: "Treating a shared calibration error as independent per data point
        # makes it appear to shrink as 1/sqrt(N) with accumulation. It does not shrink at
        # all." The variance of the sample mean under a rank-one shared covariance is
        # exactly the relative sigma squared times the mean prediction squared, for any N.
        for n in (4, 64, 1024):
            predicted = np.full(n, 3.0)
            covariance = shared_systematic_covariance(predicted, relative_sigma=0.07)
            weights = np.full(n, 1.0 / n)

            variance_of_mean = float(weights @ covariance @ weights)

            assert variance_of_mean == pytest.approx((0.07 * 3.0) ** 2, rel=1e-13)

    def test_an_independent_error_does_average_down(self) -> None:
        # The control: the same 7 % applied per-sample and independently *does* shrink,
        # which is the mistake doc 06 §3 is warning about.
        for n, expected in ((4, 4), (64, 64), (1024, 1024)):
            predicted = np.full(n, 3.0)
            covariance = diagonal_covariance(0.07 * predicted)
            weights = np.full(n, 1.0 / n)

            variance_of_mean = float(weights @ covariance @ weights)

            assert variance_of_mean == pytest.approx((0.07 * 3.0) ** 2 / expected, rel=1e-13)

    def test_it_is_rank_one(self) -> None:
        covariance = shared_systematic_covariance(np.array([1.0, 2.0, 4.0]), relative_sigma=0.06)

        assert np.linalg.matrix_rank(covariance) == 1

    def test_it_sums_with_a_statistical_term_into_a_usable_covariance(self) -> None:
        # doc 06 §4 wants terms attributable separately and combined in one covariance:
        # term 8 (Thomson photon statistics, 3 %) plus term 3 (Rayleigh calibration, 7 %).
        predicted = np.full(5, 100.0)
        statistical = diagonal_covariance(0.03 * predicted)
        systematic = shared_systematic_covariance(predicted, relative_sigma=0.07)

        combined = statistical + systematic
        value = correlated_gaussian_log_likelihood(predicted + 1.0, predicted, combined)

        assert math.isfinite(value)
        assert value < correlated_gaussian_log_likelihood(predicted, predicted, combined)

    def test_a_negative_relative_sigma_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            shared_systematic_covariance(np.ones(3), relative_sigma=-0.01)


class TestTheOesSwitch:
    """doc 05 §3.1 OES: "Poisson for weak lines; Gaussian for bright lines above ~100 pe"."""

    def test_the_threshold_comes_from_the_registry(self) -> None:
        assert pytest.approx(100.0) == OES_GAUSSIAN_SWITCH_COUNTS

    def test_below_the_threshold_it_is_exactly_the_poisson(self) -> None:
        counts = np.array([0.0, 3.0, 11.0])
        expected = np.array([0.5, 2.5, 9.0])

        assert switched_poisson_gaussian_log_likelihood(
            counts, expected, threshold=100.0
        ) == pytest.approx(poisson_log_likelihood(counts, expected), rel=1e-14)

    def test_above_the_threshold_it_is_the_gaussian_with_poisson_variance(self) -> None:
        counts = np.array([980.0, 1020.0])
        expected = np.array([1000.0, 1000.0])

        gaussian = gaussian_log_likelihood(counts, expected, np.sqrt(expected))

        assert switched_poisson_gaussian_log_likelihood(
            counts, expected, threshold=100.0
        ) == pytest.approx(gaussian, rel=1e-14)

    def test_the_switch_is_per_sample_not_per_channel(self) -> None:
        # An OES frame holds faint and bright lines together, so the switch has to act
        # sample by sample or a bright line drags the faint ones into the wrong regime.
        counts = np.array([2.0, 1020.0])
        expected = np.array([2.5, 1000.0])

        combined = switched_poisson_gaussian_log_likelihood(counts, expected, threshold=100.0)
        piecewise = poisson_log_likelihood(counts[:1], expected[:1]) + gaussian_log_likelihood(
            counts[1:], expected[1:], np.sqrt(expected[1:])
        )

        assert combined == pytest.approx(piecewise, rel=1e-14)

    def test_the_two_families_agree_at_the_documented_threshold(self) -> None:
        # Why ~100 is a defensible place to switch, measured rather than asserted, and
        # reported as what it is rather than as "close enough". At lambda = 100 the two
        # log-densities differ by 8.3e-4 nats at the mean (the Stirling remainder,
        # 1/(12 lambda)), by under 0.04 nats within +/- 2 sigma, and by 0.36 nats at the
        # -3 sigma edge, where the Poisson's skew starts to show. The +/- 2 sigma figure is
        # what matters: it is the region a credible interval is drawn from.
        expected = np.full(1, OES_GAUSSIAN_SWITCH_COUNTS)
        sigma = math.sqrt(OES_GAUSSIAN_SWITCH_COUNTS)

        def discrepancy(offset: float) -> float:
            counts = np.round(expected + offset)
            return abs(
                float(poisson_log_likelihood(counts, expected))
                - float(gaussian_log_likelihood(counts, expected, np.sqrt(expected)))
            )

        assert discrepancy(0.0) == pytest.approx(1.0 / (12.0 * 100.0), rel=0.02)
        assert max(discrepancy(o) for o in np.linspace(-2.0 * sigma, 2.0 * sigma, 9)) < 0.04
        assert max(discrepancy(o) for o in np.linspace(-3.0 * sigma, 3.0 * sigma, 13)) < 0.4

    def test_they_disagree_well_below_the_threshold(self) -> None:
        # The negative control for the test above: at lambda = 3 the Gaussian is a
        # different distribution, and it is the *upper* tail that shows it, because the
        # Poisson is right-skewed and the Gaussian is not. At k = 9 (a +3.5 sigma
        # fluctuation, which a 200-sample OES frame will contain) the two differ by 1.55
        # nats — enough to move a line ratio. Hence doc 05 §3.1's refusal.
        expected = np.full(1, 3.0)
        counts = np.full(1, 9.0)

        difference = abs(
            float(poisson_log_likelihood(counts, expected))
            - float(gaussian_log_likelihood(counts, expected, np.sqrt(expected)))
        )

        assert difference == pytest.approx(1.5539, abs=1e-3)
        assert difference > 1.0

    def test_a_non_positive_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            switched_poisson_gaussian_log_likelihood(np.zeros(2), np.ones(2), threshold=0.0)


class TestTheDetectionGate:
    """doc 01 IF-6 — "gated to zero below the detection floor" (doc 05 §3.1)."""

    def test_a_prediction_below_the_floor_is_masked_out(self) -> None:
        predicted = np.array([1.0e16, 3.3e16, 5.0e16])

        assert detection_mask(predicted, floor=3.3e16).tolist() == [False, True, True]

    def test_a_gated_channel_contributes_exactly_zero(self) -> None:
        # doc 01 IF-6: "must be modelled as absent there". Not "weakly weighted" — absent.
        channel = GaussianChannel(detection_floor=1.0e17)
        observed = _measurement([2.0, 3.0], [0.5, 0.5], instrument_id="interf", units="rad")
        predicted = _observable([1.0e16, 2.0e16], instrument_id="interf", units="rad")

        assert channel.log_prob(observed, predicted) == 0.0

    def test_an_ungated_channel_is_unaffected(self) -> None:
        with_floor = GaussianChannel(detection_floor=1.0)
        without = GaussianChannel()
        observed = _measurement([2.0, 3.0], [0.5, 0.5])
        predicted = _observable([2.2, 2.7])

        assert with_floor.log_prob(observed, predicted) == pytest.approx(
            without.log_prob(observed, predicted), rel=1e-14
        )

    def test_the_gate_is_per_sample(self) -> None:
        channel = GaussianChannel(detection_floor=2.0)
        observed = _measurement([1.0, 5.0], [0.5, 0.5])
        predicted = _observable([0.5, 5.2])

        expected = float(stats.norm.logpdf(5.0, loc=5.2, scale=0.5))

        assert channel.log_prob(observed, predicted) == pytest.approx(expected, rel=SCIPY_TOL)

    def test_a_negative_floor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            GaussianChannel(detection_floor=-1.0)


class TestOutlierRobustness:
    """doc 05 §3.3 — "a Gaussian likelihood will contort the entire fit"."""

    def test_the_student_t_matches_scipy(self) -> None:
        observed = np.array([0.4, -12.0, 1.1])
        predicted = np.zeros(3)
        sigma = np.array([1.0, 1.0, 2.0])

        expected = float(np.sum(stats.t.logpdf(observed, 4.0, loc=predicted, scale=sigma)))

        assert student_t_log_likelihood(observed, predicted, sigma, dof=4.0) == pytest.approx(
            expected, rel=SCIPY_TOL
        )

    def test_the_student_t_tends_to_the_gaussian_as_the_dof_grows(self) -> None:
        observed = np.array([0.4, -1.2, 1.1])
        predicted = np.zeros(3)
        sigma = np.ones(3)

        assert student_t_log_likelihood(observed, predicted, sigma, dof=1.0e7) == pytest.approx(
            gaussian_log_likelihood(observed, predicted, sigma), abs=1e-5
        )

    def test_the_student_t_is_barely_moved_by_a_cosmic_ray(self) -> None:
        # doc 04 §7.2's discrete failures. One 40-sigma point should not be allowed to
        # dominate; under a Gaussian it contributes -800 nats and under a t it contributes
        # a logarithm of that.
        clean = np.zeros(8)
        outlier = np.zeros(8)
        outlier[3] = 40.0
        predicted = np.zeros(8)
        sigma = np.ones(8)

        gaussian_damage = gaussian_log_likelihood(
            clean, predicted, sigma
        ) - gaussian_log_likelihood(outlier, predicted, sigma)
        t_damage = student_t_log_likelihood(
            clean, predicted, sigma, dof=4.0
        ) - student_t_log_likelihood(outlier, predicted, sigma, dof=4.0)

        assert gaussian_damage == pytest.approx(800.0, rel=1e-9)
        assert t_damage < 20.0

    def test_a_non_positive_dof_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            student_t_log_likelihood(np.zeros(2), np.zeros(2), np.ones(2), dof=0.0)

    def test_the_mixture_is_the_documented_two_component_density(self) -> None:
        # doc 05 §3.3: "an explicit mixture with a broad outlier component".
        observed = np.array([0.5, 9.0])
        predicted = np.zeros(2)
        sigma = np.ones(2)

        expected = float(
            np.sum(
                np.log(
                    0.95 * stats.norm.pdf(observed, loc=predicted, scale=sigma)
                    + 0.05 * stats.norm.pdf(observed, loc=predicted, scale=10.0 * sigma)
                )
            )
        )

        assert outlier_mixture_log_likelihood(
            observed, predicted, sigma, outlier_fraction=0.05, outlier_scale=10.0
        ) == pytest.approx(expected, rel=SCIPY_TOL)

    def test_a_zero_outlier_fraction_is_exactly_the_gaussian(self) -> None:
        observed = np.array([0.5, -1.2])
        predicted = np.zeros(2)
        sigma = np.ones(2)

        assert outlier_mixture_log_likelihood(
            observed, predicted, sigma, outlier_fraction=0.0, outlier_scale=10.0
        ) == pytest.approx(gaussian_log_likelihood(observed, predicted, sigma), rel=1e-13)

    def test_an_outlier_fraction_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            outlier_mixture_log_likelihood(
                np.zeros(2), np.zeros(2), np.ones(2), outlier_fraction=1.5, outlier_scale=10.0
            )

    def test_an_outlier_scale_below_one_is_rejected(self) -> None:
        # The outlier component is the *broad* one; a narrow one would make the mixture a
        # way of over-fitting rather than of resisting an over-fit.
        with pytest.raises(ValueError, match="broad"):
            outlier_mixture_log_likelihood(
                np.zeros(2), np.zeros(2), np.ones(2), outlier_fraction=0.05, outlier_scale=0.5
            )


class TestTheChannels:
    """Each channel of doc 05 §3.1, independently."""

    def test_the_poisson_channel_uses_the_counts(self) -> None:
        observed = _measurement([0.0, 1.0, 2.0], [0.0, 1.0, 1.4], instrument_id="thomson")
        predicted = _observable([0.008, 0.5, 3.0], instrument_id="thomson")

        assert PoissonChannel().log_prob(observed, predicted) == pytest.approx(
            float(np.sum(stats.poisson.logpmf(observed.values, predicted.values))), rel=SCIPY_TOL
        )

    def test_the_poisson_channel_ignores_the_reported_uncertainty(self) -> None:
        # A Poisson channel's uncertainty is the rate, not a separately-supplied sigma.
        # Silently using both would count the photon statistics twice.
        predicted = _observable([3.0], instrument_id="thomson")
        wide = _measurement([2.0], [10.0], instrument_id="thomson")
        narrow = _measurement([2.0], [0.1], instrument_id="thomson")

        assert PoissonChannel().log_prob(wide, predicted) == PoissonChannel().log_prob(
            narrow, predicted
        )

    def test_the_correlated_channel_uses_the_supplied_covariance(self) -> None:
        times = np.linspace(0.0, 1.0e-3, 5)
        covariance = coloured_noise_covariance(times, sigma=1.0e-4, correlation_time=3.0e-4)
        channel = CorrelatedGaussianChannel(covariance=covariance)
        observed = _measurement(
            [1e-4, -2e-4, 0.0, 3e-5, -1e-5], [1e-4] * 5, instrument_id="interf", units="rad"
        )
        predicted = _observable([0.0] * 5, instrument_id="interf", units="rad")

        assert channel.log_prob(observed, predicted) == pytest.approx(
            correlated_gaussian_log_likelihood(observed.values, predicted.values, covariance),
            rel=1e-14,
        )

    def test_the_correlated_channel_rejects_a_covariance_of_the_wrong_size(self) -> None:
        channel = CorrelatedGaussianChannel(covariance=np.eye(3))
        observed = _measurement([1.0, 2.0], [0.5, 0.5], instrument_id="interf", units="rad")
        predicted = _observable([1.0, 2.0], instrument_id="interf", units="rad")

        with pytest.raises(ChannelMismatchError, match="covariance"):
            channel.log_prob(observed, predicted)

    def test_the_switched_channel_defaults_to_the_registry_threshold(self) -> None:
        assert SwitchedPoissonGaussianChannel().threshold == pytest.approx(
            OES_GAUSSIAN_SWITCH_COUNTS
        )

    def test_the_student_t_channel_matches_the_free_function(self) -> None:
        channel = StudentTChannel(dof=4.0)
        observed = _measurement([1.0, -9.0], [0.5, 0.5], instrument_id="lif", units="count")
        predicted = _observable([1.1, 0.0], instrument_id="lif", units="count")

        assert channel.log_prob(observed, predicted) == pytest.approx(
            student_t_log_likelihood(
                observed.values, predicted.values, observed.uncertainty, dof=4.0
            ),
            rel=1e-14,
        )

    def test_the_mixture_channel_matches_the_free_function(self) -> None:
        channel = MixtureChannel(outlier_fraction=0.05, outlier_scale=10.0)
        observed = _measurement([1.0, -9.0], [0.5, 0.5], instrument_id="lif", units="count")
        predicted = _observable([1.1, 0.0], instrument_id="lif", units="count")

        assert channel.log_prob(observed, predicted) == pytest.approx(
            outlier_mixture_log_likelihood(
                observed.values,
                predicted.values,
                observed.uncertainty,
                outlier_fraction=0.05,
                outlier_scale=10.0,
            ),
            rel=1e-14,
        )


class TestTheAsynchronousSum:
    """doc 05 §3.2 — one term per observation, each with its own acquisition window."""

    def test_it_sums_over_channels_and_observations(self) -> None:
        thomson = _measurement([0.0, 1.0], [0.0, 1.0], instrument_id="thomson", window=LONG_WINDOW)
        oes = _measurement([12.0, 950.0], [3.5, 31.0], instrument_id="oes")
        measurements = MeasurementSet.of(thomson, oes)
        predictions = {
            "thomson": [_observable([0.008, 0.9], instrument_id="thomson", window=LONG_WINDOW)],
            "oes": [_observable([10.0, 1000.0], instrument_id="oes")],
        }
        channels = {"thomson": PoissonChannel(), "oes": SwitchedPoissonGaussianChannel()}

        expected = PoissonChannel().log_prob(
            thomson, predictions["thomson"][0]
        ) + SwitchedPoissonGaussianChannel().log_prob(oes, predictions["oes"][0])

        assert total_log_likelihood(measurements, predictions, channels) == pytest.approx(
            expected, rel=1e-14
        )

    def test_a_prediction_from_a_different_window_is_refused(self) -> None:
        # doc 05 §3.2: treating a 700 s Thomson accumulation and a 2 ns OES gate as
        # measurements of "the same instant" would be straightforwardly false. The
        # structure refuses to let it happen silently.
        observed = _measurement([1.0], [1.0], instrument_id="thomson", window=LONG_WINDOW)
        predicted = _observable([1.0], instrument_id="thomson", window=WINDOW)

        with pytest.raises(ChannelMismatchError, match="window"):
            total_log_likelihood(
                MeasurementSet.of(observed),
                {"thomson": [predicted]},
                {"thomson": PoissonChannel()},
            )

    def test_a_prediction_for_the_wrong_instrument_is_refused(self) -> None:
        observed = _measurement([1.0], [1.0], instrument_id="thomson")
        predicted = _observable([1.0], instrument_id="oes")

        with pytest.raises(ChannelMismatchError, match="instrument"):
            total_log_likelihood(
                MeasurementSet.of(observed),
                {"thomson": [predicted]},
                {"thomson": PoissonChannel()},
            )

    def test_a_prediction_in_different_units_is_refused(self) -> None:
        observed = _measurement([1.0], [1.0], instrument_id="thomson", units="count")
        predicted = _observable([1.0], instrument_id="thomson", units="rad")

        with pytest.raises(ChannelMismatchError, match="units"):
            total_log_likelihood(
                MeasurementSet.of(observed),
                {"thomson": [predicted]},
                {"thomson": PoissonChannel()},
            )

    def test_a_missing_channel_model_is_refused(self) -> None:
        observed = _measurement([1.0], [1.0], instrument_id="thomson")

        with pytest.raises(ChannelMismatchError, match="no likelihood"):
            total_log_likelihood(
                MeasurementSet.of(observed),
                {"thomson": [_observable([1.0], instrument_id="thomson")]},
                {},
            )

    def test_a_missing_prediction_is_refused(self) -> None:
        observed = _measurement([1.0], [1.0], instrument_id="thomson")

        with pytest.raises(ChannelMismatchError, match="no prediction"):
            total_log_likelihood(MeasurementSet.of(observed), {}, {"thomson": PoissonChannel()})

    def test_a_prediction_count_mismatch_is_refused(self) -> None:
        observed = _measurement([1.0], [1.0], instrument_id="thomson")

        with pytest.raises(ChannelMismatchError, match="1 observation"):
            total_log_likelihood(
                MeasurementSet.of(observed),
                {"thomson": [_observable([1.0], instrument_id="thomson")] * 2},
                {"thomson": PoissonChannel()},
            )

    def test_an_empty_measurement_set_has_zero_log_likelihood(self) -> None:
        # doc 02 §13's degraded configurations remove channels; the fully degraded case
        # must be the prior and not a crash.
        assert total_log_likelihood(MeasurementSet.of(), {}, {}) == 0.0

    def test_dropping_a_channel_reduces_the_information(self) -> None:
        # doc 05 §6.3's leave-one-channel-out, in its crudest form: the joint
        # log-likelihood is the sum, so removing a channel removes its term exactly.
        thomson = _measurement([0.0, 1.0], [0.0, 1.0], instrument_id="thomson", window=LONG_WINDOW)
        oes = _measurement([12.0], [3.5], instrument_id="oes")
        predictions = {
            "thomson": [_observable([0.008, 0.9], instrument_id="thomson", window=LONG_WINDOW)],
            "oes": [_observable([10.0], instrument_id="oes")],
        }
        channels = {"thomson": PoissonChannel(), "oes": SwitchedPoissonGaussianChannel()}

        full = total_log_likelihood(MeasurementSet.of(thomson, oes), predictions, channels)
        without_oes = total_log_likelihood(
            MeasurementSet.of(thomson, oes).without_instrument("oes"), predictions, channels
        )

        assert full - without_oes == pytest.approx(
            SwitchedPoissonGaussianChannel().log_prob(oes, predictions["oes"][0]), rel=1e-14
        )


class TestLayering:
    """doc 08 §8: "`vpl-inverse` must not import `vpl-physics.fluid`" — the whole of it.

    Stated more strongly than doc 08 §8 does, and deliberately: this package must not
    import any forward-model package at all. doc 05 §7 says the inverse crime is "guarded
    against structurally rather than by good intentions", and the structure that does the
    guarding is exactly this — the inversion cannot reach for the solver that generated
    the truth, because it cannot reach for any solver.

    Checked by walking the import graph of the loaded modules rather than by grepping, so
    a transitive import through some future helper is caught too.
    """

    def test_no_forward_model_package_is_reachable_from_vpl_inverse(self) -> None:
        import importlib
        import sys

        forbidden = ("vpl.physics", "vpl.instruments", "vpl.optics", "vpl.detectors")
        for name in (
            "vpl.inverse",
            "vpl.inverse.parameters",
            "vpl.inverse.priors",
            "vpl.inverse.likelihood",
        ):
            sys.modules.pop(name, None)
        for name in list(sys.modules):
            if name.startswith(forbidden):
                sys.modules.pop(name, None)

        importlib.import_module("vpl.inverse")

        leaked = sorted(n for n in sys.modules if n.startswith(forbidden))
        assert leaked == []

    def test_vpl_core_is_reachable(self) -> None:
        # The other half of the statement: the inverse layer *does* depend on the core's
        # protocols and data model (doc 08 §1 principle 1), so an isolation test that
        # passed because nothing was imported at all would be worthless.
        import sys

        import vpl.inverse  # noqa: F401  — imported for its side effect on sys.modules

        assert "vpl.core.state" in sys.modules
        assert "vpl.core.params" in sys.modules
