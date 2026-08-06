"""The Laplace posterior — doc 05 §5, doc 11 §9 item 5.

## What these tests are for

A covariance is the easiest thing in an inference stack to get wrong without anyone
noticing. It is never printed next to a truth, nothing about it looks implausible, and the
error surfaces only as intervals that are quietly too narrow — which is the failure that
matters, because a credible interval that is too narrow is an overconfident claim and this
project's entire pitch is *calibrated* uncertainty.

So the tests are anchored on a problem whose posterior covariance is known exactly. For a
linear-Gaussian model the posterior is Gaussian and its covariance is
``(A' Sigma^-1 A + S^-1)^-1``, written out here independently of the module. Agreement is
then evidence rather than a tautology — the standing lesson of ADR-011, where a solver
agreed with itself to 52 % error.

## The three specific errors these pin

1. **A factor of two in the Hessian.** ``-log p`` has Hessian ``H``, and the covariance is
   ``H^-1``, not ``2 H^-1`` or ``H^-1 / 2``. Every one of those produces a plausible
   interval; only one is right.
2. **Inverting a Hessian that is not positive definite.** At a flat direction — doc 05 §6's
   null space, which this problem *has* — the Hessian is singular and its numerical inverse
   is an enormous finite matrix rather than an error. Reporting that as a covariance turns
   "this direction is not identified" into "this direction is measured, with a big error
   bar", which is a different and much worse claim.
3. **Transforming a mean instead of a median.** The posterior is Gaussian in *unconstrained*
   space; the transforms to physical units are non-linear, so the physical distribution is
   not Gaussian and its mean is not the transform of the unconstrained mean. The median is,
   because every transform in :mod:`vpl.inverse.parameters` is strictly increasing.

## What a Laplace result is allowed to claim

doc 05 §5 is unusually direct: Laplace "is valid only when the posterior is near-Gaussian;
**its validity is tested, not assumed**", and the framework "reports the divergence between
[Laplace and NUTS]; Laplace is only permitted where that divergence is below a threshold."
NUTS is not built. So every Laplace posterior this project can currently produce is
*unvalidated*, and the label is carried on the object rather than left to a footnote — the
same discipline :mod:`vpl.validation.sealed` applies to T1-reported-as-T2.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.inverse.laplace import (
    LaplaceValidity,
    PosteriorNotPositiveDefiniteError,
    laplace_posterior,
)
from vpl.inverse.parameters import CONTROL_PARAMETERS, N_CONTROL, ControlParameters
from vpl.inverse.priors import default_control_prior


class _GaussianPrior:
    """``N(0, variance I)`` in unconstrained space, with no Jacobian to carry."""

    def __init__(self, variance: float, size: int) -> None:
        self.variance = variance
        self.size = size

    def log_prob_unconstrained(self, u: np.ndarray) -> float:
        return float(-0.5 * u @ u / self.variance)


class _FlatPrior:
    """Improper and constant — so a flat likelihood direction stays flat in the posterior."""

    def log_prob_unconstrained(self, u: np.ndarray) -> float:
        del u  # constant density: the point is that it contributes no curvature
        return 0.0


class TestAgainstTheAnalyticPosterior:
    def test_the_covariance_matches_the_linear_gaussian_closed_form(self) -> None:
        # The anchor. For y = A u + eps, eps ~ N(0, s^2 I), prior N(0, v I), the posterior
        # covariance is (A'A/s^2 + I/v)^-1 exactly. A factor of two anywhere in the Hessian
        # shows up here and nowhere else.
        rng = np.random.default_rng(20260806)
        n = 4
        design = rng.normal(size=(7, n))
        sigma, variance = 0.5, 2.0
        observed = design @ rng.normal(size=n)

        def log_likelihood(u: np.ndarray) -> float:
            r = design @ u - observed
            return float(-0.5 * r @ r / sigma**2)

        precision = design.T @ design / sigma**2 + np.eye(n) / variance
        expected = np.linalg.inv(precision)
        mean = np.linalg.solve(precision, design.T @ observed / sigma**2)

        posterior = laplace_posterior(
            mean, log_likelihood=log_likelihood, prior=_GaussianPrior(variance, n)
        )

        np.testing.assert_allclose(posterior.covariance, expected, rtol=1e-5, atol=1e-8)

    def test_the_marginal_standard_deviations_match_the_closed_form(self) -> None:
        # The quantity actually reported. Correct covariance with a mis-taken diagonal
        # would still pass nothing else in this file.
        rng = np.random.default_rng(11)
        n = 3
        design = rng.normal(size=(5, n))
        observed = design @ rng.normal(size=n)

        def log_likelihood(u: np.ndarray) -> float:
            r = design @ u - observed
            return float(-0.5 * r @ r)

        expected = np.sqrt(np.diag(np.linalg.inv(design.T @ design + np.eye(n))))

        posterior = laplace_posterior(
            np.zeros(n), log_likelihood=log_likelihood, prior=_GaussianPrior(1.0, n)
        )

        np.testing.assert_allclose(posterior.marginal_std, expected, rtol=1e-5)

    def test_the_hessian_is_symmetric(self) -> None:
        # A finite-difference Hessian is only symmetric if the cross terms are formed
        # correctly; an asymmetric one means the mixed differences are mismatched, and it
        # would still invert to something plausible.
        rng = np.random.default_rng(3)
        design = rng.normal(size=(6, 3))

        def log_likelihood(u: np.ndarray) -> float:
            r = design @ u
            return float(-0.5 * r @ r)

        posterior = laplace_posterior(
            np.zeros(3), log_likelihood=log_likelihood, prior=_GaussianPrior(1.0, 3)
        )

        np.testing.assert_allclose(posterior.precision, posterior.precision.T, rtol=1e-8)


class TestFlatDirections:
    def test_a_singular_hessian_is_refused_rather_than_inverted(self) -> None:
        # doc 05 §6's null space, made concrete: the likelihood constrains u0 + u1 and says
        # nothing about u0 - u1. numpy will happily return an enormous finite inverse, and
        # reporting that as a covariance converts "not identified" into "measured badly".
        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * (u[0] + u[1]) ** 2)

        with pytest.raises(PosteriorNotPositiveDefiniteError):
            laplace_posterior(np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior())

    def test_the_refusal_names_the_flat_direction(self) -> None:
        # An error saying only "not positive definite" sends the reader to the optimiser.
        # The useful information is *which combination* of parameters is unconstrained,
        # because that is a statement about the experiment, not about the code.
        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * (u[0] + u[1]) ** 2)

        with pytest.raises(PosteriorNotPositiveDefiniteError) as excinfo:
            laplace_posterior(np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior())

        message = str(excinfo.value)
        assert "eigenvalue" in message.lower()
        assert "doc 05 §6" in message

    def test_a_maximum_rather_than_a_minimum_is_refused(self) -> None:
        # A MAP that stopped at a saddle or a maximum reports converged=True just as
        # readily. The Hessian is the cheapest place to catch it, and a negative curvature
        # direction is not a wide error bar — it is a point that is not an optimum.
        def log_likelihood(u: np.ndarray) -> float:
            return float(+0.5 * u @ u)

        with pytest.raises(PosteriorNotPositiveDefiniteError):
            laplace_posterior(np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior())


class TestCredibleIntervals:
    def test_the_unconstrained_interval_is_the_gaussian_quantile(self) -> None:
        # In unconstrained space the approximation *is* Gaussian, so the 95 % interval is
        # exactly mean +/- 1.959964 sigma. Anything else is a z/t confusion or a one-sided
        # tail mistaken for two.
        n = 2

        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * u @ u * 3.0)

        posterior = laplace_posterior(
            np.zeros(n), log_likelihood=log_likelihood, prior=_FlatPrior()
        )
        low, high = posterior.credible_interval(0.95)

        np.testing.assert_allclose(
            high - low, 2 * 1.959963984540054 * posterior.marginal_std, rtol=1e-6
        )

    def test_a_wider_level_gives_a_wider_interval(self) -> None:
        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * u @ u)

        posterior = laplace_posterior(
            np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior()
        )

        narrow = np.subtract(*reversed(posterior.credible_interval(0.5)))
        wide = np.subtract(*reversed(posterior.credible_interval(0.99)))

        assert np.all(wide > narrow)

    def test_a_level_outside_the_unit_interval_is_refused(self) -> None:
        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * u @ u)

        posterior = laplace_posterior(
            np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior()
        )

        with pytest.raises(ValueError, match="level"):
            posterior.credible_interval(1.0)


class TestThePhysicalInterval:
    def test_the_endpoints_are_the_transformed_unconstrained_endpoints(self) -> None:
        # Every transform in vpl.inverse.parameters is strictly increasing, so a marginal
        # interval maps through *exactly* — no delta method, no linearisation error. This
        # test is what licenses that shortcut; if a non-monotone transform is ever added it
        # fails, which is the point.
        prior = default_control_prior()
        mean = np.asarray(prior.median().to_unconstrained())

        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * (u - mean) @ (u - mean) * 4.0)

        posterior = laplace_posterior(mean, log_likelihood=log_likelihood, prior=prior)
        low, high = posterior.credible_interval(0.9)
        physical = posterior.physical_credible_interval(0.9)

        for index, spec in enumerate(CONTROL_PARAMETERS):
            expected_low = spec.transform.to_constrained(float(low[index]))
            expected_high = spec.transform.to_constrained(float(high[index]))
            actual_low, actual_high = physical[spec.name]
            assert actual_low == pytest.approx(expected_low, rel=1e-10)
            assert actual_high == pytest.approx(expected_high, rel=1e-10)
            assert actual_low < actual_high, f"{spec.name} interval is inverted"

    def test_the_physical_point_estimate_is_the_median_not_the_mean(self) -> None:
        # The transform is non-linear, so E[x] != x(E[u]) — but median(x) = x(median(u))
        # under a strictly increasing map. Reporting the transformed mean and calling it a
        # posterior mean is a real, quiet bias; the median is exact and is what is reported.
        prior = default_control_prior()
        mean = np.asarray(prior.median().to_unconstrained())

        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * (u - mean) @ (u - mean))

        posterior = laplace_posterior(mean, log_likelihood=log_likelihood, prior=prior)

        median = posterior.physical_median()
        assert isinstance(median, ControlParameters)
        np.testing.assert_allclose(median.to_unconstrained(), posterior.mean, rtol=1e-10)

    def test_a_vector_that_is_not_the_control_vector_has_no_physical_reading(self) -> None:
        # The engine is dimension-agnostic on purpose (see map.py). A two-parameter
        # verification problem has no ControlParameters representation and inventing one
        # would be worse than refusing.
        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * u @ u)

        posterior = laplace_posterior(
            np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior()
        )

        assert N_CONTROL != 2, "this test is vacuous if the control vector is 2-dimensional"
        with pytest.raises(ValueError, match="control vector"):
            posterior.physical_credible_interval(0.9)


class TestSampling:
    def test_the_empirical_covariance_recovers_the_covariance(self) -> None:
        # Sampling is how a *derived* quantity like Gamma_E gets an interval: it is a
        # non-linear function of several parameters, so no endpoint transform applies and
        # the only honest route is to push samples through the forward model.
        rng = np.random.default_rng(5)
        design = rng.normal(size=(6, 3))

        def log_likelihood(u: np.ndarray) -> float:
            r = design @ u
            return float(-0.5 * r @ r)

        posterior = laplace_posterior(
            np.zeros(3), log_likelihood=log_likelihood, prior=_GaussianPrior(1.0, 3)
        )
        draws = posterior.sample(np.random.default_rng(7), 200_000)

        assert draws.shape == (200_000, 3)
        np.testing.assert_allclose(np.cov(draws, rowvar=False), posterior.covariance, atol=2e-3)

    def test_samples_are_reproducible_from_the_generator(self) -> None:
        # doc 00 E3: bit-for-bit reproduction.
        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * u @ u)

        posterior = laplace_posterior(
            np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior()
        )

        first = posterior.sample(np.random.default_rng(1), 50)
        second = posterior.sample(np.random.default_rng(1), 50)

        np.testing.assert_array_equal(first, second)


class TestTheValidityLabel:
    def test_a_laplace_posterior_is_unvalidated_by_default(self) -> None:
        # doc 05 §5: "its validity is tested, not assumed", against NUTS. NUTS is not built,
        # so nothing in this project can currently validate a Laplace posterior, and the
        # object says so rather than leaving it to a footnote somebody drops.
        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * u @ u)

        posterior = laplace_posterior(
            np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior()
        )

        assert posterior.validity is LaplaceValidity.UNVALIDATED

    def test_the_repr_carries_the_validity_so_it_cannot_be_quoted_bare(self) -> None:
        def log_likelihood(u: np.ndarray) -> float:
            return float(-0.5 * u @ u)

        posterior = laplace_posterior(
            np.zeros(2), log_likelihood=log_likelihood, prior=_FlatPrior()
        )

        assert "UNVALIDATED" in repr(posterior)
