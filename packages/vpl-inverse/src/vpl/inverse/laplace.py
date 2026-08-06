"""The Laplace posterior — doc 05 §5, doc 11 WBS 3.7, doc 11 §9 item 5.

doc 05 §5 lists the Laplace approximation second, as "a cheap Gaussian posterior at the
MAP" costing "+ one Hessian", and then immediately constrains it:

    Valid only when the posterior is near-Gaussian; **its validity is tested, not assumed**.

and

    **The Laplace-validity test is a small thing that prevents a large error.** Laplace is
    tempting because it is cheap, and it is wrong whenever the posterior is skewed or
    multimodal — which, in this problem, it demonstrably is near the identifiability
    boundary. The framework runs Laplace and NUTS on a subset of cases and reports the
    divergence between them; Laplace is only permitted where that divergence is below a
    threshold.

**NUTS is not built.** So the permission doc 05 §5 describes cannot currently be granted for
any case, and this module says so on the object: every posterior it returns carries
:attr:`LaplacePosterior.validity` = :attr:`LaplaceValidity.UNVALIDATED`, and the ``repr``
carries it too, so a number cannot be lifted into a figure with the caveat left behind. This
is the same discipline :mod:`vpl.validation.sealed` applies to reporting T1 as T2, and for
the same reason: a label that lives in a document is a label that gets dropped.

What *can* be said without NUTS is narrower and still worth having. doc 11 §9 item 5's
coverage test — "uncertainty that has been *checked*", the item the roadmap says never to
cut — is an empirical calibration check on exactly the intervals this module produces. If
the 90 % interval contains the truth in 90 % of cases, these intervals are calibrated **for
that quantity, on that ensemble**. That is a genuine and strong statement, and it is not the
same as the posterior being correct: a set of intervals can be calibrated on average while
being wrong case by case, and coverage cannot see skew that averages out. It is necessary,
not sufficient, and it does not upgrade the label.

## Why the Hessian is refused rather than regularised

At a flat direction the Hessian is singular, and ``numpy`` will return an enormous finite
inverse rather than an error. Reporting that as a covariance converts doc 05 §6's honest
statement — *this combination of parameters is not identified by this measurement* — into
the very different claim *this combination is measured, with a large error bar*. The first
is a result about the experiment; the second is a fabrication. doc 05 §6.2 predicts a
specific instance of this (``Gamma_i ~ n_0 sqrt(T_e)`` makes ``n_0`` and ``T_e`` correlated),
so the singular case is expected here, not hypothetical.

Adding a ridge to make the inverse exist would produce a covariance whose flat direction is
governed entirely by the size of the ridge — a number chosen for numerical convenience,
reported as physics. So :func:`laplace_posterior` raises, and names the offending
eigenvector, because *which* combination is unconstrained is the informative part.

## Why the physical intervals are exact rather than linearised

The posterior is Gaussian in **unconstrained** space, and the transforms to physical units
are non-linear, so the physical distribution is not Gaussian. The usual move is the delta
method, which is a linearisation and carries an error nobody tracks. It is not needed here:
every transform in :mod:`vpl.inverse.parameters` — identity, ``exp``, and the scaled logistic
— is **strictly increasing**, and a strictly increasing map carries quantiles to quantiles
exactly. So a marginal credible interval maps endpoint-to-endpoint with no approximation at
all, and the *median* — not the mean — is the point estimate that transforms exactly.
``test_the_endpoints_are_the_transformed_unconstrained_endpoints`` is what licenses this,
and it fails if a non-monotone transform is ever added.

A **derived** quantity such as ``Gamma_E`` is a non-linear function of several parameters,
so no endpoint argument applies to it. For those, :meth:`LaplacePosterior.sample` is the
honest route: draw from the Gaussian in unconstrained space, push each draw through the
forward model, and take empirical quantiles of the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky
from scipy.special import ndtri

from vpl.inverse.map import LogLikelihood, LogPosteriorPrior, negative_log_posterior
from vpl.inverse.parameters import CONTROL_PARAMETERS, N_CONTROL, ControlParameters

__all__ = [
    "LaplacePosterior",
    "LaplaceValidity",
    "PosteriorNotPositiveDefiniteError",
    "laplace_posterior",
]

type FloatArray = NDArray[np.float64]

#: Central-difference step for the **second** derivative, and deliberately not the ``1e-6``
#: :mod:`vpl.inverse.map` uses for the first. A central first difference balances truncation
#: against roundoff at ``eps^(1/3) ~ 6e-6``; a second difference divides by ``h^2`` instead
#: of ``h`` and balances at ``eps^(1/4) ~ 1.2e-4``. Reusing the gradient's step here would
#: amplify roundoff by two orders of magnitude, and the symptom — a Hessian that is noisy
#: rather than wrong — is one that looks like a hard problem instead of like a bug. The
#: coordinates are O(1) in unconstrained space by construction, so one absolute step is
#: appropriate and there is no scale to chase.
_HESSIAN_STEP: Final[float] = 1e-4

#: Smallest curvature, relative to the best-constrained direction, that is still reported as
#: a measurement rather than refused as a null-space direction. At this ratio the marginal
#: standard deviation along that direction is ``sqrt(1/1e-9) ~ 3e4`` times the tightest one:
#: an "error bar" spanning four and a half orders of magnitude more than the data resolves
#: is doc 05 §6's null space, and calling it an uncertainty would be a category error.
_MINIMUM_RELATIVE_CURVATURE: Final[float] = 1e-9


class LaplaceValidity(StrEnum):
    """Whether the Gaussian assumption behind a Laplace posterior has been *checked*.

    A ``StrEnum`` so the label survives into a manifest, a filename and a figure caption
    without a lookup table, per doc 08 §1 principle 4.
    """

    UNVALIDATED = "UNVALIDATED"
    """No reference posterior has been compared against this one — doc 05 §5's condition is
    unmet. This is the only value this project can currently produce, because doc 05 §5's
    stated reference (NUTS) is not built."""

    VALIDATED = "VALIDATED"
    """A reference posterior was compared and the divergence was below doc 05 §5's
    threshold. Reserved; nothing sets it yet, and it must not be set by anything that has
    not actually run the comparison."""


class PosteriorNotPositiveDefiniteError(RuntimeError):
    """The Hessian at the reported optimum is singular, or has negative curvature.

    Either means the Laplace approximation does not exist there: a flat direction is doc 05
    §6's null space, and a negative direction means the point is not a minimum of the
    negative log posterior at all — a MAP that stopped at a saddle reports
    ``converged=True`` exactly as readily as one that found the mode.
    """


@dataclass(frozen=True, slots=True)
class LaplacePosterior:
    """A Gaussian posterior at the MAP, in unconstrained space — doc 05 §5.

    Attributes:
        mean: The MAP point. Gaussian in *unconstrained* coordinates; see the module
            docstring for why the physical reading is a median rather than a mean.
        covariance: ``H^-1``, the posterior covariance in unconstrained space.
        precision: ``H``, the Hessian of the negative log posterior at :attr:`mean`.
            Carried alongside the covariance because it is what was actually measured —
            the covariance is derived from it — and because doc 05 §6.1's Fisher
            information is a statement about the precision, not its inverse.
        curvature: Eigenvalues of :attr:`precision`, ascending. The smallest one is how
            close this problem sits to doc 05 §6's null space, which is a result about the
            experiment worth reporting rather than an internal detail.
        validity: Always :attr:`LaplaceValidity.UNVALIDATED` for now. See the module
            docstring.
    """

    mean: FloatArray
    covariance: FloatArray
    precision: FloatArray
    curvature: FloatArray
    validity: LaplaceValidity = LaplaceValidity.UNVALIDATED

    @property
    def marginal_std(self) -> FloatArray:
        """Per-coordinate posterior standard deviation, in unconstrained space."""
        return np.sqrt(np.diag(self.covariance))

    @property
    def condition_number(self) -> float:
        """Ratio of largest to smallest curvature — how anisotropic the posterior is.

        A large value is not a defect. It is doc 05 §6.2's ``n_0``-``T_e`` correlation
        showing up as a ridge, and it is the quantity that decides whether an ensemble
        needs a sampler rather than this approximation.
        """
        return float(self.curvature[-1] / self.curvature[0])

    def credible_interval(self, level: float) -> tuple[FloatArray, FloatArray]:
        """Equal-tailed marginal credible interval in unconstrained space.

        Exact for this approximation: the marginals of a Gaussian are Gaussian, so the
        interval is ``mean +/- z sigma`` with ``z`` the two-sided normal quantile.

        Args:
            level: Central probability mass, strictly between 0 and 1. ``0.9`` gives the
                interval doc 11 §9 item 5's coverage test checks.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"credible level must lie strictly in (0, 1), got {level}")
        z = float(ndtri(0.5 + 0.5 * level))
        half_width = z * self.marginal_std
        return self.mean - half_width, self.mean + half_width

    def physical_credible_interval(self, level: float) -> dict[str, tuple[float, float]]:
        """The same interval in each parameter's own units — exact, not linearised.

        Every transform of doc 05 §2.1 is strictly increasing, so the endpoints map through
        without approximation. See the module docstring; ``test_the_endpoints_are_the_
        transformed_unconstrained_endpoints`` is what keeps that true.

        Raises:
            ValueError: If this posterior is not over the Level A control vector. A
                low-dimensional verification problem has no physical reading, and inventing
                one would be worse than reporting its absence (the same choice
                :class:`~vpl.inverse.map.MapResult` makes).
        """
        self._require_control_vector("a physical credible interval")
        low, high = self.credible_interval(level)
        return {
            spec.name: (
                spec.transform.to_constrained(float(low[index])),
                spec.transform.to_constrained(float(high[index])),
            )
            for index, spec in enumerate(CONTROL_PARAMETERS)
        }

    def physical_median(self) -> ControlParameters:
        """The posterior **median** in physical units.

        The median and not the mean, and the distinction is not pedantry: under a
        non-linear transform ``E[x] != x(E[u])``, so transforming the unconstrained mean and
        calling the result a posterior mean introduces a bias that is smooth, plausible and
        invisible. Under a strictly increasing transform the median maps exactly, so this
        number is the one that needs no caveat.
        """
        self._require_control_vector("a physical median")
        return ControlParameters.from_unconstrained(self.mean)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        """``n`` draws from the Gaussian, shape ``(n, d)``, in unconstrained space.

        The route for any **derived** quantity — ``Gamma_E`` above all — because a
        non-linear function of several parameters has no endpoint-transform shortcut. Push
        each draw through the forward model and take empirical quantiles of the output.

        Drawn through the Cholesky factor rather than by eigendecomposition: the factor
        exists precisely because :func:`laplace_posterior` already refused anything that is
        not positive definite, so a failure here would mean the invariant was violated.
        """
        if n < 0:
            raise ValueError(f"sample count cannot be negative, got {n}")
        factor = cholesky(self.covariance, lower=True)
        draws = self.mean + rng.standard_normal((n, self.mean.size)) @ factor.T
        return np.asarray(draws, dtype=np.float64)

    def sample_parameters(
        self, rng: np.random.Generator, n: int
    ) -> tuple[ControlParameters, ...]:
        """:meth:`sample`, mapped into physical units."""
        self._require_control_vector("physical samples")
        return tuple(ControlParameters.from_unconstrained(u) for u in self.sample(rng, n))

    def _require_control_vector(self, what: str) -> None:
        if self.mean.size != N_CONTROL:
            raise ValueError(
                f"{what} needs the Level A control vector of doc 05 §2.1 "
                f"({N_CONTROL} parameters); this posterior has {self.mean.size}. The engine "
                f"is dimension-agnostic on purpose, so that it can be verified against "
                f"low-dimensional problems with known answers."
            )

    def __repr__(self) -> str:
        """Carries the validity label, so a bare quote cannot lose it — doc 05 §5."""
        return (
            f"LaplacePosterior(d={self.mean.size}, "
            f"condition={self.condition_number:.3g}, {self.validity.value})"
        )


def laplace_posterior(
    mean: NDArray[np.float64],
    *,
    log_likelihood: LogLikelihood,
    prior: LogPosteriorPrior,
    step: float = _HESSIAN_STEP,
) -> LaplacePosterior:
    """Gaussian approximation to the posterior at ``mean`` — doc 05 §5's "+ one Hessian".

    Args:
        mean: The MAP point in unconstrained space —
            :attr:`~vpl.inverse.map.MapResult.unconstrained`. Taken as an argument rather
            than a :class:`~vpl.inverse.map.MapResult` so this module can be verified
            against problems whose Hessian is known in closed form, without running an
            optimiser first and confounding the two.
        log_likelihood: The same opaque callable :func:`~vpl.inverse.map.maximum_a_posteriori`
            takes, keeping the inverse layer ignorant of any forward model (doc 08 §3).
        prior: Anything satisfying :class:`~vpl.inverse.map.LogPosteriorPrior`. Its
            ``log_prob_unconstrained`` carries the log-Jacobian, so the curvature measured
            here is the curvature of the posterior actually being reported.
        step: Finite-difference step. See :data:`_HESSIAN_STEP`.

    Returns:
        A :class:`LaplacePosterior`, labelled :attr:`LaplaceValidity.UNVALIDATED`.

    Raises:
        PosteriorNotPositiveDefiniteError: If the Hessian has a non-positive eigenvalue, or
            one small enough relative to the largest to be doc 05 §6's null space rather
            than an uncertainty. Refused rather than regularised — see the module docstring.
    """
    point = np.asarray(mean, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError(f"the MAP point must be a vector, got shape {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError("the MAP point contains non-finite entries; there is no mode there")

    hessian = _hessian(point, log_likelihood=log_likelihood, prior=prior, step=step)
    # Symmetrise before the eigendecomposition. The mixed differences are symmetric in
    # exact arithmetic and differ in the last bits in practice; `eigvalsh` would silently
    # read only one triangle, so averaging is both cheaper and more honest than choosing.
    hessian = 0.5 * (hessian + hessian.T)
    curvature = np.linalg.eigvalsh(hessian)
    _require_positive_definite(hessian, curvature)

    return LaplacePosterior(
        mean=point,
        covariance=np.linalg.inv(hessian),
        precision=hessian,
        curvature=curvature,
        validity=LaplaceValidity.UNVALIDATED,
    )


def _hessian(
    point: FloatArray,
    *,
    log_likelihood: LogLikelihood,
    prior: LogPosteriorPrior,
    step: float,
) -> FloatArray:
    """Central-difference Hessian of the negative log posterior.

    The same objective :mod:`vpl.inverse.map` minimises, called through the same function,
    so the curvature reported here belongs to the surface the MAP actually descended. A
    second, locally re-derived objective would be a second thing to keep in step.
    """
    if not step > 0.0:
        raise ValueError(f"the finite-difference step must be positive, got {step}")

    def objective(u: FloatArray) -> float:
        return negative_log_posterior(u, log_likelihood=log_likelihood, prior=prior)

    size = point.size
    hessian = np.empty((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i, size):
            offset_i, offset_j = np.zeros(size), np.zeros(size)
            offset_i[i], offset_j[j] = step, step
            value = (
                objective(point + offset_i + offset_j)
                - objective(point + offset_i - offset_j)
                - objective(point - offset_i + offset_j)
                + objective(point - offset_i - offset_j)
            ) / (4.0 * step * step)
            hessian[i, j] = hessian[j, i] = value
    return hessian


def _require_positive_definite(hessian: FloatArray, curvature: FloatArray) -> None:
    """Refuse a Hessian that is not a positive-definite minimum, and say which direction.

    Naming the eigenvector matters more than naming the eigenvalue: *which combination of
    parameters* is unconstrained is a statement about what the measurement can and cannot
    see (doc 05 §6), and it is what a reader needs in order to decide whether to add a
    channel or to accept the degeneracy.
    """
    smallest = float(curvature[0])
    largest = float(curvature[-1])
    if largest <= 0.0:
        raise PosteriorNotPositiveDefiniteError(
            f"every eigenvalue of the Hessian is non-positive (largest {largest:.3e}), so "
            "this point is a maximum or a saddle of the negative log posterior, not a mode. "
            "A MAP that stopped here would still report converged=True."
        )
    if smallest <= 0.0 or smallest < _MINIMUM_RELATIVE_CURVATURE * largest:
        direction = np.linalg.eigh(hessian)[1][:, 0]
        dominant = int(np.argmax(np.abs(direction)))
        raise PosteriorNotPositiveDefiniteError(
            f"the Hessian is not positive definite: its smallest eigenvalue is "
            f"{smallest:.3e} against a largest of {largest:.3e}, a ratio of "
            f"{smallest / largest:.3e} below the {_MINIMUM_RELATIVE_CURVATURE:.0e} floor. "
            f"The unconstrained direction is {np.array2string(direction, precision=3)} "
            f"(dominated by coordinate {dominant}). This is doc 05 §6's null space, not a "
            f"wide error bar: inverting it would report an unidentified combination of "
            f"parameters as a measured one. Add an informative channel, fix a parameter, or "
            f"report the degeneracy — do not regularise it away."
        )
