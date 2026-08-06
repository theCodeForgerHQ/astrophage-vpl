"""Maximum a posteriori estimation — doc 05 §5, doc 11 WBS 3.7.

doc 05 §5 lists MAP first, for two jobs: a fast point estimate, and initialisation for the
more expensive engines. Both matter, and the second is the reason this module is careful
about a detail that would otherwise look pedantic — see :func:`negative_log_posterior`.

## The objective, and the two ways it goes wrong

The quantity minimised is

    -(log L(u) + log pi(u))

with ``u`` the **unconstrained** parameter vector of :mod:`vpl.inverse.parameters`, and
``log pi`` the prior **including the log-Jacobian of the transform**. Two failure modes,
both of which converge cleanly to the wrong answer:

- **Dropping the prior.** The optimiser then finds the maximum-likelihood estimate, which
  is a perfectly good number and is not the MAP. Nothing about the convergence looks wrong.
- **Getting the Jacobian wrong** — omitting it, or applying it twice. The estimate is then
  biased toward or away from the transform's fixed point by a smooth, plausible amount. The
  priors module already measured what this costs: without the Jacobian, the transformed
  priors integrate to 0.0, 0.361 and 40.0 instead of 1.

So this module never calls the constrained ``log_prob``. It calls
``log_prob_unconstrained``, which carries the Jacobian, and a test asserts the two differ so
the check cannot become vacuous.

## Why bounded L-BFGS-B rather than plain L-BFGS

Several of doc 05 §2.1's priors are bounded — the log-uniform box survives into log space,
and the logit-transformed parameters are unbounded in ``u`` but their *finite* prior support
means the objective is ``+inf`` outside. Feeding ``+inf`` to an unbounded quasi-Newton
method wastes line-search evaluations and occasionally strands it. Passing the box to
L-BFGS-B keeps every trial point inside the support.

## Why the default start is the prior median

``ControlParameters.reference()`` is RP-1 from the registry, and RP-1's ``eedf.kappa`` is
1.0 — which is *exactly* the lower edge of doc 05 §2.1's ``uniform [1, 5]``. Its logit is
``-inf``, so an engine initialised at the reference initialises nowhere. That is
mathematically correct behaviour and a genuine trap, so the default here is
``prior.median()``, which is interior by construction.

## Simplification, stated

The gradient is obtained by finite differences, not by autodiff or an adjoint. doc 05 §5
says "gradients from JAX/adjoint **where available**", and they are not available through an
arbitrary caller-supplied ``log_likelihood`` — which is the signature this engine needs in
order to stay ignorant of the forward model, per doc 08 §3's layering. The cost is
``n+1`` objective evaluations per gradient, i.e. 9 at the Level A dimension; the accuracy
cost is bounded by the step size and is checked against a central difference in the tests.
When the L3 surrogate lands and is differentiable, this is the function to give an analytic
gradient to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from vpl.inverse.parameters import ControlParameters

__all__ = [
    "LogPosteriorPrior",
    "MapResult",
    "maximum_a_posteriori",
    "negative_log_posterior",
]

type LogLikelihood = Callable[[NDArray[np.float64]], float]

#: Returned in place of the objective where the prior has no support. Large and finite
#: rather than ``inf``: L-BFGS-B's line search handles a big number gracefully and can be
#: derailed by a non-finite one, and the gradient of ``inf`` is ``nan``.
_OUT_OF_SUPPORT: Final[float] = 1e30

#: Central-difference step in unconstrained space. The coordinates are O(1) there by
#: construction — that is the point of the transform — so a single absolute step is
#: appropriate and there is no scale to chase.
_GRADIENT_STEP: Final[float] = 1e-6

#: doc 05 §5 puts MAP at 10^2-10^3 evaluations. This bounds a pathological case rather than
#: a normal one; hitting it is reported as non-convergence, never silently returned.
_DEFAULT_MAX_ITERATIONS: Final[int] = 500

#: Projected-gradient tolerance. SciPy's L-BFGS-B default is 1e-5, which stops while the
#: estimate is still ~1e-6 from the true mode on a well-conditioned quadratic. That is
#: comfortably good enough for a point estimate and *not* good enough for doc 05 §7.2's T0,
#: which requires "recovery to numerical tolerance" and treats a shortfall as a bug rather
#: than a result. Tightened deliberately: the cost is a few extra iterations, and the
#: alternative was to loosen the verification test until the default passed.
_GRADIENT_TOLERANCE: Final[float] = 1e-10

#: Relative function tolerance, tightened for the same reason. ``ftol`` is expressed by
#: L-BFGS-B as a multiple of machine epsilon; this asks it to stop on the gradient rather
#: than on a function-value plateau, which is the more meaningful criterion at an optimum.
_FUNCTION_TOLERANCE: Final[float] = 1e-15


@runtime_checkable
class LogPosteriorPrior(Protocol):
    """The only thing this engine needs from a prior.

    A protocol rather than the concrete :class:`~vpl.inverse.priors.ControlPrior` so that a
    test can supply an analytically tractable prior — which is how the closed-form recovery
    test is possible at all — and so the engine stays usable for the Level B and C vectors
    of doc 05 §2.2 and §2.3 when they arrive.
    """

    def log_prob_unconstrained(self, u: NDArray[np.float64]) -> float:
        """Log prior density in unconstrained space, **including** the log-Jacobian."""
        ...


@dataclass(frozen=True, slots=True)
class MapResult:
    """The MAP estimate and enough context to judge whether to believe it.

    Attributes:
        unconstrained: The optimum in unconstrained space, which is what the sampler and
            the Laplace approximation both want.
        parameters: The same point in physical units, or ``None`` when the vector is not
            the Level A control vector. Carried so that callers reason in eV and m^-3
            rather than in logits; leaving them to invert the transform themselves means
            one call site eventually gets it wrong.

            It is ``None`` rather than an error for anything else because the engine is
            deliberately dimension-agnostic — doc 05 §2.2 and §2.3 add Level B and C
            vectors, and the closed-form verification problems are necessarily
            low-dimensional. An engine that could only run on exactly eight parameters
            could not be checked against a case with a known answer, which is the only kind
            of check worth having (ADR-011).
        objective: The minimised negative log posterior.
        gradient: The gradient there. Should be near zero at a true optimum, and is the
            cheapest available check that it is one.
        converged: Whether the optimiser reported success. A MAP that silently returns its
            iteration limit is how a bad fit becomes a published number.
        iterations: How many iterations were taken.
        message: The optimiser's own explanation, kept verbatim.
    """

    unconstrained: NDArray[np.float64]
    parameters: ControlParameters | None
    objective: float
    gradient: NDArray[np.float64]
    converged: bool
    iterations: int
    message: str


def negative_log_posterior(
    u: NDArray[np.float64],
    *,
    log_likelihood: LogLikelihood,
    prior: LogPosteriorPrior,
) -> float:
    """``-(log L + log pi)`` in unconstrained space — the objective, and nothing else.

    Returns :data:`_OUT_OF_SUPPORT` rather than raising when the prior has no support at
    ``u``. L-BFGS-B *will* step outside a bounded support during its line search; raising
    there aborts an otherwise healthy optimisation, whereas a large finite value tells it to
    back off.
    """
    log_prior = prior.log_prob_unconstrained(u)
    if not np.isfinite(log_prior):
        return _OUT_OF_SUPPORT

    value = log_likelihood(u)
    if not np.isfinite(value):
        return _OUT_OF_SUPPORT

    return float(-(value + log_prior))


def _gradient(
    u: NDArray[np.float64], *, log_likelihood: LogLikelihood, prior: LogPosteriorPrior
) -> NDArray[np.float64]:
    """Central-difference gradient of :func:`negative_log_posterior`.

    Central rather than forward: the extra ``n`` evaluations buy second-order accuracy, and
    at eight parameters the cost is irrelevant next to a forward-model evaluation. A
    one-sided difference near a bound would also silently sample outside the support.
    """
    gradient = np.empty_like(u)
    for i in range(u.size):
        plus, minus = u.copy(), u.copy()
        plus[i] += _GRADIENT_STEP
        minus[i] -= _GRADIENT_STEP
        gradient[i] = (
            negative_log_posterior(plus, log_likelihood=log_likelihood, prior=prior)
            - negative_log_posterior(minus, log_likelihood=log_likelihood, prior=prior)
        ) / (2.0 * _GRADIENT_STEP)
    return gradient


def maximum_a_posteriori(
    *,
    log_likelihood: LogLikelihood,
    prior: LogPosteriorPrior,
    initial: NDArray[np.float64] | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> MapResult:
    """Maximise the posterior in unconstrained space — doc 05 §5's MAP engine.

    Args:
        log_likelihood: Takes the unconstrained vector, returns a log density. Deliberately
            an opaque callable: doc 08 §3 keeps the inverse layer ignorant of any particular
            forward model, which is what makes doc 05 §7.1's inverse-crime mismatch
            structurally possible rather than a matter of discipline.
        prior: Anything satisfying :class:`LogPosteriorPrior`.
        initial: Starting point. Defaults to the prior median — see the module docstring for
            why *not* the RP-1 reference.
        max_iterations: Iteration cap. ``0`` evaluates the objective and gradient at the
            starting point without optimising, which is how the gradient check is done.

    Returns:
        A :class:`MapResult`. Non-convergence is reported, never raised: an ensemble driver
        running doc 11 WBS 3.1's thousands of cases needs to record a failed point and carry
        on, not stop.
    """
    start = _default_start(prior) if initial is None else np.asarray(initial, dtype=np.float64)

    def objective(u: NDArray[np.float64]) -> float:
        return negative_log_posterior(u, log_likelihood=log_likelihood, prior=prior)

    def jacobian(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _gradient(u, log_likelihood=log_likelihood, prior=prior)

    if max_iterations == 0:
        return _result_at(
            start,
            objective,
            jacobian,
            converged=False,
            iterations=0,
            message="evaluated without optimising (max_iterations=0)",
        )

    outcome = minimize(
        objective,
        start,
        jac=jacobian,
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "gtol": _GRADIENT_TOLERANCE,
            "ftol": _FUNCTION_TOLERANCE,
        },
    )
    return _result_at(
        np.asarray(outcome.x, dtype=np.float64),
        objective,
        jacobian,
        converged=bool(outcome.success),
        iterations=int(outcome.nit),
        message=str(outcome.message),
    )


def _default_start(prior: LogPosteriorPrior) -> NDArray[np.float64]:
    """The prior median where the prior can supply one, else the origin.

    The origin is a sound fallback in unconstrained space precisely because the transforms
    are built to make it O(1) there; it is not a guess about the physics.
    """
    median = getattr(prior, "median", None)
    if median is None:
        return np.zeros(len(ControlParameters.__dataclass_fields__), dtype=np.float64)
    return np.asarray(median().to_unconstrained(), dtype=np.float64)


def _result_at(
    u: NDArray[np.float64],
    objective: Callable[[NDArray[np.float64]], float],
    jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    converged: bool,
    iterations: int,
    message: str,
) -> MapResult:
    value = objective(u)
    gradient = jacobian(u)
    # Only the Level A vector maps back to named physical quantities. Anything else — a
    # Level B/C vector, or one of the low-dimensional problems the engine is verified on —
    # simply has no ControlParameters representation, and inventing one would be worse than
    # reporting its absence.
    parameters = (
        ControlParameters.from_unconstrained(u)
        if u.size == len(ControlParameters.__dataclass_fields__)
        else None
    )
    # A point the objective cannot evaluate is not an estimate, whatever the optimiser said.
    healthy = converged and np.isfinite(value) and value < _OUT_OF_SUPPORT
    return MapResult(
        unconstrained=u,
        parameters=parameters,
        objective=value,
        gradient=gradient,
        converged=bool(healthy),
        iterations=iterations,
        message=message,
    )
