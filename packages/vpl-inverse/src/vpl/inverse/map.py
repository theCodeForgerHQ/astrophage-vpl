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

## Multi-start, and why it exists

A single L-BFGS-B run reports the *local* optimum nearest its start; it has no way to tell
that a nearby-looking mode is not the global one. OES's likelihood surface was measured to
be exactly this kind of trap: over 15 independent single-start fits (matched physics,
noise on, imperfect calibration on), all 15 reported ``converged=True``, but only 6 landed
near the true optimum. 5 raised a singular Hessian downstream and 4 landed on a *wrong*
mode that also passed the positive-definiteness check — a confident, tight interval around
the wrong answer, which is worse than refusing to produce one.

``n_starts`` is the fix, and it is deliberately narrow in scope: this module still runs
exactly the same L-BFGS-B search it always has, just from more than one point, and reports
whichever run reached the lowest objective (best log posterior). Two things are pinned:

- **``n_starts=1`` is bit-for-bit identical to today's behaviour.** The first start is
  always the caller's ``initial`` (or the prior median, per the section above) and nothing
  reads :mod:`vpl.core.random` unless ``n_starts > 1`` — every number already published for
  this project was measured single-start, and this must not perturb any of them.
- **Every extra start is deterministic given ``seed``.** doc 00 E3's reproducibility story
  is one recorded seed per run; multi-start must not introduce a second source of
  randomness that a rerun cannot reproduce. The extra starts are drawn from
  ``vpl.core.random.generator(seed, Stream.MAP_MULTISTART)`` — the project's own per-stream
  scheme, so a run's optimiser starts are exactly as reproducible as its noise draws are.

**Perturb around the reference start, not a fresh draw from the prior.** The alternative —
drawing every start i.i.d. from the prior in unconstrained space — was considered and
rejected: several of doc 05 §2.1's priors are bounded box-uniforms whose *unconstrained*
support is the whole real line near the box edges (the logit blows up), so a prior draw can
land absurdly far from anything physically plausible and burn its whole iteration budget
walking back. A Gaussian perturbation of the reference start stays where the objective is
actually evaluable — unconstrained coordinates are O(1) by the transform's construction —
while still being large enough to cross into a neighbouring basin of attraction, which is
the only property multi-start actually needs.

**Every extra start is bounded to the prior's own declared support, not merely centred on
the reference.** Measured directly: at ``n_starts=10`` on OES alone, a handful of the wider
perturbations landed somewhere ``OesInstrument``'s own likelihood normalisation underflows
to zero and raises — a caller-side defect this module cannot fix (its ``log_likelihood`` is
deliberately opaque, per the simplification section below), but a start this module handed
to the optimiser regardless. :func:`_bounded_perturbation` rejects and redraws any candidate
whose prior support is not finite, which is the one validity check available generically —
without depending on any particular forward model — through :class:`LogPosteriorPrior`
itself. The alternative, wrapping the optimisation in a broad ``except`` and skipping the
start, was rejected on purpose: a start that disappears into a caught exception makes
``n_starts`` a number the result cannot actually back up.

## What "distinct modes" means here, and why it is reported

doc 00's house style is that a claim a reader can check beats one they must trust. "The
surface is multimodal" is exactly such a claim, and :attr:`MapResult.n_distinct_modes`
makes it checkable per fit rather than only across an externally-run ensemble: any two
starts whose optimised unconstrained points are farther apart than
:data:`_MODE_CLUSTER_TOLERANCE` are counted as different modes. It is a strict local
diagnostic (all starts share the objective function and prior of a single call) and is not
a claim about *global* mode enumeration — a start that never found a third mode does not
prove one does not exist.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from vpl.core.random import Stream, generator
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

#: Gradient infinity-norm below which a point is certified a stationary point regardless of
#: what the optimiser reported. L-BFGS-B returns ``ABNORMAL_TERMINATION_IN_LNSRCH`` when its
#: line search stops making progress, and at the tolerance above that happens routinely once
#: the **finite-difference** gradient reaches its own noise floor — which for
#: :data:`_GRADIENT_STEP` and an O(1) objective is around 1e-8. The point is then already the
#: optimum to ~1e-9, and calling it a failure is a false negative.
#:
#: That false negative is expensive rather than cosmetic: measured on 60 linear-Gaussian
#: problems it fired on 3 of them, i.e. 5 % of an ensemble, and
#: :mod:`vpl.validation.coverage` refuses to report coverage at all past a 10 % failure rate.
#: So ``converged`` means "this is a mode", not "SciPy was happy". A saddle with a
#: coincidentally small gradient is not admitted by this: the Hessian check in
#: :func:`~vpl.inverse.laplace.laplace_posterior` rejects non-positive curvature separately.
_STATIONARITY_TOLERANCE: Final[float] = 1e-6

#: Relative function tolerance, tightened for the same reason. ``ftol`` is expressed by
#: L-BFGS-B as a multiple of machine epsilon; this asks it to stop on the gradient rather
#: than on a function-value plateau, which is the more meaningful criterion at an optimum.
_FUNCTION_TOLERANCE: Final[float] = 1e-15

#: Standard deviation of the Gaussian perturbation added to the reference start for the
#: extra points a multi-start search adds beyond the first — see the module docstring's
#: "Multi-start" section for why a perturbation rather than a fresh prior draw. A few units
#: in a space that is O(1) by construction is enough to reach a neighbouring basin without
#: routinely leaving the region the objective evaluates sensibly in.
_MULTI_START_PERTURBATION_SCALE: Final[float] = 3.0

#: Two optimised points farther apart than this (Euclidean, unconstrained space) are counted
#: as different modes for :attr:`MapResult.n_distinct_modes`. Set well above the optimiser's
#: own convergence tolerance (:data:`_STATIONARITY_TOLERANCE` operates on the gradient, not
#: on position, but a converged L-BFGS-B run is typically within 1e-6 to 1e-4 of its true
#: local optimum) and well below :data:`_MULTI_START_PERTURBATION_SCALE`, so it separates
#: "two starts converged to the same point" from "two starts converged to different points"
#: without being sensitive to the exact perturbation size chosen above.
_MODE_CLUSTER_TOLERANCE: Final[float] = 1e-2

#: How many times :func:`_bounded_perturbation` will redraw a single extra start before
#: giving up and falling back to the reference point — see its docstring for why a draw is
#: ever rejected at all. Generous rather than tight: a rejection only costs a prior
#: evaluation (cheap; no forward model runs), so there is no reason to economise on
#: attempts, and doc 05 §2.1's tightest boxes (e.g. ``kappa``'s ``[1, 5]``) can need several
#: redraws at :data:`_MULTI_START_PERTURBATION_SCALE` before landing inside.
_MAX_PERTURBATION_ATTEMPTS: Final[int] = 20

#: Factor the perturbation scale is multiplied by after each rejected draw within one
#: start's attempt budget. A rejected draw means the *current* scale is too large for this
#: particular coordinate's prior box; shrinking geometrically rather than retrying the same
#: scale converges toward a valid point within the attempt budget instead of repeating the
#: same failure, while still trying the full requested scale first.
_PERTURBATION_SHRINK: Final[float] = 0.7


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
        n_starts: How many starting points this call actually optimised from. ``1`` for
            every call that did not ask for multi-start (the default), so this field alone
            tells a reader whether a given number is comparable to the project's
            single-start baseline.
        n_distinct_modes: How many of those starts converged to a point counted as a
            distinct optimum — see the module docstring's "What 'distinct modes' means
            here" section. ``1`` whenever ``n_starts == 1``, since one run cannot see a
            second mode; a value ``> 1`` is a checkable, per-fit demonstration that the
            surface is multimodal, not merely an assertion from a separate ensemble.
        start_objectives: The final objective from every start, in the order they were run
            (index 0 is always the reference start). Kept so a caller auditing a single fit
            can see the actual spread between modes, not just the count.
    """

    unconstrained: NDArray[np.float64]
    parameters: ControlParameters | None
    objective: float
    gradient: NDArray[np.float64]
    converged: bool
    iterations: int
    message: str
    n_starts: int = 1
    n_distinct_modes: int = 1
    start_objectives: tuple[float, ...] = ()


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
    n_starts: int = 1,
    seed: int = 0,
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
            Applied identically to every start when ``n_starts > 1``.
        n_starts: How many points to optimise from. ``1`` (the default) runs exactly the
            single-start search this engine has always run, from exactly the point
            ``initial``/the prior median would have given it — see the module docstring's
            "Multi-start" section for why this is bit-for-bit unchanged. ``n_starts > 1``
            adds ``n_starts - 1`` further starts, deterministic given ``seed``, and returns
            whichever run reached the best (lowest) objective.
        seed: Root seed for the extra starts a multi-start search adds. Unused, and reads
            no randomness at all, when ``n_starts == 1``. Should be the same seed the
            caller's run is already recorded under (doc 00 E3) rather than an
            independently chosen number, so the run's reproducibility story stays single-
            seeded.

    Returns:
        A :class:`MapResult`. Non-convergence is reported, never raised: an ensemble driver
        running doc 11 WBS 3.1's thousands of cases needs to record a failed point and carry
        on, not stop.

    Raises:
        ValueError: If ``n_starts < 1``.
    """
    if n_starts < 1:
        raise ValueError(f"n_starts must be at least 1, got {n_starts}")

    reference_start = (
        _default_start(prior) if initial is None else np.asarray(initial, dtype=np.float64)
    )

    def objective(u: NDArray[np.float64]) -> float:
        return negative_log_posterior(u, log_likelihood=log_likelihood, prior=prior)

    def jacobian(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _gradient(u, log_likelihood=log_likelihood, prior=prior)

    starts = _start_points(reference_start, n_starts=n_starts, seed=seed, prior=prior)
    results = [
        _optimise_from(s, objective, jacobian, max_iterations=max_iterations) for s in starts
    ]

    best = min(results, key=lambda r: r.objective)
    distinct = _count_distinct_modes([r.unconstrained for r in results])
    return MapResult(
        unconstrained=best.unconstrained,
        parameters=best.parameters,
        objective=best.objective,
        gradient=best.gradient,
        converged=best.converged,
        iterations=best.iterations,
        message=best.message,
        n_starts=n_starts,
        n_distinct_modes=distinct,
        start_objectives=tuple(r.objective for r in results),
    )


def _optimise_from(
    start: NDArray[np.float64],
    objective: Callable[[NDArray[np.float64]], float],
    jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    max_iterations: int,
) -> MapResult:
    """One L-BFGS-B run from ``start`` — the single-start search, unchanged."""
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


def _start_points(
    reference: NDArray[np.float64], *, n_starts: int, seed: int, prior: LogPosteriorPrior
) -> list[NDArray[np.float64]]:
    """``n_starts`` deterministic starting points, the first always ``reference``.

    Starts beyond the first are ``reference`` plus i.i.d. Gaussian noise, **bounded to the
    prior's own declared support** — see :func:`_bounded_perturbation` for why and how.
    Drawn from :data:`~vpl.core.random.Stream.MAP_MULTISTART`, so that stream is never
    touched at all when ``n_starts == 1``.
    """
    if n_starts == 1:
        return [reference]
    rng = generator(seed, Stream.MAP_MULTISTART)
    extra = [_bounded_perturbation(reference, rng=rng, prior=prior) for _ in range(n_starts - 1)]
    return [reference, *extra]


def _bounded_perturbation(
    reference: NDArray[np.float64], *, rng: np.random.Generator, prior: LogPosteriorPrior
) -> NDArray[np.float64]:
    """One extra start: ``reference`` plus Gaussian noise, rejected and redrawn if it lands
    outside the prior's own support.

    A wide-enough perturbation to cross into a neighbouring basin can also land somewhere a
    caller-supplied ``log_likelihood`` breaks in a way this module cannot generically guard
    against — it is an opaque callable by design (see the module docstring's simplification
    section and doc 08 §3's layering), so it may do anything from a physics forward model up
    to raising an exception this module has never seen. What this module *can* check without
    depending on that forward model is the one thing every :class:`LogPosteriorPrior`
    already commits to reporting honestly: whether a point has finite prior support at all.
    Checking that before a start is ever handed to the optimiser is the difference between
    fixing the draw and papering over its symptom with a broad ``except`` clause around the
    optimisation itself — which was considered and rejected, because a start that silently
    disappears into a caught exception makes ``n_starts`` a claim the result cannot back up.

    Redraws at a shrinking scale (:data:`_PERTURBATION_SHRINK` per rejection) for up to
    :data:`_MAX_PERTURBATION_ATTEMPTS` tries — cheap, since each attempt costs one prior
    evaluation and no forward-model calls at all — and falls back to ``reference`` itself if
    every attempt is rejected. That fallback is never silent: it is a valid, already-known
    point rather than a missing one, and :func:`_count_distinct_modes` reports it as a
    duplicate of the base start rather than pretending it explored anywhere new.
    """
    scale = _MULTI_START_PERTURBATION_SCALE
    for _ in range(_MAX_PERTURBATION_ATTEMPTS):
        candidate = reference + rng.normal(scale=scale, size=reference.shape)
        if np.isfinite(prior.log_prob_unconstrained(candidate)):
            return candidate
        scale *= _PERTURBATION_SHRINK
    return reference.copy()


def _count_distinct_modes(points: list[NDArray[np.float64]]) -> int:
    """Single-linkage cluster count at :data:`_MODE_CLUSTER_TOLERANCE` — see module doc.

    ``O(n^2)`` in the number of starts, which is fine: ``n_starts`` is a handful, not a
    sweep parameter.
    """
    clusters: list[NDArray[np.float64]] = []
    for point in points:
        if not any(
            np.linalg.norm(point - representative) <= _MODE_CLUSTER_TOLERANCE
            for representative in clusters
        ):
            clusters.append(point)
    return len(clusters)


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
    # Conversely, a point whose gradient has vanished *is* a mode even when the optimiser
    # reported an abnormal line-search termination — see _STATIONARITY_TOLERANCE.
    stationary = bool(np.max(np.abs(gradient)) <= _STATIONARITY_TOLERANCE)
    healthy = (converged or stationary) and np.isfinite(value) and value < _OUT_OF_SUPPORT
    return MapResult(
        unconstrained=u,
        parameters=parameters,
        objective=value,
        gradient=gradient,
        converged=bool(healthy),
        iterations=iterations,
        message=message,
    )
