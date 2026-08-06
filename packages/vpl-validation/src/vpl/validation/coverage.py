"""The coverage ensemble driver — doc 11 §9 item 5, doc 05 §5, doc 11 WBS 3.7.

doc 11 §9 item 5 is "Coverage test, 400+ cases — uncertainty that has been *checked*, the
strongest differentiator", and it is the only item on that critical path marked **Do not
cut**. :mod:`vpl.validation.calibration` holds its statistics. This module runs the
ensemble that feeds them.

## What one case is

A coverage case is a truth, a likelihood and a prior. The driver runs
:func:`~vpl.inverse.map.maximum_a_posteriori`, then
:func:`~vpl.inverse.laplace.laplace_posterior`, then takes the probability integral
transform of the truth under the resulting marginals. Nothing about it is specific to a
sheath, an instrument or a fidelity level: the driver takes an opaque
:class:`CoverageCase`, exactly as :mod:`vpl.inverse.map` takes an opaque likelihood, so
that doc 08 §3's layering holds and the same ensemble machinery serves the L0 closed loop,
a T2 mismatched run, and whatever replaces them.

## Why the starting point is part of the case

``CoverageCase.start`` is a required field with no default, and it must not be derived from
the truth. doc 05 §7 names "an initial guess, just to start the optimiser somewhere
sensible" as one of the canonical ways an inverse crime is committed by accident — and in
an ensemble the temptation is far stronger than in a single run, because the truth is
sitting right there in the loop variable for every one of the 400 cases. Seeding from it
would make coverage look excellent and mean nothing.

The driver therefore touches ``truth_unconstrained`` at exactly one point: computing the
PIT, after the posterior already exists. A test asserts the access count.

## The bias that would sink this quietly

Some cases fail. A MAP hits its iteration limit; a Hessian comes back singular because
doc 05 §6.2's ``n_0``-``T_e`` ridge went flat for that particular draw. The obvious
implementation catches the exception, skips the case, and reports coverage over the
survivors.

That is **survivorship bias, and it is optimistic**, because the cases that fail are
precisely the badly-conditioned ones whose posteriors were least trustworthy. A run that
discards a third of its ensemble and reports 90 % coverage over the rest has reported
nothing, and nothing in the output would look wrong. So every attempt is counted, failures
are reported by cause, and past :data:`MAXIMUM_FAILURE_FRACTION` the driver refuses to
produce a coverage number at all — at that point the survivors are not a sample of the
intended ensemble, and a number computed from them is worse than no number.

A singular Hessian is a *result* rather than a crash: it identifies a null direction doc 05
§6 predicts (ADR-012). It is counted separately from a MAP that simply failed to converge,
because the two say different things about the experiment.

## A stated limitation of pooling

For a ``d``-dimensional posterior the driver pools ``d`` marginal PIT values per case. Each
marginal is individually Uniform(0, 1) under calibration, so the *coverage fractions* are
unbiased estimates of marginal coverage — which is what "our 90 % intervals contain the
truth 90 % of the time" means for a parameter.

The ``d`` values within one case are **not independent**, however, so the KS p-value
computed over the pooled sample has a smaller effective sample size than its nominal one and
is correspondingly optimistic. It remains a sound *ordering* statistic — it will not call a
badly miscalibrated ensemble calibrated — but the p-value should not be quoted as though the
sample were independent. Reporting per-parameter PITs separately would remove the caveat and
is the natural refinement when there is time for it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.inverse.laplace import PosteriorNotPositiveDefiniteError, laplace_posterior
from vpl.inverse.map import LogPosteriorPrior, maximum_a_posteriori
from vpl.validation.calibration import CoverageReport, coverage_report, gaussian_pit

__all__ = [
    "MAXIMUM_FAILURE_FRACTION",
    "CoverageCase",
    "CoverageEnsembleResult",
    "EnsembleTooDepletedError",
    "run_coverage_ensemble",
]

type FloatArray = NDArray[np.float64]
type LogLikelihood = Callable[[NDArray[np.float64]], float]

#: Fraction of cases that may fail before the coverage number is refused rather than
#: reported. One in ten: enough headroom that a handful of genuinely degenerate draws out of
#: 400 do not abort a run, and tight enough that the surviving ensemble is still recognisably
#: the intended one. Above it, survivorship bias dominates whatever the point estimate says.
MAXIMUM_FAILURE_FRACTION: Final[float] = 0.1


class EnsembleTooDepletedError(RuntimeError):
    """Too many cases failed for the surviving coverage estimate to mean anything."""


@dataclass(frozen=True, slots=True)
class CoverageCase:
    """One case of the coverage ensemble.

    Attributes:
        truth_unconstrained: The generating parameters, in unconstrained coordinates. Read
            **only** to compute the PIT, after the posterior exists — see the module
            docstring on why this is load-bearing rather than fussy.
        log_likelihood: The case's likelihood, as an opaque callable (doc 08 §3).
        prior: The prior used by the inversion. For coverage to mean anything this must be
            the distribution the truth was actually drawn from; measuring calibration
            against a different reference measures nothing.
        start: Where the optimiser starts. Required, with no default, and it must not come
            from the truth: doc 05 §7 names exactly that as an accidental inverse crime.
    """

    truth_unconstrained: FloatArray
    log_likelihood: LogLikelihood
    prior: LogPosteriorPrior
    start: FloatArray


@dataclass(frozen=True, slots=True)
class CoverageEnsembleResult:
    """doc 11 §9 item 5's result, with its denominator visible.

    Attributes:
        report: The calibration statistics over the cases that produced a posterior.
        n_attempted: Cases offered. doc 11 §9's "400+" is a claim about this number.
        n_used: Cases that produced a posterior and contributed to :attr:`report`.
        n_singular: Cases refused for a non-positive-definite Hessian — doc 05 §6's null
            space (ADR-012). A result about the experiment, not a crash.
        n_not_converged: Cases whose MAP did not converge. Counted apart from
            :attr:`n_singular` because the two say different things.
    """

    report: CoverageReport
    n_attempted: int
    n_used: int
    n_singular: int
    n_not_converged: int

    @property
    def failure_fraction(self) -> float:
        """Share of offered cases that produced no posterior."""
        return 0.0 if self.n_attempted == 0 else 1.0 - self.n_used / self.n_attempted

    def __repr__(self) -> str:
        return (
            f"CoverageEnsembleResult({self.n_used}/{self.n_attempted} cases used, "
            f"{self.n_singular} singular, {self.n_not_converged} unconverged, "
            f"{self.report!r})"
        )


def run_coverage_ensemble(
    cases: Sequence[CoverageCase], *, levels: tuple[float, ...]
) -> CoverageEnsembleResult:
    """Run the ensemble and measure how well its credible intervals are calibrated.

    Args:
        cases: The ensemble. doc 11 §9 item 5 asks for 400 or more.
        levels: Nominal credible levels to report coverage at.

    Returns:
        A :class:`CoverageEnsembleResult`.

    Raises:
        EnsembleTooDepletedError: If more than :data:`MAXIMUM_FAILURE_FRACTION` of the cases
            produced no posterior. Refusing is the point — see the module docstring.
    """
    if len(cases) == 0:
        raise ValueError(
            "cannot measure coverage over an empty ensemble; there is no number to report"
        )

    pit_values: list[FloatArray] = []
    n_singular = 0
    n_not_converged = 0

    for case in cases:
        outcome = _posterior_for(case)
        if outcome is None:
            n_not_converged += 1
            continue
        if isinstance(outcome, str):
            n_singular += 1
            continue
        mean, std = outcome
        # The single point at which the truth is read, and it is after the posterior
        # exists. Everything above this line ran without it.
        truth = np.asarray(case.truth_unconstrained, dtype=np.float64)
        pit_values.append(gaussian_pit(truth, mean=mean, std=std))

    n_used = len(pit_values)
    failed = len(cases) - n_used
    if failed > MAXIMUM_FAILURE_FRACTION * len(cases):
        raise EnsembleTooDepletedError(
            f"{failed} of {len(cases)} cases produced no posterior "
            f"({n_singular} singular, {n_not_converged} unconverged), above the "
            f"{MAXIMUM_FAILURE_FRACTION:.0%} limit. Coverage over the survivors would be "
            f"optimistic by survivorship: the cases that fail are the badly-conditioned "
            f"ones whose posteriors were least trustworthy. Fix the conditioning, or "
            f"report the failure rate as the result."
        )

    return CoverageEnsembleResult(
        report=coverage_report(np.concatenate(pit_values), levels=levels),
        n_attempted=len(cases),
        n_used=n_used,
        n_singular=n_singular,
        n_not_converged=n_not_converged,
    )


def _posterior_for(case: CoverageCase) -> tuple[FloatArray, FloatArray] | str | None:
    """``(mean, marginal_std)`` for one case, ``"singular"``, or ``None`` if MAP failed.

    Three outcomes rather than an exception, because an ensemble driver over doc 11 WBS
    3.1's thousands of cases must record a failure and carry on rather than stop — and
    because the two failure causes are counted separately.
    """
    result = maximum_a_posteriori(
        log_likelihood=case.log_likelihood, prior=case.prior, initial=case.start
    )
    if not result.converged:
        return None
    try:
        posterior = laplace_posterior(
            result.unconstrained, log_likelihood=case.log_likelihood, prior=case.prior
        )
    except PosteriorNotPositiveDefiniteError:
        return "singular"
    return posterior.mean, posterior.marginal_std
