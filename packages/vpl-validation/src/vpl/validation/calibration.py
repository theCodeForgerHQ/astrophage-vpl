"""Calibration statistics for the coverage test — doc 05 §5, doc 11 §9 item 5.

doc 11 §9 lists the coverage test as item 5 — "uncertainty that has been *checked* — the
strongest differentiator" — and marks it **Do not cut**. It is the only item on that list
carrying that instruction, and this module is the statistics behind it.

## The claim being checked

A posterior is *calibrated* if its 90 % credible interval contains the truth in 90 % of
cases. That is a falsifiable statement about an ensemble, and it is the only statement in
this project that turns "we report uncertainty" into "our uncertainty has been tested".

## The error this module exists to prevent

The naive implementation counts hits at one level and prints a fraction. That fraction is
itself an estimate from a finite ensemble and carries its own uncertainty: at doc 11 §9's
400 cases, the standard error on a coverage estimate near 0.9 is
``sqrt(0.9 x 0.1 / 400) ~ 1.5 %``, so the 95 % interval around it is roughly +/- 3 %.

A measured 0.88 is therefore **not** evidence of overconfidence — it is entirely consistent
with a perfectly calibrated posterior — and 0.91 is not evidence of success. Reporting
either as if it were is the same mistake with opposite signs. A project whose central claim
is calibrated uncertainty cannot make an uncalibrated statement about its own calibration,
so :class:`CoverageReport` has no way to report a coverage number without its interval.

The interval is Wilson's score interval rather than the normal approximation. At the edges
the normal approximation degenerates: 20 hits from 20 cases gives ``[1, 1]`` — "coverage is
exactly 100 %, with certainty", from twenty cases — and it can also produce bounds outside
``[0, 1]``. Wilson does neither, and costs one extra line.

## Why the PIT rather than a hit count

Counting hits at one level discards nearly everything the ensemble measured. The
probability integral transform keeps it: if the truth is drawn from the posterior then
``F_posterior(truth)`` is Uniform(0, 1) — exactly, for any posterior. One array of PIT
values therefore tests *every* credible level at once, and a Kolmogorov-Smirnov test
against the uniform is a single decision rule.

It also separates two failure modes that a coverage number alone cannot, and which matter
very differently:

- **Overconfident** — intervals too narrow, truth outside more often than advertised. PIT
  piles up at 0 and 1. This is the one that misleads a reader.
- **Underconfident** — intervals too wide. PIT piles up at 0.5. Wasteful, not misleading.
- **Biased** — right width, wrong place. PIT is skewed to one side. Coverage at a single
  level can look almost correct while the distribution is visibly wrong, which is the
  clearest argument for testing the whole PIT rather than one number.

:attr:`CoverageReport.diagnosis` reports which, because "not calibrated" alone does not tell
a reader whether the published intervals are too generous or too mean.

## What this module does not establish

Calibration is a statement about an *ensemble average*. A set of intervals can be
well-calibrated on average while being wrong case by case — over-wide in one region of
parameter space and over-narrow in another, cancelling. Passing here is necessary, not
sufficient, and it does not upgrade
:class:`~vpl.inverse.laplace.LaplaceValidity.UNVALIDATED`, which is a statement about the
Gaussian *shape* assumption that doc 05 §5 says must be tested against NUTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri
from scipy.stats import kstest

__all__ = [
    "COVERAGE_INTERVAL_LEVEL",
    "CoverageReport",
    "coverage_report",
    "gaussian_pit",
    "wilson_interval",
]

type FloatArray = NDArray[np.float64]

#: Confidence level of the interval placed *around each coverage estimate*. Distinct from
#: the credible levels being tested, and easy to conflate with them: 0.9 here would mean
#: "a 90 % confidence interval on the measured coverage of the 90 % credible interval".
COVERAGE_INTERVAL_LEVEL: Final[float] = 0.95

#: KS p-value below which the PIT is judged not uniform. Lower than the conventional 0.05
#: deliberately: at doc 11 §9's 400+ cases the test has ample power against the
#: miscalibrations that matter (a factor of two in the interval width is caught with
#: p < 1e-30), so the binding consideration is the false-alarm rate on a figure that will be
#: presented — at 0.05 a perfectly calibrated run is condemned one time in twenty.
_UNIFORMITY_THRESHOLD: Final[float] = 0.01

#: Mean absolute deviation of a Uniform(0, 1) variate from its own centre, ``E|U - 1/2|``.
#: The dividing line between PIT mass at the edges (overconfident) and at the centre
#: (underconfident). Exact, not fitted.
_UNIFORM_MEAN_ABSOLUTE_DEVIATION: Final[float] = 0.25

#: Standard deviations of the PIT mean away from 1/2 before the miscalibration is reported
#: as a *shift* rather than as a width error. Three, so that ordinary sampling scatter on a
#: calibrated ensemble is not diagnosed as bias.
_BIAS_SIGMAS: Final[float] = 3.0

#: Variance of a Uniform(0, 1) variate. Structure, not measurement.
_UNIFORM_VARIANCE: Final[float] = 1.0 / 12.0


def gaussian_pit(truth: ArrayLike, *, mean: ArrayLike, std: ArrayLike) -> FloatArray:
    """``Phi((truth - mean) / std)`` — the PIT of a Gaussian marginal posterior.

    Gaussian because that is what :mod:`vpl.inverse.laplace` produces, and because the
    marginals are taken in **unconstrained** space where the Laplace approximation actually
    is Gaussian. Taking them in physical space would require the transformed density, and
    the transform is non-linear — but it is also strictly increasing, so the PIT is
    *identical* either way and the unconstrained form is the one with a closed expression.

    Args:
        truth: The sealed true value for each case, in unconstrained coordinates.
        mean: The posterior mean for each case.
        std: The posterior marginal standard deviation for each case.

    Returns:
        One PIT value per case. Uniform(0, 1) if and only if the posterior is calibrated.
    """
    true_values = np.asarray(truth, dtype=np.float64)
    centres = np.asarray(mean, dtype=np.float64)
    spreads = np.asarray(std, dtype=np.float64)
    if not np.all(spreads > 0.0):
        raise ValueError(
            "every posterior standard deviation must be positive; a zero-width posterior "
            "claims infinite precision, and dividing by it silently yields a PIT of 0 or 1 "
            "that reads as a catastrophic miss rather than as the degenerate input it is."
        )
    return np.asarray(ndtr((true_values - centres) / spreads), dtype=np.float64)


def wilson_interval(
    *, successes: int, trials: int, level: float = COVERAGE_INTERVAL_LEVEL
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because coverage estimates live near the edges of
    ``[0, 1]``, which is exactly where the normal approximation fails: it returns the
    degenerate ``[1, 1]`` at a perfect hit rate, and can place bounds outside ``[0, 1]``.

    Args:
        successes: Cases whose interval contained the truth.
        trials: Cases run.
        level: Confidence level for this interval — see :data:`COVERAGE_INTERVAL_LEVEL`.
    """
    if trials <= 0:
        raise ValueError(f"cannot form an interval from {trials} trials")
    if not 0 <= successes <= trials:
        raise ValueError(f"{successes} successes out of {trials} trials is impossible")
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence level must lie strictly in (0, 1), got {level}")

    z = float(ndtri(0.5 + 0.5 * level))
    n = float(trials)
    observed = successes / n
    denominator = 1.0 + z * z / n
    centre = (observed + z * z / (2.0 * n)) / denominator
    spread = z / denominator * math.sqrt(observed * (1.0 - observed) / n + z * z / (4.0 * n * n))
    return max(0.0, centre - spread), min(1.0, centre + spread)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Measured coverage at several credible levels, each with its own uncertainty.

    Attributes:
        n_cases: Cases in the ensemble. doc 11 §9's "400+" is a claim about this number, and
            a coverage figure without it is unreadable.
        empirical: Measured coverage, keyed by nominal credible level.
        interval: :data:`COVERAGE_INTERVAL_LEVEL` Wilson interval on each estimate. There is
            no way to obtain :attr:`empirical` without this alongside it, by design.
        ks_statistic: KS distance between the PIT sample and Uniform(0, 1).
        ks_pvalue: Its p-value. The decision rule, because it uses the whole distribution
            rather than one level.
        diagnosis: ``"calibrated"``, ``"overconfident"``, ``"underconfident"`` or
            ``"biased"``. "Not calibrated" alone does not tell a reader whether the
            published intervals are too generous or too mean.
    """

    n_cases: int
    empirical: dict[float, float]
    interval: dict[float, tuple[float, float]]
    ks_statistic: float
    ks_pvalue: float
    diagnosis: str

    @property
    def is_calibrated(self) -> bool:
        """Whether the PIT is consistent with Uniform(0, 1) — see :data:`_UNIFORMITY_THRESHOLD`."""
        return self.ks_pvalue >= _UNIFORMITY_THRESHOLD

    def half_width(self, level: float) -> float:
        """Half the width of the interval on the coverage estimate at ``level``.

        The resolution of the test: a discrepancy smaller than this is not measurable with
        the ensemble that was run, whatever the point estimate suggests.
        """
        low, high = self.interval[level]
        return 0.5 * (high - low)

    def __repr__(self) -> str:
        verdict = "calibrated" if self.is_calibrated else f"NOT calibrated ({self.diagnosis})"
        return f"CoverageReport(n={self.n_cases}, {verdict}, KS p={self.ks_pvalue:.3g})"


def coverage_report(pit: ArrayLike, *, levels: tuple[float, ...]) -> CoverageReport:
    """Turn an ensemble of PIT values into doc 11 §9 item 5's result.

    Args:
        pit: One PIT value per case, from :func:`gaussian_pit`.
        levels: Nominal credible levels to report coverage at. Several rather than one,
            because a calibration *curve* says far more than a single hit rate — and
            because a posterior can be right at one level and wrong at another.

    Returns:
        A :class:`CoverageReport`.
    """
    values = np.asarray(pit, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError(
            "cannot report coverage for an empty ensemble; 0 of 0 is not 0 % coverage, it "
            "is no measurement at all"
        )
    if not all(0.0 < level < 1.0 for level in levels):
        raise ValueError(f"every credible level must lie strictly in (0, 1), got {levels}")

    n = int(values.size)
    empirical: dict[float, float] = {}
    interval: dict[float, tuple[float, float]] = {}
    for level in levels:
        # The equal-tailed interval at `level` contains the truth exactly when the PIT
        # falls in the central `level` of the unit interval — which is what makes one PIT
        # sample sufficient for every level at once.
        margin = 0.5 * (1.0 - level)
        hits = int(np.count_nonzero((values > margin) & (values < 1.0 - margin)))
        empirical[level] = hits / n
        interval[level] = wilson_interval(successes=hits, trials=n)

    result = kstest(values, "uniform")
    return CoverageReport(
        n_cases=n,
        empirical=empirical,
        interval=interval,
        ks_statistic=float(result.statistic),
        ks_pvalue=float(result.pvalue),
        diagnosis=_diagnose(values, pvalue=float(result.pvalue)),
    )


def _diagnose(values: FloatArray, *, pvalue: float) -> str:
    """Name the failure mode, not just its existence.

    Bias is tested first: a shifted PIT also has an unusual spread, so checking width first
    would report a displaced posterior as an over- or under-confident one and send the
    reader to fix the wrong thing.
    """
    if pvalue >= _UNIFORMITY_THRESHOLD:
        return "calibrated"

    n = float(values.size)
    bias = float(np.mean(values)) - 0.5
    if abs(bias) > _BIAS_SIGMAS * math.sqrt(_UNIFORM_VARIANCE / n):
        return "biased"

    spread = float(np.mean(np.abs(values - 0.5)))
    return "overconfident" if spread > _UNIFORM_MEAN_ABSOLUTE_DEVIATION else "underconfident"
