"""Calibration statistics for the coverage test — doc 05 §5, doc 11 §9 item 5.

doc 11 §9 lists the coverage test as item 5, describes it as "uncertainty that has been
*checked* — the strongest differentiator", and says in bold: **Do not cut**. It is the only
item on that list with that instruction. This module is its statistics.

## What is being checked, and the one thing that must not be got wrong

A posterior is *calibrated* if its 90 % credible interval contains the truth in 90 % of
cases. The naive implementation counts hits at one level and prints a fraction — and that
fraction is itself an estimate from a finite ensemble, with its own uncertainty. At the 400
cases doc 11 §9 asks for, the standard error on a coverage estimate near 0.9 is
``sqrt(0.9 * 0.1 / 400) ~ 1.5 %``. So a measured 0.88 is **not** evidence of
overconfidence — it is entirely consistent with a perfectly calibrated posterior — and
reporting it as a failure, or reporting 0.91 as a success, are the same error with opposite
signs.

Quoting a coverage number without an interval on it is therefore the specific defect these
tests exist to prevent. A project whose entire claim is calibrated uncertainty cannot
present an uncalibrated statement about its own calibration.

## Why the PIT, and not just a hit count at one level

Counting hits at 90 % throws away almost everything the ensemble measured. The probability
integral transform keeps it: if ``u_true`` is drawn from the posterior, then
``F_posterior(u_true)`` is Uniform(0, 1) — exactly, for any posterior, in any dimension the
marginal is taken over. So one array of PIT values tests *every* credible level at once, and
a KS test against the uniform gives a single decisive statistic.

It also distinguishes the two failure modes, which a single coverage number cannot:

- **Overconfident** (intervals too narrow — the dangerous one): PIT piles up at 0 and 1.
- **Underconfident** (too wide, merely wasteful): PIT piles up at 0.5.

Both give "coverage != nominal". Only the shape says which, and only one of them is a claim
that would mislead a reader.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.validation.calibration import (
    CoverageReport,
    coverage_report,
    gaussian_pit,
    wilson_interval,
)

_LEVELS = (0.5, 0.68, 0.9, 0.95)


def _calibrated_pit(n: int, seed: int = 20260806) -> np.ndarray:
    """PIT values from a perfectly calibrated posterior — uniform by construction."""
    return np.random.default_rng(seed).uniform(size=n)


class TestTheProbabilityIntegralTransform:
    def test_a_truth_drawn_from_the_posterior_gives_a_uniform_pit(self) -> None:
        # The property the whole method rests on. If this does not hold, every coverage
        # number computed from it is meaningless.
        rng = np.random.default_rng(1)
        mean, std = 3.0, 0.7
        truth = rng.normal(mean, std, size=40_000)

        pit = gaussian_pit(truth, mean=np.full(truth.size, mean), std=np.full(truth.size, std))

        assert pit.min() > 0.0
        assert pit.max() < 1.0
        # Uniform mean 0.5, variance 1/12. Four sigma on 40 000 draws is ~0.0058.
        assert float(np.mean(pit)) == pytest.approx(0.5, abs=0.01)
        assert float(np.var(pit)) == pytest.approx(1.0 / 12.0, abs=0.01)

    def test_a_truth_at_the_posterior_mean_sits_at_one_half(self) -> None:
        pit = gaussian_pit(np.array([2.0]), mean=np.array([2.0]), std=np.array([0.5]))

        assert float(pit[0]) == pytest.approx(0.5)

    def test_a_non_positive_standard_deviation_is_refused(self) -> None:
        # A zero-width posterior is a claim of infinite precision; dividing by it would
        # silently produce a PIT of 0 or 1 and read as a catastrophic miss.
        with pytest.raises(ValueError, match="standard deviation"):
            gaussian_pit(np.array([1.0]), mean=np.array([0.0]), std=np.array([0.0]))


class TestTheCoverageNumberCarriesItsOwnUncertainty:
    def test_the_report_gives_an_interval_on_every_coverage_estimate(self) -> None:
        # The defect this module exists to prevent: a bare fraction, presented as if the
        # ensemble had measured it exactly.
        report = coverage_report(_calibrated_pit(400), levels=_LEVELS)

        for level in _LEVELS:
            low, high = report.interval[level]
            assert low < report.empirical[level] < high

    def test_four_hundred_cases_buy_about_three_percent(self) -> None:
        # doc 11 §9 asks for "400+", and this is what 400 actually resolves. The standard
        # error on a coverage estimate near 0.9 is sqrt(0.9*0.1/400) ~ 1.5 %, so the 95 %
        # interval reported alongside it is ~2 x that, ~3 % wide either side. The ensemble
        # can therefore separate 0.90 from 0.85 and cannot separate it from 0.89. Pinning
        # the number keeps the report from implying a precision 400 cases do not buy.
        report = coverage_report(_calibrated_pit(400), levels=(0.9,))

        low, high = report.interval[0.9]
        assert 0.5 * (high - low) == pytest.approx(0.029, abs=0.010)

    def test_more_cases_narrow_the_interval(self) -> None:
        few = coverage_report(_calibrated_pit(400, seed=2), levels=(0.9,))
        many = coverage_report(_calibrated_pit(6400, seed=3), levels=(0.9,))

        assert many.half_width(0.9) < few.half_width(0.9)

    def test_the_wilson_interval_beats_the_normal_approximation_at_the_edge(self) -> None:
        # At 20/20 hits the normal approximation gives the degenerate [1, 1] — "coverage is
        # exactly 100 %, with certainty", from twenty cases. Wilson does not.
        low, high = wilson_interval(successes=20, trials=20, level=0.95)

        assert low < 1.0
        assert high == pytest.approx(1.0, abs=1e-9)
        assert low > 0.5

    def test_an_empty_ensemble_is_refused_rather_than_reported_as_zero(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            coverage_report(np.empty(0), levels=_LEVELS)


class TestItDetectsMiscalibration:
    def test_a_calibrated_posterior_passes(self) -> None:
        report = coverage_report(_calibrated_pit(2000), levels=_LEVELS)

        assert report.is_calibrated
        for level in _LEVELS:
            assert report.empirical[level] == pytest.approx(level, abs=0.04)

    def test_an_overconfident_posterior_is_caught(self) -> None:
        # THE failure mode that matters: intervals too narrow, so the truth falls outside
        # more often than advertised. PIT piles up at both ends.
        rng = np.random.default_rng(11)
        # Truth drawn with the real spread, posterior claiming half of it.
        truth = rng.normal(0.0, 1.0, size=2000)
        pit = gaussian_pit(truth, mean=np.zeros(2000), std=np.full(2000, 0.5))

        report = coverage_report(pit, levels=_LEVELS)

        assert not report.is_calibrated
        assert report.empirical[0.9] < 0.9 - report.half_width(0.9)
        assert report.diagnosis == "overconfident"

    def test_an_underconfident_posterior_is_caught_and_named_differently(self) -> None:
        # Wasteful rather than misleading, and the report must not conflate the two: only
        # one of them invalidates a published interval.
        rng = np.random.default_rng(12)
        truth = rng.normal(0.0, 1.0, size=2000)
        pit = gaussian_pit(truth, mean=np.zeros(2000), std=np.full(2000, 2.0))

        report = coverage_report(pit, levels=_LEVELS)

        assert not report.is_calibrated
        assert report.empirical[0.9] > 0.9
        assert report.diagnosis == "underconfident"

    def test_a_biased_posterior_is_caught(self) -> None:
        # A posterior of the right width in the wrong place. Coverage at a single level can
        # look almost right while the PIT is visibly skewed, which is why the KS test on
        # the whole distribution is the decision rule rather than one hit count.
        rng = np.random.default_rng(13)
        truth = rng.normal(1.5, 1.0, size=2000)
        pit = gaussian_pit(truth, mean=np.zeros(2000), std=np.ones(2000))

        report = coverage_report(pit, levels=_LEVELS)

        assert not report.is_calibrated


class TestTheReport:
    def test_it_records_how_many_cases_it_saw(self) -> None:
        # A coverage figure without its sample size is unreadable, and doc 11 §9's "400+"
        # is a claim about this number.
        report = coverage_report(_calibrated_pit(400), levels=_LEVELS)

        assert isinstance(report, CoverageReport)
        assert report.n_cases == 400

    def test_the_repr_states_the_sample_size_and_the_verdict(self) -> None:
        report = coverage_report(_calibrated_pit(400), levels=_LEVELS)

        text = repr(report)
        assert "400" in text
        assert "calibrated" in text.lower()

    def test_a_level_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="level"):
            coverage_report(_calibrated_pit(100), levels=(0.9, 1.0))

    def test_it_is_immutable(self) -> None:
        report = coverage_report(_calibrated_pit(100), levels=_LEVELS)

        with pytest.raises(AttributeError):
            report.n_cases = 1  # type: ignore[misc]
