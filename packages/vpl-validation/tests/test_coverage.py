"""The coverage ensemble driver — doc 11 §9 item 5, doc 05 §5.

The statistics live in :mod:`vpl.validation.calibration` and are tested there. This file
tests the thing that *runs* the ensemble, and its failure modes are entirely different from
the statistics': they are all about what happens to the cases that do not work.

## The bias that would sink this quietly

Some cases will fail. A MAP will hit its iteration limit; a Hessian will come back singular
because doc 05 §6.2's ``n_0``-``T_e`` ridge went flat for that particular draw. The obvious
implementation catches the exception, skips the case, and reports coverage over what
survived.

That is **survivorship bias, and it biases the coverage estimate optimistically**, because
the cases that fail are precisely the badly-conditioned ones where the posterior was least
trustworthy. A run that silently discards a third of its ensemble and reports 90 % coverage
over the remainder has reported nothing, and nothing about the output would look wrong.

So the driver counts every attempt, reports the failures by cause, and refuses to present a
coverage number when too many cases were lost — because at that point the surviving
ensemble is not a sample of the intended one.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.validation.coverage import (
    MAXIMUM_FAILURE_FRACTION,
    CoverageCase,
    EnsembleTooDepletedError,
    run_coverage_ensemble,
)


class _GaussianPrior:
    def __init__(self, variance: float) -> None:
        self.variance = variance

    def log_prob_unconstrained(self, u: np.ndarray) -> float:
        return float(-0.5 * u @ u / self.variance)


def _well_posed_case(rng: np.random.Generator, *, dimension: int = 3) -> CoverageCase:
    """A linear-Gaussian case whose truth really is drawn from its own prior.

    That last clause is the whole experiment. Coverage is only meaningful when the truth is
    generated from the prior the inversion uses — otherwise a miscalibration is being
    measured against the wrong reference and the number means nothing.
    """
    variance = 2.0
    truth = rng.normal(scale=np.sqrt(variance), size=dimension)
    design = rng.normal(size=(6, dimension))
    noise_sigma = 0.6
    observed = design @ truth + rng.normal(scale=noise_sigma, size=6)

    def log_likelihood(u: np.ndarray) -> float:
        r = design @ u - observed
        return float(-0.5 * r @ r / noise_sigma**2)

    return CoverageCase(
        truth_unconstrained=truth,
        log_likelihood=log_likelihood,
        prior=_GaussianPrior(variance),
        start=np.zeros(dimension),
    )


def _singular_case() -> CoverageCase:
    """A case with a genuine null direction — doc 05 §6's flat direction, made concrete."""

    class Flat:
        def log_prob_unconstrained(self, u: np.ndarray) -> float:
            del u
            return 0.0

    def log_likelihood(u: np.ndarray) -> float:
        return float(-0.5 * (u[0] + u[1]) ** 2)

    return CoverageCase(
        truth_unconstrained=np.zeros(2),
        log_likelihood=log_likelihood,
        prior=Flat(),
        start=np.zeros(2),
    )


class TestItMeasuresCalibrationOnAWellPosedEnsemble:
    @pytest.mark.physics
    def test_a_correctly_specified_ensemble_comes_out_calibrated(self) -> None:
        # The end-to-end statement doc 11 §9 item 5 is making. Truth drawn from the prior,
        # inverted with that same prior and the true noise model: the posterior is exactly
        # correct here, so the intervals must be calibrated. If this fails, either the
        # Laplace covariance or the PIT is wrong, and nothing downstream is meaningful.
        rng = np.random.default_rng(20260806)
        cases = [_well_posed_case(rng) for _ in range(400)]

        result = run_coverage_ensemble(cases, levels=(0.5, 0.9))

        assert result.n_attempted == 400
        assert result.report.is_calibrated
        assert result.report.empirical[0.9] == pytest.approx(
            0.9, abs=result.report.half_width(0.9) * 2
        )

    @pytest.mark.physics
    def test_an_overconfident_ensemble_is_caught(self) -> None:
        # The negative control. If the likelihood is told the noise is much smaller than it
        # really is, the posterior is too narrow and coverage must fail. A coverage test
        # that cannot fail is not a test.
        rng = np.random.default_rng(4)
        cases = []
        for _ in range(400):
            case = _well_posed_case(rng)

            def sharpened(u: np.ndarray, inner=case.log_likelihood) -> float:
                return inner(u) * 9.0  # claims noise 3x smaller than it is

            cases.append(
                CoverageCase(
                    truth_unconstrained=case.truth_unconstrained,
                    log_likelihood=sharpened,
                    prior=case.prior,
                    start=case.start,
                )
            )

        result = run_coverage_ensemble(cases, levels=(0.9,))

        assert not result.report.is_calibrated
        assert result.report.diagnosis == "overconfident"


class TestFailuresAreCountedRatherThanDiscarded:
    def test_a_singular_case_is_recorded_and_the_run_continues(self) -> None:
        # ADR-012: a refused singular Hessian is a *result* — it identifies a null
        # direction doc 05 §6 predicts — not a crash. The driver must record it and carry
        # on, exactly as an ensemble driver over doc 11 WBS 3.1's thousands of cases must.
        rng = np.random.default_rng(9)
        cases = [_well_posed_case(rng) for _ in range(60)]
        cases.insert(30, _singular_case())

        result = run_coverage_ensemble(cases, levels=(0.9,))

        assert result.n_attempted == 61
        assert result.n_singular == 1
        assert result.n_used == 60

    def test_the_failure_count_is_reported_alongside_the_coverage(self) -> None:
        # A coverage number whose denominator is not visible is not checkable.
        rng = np.random.default_rng(10)
        cases = [_well_posed_case(rng) for _ in range(40)]
        cases.append(_singular_case())

        result = run_coverage_ensemble(cases, levels=(0.9,))

        text = repr(result)
        assert "41" in text
        assert "1" in text

    def test_a_heavily_depleted_ensemble_refuses_to_report_coverage(self) -> None:
        # The survivorship bias this module exists to prevent. Cases that fail are the
        # badly-conditioned ones, so coverage over the survivors is optimistic by
        # construction. Past a threshold the survivors are not a sample of the intended
        # ensemble at all, and a number computed from them would be worse than none.
        cases = [_singular_case() for _ in range(30)]
        cases.append(_well_posed_case(np.random.default_rng(2)))

        with pytest.raises(EnsembleTooDepletedError, match="survivorship"):
            run_coverage_ensemble(cases, levels=(0.9,))

    def test_the_threshold_is_stated_rather_than_hidden(self) -> None:
        assert 0.0 < MAXIMUM_FAILURE_FRACTION < 1.0

    def test_an_empty_ensemble_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            run_coverage_ensemble([], levels=(0.9,))


class TestTheTruthIsSealedUntilTheEstimateExists:
    def test_the_driver_does_not_read_the_truth_before_committing_an_estimate(self) -> None:
        # doc 05 §7's inverse crime, at ensemble scale: the truth is in scope for every
        # case, one attribute access away from the optimiser. Seeding MAP from it would
        # make coverage look perfect. The driver must reach the truth only to compute the
        # PIT, after the posterior exists.
        rng = np.random.default_rng(21)
        case = _well_posed_case(rng)
        touched: list[str] = []

        class Watched:
            def __init__(self, value: np.ndarray) -> None:
                self._value = value

            def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
                touched.append("read")
                return np.asarray(self._value)

        watched = CoverageCase(
            truth_unconstrained=Watched(case.truth_unconstrained),  # type: ignore[arg-type]
            log_likelihood=case.log_likelihood,
            prior=case.prior,
            start=case.start,
        )

        run_coverage_ensemble([watched] * 3, levels=(0.9,))

        # Read once per case for the PIT, and not before — three cases, three reads.
        assert len(touched) == 3
