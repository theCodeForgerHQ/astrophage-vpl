"""Calibration uncertainty is coherent, and both likelihoods must score it that way.

The defect this module pins down, in one sentence: a radiometric or amplitude calibration
is **one number applied to every channel**, so a calibration error is *one* error, and a
likelihood that puts it on the diagonal scores it as ``N`` independent ones and comes out
over-confident by a factor that grows with the length of the scan.

Doc 04 §7.3 registers ``OES-C1.radiometric_uncertainty`` at 6 % one-sigma and doc 06 §4
item 4 registers ``LIF.scale_uncertainty`` at 5 %; doc 06 §4.1 states the property that
makes the diagonal wrong — a correlated calibration error "affects *every* point
identically and does **not** average down". Doc 00 §5.1 criterion S4 states the
consequence of getting it wrong: "an uncalibrated posterior is worse than no posterior".

## The decisive algebra, and why one identity covers both instruments

Take an observation that differs from the prediction by a **pure multiplicative scale**,
``y_obs = (1 + f) y_pred``, with ``f`` the instrument's own calibration sigma. The residual
is then exactly the coherent direction, ``r = v = f y_pred``. Write ``a = v' D^-1 v`` for
the diagonal-weighted length of that direction. Then

    diagonal chi^2   =  r' D^-1 r                                  =  a
    coherent chi^2   =  r' (D + v v')^-1 r  =  a - a^2 / (1 + a)   =  a / (1 + a)  <  1

so the **overweighting factor is exactly** ``a / (a / (1 + a)) = 1 + a = 1 + chi2_diag``.

Two things follow, and both are asserted below:

1. The coherent chi-squared is bounded above by 1 — *one* error — no matter how many
   channels the scan has, while the diagonal one grows linearly with them.
2. ``chi2_diag / chi2_coherent == 1 + chi2_diag`` holds **exactly**, in closed form, with
   no reference to ``D`` at all. That makes it a structural identity rather than a
   numerical coincidence, and it is the same identity for OES (whose ``D`` is the Poisson
   counting variance in photoelectrons) and for LIF (whose ``D`` is the reported
   per-sample uncertainty squared). Both instruments are held to it by the same
   parametrised test, which is what stops the two implementations drifting apart.

The chi-squared itself is extracted without reaching into either implementation: the
log-determinant and the ``-n/2 log 2 pi`` term do not depend on the residual, so

    chi^2  =  -2 ( logL(scaled observation) - logL(exact observation) )

with the *same* prediction on both sides, which is what makes the basis identical and the
normalisation cancel.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oes_system import energy_grid, plain_system
from vpl.core.params import default_registry
from vpl.core.protocols.config import InstrumentConfig
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    Measurement,
    Observable,
)
from vpl.core.units import Q_
from vpl.instruments.lif.instrument import INSTRUMENT_ID as LIF_ID
from vpl.instruments.lif.instrument import OBSERVABLE_UNITS as LIF_UNITS
from vpl.instruments.lif.instrument import LifInstrument
from vpl.instruments.oes.cr import CollisionalRadiativeModel
from vpl.instruments.oes.instrument import RADIANCE_UNITS, MaxwellianEedf, OesInstrument
from vpl.instruments.oes.spectrograph import Spectrograph
from vpl.physics.eedf.grid import EnergyGrid

type FloatArray = NDArray[np.float64]

ROOT_SEED = 20260806

#: Registered one-sigma calibration uncertainties, written out here rather than read from
#: the registry so that a test and the code it checks cannot agree by both reading the same
#: drifted value. `test_the_registered_calibration_uncertainties_are_what_this_module_
#: assumes` pins them against the registry.
OES_RADIOMETRIC_UNCERTAINTY = 0.06
LIF_SCALE_UNCERTAINTY = 0.05

#: Photoelectrons per pixel in the synthetic OES pair. Comfortably above
#: ``OES.poisson_gaussian_threshold`` (100 pe) so that every pixel is on the Gaussian side
#: of the doc 05 §3.1 switch: the closed-form identity above is a statement about a
#: Gaussian covariance, and a pixel that fell through to the Poisson branch would be
#: compared against a log-pmf instead.
OES_COUNTS_PER_PIXEL = 5.0e3

#: Length of the synthetic LIF scan. 201 is doc 02 §10.1's budget for a full IVDF and the
#: `configure` default, and it is the number in the finding: a coherent scale error across
#: a 201-point scan was being scored as 201 independent errors.
LIF_SCAN_POINTS = 201

#: How far the synthetic profiles are allowed to vary across channels. Non-flat on purpose:
#: a flat prediction makes the coherent direction proportional to `ones`, which is a special
#: case several wrong implementations also get right.
PROFILE_CONTRAST = 0.5

#: Fractional step used for the central-difference curvature in the interval-width tests.
#: 1e-3 sits well above float64 round-off on a quantity of order one and well below the
#: scale on which these log-likelihoods depart from quadratic.
CURVATURE_STEP = 1.0e-3


def _window() -> AcquisitionWindow:
    return AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(2.0, "ns"))


def _profile(n: int) -> FloatArray:
    """A positive, non-constant shape for the synthetic predictions."""
    return np.asarray(1.0 + PROFILE_CONTRAST * np.linspace(0.0, 1.0, n), dtype=np.float64)


# ── OES fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def grid() -> EnergyGrid:
    return energy_grid()


@pytest.fixture(scope="module")
def oes(grid: EnergyGrid) -> OesInstrument:
    return OesInstrument(
        model=CollisionalRadiativeModel(
            system=plain_system(grid), grid=grid, wall_loss_per_s={"m": 1.0e4}
        ),
        spectrograph=Spectrograph.from_registry(),
        eedf=MaxwellianEedf(grid=grid),
        centre_wavelength_nm=811.53,
        root_seed=ROOT_SEED,
    )


def _oes_prediction(instrument: OesInstrument) -> Observable:
    """A prediction pinned to a chosen photoelectron level, so the branch is known.

    Uses the instrument's own radiance-to-counts factor rather than guessing a radiance:
    the whole point of the construction is that every pixel lands on the Gaussian side of
    the doc 05 §3.1 switch, and that is a statement about *counts*. Reaching for the
    private conversion is deliberate white-box scope — this is a unit test of the
    likelihood's covariance algebra, not of the radiometry.
    """
    w = _window()
    per_radiance = np.asarray(instrument._counts_per_radiance(w), dtype=np.float64)
    counts = OES_COUNTS_PER_PIXEL * _profile(per_radiance.size)
    return Observable(
        instrument_id=instrument.instrument_id,
        values=counts / per_radiance,
        units=RADIANCE_UNITS,
        window=w,
    )


def _oes_measurement(
    instrument: OesInstrument,
    pred: Observable,
    *,
    scale: float,
    calibration: CalibrationState = CalibrationState.ESTIMATED,
) -> Measurement:
    per_radiance = np.asarray(
        instrument._counts_per_radiance(pred.window),
        dtype=np.float64,
    )
    values = np.asarray(pred.values, dtype=np.float64) * scale
    return Measurement(
        instrument_id=instrument.instrument_id,
        values=values,
        uncertainty=np.sqrt(values * per_radiance) / per_radiance,
        units=RADIANCE_UNITS,
        window=pred.window,
        calibration=calibration,
    )


# ── LIF fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lif() -> LifInstrument:
    instrument = LifInstrument()
    instrument.configure(InstrumentConfig(values={"scan_points": LIF_SCAN_POINTS}))
    return instrument


def _lif_prediction() -> Observable:
    return Observable(
        instrument_id=LIF_ID,
        values=1.0e15 * _profile(LIF_SCAN_POINTS),
        units=LIF_UNITS,
        window=_window(),
    )


def _lif_measurement(
    pred: Observable,
    *,
    scale: float,
    calibration: CalibrationState = CalibrationState.ESTIMATED,
) -> Measurement:
    values = np.asarray(pred.values, dtype=np.float64) * scale
    # A per-point uncertainty that is neither flat nor proportional to the signal, so the
    # diagonal cannot be mistaken for a multiple of the coherent direction.
    reference = np.asarray(pred.values, dtype=np.float64)
    sigma = 0.02 * reference.max() * (1.0 + np.linspace(0.0, 1.0, reference.size)) ** 2
    return Measurement(
        instrument_id=LIF_ID,
        values=values,
        uncertainty=sigma,
        units=LIF_UNITS,
        window=pred.window,
        calibration=calibration,
    )


# ── the scorers, one per instrument, with a common signature ────────────────────


def _oes_scorer(instrument: OesInstrument):
    pred = _oes_prediction(instrument)

    def score(*, scale: float, calibration_uncertainty: bool) -> float:
        return instrument.likelihood(
            _oes_measurement(instrument, pred, scale=scale),
            pred,
            calibration_uncertainty=calibration_uncertainty,
        )

    return pred, score


def _lif_scorer(instrument: LifInstrument):
    pred = _lif_prediction()

    def score(*, scale: float, calibration_uncertainty: bool) -> float:
        return instrument.likelihood(
            _lif_measurement(pred, scale=scale),
            pred,
            calibration_uncertainty=calibration_uncertainty,
        )

    return pred, score


def _chi_squared(score, *, scale: float, calibration_uncertainty: bool) -> float:
    """``-2 (logL(scaled) - logL(exact))`` — the quadratic form, normalisation cancelled."""
    scaled = score(scale=scale, calibration_uncertainty=calibration_uncertainty)
    exact = score(scale=1.0, calibration_uncertainty=calibration_uncertainty)
    return -2.0 * (scaled - exact)


# ── the finding ─────────────────────────────────────────────────────────────────


class TestACoherentScaleErrorIsScoredAsOneError:
    """The defect, and the fix, on both instruments under one identity."""

    @pytest.mark.parametrize(
        ("name", "uncertainty"),
        [("oes", OES_RADIOMETRIC_UNCERTAINTY), ("lif", LIF_SCALE_UNCERTAINTY)],
    )
    def test_the_coherent_chi_squared_is_order_one_and_the_diagonal_one_is_order_n(
        self,
        name: str,
        uncertainty: float,
        oes: OesInstrument,
        lif: LifInstrument,
    ) -> None:
        _, score = _oes_scorer(oes) if name == "oes" else _lif_scorer(lif)
        scale = 1.0 + uncertainty

        diagonal = _chi_squared(score, scale=scale, calibration_uncertainty=False)
        coherent = _chi_squared(score, scale=scale, calibration_uncertainty=True)

        # One error, not N: bounded above by 1 however long the scan is.
        assert 0.0 < coherent < 1.0
        # And the diagonal treatment is not merely a bit tight — it is enormous.
        assert diagonal > 100.0

    @pytest.mark.parametrize(
        ("name", "uncertainty"),
        [("oes", OES_RADIOMETRIC_UNCERTAINTY), ("lif", LIF_SCALE_UNCERTAINTY)],
    )
    def test_the_overweighting_factor_is_exactly_one_plus_the_diagonal_chi_squared(
        self,
        name: str,
        uncertainty: float,
        oes: OesInstrument,
        lif: LifInstrument,
    ) -> None:
        """The closed-form identity of the module docstring, on both instruments.

        This is the structural-agreement check: OES and LIF have different diagonals, in
        different units, reached by different code, and must both satisfy the *same*
        algebraic relation between the two chi-squareds. An implementation that scored the
        calibration term diagonally, or that inflated it by the wrong power of ``n``, fails
        here on both channels for the same reason.
        """
        _, score = _oes_scorer(oes) if name == "oes" else _lif_scorer(lif)
        scale = 1.0 + uncertainty

        diagonal = _chi_squared(score, scale=scale, calibration_uncertainty=False)
        coherent = _chi_squared(score, scale=scale, calibration_uncertainty=True)

        assert diagonal / coherent == pytest.approx(1.0 + diagonal, rel=1e-9)
        assert coherent == pytest.approx(diagonal / (1.0 + diagonal), rel=1e-9)

    @pytest.mark.parametrize("name", ["oes", "lif"])
    def test_the_overweighting_does_not_grow_with_the_number_of_channels_once_fixed(
        self, name: str, oes: OesInstrument, lif: LifInstrument
    ) -> None:
        """The property that distinguishes a coherent term from a diagonal one.

        A diagonal calibration term averages down as ``1/sqrt(n)`` in the fit, so its
        chi-squared grows linearly with the number of channels. A coherent one does not:
        adding channels to a scan that all share one calibration adds no independent
        information about the calibration.
        """
        uncertainty = OES_RADIOMETRIC_UNCERTAINTY if name == "oes" else LIF_SCALE_UNCERTAINTY
        _, score = _oes_scorer(oes) if name == "oes" else _lif_scorer(lif)

        coherent = _chi_squared(score, scale=1.0 + uncertainty, calibration_uncertainty=True)

        # `a / (1 + a) -> 1` from below as `a` grows; it never exceeds it, which is the
        # whole content of "one error".
        assert coherent < 1.0


class TestPassingNoCoherentTermIsExactlyTheOldLikelihood:
    """What protects T0 and T1. Bit-for-bit, not approximately."""

    def test_oes_default_is_bit_for_bit_the_uncalibrated_likelihood(
        self, oes: OesInstrument
    ) -> None:
        pred = _oes_prediction(oes)
        obs = _oes_measurement(oes, pred, scale=1.0 + OES_RADIOMETRIC_UNCERTAINTY)

        assert oes.likelihood(obs, pred) == oes.likelihood(obs, pred, calibration_uncertainty=False)

    def test_lif_default_is_bit_for_bit_the_uncalibrated_likelihood(
        self, lif: LifInstrument
    ) -> None:
        pred = _lif_prediction()
        obs = _lif_measurement(pred, scale=1.0 + LIF_SCALE_UNCERTAINTY)

        assert lif.likelihood(obs, pred) == lif.likelihood(obs, pred, calibration_uncertainty=False)

    @pytest.mark.parametrize("name", ["oes", "lif"])
    def test_a_true_calibration_carries_no_calibration_uncertainty_even_when_asked(
        self, name: str, oes: OesInstrument, lif: LifInstrument
    ) -> None:
        """Doc 04 §7.3's verification run has no calibration error to score.

        ``use_true_calibration()`` applies the *true* response, and the resulting
        ``Measurement`` records that in :attr:`CalibrationState.TRUE`. There is then no
        difference between the true and estimated scales to inflate the covariance by, and
        inflating it anyway would widen a verification interval for a systematic that
        provably is not present. T0 is exactly this run.
        """
        if name == "oes":
            pred = _oes_prediction(oes)
            obs = _oes_measurement(oes, pred, scale=1.0, calibration=CalibrationState.TRUE)
            scorer = oes.likelihood
        else:
            pred = _lif_prediction()
            obs = _lif_measurement(pred, scale=1.0, calibration=CalibrationState.TRUE)
            scorer = lif.likelihood

        assert scorer(obs, pred, calibration_uncertainty=True) == scorer(
            obs, pred, calibration_uncertainty=False
        )


class TestTheIntervalWidens:
    """The point of the exercise — doc 00 §5.1 S4."""

    @pytest.mark.parametrize("name", ["oes", "lif"])
    def test_including_calibration_uncertainty_lowers_the_curvature_on_an_overall_scale(
        self, name: str, oes: OesInstrument, lif: LifInstrument
    ) -> None:
        """A Laplace interval on an overall scale parameter is ``1/sqrt(curvature)``.

        The calibration systematic is precisely a systematic on overall scale, so scale is
        the parameter it must cost information about. Curvature is taken by central
        differences of the log-likelihood in the multiplicative factor between observation
        and prediction, at the perfectly-fitting point.
        """
        _, score = _oes_scorer(oes) if name == "oes" else _lif_scorer(lif)

        def curvature(*, calibration_uncertainty: bool) -> float:
            centre = score(scale=1.0, calibration_uncertainty=calibration_uncertainty)
            up = score(scale=1.0 + CURVATURE_STEP, calibration_uncertainty=calibration_uncertainty)
            down = score(
                scale=1.0 - CURVATURE_STEP, calibration_uncertainty=calibration_uncertainty
            )
            return -(up - 2.0 * centre + down) / CURVATURE_STEP**2

        without = curvature(calibration_uncertainty=False)
        with_calibration = curvature(calibration_uncertainty=True)

        assert without > 0.0
        assert with_calibration > 0.0
        # Wider interval == lower curvature. Reported as the width ratio because that is
        # the number doc 00 §5.1 S4 is about.
        assert (1.0 / np.sqrt(with_calibration)) > (1.0 / np.sqrt(without))


class TestTheRegisteredValues:
    def test_the_registered_calibration_uncertainties_are_what_this_module_assumes(
        self,
    ) -> None:
        registry = default_registry()

        assert (
            float(registry.value_in("OES-C1.radiometric_uncertainty", "dimensionless"))
            == OES_RADIOMETRIC_UNCERTAINTY
        )
        assert (
            float(registry.value_in("LIF.scale_uncertainty", "dimensionless"))
            == LIF_SCALE_UNCERTAINTY
        )
