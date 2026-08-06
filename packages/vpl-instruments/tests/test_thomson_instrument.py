"""``ThomsonInstrument`` — doc 02 §7, doc 04 §4, doc 08 §4.

Plain module-level helpers rather than a shared ``conftest.py`` — see the note at the top
of ``oes_system.py`` for why.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.protocols import (
    CalibrationReference,
    CalibrationSet,
    Instrument,
    InstrumentConfig,
)
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    Fidelity,
    PhaseGrid,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    Species,
)
from vpl.core.units import Q_
from vpl.instruments.thomson import photons
from vpl.instruments.thomson.instrument import (
    CHANNEL_WIDTH_NM,
    N_SPECTRAL_CHANNELS,
    PHOTON_COUNT_UNITS,
    THOMSON_INSTRUMENT_ID,
    ThomsonBlindWindowError,
    ThomsonInstrument,
)
from vpl.instruments.thomson.spectrum import CoherentScatteringRegimeError

_GRID = SpatialGrid(z_m=np.linspace(0.0, 5.0e-2, 5))
_RP1_N_E = 1.0e17
_RP1_T_E = 3.0
_RP1_VP = 12.0

#: The doc 02 §7.1 "~700 s" accumulation for 3 % at RP-1 (doc 06 §4 term 8).
_RP1_WINDOW = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(696.7, "s"))

#: A window too short to see anything at all — a single laser shot.
_SINGLE_SHOT_WINDOW = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(0.1, "s"))


def _argon() -> Species:
    return Species(name="Ar+", mass=default_registry().quantity("species.Ar.mass"), charge_number=1)


def _params(*, n_e: float = _RP1_N_E, t_e: float = _RP1_T_E) -> PlasmaParams:
    return PlasmaParams(
        species=_argon(),
        n_0=Q_(n_e, "m**-3"),
        T_e=Q_(t_e, "eV"),
        T_i=Q_(0.05, "eV"),
        T_g=Q_(300.0, "K"),
        pressure=Q_(5.0, "mTorr"),
        bias=Q_(-60.0, "V"),
        gamma_se=0.1,
        kappa=1.0,
    )


def _state(*, n_e: float = _RP1_N_E, t_e: float = _RP1_T_E) -> PlasmaState:
    fields = {
        "n_e": ScalarField(
            name="n_e", values=np.full(_GRID.n_points, n_e), units="m**-3", grid=_GRID, time=None
        ),
        "n_i": ScalarField(
            name="n_i", values=np.full(_GRID.n_points, n_e), units="m**-3", grid=_GRID, time=None
        ),
        "Phi": ScalarField(
            name="Phi", values=np.full(_GRID.n_points, _RP1_VP), units="V", grid=_GRID, time=None
        ),
        "T_e": ScalarField(
            name="T_e", values=np.full(_GRID.n_points, t_e), units="eV", grid=_GRID, time=None
        ),
    }
    return PlasmaState(
        params=_params(n_e=n_e, t_e=t_e),
        grid=_GRID,
        time=None,
        fields=fields,
        ion_distribution=None,
        fidelity=Fidelity.L0,
    )


def _configured(**overrides: object) -> ThomsonInstrument:
    instrument = ThomsonInstrument(root_seed=7)
    instrument.configure(InstrumentConfig(values=dict(overrides)))
    return instrument


def _references() -> CalibrationSet:
    return CalibrationSet.of(
        CalibrationReference(
            name="Rayleigh scattering in Ar, doc 02 §7.3",
            quantity=photons.RAYLEIGH_CALIBRATION_QUANTITY,
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=photons.RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY,
            traceable_to="doc 02 §11-style Rayleigh cross-section standard",
        )
    )


class TestTheContract:
    def test_it_is_recognised_as_an_instrument(self) -> None:
        assert isinstance(ThomsonInstrument(root_seed=1), Instrument)

    def test_forward_before_configure_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="configure"):
            ThomsonInstrument(root_seed=1).forward(_state(), _RP1_WINDOW)

    def test_observe_before_calibrate_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="calibrate"):
            _configured().observe(_state(), _RP1_WINDOW)

    def test_the_metadata_carries_citations_and_a_detection_floor(self) -> None:
        metadata = _configured().metadata()
        assert metadata.instrument_id == THOMSON_INSTRUMENT_ID
        assert len(metadata.citations) >= 3
        assert metadata.detection_floor.quantity == "n_0"
        assert metadata.detection_floor.requirement == "IF-6"
        assert metadata.detection_floor.threshold.magnitude > 0.0


class TestForward:
    def test_it_returns_one_sample_per_ts_s3_channel(self) -> None:
        observable = _configured().forward(_state(), _RP1_WINDOW)
        assert observable.n_samples == N_SPECTRAL_CHANNELS
        assert observable.instrument_id == THOMSON_INSTRUMENT_ID
        assert observable.units == PHOTON_COUNT_UNITS

    def test_it_is_deterministic(self) -> None:
        state = _state()
        first = _configured().forward(state, _RP1_WINDOW)
        second = _configured().forward(state, _RP1_WINDOW)
        np.testing.assert_array_equal(first.values, second.values)

    def test_it_is_non_negative(self) -> None:
        observable = _configured().forward(_state(), _RP1_WINDOW)
        assert np.all(observable.values >= 0.0)

    def test_the_integrated_signal_matches_the_doc_02_photon_budget(self) -> None:
        """The channel-summed forward() total is the doc 02 §7.1 photon budget: the
        signal-only, background-subtracted truth, exactly the total this instrument's
        Rayleigh calibration is meant to make an absolute density measurement out of."""
        observable = _configured().forward(_state(), _RP1_WINDOW)
        expected_total = photons.total_photoelectrons(
            electron_density_m3=_RP1_N_E, duration_s=696.7
        )
        assert float(np.sum(observable.values)) == pytest.approx(expected_total, rel=0.02)

    def test_a_700_second_window_at_rp1_gives_about_3_percent_on_the_integrated_signal(
        self,
    ) -> None:
        observable = _configured().forward(_state(), _RP1_WINDOW)
        total = float(np.sum(observable.values))
        assert 1.0 / np.sqrt(total) == pytest.approx(0.03, rel=0.05)

    def test_the_spectral_weights_sum_to_close_to_one(self) -> None:
        """TS-S2's 25 nm window comfortably covers the doc 04 §4.2 line shape at RP-1, so
        (signal channel sum) / (total signal) should be close to 1, not concentrated at
        the edges of the window."""
        instrument = _configured()
        weights = instrument._spectral_weights(electron_temperature_ev=_RP1_T_E)
        assert float(np.sum(weights)) == pytest.approx(1.0, rel=0.02)

    def test_a_single_shot_window_is_refused_as_blind(self) -> None:
        """Doc 02 §7.1 consequence 2: "any single-shot or few-shot event is invisible"."""
        with pytest.raises(ThomsonBlindWindowError, match="required_accumulation_s"):
            _configured().forward(_state(), _SINGLE_SHOT_WINDOW)

    def test_a_low_density_low_temperature_state_is_refused_as_coherent(self) -> None:
        """TS-2 / doc 04 §4.1: the single-particle spectrum does not describe a
        collective (coherent) scattering regime."""
        with pytest.raises(CoherentScatteringRegimeError, match="TS-2"):
            _configured().forward(_state(n_e=1.0e19, t_e=0.01), _RP1_WINDOW)


class TestPhaseLockedWindows:
    def _grid(self) -> PhaseGrid:
        return PhaseGrid(n_bins=16, period=Q_(1.0 / 13.56e6, "s"))

    def test_a_phase_locked_window_with_no_accumulation_is_refused(self) -> None:
        w = AcquisitionWindow.phase_locked(grid=self._grid(), bin_index=0)
        with pytest.raises(ValueError, match="accumulation"):
            _configured().forward(_state(), w)

    def test_a_phase_locked_window_needs_about_16x_the_shots_of_an_unbinned_one(
        self,
    ) -> None:
        """Doc 02 §7.1 consequence 2: 16 phase bins multiply the required shots by 16."""
        instrument = _configured()
        unbinned = float(np.sum(instrument.forward(_state(), _RP1_WINDOW).values))

        phase_window = AcquisitionWindow.phase_locked(
            grid=self._grid(), bin_index=3, accumulation=Q_(696.7 * 16.0, "s")
        )
        phase_locked = float(np.sum(instrument.forward(_state(), phase_window).values))

        assert phase_locked == pytest.approx(unbinned, rel=0.02)


class TestRequiredAccumulation:
    def test_it_matches_the_module_level_photon_budget_function(self) -> None:
        instrument = _configured()
        params = _params()
        instrument_value = instrument.required_accumulation_s(
            params, target_relative_uncertainty=0.03
        )
        module_value = photons.required_accumulation_s(
            electron_density_m3=params.n_0_per_m3, target_relative_uncertainty=0.03
        )
        assert instrument_value == pytest.approx(module_value)

    def test_16_phase_bins_takes_about_3_1_hours_at_rp1(self) -> None:
        instrument = _configured()
        hours = (
            instrument.required_accumulation_s(
                _params(), target_relative_uncertainty=0.03, n_phase_bins=16.0
            )
            / 3600.0
        )
        assert hours == pytest.approx(3.1, rel=0.02)


class TestObserveSharesTheForwardPath:
    def test_observe_equals_forward_with_noise_off_and_true_calibration(self) -> None:
        """Doc 04 §9: with noise off and the true calibration applied, ``observe`` and
        ``forward`` must come from the same code path and agree *exactly*."""
        instrument = _configured(noise=False)
        instrument.calibrate(_references())
        instrument.use_true_calibration()

        state = _state()
        predicted = instrument.forward(state, _RP1_WINDOW)
        observed = instrument.observe(state, _RP1_WINDOW)

        assert np.array_equal(observed.values, predicted.values)
        assert observed.as_observable() == predicted

    def test_observe_is_noisy_and_carries_a_positive_uncertainty(self) -> None:
        instrument = _configured()
        instrument.calibrate(_references())
        measurement = instrument.observe(_state(), _RP1_WINDOW)

        assert measurement.calibration is CalibrationState.ESTIMATED
        assert np.all(measurement.uncertainty > 0.0)
        assert not np.array_equal(
            measurement.values, instrument.forward(_state(), _RP1_WINDOW).values
        )

    def test_observe_is_reproducible_from_the_seed(self) -> None:
        """Doc 00 E3: two instruments with the same root seed draw the same noise."""
        state = _state()
        first, second = _configured(), _configured()
        for instrument in (first, second):
            instrument.calibrate(_references())
        np.testing.assert_array_equal(
            first.observe(state, _RP1_WINDOW).values, second.observe(state, _RP1_WINDOW).values
        )


class TestCalibrationDrawnOnceAndKept:
    def test_estimated_calibration_carries_the_doc_06_uncertainty_and_is_not_trivial(
        self,
    ) -> None:
        calibration = _configured().calibrate(_references())
        assert calibration.state is CalibrationState.ESTIMATED
        assert calibration.is_inverse_crime is False
        assert calibration.relative_uncertainty["count_scale"] == pytest.approx(
            photons.RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY
        )
        assert calibration.coefficients["count_scale"] != 1.0

    def test_calibration_needs_a_rayleigh_standard(self) -> None:
        empty = CalibrationSet(references=())
        with pytest.raises(KeyError, match=photons.RAYLEIGH_CALIBRATION_QUANTITY):
            _configured().calibrate(empty)

    def test_the_drawn_scale_is_unchanged_across_repeated_observations(self) -> None:
        """Doc 06 §4.1: "a single Rayleigh calibration error affects *every* Thomson
        point identically and does **not** average down" — because it is drawn once in
        ``calibrate()`` and never re-drawn, unlike the per-shot Poisson noise."""
        instrument = _configured()
        calibration = instrument.calibrate(_references())
        fixed_scale = calibration.coefficients["count_scale"]

        instrument.observe(_state(), _RP1_WINDOW)
        instrument.observe(_state(), _RP1_WINDOW)

        assert instrument._calibration is not None
        assert instrument._calibration.coefficients["count_scale"] == fixed_scale

    def test_two_calibrate_calls_on_fresh_instruments_draw_different_scales(self) -> None:
        """The draw is genuinely random (not a fixed constant mistaken for one) — two
        differently-seeded instruments must not coincide."""
        first = ThomsonInstrument(root_seed=1).calibrate(_references())
        second = ThomsonInstrument(root_seed=2).calibrate(_references())
        assert first.coefficients["count_scale"] != second.coefficients["count_scale"]


class TestStrayLightInflatesTheEstimatedUncertainty:
    def test_the_estimated_calibration_channel_uncertainty_near_the_laser_line_exceeds_the_true_one(
        self,
    ) -> None:
        """Doc 02 §4.3: the residual stray-light background "appears in the error budget
        as a background-subtraction systematic" once an imperfect (estimated) background
        estimate is subtracted, rather than the exact true one."""
        instrument = _configured(noise=False)
        instrument.calibrate(_references())
        state = _state()

        instrument.use_true_calibration()
        true_uncertainty = instrument.observe(state, _RP1_WINDOW).uncertainty

        instrument._applied_state = CalibrationState.ESTIMATED
        estimated_uncertainty = instrument.observe(state, _RP1_WINDOW).uncertainty

        axis = instrument.wavelength_axis_nm()
        laser_channel = int(np.argmin(np.abs(axis - 532.0)))
        assert estimated_uncertainty[laser_channel] > true_uncertainty[laser_channel]


class TestLikelihood:
    def test_it_peaks_at_the_truth(self) -> None:
        instrument = _configured()
        instrument.calibrate(_references())
        state = _state()
        measurement = instrument.observe(state, _RP1_WINDOW)

        at_truth = instrument.likelihood(measurement, instrument.forward(state, _RP1_WINDOW))
        at_wrong = instrument.likelihood(
            measurement, instrument.forward(_state(t_e=8.0), _RP1_WINDOW)
        )
        assert at_truth > at_wrong

    def test_it_is_finite(self) -> None:
        instrument = _configured()
        instrument.calibrate(_references())
        state = _state()
        value = instrument.likelihood(
            instrument.observe(state, _RP1_WINDOW), instrument.forward(state, _RP1_WINDOW)
        )
        assert np.isfinite(value)

    def test_it_refuses_a_mismatched_prediction(self) -> None:
        from vpl.core.state import Observable

        instrument = _configured()
        instrument.calibrate(_references())
        state = _state()
        measurement = instrument.observe(state, _RP1_WINDOW)
        truncated = Observable(
            instrument_id=THOMSON_INSTRUMENT_ID,
            values=np.ones(3),
            units=PHOTON_COUNT_UNITS,
            window=_RP1_WINDOW,
        )
        with pytest.raises(ValueError, match="same shape"):
            instrument.likelihood(measurement, truncated)


class TestIsInformative:
    def test_it_admits_rp1(self) -> None:
        assert _configured().is_informative(_params(n_e=_RP1_N_E))

    def test_it_refuses_a_density_far_below_the_floor(self) -> None:
        assert not _configured().is_informative(_params(n_e=1.0e12))


class TestWavelengthAxis:
    def test_it_spans_ts_s2_with_ts_s3s_channel_count(self) -> None:
        axis = _configured().wavelength_axis_nm()
        assert axis.shape == (N_SPECTRAL_CHANNELS,)
        assert axis[0] > 520.0
        assert axis[-1] < 545.0
        assert axis[1] - axis[0] == pytest.approx(CHANNEL_WIDTH_NM)
