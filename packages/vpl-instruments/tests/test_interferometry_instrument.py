"""``InterferometryInstrument`` — the doc 08 §4 contract over the interferometry channel.

Plain module-level helpers, not a shared ``conftest.py`` — see the note at the top of
``oes_system.py`` for why this package avoids one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.protocols import (
    CalibrationReference,
    CalibrationSet,
    Instrument,
    InstrumentConfig,
)
from vpl.core.random import Stream, generator
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    Fidelity,
    Measurement,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    Species,
    TimeGrid,
)
from vpl.core.units import Q_
from vpl.instruments.interferometry import noise as interferometry_noise
from vpl.instruments.interferometry.instrument import (
    PHASE_UNITS,
    InterferometryInstrument,
)
from vpl.instruments.interferometry.phase import N_CHORDS

_GRID = SpatialGrid(z_m=np.linspace(0.0, 5.0e-2, 26))
_WINDOW = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(1.0e-3, "s"))

_TRUE_N_E = 1.0e17
_TRUE_T_E = 3.0


def _argon() -> Species:
    return Species(name="Ar+", mass=default_registry().quantity("species.Ar.mass"), charge_number=1)


def _params(*, n_0: float = _TRUE_N_E, pressure_mtorr: float = 5.0) -> PlasmaParams:
    return PlasmaParams(
        species=_argon(),
        n_0=Q_(n_0, "m**-3"),
        T_e=Q_(_TRUE_T_E, "eV"),
        T_i=Q_(0.05, "eV"),
        T_g=Q_(300.0, "K"),
        pressure=Q_(pressure_mtorr, "mTorr"),
        bias=Q_(-250.0, "V"),
        gamma_se=0.1,
        kappa=1.0,
    )


def _profile(grid: SpatialGrid) -> np.ndarray:
    """A non-uniform n_e(z), higher in the bulk than near the wall, so distinct chords
    at distinct z genuinely see distinct densities."""
    return _TRUE_N_E * (1.0 + grid.z_m / grid.z_m[-1])


def _state(*, n_0: float = _TRUE_N_E, pressure_mtorr: float = 5.0) -> PlasmaState:
    n_e_values = _profile(_GRID)
    fields = {
        "n_e": ScalarField(name="n_e", values=n_e_values, units="m**-3", grid=_GRID, time=None),
        "n_i": ScalarField(name="n_i", values=n_e_values, units="m**-3", grid=_GRID, time=None),
        "Phi": ScalarField(
            name="Phi", values=np.full(_GRID.n_points, 12.0), units="V", grid=_GRID, time=None
        ),
        "T_e": ScalarField(
            name="T_e", values=np.full(_GRID.n_points, _TRUE_T_E), units="eV", grid=_GRID, time=None
        ),
    }
    return PlasmaState(
        params=_params(n_0=n_0, pressure_mtorr=pressure_mtorr),
        grid=_GRID,
        time=None,
        fields=fields,
        ion_distribution=None,
        fidelity=Fidelity.L0,
    )


def _configured(**overrides: object) -> InterferometryInstrument:
    instrument = InterferometryInstrument(root_seed=11)
    instrument.configure(InstrumentConfig(values=dict(overrides)))
    return instrument


def _references() -> CalibrationSet:
    return CalibrationSet.of(
        CalibrationReference(
            name="AOM frequency reference / path-length metrology standard",
            quantity="interferometer_phase_scale",
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=0.02,
            traceable_to="doc 02 §11-style frequency and path-length calibration chain",
        )
    )


class TestTheContract:
    def test_it_is_recognised_as_an_instrument(self) -> None:
        assert isinstance(InterferometryInstrument(root_seed=1), Instrument)

    def test_forward_before_configure_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="configure"):
            InterferometryInstrument(root_seed=1).forward(_state(), _WINDOW)

    def test_the_metadata_carries_a_citation_and_a_detection_floor(self) -> None:
        metadata = _configured().metadata()

        assert metadata.citations
        assert metadata.detection_floor.quantity == "n_0"
        assert metadata.detection_floor.requirement == "IF-6"


class TestChordMapping:
    def test_forward_returns_one_phase_per_chord(self) -> None:
        observable = _configured().forward(_state(), _WINDOW)

        assert observable.n_samples == N_CHORDS
        assert observable.units == PHASE_UNITS

    def test_distinct_chords_at_distinct_z_see_distinct_phases(self) -> None:
        # The test profile increases with z, so a chord anchored near the wall must read
        # a smaller phase than one further into the bulk.
        observable = _configured().forward(_state(), _WINDOW)

        assert observable.values[0] < observable.values[-1]

    def test_matches_an_independently_interpolated_profile(self) -> None:
        from vpl.instruments.interferometry.phase import net_phase_shift_rad

        instrument = _configured()
        state = _state()
        observable = instrument.forward(state, _WINDOW)

        chord_z = instrument.chord_positions_m()
        expected_n_e = np.interp(chord_z, _GRID.z_m, _profile(_GRID))
        expected = net_phase_shift_rad(
            n_e_per_m3=expected_n_e,
            n_neutral_per_m3=state.params.n_g_per_m3,
            wavelength_m=instrument.wavelength_m,
            chord_length_m=instrument.chord_length_m,
        )

        np.testing.assert_allclose(observable.values, expected, rtol=1.0e-9)


class TestForward:
    def test_it_is_deterministic(self) -> None:
        state = _state()
        first = _configured().forward(state, _WINDOW)
        second = _configured().forward(state, _WINDOW)

        np.testing.assert_array_equal(first.values, second.values)

    def test_chord_length_is_configurable_and_changes_the_phase(self) -> None:
        default_chord = _configured().forward(_state(), _WINDOW)
        short_chord = _configured(chord_length_m=0.1).forward(_state(), _WINDOW)

        assert np.all(short_chord.values < default_chord.values)


class TestObserveSharesTheForwardPath:
    def test_observe_reduces_to_forward_with_noise_off(self) -> None:
        state = _state()
        instrument = _configured(noise=False)

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        np.testing.assert_array_equal(observed.values, predicted.values)
        assert observed.as_observable() == predicted

    def test_observe_adds_noise_by_default(self) -> None:
        state = _state()
        instrument = _configured()
        instrument.calibrate(_references())

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        assert not np.array_equal(observed.values, predicted.values)
        assert np.all(observed.uncertainty > 0.0)

    def test_the_same_seed_gives_the_same_observation(self) -> None:
        state = _state()
        first = InterferometryInstrument(root_seed=9)
        first.configure(InstrumentConfig(values={}))
        first.calibrate(_references())
        second = InterferometryInstrument(root_seed=9)
        second.configure(InstrumentConfig(values={}))
        second.calibrate(_references())

        np.testing.assert_array_equal(
            first.observe(state, _WINDOW).values, second.observe(state, _WINDOW).values
        )

    def test_observe_before_calibrate_with_noise_enabled_is_refused(self) -> None:
        # doc 04 §7.3: noise on simulates a real measurement, and there is nothing to
        # apply until a calibration measurement has produced an estimated response.
        with pytest.raises(RuntimeError, match="calibrate"):
            _configured().observe(_state(), _WINDOW)


class TestCalibrate:
    def test_the_estimated_calibration_carries_uncertainty(self) -> None:
        calibration = _configured().calibrate(_references())

        assert calibration.state is CalibrationState.ESTIMATED
        assert any(value > 0.0 for value in calibration.relative_uncertainty.values())

    def test_a_calibration_set_missing_the_standard_is_refused(self) -> None:
        with pytest.raises(KeyError):
            _configured().calibrate(CalibrationSet.of())

    def test_use_true_calibration_is_recorded_on_the_measurement(self) -> None:
        instrument = _configured()
        instrument.use_true_calibration()

        observed = instrument.observe(_state(), _WINDOW)

        assert observed.calibration is CalibrationState.TRUE
        assert observed.is_inverse_crime

    def test_use_true_calibration_still_lets_noise_off_match_forward_exactly(self) -> None:
        # doc 04 §9's exact-equality requirement, with the inverse crime committed on
        # purpose (doc 04 §7.3) rather than by accident.
        state = _state()
        instrument = _configured(noise=False)
        instrument.use_true_calibration()

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        np.testing.assert_array_equal(observed.values, predicted.values)
        assert observed.as_observable() == predicted


class TestLikelihood:
    def test_a_perfect_prediction_is_the_most_probable_one(self) -> None:
        instrument = _configured()
        instrument.calibrate(_references())
        observed = instrument.observe(_state(), _WINDOW)
        predicted = observed.as_observable()

        exact = instrument.likelihood(observed, predicted)
        shifted = type(predicted)(
            instrument_id=predicted.instrument_id,
            values=predicted.values + 1.0e-6,
            units=predicted.units,
            window=predicted.window,
        )
        offset = instrument.likelihood(observed, shifted)

        assert exact > offset

    def test_a_prediction_of_the_wrong_shape_is_refused(self) -> None:
        instrument = _configured()
        instrument.calibrate(_references())
        observed = instrument.observe(_state(), _WINDOW)
        predicted = observed.as_observable()

        with pytest.raises(ValueError, match="shape"):
            instrument.likelihood(
                observed,
                type(predicted)(
                    instrument_id=predicted.instrument_id,
                    values=predicted.values[:-1],
                    units=predicted.units,
                    window=predicted.window,
                ),
            )

    def test_a_common_mode_residual_is_penalised_less_than_a_differential_one(self) -> None:
        """The load-bearing test for doc 05 Section 3.1's "correlated noise covariance".

        Doc 02 IF-G2 calls mechanical drift "the dominant systematic", and it is a
        *common-path* effect: all 8 chords share the same optical table and, largely, the
        same vibration. A diagonal covariance would let 8 independent chords average that
        shared error down as 1/sqrt(8), which they cannot physically do, and would claim
        far more information than the channel has. The correlated (rank-one-plus-diagonal)
        covariance built here must therefore be *more* forgiving of a residual that moves
        every chord the same way than of one of equal magnitude that moves chords in
        opposite directions - the second pattern cannot be explained by common-mode
        vibration at all, and the model correctly refuses to explain it away.
        """
        instrument = _configured()
        predicted = instrument.forward(_state(), _WINDOW)
        epsilon = 5.0e-5

        common_mode = predicted.values + epsilon
        alternating = predicted.values + epsilon * np.resize([1.0, -1.0], predicted.n_samples)

        common_obs = Measurement(
            instrument_id=predicted.instrument_id,
            values=common_mode,
            uncertainty=np.ones(predicted.n_samples),
            units=predicted.units,
            window=predicted.window,
        )
        alternating_obs = Measurement(
            instrument_id=predicted.instrument_id,
            values=alternating,
            uncertainty=np.ones(predicted.n_samples),
            units=predicted.units,
            window=predicted.window,
        )

        common_ll = instrument.likelihood(common_obs, predicted)
        alternating_ll = instrument.likelihood(alternating_obs, predicted)

        assert common_ll > alternating_ll


class TestTheDetectionGate:
    def test_a_dense_plasma_is_informative(self) -> None:
        assert _configured().is_informative(_params(n_0=_TRUE_N_E))

    def test_a_plasma_below_the_floor_is_not_informative(self) -> None:
        assert not _configured().is_informative(_params(n_0=1.0e14))

    def test_a_plasma_exactly_at_the_floor_is_informative(self) -> None:
        # DetectionFloor.admits is >=; doc 01 IF-6 declares the blind region strictly
        # below the floor, so the boundary itself is informative.
        instrument = _configured()
        floor = instrument.metadata().detection_floor.threshold

        params = _params(n_0=float(floor.magnitude))
        assert instrument.is_informative(params)

    def test_the_floor_scales_with_a_configured_chord_length(self) -> None:
        default_floor = _configured().metadata().detection_floor.threshold
        short_chord_floor = _configured(chord_length_m=0.1).metadata().detection_floor.threshold

        # A quarter of the chord length raises the floor by 4x (doc 02 Section 8.2's
        # "the 400 mm chamber diameter improves the interferometry detection floor by 4x").
        assert float(short_chord_floor.magnitude) == pytest.approx(
            4.0 * float(default_floor.magnitude), rel=1.0e-9
        )


class TestTimeResolvedState:
    def test_the_window_average_does_not_crash_and_stays_finite(self) -> None:
        time_grid = TimeGrid(t_s=np.array([0.0, 5.0e-4, 1.0e-3]))
        values = np.stack([_profile(_GRID), _profile(_GRID) * 1.1, _profile(_GRID) * 1.2])
        fields = {
            "n_e": ScalarField(
                name="n_e", values=values, units="m**-3", grid=_GRID, time=time_grid
            ),
            "n_i": ScalarField(
                name="n_i", values=values, units="m**-3", grid=_GRID, time=time_grid
            ),
            "Phi": ScalarField(
                name="Phi",
                values=np.full((3, _GRID.n_points), 12.0),
                units="V",
                grid=_GRID,
                time=time_grid,
            ),
            "T_e": ScalarField(
                name="T_e",
                values=np.full((3, _GRID.n_points), _TRUE_T_E),
                units="eV",
                grid=_GRID,
                time=time_grid,
            ),
        }
        state = PlasmaState(
            params=_params(),
            grid=_GRID,
            time=time_grid,
            fields=fields,
            ion_distribution=None,
            fidelity=Fidelity.L0,
        )

        observable = _configured().forward(state, _WINDOW)

        assert np.all(np.isfinite(observable.values))
        assert observable.n_samples == N_CHORDS


class TestFringeJumpRateIsConfigurable:
    def test_a_certain_rate_produces_a_full_fringe_jump_somewhere_in_the_observation(
        self,
    ) -> None:
        state = _state()
        instrument = _configured(fringe_jump_rate=1.0)
        instrument.calibrate(_references())

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        # With the rate at certainty every chord gets a jump; even after the (much
        # smaller) vibration and detector noise is added on top, the residual must stay
        # close to a whole 2*pi.
        residual = observed.values - predicted.values
        assert np.all(np.abs(np.abs(residual) - 2.0 * math.pi) < 1.0)


class TestCalibrationScaleDoesNotAverageDown:
    """doc 06 §4.1: a correlated calibration error affects every observation identically
    and must not be re-drawn per observation, or repeated observations would average it
    away and understate the very error budget this channel exists to report honestly.

    This reconstructs, from the public noise model and the calibration coefficient
    ``calibrate()`` returned, exactly what a *correct* (draw-once) implementation must
    produce on the detector-noise stream alone - and therefore what it must **not**
    produce if ``observe()`` is secretly also drawing from the calibration stream on
    every call, which is the defect this test guards against.
    """

    def test_two_observations_apply_the_same_drawn_calibration_scale(self) -> None:
        root_seed = 77
        instrument = InterferometryInstrument(root_seed=root_seed)
        instrument.configure(InstrumentConfig(values={"fringe_jump_rate": 0.0}))
        state = _state()

        calibration = instrument.calibrate(_references())
        scale = calibration.coefficients["phase_scale"]

        predicted = instrument.forward(state, _WINDOW).values
        sigma_common = interferometry_noise.vibration_phase_std_rad(_WINDOW.duration_s)
        sigma_independent = interferometry_noise.independent_phase_std_rad()
        detector_rng = generator(root_seed, Stream.DETECTOR_NOISE)

        def expected_next() -> np.ndarray:
            common = float(detector_rng.normal(0.0, sigma_common)) if sigma_common > 0.0 else 0.0
            independent = detector_rng.normal(0.0, sigma_independent, size=predicted.size)
            # sample_fringe_jumps draws exactly one array from the same stream, even at
            # rate=0.0 - reproduce that consumption so the two streams stay in lock-step.
            detector_rng.random(predicted.size)
            return scale * predicted + common + independent

        first_expected = expected_next()
        second_expected = expected_next()

        first_observed = instrument.observe(state, _WINDOW).values
        second_observed = instrument.observe(state, _WINDOW).values

        np.testing.assert_allclose(first_observed, first_expected, rtol=1.0e-9)
        np.testing.assert_allclose(second_observed, second_expected, rtol=1.0e-9)
