"""The simulated single Langmuir probe — WBS 2.12, doc 11 §9 item 7.

This is a *reference instrument*: it exists so that WBS 5.4's comparative study has
something concrete to compare the framework against, and doc 11 §9 item 7 is explicit that
the comparison "converts 'better than probes' into a measurement" rather than a slogan. The
project's own honest T2 error is 36.5 % — squarely inside the range real probes achieve —
so this module is built to measure the probe's error honestly, not to make it lose.

Two properties are load-bearing and both are tested directly:

* The naive single-probe analysis (log-slope ``T_e``, ``I_sat``-derived ``n_e``) recovers
  the truth **only** for a Maxwellian EEDF. That is doc 03 §3.2's point: the sheath EEDF is
  typically Druyvesteyn-like, and a log-slope fit to a Druyvesteyn retarding current is
  biased. The bias is not a bug to fix; it is the reason probes are uncertain.
* Sheath expansion inflates the apparent ion-saturation current, which the naive analysis
  reads off the curve uncorrected — so it biases the derived ``n_e`` high. That is Chen's
  and Merlino's textbook explanation for why cylindrical probe I-V curves do not show a
  flat ion-saturation plateau.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
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
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    Species,
)
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.probe.langmuir import (
    CURRENT_UNITS,
    LANGMUIR_INSTRUMENT_ID,
    LangmuirEstimate,
    LangmuirProbe,
    ProbeGeometry,
    bohm_speed_m_per_s,
    electron_current_a,
    estimate_from_iv_curve,
    ion_saturation_current_a,
    probe_current_a,
    sheath_expansion_radius_m,
)
from vpl.physics.eedf.analytic import DRUYVESTEYN_KAPPA, MAXWELLIAN_KAPPA

_E_C = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_GRID = SpatialGrid(z_m=np.linspace(0.0, 5.0e-2, 5))
_WINDOW = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(1.0, "ms"))
_BULK_INDEX = _GRID.n_points - 1

_TRUE_N_E = 5.0e16
_TRUE_T_E = 3.0
_TRUE_VP = 12.0


def _argon() -> Species:
    return Species(name="Ar+", mass=default_registry().quantity("species.Ar.mass"), charge_number=1)


def _params(*, kappa: float = 1.0) -> PlasmaParams:
    return PlasmaParams(
        species=_argon(),
        n_0=Q_(_TRUE_N_E, "m**-3"),
        T_e=Q_(_TRUE_T_E, "eV"),
        T_i=Q_(0.05, "eV"),
        T_g=Q_(300.0, "K"),
        pressure=Q_(5.0, "mTorr"),
        bias=Q_(-60.0, "V"),
        gamma_se=0.1,
        kappa=kappa,
    )


def _state(*, n_e: float = _TRUE_N_E, T_e: float = _TRUE_T_E, kappa: float = 1.0) -> PlasmaState:
    fields = {
        "n_e": ScalarField(
            name="n_e", values=np.full(_GRID.n_points, n_e), units="m**-3", grid=_GRID, time=None
        ),
        "n_i": ScalarField(
            name="n_i", values=np.full(_GRID.n_points, n_e), units="m**-3", grid=_GRID, time=None
        ),
        "Phi": ScalarField(
            name="Phi", values=np.full(_GRID.n_points, _TRUE_VP), units="V", grid=_GRID, time=None
        ),
        "T_e": ScalarField(
            name="T_e", values=np.full(_GRID.n_points, T_e), units="eV", grid=_GRID, time=None
        ),
    }
    return PlasmaState(
        params=_params(kappa=kappa),
        grid=_GRID,
        time=None,
        fields=fields,
        ion_distribution=None,
        fidelity=Fidelity.L0,
    )


def _geometry() -> ProbeGeometry:
    return ProbeGeometry(radius_m=1.25e-4, length_m=5.0e-3)


def _configured(**overrides: object) -> LangmuirProbe:
    instrument = LangmuirProbe(root_seed=3)
    settings: dict[str, object] = {
        "sweep_start_v": -60.0,
        "sweep_stop_v": 30.0,
        "sweep_points": 241,
    }
    settings.update(overrides)
    instrument.configure(InstrumentConfig(values=settings))
    return instrument


def _references() -> CalibrationSet:
    return CalibrationSet.of(
        CalibrationReference(
            name="Transimpedance amplifier gain standard",
            quantity="langmuir_current_scale",
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=0.03,
            traceable_to="doc 02 §11-style current-measurement calibration",
        )
    )


class TestBohmSpeed:
    def test_it_matches_the_cold_ion_formula(self) -> None:
        c_s = bohm_speed_m_per_s(electron_temperature_ev=3.0, ion_mass_kg=_argon().mass_kg)

        expected = math.sqrt(_E_C * 3.0 / _argon().mass_kg)
        assert c_s == pytest.approx(expected, rel=1e-12)

    def test_it_is_refused_for_a_non_positive_temperature(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            bohm_speed_m_per_s(electron_temperature_ev=0.0, ion_mass_kg=_argon().mass_kg)


class TestElectronCurrent:
    def test_it_matches_the_maxwellian_thermal_flux_formula_at_the_plasma_potential(self) -> None:
        """The one closed-form cross-check available: the random thermal electron flux
        ``(1/4) n_e <v_e> A`` at ``V = V_p``, where the retardation vanishes."""
        area = _geometry().area_m2
        i_e = electron_current_a(
            np.array([_TRUE_VP]),
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            probe_area_m2=area,
            kappa=MAXWELLIAN_KAPPA,
        )

        m_e = float(magnitude_in(ELECTRON_MASS, "kg"))
        mean_speed = math.sqrt(8.0 * _E_C * _TRUE_T_E / (math.pi * m_e))
        expected = 0.25 * _TRUE_N_E * _E_C * mean_speed * area

        assert float(i_e[0]) == pytest.approx(expected, rel=1e-3)

    def test_it_decreases_as_the_probe_is_biased_further_negative(self) -> None:
        area = _geometry().area_m2
        voltages = np.array([_TRUE_VP - 5.0, _TRUE_VP - 20.0])
        i_e = electron_current_a(
            voltages,
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            probe_area_m2=area,
        )

        assert i_e[1] < i_e[0]
        assert np.all(i_e > 0.0)

    def test_it_is_positive_and_finite_well_above_the_plasma_potential(self) -> None:
        area = _geometry().area_m2
        i_e = electron_current_a(
            np.array([_TRUE_VP + 50.0]),
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            probe_area_m2=area,
        )

        assert np.all(np.isfinite(i_e))
        assert i_e[0] > 0.0


class TestSheathExpansion:
    def test_the_effective_radius_is_the_bare_radius_at_the_plasma_potential(self) -> None:
        radius = sheath_expansion_radius_m(
            np.array([_TRUE_VP]),
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            probe_radius_m=_geometry().radius_m,
        )

        assert float(radius[0]) == pytest.approx(_geometry().radius_m, rel=1e-9)

    def test_the_effective_radius_grows_as_the_bias_goes_more_negative(self) -> None:
        radius = sheath_expansion_radius_m(
            np.array([_TRUE_VP - 5.0, _TRUE_VP - 50.0]),
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            probe_radius_m=_geometry().radius_m,
        )

        assert radius[1] > radius[0] > _geometry().radius_m


class TestIonSaturationCurrent:
    def test_it_grows_in_magnitude_with_more_negative_bias(self) -> None:
        """Doc 11 §9 item 7 / Chen's review: sheath expansion is why real cylindrical
        probe I-V curves do not show a flat ion-saturation plateau."""
        voltages = np.array([_TRUE_VP - 10.0, _TRUE_VP - 55.0])
        i_i = ion_saturation_current_a(
            voltages,
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            ion_mass_kg=_argon().mass_kg,
            geometry=_geometry(),
        )

        assert i_i[1] > i_i[0] > 0.0

    def test_it_matches_the_bare_area_bohm_flux_at_the_plasma_potential(self) -> None:
        i_i = ion_saturation_current_a(
            np.array([_TRUE_VP]),
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            ion_mass_kg=_argon().mass_kg,
            geometry=_geometry(),
            edge_to_centre_ratio=0.61,
        )

        c_s = bohm_speed_m_per_s(electron_temperature_ev=_TRUE_T_E, ion_mass_kg=_argon().mass_kg)
        expected = _E_C * 0.61 * _TRUE_N_E * c_s * _geometry().area_m2
        assert float(i_i[0]) == pytest.approx(expected, rel=1e-6)


class TestProbeCurrentIsMonotonic:
    def test_the_full_curve_rises_from_ion_saturation_to_electron_saturation(self) -> None:
        """Ion-saturated (negative) below, electron-saturated (positive, flat — this
        planar-retardation model does not grow the electron branch further once
        ``V >= V_p``) above, and strictly increasing through the retarding region that
        joins them."""
        voltages = np.linspace(_TRUE_VP - 60.0, _TRUE_VP + 30.0, 121)
        current = probe_current_a(
            voltages,
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            ion_mass_kg=_argon().mass_kg,
            geometry=_geometry(),
        )

        assert current[0] < 0.0 < current[-1]
        assert np.all(np.diff(current) >= -1e-30)
        below_vp = voltages < _TRUE_VP
        assert np.all(np.diff(current[below_vp]) > 0.0)


class TestEstimateFromIvCurve:
    def test_it_recovers_the_true_temperature_for_a_maxwellian_eedf(self) -> None:
        """The case the exponential-slope analysis is derived for."""
        voltages = np.linspace(_TRUE_VP - 40.0, _TRUE_VP + 20.0, 201)
        current = probe_current_a(
            voltages,
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            ion_mass_kg=_argon().mass_kg,
            geometry=_geometry(),
            kappa=MAXWELLIAN_KAPPA,
        )

        estimate = estimate_from_iv_curve(
            voltages,
            current,
            ion_mass_kg=_argon().mass_kg,
            probe_area_m2=_geometry().area_m2,
        )

        assert isinstance(estimate, LangmuirEstimate)
        assert estimate.electron_temperature_ev == pytest.approx(_TRUE_T_E, rel=0.05)

    def test_a_druyvesteyn_eedf_biases_the_extracted_temperature(self) -> None:
        """Doc 03 §3.2: the sheath EEDF is typically Druyvesteyn-like, and a log-slope
        fit to it is not the exponential law it is derived from. The bias is the point."""
        voltages = np.linspace(_TRUE_VP - 40.0, _TRUE_VP + 20.0, 201)

        maxwellian_current = probe_current_a(
            voltages,
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            ion_mass_kg=_argon().mass_kg,
            geometry=_geometry(),
            kappa=MAXWELLIAN_KAPPA,
        )
        druyvesteyn_current = probe_current_a(
            voltages,
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            ion_mass_kg=_argon().mass_kg,
            geometry=_geometry(),
            kappa=DRUYVESTEYN_KAPPA,
        )

        maxwellian_estimate = estimate_from_iv_curve(
            voltages,
            maxwellian_current,
            ion_mass_kg=_argon().mass_kg,
            probe_area_m2=_geometry().area_m2,
        )
        druyvesteyn_estimate = estimate_from_iv_curve(
            voltages,
            druyvesteyn_current,
            ion_mass_kg=_argon().mass_kg,
            probe_area_m2=_geometry().area_m2,
        )

        maxwellian_error = abs(maxwellian_estimate.electron_temperature_ev - _TRUE_T_E) / _TRUE_T_E
        druyvesteyn_error = (
            abs(druyvesteyn_estimate.electron_temperature_ev - _TRUE_T_E) / _TRUE_T_E
        )

        assert druyvesteyn_error > maxwellian_error
        assert druyvesteyn_error > 0.10

    def test_sheath_expansion_biases_the_estimated_density_high(self) -> None:
        """The naive analysis divides I_sat by the *bare* probe area, so the sheath
        expansion this module models (see TestIonSaturationCurrent) is not corrected for
        and n_e is overestimated — exactly the textbook failure mode this comparison
        exists to quantify."""
        voltages = np.linspace(_TRUE_VP - 40.0, _TRUE_VP + 20.0, 201)
        current = probe_current_a(
            voltages,
            plasma_potential_v=_TRUE_VP,
            electron_density_m3=_TRUE_N_E,
            electron_temperature_ev=_TRUE_T_E,
            ion_mass_kg=_argon().mass_kg,
            geometry=_geometry(),
            kappa=MAXWELLIAN_KAPPA,
        )

        estimate = estimate_from_iv_curve(
            voltages, current, ion_mass_kg=_argon().mass_kg, probe_area_m2=_geometry().area_m2
        )

        assert estimate.electron_density_m3 > _TRUE_N_E


class TestTheContract:
    def test_it_is_recognised_as_an_instrument(self) -> None:
        assert isinstance(LangmuirProbe(root_seed=1), Instrument)

    def test_forward_before_configure_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="configure"):
            LangmuirProbe(root_seed=1).forward(_state(), _WINDOW)

    def test_the_metadata_carries_a_citation_and_a_detection_floor(self) -> None:
        metadata = _configured().metadata()

        assert metadata.instrument_id == LANGMUIR_INSTRUMENT_ID
        assert metadata.citations
        assert metadata.detection_floor.quantity == "n_0"


class TestForward:
    def test_it_returns_one_sample_per_sweep_point(self) -> None:
        observable = _configured(sweep_points=51).forward(_state(), _WINDOW)

        assert observable.n_samples == 51
        assert observable.instrument_id == LANGMUIR_INSTRUMENT_ID
        assert observable.units == CURRENT_UNITS

    def test_it_is_deterministic(self) -> None:
        state = _state()
        first = _configured().forward(state, _WINDOW)
        second = _configured().forward(state, _WINDOW)

        np.testing.assert_array_equal(first.values, second.values)

    def test_it_reflects_the_configured_kappa_via_the_state(self) -> None:
        maxwellian = _configured().forward(_state(kappa=1.0), _WINDOW)
        druyvesteyn = _configured().forward(_state(kappa=2.0), _WINDOW)

        assert not np.allclose(maxwellian.values, druyvesteyn.values, rtol=1e-6)


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

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        assert not np.array_equal(observed.values, predicted.values)
        assert np.all(observed.uncertainty > 0.0)

    def test_the_same_seed_gives_the_same_observation(self) -> None:
        state = _state()
        first = LangmuirProbe(root_seed=9)
        first.configure(InstrumentConfig(values={"sweep_points": 41}))
        second = LangmuirProbe(root_seed=9)
        second.configure(InstrumentConfig(values={"sweep_points": 41}))

        np.testing.assert_array_equal(
            first.observe(state, _WINDOW).values, second.observe(state, _WINDOW).values
        )


class TestCalibrate:
    def test_the_estimated_calibration_carries_uncertainty(self) -> None:
        calibration = _configured().calibrate(_references())

        assert calibration.state is CalibrationState.ESTIMATED
        assert any(value > 0.0 for value in calibration.relative_uncertainty.values())

    def test_a_calibration_set_missing_the_standard_is_refused(self) -> None:
        with pytest.raises(KeyError):
            _configured().calibrate(CalibrationSet.of())


class TestLikelihood:
    def test_a_perfect_prediction_is_the_most_probable_one(self) -> None:
        instrument = _configured()
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

    def test_a_prediction_of_the_wrong_length_is_refused(self) -> None:
        instrument = _configured()
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


class TestTheDetectionGate:
    def test_a_dense_plasma_is_informative(self) -> None:
        assert _configured().is_informative(_params())

    def test_a_plasma_below_the_density_floor_is_not(self) -> None:
        sparse = PlasmaParams(
            species=_argon(),
            n_0=Q_(1.0e12, "m**-3"),
            T_e=Q_(_TRUE_T_E, "eV"),
            T_i=Q_(0.05, "eV"),
            T_g=Q_(300.0, "K"),
            pressure=Q_(5.0, "mTorr"),
            bias=Q_(-60.0, "V"),
            gamma_se=0.1,
            kappa=1.0,
        )
        assert not _configured().is_informative(sparse)


class TestTheBiasFindingIsRobustButItsMagnitudeIsNot:
    """What doc 11 §9 item 7's comparative figure may and may not claim.

    The probe's non-Maxwellian ``T_e`` bias is about to be used as the baseline this
    project is measured against. Before that number is quoted anywhere, it is worth knowing
    which part of it is physics and which part is an analysis choice — because the log-slope
    fit window is a genuinely free choice a real experimentalist makes, and if the headline
    bias moves with it, quoting a single value overstates our own advantage.

    Measured across seven defensible windows: the Druyvesteyn error exceeds the Maxwellian
    error in every one, by a factor of 2.0 to 4.7 — so the *finding* is robust. But its
    magnitude ranges from 4.0 % to 24.4 %, a factor of six. **The comparative figure must
    therefore report a range over analysis choices, not the single number the default window
    happens to give.** Quoting 16.7 % as "the" probe error would be choosing the baseline
    that flatters us.
    """

    @staticmethod
    def _error(kappa: float, window: tuple[float, float]) -> float:
        import vpl.instruments.probe.langmuir.analysis as analysis

        lower, upper = analysis._FIT_WINDOW_LOWER_FRACTION, analysis._FIT_WINDOW_UPPER_FRACTION
        analysis._FIT_WINDOW_LOWER_FRACTION, analysis._FIT_WINDOW_UPPER_FRACTION = window
        try:
            voltages = np.linspace(_TRUE_VP - 40.0, _TRUE_VP + 20.0, 201)
            current = probe_current_a(
                voltages,
                plasma_potential_v=_TRUE_VP,
                electron_density_m3=_TRUE_N_E,
                electron_temperature_ev=_TRUE_T_E,
                ion_mass_kg=_argon().mass_kg,
                geometry=_geometry(),
                kappa=kappa,
            )
            estimate = estimate_from_iv_curve(
                voltages,
                current,
                ion_mass_kg=_argon().mass_kg,
                probe_area_m2=_geometry().area_m2,
            )
            return abs(estimate.electron_temperature_ev - _TRUE_T_E) / _TRUE_T_E
        finally:
            analysis._FIT_WINDOW_LOWER_FRACTION = lower
            analysis._FIT_WINDOW_UPPER_FRACTION = upper

    _WINDOWS = (
        (0.05, 0.4),
        (0.1, 0.4),
        (0.2, 0.4),
        (0.3, 0.5),
        (0.2, 0.6),
        (0.25, 0.45),
        (0.4, 0.7),
    )

    @pytest.mark.physics
    def test_the_druyvesteyn_bias_exceeds_the_maxwellian_one_in_every_window(self) -> None:
        # The defensible claim, and the only one the comparative figure may make
        # unqualified.
        for window in self._WINDOWS:
            maxwellian = self._error(MAXWELLIAN_KAPPA, window)
            druyvesteyn = self._error(DRUYVESTEYN_KAPPA, window)
            assert druyvesteyn > maxwellian, f"window {window} does not show the bias"

    @pytest.mark.physics
    def test_the_magnitude_swings_by_several_fold_across_defensible_windows(self) -> None:
        # Deliberately asserts that the number is NOT stable, so that anyone tempted to
        # quote a single probe error has to come here and read why they must not.
        errors = [self._error(DRUYVESTEYN_KAPPA, window) for window in self._WINDOWS]

        assert max(errors) / min(errors) > 3.0, (
            "if the magnitude has become stable, this guard is obsolete and the comparative "
            "figure may quote a single number — verify that before deleting this test"
        )
