"""The LIF channel as an ``Instrument`` — doc 08 §4, doc 04 §9, doc 01 IF-6.

Doc 04 §9 states the contract the tests here hold the class to:

    ``observe`` returns raw data with noise and imperfect calibration; ``forward`` returns
    the noiseless expectation used by the likelihood. **Both come from the same code
    path**, with noise and calibration error as switchable stages, which guarantees they
    cannot drift apart — a class of bug that would silently invalidate every result.

and doc 01 IF-6 states the gate: a channel below its floor "contributes *no* likelihood
term rather than a noisy one". For LIF the floor has two parts — density, and the doc 01
§5.1 tuning range — and both are tested.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.core.constants import ELEMENTARY_CHARGE
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
    VelocityDistribution,
)
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.lif.instrument import LifInstrument

_E_C = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_GRID = SpatialGrid(z_m=np.linspace(0.0, 1.0e-3, 5))
_WINDOW = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(200.0, "ns"))


def _argon() -> Species:
    return Species(name="Ar+", mass=default_registry().quantity("species.Ar.mass"), charge_number=1)


def _params(*, n_0: float = 1.0e17, T_i_ev: float = 0.05, bias_v: float = -250.0) -> PlasmaParams:
    return PlasmaParams(
        species=_argon(),
        n_0=Q_(n_0, "m**-3"),
        T_e=Q_(3.0, "eV"),
        T_i=Q_(T_i_ev, "eV"),
        T_g=Q_(300.0, "K"),
        pressure=Q_(5.0, "mTorr"),
        bias=Q_(bias_v, "V"),
        gamma_se=0.1,
        kappa=1.0,
    )


def _state(*, drift: float = -2.7e3, T_i_ev: float = 0.05, n_0: float = 1.0e17) -> PlasmaState:
    species = _argon()
    sigma = math.sqrt(T_i_ev * _E_C / species.mass_kg)
    velocity = np.linspace(drift - 8.0 * sigma, drift + 8.0 * sigma, 2001)
    profile = (
        n_0
        / (math.sqrt(2.0 * math.pi) * sigma)
        * np.exp(-((velocity - drift) ** 2) / (2.0 * sigma**2))
    )
    fields = {
        "n_e": ScalarField(
            name="n_e", values=np.full(_GRID.n_points, n_0), units="m**-3", grid=_GRID, time=None
        ),
        "n_i": ScalarField(
            name="n_i", values=np.full(_GRID.n_points, n_0), units="m**-3", grid=_GRID, time=None
        ),
        "Phi": ScalarField(
            name="Phi", values=np.zeros(_GRID.n_points), units="V", grid=_GRID, time=None
        ),
        "T_e": ScalarField(
            name="T_e", values=np.full(_GRID.n_points, 3.0), units="eV", grid=_GRID, time=None
        ),
    }
    return PlasmaState(
        params=_params(n_0=n_0, T_i_ev=T_i_ev),
        grid=_GRID,
        time=None,
        fields=fields,
        ion_distribution=VelocityDistribution(
            grid=_GRID,
            v_m_per_s=velocity,
            values=np.tile(profile, (_GRID.n_points, 1)),
            species=species,
        ),
        fidelity=Fidelity.L0,
    )


def _configured(**overrides: object) -> LifInstrument:
    instrument = LifInstrument()
    settings: dict[str, object] = {"scan_points": 121, "scan_half_span_ghz": 4.0, "seed": 7}
    settings.update(overrides)
    instrument.configure(InstrumentConfig(values=settings))
    return instrument


def _references() -> CalibrationSet:
    return CalibrationSet.of(
        CalibrationReference(
            name="Wavemeter + Fabry-Perot etalon",
            quantity="lif_frequency_axis",
            value=Q_(2.0, "MHz"),
            relative_uncertainty=0.0,
            traceable_to="NIST-traceable wavemeter, doc 02 §11",
        ),
        CalibrationReference(
            name="Relative LIF amplitude scale",
            quantity="lif_relative_scale",
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=0.05,
            traceable_to="doc 06 §4 item 4 (metastable fraction)",
        ),
    )


class TestTheContract:
    def test_it_is_recognised_as_an_instrument(self) -> None:
        assert isinstance(LifInstrument(), Instrument)

    def test_the_metadata_carries_a_citation_and_a_detection_floor(self) -> None:
        metadata = _configured().metadata()

        assert metadata.instrument_id == "lif"
        assert metadata.citations
        assert metadata.detection_floor.quantity == "n_0"

    def test_forward_before_configure_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="configure"):
            LifInstrument().forward(_state(), _WINDOW)


class TestForward:
    def test_it_returns_one_sample_per_scan_point(self) -> None:
        observable = _configured(scan_points=51).forward(_state(), _WINDOW)

        assert observable.n_samples == 51
        assert observable.instrument_id == "lif"
        assert observable.window == _WINDOW

    def test_the_observable_is_an_upper_state_density(self) -> None:
        observable = _configured().forward(_state(), _WINDOW)

        assert observable.units == "m**-3"
        assert np.all(observable.values > 0.0)

    def test_a_state_with_no_ion_distribution_is_refused(self) -> None:
        state = _state()
        without = PlasmaState(
            params=state.params,
            grid=state.grid,
            time=None,
            fields=dict(state.fields),
            ion_distribution=None,
            fidelity=Fidelity.L0,
        )

        with pytest.raises(ValueError, match="ion distribution"):
            _configured().forward(without, _WINDOW)

    def test_it_scales_with_the_metastable_fraction(self) -> None:
        state = _state()

        base = _configured(metastable_fraction=0.01).forward(state, _WINDOW)
        doubled = _configured(metastable_fraction=0.02).forward(state, _WINDOW)

        np.testing.assert_allclose(doubled.values, 2.0 * base.values, rtol=1e-12)

    def test_it_is_deterministic(self) -> None:
        state = _state()

        first = _configured().forward(state, _WINDOW)
        second = _configured().forward(state, _WINDOW)

        np.testing.assert_array_equal(first.values, second.values)

    def test_a_scan_wider_than_the_laser_can_tune_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mode-hop-free"):
            _configured(scan_half_span_ghz=25.0)

    def test_the_field_changes_the_lineshape(self) -> None:
        state = _state()

        unmagnetised = _configured(magnetic_field_mt=0.0).forward(state, _WINDOW)
        magnetised = _configured(magnetic_field_mt=5.0).forward(state, _WINDOW)

        assert not np.allclose(unmagnetised.values, magnetised.values, rtol=1e-6)

    def test_the_polarisation_changes_the_lineshape(self) -> None:
        state = _state()

        pi_pumped = _configured(magnetic_field_mt=5.0, polarisation="pi").forward(state, _WINDOW)
        sigma = _configured(magnetic_field_mt=5.0, polarisation="sigma_plus").forward(
            state, _WINDOW
        )

        assert not np.allclose(pi_pumped.values, sigma.values, rtol=1e-6)


class TestObserveSharesTheForwardPath:
    def test_observe_reduces_to_forward_with_calibration_error_switched_off(self) -> None:
        """Doc 04 §9: the two must come from one code path with noise as a stage.

        Switching the stage off must reproduce ``forward`` **exactly** — the same standard
        doc 04 V-30 sets for the noise sources.
        """
        state = _state()
        instrument = _configured(apply_calibration_error=False)

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        np.testing.assert_array_equal(observed.values, predicted.values)
        assert observed.as_observable() == predicted

    def test_observe_applies_the_estimated_calibration_by_default(self) -> None:
        observed = _configured().observe(_state(), _WINDOW)

        assert observed.calibration is CalibrationState.ESTIMATED
        assert not observed.is_inverse_crime

    def test_the_calibration_error_is_one_common_factor_and_does_not_average_down(self) -> None:
        """Doc 06 §4.1: a correlated calibration error "affects *every* point identically
        and does **not** average down"."""
        state = _state()
        instrument = _configured()

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        ratio = observed.values / predicted.values

        assert np.ptp(ratio) < 1.0e-12
        assert not math.isclose(float(ratio[0]), 1.0, abs_tol=1e-9)

    def test_the_uncertainty_is_the_registered_relative_scale(self) -> None:
        observed = _configured().observe(_state(), _WINDOW)

        expected = float(default_registry().value_in("LIF.scale_uncertainty", "dimensionless"))

        np.testing.assert_allclose(observed.uncertainty / observed.values, expected, rtol=1e-12)

    def test_the_same_seed_gives_the_same_observation(self) -> None:
        state = _state()

        first = _configured(seed=11).observe(state, _WINDOW)
        second = _configured(seed=11).observe(state, _WINDOW)

        np.testing.assert_array_equal(first.values, second.values)

    def test_a_different_seed_gives_a_different_observation(self) -> None:
        state = _state()

        first = _configured(seed=11).observe(state, _WINDOW)
        second = _configured(seed=12).observe(state, _WINDOW)

        assert not np.array_equal(first.values, second.values)


class TestCalibrate:
    def test_the_estimated_calibration_carries_uncertainty(self) -> None:
        calibration = _configured().calibrate(_references())

        assert calibration.state is CalibrationState.ESTIMATED
        assert any(value > 0.0 for value in calibration.relative_uncertainty.values())

    def test_a_calibration_set_missing_the_frequency_standard_is_refused(self) -> None:
        partial = CalibrationSet.of(
            CalibrationReference(
                name="Relative LIF amplitude scale",
                quantity="lif_relative_scale",
                value=Q_(1.0, "dimensionless"),
                relative_uncertainty=0.05,
                traceable_to="doc 06 §4",
            )
        )

        with pytest.raises(KeyError, match="lif_frequency_axis"):
            _configured().calibrate(partial)


class TestLikelihood:
    def test_a_perfect_prediction_is_the_most_probable_one(self) -> None:
        instrument = _configured()
        state = _state()
        observed = instrument.observe(state, _WINDOW)
        predicted = observed.as_observable()

        exact = instrument.likelihood(observed, predicted)
        offset = instrument.likelihood(
            observed,
            type(predicted)(
                instrument_id=predicted.instrument_id,
                values=predicted.values * 1.05,
                units=predicted.units,
                window=predicted.window,
            ),
        )

        assert exact > offset

    def test_it_is_the_gaussian_log_density(self) -> None:
        instrument = _configured()
        observed = instrument.observe(_state(), _WINDOW)
        predicted = observed.as_observable()
        shifted = type(predicted)(
            instrument_id=predicted.instrument_id,
            values=predicted.values * 1.10,
            units=predicted.units,
            window=predicted.window,
        )

        residual = (observed.values - shifted.values) / observed.uncertainty
        expected = -0.5 * float(np.sum(residual**2)) - float(
            np.sum(np.log(observed.uncertainty * math.sqrt(2.0 * math.pi)))
        )

        assert instrument.likelihood(observed, shifted) == pytest.approx(expected, rel=1e-12)

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
    def test_a_dense_slow_plasma_is_informative(self) -> None:
        assert _configured().is_informative(_params(bias_v=-20.0))

    def test_a_plasma_below_the_density_floor_is_not(self) -> None:
        """Doc 01 IF-6: below the floor the channel is modelled as *absent*."""
        assert not _configured().is_informative(_params(n_0=1.0e13, bias_v=-20.0))

    def test_a_high_bias_sheath_is_gated_off_by_the_tuning_range(self) -> None:
        """The doc 01 §5.1 limitation, enforced at the doc 08 §4 protocol boundary.

        At a 250 V bias the ions entering the wall are far outside the reachable velocity
        window, so the channel is not weakly informative — it is uninformative, and doc 01
        IF-6 requires that to be expressed by the term not being formed.
        """
        assert not _configured().is_informative(_params(bias_v=-250.0, T_i_ev=0.05))

    def test_a_low_bias_sheath_stays_inside_the_tuning_range(self) -> None:
        assert _configured().is_informative(_params(bias_v=-20.0))

    def test_the_gate_moves_with_the_grazing_angle(self) -> None:
        """A steeper beam sees more of ``v_z`` per hertz and therefore reaches less far."""
        shallow = _configured()
        steep = _configured(grazing_angle_deg=90.0)

        assert shallow.is_informative(_params(bias_v=-100.0))
        assert not steep.is_informative(_params(bias_v=-100.0))


class TestTheReportedRunConditions:
    def test_the_saturation_parameter_is_reported(self) -> None:
        """Doc 04 §3.4: ``S`` "is computed and reported for every run"."""
        report = _configured().run_conditions()

        assert report.saturation > 0.0
        assert magnitude_in(report.power_broadened_fwhm, "Hz") > 0.0

    def test_the_nominal_beam_is_flagged_as_saturated(self) -> None:
        assert _configured().run_conditions().is_saturated

    def test_an_attenuated_beam_is_not(self) -> None:
        assert not _configured(saturation=0.01).run_conditions().is_saturated

    def test_the_coverage_of_a_state_is_reported(self) -> None:
        coverage = _configured().coverage(_state())

        assert coverage.visible_fraction == pytest.approx(1.0, abs=1e-6)

    def test_a_fast_population_is_reported_as_truncated(self) -> None:
        coverage = _configured().coverage(_state(drift=-3.47e4, T_i_ev=0.5))

        assert coverage.is_truncated


class TestConfigurationAndReprs:
    def test_the_scan_velocity_axis_is_the_ivdf_abscissa(self) -> None:
        instrument = _configured(scan_points=51, scan_half_span_ghz=4.0)

        velocities = instrument.scan_velocities()

        assert magnitude_in(velocities, "m/s").size == 51
        # +/-4 GHz over sin(15 deg) at 668.614 nm.
        expected = 668.614e-9 * 4.0e9 / math.sin(math.radians(15.0))
        assert magnitude_in(velocities, "m/s").max() == pytest.approx(expected, rel=1e-9)

    def test_the_beam_geometry_can_be_overridden_from_the_manifest(self) -> None:
        state = _state()

        narrow = _configured(beam_waist_um=25.0, laser_power_mw=5.0).forward(state, _WINDOW)
        nominal = _configured().forward(state, _WINDOW)

        assert narrow.n_samples == nominal.n_samples

    def test_a_non_integer_scan_point_count_is_refused(self) -> None:
        with pytest.raises(TypeError, match="scan_points"):
            _configured(scan_points=51.5)

    def test_a_non_numeric_saturation_is_refused(self) -> None:
        with pytest.raises(TypeError, match="saturation"):
            _configured(saturation="high")

    def test_the_likelihood_refuses_a_unit_mismatch(self) -> None:
        instrument = _configured()
        observed = instrument.observe(_state(), _WINDOW)
        predicted = observed.as_observable()

        with pytest.raises(ValueError, match="m\\*\\*-3"):
            instrument.likelihood(
                observed,
                type(predicted)(
                    instrument_id=predicted.instrument_id,
                    values=predicted.values,
                    units="m**-2",
                    window=predicted.window,
                ),
            )

    def test_coverage_of_a_state_with_no_distribution_is_refused(self) -> None:
        state = _state()
        without = PlasmaState(
            params=state.params,
            grid=state.grid,
            time=None,
            fields=dict(state.fields),
            ion_distribution=None,
            fidelity=Fidelity.L0,
        )

        with pytest.raises(ValueError, match="has none"):
            _configured().coverage(without)

    def test_the_run_conditions_repr_names_the_regime(self) -> None:
        assert "saturated" in repr(_configured().run_conditions())
        assert "linear" in repr(_configured(saturation=0.01).run_conditions())
