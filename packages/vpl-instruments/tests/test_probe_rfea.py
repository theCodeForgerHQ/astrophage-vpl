"""The simulated retarding field energy analyser (RFEA) — WBS 2.12, doc 11 §9 item 7.

The second of the two reference instruments doc 11 §9 item 7 asks for: "a comparative
figure vs simulated probe / RFEA — converts 'better than probes' into a measurement".
Built to measure, not to win — see the module docstring of
:mod:`vpl.instruments.probe.langmuir` for the framework-level statement of that rule.

Two properties are load-bearing:

* ``-dI/dV`` recovers the ion energy distribution function (IEDF) the grid was swept
  against — the textbook RFEA analysis (Böhm & Perrin 1993; Sudit & Woods 1994).
* Finite grid transparency has an energy resolution set by the wire spacing relative to
  the inter-grid separation (fringing-field smearing), and that resolution broadens a
  narrow input IEDF in the recovered spectrum. That is what actually limits a real RFEA's
  energy resolution, and it is the reason its answer is uncertain in exactly the same
  spirit as the Langmuir probe's log-slope bias.
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
from vpl.instruments.probe.rfea import (
    CURRENT_UNITS,
    RFEA_INSTRUMENT_ID,
    GridGeometry,
    IedfEstimate,
    RfeaInstrument,
    collected_current_a,
    estimate_iedf,
    flux_iedf_ev,
)

_E_C = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_GRID = SpatialGrid(z_m=np.linspace(0.0, 5.0e-2, 5))
_WINDOW = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(1.0, "ms"))


def _argon() -> Species:
    return Species(name="Ar+", mass=default_registry().quantity("species.Ar.mass"), charge_number=1)


def _drift_speed_for_energy_ev(energy_ev: float, *, mass_kg: float) -> float:
    """The (negative, toward-wall) drift speed whose kinetic energy is ``energy_ev``."""
    return -math.sqrt(2.0 * _E_C * energy_ev / mass_kg)


def _narrow_distribution(
    *, peak_energy_ev: float, relative_spread: float = 0.02, n_points: int = 4001
) -> VelocityDistribution:
    """A narrow drifting population toward the wall, peaked at a known ion energy."""
    species = _argon()
    drift = _drift_speed_for_energy_ev(peak_energy_ev, mass_kg=species.mass_kg)
    sigma = abs(drift) * relative_spread
    velocity = np.linspace(drift - 8.0 * sigma, drift + 8.0 * sigma, n_points)
    density = 1.0e16
    profile = (
        density
        / (math.sqrt(2.0 * math.pi) * sigma)
        * np.exp(-((velocity - drift) ** 2) / (2.0 * sigma**2))
    )
    return VelocityDistribution(
        grid=_GRID,
        v_m_per_s=velocity,
        values=np.tile(profile, (_GRID.n_points, 1)),
        species=species,
    )


def _geometry(*, wire_spacing_m: float = 5.0e-5, grid_separation_m: float = 1.0e-3) -> GridGeometry:
    return GridGeometry(
        collector_area_m2=math.pi * (1.0e-3) ** 2,
        wire_spacing_m=wire_spacing_m,
        grid_separation_m=grid_separation_m,
        transmission=0.15,
    )


def _state(*, distribution: VelocityDistribution) -> PlasmaState:
    fields = {
        "n_e": ScalarField(
            name="n_e", values=np.full(_GRID.n_points, 1.0e16), units="m**-3", grid=_GRID, time=None
        ),
        "n_i": ScalarField(
            name="n_i", values=np.full(_GRID.n_points, 1.0e16), units="m**-3", grid=_GRID, time=None
        ),
        "Phi": ScalarField(
            name="Phi", values=np.zeros(_GRID.n_points), units="V", grid=_GRID, time=None
        ),
        "T_e": ScalarField(
            name="T_e", values=np.full(_GRID.n_points, 3.0), units="eV", grid=_GRID, time=None
        ),
    }
    return PlasmaState(
        params=PlasmaParams(
            species=_argon(),
            n_0=Q_(1.0e16, "m**-3"),
            T_e=Q_(3.0, "eV"),
            T_i=Q_(0.05, "eV"),
            T_g=Q_(300.0, "K"),
            pressure=Q_(5.0, "mTorr"),
            bias=Q_(-200.0, "V"),
            gamma_se=0.1,
            kappa=1.0,
        ),
        grid=_GRID,
        time=None,
        fields=fields,
        ion_distribution=distribution,
        fidelity=Fidelity.L2,
    )


def _configured(**overrides: object) -> RfeaInstrument:
    instrument = RfeaInstrument(root_seed=5)
    settings: dict[str, object] = {"sweep_stop_v": 300.0, "sweep_points": 301}
    settings.update(overrides)
    instrument.configure(InstrumentConfig(values=settings))
    return instrument


def _references() -> CalibrationSet:
    return CalibrationSet.of(
        CalibrationReference(
            name="Transimpedance amplifier gain standard",
            quantity="rfea_current_scale",
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=0.03,
            traceable_to="doc 02 §11-style current-measurement calibration",
        )
    )


class TestFluxIedf:
    def test_it_conserves_the_total_particle_flux_toward_the_wall(self) -> None:
        distribution = _narrow_distribution(peak_energy_ev=200.0)

        energy_ev, gamma_ev = flux_iedf_ev(distribution, z_index=0, mass_kg=_argon().mass_kg)

        integrated = float(np.trapezoid(gamma_ev, energy_ev))
        expected = float(distribution.particle_flux_toward_wall_per_m2_s()[0])
        assert integrated == pytest.approx(expected, rel=0.02)

    def test_the_energy_axis_is_strictly_increasing_and_non_negative(self) -> None:
        distribution = _narrow_distribution(peak_energy_ev=200.0)

        energy_ev, _ = flux_iedf_ev(distribution, z_index=0, mass_kg=_argon().mass_kg)

        assert np.all(energy_ev >= 0.0)
        assert np.all(np.diff(energy_ev) > 0.0)

    def test_the_flux_peaks_near_the_known_drift_energy(self) -> None:
        distribution = _narrow_distribution(peak_energy_ev=180.0)

        energy_ev, gamma_ev = flux_iedf_ev(distribution, z_index=0, mass_kg=_argon().mass_kg)

        peak_energy = float(energy_ev[np.argmax(gamma_ev)])
        assert peak_energy == pytest.approx(180.0, rel=0.05)


class TestCollectedCurrent:
    def test_it_approaches_the_total_flux_current_at_full_transmission(self) -> None:
        distribution = _narrow_distribution(peak_energy_ev=200.0)
        energy_ev, gamma_ev = flux_iedf_ev(distribution, z_index=0, mass_kg=_argon().mass_kg)
        geometry = _geometry()

        current = collected_current_a(
            np.array([-10.0]),
            energy_ev=energy_ev,
            gamma_ev=gamma_ev,
            geometry=geometry,
            resolution_fwhm_ev=0.0,
        )

        expected = (
            _E_C
            * geometry.collector_area_m2
            * geometry.transmission
            * float(distribution.particle_flux_toward_wall_per_m2_s()[0])
        )
        assert float(current[0]) == pytest.approx(expected, rel=0.02)

    def test_it_vanishes_above_the_peak_energy(self) -> None:
        distribution = _narrow_distribution(peak_energy_ev=200.0)
        energy_ev, gamma_ev = flux_iedf_ev(distribution, z_index=0, mass_kg=_argon().mass_kg)
        geometry = _geometry()

        current = collected_current_a(
            np.array([260.0]),
            energy_ev=energy_ev,
            gamma_ev=gamma_ev,
            geometry=geometry,
            resolution_fwhm_ev=0.0,
        )

        assert float(current[0]) == pytest.approx(0.0, abs=1e-16)

    def test_current_decreases_monotonically_with_discriminator_voltage(self) -> None:
        distribution = _narrow_distribution(peak_energy_ev=200.0)
        energy_ev, gamma_ev = flux_iedf_ev(distribution, z_index=0, mass_kg=_argon().mass_kg)
        geometry = _geometry()

        voltages = np.linspace(-10.0, 260.0, 55)
        current = collected_current_a(
            voltages,
            energy_ev=energy_ev,
            gamma_ev=gamma_ev,
            geometry=geometry,
            resolution_fwhm_ev=1.0,
        )

        assert np.all(np.diff(current) <= 1e-30)


class TestEstimateIedf:
    def test_minus_di_dv_recovers_the_known_input_energy(self) -> None:
        distribution = _narrow_distribution(peak_energy_ev=200.0)
        energy_ev, gamma_ev = flux_iedf_ev(distribution, z_index=0, mass_kg=_argon().mass_kg)
        geometry = _geometry(wire_spacing_m=1.0e-6, grid_separation_m=1.0e-3)

        voltages = np.linspace(100.0, 300.0, 401)
        current = collected_current_a(
            voltages,
            energy_ev=energy_ev,
            gamma_ev=gamma_ev,
            geometry=geometry,
            resolution_fwhm_ev=0.5,
        )

        estimate = estimate_iedf(voltages, current, geometry=geometry)

        assert isinstance(estimate, IedfEstimate)
        peak_energy = float(estimate.energy_ev[np.argmax(estimate.gamma_ev)])
        assert peak_energy == pytest.approx(200.0, rel=0.05)

    def test_finite_grid_resolution_broadens_a_narrow_iedf(self) -> None:
        """The load-bearing behaviour: real fringing-field resolution smears a narrow
        input IEDF, and the naive analysis reports the smeared width, not the true one."""
        distribution = _narrow_distribution(peak_energy_ev=200.0, relative_spread=0.005)
        energy_ev, gamma_ev = flux_iedf_ev(distribution, z_index=0, mass_kg=_argon().mass_kg)
        voltages = np.linspace(100.0, 300.0, 801)

        fine_geometry = _geometry(wire_spacing_m=1.0e-7, grid_separation_m=1.0e-2)
        coarse_geometry = _geometry(wire_spacing_m=2.0e-4, grid_separation_m=1.0e-3)

        fine_current = collected_current_a(
            voltages,
            energy_ev=energy_ev,
            gamma_ev=gamma_ev,
            geometry=fine_geometry,
            resolution_fwhm_ev=fine_geometry.resolution_fwhm_ev(),
        )
        coarse_current = collected_current_a(
            voltages,
            energy_ev=energy_ev,
            gamma_ev=gamma_ev,
            geometry=coarse_geometry,
            resolution_fwhm_ev=coarse_geometry.resolution_fwhm_ev(),
        )

        fine_estimate = estimate_iedf(voltages, fine_current, geometry=fine_geometry)
        coarse_estimate = estimate_iedf(voltages, coarse_current, geometry=coarse_geometry)

        def _fwhm(energy: np.ndarray, gamma: np.ndarray) -> float:
            half_max = 0.5 * float(np.max(gamma))
            above = energy[gamma >= half_max]
            return float(above.max() - above.min())

        assert _fwhm(coarse_estimate.energy_ev, coarse_estimate.gamma_ev) > _fwhm(
            fine_estimate.energy_ev, fine_estimate.gamma_ev
        )


class TestGridGeometryResolution:
    def test_a_wider_wire_spacing_gives_worse_resolution(self) -> None:
        narrow = _geometry(wire_spacing_m=1.0e-5, grid_separation_m=1.0e-3)
        wide = _geometry(wire_spacing_m=5.0e-4, grid_separation_m=1.0e-3)

        assert wide.resolution_fwhm_ev() > narrow.resolution_fwhm_ev()

    def test_a_larger_grid_separation_gives_better_resolution(self) -> None:
        close = _geometry(wire_spacing_m=1.0e-4, grid_separation_m=5.0e-4)
        far = _geometry(wire_spacing_m=1.0e-4, grid_separation_m=5.0e-3)

        assert far.resolution_fwhm_ev() < close.resolution_fwhm_ev()


class TestTheContract:
    def test_it_is_recognised_as_an_instrument(self) -> None:
        assert isinstance(RfeaInstrument(root_seed=1), Instrument)

    def test_forward_before_configure_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="configure"):
            RfeaInstrument(root_seed=1).forward(
                _state(distribution=_narrow_distribution(peak_energy_ev=200.0)), _WINDOW
            )

    def test_the_metadata_carries_a_citation_and_a_detection_floor(self) -> None:
        metadata = _configured().metadata()

        assert metadata.instrument_id == RFEA_INSTRUMENT_ID
        assert metadata.citations
        assert metadata.detection_floor.quantity == "n_0"


class TestForward:
    def test_it_returns_one_sample_per_sweep_point(self) -> None:
        state = _state(distribution=_narrow_distribution(peak_energy_ev=200.0))
        observable = _configured(sweep_points=61).forward(state, _WINDOW)

        assert observable.n_samples == 61
        assert observable.instrument_id == RFEA_INSTRUMENT_ID
        assert observable.units == CURRENT_UNITS

    def test_a_state_with_no_ion_distribution_is_refused(self) -> None:
        state = _state(distribution=_narrow_distribution(peak_energy_ev=200.0))
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

    def test_it_is_deterministic(self) -> None:
        state = _state(distribution=_narrow_distribution(peak_energy_ev=200.0))
        first = _configured().forward(state, _WINDOW)
        second = _configured().forward(state, _WINDOW)

        np.testing.assert_array_equal(first.values, second.values)


class TestObserveSharesTheForwardPath:
    def test_observe_reduces_to_forward_with_noise_off(self) -> None:
        state = _state(distribution=_narrow_distribution(peak_energy_ev=200.0))
        instrument = _configured(noise=False)

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        np.testing.assert_array_equal(observed.values, predicted.values)
        assert observed.as_observable() == predicted

    def test_observe_adds_noise_by_default(self) -> None:
        state = _state(distribution=_narrow_distribution(peak_energy_ev=200.0))
        instrument = _configured()

        predicted = instrument.forward(state, _WINDOW)
        observed = instrument.observe(state, _WINDOW)

        assert not np.array_equal(observed.values, predicted.values)
        assert np.all(observed.uncertainty > 0.0)


class TestCalibrate:
    def test_the_estimated_calibration_carries_uncertainty(self) -> None:
        calibration = _configured().calibrate(_references())

        assert calibration.state is CalibrationState.ESTIMATED
        assert any(value > 0.0 for value in calibration.relative_uncertainty.values())


class TestLikelihood:
    def test_a_perfect_prediction_is_the_most_probable_one(self) -> None:
        state = _state(distribution=_narrow_distribution(peak_energy_ev=200.0))
        instrument = _configured()
        observed = instrument.observe(state, _WINDOW)
        predicted = observed.as_observable()

        exact = instrument.likelihood(observed, predicted)
        shifted = type(predicted)(
            instrument_id=predicted.instrument_id,
            values=predicted.values + 1.0e-9,
            units=predicted.units,
            window=predicted.window,
        )
        offset = instrument.likelihood(observed, shifted)

        assert exact > offset


class TestTheDetectionGate:
    def test_a_dense_plasma_is_informative(self) -> None:
        assert _configured().is_informative(
            PlasmaParams(
                species=_argon(),
                n_0=Q_(1.0e16, "m**-3"),
                T_e=Q_(3.0, "eV"),
                T_i=Q_(0.05, "eV"),
                T_g=Q_(300.0, "K"),
                pressure=Q_(5.0, "mTorr"),
                bias=Q_(-200.0, "V"),
                gamma_se=0.1,
                kappa=1.0,
            )
        )

    def test_a_plasma_below_the_density_floor_is_not(self) -> None:
        assert not _configured().is_informative(
            PlasmaParams(
                species=_argon(),
                n_0=Q_(1.0e12, "m**-3"),
                T_e=Q_(3.0, "eV"),
                T_i=Q_(0.05, "eV"),
                T_g=Q_(300.0, "K"),
                pressure=Q_(5.0, "mTorr"),
                bias=Q_(-200.0, "V"),
                gamma_se=0.1,
                kappa=1.0,
            )
        )
