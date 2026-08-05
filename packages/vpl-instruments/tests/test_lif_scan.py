"""The detuning scan and the velocity-resolved signal — doc 04 §3.2, doc 04 V-22.

Doc 04 §3.2 is the whole point of the diagnostic: scanning ``nu_L`` maps out ``f_i(v_par)``.
So the decisive question is whether a scan of a *known* distribution returns that
distribution, and these tests answer it against references the module does not supply:

- an **analytic Voigt** from ``scipy.special.wofz``, which is what a weak-pump scan of a
  Maxwellian must equal (doc 04 V-22, "width within 1 %");
- a **bimodal** input whose two peaks must reappear at the two detunings the doc 04 §3.2
  mapping predicts, which checks the projection factor rather than assuming it;
- a **drifted** Maxwellian, whose line centre must move by ``nu_0 u sin(theta_L)/c``;
- the **doc 04 §3.3 broadening budget**, whose 734 MHz entry the model must reproduce.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import wofz

from vpl.core.constants import ELEMENTARY_CHARGE
from vpl.core.params import default_registry
from vpl.core.state import SpatialGrid, Species, VelocityDistribution
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.lif.laser import Laser, TuningRange
from vpl.instruments.lif.scan import (
    DetuningScan,
    doppler_detuning_hz,
    fluorescence_response,
    resonant_velocity_m_per_s,
    thermal_doppler_1e_halfwidth,
    tuning_coverage,
)
from vpl.instruments.lif.transition import ProbeTransition
from vpl.instruments.lif.zeeman import (
    PumpPolarisation,
    ZeemanComponent,
    unsplit_pattern,
    zeeman_pattern,
)

_E_C = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_WEAK_PUMP = 1.0e-6


def _argon() -> Species:
    return Species(name="Ar+", mass=default_registry().quantity("species.Ar.mass"), charge_number=1)


def _thermal_speed(*, temperature_ev: float) -> float:
    """``sqrt(k T / m)`` — the standard deviation of a Maxwellian in one velocity component."""
    return math.sqrt(temperature_ev * _E_C / _argon().mass_kg)


def _maxwellian(
    *, temperature_ev: float, drift: float = 0.0, density: float = 1.0e17, n_v: int = 2001
) -> VelocityDistribution:
    sigma = _thermal_speed(temperature_ev=temperature_ev)
    velocity = np.linspace(drift - 8.0 * sigma, drift + 8.0 * sigma, n_v)
    values = (
        density
        / (math.sqrt(2.0 * math.pi) * sigma)
        * np.exp(-((velocity - drift) ** 2) / (2.0 * sigma**2))
    )
    return VelocityDistribution(
        grid=SpatialGrid(z_m=np.array([0.0, 1.0e-3])),
        v_m_per_s=velocity,
        values=np.vstack([values, values]),
        species=_argon(),
    )


def _bimodal(
    *, speeds: tuple[float, float], width: float, n_v: int = 30001
) -> VelocityDistribution:
    velocity = np.linspace(-3.0e4, 3.0e4, n_v)
    values = np.zeros_like(velocity)
    for centre in speeds:
        values += np.exp(-((velocity - centre) ** 2) / (2.0 * width**2))
    return VelocityDistribution(
        grid=SpatialGrid(z_m=np.array([0.0, 1.0e-3])),
        v_m_per_s=velocity,
        values=np.vstack([values, values]),
        species=_argon(),
    )


def _analytic_voigt(
    detuning_hz: NDArray[np.float64], *, gaussian_sigma_hz: float, lorentzian_fwhm_hz: float
) -> NDArray[np.float64]:
    """The Voigt profile, from the Faddeeva function — independent of the module tested."""
    z = (detuning_hz + 1j * lorentzian_fwhm_hz / 2.0) / (gaussian_sigma_hz * math.sqrt(2.0))
    return np.asarray(
        np.real(wofz(z)) / (gaussian_sigma_hz * math.sqrt(2.0 * math.pi)), dtype=np.float64
    )


def _fwhm_of(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Full width at half maximum of a sampled, single-peaked profile, by interpolation."""
    half = y.max() / 2.0
    above = np.flatnonzero(y >= half)
    first, last = above[0], above[-1]
    left = np.interp(half, [y[first - 1], y[first]], [x[first - 1], x[first]])
    right = np.interp(half, [y[last + 1], y[last]], [x[last + 1], x[last]])
    return float(right - left)


# ── the Doppler mapping (doc 04 §3.2) ───────────────────────────────────────────


class TestTheDopplerMapping:
    def test_the_mapping_inverts_itself(self) -> None:
        transition = ProbeTransition.from_registry()
        laser = Laser.from_registry()
        velocity = np.linspace(-2.0e4, 2.0e4, 17)

        detuning = doppler_detuning_hz(
            velocity,
            pump_frequency_hz=transition.pump_frequency_hz,
            projection_factor=laser.projection_factor,
        )
        recovered = resonant_velocity_m_per_s(
            detuning,
            pump_frequency_hz=transition.pump_frequency_hz,
            projection_factor=laser.projection_factor,
        )

        np.testing.assert_allclose(recovered, velocity, rtol=1e-12, atol=1e-9)

    def test_the_shift_is_the_projected_velocity_over_the_wavelength(self) -> None:
        transition = ProbeTransition.from_registry()
        laser = Laser.from_registry()

        detuning = doppler_detuning_hz(
            np.array([1.0e4]),
            pump_frequency_hz=transition.pump_frequency_hz,
            projection_factor=laser.projection_factor,
        )

        expected = 1.0e4 * math.sin(math.radians(15.0)) / 668.614e-9

        assert float(detuning[0]) == pytest.approx(expected, rel=1e-9)

    def test_a_beam_at_normal_incidence_would_see_the_full_velocity(self) -> None:
        transition = ProbeTransition.from_registry()

        detuning = doppler_detuning_hz(
            np.array([3.47e4]),
            pump_frequency_hz=transition.pump_frequency_hz,
            projection_factor=1.0,
        )

        # doc 01 §5.1: "A 250 eV Ar+ ion travels at 3.47e4 m/s, giving a Doppler shift of
        # 51.9 GHz at 668.6 nm."
        assert float(detuning[0]) / 1.0e9 == pytest.approx(51.9, rel=1e-3)


class TestTheDopplerWidth:
    def test_the_1_over_e_half_width_reproduces_the_doc_04_broadening_budget(self) -> None:
        """Doc 04 §3.3 tabulates "~734 MHz" at ``T_i`` = 0.05 eV.

        734 MHz is the 1/e half-width ``(nu_0/c) sqrt(2 k T/m)``, **not** the FWHM the
        table calls it: the FWHM is ``sqrt(4 ln 2)`` = 1.665 times larger, 1224 MHz. Both
        are asserted here so the mislabelling in the document cannot propagate into a
        resolution requirement that is 40 % too loose.
        """
        width = thermal_doppler_1e_halfwidth(
            temperature=Q_(0.05, "eV"),
            species=_argon(),
            transition=ProbeTransition.from_registry(),
            projection_factor=1.0,
        )

        assert magnitude_in(width, "MHz") == pytest.approx(734.0, rel=0.01)

    def test_the_corresponding_fwhm_is_1224_megahertz(self) -> None:
        width_mhz = float(
            magnitude_in(
                thermal_doppler_1e_halfwidth(
                    temperature=Q_(0.05, "eV"),
                    species=_argon(),
                    transition=ProbeTransition.from_registry(),
                    projection_factor=1.0,
                ),
                "MHz",
            )
        )

        assert width_mhz * math.sqrt(4.0 * math.log(2.0)) == pytest.approx(1224.0, rel=0.01)

    def test_the_grazing_geometry_shrinks_the_observed_width(self) -> None:
        laser = Laser.from_registry()
        transition = ProbeTransition.from_registry()

        projected = thermal_doppler_1e_halfwidth(
            temperature=Q_(0.05, "eV"),
            species=_argon(),
            transition=transition,
            projection_factor=laser.projection_factor,
        )

        assert magnitude_in(projected, "MHz") == pytest.approx(734.0 * 0.2588, rel=0.01)


# ── the scan and the tuning-range limit ─────────────────────────────────────────


class TestDetuningScan:
    def test_a_uniform_scan_spans_the_mode_hop_free_range(self) -> None:
        scan = DetuningScan.uniform(tuning=TuningRange.from_registry(), n_points=201)

        assert magnitude_in(scan.detuning, "GHz").min() == pytest.approx(-10.0)
        assert magnitude_in(scan.detuning, "GHz").max() == pytest.approx(10.0)
        assert scan.n_points == 201

    def test_a_scan_beyond_the_tuning_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mode-hop-free"):
            DetuningScan.uniform(
                tuning=TuningRange.from_registry(), n_points=51, half_span=Q_(30.0, "GHz")
            )

    def test_the_scan_resolves_the_documented_velocity_window(self) -> None:
        scan = DetuningScan.uniform(tuning=TuningRange.from_registry(), n_points=201)

        velocities = scan.velocities(
            transition=ProbeTransition.from_registry(), laser=Laser.from_registry()
        )

        assert magnitude_in(velocities, "km/s").max() == pytest.approx(25.8, rel=0.01)

    def test_a_scan_of_fewer_than_two_points_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            DetuningScan.uniform(tuning=TuningRange.from_registry(), n_points=1)


class TestTuningCoverage:
    def test_a_cold_sheath_edge_distribution_is_fully_visible(self) -> None:
        distribution = _maxwellian(temperature_ev=0.05, drift=-2.7e3)

        coverage = tuning_coverage(
            distribution=distribution,
            z_index=0,
            tuning=TuningRange.from_registry(),
            transition=ProbeTransition.from_registry(),
            laser=Laser.from_registry(),
        )

        assert coverage.visible_fraction == pytest.approx(1.0, abs=1e-6)
        assert not coverage.is_truncated

    def test_a_fast_beam_inside_the_sheath_is_mostly_invisible(self) -> None:
        """The doc 01 §5.1 limitation, as a number rather than as a caveat.

        A 250 eV beam sits at 34.7 km/s, above the 25.8 km/s ceiling, so most of it never
        comes into resonance anywhere in the scan.
        """
        distribution = _maxwellian(temperature_ev=0.5, drift=-3.47e4)

        coverage = tuning_coverage(
            distribution=distribution,
            z_index=0,
            tuning=TuningRange.from_registry(),
            transition=ProbeTransition.from_registry(),
            laser=Laser.from_registry(),
        )

        assert coverage.is_truncated
        assert coverage.visible_fraction < 0.1

    def test_the_coverage_reports_the_energy_ceiling(self) -> None:
        coverage = tuning_coverage(
            distribution=_maxwellian(temperature_ev=0.05),
            z_index=0,
            tuning=TuningRange.from_registry(),
            transition=ProbeTransition.from_registry(),
            laser=Laser.from_registry(),
        )

        assert magnitude_in(coverage.energy_ceiling, "eV") == pytest.approx(138.0, rel=0.02)


# ── the signal itself ───────────────────────────────────────────────────────────


def _weak_pump_scan(
    *,
    distribution: VelocityDistribution,
    half_span_ghz: float = 6.0,
    n_points: int = 1201,
    components: tuple[ZeemanComponent, ...] | None = None,
    saturation: float = _WEAK_PUMP,
    transition: ProbeTransition | None = None,
    laser: Laser | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    transition = transition if transition is not None else ProbeTransition.from_registry()
    laser = laser if laser is not None else Laser.from_registry()
    scan = DetuningScan.uniform(
        tuning=TuningRange.from_registry(), n_points=n_points, half_span=Q_(half_span_ghz, "GHz")
    )
    signal = fluorescence_response(
        distribution=distribution,
        z_index=0,
        scan=scan,
        transition=transition,
        laser=laser,
        components=components if components is not None else unsplit_pattern(),
        saturation=saturation,
    )
    return np.asarray(magnitude_in(scan.detuning, "Hz")), signal


class TestTheWeakPumpLineshape:
    @pytest.mark.physics
    def test_a_maxwellian_returns_the_analytic_voigt(self) -> None:
        """Doc 04 V-22: "LIF lineshape vs analytic Voigt, low saturation | width within 1 %"."""
        temperature_ev = 0.05
        transition = ProbeTransition.from_registry()
        laser = Laser.from_registry()

        detuning, signal = _weak_pump_scan(distribution=_maxwellian(temperature_ev=temperature_ev))

        sigma_v = _thermal_speed(temperature_ev=temperature_ev)
        gaussian_sigma_hz = (
            transition.pump_frequency_hz * sigma_v * laser.projection_factor / 2.99792458e8
        )
        lorentzian_fwhm_hz = (
            float(magnitude_in(transition.homogeneous_fwhm, "Hz")) + laser.linewidth_hz
        )
        voigt = _analytic_voigt(
            detuning,
            gaussian_sigma_hz=gaussian_sigma_hz,
            lorentzian_fwhm_hz=lorentzian_fwhm_hz,
        )

        np.testing.assert_allclose(signal / signal.max(), voigt / voigt.max(), rtol=2e-3, atol=2e-4)

    @pytest.mark.physics
    def test_the_recovered_width_matches_the_voigt_width_to_one_part_in_a_thousand(self) -> None:
        temperature_ev = 0.05
        transition = ProbeTransition.from_registry()
        laser = Laser.from_registry()

        detuning, signal = _weak_pump_scan(distribution=_maxwellian(temperature_ev=temperature_ev))

        sigma_v = _thermal_speed(temperature_ev=temperature_ev)
        gaussian_sigma_hz = (
            transition.pump_frequency_hz * sigma_v * laser.projection_factor / 2.99792458e8
        )
        voigt = _analytic_voigt(
            detuning,
            gaussian_sigma_hz=gaussian_sigma_hz,
            lorentzian_fwhm_hz=float(magnitude_in(transition.homogeneous_fwhm, "Hz"))
            + laser.linewidth_hz,
        )

        assert _fwhm_of(detuning, signal) == pytest.approx(_fwhm_of(detuning, voigt), rel=1e-3)

    @pytest.mark.physics
    def test_the_signal_is_linear_in_laser_power(self) -> None:
        """The weak-pump requirement of doc 04 §3.4, at the level of the whole scan."""
        distribution = _maxwellian(temperature_ev=0.05)

        _, weak = _weak_pump_scan(distribution=distribution, saturation=1.0e-8, n_points=201)
        _, weaker = _weak_pump_scan(distribution=distribution, saturation=5.0e-9, n_points=201)

        np.testing.assert_allclose(weak, 2.0 * weaker, rtol=1e-7)

    @pytest.mark.physics
    def test_the_scan_recovers_a_known_distribution_up_to_the_homogeneous_width(self) -> None:
        """The claim of doc 04 §3.2, stated with the qualification it needs.

        A detuning scan does **not** return ``f_i`` — it returns ``f_i`` convolved with the
        homogeneous response, which is why doc 04 V-22's reference is a Voigt. The
        recovered 1/e width is therefore larger than the input thermal speed by a known,
        computable amount, and the recovery becomes exact only as the homogeneous width
        goes to zero. That limit is what is tested here: shrink the homogeneous width and
        the recovered width must converge on the input, monotonically.

        Nothing about this is a modelling choice to be corrected later. It is what the
        instrument does, and the framework's answer is doc 04 §6.2's: apply the same
        broadening inside the inverse model rather than deconvolving the data.
        """
        temperature_ev = 0.05
        # Fine in velocity, so the shrinking homogeneous width stays resolved by the
        # quadrature. Under-resolving it is a real failure mode and has its own guard and
        # its own test below; conflating the two here would hide both.
        distribution = _maxwellian(temperature_ev=temperature_ev, n_v=12001)
        base = ProbeTransition.from_registry()
        input_sigma = _thermal_speed(temperature_ev=temperature_ev)

        ratios = []
        for natural_mhz in (20.0, 6.0, 2.0):
            transition = replace(
                base,
                natural_linewidth=Q_(natural_mhz, "MHz"),
                pressure_broadening=Q_(0.0, "MHz"),
            )
            laser = replace(Laser.from_registry(), linewidth=Q_(0.0, "MHz"))
            detuning, signal = _weak_pump_scan(
                distribution=distribution,
                transition=transition,
                laser=laser,
                half_span_ghz=1.5,
                n_points=601,
            )
            velocity = resonant_velocity_m_per_s(
                detuning,
                pump_frequency_hz=transition.pump_frequency_hz,
                projection_factor=laser.projection_factor,
            )
            recovered = _fwhm_of(velocity, signal) / math.sqrt(8.0 * math.log(2.0))
            ratios.append(recovered / input_sigma)

        assert ratios[0] > ratios[1] > ratios[2]
        assert ratios[2] == pytest.approx(1.0, abs=5e-3)

    @pytest.mark.physics
    def test_the_recovered_width_is_the_voigt_width_of_the_input(self) -> None:
        """The quantitative form of the same statement, at the nominal homogeneous width.

        Olivero and Longbothum's closed form for the Voigt FWHM is accurate to 0.02 % and
        is written out here rather than imported: the recovered width must equal it, so
        the 2.7 % excess over the input Gaussian is an accounted-for instrumental
        broadening rather than an unexplained discrepancy.
        """
        temperature_ev = 0.05
        transition = ProbeTransition.from_registry()
        laser = Laser.from_registry()

        detuning, signal = _weak_pump_scan(
            distribution=_maxwellian(temperature_ev=temperature_ev),
            half_span_ghz=2.0,
            n_points=2001,
        )

        gaussian_fwhm = math.sqrt(8.0 * math.log(2.0)) * _thermal_speed(
            temperature_ev=temperature_ev
        )
        lorentzian_fwhm_hz = (
            float(magnitude_in(transition.homogeneous_fwhm, "Hz")) + laser.linewidth_hz
        )
        lorentzian_fwhm = float(
            resonant_velocity_m_per_s(
                np.array([lorentzian_fwhm_hz]),
                pump_frequency_hz=transition.pump_frequency_hz,
                projection_factor=laser.projection_factor,
            )[0]
        )
        expected = 0.5346 * lorentzian_fwhm + math.sqrt(
            0.2166 * lorentzian_fwhm**2 + gaussian_fwhm**2
        )

        velocity = resonant_velocity_m_per_s(
            detuning,
            pump_frequency_hz=transition.pump_frequency_hz,
            projection_factor=laser.projection_factor,
        )

        assert _fwhm_of(velocity, signal) == pytest.approx(expected, rel=2e-3)

    @pytest.mark.physics
    def test_a_bimodal_distribution_appears_as_two_peaks_at_the_predicted_detunings(self) -> None:
        """Checks the projection factor rather than assuming it: the two peaks land where
        ``nu_0 v sin(theta_L)/c`` puts them, 3.86x closer together than an unprojected
        beam would place them."""
        speeds = (-1.0e4, 6.0e3)
        transition = ProbeTransition.from_registry()
        laser = Laser.from_registry()

        detuning, signal = _weak_pump_scan(
            distribution=_bimodal(speeds=speeds, width=6.0e2), half_span_ghz=8.0, n_points=4001
        )

        peaks = sorted(
            float(detuning[index])
            for index in range(1, signal.size - 1)
            if signal[index] > signal[index - 1] and signal[index] > signal[index + 1]
        )
        expected = sorted(
            doppler_detuning_hz(
                np.array(speeds),
                pump_frequency_hz=transition.pump_frequency_hz,
                projection_factor=laser.projection_factor,
            ).tolist()
        )

        assert len(peaks) == 2
        np.testing.assert_allclose(peaks, expected, rtol=2e-3)

    @pytest.mark.physics
    def test_a_drifting_population_shifts_the_line_centre(self) -> None:
        drift = -4.0e3
        transition = ProbeTransition.from_registry()
        laser = Laser.from_registry()

        detuning, signal = _weak_pump_scan(
            distribution=_maxwellian(temperature_ev=0.05, drift=drift), n_points=4001
        )

        centroid = float(np.trapezoid(detuning * signal, detuning) / np.trapezoid(signal, detuning))
        expected = float(
            doppler_detuning_hz(
                np.array([drift]),
                pump_frequency_hz=transition.pump_frequency_hz,
                projection_factor=laser.projection_factor,
            )[0]
        )

        assert centroid == pytest.approx(expected, rel=2e-3)

    @pytest.mark.physics
    def test_the_signal_is_proportional_to_density(self) -> None:
        _, dense = _weak_pump_scan(
            distribution=_maxwellian(temperature_ev=0.05, density=2.0e17), n_points=201
        )
        _, sparse = _weak_pump_scan(
            distribution=_maxwellian(temperature_ev=0.05, density=1.0e17), n_points=201
        )

        np.testing.assert_allclose(dense, 2.0 * sparse, rtol=1e-12)


class TestSaturationAcrossTheScan:
    @pytest.mark.physics
    def test_saturation_broadens_the_measured_lineshape(self) -> None:
        """Doc 04 §3.4's distortion, at the level of the observed IVDF.

        The saturated homogeneous width is ``Gamma sqrt(1+S)``, so at ``S = 1e4`` the
        16 MHz homogeneous line becomes 1.6 GHz and swamps the 190 MHz projected Doppler
        width. A model that applied one scalar saturation factor to the whole line would
        show no broadening at all, which is why this is the test that separates the two.
        """
        distribution = _maxwellian(temperature_ev=0.05)

        detuning, linear = _weak_pump_scan(distribution=distribution, n_points=2001)
        _, saturated = _weak_pump_scan(distribution=distribution, n_points=2001, saturation=1.0e4)

        assert _fwhm_of(detuning, saturated) > 4.0 * _fwhm_of(detuning, linear)

    @pytest.mark.physics
    def test_the_saturated_signal_is_sublinear_in_power(self) -> None:
        distribution = _maxwellian(temperature_ev=0.05)

        _, low = _weak_pump_scan(distribution=distribution, n_points=201, saturation=10.0)
        _, high = _weak_pump_scan(distribution=distribution, n_points=201, saturation=20.0)

        peak_ratio = float(high.max() / low.max())

        assert 1.0 < peak_ratio < 2.0

    @pytest.mark.physics
    def test_saturation_biases_a_naive_temperature_upward(self) -> None:
        """The consequence a user has to be warned about, quantified.

        Fitting a Gaussian width to a saturated scan and calling it ``T_i`` overstates the
        temperature, because power broadening is instrumental. The bias is asserted here
        so the framework cannot silently stop modelling it.
        """
        distribution = _maxwellian(temperature_ev=0.05)

        detuning, linear = _weak_pump_scan(distribution=distribution, n_points=2001)
        _, saturated = _weak_pump_scan(distribution=distribution, n_points=2001, saturation=100.0)

        apparent_temperature_ratio = (
            _fwhm_of(detuning, saturated) / _fwhm_of(detuning, linear)
        ) ** 2

        assert apparent_temperature_ratio > 1.05


class TestZeemanInTheScan:
    @pytest.mark.physics
    def test_a_zero_field_pattern_reproduces_the_unsplit_scan_exactly(self) -> None:
        """The ``B -> 0`` limit must be exact, not approximate — the weights sum to one."""
        transition = ProbeTransition.from_registry()
        distribution = _maxwellian(temperature_ev=0.05)

        _, unsplit = _weak_pump_scan(distribution=distribution, n_points=401)
        _, zero_field = _weak_pump_scan(
            distribution=distribution,
            n_points=401,
            components=zeeman_pattern(
                lower=transition.lower,
                upper=transition.upper,
                magnetic_field=Q_(0.0, "mT"),
                polarisation=PumpPolarisation.ISOTROPIC,
            ),
        )

        np.testing.assert_allclose(zero_field, unsplit, rtol=1e-12)

    @pytest.mark.physics
    def test_the_field_broadens_the_line_and_leaves_its_centre_alone(self) -> None:
        """Doc 04 §3.3: ignoring the pattern "would bias the inferred ion temperature and
        distort the IVDF shape" — a width effect, not a shift."""
        transition = ProbeTransition.from_registry()
        distribution = _maxwellian(temperature_ev=0.05)
        split = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=Q_(5.0, "mT"),
            polarisation=PumpPolarisation.ISOTROPIC,
        )

        detuning, unsplit = _weak_pump_scan(distribution=distribution, n_points=4001)
        _, with_field = _weak_pump_scan(distribution=distribution, n_points=4001, components=split)

        centroid = float(
            np.trapezoid(detuning * with_field, detuning) / np.trapezoid(with_field, detuning)
        )

        assert _fwhm_of(detuning, with_field) > _fwhm_of(detuning, unsplit)
        assert abs(centroid) < 2.0e6

    @pytest.mark.physics
    def test_the_temperature_bias_from_ignoring_zeeman_is_the_documented_size(self) -> None:
        """Doc 04 §3.3 puts the splitting at 10 % of the Doppler width.

        A width that grows in quadrature by 10 % of itself is a 1 % width increase and
        therefore a ~2 % ``T_i`` bias at the *unprojected* Doppler width. Under the doc 02
        §4.2 grazing geometry the observed Doppler width is 3.86x *narrower* while the
        Zeeman comb is unchanged, so the bias is far larger — the number asserted here.
        Stating it is the point: the projection makes the Zeeman correction more important,
        not less, and doc 04 §3.3's "10 %" is quoted against the unprojected width.
        """
        transition = ProbeTransition.from_registry()
        distribution = _maxwellian(temperature_ev=0.05)
        split = zeeman_pattern(
            lower=transition.lower,
            upper=transition.upper,
            magnetic_field=Q_(5.0, "mT"),
            polarisation=PumpPolarisation.ISOTROPIC,
        )

        detuning, unsplit = _weak_pump_scan(distribution=distribution, n_points=4001)
        _, with_field = _weak_pump_scan(distribution=distribution, n_points=4001, components=split)

        bias = (_fwhm_of(detuning, with_field) / _fwhm_of(detuning, unsplit)) ** 2 - 1.0

        assert bias > 0.05

    @pytest.mark.physics
    def test_pi_and_sigma_pumping_separate_the_components_by_shift_not_by_width(self) -> None:
        """The extra observable of doc 04 §3.3, and what it actually is.

        A sigma-pumped scan is *displaced* by the centre of gravity of its sub-comb —
        ``mu_B B g_u / h`` = 75 MHz at 50 G — while a pi-pumped scan stays on line centre.
        Their **widths** are nearly identical (the sigma comb's internal spread, 12.1 MHz,
        is if anything narrower than the pi comb's 14.0 MHz), so the discriminating
        observable is the shift and not the broadening. Worth pinning, because "sigma
        splits more" is the intuitive and wrong reading of the pattern.
        """
        transition = ProbeTransition.from_registry()
        distribution = _maxwellian(temperature_ev=0.05)

        centroids: list[float] = []
        widths: list[float] = []
        for polarisation in (PumpPolarisation.PI, PumpPolarisation.SIGMA_PLUS):
            detuning, signal = _weak_pump_scan(
                distribution=distribution,
                n_points=4001,
                components=zeeman_pattern(
                    lower=transition.lower,
                    upper=transition.upper,
                    magnetic_field=Q_(5.0, "mT"),
                    polarisation=polarisation,
                ),
            )
            centroids.append(
                float(np.trapezoid(detuning * signal, detuning) / np.trapezoid(signal, detuning))
            )
            widths.append(_fwhm_of(detuning, signal))

        assert abs(centroids[0]) < 2.0e6
        assert centroids[1] / 1.0e6 == pytest.approx(74.98, rel=1e-2)
        assert widths[1] / widths[0] == pytest.approx(1.0, abs=0.02)


class TestGuards:
    def test_an_out_of_range_grid_index_is_refused(self) -> None:
        transition = ProbeTransition.from_registry()

        with pytest.raises(IndexError, match="grid"):
            fluorescence_response(
                distribution=_maxwellian(temperature_ev=0.05),
                z_index=7,
                scan=DetuningScan.uniform(tuning=TuningRange.from_registry(), n_points=11),
                transition=transition,
                laser=Laser.from_registry(),
                components=unsplit_pattern(),
                saturation=1.0e-6,
            )

    def test_an_under_resolved_velocity_grid_is_refused(self) -> None:
        """The failure mode found while verifying the narrow-linewidth limit.

        If the velocity grid is coarser than the homogeneous resonance width, the
        trapezoid over ``v`` steps straight over the Lorentzian and returns a signal that
        is smooth, positive, plausible and several times too small. Nothing downstream
        would notice, so it is refused here.
        """
        coarse = VelocityDistribution(
            grid=SpatialGrid(z_m=np.array([0.0, 1.0e-3])),
            v_m_per_s=np.linspace(-3.0e3, 3.0e3, 31),
            values=np.ones((2, 31)),
            species=_argon(),
        )

        with pytest.raises(ValueError, match="step over the resonance"):
            fluorescence_response(
                distribution=coarse,
                z_index=0,
                scan=DetuningScan.uniform(tuning=TuningRange.from_registry(), n_points=11),
                transition=replace(
                    ProbeTransition.from_registry(),
                    natural_linewidth=Q_(0.1, "MHz"),
                    pressure_broadening=Q_(0.0, "MHz"),
                ),
                laser=replace(Laser.from_registry(), linewidth=Q_(0.0, "MHz")),
                components=unsplit_pattern(),
                saturation=1.0e-6,
            )

    def test_an_empty_component_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="component"):
            fluorescence_response(
                distribution=_maxwellian(temperature_ev=0.05),
                z_index=0,
                scan=DetuningScan.uniform(tuning=TuningRange.from_registry(), n_points=11),
                transition=ProbeTransition.from_registry(),
                laser=Laser.from_registry(),
                components=(),
                saturation=1.0e-6,
            )
