"""``OesInstrument`` — the doc 08 §4 contract over the OES channel.

Doc 04 §9 states the contract and the one property that makes it worth having:

    ``observe`` returns raw data with noise and imperfect calibration; ``forward`` returns
    the noiseless expectation used by the likelihood. **Both come from the same code
    path**, with noise and calibration error as switchable stages, which guarantees they
    cannot drift apart — a class of bug that would silently invalidate every result.

So there is one private method, :meth:`OesInstrument._sample`, and both public methods are
one call to it with the stages set differently. ``test_instrument.py`` asserts they agree
*exactly* with the stages off; anything less than exact equality means two code paths.

## The chain, per spatial point

1. The EEDF at the local ``T_e`` — from an :class:`EedfProvider`, which is a **required**
   constructor argument. There is no default, and that is doc 04 §2.2's doing: the rate
   coefficients must come from the Boltzmann solver's distribution, "not a Maxwellian", so
   a Maxwellian default would be the wrong answer arriving silently.
   :class:`MaxwellianEedf` exists and is clearly labelled as the verification fallback.
2. The CR solve of :mod:`vpl.instruments.oes.cr` for the upper-state populations.
3. Doc 04 §2.1's emissivity, times the chord of
   :func:`~vpl.instruments.oes.emissivity.chord_radiance`.
4. The spectrograph of :mod:`vpl.instruments.oes.spectrograph`, which puts each line on the
   detector axis with its Doppler width and the instrument function.

## The noise, and what is not here

**Only N1.** Doc 04 §7.2 enumerates eighteen noise sources and every one of N2-N18 belongs
to ``vpl-detectors``, because doc 04 §1's rule is that "``F4`` never sees the plasma" and
this module does. What is applied here is photon shot noise on the photoelectron count —
the boundary between ``F2`` and ``F3``, and the only source that is a property of the light
rather than of the camera.

The conversion to photoelectrons uses a thin-lens etendue from ``OES-S4.slit_width``,
``OES.slit_height`` and ``OES-S1.f_number``. Doc 04 §6.3 buys Raysect for the real thing
and doc 04 §6.2 is emphatic that the measurement volume is an integral rather than a point;
this is a stand-in whose error is a **scale factor on the noise level and not on the
signal**, because the signal is returned in radiance units and only the shot-noise
magnitude depends on how many photons a real instrument would have caught. That is the
right place for the approximation to sit, and it is why it is tolerable here.

## Calibration — doc 04 §7.3

``calibrate`` draws the *estimated* radiometric response once, from the doc 02 §11 6 %
standard, on the :attr:`~vpl.core.random.Stream.CALIBRATION` stream. It is drawn once per
instrument and never re-drawn, because doc 06 §4.1 says a correlated calibration error
"affects *every* Thomson point identically and does **not** average down" — the same is
true here, and re-drawing it per observation would make it average away and understate the
budget.

:meth:`use_true_calibration` exists for the deliberate verification runs doc 04 §7.3
permits, and the resulting :class:`~vpl.core.state.CalibrationState` says so in the
artifact.

## What is deliberately not modelled

- **One spectral setting.** The detector spans about 8 nm at 1800 gr/mm, so the ten lines
  of doc 02 §6.3 cannot be recorded together and this instrument sees one setting at a
  time. A real campaign steps the grating; a manifest models that as several configured
  instruments.
- **No spatial point-spread function.** Each grid point becomes an independent spectrum.
  The OES-S5 magnification is registered but the row-to-row cross-talk a real imaging
  spectrograph has is not applied, so the modelled spatial resolution is the grid's rather
  than the instrument's — which flatters the sheath reconstruction, and is the most
  optimistic simplification in this module.
- **No time dependence inside the CR solve.** The populations are quasi-static (doc 04
  §2.2), and a window spanning several samples is handled by averaging the *radiance* over
  them rather than the fields, which is the right order but still assumes the levels
  respond instantaneously. See :mod:`vpl.instruments.oes.cr` for the bound; for the
  metastables it does not hold.
"""

from __future__ import annotations

import math
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import special

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.protocols.config import InstrumentConfig
from vpl.core.protocols.instrument import Calibration, CalibrationSet, LogProb
from vpl.core.protocols.metadata import Citation, DetectionFloor, InstrumentMetadata
from vpl.core.random import Stream, generator
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    InstrumentId,
    Measurement,
    Observable,
    PlasmaParams,
    PlasmaState,
)
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.oes.cr import CollisionalRadiativeModel
from vpl.instruments.oes.emissivity import chord_radiance, emission_spectrum
from vpl.instruments.oes.lineshape import doppler_fwhm_nm
from vpl.instruments.oes.spectrograph import Spectrograph
from vpl.physics.eedf.analytic import maxwellian_eedf
from vpl.physics.eedf.grid import EnergyGrid

__all__ = [
    "RADIANCE_UNITS",
    "EedfProvider",
    "MaxwellianEedf",
    "OesInstrument",
]

type FloatArray = NDArray[np.float64]

#: Units of everything this instrument returns. Spectral radiance, as doc 04 §1 defines the
#: output of ``F2``: "spectral radiance L(lambda, r, Omega, t) [W m^-2 sr^-1 nm^-1]".
RADIANCE_UNITS: Final[str] = "W / m**2 / sr / nm"

#: The default channel key, and the one doc 08 §6's manifest uses.
_DEFAULT_INSTRUMENT_ID: Final[InstrumentId] = "oes"

#: Smallest photoelectron count the Gaussian branch of the likelihood will divide by. A
#: pixel that expects zero photons still receives none with probability one, and a
#: zero-variance Gaussian there would return ``-inf`` for a perfectly ordinary observation.
_MINIMUM_VARIANCE_COUNTS: Final[float] = 1.0


class EedfProvider(Protocol):
    """Supplies the EEDF the CR rate coefficients are integrated over — doc 04 §2.2.

    Structural rather than nominal, so that a tabulated
    :class:`~vpl.physics.eedf.tabulate.RateTable` sweep, a live
    :class:`~vpl.physics.eedf.solver.TwoTermSolver` or a stored L2 distribution can each be
    plugged in without this module knowing which.
    """

    def f0(self, *, electron_temperature_ev: float) -> FloatArray:
        """The isotropic EEDF at cell centres, normalised by ``integral f0 sqrt(eps) deps``."""
        ...


class MaxwellianEedf:
    """A Maxwellian at the local ``T_e`` — **the verification fallback, not the default**.

    Doc 04 §2.2 requires the rate coefficients to come from the Boltzmann solver's
    distribution "not a Maxwellian", and doc 03 §3.2 states that the argon EEDF at RP-1 is
    Druyvesteyn-like with a depleted tail — which is the part that drives the excitation.
    Using this in anger over-estimates every excitation rate coefficient, and the error
    grows with the threshold, so it hits the ``T_e``-sensitive lines of doc 02 §6.3 hardest.

    It exists because the two analytic limits of doc 04 §8 V-25 are stated for a Maxwellian
    and cannot be checked without one.
    """

    __slots__ = ("_cache", "grid")

    def __init__(self, *, grid: EnergyGrid) -> None:
        self.grid = grid
        self._cache: dict[float, FloatArray] = {}

    def f0(self, *, electron_temperature_ev: float) -> FloatArray:
        if not electron_temperature_ev > 0.0:
            raise ValueError(f"T_e must be positive, got {electron_temperature_ev} eV")
        cached = self._cache.get(electron_temperature_ev)
        if cached is None:
            cached = self.grid.normalise(
                maxwellian_eedf(
                    self.grid.centres_ev, electron_temperature_ev=electron_temperature_ev
                )
            )
            self._cache[electron_temperature_ev] = cached
        return cached

    def __repr__(self) -> str:
        return f"MaxwellianEedf({self.grid!r})"


class OesInstrument:
    """The optical-emission channel — doc 02 §6, doc 04 §2 and §5, doc 08 §4.

    Mutable, because :meth:`configure` and :meth:`calibrate` are declared as mutating in
    doc 08 §4 and returning a new instrument from them would leave the manifest and the
    object it configured pointing at different things.
    """

    def __init__(
        self,
        *,
        model: CollisionalRadiativeModel,
        spectrograph: Spectrograph,
        eedf: EedfProvider,
        centre_wavelength_nm: float,
        root_seed: int,
        chord_length_m: float | None = None,
        accumulations: int = 1,
        instrument_id: InstrumentId = _DEFAULT_INSTRUMENT_ID,
        registry: ParameterRegistry | None = None,
    ) -> None:
        self._registry = registry if registry is not None else default_registry()
        self.model = model
        self.spectrograph = spectrograph
        self.eedf = eedf
        self.instrument_id = instrument_id

        self.chord_length_m = (
            chord_length_m
            if chord_length_m is not None
            else float(self._registry.value_in("OES.chamber_diameter", "m"))
        )
        self.accumulations = accumulations
        self.centre_wavelength_nm = centre_wavelength_nm

        self._noise_enabled = True
        self._calibration: Calibration | None = None
        self._applied_state = CalibrationState.ESTIMATED
        self._rng = generator(root_seed, Stream.PHOTONS)
        self._calibration_rng = generator(root_seed, Stream.CALIBRATION)

    # ── configuration ───────────────────────────────────────────────────────────

    @property
    def centre_wavelength_nm(self) -> float:
        """Where the detector is centred. Setting it re-tunes the grating."""
        return self._centre_wavelength_nm

    @centre_wavelength_nm.setter
    def centre_wavelength_nm(self, value: float) -> None:
        # Validate by tuning: the grating raises if the wavelength is unreachable, and
        # discovering that at the first `forward` rather than at configuration time would
        # put the failure a long way from the manifest line that caused it.
        self.spectrograph.reciprocal_dispersion_nm_per_mm(value)
        self._centre_wavelength_nm = value

    def configure(self, cfg: InstrumentConfig) -> None:
        """Apply one entry of the manifest's ``instruments:`` list — doc 08 §6."""
        if "centre_wavelength_nm" in cfg:
            self.centre_wavelength_nm = cfg.require_float("centre_wavelength_nm")
        if "chord_length_m" in cfg:
            self.chord_length_m = cfg.require_float("chord_length_m")
        if "accumulations" in cfg:
            self.accumulations = cfg.require_int("accumulations")
        if "noise" in cfg:
            self._noise_enabled = cfg.require_bool("noise")

    def set_noise_enabled(self, enabled: bool) -> None:
        """Switch the doc 04 §7.2 N1 stage — doc 04 §8 V-30 needs it switchable."""
        self._noise_enabled = enabled

    def use_true_calibration(self) -> None:
        """Apply the true response instead of the estimated one — doc 04 §7.3.

        The inverse crime, committed on purpose. Doc 04 §7.3 permits it for verification
        and the resulting :class:`~vpl.core.state.CalibrationState` records it, so a run
        that did this cannot be mistaken for an honest one in the artifact.
        """
        self._applied_state = CalibrationState.TRUE

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        """Derive the radiometric response from a reference standard — doc 02 §11.

        The estimated response is the truth plus one draw from the standard's own
        uncertainty. Drawn once and kept: see the module docstring on why it must not
        average down.
        """
        standard = refs.for_quantity("absolute_radiometric")
        uncertainty = standard.relative_uncertainty
        scale = 1.0 + uncertainty * float(self._calibration_rng.standard_normal())

        self._calibration = Calibration(
            instrument_id=self.instrument_id,
            coefficients={"radiance_scale": scale},
            relative_uncertainty={"radiance_scale": uncertainty},
            state=CalibrationState.ESTIMATED,
            reference=standard.name,
        )
        return self._calibration

    # ── the axis ────────────────────────────────────────────────────────────────

    def wavelength_axis(self) -> FloatArray:
        """The wavelength of each detector pixel at the current setting."""
        return self.spectrograph.wavelength_axis(centre_nm=self.centre_wavelength_nm)

    # ── the shared code path — doc 04 §9 ────────────────────────────────────────

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        """The noiseless spectral radiance the likelihood compares against — doc 04 §9."""
        values, _ = self._sample(state, w, noisy=False, scale=1.0)
        return Observable(
            instrument_id=self.instrument_id,
            values=values,
            units=RADIANCE_UNITS,
            window=w,
        )

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        """A shot-noise-limited, imperfectly calibrated observation — doc 04 §9."""
        if self._calibration is None:
            raise RuntimeError(
                f"{self.instrument_id}: calibrate() must be called before observe(). "
                "doc 04 §7.3 requires the pipeline to apply an *estimated* response, and "
                "there is nothing to apply until a reference standard has been measured."
            )
        scale = (
            1.0
            if self._applied_state is CalibrationState.TRUE
            else self._calibration.coefficients["radiance_scale"]
        )
        values, uncertainty = self._sample(state, w, noisy=self._noise_enabled, scale=scale)
        return Measurement(
            instrument_id=self.instrument_id,
            values=values,
            uncertainty=uncertainty,
            units=RADIANCE_UNITS,
            window=w,
            calibration=self._applied_state,
        )

    def _sample(
        self, state: PlasmaState, w: AcquisitionWindow, *, noisy: bool, scale: float
    ) -> tuple[FloatArray, FloatArray]:
        """The one code path doc 04 §9 requires. Both public methods are this call."""
        radiance = self._radiance(state, w)
        per_radiance = self._counts_per_radiance(w)

        if not noisy:
            expected = radiance * per_radiance
            return radiance / scale, np.sqrt(np.maximum(expected, _MINIMUM_VARIANCE_COUNTS)) / (
                per_radiance * scale
            )

        counts = self._rng.poisson(radiance * per_radiance).astype(np.float64)
        observed = counts / (per_radiance * scale)
        # The uncertainty a real pipeline reports comes from the data it has, not from the
        # truth it does not: sqrt of the *observed* count, floored so an empty pixel does
        # not claim zero uncertainty and take the likelihood to -inf.
        return observed, np.sqrt(np.maximum(counts, _MINIMUM_VARIANCE_COUNTS)) / (
            per_radiance * scale
        )

    # ── the physics ─────────────────────────────────────────────────────────────

    def _radiance(self, state: PlasmaState, w: AcquisitionWindow) -> FloatArray:
        """Spectral radiance at every grid point, averaged over the acquisition window."""
        axis = self.wavelength_axis()
        n_e_field = state.field("n_e")
        t_e_field = state.field("T_e")

        n_e_all = np.atleast_2d(magnitude_in(n_e_field.quantity, "m**-3"))
        t_e_all = np.atleast_2d(magnitude_in(t_e_field.quantity, "eV"))
        samples = self._window_samples(state, w, n_samples=n_e_all.shape[0])

        n_g = float(magnitude_in(state.params.n_g, "m**-3"))
        temperature_k = float(magnitude_in(state.params.T_g, "K"))
        mass_u = float(magnitude_in(state.params.species.mass, "u"))

        radiance = np.zeros((n_e_all.shape[1], axis.size), dtype=np.float64)
        for time_index in samples:
            for point in range(n_e_all.shape[1]):
                radiance[point] += self._point_spectrum(
                    axis,
                    electron_density_per_m3=float(n_e_all[time_index, point]),
                    electron_temperature_ev=float(t_e_all[time_index, point]),
                    ground_density_per_m3=n_g,
                    gas_temperature_k=temperature_k,
                    mass_u=mass_u,
                )
        return np.asarray(radiance / len(samples))

    @staticmethod
    def _window_samples(
        state: PlasmaState, w: AcquisitionWindow, *, n_samples: int
    ) -> tuple[int, ...]:
        """Which time indices the doc 05 §3.2 window integral runs over.

        A steady state has one. A time-dependent state contributes every sample inside the
        window, and at least the nearest one — a 2 ns gate on a grid whose spacing is
        coarser than that would otherwise integrate over nothing and return zeros, which is
        a plausible-looking dark frame rather than an error.
        """
        if state.time is None:
            return (0,)
        times = np.asarray(magnitude_in(state.time.t, "s"), dtype=np.float64)
        inside = np.flatnonzero((times >= w.start_s) & (times <= w.start_s + w.duration_s))
        if inside.size == 0:
            return (int(np.argmin(np.abs(times - (w.start_s + 0.5 * w.duration_s)))),)
        return tuple(int(index) for index in inside[:n_samples])

    def _point_spectrum(
        self,
        axis: FloatArray,
        *,
        electron_density_per_m3: float,
        electron_temperature_ev: float,
        ground_density_per_m3: float,
        gas_temperature_k: float,
        mass_u: float,
    ) -> FloatArray:
        """One CR solve and one synthesised spectrum, at one point."""
        populations = self.model.solve(
            electron_density_per_m3=electron_density_per_m3,
            ground_density_per_m3=ground_density_per_m3,
            f0=self.eedf.f0(electron_temperature_ev=electron_temperature_ev),
        )
        lines = emission_spectrum(populations)
        return self.spectrograph.synthesise(
            axis,
            wavelengths_nm=[line.wavelength_nm for line in lines],
            amplitudes=[chord_radiance(line, path_length_m=self.chord_length_m) for line in lines],
            intrinsic_fwhm_nm=[
                doppler_fwhm_nm(
                    wavelength_nm=line.wavelength_nm,
                    temperature_k=gas_temperature_k,
                    mass_u=mass_u,
                )
                for line in lines
            ],
        )

    # ── radiometry ──────────────────────────────────────────────────────────────

    @property
    def etendue_m2_sr(self) -> float:
        """``A Omega`` for the slit and the working f-number — a thin-lens stand-in.

        ``Omega = pi / (4 F^2)`` is the solid angle of an ``f/F`` cone. See the module
        docstring: the real quantity is a ray trace, and this enters only the noise level.
        """
        slit_area_m2 = float(self._registry.value_in("OES-S4.slit_width", "m")) * float(
            self._registry.value_in("OES.slit_height", "m")
        )
        f_number = float(self._registry.value_in("OES-S1.f_number", "dimensionless"))
        return slit_area_m2 * math.pi / (4.0 * f_number**2)

    def _counts_per_radiance(self, w: AcquisitionWindow) -> FloatArray:
        """Photoelectrons per unit spectral radiance, per pixel.

        ``N_pe = L . (A Omega) . dlambda_pixel . t_gate . QE / (h nu)``. Returned per pixel
        because the photon energy and the pixel bandpass both vary along the axis.
        """
        axis = self.wavelength_axis()
        quantum_efficiency = float(
            self._registry.value_in("OES-D3.quantum_efficiency", "dimensionless")
        )
        pixel_bandpass_nm = self.spectrograph.pixel_bandpass_nm(self.centre_wavelength_nm)
        photon_energy_j = np.asarray(
            magnitude_in(Q_(axis, "nm").to("J", "spectroscopy"), "J"), dtype=np.float64
        )
        return np.asarray(
            self.etendue_m2_sr
            * pixel_bandpass_nm
            * w.duration_s
            * self.accumulations
            * quantum_efficiency
            / photon_energy_j
        )

    # ── the likelihood and the gate ─────────────────────────────────────────────

    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb:
        """This channel's term in the doc 05 §3.2 sum, with doc 05 §3.1's switch.

        Doc 05 §3.1: "Poisson for weak lines; Gaussian for bright lines above ~100 pe;
        explicit switch at a configured threshold". The threshold is
        ``OES.poisson_gaussian_threshold``, and the switch is per *pixel* rather than per
        line, because within one line profile the wings are in the counting regime while
        the core is not.
        """
        if obs.shape != pred.shape:
            raise ValueError(
                f"an observation and its prediction must have the same shape, got "
                f"{obs.shape} and {pred.shape}"
            )

        per_radiance = self._counts_per_radiance(obs.window)
        observed = np.asarray(obs.values) * per_radiance
        expected = np.maximum(np.asarray(pred.values) * per_radiance, np.finfo(np.float64).tiny)

        threshold = float(
            self._registry.value_in("OES.poisson_gaussian_threshold", "dimensionless")
        )
        bright = expected > threshold

        counts = np.rint(observed)
        # Poisson log-pmf, written out rather than imported: scipy.stats would be a new
        # dependency for one expression, and doc 08 §2's "buy" applies to capabilities and
        # not to three terms of arithmetic.
        poisson = counts * np.log(expected) - expected - special.gammaln(counts + 1.0)
        residual = observed - expected
        gaussian = -0.5 * residual**2 / expected - 0.5 * np.log(2.0 * math.pi * expected)
        return float(np.sum(np.where(bright, gaussian, poisson)))

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        """The doc 01 IF-6 gate applied to OES — see ``OES.detection_floor_n_0``."""
        return self.metadata().detection_floor.admits(state_guess.n_0)

    def metadata(self) -> InstrumentMetadata:
        """Identity, citations and the detection floor — doc 00 C2, doc 01 IF-6."""
        return InstrumentMetadata(
            instrument_id=self.instrument_id,
            name="Czerny-Turner f/9.7 750 mm imaging spectrograph + gated ICCD (doc 02 §6.2)",
            version="0.1.0",
            citations=(
                Citation(
                    key="holstein-1947",
                    reference=(
                        "T Holstein, 'Imprisonment of resonance radiation in gases', "
                        "Phys. Rev. 72 (1947) 1212"
                    ),
                    doi="10.1103/PhysRev.72.1212",
                ),
                Citation(
                    key="irons-1979",
                    reference=(
                        "F E Irons, 'The escape factor in plasma spectroscopy', "
                        "J. Quant. Spectrosc. Radiat. Transfer 22 (1979) 1"
                    ),
                    doi="10.1016/0022-4073(79)90102-X",
                ),
                Citation(
                    key="olivero-longbothum-1977",
                    reference=(
                        "J J Olivero and R L Longbothum, 'Empirical fits to the Voigt line "
                        "width', J. Quant. Spectrosc. Radiat. Transfer 17 (1977) 233"
                    ),
                    doi="10.1016/0022-4073(77)90161-3",
                ),
                Citation(
                    key="nist-asd",
                    reference=(
                        "A Kramida, Yu Ralchenko, J Reader and NIST ASD Team, NIST Atomic "
                        "Spectra Database v5, National Institute of Standards and Technology"
                    ),
                ),
            ),
            detection_floor=DetectionFloor(
                quantity="n_0",
                threshold=Q_(
                    float(self._registry.value_in("OES.detection_floor_n_0", "m**-3")), "m**-3"
                ),
                requirement="IF-6",
            ),
            description=(
                "Collisional-radiative emission (doc 04 §2) through the doc 02 §6.2 "
                "spectrograph. Shot noise only; N2-N18 belong to vpl-detectors."
            ),
        )

    def __repr__(self) -> str:
        return (
            f"OesInstrument({self.instrument_id!r}, centred on "
            f"{self.centre_wavelength_nm} nm, {self.model!r})"
        )
