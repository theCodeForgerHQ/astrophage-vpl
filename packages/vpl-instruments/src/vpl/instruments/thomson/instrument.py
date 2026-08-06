"""``ThomsonInstrument`` — the doc 08 §4 contract over the Thomson scattering channel.

Doc 04 §9 states the contract this module has to satisfy, the same one
:class:`~vpl.instruments.oes.instrument.OesInstrument` and
:class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe` are built to:

    ``observe`` returns raw data with noise and imperfect calibration; ``forward``
    returns the noiseless expectation used by the likelihood. **Both come from the same
    code path**, with noise and calibration error as switchable stages, which guarantees
    they cannot drift apart.

So there is one private method, :meth:`ThomsonInstrument._sample`, and both public
methods are one call into it with the stages set differently.
``test_thomson_instrument.py`` asserts they agree *exactly* with noise off and the true
calibration applied; anything less than exact equality means two code paths.

## Why this channel exists — doc 02 §7.1

A single channel cannot separate ``n_e`` from ``T_e`` because they enter the physics
multiplied together (OES's line ratios, interferometry's line-integrated phase). Thomson
scattering breaks that degeneracy from a different physical mechanism entirely: the
*integrated* scattered intensity is proportional to ``n_e`` (:mod:`.photons`), and the
*width* of the scattered spectrum gives ``T_e`` (:mod:`.spectrum`) — two independent
observables out of one measurement, which is exactly the shape doc 02 §7.1 calls
Thomson's role as "the anchor" for.

## The chain, per spatial point

1. ``n_e``, ``T_e`` at the configured spatial index (mirrors
   :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe`'s single-point
   convention — Thomson's 90 um measurement volume, TS-O3, is a point diagnostic, not a
   chord or an image).
2. :func:`~vpl.instruments.thomson.spectrum.salpeter_alpha` and
   :func:`~vpl.instruments.thomson.spectrum.check_incoherent_regime` — refuse a state
   the single-particle spectrum does not describe (doc 04 §4.1, TS-2) rather than
   silently returning one anyway.
3. The doc 02 §7.1 photon budget (:mod:`.photons`) converts ``n_e`` and the window's
   accumulation into an expected total photoelectron count; :func:`.photons.is_blind`
   gates windows too short to see anything at all (doc 02 §7.1 consequence 2).
4. The relativistic spectrum (:mod:`.spectrum`) distributes that total across the TS-S3
   20-channel detector axis.
5. The doc 02 §4.3 stray-light pedestal (:mod:`.photons`) adds in, gets Poisson-drawn
   along with the signal, and is subtracted back out using either the true (forward) or
   an imperfectly estimated (observe) background level.

## Thomson cannot follow transients — visible in the API, not a comment

Doc 02 §7.1 consequence 2 is explicit that a single shot is invisible and that
phase-resolved operation is far more expensive again. This module makes that refusal an
actual, catchable failure mode rather than a silently very noisy number:

* :meth:`required_accumulation_s` lets a caller ask, before scheduling a point, how long
  an accumulation needs to be for a given target uncertainty (and phase-bin count).
* :meth:`forward` and :meth:`observe` — through the shared :meth:`_sample` — raise
  :class:`ThomsonBlindWindowError` for a window whose expected total photoelectron count
  falls below :data:`~vpl.instruments.thomson.photons.MINIMUM_EXPECTED_TOTAL_PHOTOELECTRONS`,
  rather than returning a value indistinguishable from a dark frame and calling it an
  observation.
* A phase-locked :class:`~vpl.core.state.AcquisitionWindow` with no explicit
  ``accumulation`` is refused outright (:meth:`_accumulation_s`): the phase bin alone does
  not say how many of the free-running 10 Hz laser's shots landed in it, and reading
  ``AcquisitionWindow.wall_clock_span``'s single-RF-period default here (a convention
  built for a channel gated every RF cycle, which Thomson is not) would understate the
  needed shot count by many orders of magnitude.

``is_informative`` (the doc 01 IF-6 gate) is deliberately *not* where the transient-
blindness lives: the protocol only hands it a :class:`~vpl.core.state.PlasmaParams` guess,
with no acquisition window, so it cannot know how long a proposed accumulation would run.
It instead gates on a detection floor tied to a *reasonable single-campaign accumulation
budget* — see :data:`~vpl.instruments.thomson.photons.REFERENCE_OPERATING_DENSITY_M3` and
the metadata for the derivation — which is the doc 01 IF-6 question ("at what density does
this channel stop being usable at all") rather than the doc 02 §7.1 consequence 2 question
("is *this* window long enough"), and the two are answered by different mechanisms on
purpose.

## Calibration — doc 02 §7.3, §11; doc 06 §4.1, §5

``calibrate`` draws two coefficients once, on the
:attr:`~vpl.core.random.Stream.CALIBRATION` stream, and keeps them: the Rayleigh-derived
absolute count scale, and the stray-light background-subtraction estimate. Doc 06 §4.1 is
explicit that "a single Rayleigh calibration error affects *every* Thomson point
identically and does **not** average down" — re-drawing either per observation would make
it average away and understate the budget, exactly as
:class:`~vpl.instruments.oes.instrument.OesInstrument`'s module docstring argues for its
own radiometric scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import special

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.protocols import (
    Calibration,
    CalibrationSet,
    Citation,
    DetectionFloor,
    InstrumentConfig,
    InstrumentMetadata,
    LogProb,
)
from vpl.core.random import Stream, generator
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    InstrumentId,
    Measurement,
    Observable,
    PlasmaParams,
    PlasmaState,
    ScalarField,
)
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.coherent import (
    coherent_gaussian_log_prob,
    fractional_calibration_row,
    stack_coherent_rows,
)
from vpl.instruments.thomson import photons, spectrum
from vpl.instruments.thomson.spectrum import LASER_WAVELENGTH_NM

__all__ = [
    "CHANNEL_WIDTH_NM",
    "N_SPECTRAL_CHANNELS",
    "PHOTON_COUNT_UNITS",
    "SPECTRAL_COVERAGE_MAX_NM",
    "SPECTRAL_COVERAGE_MIN_NM",
    "THOMSON_INSTRUMENT_ID",
    "ThomsonBlindWindowError",
    "ThomsonInstrument",
]

type FloatArray = NDArray[np.float64]

#: The doc 08 §7 artifact group this channel writes to.
THOMSON_INSTRUMENT_ID: Final[InstrumentId] = "thomson"

#: What :meth:`ThomsonInstrument.forward` and :meth:`observe` return: a photoelectron
#: count per spectral channel, doc 02 §7.1's "photon-counting experiment" made explicit
#: in the returned quantity rather than only in the noise model.
PHOTON_COUNT_UNITS: Final[str] = "dimensionless"

# ── the TS-S2 / TS-S3 detector axis ─────────────────────────────────────────────────

#: TS-S2.
SPECTRAL_COVERAGE_MIN_NM: Final[float] = 520.0
#: TS-S2.
SPECTRAL_COVERAGE_MAX_NM: Final[float] = 545.0
#: TS-S3.
N_SPECTRAL_CHANNELS: Final[int] = 20
#: The width of one TS-S3 channel, derived from TS-S2's coverage: 1.25 nm.
CHANNEL_WIDTH_NM: Final[float] = (
    SPECTRAL_COVERAGE_MAX_NM - SPECTRAL_COVERAGE_MIN_NM
) / N_SPECTRAL_CHANNELS

#: Which spatial grid point the measurement volume sits at. Mirrors
#: :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe`'s
#: ``_DEFAULT_Z_INDEX``: RP-1 is a bulk-plasma reference point (doc 02 §3.3 regime B), and
#: Thomson's 90 um resolution element (TS-O3) is small enough that a manifest scanning it
#: through the sheath overrides this per point rather than this default trying to guess
#: one.
_DEFAULT_Z_INDEX: Final[int] = -1

#: Smallest counting variance, in photoelectrons, the Poisson floor below will divide by
#: — mirrors ``OesInstrument._MINIMUM_VARIANCE_COUNTS`` for the same reason: a channel
#: expecting zero counts still receives none with probability one, and a zero-variance
#: report there is infinitely (and wrongly) confident about an ordinary observation.
_MINIMUM_VARIANCE_COUNTS: Final[float] = 1.0

#: Smallest per-channel *expected* photoelectron count :meth:`ThomsonInstrument.likelihood`
#: will accept a ``coherent_discrepancy`` term against.
#:
#: A Poisson(lambda) random variable has skewness ``1 / sqrt(lambda)``; at lambda = 20 that
#: is ~0.22, the conventional rule-of-thumb point at which a symmetric Gaussian becomes a
#: defensible stand-in for a Poisson pmf (e.g. Casella & Berger, *Statistical Inference*,
#: 2nd ed., §5.3's discussion of the normal approximation to a sum of count variables — a
#: Poisson is itself a limit of a Binomial sum). See ``likelihood``'s docstring, "Model
#: discrepancy on a Poisson channel", for why this floor exists and why it is expected to
#: bind at this project's own RP-1 reference point.
_COHERENT_GAUSSIAN_MINIMUM_EXPECTED_COUNTS: Final[float] = 20.0


class ThomsonBlindWindowError(ValueError):
    """A window's expected photon count is below the doc 02 §7.1 consequence 2 floor.

    Raised by :meth:`ThomsonInstrument.forward` and :meth:`ThomsonInstrument.observe`
    (through :meth:`ThomsonInstrument._sample`) rather than silently returning a value
    indistinguishable from a dark frame. See
    :data:`~vpl.instruments.thomson.photons.MINIMUM_EXPECTED_TOTAL_PHOTOELECTRONS` and
    :meth:`ThomsonInstrument.required_accumulation_s`.
    """


def _local_value(field: ScalarField, *, z_index: int) -> float:
    """One grid point's value, at the first time sample if the field is time-resolved.

    A Thomson point is effectively instantaneous on the multi-minute accumulation scale
    this channel operates on, exactly the same simplification
    :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe`'s own
    ``_local_value`` makes for a probe sweep.
    """
    if field.is_steady:
        return float(field.values[z_index])
    return float(field.values[0, z_index])


@dataclass(frozen=True, slots=True)
class _Settings:
    z_index: int


class ThomsonInstrument:
    """The 90 degree Thomson scattering channel — doc 02 §7, doc 04 §4, doc 08 §4."""

    __slots__ = (
        "_applied_state",
        "_calibration",
        "_calibration_rng",
        "_noise_enabled",
        "_registry",
        "_rng",
        "_settings",
        "instrument_id",
    )

    def __init__(
        self,
        *,
        root_seed: int,
        instrument_id: InstrumentId = THOMSON_INSTRUMENT_ID,
        registry: ParameterRegistry | None = None,
    ) -> None:
        self.instrument_id = instrument_id
        self._registry = registry if registry is not None else default_registry()
        self._settings: _Settings | None = None
        self._calibration: Calibration | None = None
        self._applied_state = CalibrationState.ESTIMATED
        self._noise_enabled = True
        self._rng = generator(root_seed, Stream.PHOTONS)
        self._calibration_rng = generator(root_seed, Stream.CALIBRATION)

    # ── configuration ───────────────────────────────────────────────────────────

    def configure(self, cfg: InstrumentConfig) -> None:
        """Apply one entry of the manifest's ``instruments:`` list — doc 08 §6.

        ``z_index``
            Spatial grid point the measurement volume sits at. Defaults to
            :data:`_DEFAULT_Z_INDEX`.
        ``noise``
            Whether :meth:`observe` applies photon shot noise and the calibration draws.
        """
        z_index = _DEFAULT_Z_INDEX
        if "z_index" in cfg:
            z_index = cfg.require_int("z_index")
        self._settings = _Settings(z_index=z_index)

        if "noise" in cfg:
            self._noise_enabled = cfg.require_bool("noise")

    def set_noise_enabled(self, enabled: bool) -> None:
        """Switch photon shot noise and the calibration draws — doc 04 §8 V-30."""
        self._noise_enabled = enabled

    def use_true_calibration(self) -> None:
        """Apply the true response instead of the estimated one — doc 04 §7.3.

        The inverse crime, committed on purpose, exactly as
        :meth:`~vpl.instruments.oes.instrument.OesInstrument.use_true_calibration`
        documents: doc 04 §7.3 permits it for verification and the resulting
        :class:`~vpl.core.state.CalibrationState` records it.
        """
        self._applied_state = CalibrationState.TRUE

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        """Derive the Rayleigh count scale and stray-background estimate — doc 02 §7.3.

        Two coefficients, both drawn once and kept (module docstring): ``count_scale``
        from the Rayleigh standard's own uncertainty (doc 06 §5's five-term chain,
        :data:`~vpl.instruments.thomson.photons.RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY`),
        and ``stray_pedestal_scale`` from the doc 06 §4 term 11 background-subtraction
        uncertainty.
        """
        standard = refs.for_quantity(photons.RAYLEIGH_CALIBRATION_QUANTITY)
        count_uncertainty = standard.relative_uncertainty
        count_scale = 1.0 + count_uncertainty * float(self._calibration_rng.standard_normal())
        stray_uncertainty = photons.STRAY_LIGHT_BACKGROUND_RELATIVE_UNCERTAINTY
        stray_scale = 1.0 + stray_uncertainty * float(self._calibration_rng.standard_normal())

        self._calibration = Calibration(
            instrument_id=self.instrument_id,
            coefficients={"count_scale": count_scale, "stray_pedestal_scale": stray_scale},
            relative_uncertainty={
                "count_scale": count_uncertainty,
                "stray_pedestal_scale": stray_uncertainty,
            },
            state=CalibrationState.ESTIMATED,
            reference=standard.name,
        )
        return self._calibration

    # ── the axis ────────────────────────────────────────────────────────────────

    def wavelength_axis_nm(self) -> FloatArray:
        """The TS-S3 channel centre wavelengths — 20 bins tiling TS-S2's 520-545 nm."""
        edges = np.linspace(
            SPECTRAL_COVERAGE_MIN_NM, SPECTRAL_COVERAGE_MAX_NM, N_SPECTRAL_CHANNELS + 1
        )
        return np.asarray((edges[:-1] + edges[1:]) / 2.0, dtype=np.float64)

    # ── the doc 02 §7.1 consequence 2 refusal, exposed rather than hidden ────────

    def required_accumulation_s(
        self,
        state_guess: PlasmaParams,
        *,
        target_relative_uncertainty: float = photons.DEFAULT_TARGET_RELATIVE_UNCERTAINTY,
        n_phase_bins: float = 1.0,
    ) -> float:
        """How long an accumulation needs to be — doc 02 §7.1 consequence 2.

        A caller can ask this *before* scheduling a Thomson point; see the module
        docstring for why this, rather than ``is_informative``, is where the transient-
        blindness question lives.
        """
        return photons.required_accumulation_s(
            electron_density_m3=state_guess.n_0_per_m3,
            target_relative_uncertainty=target_relative_uncertainty,
            n_phase_bins=n_phase_bins,
        )

    # ── the shared code path — doc 04 §9 ────────────────────────────────────────

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        """The noiseless photoelectron spectrum the likelihood compares against — doc 04 §9."""
        values, _ = self._sample(state, w, noisy=False, applied_state=CalibrationState.TRUE)
        return Observable(
            instrument_id=self.instrument_id, values=values, units=PHOTON_COUNT_UNITS, window=w
        )

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        """A shot-noise-limited, imperfectly calibrated observation — doc 04 §9."""
        self._require_calibration()
        values, uncertainty = self._sample(
            state, w, noisy=self._noise_enabled, applied_state=self._applied_state
        )
        return Measurement(
            instrument_id=self.instrument_id,
            values=values,
            uncertainty=uncertainty,
            units=PHOTON_COUNT_UNITS,
            window=w,
            calibration=self._applied_state,
        )

    def _sample(
        self,
        state: PlasmaState,
        w: AcquisitionWindow,
        *,
        noisy: bool,
        applied_state: CalibrationState,
    ) -> tuple[FloatArray, FloatArray]:
        """The one code path doc 04 §9 requires. Both public methods are this call."""
        settings = self._require_configured()
        electron_density_m3 = _local_value(state.field("n_e"), z_index=settings.z_index)
        electron_temperature_ev = _local_value(state.field("T_e"), z_index=settings.z_index)

        alpha = spectrum.salpeter_alpha(
            electron_density_m3=electron_density_m3,
            electron_temperature_ev=electron_temperature_ev,
        )
        spectrum.check_incoherent_regime(alpha)

        shots = self._shots(w)
        self._require_not_blind(electron_density_m3=electron_density_m3, shots=shots)

        axis_nm = self.wavelength_axis_nm()
        weights = self._spectral_weights(electron_temperature_ev=electron_temperature_ev)
        per_shot = photons.photoelectrons_per_shot(electron_density_m3)
        signal = weights * (per_shot * shots)
        stray = (
            photons.stray_light_spectrum_pe_per_shot(
                axis_nm, notch_width_nm=CHANNEL_WIDTH_NM, registry=self._registry
            )
            * shots
        )
        raw_expected = signal + stray

        counts = self._rng.poisson(raw_expected).astype(np.float64) if noisy else raw_expected

        if applied_state is CalibrationState.TRUE:
            stray_estimate = stray
            count_scale = 1.0
            stray_subtraction_term = np.zeros_like(stray)
        else:
            calibration = self._require_calibration()
            stray_estimate = stray * calibration.coefficients["stray_pedestal_scale"]
            count_scale = calibration.coefficients["count_scale"]
            stray_subtraction_term = photons.STRAY_LIGHT_BACKGROUND_RELATIVE_UNCERTAINTY * stray

        subtracted = counts - stray_estimate
        values = subtracted / count_scale
        counting_variance = np.maximum(counts, _MINIMUM_VARIANCE_COUNTS)
        uncertainty = np.sqrt(counting_variance + stray_subtraction_term**2) / count_scale
        return np.asarray(values, dtype=np.float64), np.asarray(uncertainty, dtype=np.float64)

    # ── the physics ─────────────────────────────────────────────────────────────

    def _spectral_weights(self, *, electron_temperature_ev: float) -> FloatArray:
        """Fraction of the total signal each TS-S3 channel receives.

        :func:`~vpl.instruments.thomson.spectrum.relativistic_spectrum` is peak-
        normalised (doc 04 §4.2's "always included" relativistic correction, unswitched:
        see the spectrum module docstring); a Gaussian of the same 1/e half-width
        integrates to ``half_width * sqrt(pi)`` over all wavelength, so dividing the
        per-channel shape by that and multiplying by the channel width turns "shape" into
        "fraction of the total signal in this channel" — verified in
        ``test_thomson_instrument.py`` to sum close to 1 over TS-S2's window.
        """
        axis_nm = self.wavelength_axis_nm()
        delta_lambda_nm = axis_nm - LASER_WAVELENGTH_NM
        shape = spectrum.relativistic_spectrum(
            delta_lambda_nm, electron_temperature_ev=electron_temperature_ev
        )
        half_width_nm = spectrum.gaussian_half_width_nm(
            electron_temperature_ev=electron_temperature_ev
        )
        normalisation = half_width_nm * np.sqrt(np.pi)
        return np.asarray(shape * (CHANNEL_WIDTH_NM / normalisation), dtype=np.float64)

    def _accumulation_s(self, w: AcquisitionWindow) -> float:
        """Real elapsed time the 10 Hz laser fired over — not the ICCD gate width.

        Doc 02 §10.3: Thomson's 5 ns gate is locked to the Q-switch, not to the RF cycle,
        so ``AcquisitionWindow.duration`` (the gate) is not the accumulation time an
        absolute Thomson point actually needs (~700 s at RP-1, doc 02 §7.1). For an
        absolute window, ``duration`` *is* that accumulation —
        ``AcquisitionWindow.absolute(start=..., duration=Q_(700.0, "s"))`` is exactly how
        doc 02's "700 s point" is expressed. For a phase-locked window, only
        ``accumulation`` — the real wall-clock span the laser fired over while filling
        every phase bin — carries that information; ``wall_clock_span``'s single-RF-
        period default exists for a channel gated every cycle (OES) and understates a
        free-running 10 Hz laser's needed shot count by many orders of magnitude, so a
        phase-locked window with no explicit ``accumulation`` is refused rather than
        silently read as one RF period.

        Raises:
            ValueError: If ``w`` is phase-locked and carries no ``accumulation``.
        """
        if w.is_absolute:
            return w.duration_s
        if w.accumulation is None:
            raise ValueError(
                f"{self.instrument_id}: a phase-locked window needs an explicit "
                "`accumulation` (the real wall-clock span the 10 Hz laser fired over). "
                "doc 02 §10.3's phase binning does not by itself say how many shots "
                "landed in this bin, and reading `wall_clock_span`'s one-RF-period "
                "default here would understate the shot count by orders of magnitude "
                "(doc 02 §7.1 consequence 2)."
            )
        return float(magnitude_in(w.accumulation, "s"))

    @staticmethod
    def _n_phase_bins(w: AcquisitionWindow) -> float:
        """How many phase bins share the accumulated shots — doc 02 §7.1 consequence 2.

        ``1.0`` for an absolute window. For a phase-locked window, doc 02 §10.3's 16
        bins each receive roughly ``1/16`` of the shots the free-running 10 Hz laser
        fires (it is not synchronised to the 13.56 MHz RF cycle), so reaching the same
        per-bin count needs 16 times the shots an unbinned accumulation would.
        """
        if w.phase_grid is None:
            return 1.0
        return float(w.phase_grid.n_bins)

    def _shots(self, w: AcquisitionWindow) -> float:
        """Laser shots landing in this window's phase bin (or all of them, if unbinned)."""
        accumulation_s = self._accumulation_s(w)
        return photons.shots_in_window(accumulation_s) / self._n_phase_bins(w)

    def _require_not_blind(self, *, electron_density_m3: float, shots: float) -> None:
        if not photons.is_blind(electron_density_m3=electron_density_m3, shots=shots):
            return
        total_pe = photons.photoelectrons_per_shot(electron_density_m3) * shots
        raise ThomsonBlindWindowError(
            f"{self.instrument_id}: {shots:.4g} shots at n_e={electron_density_m3:.3g} m^-3 "
            f"give an expected {total_pe:.3g} total photoelectrons, below the "
            f"{photons.MINIMUM_EXPECTED_TOTAL_PHOTOELECTRONS:.3g}-pe floor doc 02 §7.1 "
            "consequence 2 calls a single-shot event: 'invisible'. Use "
            "required_accumulation_s() to find a window this channel can actually see."
        )

    # ── the likelihood and the gate ─────────────────────────────────────────────

    def likelihood(
        self,
        obs: Measurement,
        pred: Observable,
        *,
        coherent_discrepancy: FloatArray | None = None,
        calibration_uncertainty: bool = False,
    ) -> LogProb:
        """This channel's Poisson term in the doc 05 §3.2 sum — doc 05 §3.1: "Poisson for
        Thomson", unconditionally (unlike OES's bright/faint switch): doc 02 §7.1's
        0.008 pe/channel/shot at RP-1 means the bright-pixel regime that switch exists
        for essentially never applies here.

        ``pred.values`` is already the true, background-subtracted signal photoelectron
        count per channel (:meth:`forward`'s definition), and ``obs.values`` is the same
        quantity's calibrated, background-subtracted estimate, so no unit conversion is
        needed before comparing them as counts — unlike
        :meth:`~vpl.instruments.oes.instrument.OesInstrument.likelihood`, which has to
        convert radiance back to counts using the pixel bandpass and etendue first.

        Args:
            obs: The measurement.
            pred: The prediction to score it against.
            coherent_discrepancy: Optional per-channel **standard deviation** of model
                discrepancy, in photoelectrons, treated as fully coherent across channels
                (doc 05 §4, doc 11 WBS 3.6). ``None`` keeps the exact pre-existing
                behaviour, which is what T0 and T1 need — their model is correct and their
                intervals already cover, so any perturbation is a regression. See **Model
                discrepancy on a Poisson channel** below for what supplying one actually
                does to this particular likelihood, and when it is refused.
            calibration_uncertainty: Whether to score the doc 02 §7.3 Rayleigh calibration
                as the coherent systematic it is — see **Calibration** below.

        Model discrepancy on a Poisson channel — the choice made here, and why:
            A coherent additive *standard deviation* has no natural home in a pure Poisson
            likelihood. A Poisson distribution has exactly one free parameter, its mean,
            and mean and variance are the same number by construction — there is no
            separate variance slot doc 05 §4's discrepancy term can inflate the way it
            inflates OES's or LIF's Gaussian covariance. Three ways to handle that were
            weighed:

            (a) Switch to the Gaussian normal approximation to the Poisson once a
                discrepancy is supplied, and route it through the same rank-``k`` Woodbury
                machinery every other channel's coherent term already uses — valid once
                the counts involved are large enough for a Gaussian to be a defensible
                stand-in for a Poisson pmf, and refused otherwise.
            (b) Add the discrepancy as extra variance in an overdispersed count model
                (a negative-binomial-style excess-variance term) that stays a genuine
                count distribution throughout.
            (c) Refuse to support a coherent discrepancy on this channel at all.

            **(a) is what is implemented, gated by an explicit count floor, and (b) and
            (c) were both rejected.** (b) would give this one channel a second
            count-noise family that exists solely for the discrepancy term, so which
            distribution describes the same data would depend on which optional argument
            happened to be supplied — the kind of quietly inconsistent modelling doc 00 §6
            calls parameter fog, and it does not reuse any machinery this project has
            already validated. (c) was rejected because it is not the only defensible
            answer available: (a) reuses, rather than invents, the precedent
            :meth:`~vpl.instruments.oes.instrument.OesInstrument.likelihood` already sets
            — that method also abandons its Poisson pmf entirely and switches every pixel
            to the shared correlated Gaussian the moment *any* coherent term is supplied,
            for the reason its own docstring gives: "once systematic error dominates the
            counting statistics, mixing a Poisson pmf with a correlated Gaussian is not a
            well-defined joint density anyway". Applying that same, already-adopted
            reasoning here — rather than re-litigating it per channel — is (a).

            **The count floor is not decorative.** A Poisson(lambda) distribution's
            skewness is ``1 / sqrt(lambda)``;
            :data:`_COHERENT_GAUSSIAN_MINIMUM_EXPECTED_COUNTS` (20 pe) is the point at
            which that skewness (~0.22) is small enough for the Gaussian stand-in to be
            defensible — see that constant's own docstring for the citation. Every
            channel's *expected* count (``pred.values``, not the noisier observed one) is
            checked against it before the Gaussian branch is taken, and a channel below
            the floor raises rather than silently returning a number computed under an
            approximation known to be poor there.

            **This floor is expected to bind at this project's own reference point, and
            that is reported here rather than tuned around.** This method's opening
            paragraph already cites doc 02 §7.1's 0.008 pe/channel/shot as the reason the
            OES-style bright/faint switch "essentially never applies here" even *without*
            a discrepancy term. A ~700 s RP-1 accumulation raises the *total* signal
            considerably above one shot's worth — doc 02 §7.1's own worked example puts
            the channel-summed total near ``1 / 0.03**2 ~ 1.1e3`` counts for a 3 %
            measurement — but that total is spread unevenly across
            :data:`N_SPECTRAL_CHANNELS` channels by :meth:`_spectral_weights`'s line
            shape, so individual channels away from line centre fall below the 20 pe floor
            well before the channel-summed total does. A coherent discrepancy correction
            is therefore genuinely not defensible for every RP-1 Thomson acquisition this
            project runs, and this method says so by raising rather than by quietly
            approximating past it — a stated limitation of applying a Kennedy & O'Hagan
            (2001)-style discrepancy correction to this particular channel, not a defect
            in the check.

        Calibration — doc 06 §5, doc 06 §4.1:
            :meth:`calibrate` draws a ``count_scale`` from the Rayleigh standard, and this
            module's own docstring already quotes doc 06 §4.1 on why it is drawn once and
            kept: "a single Rayleigh calibration error affects *every* Thomson point
            identically and does **not** average down". Until this argument existed, the
            *likelihood* did not act on that at all — it asserted the absolute count scale
            was exactly 1.0, which is infinite confidence in a quantity
            :data:`~vpl.instruments.thomson.photons.RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY`
            puts at ~6.6 % one-sigma (doc 06 §5's five-term chain). That is the same defect
            :mod:`vpl.instruments.coherent`'s module docstring records for OES and LIF, on a
            channel whose calibration is no better than OES's.

            Enabled, the calibration enters as one coherent row, ``f * y_hat`` — a fractional
            error on the *whole* prediction, which is what a single absolute scale factor is
            — and the channel switches to the correlated Gaussian ``D + U'U`` with ``D`` the
            Poisson counting variance. Same reasoning
            :meth:`~vpl.instruments.oes.instrument.OesInstrument.likelihood` gives for its
            own switch: once a systematic dominates the counting statistics, mixing a Poisson
            pmf with a correlated Gaussian is not a well-defined joint density, and the
            Woodbury algebra is taken from :mod:`vpl.instruments.coherent` rather than
            re-derived so this channel's treatment cannot drift from the other three's.

            The uncertainty is read from the module constant, not from :attr:`_calibration`,
            for the reason
            :meth:`~vpl.instruments.oes.instrument.OesInstrument._radiometric_uncertainty`
            gives: the likelihood is used by the *inversion*, which is deliberately
            uncalibrated and holds no :class:`~vpl.core.protocols.instrument.Calibration` of
            its own.

            **Why opt-in**, and **why a ``CalibrationState.TRUE`` measurement is a no-op**:
            identical reasoning to OES's own "Calibration" section — a flipped default would
            silently re-score every existing caller's posterior, and doc 04 §7.3 permits the
            true response "for verification", which leaves no calibration error to score. The
            guard is on the measurement's own record rather than on the caller remembering.

        Raises:
            ValueError: If ``obs`` and ``pred`` have different shapes, or if
                ``coherent_discrepancy`` is supplied while any channel's *expected*
                photoelectron count is below
                :data:`_COHERENT_GAUSSIAN_MINIMUM_EXPECTED_COUNTS` — see "Model
                discrepancy on a Poisson channel" above.
        """
        if obs.shape != pred.shape:
            raise ValueError(
                f"an observation and its prediction must have the same shape, got "
                f"{obs.shape} and {pred.shape}"
            )
        counts = np.maximum(np.rint(np.asarray(obs.values, dtype=np.float64)), 0.0)
        expected = np.maximum(np.asarray(pred.values, dtype=np.float64), np.finfo(np.float64).tiny)

        rows: list[FloatArray] = []
        if calibration_uncertainty and obs.calibration is not CalibrationState.TRUE:
            rows.append(
                fractional_calibration_row(
                    np.asarray(pred.values, dtype=np.float64),
                    relative_uncertainty=photons.RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY,
                )
            )
        if coherent_discrepancy is not None:
            if np.any(expected < _COHERENT_GAUSSIAN_MINIMUM_EXPECTED_COUNTS):
                floor_hit = float(np.min(expected))
                raise ValueError(
                    f"{self.instrument_id}: a coherent model discrepancy was supplied but "
                    f"the expected photoelectron count falls as low as {floor_hit:.3g} in "
                    "at least one channel, below the "
                    f"{_COHERENT_GAUSSIAN_MINIMUM_EXPECTED_COUNTS:.3g}-pe floor this "
                    "channel's Gaussian normal-approximation branch needs to be a "
                    "defensible stand-in for the Poisson pmf a discrepancy term has no "
                    "variance slot in — see this method's docstring, 'Model discrepancy "
                    "on a Poisson channel'. Doc 02 §7.1's 0.008 pe/channel/shot means this "
                    "is expected at RP-1; a coherent discrepancy correction is not "
                    "defensible for this channel at that operating point."
                )
            rows.append(np.asarray(coherent_discrepancy, dtype=np.float64))
        # `None` here means nothing coherent was supplied, and the pre-existing Poisson
        # likelihood is returned unchanged and bit for bit — which is what protects every
        # existing caller of this method from a change only opt-in callers asked for.
        basis = stack_coherent_rows(rows, expected_shape=expected.shape)
        if basis is None:
            poisson = counts * np.log(expected) - expected - special.gammaln(counts + 1.0)
            return float(np.sum(poisson))

        # Floor the counting variance at one photoelectron, exactly as
        # `OesInstrument.likelihood`'s own coherent branch does and for the same reason: this
        # branch is Gaussian on every channel, and a channel expecting ~0 counts would
        # otherwise get weight 1/epsilon and dominate the whole likelihood. Same constant and
        # same reasoning as `_MINIMUM_VARIANCE_COUNTS` above.
        return coherent_gaussian_log_prob(
            residual=(counts - expected).reshape(-1),
            variance=np.maximum(expected.reshape(-1), _MINIMUM_VARIANCE_COUNTS),
            basis=basis,
        )

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        """The doc 01 IF-6 gate — see the module docstring for why the transient-
        blindness question (doc 02 §7.1 consequence 2) is answered elsewhere."""
        return self.metadata().detection_floor.admits(state_guess.n_0)

    def metadata(self) -> InstrumentMetadata:
        """Identity, citations and the detection floor — doc 00 C2, doc 01 IF-6."""
        return InstrumentMetadata(
            instrument_id=self.instrument_id,
            name=(
                "90 degree Thomson scattering: Nd:YAG 532 nm, triple-grating "
                "spectrometer, gated ICCD (doc 02 §7.2)"
            ),
            version="0.1.0",
            citations=(
                Citation(
                    key="salpeter-1960",
                    reference=(
                        "E. E. Salpeter, 'Electron Density Fluctuations in a Plasma', "
                        "Phys. Rev. 120 (1960) 1528"
                    ),
                    doi="10.1103/PhysRev.120.1528",
                ),
                Citation(
                    key="evans-katzenstein-1969",
                    reference=(
                        "D. E. Evans and J. Katzenstein, 'Laser light scattering in "
                        "laboratory plasmas', Rep. Prog. Phys. 32 (1969) 207"
                    ),
                    doi="10.1088/0034-4885/32/1/305",
                ),
                Citation(
                    key="selden-1980",
                    reference=(
                        "A. C. Selden, 'Simple analytic form of the relativistic "
                        "Thomson scattering spectrum', Phys. Lett. A 79 (1980) 405"
                    ),
                    doi="10.1016/0375-9601(80)90276-5",
                ),
                Citation(
                    key="sheffield-froula-2011",
                    reference=(
                        "J. Sheffield, D. Froula, S. H. Glenzer and N. C. Luhmann Jr., "
                        "'Plasma Scattering of Electromagnetic Radiation: Theory and "
                        "Measurement Techniques', 2nd ed., Academic Press (2011)"
                    ),
                ),
            ),
            detection_floor=DetectionFloor(
                quantity="n_0",
                threshold=Q_(_detection_floor_n_0_m3(registry=self._registry), "m**-3"),
                requirement="IF-6",
            ),
            description=(
                "Doc 02 §7.1 photon budget (Poisson photoelectron counting, doc 05 §3.1) "
                "plus the doc 04 §4 relativistic scattered spectrum. Absolute density "
                "from Rayleigh calibration (doc 02 §7.3); T_e from the spectral width."
            ),
        )

    def __repr__(self) -> str:
        return f"ThomsonInstrument({self.instrument_id!r})"

    # ── internal guards ─────────────────────────────────────────────────────────

    def _require_configured(self) -> _Settings:
        if self._settings is None:
            raise RuntimeError(
                f"{self.instrument_id}: configure() must be called before use. doc 08 §6 "
                "gives every instrument a manifest block; an unconfigured Thomson channel "
                "would otherwise silently measure at a default spatial point that appears "
                "in no artifact."
            )
        return self._settings

    def _require_calibration(self) -> Calibration:
        if self._calibration is None:
            raise RuntimeError(
                f"{self.instrument_id}: calibrate() must be called before observe(). "
                "doc 04 §7.3 requires the pipeline to apply an *estimated* response, and "
                "there is nothing to apply until the Rayleigh standard has been measured."
            )
        return self._calibration


def _detection_floor_n_0_m3(*, registry: ParameterRegistry) -> float:
    """The doc 01 IF-6 floor: below this, even the doc 02 §7.1 3 % target needs longer
    than ``TS.maximum_reasonable_accumulation_s``.

    ``TS.maximum_reasonable_accumulation_s`` (registered ``ASSUMED``, doc 00 C1) is the
    maximum accumulation a single Thomson point is treated as schedulable within — one
    working shift, nominally. Doc 02 does not give an operational campaign-length bound;
    see the registry entry's description for the retirement path. Read from the registry
    rather than held as a module constant so the doc 00 C1 defect stays a single,
    greppable entry.

    ``required_accumulation_s`` scales as ``1/n_e`` (:func:`photoelectrons_per_shot` is
    linear in ``n_e``), so inverting it against the accumulation budget gives the density
    floor directly rather than a second independently-chosen number.
    """
    maximum_accumulation_s = float(registry.value_in("TS.maximum_reasonable_accumulation_s", "s"))
    accumulation_at_rp1_s = photons.required_accumulation_s(
        electron_density_m3=photons.REFERENCE_OPERATING_DENSITY_M3,
        target_relative_uncertainty=photons.DEFAULT_TARGET_RELATIVE_UNCERTAINTY,
    )
    return photons.REFERENCE_OPERATING_DENSITY_M3 * (accumulation_at_rp1_s / maximum_accumulation_s)
