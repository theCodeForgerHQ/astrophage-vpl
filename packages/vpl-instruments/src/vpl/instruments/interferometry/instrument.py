"""``InterferometryInstrument`` — the doc 08 §4 contract over WBS 2.8, doc 02 §8.

## Why this channel exists (put here because it is the reason for the whole module)

The project's honest end-to-end error is 36.5 % (doc 06). The diagnosed cause is that a
single channel cannot separate plasma density from electron temperature: they enter the
physics multiplied together, so one spectrum or one I-V curve constrains a combination of
``n_e`` and ``T_e`` rather than either alone. Interferometry attacks that from the one
angle OES and the probe cannot: doc 04 §5.1's phase shift, ``Delta phi = r_e lambda
INTEGRAL n_e dl``, depends on the electron density **alone** — there is no ``T_e`` anywhere
in it. That is what makes a coarse, line-integrated, sheath-blind measurement worth having
at all: it is not a better density measurement than the others, it is an *independent*
one, and breaking the ``n_e``-``T_e`` degeneracy is worth more to the joint fit than a
second, more detailed measurement of the same combination would be.

It is deliberately not asked to do more than that. Doc 02 §8.3 is explicit about the
limits: "no intrinsic z resolution — it cannot see the sheath structure at all"; the
0.89 mm sheath is far below what 8 chords spaced 5 mm apart can resolve; "interferometry
constrains the *boundary condition* of the sheath problem, not the sheath... the inversion
must not be allowed to over-weight it." Nothing in this module tries to make the channel
more informative than that description — see the "what is deliberately not modelled"
section below.

## The chain, per acquisition

1. :meth:`InterferometryInstrument.chord_positions_m` places 8 chords (doc 02 §8.2 IF-G1)
   at 5 mm spacing in ``z``, anchored at the wall (``z = 0``) by default — the most
   conservative anchor, since it maximises the fraction of this coarse, bulk-averaging
   diagnostic that overlaps the near-wall region doc 02 §8.3 says it must not be trusted
   to resolve, rather than quietly placing the ladder somewhere it looks better behaved.
2. Each chord's density is the linear interpolation (:func:`numpy.interp`, clamped to the
   grid's own range at the edges) of the solver's ``n_e(z)`` onto that chord's position.
   Linear interpolation, not nearest-grid-point: it is the standard reconstruction of a
   continuous field from a discretised solver grid, and does not introduce a spurious step
   between adjacent chords whenever the grid happens to be coarser than the 5 mm chord
   spacing, the way snapping to the nearest node would.
3. :func:`~vpl.instruments.interferometry.phase.net_phase_shift_rad` turns each chord's
   density (plus the state's neutral density, doc 04 §5.2) into a phase, under the
   transverse-homogeneity assumption documented in
   :mod:`vpl.instruments.interferometry.phase`.
4. Noise (correlated vibration plus independent detector noise, doc 04 §5.2, doc 05 §3.1)
   and fringe jumps (:mod:`vpl.instruments.interferometry.noise`) are switchable stages on
   top of that one physics evaluation — :meth:`_predict` is the single code path doc 04 §9
   requires, and both :meth:`forward` and :meth:`observe` are one call into it.

## Heterodyne detection and what it means for the noise model

Doc 02 §8.1: the 40 MHz AOM offset (IF-L2) "makes the detection heterodyne, which recovers
the sign of the phase and rejects low-frequency amplitude drift". Two consequences for
this module, both about what is *not* here rather than what is: the phase this channel
reports is signed (no ``|phi|`` ambiguity to model), and no separate low-frequency
amplitude-drift noise term is added — IF-L2 is why not, and the vibration term that *is*
modelled (:mod:`vpl.instruments.interferometry.noise`) is the residual *phase* noise the
heterodyne scheme does not reject, not amplitude drift.

## The likelihood: a correlated Gaussian, not a diagonal one — doc 05 §3.1

Doc 05 §3.1's table is explicit: interferometry gets "Gaussian, with a **correlated**
noise covariance... vibration noise is coloured, not white — a diagonal covariance would
overstate the information." The reason connects directly to the vibration model: all 8
chords share the same floated optical table (doc 02 IF-G2), so the residual vibration
after common-path rejection moves every chord together. A diagonal covariance would let 8
chords average that shared error down as ``1/sqrt(8)``, which they cannot physically do.

:meth:`likelihood` builds ``Sigma = D + v v^T`` — ``D`` the independent per-chord detector
variance, ``v`` the common-mode vibration standard deviation broadcast identically across
chords — and applies the rank-one Sherman-Morrison identity to invert it in ``O(n)``,
exactly the pattern :class:`~vpl.instruments.oes.instrument.OesInstrument.likelihood`
uses for its own (differently-motivated) rank-one discrepancy term; see that module's
docstring for the derivation this one specialises to ``k = 1``. Unlike OES's optional
``coherent_discrepancy`` parameter, the correlated structure here is not optional: doc 05
§3.1 makes it the channel's default statistics, not an add-on for a robustness study.
``obs.uncertainty`` is deliberately not read for this — a flat per-sample array has no way
to represent an off-diagonal correlation, so the covariance is reconstructed from
``obs.window.duration_s`` through the same vibration model :meth:`observe` draws its own
noise from, which is what keeps the two from silently disagreeing about how correlated the
channel actually is.

## What is deliberately not modelled

- **Beam refraction.** Doc 04 §5.2 calls for ray tracing rather than an assumed-straight
  chord; that is out of scope for this module, which integrates along a straight line.
  The consequence: in a steep density gradient the real beam samples a slightly different
  path than the nominal chord, which would bias the line integral this module returns.
  Not modelled, and not pretended to be.
- **Abel/tomographic inversion.** Doc 01 IF-5 names it as the way a radial profile would
  be recovered from several chords; :meth:`forward` and :meth:`observe` return one phase
  per chord and stop there, matching doc 02 §8.3's "no intrinsic z resolution" — inverting
  the chord set into a profile is downstream work in ``vpl-inverse``, not this module's.
- **Amplitude noise and low-frequency drift** — see the heterodyne paragraph above.

## Citations

See :mod:`vpl.instruments.interferometry.phase` and
:mod:`vpl.instruments.interferometry.noise` for the citations behind the physics and noise
models this instrument wraps; :meth:`metadata` republishes the full set doc 00 C2 requires.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

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
from vpl.core.units import Q_
from vpl.instruments.interferometry import noise
from vpl.instruments.interferometry.phase import (
    CHAMBER_DIAMETER_M,
    CHORD_SPACING_M,
    CO2_WAVELENGTH_M,
    N_CHORDS,
    PHASE_RESOLUTION_RAD,
    detection_floor_n_e_per_m3,
    net_phase_shift_rad,
)

__all__ = ["INTERFEROMETRY_INSTRUMENT_ID", "PHASE_UNITS", "InterferometryInstrument"]

type FloatArray = NDArray[np.float64]

#: The doc 08 §7 artifact group key.
INTERFEROMETRY_INSTRUMENT_ID: Final[InstrumentId] = "interferometry"

#: Units of what :meth:`InterferometryInstrument.forward` and :meth:`observe` return.
#: This channel reports a signed phase (doc 02 §8.1's heterodyne detection recovers the
#: sign), never a density — see the module docstring's "honesty point".
PHASE_UNITS: Final[str] = "rad"

#: Which grid point the chord ladder starts at, in metres from the wall. Doc 02's port
#: table (§4.1) fixes the chords' azimuth and elevation but not their z-height relative
#: to whatever domain a given solver's z-grid spans; anchoring at the wall is the most
#: conservative default — see the module docstring.
_DEFAULT_START_Z_M: Final[float] = 0.0

#: The quantity name :meth:`InterferometryInstrument.calibrate` looks up. The absolute
#: scale of a phase measurement is set by the AOM frequency reference and path-length
#: metrology chain (doc 02 §8.1), not by anything radiometric, so this is a distinct
#: calibration quantity from the other channels' — see doc 01 SYS-3's "single documented
#: reference source per chain".
_PHASE_SCALE_QUANTITY: Final[str] = "interferometer_phase_scale"


def _as_float(value: object, *, default: float, key: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number, got {value!r}")
    return float(value)


def _time_averaged_values(field: ScalarField, w: AcquisitionWindow) -> FloatArray:
    """The field's ``z``-profile, averaged over the acquisition window's overlapping
    time samples — or, if none overlap, the single nearest sample to the window's centre.

    A steady field has no time axis and is returned as-is. This mirrors
    :class:`~vpl.instruments.oes.instrument.OesInstrument`'s window-sample handling, but
    averages the field directly rather than an integrated radiance, since there is no
    intervening emission model here for a window integral to run through.
    """
    if field.is_steady:
        return field.values
    assert field.time is not None  # doc 03 §5.2: non-steady implies a time grid.
    times = field.time.t_s
    inside = np.flatnonzero((times >= w.start_s) & (times <= w.start_s + w.duration_s))
    if inside.size == 0:
        nearest = int(np.argmin(np.abs(times - (w.start_s + 0.5 * w.duration_s))))
        return np.asarray(field.values[nearest, :], dtype=np.float64)
    return np.asarray(field.values[inside, :].mean(axis=0), dtype=np.float64)


class InterferometryInstrument:
    """The heterodyne Mach-Zehnder CO2 interferometer of doc 02 §8 — WBS 2.8.

    Mutable, because :meth:`configure` and :meth:`calibrate` are declared as mutating in
    doc 08 §4 and returning a new instrument from them would leave the manifest and the
    object it configured pointing at different things — the same convention
    :class:`~vpl.instruments.oes.instrument.OesInstrument` and
    :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe` follow.
    """

    __slots__ = (
        "_applied_state",
        "_calibration",
        "_calibration_rng",
        "_chord_length_m",
        "_fringe_jump_rate",
        "_noise_enabled",
        "_rng",
        "_start_z_m",
        "_wavelength_m",
        "instrument_id",
    )

    def __init__(
        self, *, root_seed: int, instrument_id: InstrumentId = INTERFEROMETRY_INSTRUMENT_ID
    ) -> None:
        self.instrument_id = instrument_id
        self._chord_length_m: float | None = None
        self._wavelength_m: float | None = None
        self._start_z_m: float | None = None
        self._noise_enabled: bool | None = None
        self._fringe_jump_rate: float | None = None
        self._calibration: Calibration | None = None
        self._applied_state = CalibrationState.ESTIMATED
        # Vibration and fringe-jump noise are mechanical/electronic, not photon
        # statistics, so this draws on Stream.DETECTOR_NOISE (doc 10 §5) — the same
        # convention LangmuirProbe uses for its own amplifier/digitiser noise.
        self._rng = generator(root_seed, Stream.DETECTOR_NOISE)
        self._calibration_rng = generator(root_seed, Stream.CALIBRATION)

    # ── configuration ───────────────────────────────────────────────────────────

    def configure(self, cfg: InstrumentConfig) -> None:
        """Apply one entry of the manifest's ``instruments:`` list — doc 08 §6.

        ``chord_length_m``
            Overrides :data:`~vpl.instruments.interferometry.phase.CHAMBER_DIAMETER_M`.
            Doc 02 §8.2 IF-P2 ties this to the chamber diameter; it is configurable so a
            sensitivity study can vary it (doc 02 §8.2's own worked comparison against
            the 0.1 m chord doc 01 §5.4 originally assumed).
        ``wavelength_m``
            Overrides :data:`~vpl.instruments.interferometry.phase.CO2_WAVELENGTH_M`.
            Doc 01 IF-2 makes 10.6 micron mandatory for the deployed design; configurable
            here only so the doc 01 §5.4 HeNe-vs-CO2 comparison can be reproduced.
        ``start_z_m``
            Where the 8-chord ladder begins, in metres from the wall. See the module
            docstring for why the wall is the default anchor.
        ``noise``
            Whether :meth:`observe` applies vibration, independent detector noise,
            fringe jumps, and the stored calibration scale (see :meth:`calibrate`). All
            four are one switch, matching
            :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe`'s
            convention that "noise off" means the uncalibrated, noiseless model exactly.
            With noise enabled, :meth:`calibrate` (or :meth:`use_true_calibration`) must
            have already run — see :meth:`observe`'s docstring for why that gate is on
            this side rather than the noise-off side.
        ``fringe_jump_rate``
            Per-chord, per-observation probability of a
            :data:`~vpl.instruments.interferometry.noise.FRINGE_JUMP_MAGNITUDE_RAD` event
            (doc 04 §5.2). Only sampled when ``noise`` is enabled.
        """
        self._chord_length_m = _as_float(
            cfg.get("chord_length_m"), default=CHAMBER_DIAMETER_M, key="chord_length_m"
        )
        self._wavelength_m = _as_float(
            cfg.get("wavelength_m"), default=CO2_WAVELENGTH_M, key="wavelength_m"
        )
        self._start_z_m = _as_float(
            cfg.get("start_z_m"), default=_DEFAULT_START_Z_M, key="start_z_m"
        )
        self._noise_enabled = bool(cfg.get("noise", True))
        self._fringe_jump_rate = _as_float(
            cfg.get("fringe_jump_rate"),
            default=noise.DEFAULT_FRINGE_JUMP_RATE,
            key="fringe_jump_rate",
        )

    @property
    def chord_length_m(self) -> float:
        return self._require_configured("chord_length_m", self._chord_length_m)

    @property
    def wavelength_m(self) -> float:
        return self._require_configured("wavelength_m", self._wavelength_m)

    def chord_positions_m(self) -> FloatArray:
        """The 8 fixed chord heights of doc 02 §8.2 IF-G1, 5 mm apart."""
        start = self._require_configured("start_z_m", self._start_z_m)
        return start + CHORD_SPACING_M * np.arange(N_CHORDS, dtype=np.float64)

    def _require_configured(self, name: str, value: float | None) -> float:
        if value is None:
            raise RuntimeError(
                f"{self.instrument_id}: configure() must be called before use. doc 08 §6 "
                f"gives every instrument a manifest block; {name} has no value until it "
                "has run."
            )
        return value

    def _require_noise_enabled(self) -> bool:
        if self._noise_enabled is None:
            raise RuntimeError(f"{self.instrument_id}: configure() must be called before use.")
        return self._noise_enabled

    def _require_fringe_jump_rate(self) -> float:
        if self._fringe_jump_rate is None:
            raise RuntimeError(f"{self.instrument_id}: configure() must be called before use.")
        return self._fringe_jump_rate

    # ── calibration ─────────────────────────────────────────────────────────────

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        """Derive the phase-measurement scale from a reference standard — doc 04 §7.3.

        Unlike OES's radiometric response or the probe's current-measurement gain, the
        quantity being calibrated here is the AOM frequency reference / path-length
        metrology chain (doc 02 §8.1) — see :data:`_PHASE_SCALE_QUANTITY`.
        """
        standard = refs.for_quantity(_PHASE_SCALE_QUANTITY)
        uncertainty = standard.relative_uncertainty
        scale = 1.0 + uncertainty * float(self._calibration_rng.standard_normal())

        self._calibration = Calibration(
            instrument_id=self.instrument_id,
            coefficients={"phase_scale": scale},
            relative_uncertainty={"phase_scale": uncertainty},
            state=CalibrationState.ESTIMATED,
            reference=standard.name,
        )
        return self._calibration

    def use_true_calibration(self) -> None:
        """Apply the true (unit) phase scale instead of a drawn estimate — doc 04 §7.3.

        The inverse crime, committed on purpose: doc 04 §7.3 permits it for the deliberate
        verification runs doc 07 describes, and the resulting
        :class:`~vpl.core.state.CalibrationState` on every :meth:`observe` result records
        that it was done, the same convention
        :class:`~vpl.instruments.oes.instrument.OesInstrument` follows.
        """
        self._applied_state = CalibrationState.TRUE

    # ── the shared code path — doc 04 §9 ────────────────────────────────────────

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        """The noiseless phase per chord the likelihood compares against — doc 04 §9."""
        return Observable(
            instrument_id=self.instrument_id,
            values=self._predict(state, w),
            units=PHASE_UNITS,
            window=w,
        )

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        """A noisy, imperfectly-calibrated phase per chord — doc 04 §9.

        **With noise disabled**, this returns exactly what :meth:`forward` returns
        (values identical bit for bit), and calibration is not required first: doc 04 §9's
        noiseless path is a check on the physics alone, and gating it on a calibration
        chain that has nothing to do with that check would be an arbitrary obstacle to the
        one call this project uses most for verification (``test_...`` files across this
        package call ``forward``/``observe(noise=False)`` far more than they run a full
        calibrated pipeline). The reported uncertainty always reflects the channel's
        modelled noise level regardless, since a real pipeline's uncertainty estimate does
        not know it was handed a noiseless verification run.

        **With noise enabled**, :meth:`calibrate` must have already run (or
        :meth:`use_true_calibration` must have been called) — the opposite choice from the
        noiseless path, and deliberately so: doc 04 §7.3 states the chain as "true
        instrument response -> calibration measurement -> estimated response -> applied by
        the analysis pipeline", and *simulating a measurement* (which is what enabling
        noise means) with no calibration measurement to draw the estimated response from
        would apply a response that chain never produced. This is
        :class:`~vpl.instruments.oes.instrument.OesInstrument`'s convention, chosen here
        over :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe`'s more
        permissive one because the stored scale must be reused verbatim across every call
        (see below) — there is no defensible "default" scale to redraw from once redrawing
        is off the table.

        The applied scale is read from :attr:`_calibration` — set once by
        :meth:`calibrate` — rather than drawn fresh on every call. This is not a style
        choice: doc 06 §4.1 states that a correlated calibration error "affects *every*
        [...] point identically and does **not** average down", and a fresh draw per
        ``observe`` call would let repeated observations do exactly that, understating the
        one systematic doc 02 §8.2 IF-G2 calls dominant.
        ``test_interferometry_instrument.py``'s
        ``TestCalibrationScaleDoesNotAverageDown`` is the regression guard.
        """
        predicted = self._predict(state, w)
        n_chords = predicted.size

        sigma_common = noise.vibration_phase_std_rad(w.duration_s)
        sigma_independent = noise.independent_phase_std_rad()
        uncertainty = np.full(
            n_chords, math.sqrt(sigma_common**2 + sigma_independent**2), dtype=np.float64
        )

        if not self._require_noise_enabled():
            return Measurement(
                instrument_id=self.instrument_id,
                values=predicted,
                uncertainty=uncertainty,
                units=PHASE_UNITS,
                window=w,
                calibration=CalibrationState.ESTIMATED,
            )

        if self._applied_state is CalibrationState.TRUE:
            # doc 04 §7.3's deliberate inverse crime: the unit scale, not a draw.
            scale = 1.0
        else:
            if self._calibration is None:
                raise RuntimeError(
                    f"{self.instrument_id}: calibrate() must be called before observe() "
                    "with noise enabled (or use_true_calibration() for a deliberate "
                    "doc 04 §7.3 verification run). Noise off needs no calibration chain; "
                    "noise on simulates a measurement, and doc 04 §7.3's chain has "
                    "nothing to apply until a calibration measurement has produced it."
                )
            scale = self._calibration.coefficients["phase_scale"]
        common_draw = float(self._rng.normal(0.0, sigma_common)) if sigma_common > 0.0 else 0.0
        independent_draw = self._rng.normal(0.0, sigma_independent, size=n_chords)
        jumps = noise.sample_fringe_jumps(
            self._rng, n_chords=n_chords, rate=self._require_fringe_jump_rate()
        )

        values = scale * predicted + common_draw + independent_draw + jumps
        return Measurement(
            instrument_id=self.instrument_id,
            values=values,
            uncertainty=uncertainty,
            units=PHASE_UNITS,
            window=w,
            calibration=self._applied_state,
        )

    def _predict(self, state: PlasmaState, w: AcquisitionWindow) -> FloatArray:
        """The one physics evaluation both :meth:`forward` and :meth:`observe` share."""
        chord_z = self.chord_positions_m()
        field = state.field("n_e")
        n_e_profile = _time_averaged_values(field, w)
        n_e_per_chord = np.interp(chord_z, state.grid.z_m, n_e_profile)

        return np.asarray(
            net_phase_shift_rad(
                n_e_per_m3=n_e_per_chord,
                n_neutral_per_m3=state.params.n_g_per_m3,
                wavelength_m=self.wavelength_m,
                chord_length_m=self.chord_length_m,
            ),
            dtype=np.float64,
        )

    # ── the likelihood and the gate ─────────────────────────────────────────────

    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb:
        """The correlated-Gaussian term of doc 05 §3.2 — see the module docstring.

        Builds ``Sigma = D + v v^T`` (``D`` independent per-chord variance, ``v`` the
        common-mode vibration standard deviation broadcast across chords) and applies
        the rank-one Sherman-Morrison identity:

            r' Sigma^-1 r  =  r'D^-1 r - (v'D^-1 r)^2 / (1 + v'D^-1 v)
            log|Sigma|     =  log|D| + log(1 + v'D^-1 v)

        the ``k = 1`` case of the general Woodbury identity
        :class:`~vpl.instruments.oes.instrument.OesInstrument.likelihood` documents in
        full for its own rank-``k`` discrepancy term.
        """
        if obs.shape != pred.shape:
            raise ValueError(
                f"an observation and its prediction must have the same shape, got "
                f"{obs.shape} and {pred.shape}"
            )
        if obs.units != pred.units:
            raise ValueError(f"observation is in {obs.units!r} and prediction in {pred.units!r}")

        residual = np.asarray(obs.values, dtype=np.float64) - np.asarray(
            pred.values, dtype=np.float64
        )
        n = residual.size

        sigma_independent = noise.independent_phase_std_rad()
        sigma_common = noise.vibration_phase_std_rad(obs.window.duration_s)
        d = np.full(n, sigma_independent**2, dtype=np.float64)
        v = np.full(n, sigma_common, dtype=np.float64)

        d_inv_r = residual / d
        d_inv_v = v / d
        quad_diag = float(residual @ d_inv_r)
        vt_dinv_v = float(v @ d_inv_v)
        vt_dinv_r = float(v @ d_inv_r)
        m = 1.0 + vt_dinv_v
        quadratic = quad_diag - (vt_dinv_r**2) / m

        log_det = float(np.sum(np.log(d))) + math.log(m)
        return float(-0.5 * quadratic - 0.5 * log_det - 0.5 * n * math.log(2.0 * math.pi))

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        """The doc 01 IF-6 gate: whether ``n_0`` is at or above this chord's floor."""
        return self.metadata().detection_floor.admits(state_guess.n_0)

    def metadata(self) -> InstrumentMetadata:
        """Identity, citations and the detection floor — doc 00 C2, doc 01 IF-6."""
        floor = detection_floor_n_e_per_m3(
            phase_resolution_rad=PHASE_RESOLUTION_RAD,
            wavelength_m=self.wavelength_m,
            chord_length_m=self.chord_length_m,
        )
        return InstrumentMetadata(
            instrument_id=self.instrument_id,
            name="Heterodyne Mach-Zehnder CO2 (10.6 um) interferometer, 8 chords (doc 02 §8)",
            version="0.1.0",
            citations=(
                Citation(
                    key="hutchinson-2002",
                    reference=(
                        "I. H. Hutchinson, Principles of Plasma Diagnostics, 2nd ed., "
                        "Cambridge University Press (2002), ch. 6"
                    ),
                ),
                Citation(
                    key="veron-1979",
                    reference=(
                        "D. Veron, 'Submillimeter Interferometry of High-Density "
                        "Plasmas', in Infrared and Millimeter Waves, Vol. 2, ed. K. J. "
                        "Button, Academic Press (1979), pp. 69-135"
                    ),
                ),
                Citation(
                    key="dalgarno-kingston-1960",
                    reference=(
                        "A. Dalgarno and A. E. Kingston, 'The refractive indices and "
                        "Verdet constants of the inert gases', Proc. R. Soc. Lond. A "
                        "259 (1960) 424"
                    ),
                    doi="10.1098/rspa.1960.0237",
                ),
                Citation(
                    key="peck-fisher-1964",
                    reference=(
                        "E. R. Peck and D. J. Fisher, 'Dispersion of argon', J. Opt. "
                        "Soc. Am. 54 (1964) 1362"
                    ),
                    doi="10.1364/JOSA.54.001362",
                ),
            ),
            detection_floor=DetectionFloor(
                quantity="n_0", threshold=Q_(floor, "m**-3"), requirement="IF-6"
            ),
            description=(
                "Line-integrated electron-density phase shift (doc 04 §5.1) across 8 "
                "fixed chords (doc 02 §8.2 IF-G1); no z-resolution within a chord "
                "(doc 02 §8.3). Gaussian likelihood with a correlated (common-mode "
                "vibration) covariance, doc 05 §3.1."
            ),
        )

    def __repr__(self) -> str:
        return (
            f"InterferometryInstrument({self.instrument_id!r}, "
            f"{N_CHORDS} chords x {CHORD_SPACING_M:.4g} m spacing)"
        )
