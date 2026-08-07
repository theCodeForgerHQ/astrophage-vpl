"""The CO2 heterodyne interferometry diagnostic channel — doc 04 §5.1, doc 05 §3.2, doc 02
§8.3, doc 11 §9 item 3.

## Why this channel, on top of OES and LIF

The project's honest end-to-end error is 36.5 % against a +/-3.11 % credible interval — the
truth sits outside the interval, so doc 00 §5.1 criterion S4 fails. The diagnosed root cause
(see :mod:`vpl.instruments.coherent`'s module docstring and
``packages/vpl-instruments/src/vpl/instruments/oes/instrument.py``'s calibration section) is
a calibration error scored as ``N`` independent pixel errors instead of one coherent draw —
fixed there, in ``vpl.instruments.coherent``. But fixing the *arithmetic* exposes the
*physics*: with an honest, un-overweighted 6 % calibration uncertainty, a single OES line
cannot identify plasma density at all. Overall brightness carries all of OES's density
information, and "the plasma is 6 % denser" is observationally identical to "my lamp reads
6 % bright" — the amplitude-scale degeneracy doc 05 §6.2 predicts algebraically
(``Gamma_i ~ n_0 sqrt(T_e)``, and OES's own radiometric response is exactly one more
multiplicative unknown riding along the same axis).

Interferometry breaks that because it is not radiometric at all. Doc 04 §5.1's phase shift,

    Delta phi = r_e lambda INTEGRAL n_e dl,

depends on electron density **alone** — there is no ``T_e`` in it, and its calibration chain
(doc 02 §8.1: an AOM frequency reference and a path-length metrology standard) shares no
component with a radiometric lamp. A calibration error on this channel cannot masquerade as
"the plasma is denser" the way OES's can, because this channel's scale has nothing to do
with how bright anything looks.

## The measured finding that forced a reframing, not a re-tuning

Wiring the doc 02 §8.2 IF-G1 hardware literally — 8 chords, 5 mm apart, starting at the
wall, sampling whatever ``n_e(z)`` field the solver produced — against the closed loop's
fixed observation grid (``vpl.experiment.closed_loop._fixed_spatial_grid``, sized to two
sheath thicknesses at RP-1: **z = 0 to z = 2.28 mm**, 11 points) put only the first chord
(z = 0) inside that grid. The other seven, at z = 5, 10, ..., 35 mm, sit past its outer edge,
where ``numpy.interp`` clamps to the last grid point's density rather than extrapolating.
Measured directly: seven of the eight chords reported the identical clamped value, so a
channel doc 02 §8.2 IF-G1 specifies as 8 independent line integrals degenerated into one
interior reading plus seven repeats of it — and, measured end to end, adding that channel to
the four-channel fusion *widened* the credible interval (+/-0.51 % to +/-2.29 %) rather than
narrowing it. Information cannot do that; only mis-modelled information can.

Two ways to respond to a measured pathology exist: retune the hardware numbers (move
``start_z_m``, shrink the 5 mm spacing) until the grid mismatch looks better, or ask whether
the *model* — not the hardware — was wrong. This module takes the second path, because the
first would be exactly the kind of tuning-to-a-result doc 00 §6 calls parameter fog, and
because doc 02 §8.3 already says, in so many words, what this channel actually is: "no
intrinsic z resolution — it cannot see the sheath structure at all... interferometry
constrains the *boundary condition* of the sheath problem, not the sheath... the inversion
must not be allowed to over-weight it." A diagnostic with no z-resolution was never a sheath
sampler to begin with; the 8-chord ladder against a 2.28 mm sheath-scale grid was the wrong
model for what doc 02 §8.3 itself already describes.

## What this module measures instead: the bulk density, integrated once (revised, Block F)

In real laboratory practice a line-integrated interferometer is a **bulk-plasma**
diagnostic. The sheath is thin (0.89 mm at RP-1, doc 02 §3.3) and electron-depleted —
precisely where the electrons this channel is sensitive to are *not* — so a real beam
crossing the discharge picks up almost none of its phase from the sheath and almost all of
it from the bulk. That is exactly what doc 02 §8.3's "constrains the boundary condition...
not the sheath" already says, and it motivates measuring one number, not eight — the
8-chord-ladder-to-one-line-integral reframing above is unchanged by what follows.

**What did change (recovery plan, defect #2, "Block F"): the density source.** The version
of this module that shipped after the reframing above read ``state.params.n_0`` — the
solver's own *boundary-condition parameter* — rather than the ``n_e`` field the solver
actually computed. That was measured to be a genuine bug, not a simplification: it made
this channel provably blind to the density field. Halving ``state.field("n_e")`` while
leaving ``n_0`` untouched changed this channel's reading by exactly zero bits, and a
model-discrepancy estimator built to compare this channel's prediction against a truth's
own field measured that divergence as identically zero on every draw, because neither side
of that comparison had ever touched the field. See
``TestObservableReadsTheFieldNotJustTheParameter.
test_halving_the_n_e_field_materially_changes_the_reading`` in
``test_channels_interferometry.py`` for the checked regression.

:class:`_BulkInterferometer` now computes a genuine chord-averaged line integral,
:meth:`~_BulkInterferometer._line_averaged_density_per_m3`, from ``state.field("n_e")`` and
``state.grid``, in two pieces:

1. A trapezoidal quadrature (:func:`numpy.trapezoid`) of the solver's own ``n_e(z)`` over
   the resolved near-wall domain (``state.grid.z_m`` — two sheath thicknesses at RP-1,
   z = 0 to 2.28 mm, doc 02 §3.3).
2. The unresolved remainder of the 400 mm chord, held at the resolved domain's own
   outermost (bulk-most, least-depleted) sample, ``n_e(z_max)`` — the best density estimate
   this project's 1-D sheath-normal solvers have for anything past the point they actually
   resolve.

That chord average is what feeds
:func:`~vpl.instruments.interferometry.phase.net_phase_shift_rad` as ``n_e_per_m3`` —
algebraically identical to the formula's existing "uniform density times chord length"
structure, just fed a genuine average instead of a parameter. The neutral-gas correction
(doc 04 §5.2) is unchanged: ``state.params.n_g_per_m3`` never depended on either the old
chord ladder or the ``n_0`` parameter read.

**The chord length is machine geometry, not a fitted number.** ``L`` is
:data:`~vpl.instruments.interferometry.phase.CHAMBER_DIAMETER_M`, the 400 mm chamber
diameter doc 02 §8.2 IF-P2 already names as "the full 400 mm chamber diameter, used as the
default chord length... a design decision made because of a requirements calculation" (see
that module's docstring). It is a ``Final`` constant in the unedited
:mod:`vpl.instruments.interferometry.phase`, not a registry entry and not invented here —
this module reads it, it does not choose it. There is nothing else in doc 02 §8 for a chord
length to be; a beam crossing a cylindrical chamber travels the chamber's own diameter.

## The bulk-extension assumption — stated plainly, and now expected to cost accuracy

Reporting the line-averaged density above still assumes the unresolved remainder of the
chord sees a uniform density, and now states explicitly which density that is: the
resolved domain's own edge sample, not the undepleted parameter ``n_0``.

1. **Transverse uniformity along the beam's own path** was already assumed by the
   (unedited) :func:`~vpl.instruments.interferometry.phase.electron_phase_shift_rad`: its
   own docstring states plainly that ``INTEGRAL n_e dl`` "collapses to ``n_e L``" for any
   single density fed to it, because the plasma solvers this project carries are
   one-dimensional in the sheath-normal coordinate ``z`` and have no information about how
   density varies along the chord's own transverse axis. This module inherits that
   assumption unchanged.
2. **The genuinely new cost**: the resolved domain (``_DOMAIN_SHEATHS = 2.0`` in
   ``vpl.experiment.closed_loop``) is sized to resolve the *sheath*, not to reach
   field-free bulk plasma — its outermost sample is the **presheath edge** density,
   ``n_s = h_l n_0`` (doc 03 §2.1's Godyak ratio, nominally 0.61 —
   :mod:`vpl.physics.analytic.sheath`'s module docstring), not the undepleted bulk value
   ``n_0`` a beam genuinely far from the wall would see. Extending that edge sample across
   the remaining ~99 % of the 400 mm chord therefore carries roughly 39 % of presheath
   depletion across nearly the whole chord, instead of recovering ``n_0`` exactly the way
   reading the parameter did by construction.

**This is why Block F is expected to make this channel's reading — and any fused result
that leans on it — measurably worse, not better**, and why that is the correct outcome
rather than a regression to fix away: the previous behaviour was right for the wrong
reason (both sides of the inverse problem read the identical scalar, so they could never
disagree about it), and this behaviour is a real, checkable computation off a field a
truth and a trial can genuinely disagree about — including, now, disagreeing for reasons
(presheath depletion baked into where the grid stops) that have nothing to do with either
side being "wrong" about ``n_0`` at all. Nothing here attempts to correct that depletion
bias — doing so honestly would need either a bulk-reaching grid or a stated presheath
model to extrapolate through, and inventing one to make this channel's numbers look better
would be exactly the parameter fog doc 00 §6 warns against.

## Is this an inverse crime? Doc 05 §7's sense, checked rather than assumed

Doc 05 §7 names the inverse crime as *the same model and discretisation* generating and
inverting synthetic data, so a numerical error the model makes is invisible because both
sides make it identically. That is not what is happening here, and the reason is that the
physics genuinely is this simple: a beam crossing a chord of known length picks up a phase
that is *exactly* linear in the average density it crosses (doc 04 §5.1's
``r_e lambda n_e L``, not a discretised numerical solve of anything), and truth and
inversion agree about that relationship for the same reason two calculators agree that
``2 * 2 = 4`` — because reality agrees about it, not because the harness is marking its own
homework. Four places this module could plausibly hide a crime, checked directly:

- **The chord length and wavelength are shared, unrandomised constants on both sides.** Is
  that a crime? No: doc 02 §8.2 states both as fixed hardware specifications (a machined
  chamber dimension, a mandated CO2 laser line), not calibrated quantities with a meaningful
  uncertainty the way the phase *scale* is — the task that motivated this reframing is
  explicit that "the chord length is machine geometry... not a fitted parameter", and
  sharing a genuinely fixed, non-uncertain constant between the state that generates data and
  the model that inverts it is not committing the doc 05 §7 crime; it is correctly modelling
  a quantity that has no meaningful uncertainty to disagree about.
- **``n_e``'s field values differ between the truth state and a trial state**, and the
  reported phase is a strictly monotonic function of the resulting chord average. So truth
  and inversion *can* and routinely *do* disagree — that disagreement is the entire
  likelihood signal :class:`TestDensitySensitivityAndTemperatureBlindness` in
  ``test_channels_interferometry.py`` measures, and it is now a disagreement about a field,
  not just a scalar (see the paragraph below on why that distinction matters for an L2
  truth).
- **The calibration draw and noise are genuinely one-sided**: :meth:`_BulkInterferometer.
  calibrate` draws a random phase-scale estimate only on the truth-side instrument (doc 04
  §7.3's chain); the inversion-side instrument always calls
  :meth:`~_BulkInterferometer.use_true_calibration` for its *predictions* and separately
  records the standard's *uncertainty* only so ``likelihood(calibration_uncertainty=True)``
  can score it as a systematic — see :class:`InterferometryChannel`'s own docstring. The two
  sides read different random draws from different :class:`~vpl.core.random.Stream` objects,
  never the same number.
- **The averaging formula itself (trapezoid over the resolved domain, then hold the edge
  value) is shared, unrandomised code on both sides.** Is *that* a crime? Argued below, in
  its own section, because the answer is different depending on which fidelity level
  produced the truth, and the pre-Block-F version of this docstring did not distinguish
  them.

## Why sharing the averaging formula is not a crime for L0/L1 truths, but an L2 truth differs

The pre-Block-F version of this argument stopped at "``n_0`` differs between the truth and
a trial, so truth and inversion can disagree" and treated that as settling the question.
It does settle it for **L0 and L1 truths**: those levels are generated by the same closed
family of equations (Boltzmann electrons over an analytically or numerically solved
potential profile) that the inversion's own forward model assumes, so a truth ``n_e(z)``
and a trial ``n_e(z)`` built from different ``theta`` values disagree for the ordinary,
intended reason — different parameters, same equations — and the averaging formula reduces
that disagreement to a phase difference without adding or hiding an error either state's
own solver did not already make on its own terms. Two states from the *same* solver family
computing the *same* line-integral formula over their own two different fields is not the
doc 05 §7 crime; it is the correctly-modelled case doc 05 §7 was always written to permit.

**It is not obviously sound for an L2 (particle) truth, and this docstring did not say so
before Block F.** An L2 state's ``n_e(z)`` comes from a kinetic/particle solver that need
not obey the Boltzmann relation ``n_e = n_s exp(Phi / T_e)`` the L0/L1 forward model
assumes at all — non-Maxwellian tails, finite-sampling noise, and genuine kinetic
structure near the sheath edge can all put an L2 truth's actual field somewhere the
inversion's own model family cannot reach for *any* choice of ``theta``. Before Block F,
none of that mattered to this channel, because both the L2 truth and every trial
collapsed onto the identical ``theta.n_0`` scalar the moment ``_predict`` ran — the
particle solver's own field, and however it disagreed with the Boltzmann assumption, was
computed and then discarded unread. That is precisely the failure mode this docstring's
"is this an inverse crime" section could not have ruled out for L2, because it never
described a code path where an L2 truth's field was read at all: an argument that the
*parameter* comparison was fair said nothing about whether the *field* comparison — the
one this channel now actually performs — is equally fair, and it is not guaranteed to be.
What Block F buys is not a proof that this channel is unbiased against an L2 truth; it is
that the channel is now capable of being *wrong about* an L2 truth in a way a discrepancy
estimator can measure, rather than structurally incapable of seeing the disagreement at
all. See the recovery plan's Block G for the companion effort on characterising that
model-discrepancy term from real solver output rather than assuming it away.

So the honest residual here is the bulk-extension assumption above plus, for an L2 truth
specifically, the open question this section states rather than resolves — smaller in
scope than, and different in kind from, the doc 05 §7 crime, and recorded here rather than
hidden inside a covariance that happens to look tight.

## Scoring the calibration coherently — doc 11 §9's amendment, doc 06 §4.1

:meth:`_BulkInterferometer.likelihood` builds a correlated covariance,
``Sigma = D + v v^T`` for the common-mode *vibration* term (doc 05 §3.1), reusing
:mod:`vpl.instruments.coherent`'s shared Woodbury kernel exactly as the pre-reframing
version of this module did — with ``n = 1`` now rather than ``n = 8``, since there is only
one observable per acquisition. With one sample, "correlated across chords" has nothing left
to describe (there is only one chord), but the same kernel is still the right tool: it is
what lets the vibration term and, when requested, the phase-scale calibration term
(``calibration_uncertainty=True``, doc 06 §4.1's coherent-systematic treatment) compose
without a second, hand-derived path that could disagree with the multi-chord one this
project already validated elsewhere.

**Why the inversion instrument is calibrated at all, when doc 04 §7.3 says only the truth
side should be.** ``IF.phase_scale_uncertainty`` (ASSUMED, 0.03 — doc 02 §11 quotes 3 % as
"vibration-limited", already double-counting the separately-modelled vibration term, so 0.03
is used as the conservative stand-in rather than an invented tighter figure) is a registry
entry, but ``InterferometryInstrument``-style channels have no ``calibrate()``-free path to
it the way OES and LIF's registry-backed likelihoods do (see the previous version of this
module's docstring, preserved in version control, for the fuller history of that gap). So the
inversion instrument is given :meth:`~_BulkInterferometer.calibrate` *and*
:meth:`~_BulkInterferometer.use_true_calibration`, in that order: ``calibrate()`` records the
standard's ``relative_uncertainty`` where ``likelihood(calibration_uncertainty=True)`` can
read it, and ``use_true_calibration()`` (which touches only ``observe()``'s applied state,
not ``forward()``, which never applies a scale at all) keeps the inversion side predicting
through the unit scale exactly as before.

:class:`_InterferometryWithOptionalCoherence` exists because
:class:`~vpl.inverse.fusion.JointLikelihood` calls ``instrument.likelihood(obs, pred)`` with
no keyword arguments — the same fixed two-positional-argument shape
:mod:`vpl.experiment.channels`'s own ``_LifOnReconstructedIvdf`` exists to adapt a per-call
transform onto. This pins ``calibration_uncertainty`` at construction time so fusion's fixed
call signature reaches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.params import ParameterRegistry
from vpl.core.protocols import (
    Calibration,
    CalibrationReference,
    CalibrationSet,
    Citation,
    DetectionFloor,
    InstrumentMetadata,
    LogProb,
)
from vpl.core.random import Stream, generator
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    Measurement,
    Observable,
    PlasmaParams,
    PlasmaState,
)
from vpl.core.units import Q_
from vpl.instruments.coherent import (
    coherent_gaussian_log_prob,
    fractional_calibration_row,
    stack_coherent_rows,
)
from vpl.instruments.interferometry import noise as if_noise
from vpl.instruments.interferometry.instrument import (
    INTERFEROMETER_PHASE_SCALE_QUANTITY,
    INTERFEROMETRY_INSTRUMENT_ID,
)
from vpl.instruments.interferometry.phase import (
    CHAMBER_DIAMETER_M,
    CO2_WAVELENGTH_M,
    PHASE_RESOLUTION_RAD,
    detection_floor_n_e_per_m3,
    net_phase_shift_rad,
)

__all__ = [
    "INTERFEROMETRY_CHANNEL",
    "InterferometryChannel",
    "build_interferometry_channel",
    "interferometry_acquisition_window",
    "interferometry_calibration_set",
]

type FloatArray = NDArray[np.float64]

#: Channel name — bound to the instrument's own stamped identifier rather than duplicated as
#: a second string literal, so an ablation table, a doc 08 §7 artifact group and a fusion
#: exclusion list cannot end up calling this channel two different things.
INTERFEROMETRY_CHANNEL: Final[str] = INTERFEROMETRY_INSTRUMENT_ID

#: Units of what :meth:`_BulkInterferometer.forward` and :meth:`observe` return. This
#: channel reports a signed phase (doc 02 §8.1's heterodyne detection recovers the sign),
#: never a density — see the module docstring's "honesty point" in
#: :mod:`vpl.instruments.interferometry.phase`.
PHASE_UNITS: Final[str] = "rad"

#: One observable per acquisition — the reframing's structural signature. The old 8-chord
#: ladder returned :data:`~vpl.instruments.interferometry.phase.N_CHORDS` samples per call;
#: a bulk-density line integral has exactly one number to report, because there is exactly
#: one boundary-condition density to report it about.
_N_SAMPLES: Final[int] = 1

#: One-sigma relative uncertainty of the interferometer's phase-scale calibration — the
#: AOM frequency reference and path-length metrology chain of doc 02 §8.1. Read from the
#: registry (``IF.phase_scale_uncertainty``, ASSUMED, 0.03) rather than restated as a
#: literal, so this module and the registry cannot silently disagree about the figure.
_PHASE_SCALE_UNCERTAINTY_ID: Final[str] = "IF.phase_scale_uncertainty"


def interferometry_calibration_set(registry: ParameterRegistry) -> CalibrationSet:
    """The one standard :meth:`_BulkInterferometer.calibrate` requires — doc 02 §8.1.

    What is being certified is the AOM frequency reference and the path-length metrology,
    not a radiometric response — there is no lamp in this channel and nothing about it
    depends on how bright the plasma looks. That independence is the point: the calibration
    error which was measured to make the OES amplitude scale unidentifiable cannot reach
    this measurement at all.
    """
    return CalibrationSet.of(
        CalibrationReference(
            name="AOM frequency reference + path-length metrology",
            quantity=INTERFEROMETER_PHASE_SCALE_QUANTITY,
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=float(
                registry.value_in(_PHASE_SCALE_UNCERTAINTY_ID, "dimensionless")
            ),
            traceable_to="doc 02 §8.1 frequency/length chain",
        )
    )


def interferometry_acquisition_window(registry: ParameterRegistry) -> AcquisitionWindow:
    """The window both the truth's noise draw and the inversion's covariance are built from.

    Not cosmetic: :meth:`_BulkInterferometer.observe` draws the common-mode vibration term
    from :func:`~vpl.instruments.interferometry.noise.vibration_phase_std_rad` evaluated at
    *this* window's duration, and :meth:`_BulkInterferometer.likelihood` rebuilds the same
    function at the duration recorded on the *observation* it is scoring. If the truth-side
    window this module hands to ``observe`` and the window the resulting
    :class:`~vpl.core.state.Measurement` carries were ever two different objects, the two
    sides would silently disagree about how correlated the channel is.

    Duration is ``IF.vibration_reference_window_s`` — the one window at which
    :mod:`vpl.instruments.interferometry.noise`'s own module docstring says the
    vibration-variance calibration is anchored to doc 02 §8.2 IF-P1's quoted 0.1 mrad total.
    """
    return AcquisitionWindow.absolute(
        start=Q_(0.0, "s"), duration=registry.quantity("IF.vibration_reference_window_s")
    )


class _BulkInterferometer:
    """A single line-of-sight phase measurement of the bulk density — see the module
    docstring for the physics, the chord-length sourcing and the bulk-extension assumption
    (Block F) this trades the old 8-chord sheath sampler, and then the ``n_0``-parameter
    read, for.

    Deliberately does not delegate to
    :class:`~vpl.instruments.interferometry.instrument.InterferometryInstrument`: that
    class's ``_predict`` interpolates a solver's ``n_e(z)`` field onto an 8-position chord
    ladder, which is precisely the sheath-sampling behaviour the original reframing
    replaced, and it is not this module's file to edit. This class reuses the same
    *physics primitives* (:func:`~vpl.instruments.interferometry.phase.net_phase_shift_rad`,
    the doc 02 §8.2 hardware constants, the doc 04 §5.2 vibration and fringe-jump noise
    models, and :mod:`vpl.instruments.coherent`'s shared Woodbury kernel) with a different
    binding from state to density — post-Block-F, a chord-averaged integral of
    ``state.field("n_e")`` (:meth:`_line_averaged_density_per_m3`), not a single chord
    ladder and not the ``state.params.n_0`` scalar either.

    Mutable for the same reason
    :class:`~vpl.instruments.interferometry.instrument.InterferometryInstrument` is:
    :meth:`calibrate` is declared as mutating in doc 08 §4, and returning a new instrument
    from it would leave the manifest and the object it configured pointing at different
    things.
    """

    __slots__ = (
        "_applied_state",
        "_calibration",
        "_calibration_rng",
        "_chord_length_m",
        "_fringe_jump_rate",
        "_noise_enabled",
        "_registry",
        "_rng",
        "_wavelength_m",
        "instrument_id",
    )

    def __init__(
        self,
        *,
        root_seed: int,
        registry: ParameterRegistry,
        noise: bool,
        instrument_id: str = INTERFEROMETRY_INSTRUMENT_ID,
    ) -> None:
        self.instrument_id = instrument_id
        self._registry = registry
        # doc 02 §8.2 IF-P2: the chord length is the chamber diameter, fixed machine
        # geometry — see the module docstring. Not configurable here: unlike the old
        # chord-ladder anchor, there is no sensitivity study this module's caller needs to
        # run against a different chord length, and doc 08 §5's escape hatch for a named,
        # doc-cited `Final` constant is exactly what `CHAMBER_DIAMETER_M` already is.
        self._chord_length_m = CHAMBER_DIAMETER_M
        self._wavelength_m = CO2_WAVELENGTH_M
        self._noise_enabled = noise
        self._fringe_jump_rate = if_noise.DEFAULT_FRINGE_JUMP_RATE
        self._calibration: Calibration | None = None
        self._applied_state = CalibrationState.ESTIMATED
        # Vibration and fringe-jump noise are mechanical/electronic, not photon statistics
        # — the same convention
        # :class:`~vpl.instruments.interferometry.instrument.InterferometryInstrument` uses.
        self._rng = generator(root_seed, Stream.DETECTOR_NOISE)
        self._calibration_rng = generator(root_seed, Stream.CALIBRATION)

    # ── geometry, read-only ─────────────────────────────────────────────────────

    @property
    def chord_length_m(self) -> float:
        """The fixed doc 02 §8.2 IF-P2 chamber-diameter path length — never state-derived.

        ``TestChordGeometryDoesNotDependOnTheta`` in ``test_channels_interferometry.py`` is
        the checked guard that this stays true.
        """
        return self._chord_length_m

    @property
    def wavelength_m(self) -> float:
        return self._wavelength_m

    # ── calibration ─────────────────────────────────────────────────────────────

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        """Derive the phase-measurement scale from a reference standard — doc 04 §7.3."""
        standard = refs.for_quantity(INTERFEROMETER_PHASE_SCALE_QUANTITY)
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
        verification runs doc 07 describes.
        """
        self._applied_state = CalibrationState.TRUE

    # ── the shared code path — doc 04 §9 ────────────────────────────────────────

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        """The noiseless phase the likelihood compares against — doc 04 §9.

        Post-Block-F: reads ``state.grid`` and ``state.field("n_e")`` through
        :meth:`_line_averaged_density_per_m3`, not ``state.params.n_0`` — see the module
        docstring's "what this module measures instead" section for why, and its
        "bulk-extension assumption" section for the honest cost of doing so. This does
        *not* reintroduce the old 8-chord ladder's grid-domain mismatch (the "measured
        finding" section, still accurate for the pre-reframing history it describes):
        there is still exactly one chord and one reported value, and the resolved grid is
        used as a genuine average over its own domain rather than sampled at positions
        that can fall past its edge.
        """
        return Observable(
            instrument_id=self.instrument_id,
            values=self._predict(state),
            units=PHASE_UNITS,
            window=w,
        )

    def _predict(self, state: PlasmaState) -> FloatArray:
        """The bulk-density line integral both :meth:`forward` and :meth:`observe` share.

        Genuinely integrates :meth:`_line_averaged_density_per_m3`'s reading of
        ``state.field("n_e")`` over the chord — see that method and the module
        docstring's "what this module measures instead" section (post-Block-F revision)
        for why this reads the field rather than ``state.params.n_0`` and what that
        change costs.
        """
        predicted = np.asarray(
            net_phase_shift_rad(
                n_e_per_m3=self._line_averaged_density_per_m3(state),
                n_neutral_per_m3=state.params.n_g_per_m3,
                wavelength_m=self._wavelength_m,
                chord_length_m=self._chord_length_m,
            ),
            dtype=np.float64,
        )
        assert predicted.size == _N_SAMPLES, (
            "unreachable: net_phase_shift_rad given a scalar density always returns one "
            "value — see _N_SAMPLES"
        )
        return predicted

    def _line_averaged_density_per_m3(self, state: PlasmaState) -> float:
        """The chord-averaged density a genuine line integral over ``n_e(z)`` implies.

        Block F (doc: 2026-08-07 recovery plan, defect #2): the previous version of this
        method never read ``state.field("n_e")`` at all — it read ``state.params.n_0``, a
        scalar control parameter, so halving the *field* changed this channel's reading by
        zero bits, and a discrepancy estimator built to measure how far this channel's model
        diverges from a truth's own field measured that divergence as exactly zero, always,
        by construction. That is the doc 05 §7 inverse crime the module docstring's "is this
        an inverse crime" section argued against for the *parameter* ``n_0`` and was correct
        to: two independently drawn ``n_0`` scalars really do disagree. It did not
        contemplate a truth whose ``n_0`` and whose actual ``n_e`` field disagree with each
        other — which is exactly what an L2 (particle) truth can produce, and what a
        discrepancy sweep exists to characterise. Reading the parameter made that
        disagreement structurally invisible to this channel.

        This method instead computes ``(1 / L) INTEGRAL n_e dl`` over the physical chord,
        in two pieces:

        1. **The resolved near-wall domain** (``state.grid.z_m``, doc 02 §2's sheath-normal
           coordinate — two sheath thicknesses at RP-1, doc 02 §3.3): a trapezoidal
           quadrature of the solver's own ``n_e(z)`` samples, via :func:`numpy.trapezoid`.
        2. **The unresolved remainder** of the chord (``chord_length_m`` minus the resolved
           domain's own length — at RP-1, ~99.4 % of the 400 mm chamber diameter): no 1-D
           sheath-normal solver this project carries has anything to say about density that
           far from the wall, so this method holds it at the resolved domain's own outermost
           (bulk-most) sample, ``n_e(z_max)`` — the least-depleted density the solver
           actually computed, and the same "extend the last resolved value" convention
           :func:`numpy.interp`'s clamping already used for the old chord ladder, applied
           deliberately here rather than as :func:`numpy.interp`'s incidental edge behaviour.

        **The honest cost, stated plainly.** ``n_e(z_max)`` is the presheath-*edge* density
        ``h_l n_0`` (doc 03 §2.1's Godyak ratio, 0.61 nominal — see
        :mod:`vpl.physics.analytic.sheath`'s module docstring), not the undepleted bulk
        value ``n_0`` a beam far from any wall would actually cross. The resolved domain
        stops at the presheath edge by construction (``_DOMAIN_SHEATHS = 2.0`` in
        ``vpl.experiment.closed_loop``) — it was sized to resolve the sheath, not to reach
        genuinely field-free bulk — so extending with its own edge sample carries that
        ~39 % presheath depletion across the remaining 99+% of the chord instead of
        recovering the truly uniform ``n_0`` the old parameter read reported by
        construction. That is why this reframing is expected to make this channel's
        reading, and therefore fusion results that lean on it, **worse**, not better: it
        trades a value that was right by construction (because both sides of the inverse
        problem read the same scalar) for one that is a real, checkable computation off a
        field that a truth and a trial can genuinely disagree about — including, now, in
        the L0/L1 configurations that do not need that disagreement to expose the bias
        already sitting in ``h_l``.
        """
        z_m = state.grid.z_m
        n_e_per_m3 = np.asarray(state.field("n_e").values, dtype=np.float64)
        resolved_length_m = float(z_m[-1] - z_m[0])
        remainder_length_m = self._chord_length_m - resolved_length_m
        if remainder_length_m < 0.0:
            raise ValueError(
                f"the resolved grid ({resolved_length_m:.4g} m) is longer than the "
                f"{self._chord_length_m:.4g} m chord; the bulk-extension approximation in "
                f"{type(self).__name__}._line_averaged_density_per_m3 assumes the grid is "
                "a near-wall subset of the full chord, not the whole of it."
            )
        resolved_integral = float(np.trapezoid(n_e_per_m3, z_m))
        bulk_edge_density_per_m3 = float(n_e_per_m3[-1])
        total_integral = resolved_integral + bulk_edge_density_per_m3 * remainder_length_m
        return total_integral / self._chord_length_m

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        """A noisy, imperfectly-calibrated phase — doc 04 §9.

        **With noise disabled**, this returns exactly what :meth:`forward` returns, and
        calibration is not required first — the same convention
        :class:`~vpl.instruments.interferometry.instrument.InterferometryInstrument.observe`
        follows, for the same reason: doc 04 §9's noiseless path is a check on the physics
        alone.

        **With noise enabled**, :meth:`calibrate` must have already run (or
        :meth:`use_true_calibration` must have been called): doc 04 §7.3's chain has
        nothing to apply until a calibration measurement has produced it, and *simulating a
        measurement* (what enabling noise means) with no calibration measurement to draw the
        estimated response from would apply a response that chain never produced.

        The applied scale is read from :attr:`_calibration` — set once by
        :meth:`calibrate` — rather than drawn fresh on every call: doc 06 §4.1 states that a
        correlated calibration error "affects *every* [...] point identically and does
        **not** average down", and with a single sample per acquisition there is only one
        point for it to apply to regardless, but the stored-scale discipline is kept
        identical to the multi-sample convention this project follows everywhere else.
        """
        predicted = self._predict(state)
        n = predicted.size

        sigma_common = if_noise.vibration_phase_std_rad(w.duration_s, registry=self._registry)
        sigma_independent = if_noise.independent_phase_std_rad(registry=self._registry)
        uncertainty = np.full(
            n, math.sqrt(sigma_common**2 + sigma_independent**2), dtype=np.float64
        )

        if not self._noise_enabled:
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
        independent_draw = self._rng.normal(0.0, sigma_independent, size=n)
        jumps = if_noise.sample_fringe_jumps(self._rng, n_chords=n, rate=self._fringe_jump_rate)

        values = scale * predicted + common_draw + independent_draw + jumps
        return Measurement(
            instrument_id=self.instrument_id,
            values=values,
            uncertainty=uncertainty,
            units=PHASE_UNITS,
            window=w,
            calibration=self._applied_state,
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
        """The correlated-Gaussian term of doc 05 §3.2 — see the module docstring's
        "scoring the calibration coherently" section.

        Builds ``Sigma = D + v v^T`` (``D`` the independent detector variance, ``v`` the
        common-mode vibration standard deviation) and, when ``calibration_uncertainty=True``
        was requested against a non-``TRUE``-calibrated measurement, stacks the phase-scale
        calibration in as a second coherent row via
        :func:`~vpl.instruments.coherent.stack_coherent_rows`, scored by
        :func:`~vpl.instruments.coherent.coherent_gaussian_log_prob` — the same shared
        kernel :class:`~vpl.instruments.interferometry.instrument.InterferometryInstrument`
        uses, now with a single sample rather than eight.

        Args:
            obs: The measurement.
            pred: The prediction to score it against.
            coherent_discrepancy: Optional model-discrepancy **standard deviation**, in
                radians (doc 05 §4, doc 11 WBS 3.6), stacked in as one more coherent row on
                top of the mandatory vibration term — exactly the same rank-``k`` extension
                :class:`~vpl.instruments.interferometry.instrument.InterferometryInstrument.
                likelihood` documents at length for why it is not a special case here: the
                vibration term already makes this channel's covariance non-diagonal, so a
                discrepancy adds another coherent direction rather than introducing
                correlation that was not already there. ``None`` reproduces the pre-existing
                likelihood exactly.
            calibration_uncertainty: Whether to score the doc 04 §7.3 phase-scale
                calibration as the coherent systematic it is.

        Raises:
            RuntimeError: If ``calibration_uncertainty=True`` is requested for a
                measurement that was not generated with the true calibration and
                :meth:`calibrate` has not been called, so there is no registered
                uncertainty to build the coherent row from.
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

        sigma_independent = if_noise.independent_phase_std_rad(registry=self._registry)
        sigma_common = if_noise.vibration_phase_std_rad(
            obs.window.duration_s, registry=self._registry
        )
        d = np.full(n, sigma_independent**2, dtype=np.float64)
        v = np.full(n, sigma_common, dtype=np.float64)

        # `rows` starts at `[v]` (the mandatory vibration term) whether or not either
        # optional term is requested, so the no-extra-term call below is bit for bit
        # unchanged from what this method computed before either argument existed.
        rows: list[FloatArray] = [v]
        apply_calibration = calibration_uncertainty and obs.calibration is not CalibrationState.TRUE
        if apply_calibration:
            if self._calibration is None:
                raise RuntimeError(
                    f"{self.instrument_id}: calibration_uncertainty=True needs calibrate() "
                    "to have been called first — that is where the phase-scale standard's "
                    "registered relative_uncertainty comes from. use_true_calibration() "
                    "alone does not supply it."
                )
            rows.append(
                fractional_calibration_row(
                    np.asarray(pred.values, dtype=np.float64),
                    relative_uncertainty=self._calibration.relative_uncertainty["phase_scale"],
                )
            )
        if coherent_discrepancy is not None:
            rows.append(np.asarray(coherent_discrepancy, dtype=np.float64))

        basis = stack_coherent_rows(rows, expected_shape=(n,))
        if basis is None:
            raise AssertionError("unreachable: the vibration row is never empty")
        return coherent_gaussian_log_prob(residual=residual, variance=d, basis=basis)

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        """The doc 01 IF-6 gate: whether ``n_0`` is at or above this channel's floor.

        The floor formula is unchanged from
        :meth:`~vpl.instruments.interferometry.instrument.InterferometryInstrument.metadata`
        — :func:`~vpl.instruments.interferometry.phase.detection_floor_n_e_per_m3` evaluated
        at the same chord length — so doc 02 §3.3's regime F still gates this channel off at
        the same density it always did; the reframing changes what the channel measures
        above its floor, not where the floor sits.
        """
        return bool(self.metadata().detection_floor.admits(state_guess.n_0))

    def metadata(self) -> InstrumentMetadata:
        """Identity, citations and the detection floor — doc 00 C2, doc 01 IF-6."""
        floor = detection_floor_n_e_per_m3(
            phase_resolution_rad=PHASE_RESOLUTION_RAD,
            wavelength_m=self._wavelength_m,
            chord_length_m=self._chord_length_m,
        )
        return InstrumentMetadata(
            instrument_id=self.instrument_id,
            name="Heterodyne Mach-Zehnder CO2 (10.6 um) interferometer, bulk line integral "
            "(doc 02 §8)",
            version="0.2.0",
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
                "Line-integrated bulk electron-density phase shift (doc 04 §5.1) over the "
                "fixed doc 02 §8.2 IF-P2 chord length; measures the sheath problem's "
                "boundary condition n_0 directly under a beam-uniformity assumption "
                "(doc 02 §8.3), not the sheath itself."
            ),
        )

    def __repr__(self) -> str:
        return (
            f"_BulkInterferometer({self.instrument_id!r}, L={self._chord_length_m:.4g} m, "
            f"lambda={self._wavelength_m:.4g} m)"
        )


class _InterferometryWithOptionalCoherence:
    """``_BulkInterferometer`` with ``calibration_uncertainty`` and a per-channel
    ``coherent_discrepancy`` pinned for fusion.

    :class:`~vpl.inverse.fusion.JointLikelihood` calls ``instrument.likelihood(obs, pred)``
    through a fixed two-positional-argument protocol, so a caller that wants
    :meth:`_BulkInterferometer.likelihood`'s ``calibration_uncertainty`` flag — or its doc
    05 §4 ``coherent_discrepancy`` term — on every call needs both pinned somewhere
    reachable through that fixed shape rather than threaded through it at call time — the
    same problem :mod:`vpl.experiment.channels`'s ``_LifOnReconstructedIvdf`` solves for a
    different per-call transform.
    """

    __slots__ = ("_calibration_uncertainty", "_coherent_discrepancy", "_instrument")

    def __init__(
        self,
        instrument: _BulkInterferometer,
        *,
        calibration_uncertainty: bool,
        coherent_discrepancy: FloatArray | None = None,
    ) -> None:
        self._instrument = instrument
        self._calibration_uncertainty = calibration_uncertainty
        self._coherent_discrepancy = coherent_discrepancy

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        return self._instrument.forward(state, w)

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        return self._instrument.observe(state, w)

    def likelihood(self, obs: Measurement, pred: Observable) -> float:
        return float(
            self._instrument.likelihood(
                obs,
                pred,
                coherent_discrepancy=self._coherent_discrepancy,
                calibration_uncertainty=self._calibration_uncertainty,
            )
        )

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        return bool(self._instrument.is_informative(state_guess))

    @property
    def instrument(self) -> _BulkInterferometer:
        """The wrapped channel, for ``chord_length_m`` and ``metadata``."""
        return self._instrument

    def __repr__(self) -> str:
        return (
            f"_InterferometryWithOptionalCoherence({self._instrument!r}, "
            f"calibration_uncertainty={self._calibration_uncertainty})"
        )


@dataclass(frozen=True, slots=True)
class InterferometryChannel:
    """Both instruments for one closed-loop configuration, plus the window they share.

    Truth-side and inversion-side instruments are separate objects for the reason doc 04 §9
    gives and :class:`~vpl.experiment.channels.ChannelSet`'s own docstring restates: only the
    instrument that generates the measurement is calibrated with a *drawn* estimate, while
    every trial prediction goes through the true response — sharing one object would silently
    apply the calibration error to both sides and cancel it, the doc 04 §7.3 inverse crime.

    Attributes:
        window: The acquisition window both :attr:`truth` and :attr:`inversion` are read
            against — see :func:`interferometry_acquisition_window` for why this one object,
            not two equal-valued ones, is what keeps the truth-side noise draw and the
            inversion-side covariance from silently disagreeing about the channel's
            correlation structure.
        chord_length_m: The resolved doc 02 §8.2 IF-P2 chord length both instruments below
            were actually built with — read back from :attr:`_BulkInterferometer.
            chord_length_m` rather than duplicated as a second literal, so this field can
            never disagree with what the instruments were actually built with.
        truth: The calibrated, possibly-noisy instrument that generates the synthetic
            measurement — doc 07 §3 step 2.
        inversion: The instrument every trial ``theta`` is scored against, wrapped in
            :class:`_InterferometryWithOptionalCoherence` so
            :class:`~vpl.inverse.fusion.JointLikelihood`'s fixed ``likelihood(obs, pred)``
            call shape still reaches the doc 06 §4.1 coherent calibration term when
            ``calibration_uncertainty=True`` was requested at build time.
    """

    window: AcquisitionWindow
    chord_length_m: float
    truth: _BulkInterferometer
    inversion: _InterferometryWithOptionalCoherence


def build_interferometry_channel(
    *,
    reference_state: PlasmaState,  # noqa: ARG001 - signature parity, see Args below
    seed: int,
    registry: ParameterRegistry,
    noise: bool = True,
    imperfect_calibration: bool = True,
    calibration_uncertainty: bool = False,
    discrepancy: FloatArray | None = None,
    start_z_m: float | None = None,  # noqa: ARG001 - accepted for call-signature parity only
) -> InterferometryChannel:
    """Assemble one interferometry channel for a closed-loop configuration.

    Args:
        reference_state: Accepted for call-signature parity with
            :func:`~vpl.experiment.channels.build_channels`'s LIF half (which genuinely needs
            a reference state to size its scan span). Deliberately **unused** here: the
            module docstring's reframing is precisely the finding that this channel's
            geometry never depends on any state at all — a fixed chamber diameter has
            nothing for a reference state to size.
        seed: The single recorded seed (doc 00 E3) both instruments derive their streams
            from.
        registry: Parameter source. Unlike
            :func:`~vpl.experiment.channels.build_channels`, this has no ``None`` default —
            :func:`interferometry_acquisition_window` and :class:`_BulkInterferometer` both
            need one, and requiring the caller to resolve
            :func:`~vpl.core.params.default_registry` explicitly keeps this module from
            quietly picking a different default than whatever the rest of one call's channel
            set is using.
        noise: Whether the truth instrument applies vibration, independent detector noise,
            fringe jumps and the drawn calibration scale — doc 05 §3.1.
        imperfect_calibration: Whether the truth instrument applies the doc 04 §7.3
            estimated phase scale rather than the true one.
        calibration_uncertainty: Whether the inversion instrument's likelihood scores the
            phase-scale calibration as the coherent systematic doc 06 §4.1 says it is, rather
            than asserting the scale is known exactly. See the module docstring's "scoring
            the calibration coherently" section.
        discrepancy: Optional doc 05 §4 model-discrepancy standard deviation, pinned onto
            the inversion instrument's :class:`_InterferometryWithOptionalCoherence`
            wrapper so it reaches :meth:`_BulkInterferometer.likelihood`'s
            ``coherent_discrepancy`` through fusion's fixed ``likelihood(obs, pred)`` call
            shape. ``None`` (the default) leaves the likelihood exactly as it was before
            this parameter existed.
        start_z_m: **Accepted but unused.** The old 8-chord ladder anchor this parameter
            named no longer exists — see the module docstring's reframing: there is no chord
            ladder left to anchor, only a single fixed chamber-diameter chord length (doc 02
            §8.2 IF-P2), which is not a free parameter to relocate. Kept in this signature
            solely because :func:`~vpl.experiment.channels.build_channels` calls this
            function with ``start_z_m=interferometry_start_z_m`` by keyword, and that call
            site is out of scope for this change. ``test_start_z_m_is_accepted_but_no_
            longer_moves_anything`` in ``test_channels_interferometry.py`` is the checked
            claim that supplying a value here changes nothing; a future cleanup that removes
            ``interferometry_start_z_m`` from ``build_channels`` could drop this parameter
            too.
    """
    truth = _BulkInterferometer(root_seed=seed, registry=registry, noise=noise)
    truth.calibrate(interferometry_calibration_set(registry))
    if not imperfect_calibration:
        truth.use_true_calibration()

    inversion = _BulkInterferometer(root_seed=seed, registry=registry, noise=False)
    # calibrate() first, so the phase-scale standard's registered relative_uncertainty is on
    # record for likelihood(calibration_uncertainty=True) to read; use_true_calibration()
    # second, so forward()'s predictions still go through the unit scale exactly as before.
    inversion.calibrate(interferometry_calibration_set(registry))
    inversion.use_true_calibration()

    return InterferometryChannel(
        window=interferometry_acquisition_window(registry),
        chord_length_m=truth.chord_length_m,
        truth=truth,
        inversion=_InterferometryWithOptionalCoherence(
            inversion,
            calibration_uncertainty=calibration_uncertainty,
            coherent_discrepancy=discrepancy,
        ),
    )
