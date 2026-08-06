"""The CO2 heterodyne interferometry diagnostic channel — doc 04 §5.1, doc 05 §3.2, doc 11
§9 item 3.

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
with how bright anything looks. This module wires the already-built, already-tested
:class:`~vpl.instruments.interferometry.instrument.InterferometryInstrument` into the
experiment layer exactly as :mod:`vpl.experiment.channels` wires in LIF — full physics
untouched, only the assembly is new.

## The chord ladder does not move with theta, and here is why that had to be checked

:mod:`vpl.experiment.channels`'s own module docstring records the inverse crime this project
found once already: ``LifInstrument``'s default measurement volume resolves to the
sheath-edge node *of whatever state it is handed*, which moves with the trial ``theta`` a
MAP search is exploring, and a relocating observation volume let an optimiser "improve" its
score by moving where it looked rather than by finding the right parameters (15-20 % error,
reported ``converged=True``). Interferometry integrates along chords rather than sampling a
single node, which is a different enough mechanism that the same failure needed to be ruled
out by reading the code, not assumed away by analogy.

It does not apply here, structurally. ``InterferometryInstrument.chord_positions_m`` is
``start + CHORD_SPACING_M * arange(N_CHORDS)`` — a function of the instrument's own
*configured* ``start_z_m`` alone, with no ``state`` parameter anywhere in its signature, and
``_predict`` calls it before it ever touches ``state``. There is consequently nothing in this
module that could feed a trial state into the chord ladder even by accident: unlike LIF's
``z_index``, which defaults to a value ``_resolve_z_index`` computes *from* the state it is
given, ``start_z_m`` is either the caller's explicit override or the instrument's own fixed
default, resolved once at :meth:`~vpl.instruments.interferometry.instrument.
InterferometryInstrument.configure` time.
``TestChordGeometryDoesNotDependOnTheta.test_chord_positions_are_identical_before_and_after_
forward_across_the_map_search_region`` in ``test_channels_interferometry.py`` checks this
directly — the same chord ladder before and after ``forward`` is evaluated at several
materially different thetas spanning the region :func:`~vpl.experiment.closed_loop.
_reduced_prior` and :func:`~vpl.experiment.closed_loop._reference_theta` describe —
precisely so that this finding is a checked property of the code and not a claim resting on
reading it once.

## A second finding this module's construction produced: the chord ladder mostly overlaps a
## grid that is far too short for it

The closed loop's fixed observation grid (``vpl.experiment.closed_loop._fixed_spatial_grid``)
is sized to two sheath thicknesses at the RP-1 reference point — **measured here at
z = 0 to z = 2.28 mm** (11 points). The 8 IF-G1 chords this instrument places are 5 mm apart
starting at the wall, i.e. **z = 0, 5, 10, ..., 35 mm** — a 35 mm span. Only the first chord
(z = 0) falls inside that grid; the other seven all sit past its outer edge, where
``InterferometryInstrument._predict``'s ``np.interp`` clamps to the last grid point's density
rather than extrapolating.

The consequence is not cosmetic: **as wired against this grid, seven of the eight chords
report the identical clamped density**, so the channel that doc 02 §8.2 IF-G1 specifies as 8
independent line integrals degenerates, against this particular grid, into effectively one
interior reading (chord 0) plus one repeated edge reading (chords 1-7) — nowhere near the
``1/sqrt(8)`` noise reduction 8 genuinely independent chords would buy the fit, and a serious
under-use of a channel doc 02 §8.3 already restricts to "no intrinsic z resolution". This is
recorded here rather than fixed by moving ``start_z_m`` or the chord spacing, because doing
either would be tuning the channel's own hardware specification (doc 02 §8.2 IF-G1, a
committed number) to flatter a grid this module does not own and has no mandate to resize —
that decision belongs to whoever next sizes ``_fixed_spatial_grid`` for a genuinely
multi-diagnostic fixed grid, whether interferometry gets wired into the closed loop, and
:func:`build_interferometry_channel`'s own ``start_z_m`` parameter is exactly the knob that
decision would use. What the channel already measurably adds — the amplitude-scale
degeneracy break the module docstring above describes — does not depend on more than one
interior reading: chord 0's reading alone already carries a density signal no
purely-radiometric channel has, which is what
``TestDensitySensitivityAndTemperatureBlindness`` in ``test_channels_interferometry.py``
measures directly (a 6 % change in ``n_0`` moves this channel's log-likelihood by O(1); a
comparable change in ``T_e`` does not).

## The phase-scale uncertainty is a genuinely unsourced number, kept rather than hidden

:data:`_INTERFEROMETER_PHASE_SCALE_UNCERTAINTY` (0.5 %) has no registry entry: the parameter
registry's own ``instruments-interferometry.yaml`` documents that every *other* interferometry
number is either a named ``Final`` constant in
:mod:`vpl.instruments.interferometry.phase` (doc- or paper-cited) or one of the six
``IF.vibration_*`` ASSUMED entries for the vibration-noise *shape* — nothing registers a
phase-scale *calibration* uncertainty at all. Searched directly (``grep`` across every
``*.yaml`` under ``vpl-core/src/vpl/core/params/data``): no ``IF.phase_scale*`` or
``interferometer_phase_scale*`` entry exists. So 0.5 % is kept as the module-level constant
below — "an order of magnitude tighter than OES's registered 6 %" is the qualitative claim
doc 02 §8.1's frequency-and-length chain supports, and this module does not invent a citation
for the specific figure beyond that qualitative comparison. This is reported rather than
silently registered as a new ``ASSUMED`` entry (doc 00 C1's discipline: a number this module
did not measure and cannot source does not get to look sourced by living in a YAML file
instead of a Python one) — the retirement path is the same AOM-and-length-metrology bench
characterisation :data:`~vpl.instruments.interferometry.phase.PHASE_RESOLUTION_RAD`'s own
IF-P1 figure would need if it, too, were ever pinned down further than doc 02 §8.2 states it.

## Scoring the calibration coherently — doc 11 §9's amendment, doc 06 §4.1

``InterferometryInstrument.likelihood`` already built a correlated covariance,
``Sigma = D + v v^T``, for the common-mode *vibration* term (doc 05 §3.1) — but, checked
directly by reading it, that covariance never included the phase-scale calibration
:meth:`~vpl.instruments.interferometry.instrument.InterferometryInstrument.observe` applies
multiplicatively. That is the identical defect this project already found and fixed for OES
and LIF (:mod:`vpl.instruments.coherent`'s module docstring calls it "found and fixed
independently in four places"): asserting perfect knowledge of an uncertain scale, here one
order of magnitude smaller than OES's. ``InterferometryInstrument.likelihood`` now takes a
``calibration_uncertainty`` flag, mirroring the OES and LIF signatures exactly and reusing
:mod:`vpl.instruments.coherent`'s shared Woodbury kernel for the ``k = 2`` case (vibration row
plus calibration row) rather than a fifth hand-derived rank-one path.

**Why the inversion instrument is calibrated at all, when doc 04 §7.3 says only the truth
side should be.** ``OES-C1.radiometric_uncertainty`` and ``LIF.scale_uncertainty`` are
registry entries, so those instruments' likelihoods read the uncertainty figure straight from
the registry and need no calibration object on the inversion side at all. Interferometry has
no equivalent registry entry (see above), so the only place the 0.5 % figure exists is inside
a :class:`~vpl.core.protocols.instrument.CalibrationSet` this module builds. The inversion
instrument is therefore given :meth:`~vpl.instruments.interferometry.instrument.
InterferometryInstrument.calibrate` *and* :meth:`~vpl.instruments.interferometry.instrument.
InterferometryInstrument.use_true_calibration`, in that order: ``calibrate()`` records the
standard's ``relative_uncertainty`` where ``likelihood(calibration_uncertainty=True)`` can
read it, and ``use_true_calibration()`` (which touches only ``observe()``'s applied state, not
``forward()``, which never applies a scale at all) keeps the inversion side predicting through
the unit scale exactly as before. This is a stated deviation from the OES/LIF convention,
made necessary by the registry gap rather than chosen for its own sake.

:class:`_InterferometryWithOptionalCoherence` exists because
:class:`~vpl.inverse.fusion.JointLikelihood` calls ``instrument.likelihood(obs, pred)`` with
no keyword arguments — the same fixed two-positional-argument shape
:mod:`vpl.experiment.channels`'s own ``_LifOnReconstructedIvdf`` exists to adapt a per-call
transform onto. This is that adapter's shape without the IVDF reconstruction: it pins
``calibration_uncertainty`` at construction time so fusion's fixed call signature reaches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from vpl.core.params import ParameterRegistry
from vpl.core.protocols import CalibrationReference, CalibrationSet, InstrumentConfig
from vpl.core.state import (
    AcquisitionWindow,
    Measurement,
    Observable,
    PlasmaParams,
    PlasmaState,
)
from vpl.core.units import Q_
from vpl.instruments.interferometry.instrument import (
    INTERFEROMETER_PHASE_SCALE_QUANTITY,
    INTERFEROMETRY_INSTRUMENT_ID,
    InterferometryInstrument,
)

__all__ = [
    "INTERFEROMETRY_CHANNEL",
    "InterferometryChannel",
    "build_interferometry_channel",
    "interferometry_acquisition_window",
    "interferometry_calibration_set",
]

#: Channel name — bound to the instrument's own stamped identifier rather than duplicated as
#: a second string literal, so an ablation table, a doc 08 §7 artifact group and a fusion
#: exclusion list cannot end up calling this channel two different things (the same
#: discipline :data:`~vpl.experiment.channels.LIF_CHANNEL` follows for LIF's identifier).
INTERFEROMETRY_CHANNEL: Final[str] = INTERFEROMETRY_INSTRUMENT_ID

#: One-sigma relative uncertainty of the interferometer's phase-scale calibration — the
#: AOM frequency reference and path-length metrology chain of doc 02 §8.1. Unlike Thomson's
#: Rayleigh standard, the interferometry package defines no constant of its own because
#: `calibrate()` takes the figure from the standard the caller supplies, so it is named here
#: rather than passed as a literal. A frequency reference and a length metrology chain are
#: both far better controlled than a radiometric lamp, which is exactly why this channel is
#: worth having: at 0.5 % it is an order of magnitude tighter than OES's registered 6 %, and
#: it is that tightness which lets it carry the density scale OES demonstrably cannot.
#:
#: **Unsourced beyond the qualitative comparison above** — see the module docstring's
#: "phase-scale uncertainty" section. No ``IF.phase_scale*`` registry entry exists (checked
#: directly against every YAML file under ``vpl-core/src/vpl/core/params/data``), so this
#: stays a named module constant rather than a registry lookup, and is reported as unsourced
#: rather than silently registered.
_PHASE_SCALE_UNCERTAINTY_ID: Final[str] = "IF.phase_scale_uncertainty"


def interferometry_calibration_set(registry: ParameterRegistry) -> CalibrationSet:
    """The one standard :meth:`InterferometerInstrument.calibrate` requires — doc 02 §8.1.

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

    Not cosmetic: :meth:`~vpl.instruments.interferometry.instrument.InterferometryInstrument.
    observe` draws the common-mode vibration term from
    :func:`~vpl.instruments.interferometry.noise.vibration_phase_std_rad` evaluated at *this*
    window's duration, and :meth:`~vpl.instruments.interferometry.instrument.
    InterferometryInstrument.likelihood` rebuilds the same function at the duration recorded
    on the *observation* it is scoring. If the truth-side window this module hands to
    ``observe`` and the window the resulting :class:`~vpl.core.state.Measurement` carries were
    ever two different objects, the two sides would silently disagree about how correlated
    the channel is — the exact failure mode :meth:`~vpl.instruments.interferometry.instrument.
    InterferometryInstrument.observe`'s own docstring calls out for the calibration scale.
    Returning one window here and threading it through both call sites is what keeps that
    from being possible to get wrong.

    Duration is ``IF.vibration_reference_window_s`` — not an arbitrary choice: it is the one
    window at which :mod:`vpl.instruments.interferometry.noise`'s own module docstring says
    the vibration-variance calibration is anchored to doc 02 §8.2 IF-P1's quoted 0.1 mrad
    total, so acquiring at this window is the one point where the channel's reported
    uncertainty is tied to a sourced figure rather than an extrapolation of an unmeasured
    spectral shape into an untested regime. Read from the registry rather than restated as a
    second literal, so this module and ``vpl.instruments.interferometry.noise`` cannot drift
    apart on what "the reference window" means.
    """
    return AcquisitionWindow.absolute(
        start=Q_(0.0, "s"), duration=registry.quantity("IF.vibration_reference_window_s")
    )


class _InterferometryWithOptionalCoherence:
    """``InterferometryInstrument`` with ``calibration_uncertainty`` pinned for fusion.

    :class:`~vpl.inverse.fusion.JointLikelihood` calls ``instrument.likelihood(obs, pred)``
    through a fixed two-positional-argument protocol (``vpl.inverse.fusion``'s own local
    ``_Instrument`` protocol), so a caller that wants
    :meth:`~vpl.instruments.interferometry.instrument.InterferometryInstrument.likelihood`'s
    ``calibration_uncertainty`` flag on every call needs it pinned somewhere reachable through
    that fixed shape rather than threaded through it at call time — the same problem
    :mod:`vpl.experiment.channels`'s ``_LifOnReconstructedIvdf`` solves for a different
    per-call transform (reconstructing an ion distribution rather than fixing a keyword).
    This is that adapter's shape, minus the reconstruction: it satisfies the structural
    instrument protocol :mod:`vpl.inverse.fusion` needs (``forward``/``likelihood``/
    ``is_informative``) plus ``observe``, and delegates everything else — ``metadata()``,
    ``chord_positions_m()`` — through the :attr:`instrument` property, unchanged.
    """

    __slots__ = ("_calibration_uncertainty", "_instrument")

    def __init__(
        self, instrument: InterferometryInstrument, *, calibration_uncertainty: bool
    ) -> None:
        self._instrument = instrument
        self._calibration_uncertainty = calibration_uncertainty

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        return self._instrument.forward(state, w)

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        return self._instrument.observe(state, w)

    def likelihood(self, obs: Measurement, pred: Observable) -> float:
        return float(
            self._instrument.likelihood(
                obs, pred, calibration_uncertainty=self._calibration_uncertainty
            )
        )

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        return bool(self._instrument.is_informative(state_guess))

    @property
    def instrument(self) -> InterferometryInstrument:
        """The wrapped channel, for ``chord_positions_m``, ``metadata`` and ``coverage``."""
        return self._instrument

    def __repr__(self) -> str:
        return (
            f"_InterferometryWithOptionalCoherence({self._instrument!r}, "
            f"calibration_uncertainty={self._calibration_uncertainty})"
        )


def _interferometry_instrument(
    *, seed: int, registry: ParameterRegistry, start_z_m: float | None, noise: bool
) -> InterferometryInstrument:
    """One configured, unconfigured-elsewhere interferometer — doc 08 §6's manifest block.

    ``start_z_m`` is passed through, possibly ``None``, rather than resolved here against a
    literal default: :meth:`~vpl.instruments.interferometry.instrument.
    InterferometryInstrument.configure` already owns the doc 02 §8.2 default (anchored at the
    wall), and restating that default as a second number in this module would be exactly the
    kind of duplicate-source-of-truth doc 08 §5 exists to prevent.
    """
    instrument = InterferometryInstrument(root_seed=seed, registry=registry)
    instrument.configure(InstrumentConfig(values={"start_z_m": start_z_m, "noise": noise}))
    return instrument


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
        start_z_m: The resolved chord-ladder anchor actually configured on both instruments
            below — read back from :meth:`~vpl.instruments.interferometry.instrument.
            InterferometryInstrument.chord_positions_m` rather than duplicated as a second
            copy of whatever default or override was supplied, so this field can never
            disagree with what the instruments were actually built with.
        truth: The calibrated, possibly-noisy instrument that generates the synthetic
            measurement — doc 07 §3 step 2.
        inversion: The instrument every trial ``theta`` is scored against, wrapped in
            :class:`_InterferometryWithOptionalCoherence` so
            :class:`~vpl.inverse.fusion.JointLikelihood`'s fixed ``likelihood(obs, pred)``
            call shape still reaches the doc 06 §4.1 coherent calibration term when
            ``calibration_uncertainty=True`` was requested at build time.
    """

    window: AcquisitionWindow
    start_z_m: float
    truth: InterferometryInstrument
    inversion: _InterferometryWithOptionalCoherence


def build_interferometry_channel(
    *,
    reference_state: PlasmaState,  # noqa: ARG001 - signature parity, see Args below
    seed: int,
    registry: ParameterRegistry,
    noise: bool = True,
    imperfect_calibration: bool = True,
    calibration_uncertainty: bool = False,
    start_z_m: float | None = None,
) -> InterferometryChannel:
    """Assemble one interferometry channel for a closed-loop configuration.

    Args:
        reference_state: Accepted for call-signature parity with
            :func:`~vpl.experiment.channels.build_channels`'s LIF half (which genuinely needs
            a reference state to size its scan span — see
            :func:`~vpl.experiment.channels._lif_scan_half_span_ghz`) so that the integration
            lead's assembly can call every channel builder the same way. Deliberately
            **unused** here: the module docstring's chord-ladder section is precisely the
            finding that this channel's geometry — unlike LIF's default measurement volume —
            never depends on any state at all, so there is nothing for this argument to size.
            Accepting it and ignoring it, rather than dropping it from the signature, is what
            keeps that asymmetry a documented fact about the two channels instead of a silent
            one a caller could not tell from the call site.
        seed: The single recorded seed (doc 00 E3) both instruments derive their streams
            from.
        registry: Parameter source. Unlike
            :func:`~vpl.experiment.channels.build_channels`, this has no ``None`` default —
            :func:`interferometry_acquisition_window` and
            :func:`~vpl.experiment.channels_interferometry._interferometry_instrument` both
            need one, and requiring the caller to resolve
            :func:`~vpl.core.params.default_registry` explicitly keeps this module from
            quietly picking a different default than whatever the rest of one call's channel
            set is using.
        noise: Whether the truth instrument applies vibration, independent detector noise,
            fringe jumps and the drawn calibration scale — doc 05 §3.1.
        imperfect_calibration: Whether the truth instrument applies the doc 04 §7.3 estimated
            phase scale rather than the true one.
        calibration_uncertainty: Whether the inversion instrument's likelihood scores the
            phase-scale calibration as the coherent systematic doc 06 §4.1 says it is, rather
            than asserting the scale is known exactly. See the module docstring's "Scoring
            the calibration coherently" section for why this requires calibrating the
            inversion instrument too, and
            :meth:`~vpl.instruments.interferometry.instrument.InterferometryInstrument.
            likelihood` for the mechanics.
        start_z_m: Where the 8-chord ladder begins, in metres from the wall. ``None`` uses
            the instrument's own doc 02 §8.2 default (the wall itself, the most conservative
            anchor — see :mod:`vpl.instruments.interferometry.instrument`'s module
            docstring). See the module docstring's grid-domain finding before assuming a
            non-default anchor helps: this parameter can move where the chords sit, but it
            was deliberately not used here to paper over the fact that the closed loop's
            current fixed observation grid is far shorter than the chord ladder's fixed 35 mm
            span.
    """
    truth = _interferometry_instrument(
        seed=seed, registry=registry, start_z_m=start_z_m, noise=noise
    )
    truth.calibrate(interferometry_calibration_set(registry))
    if not imperfect_calibration:
        truth.use_true_calibration()

    inversion = _interferometry_instrument(
        seed=seed, registry=registry, start_z_m=start_z_m, noise=False
    )
    # calibrate() first, so the phase-scale standard's registered relative_uncertainty is on
    # record for likelihood(calibration_uncertainty=True) to read (see the module docstring's
    # "Scoring the calibration coherently" section for why this instrument, unlike OES's or
    # LIF's inversion-side counterpart, needs a Calibration object at all); use_true_
    # calibration() second, so forward()'s predictions — and observe(), if anything ever
    # calls it on this side — still go through the unit scale exactly as before.
    inversion.calibrate(interferometry_calibration_set(registry))
    inversion.use_true_calibration()

    resolved_start_z_m = float(truth.chord_positions_m()[0])

    return InterferometryChannel(
        window=interferometry_acquisition_window(registry),
        start_z_m=resolved_start_z_m,
        truth=truth,
        inversion=_InterferometryWithOptionalCoherence(
            inversion, calibration_uncertainty=calibration_uncertainty
        ),
    )
