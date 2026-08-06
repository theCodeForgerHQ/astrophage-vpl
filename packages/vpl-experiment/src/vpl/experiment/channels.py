"""Diagnostic channel assembly for the closed loop — doc 05 §3.2, doc 11 §9 item 3.

:mod:`vpl.inverse.fusion` is the sum over channels; this module is what fills it. It
builds the OES channel the closed loop already runs, builds the LIF channel that has been
sitting fully implemented and unconnected in :mod:`vpl.instruments.lif`, and hands back a
:class:`~vpl.inverse.fusion.JointLikelihood` of the two together with the truth-side
observation for each.

## Why a second channel at all

doc 11 §9 item 3 asks for two channels and the project runs one. The reason is concrete
rather than architectural: plasma density and electron temperature enter the sheath
physics multiplied together (``Gamma_i ~ n_0 sqrt(T_e)``), so a single channel constrains a
product and slides along the degeneracy. LIF measures the ion velocity distribution
directly — line centre from the drift, width from ``T_i``, amplitude from ``n_i`` — which
is a different functional of the same state, and different is the whole point.

## Three findings this module ran into, all stated rather than worked around

**1. The RP-1 tuning-range blocker — found here, since fixed upstream, and the fix carries
its own hazard.** doc 01 §5.1 and doc 14 RS-03 give the laser a 20 GHz mode-hop-free range,
which with doc 02 §4.2's 15 degree grazing geometry is a hard ceiling of 25.8 km/s in
``v_z``, and RP-1's -250 V sheath accelerates argon to 34.9 km/s. When this module was
written, :meth:`~vpl.instruments.lif.instrument.LifInstrument.is_informative` anchored its
gate on that full free-fall speed, so it returned ``False`` at RP-1, the fusion layer
excluded LIF by name, and **connecting the channel did nothing at the operating point the
project actually cares about** — the joint likelihood was OES alone, wearing two channels'
clothes. Nothing here overrode it, because doc 01 IF-6 is explicit that a blind channel
contributes no term rather than a weak one.

That gate has since been re-anchored, in ``vpl-instruments``, at the **sheath-edge Bohm
speed**, on doc 01 §5.1's own stated mitigation: measure at the edge, where the drift is
order ``c_s`` and the Doppler shift is comfortably reachable, and *propagate* to the wall
rather than pretending the wall IVDF is directly measurable. LIF is now informative at
RP-1 and both channels contribute there.

**The hazard the fix introduces, and the reason this module pins ``lif_z_index``.** The
same change made ``LifInstrument._resolve_z_index`` default the measurement volume to the
sheath-edge node *of whatever state it is handed*. That node moves with ``theta``: measured
across the region a MAP search explores, it comes out at index 3, 2 or 1 depending on
``n_0`` and ``T_e``. A measurement volume that relocates with the trial parameters is a
**theta-dependent observation location** — the identical class of inverse crime
:mod:`vpl.experiment.closed_loop`'s module docstring documents for the spatial grid, where
it produced a 15-20 % error and reported ``converged=True``. Comparing a trial's scan
against the truth's scan point by point then compares two different physical positions, and
the optimiser can reduce the residual by relocating the volume instead of by finding the
right parameters. So :func:`build_channels` always passes an explicit ``lif_z_index`` and
never lets the default resolve per state, exactly as ``_fixed_spatial_grid`` sizes one grid
from the reference and reuses it for every trial. ``test_the_measurement_volume_is_pinned_
and_does_not_move_with_theta`` is what stops that being quietly simplified away.

**2. At the registered nominal beam the channel is saturation-broadened past uselessness,
so the beam is attenuated here and the attenuation is declared.** ``lif.yaml``'s laser
power and beam waist with its (deliberately upper-bound) ``pump_branching_ratio`` give
``S = I/I_sat`` of 1.6e4, and doc 04 §3.4's power-broadened width ``Gamma sqrt(1 + S)`` is
then 2.05 GHz — about 5.3 km/s in ``v_z``, fifteen times the 347 m/s thermal width the
measurement is supposed to resolve. Measured directly: at the nominal beam a doubling of
``T_i`` moves this channel's log-likelihood by 2.1, and at
:data:`~vpl.instruments.lif.instrument.SATURATION_REPORTING_THRESHOLD` it moves it by
4.7e4. So the channel is configured at that threshold — the largest ``S`` at which doc 04
§3.4 says the lineshape distortion stays under the doc 06 §4 systematics it competes with
— which physically is an experimenter putting an attenuator in the beam, not a fitted
knob. It is called out here because "we turned the laser down" is exactly the kind of
configuration choice that otherwise disappears into a constructor.

## The ion distribution is reconstructed, not solved for, and that is an assumption

:meth:`~vpl.instruments.lif.instrument.LifInstrument.forward` needs
``state.ion_distribution`` and refuses a state without one, quoting doc 03 §6: L0 and L1
both return ``ion_distribution=None`` because neither solves for ``f_i``, and
``AnalyticSheathSolver.solve``'s own docstring says recording the absence is more honest
than manufacturing a distribution, so "the assumption then has to be made explicitly by
whoever needs it, where it can be swept and budgeted". This module is whoever needs it.
:func:`reconstruct_ivdf` makes it explicitly: doc 03 §6's drifting Maxwellian built from
the state's own ``(n_i, u_i)`` fields and ``T_i``, the same closure
:meth:`~vpl.physics.fluid.sheath.FluidSheathSolver.flux` already uses for the L1 energy
flux and which doc 03 §6 itself calls "a genuine weakness" — it cannot represent a bimodal
RF IEDF or a collisional low-energy tail. The consequence for a fused result: at L0 and L1
the LIF channel measures the shape this module assumed, not a shape the physics computed,
so what it genuinely adds there is the ``(n_i, u_i, T_i)`` moments carried through a
different instrument, not independent evidence about the distribution's form. Against an
L2 truth, which does resolve ``f_i``, that stops being circular.

## The acquisition-window assumption, stated because doc 05 §3.2 requires it to be

The two channels integrate for genuinely different times. OES gates for
``OES-D2.gate_width`` = 2 ns; LIF must gate for at least a few times the upper level's
32 ns radiative relaxation or :mod:`vpl.instruments.lif.scan`'s steady-state solution
over-predicts the signal, so its window here is
:data:`_LIF_GATE_RELAXATION_TIMES` relaxation times, ~318 ns. That is a factor of 159, and
both windows start at ``t = 0``.

**The assumption is that the plasma is steady over the longer of the two windows**, which
is what makes "the same instant" a defensible description of two acquisitions that are not
simultaneous. It holds for the states this module is used with: L0 and L1 are both
steady-state solves (``FluidSheathSolver.solve``'s own docstring is explicit that L1 "as
implemented is steady"), and every state reaching here carries ``time=None``.
doc 05 §3.2 says plainly that treating windows separated by orders of magnitude as the same
instant is "straightforwardly false" for a transient, so **this module refuses a
time-resolved state** rather than fusing across it — see :func:`reconstruct_ivdf`. A
phase-resolved RF run is exactly that case and needs a window-aware fusion this does not
have.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.protocols import (
    CalibrationReference,
    CalibrationSet,
    InstrumentConfig,
)
from vpl.core.state import (
    AcquisitionWindow,
    Measurement,
    Observable,
    PlasmaParams,
    PlasmaState,
    VelocityDistribution,
)
from vpl.core.units import Q_, magnitude_in
from vpl.instruments.lif.instrument import (
    FREQUENCY_AXIS_STANDARD,
    RELATIVE_SCALE_STANDARD,
    SATURATION_REPORTING_THRESHOLD,
    LifInstrument,
)
from vpl.instruments.lif.instrument import (
    INSTRUMENT_ID as LIF_INSTRUMENT_ID,
)
from vpl.instruments.lif.laser import Laser
from vpl.instruments.lif.scan import doppler_detuning_hz, resonant_velocity_m_per_s
from vpl.instruments.lif.transition import ProbeTransition
from vpl.instruments.oes.instrument import OesInstrument
from vpl.inverse.fusion import Channel, JointLikelihood

__all__ = [
    "LIF_CHANNEL",
    "OES_CHANNEL",
    "ChannelSet",
    "build_channels",
    "reconstruct_ivdf",
]

type FloatArray = NDArray[np.float64]

#: Channel names — the same strings the instruments stamp on their own observables, so an
#: ablation table, a doc 08 §7 artifact group and a fusion exclusion list cannot end up
#: calling one channel by two names.
#:
#: LIF exports its identifier publicly and it is imported. OES's equivalent is private
#: (``vpl.instruments.oes.instrument._DEFAULT_INSTRUMENT_ID``), so it is duplicated here
#: and the duplicate is pinned by ``test_the_channel_names_match_what_the_instruments_stamp``
#: — a checked copy rather than a reach into another package's privates.
OES_CHANNEL: Final[str] = "oes"
LIF_CHANNEL: Final[str] = LIF_INSTRUMENT_ID

#: Frequency steps in the LIF scan. doc 02 §10.1 budgets "~200 for a full IVDF"; odd so
#: that one sample sits exactly on line centre, which is where the drift is read off.
_LIF_SCAN_POINTS: Final[int] = 201

#: Half-widths of the assumed drifting Maxwellian that the velocity axis and the frequency
#: scan are both sized to reach.
#:
#: Six. The Gaussian tail beyond 6 sigma is 2e-9 of the population, five orders of
#: magnitude below :data:`~vpl.instruments.lif.scan.TRUNCATION_TOLERANCE`, so a scan sized
#: this way is not clipping any part of the distribution the instrument would have to
#: report as truncated.
_DISTRIBUTION_HALF_WIDTH_SIGMAS: Final[float] = 6.0

#: Samples per line width on the reconstructed velocity axis.
#:
#: :data:`~vpl.instruments.lif.scan.MIN_SAMPLES_PER_HOMOGENEOUS_WIDTH` is 1.0 and that
#: module's own docstring calls one sample per full width "already marginal — the
#: trapezoid error on a Lorentzian sampled that coarsely is tens of percent". Four is the
#: smallest multiple of it that puts the quadrature error below the 5 % scale uncertainty
#: the LIF likelihood divides by, and the axis length scales inversely with it: at RP-1's
#: 35 km/s of drift range this is ~3500 velocity points per state, and every trial theta
#: in a MAP search rebuilds them.
_VELOCITY_SAMPLES_PER_WIDTH: Final[float] = 4.0

#: Length of the LIF acquisition window, in upper-state radiative relaxation times.
#:
#: :mod:`vpl.instruments.lif.scan` takes the upper level to have reached its steady state
#: within the gate and states the failure directly: the relaxation time is 32 ns, "a 2 ns
#: gate samples the transient, and the steady-state model over-predicts the signal there by
#: up to ``1 - exp(-t/tau)``, a factor of 16". Ten relaxation times leaves that error at
#: ``exp(-10)`` = 5e-5, which is four orders of magnitude below the doc 06 §4 systematics,
#: so the steady-state model this channel is built on is actually valid at this gate.
#: There is no registered LIF gate width to read instead; doc 02 LIF-D1 gives only a >= 2 ns
#: minimum, which is the value that would be wrong.
_LIF_GATE_RELAXATION_TIMES: Final[float] = 10.0


# ── doc 03 §6's assumed distribution, made explicit ─────────────────────────────


def _velocity_axis(*, drift: FloatArray, sigma: float, homogeneous_width: float) -> FloatArray:
    """The shared ``v_z`` axis a :class:`VelocityDistribution` needs.

    One axis serves every spatial point, so it must span the drift at all of them — from
    the Bohm speed at the sheath edge to the full free-fall speed at the wall — while
    still resolving the narrowest feature in the integrand. That narrowest feature is not
    the thermal width: :func:`~vpl.instruments.lif.scan.fluorescence_response` integrates
    ``f_i`` against a homogeneous Lorentzian and refuses a grid the Lorentzian would fall
    between the points of, so the spacing is set by whichever of the two is smaller.
    """
    span = _DISTRIBUTION_HALF_WIDTH_SIGMAS * sigma
    low = float(np.min(drift)) - span
    high = float(np.max(drift)) + span
    spacing = min(sigma, homogeneous_width) / _VELOCITY_SAMPLES_PER_WIDTH
    n_points = math.ceil((high - low) / spacing) + 1
    return np.linspace(low, high, n_points, dtype=np.float64)


def _homogeneous_width_m_per_s(registry: ParameterRegistry) -> float:
    """The homogeneous line width expressed in ``v_z``, via doc 04 §3.2's mapping.

    Computed from the same registry entries and the same public helpers
    :func:`~vpl.instruments.lif.scan.fluorescence_response` uses for its own grid check,
    so the axis this module builds cannot be refused by a check it was sized against a
    different number to satisfy.
    """
    transition = ProbeTransition.from_registry(registry)
    laser = Laser.from_registry(registry)
    fwhm_hz = float(magnitude_in(transition.homogeneous_fwhm, "Hz")) + laser.linewidth_hz
    return float(
        resonant_velocity_m_per_s(
            np.array([fwhm_hz], dtype=np.float64),
            pump_frequency_hz=transition.pump_frequency_hz,
            projection_factor=laser.projection_factor,
        )[0]
    )


def reconstruct_ivdf(
    state: PlasmaState, *, registry: ParameterRegistry | None = None
) -> PlasmaState:
    """Attach doc 03 §6's assumed drifting Maxwellian to a state that carries none.

    ``f_i(z, v_z) = n_i(z) / (sqrt(2 pi) sigma) exp(-(v_z + u_i(z))^2 / 2 sigma^2)`` with
    ``sigma = sqrt(k T_i / m_i)``. The plus sign is doc 02 §2's convention, carried through
    :meth:`~vpl.core.state.VelocityDistribution.particle_flux_toward_wall_per_m2_s`: the
    wall is at ``z = 0`` and ``z`` is positive into the plasma, so an ion reaching the wall
    has ``v_z < 0`` while the ``u_i`` field a sheath solver emits is a positive speed.
    Getting this backwards would put the LIF line on the wrong side of centre and still
    return a plausible-looking scan, which is why the sign lives in one named place here
    rather than inline at the call site.

    See the module docstring for what this assumption costs. Briefly: it is the same
    closure L1 already uses for its energy flux, doc 03 §6 calls it "a genuine weakness",
    and against an L0 or L1 truth it makes the LIF channel a re-measurement of moments
    rather than independent evidence about the shape of ``f_i``.

    Args:
        state: Any state carrying ``n_i`` and ``u_i``. A state that already carries an ion
            distribution — an L2 solve does — is returned unchanged, because a computed
            distribution always beats an assumed one.
        registry: Where the line widths that size the velocity axis come from.

    Raises:
        ValueError: If the state is time-resolved. See the module docstring's
            acquisition-window section: two channels integrating for 2 ns and 318 ns can be
            fused as one instant only while nothing is changing, and doc 05 §3.2 calls the
            transient case "straightforwardly false" rather than approximate.
    """
    if state.ion_distribution is not None:
        return state
    if state.time is not None:
        raise ValueError(
            "a time-resolved state reached the IVDF reconstruction. The OES and LIF "
            "windows here differ by a factor of 159 (2 ns against ~318 ns), which doc 05 "
            "§3.2 permits to be treated as one instant only for a steady plasma; fusing "
            "them across a transient would report two different times as the same "
            "measurement."
        )

    resolved = registry if registry is not None else default_registry()
    mass_kg = state.params.species.mass_kg
    sigma = math.sqrt(state.params.T_i_J / mass_kg)
    if not sigma > 0.0:
        raise ValueError(
            f"T_i = {state.params.T_i_J} J gives a zero thermal width, and a Maxwellian of "
            "zero width is a delta function no velocity grid can represent"
        )

    drift = -np.asarray(state.field("u_i").values, dtype=np.float64)
    density = np.asarray(state.field("n_i").values, dtype=np.float64)
    axis = _velocity_axis(
        drift=drift, sigma=sigma, homogeneous_width=_homogeneous_width_m_per_s(resolved)
    )

    normalisation = density / (math.sqrt(math.tau) * sigma)
    exponent = -((axis[np.newaxis, :] - drift[:, np.newaxis]) ** 2) / (2.0 * sigma**2)
    return PlasmaState(
        params=state.params,
        grid=state.grid,
        time=None,
        fields=dict(state.fields),
        ion_distribution=VelocityDistribution(
            grid=state.grid,
            v_m_per_s=axis,
            values=normalisation[:, np.newaxis] * np.exp(exponent),
            species=state.params.species,
        ),
        fidelity=state.fidelity,
    )


# ── the LIF channel, as something the closed loop can hand an L0 state ──────────


class _LifOnReconstructedIvdf:
    """:class:`LifInstrument` with :func:`reconstruct_ivdf` in front of it.

    The closed loop's inversion produces L0 states, which carry no ion distribution, and
    :class:`~vpl.inverse.fusion.JointLikelihood` calls ``instrument.forward(state, window)``
    with whatever state it is given. Putting the reconstruction in an adapter rather than
    asking every caller to remember it means the assumption is applied identically to the
    truth's state and to every trial state — if it were applied to only one of them the
    residual would contain the difference between two distribution models and be reported
    as a measurement error.

    Satisfies the structural instrument protocol :mod:`vpl.inverse.fusion` needs
    (``forward``/``likelihood``/``is_informative``) plus ``observe``, and delegates
    everything else unchanged.
    """

    __slots__ = ("_instrument", "_registry")

    def __init__(self, instrument: LifInstrument, *, registry: ParameterRegistry) -> None:
        self._instrument = instrument
        self._registry = registry

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        return self._instrument.forward(reconstruct_ivdf(state, registry=self._registry), w)

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        return self._instrument.observe(reconstruct_ivdf(state, registry=self._registry), w)

    def likelihood(self, obs: Measurement, pred: Observable) -> float:
        return float(self._instrument.likelihood(obs, pred))

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        return bool(self._instrument.is_informative(state_guess))

    @property
    def instrument(self) -> LifInstrument:
        """The wrapped channel, for ``run_conditions``, ``coverage`` and ``metadata``."""
        return self._instrument

    def __repr__(self) -> str:
        return f"_LifOnReconstructedIvdf({self._instrument!r})"


def _lif_calibration_set(registry: ParameterRegistry) -> CalibrationSet:
    """The two standards :meth:`LifInstrument.calibrate` requires — doc 02 §11.

    No radiometric standard, and there never will be one: doc 02 §11 records that LIF is
    not absolutely calibrated. What is certified is the frequency axis (wavemeter plus
    etalon) and a relative amplitude scale whose uncertainty doc 06 §4 puts at the
    metastable fraction's 5 %.
    """
    return CalibrationSet.of(
        CalibrationReference(
            name="Wavemeter + Fabry-Perot etalon",
            quantity=FREQUENCY_AXIS_STANDARD,
            value=registry.quantity("LIF.frequency_axis_uncertainty"),
            relative_uncertainty=0.0,
            traceable_to="NIST-traceable wavemeter, doc 02 §11",
        ),
        CalibrationReference(
            name="Relative LIF amplitude scale",
            quantity=RELATIVE_SCALE_STANDARD,
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=float(registry.value_in("LIF.scale_uncertainty", "dimensionless")),
            traceable_to="doc 06 §4 item 4 (metastable fraction)",
        ),
    )


def _lif_scan_half_span_ghz(reference_state: PlasmaState, *, z_index: int) -> float:
    """Scan half-width, sized from the **reference** state — never from the truth.

    This is the same discipline :func:`~vpl.experiment.closed_loop._fixed_spatial_grid`
    applies to the spatial grid, for the same reason its module docstring gives at length:
    a real laser's scan range is set before the plasma is characterised, and sizing an
    observation axis from the unknown state about to be measured is the theta-dependent
    grid inverse crime wearing a different hat. A scan sized from the truth's own drift
    would let the likelihood distinguish thetas by how well the scan happened to frame
    them.

    The span reaches the reference drift at the measurement volume plus
    :data:`_DISTRIBUTION_HALF_WIDTH_SIGMAS` thermal widths, mapped to frequency by doc 04
    §3.2. :meth:`~vpl.instruments.lif.scan.DetuningScan.uniform` refuses it outright if it
    exceeds the mode-hop-free range, which is doc 01 §5.1's limit doing its job.
    """
    registry = default_registry()
    sigma = math.sqrt(reference_state.params.T_i_J / reference_state.params.species.mass_kg)
    drift = abs(float(reference_state.field("u_i").values[z_index]))
    half_span_m_per_s = drift + _DISTRIBUTION_HALF_WIDTH_SIGMAS * sigma

    transition = ProbeTransition.from_registry(registry)
    laser = Laser.from_registry(registry)
    half_span_hz = float(
        doppler_detuning_hz(
            np.array([half_span_m_per_s], dtype=np.float64),
            pump_frequency_hz=transition.pump_frequency_hz,
            projection_factor=laser.projection_factor,
        )[0]
    )
    return float(magnitude_in(Q_(half_span_hz, "Hz"), "GHz"))


def _lif_instrument(
    *,
    reference_state: PlasmaState,
    z_index: int,
    seed: int,
    registry: ParameterRegistry,
    imperfect_calibration: bool,
) -> _LifOnReconstructedIvdf:
    """One configured LIF channel — doc 08 §6's manifest block, filled in from code.

    ``saturation`` is pinned at
    :data:`~vpl.instruments.lif.instrument.SATURATION_REPORTING_THRESHOLD` rather than left
    at the laser's own; see the module docstring's second finding for the measurement that
    made that necessary and why it is an attenuator rather than a fitted knob.
    """
    instrument = LifInstrument(registry)
    instrument.configure(
        InstrumentConfig(
            values={
                "z_index": z_index,
                "scan_points": _LIF_SCAN_POINTS,
                "scan_half_span_ghz": _lif_scan_half_span_ghz(reference_state, z_index=z_index),
                "saturation": SATURATION_REPORTING_THRESHOLD,
                "apply_calibration_error": imperfect_calibration,
                "seed": seed,
            }
        )
    )
    return _LifOnReconstructedIvdf(instrument, registry=registry)


def _lif_acquisition_window(registry: ParameterRegistry) -> AcquisitionWindow:
    """A gate long enough for doc 04 §3.1's steady state to be reached — see the constant.

    ``LIF.natural_linewidth`` is registered as ``A_total / 2 pi``, so the upper level's
    radiative relaxation time is ``1 / (2 pi Delta nu_nat)``.
    """
    natural_linewidth_hz = float(registry.value_in("LIF.natural_linewidth", "Hz"))
    relaxation_s = 1.0 / (math.tau * natural_linewidth_hz)
    return AcquisitionWindow.absolute(
        start=Q_(0.0, "s"), duration=Q_(_LIF_GATE_RELAXATION_TIMES * relaxation_s, "s")
    )


# ── the assembled set ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ChannelSet:
    """Both channels, on the truth side and the inversion side, plus their windows.

    Truth-side and inversion-side instruments are separate objects for the reason doc 04 §9
    gives: only the instrument that generates the measurement is calibrated, and at T1/T2
    it applies the *estimated* radiometric response while every trial prediction goes
    through the true one. Sharing one object would silently apply the calibration error to
    both sides and cancel it — the inverse crime doc 04 §7.3 names.

    Attributes:
        oes_window: doc 02 §6.2 OES-D2's minimum ICCD gate, at ``t = 0``.
        lif_window: :data:`_LIF_GATE_RELAXATION_TIMES` upper-state relaxation times, also
            at ``t = 0``. See the module docstring for the steady-state assumption that
            makes two windows this different fusable at all.
    """

    oes_window: AcquisitionWindow
    lif_window: AcquisitionWindow
    _truth_oes: OesInstrument
    _truth_lif: _LifOnReconstructedIvdf
    _oes: OesInstrument
    _lif: _LifOnReconstructedIvdf

    def observe(self, truth_state: PlasmaState) -> dict[str, Measurement]:
        """One synthetic measurement per channel — doc 07 §3 step 2, for both channels.

        The closed loop has to *generate* data as well as score it, and it must generate
        one dataset per channel from the same sealed truth state. Returned as a mapping
        keyed by channel name so that :meth:`joint` and an ablation sweep refer to a
        channel by the same string the fusion layer reports it as.
        """
        return {
            OES_CHANNEL: self._truth_oes.observe(truth_state, self.oes_window),
            LIF_CHANNEL: self._truth_lif.observe(truth_state, self.lif_window),
        }

    def joint(self, observations: Mapping[str, Measurement]) -> JointLikelihood:
        """doc 05 §3.2's sum, bound to one set of observations.

        The inversion-side instruments are used here, never the truth-side ones. Channels
        appear in a fixed order so that a fusion detail and an ablation table read the same
        way from run to run.

        Raises:
            KeyError: If an observation is missing for either channel. Silently fusing the
                one channel that happened to be supplied would report a single-channel
                result as a two-channel one.
        """
        missing = sorted({OES_CHANNEL, LIF_CHANNEL} - set(observations))
        if missing:
            raise KeyError(
                f"no observation supplied for {', '.join(missing)}; a joint likelihood "
                f"built from a subset would report fewer channels than it claims. Use "
                f"JointLikelihood.without() to ablate a channel deliberately."
            )
        return JointLikelihood(
            (
                Channel(
                    name=OES_CHANNEL,
                    instrument=self._oes,
                    observation=observations[OES_CHANNEL],
                    window=self.oes_window,
                ),
                Channel(
                    name=LIF_CHANNEL,
                    instrument=self._lif,
                    observation=observations[LIF_CHANNEL],
                    window=self.lif_window,
                ),
            )
        )

    @property
    def lif(self) -> LifInstrument:
        """The inversion-side LIF instrument, for ``run_conditions`` and ``coverage``.

        doc 04 §3.4 requires every LIF run to report its saturation state, and a caller
        that cannot reach the instrument cannot report it.
        """
        return self._lif.instrument


def build_channels(
    *,
    reference_state: PlasmaState,
    seed: int,
    registry: ParameterRegistry | None = None,
    noise: bool = True,
    imperfect_calibration: bool = True,
    lif_z_index: int | None = None,
) -> ChannelSet:
    """Assemble the OES and LIF channels for one closed-loop configuration.

    The OES half is built from :mod:`vpl.experiment.closed_loop`'s own helpers rather than
    reconstructed here — same registry, same acquisition window, same Maxwellian EEDF on
    the inversion side — so that the channel this module fuses and the channel that module
    runs alone cannot drift apart into two subtly different instruments that are then
    compared against each other.

    Args:
        reference_state: An L0 solve at the RP-1 reference parameters, on the fixed
            observation grid. Used only to *size* the LIF scan, never to generate data —
            see :func:`_lif_scan_half_span_ghz`.
        seed: The single recorded seed (doc 00 E3) both channels derive their streams from.
        registry: Parameter source. Defaults to :func:`~vpl.core.params.default_registry`.
        noise: Whether the OES truth instrument applies photon statistics — doc 05 §3.1.
            The LIF channel has no detector noise to switch: doc 04 §1 puts the eighteen
            sources of doc 04 §7.2 in ``F4``, and ``LifInstrument.observe``'s docstring is
            explicit that its output is not a complete synthetic measurement.
        imperfect_calibration: Whether both truth instruments apply the doc 04 §7.3
            estimated response rather than the true one.
        lif_z_index: Which spatial grid point the LIF measurement volume sits at.

            Defaults to the outermost point, where the L0 profile has relaxed to the
            quasineutral presheath and the drift is exactly the Bohm speed. Two reasons,
            and the first is the load-bearing one. **It is a fixed index, resolved once
            here rather than per state**, which is what keeps the observation location
            independent of the trial ``theta`` — see the module docstring's first finding
            for the inverse crime that the instrument's own per-state default would
            otherwise walk into. Second, ``c_s`` carries ``sqrt(T_e)`` with no ``n_0`` in
            it while the amplitude carries ``n_0`` with no ``T_e``, which is precisely the
            separation the OES emission rate does not provide and the reason this channel
            breaks the degeneracy at all.

            Index 0 is the wall and is doc 01 §1.1's deliverable, but the wall IVDF is not
            directly measurable here at any interesting bias — 34.9 km/s of drift at RP-1
            against a 25.8 km/s ceiling. doc 01 §5.1's mitigation is to propagate the
            sheath-edge distribution to the wall through the reconstructed field, and doc
            06 §4 makes that propagation's error a budgeted term rather than a silent one.
    """
    # Imported here, not at module scope: the closed loop is expected to import *this*
    # module once the fusion is wired in, and a module-level import in both directions is
    # a cycle. Deferring the direction that is only needed at call time keeps this module
    # importable from there.
    from vpl.experiment.closed_loop import (
        _acquisition_window,
        _calibration_set,
        _maxwellian_eedf_factory,
        _oes_instrument,
    )

    resolved = registry if registry is not None else default_registry()
    z_index = reference_state.grid.n_points - 1 if lif_z_index is None else lif_z_index

    truth_oes = _oes_instrument(seed=seed, registry=resolved, eedf_factory=_maxwellian_eedf_factory)
    truth_oes.calibrate(_calibration_set(resolved))
    truth_oes.set_noise_enabled(noise)
    if not imperfect_calibration:
        truth_oes.use_true_calibration()

    inversion_oes = _oes_instrument(
        seed=seed, registry=resolved, eedf_factory=_maxwellian_eedf_factory
    )

    truth_lif = _lif_instrument(
        reference_state=reference_state,
        z_index=z_index,
        seed=seed,
        registry=resolved,
        imperfect_calibration=imperfect_calibration,
    )
    truth_lif.instrument.calibrate(_lif_calibration_set(resolved))
    inversion_lif = _lif_instrument(
        reference_state=reference_state,
        z_index=z_index,
        seed=seed,
        registry=resolved,
        imperfect_calibration=False,
    )

    return ChannelSet(
        oes_window=_acquisition_window(resolved),
        lif_window=_lif_acquisition_window(resolved),
        _truth_oes=truth_oes,
        _truth_lif=truth_lif,
        _oes=inversion_oes,
        _lif=inversion_lif,
    )
