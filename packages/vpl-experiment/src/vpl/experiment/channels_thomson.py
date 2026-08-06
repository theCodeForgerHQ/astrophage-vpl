"""The Thomson scattering diagnostic channel — doc 05 §3.2, doc 11 §9 item 3.

:mod:`vpl.experiment.channels` documents the reason a second channel exists at all: `n_0`
and `T_e` enter the sheath physics multiplied together, so a single optical channel
constrains the product and slides along the degeneracy. Doc 02 §7.1 gives Thomson
scattering a different route to the same two numbers than either OES or LIF: the
*integrated* scattered intensity is proportional to `n_e` (an absolute density, from a
Rayleigh calibration rather than a radiometric one), and the *width* of the scattered
spectrum gives `T_e` directly from a Doppler mechanism (doc 04 §4). It carries none of
OES's brightness-scale degeneracy (module note on the project's honest 36.5 % T2 error, doc
06 §5) and needs no reconstructed distribution the way LIF does (doc 04 §9:
:meth:`~vpl.instruments.thomson.instrument.ThomsonInstrument.forward` reads `n_e` and `T_e`
directly off the state — both are L0/L1 outputs already), so unlike
:class:`~vpl.experiment.channels._LifOnReconstructedIvdf` this module needs no adapter: a
configured :class:`~vpl.instruments.thomson.instrument.ThomsonInstrument` already satisfies
the structural protocol :mod:`vpl.inverse.fusion` calls.

## The trap this module exists to avoid — read `channels.py`'s module docstring first

That docstring records the finding in full: LIF's per-state default measurement volume
(the sheath-edge node of *whatever state it is handed*) moves with `theta` — index 3, 2 or
1 across a MAP search region — and a measurement volume that relocates with the trial
parameters is a theta-dependent observation location, the same class of inverse crime
`closed_loop.py`'s own module docstring documents for the spatial grid (15-20 % error,
`converged=True`). `ThomsonInstrument.configure` also takes a `z_index`
(`vpl.instruments.thomson.instrument._DEFAULT_Z_INDEX` is a fixed array position, `-1`,
rather than a per-state physics-resolved node the way LIF's default is — but relying on an
instrument's own unstated default for *where a manifest-level measurement is taken* is the
same class of implicit-location risk doc 08 §6's manifest discipline exists to close, and
future-proofing against `_DEFAULT_Z_INDEX` growing a dynamic resolver later is cheap). So
:func:`build_thomson_channel` resolves one explicit, non-negative index **once**, from the
reference state's grid, via :func:`_pinned_z_index`, and passes it into `configure`
directly — never the instrument's own default. `test_the_measurement_volume_is_pinned_and_
does_not_move_with_theta` in this module's test file is what stops that being quietly
simplified away, mirroring the LIF guard by name.

## The acquisition window is sized from physics, not quoted from doc 02 §7.1's prose

Doc 02 §7.1 states "~700 s" for a 3 % point at RP-1 (`n_e ~ 1e17 m^-3`) as a worked example,
not a constant to copy. :func:`thomson_acquisition_window` instead calls
:func:`~vpl.instruments.thomson.photons.required_accumulation_s` at the **reference**
state's own `n_e` (at the same pinned index the truth and inversion instruments are
configured at) and :data:`~vpl.instruments.thomson.photons.DEFAULT_TARGET_RELATIVE_UNCERTAINTY`
— so the ~700 s figure is *reproduced*, not hard-coded, and a change to the reference
operating point or to the target precision moves this window with it rather than leaving a
stale literal behind.

**Sized from the reference, never from the truth — the same discipline
`_lif_scan_half_span_ghz` documents at length for the same reason.** Doc 02 §7.1
consequence 1 makes Thomson a genuinely scheduled measurement: a real experiment commits to
an accumulation time before the plasma is characterised, on the basis of the best density
estimate then available (the reference operating point), not on the unknown state a MAP
search is trying to recover. Sizing the window from the state currently being scored would
let a trial's own likelihood advantage depend on how convenient a window its own density
happens to justify — the theta-dependent-observation inverse crime in a different hat.

## Truth-side and inversion-side instruments are separate objects — doc 04 §9

Exactly the reason `ChannelSet` gives for OES and LIF: only the instrument that generates
the synthetic measurement is calibrated with the *estimated* Rayleigh scale doc 04 §7.3
describes, while every trial prediction during inversion goes through the true response.
Sharing one object would apply the calibration error to both sides and have it cancel
out of the residual — doc 04 §7.3's named inverse crime, not a hypothetical one; it is the
same failure mode :meth:`~vpl.instruments.thomson.instrument.ThomsonInstrument.use_true_
calibration` documents on the OES precedent it mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from vpl.core.params import ParameterRegistry
from vpl.core.protocols import CalibrationReference, CalibrationSet, InstrumentConfig
from vpl.core.state import AcquisitionWindow, PlasmaState
from vpl.core.units import Q_
from vpl.instruments.thomson import photons
from vpl.instruments.thomson.instrument import (
    THOMSON_INSTRUMENT_ID,
    ThomsonInstrument,
)
from vpl.instruments.thomson.photons import (
    RAYLEIGH_CALIBRATION_QUANTITY,
    RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY,
)

__all__ = [
    "THOMSON_CHANNEL",
    "ThomsonChannel",
    "build_thomson_channel",
    "thomson_acquisition_window",
    "thomson_calibration_set",
]

#: Channel name — bound to the identifier the instrument itself stamps on its output
#: (`ThomsonInstrument.instrument_id` defaults to this), so an ablation table, a doc 08 §7
#: artifact group and a fusion exclusion list cannot end up calling this channel by two
#: different names. Mirrors `LIF_CHANNEL`'s binding to `LIF_INSTRUMENT_ID` in `channels.py`.
THOMSON_CHANNEL: Final[str] = THOMSON_INSTRUMENT_ID


def thomson_calibration_set() -> CalibrationSet:
    """The one standard :meth:`ThomsonInstrument.calibrate` requires — doc 02 §7.3.

    Rayleigh scattering from a known neutral fill: the standard practice for absolute
    Thomson calibration, because it uses the same optics, the same detector and the same
    solid angle as the measurement, so most of the chain cancels. Its uncertainty is the
    package's own five-term figure from doc 06 §5, taken from there rather than restated
    here so the two cannot drift apart.
    """
    return CalibrationSet.of(
        CalibrationReference(
            name="Rayleigh scattering from a known neutral fill",
            quantity=RAYLEIGH_CALIBRATION_QUANTITY,
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY,
            traceable_to="doc 02 §7.3; doc 06 §5 five-term chain",
        )
    )


def _pinned_z_index(reference_state: PlasmaState) -> int:
    """The one explicit, non-negative spatial index this module ever configures Thomson at.

    `reference_state.grid.n_points - 1` — the outermost point of the fixed spatial grid
    every state in a closed-loop run shares (`closed_loop._fixed_spatial_grid` sizes one
    grid from the reference and reuses it for every trial), which is also
    `ThomsonInstrument`'s own default index expressed explicitly rather than left implicit:
    doc 02 §3.3 regime B calls RP-1 "a bulk-plasma reference point", and this is where an
    L0/L1 profile has relaxed to the quasineutral bulk, away from the sheath the analytic
    solver resolves near `z = 0`.

    Resolved from the grid's *shape* only — never from a field value, and never
    recomputed per trial state — which is what keeps it independent of `theta`. See the
    module docstring's account of the LIF-shaped inverse crime this sidesteps.
    """
    return reference_state.grid.n_points - 1


def _reference_electron_density_m3(reference_state: PlasmaState, *, z_index: int) -> float:
    """`n_e` at the pinned index, read from the reference state only — never the truth's.

    Mirrors the steady-state branch of `ThomsonInstrument`'s own private `_local_value`:
    the reference state this module is built from is always an L0/L1 steady solve (doc 05
    §3.2's steady-plasma assumption, the same one `channels.py`'s `reconstruct_ivdf`
    documents and enforces for LIF), so the time-resolved branch that helper also handles
    is refused here rather than silently mishandled.
    """
    field = reference_state.field("n_e")
    if not field.is_steady:
        raise ValueError(
            "thomson_acquisition_window: the reference state must be steady (time=None); "
            "sizing a Thomson accumulation window from a time-resolved density is not "
            "defined here — see doc 05 §3.2 and channels.py's reconstruct_ivdf for the "
            "identical refusal on the LIF side"
        )
    return float(field.values[z_index])


def thomson_acquisition_window(reference_state: PlasmaState) -> AcquisitionWindow:
    """The one absolute accumulation window every Thomson call in this module uses.

    Duration comes from :func:`~vpl.instruments.thomson.photons.required_accumulation_s`
    at the reference state's own `n_e`, at the pinned index (:func:`_pinned_z_index`), and
    :data:`~vpl.instruments.thomson.photons.DEFAULT_TARGET_RELATIVE_UNCERTAINTY` — doc 02
    §7.1's "3 %" column. At RP-1 this reproduces doc 02 §7.1's own "~700 s" figure rather
    than hard-coding it; see the module docstring for why the window must be sized from the
    reference and never from the state under test.

    Starts at `t = 0`, matching every other channel's window in this project — doc 05
    §3.2's steady-plasma treatment of "the same instant" needs one shared time origin, not
    that every window be the same length.
    """
    z_index = _pinned_z_index(reference_state)
    density_m3 = _reference_electron_density_m3(reference_state, z_index=z_index)
    duration_s = photons.required_accumulation_s(
        electron_density_m3=density_m3,
        target_relative_uncertainty=photons.DEFAULT_TARGET_RELATIVE_UNCERTAINTY,
    )
    return AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(duration_s, "s"))


@dataclass(frozen=True, slots=True)
class ThomsonChannel:
    """One configured Thomson channel — the truth side, the inversion side, and the window.

    Truth-side and inversion-side instruments are always separate objects; see the module
    docstring's doc 04 §9 section for why sharing one would be an inverse crime.

    Attributes:
        window: :func:`thomson_acquisition_window`'s absolute window, shared by both sides.
        z_index: The pinned spatial index both instruments were configured at
            (:func:`_pinned_z_index`), carried here so a caller — or a test — can check it
            against a state's own grid without reaching into either instrument's private
            settings.
        truth: Calibrated with :func:`thomson_calibration_set`; this is what
            :meth:`~vpl.instruments.thomson.instrument.ThomsonInstrument.observe` is called
            on to generate the synthetic measurement.
        inversion: Uncalibrated — `ThomsonInstrument.forward` never consults the applied
            calibration state at all (it hardcodes the true, noiseless code path; see the
            instrument module's own doc 04 §9 docstring), so there is nothing for this side
            to apply. It is still built with `use_true_calibration()` set for symmetry with
            `channels.py`'s LIF pairing and so that a future refactor of `forward` cannot
            silently start reading a calibration state this side was never given.
    """

    window: AcquisitionWindow
    z_index: int
    truth: ThomsonInstrument
    inversion: ThomsonInstrument


def build_thomson_channel(
    *,
    reference_state: PlasmaState,
    seed: int,
    registry: ParameterRegistry,
    noise: bool = True,
    imperfect_calibration: bool = True,
    z_index: int | None = None,
) -> ThomsonChannel:
    """Assemble one Thomson channel for one closed-loop configuration.

    Args:
        reference_state: An L0/L1 solve at the reference operating parameters, on the fixed
            observation grid every trial state shares. Used only to *size* the acquisition
            window and to *resolve* the pinned spatial index — never to generate data. See
            the module docstring for both disciplines.
        seed: The single recorded seed (doc 00 E3) both instruments derive their streams
            from — `ThomsonInstrument.__init__`'s `root_seed`.
        registry: Parameter source, threaded through to both instruments.
        noise: Whether the truth instrument applies photon shot noise and re-draws its
            calibration coefficients on `observe()` — doc 05 §3.1. Has no effect on
            `forward()`, which is always noiseless (doc 04 §9's shared code path).
        imperfect_calibration: Whether the truth instrument applies the doc 04 §7.3
            *estimated* Rayleigh response rather than the true one on `observe()`.
        z_index: Which spatial grid point the measurement volume sits at.

            Defaults to :func:`_pinned_z_index`'s outermost-point resolution — **a fixed
            index, resolved once here rather than per state**, which is what keeps the
            observation location independent of the trial `theta`. See the module
            docstring's first section for the inverse crime this avoids.

    Returns:
        A :class:`ThomsonChannel` with separate truth-side and inversion-side instruments
        (doc 04 §9) and one shared, reference-sized acquisition window.
    """
    resolved_z_index = _pinned_z_index(reference_state) if z_index is None else z_index
    window = thomson_acquisition_window(reference_state)

    truth = ThomsonInstrument(root_seed=seed, registry=registry)
    truth.configure(InstrumentConfig(values={"z_index": resolved_z_index, "noise": noise}))
    truth.calibrate(thomson_calibration_set())
    if not imperfect_calibration:
        truth.use_true_calibration()

    inversion = ThomsonInstrument(root_seed=seed, registry=registry)
    inversion.configure(InstrumentConfig(values={"z_index": resolved_z_index, "noise": False}))
    inversion.use_true_calibration()

    return ThomsonChannel(
        window=window,
        z_index=resolved_z_index,
        truth=truth,
        inversion=inversion,
    )
