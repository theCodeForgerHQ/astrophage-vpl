"""The LIF channel as an ``Instrument`` — doc 08 §4, doc 04 §9, doc 01 IF-6.

One class, seven methods, and the two that matter share a code path. Doc 04 §9:

    ``observe`` returns raw data with noise and imperfect calibration; ``forward`` returns
    the noiseless expectation used by the likelihood. **Both come from the same code
    path**, with noise and calibration error as switchable stages, which guarantees they
    cannot drift apart.

:meth:`LifInstrument.observe` therefore calls :meth:`LifInstrument.forward` and applies
stages to its result; there is no second evaluation of the physics anywhere in this module,
and ``test_lif_instrument.py`` checks that switching the stages off reproduces ``forward``
*exactly* rather than to a tolerance.

## What ``observe`` does and does not add

It adds the doc 04 §7.3 imperfect calibration: one scale factor, drawn once, applied to
every sample. That is the right shape as well as the right magnitude — doc 06 §4.1 records
that a calibration error "affects *every* point identically and does **not** average down",
so it is deliberately not per-sample scatter.

It adds **no detector noise**, and that is a layering decision rather than an omission.
Doc 04 §1's rule is that ``F4`` never sees the plasma; the converse holds too, and the
eighteen sources of doc 04 §7.2 belong to ``vpl-detectors`` operating on photons that have
already been through ``vpl-optics``. A synthetic LIF dataset is this channel's output
passed through ``F3`` and ``F4``, not this channel pretending to be all three. The
consequence for a caller: :meth:`observe` here is **not** a complete synthetic measurement,
and using it as one would understate the error budget by every term in doc 04 §7.2.

## The detection gate has two parts

Doc 01 IF-6 requires a channel below its floor to contribute *no* likelihood term rather
than a weak one. For LIF the floor is not only a density:

1. **Density.** Too few metastables and there are no photons. The threshold is
   ``LIF.detection_floor_n_0``, and ``lif.yaml`` labels it the weakest entry in the file.
2. **Tuning range.** Doc 01 §5.1 and doc 14 RS-03: past a certain ion speed the
   population never comes into resonance at any frequency the laser can reach. This is a
   *harder* gate than the density one — it is not a matter of signal-to-noise, the
   measurement simply does not exist — and it is why :meth:`is_informative` estimates the
   largest ion speed the channel must resolve and compares it against the velocity
   ceiling.

   **That speed is the sheath-edge speed, not the wall speed.** Doc 01 §5.1's LIF-3 note
   is explicit that a scanning single-frequency laser cannot resolve the *full-energy*
   IVDF inside a high-bias sheath — a 250 eV Ar+ ion's Doppler shift is 51.9 GHz against
   a 20 GHz mode-hop-free range — and that this is not a defect to route around but the
   reason the framework measures at (and just inside) the **sheath edge** instead, where
   the drift is order ``c_s`` (4-12 GHz, comfortably reachable). :meth:`forward` and
   :meth:`coverage` sample there by default (see :meth:`_resolve_z_index`), and
   :meth:`_expected_ion_speed` gates on the same location: ``c_s`` plus a few thermal
   widths, never the free-fall speed across the full sheath drop. The earlier version of
   this gate compared the wall speed regardless of where the measurement volume actually
   sat, which made the channel declare itself blind at every high-bias operating point
   this project cares about — RP-1 included — even though the sheath-edge measurement it
   actually performs was never out of the laser's reach.

   **The wall IVDF this channel implies is consequently an inferred quantity, not a
   measured one.** Doc 01 §5.1's mitigation is to *propagate* the sheath-edge entry
   distribution through the reconstructed sheath field to obtain it, and doc 06 §4 makes
   that propagation's error a first-class term in the error budget — not an unstated
   approximation this gate, or anything downstream of it, is entitled to ignore.

Both gates are conservative in the same direction: they return ``False`` where the channel
might still carry a little information, never ``True`` where it carries none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.protocols import (
    Calibration,
    CalibrationSet,
    Citation,
    DetectionFloor,
    InstrumentConfig,
    InstrumentMetadata,
    LogProb,
    satisfies,
    verified_by,
)
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    Measurement,
    Observable,
    PlasmaParams,
    PlasmaState,
)
from vpl.core.units import Q_, Quantity, magnitude_in
from vpl.instruments.coherent import (
    coherent_gaussian_log_prob,
    fractional_calibration_row,
    stack_coherent_rows,
)
from vpl.instruments.lif.laser import Laser, TuningRange
from vpl.instruments.lif.rates import power_broadened_fwhm
from vpl.instruments.lif.scan import (
    DetuningScan,
    TuningCoverage,
    fluorescence_response,
    tuning_coverage,
)
from vpl.instruments.lif.transition import ProbeTransition
from vpl.instruments.lif.zeeman import (
    PumpPolarisation,
    ZeemanComponent,
    unsplit_pattern,
    zeeman_pattern,
)
from vpl.physics.analytic.sheath import bohm_speed

__all__ = ["INSTRUMENT_ID", "LifInstrument", "RunConditions"]

type FloatArray = NDArray[np.float64]

#: The doc 08 §7 artifact group this channel writes to.
INSTRUMENT_ID: str = "lif"

#: Units of what :meth:`LifInstrument.forward` returns — see :mod:`vpl.instruments.lif.scan`
#: for why it is an upper-state density and not a radiance.
OBSERVABLE_UNITS: str = "m**-3"

#: The calibration standards :meth:`LifInstrument.calibrate` requires, as doc 02 §11 names
#: the chains they come from.
FREQUENCY_AXIS_STANDARD: str = "lif_frequency_axis"
RELATIVE_SCALE_STANDARD: str = "lif_relative_scale"

#: Above this the run is reported as saturated and its lineshape as distorted (doc 04 §3.4).
#: Ten percent: the power-broadened width is then 5 % wider than the natural one, which is
#: where the distortion stops being smaller than the doc 06 §4 systematics it competes with.
SATURATION_REPORTING_THRESHOLD: float = 0.1

#: Multiple of the mean ion speed that must fit inside the tuning window for the channel to
#: be declared informative.
#:
#: Three, so that the thermal spread either side of the drift is included rather than only
#: its centre. A gate on the mean alone would call a distribution visible when half of it
#: had already run off the end of the scan.
INFORMATIVE_SPEED_MARGIN: float = 3.0


@dataclass(frozen=True, slots=True)
class RunConditions:
    """What doc 04 §3.4 requires every run to report.

    Attributes:
        saturation: ``S = I / I_sat``.
        homogeneous_fwhm: The unsaturated homogeneous width — natural, collisional and
            laser linewidth summed as Lorentzians.
        power_broadened_fwhm: ``Gamma sqrt(1 + S)``, the width actually observed.
        is_saturated: Whether the distortion is large enough to matter.
    """

    saturation: float
    homogeneous_fwhm: Quantity
    power_broadened_fwhm: Quantity
    is_saturated: bool

    def __repr__(self) -> str:
        state = "saturated" if self.is_saturated else "linear"
        return (
            f"RunConditions(S={self.saturation:.4g}, {state}, "
            f"width {self.homogeneous_fwhm:.4g~P} -> {self.power_broadened_fwhm:.4g~P})"
        )


@dataclass(frozen=True, slots=True)
class _Settings:
    """The resolved manifest block — doc 08 §6's ``instruments:`` entry for this channel.

    ``z_index`` is ``None`` unless the manifest names one explicitly: see
    :meth:`LifInstrument._resolve_z_index` for what a ``None`` resolves to per state, and
    the module docstring's "detection gate" section for why the default changed from the
    wall to the sheath edge.
    """

    z_index: int | None
    scan: DetuningScan
    transition: ProbeTransition
    laser: Laser
    tuning: TuningRange
    magnetic_field: Quantity
    polarisation: PumpPolarisation
    saturation: float
    metastable_fraction: float
    scale_uncertainty: float
    detection_floor: Quantity
    edge_tolerance: float
    apply_calibration_error: bool
    seed: int


def _as_int(value: object, *, default: int, key: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer, got {value!r}")
    return value


def _as_float(value: object, *, default: float, key: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number, got {value!r}")
    return float(value)


def _sheath_edge_index(state: PlasmaState, *, tolerance: float) -> int:
    """The outermost grid node still inside the doc 03 §2.1 sheath, as a ``z_index``.

    ``(n_i - n_e) / n_i >= delta`` is doc 03 §2.1's own definition of the sheath edge,
    ``delta`` the registered ``sheath.edge_tolerance`` (doc 03 §2.1 names it explicitly
    as a swept DESIGN parameter, not a hidden convention). The search direction matches
    ``vpl.physics.fluid.sheath._sheath_edge`` and ``vpl.physics.kinetic.solver.
    _sheath_edge_m``, which locate the same crossing for the same reason: the
    quasineutrality departure is not guaranteed monotone — a ripple in the presheath, shot
    noise in a particle code — so the *outermost* index, scanning from the wall
    (``z = 0``, doc 02 §2) toward the bulk, is the one adjacent to the quasineutral
    boundary rather than a spurious inner crossing.

    Returns a concrete row index rather than an interpolated position, because
    :func:`~vpl.instruments.lif.scan.fluorescence_response` needs one:
    :class:`~vpl.core.state.VelocityDistribution` is only ever known at grid points, and
    doc 04 §6.2's point-measurement simplification this channel already makes is exactly
    that — a value at one index, not a continuum sample.

    Raises:
        ValueError: If no grid point departs from quasineutrality by ``tolerance`` — a
            flat, quasineutral state has no sheath edge to default to, and inventing one
            (the wall, the bulk boundary) would be a silent modelling choice rather than
            the caller's.
    """
    n_i = state.field("n_i").values
    n_e = state.field("n_e").values
    departure = (n_i - n_e) / n_i
    crossings = np.flatnonzero(departure >= tolerance)
    if crossings.size == 0:
        raise ValueError(
            "no grid point departs from quasineutrality by sheath.edge_tolerance = "
            f"{tolerance}; this state has no sheath edge for the LIF channel to default "
            "to. Pass an explicit z_index (e.g. 0 for the wall) if a flat, quasineutral "
            "state is genuinely what is being measured."
        )
    return int(crossings[-1])


@satisfies("LIF-1", "LIF-2", "LIF-3", "LIF-7")
@verified_by("V-22", "V-23", "F-13")
class LifInstrument:
    """Laser-induced fluorescence on the doc 02 §5.3 Ar II scheme.

    Implements :class:`vpl.core.protocols.Instrument`. Configure it from a manifest block,
    then :meth:`forward` a :class:`~vpl.core.state.PlasmaState` into a detuning scan.
    """

    __slots__ = ("_registry", "_settings")

    def __init__(self, registry: ParameterRegistry | None = None) -> None:
        self._registry = registry if registry is not None else default_registry()
        self._settings: _Settings | None = None

    # ── the doc 08 §4 contract ──────────────────────────────────────────────────

    def configure(self, cfg: InstrumentConfig) -> None:
        """Apply one entry of the manifest's ``instruments:`` list — doc 08 §6.

        Every key is optional and falls back to the registry, so an empty block gives the
        doc 02 §5.2 nominal instrument. Keys:

        ``z_index``
            Spatial grid point the measurement volume sits at. Left unset, it resolves
            per state to the outermost grid node still inside the doc 03 §2.1 sheath —
            the sheath-edge location doc 01 §5.1's LIF-3 note names as where this
            channel actually measures (see :meth:`_resolve_z_index`). ``0`` is the wall;
            doc 07's sensitivity study passes it explicitly to sweep the measurement
            location, wall included.
        ``scan_points``
            Frequency steps. Doc 02 §10.1 budgets ~200 for a full IVDF.
        ``scan_half_span_ghz``
            Half-width of the scan. Refused above the mode-hop-free range.
        ``magnetic_field_mt``
            Helmholtz field. ``0`` selects the unsplit line.
        ``polarisation``
            ``pi``, ``sigma_plus``, ``sigma_minus`` or ``isotropic``.
        ``saturation``
            ``S``, overriding the laser's own — the doc 07 F-13 sweep knob.
        ``grazing_angle_deg``, ``beam_waist_um``, ``laser_power_mw``
            Doc 02 §5.2 geometry and beam, for the sensitivity study.
        ``metastable_fraction``
            The doc 05 §2.2 nuisance parameter, at the value being sampled.
        ``apply_calibration_error``
            Whether :meth:`observe` applies the doc 04 §7.3 imperfect calibration.
            ``False`` is the deliberate inverse crime of doc 04 §7.3, for verification.
        ``seed``
            Per-instrument stream seed. Doc 00 E3.
        """
        registry = self._registry

        laser = Laser.from_registry(registry)
        grazing = cfg.get("grazing_angle_deg")
        if grazing is not None:
            laser = laser.with_grazing_angle(
                Q_(_as_float(grazing, default=0.0, key="grazing_angle_deg"), "deg")
            )
        waist = cfg.get("beam_waist_um")
        if waist is not None:
            laser = laser.with_beam_waist(
                Q_(_as_float(waist, default=0.0, key="beam_waist_um"), "um")
            )
        power = cfg.get("laser_power_mw")
        if power is not None:
            laser = laser.with_power(Q_(_as_float(power, default=0.0, key="laser_power_mw"), "mW"))

        transition = ProbeTransition.from_registry(registry)
        tuning = TuningRange.from_registry(registry)

        half_span = cfg.get("scan_half_span_ghz")
        scan = DetuningScan.uniform(
            tuning=tuning,
            n_points=_as_int(cfg.get("scan_points"), default=201, key="scan_points"),
            half_span=(
                Q_(_as_float(half_span, default=0.0, key="scan_half_span_ghz"), "GHz")
                if half_span is not None
                else None
            ),
        )

        field = cfg.get("magnetic_field_mt")
        magnetic_field = (
            Q_(_as_float(field, default=0.0, key="magnetic_field_mt"), "mT")
            if field is not None
            else registry.quantity("LIF.magnetic_field")
        )

        saturation = cfg.get("saturation")
        z_index_cfg = cfg.get("z_index")
        self._settings = _Settings(
            z_index=(
                None if z_index_cfg is None else _as_int(z_index_cfg, default=0, key="z_index")
            ),
            scan=scan,
            transition=transition,
            laser=laser,
            tuning=tuning,
            magnetic_field=magnetic_field,
            polarisation=PumpPolarisation(str(cfg.get("polarisation", "isotropic"))),
            saturation=(
                _as_float(saturation, default=0.0, key="saturation")
                if saturation is not None
                else laser.saturation_parameter(transition)
            ),
            metastable_fraction=_as_float(
                cfg.get("metastable_fraction"),
                default=float(registry.value_in("LIF.metastable_fraction", "dimensionless")),
                key="metastable_fraction",
            ),
            scale_uncertainty=float(registry.value_in("LIF.scale_uncertainty", "dimensionless")),
            detection_floor=registry.quantity("LIF.detection_floor_n_0"),
            edge_tolerance=float(registry.value_in("sheath.edge_tolerance", "dimensionless")),
            apply_calibration_error=bool(cfg.get("apply_calibration_error", True)),
            seed=_as_int(cfg.get("seed"), default=0, key="seed"),
        )

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        """Derive the applied response from reference standards — doc 02 §11, doc 04 §7.3.

        LIF is **not absolutely calibrated** (doc 02 §11), so there is no radiometric
        coefficient here and there never will be. What is calibrated is the frequency axis
        — wavemeter plus etalon, 2 MHz — and a relative amplitude scale whose uncertainty
        is dominated by the metastable fraction doc 06 §4 budgets at 5 %.
        """
        settings = self._require_configured()
        frequency = refs.for_quantity(FREQUENCY_AXIS_STANDARD)
        scale = refs.for_quantity(RELATIVE_SCALE_STANDARD)

        return Calibration(
            instrument_id=INSTRUMENT_ID,
            coefficients={
                FREQUENCY_AXIS_STANDARD: float(magnitude_in(frequency.value, "Hz")),
                RELATIVE_SCALE_STANDARD: float(magnitude_in(scale.value, "dimensionless")),
            },
            relative_uncertainty={
                FREQUENCY_AXIS_STANDARD: frequency.relative_uncertainty,
                RELATIVE_SCALE_STANDARD: max(
                    scale.relative_uncertainty, settings.scale_uncertainty
                ),
            },
            state=CalibrationState.ESTIMATED,
            reference=f"{frequency.name}; {scale.name}",
        )

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        """The noiseless detuning scan — doc 04 §3.1, §3.2.

        Returns the upper-state density at each scan point, multiplied by the metastable
        fraction of doc 02 §5.3. See :mod:`vpl.instruments.lif.scan` for why the
        radiometric constant is deliberately absent.
        """
        settings = self._require_configured()
        if state.ion_distribution is None:
            raise ValueError(
                f"a {state.fidelity.name} state reached the LIF forward model with no ion "
                "distribution. LIF measures f_i(v_z) and nothing else (doc 04 §3.2), so "
                "there is no signal to compute from moments — doc 03 §6 records that "
                "reconstructing one from a drifting Maxwellian is a genuine weakness, and "
                "doing it silently here would hide it inside a measurement."
            )

        signal = fluorescence_response(
            distribution=state.ion_distribution,
            z_index=self._resolve_z_index(state),
            scan=settings.scan,
            transition=settings.transition,
            laser=settings.laser,
            components=self._components(),
            saturation=settings.saturation,
        )
        return Observable(
            instrument_id=INSTRUMENT_ID,
            values=settings.metastable_fraction * signal,
            units=OBSERVABLE_UNITS,
            window=w,
        )

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        """A calibrated observation — doc 04 §9, doc 04 §7.3.

        :meth:`forward` with the imperfect-calibration stage applied. See the module
        docstring for what is deliberately *not* applied and why.
        """
        settings = self._require_configured()
        predicted = self.forward(state, w)

        scale = 1.0
        if settings.apply_calibration_error:
            # One draw, applied to every sample: doc 06 §4.1's correlated calibration
            # error, which does not average down. Per-sample scatter here would be the
            # detector's, and the detector is F4's (doc 04 §1).
            rng = np.random.default_rng(settings.seed)
            scale = float(rng.normal(1.0, settings.scale_uncertainty))

        values = scale * predicted.values
        return Measurement(
            instrument_id=INSTRUMENT_ID,
            values=values,
            uncertainty=settings.scale_uncertainty * np.abs(values),
            units=OBSERVABLE_UNITS,
            window=w,
            calibration=CalibrationState.ESTIMATED,
        )

    def likelihood(
        self,
        obs: Measurement,
        pred: Observable,
        *,
        coherent_discrepancy: FloatArray | None = None,
        calibration_uncertainty: bool = False,
    ) -> LogProb:
        """This channel's Gaussian term in the doc 05 §3.2 sum.

        Gaussian rather than Poisson because what is compared here is a calibrated
        amplitude dominated by a correlated systematic, not a photon count; the counting
        statistics live in ``F4`` and enter through the uncertainty this is handed. Doc 05
        §3.1's point is that the channels have *different* statistics, and this is LIF's.

        The **structure** below is deliberately identical to
        :meth:`~vpl.instruments.oes.instrument.OesInstrument.likelihood`'s, down to sharing
        :mod:`vpl.instruments.coherent`'s Woodbury kernel: the two channels differ in what
        their diagonal ``D`` means and in nothing else, and a coherent error must be scored
        the same way in both or the doc 05 §3.2 sum is inconsistent with itself. See that
        module's docstring for the algebra and the ``1 + a`` overweighting factor.

        Args:
            obs: The measurement.
            pred: The prediction to score it against.
            coherent_discrepancy: Optional per-sample **standard deviation** of model
                discrepancy, in the same units as ``pred``, treated as coherent across the
                scan (doc 05 §4, doc 11 WBS 3.6). Either one direction shaped like the
                prediction or a ``(k, n)`` basis. ``None`` reproduces the pre-existing
                diagonal likelihood exactly.
            calibration_uncertainty: Whether to score the doc 04 §7.3 amplitude calibration
                as the coherent systematic it is — see **Calibration** below.

        Calibration — doc 02 §11, doc 06 §4.1:
            LIF is **not absolutely calibrated**, so what is uncertain here is a *relative*
            amplitude scale: ``LIF.scale_uncertainty``, 5 % one-sigma, dominated by the
            metastable fraction doc 06 §4 item 4 budgets at that figure.
            :meth:`observe` applies it as **one draw across the whole scan** — doc 06 §4.1's
            correlated error that "affects every point identically and does not average
            down" — and the diagonal residual below then scored that single error once per
            scan point. At the ``configure`` default of 201 points that is a ~201x
            over-weighting of the calibration systematic in the chi-squared, and it is why
            this channel's contribution to a posterior was over-confident.

            Enabled, the calibration enters as one coherent row ``f * y_hat``: fractional,
            because a scale factor mis-scales a large signal by more absolute density than a
            small one, in exactly the ratio of their amplitudes.

            **Why opt-in.** Same reasoning as OES, and stated in full there: the inversion
            side genuinely should account for it, but which runs carry calibration
            uncertainty is a property of the campaign rather than of this method, and T0/T1
            are published regression baselines that a flipped default would silently
            re-score. The guard against forgetting is the same too — asking for it on a
            measurement whose :class:`~vpl.core.state.CalibrationState` is ``TRUE`` is a
            no-op, because a run configured with ``apply_calibration_error: false`` (doc 04
            §7.3's deliberate verification crime) has no calibration error to score.
        """
        settings = self._require_configured()
        if obs.shape != pred.shape:
            raise ValueError(
                f"observation and prediction have different shapes, {obs.shape} and "
                f"{pred.shape}. A broadcast here would compare a scan against a scan of a "
                "different length and report a finite log-likelihood for it."
            )
        if obs.units != pred.units:
            raise ValueError(f"observation is in {obs.units!r} and prediction in {pred.units!r}")
        if np.any(obs.uncertainty <= 0.0):
            raise ValueError(
                "a zero uncertainty reached the LIF likelihood. Doc 01 IF-6 expresses an "
                "uninformative channel by not forming its term at all; check "
                "is_informative before asking for one."
            )

        predicted = np.asarray(pred.values, dtype=np.float64)

        # Assemble the coherent directions. Calibration first, because it is a property of
        # the instrument rather than of the model being scored; the Woodbury form is
        # invariant to row order, so this is for readability only.
        rows: list[FloatArray] = []
        if calibration_uncertainty and obs.calibration is not CalibrationState.TRUE:
            rows.append(
                fractional_calibration_row(
                    predicted, relative_uncertainty=settings.scale_uncertainty
                )
            )
        if coherent_discrepancy is not None:
            rows.append(np.asarray(coherent_discrepancy, dtype=np.float64))

        basis = stack_coherent_rows(rows, expected_shape=predicted.shape)
        if basis is None:
            residual = (obs.values - pred.values) / obs.uncertainty
            return float(
                -0.5 * np.sum(residual**2)
                - np.sum(np.log(obs.uncertainty * math.sqrt(2.0 * math.pi)))
            )

        # No variance floor here, unlike OES: a zero uncertainty was already refused above,
        # by doc 01 IF-6's rule that an uninformative channel is expressed by not forming
        # its term at all rather than by forming a degenerate one.
        return coherent_gaussian_log_prob(
            residual=(np.asarray(obs.values, dtype=np.float64) - predicted).reshape(-1),
            variance=(np.asarray(obs.uncertainty, dtype=np.float64) ** 2).reshape(-1),
            basis=basis,
        )

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        """The doc 01 IF-6 gate, in its two parts. See the module docstring."""
        settings = self._require_configured()

        if not self.metadata().detection_floor.admits(state_guess.n_0):
            return False

        ceiling = float(
            magnitude_in(
                settings.tuning.velocity_ceiling(
                    transition=settings.transition, laser=settings.laser
                ),
                "m/s",
            )
        )
        return self._expected_ion_speed(state_guess) <= ceiling

    def metadata(self) -> InstrumentMetadata:
        """Identity, citations and the doc 01 IF-6 floor — doc 00 C2."""
        settings = self._require_configured()
        return InstrumentMetadata(
            instrument_id=INSTRUMENT_ID,
            name="Tunable ECDL at 668.6 nm + f/4 collection + gated ICCD (doc 02 §5)",
            version="0.1.0",
            citations=(
                Citation(
                    key="doc04-lif",
                    reference=(
                        "Astrophage VPL doc 04 §3 — LIF rate-equation fluorescence, "
                        "Doppler mapping, Zeeman structure and saturation"
                    ),
                ),
                Citation(
                    key="stern-johnson-1975",
                    reference=(
                        "R. A. Stern and J. A. Johnson III, 'Plasma ion diagnostics using "
                        "resonant fluorescence', Phys. Rev. Lett. 34, 1548 (1975) — the "
                        "origin of velocity-resolved LIF in plasmas"
                    ),
                    doi="10.1103/PhysRevLett.34.1548",
                ),
                Citation(
                    key="condon-shortley-1935",
                    reference=(
                        "E. U. Condon and G. H. Shortley, 'The Theory of Atomic Spectra', "
                        "Cambridge University Press (1935) — Lande factors and the "
                        "electric-dipole Zeeman intensity tables of "
                        "vpl.instruments.lif.zeeman"
                    ),
                ),
            ),
            detection_floor=DetectionFloor(
                quantity="n_0",
                threshold=settings.detection_floor,
                requirement="LIF-3",
            ),
            description=(
                "Ar II 668.614 nm pump / 442.72 nm fluorescence at 15 deg grazing "
                "incidence. Constrains IVDF shape and relative amplitude only: doc 02 §11 "
                "records that LIF is not absolutely calibrated."
            ),
        )

    # ── what doc 04 §3.4 requires every run to report ───────────────────────────

    def run_conditions(self) -> RunConditions:
        """The saturation state of this configuration — doc 04 §3.4, doc 07 F-13."""
        settings = self._require_configured()
        homogeneous_hz = (
            float(magnitude_in(settings.transition.homogeneous_fwhm, "Hz"))
            + settings.laser.linewidth_hz
        )
        return RunConditions(
            saturation=settings.saturation,
            homogeneous_fwhm=Q_(homogeneous_hz, "Hz"),
            power_broadened_fwhm=Q_(
                power_broadened_fwhm(fwhm_hz=homogeneous_hz, saturation=settings.saturation), "Hz"
            ),
            is_saturated=settings.saturation > SATURATION_REPORTING_THRESHOLD,
        )

    def coverage(self, state: PlasmaState) -> TuningCoverage:
        """How much of this state's IVDF the laser can reach — doc 01 §5.1, doc 14 RS-03.

        Reported alongside every forward evaluation rather than checked once at
        configuration time, because the answer depends on the state: the same instrument
        sees the whole distribution at the sheath edge and almost none of it at the wall.
        """
        settings = self._require_configured()
        if state.ion_distribution is None:
            raise ValueError("coverage is a property of an ion distribution; this state has none")

        return tuning_coverage(
            distribution=state.ion_distribution,
            z_index=self._resolve_z_index(state),
            tuning=settings.tuning,
            transition=settings.transition,
            laser=settings.laser,
        )

    def scan_velocities(self) -> Quantity:
        """The ``v_z`` axis this configuration resolves — the IVDF's abscissa."""
        settings = self._require_configured()
        return settings.scan.velocities(transition=settings.transition, laser=settings.laser)

    # ── internals ───────────────────────────────────────────────────────────────

    def _require_configured(self) -> _Settings:
        if self._settings is None:
            raise RuntimeError(
                "the LIF channel has not been configured. doc 08 §6 gives every instrument "
                "a manifest block; an unconfigured instrument would otherwise silently "
                "adopt defaults that appear in no artifact."
            )
        return self._settings

    def _components(self) -> tuple[ZeemanComponent, ...]:
        settings = self._require_configured()
        if float(magnitude_in(settings.magnetic_field, "T")) == 0.0:
            return unsplit_pattern()
        return zeeman_pattern(
            lower=settings.transition.lower,
            upper=settings.transition.upper,
            magnetic_field=settings.magnetic_field,
            polarisation=settings.polarisation,
        )

    def _resolve_z_index(self, state: PlasmaState) -> int:
        """Which grid point the measurement volume sits at, for this particular state.

        An explicit ``z_index`` from the manifest is honoured exactly — doc 07's
        sensitivity study sweeps it deliberately, including to the wall (``0``), and a
        caller that names an index has already decided where the volume sits. Left
        unset, this resolves to :func:`_sheath_edge_index`: the doc 01 §5.1 sheath-edge
        location this channel measures at by default. See the module docstring's
        "detection gate" section for why the wall used to be the default and is not one
        any more, and :meth:`_expected_ion_speed` for the matching change to the
        tuning-range gate.
        """
        settings = self._require_configured()
        if settings.z_index is not None:
            return settings.z_index
        return _sheath_edge_index(state, tolerance=settings.edge_tolerance)

    def _expected_ion_speed(self, params: PlasmaParams) -> float:
        """Speed the gate compares against the tuning ceiling.

        The **sheath-edge** speed the channel actually has to resolve — the Bohm speed
        ``c_s`` (doc 03 §2.1's marginal Bohm criterion, the speed ions have *at* the
        edge by definition) plus a few thermal widths — never the free-fall speed across
        the full sheath drop. Doc 01 §5.1's LIF-3 note is explicit that the wall speed is
        the wrong question for a scanning single-frequency laser: a 250 eV Ar+ ion's
        51.9 GHz Doppler shift is far outside the 20 GHz mode-hop-free range, and the
        framework's answer is not to pretend otherwise but to measure at (and just
        inside) the sheath edge instead, where the drift is order ``c_s`` and the shift
        (4-12 GHz) is comfortably reachable — see :meth:`_resolve_z_index`, which is
        where this channel now actually samples by default.

        **The wall IVDF this implies is consequently an inferred quantity, not a
        measured one.** Doc 01 §5.1's mitigation is to *propagate* the sheath-edge entry
        distribution this channel measures through the reconstructed sheath field to
        obtain the wall IVDF, and doc 06 §4 makes that propagation's error a first-class
        term in the error budget — not an unstated approximation this gate, or any
        caller of :meth:`is_informative`, is entitled to ignore.

        Still deliberately the *largest* speed the channel would have to resolve rather
        than a typical one: the gate exists to refuse a measurement the laser cannot
        make, and refusing on the basis of an average would admit configurations where
        the informative half of the distribution is off the end of the scan. What moved
        is *where* that largest speed is anchored, not the reasoning for adding a margin
        to it.
        """
        c_s = float(magnitude_in(bohm_speed(params), "m/s"))
        thermal = math.sqrt(params.T_i_J / params.species.mass_kg)
        return c_s + INFORMATIVE_SPEED_MARGIN * thermal
