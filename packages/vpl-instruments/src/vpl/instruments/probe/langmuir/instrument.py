"""``LangmuirProbe`` — the doc 08 §4 contract over
:mod:`vpl.instruments.probe.langmuir.physics`.

One class, seven methods, and the two that matter — ``forward`` and ``observe`` — share
:meth:`LangmuirProbe._predict` exactly the way
:class:`~vpl.instruments.oes.instrument.OesInstrument` and
:class:`~vpl.instruments.lif.instrument.LifInstrument` share their own single physics
path (doc 04 §9). See the package's ``__init__.py`` for the physics this wraps and the
citations behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
from vpl.instruments.probe.langmuir.physics import (
    DEFAULT_PROBE_LENGTH_M,
    DEFAULT_PROBE_RADIUS_M,
    ProbeGeometry,
    probe_current_a,
)

__all__ = ["CURRENT_UNITS", "LANGMUIR_INSTRUMENT_ID", "LangmuirProbe"]

type FloatArray = NDArray[np.float64]

#: The doc 08 §7 artifact group this channel would write to.
LANGMUIR_INSTRUMENT_ID: Final[InstrumentId] = "langmuir_probe"

#: Units of what :meth:`LangmuirProbe.forward` and :meth:`observe` return: a collected
#: current at every swept bias, the raw quantity a real probe circuit reports.
CURRENT_UNITS: Final[str] = "A"

# ── default sweep ───────────────────────────────────────────────────────────────────

_DEFAULT_SWEEP_START_V: Final[float] = -60.0
_DEFAULT_SWEEP_STOP_V: Final[float] = 30.0
_DEFAULT_SWEEP_POINTS: Final[int] = 201

#: Which spatial grid point the probe tip sits at. The last index: doc 02 §2 places the
#: wall at ``z = 0`` with ``z`` increasing into the plasma, and a Langmuir probe is a bulk
#: diagnostic inserted away from the wall, unlike OES's line-of-sight or LIF's near-wall
#: default. A manifest overrides this for a probe scanned through the sheath.
_DEFAULT_Z_INDEX: Final[int] = -1

# ── noise and calibration ───────────────────────────────────────────────────────────
#
# Real probe circuits are dominated by transimpedance-amplifier and digitiser noise
# rather than by photon shot statistics, so this is drawn on Stream.DETECTOR_NOISE
# (doc 10 §5) rather than Stream.PHOTONS, and modelled as the usual current-measurement
# combination of a relative term and an absolute floor rather than a counting statistic.

#: Relative current noise, representative of a mid-range transimpedance amplifier chain.
_CURRENT_RELATIVE_NOISE: Final[float] = 0.02

#: Absolute noise floor, so a near-zero current does not report a near-zero uncertainty.
_MINIMUM_CURRENT_UNCERTAINTY_A: Final[float] = 1.0e-7

#: 1-sigma relative uncertainty of the current-measurement gain calibration — comparable
#: to the transimpedance-amplifier calibration uncertainties doc 02 §11 tabulates for the
#: real channels.
_DEFAULT_SCALE_UNCERTAINTY: Final[float] = 0.03

#: Below this bulk density, the probe's own sheath is no longer thin compared with its
#: radius (`lambda_D ~ r_p`) and the sheath-expansion model of
#: :mod:`vpl.instruments.probe.langmuir.physics` is not defensible — doc 01 IF-6's
#: detection-floor gate, applied here to keep the channel out of the likelihood where its
#: own forward model is not trustworthy.
_DETECTION_FLOOR_N_0: Final[float] = 1.0e14


@dataclass(frozen=True, slots=True)
class _Settings:
    geometry: ProbeGeometry
    sweep_start_v: float
    sweep_stop_v: float
    sweep_points: int
    z_index: int
    noise_enabled: bool
    scale_uncertainty: float


def _as_float(value: object, *, default: float, key: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number, got {value!r}")
    return float(value)


def _as_int(value: object, *, default: int, key: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer, got {value!r}")
    return value


def _local_value(field: ScalarField, *, z_index: int) -> float:
    """One grid point's value, at the first time sample if the field is time-resolved.

    A probe sweep is effectively instantaneous next to any recorded transient in the
    project's test states, so the simplification is to read the first sample rather than
    integrate over the acquisition window the way :mod:`vpl.instruments.oes.instrument`
    does for a gated exposure — there is no gate here to integrate.
    """
    if field.is_steady:
        return float(field.values[z_index])
    return float(field.values[0, z_index])


class LangmuirProbe:
    """A simulated single Langmuir probe implementing :class:`~vpl.core.protocols.Instrument`.

    Doc 00 §5.2 E1: a plugin over the same contract every other channel uses, with the
    noise and calibration stages sharing :meth:`_predict` with ``forward`` so the two
    "cannot drift apart" (doc 04 §9) exactly as
    :class:`~vpl.instruments.oes.instrument.OesInstrument` and
    :class:`~vpl.instruments.lif.instrument.LifInstrument` are built.
    """

    __slots__ = (
        "_calibration_rng",
        "_geometry",
        "_rng",
        "_settings",
        "instrument_id",
    )

    def __init__(
        self,
        *,
        root_seed: int,
        geometry: ProbeGeometry | None = None,
        instrument_id: InstrumentId = LANGMUIR_INSTRUMENT_ID,
    ) -> None:
        self.instrument_id = instrument_id
        self._geometry = (
            geometry
            if geometry is not None
            else ProbeGeometry(radius_m=DEFAULT_PROBE_RADIUS_M, length_m=DEFAULT_PROBE_LENGTH_M)
        )
        self._settings: _Settings | None = None
        self._rng = generator(root_seed, Stream.DETECTOR_NOISE)
        self._calibration_rng = generator(root_seed, Stream.CALIBRATION)

    # ── configuration ───────────────────────────────────────────────────────────

    def configure(self, cfg: InstrumentConfig) -> None:
        """Apply one entry of the manifest's ``instruments:`` list — doc 08 §6.

        ``sweep_start_v``, ``sweep_stop_v``, ``sweep_points``
            The swept bias. Defaults span the ion- through electron-saturation regions
            at a several-eV plasma.
        ``z_index``
            Spatial grid point the probe tip sits at. Defaults to the bulk-most point
            (doc 02 §2: highest ``z``), since a Langmuir probe is a bulk diagnostic.
        ``probe_radius_m``, ``probe_length_m``
            Override the constructed geometry.
        ``noise``
            Whether :meth:`observe` applies current noise and the calibration-gain draw.
        """
        radius = cfg.get("probe_radius_m")
        length = cfg.get("probe_length_m")
        geometry = self._geometry
        if radius is not None or length is not None:
            geometry = ProbeGeometry(
                radius_m=_as_float(radius, default=geometry.radius_m, key="probe_radius_m"),
                length_m=_as_float(length, default=geometry.length_m, key="probe_length_m"),
            )

        self._settings = _Settings(
            geometry=geometry,
            sweep_start_v=_as_float(
                cfg.get("sweep_start_v"), default=_DEFAULT_SWEEP_START_V, key="sweep_start_v"
            ),
            sweep_stop_v=_as_float(
                cfg.get("sweep_stop_v"), default=_DEFAULT_SWEEP_STOP_V, key="sweep_stop_v"
            ),
            sweep_points=_as_int(
                cfg.get("sweep_points"), default=_DEFAULT_SWEEP_POINTS, key="sweep_points"
            ),
            z_index=_as_int(cfg.get("z_index"), default=_DEFAULT_Z_INDEX, key="z_index"),
            noise_enabled=bool(cfg.get("noise", True)),
            scale_uncertainty=_DEFAULT_SCALE_UNCERTAINTY,
        )

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        """Derive the current-measurement gain from a reference standard — doc 04 §7.3.

        A Langmuir probe's absolute scale is set by the current-measurement chain (the
        transimpedance amplifier and digitiser), not by anything optical, so there is one
        coefficient here rather than the several
        :class:`~vpl.instruments.oes.instrument.OesInstrument` carries.
        """
        standard = refs.for_quantity("langmuir_current_scale")
        uncertainty = max(
            standard.relative_uncertainty, self._require_configured().scale_uncertainty
        )
        scale = 1.0 + uncertainty * float(self._calibration_rng.standard_normal())
        return Calibration(
            instrument_id=self.instrument_id,
            coefficients={"current_scale": scale},
            relative_uncertainty={"current_scale": uncertainty},
            state=CalibrationState.ESTIMATED,
            reference=standard.name,
        )

    # ── the shared code path — doc 04 §9 ────────────────────────────────────────

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        """The noiseless I-V curve the likelihood compares against — doc 04 §9."""
        values = self._predict(state)
        return Observable(
            instrument_id=self.instrument_id, values=values, units=CURRENT_UNITS, window=w
        )

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        """A noisy, imperfectly-calibrated I-V curve — doc 04 §9.

        Shares :meth:`_predict` with :meth:`forward`; noise and the calibration-gain draw
        are the only things :attr:`_Settings.noise_enabled` switches, so switching it off
        reproduces ``forward`` exactly (``test_probe_langmuir.py`` checks this bit for
        bit, the same standard doc 04 V-30 sets for the OES and LIF channels).
        """
        settings = self._require_configured()
        predicted = self._predict(state)
        uncertainty = np.sqrt(
            _MINIMUM_CURRENT_UNCERTAINTY_A**2 + (_CURRENT_RELATIVE_NOISE * np.abs(predicted)) ** 2
        )
        if not settings.noise_enabled:
            return Measurement(
                instrument_id=self.instrument_id,
                values=predicted,
                uncertainty=uncertainty,
                units=CURRENT_UNITS,
                window=w,
                calibration=CalibrationState.ESTIMATED,
            )

        scale = 1.0 + settings.scale_uncertainty * float(self._calibration_rng.standard_normal())
        values = scale * predicted + self._rng.normal(0.0, uncertainty)
        return Measurement(
            instrument_id=self.instrument_id,
            values=values,
            uncertainty=uncertainty,
            units=CURRENT_UNITS,
            window=w,
            calibration=CalibrationState.ESTIMATED,
        )

    def _predict(self, state: PlasmaState) -> FloatArray:
        """The one physics evaluation both :meth:`forward` and :meth:`observe` share."""
        settings = self._require_configured()
        z_index = settings.z_index

        plasma_potential_v = _local_value(state.field("Phi"), z_index=z_index)
        electron_density_m3 = _local_value(state.field("n_e"), z_index=z_index)
        electron_temperature_ev = _local_value(state.field("T_e"), z_index=z_index)

        voltages = self._sweep_voltages(settings)
        return probe_current_a(
            voltages,
            plasma_potential_v=plasma_potential_v,
            electron_density_m3=electron_density_m3,
            electron_temperature_ev=electron_temperature_ev,
            ion_mass_kg=state.params.species.mass_kg,
            geometry=settings.geometry,
            kappa=state.params.kappa,
        )

    @staticmethod
    def _sweep_voltages(settings: _Settings) -> FloatArray:
        return np.linspace(settings.sweep_start_v, settings.sweep_stop_v, settings.sweep_points)

    # ── the likelihood and the gate ─────────────────────────────────────────────

    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb:
        """This channel's Gaussian term in the doc 05 §3.2 sum.

        Gaussian, not Poisson: the current-measurement noise this module models is an
        amplifier/digitiser combination (see the module docstring), not a counting
        statistic, so the Poisson/Gaussian switch doc 05 §3.1 makes for OES has nothing
        to switch on here.
        """
        if obs.shape != pred.shape:
            raise ValueError(
                f"an observation and its prediction must have the same shape, got "
                f"{obs.shape} and {pred.shape}"
            )
        if obs.units != pred.units:
            raise ValueError(f"observation is in {obs.units!r} and prediction in {pred.units!r}")
        if np.any(obs.uncertainty <= 0.0):
            raise ValueError(
                "a zero uncertainty reached the Langmuir likelihood; check "
                "is_informative before forming a term"
            )

        residual = (obs.values - pred.values) / obs.uncertainty
        return float(
            -0.5 * np.sum(residual**2) - np.sum(np.log(obs.uncertainty * math.sqrt(2.0 * math.pi)))
        )

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        """The doc 01 IF-6 gate, applied to the probe's own thin-sheath validity floor."""
        return self.metadata().detection_floor.admits(state_guess.n_0)

    def metadata(self) -> InstrumentMetadata:
        """Identity, citations and the detection floor — doc 00 C2, doc 01 IF-6."""
        return InstrumentMetadata(
            instrument_id=self.instrument_id,
            name="Simulated cylindrical Langmuir probe (reference instrument, doc 11 §9 item 7)",
            version="0.1.0",
            citations=(
                Citation(
                    key="langmuir-1923",
                    reference=(
                        "I. Langmuir, 'The pressure effect and other phenomena in gaseous "
                        "discharges', J. Franklin Inst. 196 (1923) 751"
                    ),
                ),
                Citation(
                    key="mott-smith-langmuir-1926",
                    reference=(
                        "H. M. Mott-Smith and I. Langmuir, 'The theory of collectors in "
                        "gaseous discharges', Phys. Rev. 28 (1926) 727"
                    ),
                    doi="10.1103/PhysRev.28.727",
                ),
                Citation(
                    key="druyvesteyn-1930",
                    reference="M. J. Druyvesteyn, 'Der Niedervoltbogen', Z. Phys. 64 (1930) 781",
                    doi="10.1007/BF01773007",
                ),
                Citation(
                    key="merlino-2007",
                    reference=(
                        "R. L. Merlino, 'Understanding Langmuir probe current-voltage "
                        "characteristics', Am. J. Phys. 75 (2007) 1078"
                    ),
                    doi="10.1119/1.2772282",
                ),
                Citation(
                    key="godyak-demidov-2011",
                    reference=(
                        "V. A. Godyak and V. I. Demidov, 'Probe measurements of "
                        "electron-energy distribution functions: over thirty years of "
                        "history', Plasma Sources Sci. Technol. 20 (2011) 062001"
                    ),
                    doi="10.1088/0963-0252/20/6/062001",
                ),
            ),
            detection_floor=DetectionFloor(
                quantity="n_0",
                threshold=Q_(_DETECTION_FLOOR_N_0, "m**-3"),
                requirement="IF-6",
            ),
            description=(
                "WBS 2.12 reference instrument for the doc 11 §9 item 7 / WBS 5.4 "
                "comparative study. Not part of the doc 01 diagnostic suite; built to "
                "measure this framework against, not to be beaten by construction."
            ),
        )

    def __repr__(self) -> str:
        return f"LangmuirProbe({self.instrument_id!r}, {self._geometry!r})"

    def _require_configured(self) -> _Settings:
        if self._settings is None:
            raise RuntimeError(
                f"{self.instrument_id}: configure() must be called before use. doc 08 §6 "
                "gives every instrument a manifest block; an unconfigured probe would "
                "otherwise silently sweep defaults that appear in no artifact."
            )
        return self._settings
