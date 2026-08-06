"""A simulated retarding field energy analyser (RFEA) — WBS 2.12, doc 11 §9 item 7.

The second of the two reference instruments this comparison needs — see the package
docstring of :mod:`vpl.instruments.probe` for the framework-level statement of "built to
measure, not to win", and :mod:`vpl.instruments.probe.langmuir` for the sibling that
statement was written against.

## The physics

A discriminator grid at potential ``V`` transmits only ions with normal kinetic energy
``E > e V``; the collected current is therefore

    I(V) = e A T integral_{E >= eV} Gamma(E) dE

where ``Gamma(E)`` is the *flux*-weighted ion energy distribution (particles per unit
area, time and energy — what a current measurement actually samples, as opposed to the
density-weighted IEDF), ``A`` the collector area and ``T`` the grids' combined geometric
transmission. Differentiating,

    -dI/dV = e^2 A T Gamma(eV)

which is the textbook RFEA analysis (Böhm & Perrin 1993; Sudit & Woods 1994) and is what
:func:`estimate_iedf` computes by finite differences on raw ``(V, I)`` data — the naive
analysis a real experimentalist runs, exactly as
:func:`vpl.instruments.probe.langmuir.estimate_from_iv_curve` is for the probe.

``Gamma(E)`` is built directly from the framework's own
:class:`~vpl.core.state.VelocityDistribution` (:func:`flux_iedf_ev`), which is the
quantity doc 03 §5.2 already emulates and :mod:`vpl.instruments.lif.instrument` already
reads for the same field — this module adds no new physics to the ion side, only the
analyser's own transmission and resolution.

## The systematic this module exists to reproduce

Real RFEA grids are not infinitely fine meshes: the potential between two neighbouring
wires sags into the gap between grids (Böhm & Perrin 1993 call this the "fringing field"),
so an ion's actual turning point is smeared over an energy window set by the wire spacing
relative to the inter-grid separation rather than by the swept voltage's own precision.
:func:`GridGeometry.resolution_fwhm_ev` is that window, and :func:`collected_current_a`
convolves ``Gamma(E)`` with it *before* integrating — so the smearing is in the generative
model, and the naive analysis of :func:`estimate_iedf` recovers the smeared spectrum, not
the true one. A narrow input IEDF comes back broadened by exactly this amount; that is the
load-bearing behaviour ``test_probe_rfea.py`` checks, and it is what actually limits a real
RFEA's energy resolution (Sudit & Woods 1994 report ~1-2 eV FWHM for a typical four-grid
design at few-hundred-eV ion energies, which is the regime :data:`_RESOLUTION_COEFFICIENT`
is chosen to reproduce at the default geometry).

## What is deliberately not modelled

- **Space-charge distortion inside the analyser** (the transmitted ion beam is dense
  enough to perturb its own potential between grids) — a second-order effect next to the
  fringing-field resolution this module exists to demonstrate, and the reason real
  designs keep the aperture small.
- **Secondary-electron suppression grid imperfection** and **collector work-function
  offsets** — both shift the effective zero of the voltage axis, not its resolution, and
  a shifted zero does not change the width comparison this module is built to make.
- **Angular acceptance.** :func:`flux_iedf_ev` uses only the sheath-normal component of
  the ion distribution, the same one-dimensional treatment
  :mod:`vpl.instruments.lif.instrument` uses for the IVDF; a finite acceptance angle
  would additionally smear the low-energy edge, which would only strengthen (never
  weaken) the resolution-broadening effect this module already demonstrates.

## Citations

- C. Böhm and J. Perrin, "Retarding-field analyzer for measurements of ion energy
  distributions and secondary electron emission coefficients in low-pressure radio
  frequency discharges", Rev. Sci. Instrum. 64 (1993) 31 — the fringing-field resolution
  limit this module's :func:`GridGeometry.resolution_fwhm_ev` follows.
- I. D. Sudit and R. C. Woods, "A workable retarding field energy analyzer for use in
  low pressure discharges", J. Appl. Phys. 76 (1994) 4488 — practical grid design,
  transmission and achieved resolution.
- J. R. Woodworth, N. B. Wood and B. K. Pikkert, "Ion energy distributions on the
  radio-frequency biased electrode of a plasma etching reactor", J. Appl. Phys. 90
  (2001) 6588 — the ``-dI/dV`` derivation quoted directly here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.core.constants import ELEMENTARY_CHARGE
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
    VelocityDistribution,
)
from vpl.core.units import Q_, magnitude_in

__all__ = [
    "CURRENT_UNITS",
    "RFEA_INSTRUMENT_ID",
    "GridGeometry",
    "IedfEstimate",
    "RfeaInstrument",
    "collected_current_a",
    "estimate_iedf",
    "flux_iedf_ev",
]

type FloatArray = NDArray[np.float64]

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))

#: The doc 08 §7 artifact group this channel would write to.
RFEA_INSTRUMENT_ID: Final[InstrumentId] = "rfea"

#: Units of what :meth:`RfeaInstrument.forward` and :meth:`observe` return.
CURRENT_UNITS: Final[str] = "A"

# ── default geometry — a small aperture, few-grid analyser (Sudit & Woods 1994 style) ──

_DEFAULT_COLLECTOR_RADIUS_M: Final[float] = 1.0e-3
_DEFAULT_WIRE_SPACING_M: Final[float] = 5.0e-5
_DEFAULT_GRID_SEPARATION_M: Final[float] = 1.0e-3

#: Combined geometric transparency of the discriminator, suppressor and shield grids —
#: Böhm & Perrin (1993) tabulate ~0.4-0.5 per nickel mesh grid; three in series gives the
#: product used here.
_DEFAULT_TRANSMISSION: Final[float] = 0.15

#: The fixed electron-repeller grid bias whose fringing field, not the swept discriminator
#: voltage, sets the resolution — see the module docstring and
#: :func:`GridGeometry.resolution_fwhm_ev`.
_DEFAULT_REPELLER_BIAS_V: Final[float] = 40.0

#: Order-unity geometric factor relating the fringing-field energy spread to the wire
#: spacing / grid separation ratio and the repeller bias — chosen so the default geometry
#: reproduces the ~1-2 eV FWHM Sudit & Woods (1994) report for a comparable four-grid
#: design at a similar aperture.
_RESOLUTION_COEFFICIENT: Final[float] = 0.5

# ── default sweep ───────────────────────────────────────────────────────────────────

_DEFAULT_SWEEP_START_V: Final[float] = 0.0
_DEFAULT_SWEEP_STOP_V: Final[float] = 400.0
_DEFAULT_SWEEP_POINTS: Final[int] = 201

#: Which spatial grid point the analyser looks at — the wall, doc 02 §2's ``z = 0``,
#: matching an RFEA's usual placement flush with (or just behind an aperture in) the
#: target electrode.
_DEFAULT_Z_INDEX: Final[int] = 0

#: Points the interpolated energy axis is built on, for the resolution convolution and
#: the transmitted-current integral.
_ENERGY_AXIS_POINTS: Final[int] = 2000

# ── noise and calibration ───────────────────────────────────────────────────────────

_CURRENT_RELATIVE_NOISE: Final[float] = 0.02
_MINIMUM_CURRENT_UNCERTAINTY_A: Final[float] = 1.0e-9
_DEFAULT_SCALE_UNCERTAINTY: Final[float] = 0.03

#: Below this bulk density there are too few ions in the beam for the collector current
#: to clear the analyser's own noise floor at the default geometry.
_DETECTION_FLOOR_N_0: Final[float] = 1.0e14


def flux_iedf_ev(
    distribution: VelocityDistribution, *, z_index: int, mass_kg: float
) -> tuple[FloatArray, FloatArray]:
    """The flux-weighted IEDF an RFEA samples, from the framework's own ion distribution.

    ``Gamma(E) dE`` is the particle flux toward the wall carried by ions in ``[E, E+dE]``.
    For the sheath-normal-only :class:`~vpl.core.state.VelocityDistribution` doc 03 §5.2
    emulates, with ``E = (1/2) m v_z^2`` and only the wall-bound branch (``v_z < 0``,
    doc 02 §2's sign convention),

        Gamma(E) dE = -v_z f_i(v_z) dv_z  =>  Gamma(E) = e f_i(v_z(E)) / m

    (derivation: substitute ``dE = -m v_z dv_z`` and simplify; the extra factor of ``e``
    relative to the naive ``f_i / m`` is exactly what makes
    ``integral Gamma(E) dE = particle_flux_toward_wall`` when ``E`` is expressed in eV —
    checked directly in ``test_probe_rfea.py``, which is the safest verification for a
    unit-conversion step like this one).

    Args:
        distribution: The ion velocity distribution, e.g. ``state.ion_distribution``.
        z_index: Which grid point to read — the analyser's position.
        mass_kg: Ion mass.

    Returns:
        ``(energy_ev, gamma_ev)``, ascending in energy, ``gamma_ev`` in
        particles / (m^2 s eV). Only the wall-bound branch (``E >= 0``) is included.
    """
    velocities = distribution.v_m_per_s
    values = distribution.values[z_index, :]

    toward_wall = velocities < 0.0
    v_neg = velocities[toward_wall]
    f_neg = values[toward_wall]
    if v_neg.size < 2:
        raise ValueError(
            "the ion distribution carries fewer than two wall-bound (v_z < 0) samples; "
            "an RFEA has nothing to transmit from this state"
        )

    energy_ev_descending = 0.5 * mass_kg * v_neg**2 / _E_C
    gamma_ev_descending = _E_C * f_neg / mass_kg

    # v_neg is ascending (least negative last), so energy is descending; reverse both to
    # hand back the ascending axis every caller here (and np.interp) needs.
    energy_ev = energy_ev_descending[::-1]
    gamma_ev = gamma_ev_descending[::-1]
    return np.asarray(energy_ev), np.asarray(gamma_ev)


def _reverse_cumulative_tail(y: FloatArray, x: FloatArray) -> FloatArray:
    """``tail[i] = integral_{x_i}^{x_max} y dx``. See
    :func:`vpl.instruments.probe.langmuir._reverse_cumulative_tail`, whose docstring
    this repeats: the two modules do not share a private helper across a package
    boundary that doc 08 §1 principle 3 keeps between plugins."""
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    forward = np.concatenate(([0.0], np.cumsum(increments)))
    return np.asarray(forward[-1] - forward)


def _gaussian_convolve(y: FloatArray, x: FloatArray, *, fwhm: float) -> FloatArray:
    """Convolve ``y(x)`` (on a uniform grid) with a Gaussian of the given FWHM.

    Zero FWHM is the identity — the "ideal grid" limit — rather than a degenerate kernel,
    which is what lets :func:`collected_current_a` be called with
    ``resolution_fwhm_ev=0.0`` for the exact-transmission checks in
    ``test_probe_rfea.py`` without a special case at the call site.
    """
    if fwhm <= 0.0:
        return y
    spacing = float(x[1] - x[0])
    sigma = fwhm / math.sqrt(8.0 * math.log(2.0))
    sigma_points = sigma / spacing
    half_width = max(1, math.ceil(4.0 * sigma_points))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_points) ** 2)
    kernel /= np.sum(kernel)
    return np.asarray(np.convolve(y, kernel, mode="same"))


@dataclass(frozen=True, slots=True)
class GridGeometry:
    """An RFEA's collector and grid stack — Böhm & Perrin (1993); Sudit & Woods (1994).

    Attributes:
        collector_area_m2: Ion-collecting area.
        wire_spacing_m: Grid mesh pitch — the aperture size the fringing field leaks
            through.
        grid_separation_m: Distance between the discriminator and its neighbouring
            grids, over which the fringing field decays.
        transmission: Combined geometric transparency of the grid stack, ``(0, 1]``.
    """

    collector_area_m2: float
    wire_spacing_m: float
    grid_separation_m: float
    transmission: float

    def __post_init__(self) -> None:
        if not self.collector_area_m2 > 0.0:
            raise ValueError(f"collector area must be positive, got {self.collector_area_m2}")
        if not self.wire_spacing_m > 0.0:
            raise ValueError(f"wire spacing must be positive, got {self.wire_spacing_m}")
        if not self.grid_separation_m > 0.0:
            raise ValueError(f"grid separation must be positive, got {self.grid_separation_m}")
        if not 0.0 < self.transmission <= 1.0:
            raise ValueError(f"transmission must lie in (0, 1], got {self.transmission}")

    def resolution_fwhm_ev(self, *, repeller_bias_v: float = _DEFAULT_REPELLER_BIAS_V) -> float:
        """The fringing-field energy resolution — Böhm & Perrin (1993) §III.

        ``FWHM ~ kappa (d / D) e V_r``: the fringing potential at a grid wire leaks a
        distance of order the wire spacing ``d`` into the inter-grid gap ``D``, so a
        finer mesh or a wider gap both narrow the window. Set by the *fixed*
        electron-repeller bias rather than the swept discriminator voltage — see the
        module docstring — which is what makes this a single, sweep-independent number
        the resolution convolution can use.
        """
        return (
            _RESOLUTION_COEFFICIENT
            * (self.wire_spacing_m / self.grid_separation_m)
            * repeller_bias_v
        )

    def __repr__(self) -> str:
        return (
            f"GridGeometry(A={self.collector_area_m2:.3g} m^2, "
            f"d/D={self.wire_spacing_m / self.grid_separation_m:.3g}, "
            f"T={self.transmission:.2f})"
        )


def collected_current_a(
    voltage_v: ArrayLike,
    *,
    energy_ev: FloatArray,
    gamma_ev: FloatArray,
    geometry: GridGeometry,
    resolution_fwhm_ev: float,
) -> FloatArray:
    """``I(V) = e A T integral_{E >= eV} Gamma_smeared(E) dE`` — see the module docstring.

    ``Gamma`` is convolved with the resolution kernel *before* the integral, so the
    smearing is part of the generative model and the naive :func:`estimate_iedf` recovers
    the smeared spectrum rather than the true one.

    Args:
        voltage_v: Swept discriminator voltage.
        energy_ev: Ascending energy axis of ``gamma_ev`` — from :func:`flux_iedf_ev`.
        gamma_ev: The flux IEDF on ``energy_ev``.
        geometry: The analyser's collector area and transmission.
        resolution_fwhm_ev: The energy resolution to convolve with. ``0.0`` for the
            ideal-grid limit.

    Returns:
        The collected current at each swept voltage.
    """
    voltages = np.atleast_1d(np.asarray(voltage_v, dtype=np.float64))

    axis_max = float(energy_ev[-1])
    axis = np.linspace(0.0, axis_max, _ENERGY_AXIS_POINTS)
    gamma_on_axis = np.interp(axis, energy_ev, gamma_ev, left=0.0, right=0.0)
    smeared = _gaussian_convolve(gamma_on_axis, axis, fwhm=resolution_fwhm_ev)

    tail = _reverse_cumulative_tail(smeared, axis)
    tail_at = np.interp(
        np.clip(voltages, 0.0, axis_max), axis, tail, left=float(tail[0]), right=0.0
    )

    return np.asarray(_E_C * geometry.collector_area_m2 * geometry.transmission * tail_at)


@dataclass(frozen=True, slots=True)
class IedfEstimate:
    """What :func:`estimate_iedf` extracts from a raw ``(V, I)`` sweep.

    Attributes:
        energy_ev: The (uniform) energy axis the estimate is reported on.
        gamma_ev: The recovered flux IEDF, ``-dI/dV`` rescaled to particles/(m^2 s eV).
        mean_energy_ev: The flux-weighted mean energy of the recovered spectrum.
    """

    energy_ev: FloatArray
    gamma_ev: FloatArray
    mean_energy_ev: float

    def __repr__(self) -> str:
        return f"IedfEstimate(<E>={self.mean_energy_ev:.4g} eV, {self.energy_ev.size} points)"


def estimate_iedf(
    voltage_v: ArrayLike, current_a: ArrayLike, *, geometry: GridGeometry
) -> IedfEstimate:
    """The naive ``-dI/dV`` analysis a real RFEA measurement is reduced with.

    Not tied to :class:`RfeaInstrument` or to a simulated state — this operates on raw
    ``(V, I)`` arrays exactly as they come off a real digitiser, for the same reason
    :func:`vpl.instruments.probe.langmuir.estimate_from_iv_curve` does: the comparison
    doc 11 §9 item 7 asks for is against what an experimentalist does with the data.

    Args:
        voltage_v: Swept discriminator voltage, any order.
        current_a: Collected current at each voltage.
        geometry: The analyser's collector area and transmission, needed to convert the
            raw derivative into a properly scaled flux IEDF.

    Returns:
        The recovered :class:`IedfEstimate`.

    Raises:
        ValueError: If the arrays disagree in shape or there are too few points to
            differentiate.
    """
    voltages = np.asarray(voltage_v, dtype=np.float64)
    currents = np.asarray(current_a, dtype=np.float64)
    if voltages.shape != currents.shape:
        raise ValueError(
            f"voltage and current arrays must have the same shape, got {voltages.shape} "
            f"and {currents.shape}"
        )
    if voltages.size < 2:
        raise ValueError("at least two sweep points are needed to estimate a derivative")

    order = np.argsort(voltages)
    voltages = voltages[order]
    currents = currents[order]

    derivative = np.gradient(currents, voltages)
    gamma_ev = np.clip(
        -derivative / (_E_C**2 * geometry.collector_area_m2 * geometry.transmission), 0.0, None
    )

    total = float(np.trapezoid(gamma_ev, voltages))
    mean_energy_ev = (
        float(np.trapezoid(voltages * gamma_ev, voltages)) / total if total > 0.0 else float("nan")
    )

    return IedfEstimate(energy_ev=voltages, gamma_ev=gamma_ev, mean_energy_ev=mean_energy_ev)


# ── the doc 08 §4 Instrument contract ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Settings:
    geometry: GridGeometry
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


class RfeaInstrument:
    """A simulated RFEA implementing :class:`~vpl.core.protocols.Instrument`.

    Doc 00 §5.2 E1: a plugin over the same contract every other channel uses. Like
    :class:`~vpl.instruments.probe.langmuir.LangmuirProbe`, :meth:`forward` and
    :meth:`observe` share :meth:`_predict` so they cannot drift apart (doc 04 §9).
    """

    __slots__ = ("_calibration_rng", "_geometry", "_rng", "_settings", "instrument_id")

    def __init__(
        self,
        *,
        root_seed: int,
        geometry: GridGeometry | None = None,
        instrument_id: InstrumentId = RFEA_INSTRUMENT_ID,
    ) -> None:
        self.instrument_id = instrument_id
        self._geometry = (
            geometry
            if geometry is not None
            else GridGeometry(
                collector_area_m2=math.pi * _DEFAULT_COLLECTOR_RADIUS_M**2,
                wire_spacing_m=_DEFAULT_WIRE_SPACING_M,
                grid_separation_m=_DEFAULT_GRID_SEPARATION_M,
                transmission=_DEFAULT_TRANSMISSION,
            )
        )
        self._settings: _Settings | None = None
        self._rng = generator(root_seed, Stream.DETECTOR_NOISE)
        self._calibration_rng = generator(root_seed, Stream.CALIBRATION)

    # ── configuration ───────────────────────────────────────────────────────────

    def configure(self, cfg: InstrumentConfig) -> None:
        """Apply one entry of the manifest's ``instruments:`` list — doc 08 §6.

        ``sweep_start_v``, ``sweep_stop_v``, ``sweep_points``
            The swept discriminator voltage.
        ``z_index``
            Spatial grid point the analyser looks at — defaults to the wall.
        ``wire_spacing_m``, ``grid_separation_m``, ``transmission``, ``collector_area_m2``
            Override the constructed grid geometry.
        ``noise``
            Whether :meth:`observe` applies current noise and the calibration-gain draw.
        """
        geometry = self._geometry
        overrides = {
            key: cfg.get(key)
            for key in ("wire_spacing_m", "grid_separation_m", "transmission", "collector_area_m2")
        }
        if any(value is not None for value in overrides.values()):
            geometry = GridGeometry(
                collector_area_m2=_as_float(
                    overrides["collector_area_m2"],
                    default=geometry.collector_area_m2,
                    key="collector_area_m2",
                ),
                wire_spacing_m=_as_float(
                    overrides["wire_spacing_m"],
                    default=geometry.wire_spacing_m,
                    key="wire_spacing_m",
                ),
                grid_separation_m=_as_float(
                    overrides["grid_separation_m"],
                    default=geometry.grid_separation_m,
                    key="grid_separation_m",
                ),
                transmission=_as_float(
                    overrides["transmission"], default=geometry.transmission, key="transmission"
                ),
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
        """Derive the current-measurement gain from a reference standard — doc 04 §7.3."""
        standard = refs.for_quantity("rfea_current_scale")
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
        """The noiseless transmitted-current curve — doc 04 §9."""
        values = self._predict(state)
        return Observable(
            instrument_id=self.instrument_id, values=values, units=CURRENT_UNITS, window=w
        )

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        """A noisy, imperfectly-calibrated transmitted-current curve — doc 04 §9.

        Shares :meth:`_predict` with :meth:`forward`; see
        :meth:`~vpl.instruments.probe.langmuir.LangmuirProbe.observe` for the identical
        pattern and the same doc 04 V-30 exactness standard this is held to.
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
        if state.ion_distribution is None:
            raise ValueError(
                f"a {state.fidelity.name} state reached the RFEA forward model with no "
                "ion distribution. An RFEA measures the IEDF and nothing else, so there "
                "is no signal to compute without one — see "
                "vpl.instruments.lif.instrument.LifInstrument.forward for the identical "
                "requirement on the LIF channel."
            )

        energy_ev, gamma_ev = flux_iedf_ev(
            state.ion_distribution, z_index=settings.z_index, mass_kg=state.params.species.mass_kg
        )
        voltages = np.linspace(settings.sweep_start_v, settings.sweep_stop_v, settings.sweep_points)
        return collected_current_a(
            voltages,
            energy_ev=energy_ev,
            gamma_ev=gamma_ev,
            geometry=settings.geometry,
            resolution_fwhm_ev=settings.geometry.resolution_fwhm_ev(),
        )

    # ── the likelihood and the gate ─────────────────────────────────────────────

    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb:
        """This channel's Gaussian term in the doc 05 §3.2 sum — see
        :meth:`~vpl.instruments.probe.langmuir.LangmuirProbe.likelihood`, whose reasoning
        is identical: the noise here is a current-measurement chain, not a counting
        statistic."""
        if obs.shape != pred.shape:
            raise ValueError(
                f"an observation and its prediction must have the same shape, got "
                f"{obs.shape} and {pred.shape}"
            )
        if obs.units != pred.units:
            raise ValueError(f"observation is in {obs.units!r} and prediction in {pred.units!r}")
        if np.any(obs.uncertainty <= 0.0):
            raise ValueError(
                "a zero uncertainty reached the RFEA likelihood; check is_informative "
                "before forming a term"
            )

        residual = (obs.values - pred.values) / obs.uncertainty
        return float(
            -0.5 * np.sum(residual**2) - np.sum(np.log(obs.uncertainty * math.sqrt(2.0 * math.pi)))
        )

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        """The doc 01 IF-6 gate, applied to the analyser's own signal-noise floor."""
        return self.metadata().detection_floor.admits(state_guess.n_0)

    def metadata(self) -> InstrumentMetadata:
        """Identity, citations and the detection floor — doc 00 C2, doc 01 IF-6."""
        return InstrumentMetadata(
            instrument_id=self.instrument_id,
            name=(
                "Simulated retarding field energy analyser (reference instrument, doc 11 §9 item 7)"
            ),
            version="0.1.0",
            citations=(
                Citation(
                    key="bohm-perrin-1993",
                    reference=(
                        "C. Böhm and J. Perrin, 'Retarding-field analyzer for "
                        "measurements of ion energy distributions and secondary "
                        "electron emission coefficients in low-pressure radio frequency "
                        "discharges', Rev. Sci. Instrum. 64 (1993) 31"
                    ),
                    doi="10.1063/1.1144402",
                ),
                Citation(
                    key="sudit-woods-1994",
                    reference=(
                        "I. D. Sudit and R. C. Woods, 'A workable retarding field "
                        "energy analyzer for use in low pressure discharges', "
                        "J. Appl. Phys. 76 (1994) 4488"
                    ),
                    doi="10.1063/1.357280",
                ),
                Citation(
                    key="woodworth-2001",
                    reference=(
                        "J. R. Woodworth, N. B. Wood and B. K. Pikkert, 'Ion energy "
                        "distributions on the radio-frequency biased electrode of a "
                        "plasma etching reactor', J. Appl. Phys. 90 (2001) 6588"
                    ),
                    doi="10.1063/1.1418001",
                ),
            ),
            detection_floor=DetectionFloor(
                quantity="n_0", threshold=Q_(_DETECTION_FLOOR_N_0, "m**-3"), requirement="IF-6"
            ),
            description=(
                "WBS 2.12 reference instrument for the doc 11 §9 item 7 / WBS 5.4 "
                "comparative study. Not part of the doc 01 diagnostic suite."
            ),
        )

    def __repr__(self) -> str:
        return f"RfeaInstrument({self.instrument_id!r}, {self._geometry!r})"

    def _require_configured(self) -> _Settings:
        if self._settings is None:
            raise RuntimeError(
                f"{self.instrument_id}: configure() must be called before use. doc 08 §6 "
                "gives every instrument a manifest block; an unconfigured analyser would "
                "otherwise silently sweep defaults that appear in no artifact."
            )
        return self._settings
