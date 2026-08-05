"""The detuning scan and the velocity-resolved signal — doc 04 §3.1 and §3.2.

Doc 04 §3.2 states what the diagnostic *is*::

    nu_L (1 - v.k/c)  =  nu_0        =>       v_par  =  c (1 - nu_0/nu_L)

    so scanning nu_L maps out f_i(v_par) where v_par is along the laser propagation
    direction.

and doc 02 §4.2 supplies the geometry that makes ``v_par`` informative about the sheath:
``v_par = v_z sin(15 deg) = 0.259 v_z``. Together they give the mapping this module
implements, and its inverse, which is the measurement.

## The signal

Doc 04 §3.1 writes ``S(nu_L) ~ integral n_2(v; nu_L) A_23 h nu_23 dv``. The proportionality
is not laziness on the specification's part — doc 02 §11 records that LIF is **not**
absolutely calibrated and doc 05 §2.2 makes the amplitude an inferred nuisance parameter —
so this module returns the integral itself,

    n_2(nu_L)  =  eta_meta  integral  f_i(v_z)  [n_2/n](v_z; nu_L)  dv_z      [m^-3]

and stops there. ``A_23 h nu_23 / 4 pi`` is a constant, and it is a constant the project
cannot source (see ``lif.yaml``); folding an invented Einstein coefficient into the output
would put a fabricated number inside a quantity whose scale is fitted anyway. The
consequence, stated plainly: **the LIF channel constrains the shape and the relative
amplitude of the IVDF, never an absolute ion density.** That is exactly what doc 02 §11
says a real LIF channel does.

## Why saturation is inside the velocity integral

See :mod:`vpl.instruments.lif.rates`. The excited fraction is evaluated per velocity class
and per Zeeman component *before* the integral over ``f_i``, because saturation is a
property of the ion's own detuning. Pulling it outside would reproduce every amplitude the
model predicts and none of the width, which is the failure mode that looks like success.

## Stated simplifications, with bounds

- **The measurement volume is a point.** Doc 04 §6.2 calls the volume integral "the single
  most commonly mishandled aspect of optical diagnostics of sheaths" and specifies the
  weighting function ``W(r)``. That is ``F3``, and ``vpl-optics`` owns it; this module
  evaluates ``f_i`` at one grid index. Composing the two is the caller's job, and the
  weighting must also be applied inside the inverse model per doc 04 §6.2. **Bound:** doc 04
  §6.2 gives the numbers — ``n_e`` varies by an order of magnitude over 890 um, and the
  measurement volume is 75 um. The volume therefore spans ``75/890`` of a decade, a factor
  of 1.2 in density across it, so a point evaluation at the centre is a **20 %-scale**
  amplitude bias near the wall and not a percent-level one. Doc 04 §6.2 calls exactly this
  "a systematic bias, not a resolution limit". It is not corrected here and must not be
  ignored downstream.
- **The pump is monochromatic apart from its linewidth**, which enters as a third
  Lorentzian. Valid because doc 02 LIF-L2's < 1 MHz is 16x below the natural plus
  collisional width and 1000x below the Doppler width.
- **Steady state.** The upper level is taken to have reached the steady state of doc 04
  §3.1 within the gate. The relaxation time is ``1/Gamma`` = 32 ns against doc 02 LIF-D1's
  >= 2 ns gate, so this is *not* automatically satisfied for the shortest gates: a 2 ns
  gate samples the transient, and the steady-state model over-predicts the signal there by
  up to ``1 - exp(-t/tau)``, a factor of 16 at 2 ns. Time-resolved LIF is out of WBS 2.6
  scope; the framework should not be used with gates below a few times 32 ns without
  adding it.
- **No radiation trapping of the fluorescence and no absorption of the pump along the
  beam.** Both are optically-thin assumptions, reasonable for a metastable fraction of
  1e-2 at 1e17 m^-3 over a 150 mm chord, and both belong to ``F3`` if they stop being.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vpl.core.constants import BOLTZMANN, SPEED_OF_LIGHT
from vpl.core.state import Species, VelocityDistribution
from vpl.core.units import Q_, ArrayQuantity, Quantity, ScalarQuantity, magnitude_in
from vpl.instruments.lif.laser import Laser, TuningRange
from vpl.instruments.lif.rates import lorentzian_response, steady_state_excited_fraction
from vpl.instruments.lif.transition import ProbeTransition
from vpl.instruments.lif.zeeman import ZeemanComponent

__all__ = [
    "DetuningScan",
    "TuningCoverage",
    "doppler_detuning_hz",
    "fluorescence_response",
    "resonant_velocity_m_per_s",
    "thermal_doppler_1e_halfwidth",
    "tuning_coverage",
]

type FloatArray = NDArray[np.float64]

_C_M_PER_S: float = float(magnitude_in(SPEED_OF_LIGHT, "m/s"))
_K_B_J_PER_K: float = float(magnitude_in(BOLTZMANN, "J/K"))

#: Fraction of the population that may fall outside the tuning window before
#: :attr:`TuningCoverage.is_truncated` reports the scan as clipped.
#:
#: One part in a thousand. Doc 06 §4 budgets whole systematics at the percent level, so a
#: truncation an order of magnitude below that is genuinely negligible and anything above
#: it is a piece of the distribution the instrument did not measure and the inversion must
#: not be allowed to assume.
TRUNCATION_TOLERANCE: float = 1.0e-3

#: Largest velocity-grid spacing, in units of the homogeneous line width, that the
#: velocity integral will accept.
#:
#: One sample per full width is already marginal — the trapezoid error on a Lorentzian
#: sampled that coarsely is tens of percent — but it is where the answer stops being
#: merely inaccurate and starts being arbitrary, and it is the threshold a caller can be
#: told about without the guard firing on a legitimately coarse but adequate grid. See
#: :func:`_check_velocity_grid_resolves` for how this was found.
MIN_SAMPLES_PER_HOMOGENEOUS_WIDTH: float = 1.0


# ── the doc 04 §3.2 mapping ─────────────────────────────────────────────────────


def doppler_detuning_hz(
    velocity_m_per_s: FloatArray, *, pump_frequency_hz: float, projection_factor: float
) -> FloatArray:
    """Laser detuning at which an ion of velocity ``v_z`` comes into resonance.

    ``dnu = nu_0 v_z sin(theta_L) / c``, the first-order form of doc 04 §3.2's
    ``nu_L (1 - v.k/c) = nu_0``. First order is exact to two parts in ``1e9`` at the
    25.8 km/s tuning ceiling, which is four orders of magnitude below the 2 MHz frequency
    axis uncertainty of doc 02 §11.
    """
    velocity = np.asarray(velocity_m_per_s, dtype=np.float64)
    return np.asarray(pump_frequency_hz * velocity * projection_factor / _C_M_PER_S)


def resonant_velocity_m_per_s(
    detuning_hz: FloatArray, *, pump_frequency_hz: float, projection_factor: float
) -> FloatArray:
    """``v_z`` the scan is sampling at each detuning — the inverse of the mapping.

    The division by ``sin(theta_L)`` is doc 04 §3.2's 3.86x error amplification made
    explicit: every uncertainty on the frequency axis arrives here multiplied by it.
    """
    if abs(projection_factor) < np.finfo(np.float64).tiny:
        raise ValueError("a beam parallel to the electrode resolves no v_z (doc 02 §4.2)")

    detuning = np.asarray(detuning_hz, dtype=np.float64)
    return np.asarray(detuning * _C_M_PER_S / (pump_frequency_hz * projection_factor))


def thermal_doppler_1e_halfwidth(
    *,
    temperature: Quantity,
    species: Species,
    transition: ProbeTransition,
    projection_factor: float = 1.0,
) -> ScalarQuantity:
    """``(nu_0/c) sqrt(2 k T / m)`` — the Doppler width of doc 04 §3.3's budget.

    **Named for what it is.** Doc 04 §3.3 tabulates "~734 MHz FWHM" at ``T_i`` = 0.05 eV;
    734 MHz is the 1/e half-width, and the FWHM is ``sqrt(4 ln 2) = 1.665`` times larger,
    1224 MHz. Quoting one as the other is a 40 % error in any resolution requirement
    derived from it, so the two are kept distinct here and both are pinned in
    ``test_lif_scan.py``.
    """
    kelvin = float(magnitude_in(temperature.to("K", "boltzmann"), "K"))
    speed = math.sqrt(2.0 * _K_B_J_PER_K * kelvin / species.mass_kg)
    return Q_(transition.pump_frequency_hz * speed * projection_factor / _C_M_PER_S, "Hz")


# ── the scan ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, slots=True)
class DetuningScan:
    """The frequency axis of one IVDF measurement — doc 02 §10.1's "~200-point scan".

    Attributes:
        detuning_hz: Detunings from line centre, ascending, in Hz. Read-only.
    """

    detuning_hz: FloatArray

    def __post_init__(self) -> None:
        axis = np.array(self.detuning_hz, dtype=np.float64, copy=True)
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(f"a scan needs at least 2 points on one axis, got {axis.shape}")
        if not np.all(np.isfinite(axis)):
            raise ValueError("scan detunings must be finite")
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError("scan detunings must be strictly increasing")
        axis.flags.writeable = False
        object.__setattr__(self, "detuning_hz", axis)

    @classmethod
    def uniform(
        cls, *, tuning: TuningRange, n_points: int, half_span: Quantity | None = None
    ) -> DetuningScan:
        """An evenly spaced scan, checked against the mode-hop-free range.

        Args:
            tuning: The laser's mode-hop-free range — doc 02 LIF-L3.
            n_points: Number of frequency steps. Doc 02 §10.1 budgets ~200 for a full IVDF.
            half_span: How far to scan either side of line centre. Defaults to the full
                mode-hop-free half-span.

        Raises:
            ValueError: If ``half_span`` exceeds what the laser can reach. This is the
                doc 01 §5.1 limit refusing to be scanned past, not a validation nicety.
        """
        if n_points < 2:
            raise ValueError(f"a scan needs at least 2 points, got {n_points}")

        span = half_span if half_span is not None else tuning.half_span
        tuning.require_reachable(span)
        edge = float(magnitude_in(span, "Hz"))
        return cls(detuning_hz=np.linspace(-edge, edge, n_points))

    @property
    def detuning(self) -> ArrayQuantity:
        """The frequency axis as a dimensional quantity, for module boundaries."""
        return Q_(self.detuning_hz, "Hz")

    @property
    def n_points(self) -> int:
        return int(self.detuning_hz.size)

    def velocities(self, *, transition: ProbeTransition, laser: Laser) -> ArrayQuantity:
        """The ``v_z`` each scan point resolves — the axis the IVDF is reported on."""
        return Q_(
            resonant_velocity_m_per_s(
                self.detuning_hz,
                pump_frequency_hz=transition.pump_frequency_hz,
                projection_factor=laser.projection_factor,
            ),
            "m/s",
        )

    def __repr__(self) -> str:
        # Converted through pint rather than divided by 1e9: doc 08 §5's literal rule has
        # no carve-out for numbers whose author was confident they did not matter.
        edge = Q_(float(self.detuning_hz[-1]), "Hz").to("GHz")
        return f"DetuningScan({self.n_points} points, +/-{edge:.3g~P})"


# ── the tuning-range limit, as a measured quantity ──────────────────────────────


@dataclass(frozen=True, slots=True)
class TuningCoverage:
    """How much of a distribution the laser can actually reach — doc 01 §5.1, doc 14 RS-03.

    Attributes:
        visible_fraction: Fraction of ``integral f dv`` inside the reachable velocity
            window. One means the scan sees the whole distribution.
        velocity_ceiling: The window half-width in ``v_z``.
        energy_ceiling: The same limit as an ion kinetic energy — the form doc 01 §5.1
            argues in.
        is_truncated: Whether enough population sits outside the window to matter. The
            flag exists so a caller cannot ignore truncation by ignoring a float.
    """

    visible_fraction: float
    velocity_ceiling: Quantity
    energy_ceiling: Quantity
    is_truncated: bool

    def __repr__(self) -> str:
        state = "TRUNCATED" if self.is_truncated else "complete"
        return (
            f"TuningCoverage({self.visible_fraction:.4%} visible, {state}, "
            f"|v_z| <= {self.velocity_ceiling:.4g~P} = {self.energy_ceiling:.4g~P})"
        )


def tuning_coverage(
    *,
    distribution: VelocityDistribution,
    z_index: int,
    tuning: TuningRange,
    transition: ProbeTransition,
    laser: Laser,
) -> TuningCoverage:
    """What fraction of ``f_i`` at ``z_index`` the laser can reach.

    This is the honesty gate of doc 01 §5.1. Inside a high-bias sheath the ion population
    accelerates past the ceiling and the LIF channel stops seeing it — not gradually and
    not noisily, but by never coming into resonance at any reachable laser frequency. A
    forward model that integrated over the whole distribution regardless would hand the
    inversion information the experiment cannot produce.
    """
    _check_z_index(distribution, z_index)

    ceiling = tuning.velocity_ceiling(transition=transition, laser=laser)
    ceiling_m_per_s = float(magnitude_in(ceiling, "m/s"))

    velocity = distribution.v_m_per_s
    values = distribution.values[z_index, :]
    total = float(np.trapezoid(values, velocity))
    if total <= 0.0:
        raise ValueError(
            f"the distribution at grid index {z_index} carries no population, so 'the "
            "fraction the laser can reach' is undefined rather than zero"
        )

    inside = np.where(np.abs(velocity) <= ceiling_m_per_s, values, 0.0)
    visible = float(np.trapezoid(inside, velocity)) / total

    return TuningCoverage(
        visible_fraction=visible,
        velocity_ceiling=ceiling,
        energy_ceiling=tuning.energy_ceiling(
            species=distribution.species, transition=transition, laser=laser
        ),
        is_truncated=visible < 1.0 - TRUNCATION_TOLERANCE,
    )


# ── the signal ──────────────────────────────────────────────────────────────────


def _check_z_index(distribution: VelocityDistribution, z_index: int) -> None:
    n_points = distribution.grid.n_points
    if not 0 <= z_index < n_points:
        # Negative indices are refused rather than wrapped: Python would read -1 as the
        # far boundary of the domain, which is bulk plasma, and report it as the wall.
        raise IndexError(
            f"grid index {z_index} is outside the {n_points}-point spatial grid. "
            "The wall is index 0 (doc 02 §2); a negative index would silently select the "
            "far boundary instead."
        )


def _check_velocity_grid_resolves(
    velocity: FloatArray,
    *,
    homogeneous_fwhm_hz: float,
    transition: ProbeTransition,
    laser: Laser,
) -> None:
    """Refuse a velocity grid the homogeneous line would fall between the points of.

    The integral over ``v_z`` is a trapezoid, and the integrand is the homogeneous
    response — a Lorentzian whose width in velocity is
    ``lambda Delta_nu_hom / sin(theta_L)``, 41 m/s at the registered widths. If the grid
    spacing exceeds that, the quadrature steps over the resonance and returns a signal
    that is smooth, positive, correctly shaped and several times too small.

    Found while verifying the narrow-linewidth limit in ``test_lif_scan.py``, where
    shrinking the homogeneous width to 0.1 MHz silently cut the recovered amplitude to a
    third. It is exactly the class of failure doc 07 §1 is about: the model was right and
    the answer was wrong, with nothing in the output to say which.
    """
    width_m_per_s = float(
        resonant_velocity_m_per_s(
            np.array([homogeneous_fwhm_hz]),
            pump_frequency_hz=transition.pump_frequency_hz,
            projection_factor=laser.projection_factor,
        )[0]
    )
    spacing = float(np.max(np.diff(velocity)))
    if spacing > width_m_per_s * MIN_SAMPLES_PER_HOMOGENEOUS_WIDTH:
        raise ValueError(
            f"the velocity grid steps {spacing:.3g} m/s but the homogeneous line is only "
            f"{width_m_per_s:.3g} m/s wide in velocity, so the quadrature would step over "
            "the resonance and under-report the signal without any other symptom. Refine "
            "the distribution's velocity axis, or widen the line."
        )


def fluorescence_response(
    *,
    distribution: VelocityDistribution,
    z_index: int,
    scan: DetuningScan,
    transition: ProbeTransition,
    laser: Laser,
    components: Sequence[ZeemanComponent],
    saturation: float | None = None,
) -> FloatArray:
    """Upper-state density at each scan point — ``S(nu_L)`` of doc 04 §3.1, up to scale.

    Args:
        distribution: ``f_i(z, v_z)`` from the forward physics.
        z_index: Which grid point the measurement volume sits at. Index 0 is the wall.
        scan: The laser frequency axis.
        transition: The doc 02 §5.3 probe pair.
        laser: The beam, whose grazing angle sets the projection and whose intensity sets
            the saturation.
        components: The Zeeman pattern — a one-element comb for an unsplit line.
        saturation: ``S = I/I_sat``. Defaults to the laser's own; supplied explicitly by
            the doc 07 F-13 sweep, which varies ``S`` without varying anything else.

    Returns:
        ``n_2(nu_L)`` in ``m^-3``, one value per scan point. **Not** multiplied by the
        metastable fraction: that is a doc 05 §2.2 nuisance parameter and belongs to the
        instrument, which knows whether it is being sampled.
    """
    _check_z_index(distribution, z_index)
    if len(components) == 0:
        raise ValueError(
            "no Zeeman component was supplied. An unsplit line is a one-element comb "
            "(vpl.instruments.lif.zeeman.unsplit_pattern), not an empty one; an empty "
            "sequence would silently return a zero signal."
        )

    pump = saturation if saturation is not None else laser.saturation_parameter(transition)
    homogeneous_fwhm_hz = (
        float(magnitude_in(transition.homogeneous_fwhm, "Hz")) + laser.linewidth_hz
    )

    velocity = distribution.v_m_per_s
    density = distribution.values[z_index, :]
    _check_velocity_grid_resolves(
        velocity,
        homogeneous_fwhm_hz=homogeneous_fwhm_hz,
        transition=transition,
        laser=laser,
    )

    # Detuning of each velocity class from line centre: an (n_v,) row broadcast against
    # the (n_detuning, 1) scan column below.
    ion_shift = doppler_detuning_hz(
        velocity,
        pump_frequency_hz=transition.pump_frequency_hz,
        projection_factor=laser.projection_factor,
    )
    residual = scan.detuning_hz[:, np.newaxis] - ion_shift[np.newaxis, :]

    excited = np.zeros_like(residual)
    for component in components:
        # Each component is treated as an independent saturating sub-transition carrying
        # weight `component.weight`. At B = 0 every component sits at the same shift and
        # the weights sum to one, so this reduces *exactly* to the unsplit result — which
        # is what makes the zero-field case a verification anchor rather than a limit.
        response = lorentzian_response(residual - component.shift_hz, fwhm_hz=homogeneous_fwhm_hz)
        excited += component.weight * steady_state_excited_fraction(
            saturation=pump,
            response=response,
            lower_degeneracy=transition.lower.degeneracy,
            upper_degeneracy=transition.upper.degeneracy,
        )

    return np.asarray(np.trapezoid(excited * density[np.newaxis, :], velocity, axis=1))
