"""Vibration, fringe jumps and the correlated-noise budget — doc 04 §5.2, doc 02 §8.1-8.2.

Two of the doc 04 §5.2 "modelled beyond the ideal" effects live here:

- **Vibration.** "Mechanical path-length noise as a 1/f + resonant-peak spectrum." Doc 02
  §8.2 IF-G2 calls mechanical drift "the dominant systematic", and it is a *common-path*
  effect — every one of the eight chords of IF-G1 shares the same floated optical table
  and largely the same beam path up to the point it splits toward the plasma, so the
  residual vibration left after IF-G2's common-path rejection moves every chord together.
  That is why :mod:`vpl.instruments.interferometry.instrument`'s likelihood treats it as a
  *correlated* (common-mode) term rather than folding it into each chord's independent
  noise — doc 05 §3.1 states the requirement directly: "Gaussian, with a correlated noise
  covariance... a diagonal covariance would overstate the information."
- **Fringe jumps.** "Phase unwrapping failure under fast transients — a discrete failure
  mode, modelled." Doc 04 §5.2: "Modelling them means the robustness study can ask whether
  the inversion detects and rejects such an event."

## What the vibration model is honest about not knowing

Doc 04 §5.2 asks for a 1/f-plus-resonance *shape*; nothing in doc 01 or doc 02 supplies a
measured vibration power spectral density to normalise it against. Inventing an absolute
PSD amplitude from nothing would be exactly the "parameter fog" doc 00 §6 warns against —
a number with no source. Instead, the shape (flicker-noise divergence at low frequency,
plus a resonance bump at a representative optical-table structural frequency) is used only
to say how the *fraction* of the noise budget already fixed by doc 02 §8.2 IF-P1 (0.1 mrad)
distributes between the correlated vibration term and the independent per-chord detector
floor, and how the vibration share scales with the acquisition window's length. The total
variance at one specific, stated reference window is pinned to IF-P1's number; nothing here
claims a vibration amplitude the docs do not supply. See
:func:`vibration_phase_std_rad`'s docstring for the exact calibration, and treat the
absolute magnitude — not the direction of its dependence on window length, and not the
existence of the correlated/independent split — as the approximate part of this module.

**That honesty was right about the amplitude and incomplete about the shape.** Every
number that fixes *where* the shape sits — the resonance frequency and its width, how far
the model integrates in frequency, how the shape's own variance splits between the 1/f and
resonant components, and what window the IF-P1 calibration is anchored at — is exactly as
unsourced as a free amplitude would have been, and none of it was registered when this
module was first written (doc 00 C1 calls that a defect). All six are now
``IF.vibration_*`` entries in the parameter registry, class ``ASSUMED``, each with a
retirement path in its description; :func:`_load_vibration_params` reads them at call
time rather than the module holding them as bare literals.

## Citations

- I. H. Hutchinson, *Principles of Plasma Diagnostics*, 2nd ed., Cambridge University
  Press (2002), ch. 6 — heterodyne interferometry and its sensitivity to path-length
  vibration (no DOI: a book).
- D. Veron, "Submillimeter Interferometry of High-Density Plasmas", in *Infrared and
  Millimeter Waves*, Vol. 2, ed. K. J. Button, Academic Press (1979), pp. 69-135 —
  heterodyne CO2/FIR plasma interferometry and the mechanical-vibration systematics that
  motivate common-path referencing (no DOI: a book chapter predating the DOI system).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.params import ParameterRegistry, default_registry
from vpl.instruments.interferometry.phase import PHASE_RESOLUTION_RAD

__all__ = [
    "DEFAULT_FRINGE_JUMP_RATE",
    "FRINGE_JUMP_MAGNITUDE_RAD",
    "independent_phase_std_rad",
    "sample_fringe_jumps",
    "vibration_phase_std_rad",
]

type FloatArray = NDArray[np.float64]

# ── vibration spectrum shape and calibration ────────────────────────────────────
#
# Every parameter of the vibration shape is an invented number with no citable source
# (module docstring) — registered ASSUMED in instruments-interferometry.yaml rather than
# held as a bare module literal (doc 00 C1). :func:`_load_vibration_params` is the single
# place this module reads them.


@dataclass(frozen=True, slots=True)
class _VibrationParams:
    """The six ``IF.vibration_*`` ASSUMED-class registry entries, resolved once per call."""

    reference_window_s: float
    common_mode_variance_fraction: float
    resonance_frequency_hz: float
    resonance_width_hz: float
    upper_bandwidth_hz: float
    one_over_f_share: float


def _load_vibration_params(registry: ParameterRegistry | None) -> _VibrationParams:
    """Resolve the vibration-model registry entries, defaulting to the shipped registry."""
    entries = registry if registry is not None else default_registry()
    return _VibrationParams(
        reference_window_s=float(entries.value_in("IF.vibration_reference_window_s", "s")),
        common_mode_variance_fraction=float(
            entries.value_in("IF.vibration_common_mode_fraction", "dimensionless")
        ),
        resonance_frequency_hz=float(entries.value_in("IF.vibration_resonance_frequency_hz", "Hz")),
        resonance_width_hz=float(entries.value_in("IF.vibration_resonance_width_hz", "Hz")),
        upper_bandwidth_hz=float(entries.value_in("IF.vibration_upper_bandwidth_hz", "Hz")),
        one_over_f_share=float(entries.value_in("IF.vibration_one_over_f_share", "dimensionless")),
    )


def _one_over_f_shape(f_min_hz: float, f_max_hz: float) -> float:
    """Un-normalised ``INTEGRAL df/f`` — a flicker-noise PSD integrates to a logarithm."""
    if f_min_hz >= f_max_hz:
        return 0.0
    return math.log(f_max_hz / f_min_hz)


def _resonance_shape(
    f_min_hz: float, f_max_hz: float, *, resonance_frequency_hz: float, resonance_width_hz: float
) -> float:
    """Un-normalised Lorentzian-resonance integral, closed form via arctan.

    ``INTEGRAL width^2 / ((f - f0)^2 + width^2) df = width * atan((f - f0)/width) + C``.
    """
    if f_min_hz >= f_max_hz:
        return 0.0
    width = resonance_width_hz
    centre = resonance_frequency_hz
    return width * (math.atan((f_max_hz - centre) / width) - math.atan((f_min_hz - centre) / width))


def _vibration_shape(duration_s: float, *, params: _VibrationParams) -> tuple[float, float]:
    """``(1/f shape, resonance shape)`` integrated over this window's resolvable band.

    The lowest frequency a finite window of length ``duration_s`` can resolve is one
    cycle per window, ``1/duration_s``: a longer window sees further into the 1/f tail,
    which is the physically correct direction — slow drift is exactly what a long
    accumulation accrues, and doc 02 IF-G2 calling mechanical drift "the dominant
    systematic" is a statement about long, not short, integrations.
    """
    f_min = 1.0 / duration_s
    f_max = params.upper_bandwidth_hz
    return (
        _one_over_f_shape(f_min, f_max),
        _resonance_shape(
            f_min,
            f_max,
            resonance_frequency_hz=params.resonance_frequency_hz,
            resonance_width_hz=params.resonance_width_hz,
        ),
    )


def vibration_phase_std_rad(
    duration_s: float, *, registry: ParameterRegistry | None = None
) -> float:
    """Common-mode (correlated-across-chords) mechanical phase-noise standard deviation.

    See the module docstring for the calibration: the *shape* (1/f divergence plus a
    resonance bump) is physically motivated; the *absolute scale* is fixed so that, at
    ``IF.vibration_reference_window_s``, the vibration term accounts for
    ``IF.vibration_common_mode_fraction`` of doc 02 §8.2 IF-P1's quoted total phase-noise
    variance (``PHASE_RESOLUTION_RAD^2``) — there being no measured spectrum in the design
    docs to normalise against independently.

    Returns 0.0 for a window too short to resolve the model's lowest frequency
    (``1/duration_s >= IF.vibration_upper_bandwidth_hz``): a gate that fast cannot see
    mechanical vibration at all, and it would be a modelling error to extrapolate noise
    power into a band the acquisition never looked at.

    Args:
        duration_s: The acquisition window's length.
        registry: Where the ``IF.vibration_*`` entries are read from. ``None`` (the
            default) uses the shipped registry.
    """
    if not duration_s > 0.0:
        raise ValueError(f"duration must be positive, got {duration_s}")

    params = _load_vibration_params(registry)

    reference_1f, reference_resonance = _vibration_shape(params.reference_window_s, params=params)
    reference_shape = (
        params.one_over_f_share * reference_1f
        + (1.0 - params.one_over_f_share) * reference_resonance
    )
    one_f, resonance = _vibration_shape(duration_s, params=params)
    shape = params.one_over_f_share * one_f + (1.0 - params.one_over_f_share) * resonance

    if shape <= 0.0 or reference_shape <= 0.0:
        return 0.0

    common_variance_at_reference = params.common_mode_variance_fraction * PHASE_RESOLUTION_RAD**2
    scaled_variance = common_variance_at_reference * shape / reference_shape
    return math.sqrt(max(scaled_variance, 0.0))


def independent_phase_std_rad(*, registry: ParameterRegistry | None = None) -> float:
    """Per-chord, uncorrelated detector/digitiser noise — the diagonal ``D`` term.

    Constant with acquisition duration: unlike the coloured vibration term, this
    represents the HgCdTe detector and digitiser electronic-noise floor (doc 02 §8.2
    IF-D1), not a mechanical spectrum, so it is not modelled as frequency-dependent.
    Combined with :func:`vibration_phase_std_rad` at ``IF.vibration_reference_window_s``,
    the two variances sum to ``PHASE_RESOLUTION_RAD^2`` exactly, by construction.

    Args:
        registry: Where ``IF.vibration_common_mode_fraction`` is read from. ``None`` (the
            default) uses the shipped registry.
    """
    params = _load_vibration_params(registry)
    return math.sqrt(1.0 - params.common_mode_variance_fraction) * PHASE_RESOLUTION_RAD


# ── fringe jumps ─────────────────────────────────────────────────────────────────

#: One fringe of phase-unwrapping ambiguity: 2*pi rad. Doc 04 §5.2 describes the failure
#: as "a plausible-looking but completely wrong density" from an integer multiple of 2*pi
#: added by a failed unwrap; exactly one fringe (the minimal, most probable failure) is
#: modelled here rather than a distribution over multiple simultaneous skips.
FRINGE_JUMP_MAGNITUDE_RAD: Final[float] = 2.0 * math.pi

#: Default fringe-jump probability per chord, per observation. Doc 04 §5.2 calls this a
#: discrete failure "under fast transients"; well-behaved, non-transient acquisition
#: should essentially never trigger it, so the default represents a rare event the doc 07
#: robustness study exercises deliberately rather than routine operation.
DEFAULT_FRINGE_JUMP_RATE: Final[float] = 1.0e-3


def sample_fringe_jumps(rng: np.random.Generator, *, n_chords: int, rate: float) -> FloatArray:
    """Independent per-chord ``+2*pi`` events at probability ``rate``.

    Independent per chord: doc 02 §8.2 IF-G1's 8 chords are separate detector/digitiser
    channels, each running its own phase-unwrapping algorithm, so a transient that defeats
    one chord's unwrap need not defeat its neighbour's.

    Always a positive jump rather than a signed one: doc 02 §8.1's heterodyne detection
    recovers the sign of the phase (see
    :mod:`vpl.instruments.interferometry.instrument`'s module docstring), and an unwrap
    failure under that scheme drops or adds whole cycles of a phase that is itself
    monotonically related to density here, not a coin flip between two directions.

    Raises:
        ValueError: If ``rate`` is not a probability in ``[0, 1]``.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"a fringe-jump rate is a probability in [0, 1], got {rate}")

    occurred = rng.random(n_chords) < rate
    return occurred.astype(np.float64) * FRINGE_JUMP_MAGNITUDE_RAD
