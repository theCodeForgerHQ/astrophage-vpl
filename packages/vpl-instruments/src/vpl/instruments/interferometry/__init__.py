"""The interferometry channel — WBS 2.8, doc 02 §8, doc 04 §5.

## Why this channel exists

The project's honest end-to-end error is 36.5 % (doc 06), diagnosed as a single-channel
degeneracy: plasma density and electron temperature enter the physics multiplied together,
so OES or a probe alone constrain a combination of ``n_e`` and ``T_e`` rather than either
one. Interferometry attacks that degeneracy from the one angle those channels cannot: the
phase shift of a probe beam through the plasma (doc 04 §5.1) depends on the electron
density **alone**, with no ``T_e`` dependence anywhere in the formula. That is why it is
worth having even though it is coarse and line-integrated — it is not a better density
measurement, it is an *independent* one, and breaking the degeneracy is what the joint fit
needs from it.

Doc 02 §8.3 is equally direct about the limits, and this package does not try to exceed
them: "no intrinsic z resolution — it cannot see the sheath structure at all"; the channel
"constrains the *boundary condition* of the sheath problem, not the sheath", and "the
inversion must not be allowed to over-weight it".

## Package layout

- :mod:`vpl.instruments.interferometry.phase` — refractive index, the electron and
  neutral-gas phase terms, and the doc 01 IF-6 detection-floor derivation. Pure physics on
  densities and lengths; no :class:`~vpl.core.state.PlasmaState` dependency.
- :mod:`vpl.instruments.interferometry.noise` — the doc 04 §5.2 vibration (1/f plus
  resonant-peak) and fringe-jump models, and the noise-budget split the instrument's
  correlated likelihood is built from.
- :mod:`vpl.instruments.interferometry.instrument` — :class:`InterferometryInstrument`,
  the doc 08 §4 :class:`~vpl.core.protocols.Instrument` contract wrapping both.

See each module's docstring for what is modelled, what is deliberately not, and the
citations behind the physics.
"""

from vpl.instruments.interferometry.instrument import (
    INTERFEROMETRY_INSTRUMENT_ID,
    PHASE_UNITS,
    InterferometryInstrument,
)
from vpl.instruments.interferometry.noise import (
    DEFAULT_FRINGE_JUMP_RATE,
    FRINGE_JUMP_MAGNITUDE_RAD,
    independent_phase_std_rad,
    sample_fringe_jumps,
    vibration_phase_std_rad,
)
from vpl.instruments.interferometry.phase import (
    ARGON_STATIC_POLARIZABILITY_VOLUME_M3,
    CHAMBER_DIAMETER_M,
    CHORD_SPACING_M,
    CO2_WAVELENGTH_M,
    HETERODYNE_OFFSET_HZ,
    N_CHORDS,
    PHASE_RESOLUTION_RAD,
    critical_density_per_m3,
    detection_floor_n_e_per_m3,
    electron_phase_shift_rad,
    net_phase_shift_rad,
    neutral_phase_shift_rad,
    refractive_index,
    refractive_index_linearized,
)

__all__ = [
    "ARGON_STATIC_POLARIZABILITY_VOLUME_M3",
    "CHAMBER_DIAMETER_M",
    "CHORD_SPACING_M",
    "CO2_WAVELENGTH_M",
    "DEFAULT_FRINGE_JUMP_RATE",
    "FRINGE_JUMP_MAGNITUDE_RAD",
    "HETERODYNE_OFFSET_HZ",
    "INTERFEROMETRY_INSTRUMENT_ID",
    "N_CHORDS",
    "PHASE_RESOLUTION_RAD",
    "PHASE_UNITS",
    "InterferometryInstrument",
    "critical_density_per_m3",
    "detection_floor_n_e_per_m3",
    "electron_phase_shift_rad",
    "independent_phase_std_rad",
    "net_phase_shift_rad",
    "neutral_phase_shift_rad",
    "refractive_index",
    "refractive_index_linearized",
    "sample_fringe_jumps",
    "vibration_phase_std_rad",
]
