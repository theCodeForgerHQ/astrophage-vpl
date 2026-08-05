"""Laser-induced fluorescence — ``F2`` for the ion channel, doc 04 §3.

The only diagnostic in the set that sees ions directly. A tunable laser is scanned across
the Ar II 668.614 nm absorption; ions Doppler-shifted into resonance fluoresce at
442.72 nm; the scan is a map of ``f_i(v_z)`` (doc 04 §3.2). Everything else in the
framework infers the ion population; this measures it.

Start at :class:`~vpl.instruments.lif.instrument.LifInstrument`, which is the doc 08 §4
contract. Underneath it:

- :mod:`~vpl.instruments.lif.transition` — the doc 02 §5.3 probe pair and ``I_sat``.
- :mod:`~vpl.instruments.lif.laser` — the beam, the 15 deg grazing geometry, and
  :class:`~vpl.instruments.lif.laser.TuningRange`, which is where doc 01 §5.1's limitation
  lives.
- :mod:`~vpl.instruments.lif.zeeman` — the pi/sigma comb of doc 04 §3.3.
- :mod:`~vpl.instruments.lif.rates` — doc 04 §3.1's rate equation and its saturated steady
  state.
- :mod:`~vpl.instruments.lif.scan` — the Doppler mapping and the velocity integral.

## What this package will and will not tell you

**It constrains the shape and the relative amplitude of the IVDF, and never an absolute
ion density.** Doc 02 §11 records that LIF is not absolutely calibrated, doc 05 §2.2 makes
the amplitude an inferred nuisance parameter, and the forward model returns an upper-state
density rather than a radiance for that reason — see :mod:`~vpl.instruments.lif.scan`.

**It measures a metastable subpopulation, not the ion population.** Doc 02 §5.3 calls this
"the single largest systematic in LIF". The metastable fraction enters as one registered
multiplier and is a doc 05 §2.2 nuisance parameter; its *velocity dependence* is not
modelled here, and doc 02 §5.3 bounds that as a shape systematic in the doc 06 §4 budget.

**It cannot see fast ions.** Doc 01 §5.1 and doc 14 RS-03: a 20 GHz mode-hop-free tuning
range is a finite ion speed. With the doc 02 §4.2 grazing geometry the ceiling is 25.8 km/s
of ``v_z``, 138 eV for argon — above doc 01 §5.1's own arithmetic, which omits the
projection factor, and still far below a 250 V sheath. The limit is represented as a hard
refusal in :meth:`~vpl.instruments.lif.laser.TuningRange.require_reachable`, as a
population fraction in :func:`~vpl.instruments.lif.scan.tuning_coverage`, and as a gate in
:meth:`~vpl.instruments.lif.instrument.LifInstrument.is_informative`.

**It is not a complete synthetic measurement.** ``F3`` (optical transport, doc 04 §6) and
``F4`` (detection, doc 04 §7) are separate packages, and
:meth:`~vpl.instruments.lif.instrument.LifInstrument.observe` applies only the doc 04 §7.3
imperfect calibration. Treating its output as a detector reading would omit all eighteen
noise sources of doc 04 §7.2.

## Three places where the model and the Baseline documents disagree

All three are carried in code with the document's value taken where it is load-bearing,
and all three are pinned by tests so they cannot be forgotten:

1. **Doc 04 §3.3's "~734 MHz FWHM" Doppler width is a 1/e half-width.** The FWHM is
   ``sqrt(4 ln 2)`` = 1.665 times larger, 1224 MHz. See
   :func:`~vpl.instruments.lif.scan.thermal_doppler_1e_halfwidth`.
2. **Doc 04 §3.3's "~70 MHz" Zeeman splitting is the scale ``mu_B B / h``,** not the width
   of the pattern; the extreme component sits at 110 MHz. The document's conclusion — 10 %
   of the Doppler width — is reproduced by the RMS spread, 62.5 MHz. See
   :mod:`~vpl.instruments.lif.zeeman`.
3. **Doc 02 §5.3 pairs the term symbols of one Ar II scheme with the wavelengths of
   another.** The wavelengths are taken as specified (they fix real hardware); the term
   symbols are registry entries. See ``lif.yaml``.
"""

from vpl.instruments.lif.instrument import (
    INSTRUMENT_ID,
    OBSERVABLE_UNITS,
    LifInstrument,
    RunConditions,
)
from vpl.instruments.lif.laser import Laser, TuningRange
from vpl.instruments.lif.rates import (
    lorentzian_response,
    power_broadened_fwhm,
    pump_rate_from_saturation,
    saturation_ceiling,
    steady_state_excited_fraction,
    upper_state_rate,
)
from vpl.instruments.lif.scan import (
    MIN_SAMPLES_PER_HOMOGENEOUS_WIDTH,
    TRUNCATION_TOLERANCE,
    DetuningScan,
    TuningCoverage,
    doppler_detuning_hz,
    fluorescence_response,
    resonant_velocity_m_per_s,
    thermal_doppler_1e_halfwidth,
    tuning_coverage,
)
from vpl.instruments.lif.transition import Level, ProbeTransition
from vpl.instruments.lif.zeeman import (
    PumpPolarisation,
    ZeemanComponent,
    lande_g_factor,
    rms_spread,
    unsplit_pattern,
    zeeman_pattern,
)

__all__ = [
    "INSTRUMENT_ID",
    "MIN_SAMPLES_PER_HOMOGENEOUS_WIDTH",
    "OBSERVABLE_UNITS",
    "TRUNCATION_TOLERANCE",
    "DetuningScan",
    "Laser",
    "Level",
    "LifInstrument",
    "ProbeTransition",
    "PumpPolarisation",
    "RunConditions",
    "TuningCoverage",
    "TuningRange",
    "ZeemanComponent",
    "doppler_detuning_hz",
    "fluorescence_response",
    "lande_g_factor",
    "lorentzian_response",
    "power_broadened_fwhm",
    "pump_rate_from_saturation",
    "resonant_velocity_m_per_s",
    "rms_spread",
    "saturation_ceiling",
    "steady_state_excited_fraction",
    "thermal_doppler_1e_halfwidth",
    "tuning_coverage",
    "unsplit_pattern",
    "upper_state_rate",
    "zeeman_pattern",
]
