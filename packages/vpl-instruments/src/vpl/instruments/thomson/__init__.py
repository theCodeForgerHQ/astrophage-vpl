"""``ThomsonInstrument`` — WBS 2.7, doc 02 §7, doc 04 §4.

## Why this channel exists

The project's honest end-to-end error currently sits at 36.5 %, and the diagnosed cause
(doc 05 §6, doc 06 §4) is a degeneracy: a single channel cannot separate plasma density
from electron temperature, because they enter the physics multiplied together — OES's
line-ratio excitation rates and interferometry's line-integrated refractive index both
depend on some product or convolution of ``n_e`` and ``T_e``, not on either alone.

Thomson scattering breaks that degeneracy from an entirely different physical mechanism.
Laser light scattered incoherently off free electrons (doc 04 §4.1) carries two
independent pieces of information in one measurement:

* the **integrated** scattered intensity is proportional to ``n_e`` directly
  (:mod:`.photons`, doc 02 §7.1's photon budget), calibrated absolutely against a known
  Rayleigh-scattering standard (doc 02 §7.3);
* the **width** of the scattered spectrum gives ``T_e`` directly, from the thermal Doppler
  broadening of the scattering electrons (:mod:`.spectrum`, doc 04 §4.2) — a mechanism
  that has nothing to do with excitation cross-sections or radiation trapping.

That is a genuinely different degeneracy structure from OES's, which is the point: doc 02
§7.1 consequence 3 calls Thomson "the anchor" for exactly this reason, and doc 02 §11
gives Thomson the tightest calibration chain of any channel (7 % combined Rayleigh
uncertainty, doc 06 §5) because so much rides on it.

## What it costs

Doc 02 §7.1's photon budget is not a footnote — it is the finding that reshapes the whole
channel. At the reference operating point (RP-1, ``n_e`` = 1e17 m^-3) a single laser shot
yields **0.16 expected photoelectrons across the entire 20-channel spectrum**. This is a
single-photoelectron counting experiment, not a signal-averaging one (doc 05 §3.1:
"Poisson for Thomson"), and reaching a useful 3 % measurement takes ~700 s of accumulation
at the TS-L1 10 Hz repetition rate. Phase-resolved operation multiplies that by the number
of RF phase bins doc 02 §10.3 defines (16), pushing a single phase-resolved point to
~3.1 hours. **Thomson cannot follow transients**, and regime G of doc 02 §3.3 ("transient
... Thomson blind by construction") exists specifically to exercise that limitation in the
benchmark suite rather than let it go untested.

## What is modelled, and what is deliberately not

* :mod:`.spectrum` — the incoherent-regime check (Salpeter 1960; Evans & Katzenstein
  1969), the doc 04 §4.2 Gaussian, and a relativistic correction in the spirit of Selden
  (1980) (see that module's docstring for the honest caveat on how closely it tracks
  Selden's specific published closed form).
* :mod:`.photons` — the doc 02 §7.1 photon budget, the accumulation/blindness
  arithmetic of doc 02 §7.1 consequence 2, the doc 02 §4.3 stray-light rejection stack
  and residual pedestal, and the doc 06 §5 Rayleigh calibration uncertainty chain.
* :mod:`.instrument` — :class:`~vpl.instruments.thomson.instrument.ThomsonInstrument`,
  the doc 08 §4 :class:`~vpl.core.protocols.Instrument` contract over the above.
* **Not modelled**: the detector electronics chain (MCP gain, phosphor, CCD, ADC — doc 02
  §9, doc 04 §7.1's ``F4``) belongs to ``vpl-detectors`` by the doc 04 §1 layering rule
  ("``F4`` never sees the plasma"), exactly as
  :class:`~vpl.instruments.oes.instrument.OesInstrument`'s module docstring notes for the
  same reason; only photon shot noise (the ``F2``/``F3`` boundary) is applied here. Ray
  tracing of the collection optics (doc 04 §6) is stood in for by the etendue-style
  radiometry of :mod:`.photons`, the same simplification
  :class:`~vpl.instruments.oes.instrument.OesInstrument` makes for its own slit optics.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
