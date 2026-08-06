"""A simulated single Langmuir probe — WBS 2.12, doc 11 §9 item 7, WBS 5.4.

Doc 11 §9 item 7 names the deliverable this package exists for: "comparative figure vs
simulated probe / RFEA — converts 'better than probes' into a measurement". Doc 00 §5.2
criterion E1 requires it to be a plugin implementing the same
:class:`~vpl.core.protocols.instrument.Instrument` contract every other channel does, with
no change to the core — see :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe`,
and :mod:`vpl.instruments.oes.instrument` / :mod:`vpl.instruments.lif.instrument` for the
convention it follows: one shared physics path, noise and calibration as switchable stages.

**This package is built to measure, not to win.** The project's own honest T2 error is
36.5 % — inside the range real probes and RFEAs achieve — so nothing here is tuned to make
the probe look worse than it is. Two systematic errors are modelled because they are the
textbook reasons single-probe analysis is uncertain, and both are left in even though they
sometimes make the probe's numbers *better* in some regime:

1. **A non-Maxwellian EEDF biases the log-slope ``T_e``.** The classic single-probe
   analysis (Langmuir; Mott-Smith & Langmuir 1926; Druyvesteyn 1930) assumes
   ``I_e(V) ~ exp((V - V_p)/T_e)``, which is exact only for a Maxwellian retarding
   population. Doc 03 §3.2 states that the sheath EEDF is "typically bi-Maxwellian or
   Druyvesteyn";
   :func:`~vpl.instruments.probe.langmuir.physics.electron_current_a` computes the
   *exact* retarding current for whichever EEDF shape
   :attr:`~vpl.core.state.PlasmaParams.kappa` selects (the same
   Maxwellian-through-Druyvesteyn family :func:`vpl.physics.eedf.analytic.generalised_eedf`
   defines), and
   :func:`~vpl.instruments.probe.langmuir.analysis.estimate_from_iv_curve` always fits
   the *linear* exponential law a real experimentalist would, regardless of the true
   shape. The mismatch is the bias.
2. **Sheath expansion inflates the ion-saturation current.** A cylindrical probe's
   collecting sheath grows with the retarding bias (Mott-Smith & Langmuir 1926;
   Lieberman & Lichtenberg §6.3), so ``I_sat`` measured deep in the ion-saturation region
   is larger than ``e n_s c_s`` at the *bare* probe area. ``estimate_from_iv_curve``
   divides by the bare area, as a real analysis that has not corrected for sheath
   expansion does, and the resulting ``n_e`` is biased high. This is Chen's and Merlino's
   textbook explanation for why a real cylindrical probe's I-V curve does not show a flat
   ion-saturation plateau.

## Package layout

- :mod:`~vpl.instruments.probe.langmuir.physics` — the generative I-V model: electron and
  ion branches, the probe's own micro-sheath expansion.
- :mod:`~vpl.instruments.probe.langmuir.analysis` — the naive log-slope / ``I_sat``
  analysis a real experimentalist runs, blind to which EEDF or sheath model produced the
  curve.
- :mod:`~vpl.instruments.probe.langmuir.instrument` — :class:`LangmuirProbe`, the doc 08
  §4 ``Instrument`` contract wrapping the physics module.

## What is deliberately not modelled

- **Orbital-motion-limited (OML) collection.** The sheath-expansion model here is the
  thin-sheath planar-front growth of Mott-Smith & Langmuir (1926) and Lieberman &
  Lichtenberg §6.3, not the full OML ``I_i ~ (V_p - V)^{1/2}`` law. OML would change the
  ion branch's shape as well as its magnitude; the simplification here still produces the
  qualitatively correct and well-cited failure mode (inflated ``I_sat``) without a second
  free regime to validate against doc 00 C2.
- **RF and floating-potential ripple**, secondary electron emission from the probe tip,
  and contamination/sputtering of the collecting surface. All are real systematics in
  laboratory probe data; none change the two effects this comparison is built to isolate.
- **Electron-branch sheath expansion.** Only the ion branch's area grows here, matching
  the literature's usual attribution of the effect to ion collection.

## Citations

- I. Langmuir, "The pressure effect and other phenomena in gaseous discharges", J. Franklin
  Inst. 196 (1923) 751 — the single-probe I-V characteristic.
- H. M. Mott-Smith and I. Langmuir, "The theory of collectors in gaseous discharges",
  Phys. Rev. 28 (1926) 727 — sheath collection and its expansion with bias.
- M. J. Druyvesteyn, "Der Niedervoltbogen", Z. Phys. 64 (1930) 781 — the non-Maxwellian
  retarding-current / second-derivative relation the electron branch specialises.
- R. L. Merlino, "Understanding Langmuir probe current-voltage characteristics", Am. J.
  Phys. 75 (2007) 1078 — the modern derivation the electron and ion branches follow most
  closely.
- V. A. Godyak and V. I. Demidov, "Probe measurements of electron-energy distribution
  functions: over thirty years of history", Plasma Sources Sci. Technol. 20 (2011) 062001 —
  review of the EEPF/second-derivative method and its limitations.
- M. A. Lieberman and A. J. Lichtenberg, *Principles of Plasma Discharges and Materials
  Processing*, 2nd ed., Wiley (2005), §6.3 — the Child-Langmuir sheath-thickness form
  reused here for the probe's own micro-sheath.
"""

from vpl.instruments.probe.langmuir.analysis import LangmuirEstimate, estimate_from_iv_curve
from vpl.instruments.probe.langmuir.instrument import (
    CURRENT_UNITS,
    LANGMUIR_INSTRUMENT_ID,
    LangmuirProbe,
)
from vpl.instruments.probe.langmuir.physics import (
    ProbeGeometry,
    bohm_speed_m_per_s,
    electron_current_a,
    ion_saturation_current_a,
    probe_current_a,
    sheath_expansion_radius_m,
)

__all__ = [
    "CURRENT_UNITS",
    "LANGMUIR_INSTRUMENT_ID",
    "LangmuirEstimate",
    "LangmuirProbe",
    "ProbeGeometry",
    "bohm_speed_m_per_s",
    "electron_current_a",
    "estimate_from_iv_curve",
    "ion_saturation_current_a",
    "probe_current_a",
    "sheath_expansion_radius_m",
]
