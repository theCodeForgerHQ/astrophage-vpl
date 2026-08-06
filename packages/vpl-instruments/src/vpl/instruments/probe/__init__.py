"""Simulated reference instruments — WBS 2.12, doc 11 §9 item 7, WBS 5.4.

Doc 11 §9 item 7 lists "comparative figure vs simulated probe / RFEA" among what turns
"better than probes" from a slogan into a measurement, and WBS 5.4 is the comparative
study that figure belongs to. Neither can exist without something to compare against, so
this package is that something: a simulated single Langmuir probe
(:mod:`~vpl.instruments.probe.langmuir`) and a simulated retarding field energy analyser
(:mod:`~vpl.instruments.probe.rfea`), each a plugin over the same
:class:`~vpl.core.protocols.instrument.Instrument` contract every other channel in
:mod:`vpl.instruments` implements (doc 00 §5.2 E1).

**Built to measure, not to win.** The project's own honest T2 error is 36.5 % — inside
the range real probes and RFEAs achieve — so the comparison this package exists for is
genuinely open, and nothing here is tuned to make either reference instrument look worse
than the literature says it is. Both modules model the textbook systematic errors that
make single-probe and RFEA analysis uncertain (a non-Maxwellian EEDF biasing the Langmuir
log-slope ``T_e``; sheath expansion inflating the Langmuir ``I_sat``; finite grid
transparency limiting the RFEA's energy resolution) rather than either an idealised or an
artificially degraded instrument. See each module's docstring for the citations and for
what is deliberately left out.

Neither instrument is part of the doc 01 diagnostic suite (OES, LIF, Thomson scattering,
interferometry); both exist solely for WBS 5.4's comparison and are not wired into the
doc 08 §6 manifest's default instrument list.
"""

from vpl.instruments.probe.langmuir import (
    CURRENT_UNITS as LANGMUIR_CURRENT_UNITS,
)
from vpl.instruments.probe.langmuir import (
    LANGMUIR_INSTRUMENT_ID,
    LangmuirEstimate,
    LangmuirProbe,
    ProbeGeometry,
    bohm_speed_m_per_s,
    electron_current_a,
    estimate_from_iv_curve,
    ion_saturation_current_a,
    probe_current_a,
    sheath_expansion_radius_m,
)
from vpl.instruments.probe.rfea import (
    CURRENT_UNITS as RFEA_CURRENT_UNITS,
)
from vpl.instruments.probe.rfea import (
    RFEA_INSTRUMENT_ID,
    GridGeometry,
    IedfEstimate,
    RfeaInstrument,
    collected_current_a,
    estimate_iedf,
    flux_iedf_ev,
)

__all__ = [
    "LANGMUIR_CURRENT_UNITS",
    "LANGMUIR_INSTRUMENT_ID",
    "RFEA_CURRENT_UNITS",
    "RFEA_INSTRUMENT_ID",
    "GridGeometry",
    "IedfEstimate",
    "LangmuirEstimate",
    "LangmuirProbe",
    "ProbeGeometry",
    "RfeaInstrument",
    "bohm_speed_m_per_s",
    "collected_current_a",
    "electron_current_a",
    "estimate_from_iv_curve",
    "estimate_iedf",
    "flux_iedf_ev",
    "ion_saturation_current_a",
    "probe_current_a",
    "sheath_expansion_radius_m",
]
