"""A simulated single Langmuir probe — WBS 2.12, doc 11 §9 item 7, WBS 5.4.

Doc 11 §9 item 7 names the deliverable this module exists for: "comparative figure vs
simulated probe / RFEA — converts 'better than probes' into a measurement". Doc 00 §5.2
criterion E1 requires it to be a plugin implementing the same
:class:`~vpl.core.protocols.instrument.Instrument` contract every other channel does, with
no change to the core — see :class:`LangmuirProbe` below, and
:mod:`vpl.instruments.oes.instrument` / :mod:`vpl.instruments.lif.instrument` for the
convention it follows: one shared physics path, noise and calibration as switchable stages.

**This module is built to measure, not to win.** The project's own honest T2 error is
36.5 % — inside the range real probes and RFEAs achieve — so nothing here is tuned to make
the probe look worse than it is. Two systematic errors are modelled because they are the
textbook reasons single-probe analysis is uncertain, and both are left in even though they
sometimes make the probe's numbers *better* in some regime:

1. **A non-Maxwellian EEDF biases the log-slope ``T_e``.** The classic single-probe
   analysis (Langmuir; Mott-Smith & Langmuir 1926; Druyvesteyn 1930) assumes
   ``I_e(V) ~ exp((V - V_p)/T_e)``, which is exact only for a Maxwellian retarding
   population. Doc 03 §3.2 states that the sheath EEDF is "typically bi-Maxwellian or
   Druyvesteyn"; :func:`electron_current_a` computes the *exact* retarding current for
   whichever EEDF shape :attr:`~vpl.core.state.PlasmaParams.kappa` selects (the same
   Maxwellian-through-Druyvesteyn family :func:`vpl.physics.eedf.analytic.generalised_eedf`
   defines), and :func:`estimate_from_iv_curve` always fits the *linear* exponential law a
   real experimentalist would, regardless of the true shape. The mismatch is the bias.
2. **Sheath expansion inflates the ion-saturation current.** A cylindrical probe's
   collecting sheath grows with the retarding bias (Mott-Smith & Langmuir 1926;
   Lieberman & Lichtenberg §6.3), so ``I_sat`` measured deep in the ion-saturation region
   is larger than ``e n_s c_s`` at the *bare* probe area. :func:`estimate_from_iv_curve`
   divides by the bare area, as a real analysis that has not corrected for sheath
   expansion does, and the resulting ``n_e`` is biased high. This is Chen's and Merlino's
   textbook explanation for why a real cylindrical probe's I-V curve does not show a flat
   ion-saturation plateau.

## What is deliberately not modelled

- **Orbital-motion-limited (OML) collection.** The sheath-expansion model here is the
  thin-sheath (`s << r_p` is not assumed, but no orbital angular-momentum barrier is
  either) planar-front growth of Mott-Smith & Langmuir (1926) and Lieberman &
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
  retarding-current / second-derivative relation this module's electron branch specialises.
- R. L. Merlino, "Understanding Langmuir probe current-voltage characteristics", Am. J.
  Phys. 75 (2007) 1078 — the modern derivation this module's electron and ion branches
  follow most closely.
- V. A. Godyak and V. I. Demidov, "Probe measurements of electron-energy distribution
  functions: over thirty years of history", Plasma Sources Sci. Technol. 20 (2011) 062001 —
  review of the EEPF/second-derivative method and its limitations.
- M. A. Lieberman and A. J. Lichtenberg, *Principles of Plasma Discharges and Materials
  Processing*, 2nd ed., Wiley (2005), §6.3 — the Child-Langmuir sheath-thickness form this
  module reuses for the probe's own micro-sheath.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
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
from vpl.core.units import Q_, magnitude_in
from vpl.physics.analytic.sheath import DEFAULT_EDGE_TO_CENTRE_RATIO
from vpl.physics.eedf.analytic import MAXWELLIAN_KAPPA, generalised_eedf

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

type FloatArray = NDArray[np.float64]

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))
_EPS0: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

#: The doc 08 §7 artifact group this channel would write to.
LANGMUIR_INSTRUMENT_ID: Final[InstrumentId] = "langmuir_probe"

#: Units of what :meth:`LangmuirProbe.forward` and :meth:`observe` return: a collected
#: current at every swept bias, the raw quantity a real probe circuit reports.
CURRENT_UNITS: Final[str] = "A"

# ── the electron-retardation energy grid ────────────────────────────────────────────
#
# generalised_eedf underflows to exactly zero far into the tail (see its module
# docstring), so a generous span costs accuracy nowhere; it only has to reach past the
# largest retarding energy the sweep can present.

#: Grid points spanning ``[0, span * T_e]`` for the retarding-current integral.
_ENERGY_GRID_POINTS: Final[int] = 4000

#: How many electron temperatures the energy grid spans. The widest sweep this module
#: defaults to is 90 V below the plasma potential at T_e as low as ~2 eV; 30 T_e covers
#: that with a wide margin before the EEDF has underflowed.
_ENERGY_GRID_SPAN_IN_TE: Final[float] = 30.0

# ── default geometry — a thin tungsten wire probe, the common laboratory choice ────
#
# Chosen so a cylindrical probe's sheath-expansion systematic (the effect this module
# exists to reproduce) is visible: Mott-Smith & Langmuir (1926) derive it specifically
# for a thin cylinder, where the sheath thickness is not negligible next to the probe
# radius.

#: Probe wire radius — a 0.25 mm diameter tip, typical of a laboratory W probe.
_DEFAULT_PROBE_RADIUS_M: Final[float] = 1.25e-4

#: Exposed collecting length.
_DEFAULT_PROBE_LENGTH_M: Final[float] = 5.0e-3

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

# ── the naive analysis's fit windows ────────────────────────────────────────────────

#: Fraction of the (sorted) sweep used to estimate the ion-saturation current — the most
#: negative points, where the electron branch has underflowed.
_ION_SAT_FRACTION_OF_SWEEP: Final[float] = 0.1

#: Lower and upper bounds, as a fraction of the recovered electron-saturation current, of
#: the window the log-slope fit uses. Excludes the ion-saturation-dominated points below
#: and the already-saturated points above, the same practical window a real analysis uses.
#: The lower bound is not arbitrary: below ~10-15 % of the electron-saturation current the
#: constant-ion-current subtraction of :func:`estimate_from_iv_curve` step 2 is no longer a
#: good approximation here (the sheath-expanded ion current at the bottom of the sweep is
#: itself a comparable fraction of the electron-saturation current at this geometry), and
#: fitting through that region biases even the Maxwellian recovery. 0.2-0.4 is clear of it.
_FIT_WINDOW_LOWER_FRACTION: Final[float] = 0.2
_FIT_WINDOW_UPPER_FRACTION: Final[float] = 0.4

#: Below this bulk density, the probe's own sheath is no longer thin compared with its
#: radius (`lambda_D ~ r_p`) and the sheath-expansion model above is not defensible —
#: doc 01 IF-6's detection-floor gate, applied here to keep the channel out of the
#: likelihood where its own forward model is not trustworthy.
_DETECTION_FLOOR_N_0: Final[float] = 1.0e14


def bohm_speed_m_per_s(*, electron_temperature_ev: float, ion_mass_kg: float) -> float:
    """Cold-ion Bohm speed ``c_s = sqrt(e T_e / m_i)`` — doc 03 §2.1, Merlino (2007) eq. 5.

    A local, field-valued re-statement of
    :func:`vpl.physics.analytic.sheath.bohm_speed`'s ``gamma_i = 0`` default: that
    function takes a :class:`~vpl.core.state.PlasmaParams` (the *bulk* control
    parameters), while a probe reading is a function of the *local* ``T_e`` field at
    wherever the tip sits, which is not in general the bulk value. Re-deriving the
    one-line formula from local floats is simpler and less coupled than reconstructing a
    synthetic ``PlasmaParams`` at every grid point just to call the bulk function.
    """
    if not electron_temperature_ev > 0.0:
        raise ValueError(f"electron temperature must be positive, got {electron_temperature_ev} eV")
    if not ion_mass_kg > 0.0:
        raise ValueError(f"ion mass must be positive, got {ion_mass_kg} kg")
    return math.sqrt(_E_C * electron_temperature_ev / ion_mass_kg)


def _debye_length_m(*, electron_density_m3: float, electron_temperature_ev: float) -> float:
    """``lambda_D = sqrt(eps_0 k T_e / (n_e e^2))``, at local field values."""
    t_e_j = electron_temperature_ev * _E_C
    return math.sqrt(_EPS0 * t_e_j / (electron_density_m3 * _E_C**2))


def _reverse_cumulative_tail(y: FloatArray, x: FloatArray) -> FloatArray:
    """``tail[i] = integral_{x_i}^{x_max} y dx``, by the trapezoid rule.

    Built as "total minus the running sum from the start" rather than by reversing and
    re-cumulating, so the same helper serves both the electron-retardation integral here
    and (with a different y) the RFEA's transmitted-flux integral without either caller
    having to reason about array order.
    """
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    forward = np.concatenate(([0.0], np.cumsum(increments)))
    return np.asarray(forward[-1] - forward)


@dataclass(frozen=True, slots=True)
class ProbeGeometry:
    """A cylindrical probe tip — Mott-Smith & Langmuir (1926).

    Attributes:
        radius_m: Wire radius.
        length_m: Exposed collecting length.
    """

    radius_m: float
    length_m: float

    def __post_init__(self) -> None:
        if not self.radius_m > 0.0:
            raise ValueError(f"probe radius must be positive, got {self.radius_m} m")
        if not self.length_m > 0.0:
            raise ValueError(f"probe length must be positive, got {self.length_m} m")

    @property
    def area_m2(self) -> float:
        """Bare cylindrical side area, ``2 pi r L``. End-cap area is neglected — the
        standard thin-wire approximation, valid for ``L >> r`` and true here by two
        orders of magnitude at the default geometry."""
        return 2.0 * math.pi * self.radius_m * self.length_m

    def __repr__(self) -> str:
        return f"ProbeGeometry(r={self.radius_m:.3g} m, L={self.length_m:.3g} m)"


def sheath_expansion_radius_m(
    voltage_v: ArrayLike,
    *,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    probe_radius_m: float,
) -> FloatArray:
    """The probe's own micro-sheath radius — Mott-Smith & Langmuir (1926) planar-front
    growth, using the Child-Langmuir thickness of Lieberman & Lichtenberg §6.3.

    Reuses the functional form of
    :func:`vpl.physics.analytic.sheath.child_langmuir_thickness` (doc 03 §2.3), not the
    function itself: that one takes a bulk :class:`~vpl.core.state.PlasmaParams` biased
    at the *wall* electrode's potential, and this is the same closed-form sheath law
    applied to the probe's own local Debye length and its own swept bias instead. Zero at
    the plasma potential — no retarding drop, no space-charge sheath to hold one — which
    is the same continuous limit :func:`~vpl.physics.analytic.sheath.child_langmuir_thickness`
    has at zero bias.

    Args:
        voltage_v: Swept probe bias.
        plasma_potential_v: Local plasma potential the bias is measured against.
        electron_density_m3: Local bulk electron density (sets the local Debye length).
        electron_temperature_ev: Local electron temperature.
        probe_radius_m: Bare probe radius.

    Returns:
        The effective collecting radius at each bias: ``r_p`` where ``V >= V_p``,
        growing as the bias is driven more negative.
    """
    voltages = np.atleast_1d(np.asarray(voltage_v, dtype=np.float64))
    retarding_v = np.clip(plasma_potential_v - voltages, 0.0, None)
    lambda_d = _debye_length_m(
        electron_density_m3=electron_density_m3, electron_temperature_ev=electron_temperature_ev
    )
    normalised_bias = 2.0 * retarding_v / electron_temperature_ev
    thickness = (math.sqrt(2.0) / 3.0) * lambda_d * np.power(normalised_bias, 0.75)
    return np.asarray(probe_radius_m + thickness)


def electron_current_a(
    voltage_v: ArrayLike,
    *,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    probe_area_m2: float,
    kappa: float = MAXWELLIAN_KAPPA,
) -> FloatArray:
    """The exact planar-retardation electron current, for any EEDF shape doc 05 §2.1's
    ``kappa`` selects — Druyvesteyn (1930); Merlino (2007) eq. 2.9; Godyak & Demidov (2011).

    For an isotropic EEDF ``f0`` (:func:`vpl.physics.eedf.analytic.generalised_eedf`'s
    convention, ``integral f0 sqrt(eps) deps = 1``), the current a planar collector draws
    at retarding energy ``eps_r = e(V_p - V)`` is

        I_e(V) = (e A / (2 sqrt(2 m_e))) integral_{eps_r}^inf f0(eps) (eps - eps_r) deps

    (energies integrated in SI, ``f0`` evaluated on an eV grid and rescaled once at
    construction). At ``kappa = 1`` (Maxwellian) this integral is exactly
    ``I_e0 exp((V - V_p)/T_e)`` — the textbook exponential law — and at ``V = V_p`` it
    reduces to the random thermal flux ``(1/4) n_e <v_e> A``; both are checked directly
    against this function in ``test_probe_langmuir.py``. For any other ``kappa`` the
    integral is *not* a pure exponential in ``V``, which is exactly why a log-slope fit
    to it is biased — the effect :func:`estimate_from_iv_curve` is built to expose.

    Args:
        voltage_v: Swept probe bias.
        plasma_potential_v: Local plasma potential.
        electron_density_m3: Local electron density.
        electron_temperature_ev: Local electron temperature — the EEDF's mean energy is
            ``1.5`` times this, for every ``kappa`` (doc 05 §2.1's convention).
        probe_area_m2: Bare collecting area. Never the sheath-expanded one: the
            literature attributes sheath expansion to ion, not electron, collection.
        kappa: The Maxwellian-through-Druyvesteyn shape parameter of
            :func:`vpl.physics.eedf.analytic.generalised_eedf` — ``1`` Maxwellian,
            ``2`` Druyvesteyn.

    Returns:
        The electron current at every swept bias, always positive.
    """
    voltages = np.atleast_1d(np.asarray(voltage_v, dtype=np.float64))
    energy_grid_ev = np.linspace(
        0.0, _ENERGY_GRID_SPAN_IN_TE * electron_temperature_ev, _ENERGY_GRID_POINTS
    )
    shape = generalised_eedf(
        energy_grid_ev, mean_energy_ev=1.5 * electron_temperature_ev, kappa=kappa
    )
    # Rescale once from the eV-normalised f0 (integral f0 sqrt(eps_eV) d(eps_eV) = 1) to
    # the SI-consistent g_J with integral g_J(eps_J) sqrt(eps_J) d(eps_J) = 1, so the
    # tail integrals below are exact SI moments and not eV-vs-Joule algebra repeated at
    # every call. See the module docstring's Merlino (2007) / Druyvesteyn (1930) citation.
    g_j = shape / _E_C**1.5
    energy_grid_j = energy_grid_ev * _E_C

    tail_g = _reverse_cumulative_tail(g_j, energy_grid_j)
    tail_eg = _reverse_cumulative_tail(energy_grid_j * g_j, energy_grid_j)

    retarding_j = np.clip(plasma_potential_v - voltages, 0.0, None) * _E_C
    tail_g_at = np.interp(retarding_j, energy_grid_j, tail_g)
    tail_eg_at = np.interp(retarding_j, energy_grid_j, tail_eg)

    flux = (
        electron_density_m3 * (tail_eg_at - retarding_j * tail_g_at) / (2.0 * math.sqrt(2.0 * _M_E))
    )
    return np.asarray(_E_C * probe_area_m2 * flux)


def ion_saturation_current_a(
    voltage_v: ArrayLike,
    *,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    ion_mass_kg: float,
    geometry: ProbeGeometry,
    edge_to_centre_ratio: float = DEFAULT_EDGE_TO_CENTRE_RATIO,
) -> FloatArray:
    """The Bohm-flux ion current on the sheath-expanded area — Merlino (2007) eq. 2.3;
    Mott-Smith & Langmuir (1926) for the area growth.

    ``I_i(V) = e n_s c_s A_eff(V)``, ``n_s = h_l n_e`` the sheath-edge density (the same
    presheath coefficient doc 03 §2.1 uses at the wall, applied here to the probe's own
    local presheath — :data:`~vpl.physics.analytic.sheath.DEFAULT_EDGE_TO_CENTRE_RATIO`),
    and ``A_eff`` the cylindrical side area at
    :func:`sheath_expansion_radius_m`'s effective radius. This is the term that grows
    with bias and is the reason a real probe's ion branch is not flat.

    Args:
        voltage_v: Swept probe bias.
        plasma_potential_v: Local plasma potential.
        electron_density_m3: Local bulk electron density.
        electron_temperature_ev: Local electron temperature (sets ``c_s`` and the local
            Debye length).
        ion_mass_kg: Ion mass.
        geometry: The bare probe geometry.
        edge_to_centre_ratio: The presheath density ratio ``h_l``.

    Returns:
        The ion current magnitude at every swept bias, always positive.
    """
    r_eff = sheath_expansion_radius_m(
        voltage_v,
        plasma_potential_v=plasma_potential_v,
        electron_density_m3=electron_density_m3,
        electron_temperature_ev=electron_temperature_ev,
        probe_radius_m=geometry.radius_m,
    )
    area_eff = 2.0 * math.pi * r_eff * geometry.length_m
    c_s = bohm_speed_m_per_s(
        electron_temperature_ev=electron_temperature_ev, ion_mass_kg=ion_mass_kg
    )
    n_s = edge_to_centre_ratio * electron_density_m3
    return np.asarray(_E_C * n_s * c_s * area_eff)


def probe_current_a(
    voltage_v: ArrayLike,
    *,
    plasma_potential_v: float,
    electron_density_m3: float,
    electron_temperature_ev: float,
    ion_mass_kg: float,
    geometry: ProbeGeometry,
    kappa: float = MAXWELLIAN_KAPPA,
    edge_to_centre_ratio: float = DEFAULT_EDGE_TO_CENTRE_RATIO,
) -> FloatArray:
    """The net collected current, electron-positive convention — Merlino (2007) eq. 2.1:
    ``I(V) = I_e(V) - I_i(V)``. This is the quantity a real transimpedance amplifier
    reports, and the one :class:`LangmuirProbe` returns from ``forward``/``observe``."""
    i_e = electron_current_a(
        voltage_v,
        plasma_potential_v=plasma_potential_v,
        electron_density_m3=electron_density_m3,
        electron_temperature_ev=electron_temperature_ev,
        probe_area_m2=geometry.area_m2,
        kappa=kappa,
    )
    i_i = ion_saturation_current_a(
        voltage_v,
        plasma_potential_v=plasma_potential_v,
        electron_density_m3=electron_density_m3,
        electron_temperature_ev=electron_temperature_ev,
        ion_mass_kg=ion_mass_kg,
        geometry=geometry,
        edge_to_centre_ratio=edge_to_centre_ratio,
    )
    return np.asarray(i_e - i_i)


@dataclass(frozen=True, slots=True)
class LangmuirEstimate:
    """What :func:`estimate_from_iv_curve` extracts — the naive single-probe result.

    Attributes:
        electron_temperature_ev: From the log-slope of the exponential region.
        electron_density_m3: From ``I_sat`` at the *bare* probe area.
        ion_saturation_current_a: The ``I_sat`` the estimate was derived from.
    """

    electron_temperature_ev: float
    electron_density_m3: float
    ion_saturation_current_a: float

    def __repr__(self) -> str:
        return (
            f"LangmuirEstimate(T_e={self.electron_temperature_ev:.4g} eV, "
            f"n_e={self.electron_density_m3:.4g} m^-3, "
            f"I_sat={self.ion_saturation_current_a:.4g} A)"
        )


def estimate_from_iv_curve(
    voltage_v: ArrayLike,
    current_a: ArrayLike,
    *,
    ion_mass_kg: float,
    probe_area_m2: float,
    edge_to_centre_ratio: float = DEFAULT_EDGE_TO_CENTRE_RATIO,
) -> LangmuirEstimate:
    """The naive analysis a real experimentalist runs on a measured I-V curve.

    Not tied to :class:`LangmuirProbe` or to any simulated state: this operates on raw
    voltage/current arrays exactly as they would come off real hardware, which is the
    point doc 00 E2 makes about the ``Instrument`` contract generally — the comparison
    this module exists for is against what an experimentalist actually does with the
    data, not against the instrument's raw signal.

    The procedure (Langmuir 1923; Merlino 2007 §II):

    1. **Ion saturation.** Average the most negative
       :data:`_ION_SAT_FRACTION_OF_SWEEP` of the (sorted) sweep, where the electron
       branch has underflowed to nothing. This reads off whatever the true ion current
       is *there* — sheath-expanded, if the generative model included it — with no
       correction applied, exactly as a real analysis that has not modelled sheath
       expansion would.
    2. **Subtract a constant ion current** from the whole curve to isolate the electron
       branch. The standard simplified subtraction (Merlino 2007 §II.B): real practice
       also linearly extrapolates the ion branch, and the constant-subtraction here is
       the more conservative (larger-residual) of the two only in the saturated region,
       where doc 00 C4 would rather the offset were visible than hidden.
    3. **Fit ``ln(I_e)`` vs ``V``** by ordinary least squares over the window
       :data:`_FIT_WINDOW_LOWER_FRACTION` to :data:`_FIT_WINDOW_UPPER_FRACTION` of the
       recovered electron-saturation current — below is dominated by the ion-subtraction
       residual, above is already saturated. ``T_e = 1 / slope``.
    4. **``n_e`` from ``I_sat``,** using the *bare* probe area and the fitted ``T_e``
       (through ``c_s``). No sheath-expansion correction — that correction requires
       knowing the true sheath law, which a real analysis of a real probe does not have.

    Args:
        voltage_v: Swept probe bias, any order.
        current_a: Collected current at each voltage, electron-positive convention.
        ion_mass_kg: Ion mass, for the Bohm speed in the density conversion.
        probe_area_m2: The bare (not sheath-expanded) collecting area.
        edge_to_centre_ratio: The presheath density ratio ``h_l`` assumed by the analysis.

    Returns:
        The recovered :class:`LangmuirEstimate`.

    Raises:
        ValueError: If the arrays disagree in shape, if the ion-saturation region does
            not carry a net negative current, or if too few points fall in the fit
            window to determine a slope.
    """
    voltages = np.asarray(voltage_v, dtype=np.float64)
    currents = np.asarray(current_a, dtype=np.float64)
    if voltages.shape != currents.shape:
        raise ValueError(
            f"voltage and current arrays must have the same shape, got {voltages.shape} "
            f"and {currents.shape}"
        )

    order = np.argsort(voltages)
    voltages = voltages[order]
    currents = currents[order]

    n_ion_points = max(2, round(_ION_SAT_FRACTION_OF_SWEEP * voltages.size))
    i_sat = -float(np.mean(currents[:n_ion_points]))
    if not i_sat > 0.0:
        raise ValueError(
            "the most negative part of this sweep does not carry a net negative "
            "current, so no ion-saturation current can be read off it; widen the sweep "
            "toward more negative bias"
        )

    electron_current = currents + i_sat
    positive = electron_current > 0.0
    if not np.any(positive):
        raise ValueError(
            "no positive electron current recovered after subtracting I_sat; widen the "
            "sweep past the plasma potential"
        )
    peak = float(np.max(electron_current[positive]))

    window = (
        positive
        & (electron_current >= _FIT_WINDOW_LOWER_FRACTION * peak)
        & (electron_current <= _FIT_WINDOW_UPPER_FRACTION * peak)
    )
    if np.count_nonzero(window) < 2:
        raise ValueError(
            "too few points fall inside the exponential fit window "
            f"[{_FIT_WINDOW_LOWER_FRACTION}, {_FIT_WINDOW_UPPER_FRACTION}] of the "
            "recovered electron-saturation current; the sweep does not resolve the "
            "exponential region"
        )

    slope, _intercept = np.polyfit(voltages[window], np.log(electron_current[window]), 1)
    if not slope > 0.0:
        raise ValueError(
            f"the fitted log-slope is not positive ({slope!r}); a Langmuir probe's "
            "electron current must rise with bias in the exponential region"
        )
    t_e_est = 1.0 / float(slope)

    c_s = bohm_speed_m_per_s(electron_temperature_ev=t_e_est, ion_mass_kg=ion_mass_kg)
    n_e_est = i_sat / (_E_C * edge_to_centre_ratio * c_s * probe_area_m2)

    return LangmuirEstimate(
        electron_temperature_ev=t_e_est, electron_density_m3=n_e_est, ion_saturation_current_a=i_sat
    )


# ── the doc 08 §4 Instrument contract ────────────────────────────────────────────────


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
            else ProbeGeometry(radius_m=_DEFAULT_PROBE_RADIUS_M, length_m=_DEFAULT_PROBE_LENGTH_M)
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
