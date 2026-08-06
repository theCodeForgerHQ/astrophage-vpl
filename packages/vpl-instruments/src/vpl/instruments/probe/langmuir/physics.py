"""The Langmuir probe I-V physics — electron and ion branches.

Split out from :mod:`vpl.instruments.probe.langmuir` (the package) so that the physics
here, the naive analysis of :mod:`vpl.instruments.probe.langmuir.analysis`, and the
:class:`~vpl.core.protocols.Instrument` wrapper of
:mod:`vpl.instruments.probe.langmuir.instrument` each stay within the doc 08 §1 principle
7 file-size guideline. See the package's ``__init__.py`` for the physics citations, the
two systematic errors this module exists to reproduce, and what is deliberately not
modelled — none of that is repeated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.units import magnitude_in
from vpl.physics.analytic.sheath import DEFAULT_EDGE_TO_CENTRE_RATIO
from vpl.physics.eedf.analytic import MAXWELLIAN_KAPPA, generalised_eedf

__all__ = [
    "ProbeGeometry",
    "bohm_speed_m_per_s",
    "electron_current_a",
    "ion_saturation_current_a",
    "probe_current_a",
    "sheath_expansion_radius_m",
]

type FloatArray = NDArray[np.float64]

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))
_EPS0: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

# ── the electron-retardation energy grid ────────────────────────────────────────────
#
# generalised_eedf underflows to exactly zero far into the tail (see its module
# docstring), so a generous span costs accuracy nowhere; it only has to reach past the
# largest retarding energy the sweep can present.

#: Grid points spanning ``[0, span * T_e]`` for the retarding-current integral.
_ENERGY_GRID_POINTS: Final[int] = 4000

#: How many electron temperatures the energy grid spans. The widest sweep
#: :mod:`vpl.instruments.probe.langmuir.instrument` defaults to is 90 V below the plasma
#: potential at T_e as low as ~2 eV; 30 T_e covers that with a wide margin before the
#: EEDF has underflowed.
_ENERGY_GRID_SPAN_IN_TE: Final[float] = 30.0

# ── default geometry — a thin tungsten wire probe, the common laboratory choice ────
#
# Chosen so a cylindrical probe's sheath-expansion systematic (the effect this module
# exists to reproduce) is visible: Mott-Smith & Langmuir (1926) derive it specifically
# for a thin cylinder, where the sheath thickness is not negligible next to the probe
# radius. Re-exported by :mod:`vpl.instruments.probe.langmuir.instrument`, which is the
# only other consumer.

#: Probe wire radius — a 0.25 mm diameter tip, typical of a laboratory W probe.
DEFAULT_PROBE_RADIUS_M: Final[float] = 1.25e-4

#: Exposed collecting length.
DEFAULT_PROBE_LENGTH_M: Final[float] = 5.0e-3


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
    re-cumulating, so the same shape of helper serves both this module's
    electron-retardation integral and (its own copy, see
    :mod:`vpl.instruments.probe.rfea`) the RFEA's transmitted-flux integral without
    either caller reasoning about array order.
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
    to it is biased — the effect
    :func:`~vpl.instruments.probe.langmuir.analysis.estimate_from_iv_curve` is built to
    expose.

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
    reports, and the one
    :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe` returns from
    ``forward``/``observe``."""
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
