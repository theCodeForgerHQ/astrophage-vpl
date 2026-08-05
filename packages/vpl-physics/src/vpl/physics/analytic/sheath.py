"""L0 — analytic sheath models. Doc 03 §2, implemented as written.

Nothing here is new. Every expression is quoted from doc 03 §2 or from the textbook it
cites, which is doc 00 C2 in practice: the contribution of this project is integration and
rigour, not discovery, so an L0 model that could not be traced to a printed equation would
be a defect rather than a feature.

## What this level is for

Doc 03 §1 gives L0 three jobs: verification anchor, MCMC proposal, and sanity bound. The
first is the load-bearing one. **Verification gate V-03** (doc 07) requires the fluid (L1)
and kinetic (L2) solvers to reproduce this module's ``s`` and ``J_i`` to within 5 % in the
collisionless high-bias limit, and doc 03 §2.3 calls its ``Gamma_E ~ 6.6 kW/m^2`` at RP-1
"the number every other model must reproduce in this limit". If this module is wrong, the
gate that is supposed to catch a wrong L2 passes anyway.

## Two places where the Baseline specification disagrees with itself

Doc 00 C4 does not permit either of these to be resolved silently, so both are explicit
arguments whose defaults are documented rather than assumed.

**1. The adiabatic index in the Bohm speed.** Doc 03 §2.1 writes

    c_s = sqrt(e (T_e + gamma_i T_i) / m_i),   gamma_i = 3 for 1-D adiabatic ions

while doc 01 §2.2 tabulates ``c_s = sqrt(e T_e / m_i) = 2.69 km/s`` — the ``gamma_i = 0``,
cold-ion form. Both are Baseline. :func:`bohm_speed` implements the general expression and
defaults to :data:`GAMMA_I_COLD_ION`, which reproduces the tabulated number and therefore
the tabulated ``Gamma_E``. At RP-1 the two differ by 2.5 % (``T_i / T_e = 1/60``), which is
small enough to sit unnoticed in a document and large enough to matter against the 20 %
end-to-end accuracy target of doc 01 R-ACC-5. Choosing ``gamma_i = 3`` raises the L0
``Gamma_E`` from 6.58 to 6.74 kW/m^2.

**2. Which density sets ``lambda_D`` in the Child-Langmuir thickness.** Doc 03 §2.3 obtains

    s = (sqrt2 / 3) lambda_D (2 V_w / T_e)^(3/4)

by "matching ``J_i`` to the Bohm flux ``e n_s c_s``". Carrying that matching through gives
``lambda_D`` evaluated at the **sheath-edge** density ``n_s = h_l n_0``, not at the bulk
density ``n_0``. Doc 01 §2.2 evaluates it at ``n_0`` and quotes ``s ~ 0.89 mm``; the
self-consistent value is ``0.89 / sqrt(h_l) = 1.14 mm``, 28 % larger.
:func:`child_langmuir_thickness` defaults to the doc 01 convention so that the tabulated
number is reproduced, and takes ``h_l`` to select the self-consistent one. Only the latter
makes :func:`child_langmuir_current_density` return the Bohm flux it was matched to.

This matters beyond bookkeeping: doc 01 §2.2 derives the collisionality ``s / lambda_CX``
and doc 01 §2.3 the ion transit time ``tau_tr = s / sqrt(2 e V_b / m_i)`` from this ``s``.
A 28 % change moves ``tau_tr / T_RF`` from 0.35 to 0.40 — the same regime, so no downstream
conclusion in doc 01 changes, which is why the discrepancy is recorded rather than
escalated.

## Sign convention

Doc 02 §2, as resolved in :mod:`vpl.core.state.params`: the wall is at ``z = 0``, ``z`` is
positive into the plasma, and flux toward the wall is positive. Every formula below takes
``V_w`` from :attr:`PlasmaParams.bias_magnitude`, the positive sheath potential drop. The
potential *field* is negative, reaching ``-V_w`` at the wall.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.constants import ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.params import default_registry
from vpl.core.state import (
    Fidelity,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    TimeGrid,
)
from vpl.core.units import Q_, Quantity, magnitude_in

__all__ = [
    "DEFAULT_DOMAIN_SHEATHS",
    "DEFAULT_EDGE_TO_CENTRE_RATIO",
    "DEFAULT_SAMPLES_PER_SHEATH",
    "GAMMA_I_ADIABATIC",
    "GAMMA_I_COLD_ION",
    "SIGMA_CX_ARGON",
    "AnalyticSheathSolver",
    "CollisionalRegime",
    "SheathModel",
    "bohm_speed",
    "child_langmuir_current_density",
    "child_langmuir_potential",
    "child_langmuir_thickness",
    "collisional_current_density",
    "collisionality",
    "cx_mean_free_path",
    "debye_length",
    "ion_energy_flux",
    "ion_flux",
    "matrix_sheath_potential",
    "matrix_sheath_thickness",
    "matrix_sheath_validity_ratio",
    "sheath_edge_density",
]

#: A field sampled on the spatial grid, in SI magnitudes.
type FloatArray = NDArray[np.float64]

# ── constants ───────────────────────────────────────────────────────────────────
#
# Physical constants are unwrapped to SI floats exactly once, here. Doc 08 §5 puts bare
# arrays inside hot loops and quantities at module boundaries; doc 00 C1 forbids a numeric
# literal with physical meaning anywhere. Both hold if the unwrapping happens at import
# from :mod:`vpl.core.constants` and nowhere else.

#: The registry is "the sole source of numeric defaults" (doc 08 §5). Reading it once at
#: import, rather than repeating its values as literals here, is what makes that true
#: instead of aspirational: there is exactly one place to change h_l, and a divergence
#: between this module and the registry is not expressible.
_REGISTRY: Final = default_registry()

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_EPS0_F_PER_M: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

#: ``gamma_i`` reproducing doc 01 §2.2's ``c_s = sqrt(e T_e / m_i)``. See the module
#: docstring: this is the default, and it is a choice between two Baseline statements.
GAMMA_I_COLD_ION: Final[float] = float(_REGISTRY.value_in("sheath.gamma_i", "dimensionless"))

#: ``gamma_i`` for 1-D adiabatic ions — doc 03 §2.1.
GAMMA_I_ADIABATIC: Final[float] = 3.0

#: The planar edge-to-centre density ratio ``h_l`` used in doc 03 §2.3's own arithmetic
#: (``Gamma_i = 0.61e17 * 2691``). doc 03 §2.1 gives the Godyak low-pressure form
#: ``h_l ~ 0.86 / sqrt(3 + L / (2 lambda_i))``, which needs a discharge length ``L`` that
#: is not part of :class:`PlasmaParams`; ``h_l`` is therefore a registered DESIGN
#: parameter passed in, and this is the RP-1 value the specification uses.
#:
#: It is also, satisfyingly, ``exp(-1/2) = 0.607`` to two figures — the Boltzmann factor
#: for the ``T_e / 2`` presheath drop that accelerates ions to the Bohm speed. So ``h_l``
#: already carries the presheath drop, which is why the sheath models below reference
#: their Boltzmann electrons to ``n_s`` and not to ``n_0``: doing both would count the
#: presheath twice.
DEFAULT_EDGE_TO_CENTRE_RATIO: Final[float] = float(
    _REGISTRY.value_in("sheath.h_l", "dimensionless")
)

#: Ar+/Ar symmetric charge-exchange cross section at RP-1 — doc 01 §2.2, class PUBLISHED.
#: Doc 01 §2.2 is explicit that the implementation "uses the energy-dependent Ar+/Ar
#: symmetric charge-exchange cross section from LXCat rather than a constant" (doc 09), so
#: this is exported as a named reference value and never as a default argument: a caller
#: reaching for it has to name it, and the LXCat substitution stays visible.
SIGMA_CX_ARGON: Final[Quantity] = _REGISTRY["collisions.sigma_cx_Ar"].quantity

#: Default domain size in sheath thicknesses — doc 01 R-SPAT-3 ("field of view must extend
#: to >= 10 x s") so that the presheath and bulk boundary conditions are on the grid.
DEFAULT_DOMAIN_SHEATHS: Final[int] = 10

#: Default samples across one sheath thickness — doc 01 R-SPAT-1 ("resolve s ~ 0.89 mm
#: with >= 10 samples").
DEFAULT_SAMPLES_PER_SHEATH: Final[int] = 10


class SheathModel(StrEnum):
    """Which doc 03 §2 potential profile a solver uses.

    A ``StrEnum`` because manifests are data (doc 08 §1 principle 4): ``model: matrix``
    round-trips into a checked member and back out for the provenance record.
    """

    #: Doc 03 §2.3. The collisionless high-voltage limit and the V-03 target.
    CHILD_LANGMUIR = "child_langmuir"

    #: Doc 03 §2.2. Uniform ion density, no ion acceleration. Valid only for ``V << T_e``;
    #: doc 03 §2.2 uses it "as a limiting case and as a numerical initialisation".
    MATRIX = "matrix"


class CollisionalRegime(StrEnum):
    """Which doc 03 §2.4 current-voltage law applies in the mobility-limited sheath."""

    #: ``J_i ~ V_w^(3/2) / s^(5/2)`` (Warren). Doc 03 §2.4 makes this the default
    #: "because Ar+/Ar charge exchange has a weakly energy-dependent cross section,
    #: making lambda_CX the better-behaved constant".
    CONSTANT_MEAN_FREE_PATH = "constant_mean_free_path"

    #: ``J_i = (9/8) eps_0 mu_i V_w^2 / s^3``.
    CONSTANT_MOBILITY = "constant_mobility"


#: The exponent ``p`` in ``Phi(z) = -V_w (1 - z/s)^p`` for each model.
#:
#: The matrix value is quoted verbatim in doc 03 §2.2. The Child-Langmuir value follows in
#: three lines from the equations doc 03 §2.3 already states: constant ion flux with
#: ``u ~ sqrt(-Phi)`` gives ``n_i ~ (-Phi)^(-1/2)``, so Poisson gives
#: ``d^2Phi/dz'^2 ~ (-Phi)^(-1/2)``, whose solution is ``-Phi ~ z'^(4/3)`` with ``z'``
#: measured from the sheath edge (Langmuir 1913; Lieberman & Lichtenberg §6.3).
_PROFILE_EXPONENT: Final[Mapping[SheathModel, float]] = MappingProxyType(
    {
        SheathModel.MATRIX: 2.0,
        SheathModel.CHILD_LANGMUIR: 4.0 / 3.0,
    }
)


# ── validation helpers ──────────────────────────────────────────────────────────


def _checked_ratio(h_l: float) -> float:
    """Reject an edge-to-centre ratio outside ``(0, 1]``.

    ``h_l`` is the ratio of a monotonically falling presheath profile's endpoint to its
    centre, so it cannot exceed 1 and cannot vanish. A silently accepted ``h_l > 1`` would
    put more plasma at the sheath edge than in the bulk and inflate every flux downstream
    by exactly the factor nobody would think to check.
    """
    if not 0.0 < h_l <= 1.0:
        raise ValueError(
            f"h_l is an edge-to-centre density ratio and must lie in (0, 1], got {h_l}. "
            "doc 03 §2.1 gives h_l ~ 0.86 / sqrt(3 + L / (2 lambda_i))."
        )
    return h_l


def _positive_length_m(value: Quantity, *, name: str) -> float:
    """Convert a length to metres, rejecting a non-positive one.

    Every current density below goes as an inverse power of the sheath thickness. Letting
    ``s = 0`` through would return ``inf``, which propagates into an error budget looking
    like a number. Raising names the caller that asked for it.
    """
    metres = float(magnitude_in(value, "m"))
    if metres <= 0.0:
        raise ValueError(f"{name} must be a positive length, got {value}")
    return metres


def _debye_length_m(*, n_per_m3: float, T_e_J: float) -> float:
    """``lambda_D = sqrt(eps_0 k T_e / (n e^2))`` — doc 01 §2.2, in SI."""
    return float(np.sqrt(_EPS0_F_PER_M * T_e_J / (n_per_m3 * _E_C**2)))


# ── the models ──────────────────────────────────────────────────────────────────


def debye_length(params: PlasmaParams) -> Quantity:
    """Electron Debye length at the **bulk** density — doc 01 §2.2.

    ``lambda_D = sqrt(eps_0 T_e[eV] / (n_0 e))``, which is the same expression as
    ``sqrt(eps_0 k T_e / (n_0 e^2))`` with ``T_e`` carried as an energy. 40.7 um at RP-1.

    doc 03 §3.4 makes this the controlling mesh scale for L1 and doc 03 §4.3 makes it a
    hard PIC stability constraint, so it is evaluated at ``n_0`` — the largest density in
    the problem, and therefore the smallest ``lambda_D``. Inside the sheath the local
    Debye length is longer; see :func:`child_langmuir_thickness` for where that matters.
    """
    return Q_(_debye_length_m(n_per_m3=params.n_0_per_m3, T_e_J=params.T_e_J), "m")


def bohm_speed(params: PlasmaParams, *, gamma_i: float = GAMMA_I_COLD_ION) -> Quantity:
    """Ion-acoustic speed ``c_s = sqrt(e (T_e + gamma_i T_i) / m_i)`` — doc 03 §2.1.

    Doc 03 §2.1 states the Bohm criterion as ``u_s >= c_s``; this is the marginal case,
    which is what every sheath model below assumes at the edge and what verification gate
    V-09 checks (``u_s / c_s = 1.0 +/- 0.05``).

    Args:
        params: Control parameters. ``T_e``, ``T_i`` and the ion mass are used.
        gamma_i: Ion adiabatic index. Doc 03 §2.1 gives 3 for 1-D adiabatic ions;
            doc 01 §2.2 tabulates 2.69 km/s at RP-1, which is the ``gamma_i = 0``
            cold-ion value. The default reproduces the tabulated number. The module
            docstring states the discrepancy in full — this is a doc 00 C4 disclosure,
            not a preference.

    Returns:
        The ion-acoustic speed. 2.69 km/s at RP-1 with the default; 2.76 km/s with
        ``gamma_i = 3``.
    """
    if gamma_i < 0.0:
        raise ValueError(f"gamma_i is an adiabatic index and cannot be negative, got {gamma_i}")
    speed = np.sqrt((params.T_e_J + gamma_i * params.T_i_J) / params.species.mass_kg)
    return Q_(float(speed), "m/s")


def sheath_edge_density(params: PlasmaParams, h_l: float) -> Quantity:
    """``n_s = h_l n_0`` — doc 03 §2.1.

    Args:
        params: Control parameters.
        h_l: The planar edge-to-centre ratio, ``h_l ~ 0.86 / sqrt(3 + L / (2 lambda_i))``
            in the Godyak low-pressure form. Taken as an argument rather than computed:
            the discharge length ``L`` is not a member of :class:`PlasmaParams`, and
            doc 00 C1 classes ``h_l`` as a DESIGN parameter that the sensitivity study
            sweeps. See :data:`DEFAULT_EDGE_TO_CENTRE_RATIO`.
    """
    return Q_(_checked_ratio(h_l) * params.n_0_per_m3, "m**-3")


def child_langmuir_thickness(params: PlasmaParams, *, h_l: float | None = None) -> Quantity:
    """``s = (sqrt2 / 3) lambda_D (2 V_w / T_e)^(3/4)`` — doc 03 §2.3.

    The collisionless Child-Langmuir current matched to the Bohm flux at the sheath edge.
    0.89 mm at RP-1 with the default, which is the value doc 01 §2.2 tabulates and doc 03
    §8 A1 uses to justify the 1-D planar assumption.

    Args:
        params: Control parameters.
        h_l: If given, ``lambda_D`` is evaluated at the sheath-edge density ``h_l n_0``,
            which is what the doc 03 §2.3 matching actually requires and what makes
            :func:`child_langmuir_current_density` return the Bohm flux exactly. If
            ``None`` (the default), ``lambda_D`` is the bulk value and the result
            reproduces doc 01 §2.2. The two differ by ``sqrt(h_l)`` — 28 % at
            ``h_l = 0.61``. The module docstring states this in full.

    Returns:
        The sheath thickness. Zero at zero bias, which is the continuous limit of the
        expression and is correct for this model: with no potential drop there is no
        space-charge region to hold one.
    """
    density = params.n_0_per_m3 if h_l is None else _checked_ratio(h_l) * params.n_0_per_m3
    lambda_D = _debye_length_m(n_per_m3=density, T_e_J=params.T_e_J)

    # V_w and T_e both appear as potentials, so the ratio is taken in volts.
    T_e_volts = params.T_e_J / _E_C
    normalised_bias = 2.0 * float(magnitude_in(params.bias_magnitude, "V")) / T_e_volts

    return Q_((np.sqrt(2.0) / 3.0) * lambda_D * normalised_bias**0.75, "m")


def child_langmuir_current_density(params: PlasmaParams, thickness: Quantity) -> Quantity:
    """``J_i = (4/9) eps_0 sqrt(2 e / m_i) V_w^(3/2) / s^2`` — doc 03 §2.3.

    The collisionless space-charge-limited ion current. Doc 03 §2.3 calls this "the
    framework's primary analytic verification target" and doc 07 V-03 requires L1 and L2
    to reproduce it within 5 %.

    Args:
        params: Control parameters.
        thickness: The sheath thickness ``s``. Taken explicitly rather than computed
            because the caller decides which of the two thicknesses of the module
            docstring it wants, and because L1 and L2 pass their *measured* sheath width
            here when V-03 compares them against this law.

    Returns:
        The ion current density at the wall. Equals ``e n_s c_s`` exactly when
        ``thickness`` came from ``child_langmuir_thickness(params, h_l=h_l)``.
    """
    s_m = _positive_length_m(thickness, name="thickness")
    V_w = float(magnitude_in(params.bias_magnitude, "V"))

    coefficient = (4.0 / 9.0) * _EPS0_F_PER_M * np.sqrt(2.0 * _E_C / params.species.mass_kg)
    return Q_(float(coefficient * V_w**1.5 / s_m**2), "A/m**2")


def matrix_sheath_thickness(params: PlasmaParams) -> Quantity:
    """``s = lambda_D sqrt(2 V_w / T_e)`` — doc 03 §2.2. 0.526 mm at RP-1.

    Uniform ion density, no ion acceleration. See :func:`matrix_sheath_validity_ratio` for
    the regime in which that is defensible.
    """
    T_e_volts = params.T_e_J / _E_C
    normalised_bias = 2.0 * float(magnitude_in(params.bias_magnitude, "V")) / T_e_volts
    lambda_D = _debye_length_m(n_per_m3=params.n_0_per_m3, T_e_J=params.T_e_J)

    return Q_(float(lambda_D * np.sqrt(normalised_bias)), "m")


def matrix_sheath_validity_ratio(params: PlasmaParams) -> float:
    """``V_w / T_e`` — the ratio doc 03 §2.2 requires to be much less than 1.

    Reported rather than warned about. The matrix sheath's stated role in doc 03 §2.2 is
    "a limiting case and a numerical initialisation", and at the project's own reference
    point the ratio is 83, so a warning would fire on every legitimate call — which is how
    warnings get filtered and then ignored. A caller that needs the model to be *valid*
    gates on this number; a caller that needs an initial guess does not.
    """
    return float(magnitude_in(params.bias_magnitude, "V")) / (params.T_e_J / _E_C)


def _potential_profile(
    *, model: SheathModel, V_w: float, z_m: FloatArray, s_m: float
) -> FloatArray:
    """``Phi(z) = -V_w (1 - z/s)^p`` on ``z_m``, clamped to zero beyond the sheath edge.

    The clamp is not cosmetic. Both profiles turn back upward for ``z > s`` and would
    invent a *positive* potential hill in the bulk, which would then accelerate ions the
    wrong way in anything that differentiates the field. The model's domain ends at the
    sheath edge; outside it the plasma is quasineutral and field-free by construction.
    """
    xi = np.clip(1.0 - z_m / s_m, 0.0, 1.0)
    return np.asarray(-V_w * xi ** _PROFILE_EXPONENT[model], dtype=np.float64)


def matrix_sheath_potential(
    params: PlasmaParams, grid: SpatialGrid, thickness: Quantity
) -> Quantity:
    """``Phi(z) = -V_w (1 - z/s)^2`` — doc 03 §2.2, verbatim.

    Args:
        params: Control parameters; only ``bias_magnitude`` is used.
        grid: Where to evaluate. The wall is at ``z = 0`` (doc 02 §2), so ``Phi[0]`` is
            the full drop ``-V_w``.
        thickness: The matrix sheath thickness, normally from
            :func:`matrix_sheath_thickness`. Passed in so that the profile can be
            evaluated on a thickness some other model produced — its documented use as a
            numerical initialisation for L1 needs exactly that.

    Returns:
        The potential on ``grid``, zero at and beyond the sheath edge.
    """
    return Q_(
        _potential_profile(
            model=SheathModel.MATRIX,
            V_w=float(magnitude_in(params.bias_magnitude, "V")),
            z_m=grid.z_m,
            s_m=_positive_length_m(thickness, name="thickness"),
        ),
        "V",
    )


def child_langmuir_potential(
    params: PlasmaParams, grid: SpatialGrid, thickness: Quantity
) -> Quantity:
    """``Phi(z) = -V_w (1 - z/s)^(4/3)`` — the profile of the doc 03 §2.3 sheath.

    Doc 03 §2.3 states the current and the thickness of the collisionless Child-Langmuir
    sheath but not its potential profile; the 4/3 power is the solution of the same
    equations (see :data:`_PROFILE_EXPONENT` for the derivation, and Lieberman &
    Lichtenberg §6.3 for the printed version).
    """
    return Q_(
        _potential_profile(
            model=SheathModel.CHILD_LANGMUIR,
            V_w=float(magnitude_in(params.bias_magnitude, "V")),
            z_m=grid.z_m,
            s_m=_positive_length_m(thickness, name="thickness"),
        ),
        "V",
    )


def collisional_current_density(
    params: PlasmaParams,
    thickness: Quantity,
    *,
    regime: CollisionalRegime = CollisionalRegime.CONSTANT_MEAN_FREE_PATH,
    mean_free_path: Quantity | None = None,
    mobility: Quantity | None = None,
) -> Quantity:
    """The mobility-limited ion current of doc 03 §2.4, in either variant.

    Applies when ``s >~ lambda_CX``; see :func:`collisionality`. At RP-1 the ratio is
    0.072, so the collisionless law of :func:`child_langmuir_current_density` governs and
    this one is out of its domain — but doc 01 §2.2 notes that 50 mTorr in the same
    operating envelope gives 0.7, "a genuinely collisional sheath", which is why both
    limits are implemented rather than one.

    **Constant mean free path** (the default, attributed to Warren in doc 03 §2.4). Doc 03
    §2.4 gives only the scaling ``J_i ~ V_w^(3/2) / s^(5/2)``. The prefactor is the
    standard one, from the ion drift law ``u = sqrt(2 e lambda_i E / (pi m_i))`` integrated
    through Poisson's equation (Lieberman & Lichtenberg eq. 6.5.7)::

        J_i = (2/3) (5/3)^(3/2) eps_0 sqrt(2 e lambda_i / (pi m_i)) V_w^(3/2) / s^(5/2)

    **Constant mobility** (doc 03 §2.4, verbatim) — the Mott-Gurney form::

        J_i = (9/8) eps_0 mu_i V_w^2 / s^3

    Args:
        params: Control parameters.
        thickness: The sheath thickness ``s``.
        regime: Which law. Doc 03 §2.4: "Both are available; the choice is a registered
            parameter."
        mean_free_path: ``lambda_i``, required by
            :attr:`CollisionalRegime.CONSTANT_MEAN_FREE_PATH`. Normally the
            charge-exchange mean free path from :func:`cx_mean_free_path`, because doc 03
            §4.5 makes CX "the single most important collision for this project".
        mobility: ``mu_i``, required by :attr:`CollisionalRegime.CONSTANT_MOBILITY`.

    Returns:
        The ion current density at the wall.

    Raises:
        ValueError: If the argument the chosen regime needs was not supplied. The two
            regimes take genuinely different physical inputs and silently defaulting
            either would be a fabricated constant.
    """
    s_m = _positive_length_m(thickness, name="thickness")
    V_w = float(magnitude_in(params.bias_magnitude, "V"))

    if regime is CollisionalRegime.CONSTANT_MOBILITY:
        if mobility is None:
            raise ValueError(
                "the constant-mobility regime needs an ion mobility; pass mobility=..., "
                "e.g. Q_(mu_i, 'm**2 / V / s')"
            )
        mu = float(magnitude_in(mobility, "m**2 / V / s"))
        return Q_((9.0 / 8.0) * _EPS0_F_PER_M * mu * V_w**2 / s_m**3, "A/m**2")

    if mean_free_path is None:
        raise ValueError(
            "the constant-mean-free-path regime needs an ion mean_free_path; pass "
            "mean_free_path=cx_mean_free_path(params, sigma_cx)"
        )
    lambda_i = _positive_length_m(mean_free_path, name="mean_free_path")

    coefficient = (2.0 / 3.0) * (5.0 / 3.0) ** 1.5 * _EPS0_F_PER_M
    drift = np.sqrt(2.0 * _E_C * lambda_i / (np.pi * params.species.mass_kg))
    return Q_(float(coefficient * drift * V_w**1.5 / s_m**2.5), "A/m**2")


def cx_mean_free_path(params: PlasmaParams, sigma_cx: Quantity) -> Quantity:
    """``lambda_CX = 1 / (n_g sigma_CX)`` — doc 01 §2.2. 12.4 mm at RP-1.

    Args:
        params: Control parameters; ``n_g = p / (k_B T_g)`` comes from
            :attr:`PlasmaParams.n_g`.
        sigma_cx: The charge-exchange cross section. Required, never defaulted:
            doc 01 §2.2 marks the constant 5e-19 m^2 as a PUBLISHED reference value and
            states that the implementation uses the energy-dependent LXCat curve instead.
            See :data:`SIGMA_CX_ARGON`.
    """
    sigma = float(magnitude_in(sigma_cx, "m**2"))
    if sigma <= 0.0:
        raise ValueError(f"sigma_cx must be a positive cross section, got {sigma_cx}")
    return Q_(1.0 / (params.n_g_per_m3 * sigma), "m")


def collisionality(
    params: PlasmaParams, sigma_cx: Quantity, *, thickness: Quantity | None = None
) -> float:
    """``s / lambda_CX`` — doc 01 §2.2. 0.072 at RP-1, "near-collisionless".

    This is the number that decides which of the doc 03 §2.3 and §2.4 current laws
    applies, and doc 01 §2.2 makes the envelope span it: 0.072 at 5 mTorr, 0.7 at
    50 mTorr. Returned as a bare ``float`` rather than a dimensionless
    :class:`~vpl.core.units.Quantity` because it is compared against 1 everywhere
    downstream, and a dimensionless-but-not-unity unit such as percent creeping in would
    be silent.

    Args:
        params: Control parameters.
        sigma_cx: The charge-exchange cross section.
        thickness: The sheath thickness. Defaults to
            :func:`child_langmuir_thickness`, which is what doc 01 §2.2 uses.
    """
    s = child_langmuir_thickness(params) if thickness is None else thickness
    return float(magnitude_in(s, "m")) / float(
        magnitude_in(cx_mean_free_path(params, sigma_cx), "m")
    )


def ion_flux(
    params: PlasmaParams,
    *,
    h_l: float = DEFAULT_EDGE_TO_CENTRE_RATIO,
    gamma_i: float = GAMMA_I_COLD_ION,
) -> Quantity:
    """``Gamma_i = h_l n_0 c_s`` — the Bohm flux of doc 01 §1.3. 1.64e20 m^-2 s^-1 at RP-1.

    Positive toward the wall, per doc 02 §2. Doc 01 §1.3 draws the consequence worth
    remembering: the ion *particle* flux is controlled by the **electron** temperature,
    because it is the electrons that set the ambipolar field.
    """
    speed = float(magnitude_in(bohm_speed(params, gamma_i=gamma_i), "m/s"))
    return Q_(_checked_ratio(h_l) * params.n_0_per_m3 * speed, "1 / (m**2 * s)")


def ion_energy_flux(
    params: PlasmaParams,
    *,
    h_l: float = DEFAULT_EDGE_TO_CENTRE_RATIO,
    gamma_i: float = GAMMA_I_COLD_ION,
) -> Quantity:
    """``Gamma_E = Gamma_i e V_w = h_l n_0 c_s e V_w`` — doc 03 §2.3. **6.6 kW/m^2 at RP-1.**

    This is the quantity of interest of doc 01 §1.1 evaluated in the collisionless
    high-bias limit, and doc 03 §2.3 calls it "the number every other model must reproduce
    in this limit". It is verification gate V-03.

    **What this expression leaves out, deliberately.** Doc 01 §1.3 writes the mean impact
    energy as ``<E_i> = e (Phi_s - Phi_w) + <E_i>_entry - dE_collisions``. L0 keeps only
    the first term. The consequences are worth stating because doc 00 C4 does not allow
    them to be discovered later:

    - **Entry energy is dropped.** Ions cross the sheath edge at ``c_s``, carrying
      ``~T_e/2`` each. At RP-1 that is 1.5 eV against 250 eV, i.e. 0.6 %, so the omission
      is negligible where the model is meant to be used and total where it is not: at zero
      bias this function returns exactly zero while the true flux does not.
    - **Collisional losses are dropped.** Doc 03 §4.5: a charge-exchange event replaces a
      fast ion with a slow one, and "it is what makes ``<E_i>`` differ substantially from
      ``e V_w``". At ``s / lambda_CX = 0.072`` that is a small correction; at 0.7 it is not,
      and L0 is then a bound rather than an estimate.

    So the L0 flux is a **lower bound that is asymptotically exact at high bias in a
    collisionless sheath**, and that is the only regime in which V-03 asks anything of it.

    Args:
        params: Control parameters. The sheath drop is ``bias_magnitude``, so a positively
            biased electrode gives the same flux as a negatively biased one of equal
            magnitude — the model knows about the size of the drop, not its polarity.
        h_l: Edge-to-centre density ratio. See :data:`DEFAULT_EDGE_TO_CENTRE_RATIO`.
        gamma_i: Ion adiabatic index. See :func:`bohm_speed` and the module docstring.
    """
    flux = float(magnitude_in(ion_flux(params, h_l=h_l, gamma_i=gamma_i), "1 / (m**2 * s)"))
    # e V_w, in joules: the energy one ion gains falling through the whole sheath drop.
    energy_per_ion = _E_C * float(magnitude_in(params.bias_magnitude, "V"))
    return Q_(flux * energy_per_ion, "W/m**2")


# ── the assembled state ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AnalyticSheathSolver:
    """L0 of the doc 03 §1 hierarchy, assembled into a :class:`PlasmaState`.

    Shaped to the ``ForwardSolver`` contract of doc 08 §4 — ``solve``, ``flux``,
    ``fidelity`` — but not declared to implement it: ``vpl.core.protocols`` is still a
    placeholder at the time of writing, and importing a protocol that does not exist yet
    to prove conformance would prove nothing. Structural typing means the annotation can
    be added later without touching this class.

    Attributes:
        h_l: Planar edge-to-centre density ratio, a DESIGN parameter.
            See :data:`DEFAULT_EDGE_TO_CENTRE_RATIO`.
        gamma_i: Ion adiabatic index. See :func:`bohm_speed`.
        model: Which doc 03 §2 potential profile to lay down.
        domain_sheaths: Default domain size in sheath thicknesses (doc 01 R-SPAT-3).
        samples_per_sheath: Default grid resolution (doc 01 R-SPAT-1).
    """

    h_l: float = DEFAULT_EDGE_TO_CENTRE_RATIO
    gamma_i: float = GAMMA_I_COLD_ION
    model: SheathModel = SheathModel.CHILD_LANGMUIR
    domain_sheaths: int = DEFAULT_DOMAIN_SHEATHS
    samples_per_sheath: int = DEFAULT_SAMPLES_PER_SHEATH

    def __post_init__(self) -> None:
        _checked_ratio(self.h_l)
        if self.gamma_i < 0.0:
            raise ValueError(f"gamma_i cannot be negative, got {self.gamma_i}")
        if self.domain_sheaths < 1 or self.samples_per_sheath < 1:
            raise ValueError(
                "domain_sheaths and samples_per_sheath must both be at least 1, got "
                f"{self.domain_sheaths} and {self.samples_per_sheath}"
            )

    def fidelity(self) -> Fidelity:
        return Fidelity.L0

    def thickness(self, params: PlasmaParams) -> Quantity:
        """The sheath thickness this solver lays its profile on.

        Uses the self-consistent, sheath-edge-referenced form (``h_l`` supplied), because
        a state whose ``J_i`` did not equal the Bohm flux it was matched to would fail its
        own V-03 check. The 0.89 mm of doc 01 §2.2 comes from
        ``child_langmuir_thickness(params)`` with no ``h_l``; see the module docstring.
        """
        if self.model is SheathModel.MATRIX:
            return matrix_sheath_thickness(params)
        return child_langmuir_thickness(params, h_l=self.h_l)

    def default_grid(self, params: PlasmaParams) -> SpatialGrid:
        """A uniform grid sized to the requirements of doc 01 §2.4.

        R-SPAT-3 wants the field of view to reach ``>= 10 s`` so the presheath and bulk
        boundary conditions are on the grid; R-SPAT-1 wants ``>= 10`` samples across ``s``.
        Uniform rather than graded: L0 has no numerical stiffness to grade against, and
        the graded mesh of doc 03 §3.4 exists for the L1 discretisation.
        """
        s_m = float(magnitude_in(self.thickness(params), "m"))
        if s_m <= 0.0:
            raise ValueError(
                "cannot size a default grid at zero bias: the sheath thickness vanishes. "
                "Pass an explicit grid= if you want the (flat, quasineutral) zero-bias state."
            )
        return SpatialGrid.uniform(
            length=Q_(self.domain_sheaths * s_m, "m"),
            n_points=self.domain_sheaths * self.samples_per_sheath + 1,
        )

    def solve(
        self,
        params: PlasmaParams,
        *,
        grid: SpatialGrid | None = None,
        time: TimeGrid | None = None,
    ) -> PlasmaState:
        """Build the L0 state.

        The fields are assembled from three statements, none of them new:

        - ``Phi(z)``: the doc 03 §2 profile selected by :attr:`model`.
        - ``n_i(z)``: ion flux conservation, ``n_i u_i = n_s c_s``, with
          ``u_i = sqrt(c_s^2 + 2 e (-Phi) / m_i)`` from energy conservation. Retaining the
          ``c_s^2`` term is not decoration: dropping it, as the textbook Child-Langmuir
          derivation does, makes ``n_i`` diverge at the sheath edge.
        - ``n_e(z)``: Boltzmann electrons, ``n_e = n_s exp(e Phi / k T_e)`` — doc 03 §3.1.

        **The Boltzmann reference density is ``n_s``, not ``n_0``.** Doc 03 §3.1 writes
        ``n_e = n_0 exp(e Phi / k T_e)`` with ``Phi`` referenced to the *bulk*; this model
        lives inside the sheath, where ``Phi = 0`` at the sheath edge. Using ``n_0`` there
        would break quasineutrality at the outer boundary by exactly the factor ``h_l``,
        and would double-count the presheath drop that ``h_l`` already encodes — see
        :data:`DEFAULT_EDGE_TO_CENTRE_RATIO`.

        Args:
            params: Control parameters.
            grid: Where to evaluate. Defaults to :meth:`default_grid`.
            time: Carried through to the state. L0 is analytic and steady; a time grid
                would imply a time dependence this level does not model, so the default
                is ``None`` and a caller supplying one is responsible for what it means.

        Returns:
            A steady :class:`PlasmaState` at :attr:`Fidelity.L0` carrying ``n_e``, ``n_i``,
            ``Phi``, ``T_e`` and ``u_i``, and **no ion distribution**: doc 03 §6 obtains
            ``f_i`` at L0 as an *assumed* drifting Maxwellian rather than a computed one,
            and :attr:`Fidelity.L0.resolves_distribution` is ``False``. Recording the
            absence is more honest than manufacturing a distribution the level never
            solved for, and the assumption then has to be made explicitly by whoever needs
            it, where it can be swept and budgeted.
        """
        mesh = self.default_grid(params) if grid is None else grid
        s_m = float(magnitude_in(self.thickness(params), "m"))

        V_w = float(magnitude_in(params.bias_magnitude, "V"))
        c_s = float(magnitude_in(bohm_speed(params, gamma_i=self.gamma_i), "m/s"))
        n_s = self.h_l * params.n_0_per_m3
        T_e_eV = params.T_e_J / _E_C

        # At zero bias there is no sheath and the profile expressions divide by s. The
        # limit is unambiguous: a flat potential and a uniform quasineutral plasma.
        phi_volts = (
            np.zeros(mesh.n_points, dtype=np.float64)
            if s_m <= 0.0
            else _potential_profile(model=self.model, V_w=V_w, z_m=mesh.z_m, s_m=s_m)
        )

        u_i = np.sqrt(c_s**2 + 2.0 * _E_C * (-phi_volts) / params.species.mass_kg)
        n_i = n_s * c_s / u_i
        n_e = n_s * np.exp(phi_volts / T_e_eV)

        def _field(name: str, values: FloatArray, units: str) -> ScalarField:
            return ScalarField(name=name, values=values, units=units, grid=mesh, time=time)

        return PlasmaState(
            params=params,
            grid=mesh,
            time=time,
            fields={
                "n_e": _field("n_e", n_e, "m**-3"),
                "n_i": _field("n_i", n_i, "m**-3"),
                "Phi": _field("Phi", phi_volts, "V"),
                # Isothermal. doc 03 §3.1 makes the electron energy equation optional and
                # L0 does not solve it; a uniform T_e is the assumption, stated as a field.
                "T_e": _field("T_e", np.full(mesh.n_points, T_e_eV), "eV"),
                "u_i": _field("u_i", u_i, "m/s"),
            },
            ion_distribution=None,
            fidelity=Fidelity.L0,
        )

    def flux(self, state: PlasmaState) -> Quantity:
        """``Gamma_E`` at the wall for a state this solver produced — doc 03 §6.

        Delegates to :func:`ion_energy_flux` rather than integrating the state's fields,
        so that the solver cannot develop its own opinion about the quantity of interest.
        Doc 03 §6 requires "the same functional" at every level.
        """
        return ion_energy_flux(state.params, h_l=self.h_l, gamma_i=self.gamma_i)
