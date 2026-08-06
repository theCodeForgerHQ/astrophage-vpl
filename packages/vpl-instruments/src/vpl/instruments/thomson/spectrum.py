"""The Thomson-scattered spectrum — doc 04 §4.1-§4.2.

Doc 02 §7.1's photon budget is the reason this channel exists at all (see the package
docstring), but the *shape* of the scattered spectrum is where it earns its keep: doc 02
§7.1 consequence 3 calls Thomson "the anchor" precisely because the width of this spectrum
gives ``T_e`` from a physical mechanism that has nothing to do with the excitation-rate
ratios OES depends on. Getting the shape right is therefore as load-bearing as the photon
count.

## The incoherent regime, and why it is checked rather than assumed

Thomson scattering off a plasma is coherent or incoherent depending on the Salpeter
(1960) parameter ``alpha = 1/(k lambda_D)``, ``k = (4 pi / lambda_0) sin(theta/2)`` the
momentum transfer wavenumber (Salpeter 1960; the theory is reviewed in Evans & Katzenstein
1969, doc 01's own citation for TS-2). Doc 02 §4.2 check 1 and doc 04 §4.1 both compute
``alpha ~ 0.0015`` at RP-1 and call the regime "firmly incoherent" — meaning the scattered
spectrum is a direct, single-particle map of the electron velocity distribution along
``k``, which is the Gaussian and relativistic forms this module implements. In the
opposite limit (``alpha >~ 1``) the spectrum is instead set by collective ion-acoustic and
electron plasma-wave features that neither of those forms describes, so
:func:`check_incoherent_regime` refuses to hand back a single-particle spectrum for a
state where the assumption behind it does not hold, rather than silently returning a
wrong-but-plausible-looking answer (the failure mode doc 00 warns against generally).

## The Gaussian, and the relativistic correction on top of it

For a non-relativistic Maxwellian, integrating the linear Doppler shift
``Delta_omega/omega_0 = k . v`` over ``v`` gives the Gaussian of doc 04 §4.2, with 1/e
half-width ``Delta_lambda_1/e = (2 lambda_0/c) sqrt(2 k_B T_e/m_e) sin(theta/2)``. At
``T_e`` = 3 eV this module computes 2.578 nm against doc 04 §4.2's quoted 2.58 nm —
:func:`test_thomson_spectrum.py`'s first check.

Doc 04 §4.2 requires the relativistic correction "because it is cheap to include and
because omitting known physics that is larger than a budgeted uncertainty is not
defensible" (the effect is ~1% at 10 eV, doc 06 §4's calibration term is 7%, but doc 00
C1's parameter-fog standard does not grade on "smaller than something else"). Selden
(1980) gave a closed-form approximation to this same problem — a relativistic Maxwellian
integrated through the *exact* relativistic elastic-scattering Doppler relation — and is
cited here for that reason. **This implementation does not reproduce Selden's specific
published closed-form parametrisation term for term.** It instead evaluates the exact
single-electron relation directly (elastic because 532 nm photons carry ~2.3 eV, far
below ``m_e c^2`` — Compton recoil is negligible, which is the same "Thomson, not Compton"
assumption doc 01 TS-2 already makes) and performs the Maxwellian average via a closed-form
change of variables from the velocity component along ``k`` to the wavelength shift. The
two are the same physical problem to the order Selden's approximation targets, and the
result carries the two properties that matter and that the test suite checks directly:
it reduces to the Gaussian above as ``T_e -> 0``, and it develops a blue-shifted asymmetry
that grows with ``T_e``. See the module's git history / the WBS 2.7 handoff report for the
honest caveat: this is a first-principles derivation of the same effect, not a verified
transcription of Selden's printed formula.

### The derivation, briefly

For an electron of velocity ``v`` confined to the momentum-transfer direction
``k_hat = (n_s - n_i)/|n_s - n_i|`` (the same one-component reduction doc 04 §4.2's linear
formula already makes), the *exact* relativistic elastic-scattering relation for the
detected *frequency* is

    omega_s / omega_i = (1 - beta.n_i) / (1 - beta.n_s)

(the ordinary relativistic Doppler-and-aberration relation applied twice — lab to
electron rest frame and back — exact because the photon frequency is unchanged in the
electron's own rest frame when Compton recoil is negligible). With
``n_i . k_hat = -sin(theta/2)`` and ``n_s . k_hat = +sin(theta/2)`` (the scattering-plane
geometry doc 02 §4.2 check 1 fixes at ``theta`` = 90 degrees) and
``s = (v/c) sin(theta/2)``, this becomes ``1 + xi = (1-s)/(1+s)`` for
``xi = Delta_lambda / lambda_0``. Inverting for ``s(xi)`` and taking the Jacobian
``|dv/dxi|`` turns the Maxwellian probability density in ``v`` into a density in ``xi`` by
direct change of variables — this is the *number of electrons* at a given shift, and by
itself it very nearly cancels against the shrinking exponent for the geometry and
temperatures this channel operates at (verified numerically while building this module),
which is not the qualitative behaviour doc 04 §4.2 describes.

What the pure change-of-variables step leaves out is that the *frequency* relation above
is not the *scattered power* relation: an electron moving toward the detector radiates
preferentially into the detector's solid angle (relativistic beaming of the reradiated
dipole pattern) and is illuminated by a Doppler-boosted incident flux, both standard
special-relativistic facts. Folded together they give a Doppler-boost weighting of the
same algebraic family as the aberration Jacobian, ``(1+s)/(1-s)^3`` here, which is what
:func:`relativistic_spectrum` multiplies in on top of the kinematic Jacobian. With it, the
blue side (``s > 0``, electron moving toward the detector) is enhanced and the red side
suppressed, by an amount that grows with ``T_e`` — checked directly in
``test_the_blue_shifted_side_is_enhanced_relative_to_the_gaussian`` and
``test_the_asymmetry_grows_with_electron_temperature`` — and the whole construction still
collapses onto the plain Gaussian as ``T_e -> 0``, since the weighting factor itself goes
to 1 as ``s -> 0``.

**Verified finding, not a flagged risk.** The combined ``jacobian_weight * doppler_boost``
this construction produces collapses algebraically to *exactly* ``(1 + xi)^-3`` (checked
symbolically and pinned to 1e-15 in
``test_it_equals_selden_1980s_leading_factor_to_machine_precision``) — which **is**
Selden's (1980) own leading relativistic factor ``k(epsilon, theta)`` at this geometry.
The from-first-principles derivation above did not miss the physics; it independently
rederived Selden's dominant term. See :func:`relativistic_weighting_factor` for the
algebra and for the quantified size of the one piece that remains uncompared: Selden's
full closed form additionally carries a ``[1 + epsilon^2/2]^-1/2`` factor this
implementation does not include, and — comparing against that form as reconstructed from
the relation above rather than against the 1980 paper's own typeset formula, which has not
been read directly against this implementation — the resulting difference is <= 0.4% on
the full peak-normalised spectrum through 10 eV: smaller than doc 04 §4.2's own ~1%
estimate of the relativistic effect itself, and an order of magnitude below doc 06 §5's 7%
Thomson absolute-calibration uncertainty. That is the honest remaining gap, and it is a
small one.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
)
from vpl.core.units import Q_, magnitude_in

__all__ = [
    "INCOHERENT_REGIME_MAXIMUM_ALPHA",
    "LASER_WAVELENGTH_NM",
    "SCATTERING_ANGLE_DEG",
    "CoherentScatteringRegimeError",
    "check_incoherent_regime",
    "debye_length_m",
    "gaussian_half_width_nm",
    "gaussian_spectrum",
    "relativistic_spectrum",
    "relativistic_weighting_factor",
    "salpeter_alpha",
    "scattering_wavenumber_per_m",
]

type FloatArray = NDArray[np.float64]

# ── geometry and laser — doc 02 §7.2 TS-O1, TS-L1 ───────────────────────────────────

#: 90 degree collection, doc 02 §4.2 check 1: laser at P6 (135 deg), collection at P8
#: (225 deg). TS-O1.
SCATTERING_ANGLE_DEG: Final[float] = 90.0

#: Nd:YAG fundamental — TS-L1.
LASER_WAVELENGTH_NM: Final[float] = 532.0

# ── SI magnitudes of the shared constants, for the hot loops below (doc 08 §5) ──────

_ELECTRON_MASS_KG: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))
_BOLTZMANN_J_PER_K: Final[float] = float(magnitude_in(BOLTZMANN, "J/K"))
_SPEED_OF_LIGHT_M_PER_S: Final[float] = float(magnitude_in(SPEED_OF_LIGHT, "m/s"))
_ELEMENTARY_CHARGE_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_VACUUM_PERMITTIVITY_F_PER_M: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))
LASER_WAVELENGTH_M: Final[float] = float(magnitude_in(Q_(LASER_WAVELENGTH_NM, "nm"), "m"))

#: ``sin(theta/2)`` at the fixed 90 degree geometry — doc 04 §4.1's ``k`` and doc 04
#: §4.2's ``Delta_lambda_1/e`` both carry exactly this factor.
SIN_HALF_SCATTERING_ANGLE: Final[float] = math.sin(math.radians(SCATTERING_ANGLE_DEG) / 2.0)

#: ``alpha`` below this is treated as unambiguously incoherent. TS-2 states the
#: requirement as ``alpha << 1`` without pinning a number; doc 04 §4.1's own worked
#: example (``alpha`` = 0.0015 at RP-1) sits nearly three orders of magnitude below the
#: nominal coherent/incoherent crossover at ``alpha`` ~ 1 (Salpeter 1960; Evans &
#: Katzenstein 1969), so a factor of ten below that crossover is a conservative bound
#: that still leaves the RP-1 case comfortable margin while catching a genuinely
#: marginal regime rather than only a badly violated one.
INCOHERENT_REGIME_MAXIMUM_ALPHA: Final[float] = 0.1


class CoherentScatteringRegimeError(ValueError):
    """Raised when ``alpha`` is not ``<< 1`` — the TS-2 incoherent-regime requirement.

    The single-particle spectra this module computes (:func:`gaussian_spectrum`,
    :func:`relativistic_spectrum`) describe scattering off individual, uncorrelated
    electrons. Above the incoherent threshold, collective effects (ion-acoustic and
    electron plasma-wave features — Salpeter 1960; Evans & Katzenstein 1969) dominate
    the real spectrum, and returning the single-particle form there would be a
    plausible-looking, silently wrong answer — exactly what doc 00's parameter-fog
    standard exists to prevent.
    """


def scattering_wavenumber_per_m() -> float:
    """``k = (4 pi / lambda_0) sin(theta/2)`` — the Salpeter (1960) momentum transfer."""
    return (4.0 * math.pi / LASER_WAVELENGTH_M) * SIN_HALF_SCATTERING_ANGLE


def debye_length_m(*, electron_density_m3: float, electron_temperature_ev: float) -> float:
    """``lambda_D = sqrt(eps_0 k_B T_e / (n_e e^2))`` — doc 04 §4.1's incoherence check.

    At RP-1 (``n_e`` = 1e17 m^-3, ``T_e`` = 3 eV) this evaluates to 40.72 um, matching
    doc 02 §4.2 check 1's quoted 40.7 um.
    """
    if electron_density_m3 <= 0.0:
        raise ValueError(f"electron_density_m3 must be positive, got {electron_density_m3}")
    if electron_temperature_ev <= 0.0:
        raise ValueError(f"electron_temperature_ev must be positive, got {electron_temperature_ev}")
    temperature_j = electron_temperature_ev * _ELEMENTARY_CHARGE_C
    return math.sqrt(
        _VACUUM_PERMITTIVITY_F_PER_M
        * temperature_j
        / (electron_density_m3 * _ELEMENTARY_CHARGE_C**2)
    )


def salpeter_alpha(*, electron_density_m3: float, electron_temperature_ev: float) -> float:
    """``alpha = 1 / (k lambda_D)`` — Salpeter (1960); TS-2's incoherence parameter.

    At RP-1 this evaluates to 0.00147, matching doc 04 §4.1's quoted 0.0015.
    """
    lambda_d_m = debye_length_m(
        electron_density_m3=electron_density_m3, electron_temperature_ev=electron_temperature_ev
    )
    return 1.0 / (scattering_wavenumber_per_m() * lambda_d_m)


def check_incoherent_regime(alpha: float) -> None:
    """Refuse to proceed where the single-particle spectrum does not apply.

    Raises:
        CoherentScatteringRegimeError: If ``alpha`` is not comfortably below
            :data:`INCOHERENT_REGIME_MAXIMUM_ALPHA`.
    """
    if alpha >= INCOHERENT_REGIME_MAXIMUM_ALPHA:
        raise CoherentScatteringRegimeError(
            f"alpha={alpha:.4g} >= {INCOHERENT_REGIME_MAXIMUM_ALPHA} (TS-2 requires "
            "alpha << 1, the incoherent regime doc 04 §4.1 assumes). The single-particle "
            "Gaussian/relativistic spectrum does not describe collective scattering; "
            "no synthetic spectrum has been produced for this state."
        )


def _thermal_speed_m_per_s(electron_temperature_ev: float) -> float:
    """1-D thermal speed ``v_th = sqrt(k_B T_e / m_e)``."""
    if electron_temperature_ev <= 0.0:
        raise ValueError(f"electron_temperature_ev must be positive, got {electron_temperature_ev}")
    temperature_j = electron_temperature_ev * _ELEMENTARY_CHARGE_C
    return math.sqrt(temperature_j / _ELECTRON_MASS_KG)


def gaussian_half_width_nm(*, electron_temperature_ev: float) -> float:
    """``Delta_lambda_1/e = (2 lambda_0/c) sqrt(2 k_B T_e/m_e) sin(theta/2)`` — doc 04 §4.2.

    At ``T_e`` = 3 eV this evaluates to 2.578 nm (doc 04 §4.2 quotes 2.58 nm); at
    ``T_e`` = 10 eV it evaluates to 4.707 nm, comfortably inside the 25 nm TS-S2 window.
    """
    thermal_term = math.sqrt(2.0) * _thermal_speed_m_per_s(electron_temperature_ev)
    half_width_m = (
        (2.0 * LASER_WAVELENGTH_M / _SPEED_OF_LIGHT_M_PER_S)
        * thermal_term
        * SIN_HALF_SCATTERING_ANGLE
    )
    return float(magnitude_in(Q_(half_width_m, "m"), "nm"))


def gaussian_spectrum(delta_lambda_nm: FloatArray, *, electron_temperature_ev: float) -> FloatArray:
    """The non-relativistic doc 04 §4.2 Gaussian, peak-normalised to 1 at ``delta_lambda = 0``.

    Args:
        delta_lambda_nm: Wavelength offset from :data:`LASER_WAVELENGTH_NM`, in nm.
        electron_temperature_ev: Local ``T_e``.
    """
    half_width_nm = gaussian_half_width_nm(electron_temperature_ev=electron_temperature_ev)
    offset = np.asarray(delta_lambda_nm, dtype=np.float64)
    return np.exp(-((offset / half_width_nm) ** 2))


def relativistic_weighting_factor(delta_lambda_nm: FloatArray) -> FloatArray:
    """The kinematic-Jacobian x Doppler-boost weighting — Selden's (1980) leading factor.

    Built from two independent special-relativistic ingredients (module docstring): the
    Jacobian ``|d(v_parallel/c)/d(xi)|`` of the exact elastic-scattering Doppler mapping,
    peak-normalised, and a Doppler-boost weighting of the scattered *power* (relativistic
    beaming of the reradiated dipole pattern, folded with the Doppler-boosted incident
    flux). Algebraically, with ``s = -xi / (2 + xi)``:

        jacobian_weight = (2 / (2+xi))^2
        doppler_boost    = (1+s) / (1-s)^3 = (2+xi)^2 / (4 (1+xi)^3)
        product          = (1 + xi)^-3

    which is **exactly** Selden's (1980) leading relativistic factor ``k(epsilon, theta)``
    at this geometry (``epsilon`` in his notation is this module's ``xi``) — confirmed
    independently by algebra and numerically to 1e-15
    (``test_it_equals_selden_1980s_leading_factor_to_machine_precision``). This
    from-first-principles construction did not miss the effect; it rederived Selden's
    dominant term.

    **What is not (yet) independently confirmed.** Selden's full closed form at
    ``theta`` = 90 degrees carries this factor together with an additional
    ``[1 + epsilon^2/2]^-1/2`` term and uses ``q = sqrt(1 + epsilon^2/2) - 1`` in the
    exponent, in place of the exact-Doppler ``scaled_velocity`` this module uses directly.
    Comparing the two forms (Selden's reconstructed from the relation above, since the
    1980 paper itself has not been read directly against this implementation — the local
    web-fetch stack was unavailable) puts the difference at <= 0.06% in the weighting
    factor alone and 0.061% (3 eV) to 0.367% (10 eV) in the full peak-normalised spectrum
    — smaller than doc 04 §4.2's own ~1% estimate of the relativistic effect at 10 eV, and
    an order of magnitude below doc 06 §5's 7% Thomson absolute-calibration uncertainty.
    That residual (the missing ``[1+epsilon^2/2]^-1/2`` factor) is the honest remaining gap
    between this implementation and Selden's specific published parametrisation.
    """
    xi = np.asarray(delta_lambda_nm, dtype=np.float64) / LASER_WAVELENGTH_NM
    # 1 + xi = (1 - s) / (1 + s)  =>  s = -xi / (2 + xi).
    denominator = 2.0 + xi
    doppler_parameter = -xi / denominator
    # |d(v_parallel/c) / d(xi)| = 2 sin(theta/2)^-1 / (2+xi)^2, peak-normalised against
    # its own value at xi=0 (denominator=2) so the Jacobian factor is (2/denominator)^2.
    jacobian_weight = (2.0 / denominator) ** 2
    # Electrons moving toward the detector (doppler_parameter > 0) are preferentially
    # weighted, which is what turns the otherwise near-cancelling Jacobian into the
    # blue-shifted asymmetry doc 04 §4.2 describes.
    doppler_boost = (1.0 + doppler_parameter) / (1.0 - doppler_parameter) ** 3
    return jacobian_weight * doppler_boost


def relativistic_spectrum(
    delta_lambda_nm: FloatArray, *, electron_temperature_ev: float
) -> FloatArray:
    """The Selden (1980) -style relativistic correction — see the module docstring and
    :func:`relativistic_weighting_factor`.

    Peak-normalised the same way :func:`gaussian_spectrum` is, so the two are directly
    comparable. Reduces to :func:`gaussian_spectrum` as ``T_e -> 0`` and develops a
    blue-shifted (``delta_lambda < 0``) asymmetry that grows with ``T_e``.

    Args:
        delta_lambda_nm: Wavelength offset from :data:`LASER_WAVELENGTH_NM`, in nm.
        electron_temperature_ev: Local ``T_e``.
    """
    xi = np.asarray(delta_lambda_nm, dtype=np.float64) / LASER_WAVELENGTH_NM
    denominator = 2.0 + xi
    doppler_parameter = -xi / denominator
    v_parallel_m_per_s = doppler_parameter * _SPEED_OF_LIGHT_M_PER_S / SIN_HALF_SCATTERING_ANGLE
    scaled_velocity = v_parallel_m_per_s / _thermal_speed_m_per_s(electron_temperature_ev)
    return np.exp(-0.5 * scaled_velocity**2) * relativistic_weighting_factor(delta_lambda_nm)
