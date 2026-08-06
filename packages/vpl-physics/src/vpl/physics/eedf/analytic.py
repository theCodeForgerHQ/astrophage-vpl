"""Analytic EEDF families — doc 03 §3.2, doc 05 §2.1.

Doc 03 §3.2 states the problem this package exists for:

    the EEDF is typically bi-Maxwellian or Druyvesteyn, and the high-energy tail — the
    part that matters for ionisation and for OES line ratios — is depleted.

Doc 05 §2.1 then makes the shape an **inferred** quantity: ``kappa``, uniform on [1, 5],
annotated "Maxwellian → Druyvesteyn". This module is that one-parameter family.

## The family

    f0(eps)  =  A exp( -(eps / eps_0)**kappa )

with ``A`` and ``eps_0`` fixed by the two constraints every EEDF in this package obeys::

    integral f0 sqrt(eps) deps      = 1
    integral eps**(3/2) f0 deps     = <eps>

which give, in closed form,

    eps_0 = <eps> Gamma(3/(2 kappa)) / Gamma(5/(2 kappa))
    A     = kappa / ( eps_0**(3/2) Gamma(3/(2 kappa)) )

``kappa = 1`` is the Maxwellian, ``kappa = 2`` the Druyvesteyn. Both are limits the
two-term solver of :mod:`vpl.physics.eedf.solver` reaches on its own from the collision
physics — a Maxwellian at vanishing field, a Druyvesteyn for a constant momentum-transfer
cross section and a cold gas — so this module is a *comparison*, not an input to it.

## Parameterised by mean energy, not by temperature

Deliberately. A non-Maxwellian distribution has no temperature; it has a mean energy, and
``T_e`` is a derived convenience defined as ``(2/3) <eps>``. Taking a temperature would
invite the reading that ``kappa = 2`` describes a Druyvesteyn "at 3 eV", which is exactly
the confusion doc 03 §3.2 is warning about: a Maxwellian fit to a depleted distribution
reports a temperature belonging to neither population (doc 04 §4.2, benchmark B-06).
:func:`maxwellian_eedf` is the one exception, because for a Maxwellian the temperature
*is* well defined.

## The kappa distribution — a second, unrelated "kappa"

:func:`kappa_eedf` is a second EEDF family, added so that doc 05 §7.1's "EEDF
parameterisation" model mismatch has more than one shape to exercise. Its ``kappa`` is
the Summers & Thorne (1991) / Pierrard & Lazar (2010) spectral index of a power-law tail::

    f0(eps)  =  A (1 + eps / ((kappa - 3/2) T_e))**(-(kappa + 1))

**This is a different parameter from the one above**, despite the shared symbol. The
family's own ``kappa`` is an exponent inside an ``exp``, sampled on ``[1, 5]`` with
``kappa = 1`` Maxwellian and ``kappa = 2`` Druyvesteyn. The kappa distribution's ``kappa``
is a power-law index with no upper limit and a **lower** validity bound of ``3/2``, below
which the mean energy does not exist. They coincide only in that doc 05 §2.1 happens to
give ``PlasmaParams.kappa`` the same ``[1, 5]`` prior, whose lower half — ``[1, 1.5]`` — is
below the kappa distribution's domain of validity. That is a real inconsistency between
the prior's support and this family's validity domain; resolving it (narrowing the prior,
using a different field, or something else) is a modelling decision for whoever wires
:class:`~vpl.instruments.oes.instrument.KappaEedf` to that parameter, not something this
module can silently paper over by clamping or widening a bound.

Unlike ``generalised_eedf``, the kappa distribution *is* well parameterised by a
temperature: the ``(kappa - 3/2)`` scaling is chosen precisely so that
``<eps> = 1.5 T_e`` for every valid ``kappa``, exactly as it is for a Maxwellian — and as
``kappa -> inf`` the distribution converges to the Maxwellian pointwise, which is the
property that makes it a genuine generalisation rather than an unrelated shape.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma as gamma_function
from scipy.special import gammaln as log_gamma_function

__all__ = [
    "DRUYVESTEYN_KAPPA",
    "KAPPA_DISTRIBUTION_MINIMUM_KAPPA",
    "MAXWELLIAN_KAPPA",
    "AnalyticEedf",
    "druyvesteyn_eedf",
    "generalised_eedf",
    "kappa_eedf",
    "maxwellian_eedf",
]

type FloatArray = NDArray[np.float64]

#: ``kappa = 1``: ``f0 ~ exp(-eps/T_e)``. The shape the Boltzmann-electron relation of
#: doc 03 §3.1 assumes, and the shape doc 03 §3.2 says the sheath does not have.
MAXWELLIAN_KAPPA: Final[float] = 1.0

#: ``kappa = 2``: ``f0 ~ exp(-(eps/eps_0)**2)``. The exact solution of the two-term
#: equation for a constant momentum-transfer cross section and a cold gas, and the shape
#: doc 03 §3.2 names as typical of a low-pressure discharge.
DRUYVESTEYN_KAPPA: Final[float] = 2.0

#: Ratio of the mean energy to the temperature of a Maxwellian, ``<eps> = (3/2) k T_e``.
_THREE_HALVES: Final[float] = 1.5

#: The kappa distribution's spectral index below which the mean energy diverges.
#:
#: ``<eps> = integral eps**(3/2) f0 deps`` is, up to a factor, the Beta-function integral
#: ``B(5/2, kappa - 3/2)``, which requires ``kappa - 3/2 > 0`` to converge (Summers & Thorne
#: 1991; Pierrard & Lazar 2010). At and below this bound there is no temperature for the
#: distribution to carry, so :func:`kappa_eedf` refuses to evaluate rather than return a
#: value with no physical mean energy behind it. Numerically equal to :data:`_THREE_HALVES`,
#: but named separately because the two play unrelated roles: one is "how the moment
#: identity for a Maxwellian is stated", the other is "where a different family's validity
#: domain ends".
KAPPA_DISTRIBUTION_MINIMUM_KAPPA: Final[float] = _THREE_HALVES

#: Order shift between the normalisation Beta integral, ``B(3/2, kappa - 1/2)``, and the
#: mean-energy Beta integral, ``B(5/2, kappa - 3/2)``, in :func:`kappa_eedf`. Algebra, not a
#: measurement: it is the ``1/2`` that separates ``sqrt(eps)`` from ``eps**(3/2)`` under the
#: same substitution ``eps = (kappa - 3/2) T_e x``.
_HALF: Final[float] = 0.5

#: Largest exponent evaluated before the result is taken as an underflow to zero.
#:
#: ``exp(-x)`` underflows to zero at ``x ~ 745``, and NumPy reports that as a RuntimeWarning
#: which pytest's ``filterwarnings = ["error"]`` turns into a failure. The underflow is the
#: *correct* answer — the tail of a kappa = 5 distribution really is zero to double
#: precision long before the grid ends — so it is produced deliberately rather than
#: allowed to happen and then suppressed.
_MAX_EXPONENT: Final[float] = 700.0


def _checked_energy(energy_ev: ArrayLike) -> FloatArray:
    grid = np.asarray(energy_ev, dtype=np.float64)
    if np.any(grid < 0.0):
        raise ValueError("an EEDF is not defined at negative energy")
    return grid


def _checked_shape(*, mean_energy_ev: float, kappa: float) -> None:
    if not kappa > 0.0:
        raise ValueError(
            f"kappa is the exponent of the energy in the exponential and must be "
            f"positive, got {kappa}. doc 05 §2.1 samples it on [1, 5]."
        )
    if not mean_energy_ev > 0.0:
        raise ValueError(f"the mean energy must be positive, got {mean_energy_ev}")


def generalised_eedf(energy_ev: ArrayLike, *, mean_energy_ev: float, kappa: float) -> FloatArray:
    """``f0(eps) = A exp(-(eps/eps_0)**kappa)``, normalised and at the given mean energy.

    The doc 05 §2.1 family in one function: ``kappa = 1`` Maxwellian, ``kappa = 2``
    Druyvesteyn, larger ``kappa`` progressively more depleted in the tail.

    Args:
        energy_ev: Energies in eV. Non-negative.
        mean_energy_ev: ``<eps> = integral eps**(3/2) f0 deps``, in eV. For a Maxwellian
            this is ``1.5 k T_e``.
        kappa: The shape exponent. See :data:`MAXWELLIAN_KAPPA`, :data:`DRUYVESTEYN_KAPPA`.

    Returns:
        ``f0`` on ``energy_ev``, satisfying ``integral f0 sqrt(eps) deps = 1`` and
        ``integral eps**(3/2) f0 deps = mean_energy_ev`` analytically. Underflows to
        exactly zero far out in the tail rather than warning.
    """
    _checked_shape(mean_energy_ev=mean_energy_ev, kappa=kappa)
    grid = _checked_energy(energy_ev)

    lower = float(gamma_function(_THREE_HALVES / kappa))
    scale_ev = mean_energy_ev * lower / float(gamma_function(2.5 / kappa))
    amplitude = kappa / (scale_ev**_THREE_HALVES * lower)

    exponent = (grid / scale_ev) ** kappa
    return np.asarray(
        np.where(
            exponent > _MAX_EXPONENT, 0.0, amplitude * np.exp(-np.minimum(exponent, _MAX_EXPONENT))
        )
    )


def maxwellian_eedf(energy_ev: ArrayLike, *, electron_temperature_ev: float) -> FloatArray:
    """``f0(eps) = 2 (pi T_e**3)**(-1/2) exp(-eps/T_e)`` — doc 03 §3.1's assumed shape.

    Args:
        energy_ev: Energies in eV.
        electron_temperature_ev: ``k T_e / e`` in eV. The mean energy is ``1.5`` times it.
    """
    return generalised_eedf(
        energy_ev,
        mean_energy_ev=_THREE_HALVES * electron_temperature_ev,
        kappa=MAXWELLIAN_KAPPA,
    )


def _checked_kappa_distribution_shape(*, electron_temperature_ev: float, kappa: float) -> None:
    if not electron_temperature_ev > 0.0:
        raise ValueError(f"T_e must be positive, got {electron_temperature_ev} eV")
    if not kappa > KAPPA_DISTRIBUTION_MINIMUM_KAPPA:
        raise ValueError(
            f"kappa must exceed {KAPPA_DISTRIBUTION_MINIMUM_KAPPA} for the kappa "
            f"distribution's mean energy to exist (Summers & Thorne 1991; Pierrard & "
            f"Lazar 2010), got {kappa}. Note doc 05 §2.1's uniform [1, 5] prior on "
            f"PlasmaParams.kappa extends below this bound — see the module docstring."
        )


def kappa_eedf(energy_ev: ArrayLike, *, electron_temperature_ev: float, kappa: float) -> FloatArray:
    """``f0(eps) = A (1 + eps / ((kappa - 3/2) T_e))**-(kappa + 1)`` — the kappa-distribution EEDF.

    The Summers & Thorne (1991) / Pierrard & Lazar (2010) convention: a power-law tail in
    place of :func:`maxwellian_eedf`'s exponential one, added so that doc 05 §7.1's "EEDF
    parameterisation" model mismatch has a genuinely different *shape* to exercise and not
    only a different exponent inside the same ``exp``. See the module docstring for how
    this ``kappa`` differs from :func:`generalised_eedf`'s.

    ``A`` is fixed analytically from the same two constraints every EEDF in this package
    obeys (``integral f0 sqrt(eps) deps = 1``, ``integral eps**(3/2) f0 deps = 1.5 T_e``),
    via the Beta-function identity ``integral_0^inf x**(a - 1) (1 + x)**-(a + b) dx =
    B(a, b)``. The result converges pointwise to :func:`maxwellian_eedf` as
    ``kappa -> inf`` — the one property that makes this a generalisation and not an
    unrelated shape sharing a symbol.

    Computed in log space (:func:`scipy.special.gammaln`, :func:`numpy.log1p`) rather than
    by evaluating ``Gamma(kappa + 1)`` directly, because that overflows float64 for
    ``kappa`` above about 170 — exactly the regime a "does this reach the Maxwellian
    limit" check needs to probe.

    Args:
        energy_ev: Energies in eV. Non-negative.
        electron_temperature_ev: ``k T_e / e`` in eV. The mean energy is ``1.5`` times it,
            for every valid ``kappa`` — see :data:`KAPPA_DISTRIBUTION_MINIMUM_KAPPA`.
        kappa: The spectral index. Must exceed :data:`KAPPA_DISTRIBUTION_MINIMUM_KAPPA`.

    Returns:
        ``f0`` on ``energy_ev``, satisfying ``integral f0 sqrt(eps) deps = 1`` and
        ``integral eps**(3/2) f0 deps = 1.5 electron_temperature_ev`` analytically.
        Underflows to exactly zero far out in the tail rather than warning, like
        :func:`generalised_eedf`.

    Raises:
        ValueError: If ``kappa <= 1.5``, ``electron_temperature_ev <= 0``, or an energy is
            negative.
    """
    _checked_kappa_distribution_shape(electron_temperature_ev=electron_temperature_ev, kappa=kappa)
    grid = _checked_energy(energy_ev)

    scale_ev = (kappa - _THREE_HALVES) * electron_temperature_ev
    log_beta = (
        log_gamma_function(_THREE_HALVES)
        + log_gamma_function(kappa - _HALF)
        - log_gamma_function(kappa + 1.0)
    )
    log_amplitude = -_THREE_HALVES * math.log(scale_ev) - log_beta

    log_tail = -(kappa + 1.0) * np.log1p(grid / scale_ev)
    log_f0 = log_amplitude + log_tail
    return np.asarray(
        np.where(log_f0 < -_MAX_EXPONENT, 0.0, np.exp(np.maximum(log_f0, -_MAX_EXPONENT)))
    )


def druyvesteyn_eedf(energy_ev: ArrayLike, *, mean_energy_ev: float) -> FloatArray:
    """``f0(eps) ~ exp(-(eps/eps_0)**2)`` — doc 03 §3.2's depleted-tail shape.

    The exact two-term solution for a constant momentum-transfer cross section and a cold
    gas; see :class:`~vpl.physics.eedf.solver.TwoTermSolver`, which reaches it from the
    collision physics rather than being told it.
    """
    return generalised_eedf(energy_ev, mean_energy_ev=mean_energy_ev, kappa=DRUYVESTEYN_KAPPA)


class AnalyticEedf(StrEnum):
    """The named members of the family, for manifests.

    A ``StrEnum`` because manifests are data (doc 08 §1 principle 4): ``eedf: druyvesteyn``
    round-trips into a checked member and back out for the provenance record, and the
    ablation matrix of doc 07 can sweep it.
    """

    MAXWELLIAN = "maxwellian"
    DRUYVESTEYN = "druyvesteyn"

    @property
    def kappa(self) -> float:
        """The doc 05 §2.1 shape parameter this name stands for."""
        return MAXWELLIAN_KAPPA if self is AnalyticEedf.MAXWELLIAN else DRUYVESTEYN_KAPPA

    def evaluate(self, energy_ev: ArrayLike, *, mean_energy_ev: float) -> FloatArray:
        """``f0`` for this shape at the given mean energy."""
        return generalised_eedf(energy_ev, mean_energy_ev=mean_energy_ev, kappa=self.kappa)
