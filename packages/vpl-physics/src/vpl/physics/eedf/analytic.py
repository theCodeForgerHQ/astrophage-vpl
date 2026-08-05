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
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma as gamma_function

__all__ = [
    "DRUYVESTEYN_KAPPA",
    "MAXWELLIAN_KAPPA",
    "AnalyticEedf",
    "druyvesteyn_eedf",
    "generalised_eedf",
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
