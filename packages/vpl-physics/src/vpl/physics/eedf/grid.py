"""The energy grid the two-term solve lives on — doc 03 §3.2.

A finite-volume grid in electron energy: ``n`` cells, ``n + 1`` boundaries, the first
boundary at exactly zero. ``f0`` is stored at cell centres and reconstructed as
piecewise constant, which is not an incidental choice — it is what makes the quadrature
below *exact* rather than merely accurate.

## Why the quadrature has to be exact and not just good

Every quantity doc 03 §3.2 tabulates is a moment of ``f0``:

    normalisation   integral f0 sqrt(eps) deps            = 1
    mean energy     integral eps**(3/2) f0 deps           = <eps>
    rate            gamma integral eps sigma(eps) f0 deps = k

and the solver enforces particle conservation cell by cell against the same
reconstruction. If :meth:`EnergyGrid.moment` used a quadrature the solver did not, the
normalisation would hold only to the quadrature error and *every* rate coefficient would
inherit that error as a multiplicative bias — a bias that no refinement study would
attribute to the quadrature, because it looks exactly like a converged answer.

So the cell weights are the analytic integrals ``integral_cell eps**(k + 1/2) deps``, and
for a piecewise-constant ``f0`` the moment is exact to machine precision.

## Two grid families

Linear is the default and is what makes the exponential scheme's second-order accuracy easy
to see: on a uniform grid the cell boundary is *exactly* the midpoint between the two
adjacent cell centres, so the boundary-evaluated flux coefficients form a midpoint rule.

Quadratic (``b_j ~ j**2``) concentrates cells where a low-temperature EEDF actually lives.
It gives up the exact-midpoint property, and the obvious expectation is that it gives up an
order of accuracy with it. **Measured against the Druyvesteyn closed form it does not**: on
the same problem at 100/200/400/800 cells the observed orders are 1.9975 (linear) and
2.0000 (quadratic), and the quadratic grid is in fact slightly the more accurate of the two.
The offset between a boundary and its neighbours' midpoint turns out to be a constant
``eps_max / (2 n**2)`` rather than a fixed fraction of the local cell width, so it shrinks
faster than the truncation error it would have polluted. Linear remains the default because
the second-order argument for it is a proof and the one for quadratic is an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["EnergyGrid"]

#: A quantity sampled on the grid, in SI magnitudes except energies, which are eV.
FloatArray: TypeAlias = NDArray[np.float64]

#: Exponent of the geometric weight in :meth:`EnergyGrid.quadratic`.
_QUADRATIC_EXPONENT: Final[float] = 2.0

#: How closely cell widths must agree before a grid counts as uniform. Not a physical
#: tolerance: the widths of a ``linspace`` differ in their last bits, and this only has to
#: separate that from a deliberately graded grid, whose widths differ by orders of
#: magnitude. It decides which family :meth:`EnergyGrid.refined` rebuilds in.
_UNIFORMITY_TOLERANCE: Final[float] = 1e-12


def _locked(values: ArrayLike, *, what: str) -> FloatArray:
    """Copy, check finite, and lock against writes.

    The same treatment :class:`~vpl.physics.atomic.lxcat.CrossSection` gives a table, for
    the same reason: a grid is built once and read by every integral in a sweep, so an
    in-place edit anywhere would silently change results everywhere else.
    """
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} must be finite; found nan or inf")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, eq=False)
class EnergyGrid:
    """Cell boundaries in eV, from zero upwards.

    Attributes:
        boundaries_ev: ``n + 1`` strictly increasing energies starting at exactly zero.
            Read-only. Zero is not negotiable: the two-term equation's lower boundary
            condition is zero energy-space flux at ``eps = 0``, and a grid that began
            above zero would silently discard the population below its first boundary.
    """

    boundaries_ev: FloatArray

    def __post_init__(self) -> None:
        boundaries = _locked(self.boundaries_ev, what="cell boundaries")
        if boundaries.size < 2:
            raise ValueError(
                f"an energy grid needs at least one cell, i.e. two boundaries, got "
                f"{boundaries.size}"
            )
        if boundaries[0] != 0.0:
            raise ValueError(
                f"the first cell boundary must be exactly zero energy, got {boundaries[0]}. "
                "The two-term equation's lower boundary condition is zero flux at eps = 0."
            )
        if not np.all(np.diff(boundaries) > 0.0):
            raise ValueError("cell boundaries must be strictly increasing")
        object.__setattr__(self, "boundaries_ev", boundaries)

    # ── the two families ────────────────────────────────────────────────────────

    @classmethod
    def linear(cls, *, max_ev: float, n_cells: int) -> Self:
        """Uniform cells over ``[0, max_ev]``.

        The default. On a uniform grid every cell boundary is the exact midpoint of the
        two adjacent cell centres, which makes the boundary-evaluated flux coefficients
        of the exponential scheme a midpoint rule and the whole discretisation second
        order.
        """
        return cls(
            boundaries_ev=np.linspace(0.0, _positive_max(max_ev), _checked_cells(n_cells) + 1)
        )

    @classmethod
    def quadratic(cls, *, max_ev: float, n_cells: int) -> Self:
        """Cells growing as ``j**2``, so the low-energy bulk is resolved finely.

        Useful when ``<eps>`` is a small fraction of the ionisation threshold and a
        uniform grid would spend most of its cells on a tail that has underflowed. It is
        not the default because it gives up the exact-midpoint property above.
        """
        count = _checked_cells(n_cells)
        fraction = np.arange(count + 1, dtype=np.float64) / count
        return cls(boundaries_ev=_positive_max(max_ev) * fraction**_QUADRATIC_EXPONENT)

    def refined(self, factor: int) -> Self:
        """The same grid family with ``factor`` times as many cells.

        For the mesh-refinement studies of doc 07 §2.3. Refining by subdividing the
        existing boundaries would preserve the family only for a linear grid; refining by
        rebuilding from ``max_ev`` and ``n_cells`` preserves it for both, which is what a
        convergence study needs — otherwise the study measures the grid changing family
        rather than the discretisation converging.
        """
        if factor < 1:
            raise ValueError(f"a refinement factor must be at least 1, got {factor}")
        if self._is_uniform():
            return type(self).linear(max_ev=self.max_ev, n_cells=self.n_cells * factor)
        return type(self).quadratic(max_ev=self.max_ev, n_cells=self.n_cells * factor)

    def _is_uniform(self) -> bool:
        widths = self.widths_ev
        return bool(np.allclose(widths, widths[0], rtol=_UNIFORMITY_TOLERANCE, atol=0.0))

    # ── geometry ────────────────────────────────────────────────────────────────

    @property
    def n_cells(self) -> int:
        return int(self.boundaries_ev.size - 1)

    @property
    def max_ev(self) -> float:
        """The top of the grid. Also the energy at which ``f0`` is truncated."""
        return float(self.boundaries_ev[-1])

    @property
    def centres_ev(self) -> FloatArray:
        """Cell centres — where ``f0`` and the cross sections are sampled."""
        return np.asarray(0.5 * (self.boundaries_ev[:-1] + self.boundaries_ev[1:]))

    @property
    def widths_ev(self) -> FloatArray:
        """Cell widths ``d_j``. The measure of ``deps`` integrals that carry no ``sqrt(eps)``."""
        return np.asarray(np.diff(self.boundaries_ev))

    @property
    def centre_spacing_ev(self) -> FloatArray:
        """Distances between adjacent cell centres, one per interior boundary."""
        return np.asarray(np.diff(self.centres_ev))

    @property
    def cell_masses(self) -> FloatArray:
        """``integral_cell sqrt(eps) deps`` — the measure the normalisation is taken in.

        Named "mass" because that is its role: ``m_j f0_j`` is the number of electrons in
        cell ``j``, and the solver's conservation statement is written in these units.
        """
        return self._weights(0)

    # ── quadrature ──────────────────────────────────────────────────────────────

    def _weights(self, order: int) -> FloatArray:
        """``integral_cell eps**(order + 1/2) deps``, analytically."""
        power = order + 1.5
        return np.asarray(
            (self.boundaries_ev[1:] ** power - self.boundaries_ev[:-1] ** power) / power
        )

    def _checked(self, f0: ArrayLike) -> FloatArray:
        values = np.asarray(f0, dtype=np.float64)
        if values.shape != (self.n_cells,):
            raise ValueError(
                f"a distribution must carry one value per cell: expected shape "
                f"({self.n_cells},), got {values.shape}"
            )
        return values

    def moment(self, f0: ArrayLike, order: int) -> float:
        """``integral eps**order f0(eps) sqrt(eps) deps``, exact for piecewise-constant ``f0``.

        Args:
            f0: One value per cell.
            order: ``0`` for the normalisation, ``1`` for the mean energy.

        Returns:
            The moment. ``moment(f0, 1) / moment(f0, 0)`` is ``<eps>``.
        """
        return float(np.dot(self._checked(f0), self._weights(order)))

    def normalise(self, f0: ArrayLike) -> FloatArray:
        """``f0`` rescaled so that :meth:`moment` of order zero is exactly one.

        Raises:
            ValueError: If the distribution has zero or negative norm. A vanishing ``f0``
                cannot be normalised, and returning nan would put an EEDF-shaped array of
                nans into a rate integral, where it would surface as a nan rate
                coefficient several modules later.
        """
        values = self._checked(f0)
        norm = float(np.dot(values, self._weights(0)))
        if norm <= 0.0:
            raise ValueError(
                f"cannot normalise a distribution whose integral is {norm}; it must be "
                "strictly positive. A zero norm means the distribution has vanished."
            )
        return np.asarray(values / norm)

    def __len__(self) -> int:
        return self.n_cells

    def __repr__(self) -> str:
        family = "linear" if self._is_uniform() else "graded"
        return f"EnergyGrid({family}, {self.n_cells} cells, 0 to {self.max_ev:g} eV)"


def _checked_cells(n_cells: int) -> int:
    if n_cells < 1:
        raise ValueError(f"an energy grid needs at least one cell, got {n_cells}")
    return n_cells


def _positive_max(max_ev: float) -> float:
    if not max_ev > 0.0:
        raise ValueError(f"the top of the energy grid must be a positive energy, got {max_ev}")
    return max_ev
