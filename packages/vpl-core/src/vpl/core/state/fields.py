"""Scalar fields and velocity distributions — the payload of a plasma state.

doc 03 §5.2 names the set the framework actually needs: ``n_e(z)``, ``n_i(z)``,
``Phi(z)``, ``f_i(v_z; z=0)`` and ``T_e(z)``. Everything an instrument's forward model
reads is one of these, and everything a solver produces is one of these.

## The sign convention, again

doc 02 §2 fixes it: the wall is at ``z = 0``, ``z`` is positive *into* the plasma, and
flux toward the wall is positive. An ion arriving at the wall therefore has ``v_z < 0``.
Doc 02 §2 calls sign errors in flux quantities "common, silent and catastrophic", so the
flux moment carries the convention in its name — ``particle_flux_toward_wall`` — rather
than leaving an inline minus sign somewhere downstream for a reader to find.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pint
from numpy.typing import NDArray

from vpl.core.state.grid import SpatialGrid, TimeGrid
from vpl.core.state.species import Species
from vpl.core.units import Q_, UREG, ArrayQuantity, DimensionalityError

__all__ = ["ScalarField", "VelocityDistribution"]


def _frozen_values(values: NDArray[np.float64], *, what: str) -> NDArray[np.float64]:
    """Copy, check finiteness, and lock against writes."""
    array = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} must be finite; found nan or inf")
    array.flags.writeable = False
    return array


def _validated_units(units: str) -> str:
    """Reject units pint cannot parse, at construction rather than at first use."""
    try:
        UREG.Unit(units)
    except (pint.UndefinedUnitError, TypeError, ValueError) as exc:
        raise DimensionalityError(f"unrecognised units {units!r}: {exc}") from exc
    return units


@dataclass(frozen=True, eq=False, slots=True)
class ScalarField:
    """A named scalar quantity sampled on a spatial grid, optionally over time.

    Time-dependent values are indexed **time first**: ``(n_t, n_z)``. Fixing that
    ordering once, here, matters because a transposed field is shape-compatible and
    therefore silent whenever ``n_t`` happens to equal ``n_z``.

    Attributes:
        name: The symbol as it appears in the specification, e.g. ``"n_e"``, ``"Phi"``.
        values: ``(n_z,)`` for a steady field, ``(n_t, n_z)`` for a time-dependent one.
        units: pint-parseable units of ``values``.
        grid: The spatial grid the last axis indexes.
        time: The time grid the first axis indexes, or ``None`` for a steady field.
    """

    name: str
    values: NDArray[np.float64]
    units: str
    grid: SpatialGrid
    time: TimeGrid | None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a field needs a name; it appears in every artifact")

        object.__setattr__(self, "units", _validated_units(self.units))
        object.__setattr__(self, "values", _frozen_values(self.values, what=f"field {self.name!r}"))

        expected = (
            (self.grid.n_points,) if self.time is None else (self.time.n_points, self.grid.n_points)
        )
        if self.values.shape != expected:
            raise ValueError(
                f"field {self.name!r} has shape {self.values.shape}, expected {expected}. "
                "Time-dependent fields are indexed (n_t, n_z), time first."
            )

    @property
    def quantity(self) -> ArrayQuantity:
        """Values as a dimensional quantity, for module boundaries (doc 08 §5)."""
        return Q_(self.values, self.units)

    @property
    def is_steady(self) -> bool:
        return self.time is None

    def at_time_index(self, index: int) -> ScalarField:
        """The steady field at one instant."""
        if self.time is None:
            raise ValueError(f"field {self.name!r} is steady; there is no time axis to index")
        return ScalarField(
            name=self.name,
            values=self.values[index, :],
            units=self.units,
            grid=self.grid,
            time=None,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScalarField):
            return NotImplemented
        return (
            self.name == other.name
            and self.units == other.units
            and self.grid == other.grid
            and self.time == other.time
            and np.array_equal(self.values, other.values)
        )

    def __repr__(self) -> str:
        when = "steady" if self.time is None else f"{self.time.n_points} times"
        return f"ScalarField({self.name!r}, {self.units!r}, {when}, shape={self.values.shape})"


@dataclass(frozen=True, eq=False, slots=True)
class VelocityDistribution:
    """A one-dimensional distribution ``f(z, v_z)`` in the sheath-normal velocity.

    This is the marginal in ``v_z`` — the projection LIF actually measures (doc 04 §3.2)
    and the quantity doc 03 §5.2 emulates. It deliberately does **not** carry the
    perpendicular components: the full energy flux of doc 03 §6 weights by ``|v|^2``, and
    reconstructing that from a ``v_z`` marginal requires a physical assumption about the
    perpendicular population. That assumption belongs in ``vpl-physics`` where it can be
    stated, swept and budgeted — not baked silently into the data model.

    Attributes:
        grid: Positions along the sheath normal.
        v_m_per_s: Velocity axis, strictly increasing, in m/s.
        values: ``(n_z, n_v)`` phase-space density such that ``int f dv_z = n``.
        species: Whose distribution this is.
    """

    grid: SpatialGrid
    v_m_per_s: NDArray[np.float64]
    values: NDArray[np.float64]
    species: Species

    def __post_init__(self) -> None:
        axis = np.array(self.v_m_per_s, dtype=np.float64, copy=True)
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(
                f"velocity axis must be one-dimensional with >= 2 points, got {axis.shape}"
            )
        if not np.all(np.isfinite(axis)):
            raise ValueError("velocity axis must be finite")
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError("velocity axis must be strictly increasing")
        axis.flags.writeable = False
        object.__setattr__(self, "v_m_per_s", axis)

        values = _frozen_values(self.values, what=f"{self.species.name} distribution")
        expected = (self.grid.n_points, axis.size)
        if values.shape != expected:
            raise ValueError(
                f"distribution has shape {values.shape}, expected {expected} (n_z, n_v)"
            )
        if np.any(values < 0.0):
            # A distribution function is a phase-space density. A negative entry is not
            # a small numerical wobble to tolerate: whatever produced it is wrong, and
            # every moment taken afterwards inherits the error without showing it.
            raise ValueError(
                f"{self.species.name} distribution has negative values; "
                "a phase-space density cannot be negative"
            )
        object.__setattr__(self, "values", values)

    @property
    def v(self) -> ArrayQuantity:
        """Velocity axis as a dimensional quantity."""
        return Q_(self.v_m_per_s, "m/s")

    @property
    def n_velocities(self) -> int:
        return int(self.v_m_per_s.size)

    def at_wall(self) -> NDArray[np.float64]:
        """The distribution at ``z = 0``.

        doc 03 §5.2 emulates ``f_i(v_z; z=0)`` specifically, and doc 01 §1.1 makes the
        IEDF at the surface the deliverable. Reaching it should not require the caller
        to remember whether the wall is the first or the last grid point.
        """
        return self.values[0, :]

    def _moment(self, weight: NDArray[np.float64]) -> NDArray[np.float64]:
        """``int weight(v) f(z, v) dv`` for every ``z``."""
        integrand = self.values * weight[None, :]
        return np.asarray(np.trapezoid(integrand, self.v_m_per_s, axis=1), dtype=np.float64)

    def density_per_m3(self) -> NDArray[np.float64]:
        """Zeroth moment: ``int f dv_z``."""
        return self._moment(np.ones_like(self.v_m_per_s))

    def mean_velocity_m_per_s(self) -> NDArray[np.float64]:
        """First moment normalised by density.

        A zero-density cell is physical — nothing has arrived there yet in a transient —
        so the ratio is guarded. A NaN escaping from 0/0 would poison every downstream
        aggregate without announcing itself.
        """
        density = self.density_per_m3()
        momentum = self._moment(self.v_m_per_s)
        return np.divide(momentum, density, out=np.zeros_like(momentum), where=density > 0.0)

    def particle_flux_toward_wall_per_m2_s(self) -> NDArray[np.float64]:
        """``Gamma_i`` with the doc 02 §2 sign: positive means flowing toward the wall.

        The wall is at ``z = 0`` and ``z`` is positive into the plasma, so an ion that
        reaches the wall has ``v_z < 0``. The minus sign lives here, in a named method,
        because doc 02 §2 records that sign errors in flux quantities are common, silent
        and catastrophic — and an inline negation three modules downstream is exactly how
        that happens.
        """
        return -self._moment(self.v_m_per_s)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VelocityDistribution):
            return NotImplemented
        return (
            self.species == other.species
            and self.grid == other.grid
            and np.array_equal(self.v_m_per_s, other.v_m_per_s)
            and np.array_equal(self.values, other.values)
        )

    def __repr__(self) -> str:
        return (
            f"VelocityDistribution({self.species.name}, "
            f"n_z={self.grid.n_points}, n_v={self.n_velocities})"
        )
