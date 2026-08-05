"""Coordinate grids — doc 02 §2 (frame) and doc 03 §3.4 (meshes).

The spatial frame is normative and every module uses it: the wall is at ``z = 0`` and
``z`` is positive *into* the plasma. Doc 02 §2 records that inter-channel registration
error is among the most damaging and least modelled systematics in multi-diagnostic
fusion, so a grid that does not obey the frame is rejected at construction rather than
producing a plausible answer about the wrong place.

Grids store SI magnitudes and expose both forms: :attr:`SpatialGrid.z` for module
boundaries and :attr:`SpatialGrid.z_m` for hot loops. That is the doc 08 §5 contract
made concrete — the conversion happens once, at construction, under assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from vpl.core.units import Q_, ArrayQuantity, Quantity, ScalarQuantity, magnitude_in

__all__ = ["PhaseGrid", "SpatialGrid", "TimeGrid"]


def _frozen_axis(
    values: NDArray[np.float64],
    *,
    name: str,
    reject_negative: bool,
) -> NDArray[np.float64]:
    """Validate a coordinate axis and return an immutable copy of it.

    Copying is not defensive pedantry. The caller's array is theirs to reuse, and a grid
    that aliases it would silently change shape underneath a solver that had already
    cached its spacings.
    """
    axis = np.array(values, dtype=np.float64, copy=True)

    if axis.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {axis.shape}")
    if axis.size < 2:
        raise ValueError(f"{name} needs at least two points, got {axis.size}")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be finite; found nan or inf")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if reject_negative and axis[0] < 0.0:
        raise ValueError(
            f"{name} starts at {axis[0]}, which is behind the wall. "
            "doc 02 §2 places the wall at z = 0 with z positive into the plasma."
        )

    axis.flags.writeable = False
    return axis


def _geometric_widths(*, total: float, n_cells: int, stretch: float) -> NDArray[np.float64]:
    """Cell widths in geometric progression summing exactly to ``total``."""
    if stretch <= 0.0:
        raise ValueError(f"stretch must be positive, got {stretch}")

    powers = np.arange(n_cells, dtype=np.float64)
    if stretch == 1.0:
        # The degenerate case: the closed form below divides by zero here.
        return np.full(n_cells, total / n_cells)

    first = total * (stretch - 1.0) / (stretch**n_cells - 1.0)
    return first * stretch**powers


@dataclass(frozen=True, eq=False, slots=True)
class SpatialGrid:
    """Points along the sheath normal, wall-first.

    Attributes:
        z_m: Coordinates in metres, ascending from the wall. Read-only.
    """

    z_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "z_m", _frozen_axis(self.z_m, name="z", reject_negative=True))

    # ── constructors ────────────────────────────────────────────────────────────

    @classmethod
    def from_quantity(cls, z: Quantity) -> SpatialGrid:
        """Build from an explicit coordinate array carrying length units."""
        return cls(z_m=np.atleast_1d(np.asarray(magnitude_in(z, "m"), dtype=np.float64)))

    @classmethod
    def uniform(cls, *, length: Quantity, n_points: int) -> SpatialGrid:
        """Evenly spaced points from the wall to ``length``."""
        span = magnitude_in(length, "m")
        return cls(z_m=np.linspace(0.0, float(span), n_points))

    @classmethod
    def geometric(cls, *, length: Quantity, n_points: int, stretch: float) -> SpatialGrid:
        """Points refined toward the wall, cell widths in geometric progression.

        doc 03 §3.4: the Debye length at the wall is the controlling scale, so a uniform
        mesh spends roughly 90% of its cells resolving bulk plasma that does not need
        them. ``stretch`` is the ratio of each cell to its predecessor; ``1.0`` reduces
        to :meth:`uniform`.
        """
        span = float(magnitude_in(length, "m"))
        widths = _geometric_widths(total=span, n_cells=n_points - 1, stretch=stretch)
        return cls(z_m=np.concatenate([[0.0], np.cumsum(widths)]))

    # ── views ───────────────────────────────────────────────────────────────────

    @property
    def z(self) -> ArrayQuantity:
        """Coordinates as a dimensional quantity, for module boundaries."""
        return Q_(self.z_m, "m")

    @property
    def dz_m(self) -> NDArray[np.float64]:
        """Cell widths in metres; length ``n_cells``."""
        return np.diff(self.z_m)

    @property
    def min_dz(self) -> ScalarQuantity:
        """The smallest cell.

        The stability constraints of doc 03 §4.3 bind on the smallest cell, not the mean.
        A mean-based check accepts a mesh that is locally illegal.
        """
        return Q_(float(self.dz_m.min()), "m")

    @property
    def length(self) -> ScalarQuantity:
        """Distance from the wall to the far boundary."""
        return Q_(float(self.z_m[-1] - self.z_m[0]), "m")

    @property
    def n_points(self) -> int:
        return int(self.z_m.size)

    @property
    def n_cells(self) -> int:
        return int(self.z_m.size - 1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpatialGrid):
            return NotImplemented
        return np.array_equal(self.z_m, other.z_m)

    def __repr__(self) -> str:
        return (
            f"SpatialGrid(n_points={self.n_points}, "
            f"length={self.z_m[-1] - self.z_m[0]:.4g} m, "
            f"min_dz={self.dz_m.min():.4g} m)"
        )


@dataclass(frozen=True, eq=False, slots=True)
class TimeGrid:
    """Instants on the shared latent time base of doc 05 §3.2.

    Unlike ``z``, ``t = 0`` carries no privileged meaning — a pre-trigger baseline window
    is a legitimate acquisition (doc 02 §10.2) — so negative times are allowed.

    Attributes:
        t_s: Instants in seconds, ascending. Read-only.
    """

    t_s: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_s", _frozen_axis(self.t_s, name="t", reject_negative=False))

    @classmethod
    def from_quantity(cls, t: Quantity) -> TimeGrid:
        return cls(t_s=np.atleast_1d(np.asarray(magnitude_in(t, "s"), dtype=np.float64)))

    @classmethod
    def uniform(cls, *, duration: Quantity, n_points: int) -> TimeGrid:
        span = magnitude_in(duration, "s")
        return cls(t_s=np.linspace(0.0, float(span), n_points))

    @property
    def t(self) -> ArrayQuantity:
        return Q_(self.t_s, "s")

    @property
    def dt_s(self) -> NDArray[np.float64]:
        return np.diff(self.t_s)

    @property
    def duration(self) -> ScalarQuantity:
        return Q_(float(self.t_s[-1] - self.t_s[0]), "s")

    @property
    def n_points(self) -> int:
        return int(self.t_s.size)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeGrid):
            return NotImplemented
        return np.array_equal(self.t_s, other.t_s)

    def __repr__(self) -> str:
        return f"TimeGrid(n_points={self.n_points}, duration={self.t_s[-1] - self.t_s[0]:.4g} s)"


@dataclass(frozen=True, slots=True)
class PhaseGrid:
    """RF phase binning for cyclo-stationary accumulation — doc 02 §10.3.

    Steady-state RF operation is treated as periodic in phase, so observations at the
    same phase across different cycles may be accumulated. That is what makes Thomson
    and LIF possible at all in the RF regime.

    Attributes:
        n_bins: Number of equal phase bins across one period.
        period: The RF period being divided.
    """

    n_bins: int
    period: Quantity

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"a phase grid needs at least one bin, got {self.n_bins}")
        if magnitude_in(self.period, "s") <= 0.0:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def bin_width(self) -> ScalarQuantity:
        """Duration of one phase bin.

        doc 02 §10.3 requires this to stay compatible with the 2 ns minimum ICCD gate;
        the check belongs to the instrument, which knows its own gate.
        """
        # pint's division is typed as returning Any; the operands make the result a time.
        return cast("ScalarQuantity", self.period / self.n_bins)

    @property
    def bin_width_rad(self) -> float:
        return 2.0 * np.pi / self.n_bins

    @property
    def centres_rad(self) -> NDArray[np.float64]:
        """Phase at the centre of each bin, in radians."""
        return (np.arange(self.n_bins, dtype=np.float64) + 0.5) * self.bin_width_rad

    @property
    def edges_rad(self) -> NDArray[np.float64]:
        """Phase at each bin boundary; length ``n_bins + 1``."""
        return np.arange(self.n_bins + 1, dtype=np.float64) * self.bin_width_rad

    def bin_index(self, phase_rad: float) -> int:
        """Bin containing ``phase_rad``, wrapping into ``[0, 2*pi)``.

        Wrapping is the whole point of cyclo-stationarity: a phase one full cycle on is
        the same phase. An accumulation that failed to wrap would smear the
        phase-resolved result it exists to produce.
        """
        wrapped = float(np.mod(phase_rad, 2.0 * np.pi))
        return int(np.floor(wrapped / self.bin_width_rad)) % self.n_bins

    def __repr__(self) -> str:
        return f"PhaseGrid(n_bins={self.n_bins}, period={self.period:.4g~P})"
