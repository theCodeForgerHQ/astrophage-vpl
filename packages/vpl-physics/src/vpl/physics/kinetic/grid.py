"""The uniform PIC grid — doc 03 §4.3.

A separate grid type from :class:`vpl.core.state.SpatialGrid`, and deliberately so. That one
permits geometric stretching towards the wall, which is right for the fluid solver: a FEM
solve wants cells concentrated where the gradients are, and doc 03 §3.4 specifies exactly
that.

**PIC cannot use it.** doc 03 §4.3 makes ``dz <= lambda_D`` a hard constraint, and its reason
is numerical grid heating — an explicit electrostatic PIC on cells wider than a Debye length
transfers energy from the field to the particles without any physics doing it, and the
plasma silently heats until the Debye length grows to match the cell. A graded mesh makes
that constraint position-dependent and the failure regional, which is worse than uniform:
the code would heat only near the bulk boundary, where nobody is looking.

So the PIC grid is uniform, and the constraint is checked once against the whole domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax.numpy as jnp
from jax import Array

from vpl.physics.kinetic.precision import enable_double_precision

enable_double_precision()

__all__ = ["UniformGrid"]

#: Display-only unit conversions for __repr__. Not physics — a sheath is naturally quoted in
#: millimetres and a PIC cell in micrometres — but the literal-lint rule of doc 08 §5 does not
#: distinguish, and it is right not to: a bare 1e6 in a physics module is exactly what the
#: rule exists to stop, and exempting "but this one is only for printing" would gut it.
_M_PER_MM: float = 1e3
_M_PER_UM: float = 1e6


@dataclass(frozen=True)
class UniformGrid:
    """A uniform 1-D grid of nodes spanning ``[0, length_m]``.

    ``z = 0`` is the biased wall and ``z`` increases into the plasma, matching
    :class:`vpl.core.state.SpatialGrid` and doc 02 §2's sign convention. Getting this
    backwards flips the sign of every flux the project reports.

    Attributes:
        length_m: Domain length. doc 03 §4.4 uses 20 mm at RP-1, roughly 22 sheath
            thicknesses, so the bulk boundary is far enough away not to matter.
        n_cells: Number of cells. There are ``n_cells + 1`` nodes; fields and charge live
            on nodes and both walls are nodes.
    """

    length_m: float
    n_cells: int

    def __post_init__(self) -> None:
        if self.n_cells < 1:
            raise ValueError(f"a grid needs at least one cell; cells = {self.n_cells}")
        if not self.length_m > 0.0:
            raise ValueError(f"domain length must be positive, got {self.length_m}")

    @property
    def n_nodes(self) -> int:
        """``n_cells + 1``. Fields live here, and both boundaries are nodes."""
        return self.n_cells + 1

    @property
    def dz_m(self) -> float:
        """Cell width. The quantity doc 03 §4.3 bounds by the Debye length."""
        return self.length_m / self.n_cells

    @cached_property
    def nodes_m(self) -> Array:
        """Node positions, from exactly 0 to exactly ``length_m``.

        Built with ``linspace`` rather than ``arange * dz`` so the last node lands on the
        domain length to the bit. An accumulated rounding error there would put particles
        marginally outside the domain and trip the deposition guard at random.
        """
        return jnp.linspace(0.0, self.length_m, self.n_nodes, dtype=jnp.float64)

    def __repr__(self) -> str:
        return (
            f"UniformGrid({self.length_m * _M_PER_MM:g} mm, {self.n_cells} cells, "
            f"dz={self.dz_m * _M_PER_UM:g} um)"
        )
