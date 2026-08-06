"""The L1 mesh — doc 03 §3.4, and the FEniCSx interval built on it.

Doc 03 §3.4 states two mesh requirements:

    Mesh            | Graded, refined toward the wall | ``lambda_D`` at the wall is the
                    |                                 | controlling scale; uniform
                    |                                 | meshing wastes 90 % of the cells
    Mesh resolution | ``dz <= lambda_D / 4`` in the sheath | Resolve the Debye scale;
                    |                                 | verified by V-04

**Taken literally, together, they are unsatisfiable.** A single geometric progression
from the wall with the registered ``mesh.A.stretch`` of 1.02 starts at ``lambda_D / 4``
and reaches ``2.7 lambda_D / 4`` by the sheath edge, so the second requirement fails
exactly where it is stated to apply. The construction here is therefore two-zone —
uniform at the Debye scale through the sheath, geometric from there to the bulk boundary
— which satisfies both. That is a decision doc 03 §3.4 does not make, so
:func:`max_cell_size_within` exists to turn the ``dz <= lambda_D / 4`` claim into a
measurement on the mesh that was actually built rather than an assertion about the mesh
that was intended.

The saving is real and is what doc 03 §3.4 is buying: at RP-1 a uniform mesh at
``lambda_D / 4`` over the 20 mm domain of ``mesh.domain_length`` needs 1 535 cells, and
the two-zone mesh needs 260.

## Which ``lambda_D``

The largest density in the L1 domain, and therefore the smallest Debye length, is at the
bulk boundary. That is the binding scale and it is what the caller passes in. Inside the
sheath the local Debye length is longer, so a mesh sized on the boundary value resolves
the interior with room to spare — which is the right direction to be conservative in.
"""

from __future__ import annotations

from typing import Final, TypeAlias

import basix.ufl
import numpy as np
import ufl
from dolfinx import mesh as _mesh
from mpi4py import MPI
from numpy.typing import NDArray

from vpl.core.params import default_registry
from vpl.core.state import SpatialGrid
from vpl.core.units import Q_, Quantity, magnitude_in

__all__ = [
    "BULK_FACET",
    "DEFAULT_DOMAIN_LENGTH",
    "DEFAULT_MESH_STRETCH",
    "MIN_CELLS_PER_DEBYE",
    "WALL_FACET",
    "graded_sheath_grid",
    "interval_mesh",
    "max_cell_size_within",
    "wall_facet_tags",
]

#: A field sampled on a grid, in SI magnitudes.
FloatArray: TypeAlias = NDArray[np.float64]

_REGISTRY: Final = default_registry()

#: Cells per Debye length required by doc 03 §3.4 (``dz <= lambda_D / 4``).
#:
#: Not read from the registry, because the registry's ``mesh.A.cells_per_debye`` (2) and
#: ``mesh.B.cells_per_debye`` (3) are the doc 05 §7.1 *PIC* meshes, deliberately
#: mismatched with each other to avoid an inverse crime, and both are coarser than doc 03
#: §3.4 permits for L1. Silently reusing one of them would make L1 violate its own
#: stated resolution requirement while appearing to obey the registry.
MIN_CELLS_PER_DEBYE: Final[float] = 4.0

#: Geometric cell-growth ratio away from the sheath — ``mesh.A.stretch``.
DEFAULT_MESH_STRETCH: Final[float] = float(_REGISTRY.value_in("mesh.A.stretch", "dimensionless"))

#: Domain length along the sheath normal — ``mesh.domain_length``, 20 mm.
DEFAULT_DOMAIN_LENGTH: Final[Quantity] = _REGISTRY["mesh.domain_length"].quantity

#: The facet tag :func:`wall_facet_tags` puts on ``z = 0``.
WALL_FACET: Final[int] = 1

#: The facet tag it puts on the bulk boundary.
BULK_FACET: Final[int] = 2


def _positive(value: Quantity, *, name: str) -> float:
    metres = float(magnitude_in(value, "m"))
    if metres <= 0.0:
        raise ValueError(f"{name} must be a positive length, got {value}")
    return metres


def _geometric_cell_count(*, span: float, first: float, stretch: float) -> int:
    """Most geometric cells of ratio ``stretch`` starting at ``first`` that fit in ``span``.

    A **floor**, not a ceiling, and the difference is the join between the two zones.
    Taking the floor and then recomputing the exact first width makes the first outer
    cell at least as wide as the last sheath cell, so widths never shrink with distance
    from the wall. A ceiling would make it slightly *narrower*, which is not wrong by any
    accuracy measure but does put the mesh's smallest cell somewhere other than the wall
    — and every stability and resolution statement in doc 03 §3.4 and §4.3 is written
    about the wall.
    """
    return max(1, int(np.floor(np.log1p(span * (stretch - 1.0) / first) / np.log(stretch))))


def _geometric_widths(*, span: float, n_cells: int, stretch: float) -> FloatArray:
    """Cell widths in geometric progression summing exactly to ``span``."""
    first = span * (stretch - 1.0) / (stretch**n_cells - 1.0)
    return np.asarray(first * stretch ** np.arange(n_cells, dtype=np.float64), dtype=np.float64)


def _uniform_widths(*, span: float, target: float) -> FloatArray:
    """Equal cells no wider than ``target``, spanning ``span`` exactly."""
    n_cells = max(1, int(np.ceil(span / target)))
    return np.full(n_cells, span / n_cells, dtype=np.float64)


def graded_sheath_grid(
    *,
    debye_length: Quantity,
    sheath_thickness: Quantity,
    domain_length: Quantity,
    cells_per_debye: float = MIN_CELLS_PER_DEBYE,
    stretch: float = DEFAULT_MESH_STRETCH,
) -> SpatialGrid:
    """The two-zone graded mesh of the module docstring.

    Args:
        debye_length: The controlling ``lambda_D``, at the largest density in the domain.
        sheath_thickness: How far from the wall the Debye-resolved zone extends. Normally
            :func:`~vpl.physics.analytic.sheath.child_langmuir_thickness` with ``h_l``
            supplied, i.e. the ADR-007 self-consistent value.
        domain_length: Distance from the wall to the bulk boundary.
        cells_per_debye: Cells per Debye length inside the sheath zone. The default is
            doc 03 §3.4's ``lambda_D / 4``; V-04 halves ``dz`` by doubling it.
        stretch: Geometric growth ratio outside the sheath zone. ``1.0`` gives a uniform
            mesh, which is what the mesh-independence study compares against.

    Returns:
        A grid with the wall at ``z = 0`` (doc 02 §2), cells never shrinking with
        distance from the wall, and a far boundary at exactly ``domain_length``.

    Raises:
        ValueError: If the sheath does not fit inside the domain, if any length is not
            positive, if ``cells_per_debye`` is not positive, or if ``stretch`` is below
            one — a stretch below one refines *away* from the wall, which is the opposite
            of what doc 03 §3.4 asks for and would silently under-resolve the sheath.
    """
    lambda_D = _positive(debye_length, name="debye_length")
    sheath = _positive(sheath_thickness, name="sheath_thickness")
    span = _positive(domain_length, name="domain_length")

    if cells_per_debye <= 0.0:
        raise ValueError(f"cells_per_debye must be positive, got {cells_per_debye}")
    if stretch < 1.0:
        raise ValueError(
            f"stretch must be at least one, got {stretch}. Below one the mesh refines "
            "away from the wall; doc 03 §3.4 refines toward it."
        )
    if sheath >= span:
        raise ValueError(
            f"sheath_thickness {sheath_thickness} must be shorter than domain_length "
            f"{domain_length}; doc 01 R-SPAT-3 wants the domain to reach at least ten "
            "sheath thicknesses so the bulk boundary condition is applied away from the "
            "structure being resolved."
        )

    target = lambda_D / cells_per_debye
    if stretch == 1.0:
        # The uniform reference mesh the grading is measured against. Built in one zone
        # rather than two so that it is exactly uniform: a two-zone construction leaves a
        # step at the join, and a "uniform" mesh with a step in it is a poor control.
        widths = _uniform_widths(span=span, target=target)
    else:
        inner = _uniform_widths(span=sheath, target=target)
        n_outer = _geometric_cell_count(span=span - sheath, first=inner[0], stretch=stretch)
        outer = _geometric_widths(span=span - sheath, n_cells=n_outer, stretch=stretch)
        widths = np.concatenate([inner, outer])
    return SpatialGrid.from_quantity(Q_(np.concatenate([[0.0], np.cumsum(widths)]), "m"))


def max_cell_size_within(grid: SpatialGrid, *, extent: Quantity) -> Quantity:
    """The widest cell that starts inside ``[0, extent]``.

    The measurement behind doc 03 §3.4's ``dz <= lambda_D / 4``. A mean would accept a
    mesh that is locally illegal, which is the same reason
    :attr:`~vpl.core.state.SpatialGrid.min_dz` is a minimum and not a mean.
    """
    limit = _positive(extent, name="extent")
    inside = grid.z_m[:-1] < limit
    if not np.any(inside):
        raise ValueError(
            f"no cell starts inside {extent}; the grid's first cell is {grid.dz_m[0]:.4g} m wide"
        )
    return Q_(float(grid.dz_m[inside].max()), "m")


def interval_mesh(grid: SpatialGrid, *, comm: MPI.Comm | None = None) -> _mesh.Mesh:
    """The FEniCSx interval carrying exactly the vertices of ``grid``.

    Built from explicit coordinates rather than by ``create_interval`` so that the graded
    spacing survives: the solver and the :class:`~vpl.core.state.SpatialGrid` that
    reaches the artifact must agree on where the samples are, and doc 02 §2 records
    inter-channel registration error as among the most damaging and least modelled
    systematics in multi-diagnostic fusion.

    Args:
        grid: The vertices, wall first.
        comm: MPI communicator. Defaults to ``COMM_SELF``: doc 10 §6 runs a campaign as a
            local process pool over independent cases, so an L1 solve is one serial task
            and splitting its 10^3 degrees of freedom across ranks would cost more in
            communication than it saves.
    """
    communicator = MPI.COMM_SELF if comm is None else comm
    n_cells = grid.n_cells
    cells = np.column_stack(
        [np.arange(n_cells, dtype=np.int64), np.arange(1, n_cells + 1, dtype=np.int64)]
    )
    coordinates = grid.z_m.reshape(-1, 1).copy()
    element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(1,)))
    return _mesh.create_mesh(communicator, cells, coordinates, element)


def wall_facet_tags(domain: _mesh.Mesh, *, length: float) -> _mesh.MeshTags:
    """Tag the two ends: :data:`WALL_FACET` at ``z = 0``, :data:`BULK_FACET` at ``z = L``.

    The wall carries a genuine natural boundary condition — the outflow ion flux of doc
    03 §3.3, "ion flux absorbed" — which has to be integrated over that facet and only
    that facet. An untagged ``ds`` would integrate over both ends and quietly inject the
    outflow term at the inflow boundary as well.
    """
    domain.topology.create_connectivity(0, 1)
    wall = _mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], 0.0))
    bulk = _mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], length))

    indices = np.concatenate([wall, bulk]).astype(np.int32)
    values = np.concatenate(
        [np.full(wall.size, WALL_FACET), np.full(bulk.size, BULK_FACET)]
    ).astype(np.int32)
    order = np.argsort(indices)
    return _mesh.meshtags(domain, 0, indices[order], values[order])
