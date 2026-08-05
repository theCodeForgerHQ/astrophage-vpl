"""The graded L1 mesh — doc 03 §3.4.

Doc 03 §3.4 states two mesh requirements and they pull against each other:

- *Mesh*: "graded, refined toward the wall", because "``lambda_D`` at the wall is the
  controlling scale; uniform meshing wastes 90 % of the cells".
- *Mesh resolution*: "``dz <= lambda_D / 4`` in the sheath", to "resolve the Debye scale;
  verified by mesh-refinement study V-04".

A purely geometric mesh honours the first and violates the second: starting at
``lambda_D / 4`` at the wall with the registered ``mesh.A.stretch`` of 1.02, the cell at
the sheath edge has already grown to roughly ``2.7 lambda_D / 4``. The construction
tested here is therefore two-zone — uniform at the Debye scale through the sheath, then
geometric out to the bulk boundary — which is the only shape that satisfies both
statements at once.

That is a design decision doc 03 §3.4 does not spell out, so it is tested rather than
assumed: :func:`max_cell_size_within` exists so the ``dz <= lambda_D / 4`` claim is a
measurement on the mesh that was actually built.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from vpl.core.state import SpatialGrid
from vpl.core.units import Q_, magnitude_in
from vpl.physics.fluid import (
    DEFAULT_MESH_STRETCH,
    MIN_CELLS_PER_DEBYE,
    graded_sheath_grid,
    interval_mesh,
    max_cell_size_within,
)

#: RP-1 scales — doc 01 §2.2 and ADR-007. ``lambda_D`` at the sheath-edge density and the
#: self-consistent Child-Langmuir thickness that goes with it.
LAMBDA_D = Q_(52.13e-6, "m")
SHEATH = Q_(1.140e-3, "m")
DOMAIN = Q_(20.0, "mm")


def _grid(**overrides: object) -> SpatialGrid:
    defaults: dict[str, object] = {
        "debye_length": LAMBDA_D,
        "sheath_thickness": SHEATH,
        "domain_length": DOMAIN,
    }
    return graded_sheath_grid(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestDebyeResolution:
    """The ``dz <= lambda_D / 4`` half of doc 03 §3.4."""

    @pytest.mark.physics
    def test_every_cell_in_the_sheath_resolves_a_quarter_debye_length(self) -> None:
        grid = _grid()
        largest = magnitude_in(max_cell_size_within(grid, extent=SHEATH), "m")
        assert float(largest) <= float(magnitude_in(LAMBDA_D, "m")) / MIN_CELLS_PER_DEBYE * (
            1.0 + 1e-12
        )

    @pytest.mark.physics
    def test_a_geometric_mesh_alone_would_violate_it(self) -> None:
        """The reason the construction is two-zone, stated as a measurement.

        doc 03 §3.4's "graded, refined toward the wall" taken literally — a single
        geometric progression from the wall — puts a cell of roughly 2.7 Debye quarters
        at the sheath edge. Recording it here means the two-zone choice is visible rather
        than being an unexplained departure from the document.
        """
        first = float(magnitude_in(LAMBDA_D, "m")) / MIN_CELLS_PER_DEBYE
        span = float(magnitude_in(DOMAIN, "m"))
        n_cells = int(
            np.ceil(
                np.log1p(span * (DEFAULT_MESH_STRETCH - 1.0) / first) / np.log(DEFAULT_MESH_STRETCH)
            )
        )
        geometric = SpatialGrid.geometric(
            length=DOMAIN, n_points=n_cells + 1, stretch=DEFAULT_MESH_STRETCH
        )
        largest = float(magnitude_in(max_cell_size_within(geometric, extent=SHEATH), "m"))
        assert largest > 2.0 * first

    def test_halving_the_cell_size_halves_the_smallest_cell(self) -> None:
        """The refinement knob V-04 turns.

        Not to machine precision: the sheath zone holds a whole number of cells, so
        doubling ``cells_per_debye`` doubles the count up to one cell of rounding. At the
        default that is 88 cells against 175, a 0.6 % departure from an exact halving —
        which is a property of the mesh and not something to be tuned away.
        """
        coarse = _grid(cells_per_debye=MIN_CELLS_PER_DEBYE)
        fine = _grid(cells_per_debye=2.0 * MIN_CELLS_PER_DEBYE)
        assert float(magnitude_in(fine.min_dz, "m")) == pytest.approx(
            float(magnitude_in(coarse.min_dz, "m")) / 2.0, rel=0.02
        )


class TestGrading:
    """The "refined toward the wall" half."""

    def test_the_wall_is_at_z_equals_zero(self) -> None:
        assert _grid().z_m[0] == 0.0

    def test_the_far_boundary_is_exactly_the_domain_length(self) -> None:
        grid = _grid()
        assert float(magnitude_in(grid.length, "m")) == pytest.approx(
            float(magnitude_in(DOMAIN, "m")), rel=1e-12
        )

    def test_cells_never_shrink_with_distance_from_the_wall(self) -> None:
        widths = _grid().dz_m
        assert np.all(np.diff(widths) >= -1e-18)

    def test_the_smallest_cell_is_at_the_wall(self) -> None:
        widths = _grid().dz_m
        assert widths[0] == pytest.approx(widths.min())

    def test_unit_stretch_gives_a_uniform_mesh(self) -> None:
        widths = _grid(stretch=1.0).dz_m
        assert widths.std() == pytest.approx(0.0, abs=1e-15)

    def test_grading_saves_most_of_the_cells(self) -> None:
        """doc 03 §3.4: "uniform meshing wastes 90 % of the cells"."""
        graded = _grid()
        uniform = _grid(stretch=1.0)
        assert graded.n_cells < uniform.n_cells / 3


class TestValidation:
    def test_rejects_a_sheath_wider_than_the_domain(self) -> None:
        with pytest.raises(ValueError, match="sheath_thickness"):
            _grid(sheath_thickness=Q_(30.0, "mm"))

    def test_rejects_a_non_positive_debye_length(self) -> None:
        with pytest.raises(ValueError, match="debye_length"):
            _grid(debye_length=Q_(0.0, "m"))

    def test_rejects_a_stretch_below_one(self) -> None:
        """A stretch below one refines *away* from the wall — the opposite of doc 03."""
        with pytest.raises(ValueError, match="stretch"):
            _grid(stretch=0.98)

    def test_rejects_a_resolution_coarser_than_the_documented_floor(self) -> None:
        with pytest.raises(ValueError, match="cells_per_debye"):
            _grid(cells_per_debye=0.0)


class TestDolfinxInterval:
    """The FEniCSx mesh built on the grid must be the same mesh."""

    @pytest.mark.fenicsx
    def test_vertex_coordinates_round_trip(self) -> None:
        grid = _grid()
        mesh = interval_mesh(grid)
        coordinates = np.sort(mesh.geometry.x[:, 0])
        assert coordinates == pytest.approx(grid.z_m, rel=0.0, abs=1e-15)

    @pytest.mark.fenicsx
    def test_cell_count_matches_the_grid(self) -> None:
        grid = _grid()
        mesh = interval_mesh(grid)
        assert mesh.topology.index_map(1).size_local == grid.n_cells
