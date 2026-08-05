"""V-01 — method of manufactured solutions on the Poisson operator.

doc 03 §7 V-01: *observed order = design order +/- 0.1*. doc 01 §3 explains why this
particular operator gets the tightest gate in the suite:

    We do not measure the potential. We measure the densities, and the sheath physics —
    Poisson plus the ion momentum equation plus the Boltzmann electron relation —
    converts them into a potential profile [...] **The physics prior is not decoration;
    it is load-bearing, and it is load-bearing for the one quantity nothing else can
    reach.**

The manufactured problem comes from :func:`vpl.validation.manufactured.poisson_manufactured`
and the order fit from :mod:`vpl.validation.convergence`. Neither is reimplemented here:
a hand-derived source term that is subtly wrong makes a *correct* solver converge at the
wrong order, and the resulting bug hunt goes looking in the solver.

## Two things this study had to be designed around

**Nodal superconvergence.** One-dimensional finite elements for ``-u'' = f`` are nodally
exact at any degree, so a study built on vertex values measures floating-point noise. A
P2 element's interior node is the cell *midpoint*, so sampling there instead reports
order 4 for a third-order method — measured, before it was designed around. The study
samples two Gauss points per cell, which are superconvergent for the gradient and not for
the value. :mod:`vpl.physics.fluid.forms` states the argument in full.

**Mesh self-similarity.** The graded study holds the mesh's *total* growth ratio fixed
and varies the per-cell stretch, so that refining produces a self-similar family and
``min_dz`` is proportional to a characteristic ``h``. Holding the per-cell stretch fixed
instead changes the mesh's shape between levels — at stretch 1.01 the widest-to-narrowest
ratio runs from 1.5 at 40 cells to 24 at 320 — and the fitted slope then measures the
family rather than the scheme.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from vpl.core.state import SpatialGrid
from vpl.core.units import Q_, magnitude_in
from vpl.physics.fluid import solve_poisson
from vpl.validation.convergence import (
    RefinementLevel,
    assert_design_order,
    observed_order,
    weighted_l2_error,
)
from vpl.validation.manufactured import poisson_manufactured

#: A sheath-scaled manufactured problem: 1 mm of domain at the doc 01 §2.1 bias of 250 V.
LENGTH = 1.0e-3
AMPLITUDE = 250.0
MODES = 3

#: Coarse-to-fine cell counts. Four levels, a factor of two apart, chosen so the coarsest
#: cannot resolve the third harmonic and the finest stays far above the round-off floor
#: (the P1 error at 320 cells is ~1e-3 of the amplitude, against a machine epsilon of
#: 3e-14 on 250 V).
CELL_COUNTS = (40, 80, 160, 320)


def _uniform(n_cells: int) -> SpatialGrid:
    return SpatialGrid.uniform(length=Q_(LENGTH, "m"), n_points=n_cells + 1)


#: Widest-to-narrowest cell ratio held fixed across the graded study. See the module
#: docstring: a fixed *per-cell* stretch would change the mesh's shape between levels.
GRADED_TOTAL_RATIO = 8.0


def _graded(n_cells: int) -> SpatialGrid:
    """doc 03 §3.4's graded mesh, refined toward the wall, at fixed total grading."""
    return SpatialGrid.geometric(
        length=Q_(LENGTH, "m"),
        n_points=n_cells + 1,
        stretch=GRADED_TOTAL_RATIO ** (1.0 / (n_cells - 1)),
    )


def _study(
    grids: tuple[SpatialGrid, ...], *, degree: int
) -> tuple[list[RefinementLevel], list[float]]:
    """Solve the manufactured problem on each grid; return the study and the FE norms."""
    problem = poisson_manufactured(length=LENGTH, amplitude=AMPLITUDE, modes=MODES)
    levels: list[RefinementLevel] = []
    assembled: list[float] = []

    for grid in grids:
        solution = solve_poisson(
            grid=grid,
            charge_density=problem.source,
            boundary_values=problem.boundary_values,
            degree=degree,
        )
        points = solution.quadrature_points
        error = weighted_l2_error(
            solution.quadrature_values,
            problem.exact(points),
            weights=solution.quadrature_weights,
        )
        levels.append(RefinementLevel(h=float(magnitude_in(grid.min_dz, "m")), error=error))
        assembled.append(solution.l2_error(problem.exact))

    return levels, assembled


class TestManufacturedSolutionIsWhatItClaims:
    """Guard the manufactured problem before trusting it to judge the solver."""

    @pytest.mark.physics
    def test_the_source_reproduces_the_operator_applied_to_the_exact_solution(self) -> None:
        from vpl.core.constants import VACUUM_PERMITTIVITY

        problem = poisson_manufactured(length=LENGTH, amplitude=AMPLITUDE, modes=MODES)
        eps0 = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))
        z = np.linspace(0.0, LENGTH, 8001)
        h = z[1] - z[0]
        exact = problem.exact(z)
        second = (exact[2:] - 2.0 * exact[1:-1] + exact[:-2]) / h**2
        source = problem.source(z[1:-1])
        # A second difference is second-order accurate, so the comparison is made to the
        # accuracy the difference itself has and not to the accuracy sympy has.
        assert -eps0 * second == pytest.approx(source, rel=1e-4, abs=1e-6 * np.abs(source).max())


class TestV01PoissonOrderOfAccuracy:
    """doc 03 §7 V-01 / doc 11 G-1.1."""

    @pytest.mark.physics
    def test_p1_converges_at_second_order_on_a_uniform_mesh(self) -> None:
        levels, assembled = _study(tuple(_uniform(n) for n in CELL_COUNTS), degree=1)
        result = observed_order(levels)
        print(f"\nV-01 Poisson P1, uniform mesh:\n{result}")
        assert_design_order(result, design_order=2.0)

        fe_order = observed_order(
            [RefinementLevel(h=lv.h, error=e) for lv, e in zip(levels, assembled, strict=True)]
        )
        assert fe_order.observed_order == pytest.approx(result.observed_order, abs=0.1)

    @pytest.mark.physics
    def test_p1_converges_at_second_order_on_the_graded_mesh_of_doc_03(self) -> None:
        """Grading must not cost an order. doc 03 §3.4 grades every production mesh."""
        levels, _ = _study(tuple(_graded(n) for n in CELL_COUNTS), degree=1)
        result = observed_order(levels)
        print(f"\nV-01 Poisson P1, graded mesh:\n{result}")
        assert_design_order(result, design_order=2.0)

    @pytest.mark.physics
    def test_p2_converges_at_third_order(self) -> None:
        """doc 03 §3.4 offers "P1/P2 Lagrange"; the gate applies to whichever is used."""
        levels, _ = _study(tuple(_uniform(n) for n in (20, 40, 80, 160)), degree=2)
        result = observed_order(levels)
        print(f"\nV-01 Poisson P2, uniform mesh:\n{result}")
        assert_design_order(result, design_order=3.0)


class TestPoissonContract:
    def test_dirichlet_values_are_imposed_exactly(self) -> None:
        problem = poisson_manufactured(length=LENGTH, amplitude=AMPLITUDE, modes=MODES)
        solution = solve_poisson(
            grid=_uniform(64),
            charge_density=problem.source,
            boundary_values=(-AMPLITUDE, 0.0),
            degree=1,
        )
        assert solution.nodal[0] == pytest.approx(-AMPLITUDE, rel=1e-12)
        assert solution.nodal[-1] == pytest.approx(0.0, abs=1e-12)

    def test_zero_charge_density_gives_the_linear_profile(self) -> None:
        """The vacuum sheath: no space charge, so the potential is a straight line."""
        grid = _uniform(32)
        solution = solve_poisson(
            grid=grid,
            charge_density=lambda z: np.zeros_like(z),
            boundary_values=(-AMPLITUDE, 0.0),
            degree=1,
        )
        expected = -AMPLITUDE * (1.0 - grid.z_m / LENGTH)
        assert solution.nodal == pytest.approx(expected, abs=1e-9)

    def test_rejects_an_unsupported_element_degree(self) -> None:
        with pytest.raises(ValueError, match="degree"):
            solve_poisson(
                grid=_uniform(8),
                charge_density=lambda z: np.zeros_like(z),
                boundary_values=(0.0, 0.0),
                degree=0,
            )

    def test_cell_centres_are_the_midpoints_of_the_grid(self) -> None:
        grid = _graded(16)
        solution = solve_poisson(
            grid=grid,
            charge_density=lambda z: np.zeros_like(z),
            boundary_values=(0.0, 0.0),
            degree=1,
        )
        expected = 0.5 * (grid.z_m[:-1] + grid.z_m[1:])
        assert solution.cell_centres == pytest.approx(expected, rel=1e-12)
