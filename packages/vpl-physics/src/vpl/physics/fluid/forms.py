"""FEniCSx plumbing shared by every L1 solve — doc 03 §3.4, doc 08 §5.

Three things live here and each exists because getting it wrong is silent:

**Sampling.** doc 08 §5 puts dimensional quantities at module boundaries and bare arrays
inside hot loops. A finite element ``Function`` is neither, so :class:`FieldSolution`
converts it once, at the boundary, into the two views the rest of the framework needs:
nodal values on the :class:`~vpl.core.state.SpatialGrid` (what a
:class:`~vpl.core.state.ScalarField` carries) and cell-midpoint values (what a
convergence study must measure against — see below).

**The exponentially-fitted diffusivity, in UFL.** The scalar mathematics is in
:mod:`vpl.physics.fluid.exponential_fitting` and is tested there without a PDE stack;
:func:`ufl_fitted_diffusivity` is the same expression written for assembly, so the tested
formula and the assembled one cannot drift apart.

**Newton.** doc 03 §3.4 specifies "Newton on the fully coupled system" because "the
n_e(Phi) exponential makes Gummel iteration converge poorly at high bias".
:func:`run_newton` fails loudly rather than returning an unconverged iterate: doc 07 §7
is explicit that physics regressions are otherwise silent — the code runs, the plots look
plausible, and the answer is wrong.

## Why the error is measured at Gauss points and nowhere else

One-dimensional finite elements for ``-u'' = f`` are **nodally exact**: the discrete
Green's function argument makes the computed values at the element nodes equal the exact
ones to round-off, on any mesh, graded or not, at any degree. A convergence study built
on nodal error measures floating-point noise. Worse, the trap moves: a P2 element's
interior node *is* the cell midpoint, so a study that avoided the vertices by sampling
midpoints reports fourth-order convergence of a third-order method — which is a true
statement about those points and a false one about the scheme. That failure was observed
here before it was designed around, which is the argument for measuring it rather than
reasoning about it.

:attr:`FieldSolution.quadrature_points` therefore samples two Gauss points per cell.
Gauss points are superconvergent for the *gradient*, not for the value, so a field
sampled there carries the generic error; and the two-point rule is exact for cubics, so
its own quadrature error is a fixed fraction of the norm rather than a contribution to
its order. :meth:`FieldSolution.l2_error` assembles the true ``L2(Omega)`` norm alongside
so the two cannot silently disagree.

:attr:`FieldSolution.cell_values` remains, at the midpoints, because a cell-averaged
field is what the physics wants; it is simply not what a convergence study wants.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as _mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from numpy.typing import NDArray
from petsc4py import PETSc

from vpl.core.state import SpatialGrid
from vpl.physics.fluid.exponential_fitting import COTH_UNITY_ARGUMENT

__all__ = [
    "DEFAULT_NEWTON_RTOL",
    "DEFAULT_SOURCE_DEGREE_OFFSET",
    "FieldSolution",
    "NewtonReport",
    "boundary_facets",
    "cells_in_order",
    "dirichlet_at_ends",
    "dirichlet_on_component",
    "evaluate_at",
    "interpolate_callable",
    "lagrange_space",
    "run_newton",
    "sample",
    "ufl_fitted_diffusivity",
]

#: A field sampled on a grid, in SI magnitudes.
type FloatArray = NDArray[np.float64]

#: A source term, evaluated on an array of positions in metres.
type SourceTerm = Callable[[FloatArray], FloatArray]

#: How many degrees above the solution space a manufactured or physical source term is
#: represented in. Two is enough that the source's own representation error is ``O(h^4)``
#: against an ``O(h^2)`` solution error, so the order study measures the discretisation
#: of the *operator* rather than of its right-hand side.
DEFAULT_SOURCE_DEGREE_OFFSET: Final[int] = 2

#: Relative tolerance on the Newton increment. Tight, because every gate below is a
#: physics tolerance in the percent range and an iteration error anywhere near it would
#: be indistinguishable from a discretisation error.
DEFAULT_NEWTON_RTOL: Final[float] = 1e-11

#: Absolute floor on the Newton residual, so a problem whose residual is already at
#: round-off does not iterate to the maximum count against noise.
DEFAULT_NEWTON_ATOL: Final[float] = 1e-14

#: Smallest Peclet argument the fitted diffusivity forms, to keep ``coth`` finite at a
#: stagnation point. See :func:`ufl_fitted_diffusivity`.
_MIN_PECLET_ARGUMENT: Final[float] = 1e-12

#: Slack allowed when checking that an evaluation point is inside the grid, in metres.
#: A Gauss point computed from the grid's own coordinates can land a few ulp outside it.
_POSITION_TOLERANCE: Final[float] = 1e-15


def lagrange_space(domain: _mesh.Mesh, degree: int) -> fem.FunctionSpace:
    """A scalar continuous Lagrange space — doc 03 §3.4's "P1/P2 Lagrange"."""
    if degree < 1:
        raise ValueError(f"element degree must be at least one, got {degree}")
    return fem.functionspace(domain, ("Lagrange", degree))


def interpolate_callable(domain: _mesh.Mesh, source: SourceTerm, *, degree: int) -> fem.Function:
    """Represent a Python source term in a Lagrange space of the given degree."""
    space = lagrange_space(domain, degree)
    function = fem.Function(space)
    function.interpolate(lambda x: source(np.asarray(x[0], dtype=np.float64)))
    return function


def ufl_fitted_diffusivity(
    *, velocity: ufl.core.expr.Expr, diffusivity: float, cell_size: ufl.core.expr.Expr
) -> ufl.core.expr.Expr:
    """``(|a| h / 2) coth(|a| h / (2 D))`` as a UFL expression.

    The scalar twin of
    :func:`~vpl.physics.fluid.exponential_fitting.fitted_diffusivity`, which the test
    suite checks against the Scharfetter-Gummel two-point flux directly.

    Args:
        velocity: The local advection velocity, generally a function of the unknowns.
        diffusivity: The physical diffusivity. **Zero is not a degenerate case** — it is
            the collisionless sheath of doc 03 §7 V-03, in which the ion continuity
            equation carries no diffusive flux at all, and the fitted value reduces
            exactly to the upwind ``|a| h / 2``.
        cell_size: The local cell width, normally ``ufl.CellDiameter(mesh)``.

    Returns:
        The total diffusivity to put in the flux, physical part included.
    """
    if diffusivity < 0.0:
        raise ValueError(f"diffusivity must be non-negative, got {diffusivity}")

    half_flux = abs(velocity) * cell_size / 2.0
    if diffusivity == 0.0:
        return half_flux

    # Clamped from above because coth is one to double precision past
    # COTH_UNITY_ARGUMENT, and from below because coth diverges at zero — which is
    # reached wherever the velocity vanishes, and where the correct answer is the
    # physical diffusivity that the outer maximum then supplies.
    argument = ufl.max_value(
        ufl.min_value(half_flux / diffusivity, COTH_UNITY_ARGUMENT), _MIN_PECLET_ARGUMENT
    )
    return ufl.max_value(half_flux / ufl.tanh(argument), ufl.as_ufl(diffusivity))


@dataclass(frozen=True, slots=True)
class FieldSolution:
    """One solved field, in the two views doc 08 §5 asks for.

    Attributes:
        grid: The spatial grid the solve was performed on.
        nodal: Values at the grid vertices, ascending in ``z``. What a
            :class:`~vpl.core.state.ScalarField` carries.
        cell_values: Values at the cell midpoints, ascending in ``z``. What a convergence
            study measures against — see the module docstring.
        function: The finite element function itself, retained so that the assembled
            ``L2`` norm can be taken without re-solving.
        scale: What ``function`` must be multiplied by to reach physical units. Solvers
            that non-dimensionalise their unknowns (see
            :mod:`vpl.physics.fluid.coupled`) leave the scaling here rather than
            rescaling the dof vector, so that :meth:`l2_error` and the sampled arrays
            cannot end up in different units.
    """

    grid: SpatialGrid
    nodal: FloatArray
    cell_values: FloatArray
    function: fem.Function
    scale: float = 1.0

    @property
    def cell_centres(self) -> FloatArray:
        """Midpoints of the grid cells, in metres."""
        return 0.5 * (self.grid.z_m[:-1] + self.grid.z_m[1:])

    @property
    def cell_widths(self) -> FloatArray:
        """Cell widths in metres — the weights of a midpoint-rule ``L2`` norm."""
        return self.grid.dz_m

    @property
    def quadrature_points(self) -> FloatArray:
        """Two Gauss points per cell, ascending in ``z``.

        Where a convergence study must sample, and **not** the vertices or the midpoints.
        In one dimension the finite element solution is superconvergent at the element's
        own nodes: nodally exact at the vertices for any degree, and — because a P2
        element's interior node *is* the cell midpoint — superconvergent at the midpoints
        too. A P2 study sampled at midpoints measures fourth-order convergence of a
        third-order method, which is a true statement about those points and a false one
        about the scheme.

        Gauss points are superconvergent for the gradient, not for the value, so they are
        the right place to sample a field. The two-point rule is exact for cubics, so its
        own quadrature error is a fixed fraction of the norm rather than a contribution
        to its order.
        """
        offset = self.cell_widths / (2.0 * np.sqrt(3.0))
        centres = self.cell_centres
        return np.stack([centres - offset, centres + offset], axis=1).reshape(-1)

    @property
    def quadrature_weights(self) -> FloatArray:
        """Weights matching :attr:`quadrature_points` — half a cell width each."""
        halves = 0.5 * self.cell_widths
        return np.stack([halves, halves], axis=1).reshape(-1)

    @property
    def quadrature_values(self) -> FloatArray:
        """The solution at :attr:`quadrature_points`."""
        return self.scale * evaluate_at(self.function, self.quadrature_points, self.grid)

    def l2_error(self, exact: SourceTerm, *, degree_offset: int = 2) -> float:
        """The assembled ``L2(Omega)`` norm of the difference from ``exact``.

        The finite element norm, taken by quadrature over the elements rather than by
        sampling. Reported alongside the midpoint-sampled norm so that a bug in either
        shows up as a disagreement rather than as a plausible number.
        """
        domain = self.function.function_space.mesh
        degree = self.function.function_space.ufl_element().embedded_superdegree
        reference = interpolate_callable(domain, exact, degree=degree + degree_offset)
        squared = fem.assemble_scalar(
            fem.form((self.scale * self.function - reference) ** 2 * ufl.dx)
        )
        return float(np.sqrt(max(float(squared), 0.0)))


def cells_in_order(domain: _mesh.Mesh, n_cells: int) -> NDArray[np.int32]:
    """dolfinx cell indices sorted by position along the interval.

    dolfinx does not promise that cell ``j`` of its own numbering is the ``j``-th cell
    from the wall, and on a graded mesh the difference is not subtle. Every routine here
    that needs a per-cell quantity in grid order goes through this.
    """
    midpoints = _mesh.compute_midpoints(domain, 1, np.arange(n_cells, dtype=np.int32))[:, 0]
    return np.asarray(np.argsort(midpoints), dtype=np.int32)


def evaluate_at(function: fem.Function, positions: FloatArray, grid: SpatialGrid) -> FloatArray:
    """Evaluate a finite element function at arbitrary points along the interval.

    The owning cell is found by bisection on the grid rather than by a bounding-box
    collision query. dolfinx's collision test carries a geometric tolerance scaled to the
    cell, and on the graded meshes of doc 03 §3.4 — where the narrowest cell is 13 um in
    a 20 mm domain — it reports interior points as belonging to no cell at all. Bisection
    on a strictly increasing coordinate axis cannot.

    Raises:
        ValueError: If a point lies outside the grid.
    """
    if positions.size and (
        positions.min() < grid.z_m[0] - _POSITION_TOLERANCE
        or positions.max() > grid.z_m[-1] + _POSITION_TOLERANCE
    ):
        raise ValueError(
            f"positions must lie inside [{grid.z_m[0]:.6g}, {grid.z_m[-1]:.6g}] m; got "
            f"[{positions.min():.6g}, {positions.max():.6g}]"
        )

    domain = function.function_space.mesh
    order = cells_in_order(domain, grid.n_cells)
    index = np.clip(np.searchsorted(grid.z_m, positions, side="right") - 1, 0, grid.n_cells - 1)

    points = np.zeros((positions.size, 3), dtype=np.float64)
    points[:, 0] = positions
    return np.asarray(function.eval(points, order[index]), dtype=np.float64).reshape(-1)


def sample(function: fem.Function, grid: SpatialGrid) -> FieldSolution:
    """Read a finite element function onto ``grid``, at vertices and at cell midpoints.

    Both views are produced by *interpolation into* a P1 and a DG-0 space rather than by
    reading the dof vector directly, so the routine is correct for any element degree.
    Degrees of freedom are then reordered by coordinate: dolfinx makes no promise that
    its dof numbering follows the mesh geometry, and a field written out in dof order
    would be a plausible-looking permutation of the answer.
    """
    domain = function.function_space.mesh

    vertex_space = lagrange_space(domain, 1)
    at_vertices = fem.Function(vertex_space)
    at_vertices.interpolate(function)
    vertex_order = np.argsort(vertex_space.tabulate_dof_coordinates()[:, 0])

    cell_space = fem.functionspace(domain, ("DG", 0))
    at_cells = fem.Function(cell_space)
    at_cells.interpolate(function)
    cell_dofs = np.asarray(cell_space.dofmap.list, dtype=np.int32).reshape(-1)
    cell_order = cells_in_order(domain, grid.n_cells)

    return FieldSolution(
        grid=grid,
        nodal=np.asarray(at_vertices.x.array[vertex_order], dtype=np.float64),
        cell_values=np.asarray(at_cells.x.array[cell_dofs][cell_order], dtype=np.float64),
        function=function,
    )


def boundary_facets(domain: _mesh.Mesh, position: float) -> NDArray[np.int32]:
    """The boundary vertices of a 1-D mesh at ``z = position``.

    Located by coordinate rather than by index. dolfinx does not promise that vertex 0 is
    the one at the wall, and doc 02 §2 makes the wall end normative — a boundary
    condition applied to the wrong end of the domain produces a perfectly convergent
    solution to the wrong problem.
    """
    located = _mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], position))
    return np.asarray(located, dtype=np.int32)


def dirichlet_at_ends(
    space: fem.FunctionSpace, grid: SpatialGrid, values: tuple[float, float]
) -> list[fem.DirichletBC]:
    """Pin a scalar field at the wall and at the bulk boundary."""
    domain = space.mesh
    conditions: list[fem.DirichletBC] = []
    for position, value in ((0.0, values[0]), (float(grid.z_m[-1]), values[1])):
        dofs = fem.locate_dofs_topological(space, 0, boundary_facets(domain, position))
        conditions.append(fem.dirichletbc(np.float64(value), dofs, space))
    return conditions


def dirichlet_on_component(
    mixed: fem.FunctionSpace, component: int, *, position: float, value: float
) -> fem.DirichletBC:
    """Pin one component of a mixed space at one end of the interval."""
    domain = mixed.mesh
    collapsed, _ = mixed.sub(component).collapse()
    dofs = fem.locate_dofs_topological(
        (mixed.sub(component), collapsed), 0, boundary_facets(domain, position)
    )
    pinned = fem.Function(collapsed)
    pinned.x.array[:] = value
    return fem.dirichletbc(pinned, dofs, mixed.sub(component))


@dataclass(frozen=True, slots=True)
class NewtonReport:
    """What the nonlinear solve did — doc 03 §3.4's claim, made checkable.

    Attributes:
        iterations: Newton steps taken.
        converged: Whether the increment met the tolerance. Always ``True`` on a report
            that :func:`run_newton` returned; a ``False`` one has been raised on.
        rtol: The relative tolerance that was applied.
    """

    iterations: int
    converged: bool
    rtol: float

    def __repr__(self) -> str:
        outcome = "converged" if self.converged else "DID NOT CONVERGE"
        return f"NewtonReport({self.iterations} iterations, {outcome}, rtol={self.rtol:.1e})"


def run_newton(
    residual: ufl.Form,
    unknown: fem.Function,
    *,
    bcs: list[fem.DirichletBC],
    rtol: float = DEFAULT_NEWTON_RTOL,
    atol: float = DEFAULT_NEWTON_ATOL,
    max_iterations: int = 50,
    relaxation: float = 1.0,
    what: str = "the coupled system",
) -> NewtonReport:
    """Newton on the fully coupled residual — doc 03 §3.4.

    The Jacobian is derived symbolically by UFL rather than assembled by hand, which
    removes the single most common cause of a Newton method that converges linearly and
    is then blamed on the physics.

    Raises:
        RuntimeError: If the iteration does not converge. Returning the last iterate
            instead would hand back a field that satisfies no equation and looks like a
            solution.
    """
    problem = NonlinearProblem(residual, unknown, bcs=bcs)
    solver = NewtonSolver(unknown.function_space.mesh.comm, problem)
    solver.rtol = rtol
    solver.atol = atol
    solver.max_it = max_iterations
    solver.convergence_criterion = "incremental"
    solver.error_on_nonconvergence = False
    solver.relaxation_parameter = relaxation

    _use_direct_solver(solver.krylov_solver)

    iterations, converged = solver.solve(unknown)
    if not converged:
        raise RuntimeError(
            f"Newton did not converge on {what} in {max_iterations} iterations "
            f"(rtol={rtol:.1e}). doc 03 §3.4 chose Newton over Gummel iteration because "
            "the n_e(Phi) exponential converges poorly at high bias; an unconverged "
            "iterate is not a solution and is not returned."
        )
    return NewtonReport(iterations=int(iterations), converged=True, rtol=rtol)


def _use_direct_solver(ksp: PETSc.KSP) -> None:
    """Factorise directly. The L1 system is one-dimensional and has ~10^3 unknowns.

    An iterative solver would need a preconditioner tuned to a system whose blocks span
    twenty orders of magnitude in scale, to save time that is not being spent.
    """
    options = PETSc.Options()
    prefix = ksp.getOptionsPrefix() or ""
    options[f"{prefix}ksp_type"] = "preonly"
    options[f"{prefix}pc_type"] = "lu"
    ksp.setFromOptions()
