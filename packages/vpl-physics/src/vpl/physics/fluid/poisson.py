"""Poisson — doc 03 §3.1, and the operator doc 01 §3 calls load-bearing.

    ``-eps0 d2Phi/dz2 = e (n_i - n_e)``

doc 01 §3 states plainly why this equation, of all the equations in the framework, gets
the tightest verification gate:

    ``Phi(z)`` **is essentially not measurable** by routine means. This is the crucial
    gap. And it is precisely the gap the physics prior fills: **Poisson's equation links
    ``Phi(z)`` to ``n_e(z)`` and ``n_i(z)``**, both of which *are* observable. [...] **The
    physics prior is not decoration; it is load-bearing, and it is load-bearing for the
    one quantity nothing else can reach.**

Solved here on its own, with the charge density supplied, so that doc 03 §7 V-01 tests
the discretisation of the operator without the ion transport on top of it. The coupled
version is :mod:`vpl.physics.fluid.coupled`.
"""

from __future__ import annotations

from typing import Final

import ufl
from dolfinx.fem.petsc import LinearProblem

from vpl.core.constants import VACUUM_PERMITTIVITY
from vpl.core.state import SpatialGrid
from vpl.core.units import magnitude_in
from vpl.physics.fluid.forms import (
    DEFAULT_SOURCE_DEGREE_OFFSET,
    FieldSolution,
    SourceTerm,
    dirichlet_at_ends,
    interpolate_callable,
    lagrange_space,
    sample,
)
from vpl.physics.fluid.mesh import interval_mesh

__all__ = ["solve_poisson"]

_EPS0_F_PER_M: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

#: Options for a direct factorisation. See :func:`vpl.physics.fluid.forms.run_newton`.
_DIRECT: Final[dict[str, str]] = {"ksp_type": "preonly", "pc_type": "lu"}


def solve_poisson(
    *,
    grid: SpatialGrid,
    charge_density: SourceTerm,
    boundary_values: tuple[float, float],
    degree: int = 1,
    source_degree_offset: int = DEFAULT_SOURCE_DEGREE_OFFSET,
) -> FieldSolution:
    """Solve ``-eps0 Phi'' = rho`` with Dirichlet ends.

    Args:
        grid: Where to solve. The wall is at ``z = 0`` (doc 02 §2).
        charge_density: ``rho(z)`` in C/m^3, evaluated on an array of positions in metres.
        boundary_values: ``(Phi(0), Phi(L))`` in volts. doc 03 §3.3 fixes these for the
            sheath: ``Phi = V_w`` at the wall, ``Phi = 0`` in the bulk.
        degree: Lagrange degree. doc 03 §3.4 offers P1 or P2; the design order for the
            ``L2`` norm is ``degree + 1``, and doc 03 §7 V-01 gates on it.
        source_degree_offset: How far above ``degree`` the charge density is represented.
            See :data:`~vpl.physics.fluid.forms.DEFAULT_SOURCE_DEGREE_OFFSET`.

    Returns:
        The potential, sampled at grid vertices and cell midpoints.

    Raises:
        ValueError: If ``degree`` is below one.
    """
    domain = interval_mesh(grid)
    space = lagrange_space(domain, degree)

    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    rho = interpolate_callable(domain, charge_density, degree=degree + source_degree_offset)

    bilinear = _EPS0_F_PER_M * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
    linear = rho * test * ufl.dx

    problem = LinearProblem(
        bilinear,
        linear,
        bcs=dirichlet_at_ends(space, grid, boundary_values),
        petsc_options=_DIRECT,
    )
    return sample(problem.solve(), grid)
