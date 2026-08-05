"""Ion continuity in drift-diffusion form — doc 03 §3.1, discretised per doc 03 §3.4.

    ``d/dz (u n_i - D dn_i/dz)  =  S_iz - L_rec``

This is the ion continuity equation of doc 03 §3.1 with the momentum closure folded into
a drift velocity, which is the form doc 03 §1 names when it calls L1 "drift-diffusion
ions + Boltzmann electrons + Poisson". It is also the operator
:func:`vpl.validation.manufactured.drift_diffusion_manufactured` manufactures a solution
for, and therefore the one doc 03 §7 V-02 gates.

Two discretisations are available and the choice is not a preference:

- :attr:`Stabilisation.EXPONENTIAL_FITTING` is doc 03 §3.4's Scharfetter-Gummel.
- :attr:`Stabilisation.NONE` is plain centred Galerkin, retained **so that the claim
  doc 03 §3.4 makes about it can be measured**. "Centred differences oscillate in the
  strongly-drift-dominated sheath" is a statement about this code, and a statement about
  code that the code cannot be made to demonstrate is a statement nobody has checked.

The inertial three-field system that the actual sheath solve uses is
:mod:`vpl.physics.fluid.sheath`; the drift-diffusion form here cannot recover
Child-Langmuir, because a constant-mobility ion flux gives the Mott-Gurney
``V^2 / s^3`` law rather than the ``V^{3/2} / s^2`` of doc 03 §2.3. That is a property of
the model, not of the discretisation, and doc 03 §3.1 is careful to write its momentum
equation with inertia for exactly this reason.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import ufl
from dolfinx.fem.petsc import LinearProblem

from vpl.core.state import SpatialGrid
from vpl.physics.fluid.forms import (
    DEFAULT_SOURCE_DEGREE_OFFSET,
    FieldSolution,
    SourceTerm,
    dirichlet_at_ends,
    interpolate_callable,
    lagrange_space,
    sample,
    ufl_fitted_diffusivity,
)
from vpl.physics.fluid.mesh import interval_mesh

__all__ = ["Stabilisation", "solve_drift_diffusion"]

_DIRECT: Final[dict[str, str]] = {"ksp_type": "preonly", "pc_type": "lu"}


class Stabilisation(StrEnum):
    """How the advective term is discretised — doc 03 §3.4.

    A ``StrEnum`` because manifests are data (doc 08 §1 principle 4): the choice
    round-trips into the provenance record, so a run that used the unstabilised operator
    says so rather than being indistinguishable from one that did not.
    """

    #: Doc 03 §3.4's choice. Scharfetter-Gummel, in its finite element form.
    EXPONENTIAL_FITTING = "exponential_fitting"

    #: Plain centred Galerkin. Oscillates above cell Peclet one; present so that can be
    #: demonstrated rather than asserted.
    NONE = "none"


def solve_drift_diffusion(
    *,
    grid: SpatialGrid,
    drift: float,
    diffusivity: float,
    source: SourceTerm,
    boundary_values: tuple[float, float],
    stabilisation: Stabilisation = Stabilisation.EXPONENTIAL_FITTING,
    source_degree_offset: int = DEFAULT_SOURCE_DEGREE_OFFSET,
) -> FieldSolution:
    """Solve ``(u n - D n')' = S`` on P1 with Dirichlet ends.

    P1 only. The fitted diffusivity is derived for a two-point stencil, so it has no
    meaning on an element with interior nodes; a P2 "Scharfetter-Gummel" scheme would be
    stabilised by an expression that does not correspond to the scheme doc 03 §3.4 named.

    Args:
        grid: Where to solve.
        drift: Advection velocity ``u`` in m/s, constant.
        diffusivity: Diffusivity ``D`` in m^2/s. Must be positive here; the collisionless
            limit belongs to the inertial system of :mod:`vpl.physics.fluid.sheath`.
        source: ``S(z)`` in m^-3 s^-1 — the ``S_iz - L_rec`` of doc 03 §3.1.
        boundary_values: ``(n(0), n(L))`` in m^-3.
        stabilisation: See :class:`Stabilisation`.
        source_degree_offset: How far above P1 the source is represented.

    Returns:
        The density, sampled at grid vertices and cell midpoints.

    Raises:
        ValueError: If ``diffusivity`` is not positive.
    """
    if diffusivity <= 0.0:
        raise ValueError(
            f"diffusivity must be positive here, got {diffusivity}. The collisionless "
            "limit has no diffusive flux at all and is solved by the inertial system in "
            "vpl.physics.fluid.sheath."
        )

    domain = interval_mesh(grid)
    space = lagrange_space(domain, 1)
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)

    total_diffusivity = _effective_diffusivity(
        domain=domain, drift=drift, diffusivity=diffusivity, stabilisation=stabilisation
    )

    # -integral( (u n - D n') v' ) = integral( S v ), the flux form of the residual with
    # the test function vanishing at both Dirichlet ends.
    bilinear = (
        total_diffusivity * ufl.grad(trial)[0] * ufl.grad(test)[0]
        - drift * trial * ufl.grad(test)[0]
    ) * ufl.dx
    linear = interpolate_callable(domain, source, degree=1 + source_degree_offset) * test * ufl.dx

    problem = LinearProblem(
        bilinear,
        linear,
        bcs=dirichlet_at_ends(space, grid, boundary_values),
        petsc_options=_DIRECT,
    )
    return sample(problem.solve(), grid)


def _effective_diffusivity(
    *,
    domain: object,
    drift: float,
    diffusivity: float,
    stabilisation: Stabilisation,
) -> ufl.core.expr.Expr:
    """``D`` or the exponentially-fitted total, depending on :class:`Stabilisation`."""
    if stabilisation is Stabilisation.NONE:
        return ufl.as_ufl(diffusivity)
    return ufl_fitted_diffusivity(
        velocity=ufl.as_ufl(drift),
        diffusivity=diffusivity,
        cell_size=ufl.CellDiameter(domain),
    )
