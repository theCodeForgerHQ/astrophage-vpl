"""L1 — the fluid sheath solver of doc 03 §3, on FEniCSx.

doc 03 §1 gives L1 its job: "workhorse for sweeps; valid where the IVDF is near-drifting-
Maxwellian", at roughly a second a solve against L2's minutes. doc 03 §9 makes FEniCSx a
**buy** decision — "mature, verified, adjoint-capable" — and doc 08 §3 puts it behind an
optional dependency for a stated reason:

    ``pip install vpl-core vpl-inverse`` must work without dragging in FEniCSx and Gmsh —
    a user who only wants to run the inversion on their own data should not need a PDE
    stack.

**Importing this package therefore requires FEniCSx.** Everything in
:mod:`vpl.physics.analytic` — L0, the verification anchor — works without it, and
``vpl-physics[fluid]`` is the extra that adds the stack.

## What is here

============================================  ==============================================
:mod:`~vpl.physics.fluid.sheath`              L1 itself: the coupled ``(n_i, u_i, Phi)`` system
:mod:`~vpl.physics.fluid.system`              Its weak forms and the Newton bias continuation
:mod:`~vpl.physics.fluid.coupled`             Drift-diffusion ions + Boltzmann + Poisson (V-02)
:mod:`~vpl.physics.fluid.poisson`             Poisson alone (V-01)
:mod:`~vpl.physics.fluid.drift_diffusion`     Ion continuity alone, with and without stabilisation
:mod:`~vpl.physics.fluid.exponential_fitting` Scharfetter-Gummel, as mathematics
:mod:`~vpl.physics.fluid.mesh`                The graded mesh of doc 03 §3.4
:mod:`~vpl.physics.fluid.scaling`             What makes the Newton system solvable
:mod:`~vpl.physics.fluid.forms`               FEniCSx plumbing shared by all of the above
============================================  ==============================================

The operators are exposed separately from the solver on purpose. doc 07 §2.3 applies the
method of manufactured solutions "to every PDE solver", and a manufactured solution that
exercised a re-implementation of the discretisation rather than the production one would
verify the test.
"""

from __future__ import annotations

from vpl.physics.fluid.coupled import CoupledSolution, solve_drift_diffusion_poisson
from vpl.physics.fluid.drift_diffusion import Stabilisation, solve_drift_diffusion
from vpl.physics.fluid.exponential_fitting import (
    bernoulli,
    cell_peclet,
    fitted_diffusivity,
    scharfetter_gummel_coefficients,
)
from vpl.physics.fluid.forms import FieldSolution, NewtonReport
from vpl.physics.fluid.mesh import (
    DEFAULT_DOMAIN_LENGTH,
    DEFAULT_MESH_STRETCH,
    MIN_CELLS_PER_DEBYE,
    graded_sheath_grid,
    interval_mesh,
    max_cell_size_within,
)
from vpl.physics.fluid.poisson import solve_poisson
from vpl.physics.fluid.scaling import SheathScaling
from vpl.physics.fluid.sheath import (
    DEFAULT_BOHM_MARGIN,
    SECONDARY_EMISSION_ENERGY,
    SHEATH_EDGE_TOLERANCE,
    FluidSheathSolver,
    SheathSolution,
)

__all__ = [
    "DEFAULT_BOHM_MARGIN",
    "DEFAULT_DOMAIN_LENGTH",
    "DEFAULT_MESH_STRETCH",
    "MIN_CELLS_PER_DEBYE",
    "SECONDARY_EMISSION_ENERGY",
    "SHEATH_EDGE_TOLERANCE",
    "CoupledSolution",
    "FieldSolution",
    "FluidSheathSolver",
    "NewtonReport",
    "SheathScaling",
    "SheathSolution",
    "Stabilisation",
    "bernoulli",
    "cell_peclet",
    "fitted_diffusivity",
    "graded_sheath_grid",
    "interval_mesh",
    "max_cell_size_within",
    "scharfetter_gummel_coefficients",
    "solve_drift_diffusion",
    "solve_drift_diffusion_poisson",
    "solve_poisson",
]
