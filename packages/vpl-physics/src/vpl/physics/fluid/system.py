"""The L1 discretisation — the weak forms of doc 03 §3.1 and the Newton continuation.

Split out of :mod:`vpl.physics.fluid.sheath` so that the solver's public surface — what a
manifest configures and what a caller receives — is readable without the variational forms
underneath it, and so that neither file exceeds the size at which a reviewer stops
reading.

Everything here is private to the package. :func:`solve_system` is the single entry point
and :class:`SolvedFields` is what it returns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as _mesh
from petsc4py import PETSc

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from vpl.core.state import PlasmaParams, SpatialGrid
from vpl.core.units import magnitude_in
from vpl.physics.analytic.sheath import SheathModel
from vpl.physics.fluid.forms import (
    DEFAULT_SOURCE_DEGREE_OFFSET,
    FieldSolution,
    NewtonReport,
    boundary_facets,
    dirichlet_on_component,
    interpolate_callable,
    run_newton,
    sample,
    ufl_fitted_diffusivity,
)
from vpl.physics.fluid.mesh import WALL_FACET, interval_mesh, wall_facet_tags
from vpl.physics.fluid.scaling import SheathScaling

if TYPE_CHECKING:
    from vpl.physics.fluid.sheath import FluidSheathSolver

__all__ = [
    "CONTINUATION_START",
    "MAX_BOLTZMANN_EXPONENT",
    "MIN_CONTINUATION_STEP",
    "SolvedFields",
    "solve_system",
]

#: A field sampled on the spatial grid, in SI magnitudes.
type FloatArray = np.typing.NDArray[np.float64]

_E_C = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E_KG = float(magnitude_in(ELECTRON_MASS, "kg"))

#: Where the bias continuation of :func:`_continue_to` starts, in units of the electron
#: temperature. A quarter of a ``T_e`` is inside the basin of attraction of the L0 seed at
#: every operating point tried; see that function for why a continuation is needed at all.
CONTINUATION_START: Final[float] = 0.25

#: Smallest continuation step, as a fraction of the bias already reached. Below this the
#: solve gives up rather than grinding: a step this small that still will not converge is
#: a solver problem, not a step-size problem, and saying so is more useful than a very
#: slow failure.
MIN_CONTINUATION_STEP: Final[float] = 1e-3

#: Largest ``e Phi / k T_e`` the Boltzmann relation is evaluated at before clamping.
#: ``exp(8)`` is three thousand times the reference density — far outside anything a
#: converging solve visits, and far inside the ``exp(709)`` that overflows a float64.
MAX_BOLTZMANN_EXPONENT: Final[float] = 8.0

#: Component indices in the mixed space.
_NU, _W, _PHI = 0, 1, 2


@dataclass(frozen=True, slots=True)
class SolvedFields:
    density: FieldSolution
    velocity: FieldSolution
    potential: FieldSolution
    newton: NewtonReport
    continuation_steps: int


def solve_system(
    solver: FluidSheathSolver,
    params: PlasmaParams,
    grid: SpatialGrid,
    scaling: SheathScaling,
) -> SolvedFields:
    """Assemble and solve the scaled three-field residual — doc 03 §3.1, §3.4."""
    domain = interval_mesh(grid)
    length = float(grid.z_m[-1])
    element = basix.ufl.element("Lagrange", "interval", 1)
    space = fem.functionspace(domain, basix.ufl.mixed_element([element, element, element]))

    unknown = fem.Function(space)

    nu, w, phi = ufl.split(unknown)
    q, r, v = ufl.TestFunctions(space)

    cell = ufl.CellDiameter(domain)
    injected = solver._injected_flux()

    # The wall potential is a Constant rather than a literal so that the continuation
    # schedule below can walk it up to the target without recompiling the form. FEniCSx
    # compiles each variational form to C on first use, and a form rebuilt once per
    # continuation step would spend more time in the compiler than in the solver.
    start = min(CONTINUATION_START, scaling.normalised_bias(params))
    wall_bias = fem.Constant(domain, PETSc.ScalarType(-start))

    # doc 03 §3.4's exponential fitting. The ion continuity and momentum equations carry
    # no diffusive flux of their own — the ion pressure lives in the momentum source, not
    # in a transport coefficient — so the fitted diffusivity is evaluated at zero
    # physical diffusivity and returns its Pe -> infinity limit, |a| h / 2. That limit is
    # what makes the scheme monotone across the sheath front; the finite-Peclet branch is
    # exercised by the collisional drift-diffusion form in vpl.physics.fluid.coupled and
    # by verification test V-02.
    #
    # The signed velocity is passed, not its magnitude. ufl_fitted_diffusivity takes the
    # absolute value itself, and UFL 2024.2 recurses without bound on ``abs(abs(x))``.
    fitted = ufl_fitted_diffusivity(velocity=w, diffusivity=0.0, cell_size=cell)

    facets = wall_facet_tags(domain, length=length)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)

    source = _scaled_source(solver, domain, scaling)
    drag = _drag(solver, w)
    secondary = _ufl_secondary_density(
        solver, params, scaling, phi, wall_bias=wall_bias, injected=injected
    )

    # Continuity: (nu w)' = sigma, in flux form with the wall outflow left natural.
    continuity = (
        fitted * ufl.grad(nu)[0] * ufl.grad(q)[0] - nu * w * ufl.grad(q)[0]
    ) * ufl.dx - nu * w * q * ds(WALL_FACET)
    if source is not None:
        continuity -= source * q * ufl.dx

    # Momentum: (nu w^2)' = -Theta nu phi' - c_p nu' - nu w |w| / lambda, in conservation
    # form exactly as doc 03 §3.1 writes it.
    momentum = (
        fitted * ufl.grad(nu * w)[0] * ufl.grad(r)[0]
        - nu * w * w * ufl.grad(r)[0]
        + (
            scaling.sonic_factor * nu * ufl.grad(phi)[0]
            + scaling.pressure_coefficient * ufl.grad(nu)[0]
            + nu * drag
        )
        * r
    ) * ufl.dx - nu * w * w * r * ds(WALL_FACET)

    # exp is clamped from above. phi is non-positive on any physical solution, but an
    # intermediate Newton iterate is under no such obligation, and exp of a few hundred is
    # an inf that turns the whole residual into NaN — from which no iteration recovers.
    # The clamp sits far outside the physical range, so it never binds on a converging
    # solve and only ever converts a divergence into a recoverable one.
    electrons = ufl.exp(ufl.min_value(phi, MAX_BOLTZMANN_EXPONENT))

    poisson = (
        scaling.length * ufl.grad(phi)[0] * ufl.grad(v)[0]
        - (nu - electrons - secondary) * v / scaling.length
    ) * ufl.dx

    wall_potential = _wall_potential_condition(space, position=0.0, value=-start)
    boundary = [
        dirichlet_on_component(space, _NU, position=length, value=1.0),
        dirichlet_on_component(space, _W, position=length, value=-injected),
        dirichlet_on_component(space, _PHI, position=length, value=0.0),
        wall_potential.condition,
    ]

    def solve_at(bias: float, step: int) -> NewtonReport:
        wall_bias.value = PETSc.ScalarType(-bias)
        wall_potential.value.x.array[:] = -bias
        return run_newton(
            continuity + momentum + poisson,
            unknown,
            bcs=boundary,
            rtol=solver.rtol,
            max_iterations=solver.max_iterations,
            relaxation=solver.relaxation,
            what=f"the L1 fluid sheath at V_w / T_e = {bias:.4g} (continuation step {step})",
        )

    iterations, steps = _continue_to(
        target=scaling.normalised_bias(params),
        start=start,
        factor=solver.continuation_factor,
        solve_at=solve_at,
        seed=lambda bias: _seed(solver, params, scaling, unknown, bias=bias),
        state=unknown,
    )

    return SolvedFields(
        density=sample(unknown.sub(_NU).collapse(), grid),
        velocity=sample(unknown.sub(_W).collapse(), grid),
        potential=sample(unknown.sub(_PHI).collapse(), grid),
        newton=NewtonReport(iterations=iterations, converged=True, rtol=solver.rtol),
        continuation_steps=steps,
    )


@dataclass(frozen=True, slots=True)
class _WallPotential:
    """A Dirichlet condition whose value the continuation loop can move."""

    condition: fem.DirichletBC
    value: fem.Function


def _wall_potential_condition(
    space: fem.FunctionSpace, *, position: float, value: float
) -> _WallPotential:
    """Pin the potential at the wall, keeping the value function so it can be updated."""
    collapsed, _ = space.sub(_PHI).collapse()
    pinned = fem.Function(collapsed)
    pinned.x.array[:] = value
    dofs = fem.locate_dofs_topological(
        (space.sub(_PHI), collapsed), 0, boundary_facets(space.mesh, position)
    )
    return _WallPotential(condition=fem.dirichletbc(pinned, dofs, space.sub(_PHI)), value=pinned)


def _continue_to(
    *,
    target: float,
    start: float,
    factor: float,
    solve_at: Callable[[float, int], NewtonReport],
    seed: Callable[[float], None],
    state: fem.Function,
) -> tuple[int, int]:
    """Walk the wall potential up to ``target``, halving any step that does not converge.

    **Newton on this system does not converge from the L0 seed at a sheath bias**, and
    that is a measurement rather than an expectation. At RP-1's ``V_w / T_e = 83`` the
    residual grows by four orders in four iterations and then overflows; it does so on
    every mesh, domain length and Bohm margin tried, with and without damping. Enlarging
    the basin of attraction was not possible. Walking into it is.

    So the wall potential is ramped geometrically from a quarter of an electron
    temperature — where the L0 seed *is* in the basin — with each converged solution
    seeding the next, and any step that fails is halved and retried from the last
    converged state. That last part is what carries the solve to doc 01 R-ENV-4's 1000 V,
    which a fixed geometric ramp does not reach.

    This is the standard continuation treatment of the sheath-matching problem, and it is
    what doc 03 §3.4's "Newton on the fully coupled system" costs in practice: tens of
    linear solves rather than one. That cost is the honest content of the G-1.4
    throughput measurement, and it is the single largest thing a reviewer should
    second-guess about this implementation.

    Args:
        target: The final ``V_w / T_e``, positive.
        start: Where the ramp begins.
        factor: Ratio between successive steps.
        solve_at: Solves at one bias, raising :class:`RuntimeError` if Newton fails.
        seed: Initialises the state from L0 at one bias.
        state: The unknown, snapshotted before each attempt so a failed step can be undone.

    Returns:
        ``(total Newton iterations, number of continuation steps)``.

    Raises:
        RuntimeError: If a step cannot be made to converge even at the minimum size.
    """
    if factor <= 1.0:
        raise ValueError(f"continuation_factor must exceed one, got {factor}")

    bias = min(start, target)
    seed(bias)
    report = solve_at(bias, 1)
    iterations, steps = report.iterations, 1

    while bias < target:
        step = min(bias * factor, target) - bias
        floor = MIN_CONTINUATION_STEP * bias
        while True:
            snapshot = state.x.array.copy()
            try:
                report = solve_at(bias + step, steps + 1)
            except RuntimeError:
                state.x.array[:] = snapshot
                state.x.scatter_forward()
                step *= 0.5
                if step < floor:
                    raise
                continue
            bias += step
            iterations += report.iterations
            steps += 1
            break

    return iterations, steps


def _scaled_source(
    solver: FluidSheathSolver, domain: _mesh.Mesh, scaling: SheathScaling
) -> fem.Function | None:
    """``sigma = S_iz / (n_s c_s)`` in m^-1, or ``None`` when there is no source."""
    if solver.ion_source is None:
        return None
    factor = 1.0 / scaling.flux
    term = solver.ion_source
    return interpolate_callable(
        domain,
        lambda z: np.asarray(term(z), dtype=np.float64) * factor,
        degree=1 + DEFAULT_SOURCE_DEGREE_OFFSET,
    )


def _drag(solver: FluidSheathSolver, w: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """``w |w| / lambda_mfp`` — doc 03 §3.1's ``nu_in u_i``, at constant mean free path.

    doc 03 §2.4 makes the constant-mean-free-path variant the default "because Ar+/Ar
    charge exchange has a weakly energy-dependent cross section, making lambda_CX the
    better-behaved constant", so the collision frequency is ``nu_in = |u_i| / lambda_CX``
    rather than a constant. Zero in the collisionless limit doc 03 §7 V-03 is stated in.
    """
    if solver.mean_free_path is None:
        return ufl.as_ufl(0.0)
    path = float(magnitude_in(solver.mean_free_path, "m"))
    if path <= 0.0:
        raise ValueError(f"mean_free_path must be positive, got {solver.mean_free_path}")
    return w * abs(w) / path


def _ufl_secondary_density(
    solver: FluidSheathSolver,
    params: PlasmaParams,
    scaling: SheathScaling,
    phi: ufl.core.expr.Expr,
    *,
    wall_bias: fem.Constant,
    injected: float,
) -> ufl.core.expr.Expr:
    """``nu_se(phi)`` — the doc 03 §3.3 secondary electron beam, as space charge.

    doc 03 §3.3: "**Secondary electron emission is not optional.** [...] Secondary
    electrons are accelerated back through the full sheath potential, gaining 250 eV, and
    they modify both the sheath structure and — importantly — the OES emission profile
    inside the sheath. Omitting them is a common and significant modelling error."

    With Boltzmann electrons there is no electron continuity equation to hang a flux
    boundary condition on, so the secondaries enter the only way they can: as a
    monoenergetic beam launched at the wall with :data:`SECONDARY_EMISSION_ENERGY` and
    accelerated by the sheath field (Hobbs & Wesson 1967; Lieberman & Lichtenberg §6.6).
    Flux conservation gives its density directly,

        ``n_se(z) = gamma_se Gamma_i / v_se(z)``,
        ``v_se(z) = sqrt(2 (E_se + e (Phi(z) - Phi_wall)) / m_e)``

    and it appears in Poisson as additional negative charge. The emitted flux is taken as
    the *injected* ion flux rather than the local one, which is exact wherever ion
    continuity is source-free and is the same approximation doc 03 §3.3 makes implicitly
    by writing a single ``gamma_se``.
    """
    if params.gamma_se == 0.0:
        return ufl.as_ufl(0.0)

    emission = float(magnitude_in(solver.secondary_emission_energy, "J"))
    # max_value guards the square root: an intermediate Newton iterate may put phi below
    # the wall potential, which is unphysical and would otherwise produce a NaN residual
    # that no line search can recover from.
    drop = ufl.max_value(phi - wall_bias, 0.0)
    velocity = ufl.sqrt(2.0 * (emission + _E_C * scaling.potential * drop) / _M_E_KG)
    return params.gamma_se * injected * scaling.speed / velocity


def _seed(
    solver: FluidSheathSolver,
    params: PlasmaParams,
    scaling: SheathScaling,
    unknown: fem.Function,
    *,
    bias: float,
) -> None:
    """Initialise Newton from L0 — doc 03 §2.2's stated use for the matrix sheath.

    The seed satisfies ion continuity and the collisionless momentum equation *exactly*
    for the L0 potential, so Newton only has to move the potential onto the self-
    consistent one. Which L0 profile is used must not change the converged answer, and a
    test asserts it does not.
    """
    thickness = float(magnitude_in(solver.sheath_estimate(params), "m"))
    exponent = 2.0 if solver.initial_model is SheathModel.MATRIX else 4.0 / 3.0
    injected = solver._injected_flux()
    theta = scaling.sonic_factor

    def potential(z: FloatArray) -> FloatArray:
        return np.asarray(
            -bias * np.clip(1.0 - z / thickness, 0.0, 1.0) ** exponent, dtype=np.float64
        )

    def velocity(z: FloatArray) -> FloatArray:
        return np.asarray(-np.sqrt(injected**2 + 2.0 * theta * (-potential(z))), dtype=np.float64)

    def density(z: FloatArray) -> FloatArray:
        return np.asarray(injected / np.abs(velocity(z)), dtype=np.float64)

    for component, seed in ((_NU, density), (_W, velocity), (_PHI, potential)):
        unknown.sub(component).interpolate(
            lambda x, fn=seed: fn(np.asarray(x[0], dtype=np.float64))
        )
    unknown.x.scatter_forward()
