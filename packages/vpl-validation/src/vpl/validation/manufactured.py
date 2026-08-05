"""Manufactured solutions for the method of manufactured solutions — doc 07 §2.3.

    Choose an analytic solution, substitute it into the governing equations, derive the
    source term that makes it exact, then verify that the observed convergence rate
    matches the design order.

## Why the source terms are derived symbolically

The step that goes wrong is the middle one. A hand-derived source term that is subtly
wrong makes a *correct* solver converge at the wrong order, and the ensuing bug hunt goes
looking in the solver. It is a particularly expensive failure because everything about it
looks like a solver defect.

SymPy differentiates the chosen solution through the operator and lambdifies both sides,
so the source cannot disagree with the exact solution by arithmetic slip. The tests then
close the loop the other way: they apply the operator *numerically* to the exact solution
and check it reproduces the source. Between them, a wrong manufactured problem fails in
this module rather than in the solver it was meant to verify.

## Choosing the solution

Two constraints, and they pull against each other:

- It must **not** lie in the discrete space. A manufactured solution a P1 element
  reproduces exactly gives zero error at every resolution, and the fitted order is then
  undefined rather than wrong — a study that silently proves nothing. Hence trigonometric
  content.
- It must be **smooth and resolvable** on the finest mesh, or the study never reaches its
  asymptotic range and the fit measures the coarse-mesh regime instead.

``modes`` is the knob between them: enough structure that the coarsest mesh cannot
resolve it, few enough that the finest comfortably can.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from vpl.core.constants import VACUUM_PERMITTIVITY
from vpl.core.units import magnitude_in

__all__ = [
    "ManufacturedSolution",
    "drift_diffusion_manufactured",
    "poisson_manufactured",
]

type Field = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_EPS0: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

#: Symbol for the sheath-normal coordinate, shared by every expression here.
_Z: Final[sp.Symbol] = sp.Symbol("z", real=True)


def _lambdify(expression: sp.Expr) -> Field:
    """Turn a SymPy expression into a NumPy-array callable.

    ``sympy.lambdify`` returns a scalar for a constant expression, which would silently
    break broadcasting at the one place a manufactured solution is degenerate. The
    wrapper forces an array of the caller's shape.
    """
    raw = sp.lambdify(_Z, expression, "numpy")

    def evaluate(z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.broadcast_to(np.asarray(raw(z), dtype=np.float64), np.shape(z)).astype(
            np.float64, copy=True
        )

    return evaluate


@dataclass(frozen=True, slots=True)
class ManufacturedSolution:
    """An exact solution and the source term that makes it satisfy the operator.

    Attributes:
        name: Short identifier, e.g. ``"V-01-poisson"``. Appears in the report.
        description: What operator this manufactures a solution for.
        exact: The analytic solution, evaluated on an array of positions.
        source: The source term that makes ``exact`` satisfy the operator.
        exact_expression: The solution in symbolic form, for the verification report.
            doc 07 §2.3 wants the manufactured problem *stated*, not merely asserted to
            have been used — a reader has to be able to re-derive it.
        source_expression: The derived source, likewise.
        boundary_values: ``exact`` at the two ends of the domain, since a Dirichlet
            problem needs them and computing them separately invites disagreement.
        coefficient: The leading operator coefficient, so a test can apply the operator
            numerically without re-deriving what it was built with.
        drift: Advection velocity, where the operator has one.
        diffusivity: Diffusion coefficient, where the operator has one.
    """

    name: str
    description: str
    exact: Field
    source: Field
    exact_expression: str
    source_expression: str
    boundary_values: tuple[float, float]
    coefficient: float
    drift: float = 0.0
    diffusivity: float = 0.0


def _validate_domain(length: float, modes: int) -> None:
    if length <= 0.0:
        raise ValueError(f"domain length must be positive, got {length}")
    if modes < 1:
        raise ValueError(f"need at least one mode to leave the discrete space, got {modes}")


def poisson_manufactured(*, length: float, amplitude: float, modes: int) -> ManufacturedSolution:
    """A manufactured solution for ``-eps0 d2Phi/dz2 = rho`` — verification test V-01.

    The chosen potential is a sine bump that vanishes at both ends and superposes
    ``modes`` harmonics with decaying weight, which gives the coarse mesh something it
    cannot resolve while staying smooth enough for the fine mesh to reach the asymptotic
    range.

    Args:
        length: Domain length in metres.
        amplitude: Peak potential in volts, so the manufactured problem sits at the same
            scale as the sheath it stands in for (doc 01 §2.1 biases to -250 V).
        modes: How many harmonics to superpose.
    """
    _validate_domain(length, modes)

    phi = sum(
        (amplitude / (k + 1) ** 2) * sp.sin((k + 1) * sp.pi * _Z / length) for k in range(modes)
    )
    rho = -_EPS0 * sp.diff(phi, _Z, 2)

    exact = _lambdify(sp.sympify(phi))
    return ManufacturedSolution(
        name=f"V-01-poisson-{modes}mode",
        description=(
            "Poisson operator -eps0 d2Phi/dz2 = rho with Dirichlet ends (doc 01 §3, "
            "doc 03 §3.1). This is the equation doc 01 §3 calls load-bearing: it is what "
            "converts observable densities into the potential nothing can measure."
        ),
        exact=exact,
        source=_lambdify(sp.sympify(rho)),
        exact_expression=str(sp.simplify(phi)),
        source_expression=str(sp.simplify(rho)),
        boundary_values=(
            float(exact(np.array([0.0]))[0]),
            float(exact(np.array([length]))[0]),
        ),
        coefficient=_EPS0,
    )


def drift_diffusion_manufactured(
    *, length: float, density: float, drift: float, diffusivity: float, modes: int
) -> ManufacturedSolution:
    """A manufactured solution for ``d/dz (u n - D dn/dz) = S`` — verification test V-02.

    This is the ion continuity equation of doc 03 §3.1 with the momentum closure folded
    into a constant drift, which is the operator whose *discretisation* doc 03 §3.4 warns
    about: centred differences oscillate in the strongly drift-dominated sheath, which is
    why Scharfetter-Gummel exponential fitting is specified. A manufactured solution is
    how that claim gets tested rather than trusted.

    The density is built as a positive baseline plus a bounded oscillation. A manufactured
    density that went negative is not a plasma state, and a positivity-preserving scheme
    would be penalised for refusing to reproduce something physically impossible.

    Args:
        length: Domain length in metres.
        density: Baseline density in m^-3.
        drift: Advection velocity in m/s.
        diffusivity: Diffusion coefficient in m^2/s.
        modes: How many harmonics to superpose.
    """
    _validate_domain(length, modes)
    if diffusivity <= 0.0:
        raise ValueError(f"diffusivity must be positive, got {diffusivity}")

    oscillation = sum(sp.sin((k + 1) * sp.pi * _Z / length) / (k + 1) ** 2 for k in range(modes))
    # Half the baseline as oscillation amplitude keeps n strictly positive: the harmonic
    # sum is bounded by sum 1/k^2 < pi^2/6 < 1.645, so n >= density * (1 - 0.3 * 1.645) > 0.
    n = density * (1 + sp.Rational(3, 10) * oscillation)
    flux = drift * n - diffusivity * sp.diff(n, _Z)
    source = sp.diff(flux, _Z)

    exact = _lambdify(sp.sympify(n))
    return ManufacturedSolution(
        name=f"V-02-drift-diffusion-{modes}mode",
        description=(
            "Drift-diffusion operator d/dz (u n - D dn/dz) = S, the ion continuity "
            "equation of doc 03 §3.1. doc 03 §3.4 specifies Scharfetter-Gummel "
            "exponential fitting because centred differences oscillate here; this is how "
            "that is tested rather than trusted."
        ),
        exact=exact,
        source=_lambdify(sp.sympify(source)),
        exact_expression=str(sp.simplify(n)),
        source_expression=str(sp.simplify(source)),
        boundary_values=(
            float(exact(np.array([0.0]))[0]),
            float(exact(np.array([length]))[0]),
        ),
        coefficient=diffusivity,
        drift=drift,
        diffusivity=diffusivity,
    )
