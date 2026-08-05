"""Deposition, Poisson and gather — doc 03 §4.2 steps 1-3.

## The one rule that matters

doc 03 §4.2 step 3 carries a parenthesis that is easy to read past:

    3. Interpolate field to particles  (same weighting — required for momentum conservation)

It is the single most important line in the algorithm. Deposition is a linear map ``S`` from
particle weights to node densities; the gather must be ``S^T``. When it is, the force a
particle exerts on itself through the grid cancels exactly and total momentum is conserved
to round-off. When it is not — CIC deposition with nearest-grid-point gathering, say — every
particle pushes itself. The code still runs, still looks like a plasma, and conserves
nothing. ``test_gather_is_the_transpose_of_deposit`` asserts the identity directly rather
than hoping a downstream conservation test notices.

## Why the Poisson solve is direct

The 1-D Poisson matrix is tridiagonal, so a direct solve is ``O(n)`` and exact. There is no
reason to iterate, and an iterative solver would introduce a tolerance that silently becomes
part of the physics — the sheath potential is the quantity nothing can measure (doc 01 §3)
and the one the whole inversion is built to recover, so it should not carry a convergence
knob.

JAX has no tridiagonal solve that is both differentiable and available on every backend, so
the Thomas algorithm is written out. It is twenty lines and exact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from vpl.core.constants import VACUUM_PERMITTIVITY
from vpl.core.units import magnitude_in
from vpl.physics.kinetic.grid import UniformGrid

__all__ = ["deposit_cic", "gather_cic", "solve_poisson_dirichlet"]

#: Vacuum permittivity in F/m, from CODATA via vpl.core.constants.
_EPS0: float = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

#: Relative slack on the domain-bounds check. Purely a floating-point allowance — a particle
#: pushed exactly onto the wall can land a few ulps outside it — and deliberately far tighter
#: than a cell, so a genuinely escaped particle is still caught.
_BOUNDS_TOLERANCE: float = 1e-9


def _cell_and_offset(grid: UniformGrid, positions_m: Array) -> tuple[Array, Array]:
    """Left node index and the fractional offset into its cell, for each particle.

    The index is clamped to ``n_cells - 1`` so that a particle sitting exactly on the last
    node lands in the final cell with offset 1 rather than indexing past the array. That is
    the only clamp in this module and it is a representation detail, not a physics one — a
    particle genuinely outside the domain is rejected before it gets here.
    """
    scaled = positions_m / grid.dz_m
    index = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, grid.n_cells - 1)
    return index, scaled - index


def _check_inside(grid: UniformGrid, positions_m: Array) -> None:
    """Reject particles outside ``[0, length]``.

    Not a defensive nicety. Silently clamping an escaped particle deposits its charge onto
    the wall node — which is precisely where this project measures the ion energy flux — so
    a lost particle would masquerade as wall flux. Boundary handling is the caller's job and
    must run before deposition.
    """
    if positions_m.size == 0:
        return
    low = float(jnp.min(positions_m))
    high = float(jnp.max(positions_m))
    tolerance = _BOUNDS_TOLERANCE * grid.length_m
    if low < -tolerance or high > grid.length_m + tolerance:
        raise ValueError(
            f"particles lie outside the domain [0, {grid.length_m}]: "
            f"range is [{low}, {high}]. Apply boundary conditions before depositing."
        )


def deposit_cic(grid: UniformGrid, *, positions_m: Array, weights: Array) -> Array:
    """Cloud-in-cell deposition of particle weights onto node densities.

    Returns a density per unit length, so that ``trapezoid(result, dz)`` recovers the total
    deposited weight. Dividing by ``dz`` here rather than at the call site keeps the units
    of the returned array honest: it is a density, and the Poisson solve consumes it as one.

    Args:
        grid: The uniform grid.
        positions_m: Particle positions, all within ``[0, length]``.
        weights: Physical particles represented by each computational particle. Separate
            from position because the two species carry different weights and because
            variable weighting is how a future version would resolve the tail.
    """
    if weights.shape != positions_m.shape:
        raise ValueError(
            f"weights shape {weights.shape} does not match positions shape {positions_m.shape}"
        )
    _check_inside(grid, positions_m)

    index, offset = _cell_and_offset(grid, positions_m)
    density = jnp.zeros(grid.n_nodes, dtype=jnp.float64)
    density = density.at[index].add(weights * (1.0 - offset))
    density = density.at[index + 1].add(weights * offset)
    return density / grid.dz_m


def gather_cic(grid: UniformGrid, node_values: Array, *, positions_m: Array) -> Array:
    """Interpolate a node-centred field to particle positions — the transpose of deposition.

    Same linear weights as :func:`deposit_cic`, which is what makes the self-force vanish.
    See the module docstring; this is doc 03 §4.2 step 3's parenthesis.
    """
    if node_values.shape != (grid.n_nodes,):
        raise ValueError(f"expected {grid.n_nodes} node values, got shape {node_values.shape}")
    _check_inside(grid, positions_m)

    index, offset = _cell_and_offset(grid, positions_m)
    return node_values[index] * (1.0 - offset) + node_values[index + 1] * offset


@jax.jit
def _thomas(lower: Array, diagonal: Array, upper: Array, rhs: Array) -> Array:
    """Thomas algorithm for a tridiagonal system — exact, ``O(n)``, no tolerance.

    Written out because JAX offers no tridiagonal solve that is available on every backend.
    A dense ``linalg.solve`` would work and would be ``O(n^3)``; at 1 000 cells inside a
    65 800-step loop that is the difference between minutes and days.
    """

    def forward(
        carry: tuple[Array, Array], row: tuple[Array, Array, Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        c_prev, d_prev = carry
        a, b, c, d = row
        denominator = b - a * c_prev
        c_new = c / denominator
        d_new = (d - a * d_prev) / denominator
        return (c_new, d_new), (c_new, d_new)

    init = (jnp.array(0.0, dtype=jnp.float64), jnp.array(0.0, dtype=jnp.float64))
    _, (c_star, d_star) = jax.lax.scan(forward, init, (lower, diagonal, upper, rhs))

    def backward(x_next: Array, row: tuple[Array, Array]) -> tuple[Array, Array]:
        c, d = row
        x = d - c * x_next
        return x, x

    _, solution = jax.lax.scan(
        backward, jnp.array(0.0, dtype=jnp.float64), (c_star, d_star), reverse=True
    )
    return solution


def solve_poisson_dirichlet(
    grid: UniformGrid, charge_density: Array, *, left_v: float, right_v: float
) -> Array:
    """Solve ``-eps0 d2Phi/dz2 = rho`` with Dirichlet ends — doc 03 §4.2 step 2.

    The second-order centred stencil is exact on a parabola, which is why the uniform-charge
    test asserts machine precision rather than a tolerance; a first-order or mis-scaled
    stencil cannot pass it. The sinusoidal convergence test is what distinguishes second
    order from merely-exact-on-parabolas.

    Args:
        grid: The uniform grid.
        charge_density: Net charge density at the nodes, C/m^3.
        left_v: Potential at ``z = 0``, the biased wall. This is where doc 01 §2.1's -250 V
            enters the simulation, and the only place it does.
        right_v: Potential at the bulk boundary.
    """
    if charge_density.shape != (grid.n_nodes,):
        raise ValueError(
            f"expected {grid.n_nodes} node charge densities, got shape {charge_density.shape}"
        )

    n_interior = grid.n_cells - 1
    if n_interior < 1:
        return jnp.array([left_v, right_v], dtype=jnp.float64)

    dz2 = grid.dz_m**2
    # -eps0 (phi[i-1] - 2 phi[i] + phi[i+1]) / dz^2 = rho[i], written as a positive-definite
    # system in the interior unknowns with the boundary values folded into the right side.
    diagonal = jnp.full(n_interior, 2.0 * _EPS0 / dz2, dtype=jnp.float64)
    off = jnp.full(n_interior, -_EPS0 / dz2, dtype=jnp.float64)
    lower = off.at[0].set(0.0)
    upper = off.at[-1].set(0.0)

    rhs = charge_density[1:-1]
    rhs = rhs.at[0].add(_EPS0 * left_v / dz2)
    rhs = rhs.at[-1].add(_EPS0 * right_v / dz2)

    interior = _thomas(lower, diagonal, upper, rhs)
    return jnp.concatenate(
        [jnp.array([left_v], dtype=jnp.float64), interior, jnp.array([right_v], dtype=jnp.float64)]
    )
