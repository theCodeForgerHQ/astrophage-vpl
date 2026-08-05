"""Double precision is a correctness requirement here, not a preference.

JAX defaults to **float32**, silently. Running this PIC in single precision is not a small
accuracy cost: 13 of the 21 field tests fail outright, the Poisson solve loses the exactness
that makes the parabola test meaningful, and the second-order convergence study reports
garbage because the discretisation error falls below the round-off floor almost immediately.

The failure mode that matters is subtler than a failing test, though. A 65 800-step particle
push accumulates position error every step; in float32 the accumulated phase error over ten
RF periods is comparable to a cell width, so particles arrive at the wall at the wrong time
and the IEDF — the deliverable — is quietly wrong while every moment still looks plausible.

So precision is enabled by the package at import, and pinned here. The alternative, an
environment variable in CI, is exactly the kind of configuration that travels badly and is
absent the one time somebody runs a script by hand.
"""

from __future__ import annotations

import jax.numpy as jnp

import vpl.physics.kinetic  # noqa: F401  — the import is what enables x64
from vpl.physics.kinetic.fields import solve_poisson_dirichlet
from vpl.physics.kinetic.grid import UniformGrid


def test_importing_the_kinetic_package_enables_double_precision() -> None:
    assert jnp.zeros(1).dtype == jnp.float64


def test_the_grid_nodes_are_double_precision() -> None:
    assert UniformGrid(length_m=0.02, n_cells=10).nodes_m.dtype == jnp.float64


def test_the_poisson_solution_is_double_precision() -> None:
    grid = UniformGrid(length_m=0.02, n_cells=10)

    phi = solve_poisson_dirichlet(grid, jnp.zeros(grid.n_nodes), left_v=-250.0, right_v=0.0)

    assert phi.dtype == jnp.float64
