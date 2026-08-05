"""PIC field solve — deposition, Poisson, interpolation. doc 03 §4.2 steps 1-3.

These three steps are where a PIC code is most often subtly wrong, and where it is most
cheaply verified: each has a closed form. The particle push has none until several steps
have accumulated, so it is tested by conservation instead (see ``test_kinetic_push.py``).

The one rule that is not obvious and is load-bearing: **the same weighting must be used to
deposit charge and to interpolate the field back**. doc 03 §4.2 step 3 says so parenthetically
— "same weighting — required for momentum conservation" — and a code that uses, say, CIC
deposition with nearest-grid-point gathering exerts a net self-force on every particle. It
still runs, still looks like a plasma, and conserves nothing. That asymmetry is asserted here
directly rather than left to a conservation test to catch downstream.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from vpl.physics.kinetic.fields import (
    deposit_cic,
    gather_cic,
    solve_poisson_dirichlet,
)
from vpl.physics.kinetic.grid import UniformGrid


def _grid(*, length: float = 1.0, n_cells: int = 100) -> UniformGrid:
    return UniformGrid(length_m=length, n_cells=n_cells)


class TestTheUniformGrid:
    def test_the_cell_width_is_the_length_over_the_cell_count(self) -> None:
        grid = _grid(length=0.02, n_cells=1000)

        assert grid.dz_m == pytest.approx(2e-5)

    def test_there_is_one_more_node_than_cell(self) -> None:
        # Fields live on nodes, charge is deposited to nodes, and the two Dirichlet
        # boundaries are nodes. Off-by-one here is the classic PIC indexing bug.
        grid = _grid(n_cells=100)

        assert grid.n_nodes == 101
        assert grid.nodes_m.shape == (101,)

    def test_the_last_node_is_exactly_at_the_domain_length(self) -> None:
        grid = _grid(length=0.02, n_cells=1000)

        assert float(grid.nodes_m[-1]) == pytest.approx(0.02, rel=1e-12)

    def test_a_non_positive_cell_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cells"):
            UniformGrid(length_m=1.0, n_cells=0)

    def test_a_non_positive_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="length"):
            UniformGrid(length_m=0.0, n_cells=10)


class TestChargeDeposition:
    @pytest.mark.physics
    def test_a_particle_exactly_on_a_node_deposits_entirely_to_it(self) -> None:
        grid = _grid(n_cells=10)

        density = deposit_cic(grid, positions_m=jnp.array([0.3]), weights=jnp.array([1.0]))

        assert float(density[3]) == pytest.approx(1.0 / grid.dz_m)
        assert float(jnp.sum(jnp.abs(density.at[3].set(0.0)))) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.physics
    def test_a_particle_midway_between_nodes_splits_evenly(self) -> None:
        # The defining property of cloud-in-cell: linear weighting, so a particle halfway
        # between two nodes contributes half to each.
        grid = _grid(n_cells=10)

        density = deposit_cic(grid, positions_m=jnp.array([0.35]), weights=jnp.array([1.0]))

        assert float(density[3]) == pytest.approx(0.5 / grid.dz_m)
        assert float(density[4]) == pytest.approx(0.5 / grid.dz_m)

    @pytest.mark.physics
    def test_the_weighting_is_linear_in_the_offset(self) -> None:
        grid = _grid(n_cells=10)

        density = deposit_cic(grid, positions_m=jnp.array([0.32]), weights=jnp.array([1.0]))

        assert float(density[3]) == pytest.approx(0.8 / grid.dz_m)
        assert float(density[4]) == pytest.approx(0.2 / grid.dz_m)

    @pytest.mark.physics
    def test_total_charge_is_conserved_by_deposition(self) -> None:
        # The integral of the deposited density over the domain must equal the summed
        # particle weight, for any placement. A deposition that leaks charge silently
        # rescales the whole Poisson solve.
        grid = _grid(length=0.02, n_cells=64)
        rng = np.random.default_rng(20260805)
        positions = jnp.asarray(rng.uniform(0.0, 0.02, size=5000))
        weights = jnp.asarray(rng.uniform(0.5, 1.5, size=5000))

        density = deposit_cic(grid, positions_m=positions, weights=weights)

        # The exact CIC identity is sum(density) * dz, not the trapezoidal rule. CIC
        # distributes each particle's weight between two nodes, so summing the nodes
        # recovers the total exactly; the trapezoidal rule would half-weight the two end
        # nodes and lose about a cell's worth. This holds to zero relative error, and
        # asserting it exactly is what makes it a conservation test rather than a
        # tolerance test.
        assert float(jnp.sum(density)) * grid.dz_m == pytest.approx(
            float(jnp.sum(weights)), rel=1e-14
        )

    @pytest.mark.physics
    def test_the_boundary_nodes_hold_half_the_interior_density(self) -> None:
        # Not a defect: an end node owns half a cell, so a uniform plasma deposits half as
        # much onto it. It is pinned here because it looks like a wall depletion layer to
        # anyone plotting the density, and because it is the reason the Poisson solve reads
        # only interior nodes — the boundary values are imposed Dirichlet potentials, so
        # the artifact never enters the field.
        grid = _grid(length=0.02, n_cells=64)
        rng = np.random.default_rng(20260805)
        positions = jnp.asarray(rng.uniform(0.0, 0.02, size=200000))

        density = deposit_cic(grid, positions_m=positions, weights=jnp.ones(200000))

        interior_mean = float(jnp.mean(density[1:-1]))
        assert float(density[0]) == pytest.approx(0.5 * interior_mean, rel=0.05)
        assert float(density[-1]) == pytest.approx(0.5 * interior_mean, rel=0.05)

    def test_particles_outside_the_domain_are_refused(self) -> None:
        # Silently clamping an escaped particle would quietly create charge at the wall,
        # which is exactly where this project measures. Boundary handling is the caller's
        # job and must happen before deposition.
        grid = _grid()

        with pytest.raises(ValueError, match="outside"):
            deposit_cic(grid, positions_m=jnp.array([1.5]), weights=jnp.array([1.0]))

    def test_a_weight_array_of_the_wrong_length_is_refused(self) -> None:
        grid = _grid()

        with pytest.raises(ValueError, match="shape"):
            deposit_cic(grid, positions_m=jnp.array([0.1, 0.2]), weights=jnp.array([1.0]))


class TestThePoissonSolve:
    @pytest.mark.physics
    def test_it_reproduces_a_quadratic_potential_from_a_uniform_charge_exactly(self) -> None:
        # -eps0 d2Phi/dz2 = rho with constant rho has a parabolic solution. A second-order
        # centred difference is EXACT on a parabola, so this must agree to machine
        # precision, not merely to tolerance. Anything less means the stencil is wrong.
        from vpl.core.constants import VACUUM_PERMITTIVITY
        from vpl.core.units import magnitude_in

        eps0 = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))
        grid = _grid(length=0.02, n_cells=200)
        rho = 1e-6
        charge_density = jnp.full(grid.n_nodes, rho)

        phi = solve_poisson_dirichlet(grid, charge_density, left_v=0.0, right_v=0.0)

        z = grid.nodes_m
        exact = rho * z * (grid.length_m - z) / (2.0 * eps0)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(exact), rtol=1e-10, atol=1e-10)

    @pytest.mark.physics
    def test_it_honours_both_dirichlet_boundary_values(self) -> None:
        # The wall bias enters here and nowhere else. If the boundary is not applied
        # exactly, every sheath potential in the project is offset.
        grid = _grid(length=0.02, n_cells=50)

        phi = solve_poisson_dirichlet(grid, jnp.zeros(grid.n_nodes), left_v=-250.0, right_v=0.0)

        assert float(phi[0]) == pytest.approx(-250.0)
        assert float(phi[-1]) == pytest.approx(0.0)

    @pytest.mark.physics
    def test_with_no_charge_the_potential_is_linear(self) -> None:
        grid = _grid(length=0.02, n_cells=50)

        phi = solve_poisson_dirichlet(grid, jnp.zeros(grid.n_nodes), left_v=-250.0, right_v=0.0)

        expected = -250.0 * (1.0 - grid.nodes_m / grid.length_m)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(expected), rtol=1e-9, atol=1e-9)

    @pytest.mark.physics
    def test_it_converges_at_second_order_on_a_non_polynomial_source(self) -> None:
        # The parabola test above cannot detect a first-order stencil, because a parabola
        # is in the space either way. A sinusoidal source can.
        from vpl.core.constants import VACUUM_PERMITTIVITY
        from vpl.core.units import magnitude_in
        from vpl.validation.convergence import RefinementLevel, assert_design_order, observed_order

        eps0 = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))
        length = 0.02
        levels = []
        for n_cells in (50, 100, 200, 400):
            grid = _grid(length=length, n_cells=n_cells)
            z = grid.nodes_m
            k = jnp.pi / length
            exact = jnp.sin(k * z)
            charge_density = eps0 * k**2 * exact

            phi = solve_poisson_dirichlet(grid, charge_density, left_v=0.0, right_v=0.0)

            error = float(jnp.max(jnp.abs(phi - exact)))
            levels.append(RefinementLevel(h=grid.dz_m, error=error))

        assert_design_order(observed_order(levels), design_order=2.0)

    def test_a_charge_density_of_the_wrong_length_is_refused(self) -> None:
        grid = _grid(n_cells=10)

        with pytest.raises(ValueError, match="node"):
            solve_poisson_dirichlet(grid, jnp.zeros(5), left_v=0.0, right_v=0.0)


class TestFieldGather:
    @pytest.mark.physics
    def test_a_uniform_field_is_recovered_everywhere(self) -> None:
        grid = _grid(n_cells=10)
        field = jnp.full(grid.n_nodes, -3.5)

        gathered = gather_cic(grid, field, positions_m=jnp.array([0.0, 0.137, 0.5, 0.999]))

        np.testing.assert_allclose(np.asarray(gathered), -3.5, rtol=1e-12)

    @pytest.mark.physics
    def test_a_linear_field_is_interpolated_exactly(self) -> None:
        grid = _grid(n_cells=10)
        field = 7.0 * grid.nodes_m

        gathered = gather_cic(grid, field, positions_m=jnp.array([0.23, 0.61]))

        np.testing.assert_allclose(np.asarray(gathered), [7.0 * 0.23, 7.0 * 0.61], rtol=1e-10)

    @pytest.mark.physics
    def test_gather_is_the_transpose_of_deposit(self) -> None:
        # This is the momentum-conservation requirement of doc 03 §4.2 step 3, stated as
        # the linear-algebra identity it actually is: if S is the deposition matrix, the
        # gather must be S^T. Then the self-force vanishes identically.
        #
        # Checked as <gather(F), w> == <F, deposit(w)> * dz for arbitrary F and w, which
        # holds iff the two use the same weights.
        grid = _grid(length=0.02, n_cells=32)
        rng = np.random.default_rng(7)
        positions = jnp.asarray(rng.uniform(0.0, 0.02, size=200))
        weights = jnp.asarray(rng.uniform(0.1, 2.0, size=200))
        field = jnp.asarray(rng.normal(size=grid.n_nodes))

        lhs = float(jnp.sum(gather_cic(grid, field, positions_m=positions) * weights))
        rhs = float(
            jnp.sum(field * deposit_cic(grid, positions_m=positions, weights=weights)) * grid.dz_m
        )

        assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_a_field_array_of_the_wrong_length_is_refused(self) -> None:
        grid = _grid(n_cells=10)

        with pytest.raises(ValueError, match="node"):
            gather_cic(grid, jnp.zeros(5), positions_m=jnp.array([0.1]))
