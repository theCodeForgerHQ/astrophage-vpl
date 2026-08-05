"""Particle push and stability constraints — doc 03 §4.2 step 4, §4.3.

The push has no closed form on its own once the field responds to the particles, so it is
verified in three escalating stages:

1. **Kinematics in a frozen field.** Exact, and catches a wrong sign or a factor of two in
   the charge-to-mass ratio — which is the single most common PIC bug and the one that looks
   most like plausible physics.
2. **Cold plasma oscillation.** The field solve and the push *together*, against
   ``omega_pe = sqrt(n e^2 / (eps0 m))`` — a closed form neither of them can influence. This
   is the one place in the kernel where theory hands over an exact number, and ADR-003 leans
   on it because the Smilei cross-check it originally specified is deferred.
3. **Energy conservation**, verification test V-07 of doc 03 §7: total energy drift below
   0.1 % over a collisionless run.

Stage 2 is the load-bearing one. A code can pass stage 1 with the deposition and the gather
using different weightings, and pass stage 3 by being uniformly too cold. Only the
oscillation frequency pins the coupling constant between them.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.units import magnitude_in
from vpl.physics.kinetic.constraints import StabilityViolationError, check_stability, debye_length_m
from vpl.physics.kinetic.grid import UniformGrid
from vpl.physics.kinetic.push import Species, push_leapfrog

_E = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E = float(magnitude_in(ELECTRON_MASS, "kg"))
_EPS0 = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))


def _electrons(n: int) -> Species:
    return Species(name="e", charge_c=-_E, mass_kg=_M_E, weight=1.0)


class TestTheStabilityConstraints:
    """doc 03 §4.3 — 'hard constraints, checked at runtime, and violating them is a runtime
    error rather than a warning'. Implemented exactly that way."""

    def test_the_debye_length_matches_the_closed_form(self) -> None:
        # doc 01 §2.1's RP-1: n_e = 1e17 m^-3, T_e = 3 eV -> lambda_D = 40.7 um, and
        # doc 03 §4.3 sets dz = lambda_D/2 = 20 um from it.
        lambda_d = debye_length_m(density_m3=1e17, electron_temperature_ev=3.0)

        assert lambda_d == pytest.approx(4.07e-5, rel=2e-2)

    @pytest.mark.physics
    def test_a_cell_wider_than_a_debye_length_is_a_runtime_error(self) -> None:
        # The constraint whose violation causes numerical grid heating: the plasma warms
        # with no physics doing it, until the Debye length grows to match the cell. It is
        # a runtime error because the run that violates it still produces output.
        lambda_d = debye_length_m(density_m3=1e17, electron_temperature_ev=3.0)
        grid = UniformGrid(length_m=0.02, n_cells=100)  # dz = 200 um >> lambda_D

        with pytest.raises(StabilityViolationError, match="Debye"):
            check_stability(
                grid,
                dt_s=1e-12,
                density_m3=1e17,
                electron_temperature_ev=3.0,
                n_ppc=1000,
                v_max_m_per_s=1e6,
            )
        assert grid.dz_m > lambda_d

    @pytest.mark.physics
    def test_a_timestep_that_underresolves_the_plasma_frequency_is_refused(self) -> None:
        # dt <= 0.2 / omega_pe. At RP-1 omega_pe = 1.78e10 /s, so dt <= 11.2 ps.
        grid = UniformGrid(length_m=0.02, n_cells=1000)

        with pytest.raises(StabilityViolationError, match="plasma frequency"):
            check_stability(
                grid,
                dt_s=1e-10,
                density_m3=1e17,
                electron_temperature_ev=3.0,
                n_ppc=1000,
                v_max_m_per_s=1e6,
            )

    @pytest.mark.physics
    def test_a_timestep_violating_the_cfl_condition_is_refused(self) -> None:
        # dt <= dz / v_max: no particle may cross more than one cell in a step, or the
        # deposition skips cells and the field it feels is not the field it moved through.
        grid = UniformGrid(length_m=0.02, n_cells=1000)

        with pytest.raises(StabilityViolationError, match="CFL"):
            check_stability(
                grid,
                dt_s=1e-11,
                density_m3=1e17,
                electron_temperature_ev=3.0,
                n_ppc=1000,
                v_max_m_per_s=1e9,
            )

    def test_too_few_particles_per_cell_is_refused(self) -> None:
        grid = UniformGrid(length_m=0.02, n_cells=1000)

        with pytest.raises(StabilityViolationError, match="particles per cell"):
            check_stability(
                grid,
                dt_s=1e-12,
                density_m3=1e17,
                electron_temperature_ev=3.0,
                n_ppc=10,
                v_max_m_per_s=1e6,
            )

    def test_the_rp1_configuration_of_doc_03_section_4_4_passes(self) -> None:
        # The configuration the document specifies must satisfy the constraints the same
        # document sets. If it does not, one of the two is wrong and it is worth knowing
        # before a 65 800-step run.
        grid = UniformGrid(length_m=0.02, n_cells=1000)

        check_stability(
            grid,
            dt_s=11.2e-12,
            density_m3=1e17,
            electron_temperature_ev=3.0,
            n_ppc=1000,
            v_max_m_per_s=1.5e6,
        )

    def test_the_violation_message_reports_the_offending_and_permitted_values(self) -> None:
        grid = UniformGrid(length_m=0.02, n_cells=100)

        with pytest.raises(StabilityViolationError) as excinfo:
            check_stability(
                grid,
                dt_s=1e-12,
                density_m3=1e17,
                electron_temperature_ev=3.0,
                n_ppc=1000,
                v_max_m_per_s=1e6,
            )

        # Content, not formatting: the reader of a failed ensemble point is usually not the
        # person who configured it, so the message has to carry the offending value, the
        # bound it broke, the consequence, and the fix.
        message = str(excinfo.value)
        assert "2.000e-04" in message, "the offending cell width"
        assert "4.072e-05" in message, "the Debye length it exceeded"
        assert "grid heating" in message, "the consequence"
        assert "492 cells" in message, "the actionable fix"


class TestTheLeapfrogPushInAFrozenField:
    @pytest.mark.physics
    def test_a_constant_field_produces_exact_uniform_acceleration(self) -> None:
        # Leapfrog is exact for constant acceleration in velocity, and exact in position
        # too when the half-step offset is honoured. Any deviation here is a wrong q/m.
        species = _electrons(1)
        field = -1000.0  # V/m
        dt = 1e-12
        z = jnp.array([0.01])
        v = jnp.zeros((1, 3))

        _, v1 = push_leapfrog(
            species,
            positions_m=z,
            velocities_m_per_s=v,
            e_field_v_per_m=jnp.array([field]),
            dt_s=dt,
        )

        expected_vz = species.charge_c * field / species.mass_kg * dt
        assert float(v1[0, 2]) == pytest.approx(expected_vz, rel=1e-12)

    @pytest.mark.physics
    def test_an_electron_accelerates_towards_positive_potential(self) -> None:
        # Sign discipline, checked explicitly. E = -dPhi/dz, and a negative charge in a
        # field pointing in -z moves in +z. Getting this backwards inverts the sheath and
        # every flux the project reports, while still looking like a plasma.
        species = _electrons(1)
        _, v1 = push_leapfrog(
            species,
            positions_m=jnp.array([0.01]),
            velocities_m_per_s=jnp.zeros((1, 3)),
            e_field_v_per_m=jnp.array([-1000.0]),
            dt_s=1e-12,
        )

        assert float(v1[0, 2]) > 0.0

    @pytest.mark.physics
    def test_the_transverse_velocities_are_untouched_by_the_field(self) -> None:
        # 1D3V electrostatic: the field has only a z component, so v_x and v_y change only
        # through collisions. They are carried because LIF measures a projected velocity
        # and Thomson the full electron distribution (doc 03 §4.1).
        species = _electrons(1)
        v = jnp.array([[1e5, -2e5, 0.0]])

        _, v1 = push_leapfrog(
            species,
            positions_m=jnp.array([0.01]),
            velocities_m_per_s=v,
            e_field_v_per_m=jnp.array([500.0]),
            dt_s=1e-12,
        )

        assert float(v1[0, 0]) == pytest.approx(1e5)
        assert float(v1[0, 1]) == pytest.approx(-2e5)

    @pytest.mark.physics
    def test_position_advances_by_the_updated_velocity(self) -> None:
        species = _electrons(1)
        z0 = 0.01
        z1, _ = push_leapfrog(
            species,
            positions_m=jnp.array([z0]),
            velocities_m_per_s=jnp.array([[0.0, 0.0, 3e5]]),
            e_field_v_per_m=jnp.array([0.0]),
            dt_s=1e-12,
        )

        assert float(z1[0]) == pytest.approx(z0 + 3e5 * 1e-12, rel=1e-12)

    def test_a_field_array_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            push_leapfrog(
                _electrons(2),
                positions_m=jnp.array([0.01, 0.02]),
                velocities_m_per_s=jnp.zeros((2, 3)),
                e_field_v_per_m=jnp.array([0.0]),
                dt_s=1e-12,
            )

    def test_a_velocity_array_that_is_not_three_dimensional_is_refused(self) -> None:
        with pytest.raises(ValueError, match="3"):
            push_leapfrog(
                _electrons(1),
                positions_m=jnp.array([0.01]),
                velocities_m_per_s=jnp.zeros((1, 2)),
                e_field_v_per_m=jnp.array([0.0]),
                dt_s=1e-12,
            )


class TestTheSpeciesType:
    def test_it_carries_the_charge_to_mass_ratio(self) -> None:
        species = _electrons(1)

        assert species.charge_to_mass == pytest.approx(-_E / _M_E)

    def test_a_massless_species_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            Species(name="ghost", charge_c=_E, mass_kg=0.0, weight=1.0)

    def test_a_non_positive_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            Species(name="e", charge_c=-_E, mass_kg=_M_E, weight=0.0)
