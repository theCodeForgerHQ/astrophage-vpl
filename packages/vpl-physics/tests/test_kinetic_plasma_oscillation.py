"""Cold plasma oscillation — the external check on the assembled kernel.

This is the most important verification test in the kinetic layer, and ADR-003 says why.
The decision to build a PIC kernel rather than adopt Smilei was taken with the Smilei
cross-check as its mitigation for risk RT-05, and that cross-check is **deferred** for the
two-day build window. What is left has to carry the weight, and self-consistency will not:
ADR-011 is this project's own worked example of a solver that agreed with itself 1 488 times
and was 52 % wrong.

So the check has to be against something the kernel cannot influence. Cold plasma
oscillation is the one place in electrostatic PIC where theory hands over an exact number:

    omega_pe = sqrt(n e^2 / (eps0 m_e))

Displace a cold electron slab against a fixed neutralising ion background and it oscillates
at exactly that frequency, independent of the displacement amplitude, the grid, the particle
count and the timestep. Nothing in the code knows this number — ``omega_pe`` never appears in
``fields.py`` or ``push.py`` — so recovering it tests the *product* of the deposition
constant, the Poisson scaling, the gather and the charge-to-mass ratio.

That product is exactly what the individual unit tests cannot reach. Each of them fixes one
factor; only this one fixes the chain. A code with deposition too large by two and a Poisson
constant too small by two passes every test in ``test_kinetic_fields.py`` and fails here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from vpl.core.units import magnitude_in
from vpl.physics.kinetic.constraints import electron_plasma_frequency_rad_per_s
from vpl.physics.kinetic.fields import deposit_cic, gather_cic, solve_poisson_dirichlet
from vpl.physics.kinetic.grid import UniformGrid
from vpl.physics.kinetic.push import Species, push_leapfrog

_E = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E = float(magnitude_in(ELECTRON_MASS, "kg"))


def _run_oscillation(
    *,
    density_m3: float,
    n_cells: int,
    particles_per_cell: int,
    n_steps: int,
    steps_per_period: int,
    amplitude_fraction: float,
    mass_scale: float = 1.0,
) -> tuple[Array, float]:
    """Displace a cold electron slab and record the field energy each step.

    The ions are a *fixed* uniform neutralising background rather than particles. That is
    the standard setup for this test and it is deliberate: mobile ions would add the ion
    plasma frequency to the signal, and at a mass ratio of 73 000 the two are well separated
    but the fit would no longer be against a single clean closed form.
    """
    omega_pe = electron_plasma_frequency_rad_per_s(density_m3=density_m3)
    dt = 2.0 * np.pi / omega_pe / steps_per_period

    length = 0.001
    grid = UniformGrid(length_m=length, n_cells=n_cells)
    n_particles = n_cells * particles_per_cell
    # One computational particle stands for this many electrons, chosen so the marker
    # population reproduces the physical density exactly.
    weight = density_m3 * length / n_particles

    # Uniform loading, then a sinusoidal displacement that vanishes at both walls so the
    # Dirichlet boundaries are consistent with the perturbation.
    z0 = (jnp.arange(n_particles, dtype=jnp.float64) + 0.5) * (length / n_particles)
    displacement = amplitude_fraction * grid.dz_m * jnp.sin(jnp.pi * z0 / length)
    positions = jnp.clip(z0 + displacement, 0.0, length)
    velocities = jnp.zeros((n_particles, 3), dtype=jnp.float64)

    # mass_scale is 1.0 for every real run. It exists so the negative control can perturb
    # the one constant the closed form is most sensitive to, without monkeypatching.
    electrons = Species(name="e", charge_c=-_E, mass_kg=_M_E * mass_scale, weight=weight)
    ion_background = density_m3 * _E  # C/m^3, uniform and fixed

    energies = []
    for _ in range(n_steps):
        electron_density = deposit_cic(
            grid, positions_m=positions, weights=jnp.full(n_particles, weight)
        )
        charge_density = ion_background - _E * electron_density
        phi = solve_poisson_dirichlet(grid, charge_density, left_v=0.0, right_v=0.0)
        # E = -dPhi/dz, second-order interior, one-sided at the two boundary nodes.
        e_field = -jnp.gradient(phi, grid.dz_m)
        energies.append(float(jnp.sum(e_field**2)) * grid.dz_m)

        gathered = gather_cic(grid, e_field, positions_m=positions)
        positions, velocities = push_leapfrog(
            electrons,
            positions_m=positions,
            velocities_m_per_s=velocities,
            e_field_v_per_m=gathered,
            dt_s=dt,
        )
        positions = jnp.clip(positions, 0.0, length)

    return jnp.asarray(energies), dt


def _dominant_frequency_rad_per_s(signal: Array, dt: float) -> float:
    """Peak frequency of a real signal, with its mean removed.

    The field *energy* oscillates at ``2 omega_pe`` — it goes as the square of the field —
    so the caller halves it. Stating that here rather than silently dividing keeps the
    factor of two visible, since a stray factor of two is exactly what this test exists to
    catch.
    """
    centred = np.asarray(signal) - float(np.mean(signal))
    spectrum = np.abs(np.fft.rfft(centred))
    frequencies = np.fft.rfftfreq(len(centred), d=dt)
    peak = int(np.argmax(spectrum[1:]) + 1)
    return float(2.0 * np.pi * frequencies[peak])


class TestColdPlasmaOscillation:
    @pytest.mark.physics
    def test_the_oscillation_frequency_is_the_electron_plasma_frequency(self) -> None:
        # The headline. omega_pe appears nowhere in fields.py or push.py, so recovering it
        # tests the product of every constant in the chain at once.
        density = 1e16
        expected = electron_plasma_frequency_rad_per_s(density_m3=density)

        energies, dt = _run_oscillation(
            density_m3=density,
            n_cells=64,
            particles_per_cell=64,
            n_steps=512,
            steps_per_period=32,
            amplitude_fraction=0.05,
        )

        measured = _dominant_frequency_rad_per_s(energies, dt) / 2.0

        assert measured == pytest.approx(expected, rel=0.05), (
            f"measured {measured:.4e} rad/s against the closed form {expected:.4e} rad/s"
        )

    @pytest.mark.physics
    def test_the_frequency_does_not_depend_on_the_density_used_to_set_it(self) -> None:
        # omega_pe scales as sqrt(n). Checking two densities a decade apart confirms the
        # scaling rather than a single coincidence, which a compensating pair of errors
        # could otherwise produce.
        ratios = []
        for density in (1e16, 1e17):
            expected = electron_plasma_frequency_rad_per_s(density_m3=density)
            energies, dt = _run_oscillation(
                density_m3=density,
                n_cells=64,
                particles_per_cell=32,
                n_steps=512,
                steps_per_period=32,
                amplitude_fraction=0.05,
            )
            ratios.append(_dominant_frequency_rad_per_s(energies, dt) / 2.0 / expected)

        np.testing.assert_allclose(ratios, 1.0, rtol=0.06)

    @pytest.mark.physics
    def test_the_oscillation_amplitude_does_not_grow(self) -> None:
        # A leapfrog on a symplectic system has bounded energy error. Growth here would be
        # numerical heating — the failure doc 03 §4.3's dz <= lambda_D constraint exists to
        # prevent — and it is worth catching directly rather than inferring it.
        energies, _ = _run_oscillation(
            density_m3=1e16,
            n_cells=64,
            particles_per_cell=64,
            n_steps=512,
            steps_per_period=32,
            amplitude_fraction=0.05,
        )

        first_half = float(jnp.max(energies[:256]))
        second_half = float(jnp.max(energies[256:]))

        assert second_half < 1.5 * first_half, (
            f"field energy grew from {first_half:.3e} to {second_half:.3e} — numerical heating"
        )

    @pytest.mark.physics
    def test_a_deliberately_wrong_electron_mass_fails_the_test(self) -> None:
        # Proof the test has teeth. omega_pe goes as 1/sqrt(m), so a mass wrong by four
        # halves the frequency — far outside the 5 % gate. A benchmark nobody has seen
        # reject anything is not a benchmark, and ADR-011 is this project's reminder of
        # what that costs.
        density = 1e16
        expected = electron_plasma_frequency_rad_per_s(density_m3=density)

        energies, dt = _run_oscillation(
            density_m3=density,
            n_cells=64,
            particles_per_cell=32,
            n_steps=512,
            steps_per_period=32,
            amplitude_fraction=0.05,
            mass_scale=4.0,
        )

        measured = _dominant_frequency_rad_per_s(energies, dt) / 2.0

        assert measured != pytest.approx(expected, rel=0.05)
        # And specifically, it should be low by the factor the closed form predicts.
        assert measured == pytest.approx(expected / 2.0, rel=0.10)
