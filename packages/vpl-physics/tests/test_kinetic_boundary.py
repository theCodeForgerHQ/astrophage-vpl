"""Boundary conditions — doc 03 §4.2 step 6, doc 03 §3.3.

The boundary is where this project's deliverable is actually produced. doc 01 §1.1 makes
the ion energy flux at the surface the quantity of interest, and every ion that reaches
``z = 0`` contributes to it exactly once. So the failures worth guarding against here are
bookkeeping failures, and they are all of the same shape: **a particle counted twice, or
not at all, still produces an IEDF.**

The assertions below are therefore against statements the module cannot influence:

1. **Energy is what kinematics says it is.** An absorbed macroparticle's recorded energy
   is ``m |v|^2 / 2`` from the velocity it carried, checked against the velocity that was
   handed in — not against a total the module also computed.
2. **Specular reflection is an isometry.** ``|v|`` is unchanged and ``v_z`` is negated,
   which is checkable per particle rather than in aggregate.
3. **The inflow sampler reproduces its own closed form.** The flux-weighted inflow speed
   distribution ``p(w) ~ w exp(-(w-u)^2 / 2 sigma^2)`` is a printed expression
   (Garcia & Wagner 2006); the sampler is tested against it by Kolmogorov-Smirnov with a
   numerically integrated CDF, and its mean against a quadrature of the same density.
   Neither reference is produced by the sampler.
4. **The one-way flux formula is checked against the sampler.** ``Gamma / n`` is the mean
   of ``|v_z|`` over the *number*-weighted Maxwellian restricted to inward motion, which
   is an independent route to the same number.

Point 3 matters more than it looks. A number-weighted half-Maxwellian passes every
"velocities are inward and roughly thermal" test and injects the wrong distribution — the
classic inflow-boundary error, and one that shifts the entry energy of every ion in the
run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate, stats

from vpl.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from vpl.core.params import default_registry
from vpl.core.random import Stream, generator
from vpl.core.units import magnitude_in
from vpl.physics.kinetic.boundary import (
    DEFAULT_ION_REFLECTION,
    DEFAULT_SECONDARY_YIELD,
    SECONDARY_EMISSION_ENERGY_EV,
    InjectionSource,
    SurfaceModel,
    WallModel,
    apply_boundaries,
    sample_inflow_velocities,
    sample_secondary_velocities,
)

_E = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E = float(magnitude_in(ELECTRON_MASS, "kg"))
_M_AR = float(default_registry().value_in("species.Ar.mass", "kg"))

#: One root seed for the whole file, so a failure is reproducible by re-running it.
_SEED = 20260805

#: Domain length used by the wall tests. Any positive length works; a millimetre is the
#: scale of the sheath at RP-1 (doc 01 §2.2) and keeps the numbers readable.
_LENGTH_M = 1.0e-3

#: How many standard errors a Monte-Carlo assertion is allowed, matching
#: ``test_kinetic_collisions.py``: 4 sigma is a 6e-5 two-sided false-alarm rate.
_SIGMA = 4.0

#: Kolmogorov-Smirnov critical value at the 0.1 % level, ``D_crit = 1.95 / sqrt(n)``.
_KS_CRITICAL = 1.95


def _rng() -> np.random.Generator:
    return generator(_SEED, Stream.PLASMA_INIT)


def _flux_density(w: NDArray[np.float64], *, u: float, sigma: float) -> NDArray[np.float64]:
    """The unnormalised flux-weighted inflow speed density ``w exp(-(w-u)^2/2 sigma^2)``."""
    return np.asarray(w * np.exp(-((w - u) ** 2) / (2.0 * sigma**2)), dtype=np.float64)


def _flux_cdf(w: NDArray[np.float64], *, u: float, sigma: float) -> NDArray[np.float64]:
    """CDF of the same density, by quadrature. The reference the sampler is tested on."""
    upper = u + 12.0 * sigma
    norm, _ = integrate.quad(lambda x: _flux_density(np.asarray(x), u=u, sigma=sigma), 0.0, upper)
    values = [
        integrate.quad(lambda x: _flux_density(np.asarray(x), u=u, sigma=sigma), 0.0, point)[0]
        for point in np.atleast_1d(w)
    ]
    return np.asarray(values, dtype=np.float64) / norm


class TestSurfaceModel:
    def test_the_defaults_come_from_the_parameter_registry(self) -> None:
        # doc 08 §5: the registry is the sole source of numeric defaults. If these drift
        # apart, a run reports a gamma_se the manifest never set.
        registry = default_registry()
        assert registry.value_in("sheath.gamma_se_W", "dimensionless") == DEFAULT_SECONDARY_YIELD
        assert registry.value_in("sheath.ion_reflection", "dimensionless") == (
            DEFAULT_ION_REFLECTION
        )

    def test_a_negative_yield_is_refused(self) -> None:
        with pytest.raises(ValueError, match="secondary"):
            SurfaceModel(secondary_yield=-0.1, ion_reflection=0.0)

    def test_a_reflection_coefficient_above_one_is_refused(self) -> None:
        # R_i > 1 would return more ions than arrived and manufacture flux from nothing.
        with pytest.raises(ValueError, match="reflection"):
            SurfaceModel(secondary_yield=0.0, ion_reflection=1.5)

    def test_a_negative_emission_energy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="emission energy"):
            SurfaceModel(secondary_yield=0.1, ion_reflection=0.0, secondary_energy_ev=-1.0)


class TestAbsorption:
    def test_an_ion_past_the_wall_is_absorbed_and_recorded_with_its_own_energy(self) -> None:
        # The headline bookkeeping statement: the recorded energy is m|v|^2/2 of the
        # velocity that was handed in, not of anything the module recomputed.
        velocities = np.array([[3.0e3, -4.0e3, -1.2e4], [0.0, 0.0, -5.0e3]], dtype=np.float64)
        positions = np.array([-1.0e-6, -2.0e-6], dtype=np.float64)

        outcome = apply_boundaries(
            _rng(),
            positions_m=positions,
            velocities_m_per_s=velocities,
            alive=np.ones(2, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.ABSORBING,
            surface=SurfaceModel(secondary_yield=0.0, ion_reflection=0.0),
        )

        expected = 0.5 * _M_AR * np.sum(velocities**2, axis=1) / _E
        assert outcome.n_absorbed_at_wall == 2
        np.testing.assert_allclose(np.sort(outcome.absorbed_energies_ev), np.sort(expected))
        assert not outcome.alive.any()

    def test_an_absorbed_particle_is_parked_at_rest_inside_the_domain(self) -> None:
        # The performance contract of the solver: dead particles keep the array shape and
        # must be inert. At rest they collide at nu = n sigma v = 0, and inside the domain
        # they do not trip the deposition bounds guard.
        outcome = apply_boundaries(
            _rng(),
            positions_m=np.array([-1.0e-6], dtype=np.float64),
            velocities_m_per_s=np.array([[1.0, 2.0, -3.0e3]], dtype=np.float64),
            alive=np.ones(1, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.ABSORBING,
            surface=SurfaceModel(secondary_yield=0.0, ion_reflection=0.0),
        )

        assert 0.0 <= float(outcome.positions_m[0]) <= _LENGTH_M
        np.testing.assert_array_equal(outcome.velocities_m_per_s[0], np.zeros(3))

    def test_a_particle_leaving_at_the_bulk_boundary_is_not_wall_flux(self) -> None:
        # doc 02 §2's sign convention in bookkeeping form: only z = 0 is the wall. A
        # particle that returns to the bulk reservoir must not appear in Gamma_E, which is
        # measured at the surface.
        outcome = apply_boundaries(
            _rng(),
            positions_m=np.array([_LENGTH_M * 1.01], dtype=np.float64),
            velocities_m_per_s=np.array([[0.0, 0.0, 4.0e3]], dtype=np.float64),
            alive=np.ones(1, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.ABSORBING,
            surface=SurfaceModel(secondary_yield=0.0, ion_reflection=0.0),
        )

        assert outcome.n_left_bulk == 1
        assert outcome.n_absorbed_at_wall == 0
        assert outcome.absorbed_energies_ev.size == 0
        assert not outcome.alive.any()

    def test_particles_inside_the_domain_are_untouched(self) -> None:
        positions = np.array([1.0e-4, 5.0e-4, 9.0e-4], dtype=np.float64)
        velocities = np.arange(9, dtype=np.float64).reshape(3, 3)

        outcome = apply_boundaries(
            _rng(),
            positions_m=positions,
            velocities_m_per_s=velocities,
            alive=np.ones(3, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.ABSORBING,
            surface=SurfaceModel(secondary_yield=0.0, ion_reflection=0.0),
        )

        np.testing.assert_array_equal(outcome.positions_m, positions)
        np.testing.assert_array_equal(outcome.velocities_m_per_s, velocities)
        assert outcome.alive.all()

    def test_a_dead_particle_is_never_re_absorbed(self) -> None:
        # Double counting is the failure this whole file exists to prevent: a dead
        # particle re-entering the wall record would inflate Gamma_E every step it sat
        # there, and the run would still finish.
        outcome = apply_boundaries(
            _rng(),
            positions_m=np.array([-1.0e-6], dtype=np.float64),
            velocities_m_per_s=np.array([[0.0, 0.0, -3.0e3]], dtype=np.float64),
            alive=np.zeros(1, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.ABSORBING,
            surface=SurfaceModel(secondary_yield=0.0, ion_reflection=0.0),
        )

        assert outcome.n_absorbed_at_wall == 0
        assert outcome.absorbed_energies_ev.size == 0

    def test_the_inputs_are_not_mutated(self) -> None:
        positions = np.array([-1.0e-6, 5.0e-4], dtype=np.float64)
        velocities = np.array([[0.0, 0.0, -1.0e3], [0.0, 0.0, 1.0e3]], dtype=np.float64)
        alive = np.ones(2, dtype=bool)
        before = (positions.copy(), velocities.copy(), alive.copy())

        apply_boundaries(
            _rng(),
            positions_m=positions,
            velocities_m_per_s=velocities,
            alive=alive,
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.ABSORBING,
            surface=SurfaceModel(secondary_yield=0.0, ion_reflection=0.0),
        )

        np.testing.assert_array_equal(positions, before[0])
        np.testing.assert_array_equal(velocities, before[1])
        np.testing.assert_array_equal(alive, before[2])


class TestIonReflection:
    def test_full_reflection_absorbs_nothing_and_preserves_speed(self) -> None:
        # doc 03 §8 A10 registers R_i as a swept parameter with default zero. At R_i = 1
        # every ion returns, and specular reflection is an isometry — checkable per
        # particle, which an aggregate flux comparison is not.
        velocities = np.array([[1.0e3, -2.0e3, -8.0e3], [0.0, 3.0e3, -4.0e3]], dtype=np.float64)
        positions = np.array([-1.0e-6, -3.0e-6], dtype=np.float64)

        outcome = apply_boundaries(
            _rng(),
            positions_m=positions,
            velocities_m_per_s=velocities,
            alive=np.ones(2, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.ABSORBING,
            surface=SurfaceModel(secondary_yield=0.0, ion_reflection=1.0),
        )

        assert outcome.n_absorbed_at_wall == 0
        assert outcome.n_reflected_at_wall == 2
        assert outcome.alive.all()
        np.testing.assert_allclose(
            np.linalg.norm(outcome.velocities_m_per_s, axis=1), np.linalg.norm(velocities, axis=1)
        )
        np.testing.assert_allclose(outcome.velocities_m_per_s[:, 2], -velocities[:, 2])
        np.testing.assert_allclose(outcome.positions_m, -positions)

    @pytest.mark.physics
    def test_the_reflected_fraction_matches_the_coefficient(self) -> None:
        # A Bernoulli trial per arriving ion, so the count is binomial and the tolerance
        # is derived from its variance rather than eyeballed.
        n = 20000
        coefficient = 0.25
        outcome = apply_boundaries(
            _rng(),
            positions_m=np.full(n, -1.0e-6),
            velocities_m_per_s=np.tile(np.array([0.0, 0.0, -3.0e3]), (n, 1)),
            alive=np.ones(n, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.ABSORBING,
            surface=SurfaceModel(secondary_yield=0.0, ion_reflection=coefficient),
        )

        expected = n * coefficient
        error = math.sqrt(n * coefficient * (1.0 - coefficient))
        assert abs(outcome.n_reflected_at_wall - expected) < _SIGMA * error
        assert outcome.n_reflected_at_wall + outcome.n_absorbed_at_wall == n


class TestReflectingWall:
    def test_nothing_is_lost_at_either_boundary(self) -> None:
        # The closed configuration V-07 runs in. Absorption removes energy from the
        # system, so the energy-conservation gate needs a boundary that does not.
        positions = np.array([-1.0e-5, _LENGTH_M + 2.0e-5, 5.0e-4], dtype=np.float64)
        velocities = np.array(
            [[1.0, 2.0, -3.0e3], [0.0, 0.0, 5.0e3], [1.0e3, 0.0, 1.0e2]], dtype=np.float64
        )

        outcome = apply_boundaries(
            _rng(),
            positions_m=positions,
            velocities_m_per_s=velocities,
            alive=np.ones(3, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.REFLECTING,
        )

        assert outcome.alive.all()
        assert outcome.n_absorbed_at_wall == 0
        assert outcome.n_left_bulk == 0
        assert np.all(outcome.positions_m >= 0.0)
        assert np.all(outcome.positions_m <= _LENGTH_M)
        np.testing.assert_allclose(
            np.linalg.norm(outcome.velocities_m_per_s, axis=1), np.linalg.norm(velocities, axis=1)
        )
        np.testing.assert_allclose(outcome.positions_m, [1.0e-5, _LENGTH_M - 2.0e-5, 5.0e-4])
        np.testing.assert_allclose(outcome.velocities_m_per_s[:, 2], [3.0e3, -5.0e3, 1.0e2])

    def test_an_overshoot_of_several_domains_still_lands_inside(self) -> None:
        # Not reachable under the doc 03 §4.3 CFL constraint, and folded exactly anyway:
        # a clamp would silently pile particles onto the wall node, which is where this
        # project measures its deliverable.
        outcome = apply_boundaries(
            _rng(),
            positions_m=np.array([2.5 * _LENGTH_M, -1.5 * _LENGTH_M], dtype=np.float64),
            velocities_m_per_s=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64),
            alive=np.ones(2, dtype=bool),
            domain_length_m=_LENGTH_M,
            mass_kg=_M_AR,
            wall=WallModel.REFLECTING,
        )

        # Two reflections each — 2.5 L bounces off z = L and then off z = 0 — so both
        # come back with the sign they started with. The parity is the point: an
        # implementation that flipped once per call would leave the sign wrong exactly
        # when a particle crossed twice.
        np.testing.assert_allclose(outcome.positions_m, [0.5 * _LENGTH_M, 0.5 * _LENGTH_M])
        np.testing.assert_allclose(outcome.velocities_m_per_s[:, 2], [1.0, -1.0])


class TestSecondaryElectronEmission:
    @pytest.mark.physics
    def test_the_yield_sets_the_emitted_count(self) -> None:
        # doc 03 §3.3: "Secondary electron emission is not optional." One Bernoulli trial
        # per absorbed ion, so the count is binomial in the yield.
        n = 40000
        velocities = sample_secondary_velocities(
            _rng(), n_absorbed=n, surface=SurfaceModel(secondary_yield=DEFAULT_SECONDARY_YIELD)
        )

        expected = n * DEFAULT_SECONDARY_YIELD
        error = math.sqrt(n * DEFAULT_SECONDARY_YIELD * (1.0 - DEFAULT_SECONDARY_YIELD))
        assert abs(velocities.shape[0] - expected) < _SIGMA * error

    def test_secondaries_are_emitted_into_the_plasma(self) -> None:
        # doc 02 §2: z increases into the plasma, so an electron leaving the wall has
        # v_z >= 0. The opposite sign would put every secondary straight back into the
        # surface it came from and silently switch the channel off.
        velocities = sample_secondary_velocities(
            _rng(),
            n_absorbed=2000,
            surface=SurfaceModel(secondary_yield=1.0, secondary_energy_ev=5.0),
        )

        assert velocities.shape == (2000, 3)
        assert np.all(velocities[:, 2] >= 0.0)
        energy_ev = 0.5 * _M_E * np.sum(velocities**2, axis=1) / _E
        np.testing.assert_allclose(energy_ev, 5.0, rtol=1e-12)

    def test_a_zero_yield_emits_nothing(self) -> None:
        velocities = sample_secondary_velocities(
            _rng(), n_absorbed=1000, surface=SurfaceModel(secondary_yield=0.0)
        )
        assert velocities.shape == (0, 3)

    def test_the_default_emission_energy_is_stated_and_small(self) -> None:
        # The module's documented simplification: emitted at rest, then accelerated
        # through the full sheath drop. Recorded as a test so the bound in the docstring
        # cannot drift away from the code.
        assert SECONDARY_EMISSION_ENERGY_EV >= 0.0
        assert SECONDARY_EMISSION_ENERGY_EV < 1.0


class TestInjectionSource:
    def test_the_one_way_flux_of_a_still_maxwellian_is_the_textbook_quarter_n_v_bar(
        self,
    ) -> None:
        # Gamma = n <v> / 4 with <v> = sqrt(8 k T / pi m). Stated in every kinetic theory
        # text; the module must reproduce it, not define it.
        sigma = 1.0e5
        source = InjectionSource(
            density_m3=1.0e17, thermal_speed_m_per_s=sigma, drift_speed_m_per_s=0.0
        )

        mean_speed = math.sqrt(8.0 / math.pi) * sigma
        assert source.flux_per_m2_s == pytest.approx(1.0e17 * mean_speed / 4.0, rel=1e-12)

    def test_a_strong_drift_gives_the_drift_flux(self) -> None:
        # doc 03 §3.3's bulk condition: ions enter at u = c_s. When u >> sigma the
        # one-way flux is n u to exponential accuracy, which is the Bohm flux the L0
        # cross-check compares against.
        source = InjectionSource(
            density_m3=1.0e17, thermal_speed_m_per_s=1.0e2, drift_speed_m_per_s=1.0e4
        )
        assert source.flux_per_m2_s == pytest.approx(1.0e17 * 1.0e4, rel=1e-6)

    def test_a_negative_density_is_refused(self) -> None:
        with pytest.raises(ValueError, match="density"):
            InjectionSource(density_m3=-1.0, thermal_speed_m_per_s=1.0e3, drift_speed_m_per_s=0.0)

    def test_a_drift_away_from_the_wall_is_refused(self) -> None:
        # The drift is a magnitude toward the wall; a signed one here is how the sign
        # convention of doc 02 §2 gets inverted at exactly the boundary that matters.
        with pytest.raises(ValueError, match="drift"):
            InjectionSource(
                density_m3=1.0e17, thermal_speed_m_per_s=1.0e3, drift_speed_m_per_s=-1.0e3
            )


class TestInflowSampling:
    def test_every_injected_particle_moves_toward_the_wall(self) -> None:
        velocities = sample_inflow_velocities(
            _rng(),
            2000,
            source=InjectionSource(
                density_m3=1.0e17, thermal_speed_m_per_s=1.0e3, drift_speed_m_per_s=2.7e3
            ),
        )
        assert velocities.shape == (2000, 3)
        assert np.all(velocities[:, 2] < 0.0)

    @pytest.mark.physics
    @pytest.mark.parametrize(("drift", "sigma"), [(0.0, 1.0e3), (2.7e3, 3.5e2), (1.0e3, 1.0e3)])
    def test_the_inflow_speed_follows_the_flux_weighted_maxwellian(
        self, drift: float, sigma: float
    ) -> None:
        # The reference is a quadrature of p(w) ~ w exp(-(w-u)^2/2 sigma^2), not a
        # previous run of this code. A number-weighted half-Maxwellian — the classic
        # inflow-boundary error — fails this by a wide margin at every drift.
        n = 20000
        velocities = sample_inflow_velocities(
            _rng(),
            n,
            source=InjectionSource(
                density_m3=1.0e17, thermal_speed_m_per_s=sigma, drift_speed_m_per_s=drift
            ),
        )
        speeds = -velocities[:, 2]

        result = stats.ks_1samp(speeds, lambda w: _flux_cdf(np.asarray(w), u=drift, sigma=sigma))
        assert result.statistic < _KS_CRITICAL / math.sqrt(n), (
            f"KS distance {result.statistic:.4f} against the flux-weighted Maxwellian"
        )

    @pytest.mark.physics
    def test_a_number_weighted_half_maxwellian_would_fail_that_test(self) -> None:
        # Proof the test has teeth. ADR-011: a benchmark nobody has seen reject anything
        # is not a benchmark.
        n = 20000
        sigma, drift = 1.0e3, 0.0
        rng = _rng()
        wrong = np.abs(rng.normal(scale=sigma, size=n))

        result = stats.ks_1samp(wrong, lambda w: _flux_cdf(np.asarray(w), u=drift, sigma=sigma))
        assert result.statistic > _KS_CRITICAL / math.sqrt(n)

    @pytest.mark.physics
    def test_the_transverse_components_carry_the_thermal_spread(self) -> None:
        # The flux weighting acts on v_z alone. A sampler that weighted the transverse
        # components too would narrow the LIF-visible distribution of doc 04 §3.2.
        n = 40000
        sigma = 1.0e3
        velocities = sample_inflow_velocities(
            _rng(),
            n,
            source=InjectionSource(
                density_m3=1.0e17, thermal_speed_m_per_s=sigma, drift_speed_m_per_s=2.7e3
            ),
        )

        for axis in (0, 1):
            measured = float(np.std(velocities[:, axis]))
            # Standard error of a standard deviation estimate is sigma / sqrt(2n).
            assert abs(measured - sigma) < _SIGMA * sigma / math.sqrt(2.0 * n)

    @pytest.mark.physics
    def test_the_mean_inflow_speed_matches_the_ratio_of_two_moments(self) -> None:
        # <|v_z|> over the flux-weighted density equals <v_z^2>/<|v_z|> over the
        # number-weighted one — an independent expression for the same quantity, and the
        # one that ties the sampled shape to InjectionSource.flux_per_m2_s.
        n = 40000
        sigma, drift = 3.5e2, 2.691e3
        velocities = sample_inflow_velocities(
            _rng(),
            n,
            source=InjectionSource(
                density_m3=1.0e17, thermal_speed_m_per_s=sigma, drift_speed_m_per_s=drift
            ),
        )
        speeds = -velocities[:, 2]

        upper = drift + 12.0 * sigma
        numerator, _ = integrate.quad(
            lambda w: w * _flux_density(np.asarray(w), u=drift, sigma=sigma), 0.0, upper
        )
        norm, _ = integrate.quad(
            lambda w: _flux_density(np.asarray(w), u=drift, sigma=sigma), 0.0, upper
        )
        expected = numerator / norm

        error = float(np.std(speeds)) / math.sqrt(n)
        assert abs(float(np.mean(speeds)) - expected) < _SIGMA * error

    def test_a_cold_source_injects_at_exactly_the_drift_speed(self) -> None:
        # The zero-temperature limit. Worth pinning because the rejection sampler divides
        # by sigma, and the limit is the configuration a cold-ion cross-check would use.
        velocities = sample_inflow_velocities(
            _rng(),
            16,
            source=InjectionSource(
                density_m3=1.0e17, thermal_speed_m_per_s=0.0, drift_speed_m_per_s=2.7e3
            ),
        )
        np.testing.assert_allclose(velocities[:, 2], -2.7e3)
        np.testing.assert_allclose(velocities[:, :2], 0.0)

    def test_sampling_is_reproducible_from_the_seed(self) -> None:
        # doc 00 E3: bit-for-bit reproduction from a manifest. The injection stream is
        # part of what a manifest determines.
        source = InjectionSource(
            density_m3=1.0e17, thermal_speed_m_per_s=1.0e3, drift_speed_m_per_s=2.7e3
        )
        first = sample_inflow_velocities(_rng(), 500, source=source)
        second = sample_inflow_velocities(_rng(), 500, source=source)
        np.testing.assert_array_equal(first, second)

    def test_zero_particles_gives_an_empty_but_correctly_shaped_array(self) -> None:
        # (0, 3) and not (0,): a solver concatenating this onto its particle arrays would
        # otherwise fail as a broadcast error several steps later.
        velocities = sample_inflow_velocities(
            _rng(),
            0,
            source=InjectionSource(
                density_m3=1.0e17, thermal_speed_m_per_s=1.0e3, drift_speed_m_per_s=0.0
            ),
        )
        assert velocities.shape == (0, 3)


class TestValidation:
    def test_a_mismatched_velocity_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            apply_boundaries(
                _rng(),
                positions_m=np.zeros(3),
                velocities_m_per_s=np.zeros((3, 2)),
                alive=np.ones(3, dtype=bool),
                domain_length_m=_LENGTH_M,
                mass_kg=_M_AR,
                wall=WallModel.REFLECTING,
            )

    def test_a_mismatched_alive_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="alive"):
            apply_boundaries(
                _rng(),
                positions_m=np.zeros(3),
                velocities_m_per_s=np.zeros((3, 3)),
                alive=np.ones(2, dtype=bool),
                domain_length_m=_LENGTH_M,
                mass_kg=_M_AR,
                wall=WallModel.REFLECTING,
            )

    def test_an_absorbing_wall_without_a_surface_model_is_refused(self) -> None:
        # Defaulting the surface would silently switch secondary emission off, which
        # doc 03 §3.3 calls "a common and significant modelling error".
        with pytest.raises(ValueError, match="surface"):
            apply_boundaries(
                _rng(),
                positions_m=np.zeros(1),
                velocities_m_per_s=np.zeros((1, 3)),
                alive=np.ones(1, dtype=bool),
                domain_length_m=_LENGTH_M,
                mass_kg=_M_AR,
                wall=WallModel.ABSORBING,
            )

    def test_a_non_positive_domain_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="domain length"):
            apply_boundaries(
                _rng(),
                positions_m=np.zeros(1),
                velocities_m_per_s=np.zeros((1, 3)),
                alive=np.ones(1, dtype=bool),
                domain_length_m=0.0,
                mass_kg=_M_AR,
                wall=WallModel.REFLECTING,
            )
