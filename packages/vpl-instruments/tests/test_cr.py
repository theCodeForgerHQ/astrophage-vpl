"""The collisional-radiative model — doc 04 §2.2, verification item V-25.

Every anchor here is analytic. There is no published Ar line-ratio benchmark in this file
and none anywhere else in the package: see the note in :mod:`vpl.instruments.oes.cr` about
doc 04 §8 **V-24**, which is *not* satisfied.

What is checked, and against what:

* **Klein-Rosseland.** The de-excitation rate coefficient is computed by integrating the
  excitation cross section over the actual EEDF. Handed a Maxwellian it must reproduce
  ``k_ul = (g_l/g_u) exp(dE/T_e) k_lu`` — a closed-form identity that has nothing to do
  with how the integral was discretised.
* **The corona limit** (V-25). At vanishing ``n_e`` every population reduces to
  ``n_e n_g K_gu / sum(A Lambda)``, which is written out in the test.
* **The LTE limit** (V-25). At large ``n_e`` with a Maxwellian EEDF every population
  reduces to Boltzmann at ``T_e``, ``n_u/n_g = (g_u/g_g) exp(-E_u/T_e)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oes_system import (
    METASTABLE_ENERGY_EV,
    energy_grid,
    metastable_trapped_system,
    plain_system,
    resonance_trapped_system,
)
from vpl.instruments.oes.cr import (
    CollisionalRadiativeModel,
    CrConvergenceError,
)
from vpl.instruments.oes.levels import LevelSystem
from vpl.physics.eedf.analytic import druyvesteyn_eedf, maxwellian_eedf
from vpl.physics.eedf.grid import EnergyGrid


@pytest.fixture(scope="module")
def grid() -> EnergyGrid:
    return energy_grid()


@pytest.fixture(scope="module")
def system(grid: EnergyGrid) -> LevelSystem:
    return plain_system(grid)


@pytest.fixture(scope="module")
def trapped_system(grid: EnergyGrid) -> LevelSystem:
    return resonance_trapped_system(grid)


@pytest.fixture(scope="module")
def self_trapped_system(grid: EnergyGrid) -> LevelSystem:
    return metastable_trapped_system(grid)


def maxwellian(grid: EnergyGrid, t_e_ev: float) -> NDArray[np.float64]:
    return grid.normalise(maxwellian_eedf(grid.centres_ev, electron_temperature_ev=t_e_ev))


# ── Klein-Rosseland ─────────────────────────────────────────────────────────────


@pytest.mark.physics
def test_de_excitation_obeys_detailed_balance_for_a_maxwellian(
    grid: EnergyGrid, system: LevelSystem
) -> None:
    """``k_ul = (g_l / g_u) exp(dE / T_e) k_lu``, exactly, for a Maxwellian EEDF.

    This is the single most valuable check in the module. The superelastic rate is never
    computed from this identity — it is an independent integral of the same cross section
    over the same EEDF, shifted in energy by the threshold — so agreement means the
    Klein-Rosseland relation between the cross sections, the energy shift, the quadrature
    weights and the statistical weights are all simultaneously right.
    """
    t_e_ev = 3.0
    f0 = maxwellian(grid, t_e_ev)
    weights = {level.label: level.degeneracy for level in system.levels}

    for impact in system.electron_impact:
        excitation = impact.excitation_rate_coefficient(grid, f0)
        de_excitation = impact.de_excitation_rate_coefficient(
            grid, f0, g_lower=weights[impact.lower], g_upper=weights[impact.upper]
        )
        predicted = (
            weights[impact.lower]
            / weights[impact.upper]
            * np.exp(impact.threshold_ev / t_e_ev)
            * excitation
        )
        assert de_excitation == pytest.approx(predicted, rel=1e-9)


@pytest.mark.physics
def test_detailed_balance_is_not_hard_wired(grid: EnergyGrid, system: LevelSystem) -> None:
    """A Druyvesteyn EEDF must break the identity — doc 04 §2.2 is explicit about this.

    "LXCat cross sections integrated over the **BOLSIG+ EEDF**, not a Maxwellian". If the
    superelastic rate were being obtained from the Maxwellian detailed-balance formula
    rather than from the distribution, this test would pass by accident, and the whole
    non-Maxwellian claim of the framework would be decorative.
    """
    mean_energy_ev = 4.5
    f0 = grid.normalise(druyvesteyn_eedf(grid.centres_ev, mean_energy_ev=mean_energy_ev))
    impact = system.electron_impact[0]
    weights = {level.label: level.degeneracy for level in system.levels}

    excitation = impact.excitation_rate_coefficient(grid, f0)
    de_excitation = impact.de_excitation_rate_coefficient(
        grid, f0, g_lower=weights[impact.lower], g_upper=weights[impact.upper]
    )
    maxwellian_prediction = (
        weights[impact.lower]
        / weights[impact.upper]
        * np.exp(impact.threshold_ev / ((2.0 / 3.0) * mean_energy_ev))
        * excitation
    )
    # A Druyvesteyn distribution of the same mean energy has a depleted tail and a
    # correspondingly fatter body. The excitation integral, which lives in the tail above
    # 11.5 eV, is suppressed; the superelastic integral, which samples the body, is not.
    # So the ratio between them departs from the Maxwellian relation by nearly an order of
    # magnitude, and in the direction the physics says it should.
    assert de_excitation > 2.0 * maxwellian_prediction


# ── V-25: the two density limits ────────────────────────────────────────────────


@pytest.mark.physics
def test_low_density_limit_is_corona(grid: EnergyGrid, system: LevelSystem) -> None:
    """doc 04 §8 V-25, corona half: at low ``n_e`` every level is radiatively controlled.

    ``n_u = n_e n_g K_gu / sum_l A_ul``, with no collisional de-excitation and no cascade
    contribution to speak of. Written out here from the level data rather than taken from
    the model.
    """
    t_e_ev = 3.0
    f0 = maxwellian(grid, t_e_ev)
    n_e, n_g = 1e10, 1.6e20
    # The metastable needs its wall term even here. Without one it has no loss channel
    # that survives n_e -> 0, so it climbs to its LTE value and pumps every level above it
    # collisionally — there is no corona limit for a level with no non-collisional loss,
    # which is itself worth knowing.
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})

    populations = model.solve(electron_density_per_m3=n_e, ground_density_per_m3=n_g, f0=f0)

    excitation = {
        impact.upper: impact.excitation_rate_coefficient(grid, f0)
        for impact in system.electron_impact
        if impact.lower == "g"
    }
    for label in ("r", "p"):
        total_a = sum(r.a_ul_per_s for r in system.radiative if r.upper == label)
        assert populations[label] == pytest.approx(
            n_e * n_g * excitation[label] / total_a, rel=1e-3
        )


@pytest.mark.physics
def test_a_metastable_at_low_density_is_limited_by_the_wall(
    grid: EnergyGrid, system: LevelSystem
) -> None:
    """The metastable has no radiative channel, so its corona limit is the wall term."""
    t_e_ev = 3.0
    f0 = maxwellian(grid, t_e_ev)
    n_e, n_g = 1e10, 1.6e20
    wall_loss = 1.0e4
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": wall_loss})

    populations = model.solve(electron_density_per_m3=n_e, ground_density_per_m3=n_g, f0=f0)

    # Every excitation out of the ground state that ends on a level cascading into the
    # metastable arrives there, because in the corona limit each of those levels radiates
    # everything it receives. So the source is k_gm *plus* k_gp, not k_gm alone — and
    # getting that wrong is exactly how a cascade term goes missing unnoticed.
    def rate(lower: str, upper: str) -> float:
        pair = (lower, upper)
        impact = next(c for c in system.electron_impact if (c.lower, c.upper) == pair)
        return impact.excitation_rate_coefficient(grid, f0)

    source = n_e * n_g * (rate("g", "m") + rate("g", "p"))
    assert populations["m"] == pytest.approx(source / wall_loss, rel=1e-3)


@pytest.mark.physics
def test_high_density_limit_is_boltzmann(grid: EnergyGrid, system: LevelSystem) -> None:
    """doc 04 §8 V-25, LTE half: at large ``n_e`` the populations are Boltzmann at ``T_e``.

    Only true because the superelastic rates obey Klein-Rosseland against the *same*
    distribution the excitation rates were taken over. A CR model that used a Maxwellian
    detailed-balance shortcut would also pass this and fail the non-Maxwellian test above;
    one that got the shift wrong would fail this and pass that. Both are needed.
    """
    t_e_ev = 3.0
    f0 = maxwellian(grid, t_e_ev)
    n_g = 1.6e20
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})

    # Far above anything physical, and deliberately so: this is a limit, and the approach
    # to it goes as A_ul / (n_e K_ul), which for the resonance line needs 1e28 m^-3 before
    # the radiative channel is negligible at the 1e-3 level being asserted.
    populations = model.solve(electron_density_per_m3=1e28, ground_density_per_m3=n_g, f0=f0)

    for level in system.levels[1:]:
        boltzmann = n_g * level.degeneracy * np.exp(-level.energy_ev / t_e_ev)
        assert populations[level.label] == pytest.approx(boltzmann, rel=1e-3)


@pytest.mark.physics
def test_the_approach_to_lte_is_monotone_in_density(grid: EnergyGrid, system: LevelSystem) -> None:
    """The metastable density climbs monotonically and saturates at its Boltzmann value.

    Note that the metastable *fraction* ``n_m / n_e`` does the opposite: it falls, because
    the wall term stops competing once collisional de-excitation takes over. Asserting the
    density rather than the fraction is not a convenience — the fraction is what a careless
    reading of "approach to LTE" would predict to rise, and it does not.
    """
    f0 = maxwellian(grid, 3.0)
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})
    densities = [
        model.solve(electron_density_per_m3=n_e, ground_density_per_m3=1.6e20, f0=f0)["m"]
        for n_e in np.logspace(10.0, 26.0, 33)
    ]
    assert np.all(np.diff(densities) > 0.0)
    boltzmann = 1.6e20 * 5.0 * np.exp(-METASTABLE_ENERGY_EV / 3.0)
    assert densities[-1] == pytest.approx(boltzmann, rel=0.05)


# ── structure ───────────────────────────────────────────────────────────────────


@pytest.mark.physics
def test_populations_are_linear_in_ground_density_at_fixed_electron_density(
    grid: EnergyGrid, system: LevelSystem
) -> None:
    """In the corona regime the whole system is a linear response to the ground state."""
    f0 = maxwellian(grid, 3.0)
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})
    single = model.solve(electron_density_per_m3=1e12, ground_density_per_m3=1.6e20, f0=f0)
    double = model.solve(electron_density_per_m3=1e12, ground_density_per_m3=3.2e20, f0=f0)
    for label in single.labels:
        assert double[label] == pytest.approx(2.0 * single[label], rel=1e-9)


@pytest.mark.physics
def test_every_population_is_positive(grid: EnergyGrid, system: LevelSystem) -> None:
    f0 = maxwellian(grid, 3.0)
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})
    for n_e in np.logspace(8.0, 22.0, 15):
        populations = model.solve(
            electron_density_per_m3=float(n_e), ground_density_per_m3=1.6e20, f0=f0
        )
        assert all(populations[label] > 0.0 for label in populations.labels)


@pytest.mark.physics
def test_cascade_feeds_a_level_that_nothing_excites_directly(
    grid: EnergyGrid, system: LevelSystem
) -> None:
    """Deleting the direct ``g -> m`` channel must not empty the metastable.

    ``p -> m`` radiative decay still feeds it, which is the cascade term of doc 04 §2.2's
    second sum and the reason the 811.53 nm line is called metastable-*coupled* rather
    than metastable-fed.
    """
    f0 = maxwellian(grid, 3.0)
    without_direct = LevelSystem(
        levels=system.levels,
        electron_impact=tuple(
            impact
            for impact in system.electron_impact
            if (impact.lower, impact.upper) != ("g", "m")
        ),
        radiative=system.radiative,
    )
    model = CollisionalRadiativeModel(
        system=without_direct, grid=grid, wall_loss_per_s={"m": 1.0e4}
    )
    populations = model.solve(electron_density_per_m3=1e15, ground_density_per_m3=1.6e20, f0=f0)
    assert populations["m"] > 0.0


@pytest.mark.physics
def test_wall_quenching_depletes_the_metastable(grid: EnergyGrid, system: LevelSystem) -> None:
    f0 = maxwellian(grid, 3.0)
    slow = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e3})
    fast = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e5})
    kwargs = {"electron_density_per_m3": 1e15, "ground_density_per_m3": 1.6e20, "f0": f0}
    assert fast.solve(**kwargs)["m"] < slow.solve(**kwargs)["m"]  # type: ignore[arg-type]


# ── doc 04 §2.3: trapping is not optional ───────────────────────────────────────


@pytest.mark.physics
def test_radiation_trapping_raises_the_metastable_population(
    grid: EnergyGrid, system: LevelSystem, trapped_system: LevelSystem
) -> None:
    """doc 04 §2.3's central claim, as a test.

    "The Ar I resonance lines are strongly self-absorbed, which pumps the metastable
    population far above its optically-thin value." Here the resonant level's only escape
    is a trapped line, so trapping holds it up; it feeds the metastable through electron
    collisions and cascade. The two systems differ in nothing but the escape factor.
    """
    f0 = maxwellian(grid, 3.0)
    thin = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})
    trapped = CollisionalRadiativeModel(
        system=trapped_system,
        grid=grid,
        wall_loss_per_s={"m": 1.0e4},
        trapping_path_length_m=0.4,
    )
    kwargs = {"electron_density_per_m3": 1e17, "ground_density_per_m3": 1.6e20, "f0": f0}

    thin_populations = thin.solve(**kwargs)  # type: ignore[arg-type]
    trapped_populations = trapped.solve(**kwargs)  # type: ignore[arg-type]

    assert trapped_populations["r"] > 100.0 * thin_populations["r"]
    assert trapped_populations.escape_factors[("r", "g")] < 1e-2
    assert trapped_populations.escape_factors[("p", "m")] == 1.0
    # ...and the resonant level feeds the metastable collisionally, which is the mechanism
    # doc 04 §2.3 names. The effect on the metastable is real but modest here because the
    # synthetic system also excites it directly from the ground state; in argon the
    # 1s4/1s5 mixing is much stronger than the direct channel and the pumping dominates.
    assert trapped_populations["m"] > 1.02 * thin_populations["m"]


@pytest.mark.physics
def test_an_untrapped_system_reports_unit_escape_factors(
    grid: EnergyGrid, system: LevelSystem
) -> None:
    f0 = maxwellian(grid, 3.0)
    model = CollisionalRadiativeModel(system=system, grid=grid)
    populations = model.solve(electron_density_per_m3=1e15, ground_density_per_m3=1.6e20, f0=f0)
    assert set(populations.escape_factors.values()) == {1.0}
    assert populations.iterations == 1


@pytest.mark.physics
def test_the_escape_factor_iteration_converges_and_is_reported(
    grid: EnergyGrid, self_trapped_system: LevelSystem
) -> None:
    """Trapping on a line whose absorber is a solved level makes the solve non-linear."""
    f0 = maxwellian(grid, 3.0)
    model = CollisionalRadiativeModel(
        system=self_trapped_system,
        grid=grid,
        wall_loss_per_s={"m": 1.0e4},
        trapping_path_length_m=0.4,
    )
    populations = model.solve(electron_density_per_m3=1e17, ground_density_per_m3=1.6e20, f0=f0)
    assert populations.iterations > 1
    assert populations.residual < model.relative_tolerance
    assert 0.0 < populations.escape_factors[("p", "m")] < 1.0


def test_a_starved_iteration_budget_raises(
    grid: EnergyGrid, self_trapped_system: LevelSystem
) -> None:
    f0 = maxwellian(grid, 3.0)
    model = CollisionalRadiativeModel(
        system=self_trapped_system,
        grid=grid,
        trapping_path_length_m=0.4,
        relative_tolerance=1e-30,
        max_iterations=2,
    )
    with pytest.raises(CrConvergenceError, match="escape-factor"):
        model.solve(electron_density_per_m3=1e17, ground_density_per_m3=1.6e20, f0=f0)


# ── contract ────────────────────────────────────────────────────────────────────


def test_solve_rejects_a_negative_electron_density(grid: EnergyGrid, system: LevelSystem) -> None:
    f0 = maxwellian(grid, 3.0)
    model = CollisionalRadiativeModel(system=system, grid=grid)
    with pytest.raises(ValueError, match="cannot be negative"):
        model.solve(electron_density_per_m3=-1.0, ground_density_per_m3=1e20, f0=f0)


def test_solve_rejects_an_eedf_of_the_wrong_length(grid: EnergyGrid, system: LevelSystem) -> None:
    model = CollisionalRadiativeModel(system=system, grid=grid)
    with pytest.raises(ValueError, match="one value per cell"):
        model.solve(
            electron_density_per_m3=1e15,
            ground_density_per_m3=1e20,
            f0=np.ones(grid.n_cells - 1),
        )


def test_wall_loss_on_an_unknown_level_is_rejected(grid: EnergyGrid, system: LevelSystem) -> None:
    with pytest.raises(ValueError, match="no level"):
        CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"nope": 1.0})


def test_trapping_without_a_path_length_is_rejected(
    grid: EnergyGrid, trapped_system: LevelSystem
) -> None:
    with pytest.raises(ValueError, match="path length"):
        CollisionalRadiativeModel(system=trapped_system, grid=grid)
