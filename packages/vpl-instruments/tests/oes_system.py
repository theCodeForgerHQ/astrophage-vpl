"""Builders for a synthetic argon-like level system, shared by the OES tests.

Plain functions and **not** a ``conftest.py``: this package is developed by more than one
person at a time, and a shared ``conftest`` is the file two of them collide on. Each test
module wraps what it needs in its own fixture.

The level system here is **not argon**, and that is deliberate for the same reason doc 09
§2.1's swarm benchmarks use model gases: a synthetic system has no atomic-data uncertainty,
so every number the CR model produces from it is attributable to the CR model. Its energies
and degeneracies are argon-shaped — a metastable near 11.5 eV, a resonant level just above
it, a 2p-like radiating level near 13 eV — so the structure being exercised is the one
doc 02 §6.3 actually uses.

Thresholds are multiples of the energy-grid cell width on purpose. The Klein-Rosseland
detailed-balance identity of :mod:`vpl.instruments.oes.levels` is then exact in the
discrete sense rather than accurate to an interpolation, which is what makes
``test_de_excitation_obeys_detailed_balance_for_a_maxwellian`` a check on the physics
instead of a check on ``np.interp``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vpl.instruments.oes.escape import LineProfileShape, TrappedLine
from vpl.instruments.oes.levels import (
    ElectronImpactChannel,
    Level,
    LevelSystem,
    RadiativeChannel,
)
from vpl.physics.eedf.grid import EnergyGrid

GRID_MAX_EV = 60.0
GRID_CELLS = 600

GROUND_ENERGY_EV = 0.0
METASTABLE_ENERGY_EV = 11.5
RESONANT_ENERGY_EV = 11.6
RADIATING_ENERGY_EV = 13.1


def energy_grid() -> EnergyGrid:
    """A uniform 0.1 eV grid. Uniform so the detailed-balance shift is exact."""
    return EnergyGrid.linear(max_ev=GRID_MAX_EV, n_cells=GRID_CELLS)


def threshold_cross_section(
    energy_ev: NDArray[np.float64], *, threshold_ev: float, peak_m2: float
) -> NDArray[np.float64]:
    """``sigma = peak (1 - threshold/eps)`` above threshold, zero below.

    A Born-like shape: correct threshold behaviour, monotone, bounded. Nothing here
    depends on its being argon's, and using a real cross section would make the tests
    depend on data the repository deliberately does not vendor (doc 09 §5).
    """
    above = energy_ev > threshold_ev
    return np.where(above, peak_m2 * (1.0 - threshold_ev / np.where(above, energy_ev, 1.0)), 0.0)


def channel(
    grid: EnergyGrid, *, lower: str, upper: str, threshold_ev: float, peak_m2: float
) -> ElectronImpactChannel:
    return ElectronImpactChannel.from_sampler(
        grid,
        lower=lower,
        upper=upper,
        threshold_ev=threshold_ev,
        sampler=lambda e: threshold_cross_section(e, threshold_ev=threshold_ev, peak_m2=peak_m2),
    )


def argon_like_levels() -> tuple[Level, ...]:
    return (
        Level(label="g", energy_ev=GROUND_ENERGY_EV, degeneracy=1),
        Level(label="m", energy_ev=METASTABLE_ENERGY_EV, degeneracy=5, is_metastable=True),
        Level(label="r", energy_ev=RESONANT_ENERGY_EV, degeneracy=3),
        Level(label="p", energy_ev=RADIATING_ENERGY_EV, degeneracy=7),
    )


def plain_system(grid: EnergyGrid) -> LevelSystem:
    """Ground, metastable, resonant and radiating levels, argon-shaped.

    Radiative structure: the resonant level decays to ground on a strong resonance line
    (the 106.7 nm analogue, and the one that traps), and the radiating level decays to the
    metastable on a weak near-infrared line (the 811.53 nm analogue, and the one observed).
    """
    return LevelSystem(
        levels=argon_like_levels(),
        electron_impact=(
            channel(grid, lower="g", upper="m", threshold_ev=11.5, peak_m2=3.0e-21),
            channel(grid, lower="g", upper="r", threshold_ev=11.6, peak_m2=2.0e-21),
            channel(grid, lower="g", upper="p", threshold_ev=13.1, peak_m2=1.0e-21),
            channel(grid, lower="m", upper="p", threshold_ev=1.6, peak_m2=5.0e-20),
            channel(grid, lower="m", upper="r", threshold_ev=0.1, peak_m2=1.0e-19),
        ),
        radiative=(
            RadiativeChannel(upper="r", lower="g", a_ul_per_s=1.19e8, wavelength_nm=106.7),
            RadiativeChannel(upper="p", lower="m", a_ul_per_s=3.31e7, wavelength_nm=811.53),
        ),
    )


def resonance_trapped_system(grid: EnergyGrid) -> LevelSystem:
    """The same system with the resonance line given a profile, so that it traps."""
    return LevelSystem(
        levels=argon_like_levels(),
        electron_impact=(
            channel(grid, lower="g", upper="m", threshold_ev=11.5, peak_m2=3.0e-21),
            channel(grid, lower="g", upper="r", threshold_ev=11.6, peak_m2=2.0e-21),
            channel(grid, lower="g", upper="p", threshold_ev=13.1, peak_m2=1.0e-21),
            channel(grid, lower="m", upper="p", threshold_ev=1.6, peak_m2=5.0e-20),
            channel(grid, lower="m", upper="r", threshold_ev=0.1, peak_m2=1.0e-19),
        ),
        radiative=(
            RadiativeChannel(
                upper="r",
                lower="g",
                a_ul_per_s=1.19e8,
                wavelength_nm=106.7,
                trapping=TrappedLine(
                    wavelength_nm=106.7,
                    a_ul_per_s=1.19e8,
                    g_upper=3,
                    g_lower=1,
                    shape=LineProfileShape.DOPPLER,
                    profile_fwhm_nm=2.09e-4,
                ),
            ),
            RadiativeChannel(upper="p", lower="m", a_ul_per_s=3.31e7, wavelength_nm=811.53),
        ),
    )


def metastable_trapped_system(grid: EnergyGrid) -> LevelSystem:
    """A system whose trapped line absorbs on a *solved* level, not on the ground state.

    Trapping on the resonance line is linear: its absorber is the ground state, which is
    held fixed, so the escape factor is a constant and the fixed point converges in one
    step. Trapping the 811.53 nm analogue instead puts the metastable — an unknown — in
    the absorption coefficient, which is what makes the solve genuinely non-linear and is
    the case the iteration exists for.
    """
    return LevelSystem(
        levels=argon_like_levels(),
        electron_impact=(
            channel(grid, lower="g", upper="m", threshold_ev=11.5, peak_m2=3.0e-21),
            channel(grid, lower="g", upper="r", threshold_ev=11.6, peak_m2=2.0e-21),
            channel(grid, lower="g", upper="p", threshold_ev=13.1, peak_m2=1.0e-21),
            channel(grid, lower="m", upper="p", threshold_ev=1.6, peak_m2=5.0e-20),
            channel(grid, lower="m", upper="r", threshold_ev=0.1, peak_m2=1.0e-19),
        ),
        radiative=(
            RadiativeChannel(upper="r", lower="g", a_ul_per_s=1.19e8, wavelength_nm=106.7),
            RadiativeChannel(
                upper="p",
                lower="m",
                a_ul_per_s=3.31e7,
                wavelength_nm=811.53,
                trapping=TrappedLine(
                    wavelength_nm=811.53,
                    a_ul_per_s=3.31e7,
                    g_upper=7,
                    g_lower=5,
                    shape=LineProfileShape.DOPPLER,
                    profile_fwhm_nm=1.59e-3,
                ),
            ),
        ),
    )
