"""Line emissivity — doc 04 §2.1.

``eps_ul = n_u A_ul h nu_ul / 4 pi``. Small enough that the only thing worth checking is
that it *is* that expression, evaluated with CODATA constants and a wavelength in the
units it claims — which is exactly the class of error that makes a synthetic spectrum
plausible and wrong by nine orders of magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import constants as sc

from oes_system import energy_grid, plain_system, resonance_trapped_system
from vpl.instruments.oes.cr import CollisionalRadiativeModel
from vpl.instruments.oes.emissivity import (
    chord_radiance,
    emission_spectrum,
    line_emissivity,
)
from vpl.instruments.oes.levels import LevelSystem
from vpl.physics.eedf.analytic import maxwellian_eedf
from vpl.physics.eedf.grid import EnergyGrid


@pytest.fixture(scope="module")
def grid() -> EnergyGrid:
    return energy_grid()


@pytest.fixture(scope="module")
def system(grid: EnergyGrid) -> LevelSystem:
    return plain_system(grid)


@pytest.mark.physics
def test_emissivity_is_the_doc_04_expression() -> None:
    """``n_u A_ul h c / (4 pi lambda)``, with h and c from CODATA and lambda in metres."""
    n_u, a_ul, wavelength_nm = 1e14, 3.31e7, 811.53
    expected = n_u * a_ul * sc.Planck * sc.speed_of_light / (wavelength_nm * 1e-9) / (4.0 * np.pi)
    assert line_emissivity(
        upper_density_per_m3=n_u, a_ul_per_s=a_ul, wavelength_nm=wavelength_nm
    ) == pytest.approx(expected, rel=1e-12)


@pytest.mark.physics
def test_emissivity_at_the_reference_point_is_physically_sized() -> None:
    """A sanity anchor with units in it.

    1e14 m^-3 of Ar(2p9) radiating at 3.31e7 /s emits 3.3e21 photons m^-3 s^-1, and at
    1.53 eV each that is 8.1e2 W m^-3, or 64 W m^-3 sr^-1. A model returning microwatts or
    megawatts per cubic metre is wrong by a power of ten thousand and no dimensionless test
    would say so.
    """
    emissivity = line_emissivity(upper_density_per_m3=1e14, a_ul_per_s=3.31e7, wavelength_nm=811.53)
    assert 10.0 < emissivity < 100.0


@pytest.mark.physics
def test_emissivity_is_linear_in_population_and_in_the_transition_probability() -> None:
    base = line_emissivity(upper_density_per_m3=1e14, a_ul_per_s=3.31e7, wavelength_nm=750.39)
    doubled = line_emissivity(upper_density_per_m3=2e14, a_ul_per_s=3.31e7, wavelength_nm=750.39)
    stronger = line_emissivity(upper_density_per_m3=1e14, a_ul_per_s=6.62e7, wavelength_nm=750.39)
    assert doubled == pytest.approx(2.0 * base)
    assert stronger == pytest.approx(2.0 * base)


@pytest.mark.physics
def test_a_bluer_photon_carries_more_energy_at_equal_population_and_rate() -> None:
    blue = line_emissivity(upper_density_per_m3=1e14, a_ul_per_s=1e7, wavelength_nm=400.0)
    red = line_emissivity(upper_density_per_m3=1e14, a_ul_per_s=1e7, wavelength_nm=800.0)
    assert blue == pytest.approx(2.0 * red)


def test_emissivity_rejects_a_negative_population() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        line_emissivity(upper_density_per_m3=-1.0, a_ul_per_s=1e7, wavelength_nm=500.0)


# ── the spectrum ────────────────────────────────────────────────────────────────


@pytest.mark.physics
def test_the_spectrum_carries_one_entry_per_radiative_channel(
    grid: EnergyGrid, system: LevelSystem
) -> None:
    f0 = grid.normalise(maxwellian_eedf(grid.centres_ev, electron_temperature_ev=3.0))
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})
    populations = model.solve(electron_density_per_m3=1e17, ground_density_per_m3=1.6e20, f0=f0)

    lines = emission_spectrum(populations)

    assert len(lines) == len(system.radiative)
    for line, channel in zip(lines, system.radiative, strict=True):
        assert line.wavelength_nm == channel.wavelength_nm
        assert line.emissivity_w_per_m3_sr == pytest.approx(
            line_emissivity(
                upper_density_per_m3=populations[channel.upper],
                a_ul_per_s=channel.a_ul_per_s,
                wavelength_nm=channel.wavelength_nm,
            )
        )


@pytest.mark.physics
def test_trapping_suppresses_what_escapes_but_not_what_is_emitted(grid: EnergyGrid) -> None:
    """doc 04 §2.1 writes the emissivity without ``Lambda``; the escaping flux carries it.

    Both are reported. Confusing them is a factor of ``Lambda`` — three orders of magnitude
    on the resonance line — in whichever direction the confusion runs.
    """
    f0 = grid.normalise(maxwellian_eedf(grid.centres_ev, electron_temperature_ev=3.0))
    model = CollisionalRadiativeModel(
        system=resonance_trapped_system(grid),
        grid=grid,
        wall_loss_per_s={"m": 1.0e4},
        trapping_path_length_m=0.4,
    )
    populations = model.solve(electron_density_per_m3=1e17, ground_density_per_m3=1.6e20, f0=f0)
    resonance = next(line for line in emission_spectrum(populations) if line.upper == "r")

    assert resonance.escape_factor < 1e-2
    assert resonance.escaping_emissivity_w_per_m3_sr == pytest.approx(
        resonance.emissivity_w_per_m3_sr * resonance.escape_factor
    )
    assert resonance.escaping_emissivity_w_per_m3_sr < resonance.emissivity_w_per_m3_sr


@pytest.mark.physics
def test_photon_and_energy_emissivity_differ_by_the_photon_energy(
    grid: EnergyGrid, system: LevelSystem
) -> None:
    f0 = grid.normalise(maxwellian_eedf(grid.centres_ev, electron_temperature_ev=3.0))
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})
    populations = model.solve(electron_density_per_m3=1e17, ground_density_per_m3=1.6e20, f0=f0)
    for line in emission_spectrum(populations):
        photon_energy_j = sc.Planck * sc.speed_of_light / (line.wavelength_nm * 1e-9)
        assert line.photon_emissivity_per_m3_s_sr == pytest.approx(
            line.emissivity_w_per_m3_sr / photon_energy_j
        )


# ── the chord ───────────────────────────────────────────────────────────────────


@pytest.mark.physics
def test_chord_radiance_is_emissivity_times_path(grid: EnergyGrid, system: LevelSystem) -> None:
    f0 = grid.normalise(maxwellian_eedf(grid.centres_ev, electron_temperature_ev=3.0))
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})
    populations = model.solve(electron_density_per_m3=1e17, ground_density_per_m3=1.6e20, f0=f0)
    line = emission_spectrum(populations)[0]
    assert chord_radiance(line, path_length_m=0.4) == pytest.approx(
        line.escaping_emissivity_w_per_m3_sr * 0.4
    )


def test_chord_radiance_rejects_a_non_positive_path(grid: EnergyGrid, system: LevelSystem) -> None:
    f0 = grid.normalise(maxwellian_eedf(grid.centres_ev, electron_temperature_ev=3.0))
    model = CollisionalRadiativeModel(system=system, grid=grid, wall_loss_per_s={"m": 1.0e4})
    populations = model.solve(electron_density_per_m3=1e17, ground_density_per_m3=1.6e20, f0=f0)
    with pytest.raises(ValueError, match="positive"):
        chord_radiance(emission_spectrum(populations)[0], path_length_m=0.0)
