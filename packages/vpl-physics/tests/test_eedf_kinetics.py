"""Cross sections to solver input — doc 03 §3.2, §4.5, doc 09 §5.

The two-term solve needs three things from the atomic layer and nothing else: a
momentum-transfer cross section on the energy grid, the mass ratio ``m_e / M`` that sets
the elastic energy loss, and one channel per inelastic process with its threshold. This
module is the join, and it is where the extrapolation policy of
:mod:`vpl.physics.atomic.interpolation` has to be *decided* rather than inherited by
accident.

No data is read from disk here. doc 09 §5 keeps LXCat exports out of the repository, so
every fixture below is written in this module.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.physics.atomic.interpolation import ExtrapolationPolicy, OutsideTabulatedRangeError
from vpl.physics.atomic.lxcat import CrossSection, CrossSectionSet, ProcessType
from vpl.physics.eedf.grid import EnergyGrid
from vpl.physics.eedf.kinetics import (
    DEFAULT_ABOVE_GRID,
    ElectronKinetics,
    InelasticChannel,
    kinetics_from_set,
)

# ── synthetic fixtures (doc 09 §5: no downloaded data in tests) ──────────────────

ARGON_MASS_RATIO = 1.36e-5


def _section(process: ProcessType, **overrides: object) -> CrossSection:
    defaults: dict[str, object] = {
        "process": process,
        "database": "synthetic",
        "projectile": "e",
        "target": "Ar",
        "reactants": ("E", "Ar"),
        "products": ("E", "Ar"),
        "threshold_ev": None,
        "mass_ratio": None,
        "energy_ev": np.array([0.1, 1.0, 10.0, 100.0]),
        "sigma_m2": np.array([1e-20, 2e-20, 1e-20, 5e-21]),
        "parameters": {},
    }
    return CrossSection(**{**defaults, **overrides})  # type: ignore[arg-type]


def _effective() -> CrossSection:
    return _section(ProcessType.EFFECTIVE, mass_ratio=ARGON_MASS_RATIO)


def _elastic() -> CrossSection:
    return _section(ProcessType.ELASTIC, mass_ratio=ARGON_MASS_RATIO)


def _excitation(threshold_ev: float = 11.5) -> CrossSection:
    return _section(
        ProcessType.EXCITATION,
        threshold_ev=threshold_ev,
        products=("E", "Ar*"),
        energy_ev=np.array([threshold_ev, 20.0, 100.0]),
        sigma_m2=np.array([0.0, 3e-21, 1e-21]),
    )


def _ionisation(threshold_ev: float = 15.76) -> CrossSection:
    return _section(
        ProcessType.IONIZATION,
        threshold_ev=threshold_ev,
        products=("E", "E", "Ar+"),
        energy_ev=np.array([threshold_ev, 50.0, 100.0]),
        sigma_m2=np.array([0.0, 2.5e-20, 2e-20]),
    )


def _set(*sections: CrossSection, database: str = "synthetic") -> CrossSectionSet:
    return CrossSectionSet(database=database, sections=sections, reference="synthetic fixture")


def _grid() -> EnergyGrid:
    return EnergyGrid.linear(max_ev=80.0, n_cells=200)


# ── the join ────────────────────────────────────────────────────────────────────


class TestAssembly:
    def test_the_momentum_transfer_channel_is_found_as_effective(self) -> None:
        kinetics = kinetics_from_set(_set(_effective(), _ionisation()), _grid())

        assert kinetics.momentum_transfer_m2.shape == (200,)
        assert np.all(kinetics.momentum_transfer_m2 > 0.0)

    def test_the_momentum_transfer_channel_is_found_as_elastic(self) -> None:
        # Phelps publishes argon momentum transfer as EFFECTIVE, Biagi as ELASTIC.
        # doc 09 §2.1 keeps both, so neither spelling may be the one that works.
        kinetics = kinetics_from_set(_set(_elastic(), _ionisation()), _grid())

        assert np.all(kinetics.momentum_transfer_m2 > 0.0)

    def test_the_mass_ratio_comes_from_the_momentum_transfer_block(self) -> None:
        kinetics = kinetics_from_set(_set(_effective(), _ionisation()), _grid())

        assert kinetics.mass_ratio == pytest.approx(ARGON_MASS_RATIO)

    def test_an_explicit_mass_ratio_overrides_the_published_one(self) -> None:
        kinetics = kinetics_from_set(_set(_effective(), _ionisation()), _grid(), mass_ratio=2.0e-5)

        assert kinetics.mass_ratio == pytest.approx(2.0e-5)

    def test_a_set_without_a_published_mass_ratio_demands_one(self) -> None:
        bare = _section(ProcessType.ELASTIC)  # mass_ratio is None

        with pytest.raises(ValueError, match="mass ratio"):
            kinetics_from_set(_set(bare, _ionisation()), _grid())

    def test_every_inelastic_channel_is_carried_with_its_threshold(self) -> None:
        kinetics = kinetics_from_set(
            _set(_effective(), _excitation(11.5), _excitation(13.0), _ionisation()), _grid()
        )

        assert len(kinetics.channels) == 3
        assert sorted(c.threshold_ev for c in kinetics.channels) == [11.5, 13.0, 15.76]

    def test_ionisation_and_excitation_are_separable(self) -> None:
        kinetics = kinetics_from_set(_set(_effective(), _excitation(), _ionisation()), _grid())

        assert len(kinetics.ionisation) == 1
        assert len(kinetics.excitations) == 1
        assert kinetics.ionisation[0].is_ionisation
        assert not kinetics.excitations[0].is_ionisation

    def test_channels_are_keyed_by_their_reaction(self) -> None:
        kinetics = kinetics_from_set(_set(_effective(), _ionisation()), _grid())

        assert kinetics.channels[0].reaction == "E + Ar -> E + E + Ar+"

    def test_the_database_is_carried_through(self) -> None:
        kinetics = kinetics_from_set(_set(_effective(), database="Biagi"), _grid())

        assert kinetics.database == "Biagi"

    def test_a_set_with_no_momentum_transfer_channel_is_refused(self) -> None:
        with pytest.raises(LookupError, match="momentum-transfer"):
            kinetics_from_set(_set(_ionisation()), _grid())

    def test_an_elastic_only_set_is_allowed(self) -> None:
        # The Maxwellian and Druyvesteyn limits are elastic-only by construction, so a
        # set with no inelastic channel is a legitimate verification case, not an error.
        kinetics = kinetics_from_set(_set(_effective()), _grid())

        assert kinetics.channels == ()

    def test_a_zero_momentum_transfer_cross_section_is_refused(self) -> None:
        # sigma_m appears in a denominator in every transport integral. A zero would
        # return inf, which propagates into a rate table looking like a number.
        zeroed = _section(
            ProcessType.EFFECTIVE,
            mass_ratio=ARGON_MASS_RATIO,
            energy_ev=np.array([0.1, 1.0, 100.0]),
            sigma_m2=np.array([1e-20, 0.0, 1e-20]),
        )

        with pytest.raises(ValueError, match="positive"):
            kinetics_from_set(_set(zeroed), _grid())


class TestTheExtrapolationDecision:
    """doc 03 §4.5: the tail is the most consequential atomic-data choice in the project.

    The atomic layer defaults to refusing above the table. This module keeps that
    default rather than choosing for the caller — see the module docstring of
    :mod:`vpl.physics.eedf.kinetics`.
    """

    def test_the_default_above_the_table_is_to_refuse(self) -> None:
        assert DEFAULT_ABOVE_GRID is ExtrapolationPolicy.RAISE

    @pytest.mark.physics
    def test_a_grid_running_past_the_table_raises_by_default(self) -> None:
        short = _section(
            ProcessType.EFFECTIVE,
            mass_ratio=ARGON_MASS_RATIO,
            energy_ev=np.array([0.1, 1.0, 30.0]),
            sigma_m2=np.array([1e-20, 2e-20, 1e-20]),
        )

        with pytest.raises(OutsideTabulatedRangeError):
            kinetics_from_set(_set(short), EnergyGrid.linear(max_ev=80.0, n_cells=100))

    def test_the_policy_can_be_chosen_explicitly(self) -> None:
        short = _section(
            ProcessType.EFFECTIVE,
            mass_ratio=ARGON_MASS_RATIO,
            energy_ev=np.array([0.1, 1.0, 30.0]),
            sigma_m2=np.array([1e-20, 2e-20, 1e-20]),
        )

        kinetics = kinetics_from_set(
            _set(short),
            EnergyGrid.linear(max_ev=80.0, n_cells=100),
            above=ExtrapolationPolicy.CONSTANT,
        )

        assert kinetics.momentum_transfer_m2[-1] == pytest.approx(1e-20)


class TestChannelsAboveTheGrid:
    @pytest.mark.physics
    def test_a_threshold_above_the_grid_top_makes_the_channel_identically_zero(self) -> None:
        # Not an error: a manifest may legitimately sweep a grid shorter than some
        # channel's threshold. The rate coefficient must then be exactly zero rather
        # than a small number produced by an off-grid extrapolation.
        kinetics = kinetics_from_set(
            _set(_effective(), _ionisation(threshold_ev=15.76)),
            EnergyGrid.linear(max_ev=10.0, n_cells=50),
            above=ExtrapolationPolicy.CONSTANT,
        )

        np.testing.assert_array_equal(kinetics.channels[0].sigma_m2, np.zeros(50))

    @pytest.mark.physics
    def test_a_channel_is_exactly_zero_below_its_threshold(self) -> None:
        kinetics = kinetics_from_set(_set(_effective(), _ionisation(threshold_ev=15.76)), _grid())
        channel = kinetics.channels[0]
        below = kinetics.grid.centres_ev < channel.threshold_ev

        np.testing.assert_array_equal(channel.sigma_m2[below], 0.0)


class TestConstruction:
    def test_a_channel_needs_a_positive_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            InelasticChannel(
                reaction="E + Ar -> E + Ar*",
                process=ProcessType.EXCITATION,
                threshold_ev=-1.0,
                sigma_m2=np.zeros(4),
            )

    def test_a_channel_cross_section_cannot_be_negative(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            InelasticChannel(
                reaction="E + Ar -> E + Ar*",
                process=ProcessType.EXCITATION,
                threshold_ev=11.5,
                sigma_m2=np.array([-1e-21, 0.0]),
            )

    def test_the_mass_ratio_must_be_positive_and_small(self) -> None:
        grid = _grid()
        with pytest.raises(ValueError, match="mass ratio"):
            ElectronKinetics(
                grid=grid,
                database="synthetic",
                momentum_transfer_m2=np.full(grid.n_cells, 1e-20),
                momentum_transfer_edge_m2=np.full(grid.n_cells + 1, 1e-20),
                mass_ratio=0.0,
                channels=(),
            )

    def test_arrays_must_match_the_grid(self) -> None:
        grid = _grid()
        with pytest.raises(ValueError, match="one value per cell"):
            ElectronKinetics(
                grid=grid,
                database="synthetic",
                momentum_transfer_m2=np.full(7, 1e-20),
                momentum_transfer_edge_m2=np.full(grid.n_cells + 1, 1e-20),
                mass_ratio=ARGON_MASS_RATIO,
                channels=(),
            )

    def test_the_edge_array_must_match_the_boundaries(self) -> None:
        grid = _grid()
        with pytest.raises(ValueError, match="boundaries"):
            ElectronKinetics(
                grid=grid,
                database="synthetic",
                momentum_transfer_m2=np.full(grid.n_cells, 1e-20),
                momentum_transfer_edge_m2=np.full(grid.n_cells, 1e-20),
                mass_ratio=ARGON_MASS_RATIO,
                channels=(),
            )

    def test_the_repr_names_the_database_and_the_channels(self) -> None:
        kinetics = kinetics_from_set(_set(_effective(), _ionisation()), _grid())

        assert "synthetic" in repr(kinetics)
        assert "1 channel" in repr(kinetics)


class TestMoreRejections:
    def test_a_non_finite_momentum_transfer_cross_section_is_refused(self) -> None:
        grid = _grid()
        with pytest.raises(ValueError, match="finite"):
            ElectronKinetics(
                grid=grid,
                database="synthetic",
                momentum_transfer_m2=np.full(grid.n_cells, np.inf),
                momentum_transfer_edge_m2=np.full(grid.n_cells + 1, 1e-20),
                mass_ratio=ARGON_MASS_RATIO,
                channels=(),
            )

    def test_a_two_dimensional_channel_cross_section_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            InelasticChannel(
                reaction="E + Ar -> E + Ar*",
                process=ProcessType.EXCITATION,
                threshold_ev=11.5,
                sigma_m2=np.zeros((2, 2)),
            )

    def test_a_non_finite_channel_cross_section_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            InelasticChannel(
                reaction="E + Ar -> E + Ar*",
                process=ProcessType.EXCITATION,
                threshold_ev=11.5,
                sigma_m2=np.array([np.nan, 0.0]),
            )

    def test_a_channel_that_does_not_match_the_grid_is_refused(self) -> None:
        grid = _grid()
        with pytest.raises(ValueError, match="but the grid has"):
            ElectronKinetics(
                grid=grid,
                database="synthetic",
                momentum_transfer_m2=np.full(grid.n_cells, 1e-20),
                momentum_transfer_edge_m2=np.full(grid.n_cells + 1, 1e-20),
                mass_ratio=ARGON_MASS_RATIO,
                channels=(
                    InelasticChannel(
                        reaction="E + Ar -> E + Ar*",
                        process=ProcessType.EXCITATION,
                        threshold_ev=11.5,
                        sigma_m2=np.zeros(7),
                    ),
                ),
            )

    def test_the_channel_repr_names_the_reaction_and_threshold(self) -> None:
        channel = InelasticChannel(
            reaction="E + Ar -> E + Ar*",
            process=ProcessType.EXCITATION,
            threshold_ev=11.5,
            sigma_m2=np.zeros(4),
        )

        assert "E + Ar -> E + Ar*" in repr(channel)
        assert "11.5 eV" in repr(channel)
