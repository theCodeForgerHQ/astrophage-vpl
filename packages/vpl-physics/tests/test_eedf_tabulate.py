"""Tabulation over ``E/N`` and the three-database spread — doc 03 §3.2, doc 09 §2.1.

Doc 03 §3.2 names exactly what has to come out of the solver::

    k_iz(E/N),  k_ex,j(E/N),  mu_e(E/N),  D_e(E/N),  <eps>(E/N)

Doc 09 §2.1 then says the disagreement between the three electron databases "is a term in
the error budget (doc 06 §4, term 2) rather than an unstated risk". So the spread is a
returned number, not a plot somebody makes once.

Every cross-section set below is written in this module; doc 09 §5 keeps the real ones out
of the repository.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vpl.physics.atomic.dataset import ElectronDatabase
from vpl.physics.atomic.lxcat import CrossSection, CrossSectionSet, ProcessType
from vpl.physics.eedf.grid import EnergyGrid
from vpl.physics.eedf.kinetics import kinetics_from_set
from vpl.physics.eedf.solver import TwoTermSolver
from vpl.physics.eedf.tabulate import (
    DatabaseSpread,
    RateTable,
    TabulatedQuantity,
    tabulate,
    tabulate_each_electron_set,
    tabulate_electron_sets,
)

ARGON_MASS_RATIO = 1.36e-5
FIELDS_TD = np.array([5.0, 10.0, 20.0, 40.0])


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
        "energy_ev": np.array([0.0, 1000.0]),
        "sigma_m2": np.array([2e-19, 2e-19]),
        "parameters": {},
    }
    return CrossSection(**{**defaults, **overrides})  # type: ignore[arg-type]


def _argon_like(*, scale: float = 1.0, database: str = "synthetic") -> CrossSectionSet:
    """A three-channel gas: momentum transfer, one excitation, one ionisation."""
    return CrossSectionSet(
        database=database,
        sections=(
            _section(
                ProcessType.EFFECTIVE,
                mass_ratio=ARGON_MASS_RATIO,
                sigma_m2=np.array([2e-19, 2e-19]) * scale,
            ),
            _section(
                ProcessType.EXCITATION,
                threshold_ev=11.5,
                products=("E", "Ar*"),
                energy_ev=np.array([11.5, 1000.0]),
                sigma_m2=np.array([0.0, 5e-21]) * scale,
            ),
            _section(
                ProcessType.IONIZATION,
                threshold_ev=15.76,
                products=("E", "E", "Ar+"),
                energy_ev=np.array([15.76, 1000.0]),
                sigma_m2=np.array([0.0, 3e-21]) * scale,
            ),
        ),
        reference="synthetic fixture",
    )


def _solver(*, scale: float = 1.0, database: str = "synthetic") -> TwoTermSolver:
    grid = EnergyGrid.linear(max_ev=200.0, n_cells=400)
    return TwoTermSolver(
        kinetics=kinetics_from_set(_argon_like(scale=scale, database=database), grid)
    )


# ── the table doc 03 §3.2 asks for ─────────────────────────────────────────────


class TestTheTable:
    @pytest.mark.physics
    def test_it_carries_every_quantity_doc_03_section_3_2_lists(self) -> None:
        table = tabulate(_solver(), FIELDS_TD)

        assert table.reduced_field_td.shape == FIELDS_TD.shape
        assert table.mean_energy_ev.shape == FIELDS_TD.shape
        assert table.reduced_mobility.shape == FIELDS_TD.shape
        assert table.reduced_diffusion.shape == FIELDS_TD.shape
        assert set(table.rate_coefficients) == {
            "E + Ar -> E + Ar*",
            "E + Ar -> E + E + Ar+",
        }
        assert table.ionisation.shape == FIELDS_TD.shape

    def test_the_length_is_the_number_of_fields(self) -> None:
        assert len(tabulate(_solver(), FIELDS_TD)) == 4

    def test_the_database_is_carried_through(self) -> None:
        assert tabulate(_solver(database="Biagi"), FIELDS_TD).database == "Biagi"

    @pytest.mark.physics
    def test_the_mean_energy_rises_with_the_reduced_field(self) -> None:
        table = tabulate(_solver(), FIELDS_TD)

        assert np.all(np.diff(table.mean_energy_ev) > 0.0)

    @pytest.mark.physics
    def test_the_ionisation_rate_rises_with_the_reduced_field(self) -> None:
        table = tabulate(_solver(), FIELDS_TD)

        assert np.all(np.diff(table.ionisation) > 0.0)

    @pytest.mark.physics
    def test_the_excitation_rate_exceeds_the_ionisation_rate_at_every_field(self) -> None:
        # Lower threshold, so it samples a less depleted part of the tail. True for argon
        # over this whole range and a cheap check that the channels have not been swapped.
        table = tabulate(_solver(), FIELDS_TD)

        assert np.all(table.excitations["E + Ar -> E + Ar*"] > table.ionisation)

    @pytest.mark.physics
    def test_the_characteristic_energy_is_the_einstein_ratio(self) -> None:
        table = tabulate(_solver(), FIELDS_TD)

        np.testing.assert_allclose(
            table.characteristic_energy_ev, table.reduced_diffusion / table.reduced_mobility
        )

    def test_the_quantities_are_reachable_by_their_manifest_name(self) -> None:
        table = tabulate(_solver(), FIELDS_TD)

        np.testing.assert_allclose(
            table.column(TabulatedQuantity.REDUCED_MOBILITY), table.reduced_mobility
        )
        np.testing.assert_allclose(
            table.column(TabulatedQuantity.IONISATION_RATE), table.ionisation
        )

    @pytest.mark.physics
    def test_warm_starting_does_not_change_the_answer(self) -> None:
        # The sweep warm-starts each solve from the previous field. It has to be
        # verifiably invisible in the result: an initial guess that moved the answer would
        # mean the iteration had stopped short of its fixed point rather than at it.
        solver = _solver()
        table = tabulate(solver, FIELDS_TD)

        for index, field in enumerate(FIELDS_TD):
            assert solver.solve(float(field)).mean_energy_ev == pytest.approx(
                table.mean_energy_ev[index], rel=1e-8
            )

    def test_the_iteration_count_is_reported_per_field(self) -> None:
        # Reported because a sweep whose iteration count climbs at the top end is a sweep
        # running out of grid, and that is worth seeing without re-running anything.
        table = tabulate(_solver(), FIELDS_TD)

        assert table.iterations.shape == FIELDS_TD.shape
        assert np.all(table.iterations > 0)

    def test_the_fields_must_be_increasing(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            tabulate(_solver(), np.array([30.0, 10.0]))

    def test_an_empty_field_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            tabulate(_solver(), np.array([]))

    def test_the_repr_names_the_database_and_the_range(self) -> None:
        text = repr(tabulate(_solver(), FIELDS_TD))

        assert "synthetic" in text
        assert "40" in text


# ── the three-database spread (doc 09 §2.1, doc 06 §4 term 2) ───────────────────


def _three_sets() -> Iterator[tuple[ElectronDatabase, CrossSectionSet]]:
    """Three sets that disagree by construction, as doc 09 §2.1 says the real ones do."""
    for database, scale in zip(ElectronDatabase, (1.0, 1.2, 0.85), strict=True):
        yield database, _argon_like(scale=scale, database=database.value)


class _StubStore:
    """Duck-typed stand-in for :class:`~vpl.physics.atomic.store.AtomicDataStore`.

    The real store verifies cached files against the version lock, which needs data on
    disk; doc 09 §5 keeps that out of the repository. Only ``each_electron_set`` is used.
    """

    def each_electron_set(self) -> Iterator[tuple[ElectronDatabase, CrossSectionSet]]:
        return _three_sets()


class TestTheDatabaseSpread:
    @staticmethod
    def _spread() -> DatabaseSpread:
        return tabulate_electron_sets(
            _three_sets(),
            reduced_fields_td=FIELDS_TD,
            grid=EnergyGrid.linear(max_ev=200.0, n_cells=400),
        )

    def test_every_electron_database_gets_its_own_table(self) -> None:
        spread = self._spread()

        assert set(spread.tables) == set(ElectronDatabase)
        assert all(isinstance(t, RateTable) for t in spread.tables.values())

    @pytest.mark.physics
    def test_the_spread_is_a_number_and_not_a_plot(self) -> None:
        spread = self._spread()

        relative = spread.relative_spread(TabulatedQuantity.IONISATION_RATE)

        assert relative.shape == FIELDS_TD.shape
        assert np.all(relative > 0.0)

    @pytest.mark.physics
    def test_identical_databases_produce_no_spread(self) -> None:
        identical = ((db, _argon_like(database=db.value)) for db in ElectronDatabase)

        spread = tabulate_electron_sets(
            identical,
            reduced_fields_td=FIELDS_TD,
            grid=EnergyGrid.linear(max_ev=200.0, n_cells=400),
        )

        np.testing.assert_allclose(
            spread.relative_spread(TabulatedQuantity.MEAN_ENERGY), 0.0, atol=1e-12
        )

    @pytest.mark.physics
    def test_the_worst_case_spread_is_reported_as_one_number(self) -> None:
        spread = self._spread()

        worst = spread.max_relative_spread(TabulatedQuantity.IONISATION_RATE)

        assert worst == pytest.approx(
            float(spread.relative_spread(TabulatedQuantity.IONISATION_RATE).max())
        )

    def test_the_summary_names_every_database_and_every_quantity(self) -> None:
        summary = self._spread().summary()

        for database in ElectronDatabase:
            assert database.value in summary
        for quantity in TabulatedQuantity:
            assert quantity.value in summary

    def test_the_fields_are_shared_by_every_table(self) -> None:
        spread = self._spread()

        np.testing.assert_allclose(spread.reduced_field_td, FIELDS_TD)

    def test_a_store_can_be_swept_directly(self) -> None:
        spread = tabulate_each_electron_set(
            _StubStore(),
            reduced_fields_td=FIELDS_TD,
            grid=EnergyGrid.linear(max_ev=200.0, n_cells=400),
        )

        assert set(spread.tables) == set(ElectronDatabase)

    def test_an_empty_sweep_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one database"):
            tabulate_electron_sets(
                iter(()),
                reduced_fields_td=FIELDS_TD,
                grid=EnergyGrid.linear(max_ev=100.0, n_cells=100),
            )

    def test_tables_on_different_field_grids_cannot_be_compared(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=400)
        one = tabulate(_solver(), FIELDS_TD)
        other = tabulate(
            TwoTermSolver(kinetics=kinetics_from_set(_argon_like(), grid)),
            FIELDS_TD * 2.0,
        )

        with pytest.raises(ValueError, match="same reduced-field"):
            DatabaseSpread(tables={ElectronDatabase.PHELPS: one, ElectronDatabase.BIAGI: other})


class TestSpreadPresentation:
    def test_a_spread_over_no_databases_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one database"):
            DatabaseSpread(tables={})

    def test_the_repr_names_the_database_count_and_the_field_count(self) -> None:
        spread = tabulate_electron_sets(
            _three_sets(),
            reduced_fields_td=FIELDS_TD,
            grid=EnergyGrid.linear(max_ev=200.0, n_cells=400),
        )

        assert "3 databases" in repr(spread)
        assert "4 points" in repr(spread)
