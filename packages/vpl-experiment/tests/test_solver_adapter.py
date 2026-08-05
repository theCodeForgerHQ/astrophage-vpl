"""The L0 solver as a manifest-resolvable plugin — doc 08 §10, doc 08 §4."""

from __future__ import annotations

from typing import Any

import pytest

from vpl.core.protocols.config import SolverConfig
from vpl.core.registry import PluginGroup, available, load
from vpl.core.state import Fidelity, TimeGrid
from vpl.core.units import Q_
from vpl.experiment import ManifestSolver, resolve_plasma
from vpl.experiment.solvers import AnalyticSheathForwardSolver

#: doc 03 §2.3's headline number at RP-1, to the precision ADR-007 records it.
V03_GAMMA_E_WATT_PER_M2 = 6576.94


@pytest.fixture
def solver() -> AnalyticSheathForwardSolver:
    return AnalyticSheathForwardSolver()


class TestDiscovery:
    def test_the_analytic_solver_is_declared_in_the_vpl_solvers_group(self) -> None:
        assert "vpl.physics.analytic.sheath" in available(PluginGroup.SOLVERS)

    def test_the_name_a_manifest_writes_resolves_to_the_adapter(self) -> None:
        assert load(PluginGroup.SOLVERS, "vpl.physics.analytic.sheath") is (
            AnalyticSheathForwardSolver
        )

    def test_it_satisfies_the_contract_the_manifest_engine_requires(
        self, solver: AnalyticSheathForwardSolver
    ) -> None:
        assert isinstance(solver, ManifestSolver)

    def test_it_reports_the_l0_fidelity_of_doc_03_1(
        self, solver: AnalyticSheathForwardSolver
    ) -> None:
        assert solver.fidelity() is Fidelity.L0


class TestConfiguration:
    def test_it_accepts_an_empty_configuration(self, solver: AnalyticSheathForwardSolver) -> None:
        solver.configure(SolverConfig(values={}))
        assert solver.fidelity() is Fidelity.L0

    def test_it_selects_the_matrix_sheath_model_when_the_manifest_asks_for_it(
        self, solver: AnalyticSheathForwardSolver, runnable_manifest: Any
    ) -> None:
        solver.configure(SolverConfig(values={"model": "matrix"}))
        params = resolve_plasma(runnable_manifest.plasma).params
        matrix_state = solver.solve(params, None)

        child = AnalyticSheathForwardSolver()
        child.configure(SolverConfig(values={"model": "child_langmuir"}))
        assert matrix_state != child.solve(params, None)

    def test_it_refuses_a_key_it_does_not_own(self, solver: AnalyticSheathForwardSolver) -> None:
        # doc 08 §6's own forward block sets `n_ppc`, which belongs to the PIC solver.
        # Pointing the analytic solver at that manifest is a mistake worth naming.
        with pytest.raises(ValueError, match="n_ppc"):
            solver.configure(SolverConfig(values={"n_ppc": 1000}))

    def test_it_refuses_a_sheath_model_that_does_not_exist(
        self, solver: AnalyticSheathForwardSolver
    ) -> None:
        with pytest.raises(ValueError, match="debye"):
            solver.configure(SolverConfig(values={"model": "debye"}))

    def test_it_refuses_an_edge_to_centre_ratio_outside_its_range(
        self, solver: AnalyticSheathForwardSolver
    ) -> None:
        with pytest.raises(ValueError, match="h_l"):
            solver.configure(SolverConfig(values={"h_l": 2.0}))


class TestTheFluxFunctional:
    def test_it_reproduces_the_doc_03_2_3_energy_flux_at_rp1(
        self, solver: AnalyticSheathForwardSolver, runnable_manifest: Any
    ) -> None:
        params = resolve_plasma(runnable_manifest.plasma).params
        flux = solver.flux(solver.solve(params, None), 0.0)

        assert float(flux.energy_flux_toward_wall_watt_per_m2) == pytest.approx(
            V03_GAMMA_E_WATT_PER_M2, rel=1e-5
        )

    def test_it_carries_the_doc_01_1_2_decomposition_and_not_only_the_product(
        self, solver: AnalyticSheathForwardSolver, runnable_manifest: Any
    ) -> None:
        params = resolve_plasma(runnable_manifest.plasma).params
        flux = solver.flux(solver.solve(params, None), 0.0)

        assert float(flux.particle_flux_toward_wall_per_m2_s) > 0.0
        assert float(flux.mean_impact_energy.m_as("eV")) == pytest.approx(250.0, rel=1e-3)

    def test_it_refuses_a_position_away_from_the_wall(
        self, solver: AnalyticSheathForwardSolver, runnable_manifest: Any
    ) -> None:
        params = resolve_plasma(runnable_manifest.plasma).params
        state = solver.solve(params, None)
        with pytest.raises(ValueError, match="wall"):
            solver.flux(state, 0.005)

    def test_it_is_steady_and_reports_itself_as_such(
        self, solver: AnalyticSheathForwardSolver, runnable_manifest: Any
    ) -> None:
        params = resolve_plasma(runnable_manifest.plasma).params
        assert solver.flux(solver.solve(params, None), 0.0).is_steady

    def test_it_refuses_a_time_grid_because_l0_does_not_model_time(
        self, solver: AnalyticSheathForwardSolver, runnable_manifest: Any
    ) -> None:
        params = resolve_plasma(runnable_manifest.plasma).params
        grid = TimeGrid.uniform(duration=Q_(1.0, "s"), n_points=4)
        with pytest.raises(ValueError, match="steady"):
            solver.solve(params, grid)


class TestMetadata:
    def test_it_names_itself_with_the_dotted_path_a_manifest_resolves(
        self, solver: AnalyticSheathForwardSolver
    ) -> None:
        assert solver.metadata().name == "vpl.physics.analytic.sheath"

    def test_it_carries_a_citation_because_doc_00_c2_requires_one(
        self, solver: AnalyticSheathForwardSolver
    ) -> None:
        assert solver.metadata().citations
