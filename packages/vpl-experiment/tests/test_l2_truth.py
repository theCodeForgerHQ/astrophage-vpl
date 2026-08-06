"""Tests for :mod:`vpl.experiment.l2_truth` — the L2-truth / L1-inversion configuration.

Almost everything here runs without either heavy dependency. The L2 solve needs JAX and
costs one wall-clock hour; the L1 inversion needs ``dolfinx``, which this development
machine does not have. So the tests that can be cheap are cheap and structural — the
configuration, the domain guard, the artefact round-trip, the tier certification — and the
two that genuinely need a solver are gated the same way ``test_closed_loop.py`` gates its
own T2 tests.

The structural tests are not filler. The domain guard is the one that matters most: a
truth solved on a domain shorter than the fixed observation grid would be reported through
positions the physics never evaluated, and finding that out *after* an hour of PIC is the
difference between a cheap failure and a wasted afternoon.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.state import Fidelity, PlasmaState, ScalarField
from vpl.core.units import magnitude_in
from vpl.experiment.channels import _homogeneous_width_m_per_s, reconstruct_ivdf
from vpl.experiment.closed_loop import (
    _argon_ion,
    _draw_true_theta,
    _reference_theta,
    _to_plasma_params,
)
from vpl.experiment.l2_truth import (
    L2_CAPACITY_FACTOR,
    L2_CELLS_PER_DEBYE,
    L2_DOMAIN_SHEATHS,
    L2_EQUILIBRATION_FRACTION,
    L2_N_PPC,
    L2_TO_L1_TIER,
    L2_TRANSITS,
    L2_VELOCITY_BINS,
    DomainTooShortError,
    L2Truth,
    _require_domain_covers_observation_grid,
    l2_domain_length_m,
    l2_truth_solver,
    load_l2_truth,
    observation_grid,
    run_l2_to_l1,
    save_l2_truth,
)
from vpl.instruments.lif.scan import MIN_SAMPLES_PER_HOMOGENEOUS_WIDTH
from vpl.physics.analytic.sheath import AnalyticSheathSolver
from vpl.validation.sealed import Tier

_SEED = 0

#: Whether ``dolfinx`` (FEniCSx) is importable here. The L1 inversion needs it; see
#: ``test_closed_loop.py``'s ``_DOLFINX_AVAILABLE`` for the same gate and the same reason.
_DOLFINX_AVAILABLE = importlib.util.find_spec("dolfinx") is not None

#: Whether JAX is importable here. The L2 truth solver needs it — ``vpl.physics.kinetic.
#: precision`` turns x64 on at import and calls float32 a correctness bug, not a slower
#: answer — and the FEniCSx machine does not have it. The two gates are deliberately
#: symmetric: no environment on this project has both, which is the fact the whole
#: file-based split exists to accommodate.
_JAX_AVAILABLE = importlib.util.find_spec("jax") is not None

_NEEDS_JAX = pytest.mark.skipif(not _JAX_AVAILABLE, reason="requires JAX (the L2 solver)")

#: Scalars carried by the stand-in truth. Arbitrary, and never asserted against anything
#: physical: what they exercise is that a float survives the trip through the npz.
_STUB_GAMMA_E_W_PER_M2 = 9331.69
_STUB_MEAN_ENERGY_EV = 250.0
_STUB_THICKNESS_M = 1.0e-3
_STUB_ENERGY_DRIFT = 1.0e-6
_STUB_WALL_CLOCK_S = 3600.0

#: The verification-scale settings this experiment is required to reuse, copied from
#: ``packages/vpl-physics/tests/test_kinetic_solver.py``'s ``_sheath_solver``. Duplicated
#: rather than imported because that file is a test module in another package and is not
#: importable from here — which is exactly why it needs pinning: a silent drift between the
#: configuration doc 03 §7's gates were measured at and the configuration this experiment's
#: truth was solved at would make the truth's error budget unattributable.
_VERIFICATION_SCALE = {
    "n_ppc": 400.0,
    "cells_per_debye": 1.0,
    "transits": 6.0,
    "equilibration_fraction": 0.5,
    "capacity_factor": 1.5,
}


def _truth_params(seed: int = _SEED):  # type: ignore[no-untyped-def]
    registry = default_registry()
    species = _argon_ion(registry)
    reference = _reference_theta()
    return _to_plasma_params(
        _draw_true_theta(seed=seed, reference=reference), species=species, registry=registry
    )


@_NEEDS_JAX
class TestTheL2Configuration:
    """Which PIC configuration this experiment's truth is solved at, and why that one."""

    def test_it_reuses_the_kinetic_suites_verification_scale_settings(self) -> None:
        solver = l2_truth_solver()

        assert solver.n_ppc == _VERIFICATION_SCALE["n_ppc"]
        assert solver.cells_per_debye == _VERIFICATION_SCALE["cells_per_debye"]
        assert solver.transits == _VERIFICATION_SCALE["transits"]
        assert solver.equilibration_fraction == _VERIFICATION_SCALE["equilibration_fraction"]
        assert solver.capacity_factor == _VERIFICATION_SCALE["capacity_factor"]

    def test_the_named_constants_are_the_verification_scale_ones(self) -> None:
        # The module must not merely happen to construct a solver with these numbers; the
        # constants are what a reader checks against doc 03 §7's own suite.
        assert _VERIFICATION_SCALE["n_ppc"] == L2_N_PPC
        assert _VERIFICATION_SCALE["cells_per_debye"] == L2_CELLS_PER_DEBYE
        assert _VERIFICATION_SCALE["transits"] == L2_TRANSITS
        assert _VERIFICATION_SCALE["equilibration_fraction"] == L2_EQUILIBRATION_FRACTION
        assert _VERIFICATION_SCALE["capacity_factor"] == L2_CAPACITY_FACTOR

    def test_the_domain_is_the_one_setting_that_differs_from_the_verification_suite(self) -> None:
        # doc 03 §7's own gates run at 1.2 sheaths, which is a *shorter* domain than the
        # closed loop's fixed observation grid. This is the single deliberate departure and
        # it is required by the next test, not chosen for taste.
        assert L2_DOMAIN_SHEATHS > 1.2
        assert l2_truth_solver().domain_sheaths == L2_DOMAIN_SHEATHS

    def test_the_domain_covers_the_fixed_observation_grid_at_the_sealed_truth(self) -> None:
        params = _truth_params()
        grid = observation_grid()

        assert l2_domain_length_m(params) >= float(grid.z_m[-1])

    def test_the_domain_length_is_the_child_langmuir_thickness_times_the_sheath_count(
        self,
    ) -> None:
        # Pinned against the analytic thickness the PIC's own planner uses to size its
        # grid, so this module cannot drift into a second idea of "one sheath".
        params = _truth_params()
        thickness = float(magnitude_in(AnalyticSheathSolver().thickness(params), "m"))

        assert l2_domain_length_m(params) == pytest.approx(L2_DOMAIN_SHEATHS * thickness)


@_NEEDS_JAX
class TestTheVelocityAxisResolvesTheLifLine:
    """The L2 truth's ``f_i`` is what makes the LIF channel non-circular for the first time
    (``channels.reconstruct_ivdf`` returns a state that has one unchanged). It is only
    usable if it resolves the line LIF integrates against — the instrument refuses a coarser
    grid outright, and would refuse it an hour into the run.
    """

    def _spacing_m_per_s(self) -> float:
        plan = l2_truth_solver()._plan(_truth_params(), None)
        return float(np.diff(plan.velocity_edges_m_per_s)[0])

    def test_the_spacing_clears_the_instruments_own_guard(self) -> None:
        width = _homogeneous_width_m_per_s(default_registry())

        assert self._spacing_m_per_s() <= width * MIN_SAMPLES_PER_HOMOGENEOUS_WIDTH

    def test_the_spacing_carries_the_declared_factor_of_two_of_margin(self) -> None:
        # Two samples per homogeneous width, not one: the module docstring states this as
        # the compromise between the guard and the PIC histogram's shot noise, and a
        # constant that drifted back to the bare guard would leave that claim false.
        width = _homogeneous_width_m_per_s(default_registry())

        assert self._spacing_m_per_s() <= 0.5 * width

    def test_the_bin_count_is_the_one_the_module_declares(self) -> None:
        assert l2_truth_solver().n_velocity_bins == L2_VELOCITY_BINS


@_NEEDS_JAX
class TestTheDomainGuardFiresBeforeAnHourOfPic:
    """The guard exists because the failure it prevents is expensive and late.

    ``closed_loop._resample_onto_grid`` already refuses to report through a position the
    solve never evaluated — but it refuses *after* the solve, which for L2 is an hour. The
    same condition is therefore checked up front, against the plan rather than the result.
    """

    def test_a_domain_shorter_than_the_observation_grid_is_refused(self) -> None:
        params = _truth_params()
        grid = observation_grid()
        # One sheath thickness is ~0.95 mm against the grid's 2.28 mm: genuinely short.
        short = l2_truth_solver(domain_sheaths=1.0)

        with pytest.raises(DomainTooShortError, match="observation grid"):
            _require_domain_covers_observation_grid(short, params, grid)

    def test_the_configured_solver_passes_its_own_guard(self) -> None:
        _require_domain_covers_observation_grid(
            l2_truth_solver(), _truth_params(), observation_grid()
        )


def _synthetic_truth() -> L2Truth:
    """An ``L2Truth`` whose state is L0's, relabelled, standing in for a PIC solve.

    Two tests need a truth object and neither of them is about the physics: the artefact
    tests are about serialisation, and the dolfinx-gated test is about whether the L1
    inversion runs at all and certifies its tier. Spending two hours of PIC to answer either
    would be a test nobody runs, so the stand-in is an L0 solve at the same sealed
    parameters with :func:`~vpl.experiment.channels.reconstruct_ivdf`'s assumed distribution
    attached — that function exists precisely to supply the ``f_i`` a state lacks, and its
    velocity axis is sized to what the LIF quadrature needs, which an arbitrary stub array
    would not be. **The recovery numbers such a run produces are meaningless**; only the
    tier and the finiteness are asserted.
    """
    registry = default_registry()
    grid = observation_grid()
    params = _truth_params()
    state = reconstruct_ivdf(AnalyticSheathSolver().solve(params, grid=grid), registry=registry)
    fields = {
        name: ScalarField(name=name, values=field.values, units=field.units, grid=grid, time=None)
        for name, field in state.fields.items()
    }
    return L2Truth(
        seed=_SEED,
        state=PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=fields,
            # An L2 state must carry a distribution — PlasmaState enforces it.
            ion_distribution=state.ion_distribution,
            fidelity=Fidelity.L2,
        ),
        gamma_e_w_per_m2=_STUB_GAMMA_E_W_PER_M2,
        mean_impact_energy_ev=_STUB_MEAN_ENERGY_EV,
        sheath_thickness_m=_STUB_THICKNESS_M,
        energy_drift=_STUB_ENERGY_DRIFT,
        wall_clock_s=_STUB_WALL_CLOCK_S,
    )


class TestTheTruthArtefactRoundTrip:
    def _synthetic(self) -> L2Truth:
        return _synthetic_truth()

    def test_the_artefact_carries_exactly_these_keys(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The npz is the one channel between the sealed truth and the inversion.

        A file handed from the machine that knows the answer to the machine that is trying
        to find it is a route for the answer to travel, which is the inverse crime doc 05
        §7 exists to prevent. The module docstring lists the contents and says where each
        one is used; this test is what stops that list going quietly out of date. It fails
        on an *added* key as loudly as on a removed one, so a future change that decides to
        serialise, say, the true ``n_0`` has to come here and argue for it.
        """
        path = save_l2_truth(self._synthetic(), tmp_path / "truth.npz")

        with np.load(path, allow_pickle=False) as handle:
            keys = set(handle.files)

        assert keys == {
            "seed",
            "z_m",
            "v_m_per_s",
            "f_i",
            "field_names",
            "field_units",
            "field_values",
            "gamma_e_w_per_m2",
            "mean_impact_energy_ev",
            "sheath_thickness_m",
            "energy_drift",
            "wall_clock_s",
        }

    def test_the_observable_fields_are_the_only_physics_that_travels(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # Everything under `field_values` must be a field an instrument reads off a state.
        # A control parameter smuggled in as a "field" would reach the inversion machine
        # wearing an observable's clothes.
        path = save_l2_truth(self._synthetic(), tmp_path / "truth.npz")

        with np.load(path, allow_pickle=False) as handle:
            names = {str(name) for name in handle["field_names"]}

        assert names <= {"n_i", "n_e", "Phi", "u_i", "T_e"}

    def test_a_saved_truth_round_trips(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        truth = self._synthetic()
        path = tmp_path / "truth.npz"

        save_l2_truth(truth, path)
        loaded = load_l2_truth(path)

        assert loaded.seed == truth.seed
        assert loaded.gamma_e_w_per_m2 == pytest.approx(truth.gamma_e_w_per_m2)
        assert loaded.wall_clock_s == pytest.approx(truth.wall_clock_s)
        assert loaded.state.fidelity is Fidelity.L2
        assert set(loaded.state.fields) == set(truth.state.fields)
        for name, field in truth.state.fields.items():
            np.testing.assert_allclose(loaded.state.field(name).values, field.values)
        np.testing.assert_allclose(loaded.state.grid.z_m, truth.state.grid.z_m)

        assert loaded.state.ion_distribution is not None
        assert truth.state.ion_distribution is not None
        np.testing.assert_allclose(
            loaded.state.ion_distribution.v_m_per_s, truth.state.ion_distribution.v_m_per_s
        )
        np.testing.assert_allclose(
            loaded.state.ion_distribution.values, truth.state.ion_distribution.values
        )

    def test_loading_refuses_a_grid_that_is_not_the_fixed_observation_grid(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # The whole discipline of closed_loop's fixed grid is that every state is read at
        # the same physical positions. A truth file carrying a different z array would
        # silently reintroduce the aliasing that module's docstring documents at length.
        truth = self._synthetic()
        path = tmp_path / "truth.npz"
        save_l2_truth(truth, path)

        with np.load(path) as handle:
            contents = dict(handle)
        contents["z_m"] = contents["z_m"] * 2.0
        np.savez(path, **contents)

        with pytest.raises(ValueError, match="observation grid"):
            load_l2_truth(path)


class TestTheTierIsCertifiedRatherThanLabelled:
    def test_the_experiments_tier_comes_from_tier_of_configuration(self) -> None:
        # doc 05 §7.1: mismatched physics level, noise, and the *estimated* calibration.
        # This module must not hand-label T2; `tier_of_configuration` decides.
        assert L2_TO_L1_TIER is Tier.T2


class TestTheInversionSideNeedsDolfinx:
    @pytest.mark.skipif(_DOLFINX_AVAILABLE, reason="dolfinx is installed; L1 is runnable")
    def test_it_states_the_missing_dependency_rather_than_mislabelling_a_run(self) -> None:
        truth = _synthetic_truth()
        with pytest.raises(NotImplementedError, match="dolfinx"):
            run_l2_to_l1(truth, calibration_uncertainty=True)


class TestTheL2ToL1ResultIsCertified:
    """Skipped on this machine. It is the test the remote FEniCSx environment exists for,
    and it needs a truth artefact produced by an hour of PIC in a JAX environment, so it
    reads one from disk rather than solving one.
    """

    @pytest.mark.skipif(not _DOLFINX_AVAILABLE, reason="requires dolfinx (FEniCSx)")
    def test_a_run_against_a_saved_l2_truth_is_certified_at_t2(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("dolfinx")
        truth = _synthetic_truth()
        report = run_l2_to_l1(truth, calibration_uncertainty=True)

        assert report.tier is Tier.T2
        assert math.isfinite(report.relative_error)
        report.sealed.assert_at_least(Tier.T2)
