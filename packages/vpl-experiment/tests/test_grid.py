"""What :mod:`vpl.experiment.grid` must be true for — doc 05 §7.2, doc 11 §9 items 4-6.

The module under test exists because three earlier drivers score differently and their
numbers cannot be compared. Its whole value proposition is **one driver, one likelihood,
one interval convention, every cell** — which means the properties worth pinning here are
not physics results (those belong to the one slow end-to-end test at the bottom) but the
structural guarantees that make the grid a grid rather than ten unrelated scripts:

* :attr:`~vpl.experiment.grid.Cell.tier` is *delegated* to
  :func:`~vpl.validation.sealed.tier_of_configuration`, never hand-labelled — the module
  docstring calls this "load bearing" and states plainly why: it is the reason
  :data:`~vpl.experiment.grid.STANDARD_CELLS` has no mismatched-model T0 row.
* :data:`~vpl.experiment.grid.STANDARD_CELLS` is what its own comment claims: every cell
  reachable, the mismatched pairs true pairs, every label unique.
* :func:`~vpl.experiment.grid.cell_label` is total and collision-free across the axes.
* :func:`~vpl.experiment.grid.run_cell` refuses what it cannot afford — an L2 inversion, a
  missing or mismatched L2 truth artefact — loudly, with a named exception, and cheaply.
* :func:`~vpl.experiment.grid.run_grid` records a failure in the row a cell would have
  produced rather than stopping or dropping it.

Almost everything above is cheap: a `Cell` is five booleans-and-enums and most of the
guards this file checks fire before any solve happens. The exception is
``TestRunCellEndToEndAtT0``, the one test allowed to run a full MAP search (~28 s on the
cheapest cell), marked ``slow`` and kept to a single case for exactly that reason.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import pytest

from vpl.core.params import default_registry
from vpl.core.state import PlasmaState, ScalarField
from vpl.experiment.channels import CHANNEL_NAMES, reconstruct_ivdf
from vpl.experiment.closed_loop import (
    _argon_ion,
    _draw_true_theta,
    _reference_theta,
    _to_plasma_params,
)
from vpl.experiment.grid import (
    STANDARD_CELLS,
    Cell,
    EedfShape,
    Fidelity,
    InfeasibleCellError,
    TruthArtefactRequiredError,
    cell_label,
    run_cell,
    run_grid,
)
from vpl.experiment.l2_truth import L2Truth, observation_grid, save_l2_truth
from vpl.physics.analytic.sheath import AnalyticSheathSolver
from vpl.validation.sealed import Tier, TierMismatchError, tier_of_configuration

#: Stand-in scalars for the synthetic L2 artefact built below. Never asserted against
#: anything physical — see ``_synthetic_l2_truth``'s docstring for why a physically
#: meaningless truth is the right fixture for the two tests that use it.
_STUB_GAMMA_E_W_PER_M2 = 9331.69
_STUB_MEAN_ENERGY_EV = 250.0
_STUB_THICKNESS_M = 1.0e-3
_STUB_ENERGY_DRIFT = 1.0e-6
_STUB_WALL_CLOCK_S = 3600.0


def _synthetic_l2_truth(seed: int) -> L2Truth:
    """An ``L2Truth`` whose state is L0's, relabelled — the same stand-in
    ``test_l2_truth.py``'s own ``_synthetic_truth`` uses, duplicated here rather than
    imported because the tests directory is not a package (see this package's
    ``conftest.py``).

    The two tests that use this need an artefact whose *seed round-trips*, not a
    physically meaningful one: a real L2 solve needs JAX and costs an hour, which is
    unaffordable inside a suite that must stay cheap, and neither test reads the
    recovered numbers. This is why option (a) from the task brief — build a minimal
    artefact rather than skip when one is not lying around — is the honest choice here:
    a ``pytest.skip`` would silently stop checking the seed-mismatch guard on any machine
    that has not happened to run a real L2 solve, which is every machine this suite
    normally runs on.
    """
    registry = default_registry()
    grid = observation_grid(registry)
    species = _argon_ion(registry)
    theta = _draw_true_theta(seed=seed, reference=_reference_theta())
    params = _to_plasma_params(theta, species=species, registry=registry)
    state = reconstruct_ivdf(AnalyticSheathSolver().solve(params, grid=grid), registry=registry)
    fields = {
        name: ScalarField(name=name, values=field.values, units=field.units, grid=grid, time=None)
        for name, field in state.fields.items()
    }
    return L2Truth(
        seed=seed,
        state=PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=fields,
            # An L2 state must carry a distribution; PlasmaState enforces it.
            ion_distribution=state.ion_distribution,
            fidelity=Fidelity.L2,
        ),
        gamma_e_w_per_m2=_STUB_GAMMA_E_W_PER_M2,
        mean_impact_energy_ev=_STUB_MEAN_ENERGY_EV,
        sheath_thickness_m=_STUB_THICKNESS_M,
        energy_drift=_STUB_ENERGY_DRIFT,
        wall_clock_s=_STUB_WALL_CLOCK_S,
    )


def _base_cell() -> Cell:
    """A cell with every axis at its "off" setting, for the one-flip-at-a-time tests."""
    return Cell(
        truth=Fidelity.L0,
        inversion=Fidelity.L0,
        noise=False,
        imperfect_calibration=False,
        calibration_uncertainty=False,
        truth_eedf=EedfShape.MAXWELLIAN,
    )


# ── tier is delegated, not asserted ─────────────────────────────────────────────


class TestTierIsDelegatedNotAsserted:
    """``Cell.tier`` must be ``tier_of_configuration``'s verdict, not a stored label.

    The module docstring names this property "load bearing": it is the entire reason
    ``STANDARD_CELLS`` has no mismatched-model T0 row. If ``Cell.tier`` were ever changed
    to compute a label itself (or cache one incorrectly) instead of delegating, both
    tests below would be able to catch it — the first because it checks every standard
    cell against the same function the property is supposed to call, the second because
    it checks the one case the property must refuse rather than answer.
    """

    @pytest.mark.parametrize("cell", STANDARD_CELLS, ids=cell_label)
    def test_every_standard_cell_matches_tier_of_configuration(self, cell: Cell) -> None:
        assert cell.tier == tier_of_configuration(
            same_model=cell.same_model,
            noise=cell.noise,
            imperfect_calibration=cell.imperfect_calibration,
        )

    def test_a_mismatched_model_cell_without_noise_raises_rather_than_labels_t0(self) -> None:
        # doc 05 §7.1's refusal, read off Cell.tier directly rather than off
        # tier_of_configuration: this is the property that must surface it, since a grid
        # row is built by reading .tier, not by calling the sealed module itself.
        cell = dataclasses.replace(_base_cell(), truth=Fidelity.L1, inversion=Fidelity.L0)

        with pytest.raises(TierMismatchError):
            _ = cell.tier


# ── STANDARD_CELLS is what its own comment claims ───────────────────────────────


class TestStandardCellsShape:
    def test_every_cell_certifies_a_tier_without_raising(self) -> None:
        for cell in STANDARD_CELLS:
            _ = cell.tier

    def test_same_model_cells_are_t0_or_t1(self) -> None:
        for cell in STANDARD_CELLS:
            if cell.same_model:
                assert cell.tier in (Tier.T0, Tier.T1)

    def test_mismatched_cells_are_t2(self) -> None:
        for cell in STANDARD_CELLS:
            if not cell.same_model:
                assert cell.tier is Tier.T2

    def test_each_mismatched_pair_appears_exactly_twice_differing_only_in_calibration(
        self,
    ) -> None:
        # The comment above STANDARD_CELLS states the pairing explicitly: each of the
        # three mismatched (truth, inversion) combinations appears once with
        # calibration_uncertainty off and once with it on, and nothing else differs
        # between the two. A pairing that silently became a triple, or drifted apart on
        # some other axis, would still "look like a grid" without this check.
        mismatched = [cell for cell in STANDARD_CELLS if not cell.same_model]
        by_configuration: dict[tuple[object, ...], list[Cell]] = {}
        for cell in mismatched:
            configuration: tuple[object, ...] = (
                cell.truth,
                cell.inversion,
                cell.noise,
                cell.imperfect_calibration,
                cell.truth_eedf,
            )
            by_configuration.setdefault(configuration, []).append(cell)

        for key, cells in by_configuration.items():
            assert len(cells) == 2, f"{key} appears {len(cells)} times in STANDARD_CELLS, not 2"
            assert {c.calibration_uncertainty for c in cells} == {True, False}

    def test_cell_label_has_no_collisions(self) -> None:
        # A collision would silently merge two rows in the results table — the exact
        # failure mode cell_label's own docstring says it exists to prevent.
        labels = [cell_label(cell) for cell in STANDARD_CELLS]
        assert len(set(labels)) == len(labels)


# ── cell_label is total and collision-free across the axes ─────────────────────


class TestCellLabelIsTotalAndStable:
    def test_the_label_is_stable_across_separately_constructed_equal_cells(self) -> None:
        # Two independently built Cells with the same field values must produce the same
        # label — the property a results table's join key depends on.
        assert cell_label(_base_cell()) == cell_label(_base_cell())

    @pytest.mark.parametrize(
        ("axis", "replacement"),
        [
            ("truth", {"truth": Fidelity.L1}),
            ("inversion", {"inversion": Fidelity.L1}),
            ("noise", {"noise": True}),
            ("imperfect_calibration", {"imperfect_calibration": True}),
            ("calibration_uncertainty", {"calibration_uncertainty": True}),
            ("truth_eedf", {"truth_eedf": EedfShape.DRUYVESTEYN}),
        ],
        ids=lambda value: value if isinstance(value, str) else "",
    )
    def test_flipping_one_axis_changes_the_label(
        self, axis: str, replacement: dict[str, Any]
    ) -> None:
        base = _base_cell()
        flipped = dataclasses.replace(base, **replacement)

        assert cell_label(flipped) != cell_label(base), (
            f"flipping only {axis!r} left the label unchanged; two distinct cells on this "
            "axis would report through the same row"
        )


# ── infeasible cells refuse loudly and cheaply ──────────────────────────────────


class TestInfeasibleCellsRefuseCheaply:
    """``InfeasibleCellError`` must arrive before the MAP search starts, not after it
    fails partway through. L2-as-inversion is unaffordable rather than unimplemented —
    one solve is 7 122 s measured and an inversion asks for thousands — and the guard
    that matters is that ``run_cell`` says so fast, since a slow refusal would defeat the
    entire reason this is a distinct exception type from "the search diverged".
    """

    def test_an_l2_inversion_cell_is_refused_and_names_the_cost(self) -> None:
        cell = dataclasses.replace(
            _base_cell(),
            inversion=Fidelity.L2,
            noise=True,
            imperfect_calibration=True,
        )

        started = time.perf_counter()
        with pytest.raises(InfeasibleCellError, match=r"7 122 s"):
            run_cell(cell, seed=0)
        elapsed = time.perf_counter() - started

        # A full run_cell on the cheapest cell measures ~28 s (a MAP search's thousands
        # of forward solves). This refusal does one L0 truth solve and one channel
        # forward pass before raising, with no search at all, so it should be well under
        # that — a generous bound rather than a tight one, to keep this from flaking on a
        # loaded machine.
        assert elapsed < 10.0, (
            f"refusing an infeasible cell took {elapsed:.1f} s; InfeasibleCellError is "
            "supposed to be the cheap way to discover this, not the slow one"
        )


# ── L2 truth artefact discipline ────────────────────────────────────────────────


class TestL2TruthArtefactDiscipline:
    """``run_cell`` must refuse an L2-truth cell that has no artefact, and one whose
    artefact was recorded at a different seed — doc 05 §7.1's arrangement for a truth
    that has to travel through a file (see the module docstring's final section).
    """

    def _l2_cell(self) -> Cell:
        return next(cell for cell in STANDARD_CELLS if cell.truth is Fidelity.L2)

    def test_a_missing_artefact_path_is_refused(self) -> None:
        with pytest.raises(TruthArtefactRequiredError, match="l2_truth_path"):
            run_cell(self._l2_cell(), seed=0, l2_truth_path=None)

    def test_an_artefact_recorded_at_a_different_seed_is_refused_and_names_both_seeds(
        self, tmp_path: Path
    ) -> None:
        artefact = save_l2_truth(_synthetic_l2_truth(seed=0), tmp_path / "truth.npz")

        with pytest.raises(TruthArtefactRequiredError, match="seed 0") as excinfo:
            run_cell(self._l2_cell(), seed=1, l2_truth_path=artefact)

        # Not pedantry — see TruthArtefactRequiredError's own docstring: scoring one
        # draw's truth against another draw's instrument realisations produces a number
        # that means nothing and looks entirely ordinary, so the message has to name
        # *both* seeds for a reader to be able to tell the mismatch happened at all.
        assert "seed 1" in str(excinfo.value)


# ── run_grid records failures in place of rows ──────────────────────────────────


class TestRunGridRecordsFailuresInPlace:
    """``run_grid`` must neither stop at the first infeasible cell nor drop it silently
    — see the function's own docstring. Both cells here are refused for free (one at the
    ``.tier`` property, before any solve; the other at the L2-truth guard, also before
    any solve), so the whole test stays cheap while still exercising two different
    exception types landing in the slot the corresponding row would have used.
    """

    def test_two_cheap_failures_are_recorded_rather_than_dropped(self) -> None:
        mismatched_without_noise = dataclasses.replace(
            _base_cell(), truth=Fidelity.L1, inversion=Fidelity.L0, imperfect_calibration=True
        )
        l2_without_an_artefact = next(cell for cell in STANDARD_CELLS if cell.truth is Fidelity.L2)

        rows = run_grid((mismatched_without_noise, l2_without_an_artefact), seed=0, verbose=False)

        assert len(rows) == 2
        first_cell, first_outcome = rows[0]
        second_cell, second_outcome = rows[1]
        assert first_cell is mismatched_without_noise
        assert isinstance(first_outcome, TierMismatchError)
        assert second_cell is l2_without_an_artefact
        assert isinstance(second_outcome, TruthArtefactRequiredError)


# ── the one slow end-to-end test ────────────────────────────────────────────────


class TestRunCellEndToEndAtT0:
    """The one test in this file allowed to run a full MAP search — doc 05 §7.2's
    blocking-bug detector, exercised through this module's own scoring path rather than
    ``closed_loop.run_t0``'s.

    The relative-error threshold below is intentionally loose, and the reason is the
    module docstring's own "one deliberate departure" section: ``closed_loop.run_t0``
    scores T0 with a continuum residual specifically to step around ``np.rint``'s
    quantisation floor in ``OesInstrument.likelihood``, and this module does *not*
    reproduce that departure — every cell here, T0 included, is scored by the four
    channels' own likelihoods, so this T0 inherits the floor ``closed_loop``'s T0 was
    built to avoid. Measured at commit 0a0c52d the relative error was 1.183127e-04; the
    assertion below pins the *property* — orders of magnitude below any physical effect
    doc 05 §7.2 cares about — rather than that exact float, which would make the test
    brittle against unrelated numerical noise elsewhere in the chain.
    """

    @pytest.mark.slow
    def test_the_t0_cell_recovers_gamma_e_with_every_channel_contributing(self) -> None:
        cell = STANDARD_CELLS[0]

        report = run_cell(cell, seed=0)

        assert report.tier is Tier.T0
        assert set(report.contributing) == set(CHANNEL_NAMES)
        assert report.excluded == ()
        # Loose on purpose — see the class docstring for the quantisation floor this
        # threshold sits well above.
        assert report.relative_error < 1e-3
        assert report.truth_within_interval is True
        assert report.cell is cell
        assert report.label == cell_label(cell)
