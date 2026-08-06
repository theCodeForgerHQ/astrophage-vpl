"""The channel ablation experiment — doc 11 §9 item 6, WBS 4.9.

``vpl.experiment.channels``'s own test file, ``test_channels.py``, already establishes that
LIF is reachable at -100 V and blind at RP-1's -250 V; this file does not re-derive either
fact, it checks that :mod:`vpl.experiment.ablation` reports them correctly and that the
sweep it runs cannot silently mislabel or corrupt a configuration.

A single MAP recovery through the LIF channel costs 10-25 s here (measured), so the tests
that actually run a recovery are grouped behind two module-scoped, ``slow``-marked fixtures
— one :func:`~vpl.experiment.ablation.run_ablation` call and one
:func:`~vpl.experiment.ablation.run_rp1_baseline` call — computed once and shared by every
assertion that only reads their result. Tests that do not need a recovery at all (the
``_prepare``-level grid/bias checks, and ``render_table``) are ordinary speed and unmarked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.experiment.ablation import (
    BASELINE_LABEL,
    RP1_LABEL,
    WITHOUT_LIF_LABEL,
    WITHOUT_OES_LABEL,
    AblationResult,
    _prepare,
    render_table,
    run_ablation,
    run_rp1_baseline,
)
from vpl.experiment.channels import LIF_CHANNEL, OES_CHANNEL
from vpl.inverse.parameters import ControlParameters

_SEED = 7


@pytest.fixture(scope="module")
def sweep() -> tuple[AblationResult, ...]:
    return run_ablation(seed=_SEED)


@pytest.fixture(scope="module")
def rp1() -> AblationResult:
    return run_rp1_baseline(seed=_SEED)


# ── fast: what _prepare builds, before any recovery runs ────────────────────────────


class TestPrepare:
    def test_the_joint_likelihood_has_both_channel_names(self) -> None:
        prepared = _prepare(seed=_SEED, wall_bias_v=-100.0, registry=None)

        assert set(prepared.joint.names) == {OES_CHANNEL, LIF_CHANNEL}

    def test_without_returns_a_new_object_and_leaves_the_original_untouched(self) -> None:
        # doc 11 §9 item 6's primitive, and the first risk a peer review flagged: an
        # ablation sweep that quietly degraded the baseline it is compared against would
        # make every row in the table look plausible while comparing nothing real.
        prepared = _prepare(seed=_SEED, wall_bias_v=-100.0, registry=None)
        baseline_names_before = prepared.joint.names

        without_lif = prepared.joint.without(LIF_CHANNEL)
        without_oes = prepared.joint.without(OES_CHANNEL)

        assert prepared.joint.names == baseline_names_before == (OES_CHANNEL, LIF_CHANNEL)
        assert without_lif.names == (OES_CHANNEL,)
        assert without_oes.names == (LIF_CHANNEL,)

    def test_the_grid_is_sized_from_rp1_regardless_of_the_operating_bias(self) -> None:
        # closed_loop's own module docstring: a spatial grid sized from anything other than
        # the fixed RP-1 reference is the theta-dependent-grid inverse crime wearing a
        # different hat. The operating bias is a chosen experimental setting, not the
        # unknown being estimated, but this pins that changing it does not move the grid.
        at_reachable_bias = _prepare(seed=_SEED, wall_bias_v=-100.0, registry=None)
        at_rp1_bias = _prepare(seed=_SEED, wall_bias_v=None, registry=None)

        np.testing.assert_array_equal(at_reachable_bias.grid.z_m, at_rp1_bias.grid.z_m)

    def test_wall_bias_v_is_read_off_the_operating_reference_not_hardcoded(self) -> None:
        # RP-1's bias comes from the registry via ControlParameters.reference(), not from a
        # literal duplicated in this module — this is the check that keeps the two in sync.
        at_rp1_bias = _prepare(seed=_SEED, wall_bias_v=None, registry=None)

        assert at_rp1_bias.wall_bias_v == ControlParameters.reference().V_w

    def test_a_different_operating_bias_is_actually_applied(self) -> None:
        at_reachable_bias = _prepare(seed=_SEED, wall_bias_v=-100.0, registry=None)

        assert at_reachable_bias.wall_bias_v == -100.0
        assert at_reachable_bias.wall_bias_v != ControlParameters.reference().V_w


# ── slow: the reachable-bias sweep doc 11 §9 item 6 asks for ───────────────────────


@pytest.mark.slow
class TestRunAblation:
    def test_the_baseline_is_reported_first_and_the_ablations_follow_in_a_fixed_order(
        self, sweep: tuple[AblationResult, ...]
    ) -> None:
        assert [result.label for result in sweep] == [
            BASELINE_LABEL,
            WITHOUT_LIF_LABEL,
            WITHOUT_OES_LABEL,
        ]

    def test_the_baseline_configuration_is_scored_by_both_channels(
        self, sweep: tuple[AblationResult, ...]
    ) -> None:
        baseline = sweep[0]

        assert baseline.contributing == (OES_CHANNEL, LIF_CHANNEL)
        assert baseline.excluded == ()

    def test_dropping_lif_leaves_only_oes_contributing(
        self, sweep: tuple[AblationResult, ...]
    ) -> None:
        without_lif = sweep[1]

        assert without_lif.contributing == (OES_CHANNEL,)
        assert LIF_CHANNEL not in without_lif.contributing

    def test_dropping_oes_leaves_only_lif_contributing(
        self, sweep: tuple[AblationResult, ...]
    ) -> None:
        without_oes = sweep[2]

        assert without_oes.contributing == (LIF_CHANNEL,)
        assert OES_CHANNEL not in without_oes.contributing

    def test_every_configuration_is_scored_against_the_same_sealed_truth(
        self, sweep: tuple[AblationResult, ...]
    ) -> None:
        # Only the likelihood changes between rows; a table where the truth itself drifted
        # between rows would not be an ablation of one recovery, it would be three
        # unrelated ones that happen to share a table.
        truths = {result.gamma_e_true_w_per_m2 for result in sweep}

        assert len(truths) == 1

    def test_every_configuration_reports_a_finite_nonnegative_relative_error(
        self, sweep: tuple[AblationResult, ...]
    ) -> None:
        for result in sweep:
            assert math.isfinite(result.relative_error)
            assert result.relative_error >= 0.0

    def test_every_configuration_reports_an_interval_or_states_its_absence_explicitly(
        self, sweep: tuple[AblationResult, ...]
    ) -> None:
        # doc 05 §6 / ADR-012: a non-positive-definite Hessian is a result (the combination
        # is not identified), not a crash, and `_gamma_e_interval` returns (None, None) for
        # it rather than a manufactured number. Both fields must agree about the absence.
        for result in sweep:
            has_interval = result.interval_w_per_m2 is not None
            assert has_interval == (result.interval_width_w_per_m2 is not None)
            assert has_interval == (result.truth_within_interval is not None)
            if has_interval:
                assert result.interval_width_w_per_m2 >= 0.0


# ── slow: the RP-1 datapoint, run separately from the reachable-bias sweep ──────────


@pytest.mark.slow
class TestRp1Baseline:
    def test_lif_is_named_excluded_despite_being_requested(self, rp1: AblationResult) -> None:
        # doc 01 IF-6, as it actually bites at the bias the project cares about: a
        # configuration nominally containing two channels but running on one must say so.
        assert rp1.contributing == (OES_CHANNEL,)
        assert rp1.excluded == (LIF_CHANNEL,)

    def test_the_label_names_it_as_the_rp1_configuration(self, rp1: AblationResult) -> None:
        assert rp1.label == RP1_LABEL

    def test_it_runs_at_rp1s_own_bias_not_the_reachable_sweeps(self, rp1: AblationResult) -> None:
        assert rp1.wall_bias_v == ControlParameters.reference().V_w


# ── fast: the text table, independent of any physics ────────────────────────────────


def _fabricated(
    *, label: str, contributing: tuple[str, ...], excluded: tuple[str, ...]
) -> AblationResult:
    return AblationResult(
        label=label,
        contributing=contributing,
        excluded=excluded,
        gamma_e_true_w_per_m2=1000.0,
        gamma_e_estimate_w_per_m2=990.0,
        relative_error=0.01,
        interval_w_per_m2=(980.0, 1000.0),
        interval_width_w_per_m2=20.0,
        truth_within_interval=True,
        wall_bias_v=-100.0,
        seed=_SEED,
    )


class TestRenderTable:
    def test_every_labels_row_appears_in_the_rendered_table(self) -> None:
        results = (
            _fabricated(label=BASELINE_LABEL, contributing=(OES_CHANNEL, LIF_CHANNEL), excluded=()),
            _fabricated(label=WITHOUT_LIF_LABEL, contributing=(OES_CHANNEL,), excluded=()),
        )

        table = render_table(results)

        assert BASELINE_LABEL in table
        assert WITHOUT_LIF_LABEL in table

    def test_an_absent_interval_is_rendered_as_not_available_rather_than_none_or_a_crash(
        self,
    ) -> None:
        result = AblationResult(
            label=RP1_LABEL,
            contributing=(OES_CHANNEL,),
            excluded=(LIF_CHANNEL,),
            gamma_e_true_w_per_m2=1000.0,
            gamma_e_estimate_w_per_m2=1000.0,
            relative_error=0.0,
            interval_w_per_m2=None,
            interval_width_w_per_m2=None,
            truth_within_interval=None,
            wall_bias_v=-250.0,
            seed=_SEED,
        )

        table = render_table((result,))

        assert "n/a" in table
        assert "None" not in table
