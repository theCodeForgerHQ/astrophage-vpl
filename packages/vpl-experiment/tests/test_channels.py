"""What :mod:`vpl.experiment.channels` must be true for — doc 05 §3.2, doc 11 §9 item 3.

The project's honest end-to-end error is 36.5 %, diagnosed as a single optical channel
sliding along the ``n_0``-``T_e`` degeneracy. Connecting the second, already-built channel
is the designed fix, and it is only a fix if the second channel sees something the first
cannot. ``TestWhatLifAddsThatOesCannotSee`` is therefore not a nice-to-have test in this
file: it is the test that decides whether the whole exercise is worth doing, and it is
written to fail loudly if the answer is no.

``TestTheTuningRangeBlocker`` records the answer this file actually found at the RP-1
operating point, so that it is a checked property of the code rather than a discovery
somebody makes again later.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.state import PlasmaState
from vpl.core.units import magnitude_in
from vpl.experiment.channels import (
    LIF_CHANNEL,
    OES_CHANNEL,
    ChannelSet,
    build_channels,
    reconstruct_ivdf,
)
from vpl.experiment.closed_loop import (
    _argon_ion,
    _fixed_spatial_grid,
    _reference_theta,
    _to_plasma_params,
)
from vpl.inverse.fusion import JointLikelihood
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import AnalyticSheathSolver

#: Wall bias the channel tests run at, in volts.
#:
#: **Not RP-1's -250 V, and the difference is the finding this file exists to record.** A
#: 250 V sheath accelerates argon to 34.9 km/s, past the 25.8 km/s the laser's 20 GHz
#: mode-hop-free range can reach (doc 01 §5.1, doc 14 RS-03), so `LifInstrument.
#: is_informative` refuses the channel outright there and every test about what LIF adds
#: would be vacuously testing an excluded channel. -100 V is inside the reachable window
#: with margin; `TestTheTuningRangeBlocker` pins the RP-1 behaviour separately rather than
#: letting this choice quietly hide it.
_REACHABLE_BIAS_V = -100.0

_SEED = 7


def _solver() -> AnalyticSheathSolver:
    return AnalyticSheathSolver()


def _state(theta: ControlParameters) -> PlasmaState:
    registry = default_registry()
    species = _argon_ion(registry)
    solver = _solver()
    grid = _fixed_spatial_grid(
        solver, reference=_reference_theta(), species=species, registry=registry
    )
    params = _to_plasma_params(theta, species=species, registry=registry)
    return solver.solve(params, grid=grid)


def _reference_state() -> PlasmaState:
    return _state(_reference_theta())


def _operating_theta() -> ControlParameters:
    return _reference_theta().replace(V_w=_REACHABLE_BIAS_V)


@pytest.fixture
def channels() -> ChannelSet:
    return build_channels(reference_state=_reference_state(), seed=_SEED)


@pytest.fixture
def truth_state() -> PlasmaState:
    return _state(_operating_theta())


def _joint_and_truth(channels: ChannelSet, truth_state: PlasmaState) -> JointLikelihood:
    return channels.joint(channels.observe(truth_state))


class TestTheChannelSet:
    def test_the_joint_likelihood_contains_both_channels_by_name(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        joint = channels.joint(channels.observe(truth_state))

        assert set(joint.names) == {OES_CHANNEL, LIF_CHANNEL}

    def test_the_channel_names_match_what_the_instruments_stamp_on_their_output(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        # OES's identifier is private in its own package and duplicated in channels.py;
        # this is the check that keeps the duplicate honest.
        observations = channels.observe(truth_state)

        assert observations[OES_CHANNEL].instrument_id == OES_CHANNEL
        assert observations[LIF_CHANNEL].instrument_id == LIF_CHANNEL

    def test_each_channel_alone_produces_a_finite_log_likelihood(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        joint = _joint_and_truth(channels, truth_state)

        for name in (OES_CHANNEL, LIF_CHANNEL):
            dropped = LIF_CHANNEL if name == OES_CHANNEL else OES_CHANNEL
            alone = joint.without(dropped)

            detail = alone.detail(truth_state)

            assert detail.contributing == (name,)
            assert math.isfinite(detail.log_prob)

    def test_ablation_by_name_leaves_a_likelihood_that_still_scores(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        # doc 11 §9 item 6's primitive: dropping a channel must yield something that runs,
        # not something that raises, or the ablation sweep has nothing to report.
        joint = _joint_and_truth(channels, truth_state)

        without_lif = joint.without(LIF_CHANNEL)

        assert without_lif.names == (OES_CHANNEL,)
        assert math.isfinite(without_lif.log_prob(truth_state))
        assert set(joint.names) == {OES_CHANNEL, LIF_CHANNEL}, "without() must not mutate"

    def test_the_two_channels_integrate_for_genuinely_different_times(
        self, channels: ChannelSet
    ) -> None:
        # The module docstring's acquisition-window section states the assumption that
        # makes fusing these defensible; this pins the fact that they *are* different, so
        # the assumption cannot quietly become vacuous.
        oes_s = float(magnitude_in(channels.oes_window.duration, "s"))
        lif_s = float(magnitude_in(channels.lif_window.duration, "s"))

        assert lif_s > 100.0 * oes_s
        assert float(magnitude_in(channels.oes_window.start, "s")) == 0.0
        assert float(magnitude_in(channels.lif_window.start, "s")) == 0.0

    def test_a_missing_observation_is_refused_rather_than_silently_dropped(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        observations = channels.observe(truth_state)
        del observations[LIF_CHANNEL]

        # Matched on the explanation, not just on KeyError: a bare dict lookup raises
        # KeyError too, so asserting the type alone would pass whether or not the guard
        # exists. (Found by mutation: deleting the guard left this test green.)
        with pytest.raises(KeyError, match="no observation supplied for lif"):
            channels.joint(observations)


class TestTheReconstructedIvdf:
    """doc 03 §6's assumed drifting Maxwellian, checked against its own moments."""

    def test_it_carries_the_states_ion_density(self, truth_state: PlasmaState) -> None:
        reconstructed = reconstruct_ivdf(truth_state)

        assert reconstructed.ion_distribution is not None
        np.testing.assert_allclose(
            reconstructed.ion_distribution.density_per_m3(),
            truth_state.field("n_i").values,
            rtol=1e-6,
        )

    def test_the_drift_is_toward_the_wall(self, truth_state: PlasmaState) -> None:
        # doc 02 §2's sign convention: the wall is at z = 0 and an ion reaching it has
        # v_z < 0, while the u_i field a sheath solver emits is a positive speed. Getting
        # this backwards puts the LIF line on the wrong side of centre and still returns a
        # plausible scan.
        reconstructed = reconstruct_ivdf(truth_state)

        assert reconstructed.ion_distribution is not None
        np.testing.assert_allclose(
            reconstructed.ion_distribution.mean_velocity_m_per_s(),
            -truth_state.field("u_i").values,
            rtol=1e-6,
        )
        assert np.all(reconstructed.ion_distribution.particle_flux_toward_wall_per_m2_s() > 0.0)

    def test_a_state_that_already_resolves_f_i_is_returned_untouched(
        self, truth_state: PlasmaState
    ) -> None:
        # A computed distribution always beats an assumed one; L2 states carry theirs.
        once = reconstruct_ivdf(truth_state)

        assert reconstruct_ivdf(once) is once


class TestWhatLifAddsThatOesCannotSee:
    """The test that decides whether connecting LIF is worth doing at all.

    ``T_i`` is the sharpest available probe of the question. ``closed_loop``'s own module
    docstring states that ``T_i`` "has exactly zero effect on this configuration's
    prediction" because L0 runs at ``gamma_i = GAMMA_I_COLD_ION = 0``, so the OES channel
    is not merely weakly sensitive to it — it is bit-for-bit blind. LIF's line width is
    ``sqrt(T_i)`` by construction. If the LIF response is not large here, the second
    channel is carrying no information the first lacks and the whole fusion exercise is
    pointless; the assertions are written to say so.
    """

    def test_oes_is_bit_for_bit_blind_to_the_ion_temperature(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        theta = _operating_theta()
        joint = _joint_and_truth(channels, truth_state).without(LIF_CHANNEL)

        at_truth = joint.log_prob(_state(theta))
        at_four_times_t_i = joint.log_prob(_state(theta.replace(T_i=theta.T_i * 4.0)))

        assert at_four_times_t_i == at_truth

    def test_lif_responds_sharply_to_the_ion_temperature_oes_cannot_see(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        theta = _operating_theta()
        observations = channels.observe(truth_state)
        lif = channels.joint(observations).without(OES_CHANNEL)

        at_truth = lif.log_prob(_state(theta))
        hotter = lif.log_prob(_state(theta.replace(T_i=theta.T_i * 1.5)))
        colder = lif.log_prob(_state(theta.replace(T_i=theta.T_i / 1.5)))

        # A factor of 1.5 in T_i is a factor of 1.22 in line width. Both directions must
        # be worse than the truth, and by a margin that could not be noise: 100 log units
        # is 200 in chi-squared over 201 scan points. Observed here is ~7.9e3 either way.
        assert at_truth - hotter > 100.0
        assert at_truth - colder > 100.0

    def test_lif_constrains_the_degeneracy_direction_at_least_as_hard_as_oes(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        # ``T_i`` proves LIF sees *something* OES cannot, but the parameter combination the
        # project is actually stuck on is ``Gamma_i ~ n_0 sqrt(T_e)``. This walks along
        # that curve — ``n_0 -> k n_0``, ``T_e -> T_e / k^2``, which leaves the product
        # exactly fixed — and asks each channel how much it minds.
        #
        # LIF minds more, and for a reason worth stating: at the sheath edge its line
        # centre sits at the Bohm speed, which is ``sqrt(T_e / M)`` with no ``n_0`` in it
        # at all, while its amplitude carries ``n_0`` with no ``T_e``. OES forms neither
        # separately — it sees an emission rate that mixes both. Observed: LIF -5.3e3
        # against OES -2.4e3 for k = 1.05.
        theta = _operating_theta()
        observations = channels.observe(truth_state)
        joint = channels.joint(observations)
        oes = joint.without(LIF_CHANNEL)
        lif = joint.without(OES_CHANNEL)

        k = 1.05
        along_ridge = _state(theta.replace(n_0=theta.n_0 * k, T_e=theta.T_e / k**2))
        oes_penalty = oes.log_prob(_state(theta)) - oes.log_prob(along_ridge)
        lif_penalty = lif.log_prob(_state(theta)) - lif.log_prob(along_ridge)

        assert oes_penalty > 0.0
        assert lif_penalty > oes_penalty


class TestTheTuningRangeBlocker:
    """doc 01 §5.1 and doc 14 RS-03, as they actually bite at RP-1.

    Pinned rather than left in a docstring because it is the single fact that decides
    whether connecting LIF helps the run the project actually cares about.
    """

    def test_lif_declares_itself_blind_at_the_rp1_wall_bias(self, channels: ChannelSet) -> None:
        registry = default_registry()
        rp1 = _to_plasma_params(_reference_theta(), species=_argon_ion(registry), registry=registry)

        assert not channels.lif.is_informative(rp1)

    def test_lif_is_reachable_at_the_reduced_bias_these_tests_run_at(
        self, channels: ChannelSet
    ) -> None:
        registry = default_registry()
        reachable = _to_plasma_params(
            _operating_theta(), species=_argon_ion(registry), registry=registry
        )

        assert channels.lif.is_informative(reachable)

    def test_at_rp1_the_joint_likelihood_is_oes_alone_and_says_so(
        self, channels: ChannelSet
    ) -> None:
        # doc 01 IF-6: the blind channel's term is not formed, and it is *named*. A run
        # that reported "two channels fused" here would be making a false claim.
        rp1_state = _reference_state()
        joint = channels.joint(channels.observe(rp1_state))

        detail = joint.detail(rp1_state)

        assert detail.contributing == (OES_CHANNEL,)
        assert detail.excluded == (LIF_CHANNEL,)
        assert math.isfinite(detail.log_prob)
