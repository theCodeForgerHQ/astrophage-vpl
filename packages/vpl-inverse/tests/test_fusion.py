"""Multi-channel fusion — doc 05 §3.2, doc 01 IF-6, doc 11 §9 items 3 and 6.

## Why this module exists

The project's honest end-to-end error is 36.5 %, and the diagnosed cause is a degeneracy:
plasma density and electron temperature enter the physics multiplied together, so a single
optical channel measures the combination and cannot separate the ingredients. Extra channels
that constrain those quantities through *different physics* are the designed fix — doc 11 §9
item 3 asks for two channels precisely because "two channels demonstrate fusion".

This is the module that combines them. It is also the machinery doc 11 §9 item 6 needs:
"drop each channel, show the CI inflate" is only expressible if dropping a channel is a
first-class operation rather than a code edit.

## The three ways fusion goes wrong quietly

1. **Silently ingesting a blind channel.** doc 01 IF-6 is emphatic that below its detection
   floor a channel must be modelled as *absent*, not as a weak measurement, because "an
   inversion that quietly ingests noise as data will produce confident nonsense at low
   density". doc 02 §3.3 has regimes where this genuinely happens — regime F is
   interferometry-blind, regime G is Thomson-blind. Summing a blind channel's likelihood
   adds noise-shaped information and tightens the posterior around nothing.

2. **Claiming more channels than contributed.** If two of four channels were blind, "we
   fused four channels" is false, and the excluded ones vanish without trace unless the
   result carries them. So exclusions are reported, not merely handled.

3. **Double counting.** Two channels sharing a calibration standard, or a laser, are not
   independent, and adding their log-likelihoods asserts they are. The sum is only valid
   under independence, and this module states that assumption rather than burying it.
"""

from __future__ import annotations

import pytest

from vpl.inverse.fusion import (
    BlindChannelError,
    Channel,
    JointLikelihood,
)


class _FakeInstrument:
    """Minimal stand-in with the three methods fusion actually calls."""

    def __init__(self, *, value: float, informative: bool = True, name: str = "fake") -> None:
        self.value = value
        self.informative = informative
        self.name = name
        self.forward_calls = 0

    def forward(self, state: object, w: object) -> object:
        del state, w
        self.forward_calls += 1
        return object()

    def likelihood(self, obs: object, pred: object) -> float:
        del obs, pred
        return self.value

    def is_informative(self, state_guess: object) -> bool:
        del state_guess
        return self.informative


def _channel(name: str, value: float, *, informative: bool = True) -> Channel:
    return Channel(
        name=name,
        instrument=_FakeInstrument(value=value, informative=informative, name=name),  # type: ignore[arg-type]
        observation=object(),  # type: ignore[arg-type]
        window=object(),  # type: ignore[arg-type]
    )


class _State:
    params = object()


class TestIndependentChannelsAdd:
    def test_the_joint_log_probability_is_the_sum(self) -> None:
        # doc 05 §3.2's sum. Valid only under independence, which this module asserts and
        # does not verify — see the module docstring.
        joint = JointLikelihood((_channel("oes", -3.0), _channel("lif", -5.0)))

        assert joint.log_prob(_State()) == pytest.approx(-8.0)

    def test_one_channel_alone_equals_that_channel(self) -> None:
        # Guards the commonest arithmetic slip in a fold: an extra term, or a prior added
        # once per channel instead of once.
        joint = JointLikelihood((_channel("oes", -3.0),))

        assert joint.log_prob(_State()) == pytest.approx(-3.0)

    def test_every_contributing_channel_is_named(self) -> None:
        joint = JointLikelihood((_channel("oes", -1.0), _channel("lif", -2.0)))

        detail = joint.detail(_State())

        assert detail.contributing == ("oes", "lif")
        assert detail.excluded == ()


class TestABlindChannelIsExcludedNotSummed:
    def test_a_channel_below_its_detection_floor_does_not_contribute(self) -> None:
        # doc 01 IF-6: "a False here means the term is not formed, not that it is formed and
        # small". The blind channel here would have contributed -100 if summed, so the
        # difference is unmissable.
        joint = JointLikelihood(
            (_channel("oes", -3.0), _channel("thomson", -100.0, informative=False))
        )

        assert joint.log_prob(_State()) == pytest.approx(-3.0)

    def test_the_blind_channel_is_reported_rather_than_vanishing(self) -> None:
        # "We fused three channels" is false if one was blind. The exclusion has to survive
        # into the result or the claim cannot be checked afterwards.
        joint = JointLikelihood(
            (_channel("oes", -3.0), _channel("thomson", -100.0, informative=False))
        )

        detail = joint.detail(_State())

        assert detail.contributing == ("oes",)
        assert detail.excluded == ("thomson",)

    def test_a_blind_channel_is_never_asked_for_a_prediction(self) -> None:
        # Not just an optimisation. doc 02 §7 costs a Thomson forward model at ~700 s of
        # accumulation to reach 3 %; forming a prediction for a channel that cannot see
        # anything is the most expensive possible no-op.
        blind = _FakeInstrument(value=-100.0, informative=False, name="thomson")
        joint = JointLikelihood(
            (
                _channel("oes", -3.0),
                Channel(name="thomson", instrument=blind, observation=object(), window=object()),  # type: ignore[arg-type]
            )
        )

        joint.log_prob(_State())

        assert blind.forward_calls == 0

    def test_all_channels_blind_is_an_error_not_a_zero(self) -> None:
        # A log-probability of 0.0 from no channels is indistinguishable from a perfect fit
        # and would make every parameter equally probable — the optimiser would wander and
        # report success. Refusing is the only safe answer.
        joint = JointLikelihood(
            (
                _channel("oes", -3.0, informative=False),
                _channel("lif", -5.0, informative=False),
            )
        )

        with pytest.raises(BlindChannelError, match="no channel"):
            joint.log_prob(_State())


class TestAblationIsFirstClass:
    """doc 11 §9 item 6: "drop each channel, show the CI inflate"."""

    def test_a_channel_can_be_dropped_by_name(self) -> None:
        joint = JointLikelihood((_channel("oes", -3.0), _channel("lif", -5.0)))

        ablated = joint.without("lif")

        assert ablated.log_prob(_State()) == pytest.approx(-3.0)
        assert joint.log_prob(_State()) == pytest.approx(-8.0), "the original must be unchanged"

    def test_dropping_an_unknown_channel_is_refused(self) -> None:
        # A typo'd name would silently ablate nothing, and the ablation figure would show
        # two identical bars that look like "this channel carries no information".
        joint = JointLikelihood((_channel("oes", -3.0),))

        with pytest.raises(KeyError, match="thomson"):
            joint.without("thomson")

    def test_the_channel_names_are_exposed_for_sweeping(self) -> None:
        joint = JointLikelihood((_channel("oes", -1.0), _channel("lif", -2.0)))

        assert joint.names == ("oes", "lif")

    def test_dropping_the_only_channel_is_refused(self) -> None:
        joint = JointLikelihood((_channel("oes", -3.0),))

        with pytest.raises(ValueError, match="at least one"):
            joint.without("oes")


class TestConstruction:
    def test_duplicate_channel_names_are_refused(self) -> None:
        # Two channels called "oes" would make `without("oes")` ambiguous and would let the
        # same measurement be counted twice, which is the double-counting failure the module
        # docstring warns about, wearing an obvious disguise.
        with pytest.raises(ValueError, match="duplicate"):
            JointLikelihood((_channel("oes", -1.0), _channel("oes", -2.0)))

    def test_an_empty_channel_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            JointLikelihood(())

    def test_a_non_finite_channel_contribution_is_refused(self) -> None:
        # A NaN from one channel silently poisons the sum: the optimiser sees NaN
        # everywhere, every step looks equally bad, and it stops and reports a result.
        joint = JointLikelihood((_channel("oes", -3.0), _channel("lif", float("nan"))))

        with pytest.raises(ValueError, match="lif"):
            joint.log_prob(_State())
