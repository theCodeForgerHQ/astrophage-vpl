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


def _channel(name: str, value: float, *, informative: bool = True, weight: float = 1.0) -> Channel:
    return Channel(
        name=name,
        instrument=_FakeInstrument(value=value, informative=informative, name=name),  # type: ignore[arg-type]
        observation=object(),  # type: ignore[arg-type]
        window=object(),  # type: ignore[arg-type]
        weight=weight,
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


class TestPerChannelWeights:
    """The power-likelihood correction for channels that understate their uncertainty.

    Measured failure at T2: every channel claims a tighter uncertainty than it delivers, and
    because the sum treats each as an independent confirmation, fusing four compounds the
    understatement ~50-fold — the posterior gets narrower and less correct as channels are
    added. Scaling a Gaussian log-likelihood by ``w`` inflates that channel's variance by
    ``1 / w``, which is exact algebra, not a fudge (Ibrahim & Chen 2000; Boss et al. 2025).

    The load-bearing test in this class is the bit-for-bit one. Every number this project has
    already measured was produced without weights, and those numbers stay comparable only if
    an unweighted run is *identical*, not merely close.
    """

    def test_the_default_weight_leaves_the_sum_bit_for_bit_unchanged(self) -> None:
        # Not approx. Multiplying a float by exactly 1.0 is exact in IEEE 754, so this must
        # hold to the last bit — otherwise every pre-weighting measurement in the project
        # silently stops being comparable with every post-weighting one.
        unweighted = JointLikelihood((_channel("oes", -3.7), _channel("lif", -5.25)))
        explicit_one = JointLikelihood(
            (_channel("oes", -3.7, weight=1.0), _channel("lif", -5.25, weight=1.0))
        )

        assert explicit_one.log_prob(_State()) == unweighted.log_prob(_State())
        assert unweighted.log_prob(_State()) == -3.7 + -5.25

    def test_a_weight_below_one_shrinks_that_channel_s_contribution(self) -> None:
        # A log-likelihood is negative, so scaling it by w < 1 moves it *towards* zero: the
        # channel discriminates less between states, which is what a wider variance means.
        joint = JointLikelihood((_channel("oes", -3.0), _channel("lif", -8.0, weight=0.25)))

        assert joint.log_prob(_State()) == pytest.approx(-3.0 + 0.25 * -8.0)

    def test_a_weight_below_one_strictly_widens_relative_to_unweighted(self) -> None:
        # The directional guarantee Block C depends on. The gap between a good state and a
        # bad one is what sets the posterior width; down-weighting must strictly shrink that
        # gap, for every channel, at every pair of states.
        good, bad = -2.0, -20.0
        unweighted_gap = abs(
            JointLikelihood((_channel("lif", good),)).log_prob(_State())
            - JointLikelihood((_channel("lif", bad),)).log_prob(_State())
        )
        weighted_gap = abs(
            JointLikelihood((_channel("lif", good, weight=0.25),)).log_prob(_State())
            - JointLikelihood((_channel("lif", bad, weight=0.25),)).log_prob(_State())
        )

        assert weighted_gap < unweighted_gap

    def test_the_variance_inflation_identity_holds_exactly_for_a_gaussian(self) -> None:
        # The whole justification in one assertion: w * gaussian(sigma) is gaussian with
        # sigma**2 / w. If this ever fails, the weights are a fudge and the docstring lies.
        residual, sigma, weight = 1.3, 0.4, 0.25
        scaled = weight * (-(residual**2) / (2.0 * sigma**2))
        inflated = -(residual**2) / (2.0 * (sigma**2 / weight))

        assert scaled == inflated

    def test_the_applied_weights_are_reported_alongside_the_contributors(self) -> None:
        # "We down-weighted Thomson by 4x" has to be readable off the result. Inferring it
        # from the log-probability is not checking, and trusting the brief is not either.
        joint = JointLikelihood((_channel("oes", -3.0, weight=0.5), _channel("lif", -5.0)))

        detail = joint.detail(_State())

        assert detail.contributing == ("oes", "lif")
        assert detail.weights == (0.5, 1.0)

    def test_a_blind_channel_contributes_no_weight_entry(self) -> None:
        # weights is positionally aligned with contributing, so an excluded channel must not
        # leave a phantom entry that shifts every subsequent name against its weight.
        joint = JointLikelihood(
            (
                _channel("oes", -3.0, weight=0.5),
                _channel("thomson", -100.0, weight=0.1, informative=False),
                _channel("lif", -5.0, weight=0.25),
            )
        )

        detail = joint.detail(_State())

        assert detail.contributing == ("oes", "lif")
        assert detail.weights == (0.5, 0.25)
        assert detail.excluded == ("thomson",)


class TestWithWeightsIsImmutable:
    def test_applying_weights_returns_a_new_likelihood(self) -> None:
        # Same reason `without` is immutable: a weighted run must not degrade the unweighted
        # baseline it is being compared against.
        joint = JointLikelihood((_channel("oes", -4.0), _channel("lif", -8.0)))

        weighted = joint.with_weights({"lif": 0.5})

        assert weighted.log_prob(_State()) == pytest.approx(-4.0 + 0.5 * -8.0)
        assert joint.log_prob(_State()) == pytest.approx(-12.0), "baseline must be unchanged"

    def test_unmentioned_channels_keep_their_existing_weight(self) -> None:
        joint = JointLikelihood(
            (_channel("oes", -4.0, weight=0.5), _channel("lif", -8.0, weight=0.25))
        )

        weighted = joint.with_weights({"lif": 1.0})

        assert weighted.detail(_State()).weights == (0.5, 1.0)

    def test_weighting_an_unknown_channel_is_refused(self) -> None:
        # A typo'd name would apply to nothing, and the run would be identical to the
        # unweighted baseline — which reads as "the weighting made no difference", the exact
        # opposite of what happened.
        joint = JointLikelihood((_channel("oes", -3.0),))

        with pytest.raises(KeyError, match="thomson"):
            joint.with_weights({"thomson": 0.5})

    def test_an_invalid_weight_is_refused_at_application_time(self) -> None:
        joint = JointLikelihood((_channel("oes", -3.0),))

        with pytest.raises(ValueError, match="strictly positive"):
            joint.with_weights({"oes": 0.0})

    def test_an_empty_mapping_is_a_no_op_that_still_returns_a_copy(self) -> None:
        joint = JointLikelihood((_channel("oes", -3.0),))

        weighted = joint.with_weights({})

        assert weighted is not joint
        assert weighted.log_prob(_State()) == joint.log_prob(_State())


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

    def test_a_weight_of_zero_is_refused(self) -> None:
        # Zero would remove a channel by making its contribution vanish — doc 01 IF-6's
        # "formed and small", wearing the disguise of a weight. Exclusion is the detection
        # gate's job and is recorded in FusionDetail.excluded; a zero weight leaves no trace
        # at all, so "we fused four channels" would be false and uncheckable.
        with pytest.raises(ValueError, match="strictly positive"):
            JointLikelihood((_channel("oes", -3.0), _channel("lif", -5.0, weight=0.0)))

    def test_a_negative_weight_is_refused(self) -> None:
        # A negative weight rewards a channel for disagreeing with the others: the worse the
        # fit, the higher the joint log-probability.
        with pytest.raises(ValueError, match="strictly positive"):
            JointLikelihood((_channel("oes", -3.0, weight=-1.0),))

    def test_a_non_finite_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-finite weight"):
            JointLikelihood((_channel("oes", -3.0, weight=float("inf")),))

    def test_a_non_finite_channel_contribution_is_refused(self) -> None:
        # A NaN from one channel silently poisons the sum: the optimiser sees NaN
        # everywhere, every step looks equally bad, and it stops and reports a result.
        joint = JointLikelihood((_channel("oes", -3.0), _channel("lif", float("nan"))))

        with pytest.raises(ValueError, match="lif"):
            joint.log_prob(_State())
