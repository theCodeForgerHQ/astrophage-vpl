"""Multi-channel fusion — doc 05 §3.2, doc 01 IF-6, doc 11 §9 items 3 and 6.

doc 05 §3.2 writes the joint likelihood as a sum over channels. This module is that sum,
and the three things that make it more than a ``sum()``.

## Why it matters here specifically

The project's honest end-to-end error is 36.5 %, and the cause is diagnosed rather than
suspected: plasma density and electron temperature enter the physics multiplied together
(``Gamma_i ~ n_0 sqrt(T_e)``), so a single optical channel constrains the product and slides
freely along the degeneracy. A second channel measuring the same plasma through *different*
physics breaks it. doc 11 §9 item 3 asks for two channels for exactly this reason, and the
project currently runs one while a fully built LIF channel sits unused.

This module is also what doc 11 §9 item 6 needs. "Drop each channel, show the CI inflate" is
only expressible as an experiment if dropping a channel is a first-class operation — hence
:meth:`JointLikelihood.without`, which returns a new object rather than mutating this one, so
an ablation sweep cannot corrupt the baseline it is being compared against.

## The independence assumption, stated rather than buried

Adding log-likelihoods asserts the channels are statistically independent given the plasma
state. That is not automatically true here. Two channels sharing a radiometric calibration
standard share its error; two sharing a laser share its pointing and power drift; doc 02 §4.3
notes stray light coupling between channels is "among the most damaging and least modelled
systematics in multi-diagnostic" work.

This module **assumes independence and does not verify it**, because verifying it needs a
joint calibration model that does not exist yet. The consequence if it is false is
over-confidence — shared error counted as two independent confirmations — which is precisely
the failure mode the project is already fighting at T2. Recording it here so it is a known
limitation rather than a discovered one.

## Blind channels are excluded, not down-weighted

doc 01 IF-6, quoted in the :class:`~vpl.core.protocols.instrument.Instrument` protocol
itself: "a ``False`` here means the term is not formed, not that it is formed and small",
because "an inversion that quietly ingests noise as data will produce confident nonsense at
low density". This is not hypothetical — doc 02 §3.3's regime F is interferometry-blind and
regime G is Thomson-blind, both by construction.

So a channel that reports itself blind is dropped from the sum entirely, is never even asked
for a prediction (which for Thomson would be an expensive no-op — doc 02 §7 costs a 3 %
measurement at ~700 s of accumulation), and **is named in the result**. That last part is the
one most easily skipped: if two of four channels were blind, "we fused four channels" is a
false claim, and without the record nobody can tell afterwards.

## Per-channel weights, and why they are not a fudge

The measured failure mode at T2 is that every channel *understates its own uncertainty*, and
because the sum above treats each as an independent confirmation, fusion compounds the
understatement — observed at roughly 50-fold. The symptom is a posterior that gets narrower
and less correct as channels are added: adding all four is worse than either strong channel
alone, and the credible interval covers the truth 0 times in 5.

:attr:`Channel.weight` is the correction. Scaling a Gaussian log-likelihood by ``w`` gives
exactly a Gaussian with its variance inflated by ``1 / w``, so setting

    ``w = 1 / (understatement factor)**2``

restores the width that channel *should* have claimed. For a Gaussian this is exact algebra
rather than an approximation — ``w * (-(d - mu)**2 / (2 * sigma**2))`` is identically
``-(d - mu)**2 / (2 * (sigma**2 / w))`` — and it is a named, peer-reviewed technique: the
**power likelihood** (equivalently power prior, likelihood tempering, fractional posterior).
Ibrahim & Chen 2000, *Statistical Science* **15** 46 introduced the power prior; Boss et al.
2025, *Biometrics* **81** ujaf008 re-derive precisely this ``1 / w`` variance relationship
when combining sources of differing trustworthiness.

Its virtue here is that it is measurable rather than tuned: the understatement factor comes
from an empirical audit (fix a plasma, draw N noisy measurements, invert each with that
channel alone, compare actual scatter against claimed uncertainty), not from fitting the
answer.

**What this is not.** Integrated Data Analysis practice in fusion — Fischer et al. 2003,
*PPCF* **45** 1095, and the Minerva framework of Svensson & Werner — treats channel
inconsistency by inferring per-channel nuisance/discrepancy parameters *jointly* with the
plasma state, or by selecting the tempering exponent adaptively from predictive performance.
A weight fixed once from an audit is a point estimate of that, and is the weaker form. It is
chosen here because it is measurable, auditable and cheap, not because it is the best
available. The honest statement in any report is "point-estimate power-likelihood correction,
calibrated once against a specific noise model" — and ``w`` must be re-derived if the noise
model, the channel physics or the operating point changes. Worth knowing that Fischer's
tutorial names a Student-t likelihood as the field's usual workhorse for robustness; that
would be the next upgrade in this same module, and it needs no audit number to be exactly
right.

Two limits are deliberately closed off:

- **A weight may not be zero or negative.** Zero would exclude a channel by making its
  contribution vanish, which is exactly the "formed and small" behaviour doc 01 IF-6
  forbids — exclusion is the detection gate's job and is recorded; a zero weight would not
  be. A negative weight would reward disagreement.
- **A weight above 1 narrows the posterior**, claiming a channel is *better* than it says.
  That is permitted, because a channel can genuinely overstate its uncertainty, but it
  inverts the direction this mechanism exists to fix, so it is recorded in
  :attr:`FusionDetail.weights` where a reviewer can see it rather than inferred from
  the log-probability.

Weights are carried in the result for the same reason exclusions are: "we down-weighted
Thomson by 4x" must be checkable after the fact, not trusted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

__all__ = [
    "BlindChannelError",
    "Channel",
    "FusionDetail",
    "JointLikelihood",
]


class _Instrument(Protocol):
    """The three methods fusion calls, structurally.

    A local protocol rather than an import of
    :class:`vpl.core.protocols.instrument.Instrument` so that a test can supply a stand-in
    without constructing a spectrograph — which is what makes the arithmetic here checkable
    against hand-computed sums (ADR-011: agreement with yourself is not evidence).
    """

    def forward(self, state: Any, w: Any) -> Any: ...

    def likelihood(self, obs: Any, pred: Any) -> float: ...

    def is_informative(self, state_guess: Any) -> bool: ...


class BlindChannelError(RuntimeError):
    """Every channel reported itself below its detection floor.

    Distinct from "the fit is bad": there is no measurement at all here. Returning 0.0 — the
    empty sum — would make every parameter equally probable, so the optimiser would wander
    the prior and report convergence on whatever it happened to touch last.
    """


@dataclass(frozen=True, slots=True)
class Channel:
    """One diagnostic channel: an instrument, what it recorded, and when.

    Attributes:
        name: How the channel is identified in an ablation and in a report. Unique within a
            :class:`JointLikelihood`.
        instrument: Anything satisfying the instrument protocol.
        observation: What was actually recorded.
        window: The acquisition window the observation was taken over. Channels legitimately
            differ here by many orders of magnitude — doc 02 §7 has OES gating at
            nanoseconds while Thomson accumulates for ~700 s — which is sound for a
            steady-state plasma and **is not** for a transient. doc 05 §3.2 calls treating
            such windows as "the same instant" straightforwardly false.
        weight: Multiplier on this channel's log-likelihood. ``1.0`` leaves the term exactly
            as the instrument reported it — bit-for-bit, since multiplying a float by 1.0 is
            exact. Below 1.0 inflates that channel's effective variance by ``1 / weight``,
            which is how a measured uncertainty-understatement factor is applied: see the
            module docstring. Must be finite and strictly positive.
    """

    name: str
    instrument: _Instrument
    observation: Any
    window: Any
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class FusionDetail:
    """The joint log-probability together with who actually contributed to it.

    Attributes:
        log_prob: The sum over contributing channels.
        contributing: Names of the channels that were above their detection floor, in order.
        excluded: Names of the channels that reported themselves blind. Carried so that a
            claim of "N channels fused" can be checked rather than trusted.
        weights: The weight applied to each contributing channel, positionally aligned with
            :attr:`contributing`. Carried for the same reason as :attr:`excluded`: a claim
            that a channel was down-weighted by some factor should be readable off the
            result rather than inferred from the log-probability or trusted from a brief.
    """

    log_prob: float
    contributing: tuple[str, ...]
    excluded: tuple[str, ...]
    weights: tuple[float, ...] = ()


class JointLikelihood:
    """doc 05 §3.2's sum over channels, with the detection gate and ablation built in.

    Immutable: :meth:`without` returns a new instance. An ablation sweep therefore cannot
    quietly degrade the baseline it is measured against, which it would if dropping a channel
    mutated shared state.
    """

    __slots__ = ("_channels",)

    def __init__(self, channels: Sequence[Channel]) -> None:
        if len(channels) == 0:
            raise ValueError(
                "a joint likelihood needs at least one channel; with none the log "
                "probability is the empty sum, 0.0, which is indistinguishable from a "
                "perfect fit"
            )
        names = [channel.name for channel in channels]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate channel name(s) {', '.join(duplicates)}: ablation by name would "
                f"be ambiguous, and the same measurement could be counted twice"
            )
        for channel in channels:
            weight = float(channel.weight)
            if not math.isfinite(weight):
                raise ValueError(
                    f"channel {channel.name!r} has a non-finite weight ({channel.weight}); "
                    f"it would make the joint log-probability non-finite at every state, so "
                    f"the optimiser would stop and report whatever it touched last"
                )
            if weight <= 0.0:
                raise ValueError(
                    f"channel {channel.name!r} has weight {channel.weight}, which must be "
                    f"strictly positive. A zero weight excludes a channel by making its "
                    f"contribution vanish, which is the 'formed and small' behaviour doc 01 "
                    f"IF-6 forbids — exclusion is the detection gate's job and is recorded "
                    f"in FusionDetail.excluded, whereas a zero weight would leave no trace. "
                    f"A negative weight would reward a channel for disagreeing. To drop a "
                    f"channel deliberately, use .without({channel.name!r})."
                )
        self._channels = tuple(channels)

    @property
    def names(self) -> tuple[str, ...]:
        """Channel names in order — what an ablation sweep iterates over."""
        return tuple(channel.name for channel in self._channels)

    def without(self, name: str) -> JointLikelihood:
        """A copy with one channel removed — doc 11 §9 item 6's primitive.

        Raises:
            KeyError: If no channel has that name. A typo would otherwise ablate nothing,
                and the resulting figure would show two identical bars that read as "this
                channel carries no information" — the opposite of the truth.
            ValueError: If it would empty the set.
        """
        if name not in self.names:
            raise KeyError(
                f"no channel named {name!r}; this likelihood has {', '.join(self.names)}"
            )
        remaining = tuple(channel for channel in self._channels if channel.name != name)
        if not remaining:
            raise ValueError(
                f"dropping {name!r} would leave no channels; an ablation needs at least one "
                f"to remain"
            )
        return JointLikelihood(remaining)

    def with_weights(self, weights: dict[str, float]) -> JointLikelihood:
        """A copy with per-channel weights applied — the error-budget audit's output port.

        Immutable for the same reason :meth:`without` is: a weighted run must not silently
        degrade the unweighted baseline it is being compared against.

        Args:
            weights: Channel name to weight. Channels absent from the mapping keep the
                weight they already have, so applying a correction for two channels does not
                require restating the other two.

        Raises:
            KeyError: If a name is not a channel here. Silently ignoring it would produce a
                run that looks weighted, is not, and differs from the baseline by nothing —
                which reads as "the weighting had no effect", the opposite of the truth.
            ValueError: If a weight is not finite and strictly positive.
        """
        unknown = sorted(set(weights) - set(self.names))
        if unknown:
            raise KeyError(
                f"no channel named {', '.join(repr(name) for name in unknown)}; this "
                f"likelihood has {', '.join(self.names)}. A weight aimed at a name that "
                f"does not exist would apply to nothing and the run would be "
                f"indistinguishable from the unweighted baseline."
            )
        return JointLikelihood(
            tuple(
                replace(channel, weight=weights[channel.name])
                if channel.name in weights
                else channel
                for channel in self._channels
            )
        )

    def detail(self, state: Any) -> FusionDetail:
        """The sum, plus the record of who contributed and who was blind."""
        total = 0.0
        contributing: list[str] = []
        excluded: list[str] = []
        weights: list[float] = []

        for channel in self._channels:
            # The gate is asked *before* the prediction is formed. doc 01 IF-6 makes this a
            # correctness point rather than an optimisation, but it is also the difference
            # between skipping a Thomson forward model and computing one to discard.
            if not channel.instrument.is_informative(getattr(state, "params", state)):
                excluded.append(channel.name)
                continue
            predicted = channel.instrument.forward(state, channel.window)
            value = float(channel.instrument.likelihood(channel.observation, predicted))
            if not math.isfinite(value):
                raise ValueError(
                    f"channel {channel.name!r} returned a non-finite log-likelihood "
                    f"({value}); a NaN propagates through the sum and makes every parameter "
                    f"look equally good, so the optimiser stops and reports a result"
                )
            # Multiplying a float by exactly 1.0 is exact in IEEE 754, so an unweighted
            # likelihood is bit-for-bit what it was before weights existed. That is asserted
            # in the tests rather than assumed: it is what makes every number measured
            # before this change still comparable with every number measured after it.
            total += channel.weight * value
            contributing.append(channel.name)
            weights.append(float(channel.weight))

        if not contributing:
            raise BlindChannelError(
                f"no channel is above its detection floor at this state "
                f"(blind: {', '.join(excluded)}). doc 01 IF-6 treats this as absence of "
                f"measurement, not as weak measurement, so there is no likelihood to report."
            )
        return FusionDetail(
            log_prob=total,
            contributing=tuple(contributing),
            excluded=tuple(excluded),
            weights=tuple(weights),
        )

    def log_prob(self, state: Any) -> float:
        """The joint log-likelihood — the number an inference engine consumes."""
        return self.detail(state).log_prob

    def __repr__(self) -> str:
        return f"JointLikelihood({', '.join(self.names)})"
