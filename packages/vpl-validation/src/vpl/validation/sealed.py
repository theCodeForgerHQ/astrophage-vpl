"""The sealed-truth barrier and tier labelling — doc 05 §7, doc 07 §3.

doc 05 §7 states the problem and, unusually, prescribes the shape of the solution:

    An inverse crime is committed when the same model and discretisation generate the data
    and perform the inversion. The result is an artificially perfect recovery that proves
    nothing. This is the most common way computational inverse-problem work is invalidated,
    and it is guarded against **structurally rather than by good intentions**.

That last clause is a design instruction, and it is the reason this module exists rather
than a convention in a contributing guide.

## Why good intentions are not enough

Nobody commits an inverse crime on purpose. It happens because the truth is *in scope* —
sitting in a local variable, one attribute access away from the code computing the estimate.
Then something entirely reasonable reaches for it:

- an initial guess, "just to start the optimiser somewhere sensible";
- a convergence check that stops when the estimate is close enough to the answer;
- a plotting routine that normalises by the true value;
- a test fixture that builds the observation *from* the truth and then asserts against it.

Each is a one-line change that looks like a convenience and silently converts the result
into a tautology. The recovery becomes excellent, and the excellence is the symptom.

Sealing the value makes every one of those a visible, deliberate act. It does not make the
crime impossible — nothing can, short of separate processes — but it moves it from
"accident nobody noticed" to "line of code somebody had to write".

## The tiers

doc 05 §7.2 requires every headline result to be reported at three levels, together:

- **T0** — same model, no noise. Recovery to numerical tolerance. Failing T0 is a *bug*,
  and doc 05 is explicit that nothing else is meaningful until it passes. That is why
  :meth:`SealedTruth.assert_t0_consistency` raises ``AssertionError`` and says "bug": it is
  a statement about the code, not a measurement of performance.
- **T1** — same model, with noise. Optimistic; the upper bound on achievable accuracy.
- **T2** — mismatched models, noise, imperfect calibration. **The number quoted publicly.**

And the rule this module enforces:

    **Reporting T1 as if it were T2 is treated as a project defect**, and the CI enforces
    that any figure showing accuracy carries its tier label.

A label with a default is chosen by whoever is in a hurry, which is precisely the situation
the rule is written for — so ``tier`` is a required keyword argument with no default.

## What this does not do

It does not verify that the models actually differ. :func:`tier_of_configuration` takes the
caller's word for ``same_model``, because this package cannot import a solver without
inverting the dependency that keeps it impartial. The mismatches of doc 05 §7.1 — grid,
timestep, collision set, EEDF parameterisation, calibration — are the experiment layer's to
arrange and the manifest's to record. This module makes the *claim* explicit and checkable;
it does not audit it.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "InverseCrimeError",
    "SealedTruth",
    "Tier",
    "TierMismatchError",
    "tier_of_configuration",
]

type Value = float | NDArray[np.float64]

#: Below this magnitude a truth is treated as zero and the error reported as absolute.
#: A relative error against a vanishing denominator is unbounded and uninformative, and the
#: quantities this seals — a drift, a residual — legitimately pass through zero.
_ZERO_THRESHOLD: Final[float] = 1e-300


class Tier(IntEnum):
    """doc 05 §7.2's three reporting tiers, ordered by honesty.

    ``IntEnum`` so that ``Tier.T0 < Tier.T2`` is meaningful and a report can require "at
    least T2" rather than string-matching a label. The ordering is *not* a quality ranking
    of the run — a T0 result is not worse work than a T2 one, it answers a different
    question — it is a ranking of how much a number may be claimed to mean.
    """

    T0 = 0
    """Same model, no noise. Tests that the code is self-consistent."""

    T1 = 1
    """Same model, with noise. The optimistic bound."""

    T2 = 2
    """Mismatched models, noise, imperfect calibration. The real result."""


class InverseCrimeError(RuntimeError):
    """The truth was read before an estimate was committed, or an estimate was revised."""


class TierMismatchError(RuntimeError):
    """A result was claimed at a tier its configuration does not support."""


def tier_of_configuration(*, same_model: bool, noise: bool, imperfect_calibration: bool) -> Tier:
    """Map a run configuration to the tier it may be reported at — doc 05 §7.2's table.

    A function rather than a table in a document somebody follows, because the difference
    between T1 and T2 is exactly the difference between an optimistic number and a
    publishable one, and it is decided by three booleans that are easy to get wrong.

    Raises:
        TierMismatchError: If the models are mismatched but the run is missing one of the
            other mandatory conditions of doc 05 §7.1. Mismatching the physics and then
            skipping the noise, or skipping the *estimated* calibration of doc 04 §7.3, is
            the tempting half-measure — it feels honest and is not T2. The estimated
            calibration is the one most easily forgotten, because the true response is
            already sitting in the simulation.
    """
    if same_model:
        return Tier.T1 if noise else Tier.T0

    if not noise:
        raise TierMismatchError(
            "a mismatched-model run without noise is not T2 (doc 05 §7.1 requires noise); "
            "it is not T0 or T1 either, since those are defined for the same model. "
            "Add noise, or run same_model=True."
        )
    if not imperfect_calibration:
        raise TierMismatchError(
            "a mismatched-model run with perfect calibration is not T2: doc 05 §7.1 makes "
            "the *estimated* instrument response of doc 04 §7.3 a mandatory mismatch. Using "
            "the true response is the easiest of the mismatches to forget, because the "
            "simulation already has it."
        )
    return Tier.T2


class SealedTruth:
    """A ground-truth value that cannot be read until an estimate has been committed.

    Deliberately a mutable class rather than a frozen dataclass: the seal is *state*, and
    the point is that reading it changes what the object will allow.

    Args:
        value: The truth. Stored privately and never exposed until commitment.
        name: What it is, so the error message is actionable in a stack trace nobody
            expected to be reading.

    Example:
        >>> sealed = SealedTruth(value=6577.0, name="Gamma_E")
        >>> sealed.commit_estimate(6510.0, tier=Tier.T0)
        >>> round(sealed.relative_error, 4)
        0.0102
    """

    __slots__ = ("_committed", "_estimate", "_name", "_tier", "_value")

    def __init__(self, *, value: Value, name: str) -> None:
        self._value = value
        self._name = name
        self._committed = False
        self._estimate: Value | None = None
        self._tier: Tier | None = None

    @property
    def name(self) -> str:
        """What is sealed. Always readable — the *identity* is not the secret."""
        return self._name

    @property
    def is_committed(self) -> bool:
        """Whether an estimate has been committed. Readable without breaking the seal."""
        return self._committed

    def _require_committed(self, what: str) -> None:
        if not self._committed:
            raise InverseCrimeError(
                f"{what} of {self._name!r} was read before an estimate has been committed. "
                f"doc 05 §7 guards the inverse crime structurally: call commit_estimate() "
                f"first. If you are reaching for the truth to seed an optimiser, normalise "
                f"a plot or stop an iteration, that is the crime itself."
            )

    def commit_estimate(self, estimate: Value, *, tier: Tier) -> None:
        """Commit an estimate and unseal the truth.

        Args:
            estimate: What the inversion produced.
            tier: Required, with no default, per doc 05 §7.2. A default would be chosen by
                whoever was in a hurry, which is the situation the rule exists for.

        Raises:
            InverseCrimeError: If an estimate was already committed. Without this the
                barrier is theatre — commit anything, read the truth, then revise.
        """
        if self._committed:
            raise InverseCrimeError(
                f"an estimate for {self._name!r} was already committed "
                f"({self._estimate!r} at {self._tier}). Revising it after the truth became "
                f"readable is the inverse crime with extra steps."
            )
        self._estimate = estimate
        self._tier = tier
        self._committed = True

    @property
    def value(self) -> Value:
        """The truth. Raises until an estimate is committed."""
        self._require_committed("the value")
        return self._value

    @property
    def estimate(self) -> Value:
        """The committed estimate."""
        self._require_committed("the estimate")
        assert self._estimate is not None
        return self._estimate

    @property
    def tier(self) -> Tier:
        """The tier the estimate was committed at."""
        self._require_committed("the tier")
        assert self._tier is not None
        return self._tier

    @property
    def relative_error(self) -> float:
        """``|estimate/truth - 1|``, or the absolute error where the truth is zero.

        Falls back to absolute error below :data:`_ZERO_THRESHOLD` because a relative
        error against a vanishing denominator is unbounded and says nothing, and the
        quantities worth sealing — a drift, a residual — legitimately pass through zero.
        """
        self._require_committed("the error")
        truth = np.asarray(self._value, dtype=np.float64)
        estimate = np.asarray(self._estimate, dtype=np.float64)
        scale = float(np.max(np.abs(truth))) if truth.size else 0.0
        if scale < _ZERO_THRESHOLD:
            return float(np.max(np.abs(estimate - truth)))
        return float(np.max(np.abs(estimate - truth)) / scale)

    def assert_at_least(self, required: Tier) -> None:
        """Require the committed tier to be at least ``required``.

        This is doc 05 §7.2's "reporting T1 as if it were T2 is treated as a project
        defect", as a callable a report or a CI check can invoke.
        """
        if self.tier < required:
            raise TierMismatchError(
                f"{self._name!r} was produced at {self.tier.name} but is being reported as "
                f"{required.name}. doc 05 §7.2 treats reporting T1 as if it were T2 as a "
                f"project defect: {Tier.T1.name} is the optimistic bound, not the result."
            )

    def assert_t0_consistency(self, *, tolerance: float) -> None:
        """Assert T0 recovery to numerical tolerance.

        Raises ``AssertionError`` rather than a domain error, and says "bug", because
        doc 05 §7.2 is explicit that this is a statement about the code:

            Failing T0 means a bug; nothing else is meaningful until it passes.

        Raises:
            TierMismatchError: If the result is not T0. A T1 result is noisy by
                construction and cannot recover to numerical tolerance; running this check
                on it would fail spuriously, or — with a tolerance loosened to make it pass
                — succeed while meaning nothing.
        """
        if self.tier is not Tier.T0:
            raise TierMismatchError(
                f"assert_t0_consistency is only meaningful at T0; {self._name!r} was "
                f"committed at {self.tier.name}, which is noisy by construction."
            )
        error = self.relative_error
        if not error <= tolerance:
            raise AssertionError(
                f"T0 consistency failed for {self._name!r}: relative error {error:.3e} "
                f"exceeds {tolerance:.3e}. doc 05 §7.2 — this is a bug, not a result, and "
                f"no T1 or T2 number is meaningful until it passes."
            )

    def __repr__(self) -> str:
        """Never leaks the value before commitment.

        A sealed value that prints itself in a traceback, a debugger or a log line is not
        sealed, and that is the leak that would actually happen in practice.
        """
        if not self._committed:
            return f"SealedTruth({self._name!r}, sealed)"
        return (
            f"SealedTruth({self._name!r}, value={self._value!r}, "
            f"estimate={self._estimate!r}, tier={self._tier.name if self._tier else None})"
        )

    def __format__(self, spec: str) -> str:
        """Formatting an uncommitted seal must not leak it either — ``f"{sealed}"``."""
        return repr(self) if not self._committed else format(self._value, spec)

    def __getattr__(self, item: str) -> Any:
        raise AttributeError(
            f"SealedTruth has no attribute {item!r}. If you are looking for the underlying "
            f"value, it is behind .value and only after commit_estimate()."
        )
