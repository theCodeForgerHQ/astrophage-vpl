"""``NoiseModel`` — one switchable source, applied inside ``F4``.

Doc 04 §7.2 tabulates eighteen sources, from photon shot noise to RF pickup, and opens
with the constraint that shapes this module: "Every source is separately switchable so its
contribution can be isolated in the error budget." Doc 08 §6's manifest lists them by
identifier (``enabled_sources: [N1, N2, ...]``), doc 04 §8's test V-30 requires each to
switch off *exactly*, and doc 06 §4 needs each one's contribution separately or the budget
does not decompose.

## Why ``Signal`` and not ``Measurement``

Doc 04 §1 states the rule that keeps synthetic data honest: "**``F4`` never sees the
plasma.** The detector model receives photons and nothing else. It has no access to
``n_e``, ``T_e``, or any plasma variable. If a detector module needs a plasma quantity,
that is a design error, because a real detector cannot know one."

A :class:`~vpl.core.state.Measurement` carries an instrument id, an acquisition window and
a calibration state — context a photocathode does not have. Handing one to a noise source
would put plasma-adjacent metadata inside ``F4`` and make the layering violation
type-correct. :class:`Signal` is therefore the minimum a detector stage can act on: an
array, and which stage of the doc 04 §7.1 chain it sits at.

## Why the stage is an enum and not units

Doc 04 §7.1's chain runs photons → photoelectrons → electrons → ADU. "ADU" is not a
physical unit and has no entry in any registry; treating it as one would mean either
inventing a unit or dropping the distinction. The distinction is the load-bearing part —
read noise applies in electrons, quantisation in ADU, and applying either at the wrong
stage produces a plausible photon-transfer curve that fails V-28 for reasons nobody can
see — so it is carried explicitly.

## What is deliberately not on the protocol

An identifier. Doc 08 §6's manifest switches sources by name (``N1``…``N18``), but doc 08
§4 gives :class:`NoiseModel` three methods and no identity, and doc 08 §10 already makes
the plugin's entry-point name the identity. Adding a fourth method would be this module
disagreeing with the specification about where a name lives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

__all__ = ["NoiseModel", "Signal", "SignalDomain"]


class SignalDomain(StrEnum):
    """Which stage of the doc 04 §7.1 detection chain a signal sits at.

    Each stage has its own statistics (doc 04 §7.2): shot noise is Poisson on photons,
    the MCP gain is Pólya on photoelectrons, read noise is Gaussian on electrons, and
    quantisation is uniform on ADU. A source applied one stage out of place gives a
    photon-transfer curve that looks right and fails V-28 with no visible cause.
    """

    PHOTON = "photon"
    PHOTOELECTRON = "photoelectron"
    ELECTRON = "electron"
    ADU = "adu"


@dataclass(frozen=True, eq=False, slots=True)
class Signal:
    """Samples at one stage of the detection chain — what a noise source transforms.

    Attributes:
        values: The samples. Read-only; copied at construction, because the caller's
            buffer is theirs to reuse for the next frame and a signal aliasing it would
            change under a noise stage that had already read it.
        domain: Which stage of doc 04 §7.1 these live at.
    """

    values: NDArray[np.float64]
    domain: SignalDomain

    def __post_init__(self) -> None:
        array = np.array(self.values, dtype=np.float64, copy=True)

        if array.ndim < 1:
            raise ValueError("a signal is an array of samples, got a scalar")
        if array.size < 1:
            raise ValueError("a signal needs at least one sample, got an empty array")
        if not np.all(np.isfinite(array)):
            raise ValueError("signal must be finite; found nan or inf")

        # Negative samples are legal and deliberately not rejected. An ADU trace with the
        # bias offset removed goes negative on the read-noise tail (doc 04 §7.1), and a
        # type that clipped it would bias the low end of the V-28 photon-transfer curve
        # upward — turning a verification test into a test of the clip.

        array.flags.writeable = False
        object.__setattr__(self, "values", array)

    def with_values(self, values: NDArray[np.float64]) -> Signal:
        """The same signal at the same stage, with new samples.

        What ``NoiseModel.apply`` and ``NoiseModel.variance`` return. Keeping the domain
        rather than taking it again means a noise source cannot silently move a signal to
        a different stage of the chain, which would be a doc 04 §1 layering violation
        with no symptom.
        """
        return replace(self, values=values)

    @property
    def n_samples(self) -> int:
        return int(self.values.size)

    @property
    def shape(self) -> tuple[int, ...]:
        return cast("tuple[int, ...]", self.values.shape)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            return NotImplemented
        return self.domain is other.domain and np.array_equal(self.values, other.values)

    def __repr__(self) -> str:
        return f"Signal({self.domain.value}, n={self.n_samples}, shape={self.shape})"


@runtime_checkable
class NoiseModel(Protocol):
    """One of the doc 04 §7.2 sources, individually switchable — doc 08 §4.

    Structural at runtime, nominal at type-check time; see
    :class:`~vpl.core.protocols.forward.ForwardSolver` for the reasoning.
    """

    def apply(self, signal: Signal, rng: Generator) -> Signal:
        """Corrupt ``signal`` with this source, drawing from ``rng``.

        The generator is a parameter rather than owned state. Doc 00 E3 promises
        bit-for-bit reproducibility from ``(manifest, commit, environment, seed)`` and
        doc 10 §5 derives per-stream seeds from one root; a source that seeded itself
        would put a second, undeclared entropy source into every synthetic dataset.

        A disabled source must return the signal unchanged — doc 04 §8 test V-30 requires
        each source in isolation to reproduce the noiseless limit *exactly*, not to
        within tolerance.
        """
        ...

    def variance(self, signal: Signal) -> Signal:
        """This source's contribution to the variance of ``signal``.

        Returns a :class:`Signal` whose values are variances rather than samples — the
        squared-units abuse doc 08 §4 chooses on purpose, so that variance propagates
        through the same stage-by-stage pipeline as the signal and cannot be applied at a
        stage the signal never passed through. Doc 06 §4 sums these into the budget, and
        doc 06 §4's correlated terms are why each source reports its own rather than one
        combined error bar being attached at the end.
        """
        ...

    def enabled(self) -> bool:
        """Whether this source is switched on — doc 08 §6's ``enabled_sources``.

        Doc 04 §8 V-30 is the test this exists for, and doc 07 §5.2's F-14 noise-scaling
        ablation is the study.
        """
        ...
