"""``Instrument`` — virtual and real alike, on one contract.

Doc 08 §4 declares it; doc 00 E2 states what it buys: "A real instrument can replace a
virtual one by implementing the same interface, with no change anywhere else." Doc 04 §9
spells out the real-hardware case — ``observe`` reads the device, ``forward`` raises — and
``vpl-plugin-mock-hardware`` (doc 08 §10) exists solely to prove it.

## The two returns that must not drift apart

``forward`` returns a noiseless :class:`~vpl.core.state.Observable`; ``observe`` returns a
noisy :class:`~vpl.core.state.Measurement`. Doc 04 §9 and doc 08 §4 both insist they come
from **one code path** with noise and calibration error as switchable stages, "which
guarantees they cannot drift apart — a class of bug that would silently invalidate every
result". The protocol cannot enforce that; only a test can, and
``Measurement.as_observable`` exists to make that test writable.

## Calibration is applied, not assumed

Doc 04 §7.3 fixes the chain::

    true instrument response → calibration measurement (with its own noise)
                             → estimated response (biased, uncertain)
                             → applied by the analysis pipeline

and states the consequence: "Applying the true calibration would be a form of inverse
crime and would understate the error." ``calibrate`` therefore takes a
:class:`CalibrationSet` of reference *standards* — doc 02 §11's lamps, Raman and Rayleigh
scatterers and fiducial targets — and returns the :class:`Calibration` estimated from
them. Whether the true or the estimated response was applied is recorded on the result
using :class:`~vpl.core.state.CalibrationState`, the enum the measurement types already
use; duplicating it here would let the two disagree about the same run.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

from vpl.core.protocols.config import InstrumentConfig
from vpl.core.protocols.metadata import InstrumentMetadata
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    InstrumentId,
    Measurement,
    Observable,
    PlasmaParams,
    PlasmaState,
)
from vpl.core.units import Quantity

__all__ = [
    "Calibration",
    "CalibrationReference",
    "CalibrationSet",
    "Instrument",
    "LogProb",
]

#: One channel's contribution to the doc 05 §3.2 log-likelihood, in nats.
#:
#: Deliberately a bare ``float`` and not a wrapper. Two things could be argued into it and
#: both are wrong here. A "no contribution" sentinel for a channel gated off by doc 01
#: IF-6 would be summed by accident the first time somebody wrote ``sum(terms)``; the gate
#: is ``Instrument.is_informative``, evaluated *before* the term is asked for, so the
#: absence is expressed by there being no term rather than by a value meaning none. And a
#: units wrapper would be false precision: a log-probability is dimensionless, and the
#: base is fixed at ``e`` across every engine in doc 05 §5.
#:
#: ``-inf`` is legal and meaningful: a proposal that violates a hard constraint.
LogProb: TypeAlias = float


def _required_text(value: str, *, what: str) -> str:
    if not value.strip():
        raise ValueError(f"{what} must not be empty")
    return value


def _checked_fraction(value: float, *, what: str) -> float:
    """A relative (1-sigma) uncertainty, as doc 02 §11 quotes them."""
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value}")
    if value < 0.0:
        raise ValueError(f"{what} is a relative standard uncertainty and cannot be negative")
    return value


@dataclass(frozen=True, slots=True)
class CalibrationReference:
    """One reference standard a calibration is derived from — doc 02 §11.

    Doc 02 §11 tabulates these per instrument: an Hg/Ar pencil lamp against the NIST line
    list, a NIST-traceable tungsten-halogen lamp on the FEL scale, Raman scattering in
    N2, Rayleigh scattering in Ar, a machined fiducial target measured on a CMM. Each
    carries its own uncertainty, and doc 06 §4.1 shows those uncertainties dominating the
    calibration terms of the error budget.

    Attributes:
        name: The standard itself, e.g. ``"NIST FEL tungsten-halogen lamp"``.
        quantity: What it certifies, e.g. ``"absolute_radiometric"``. The key
            :meth:`CalibrationSet.for_quantity` looks up.
        value: The certified value.
        relative_uncertainty: Its 1-sigma relative uncertainty — doc 02 §11's "6 %".
        traceable_to: The scale it is traceable to. Required by doc 01 SYS-3, which asks
            for "a single documented reference source"; an untraceable standard cannot
            support an absolute density, and doc 01 N2 says the density must be absolute.
    """

    name: str
    quantity: str
    value: Quantity
    relative_uncertainty: float
    traceable_to: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, what="reference name"))
        object.__setattr__(self, "quantity", _required_text(self.quantity, what="quantity"))
        object.__setattr__(
            self, "traceable_to", _required_text(self.traceable_to, what="traceable_to")
        )
        _checked_fraction(self.relative_uncertainty, what=f"{self.name} uncertainty")

    def __repr__(self) -> str:
        return (
            f"CalibrationReference({self.quantity!r} from {self.name!r}, "
            f"u={self.relative_uncertainty:.2%})"
        )


@dataclass(frozen=True, eq=False, slots=True)
class CalibrationSet:
    """The reference standards handed to ``Instrument.calibrate`` — doc 02 §11.

    Attributes:
        references: The standards, re-sorted by certified quantity at construction so
            that iteration is deterministic (doc 00 E3).
    """

    references: tuple[CalibrationReference, ...]

    def __post_init__(self) -> None:
        seen: dict[str, CalibrationReference] = {}
        for reference in self.references:
            existing = seen.get(reference.quantity)
            if existing is not None:
                # doc 01 SYS-3 asks for a single documented reference source per chain.
                # With two, whichever the loader reached first would set the absolute
                # scale, and doc 01 N2 propagates that error straight into Gamma_i.
                raise ValueError(
                    f"two standards certify {reference.quantity!r}: {existing.name!r} and "
                    f"{reference.name!r}. doc 01 SYS-3 requires a single documented "
                    "reference source, because whichever was applied would set the "
                    "absolute scale and nothing in the artifact would say which."
                )
            seen[reference.quantity] = reference

        object.__setattr__(
            self, "references", tuple(sorted(self.references, key=lambda r: r.quantity))
        )

    @classmethod
    def of(cls, *references: CalibrationReference) -> CalibrationSet:
        """Build from positional standards, for call sites that have them to hand."""
        return cls(references=references)

    def for_quantity(self, quantity: str) -> CalibrationReference:
        """The standard certifying ``quantity``.

        Raises:
            KeyError: If the set carries no such standard. The message lists what it does
                carry, because a calibration failing here means the manifest and the
                instrument model disagree about which chain is being simulated.
        """
        for reference in self.references:
            if reference.quantity == quantity:
                return reference

        present = ", ".join(r.quantity for r in self.references) or "nothing"
        raise KeyError(f"no standard certifies {quantity!r}; this set certifies: {present}")

    def __iter__(self) -> Iterator[CalibrationReference]:
        return iter(self.references)

    def __len__(self) -> int:
        return len(self.references)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CalibrationSet):
            return NotImplemented
        return self.references == other.references

    def __repr__(self) -> str:
        return f"CalibrationSet({', '.join(r.quantity for r in self.references)})"


@dataclass(frozen=True, eq=False, slots=True)
class Calibration:
    """An instrument response — the one the pipeline applies, doc 04 §7.3.

    Attributes:
        instrument_id: Whose response this is.
        coefficients: The response itself, keyed by the quantity it corrects. Read-only.
        relative_uncertainty: 1-sigma relative uncertainty of each coefficient, keyed
            identically. Read-only. Doc 06 §4.1 propagates these; a correlated
            calibration error "affects *every* Thomson point identically and does **not**
            average down", which is why the uncertainty travels with the coefficient
            rather than being folded into a per-sample error bar.
        state: Whether this is the true response or the estimated one. Doc 04 §7.3 makes
            the estimated one the default everywhere except deliberate verification runs.
        reference: The standard it was derived from — doc 01 SYS-3.
    """

    instrument_id: InstrumentId
    coefficients: Mapping[str, float]
    relative_uncertainty: Mapping[str, float]
    state: CalibrationState
    reference: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instrument_id", _required_text(self.instrument_id, what="instrument id")
        )
        object.__setattr__(
            self, "reference", _required_text(self.reference, what="calibration reference")
        )

        missing = sorted(set(self.coefficients) - set(self.relative_uncertainty))
        extra = sorted(set(self.relative_uncertainty) - set(self.coefficients))
        if missing or extra:
            raise ValueError(
                f"every coefficient needs exactly one uncertainty; missing {missing}, "
                f"unexpected {extra}. A coefficient with no uncertainty drops silently "
                "out of the doc 06 §4 budget."
            )

        for name, value in self.relative_uncertainty.items():
            _checked_fraction(value, what=f"{name} uncertainty")

        if self.state is CalibrationState.ESTIMATED and not any(
            value > 0.0 for value in self.relative_uncertainty.values()
        ):
            # An estimated response with no uncertainty anywhere is the true response
            # wearing the honest label: indistinguishable in the artifact, and it makes
            # the doc 06 §4 calibration terms vanish from the budget without a trace.
            raise ValueError(
                f"calibration for {self.instrument_id!r} calls itself estimated but has "
                "zero uncertainty everywhere. doc 04 §7.3 derives the estimated response "
                "from a calibration measurement that has its own noise, so it is biased "
                "and uncertain by construction; a zero here is the inverse crime with an "
                "honest label on it."
            )

        object.__setattr__(self, "coefficients", MappingProxyType(dict(self.coefficients)))
        object.__setattr__(
            self, "relative_uncertainty", MappingProxyType(dict(self.relative_uncertainty))
        )

    @property
    def is_inverse_crime(self) -> bool:
        """Whether applying this understates the error budget — doc 04 §7.3."""
        return self.state.is_inverse_crime

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Calibration):
            return NotImplemented
        return (
            self.instrument_id == other.instrument_id
            and self.state is other.state
            and self.reference == other.reference
            and dict(self.coefficients) == dict(other.coefficients)
            and dict(self.relative_uncertainty) == dict(other.relative_uncertainty)
        )

    def __repr__(self) -> str:
        return (
            f"Calibration({self.instrument_id!r}, {len(self.coefficients)} coefficients, "
            f"{self.state.value}, from {self.reference!r})"
        )


@runtime_checkable
class Instrument(Protocol):
    """One measurement channel — doc 08 §4, doc 04 §9.

    Declared exactly as doc 08 §4 declares it, including the ``AcquisitionWindow``
    parameter. (Doc 04 §9 writes the same two methods with a ``TimeWindow``; doc 08 §4 is
    the later and binding statement, and :class:`~vpl.core.state.AcquisitionWindow` is the
    type that exists and that carries the doc 02 §10.3 phase binning the RF channels need.)

    **Checking is structural at runtime and nominal at type-check time.** ``isinstance``
    verifies the seven method names — enough to reject a broken plugin at load time — and
    the signatures are enforced by mypy against the plugin's own source. See
    :class:`~vpl.core.protocols.forward.ForwardSolver` for the full reasoning; it applies
    identically here, and the near-miss it guards against is sharper: an instrument whose
    ``forward`` returned a ``Measurement`` would feed an uncertainty into the likelihood
    twice, and nothing at runtime would say so.
    """

    def configure(self, cfg: InstrumentConfig) -> None:
        """Apply one entry of the manifest's ``instruments:`` list — doc 08 §6."""
        ...

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        """Derive the response from reference standards — doc 02 §11, doc 04 §7.3."""
        ...

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        """The noiseless expectation the likelihood compares against — doc 04 §9.

        A real instrument raises here (doc 04 §9, doc 00 E2): hardware has no noiseless
        mode, and returning its noisy reading would silently double-count the noise.
        """
        ...

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        """A noisy, imperfectly-calibrated observation — doc 04 §9, doc 01 SYS-4.

        Shares a code path with :meth:`forward`, with noise and calibration error as
        switchable stages. That is a requirement on the implementation, not on the
        signature, and it is what doc 04 §9 says "guarantees they cannot drift apart".
        """
        ...

    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb:
        """This channel's term in the doc 05 §3.2 sum.

        Per-channel, because doc 05 §3.1 gives the channels genuinely different
        statistics — Poisson for Thomson, Gaussian for interferometry — and a single
        shared Gaussian would understate the low-count tails that dominate exactly where
        the answer is hardest.
        """
        ...

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        """Whether this channel contributes at all — the doc 01 IF-6 detection gate.

        Doc 01 IF-6 is "a requirement, not a caveat": below its floor a channel must be
        modelled as *absent* rather than as a weak measurement, because "an inversion that
        quietly ingests noise as data will produce confident nonsense at low density".
        A ``False`` here means the term is not formed, not that it is formed and small.

        Takes a *guess* at the state, because the gate has to be evaluated at every
        proposal the sampler visits and the true state is sealed (doc 07 §3).
        """
        ...

    def metadata(self) -> InstrumentMetadata:
        """Identity, citations and the detection floor — doc 00 C2, doc 01 IF-6."""
        ...
