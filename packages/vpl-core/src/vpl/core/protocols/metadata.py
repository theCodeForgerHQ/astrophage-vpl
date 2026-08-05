"""Who computed this, with what, and on whose authority — doc 00 C2, doc 08 §7.

Doc 00 C2 is a hard constraint, not a nicety: "The contribution is integration and rigour,
not discovery. **Every algorithm must have a citation.** If implementing something requires
solving an open problem, the feature is cut rather than fudged." The consequence in code is
that :class:`SolverMetadata` and :class:`InstrumentMetadata` cannot be constructed without
one. There is no permissive path, because the permissive path is the one that gets taken
at 2 a.m. and never revisited.

The two metadata types are close but not identical, and the differences are the
specification's rather than an oversight:

- An instrument carries an ``instrument_id`` as well as a ``name``. Doc 08 §7 stores
  measurements as "one group per instrument", so a channel needs a short machine-facing
  key ("oes") *and* a human-facing description of the hardware being modelled. A solver
  is named once, by the dotted path a manifest resolves (doc 08 §6), and needs only one.
- An instrument carries a :class:`DetectionFloor`. Doc 01 IF-6 is stated as a requirement
  and not a caveat — "the channel contributes *no information* over the lower third of
  R-ENV-1 and must be modelled as absent there" — so the floor is required, not optional.
  ``Instrument.is_informative`` is the gate; this is the number it gates on, and it names
  the requirement that fixed it so that doc 00 C4's "no hidden assumptions" holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vpl.core.protocols.traceability import RequirementId, canonical_requirement_id
from vpl.core.state import InstrumentId
from vpl.core.units import Quantity, magnitude_in

__all__ = ["Citation", "DetectionFloor", "InstrumentMetadata", "SolverMetadata"]


def _required_text(value: str, *, what: str) -> str:
    """Reject an empty field that a reader would need in order to check the work."""
    if not value.strip():
        raise ValueError(f"{what} must not be empty")
    return value


def _required_citations(citations: tuple[Citation, ...], *, what: str) -> tuple[Citation, ...]:
    """Enforce doc 00 C2 at construction rather than at review."""
    if not citations:
        raise ValueError(
            f"{what} has no citation. doc 00 C2 requires every algorithm to have one; "
            "an uncitable algorithm is either an open problem in disguise, in which case "
            "C2 says cut it, or it is cited somewhere and the reference is missing here."
        )
    return tuple(citations)


@dataclass(frozen=True, slots=True)
class Citation:
    """A reference for one algorithm — doc 00 C2.

    Attributes:
        key: Citation key, matching the bibliography under ``refs/``. Short, stable and
            greppable, because doc 00 C2 has to be auditable and "see the paper" is not.
        reference: The human-readable reference, for the artifact and the report.
        doi: Digital object identifier, without a resolver prefix.
    """

    key: str
    reference: str
    doi: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _required_text(self.key, what="citation key"))
        object.__setattr__(
            self, "reference", _required_text(self.reference, what="citation reference")
        )

        if self.doi is not None and not (self.doi.startswith("10.") and "/" in self.doi):
            # A URL stored here looks right in a repr and resolves to nothing through a
            # DOI resolver, so the failure survives every review that reads rather than
            # clicks. DOIs are registrant/suffix pairs under the "10." directory.
            raise ValueError(
                f"doi {self.doi!r} is not a DOI. A DOI is '10.<registrant>/<suffix>'; "
                "store the bare identifier, not a resolver URL."
            )

    def __repr__(self) -> str:
        return f"Citation({self.key!r}: {self.reference})"


@dataclass(frozen=True, slots=True)
class DetectionFloor:
    """Below this, a channel measures nothing — doc 01 IF-6.

    The worked example is the interferometer: doc 01 §5.4 computes a floor of
    ``3.3e16 m^-3`` for the CO2 chord and states plainly that "an inversion that quietly
    ingests noise as data will produce confident nonsense at low density". The floor is
    therefore part of the instrument's declared identity rather than a constant buried in
    its likelihood.

    Attributes:
        quantity: The state quantity the floor binds on, named as
            :class:`~vpl.core.state.PlasmaParams` spells it. Stated explicitly because the
            four channels are limited by different things — the interferometer by ``n_0``,
            LIF by metastable density — and a bare threshold could not say which.
        threshold: The smallest value at which the channel is informative. The channel is
            informative *at* the threshold; doc 01 IF-6 declares the blind region as
            strictly below its floor.
        requirement: The doc 01 requirement that fixes this number. Validated as a
            requirement id, so a floor cannot be declared without something to trace it
            to (doc 00 C4).
    """

    quantity: str
    threshold: Quantity
    requirement: RequirementId

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", _required_text(self.quantity, what="quantity"))
        object.__setattr__(self, "requirement", canonical_requirement_id(self.requirement))

        magnitude = float(self.threshold.magnitude)
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError(
                f"detection floor threshold must be finite and positive, got {self.threshold}"
            )

    def admits(self, value: Quantity) -> bool:
        """Whether ``value`` is at or above the floor — the doc 01 IF-6 gate.

        Args:
            value: The state quantity this floor binds on.

        Returns:
            ``True`` if the channel carries information at ``value``.

        Raises:
            DimensionalityError: If ``value`` is not commensurate with the threshold.
                Comparing a density against an energy would otherwise silently gate the
                wrong channel off.
        """
        units = str(self.threshold.units)
        return float(magnitude_in(value, units)) >= float(self.threshold.magnitude)

    def __repr__(self) -> str:
        return f"DetectionFloor({self.quantity} >= {self.threshold:.3g~P}, {self.requirement})"


@dataclass(frozen=True, slots=True)
class SolverMetadata:
    """What a ``ForwardSolver`` says about itself — doc 08 §4, doc 08 §7.

    The fidelity level is deliberately **not** here. ``ForwardSolver.fidelity()`` is a
    method of the protocol in its own right, and carrying the same fact in two places
    invites the two to disagree — at which point nothing in the artifact says which one
    the run actually used.

    Attributes:
        name: The dotted path a manifest resolves, e.g.
            ``"vpl.physics.kinetic.pic1d3v"`` (doc 08 §6).
        version: Solver version, as it appears in
            :attr:`~vpl.core.provenance.Provenance.solver_versions`.
        citations: At least one. Doc 00 C2.
        description: Free text for the report; never load-bearing.
    """

    name: str
    version: str
    citations: tuple[Citation, ...]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, what="solver name"))
        object.__setattr__(self, "version", _required_text(self.version, what="solver version"))
        object.__setattr__(
            self,
            "citations",
            _required_citations(tuple(self.citations), what=f"solver {self.name}"),
        )

    def __repr__(self) -> str:
        return f"SolverMetadata({self.name!r} v{self.version}, {len(self.citations)} citations)"


@dataclass(frozen=True, slots=True)
class InstrumentMetadata:
    """What an ``Instrument`` says about itself — doc 08 §4, doc 01 IF-6.

    A *real* instrument implements the same protocol and therefore returns one of these
    too (doc 04 §9, doc 00 E2). Nothing here assumes the instrument is simulated: a
    detection floor and a calibration reference are properties of hardware first.

    Attributes:
        instrument_id: Short key naming this channel's artifact group (doc 08 §7).
        name: What is being modelled, e.g. ``"Andor iStar 340T ICCD + SP-2750"``.
        version: Model version, for provenance.
        citations: At least one. Doc 00 C2 applies to a measurement model exactly as it
            applies to a solver.
        detection_floor: The doc 01 IF-6 gate. Required — see the module docstring.
        description: Free text for the report.
    """

    instrument_id: InstrumentId
    name: str
    version: str
    citations: tuple[Citation, ...]
    detection_floor: DetectionFloor
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instrument_id", _required_text(self.instrument_id, what="instrument id")
        )
        object.__setattr__(self, "name", _required_text(self.name, what="instrument name"))
        object.__setattr__(self, "version", _required_text(self.version, what="instrument version"))
        object.__setattr__(
            self,
            "citations",
            _required_citations(tuple(self.citations), what=f"instrument {self.instrument_id}"),
        )

    def __repr__(self) -> str:
        return (
            f"InstrumentMetadata({self.instrument_id!r}, {self.name!r} v{self.version}, "
            f"floor={self.detection_floor.threshold:.3g~P})"
        )
