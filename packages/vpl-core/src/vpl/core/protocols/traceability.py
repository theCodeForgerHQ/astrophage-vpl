"""Source-annotated traceability — doc 08 §9.

Doc 01 §6 fixes the chain that has to be defensible::

    Requirement ID → design element → implementation → verification test → benchmark → figure

and states the rule that makes it worth having: "A requirement with no verification test is
a project defect." Doc 08 §9 then says how it is maintained — it is not. The matrix is
**generated** from annotations in source::

    @satisfies("R-SPAT-1", "R-ACC-4")
    @verified_by("V-22", "V-23")
    class LIFInstrument: ...

"Hand-maintained traceability matrices are always stale; generated ones cannot be."

## Why the identifiers are parsed rather than stored verbatim

A typo'd identifier that registered silently is worse than no traceability at all: it puts
a row in the generated matrix that no requirement in doc 01 claims and no test in doc 07
covers, and the matrix then reads as *evidence* that a requirement nobody wrote is
satisfied. So every identifier is parsed against the families the documents actually
define, and anything else fails at import time — which is the earliest moment a typo can
be caught and the only one at which it is cheap.

Two families of identifier, kept apart deliberately. ``@satisfies("V-22")`` is a plausible
mistake — it reads well and would file a verification test as though it were a
requirement — so requirement identifiers and evidence identifiers are validated against
disjoint sets, and each rejection says which side the identifier belongs on.

## What is deliberately not accepted

- The level-0 necessities ``N1…N7`` of doc 01 §1.4. They are derived *necessities*, and
  doc 01 §2.4 and §5 restate each of them as a hyphenated, verifiable requirement. The
  hyphenated one is what belongs in the matrix.
- The charter criteria ``S1``, ``E1``, ``C2`` (doc 00 §4, §5). They are project success
  criteria, not requirements on a component, and doc 01 §6's chain does not run through
  them.
- ``ADR-nnn`` decision records, ``RS/RT/RC-nn`` risks (doc 14), ``Q-nn`` open questions
  and ``G-*`` acceptance gates. A gate is a pass criterion *over* a benchmark, not
  evidence in its own right; the benchmark identifier is the traceable artifact.

## The two CI gates

Doc 08 §9: "A requirement with no ``@satisfies``, or a ``@satisfies`` with no
``@verified_by``, fails CI." :func:`unverified_claims` answers the second directly.
The first needs the requirement list from doc 01 §6, which this package does not hold and
should not — so :func:`uncovered_requirements` takes the catalogue as an argument. That is
the seam the CI job fills in.

The index is process-global module state, like :mod:`vpl.core.registry`, and for the same
reason: an annotation is a property of *what has been imported*, and threading a collector
object through every decorated class would put the traceability mechanism into the
signature of everything it traces.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

__all__ = [
    "EVIDENCE_FAMILIES",
    "REQUIREMENT_FAMILIES",
    "EvidenceId",
    "RequirementId",
    "TraceEntry",
    "TraceabilityError",
    "canonical_evidence_id",
    "canonical_requirement_id",
    "clear_registry",
    "entries",
    "entry_for",
    "evidence_of",
    "requirements_of",
    "satisfies",
    "traceability_matrix",
    "uncovered_requirements",
    "unverified_claims",
]

#: A requirement identifier from doc 01, in the canonical spelling of that document.
type RequirementId = str

#: A verification-test or benchmark identifier from docs 03, 04, 06 and 07.
type EvidenceId = str


class TraceabilityError(ValueError):
    """An annotation cannot be filed: a malformed identifier, or an untraceable target.

    A ``ValueError`` because that is what a bad argument is, and a named subclass because
    the CI job that generates the matrix reports these differently from every other
    import-time failure — a typo'd requirement is a documentation defect, not a crash.
    """


@dataclass(frozen=True, slots=True)
class _Family:
    """One identifier family, and how the documents spell its number.

    Attributes:
        width: Zero-padding of the numeric part *as the specification writes it*.
            Verification tests run ``V-01…V-48`` and requirements run ``IF-6``; keeping
            the width per-family means the generated matrix is greppable against the
            documents instead of merely internally consistent.
        source: Where the family is defined, for the message a typo produces.
    """

    width: int
    source: str


#: Requirement families — doc 01 §2.4 (envelope), §5.1-5.4 (per instrument), §5.5 (system).
REQUIREMENT_FAMILIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "R-SPAT": "doc 01 §2.4 — spatial resolution",
        "R-TEMP": "doc 01 §2.4 — temporal resolution",
        "R-ACC": "doc 01 §2.4 — accuracy",
        "R-ENV": "doc 01 §2.4 — operating envelope",
        "R-NON": "doc 01 §2.4 — non-intrusiveness",
        "LIF": "doc 01 §5.1 — laser-induced fluorescence",
        "TS": "doc 01 §5.2 — Thomson scattering",
        "OES": "doc 01 §5.3 — optical emission spectroscopy",
        "IF": "doc 01 §5.4 — interferometry",
        "SYS": "doc 01 §5.5 — cross-instrument",
    }
)

#: Evidence families — verification tests, scenario benchmarks and ablations.
EVIDENCE_FAMILIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "V": "verification tests — doc 03 §7, doc 04 §8, doc 06 §7, doc 07 §2",
        "B": "scenario benchmarks — doc 07 §5.1",
        "F": "robustness / ablation matrix — doc 07 §5.2, doc 02 §13",
    }
)

_REQUIREMENT_WIDTHS: Final[Mapping[str, _Family]] = MappingProxyType(
    {prefix: _Family(width=1, source=source) for prefix, source in REQUIREMENT_FAMILIES.items()}
)

_EVIDENCE_WIDTHS: Final[Mapping[str, _Family]] = MappingProxyType(
    {prefix: _Family(width=2, source=source) for prefix, source in EVIDENCE_FAMILIES.items()}
)

#: ``PREFIX-number``, uppercase, with the prefix itself allowed to be hyphenated so that
#: ``R-SPAT-1`` parses as the family ``R-SPAT`` and not as the family ``R``.
_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"([A-Z][A-Z-]*[A-Z]|[A-Z])-([0-9]{1,3})")


def _canonicalise(
    text: str,
    *,
    families: Mapping[str, _Family],
    other: Mapping[str, _Family],
    kind: str,
    opposite: str,
) -> str:
    """Parse one identifier and re-render it the way the specification writes it.

    Args:
        text: The identifier as it appears in the annotation.
        families: The families this side accepts.
        other: The families the *other* side accepts, so that a swapped identifier gets
            an error naming the mistake rather than a generic one.
        kind: What this side is called, for the message.
        opposite: What the other side is called.

    Returns:
        The canonical spelling, so that ``V-1`` and ``V-01`` are one identifier rather
        than two rows in the generated matrix.

    Raises:
        TraceabilityError: If the identifier is malformed, numbered zero, or belongs to
            a family this side does not accept.
    """
    match = _ID_PATTERN.fullmatch(text.strip())
    if match is None:
        raise TraceabilityError(
            f"{text!r} is not a well-formed {kind} id. The shape is PREFIX-NUMBER in "
            f"upper case, as doc 01 §6 writes it: R-SPAT-1, IF-6, V-22, B-03, F-15."
        )

    prefix, digits = match.group(1), match.group(2)
    number = int(digits)
    if number < 1:
        raise TraceabilityError(f"{text!r} is numbered zero; no {kind} in the documents is")

    family = families.get(prefix)
    if family is None:
        if prefix in other:
            raise TraceabilityError(
                f"{text!r} comes from the {opposite} families ({other[prefix].source}) "
                f"and cannot stand where a {kind} id is expected. Filing it here would "
                "put a row in the generated matrix that doc 01 never wrote."
            )
        known = ", ".join(sorted(families))
        raise TraceabilityError(f"{prefix!r} is not a {kind} family. The families are: {known}.")

    return f"{prefix}-{number:0{family.width}d}"


def canonical_requirement_id(text: str) -> RequirementId:
    """Validate a doc 01 requirement identifier and return its canonical spelling.

    Raises:
        TraceabilityError: If ``text`` is malformed or is not a requirement.
    """
    return _canonicalise(
        text,
        families=_REQUIREMENT_WIDTHS,
        other=_EVIDENCE_WIDTHS,
        kind="requirement",
        opposite="evidence",
    )


def canonical_evidence_id(text: str) -> EvidenceId:
    """Validate a verification-test or benchmark identifier and canonicalise it.

    Raises:
        TraceabilityError: If ``text`` is malformed or is not evidence.
    """
    return _canonicalise(
        text,
        families=_EVIDENCE_WIDTHS,
        other=_REQUIREMENT_WIDTHS,
        kind="evidence",
        opposite="a requirement",
    )


@dataclass(frozen=True, slots=True)
class TraceEntry:
    """One annotated implementation — a row of the doc 08 §9 matrix.

    Attributes:
        target: Fully-qualified name of the annotated class or function. Fully qualified
            because two instruments in different plugins will be called ``Instrument``,
            and a matrix that could not tell them apart would be worse than none.
        requirements: What it claims to satisfy, canonicalised and sorted.
        evidence: What verifies that claim, canonicalised and sorted.
    """

    target: str
    requirements: tuple[RequirementId, ...]
    evidence: tuple[EvidenceId, ...]


@dataclass(slots=True)
class _Record:
    """Mutable accumulator behind one :class:`TraceEntry`."""

    requirements: set[str] = field(default_factory=set)
    evidence: set[str] = field(default_factory=set)


#: Every annotation seen in this process, keyed by fully-qualified target name.
_INDEX: Final[dict[str, _Record]] = {}


def _target_key(target: object) -> str:
    """The fully-qualified name an annotation is filed under.

    Classes and functions only. An annotation on an *instance* would file the claim
    against the class of every other instance too, and an annotation on a module-level
    constant has nothing for a reviewer to open.
    """
    if not (isinstance(target, type) or inspect.isfunction(target)):
        raise TraceabilityError(
            f"traceability annotates a class or function; got {type(target).__name__}. "
            "doc 01 §6's chain runs Requirement → implementation → test, and an "
            "implementation is something a reviewer can open."
        )

    return f"{target.__module__}.{target.__qualname__}"


def _record_for(target: object) -> _Record:
    return _INDEX.setdefault(_target_key(target), _Record())


def satisfies[T](*requirements: str) -> Callable[[T], T]:
    """Declare which doc 01 requirements an implementation satisfies — doc 08 §9.

    Repeated application accumulates rather than replaces, because splitting a long claim
    across two decorator lines is normal and silently keeping only the last would drop
    half the matrix.

    Args:
        *requirements: Requirement identifiers as doc 01 writes them, e.g. ``"R-SPAT-1"``.

    Returns:
        A decorator that records the claim and returns its target unchanged.

    Raises:
        TraceabilityError: If no identifier is given, or any is malformed or is not a
            requirement. Raised when the decorator is *built*, so a typo fails at import.
    """
    if not requirements:
        raise TraceabilityError(
            "@satisfies needs at least one requirement id; an empty claim registers a "
            "target that appears in no row of the matrix and is never gated"
        )

    canonical = frozenset(canonical_requirement_id(text) for text in requirements)

    def annotate(target: T) -> T:
        _record_for(target).requirements.update(canonical)
        return target

    return annotate


def verified_by[T](*evidence: str) -> Callable[[T], T]:
    """Declare which tests and benchmarks verify an implementation — doc 08 §9.

    Args:
        *evidence: Verification-test or benchmark identifiers, e.g. ``"V-22"``, ``"B-03"``.

    Returns:
        A decorator that records the evidence and returns its target unchanged.

    Raises:
        TraceabilityError: If no identifier is given, or any is malformed or is not
            evidence.
    """
    if not evidence:
        raise TraceabilityError(
            "@verified_by needs at least one test or benchmark id; an empty annotation "
            "reads as verified and is not"
        )

    canonical = frozenset(canonical_evidence_id(text) for text in evidence)

    def annotate(target: T) -> T:
        _record_for(target).evidence.update(canonical)
        return target

    return annotate


def entry_for(target: object) -> TraceEntry | None:
    """The annotation filed against ``target``, or ``None`` if it has none.

    Looked up by fully-qualified name rather than read off an attribute, so that a
    subclass does **not** inherit its base's claims. An inherited ``@satisfies`` would
    credit every subclass with coverage from tests that never ran against it, which is
    the precise failure mode a generated matrix exists to rule out.
    """
    record = _INDEX.get(_target_key(target))
    if record is None:
        return None

    return TraceEntry(
        target=_target_key(target),
        requirements=tuple(sorted(record.requirements)),
        evidence=tuple(sorted(record.evidence)),
    )


def requirements_of(target: object) -> tuple[RequirementId, ...]:
    """What ``target`` claims to satisfy, sorted. Empty if it is unannotated."""
    entry = entry_for(target)
    return () if entry is None else entry.requirements


def evidence_of(target: object) -> tuple[EvidenceId, ...]:
    """What verifies ``target``, sorted. Empty if it is unannotated."""
    entry = entry_for(target)
    return () if entry is None else entry.evidence


def entries() -> tuple[TraceEntry, ...]:
    """Every annotation this process has seen, in target order.

    Only what has been *imported* is visible. That is inherent to source annotations
    (doc 08 §9) and is why the CI job that generates the matrix imports every package
    rather than walking the filesystem.
    """
    return tuple(
        TraceEntry(
            target=target,
            requirements=tuple(sorted(record.requirements)),
            evidence=tuple(sorted(record.evidence)),
        )
        for target, record in sorted(_INDEX.items())
    )


def traceability_matrix() -> Mapping[RequirementId, tuple[TraceEntry, ...]]:
    """The doc 01 §6 matrix: each requirement, and everything that claims it.

    Read-only and sorted, because the generated document has to diff cleanly between
    commits — a matrix whose row order depended on import order would show spurious
    changes on every unrelated edit.
    """
    matrix: dict[RequirementId, list[TraceEntry]] = {}
    for entry in entries():
        for requirement in entry.requirements:
            matrix.setdefault(requirement, []).append(entry)

    return MappingProxyType(
        {requirement: tuple(matrix[requirement]) for requirement in sorted(matrix)}
    )


def unverified_claims() -> tuple[TraceEntry, ...]:
    """Implementations that claim a requirement with nothing verifying it.

    Doc 08 §9: "a ``@satisfies`` with no ``@verified_by`` fails CI". Returns the offending
    entries rather than a boolean so the CI job can name them; a gate that can only say
    "no" gets worked around instead of fixed.
    """
    return tuple(entry for entry in entries() if entry.requirements and not entry.evidence)


def uncovered_requirements(catalogue: Iterable[str]) -> tuple[RequirementId, ...]:
    """Requirements in ``catalogue`` that no imported implementation claims.

    The other half of the doc 08 §9 gate. The catalogue is an argument because the list
    of requirements lives in doc 01 §6, not in this package — putting it here would mean
    editing ``vpl-core`` every time a requirement is added, which is the coupling
    doc 08 §1 principle 3 exists to prevent.

    Args:
        catalogue: Every requirement identifier the project has declared.

    Returns:
        Those with no ``@satisfies``, canonicalised and sorted.

    Raises:
        TraceabilityError: If the catalogue itself contains a malformed identifier. A
            typo there would silently exempt a real requirement from the gate.
    """
    claimed = set(traceability_matrix())
    declared = {canonical_requirement_id(text) for text in catalogue}
    return tuple(sorted(declared - claimed))


def clear_registry() -> None:
    """Forget every annotation seen so far.

    For tests. Nothing in a real run should ever want this: the index is built by the
    import walk and consumed once, when the matrix is generated.
    """
    _INDEX.clear()
