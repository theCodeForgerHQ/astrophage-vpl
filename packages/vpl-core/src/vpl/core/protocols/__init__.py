"""The core contracts — doc 08 §4.

Doc 08 §1 principle 1: "Every solver, instrument and inference engine implements a
``Protocol``. **The core depends only on protocols.**" This package is that dependency,
and it is the whole of it: four protocols, plus the supporting types doc 08 §4 names in
their signatures, plus the traceability annotations of doc 08 §9.

Nothing here computes anything. A protocol module that grew a default implementation
would make the core depend on one particular way of satisfying the contract, which is the
coupling doc 08 §1 principle 3 forbids and the reason plugins can be installed rather than
merged.

## The four contracts

- :class:`ForwardSolver` — ``F1``, at one of the four fidelity levels of doc 03 §1.
- :class:`Instrument` — one measurement channel, virtual or real (doc 00 E2).
- :class:`InverseEngine` — one of the six engines of doc 05 §5.
- :class:`NoiseModel` — one individually switchable source from doc 04 §7.2.

Each is ``@runtime_checkable``, which buys exactly one thing: a plugin loaded by name
from an entry point (doc 08 §10) can be rejected at load time rather than failing with an
``AttributeError`` an hour into a sweep. It does **not** check signatures — ``isinstance``
against a ``Protocol`` compares method names only — so the signature half of each contract
is enforced by ``mypy --strict`` against the plugin's own source (doc 08 §12). Structural
at runtime, nominal at type-check time, and the two are complementary rather than
redundant.

## Module layout

Doc 08 §1 principle 7 asks for 200 to 400 line modules. The four protocols are one module
each; the types shared between them are grouped by the concern they serve rather than by
which protocol mentions them first — :mod:`~vpl.core.protocols.metadata` is doc 00 C2's
citation requirement, :mod:`~vpl.core.protocols.config` is the seam the doc 08 §6 manifest
engine will fill, and :mod:`~vpl.core.protocols.traceability` is doc 08 §9.
"""

from vpl.core.protocols.config import (
    Config,
    ConfigValue,
    InstrumentConfig,
    InverseConfig,
    SolverConfig,
)
from vpl.core.protocols.forward import (
    CostBasis,
    CostEstimate,
    Device,
    ForwardSolver,
    IonEnergyFlux,
)
from vpl.core.protocols.instrument import (
    Calibration,
    CalibrationReference,
    CalibrationSet,
    Instrument,
    LogProb,
)
from vpl.core.protocols.inverse import (
    ForwardModel,
    Identifiability,
    IdentifiabilityReport,
    InverseEngine,
    classify_by_condition_number,
)
from vpl.core.protocols.metadata import (
    Citation,
    DetectionFloor,
    InstrumentMetadata,
    SolverMetadata,
)
from vpl.core.protocols.noise import NoiseModel, Signal, SignalDomain
from vpl.core.protocols.traceability import (
    EVIDENCE_FAMILIES,
    REQUIREMENT_FAMILIES,
    EvidenceId,
    RequirementId,
    TraceabilityError,
    TraceEntry,
    canonical_evidence_id,
    canonical_requirement_id,
    clear_registry,
    entries,
    entry_for,
    evidence_of,
    requirements_of,
    satisfies,
    traceability_matrix,
    uncovered_requirements,
    unverified_claims,
    verified_by,
)

__all__ = [
    "EVIDENCE_FAMILIES",
    "REQUIREMENT_FAMILIES",
    "Calibration",
    "CalibrationReference",
    "CalibrationSet",
    "Citation",
    "Config",
    "ConfigValue",
    "CostBasis",
    "CostEstimate",
    "DetectionFloor",
    "Device",
    "EvidenceId",
    "ForwardModel",
    "ForwardSolver",
    "Identifiability",
    "IdentifiabilityReport",
    "Instrument",
    "InstrumentConfig",
    "InstrumentMetadata",
    "InverseConfig",
    "InverseEngine",
    "IonEnergyFlux",
    "LogProb",
    "NoiseModel",
    "RequirementId",
    "Signal",
    "SignalDomain",
    "SolverConfig",
    "SolverMetadata",
    "TraceEntry",
    "TraceabilityError",
    "canonical_evidence_id",
    "canonical_requirement_id",
    "classify_by_condition_number",
    "clear_registry",
    "entries",
    "entry_for",
    "evidence_of",
    "requirements_of",
    "satisfies",
    "traceability_matrix",
    "uncovered_requirements",
    "unverified_claims",
    "verified_by",
]
