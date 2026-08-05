"""The experiment manifest engine and the ``vpl`` command line — doc 08 §6, WBS 1.3.

    One file, one experiment, fully reproducible. — doc 08 §6

This package is what doc 00 E3 is measured against: *"every experiment is defined by a
single declarative manifest and is bit-for-bit reproducible given the manifest, the code
version and the seed"*. Gate **G-1.3** (doc 11 §2) is that sentence checked, and
:func:`~vpl.experiment.run.reproduce.reproduce` is the check.

Four layers, none of which knows more than it needs to:

============================================ =================================================
:mod:`~vpl.experiment.manifest`              doc 08 §6's schema, its digest and its loader
:mod:`~vpl.experiment.solvers`               the contract a ``forward.solver`` satisfies
:mod:`~vpl.experiment.run`                   the run directory, the engine and G-1.3
:mod:`~vpl.experiment.compare`               the structured diff of two runs
============================================ =================================================

with :mod:`~vpl.experiment.digest` underneath, which is where "bit-identical" is defined
for artifacts that are required to embed the time they were written.

The configuration substrate is **OmegaConf, with Hydra reserved for the doc 10 §6 sweep
layer** — a deviation from doc 08 §2's build/buy table, recorded in
``docs/adr/ADR-008-manifest-substrate.md``.
"""

from vpl.experiment.compare import (
    MetricDifference,
    RunComparison,
    ValueDifference,
    compare_runs,
)
from vpl.experiment.digest import (
    EXECUTION_ONLY_FIELDS,
    PROVENANCE_ONLY_FIELDS,
    UnknownArtifactFormatError,
    artifact_digest,
    run_content_digest,
)
from vpl.experiment.manifest import (
    MAXWELLIAN_KAPPA,
    ArtifactRequest,
    BiasMode,
    BiasSpec,
    CalibrationMode,
    ExperimentSpec,
    ForwardSpec,
    InstrumentSpec,
    InverseSpec,
    Manifest,
    ManifestConsistencyError,
    NoiseSpec,
    OutputSpec,
    PlasmaSpec,
    ResolvedPlasma,
    UnknownKeyError,
    ValidationSpec,
    load_manifest,
    manifest_from_document,
    resolve_plasma,
)
from vpl.experiment.run import (
    IMPLEMENTED_ARTIFACTS,
    IndexEntry,
    ReproductionResult,
    RunDirectory,
    RunNotFoundError,
    RunRecord,
    RunStatus,
    RunStore,
    StageNotImplementedError,
    execute,
    reproduce,
    run_id_for,
)
from vpl.experiment.solvers import ManifestSolver

__all__ = [
    "EXECUTION_ONLY_FIELDS",
    "IMPLEMENTED_ARTIFACTS",
    "MAXWELLIAN_KAPPA",
    "PROVENANCE_ONLY_FIELDS",
    "ArtifactRequest",
    "BiasMode",
    "BiasSpec",
    "CalibrationMode",
    "ExperimentSpec",
    "ForwardSpec",
    "IndexEntry",
    "InstrumentSpec",
    "InverseSpec",
    "Manifest",
    "ManifestConsistencyError",
    "ManifestSolver",
    "MetricDifference",
    "NoiseSpec",
    "OutputSpec",
    "PlasmaSpec",
    "ReproductionResult",
    "ResolvedPlasma",
    "RunComparison",
    "RunDirectory",
    "RunNotFoundError",
    "RunRecord",
    "RunStatus",
    "RunStore",
    "StageNotImplementedError",
    "UnknownArtifactFormatError",
    "UnknownKeyError",
    "ValidationSpec",
    "ValueDifference",
    "__version__",
    "artifact_digest",
    "compare_runs",
    "execute",
    "load_manifest",
    "manifest_from_document",
    "reproduce",
    "resolve_plasma",
    "run_content_digest",
    "run_id_for",
]

__version__ = "0.1.0"
