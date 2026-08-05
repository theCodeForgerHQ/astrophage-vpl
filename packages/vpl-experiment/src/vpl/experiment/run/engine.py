"""`vpl run` — manifest to provenanced artifacts, doc 08 §6.

The pipeline doc 08 §3 names is ``manifest → pipeline → artifacts``. Of its stages only
the forward one exists (doc 11 §2: WBS 1.5 landed, 2.x has not), so this module does two
things: it runs that stage, and it **refuses the others by name**.

## Why refusing is a feature

doc 08 §6's own example manifest asks for four instruments, thirteen noise sources, a NUTS
inversion and four artifacts nothing can write yet. Three responses were available:

1. Reject the manifest at load. Wrong: doc 13 §5 keeps manifests forever, so one has to be
   *readable* long before it is runnable.
2. Run the forward stage and ignore the rest. Wrong, and the worst of the three: it
   produces a run directory, a provenance block and a plausible artifact that answers a
   different question from the one the manifest asked.
3. Run nothing and say exactly which stages are missing.

Option 3 is what :class:`StageNotImplementedError` is, and the refusal happens **before
the run directory is created** — an empty directory left behind by a refused run would
show up in the index as a run that did nothing.

## Failure is recorded, not dropped

doc 10 §6: "a quarantined case is never silently dropped from statistics". Once execution
has begun, every exit path writes the record. A run that fails leaves ``status: failed``,
``quarantined_cases: 1`` and the diagnostic; the exception still propagates, so a caller
cannot mistake failure for success, but the archive never loses the case.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from vpl.core.protocols.forward import IonEnergyFlux
from vpl.core.provenance import Provenance
from vpl.core.registry import PluginGroup, load
from vpl.core.state import PlasmaParams, PlasmaState
from vpl.core.storage import MetricRecord, write_metrics, write_plasma_state
from vpl.core.units import magnitude_in
from vpl.experiment.manifest import ArtifactRequest, Manifest, ResolvedPlasma, resolve_plasma
from vpl.experiment.run.record import RunRecord
from vpl.experiment.run.store import RunDirectory, RunStore, run_id_for
from vpl.experiment.solvers import ManifestSolver

__all__ = [
    "IMPLEMENTED_ARTIFACTS",
    "METRICS_FILENAME",
    "PLASMA_STATE_FILENAME",
    "StageNotImplementedError",
    "execute",
]

#: The artifacts the framework can produce today.
IMPLEMENTED_ARTIFACTS: Final[frozenset[ArtifactRequest]] = frozenset(
    {ArtifactRequest.PLASMA_STATE, ArtifactRequest.METRICS}
)

PLASMA_STATE_FILENAME: Final[str] = "plasma_state.h5"
METRICS_FILENAME: Final[str] = "metrics.parquet"

#: Where the flux functional is evaluated — the wall, doc 02 §2.
_WALL_POSITION_M: Final[float] = 0.0

#: doc 03 §2.3's ``Gamma_E`` and doc 01 §1.2's decomposition of it, as stored units.
_ENERGY_FLUX_UNITS: Final[str] = "W/m**2"
_PARTICLE_FLUX_UNITS: Final[str] = "1/(m**2*s)"
_IMPACT_ENERGY_UNITS: Final[str] = "eV"


class StageNotImplementedError(NotImplementedError):
    """The manifest asks for a pipeline stage that does not exist yet.

    A subclass of :class:`NotImplementedError` so that it reads as what it is — a gap in
    the framework, not a mistake in the manifest. The message names every missing stage at
    once rather than the first: someone porting doc 08 §6's manifest forward wants the
    whole list, not four rounds of the same error.
    """


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """What the forward stage produced."""

    state: PlasmaState
    flux: IonEnergyFlux
    solver_name: str
    solver_version: str


def _missing_stages(manifest: Manifest) -> list[str]:
    """Every stage the manifest asks for that the framework cannot run."""
    missing: list[str] = []

    if any(instrument.enabled for instrument in manifest.instruments):
        # A *disabled* channel is not a stage. doc 07 §5.2's ablation matrix switches
        # exactly this flag, and a manifest that ablates every channel is a legitimate
        # forward-only run rather than a request for the instrument layer.
        missing.append("instruments (doc 04 — WBS 2.4-2.10)")
    if manifest.noise is not None:
        missing.append("noise (doc 04 §7.2 — WBS 2.10)")
    if manifest.inverse is not None:
        missing.append("inverse (doc 05 — WBS 3.x)")
    if manifest.validation is not None:
        missing.append("validation (doc 07 — WBS 5.x)")

    unimplemented = sorted(
        request.value
        for request in manifest.outputs.artifacts
        if request not in IMPLEMENTED_ARTIFACTS
    )
    if unimplemented:
        missing.append(f"outputs.artifacts: {', '.join(unimplemented)}")
    if manifest.outputs.figures:
        missing.append("outputs.figures (doc 13 §3 — WBS 6.x)")
    if manifest.outputs.report:
        missing.append("outputs.report (doc 13 §4 — WBS 6.x)")

    return missing


def _check_stages(manifest: Manifest) -> None:
    missing = _missing_stages(manifest)
    if not missing:
        return
    raise StageNotImplementedError(
        f"manifest {manifest.experiment.name!r} asks for stages this build does not have: "
        + "; ".join(missing)
        + ". The manifest is valid and archivable (doc 13 §5); it simply cannot run yet. "
        "Running the forward stage alone and ignoring the rest would produce an artifact "
        "answering a different question from the one the manifest asked."
    )


def _resolve_solver(name: str) -> ManifestSolver:
    """Resolve ``forward.solver`` through the entry-point registry — doc 08 §10.

    The instance is checked against :class:`ManifestSolver` at load, which verifies the
    method names and nothing more. That is the right amount: it turns a plugin that is not
    a solver into a loud failure here, before a run directory exists, instead of an
    ``AttributeError`` after one does.
    """
    plugin = load(PluginGroup.SOLVERS, name)
    if not callable(plugin):
        raise TypeError(
            f"forward.solver {name!r} resolves to {plugin!r}, which is not constructible. "
            "doc 08 §10 entry points name the solver *class*."
        )

    solver = plugin()
    if not isinstance(solver, ManifestSolver):
        raise TypeError(
            f"forward.solver {name!r} resolves to {type(solver).__name__}, which does not "
            "implement the doc 08 §4 ForwardSolver contract: it is missing one of "
            "configure, solve, flux, fidelity, metadata."
        )
    return solver


def _run_forward(manifest: Manifest, params: PlasmaParams, solver: ManifestSolver) -> ForwardResult:
    """Configure, solve and apply the doc 03 §6 flux functional."""
    solver.configure(manifest.forward.config)

    # ``None``: doc 08 §6's forward block carries no time grid, and L0/L1 are steady.
    # See vpl.experiment.solvers for why the contract had to widen to allow it.
    state = solver.solve(params, None)
    flux = solver.flux(state, _WALL_POSITION_M)
    metadata = solver.metadata()

    return ForwardResult(
        state=state,
        flux=flux,
        solver_name=metadata.name,
        solver_version=metadata.version,
    )


def _metrics(flux: IonEnergyFlux) -> list[MetricRecord]:
    """The doc 01 §1.2 decomposition, as rows of the doc 07 §7 regression store.

    All three come from the flux functional the solver applied; nothing is recomputed
    here. ``<E_i> = Gamma_E / Gamma_i`` is stored beside its two factors because doc 01
    §1.2 calls the split "the information decomposition of the problem" — the two are
    constrained by different physics and observed by different instruments, and a store
    that kept only the product could not regress them separately.
    """
    return [
        MetricRecord(
            name="gamma_E",
            value=float(magnitude_in(flux.energy_flux, _ENERGY_FLUX_UNITS)),
            units=_ENERGY_FLUX_UNITS,
        ),
        MetricRecord(
            name="gamma_i",
            value=float(magnitude_in(flux.particle_flux, _PARTICLE_FLUX_UNITS)),
            units=_PARTICLE_FLUX_UNITS,
        ),
        MetricRecord(
            name="mean_impact_energy",
            value=float(magnitude_in(flux.mean_impact_energy, _IMPACT_ENERGY_UNITS)),
            units=_IMPACT_ENERGY_UNITS,
        ),
    ]


def _write_artifacts(
    run: RunDirectory,
    *,
    requested: Sequence[ArtifactRequest],
    result: ForwardResult,
    provenance: Provenance,
) -> dict[str, str]:
    """Write what the manifest asked for, and return name-to-file for the record."""
    written: dict[str, str] = {}

    if ArtifactRequest.PLASMA_STATE in requested:
        write_plasma_state(
            run.artifacts_path / PLASMA_STATE_FILENAME, result.state, provenance=provenance
        )
        written[ArtifactRequest.PLASMA_STATE.value] = PLASMA_STATE_FILENAME

    if ArtifactRequest.METRICS in requested:
        write_metrics(
            run.artifacts_path / METRICS_FILENAME, _metrics(result.flux), provenance=provenance
        )
        written[ArtifactRequest.METRICS.value] = METRICS_FILENAME

    return written


def execute(
    manifest: Manifest,
    *,
    store: RunStore,
    force: bool = False,
    run_id: str | None = None,
) -> RunDirectory:
    """Run a manifest and write its artifacts — doc 08 §6's ``vpl run``.

    The order of operations is the load-bearing part, and each step is where it is for a
    reason:

    1. **Refuse missing stages**, before anything exists on disk.
    2. **Resolve the plasma block and the solver**, before anything exists on disk. Both
       can fail on a manifest that is well-formed but wrong — an unregistered gas, an
       uninstalled solver — and neither failure should leave an empty run behind.
    3. **Capture provenance**, which needs the solver's version, so it follows resolution.
    4. **Create the directory and write the manifest, the sidecar and a ``running``
       record**, so that an interruption from here on is discoverable (doc 10 §6).
    5. **Execute**, and write the record either way (doc 10 §6, "never silently dropped").

    Args:
        manifest: The validated manifest.
        store: Where the run directory goes.
        force: Overwrite an existing run of the same manifest. Off by default; see
            :meth:`~vpl.experiment.run.store.RunStore.create`.
        run_id: Override the derived identity. Used by ``vpl reproduce``, which
            re-executes into a scratch directory under the *original* run's name.

    Returns:
        The run directory, holding the manifest, the provenance sidecar, the doc 13 §2
        record and the artifacts.

    Raises:
        StageNotImplementedError: If the manifest asks for a stage that does not exist.
        FileExistsError: If the run already exists and ``force`` is false.
        PluginNotFoundError: If nothing declares ``forward.solver``.
    """
    _check_stages(manifest)

    started = datetime.now(UTC)
    resolved: ResolvedPlasma = resolve_plasma(manifest.plasma)
    solver = _resolve_solver(manifest.forward.solver)
    metadata = solver.metadata()

    provenance = Provenance.capture(
        manifest_sha256=manifest.sha256,
        seed=manifest.experiment.seed,
        tier=manifest.experiment.tier,
        solver_versions={metadata.name: metadata.version},
    )

    identity = run_id_for(manifest, started) if run_id is None else run_id
    run = store.create(identity, force=force)
    record = RunRecord.opened(
        run_id=identity, provenance=provenance, parameter_sources=resolved.sources
    )
    run.write_manifest(manifest)
    run.write_provenance(provenance)
    run.write_record(record)

    clock = time.perf_counter()
    try:
        result = _run_forward(manifest, resolved.params, solver)
        artifacts = _write_artifacts(
            run, requested=manifest.outputs.artifacts, result=result, provenance=provenance
        )
    except Exception as exc:
        run.write_record(
            record.quarantined(
                duration_s=time.perf_counter() - clock,
                failure=f"{type(exc).__name__}: {exc}",
            )
        )
        store.reindex()
        raise

    run.write_record(record.finished(duration_s=time.perf_counter() - clock, artifacts=artifacts))
    store.reindex()
    return run
