"""The experiment manifest — doc 08 §6, block by block.

One file, one experiment. The types here are the whole of doc 08 §6's example manifest
made checkable: every key it writes is expressible, and nothing else is.

## Blocks for stages that do not exist yet

``instruments``, ``noise``, ``inverse`` and ``validation`` parse and validate in full even
though nothing in the framework consumes them yet. That is deliberate and it is not the
same as ignoring them. A manifest is the archival record of an experiment (doc 13 §5,
"forever, in Git"), so it has to be *readable* long before it is runnable, and a schema
that only grew a block when its stage landed would make every manifest written before that
point silently invalid afterwards. What the run engine does with a block it cannot execute
is refuse loudly — see :class:`~vpl.experiment.run.engine.StageNotImplementedError`.

## The mandatory mismatch

doc 05 §7.1 makes the forward/inverse mismatch **structural rather than a matter of good
intentions**, and doc 05 §7.2 makes reporting T1 as if it were T2 a project defect. Both
are checkable from the manifest alone, so both are checked here:

- a **T2** manifest whose ``inverse.model`` equals its ``forward.solver``, or whose two
  meshes agree, is an inverse crime wearing an honest tier's label;
- a **T0/T1** manifest is defined by using the *same* model, so one that mismatches is
  claiming a pessimism it has not earned.

Neither check can be satisfied by accident, and neither costs anything to state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import yaml

from vpl.core.protocols.config import InstrumentConfig, InverseConfig, SolverConfig
from vpl.core.provenance import ManifestValue, Tier, manifest_sha256
from vpl.core.units import Quantity
from vpl.experiment.manifest.parse import (
    UnknownKeyError,
    block,
    check_keys,
    flag,
    frozen,
    integer,
    member,
    number,
    plain,
    quantity,
    required,
    strings,
    text,
)

__all__ = [
    "ArtifactRequest",
    "BiasMode",
    "BiasSpec",
    "CalibrationMode",
    "ExperimentSpec",
    "ForwardSpec",
    "InstrumentSpec",
    "InverseSpec",
    "Manifest",
    "ManifestConsistencyError",
    "NoiseSpec",
    "OutputSpec",
    "PlasmaSpec",
    "UnknownKeyError",
    "ValidationSpec",
    "manifest_from_document",
]


class ManifestConsistencyError(ValueError):
    """Two blocks of one manifest contradict each other.

    Separate from :class:`~vpl.experiment.manifest.parse.UnknownKeyError` because the fix
    is different in kind: every block is individually well-formed and the manifest still
    describes an experiment that would not mean what it claims.
    """


class BiasMode(StrEnum):
    """How the electrode is driven — doc 02 §3.3."""

    DC = "dc"
    RF = "rf"


class CalibrationMode(StrEnum):
    """Which calibration the pipeline works with — doc 04 §7.3.

    doc 08 §6 writes ``calibration: estimated`` and comments "NOT 'true'". Both are
    expressible, because doc 07 §5.2's ablation matrix needs to be able to *ask* what the
    true response would have bought — but the honest tier uses the estimate, and a
    manifest has to say which it used rather than leaving it to a default.
    """

    ESTIMATED = "estimated"
    TRUE = "true"


class ArtifactRequest(StrEnum):
    """What ``outputs.artifacts`` may ask for — doc 08 §6, doc 08 §7.

    A closed set, unlike the figure list. doc 08 §7 fixes one storage format per artifact
    kind, so an artifact this framework has no writer for is not a thing a manifest can
    coherently request; a free-form string would be a request that silently produced
    nothing.
    """

    #: Written by the forward stage. HDF5 (doc 08 §7).
    PLASMA_STATE = "plasma_state"
    #: Written by the instrument stage. HDF5, one group per instrument.
    MEASUREMENTS = "measurements"
    #: Scalars for the doc 07 §7 regression store. Parquet.
    METRICS = "metrics"
    #: Written by the inverse stage. Zarr.
    POSTERIOR = "posterior"
    #: Sampler diagnostics — doc 05 §5.
    DIAGNOSTICS = "diagnostics"
    #: FIM, profile likelihood and Sobol indices — doc 05 §9.
    IDENTIFIABILITY = "identifiability"
    #: The doc 06 §4 term-by-term budget.
    ERROR_BUDGET = "error_budget"


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """``experiment:`` — what this run is and what claim it supports.

    Attributes:
        name: Short identifier. It becomes part of the run identity (doc 13 §2).
        description: Prose for the report. Never load-bearing.
        tier: Which of doc 05 §7.2's three claims. Never defaulted — doc 05 §7.2 makes
            reporting one tier as another a project defect, and a default would be the
            flattering one.
        seed: The single root seed. Every stream derives from it (doc 10 §5).
    """

    name: str
    description: str
    tier: Tier
    seed: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment.name is what the run directory is named after")
        if self.seed < 0:
            raise ValueError(
                f"experiment.seed indexes a generator and cannot be negative, got {self.seed}"
            )


@dataclass(frozen=True, slots=True)
class BiasSpec:
    """``plasma.bias:`` — doc 08 §6's ``{mode: dc, value: -250.0, units: V}``.

    Attributes:
        mode: DC or RF.
        value: The **signed** applied electrode potential, as
            :class:`~vpl.core.state.PlasmaParams` defines it. Negative for a biased
            electrode.
        frequency: The drive frequency for an RF bias, or ``None`` to take the registered
            one. Refused outright for a DC bias.
        phase: Phase offset of the bias waveform in radians, or ``None`` where the
            manifest did not state one. ``None`` rather than a defaulted zero so that
            "the origin, by convention" and "zero, deliberately" stay distinguishable in
            the run record (doc 00 C4).
    """

    mode: BiasMode
    value: Quantity
    frequency: Quantity | None = None
    phase: float | None = None

    def __post_init__(self) -> None:
        if self.mode is BiasMode.DC and self.frequency is not None:
            raise ValueError(
                "plasma.bias: a dc bias has no drive frequency. Either drop the "
                "frequency or set mode: rf — silently ignoring it would make the "
                "manifest disagree with the run."
            )


@dataclass(frozen=True, slots=True)
class PlasmaSpec:
    """``plasma:`` — the control parameters of doc 05 §2.1 as a manifest writes them.

    Field names follow :class:`~vpl.core.state.PlasmaParams` rather than doc 08 §6's
    shorter manifest spellings (``n0``, ``Te``); the correspondence is fixed once, in
    :data:`_PLASMA_KEYS`, so that a reader of either can find the other.

    Only the five keys doc 08 §6 writes are required. The rest resolve from the parameter
    registry — see :mod:`vpl.experiment.manifest.plasma`, which also records where each
    one came from.
    """

    gas: str
    pressure: Quantity
    n_0: Quantity
    T_e: Quantity
    bias: BiasSpec
    T_i: Quantity | None = None
    T_g: Quantity | None = None
    gamma_se: float | None = None
    kappa: float | None = None


@dataclass(frozen=True, slots=True)
class ForwardSpec:
    """``forward:`` — the truth generator.

    Attributes:
        solver: The dotted name the plugin registry resolves (doc 08 §10).
        config: Every other key in the block, handed to ``ForwardSolver.configure``.

    The block is deliberately **open** below ``solver``: ``mesh``, ``n_ppc`` and whatever
    a third-party solver needs are that solver's vocabulary, and a closed schema here
    would mean editing the core every time a plugin gained an option — the coupling
    doc 08 §1 principle 3 forbids. The unknown-key rule still applies, one level down:
    ``configure`` is where a solver refuses a key it does not own, because it is the only
    thing that knows its own keys.
    """

    solver: str
    config: SolverConfig

    def __post_init__(self) -> None:
        if not self.solver.strip():
            raise ValueError("forward.solver names the plugin to resolve and cannot be empty")


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """One entry of ``instruments:`` — doc 08 §6.

    Attributes:
        id: The channel key. Becomes the artifact group name (doc 08 §7).
        enabled: Whether this channel contributes. doc 07 §5.2's ablation matrix switches
            exactly this.
        config: Path to the channel's own configuration file, relative to the manifest.
        options: Any further keys, handed to ``Instrument.configure``.
    """

    id: str
    enabled: bool
    config: Path | None = None
    options: InstrumentConfig | None = None


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """``noise:`` — doc 04 §7.2's eighteen sources, individually switchable.

    Attributes:
        enabled_sources: Source identifiers, in the order the manifest wrote them.
        calibration: Which calibration the pipeline uses. Required, because doc 04 §7.3
            makes the difference between the two the whole point of the honest tier.
    """

    enabled_sources: tuple[str, ...]
    calibration: CalibrationMode


@dataclass(frozen=True, slots=True)
class InverseSpec:
    """``inverse:`` — doc 05, and half of the doc 05 §7.1 mismatch."""

    model: str
    engine: str
    draws: int
    chains: int
    mesh: Mapping[str, ManifestValue]
    parameters: Mapping[str, ManifestValue]
    config: InverseConfig | None = None

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise ValueError(f"inverse.draws must be at least one, got {self.draws}")
        if self.chains < 1:
            # doc 05 §5 diagnoses convergence with R-hat, which is defined across chains;
            # a single chain cannot be checked against anything.
            raise ValueError(
                f"inverse.chains must be at least one, got {self.chains}. doc 05 §5 gates "
                "on R-hat, which needs more than one chain to mean anything."
            )


@dataclass(frozen=True, slots=True)
class ValidationSpec:
    """``validation:`` — doc 07."""

    seal_truth: bool
    metrics: tuple[str, ...]
    n_repeats: int

    def __post_init__(self) -> None:
        if self.n_repeats < 1:
            raise ValueError(f"validation.n_repeats must be at least one, got {self.n_repeats}")


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """``outputs:`` — doc 08 §6, doc 13 §3."""

    artifacts: tuple[ArtifactRequest, ...]
    figures: tuple[str, ...] = ()
    report: bool = False


# ── the manifest ────────────────────────────────────────────────────────────────

#: doc 08 §6's manifest spelling for each :class:`PlasmaSpec` field.
#:
#: The document uses the symbols a plasma physicist writes (``n0``, ``Te``); the code uses
#: the names :class:`~vpl.core.state.PlasmaParams` uses. One table rather than two sets of
#: field names keeps the two from drifting.
_PLASMA_KEYS: Final[Mapping[str, str]] = {
    "n0": "n_0",
    "Te": "T_e",
    "Ti": "T_i",
    "Tg": "T_g",
}

_TOP_LEVEL_REQUIRED: Final[tuple[str, ...]] = ("experiment", "plasma", "forward", "outputs")
_TOP_LEVEL_OPTIONAL: Final[tuple[str, ...]] = ("instruments", "noise", "inverse", "validation")


@dataclass(frozen=True, slots=True)
class Manifest:
    """One experiment, fully specified — doc 08 §6.

    Attributes:
        document: The loaded manifest, deep-frozen. This — not the parsed blocks — is what
            :attr:`sha256` is taken over; see the property for why.
        experiment: Identity, tier and root seed.
        plasma: Control parameters as written.
        forward: The truth generator.
        outputs: What to produce.
        instruments: Channels, in manifest order. Empty when the block is absent.
        noise: Noise configuration, or ``None``.
        inverse: Inference configuration, or ``None``.
        validation: Validation configuration, or ``None``.
    """

    document: Mapping[str, ManifestValue]
    experiment: ExperimentSpec
    plasma: PlasmaSpec
    forward: ForwardSpec
    outputs: OutputSpec
    instruments: tuple[InstrumentSpec, ...] = ()
    noise: NoiseSpec | None = None
    inverse: InverseSpec | None = None
    validation: ValidationSpec | None = None

    @property
    def sha256(self) -> str:
        """The run's identity — doc 00 E3, doc 08 §7.

        Taken over the **document**, through
        :func:`~vpl.core.provenance.manifest_sha256`, and not over the parsed blocks.
        Three consequences, all of them wanted:

        - Key order and comments do not change it, because the digest is over the loaded
          structure and JSON sorts keys.
        - Sequence order *does*, because doc 08 §6 orders instruments and noise sources
          meaningfully.
        - A manifest that leans on a registry default and one that pins the same value
          explicitly are **different manifests**. They are: the first tracks the registry
          and the second does not, so they are not the same experiment specification even
          when today they resolve to the same numbers. The resolved values are recorded
          separately, in the run record, where they can be read without being confused for
          the specification.
        """
        return manifest_sha256(self.as_document())

    def as_document(self) -> dict[str, ManifestValue]:
        """A plain, mutable copy of the loaded manifest.

        A copy because a caller that edits what it was handed must not be editing the
        manifest a run has already been provenanced against.
        """
        return cast("dict[str, ManifestValue]", plain(self.document))

    def to_yaml(self) -> str:
        """The archived form — doc 08 §7 ("Manifests, provenance | YAML + JSON sidecar").

        Round-trips through :func:`~vpl.experiment.manifest.load.load_manifest` to an
        equal digest, which is what makes ``vpl reproduce`` re-execution rather than
        replay. Comments are not preserved; they are not part of the experiment, and the
        digest already says so.
        """
        return yaml.safe_dump(
            self.as_document(), sort_keys=True, default_flow_style=False, allow_unicode=True
        )

    def __repr__(self) -> str:
        return (
            f"Manifest({self.experiment.name!r}, {self.experiment.tier}, "
            f"solver={self.forward.solver!r}, sha256={self.sha256[:8]})"
        )


# ── parsing ─────────────────────────────────────────────────────────────────────


def _experiment(data: Mapping[str, ManifestValue]) -> ExperimentSpec:
    where = "experiment"
    check_keys(
        data, required_keys=("name", "tier", "seed"), optional_keys=("description",), where=where
    )
    return ExperimentSpec(
        name=text(required(data, "name", where=where), where=f"{where}.name"),
        description=text(data.get("description", ""), where=f"{where}.description"),
        tier=member(required(data, "tier", where=where), Tier, where=f"{where}.tier"),
        seed=integer(required(data, "seed", where=where), where=f"{where}.seed"),
    )


def _bias(data: Mapping[str, ManifestValue]) -> BiasSpec:
    where = "plasma.bias"
    check_keys(
        data,
        required_keys=("mode", "value", "units"),
        optional_keys=("frequency", "phase"),
        where=where,
    )
    raw_frequency = data.get("frequency")
    raw_phase = data.get("phase")
    return BiasSpec(
        mode=member(required(data, "mode", where=where), BiasMode, where=f"{where}.mode"),
        # The potential is written flat — `{mode: dc, value: -250.0, units: V}` — rather
        # than as a nested quantity, because that is how doc 08 §6 writes it.
        value=quantity({"value": data["value"], "units": data["units"]}, where=where),
        frequency=(
            None if raw_frequency is None else quantity(raw_frequency, where=f"{where}.frequency")
        ),
        phase=None if raw_phase is None else number(raw_phase, where=f"{where}.phase"),
    )


def _plasma(data: Mapping[str, ManifestValue]) -> PlasmaSpec:
    where = "plasma"
    check_keys(
        data,
        required_keys=("gas", "pressure", "n0", "Te", "bias"),
        optional_keys=("Ti", "Tg", "gamma_se", "kappa"),
        where=where,
    )

    optional: dict[str, Quantity | float | None] = {}
    for manifest_key in ("Ti", "Tg"):
        raw = data.get(manifest_key)
        optional[_PLASMA_KEYS[manifest_key]] = (
            None if raw is None else quantity(raw, where=f"{where}.{manifest_key}")
        )
    for manifest_key in ("gamma_se", "kappa"):
        raw = data.get(manifest_key)
        optional[manifest_key] = (
            None if raw is None else number(raw, where=f"{where}.{manifest_key}")
        )

    return PlasmaSpec(
        gas=text(required(data, "gas", where=where), where=f"{where}.gas"),
        pressure=quantity(required(data, "pressure", where=where), where=f"{where}.pressure"),
        n_0=quantity(required(data, "n0", where=where), where=f"{where}.n0"),
        T_e=quantity(required(data, "Te", where=where), where=f"{where}.Te"),
        bias=_bias(block(required(data, "bias", where=where), where=f"{where}.bias")),
        **optional,  # type: ignore[arg-type]
    )


def _forward(data: Mapping[str, ManifestValue]) -> ForwardSpec:
    where = "forward"
    if "solver" not in data:
        raise ValueError(f"{where}: missing required key solver")
    rest = {key: value for key, value in data.items() if key != "solver"}
    return ForwardSpec(
        solver=text(data["solver"], where=f"{where}.solver"),
        config=SolverConfig(values=rest),
    )


def _instruments(value: ManifestValue) -> tuple[InstrumentSpec, ...]:
    where = "instruments"
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{where}: expected a list of channels, got {type(value).__name__}")

    entries: list[InstrumentSpec] = []
    for index, raw in enumerate(value):
        entry_where = f"{where}[{index}]"
        entry = block(raw, where=entry_where)
        check_keys(
            entry,
            required_keys=("id", "enabled"),
            optional_keys=("config", "options"),
            where=entry_where,
        )
        raw_config = entry.get("config")
        raw_options = entry.get("options")
        entries.append(
            InstrumentSpec(
                id=text(required(entry, "id", where=entry_where), where=f"{entry_where}.id"),
                enabled=flag(
                    required(entry, "enabled", where=entry_where), where=f"{entry_where}.enabled"
                ),
                config=(
                    None
                    if raw_config is None
                    else Path(text(raw_config, where=f"{entry_where}.config"))
                ),
                options=(
                    None
                    if raw_options is None
                    else InstrumentConfig(values=block(raw_options, where=f"{entry_where}.options"))
                ),
            )
        )
    return tuple(entries)


def _noise(data: Mapping[str, ManifestValue]) -> NoiseSpec:
    where = "noise"
    check_keys(data, required_keys=("enabled_sources", "calibration"), where=where)
    return NoiseSpec(
        enabled_sources=strings(
            required(data, "enabled_sources", where=where), where=f"{where}.enabled_sources"
        ),
        calibration=member(
            required(data, "calibration", where=where),
            CalibrationMode,
            where=f"{where}.calibration",
        ),
    )


def _inverse(data: Mapping[str, ManifestValue]) -> InverseSpec:
    where = "inverse"
    check_keys(
        data,
        required_keys=("model", "engine", "draws", "chains"),
        optional_keys=("mesh", "parameters", "config"),
        where=where,
    )
    raw_config = data.get("config")
    return InverseSpec(
        model=text(required(data, "model", where=where), where=f"{where}.model"),
        engine=text(required(data, "engine", where=where), where=f"{where}.engine"),
        draws=integer(required(data, "draws", where=where), where=f"{where}.draws"),
        chains=integer(required(data, "chains", where=where), where=f"{where}.chains"),
        mesh=block(data.get("mesh", {}), where=f"{where}.mesh"),
        parameters=block(data.get("parameters", {}), where=f"{where}.parameters"),
        config=(
            None
            if raw_config is None
            else InverseConfig(values=block(raw_config, where=f"{where}.config"))
        ),
    )


def _validation(data: Mapping[str, ManifestValue]) -> ValidationSpec:
    where = "validation"
    check_keys(data, required_keys=("seal_truth", "metrics", "n_repeats"), where=where)
    return ValidationSpec(
        seal_truth=flag(required(data, "seal_truth", where=where), where=f"{where}.seal_truth"),
        metrics=strings(required(data, "metrics", where=where), where=f"{where}.metrics"),
        n_repeats=integer(required(data, "n_repeats", where=where), where=f"{where}.n_repeats"),
    )


def _outputs(data: Mapping[str, ManifestValue]) -> OutputSpec:
    where = "outputs"
    check_keys(data, required_keys=("artifacts",), optional_keys=("figures", "report"), where=where)
    names = strings(required(data, "artifacts", where=where), where=f"{where}.artifacts")
    return OutputSpec(
        artifacts=tuple(
            member(name, ArtifactRequest, where=f"{where}.artifacts[{index}]")
            for index, name in enumerate(names)
        ),
        figures=strings(data.get("figures", ()), where=f"{where}.figures"),
        report=flag(data.get("report", False), where=f"{where}.report"),
    )


def _check_the_mandatory_mismatch(manifest: Manifest) -> None:
    """doc 05 §7.1 and §7.2, checked from the manifest alone."""
    inverse = manifest.inverse
    if inverse is None:
        return

    tier = manifest.experiment.tier
    same_model = inverse.model == manifest.forward.solver
    forward_mesh = manifest.forward.config.get("mesh")
    same_mesh = _same_mesh(forward_mesh, inverse.mesh)

    if tier is Tier.T2:
        if same_model:
            raise ManifestConsistencyError(
                f"tier T2 with inverse.model == forward.solver ({inverse.model!r}) is an "
                "inverse crime reported as the honest result. doc 05 §7.1 makes the "
                "forward/inverse mismatch mandatory precisely because an artificially "
                "perfect recovery proves nothing; doc 05 §7.2 calls reporting T1 as T2 a "
                "project defect. Use tier T1 if the crime is deliberate."
            )
        if same_mesh:
            raise ManifestConsistencyError(
                "tier T2 with forward.mesh == inverse.mesh commits half the inverse "
                "crime. doc 05 §7.1 requires the spatial discretisations to differ "
                "(dz = lambda_D/2 against lambda_D/3, graded mesh A against mesh B)."
            )
        return

    if not same_model:
        raise ManifestConsistencyError(
            f"tier {tier.value} is defined by inverting with the *same* model that "
            f"generated the data (doc 05 §7.2), but forward.solver is "
            f"{manifest.forward.solver!r} and inverse.model is {inverse.model!r}. A "
            "mismatched run is tier T2; labelling it T1 understates it rather than "
            "overstating it, which is still a mislabel."
        )
    if not same_mesh:
        raise ManifestConsistencyError(
            f"tier {tier.value} inverts with the same model but a different mesh, which "
            "is neither of doc 05 §7.2's configurations. Match the meshes for T0/T1, or "
            "declare the run T2."
        )


def _same_mesh(left: object, right: object) -> bool:
    """Whether two mesh blocks describe the same discretisation.

    Compared as plain structures rather than by identity: one arrives through a
    :class:`~vpl.core.protocols.config.SolverConfig` (which freezes and sorts) and the
    other straight off the document, and two mappings that agree key for key are the same
    mesh however they were frozen.
    """
    if left is None or right is None:
        return left is right
    return plain(left) == plain(right)  # type: ignore[arg-type]


def manifest_from_document(document: Mapping[str, ManifestValue]) -> Manifest:
    """Parse and validate a loaded manifest document — doc 08 §6.

    Args:
        document: The manifest as loaded, already resolved to plain values.

    Returns:
        The validated manifest.

    Raises:
        UnknownKeyError: If any block sets a key nothing reads.
        ManifestConsistencyError: If two blocks contradict each other — see the module
            docstring on the doc 05 §7 mismatch.
        ValueError: If a required key is missing or a value is out of range.
        TypeError: If a value is of the wrong kind.
    """
    top = block(document, where="manifest")
    check_keys(
        top,
        required_keys=_TOP_LEVEL_REQUIRED,
        optional_keys=_TOP_LEVEL_OPTIONAL,
        where="manifest",
    )

    raw_instruments = top.get("instruments")
    raw_noise = top.get("noise")
    raw_inverse = top.get("inverse")
    raw_validation = top.get("validation")

    manifest = Manifest(
        document=block(frozen(dict(top), where=""), where="manifest"),
        experiment=_experiment(block(top["experiment"], where="experiment")),
        plasma=_plasma(block(top["plasma"], where="plasma")),
        forward=_forward(block(top["forward"], where="forward")),
        outputs=_outputs(block(top["outputs"], where="outputs")),
        instruments=() if raw_instruments is None else _instruments(raw_instruments),
        noise=None if raw_noise is None else _noise(block(raw_noise, where="noise")),
        inverse=None if raw_inverse is None else _inverse(block(raw_inverse, where="inverse")),
        validation=(
            None
            if raw_validation is None
            else _validation(block(raw_validation, where="validation"))
        ),
    )
    _check_the_mandatory_mismatch(manifest)
    return manifest
