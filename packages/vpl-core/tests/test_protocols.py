"""The core contracts — doc 08 §4, and the traceability annotations of doc 08 §9.

Two things are being tested here, and they are not the same thing.

The **supporting types** — ``IonEnergyFlux``, ``CostEstimate``, ``Calibration`` and the
rest — are ordinary value objects and are tested the way every other type in this package
is: what they accept, what they refuse, and why the refusal matters.

The **protocols** are tested differently. A protocol is only worth declaring if something
fails it, so each of the four gets a complete fake implementation (proving the contract is
implementable and that the concrete types line up) *and* a near-miss that does not satisfy
it (proving the protocol constrains something). A protocol every object satisfies is
decoration, and decoration in a contract layer is worse than nothing, because it is
mistaken for a guarantee.

Two kinds of near-miss appear. A **structural** near-miss omits a method, and
:func:`isinstance` rejects it at plugin-load time. A **signature** near-miss has every
method name and the wrong types; :func:`isinstance` accepts it — ``@runtime_checkable``
checks names only — and mypy rejects it. The ``# type: ignore[arg-type]`` markers below
are the assertion for that second kind: ``warn_unused_ignores`` is part of ``--strict``,
so if a protocol stopped constraining signatures, the unused ignore would fail the type
check.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from vpl.core.protocols import (
    Calibration,
    CalibrationReference,
    CalibrationSet,
    Citation,
    Config,
    CostBasis,
    CostEstimate,
    DetectionFloor,
    Device,
    ForwardModel,
    ForwardSolver,
    Identifiability,
    IdentifiabilityReport,
    Instrument,
    InstrumentConfig,
    InstrumentMetadata,
    InverseConfig,
    InverseEngine,
    IonEnergyFlux,
    LogProb,
    NoiseModel,
    Signal,
    SignalDomain,
    SolverConfig,
    SolverMetadata,
    TraceabilityError,
    canonical_evidence_id,
    canonical_requirement_id,
    classify_by_condition_number,
    clear_registry,
    entries,
    evidence_of,
    requirements_of,
    satisfies,
    traceability_matrix,
    uncovered_requirements,
    unverified_claims,
    verified_by,
)
from vpl.core.provenance import Tier
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    Fidelity,
    Measurement,
    MeasurementSet,
    Observable,
    ParameterLevel,
    PlasmaParams,
    PlasmaState,
    Posterior,
    SamplerDiagnostics,
    ScalarField,
    SpatialGrid,
    Species,
    TimeGrid,
)
from vpl.core.units import Q_, DimensionalityError

# ── fixtures ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def argon() -> Species:
    return Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)


@pytest.fixture
def params(argon: Species) -> PlasmaParams:
    """The reference operating point RP-1 of doc 01 §2.1."""
    return PlasmaParams(
        species=argon,
        n_0=Q_(1e17, "m**-3"),
        T_e=Q_(3.0, "eV"),
        T_i=Q_(0.05, "eV"),
        T_g=Q_(300.0, "K"),
        pressure=Q_(5.0, "mTorr"),
        bias=Q_(-250.0, "V"),
        gamma_se=0.10,
        kappa=1.0,
    )


@pytest.fixture
def grid() -> SpatialGrid:
    return SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=5)


@pytest.fixture
def state(params: PlasmaParams, grid: SpatialGrid) -> PlasmaState:
    spec = {"n_e": "m**-3", "n_i": "m**-3", "Phi": "V", "T_e": "eV"}
    fields = {
        name: ScalarField(
            name=name, values=np.ones(grid.n_points), units=units, grid=grid, time=None
        )
        for name, units in spec.items()
    }
    return PlasmaState(
        params=params,
        grid=grid,
        time=None,
        fields=fields,
        ion_distribution=None,
        fidelity=Fidelity.L1,
    )


@pytest.fixture
def window() -> AcquisitionWindow:
    return AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(2.0, "ns"))


@pytest.fixture
def citation() -> Citation:
    return Citation(
        key="birdsall1991",
        reference="Birdsall, IEEE Trans. Plasma Sci. 19, 65 (1991)",
        doi="10.1109/27.106800",
    )


@pytest.fixture
def floor() -> DetectionFloor:
    """The doc 01 IF-6 blind region: the interferometer sees nothing below this."""
    return DetectionFloor(
        quantity="n_0",
        threshold=Q_(3.3e16, "m**-3"),
        requirement="IF-6",
    )


@pytest.fixture(autouse=True)
def _empty_traceability_registry() -> Iterator[None]:
    """Isolate the process-global traceability index between tests.

    The index is module state by design (doc 08 §9 generates the matrix from whatever
    the CI import walk reached), so a test that registers an annotation would otherwise
    be visible to every test that ran after it.
    """
    clear_registry()
    yield
    clear_registry()


# ── fakes: complete implementations of each doc 08 §4 protocol ──────────────────────


class SolverDemandingATimeGrid:
    """Structurally a ForwardSolver, but its ``solve`` refuses a steady request.

    doc 08 §4 declares ``t: TimeGrid``. Because parameter types are contravariant, a
    solver written to that literal signature is *not* assignable to the widened protocol
    — which is the whole reason the widening had to be deliberate rather than incidental.
    """

    def configure(self, cfg: SolverConfig) -> None: ...

    def solve(self, params: PlasmaParams, t: TimeGrid) -> PlasmaState:
        raise NotImplementedError

    def flux(self, state: PlasmaState, z: float) -> IonEnergyFlux:
        raise NotImplementedError

    def fidelity(self) -> Fidelity:
        return Fidelity.L1

    def cost_estimate(self, cfg: SolverConfig) -> CostEstimate:
        raise NotImplementedError

    def metadata(self) -> SolverMetadata:
        raise NotImplementedError


class FakeSolver:
    """A ForwardSolver that computes nothing — the contract, and only the contract."""

    def __init__(self) -> None:
        self._cfg: SolverConfig | None = None

    def configure(self, cfg: SolverConfig) -> None:
        self._cfg = cfg

    def solve(self, params: PlasmaParams, t: TimeGrid | None) -> PlasmaState:
        grid = SpatialGrid.uniform(length=Q_(20.0, "mm"), n_points=4)
        shape = (grid.n_points,) if t is None else (t.n_points, grid.n_points)
        spec = {"n_e": "m**-3", "n_i": "m**-3", "Phi": "V", "T_e": "eV"}
        return PlasmaState(
            params=params,
            grid=grid,
            time=t,
            fields={
                name: ScalarField(name=name, values=np.ones(shape), units=units, grid=grid, time=t)
                for name, units in spec.items()
            },
            ion_distribution=None,
            fidelity=Fidelity.L0,
        )

    def flux(self, state: PlasmaState, z: float) -> IonEnergyFlux:
        return IonEnergyFlux(
            position=Q_(z, "m"),
            species=state.params.species,
            energy_flux_toward_wall_watt_per_m2=np.asarray(4.0e2),
            particle_flux_toward_wall_per_m2_s=np.asarray(1.0e19),
            fidelity=state.fidelity,
            time=None,
        )

    def fidelity(self) -> Fidelity:
        return Fidelity.L0

    def cost_estimate(self, cfg: SolverConfig) -> CostEstimate:
        return CostEstimate.estimated(
            wall_clock=Q_(10.0, "us"), device=Device.CPU, source="doc 10 §3.1"
        )

    def metadata(self) -> SolverMetadata:
        return SolverMetadata(
            name="vpl.physics.analytic.child_langmuir",
            version="0.1.0",
            citations=(Citation(key="lieberman2005", reference="Lieberman & Lichtenberg (2005)"),),
        )


class FakeInstrument:
    """An Instrument whose ``forward`` and ``observe`` share a code path — doc 04 §9."""

    instrument_id = "interf"

    def __init__(self) -> None:
        self._cfg: InstrumentConfig | None = None
        self._calibration: Calibration | None = None

    def configure(self, cfg: InstrumentConfig) -> None:
        self._cfg = cfg

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        reference = refs.for_quantity("phase")
        return Calibration(
            instrument_id=self.instrument_id,
            coefficients={"phase": 1.0},
            relative_uncertainty={"phase": reference.relative_uncertainty},
            state=CalibrationState.ESTIMATED,
            reference=reference.name,
        )

    def _predict(self, state: PlasmaState) -> NDArray[np.float64]:
        return np.asarray(state.field("n_e").values[:1] * 1.0e-19, dtype=np.float64)

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        return Observable(
            instrument_id=self.instrument_id,
            values=self._predict(state),
            units="rad",
            window=w,
        )

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        noiseless = self.forward(state, w)
        return Measurement(
            instrument_id=self.instrument_id,
            values=noiseless.values,
            uncertainty=np.full_like(noiseless.values, 1.0e-4),
            units=noiseless.units,
            window=w,
            calibration=CalibrationState.ESTIMATED,
        )

    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb:
        residual = (obs.values - pred.values) / obs.uncertainty
        return float(-0.5 * np.sum(residual**2))

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        return self.metadata().detection_floor.admits(state_guess.n_0)

    def metadata(self) -> InstrumentMetadata:
        return InstrumentMetadata(
            instrument_id=self.instrument_id,
            name="Mach-Zehnder heterodyne CO2 interferometer",
            version="0.1.0",
            citations=(Citation(key="hutchinson2002", reference="Hutchinson (2002) ch. 4"),),
            detection_floor=DetectionFloor(
                quantity="n_0", threshold=Q_(3.3e16, "m**-3"), requirement="IF-6"
            ),
        )


class FakeForwardModel:
    """``F = F4 o F3 o F2 o F1`` (doc 05 §1.1), stubbed."""

    def predict(self, params: PlasmaParams, data: MeasurementSet) -> tuple[Observable, ...]:
        return tuple(
            Observable(
                instrument_id=m.instrument_id,
                values=np.zeros_like(m.values),
                units=m.units,
                window=m.window,
            )
            for m in data
        )

    def log_likelihood(self, params: PlasmaParams, data: MeasurementSet) -> LogProb:
        return -float(data.n_observations)

    def fidelity(self) -> Fidelity:
        return Fidelity.L3


class FakeEngine:
    """An InverseEngine that returns a fixed posterior."""

    def __init__(self) -> None:
        self._cfg: InverseConfig | None = None
        self.model_fidelity: Fidelity | None = None

    def configure(self, cfg: InverseConfig) -> None:
        self._cfg = cfg

    def fit(self, data: MeasurementSet, model: ForwardModel) -> Posterior:
        # doc 05 §7.1: the level used inside the inversion must differ from the one that
        # generated the truth, so an engine that never asked could not report its tier.
        self.model_fidelity = model.fidelity()
        rng = default_rng(seed=20260804 + data.n_observations)
        return Posterior(
            samples=rng.normal(size=(2, 500, 1)),
            names=("n_0",),
            levels={"n_0": ParameterLevel.CONTROL},
            tier=Tier.T2,
            diagnostics=self.diagnostics(),
        )

    def diagnostics(self) -> SamplerDiagnostics:
        return SamplerDiagnostics(r_hat={"n_0": 1.001}, ess={"n_0": 900.0}, divergences=0)

    def identifiability(self, at: PlasmaParams) -> IdentifiabilityReport:
        return IdentifiabilityReport(
            at=at,
            names=("n_0", "T_e"),
            eigenvalues=np.asarray([100.0, 1.0]),
            eigenvectors=np.eye(2),
            classification=Identifiability.IDENTIFIABLE,
        )


class FakeNoise:
    """A single switchable noise source — doc 04 §7.2, V-30."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    def apply(self, signal: Signal, rng: Generator) -> Signal:
        if not self._enabled:
            return signal
        return signal.with_values(rng.poisson(signal.values).astype(np.float64))

    def variance(self, signal: Signal) -> Signal:
        return signal.with_values(signal.values if self._enabled else np.zeros_like(signal.values))

    def enabled(self) -> bool:
        return self._enabled


# ── near-misses: structurally incomplete ────────────────────────────────────────────


class SolverWithoutCostEstimate:
    """Everything a ForwardSolver needs except the doc 10 §6 scheduling hook."""

    def configure(self, cfg: SolverConfig) -> None: ...

    def solve(self, params: PlasmaParams, t: TimeGrid) -> PlasmaState:
        raise NotImplementedError

    def flux(self, state: PlasmaState, z: float) -> IonEnergyFlux:
        raise NotImplementedError

    def fidelity(self) -> Fidelity:
        return Fidelity.L0

    def metadata(self) -> SolverMetadata:
        raise NotImplementedError


class InstrumentWithoutDetectionGate:
    """An Instrument missing ``is_informative`` — the doc 01 IF-6 gate."""

    def configure(self, cfg: InstrumentConfig) -> None: ...

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        raise NotImplementedError

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable:
        raise NotImplementedError

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        raise NotImplementedError

    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb:
        raise NotImplementedError

    def metadata(self) -> InstrumentMetadata:
        raise NotImplementedError


class EngineWithoutIdentifiability:
    """An InverseEngine that fits but cannot report a null space — doc 05 §6."""

    def configure(self, cfg: InverseConfig) -> None: ...

    def fit(self, data: MeasurementSet, model: ForwardModel) -> Posterior:
        raise NotImplementedError

    def diagnostics(self) -> SamplerDiagnostics:
        raise NotImplementedError


class NoiseWithoutSwitch:
    """A noise source that cannot be switched off, which doc 04 §8 V-30 forbids."""

    def apply(self, signal: Signal, rng: Generator) -> Signal:
        raise NotImplementedError

    def variance(self, signal: Signal) -> Signal:
        raise NotImplementedError


# ── near-misses: right names, wrong types ───────────────────────────────────────────


class SolverReturningBareFloatFlux:
    """``flux`` returns a number instead of the doc 01 §1.2 decomposition."""

    def configure(self, cfg: SolverConfig) -> None: ...

    def solve(self, params: PlasmaParams, t: TimeGrid) -> PlasmaState:
        raise NotImplementedError

    def flux(self, state: PlasmaState, z: float) -> float:
        return 0.0

    def fidelity(self) -> Fidelity:
        return Fidelity.L0

    def cost_estimate(self, cfg: SolverConfig) -> CostEstimate:
        raise NotImplementedError

    def metadata(self) -> SolverMetadata:
        raise NotImplementedError


class InstrumentReturningMeasurementFromForward:
    """``forward`` returns a noisy Measurement — the doc 04 §9 split, broken."""

    def configure(self, cfg: InstrumentConfig) -> None: ...

    def calibrate(self, refs: CalibrationSet) -> Calibration:
        raise NotImplementedError

    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        raise NotImplementedError

    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement:
        raise NotImplementedError

    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb:
        raise NotImplementedError

    def is_informative(self, state_guess: PlasmaParams) -> bool:
        return True

    def metadata(self) -> InstrumentMetadata:
        raise NotImplementedError


def _requires_solver(solver: ForwardSolver) -> ForwardSolver:
    return solver


def _requires_instrument(instrument: Instrument) -> Instrument:
    return instrument


# ── the supporting types ────────────────────────────────────────────────────────────


class TestCitation:
    def test_carries_the_reference_doc_00_c2_requires_of_every_algorithm(self) -> None:
        cited = Citation(key="birdsall1991", reference="Birdsall (1991)")

        assert cited.key == "birdsall1991"
        assert "Birdsall" in repr(cited)

    def test_rejects_an_empty_key(self) -> None:
        with pytest.raises(ValueError, match="key"):
            Citation(key="  ", reference="Birdsall (1991)")

    def test_rejects_an_empty_reference(self) -> None:
        with pytest.raises(ValueError, match="reference"):
            Citation(key="birdsall1991", reference="")

    def test_rejects_a_doi_that_is_not_a_doi(self) -> None:
        # A URL stored in the doi field looks right in a repr and resolves to nothing
        # through a DOI resolver, which is exactly the failure that survives review.
        with pytest.raises(ValueError, match="doi"):
            Citation(key="k", reference="r", doi="https://example.org/paper")


class TestDetectionFloor:
    def test_admits_a_state_above_the_floor(self, floor: DetectionFloor) -> None:
        assert floor.admits(Q_(1e17, "m**-3")) is True

    def test_refuses_a_state_below_the_floor(self, floor: DetectionFloor) -> None:
        # doc 01 IF-6: at 1e16 the CO2 interferometer is below its own detection floor,
        # and an inversion that ingests its noise as data produces confident nonsense.
        assert floor.admits(Q_(1e16, "m**-3")) is False

    def test_the_floor_itself_is_admitted(self, floor: DetectionFloor) -> None:
        assert floor.admits(Q_(3.3e16, "m**-3")) is True

    def test_rejects_a_value_of_the_wrong_dimensionality(self, floor: DetectionFloor) -> None:
        with pytest.raises(DimensionalityError):
            floor.admits(Q_(3.0, "eV"))

    def test_must_name_the_requirement_that_fixed_it(self) -> None:
        # doc 00 C4: no hidden assumptions. A detection floor with no requirement behind
        # it is a number somebody chose, and nothing in the artifact says who.
        with pytest.raises(TraceabilityError, match="IF6"):
            DetectionFloor(quantity="n_0", threshold=Q_(3.3e16, "m**-3"), requirement="IF6")

    def test_rejects_a_non_positive_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            DetectionFloor(quantity="n_0", threshold=Q_(0.0, "m**-3"), requirement="IF-6")


class TestSolverMetadata:
    def test_requires_at_least_one_citation(self) -> None:
        # doc 00 C2: "Every algorithm must have a citation."
        with pytest.raises(ValueError, match="citation"):
            SolverMetadata(name="vpl.physics.mystery", version="0.1.0", citations=())

    def test_carries_the_version_provenance_records(self, citation: Citation) -> None:
        meta = SolverMetadata(
            name="vpl.physics.kinetic.pic1d3v", version="0.3.1", citations=(citation,)
        )

        assert meta.version == "0.3.1"
        assert meta.citations == (citation,)

    def test_rejects_an_empty_version(self, citation: Citation) -> None:
        with pytest.raises(ValueError, match="version"):
            SolverMetadata(name="vpl.physics.kinetic.pic1d3v", version="", citations=(citation,))


class TestInstrumentMetadata:
    def test_carries_the_detection_floor_if_6_gates_on(
        self, citation: Citation, floor: DetectionFloor
    ) -> None:
        meta = InstrumentMetadata(
            instrument_id="interf",
            name="CO2 heterodyne interferometer",
            version="0.1.0",
            citations=(citation,),
            detection_floor=floor,
        )

        assert meta.detection_floor.requirement == "IF-6"

    def test_requires_a_citation_like_every_other_algorithm(self, floor: DetectionFloor) -> None:
        with pytest.raises(ValueError, match="citation"):
            InstrumentMetadata(
                instrument_id="interf",
                name="CO2 heterodyne interferometer",
                version="0.1.0",
                citations=(),
                detection_floor=floor,
            )

    def test_rejects_an_empty_instrument_id(
        self, citation: Citation, floor: DetectionFloor
    ) -> None:
        # The id names the HDF5 group of doc 08 §7; an empty one names nothing.
        with pytest.raises(ValueError, match="instrument id"):
            InstrumentMetadata(
                instrument_id="   ",
                name="CO2 heterodyne interferometer",
                version="0.1.0",
                citations=(citation,),
                detection_floor=floor,
            )


class TestConfig:
    def test_reads_a_manifest_block_as_typed_values(self) -> None:
        cfg = SolverConfig(values={"n_ppc": 1000, "mesh": {"dz": "lambda_D/2"}})

        assert cfg.require_int("n_ppc") == 1000
        assert cfg.section("mesh").require_str("dz") == "lambda_D/2"

    def test_an_int_is_acceptable_where_a_float_is_required(self) -> None:
        # YAML writes `1000` and `1.0e17` for quantities of the same kind, and refusing
        # the first would push a float() call into every solver's configure().
        assert SolverConfig(values={"draws": 4000}).require_float("draws") == 4000.0

    def test_a_bool_is_not_acceptable_where_a_number_is_required(self) -> None:
        # bool is a subclass of int, so an `enabled: true` landing in a count field
        # would otherwise read as 1 and run a sweep of one case.
        with pytest.raises(TypeError, match="n_ppc"):
            SolverConfig(values={"n_ppc": True}).require_int("n_ppc")

    def test_a_missing_key_names_the_keys_that_are_present(self) -> None:
        with pytest.raises(KeyError, match="n_ppc"):
            SolverConfig(values={"n_ppc": 1000}).require_int("nppc")

    def test_the_wrapped_mapping_cannot_be_mutated(self) -> None:
        cfg = SolverConfig(values={"n_ppc": 1000})

        with pytest.raises(TypeError):
            cfg.values["n_ppc"] = 2000  # type: ignore[index]

    def test_mutating_the_caller_mapping_does_not_reach_the_config(self) -> None:
        source: dict[str, object] = {"n_ppc": 1000}
        cfg = SolverConfig(values=source)  # type: ignore[arg-type]
        source["n_ppc"] = 2000

        assert cfg.require_int("n_ppc") == 1000

    def test_nested_blocks_are_frozen_too(self) -> None:
        cfg = SolverConfig(values={"mesh": {"dz": "lambda_D/2"}})
        block = cfg["mesh"]

        assert isinstance(block, dict) is False
        with pytest.raises(TypeError):
            block["dz"] = "other"  # type: ignore[index]

    def test_sequences_become_tuples_so_nothing_downstream_can_append(self) -> None:
        cfg = InverseConfig(values={"metrics": ["rel_error", "coverage"]})

        assert cfg["metrics"] == ("rel_error", "coverage")

    def test_a_solver_config_is_not_an_instrument_config(self) -> None:
        # Nominal, not structural. `configure(cfg: SolverConfig)` must not silently
        # accept the instrument block of the same manifest. Held through `object` here
        # because mypy makes the stronger statement on its own — comparing the two
        # directly is a non-overlapping equality check and fails the type gate.
        solver: object = SolverConfig(values={"a": 1})
        instrument: object = InstrumentConfig(values={"a": 1})

        assert solver != instrument

    def test_two_configs_of_the_same_kind_and_content_are_equal(self) -> None:
        assert SolverConfig(values={"a": 1}) == SolverConfig(values={"a": 1})

    def test_reports_its_keys_in_a_deterministic_order(self) -> None:
        cfg = SolverConfig(values={"n_ppc": 1, "mesh": {}, "a": 2})

        assert tuple(cfg) == ("a", "mesh", "n_ppc")

    def test_membership_and_length_read_naturally(self) -> None:
        cfg = SolverConfig(values={"n_ppc": 1000})

        assert "n_ppc" in cfg
        assert "mesh" not in cfg
        assert len(cfg) == 1

    def test_get_returns_the_default_for_an_absent_key(self) -> None:
        assert SolverConfig(values={}).get("n_ppc", 1000) == 1000

    def test_section_refuses_a_key_that_is_not_a_block(self) -> None:
        with pytest.raises(TypeError, match="n_ppc"):
            SolverConfig(values={"n_ppc": 1000}).section("n_ppc")

    def test_a_nested_block_is_not_itself_a_solver_config(self) -> None:
        # `mesh:` inside `forward:` is not something configure() should accept, so
        # section() deliberately widens the type rather than preserving it.
        block = SolverConfig(values={"mesh": {"dz": "lambda_D/2"}}).section("mesh")

        assert type(block) is Config

    def test_require_bool_reads_a_switch(self) -> None:
        assert InstrumentConfig(values={"enabled": True}).require_bool("enabled") is True

    def test_require_bool_rejects_a_number(self) -> None:
        with pytest.raises(TypeError, match="enabled"):
            InstrumentConfig(values={"enabled": 1}).require_bool("enabled")

    def test_require_str_rejects_a_number(self) -> None:
        with pytest.raises(TypeError, match="engine"):
            InverseConfig(values={"engine": 4}).require_str("engine")

    def test_require_float_rejects_a_string(self) -> None:
        # `dz: lambda_D/2` is stored as the string the manifest wrote, for the physics
        # package to resolve; reading it as a number here would have to invent one.
        with pytest.raises(TypeError, match="dz"):
            SolverConfig(values={"dz": "lambda_D/2"}).require_float("dz")

    def test_rejects_a_value_no_manifest_could_hold(self) -> None:
        with pytest.raises(TypeError, match="set"):
            SolverConfig(values={"seeds": {1, 2}})  # type: ignore[dict-item]

    def test_rejects_a_key_that_is_not_a_string(self) -> None:
        # YAML permits `1: x`. Nothing downstream addresses a manifest block by integer,
        # and a non-string key would break both the provenance hash and the error paths.
        with pytest.raises(TypeError, match="keys are strings"):
            SolverConfig(values={"mesh": {1: "x"}})  # type: ignore[dict-item]


class TestIonEnergyFlux:
    def test_carries_both_factors_of_the_doc_01_1_2_decomposition(self, argon: Species) -> None:
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.asarray(4.0e2),
            particle_flux_toward_wall_per_m2_s=np.asarray(1.0e19),
            fidelity=Fidelity.L2,
        )

        assert flux.energy_flux.units == Q_(1.0, "W/m**2").units
        assert flux.particle_flux.magnitude == pytest.approx(1.0e19)

    def test_mean_impact_energy_is_the_ratio_of_the_two_factors(self, argon: Species) -> None:
        # doc 01 §1.2: <E_i> = Gamma_E / Gamma_i, and the two factors are constrained by
        # different physics — which is why both are stored rather than just the product.
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.asarray(4.0e2),
            particle_flux_toward_wall_per_m2_s=np.asarray(1.0e19),
            fidelity=Fidelity.L2,
        )

        assert float(flux.mean_impact_energy_joules) == pytest.approx(4.0e-17)

    def test_the_mean_impact_energy_crosses_module_boundaries_dimensionally(
        self, argon: Species
    ) -> None:
        # doc 08 §5: pint quantities at module boundaries, raw arrays only in hot loops.
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.asarray(4.0e2),
            particle_flux_toward_wall_per_m2_s=np.asarray(1.0e19),
            fidelity=Fidelity.L2,
        )

        assert float(flux.mean_impact_energy.to("eV").magnitude) == pytest.approx(249.6, rel=1e-3)

    def test_a_positive_flux_means_energy_arriving_at_the_wall(self, argon: Species) -> None:
        # doc 02 §2, and consistent with VelocityDistribution.particle_flux_toward_wall:
        # the wall is at z = 0, z is positive into the plasma, so an ion that reaches the
        # wall has v_z < 0 and the sign is absorbed in the name, not in a stray minus.
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.asarray(4.0e2),
            particle_flux_toward_wall_per_m2_s=np.asarray(1.0e19),
            fidelity=Fidelity.L2,
        )

        assert flux.is_toward_wall is True

    def test_mean_impact_energy_is_undefined_rather_than_zero_where_no_ions_arrive(
        self, argon: Species
    ) -> None:
        # Zero would be a claim: "the ions that arrived carried no energy". There were no
        # ions. NaN says so, and propagates instead of quietly biasing an average down.
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.asarray([4.0e2, 0.0]),
            particle_flux_toward_wall_per_m2_s=np.asarray([1.0e19, 0.0]),
            fidelity=Fidelity.L2,
            time=TimeGrid.uniform(duration=Q_(10.0, "ns"), n_points=2),
        )
        energies = flux.mean_impact_energy_joules

        assert energies[0] == pytest.approx(4.0e-17)
        assert np.isnan(energies[1])

    def test_reports_the_fidelity_that_produced_it(self, argon: Species) -> None:
        # doc 03 §6 applies the same functional at every level "so that results are
        # comparable"; benchmark B-03 compares L1 against L2 and must not lose track of
        # which curve is which once the solver is gone and only the artifact remains.
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.asarray(1.0),
            particle_flux_toward_wall_per_m2_s=np.asarray(1.0),
            fidelity=Fidelity.L1,
        )

        assert flux.fidelity is Fidelity.L1

    def test_a_time_resolved_flux_matches_its_time_grid(self, argon: Species) -> None:
        time = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=4)
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.ones(4),
            particle_flux_toward_wall_per_m2_s=np.ones(4),
            fidelity=Fidelity.L2,
            time=time,
        )

        assert flux.is_steady is False
        assert flux.n_times == 4

    def test_rejects_a_time_axis_that_disagrees_with_the_values(self, argon: Species) -> None:
        with pytest.raises(ValueError, match="shape"):
            IonEnergyFlux(
                position=Q_(0.0, "m"),
                species=argon,
                energy_flux_toward_wall_watt_per_m2=np.ones(3),
                particle_flux_toward_wall_per_m2_s=np.ones(3),
                fidelity=Fidelity.L2,
                time=TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=4),
            )

    def test_rejects_factors_of_different_shapes(self, argon: Species) -> None:
        # Gamma_E and Gamma_i are moments of one distribution. Different shapes means
        # they came from different states, and their ratio would be meaningless.
        with pytest.raises(ValueError, match="same shape"):
            IonEnergyFlux(
                position=Q_(0.0, "m"),
                species=argon,
                energy_flux_toward_wall_watt_per_m2=np.ones(3),
                particle_flux_toward_wall_per_m2_s=np.ones(4),
                fidelity=Fidelity.L2,
                time=TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=3),
            )

    def test_rejects_a_non_finite_flux(self, argon: Species) -> None:
        with pytest.raises(ValueError, match="finite"):
            IonEnergyFlux(
                position=Q_(0.0, "m"),
                species=argon,
                energy_flux_toward_wall_watt_per_m2=np.asarray(np.inf),
                particle_flux_toward_wall_per_m2_s=np.asarray(1.0),
                fidelity=Fidelity.L2,
            )

    def test_rejects_a_position_that_is_not_a_length(self, argon: Species) -> None:
        with pytest.raises(DimensionalityError):
            IonEnergyFlux(
                position=Q_(0.0, "s"),
                species=argon,
                energy_flux_toward_wall_watt_per_m2=np.asarray(1.0),
                particle_flux_toward_wall_per_m2_s=np.asarray(1.0),
                fidelity=Fidelity.L2,
            )

    def test_two_fluxes_with_the_same_content_are_equal(self, argon: Species) -> None:
        def build() -> IonEnergyFlux:
            return IonEnergyFlux(
                position=Q_(0.0, "m"),
                species=argon,
                energy_flux_toward_wall_watt_per_m2=np.asarray(1.0),
                particle_flux_toward_wall_per_m2_s=np.asarray(2.0),
                fidelity=Fidelity.L2,
            )

        assert build() == build()
        assert build() != "not a flux"


class TestCostEstimate:
    def test_an_estimate_names_itself_an_estimate(self) -> None:
        # doc 10 §3.2 is explicit that its GPU throughput is "an estimate, not a
        # measurement", and gate G-1.4 requires the measurement to replace it. A type
        # that could not tell the two apart would let the estimate survive the gate.
        cost = CostEstimate.estimated(
            wall_clock=Q_(5.0, "s"), device=Device.GPU, source="doc 10 §3.1"
        )

        assert cost.basis is CostBasis.ESTIMATED
        assert cost.is_measured is False

    def test_a_measurement_supersedes_an_estimate(self) -> None:
        measured = CostEstimate.measured(
            wall_clock=Q_(7.5, "s"), device=Device.GPU, source="G-1.4 run 2026-08-05"
        )

        assert measured.basis is CostBasis.MEASURED
        assert measured.is_measured is True

    def test_scaling_by_a_case_count_gives_the_campaign_total(self) -> None:
        # doc 10 §3.1 multiplies unit cost by count to schedule a campaign; §6 uses that
        # total to size the work queue.
        unit = CostEstimate.estimated(
            wall_clock=Q_(5.0, "s"), device=Device.GPU, source="doc 10 §3.1"
        )
        total = unit.scaled(5500)

        assert total.wall_clock_s == pytest.approx(27500.0)
        assert total.basis is CostBasis.ESTIMATED

    def test_scaling_preserves_peak_memory_because_cases_run_one_at_a_time(self) -> None:
        # doc 10 §6 schedules a local process pool over a serial queue; a campaign of
        # 5500 cases does not need 5500x the VRAM, and pretending it did would fail the
        # doc 10 §3.3 feasibility check against the 16 GB of C3 for no reason.
        unit = CostEstimate.estimated(
            wall_clock=Q_(5.0, "s"),
            device=Device.GPU,
            source="doc 10 §3.1",
            peak_memory=Q_(2.0, "GB"),
        )

        assert unit.scaled(100).peak_memory_bytes == pytest.approx(2.0e9)

    def test_rejects_a_non_positive_wall_clock(self) -> None:
        with pytest.raises(ValueError, match="wall_clock"):
            CostEstimate.estimated(wall_clock=Q_(0.0, "s"), device=Device.CPU, source="doc 10")

    def test_rejects_a_wall_clock_that_is_not_a_time(self) -> None:
        with pytest.raises(DimensionalityError):
            CostEstimate.estimated(wall_clock=Q_(5.0, "GB"), device=Device.CPU, source="doc 10")

    def test_an_uncharacterised_workload_reports_no_peak_memory(self) -> None:
        # Doc 10 §3.3 checks every workload against the 16 GB of doc 00 C3. A workload
        # nobody has measured says so rather than reporting a zero that would pass.
        unit = CostEstimate.estimated(wall_clock=Q_(5.0, "s"), device=Device.CPU, source="doc 10")

        assert unit.peak_memory_bytes is None

    def test_rejects_a_non_positive_peak_memory(self) -> None:
        with pytest.raises(ValueError, match="peak_memory"):
            CostEstimate.estimated(
                wall_clock=Q_(5.0, "s"),
                device=Device.GPU,
                source="doc 10",
                peak_memory=Q_(0.0, "GB"),
            )

    def test_rejects_a_non_positive_case_count(self) -> None:
        unit = CostEstimate.estimated(wall_clock=Q_(5.0, "s"), device=Device.GPU, source="doc 10")

        with pytest.raises(ValueError, match="count"):
            unit.scaled(0)

    def test_requires_a_source_so_the_number_can_be_challenged(self) -> None:
        with pytest.raises(ValueError, match="source"):
            CostEstimate.estimated(wall_clock=Q_(5.0, "s"), device=Device.GPU, source="")

    def test_reports_its_basis_and_device_in_its_repr(self) -> None:
        cost = CostEstimate.measured(wall_clock=Q_(5.0, "s"), device=Device.GPU, source="G-1.4")

        assert "measured" in repr(cost)
        assert "gpu" in repr(cost)


class TestCalibration:
    def test_records_whether_the_true_or_the_estimated_response_was_used(self) -> None:
        # doc 04 §7.3: the pipeline works with the estimated response. Applying the true
        # one "would be a form of inverse crime and would understate the error".
        estimated = Calibration(
            instrument_id="thomson",
            coefficients={"absolute": 1.03},
            relative_uncertainty={"absolute": 0.07},
            state=CalibrationState.ESTIMATED,
            reference="Rayleigh scattering in Ar",
        )

        assert estimated.state is CalibrationState.ESTIMATED
        assert estimated.is_inverse_crime is False

    def test_the_true_response_declares_itself_an_inverse_crime(self) -> None:
        truth = Calibration(
            instrument_id="thomson",
            coefficients={"absolute": 1.0},
            relative_uncertainty={"absolute": 0.0},
            state=CalibrationState.TRUE,
            reference="sealed truth",
        )

        assert truth.is_inverse_crime is True

    def test_an_estimated_calibration_cannot_claim_to_be_exact(self) -> None:
        # An estimated response with zero uncertainty everywhere is the true response
        # wearing the honest label — indistinguishable in the artifact, and it makes the
        # doc 06 §4 calibration terms vanish from the error budget.
        with pytest.raises(ValueError, match="uncertainty"):
            Calibration(
                instrument_id="thomson",
                coefficients={"absolute": 1.0},
                relative_uncertainty={"absolute": 0.0},
                state=CalibrationState.ESTIMATED,
                reference="Rayleigh scattering in Ar",
            )

    def test_every_coefficient_needs_an_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="spectral"):
            Calibration(
                instrument_id="thomson",
                coefficients={"absolute": 1.0, "spectral": 1.0},
                relative_uncertainty={"absolute": 0.07},
                state=CalibrationState.ESTIMATED,
                reference="Rayleigh scattering in Ar",
            )

    def test_rejects_a_negative_relative_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="uncertainty"):
            Calibration(
                instrument_id="thomson",
                coefficients={"absolute": 1.0},
                relative_uncertainty={"absolute": -0.07},
                state=CalibrationState.ESTIMATED,
                reference="Rayleigh scattering in Ar",
            )

    def test_requires_a_named_reference_source(self) -> None:
        # doc 01 SYS-3: "traceable, with a single documented reference source".
        with pytest.raises(ValueError, match="reference"):
            Calibration(
                instrument_id="thomson",
                coefficients={"absolute": 1.0},
                relative_uncertainty={"absolute": 0.07},
                state=CalibrationState.ESTIMATED,
                reference="",
            )

    def test_rejects_a_non_finite_relative_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Calibration(
                instrument_id="thomson",
                coefficients={"absolute": 1.0},
                relative_uncertainty={"absolute": math.nan},
                state=CalibrationState.ESTIMATED,
                reference="Rayleigh scattering in Ar",
            )

    def test_two_calibrations_with_the_same_content_are_equal(self) -> None:
        def build(state: CalibrationState) -> Calibration:
            return Calibration(
                instrument_id="thomson",
                coefficients={"absolute": 1.03},
                relative_uncertainty={"absolute": 0.07},
                state=state,
                reference="Rayleigh scattering in Ar",
            )

        assert build(CalibrationState.ESTIMATED) == build(CalibrationState.ESTIMATED)
        # doc 04 §7.3: the same numbers derived the honest way and the crooked way are
        # not the same calibration, and an artifact comparison must not say they are.
        assert build(CalibrationState.ESTIMATED) != build(CalibrationState.TRUE)
        assert build(CalibrationState.ESTIMATED) != "not a calibration"

    def test_the_coefficient_mapping_cannot_be_mutated(self) -> None:
        calibration = Calibration(
            instrument_id="thomson",
            coefficients={"absolute": 1.0},
            relative_uncertainty={"absolute": 0.07},
            state=CalibrationState.ESTIMATED,
            reference="Rayleigh scattering in Ar",
        )

        with pytest.raises(TypeError):
            calibration.coefficients["absolute"] = 2.0  # type: ignore[index]


class TestCalibrationSet:
    def test_looks_a_reference_up_by_the_quantity_it_certifies(self) -> None:
        refs = CalibrationSet.of(
            CalibrationReference(
                name="NIST FEL tungsten-halogen lamp",
                quantity="absolute_radiometric",
                value=Q_(1.0, "W/(m**2*nm*sr)"),
                relative_uncertainty=0.06,
                traceable_to="NIST FEL scale",
            )
        )

        assert refs.for_quantity("absolute_radiometric").relative_uncertainty == 0.06

    def test_a_missing_reference_names_what_the_set_does_carry(self) -> None:
        refs = CalibrationSet.of(
            CalibrationReference(
                name="Hg/Ar pencil lamp",
                quantity="wavelength",
                value=Q_(546.074, "nm"),
                relative_uncertainty=1e-5,
                traceable_to="NIST line list",
            )
        )

        with pytest.raises(KeyError, match="wavelength"):
            refs.for_quantity("absolute_radiometric")

    def test_refuses_two_standards_for_one_quantity(self) -> None:
        # doc 01 SYS-3 requires a single documented reference source per chain. Two
        # standards for one quantity means the calibration chain is ambiguous, and
        # whichever the loader reached first would decide the absolute scale.
        duplicate = CalibrationReference(
            name="lamp A",
            quantity="absolute_radiometric",
            value=Q_(1.0, "W/(m**2*nm*sr)"),
            relative_uncertainty=0.06,
            traceable_to="NIST FEL scale",
        )
        other = CalibrationReference(
            name="lamp B",
            quantity="absolute_radiometric",
            value=Q_(1.1, "W/(m**2*nm*sr)"),
            relative_uncertainty=0.06,
            traceable_to="NIST FEL scale",
        )

        with pytest.raises(ValueError, match="absolute_radiometric"):
            CalibrationSet.of(duplicate, other)

    def test_iterates_in_a_deterministic_order(self) -> None:
        refs = CalibrationSet.of(
            CalibrationReference(
                name="b",
                quantity="spectral",
                value=Q_(1.0, "dimensionless"),
                relative_uncertainty=0.04,
                traceable_to="Raman cross sections",
            ),
            CalibrationReference(
                name="a",
                quantity="absolute",
                value=Q_(1.0, "dimensionless"),
                relative_uncertainty=0.07,
                traceable_to="Rayleigh cross section",
            ),
        )

        assert tuple(r.quantity for r in refs) == ("absolute", "spectral")
        assert len(refs) == 2

    def test_two_sets_with_the_same_standards_are_equal(self) -> None:
        def build(uncertainty: float) -> CalibrationSet:
            return CalibrationSet.of(
                CalibrationReference(
                    name="Rayleigh scattering in Ar",
                    quantity="absolute",
                    value=Q_(1.0, "dimensionless"),
                    relative_uncertainty=uncertainty,
                    traceable_to="Rayleigh cross section",
                )
            )

        assert build(0.07) == build(0.07)
        assert build(0.07) != build(0.08)
        assert build(0.07) != "not a calibration set"

    def test_rejects_a_reference_with_no_traceability(self) -> None:
        with pytest.raises(ValueError, match="traceable"):
            CalibrationReference(
                name="a lamp",
                quantity="absolute",
                value=Q_(1.0, "dimensionless"),
                relative_uncertainty=0.07,
                traceable_to="",
            )

    def test_rejects_a_reference_whose_uncertainty_is_not_a_fraction(self) -> None:
        with pytest.raises(ValueError, match="uncertainty"):
            CalibrationReference(
                name="a lamp",
                quantity="absolute",
                value=Q_(1.0, "dimensionless"),
                relative_uncertainty=-0.01,
                traceable_to="NIST",
            )


class TestSignal:
    def test_carries_the_stage_of_the_doc_04_7_1_chain_it_lives_at(self) -> None:
        signal = Signal(values=np.asarray([1.0, 2.0]), domain=SignalDomain.PHOTOELECTRON)

        assert signal.domain is SignalDomain.PHOTOELECTRON
        assert signal.n_samples == 2

    def test_derives_a_new_signal_at_the_same_stage(self) -> None:
        # NoiseModel.apply returns a Signal, and a noise source that moved the signal to
        # a different stage of the chain would be a layering violation (doc 04 §1).
        signal = Signal(values=np.asarray([1.0, 2.0]), domain=SignalDomain.ADU)
        noisier = signal.with_values(np.asarray([1.5, 2.5]))

        assert noisier.domain is SignalDomain.ADU
        assert noisier.values[0] == pytest.approx(1.5)

    def test_values_are_copied_and_locked(self) -> None:
        source = np.asarray([1.0, 2.0])
        signal = Signal(values=source, domain=SignalDomain.ADU)
        source[0] = 99.0

        assert signal.values[0] == pytest.approx(1.0)
        with pytest.raises(ValueError, match="read-only"):
            signal.values[0] = 5.0

    def test_a_negative_sample_is_legal_after_offset_subtraction(self) -> None:
        # An ADU trace with the bias removed goes negative on the read-noise tail. A type
        # that rejected it would force every detector model to clip, which biases the
        # photon-transfer curve of V-28 upward at the low end.
        signal = Signal(values=np.asarray([-1.0, 2.0]), domain=SignalDomain.ADU)

        assert signal.values[0] == pytest.approx(-1.0)

    def test_rejects_a_non_finite_sample(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Signal(values=np.asarray([np.nan]), domain=SignalDomain.ADU)

    def test_rejects_an_empty_signal(self) -> None:
        with pytest.raises(ValueError, match="sample"):
            Signal(values=np.asarray([]), domain=SignalDomain.ADU)

    def test_rejects_a_scalar(self) -> None:
        # A detector stage acts on a frame or a trace. A bare scalar would broadcast
        # against everything and silently collapse a whole frame's noise to one draw.
        with pytest.raises(ValueError, match="scalar"):
            Signal(values=np.asarray(1.0), domain=SignalDomain.ADU)

    def test_two_signals_with_the_same_content_are_equal(self) -> None:
        left = Signal(values=np.asarray([1.0]), domain=SignalDomain.ADU)
        right = Signal(values=np.asarray([1.0]), domain=SignalDomain.ADU)

        assert left == right
        assert left != Signal(values=np.asarray([1.0]), domain=SignalDomain.PHOTON)
        assert left != "not a signal"


class TestIdentifiabilityReport:
    def _report(self, params: PlasmaParams, **overrides: object) -> IdentifiabilityReport:
        defaults: dict[str, object] = {
            "at": params,
            "names": ("n_0", "T_e"),
            "eigenvalues": np.asarray([100.0, 1.0]),
            "eigenvectors": np.eye(2),
            "classification": Identifiability.IDENTIFIABLE,
        }
        defaults.update(overrides)
        return IdentifiabilityReport(**defaults)  # type: ignore[arg-type]

    def test_reports_the_condition_number_the_identifiability_map_is_coloured_by(
        self, params: PlasmaParams
    ) -> None:
        # doc 05 §6.2: "The condition number lambda_max/lambda_min is reported for every
        # operating point, producing the identifiability map required by doc 00 S5."
        assert self._report(params).condition_number == pytest.approx(100.0)

    def test_a_singular_information_matrix_has_an_infinite_condition_number(
        self, params: PlasmaParams
    ) -> None:
        report = self._report(
            params,
            eigenvalues=np.asarray([100.0, 0.0]),
            classification=Identifiability.NON_IDENTIFIABLE,
        )

        assert np.isinf(report.condition_number)

    def test_the_weakest_direction_names_the_degeneracy(self, params: PlasmaParams) -> None:
        # doc 05 §6.2: "The eigenvectors name the degeneracies." The expected one is an
        # n_0-T_e correlation, since Gamma_i ~ n_0 sqrt(T_e).
        report = self._report(
            params,
            eigenvalues=np.asarray([100.0, 1.0]),
            eigenvectors=np.asarray([[1.0, 0.6], [0.0, 0.8]]),
        )

        assert report.weakest_direction()["T_e"] == pytest.approx(0.8)

    def test_the_cramer_rao_bound_is_the_best_any_unbiased_estimator_could_do(
        self, params: PlasmaParams
    ) -> None:
        # doc 05 §6.1: Cov >= I^-1. With an identity eigenbasis the bound is 1/lambda.
        bound = self._report(params).cramer_rao_variance

        assert bound[0] == pytest.approx(0.01)
        assert bound[1] == pytest.approx(1.0)

    def test_an_unconstrained_direction_has_an_unbounded_cramer_rao_variance(
        self, params: PlasmaParams
    ) -> None:
        report = self._report(
            params,
            eigenvalues=np.asarray([100.0, 0.0]),
            classification=Identifiability.NON_IDENTIFIABLE,
        )

        assert np.isinf(report.cramer_rao_variance[1])

    def test_efficiency_says_whether_information_is_being_left_on_the_table(
        self, params: PlasmaParams
    ) -> None:
        # doc 05 §6.1 asks exactly this: "is the inversion extracting the information
        # that is present, or is it leaving some on the table?"
        report = self._report(params, posterior_variance=np.asarray([0.02, 1.0]))

        assert report.information_efficiency()[0] == pytest.approx(0.5)
        assert report.information_efficiency()[1] == pytest.approx(1.0)

    def test_efficiency_is_unavailable_without_a_posterior_to_compare_against(
        self, params: PlasmaParams
    ) -> None:
        with pytest.raises(ValueError, match="posterior_variance"):
            self._report(params).information_efficiency()

    def test_rejects_eigenvalues_that_are_not_in_descending_order(
        self, params: PlasmaParams
    ) -> None:
        # lambda_max/lambda_min and weakest_direction both index by position. An
        # unordered spectrum would make both silently name the wrong mode.
        with pytest.raises(ValueError, match="descending"):
            self._report(params, eigenvalues=np.asarray([1.0, 100.0]))

    def test_rejects_a_negative_eigenvalue(self, params: PlasmaParams) -> None:
        # The Fisher information matrix is positive semi-definite by construction, so a
        # negative eigenvalue is an arithmetic failure and not a weak direction.
        with pytest.raises(ValueError, match="negative"):
            self._report(params, eigenvalues=np.asarray([1.0, -1.0]))

    def test_rejects_an_eigenvector_matrix_of_the_wrong_shape(self, params: PlasmaParams) -> None:
        with pytest.raises(ValueError, match="eigenvectors"):
            self._report(params, eigenvectors=np.eye(3))

    def test_rejects_a_spectrum_that_does_not_cover_the_parameters(
        self, params: PlasmaParams
    ) -> None:
        with pytest.raises(ValueError, match="eigenvalues"):
            self._report(params, eigenvalues=np.asarray([1.0]))

    def test_rejects_duplicate_parameter_names(self, params: PlasmaParams) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            self._report(params, names=("n_0", "n_0"))

    def test_rejects_a_report_over_no_parameters(self, params: PlasmaParams) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            self._report(
                params,
                names=(),
                eigenvalues=np.asarray([]),
                eigenvectors=np.zeros((0, 0)),
            )

    def test_rejects_a_non_finite_eigenvalue(self, params: PlasmaParams) -> None:
        # An infinite Fisher eigenvalue is an arithmetic failure, not perfect knowledge.
        with pytest.raises(ValueError, match="finite"):
            self._report(params, eigenvalues=np.asarray([np.inf, 1.0]))

    def test_rejects_a_non_finite_eigenvector(self, params: PlasmaParams) -> None:
        with pytest.raises(ValueError, match="eigenvectors must be finite"):
            self._report(params, eigenvectors=np.asarray([[1.0, np.nan], [0.0, 1.0]]))

    def test_rejects_a_posterior_variance_that_does_not_cover_the_parameters(
        self, params: PlasmaParams
    ) -> None:
        with pytest.raises(ValueError, match="posterior_variance"):
            self._report(params, posterior_variance=np.asarray([0.02]))

    def test_rejects_a_non_positive_posterior_variance(self, params: PlasmaParams) -> None:
        # A zero marginal variance would divide the efficiency ratio by zero and report
        # a perfectly efficient inversion at exactly the point the sampler collapsed.
        with pytest.raises(ValueError, match="positive"):
            self._report(params, posterior_variance=np.asarray([0.0, 1.0]))

    def test_two_reports_at_the_same_point_with_the_same_spectrum_are_equal(
        self, params: PlasmaParams
    ) -> None:
        assert self._report(params) == self._report(params)
        assert self._report(params) != self._report(params, eigenvalues=np.asarray([10.0, 1.0]))
        assert self._report(params) != "not a report"

    def test_a_report_with_the_fitted_variances_differs_from_one_without(
        self, params: PlasmaParams
    ) -> None:
        # Only one of the two can answer doc 05 §6.1's question, so they are not
        # interchangeable even at the same operating point with the same spectrum.
        fitted = self._report(params, posterior_variance=np.asarray([0.02, 1.0]))

        assert fitted != self._report(params)
        assert self._report(params) != fitted
        assert fitted != self._report(params, posterior_variance=np.asarray([0.03, 1.0]))
        assert fitted == self._report(params, posterior_variance=np.asarray([0.02, 1.0]))


class TestIdentifiabilityClassification:
    def test_a_well_conditioned_point_is_identifiable(self) -> None:
        verdict = classify_by_condition_number(10.0, weak_above=1.0e3, non_identifiable_above=1.0e8)

        assert verdict is Identifiability.IDENTIFIABLE

    def test_a_badly_conditioned_point_is_weakly_identifiable(self) -> None:
        verdict = classify_by_condition_number(
            1.0e5, weak_above=1.0e3, non_identifiable_above=1.0e8
        )

        assert verdict is Identifiability.WEAKLY_IDENTIFIABLE

    def test_a_singular_point_is_not_identifiable_at_all(self) -> None:
        verdict = classify_by_condition_number(
            np.inf, weak_above=1.0e3, non_identifiable_above=1.0e8
        )

        assert verdict is Identifiability.NON_IDENTIFIABLE

    def test_refuses_thresholds_that_are_not_ordered(self) -> None:
        with pytest.raises(ValueError, match="weak_above"):
            classify_by_condition_number(10.0, weak_above=1.0e8, non_identifiable_above=1.0e3)

    def test_refuses_a_condition_number_below_one(self) -> None:
        # lambda_max >= lambda_min by construction, so cond < 1 means the caller passed
        # the ratio upside down — and it would classify a singular point as pristine.
        with pytest.raises(ValueError, match="condition number"):
            classify_by_condition_number(0.5, weak_above=1.0e3, non_identifiable_above=1.0e8)


# ── the traceability annotations of doc 08 §9 ───────────────────────────────────────


class TestTraceabilityIdShapes:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("R-SPAT-1", "R-SPAT-1"),
            ("R-ACC-4", "R-ACC-4"),
            ("IF-6", "IF-6"),
            ("SYS-4", "SYS-4"),
            ("LIF-2", "LIF-2"),
            ("OES-6", "OES-6"),
            ("TS-3", "TS-3"),
            ("R-ENV-1", "R-ENV-1"),
            ("R-NON-1", "R-NON-1"),
            ("R-TEMP-3", "R-TEMP-3"),
        ],
    )
    def test_accepts_every_requirement_family_the_documents_use(
        self, given: str, expected: str
    ) -> None:
        assert canonical_requirement_id(given) == expected

    @pytest.mark.parametrize(
        ("given", "expected"),
        [("V-22", "V-22"), ("V-1", "V-01"), ("B-03", "B-03"), ("F-15", "F-15")],
    )
    def test_accepts_the_evidence_families_and_pads_them_as_the_documents_do(
        self, given: str, expected: str
    ) -> None:
        assert canonical_evidence_id(given) == expected

    @pytest.mark.parametrize(
        "typo", ["R-SPAT1", "RSPAT-1", "V22", "r-spat-1", "V-", "V-0", "", "  ", "V-2 2"]
    )
    def test_refuses_a_malformed_id(self, typo: str) -> None:
        # doc 08 §9 generates the traceability matrix from these annotations. A typo that
        # registered silently would put a row in the matrix that no requirement claims
        # and no test covers, which is worse than having no matrix at all.
        with pytest.raises(TraceabilityError):
            canonical_requirement_id(typo)

    def test_refuses_a_family_the_documents_do_not_define(self) -> None:
        with pytest.raises(TraceabilityError, match="ZZ"):
            canonical_requirement_id("ZZ-1")

    def test_refuses_an_evidence_id_where_a_requirement_belongs(self) -> None:
        # "@satisfies('V-22')" is a real mistake: it reads plausibly and produces a
        # requirement row for a test.
        with pytest.raises(TraceabilityError, match="V"):
            canonical_requirement_id("V-22")

    def test_refuses_a_requirement_id_where_evidence_belongs(self) -> None:
        with pytest.raises(TraceabilityError, match="IF"):
            canonical_evidence_id("IF-6")

    def test_the_error_lists_the_families_that_would_have_worked(self) -> None:
        with pytest.raises(TraceabilityError, match="R-SPAT"):
            canonical_requirement_id("R-SPACE-1")


class TestTraceabilityAnnotations:
    def test_records_what_a_class_claims_and_what_verifies_it(self) -> None:
        @satisfies("R-SPAT-1", "R-ACC-4")
        @verified_by("V-22", "V-23")
        class LIFInstrument: ...

        assert requirements_of(LIFInstrument) == ("R-ACC-4", "R-SPAT-1")
        assert evidence_of(LIFInstrument) == ("V-22", "V-23")

    def test_the_decorators_return_the_object_unchanged(self) -> None:
        @satisfies("R-SPAT-1")
        @verified_by("V-22")
        class Thing:
            marker = 7

        assert Thing.marker == 7
        assert Thing().marker == 7

    def test_the_order_of_the_two_decorators_does_not_matter(self) -> None:
        @verified_by("V-22")
        @satisfies("R-SPAT-1")
        class Reversed: ...

        assert requirements_of(Reversed) == ("R-SPAT-1",)
        assert evidence_of(Reversed) == ("V-22",)

    def test_annotates_functions_as_well_as_classes(self) -> None:
        @satisfies("R-ACC-1")
        @verified_by("V-29")
        def phase_shift() -> float:
            return 0.0

        assert requirements_of(phase_shift) == ("R-ACC-1",)
        assert phase_shift() == pytest.approx(0.0)

    def test_a_second_application_adds_rather_than_replaces(self) -> None:
        # Splitting a long claim list across two decorator lines is normal; silently
        # keeping only the last would drop half the matrix.
        @satisfies("R-SPAT-1")
        @satisfies("R-ACC-4")
        @verified_by("V-22")
        class Split: ...

        assert requirements_of(Split) == ("R-ACC-4", "R-SPAT-1")

    def test_duplicate_ids_are_recorded_once(self) -> None:
        @satisfies("R-SPAT-1", "R-SPAT-1")
        @verified_by("V-1", "V-01")
        class Repeated: ...

        assert requirements_of(Repeated) == ("R-SPAT-1",)
        assert evidence_of(Repeated) == ("V-01",)

    def test_an_unannotated_object_claims_nothing(self) -> None:
        class Plain: ...

        assert requirements_of(Plain) == ()
        assert evidence_of(Plain) == ()

    def test_a_subclass_does_not_inherit_its_parents_claims(self) -> None:
        # An inherited @satisfies would credit every subclass with the base's coverage,
        # and the matrix would report requirements as verified by tests that never ran
        # against the subclass.
        @satisfies("R-SPAT-1")
        @verified_by("V-22")
        class Base: ...

        class Derived(Base): ...

        assert requirements_of(Derived) == ()

    def test_refuses_an_annotation_with_no_ids(self) -> None:
        with pytest.raises(TraceabilityError, match="at least one"):

            @satisfies()
            class Empty: ...

    def test_refuses_evidence_with_no_ids(self) -> None:
        # An empty @verified_by reads as verified and is not — which is precisely the
        # state doc 08 §9's gate exists to fail.
        with pytest.raises(TraceabilityError, match="at least one"):

            @verified_by()
            class Empty: ...

    def test_refuses_to_annotate_something_that_is_not_a_class_or_function(self) -> None:
        with pytest.raises(TraceabilityError, match="class or function"):
            satisfies("R-SPAT-1")(object())

    def test_a_malformed_id_fails_at_import_time(self) -> None:
        with pytest.raises(TraceabilityError):

            @satisfies("R-SPAT-01x")
            class Bad: ...


class TestTraceabilityMatrix:
    def test_the_matrix_maps_a_requirement_to_everything_that_claims_it(self) -> None:
        @satisfies("R-SPAT-1")
        @verified_by("V-22")
        class LIFInstrument: ...

        @satisfies("R-SPAT-1")
        @verified_by("V-27")
        class RayTracer: ...

        matrix = traceability_matrix()

        assert set(matrix) == {"R-SPAT-1"}
        assert len(matrix["R-SPAT-1"]) == 2

    def test_the_matrix_keys_are_sorted(self) -> None:
        @satisfies("SYS-4", "R-ACC-1")
        @verified_by("V-22")
        class Multi: ...

        assert tuple(traceability_matrix()) == ("R-ACC-1", "SYS-4")

    def test_the_matrix_is_read_only(self) -> None:
        @satisfies("R-SPAT-1")
        @verified_by("V-22")
        class Thing: ...

        with pytest.raises(TypeError):
            traceability_matrix()["R-ACC-1"] = ()  # type: ignore[index]

    def test_entries_carry_the_fully_qualified_target_name(self) -> None:
        @satisfies("R-SPAT-1")
        @verified_by("V-22")
        class Named: ...

        (entry,) = entries()

        assert entry.target.endswith("Named")
        assert "test_protocols" in entry.target

    def test_an_empty_registry_yields_an_empty_matrix(self) -> None:
        assert entries() == ()
        assert traceability_matrix() == {}


class TestTraceabilityCiGates:
    def test_a_claim_with_no_verification_fails_the_gate(self) -> None:
        # doc 08 §9: "a @satisfies with no @verified_by fails CI".
        @satisfies("R-SPAT-1")
        class Unverified: ...

        (failing,) = unverified_claims()

        assert failing.target.endswith("Unverified")
        assert failing.requirements == ("R-SPAT-1",)

    def test_a_verified_claim_passes_the_gate(self) -> None:
        @satisfies("R-SPAT-1")
        @verified_by("V-22")
        class Verified: ...

        assert unverified_claims() == ()

    def test_evidence_without_a_claim_is_not_a_failure(self) -> None:
        # A test-only helper may name the tests it belongs to without claiming to
        # satisfy a requirement; doc 08 §9 gates the other direction only.
        @verified_by("V-22")
        class Helper: ...

        assert unverified_claims() == ()

    def test_a_requirement_nothing_claims_fails_the_gate(self) -> None:
        # doc 08 §9: "a requirement with no @satisfies ... fails CI". The requirement
        # list comes from doc 01 §6, which this package does not hold, so the catalogue
        # is supplied by the caller.
        @satisfies("R-SPAT-1")
        @verified_by("V-22")
        class Covered: ...

        assert uncovered_requirements(("R-SPAT-1", "R-ACC-4")) == ("R-ACC-4",)

    def test_a_fully_covered_catalogue_reports_nothing(self) -> None:
        @satisfies("R-SPAT-1")
        @verified_by("V-22")
        class Covered: ...

        assert uncovered_requirements(("R-SPAT-1",)) == ()

    def test_the_catalogue_itself_is_checked_for_typos(self) -> None:
        with pytest.raises(TraceabilityError):
            uncovered_requirements(("R-SPAT-1", "R-SPAT1"))


# ── the four protocols ──────────────────────────────────────────────────────────────


class TestForwardSolverProtocol:
    def test_a_complete_implementation_satisfies_it(self) -> None:
        assert isinstance(FakeSolver(), ForwardSolver)

    def test_a_solver_missing_its_cost_estimate_does_not(self) -> None:
        # doc 10 §6 schedules campaigns from cost_estimate; a solver that cannot be
        # costed cannot be queued, and finding that out mid-sweep is too late.
        assert not isinstance(SolverWithoutCostEstimate(), ForwardSolver)

    def test_a_solver_whose_flux_returns_a_bare_number_is_rejected_by_the_type_checker(
        self,
    ) -> None:
        # isinstance checks method *names*; the signature contract is mypy's. The ignore
        # below is the assertion — warn_unused_ignores is part of --strict, so if the
        # protocol stopped constraining the return type this line would fail the build.
        assert isinstance(SolverReturningBareFloatFlux(), ForwardSolver)
        _requires_solver(SolverReturningBareFloatFlux())  # type: ignore[arg-type]

    def test_a_solver_that_demands_a_time_grid_is_rejected_by_the_type_checker(
        self,
    ) -> None:
        # The reason ForwardSolver.solve was widened from doc 08 §4's literal
        # ``t: TimeGrid``. Parameter types are contravariant, so a solver that *insists*
        # on a TimeGrid cannot stand in for one that accepts None — and L0 has no time
        # dependence at all (doc 03 §1), so demanding one would make the analytic level
        # unable to implement the contract every level is supposed to share (doc 00 E1).
        # The ignore is the assertion: warn_unused_ignores is part of --strict.
        assert isinstance(SolverDemandingATimeGrid(), ForwardSolver)
        _requires_solver(SolverDemandingATimeGrid())  # type: ignore[arg-type]

    def test_a_steady_solve_returns_a_steady_state(self, params: PlasmaParams) -> None:
        solver: ForwardSolver = FakeSolver()

        state = solver.solve(params, None)

        assert state.is_steady is True

    def test_the_contract_composes_end_to_end(self, params: PlasmaParams) -> None:
        solver: ForwardSolver = FakeSolver()
        solver.configure(SolverConfig(values={"n_ppc": 1000}))
        state = solver.solve(params, TimeGrid.uniform(duration=Q_(10.0, "ns"), n_points=3))
        flux = solver.flux(state, 0.0)

        assert state.fidelity is Fidelity.L0
        assert flux.fidelity is Fidelity.L0
        assert float(flux.mean_impact_energy_joules) == pytest.approx(4.0e-17)

    def test_the_solver_reports_a_costable_and_citable_identity(self) -> None:
        solver: ForwardSolver = FakeSolver()

        assert solver.fidelity() is Fidelity.L0
        assert solver.cost_estimate(SolverConfig(values={})).is_measured is False
        assert solver.metadata().citations


class TestInstrumentProtocol:
    def test_a_complete_implementation_satisfies_it(self) -> None:
        assert isinstance(FakeInstrument(), Instrument)

    def test_an_instrument_without_the_detection_gate_does_not(self) -> None:
        # doc 01 IF-6 is a requirement, not a caveat: a channel below its floor must
        # contribute no likelihood term rather than a noisy one.
        assert not isinstance(InstrumentWithoutDetectionGate(), Instrument)

    def test_an_instrument_whose_forward_is_noisy_is_rejected_by_the_type_checker(self) -> None:
        # doc 04 §9 splits forward (noiseless Observable) from observe (noisy
        # Measurement). Returning a Measurement from forward would feed an uncertainty
        # into the likelihood twice and nothing at runtime would say so.
        assert isinstance(InstrumentReturningMeasurementFromForward(), Instrument)
        _requires_instrument(InstrumentReturningMeasurementFromForward())  # type: ignore[arg-type]

    def test_forward_and_observe_agree_because_they_share_a_code_path(
        self, state: PlasmaState, window: AcquisitionWindow
    ) -> None:
        # doc 04 §9 / doc 08 §4: "Both come from the same code path", which is the
        # invariant this test holds them to.
        instrument: Instrument = FakeInstrument()
        prediction = instrument.forward(state, window)
        observation = instrument.observe(state, window)

        assert observation.as_observable() == prediction

    def test_an_observation_is_calibrated_with_the_estimated_response(
        self, state: PlasmaState, window: AcquisitionWindow
    ) -> None:
        instrument: Instrument = FakeInstrument()

        assert instrument.observe(state, window).is_inverse_crime is False

    def test_the_likelihood_is_a_bare_log_probability(
        self, state: PlasmaState, window: AcquisitionWindow
    ) -> None:
        instrument: Instrument = FakeInstrument()
        prediction = instrument.forward(state, window)
        observation = instrument.observe(state, window)
        value: LogProb = instrument.likelihood(observation, prediction)

        assert value == pytest.approx(0.0)

    def test_the_channel_gates_itself_on_its_own_detection_floor(
        self, params: PlasmaParams
    ) -> None:
        instrument: Instrument = FakeInstrument()
        blind = params.replace(n_0=Q_(1.0e16, "m**-3"))

        assert instrument.is_informative(params) is True
        assert instrument.is_informative(blind) is False

    def test_calibrating_from_a_reference_set_yields_an_estimated_response(self) -> None:
        instrument: Instrument = FakeInstrument()
        refs = CalibrationSet.of(
            CalibrationReference(
                name="AOM offset over a known path",
                quantity="phase",
                value=Q_(1.0, "rad"),
                relative_uncertainty=0.03,
                traceable_to="machined path length",
            )
        )
        calibration = instrument.calibrate(refs)

        assert calibration.state is CalibrationState.ESTIMATED
        assert calibration.is_inverse_crime is False


class TestInverseEngineProtocol:
    def test_a_complete_implementation_satisfies_it(self) -> None:
        assert isinstance(FakeEngine(), InverseEngine)

    def test_an_engine_that_cannot_report_a_null_space_does_not(self) -> None:
        # doc 05 §6 is "what elevates the work from an inversion to an analysis of an
        # inversion", and doc 00 S5 requires the identifiability map.
        assert not isinstance(EngineWithoutIdentifiability(), InverseEngine)

    def test_a_fake_forward_model_satisfies_the_model_contract(self) -> None:
        assert isinstance(FakeForwardModel(), ForwardModel)

    def test_the_contract_composes_end_to_end(
        self, params: PlasmaParams, state: PlasmaState, window: AcquisitionWindow
    ) -> None:
        instrument = FakeInstrument()
        data = MeasurementSet.of(instrument.observe(state, window))
        engine = FakeEngine()
        engine.configure(InverseConfig(values={"draws": 4000, "chains": 4}))
        posterior = engine.fit(data, FakeForwardModel())

        assert posterior.n_chains == 2
        assert engine.model_fidelity is Fidelity.L3
        assert engine.diagnostics().is_clean() is True

    def test_the_engine_reports_identifiability_at_a_named_operating_point(
        self, params: PlasmaParams
    ) -> None:
        engine: InverseEngine = FakeEngine()
        report = engine.identifiability(params)

        assert report.at is params
        assert report.classification is Identifiability.IDENTIFIABLE


class TestNoiseModelProtocol:
    def test_a_complete_implementation_satisfies_it(self) -> None:
        assert isinstance(FakeNoise(), NoiseModel)

    def test_a_noise_source_that_cannot_be_switched_off_does_not(self) -> None:
        # doc 04 §8 V-30: "Noise sources switch off cleanly (each in isolation
        # reproduces the noiseless limit)", and the manifest of doc 08 §6 lists
        # enabled_sources explicitly.
        assert not isinstance(NoiseWithoutSwitch(), NoiseModel)

    def test_a_disabled_source_reproduces_the_noiseless_limit_exactly(self) -> None:
        noise: NoiseModel = FakeNoise(enabled=False)
        signal = Signal(values=np.asarray([100.0, 200.0]), domain=SignalDomain.PHOTOELECTRON)

        assert noise.apply(signal, default_rng(seed=0)) == signal
        assert noise.enabled() is False

    def test_an_enabled_source_perturbs_the_signal_reproducibly(self) -> None:
        # doc 00 E3: bit-for-bit reproducibility given the seed. The generator is passed
        # in rather than owned so that the run's seed derivation stays in one place.
        noise: NoiseModel = FakeNoise()
        signal = Signal(values=np.asarray([100.0, 200.0]), domain=SignalDomain.PHOTOELECTRON)
        first = noise.apply(signal, default_rng(seed=20260804))
        again = noise.apply(signal, default_rng(seed=20260804))

        assert first == again
        assert first != signal

    def test_the_variance_stays_at_the_same_stage_of_the_chain(self) -> None:
        noise: NoiseModel = FakeNoise()
        signal = Signal(values=np.asarray([100.0]), domain=SignalDomain.PHOTOELECTRON)

        assert noise.variance(signal).domain is SignalDomain.PHOTOELECTRON


class TestReprs:
    """Every value type prints what a reviewer reading a traceback needs.

    Not cosmetic. These objects appear in pytest failure output, in MLflow parameter
    logs and in the notebook a reviewer opens against a run's artifacts, and the default
    dataclass repr for an array-carrying type is a wall of numbers with the one
    distinguishing field buried in it.
    """

    def test_a_citation_shows_the_reference_not_just_the_key(self, citation: Citation) -> None:
        assert "Birdsall" in repr(citation)

    def test_a_detection_floor_shows_the_requirement_behind_it(self, floor: DetectionFloor) -> None:
        assert "IF-6" in repr(floor)
        assert "n_0" in repr(floor)

    def test_solver_metadata_shows_name_and_version(self, citation: Citation) -> None:
        meta = SolverMetadata(
            name="vpl.physics.kinetic.pic1d3v", version="0.3.1", citations=(citation,)
        )

        assert "pic1d3v" in repr(meta)
        assert "0.3.1" in repr(meta)

    def test_instrument_metadata_shows_the_channel_and_its_floor(
        self, citation: Citation, floor: DetectionFloor
    ) -> None:
        meta = InstrumentMetadata(
            instrument_id="interf",
            name="CO2 heterodyne interferometer",
            version="0.1.0",
            citations=(citation,),
            detection_floor=floor,
        )

        assert "interf" in repr(meta)

    def test_a_config_lists_its_keys_and_not_its_values(self) -> None:
        # A manifest block can be large; the keys say which block this is, which is what
        # a "configure() got the wrong section" traceback needs.
        assert repr(SolverConfig(values={"n_ppc": 1000})) == "SolverConfig(n_ppc)"
        assert repr(InverseConfig(values={})) == "InverseConfig(empty)"

    def test_a_flux_shows_its_fidelity_and_magnitude(self, argon: Species) -> None:
        # doc 03 §6 / benchmark B-03: which level produced a curve is the whole point of
        # the comparison, so it goes in the repr rather than being looked up.
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.asarray(4.0e2),
            particle_flux_toward_wall_per_m2_s=np.asarray(1.0e19),
            fidelity=Fidelity.L2,
        )

        assert "L2" in repr(flux)
        assert "steady" in repr(flux)

    def test_a_time_resolved_flux_says_how_many_instants(self, argon: Species) -> None:
        flux = IonEnergyFlux(
            position=Q_(0.0, "m"),
            species=argon,
            energy_flux_toward_wall_watt_per_m2=np.ones(4),
            particle_flux_toward_wall_per_m2_s=np.ones(4),
            fidelity=Fidelity.L2,
            time=TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=4),
        )

        assert "4 times" in repr(flux)

    def test_a_cost_shows_its_peak_memory_when_it_has_one(self) -> None:
        cost = CostEstimate.estimated(
            wall_clock=Q_(5.0, "s"),
            device=Device.GPU,
            source="doc 10 §3.1",
            peak_memory=Q_(2.0, "GB"),
        )

        assert "peak" in repr(cost)

    def test_a_calibration_reference_shows_its_uncertainty(self) -> None:
        reference = CalibrationReference(
            name="NIST FEL tungsten-halogen lamp",
            quantity="absolute_radiometric",
            value=Q_(1.0, "W/(m**2*nm*sr)"),
            relative_uncertainty=0.06,
            traceable_to="NIST FEL scale",
        )

        assert "6.00%" in repr(reference)

    def test_a_calibration_set_lists_what_it_certifies(self) -> None:
        refs = CalibrationSet.of(
            CalibrationReference(
                name="Rayleigh scattering in Ar",
                quantity="absolute",
                value=Q_(1.0, "dimensionless"),
                relative_uncertainty=0.07,
                traceable_to="Rayleigh cross section",
            )
        )

        assert repr(refs) == "CalibrationSet(absolute)"

    def test_a_calibration_shows_which_response_was_applied(self) -> None:
        calibration = Calibration(
            instrument_id="thomson",
            coefficients={"absolute": 1.03},
            relative_uncertainty={"absolute": 0.07},
            state=CalibrationState.ESTIMATED,
            reference="Rayleigh scattering in Ar",
        )

        assert "estimated" in repr(calibration)

    def test_a_signal_shows_the_stage_of_the_chain_it_sits_at(self) -> None:
        signal = Signal(values=np.ones(3), domain=SignalDomain.PHOTOELECTRON)

        assert repr(signal) == "Signal(photoelectron, n=3, shape=(3,))"

    def test_an_identifiability_report_shows_the_verdict_and_the_condition_number(
        self, params: PlasmaParams
    ) -> None:
        report = IdentifiabilityReport(
            at=params,
            names=("n_0", "T_e"),
            eigenvalues=np.asarray([100.0, 1.0]),
            eigenvectors=np.eye(2),
            classification=Identifiability.WEAKLY_IDENTIFIABLE,
        )

        assert "weakly_identifiable" in repr(report)
        assert "100" in repr(report)


class TestProtocolExports:
    def test_every_name_the_package_advertises_is_importable(self) -> None:
        import vpl.core.protocols as protocols

        missing = [name for name in protocols.__all__ if not hasattr(protocols, name)]

        assert missing == []

    def test_nothing_a_submodule_makes_public_is_left_unexported(self) -> None:
        # A supporting type reachable only as `vpl.core.protocols.forward.CostBasis`
        # would make the import path an implementation detail, and the first plugin to
        # depend on it would pin the module layout doc 08 §1 principle 7 expects to move.
        import importlib

        import vpl.core.protocols as protocols

        submodules = ("config", "forward", "instrument", "inverse", "metadata", "noise")
        unexported = [
            name
            for module in submodules
            for name in importlib.import_module(f"vpl.core.protocols.{module}").__all__
            if name not in protocols.__all__
        ]

        assert unexported == []

    def test_no_name_is_advertised_twice(self) -> None:
        import vpl.core.protocols as protocols

        assert len(set(protocols.__all__)) == len(protocols.__all__)


def test_a_sequence_of_observables_is_what_a_forward_model_predicts(
    params: PlasmaParams, state: PlasmaState, window: AcquisitionWindow
) -> None:
    """The engine never sees an instrument — doc 04 §1's layering, at the seam."""
    data = MeasurementSet.of(FakeInstrument().observe(state, window))
    model: ForwardModel = FakeForwardModel()
    predictions: Sequence[Observable] = model.predict(params, data)

    assert len(predictions) == len(data)
    assert model.fidelity() is Fidelity.L3
    assert model.log_likelihood(params, data) < 0.0
