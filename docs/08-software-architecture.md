# 08 — Software Architecture

Version 1.0 · Status: **Baseline** · Owner: Danushika N

---

## 1. Principles

| # | Principle | Consequence |
|---|---|---|
| 1 | **Contracts before implementations** | Every solver, instrument and inference engine implements a `Protocol`. The core depends only on protocols |
| 2 | **The core contains only what does not already exist** | Build/buy table §2 is binding; deviating requires an ADR |
| 3 | **Plugins, not branches** | New capability arrives as an installable package discovered via entry points — never by editing the core |
| 4 | **Configuration is data, not code** | One declarative manifest per experiment; no parameters in source |
| 5 | **Layers do not leak** | `F₄` cannot see plasma (doc 04 §1); the inverse cannot see truth (doc 07 §3) |
| 6 | **Everything is provenanced** | Every artifact carries manifest hash, commit, seed, environment |
| 7 | **Small focused modules** | 200–400 lines typical, 800 maximum |

---

## 2. Build vs buy

Binding. Deviation requires an ADR with justification.

| Capability | Decision | Choice |
|---|---|---|
| Arrays, linear algebra | Buy | NumPy, SciPy |
| Sparse / parallel linear algebra | Buy | PETSc (via `petsc4py`) |
| FEM / PDE | Buy | FEniCSx (`dolfinx`) |
| Mesh generation | Buy | Gmsh |
| Automatic differentiation | Buy | JAX (ADR-002) |
| EEDF / Boltzmann solver | Buy | BOLSIG+ or `bolos` |
| Cross-section data | Buy | LXCat |
| Atomic data | Buy | NIST ASD, OpenADAS, ChiantiPy |
| Ray tracing | Buy | Raysect |
| Gaussian processes | Buy | GPyTorch |
| MCMC / HMC | Buy | NumPyro; PyMC as cross-check |
| Nested sampling / SMC | Buy | `dynesty` |
| Data ensemble filters | Buy | `filterpy` / custom thin layer |
| Configuration | Buy | Hydra + OmegaConf |
| Experiment tracking | Buy | MLflow (metrics) + DVC (data) |
| Storage | Buy | HDF5 (`h5py`), Zarr, Parquet |
| Plotting | Buy | Matplotlib + SciencePlots |
| Docs | Buy | MkDocs Material + mkdocstrings |
| Testing | Buy | pytest, Hypothesis |
| **1D3V electrostatic PIC kernel** | **Build** | ADR-003 — textbook algorithm, easier to verify and couple than adapting an EM 3-D code; Smilei retained as independent cross-check |
| **Detector / digitiser models** | **Build** | Every experiment differs; no library exists at this fidelity |
| **CR model assembly** | **Build** | Thin layer over bought atomic data |
| **Instrument abstraction layer** | **Build** | The core contribution |
| **Inverse framework** | **Build** | The core contribution |
| **Validation / benchmark engine** | **Build** | The core contribution |
| **Experiment manifest language** | **Build** | Thin layer over Hydra |
| **Publication engine** | **Build** | Thin layer over Matplotlib |

---

## 3. Package layout

```
astrophage-vpl/
├── packages/
│   ├── vpl-core/                     # protocols, registry, data model, units, provenance
│   │   └── src/vpl/core/
│   │       ├── protocols/            # Instrument, ForwardSolver, InverseEngine, NoiseModel…
│   │       ├── state/                # PlasmaState, Measurement, Posterior, Artifact
│   │       ├── units/                # pint-backed dimensional safety
│   │       ├── registry/             # entry-point plugin discovery
│   │       ├── provenance/           # manifest hashing, environment capture
│   │       └── params/               # the parameter registry (doc 02 §12)
│   │
│   ├── vpl-physics/                  # F₁ — doc 03
│   │   └── src/vpl/physics/
│   │       ├── analytic/             # L0
│   │       ├── fluid/                # L1  (FEniCSx)
│   │       ├── kinetic/              # L2  (PIC-MCC)
│   │       ├── surrogate/            # L3  (GP / neural operator)
│   │       ├── collisions/           # LXCat loading, MCC, null-collision
│   │       └── flux/                 # the Γ_E functional
│   │
│   ├── vpl-optics/                   # F₃ — doc 04 §6
│   ├── vpl-instruments/              # F₂ + instrument assembly
│   │   └── src/vpl/instruments/
│   │       ├── oes/                  # CR model, emissivity, spectrograph
│   │       ├── lif/                  # rate equations, Zeeman, saturation
│   │       ├── thomson/              # Selden spectrum, Rayleigh calibration
│   │       ├── interferometry/       # phase, unwrapping, vibration
│   │       ├── imaging/              # gated ICCD imaging
│   │       └── reference/            # simulated Langmuir probe, RFEA (doc 07 §5.3)
│   │
│   ├── vpl-detectors/                # F₄ — doc 04 §7
│   ├── vpl-inverse/                  # doc 05
│   │   └── src/vpl/inverse/
│   │       ├── likelihood/           # per-channel, asynchronous
│   │       ├── priors/
│   │       ├── engines/              # MAP, Laplace, NUTS, SMC, EnKF, PF
│   │       ├── identifiability/      # FIM, profile likelihood, Sobol
│   │       └── design/               # optimal experiment design
│   │
│   ├── vpl-uq/                       # doc 06 — budget, propagation, calibration tests
│   ├── vpl-validation/               # doc 07 — sealed truth, benchmarks, metrics
│   ├── vpl-experiment/               # manifest → pipeline → artifacts
│   ├── vpl-publish/                  # publication-quality figures & reports
│   └── vpl-app/                      # interactive research interface
│
├── plugins/                          # demonstrating the extension mechanism
│   ├── vpl-plugin-smilei/            # alternative PIC backend (cross-check)
│   ├── vpl-plugin-efish/             # Tier-2 diagnostic (doc 01 §4.2)
│   └── vpl-plugin-mock-hardware/     # proves doc 00 E2 — real instrument, same protocol
│
├── experiments/                      # manifests, version-controlled
├── benchmarks/                       # benchmark definitions
├── docs/
└── refs/
```

**Why a monorepo of separately-installable packages.** `pip install vpl-core vpl-inverse`
must work without dragging in FEniCSx and Gmsh — a user who only wants to run the inversion
on their own data should not need a PDE stack. The separation is enforced by dependency tests
in CI, because it decays instantly otherwise.

---

## 4. Core contracts

```python
# ─── Forward physics ──────────────────────────────────────────────────────────
class ForwardSolver(Protocol):
    def configure(self, cfg: SolverConfig) -> None: ...
    def solve(self, params: PlasmaParams, t: TimeGrid) -> PlasmaState: ...
    def flux(self, state: PlasmaState, z: float) -> IonEnergyFlux: ...
    def fidelity(self) -> Fidelity: ...            # L0 | L1 | L2 | L3
    def cost_estimate(self, cfg: SolverConfig) -> CostEstimate: ...
    def metadata(self) -> SolverMetadata: ...

# ─── Instruments (virtual AND real) ───────────────────────────────────────────
class Instrument(Protocol):
    def configure(self, cfg: InstrumentConfig) -> None: ...
    def calibrate(self, refs: CalibrationSet) -> Calibration: ...
    def forward(self, state: PlasmaState, w: AcquisitionWindow) -> Observable: ...   # noiseless
    def observe(self, state: PlasmaState, w: AcquisitionWindow) -> Measurement: ...  # noisy
    def likelihood(self, obs: Measurement, pred: Observable) -> LogProb: ...
    def is_informative(self, state_guess: PlasmaParams) -> bool: ...   # detection-floor gate
    def metadata(self) -> InstrumentMetadata: ...

# ─── Inference ────────────────────────────────────────────────────────────────
class InverseEngine(Protocol):
    def configure(self, cfg: InverseConfig) -> None: ...
    def fit(self, data: MeasurementSet, model: ForwardModel) -> Posterior: ...
    def diagnostics(self) -> SamplerDiagnostics: ...
    def identifiability(self, at: PlasmaParams) -> IdentifiabilityReport: ...

# ─── Noise ────────────────────────────────────────────────────────────────────
class NoiseModel(Protocol):
    def apply(self, signal: Signal, rng: Generator) -> Signal: ...
    def variance(self, signal: Signal) -> Signal: ...
    def enabled(self) -> bool: ...      # every source individually switchable (doc 04 §8 V-30)
```

**`Instrument.forward` and `Instrument.observe` share a code path** with noise and calibration
error as switchable stages (doc 04 §9). They cannot drift apart, which would otherwise be an
invisible and fatal bug class.

**`is_informative` implements doc 01 IF-6**: a channel below its detection floor contributes
no likelihood term rather than a noisy one.

---

## 5. Parameter registry and units

Every physical quantity is dimensional. `pint`-backed quantities are used at module
boundaries; raw arrays only inside hot loops, with the unit contract asserted at entry and
exit.

The parameter registry (doc 02 §12) is loaded at startup and is the sole source of numeric
defaults. A lint rule fails CI on any numeric literal in physics code outside a registry file
or a named constant. **The `ASSUMED`-class count is emitted as a CI metric** and its increase
fails the build (doc 00 C1).

---

## 6. Experiment manifest

One file, one experiment, fully reproducible.

```yaml
experiment:
  name: b02-reference-operating-point
  description: Closed-loop validation at RP-1, honest tier
  tier: T2
  seed: 20260804

plasma:
  gas: argon
  pressure: {value: 5.0, units: mTorr}
  n0: {value: 1.0e17, units: m^-3}
  Te: {value: 3.0, units: eV}
  bias: {mode: dc, value: -250.0, units: V}

forward:
  solver: vpl.physics.kinetic.pic1d3v      # truth generator
  mesh: {dz: lambda_D/2, grading: wall_refined_A}
  n_ppc: 1000

instruments:
  - {id: oes,     enabled: true,  config: configs/instruments/oes_iccd.yaml}
  - {id: lif,     enabled: true,  config: configs/instruments/lif_ecdl.yaml}
  - {id: thomson, enabled: true,  config: configs/instruments/thomson_yag.yaml}
  - {id: interf,  enabled: true,  config: configs/instruments/interf_co2.yaml}

noise:
  enabled_sources: [N1, N2, N3, N5, N6, N7, N8, N9, N12, N14, N15, N17, N18]
  calibration: estimated            # NOT 'true' — doc 04 §7.3

inverse:
  model: vpl.physics.surrogate.gp          # deliberately ≠ forward.solver (doc 05 §7.1)
  mesh: {dz: lambda_D/3, grading: wall_refined_B}
  engine: numpyro_nuts
  draws: 4000
  chains: 4
  parameters: {control: all, nuisance: all, discrepancy: {n_basis: 12}}

validation:
  seal_truth: true
  metrics: [rel_error, coverage, crps, wasserstein_iedf, fim_condition]
  n_repeats: 200

outputs:
  artifacts: [posterior, diagnostics, identifiability, error_budget]
  figures: [flux_profile, iedf_comparison, reliability_diagram, fim_spectrum]
  report: true
```

Running it:

```bash
vpl run experiments/b02-reference-operating-point.yaml
vpl reproduce <run-id>          # re-executes bit-for-bit from the archived manifest
vpl compare <run-id-a> <run-id-b>
```

---

## 7. Data model and storage

| Artifact | Format | Rationale |
|---|---|---|
| Plasma state fields | HDF5 (chunked, compressed) | Large, hierarchical, standard |
| Particle data (PIC) | HDF5 with subsampling | Full particle dumps are enormous; subsample deterministically by seed |
| Measurements | HDF5, one group per instrument | Carries per-sample timestamp, acquisition window, phase bin, uncertainty (doc 01 SYS-4) |
| Posterior samples | Zarr (chunked, cloud-ready) | Large, appended incrementally |
| Metrics / scalars | Parquet | Queryable across runs |
| Manifests, provenance | YAML + JSON sidecar | Human-readable and diffable |

**Never CSV for simulation output.** No dtype, no units, no metadata, no compression, no
chunking — every one of which the project needs.

Every artifact embeds:

```
manifest_sha256, git_commit, git_dirty, seed, environment_lock_hash,
created_utc, vpl_version, solver_versions, tier
```

---

## 8. Testing

| Layer | Tool | Target |
|---|---|---|
| Unit | pytest | ≥ 80 % core coverage |
| Property-based | Hypothesis | Invariants: positivity, conservation, monotonicity, unit consistency |
| Physics verification | pytest + MMS harness | doc 03 §7, doc 04 §8 |
| Statistical | pytest + SBC harness | doc 06 §7 |
| Integration | pytest | Full manifest → artifact round trip |
| Regression | metric store + threshold check | doc 07 §7 |
| Dependency isolation | import-graph test | `vpl-inverse` must not import `vpl-physics.fluid` |

**Physics tests and software tests are reported separately.** 100 % line coverage with a
first-order solver that was designed to be second-order is a passing test suite and a broken
program.

---

## 9. Traceability

Requirements (doc 01) are annotated in source:

```python
@satisfies("R-SPAT-1", "R-ACC-4")
@verified_by("V-22", "V-23")
class LIFInstrument:
    ...
```

The traceability matrix (doc 01 §6) is **generated** from these annotations. A requirement with
no `@satisfies`, or a `@satisfies` with no `@verified_by`, fails CI. Hand-maintained
traceability matrices are always stale; generated ones cannot be.

---

## 10. Plugin mechanism

Standard Python entry points — no bespoke plugin system (doc 00 §6, reimplementation trap).

```toml
# a third-party package's pyproject.toml
[project.entry-points."vpl.instruments"]
efish = "vpl_plugin_efish:EFISHInstrument"

[project.entry-points."vpl.solvers"]
smilei = "vpl_plugin_smilei:SmileiSolver"
```

`pip install vpl-plugin-efish` makes it available to every manifest with no core change. The
three bundled plugins exist to prove the mechanism works, `vpl-plugin-mock-hardware` in
particular being the demonstration of doc 00 E2.

---

## 11. Interactive interface

Deliberately last. It visualises stored artifacts; it does not compute (doc 00 §6, demo trap).

| Panel | Content |
|---|---|
| Scenario designer | Edit a manifest with live validation |
| Forward viewer | Sheath profiles, fields, IEDF, ion trajectories, time animation |
| Virtual instrument viewer | Synthetic spectra, LIF scans, interferograms, raw detector frames |
| Reconstruction | Recovered flux with credible bands, against sealed truth |
| Sensitivity | Live sliders re-querying the surrogate (ms latency) |
| Ablation | Toggle channels; watch the CI inflate |
| Information | Per-channel entropy contribution; FIM spectrum |
| Identifiability map | Envelope coloured by condition number |
| Comparison | Optical vs simulated RFEA vs simulated probe vs truth |
| Report | One-click PDF |

**Because the surrogate (L3) evaluates in milliseconds, the sensitivity panel is genuinely
interactive** — a slider moves and the reconstruction updates. That is only possible because
of the L3 decision in doc 03 §5, which was made for inference-cost reasons and happens to pay
for the demo as well.

Stack: FastAPI backend over the same artifact store; React + Plotly + VTK.js frontend. The
backend exposes exactly the same operations as the CLI, so nothing is demo-only.

---

## 12. Infrastructure

| Concern | Choice |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Environment | micromamba (conda-forge; FEniCSx/PETSc are painful under pip) |
| Container | Docker + devcontainer; CUDA base image for the A4000 |
| Build | `hatch` per package; CMake for the PIC kernel if C++ |
| CI | GitHub Actions |
| Lint / format | Ruff, Black, `clang-format`, `mypy --strict` on `vpl-core` |
| Docs | MkDocs Material, mkdocstrings, Mermaid |
| Tracking | MLflow (metrics), DVC (data) |

---

## 13. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Twelve packages, four core protocols, entry-point plugins, manifest-driven execution, generated traceability. |
