# 00 — Project Charter

**Astrophage VPL — Virtual Plasma Laboratory for Physics-Constrained Inverse Sheath Diagnostics**

Version 1.0 · Status: **Baseline** · Owner: Team Astrophage

---

## 1. Identity

### 1.1 What we are building

A **reference computational framework** for developing, verifying, validating, benchmarking
and eventually deploying physics-constrained inverse optical diagnostics of plasma sheaths.

### 1.2 What we are deliberately *not* calling it

| Rejected name | Why it is wrong |
|---|---|
| Digital twin | A twin mirrors a *physical asset*. We have no asset. The term promises a hardware coupling we cannot honour, and a reviewer who knows the term will notice. |
| Plasma simulator | The simulation is a *subsystem*, not the product. Calling it a simulator invites comparison against Smilei, WarpX and COMSOL — comparisons we would lose and do not need. |
| AI for plasma | The inference is physics-constrained, not learned. Framing it as ML invites the "your model has no physics" critique that is the single most common way projects like this die. |

The accurate description is **inverse diagnostics framework**. The plasma simulation exists
only because the inverse algorithm requires a physically correct forward operator.

### 1.3 One-sentence claim

> We built the software infrastructure required to develop, verify, validate, benchmark and
> deploy physics-constrained inverse optical diagnostics for plasma sheaths — and we
> demonstrated, on a closed-loop synthetic laboratory with hidden ground truth, the
> conditions under which ion energy flux is and is not recoverable.

Note the second clause. **Reporting where the method fails is a feature, not a hedge.** A
framework that only reports successes is a demo; one that maps its own failure boundary is
an instrument.

---

## 2. The scientific question

Let `x` denote the plasma state and `y` the vector of instrument observations. The forward
operator is

```
y = F(x) + ε ,        F = F₄ ∘ F₃ ∘ F₂ ∘ F₁
```

with `F₁` the plasma/sheath physics, `F₂` the emission and scattering response, `F₃` the
optical transport, `F₄` the detector and digitiser chain, and `ε` the stochastic
measurement noise.

The quantity of interest is a **functional of the state**, not a component of it:

```
Γ_E(z, t) = ∫ E · v_z · f_i(z, v, t) d³v          [W · m⁻²]
```

the ion energy flux through the plane at `z`, obtained by integrating the ion distribution
function `f_i` weighted by kinetic energy and normal velocity.

`F` is **not invertible**: it is non-linear, it maps a high-dimensional state to a
lower-dimensional observation set, and it has a non-trivial null space. Therefore the
problem is posed as regularised Bayesian inversion,

```
p(x | y) ∝ p(y | x) · p(x)
Γ̂_E     = Γ_E( E[x | y] )   with credible intervals from the posterior
```

where `p(x)` encodes sheath theory as a physical prior rather than as a hard constraint.

**Everything in this project exists to answer one question: for which regions of the
operating space is `Γ_E` identifiable from `y`, and to what precision?**

---

## 3. Scope

### 3.1 In scope

- Forward physics hierarchy: analytic sheath models → fluid → kinetic (PIC), behind one interface.
- Forward measurement models for four diagnostics: OES, LIF, Thomson scattering, interferometry.
- A fully parameterised instrument-reality layer (optics, detectors, electronics, timing, calibration).
- A physics-constrained inverse solver with pluggable regularisation and pluggable optimisers/samplers.
- Full uncertainty propagation and a decomposed error budget.
- Identifiability and observability analysis (Fisher information, sensitivity, null-space characterisation).
- Verification (MMS, convergence, conservation) separated cleanly from validation.
- A benchmark suite with defined scenarios, ablations and robustness sweeps.
- Reproducible experiment specification, execution, archival and publication-figure generation.
- An interactive research interface for exploring the framework live.

### 3.2 Explicitly out of scope

| Out of scope | Rationale |
|---|---|
| Building any physical hardware | No lab access. Constraint, stated up front, designed around. |
| Acquiring real experimental data | Same. Validation is closed-loop against synthetic ground truth; this is the standard methodology for inverse-method development prior to lab access. |
| Novel numerical methods or new physics | Hard constraint. Every method used must be citable to prior literature. |
| Writing our own FEM solver, PIC engine, mesh generator, ray tracer, sampler or linear algebra | Mature validated open source exists. Rewriting it is negative-value work and reviewers know it. |
| Claiming experimental validation | We will claim *computational* validation and say so precisely. Overclaiming here is the fastest way to lose credibility. |
| Real-time deployment on live hardware | Belongs to a later TRL stage; the API is designed so it is possible, but it is not delivered. |

### 3.3 The boundary that matters most

The project must never claim that a closed-loop synthetic validation proves the method works
on real plasma. It proves something narrower and still valuable: **that the inversion is
mathematically well-posed and numerically accurate under a stated, exhaustively enumerated
set of physical and instrumental assumptions**, and it quantifies how the accuracy degrades
as each assumption is violated. Stating this boundary explicitly is a strength; letting a
reviewer discover it is a fatal weakness.

---

## 4. Hard constraints

### C1 — No hardware, no lab, no real data

All data is synthetic. Consequence: **every input parameter must be traceable.** The
project maintains a provenance register (doc 09) in which each physical constant, cross
section, transition probability, material property and instrument specification is one of:

- `MEASURED` — from a standard database (NIST ASD, LXCat, OpenADAS), cited with version and access date;
- `PUBLISHED` — from a specific paper, cited;
- `SPEC` — from a real commercial component datasheet, cited with part number;
- `DESIGN` — a free design choice we made, and therefore a variable in the sensitivity study;
- `ASSUMED` — a value we could not source. **Every `ASSUMED` value is a defect.** The register tracks the count, and driving it toward zero is a tracked project metric.

### C2 — No unsolved problems

The contribution is integration and rigour, not discovery. Every algorithm must have a
citation. If implementing something requires solving an open problem, the feature is cut
rather than fudged.

### C3 — Compute: RTX A4000 (16 GB) first, better later

See doc 10 for the full analysis. The A4000 shapes the architecture in three concrete ways:

1. **16 GB VRAM** caps single-run PIC particle counts and Bayesian batch sizes. Every study
   is designed to fit, with an explicitly documented tiering to larger memory.
2. **FP64 throughput is 1/32 of FP32** on Ampere consumer/professional silicon (~0.6 TFLOP/s
   FP64 vs ~19.2 TFLOP/s FP32). Therefore: **double precision runs on CPU; the GPU is used
   in FP32/TF32 for sampling, autodiff and ray tracing only, and every such use is
   accompanied by a precision-sensitivity test.** This is a real design decision, not a
   footnote.
3. **Single GPU, no NVLink.** No multi-GPU domain decomposition is assumed. Parallelism is
   MPI-over-CPU for PDE work and embarrassingly-parallel sweeps for statistics.

Every headline result must be reproducible on one A4000 within a documented wall-clock
budget. Larger compute, when it arrives, extends sweep density and PIC fidelity — it never
becomes a prerequisite.

### C4 — No hidden assumptions

Not "no assumptions" — that is impossible. Every assumption is promoted to a **named,
typed, validated, logged and version-controlled parameter**. If a value can be changed, it
lives in the experiment manifest. If it cannot be changed, it is a physical constant with a
citation. There is no third category.

### C5 — Maximum defensible complexity, zero gratuitous complexity

The instruction is to push as hard as possible on *straightforward* work. Operationally:
complexity is spent on **coverage and rigour** (more diagnostics modelled correctly, more
noise sources, more benchmarks, more ablations, deeper UQ) and never on **novelty for its
own sake** (bespoke solvers, exotic ML, unproven methods).

---

## 5. Success criteria

The project is successful if and only if all of the following hold.

### 5.1 Scientific

- **S1** The forward operator is *verified*: method of manufactured solutions passes at the
  designed order of accuracy for every PDE solver; conservation laws hold to solver tolerance;
  mesh and timestep independence are demonstrated with published convergence plots.
- **S2** The forward operator is *cross-validated*: analytic sheath models, the fluid model
  and the PIC model agree within stated tolerances in their common region of validity
  (e.g. collisionless, high-voltage, planar limit → Child–Langmuir scaling recovered).
- **S3** The inverse solver recovers ion energy flux from noise-free synthetic measurements
  to within numerical tolerance — the *consistency* test. Failing this means the forward and
  inverse models disagree, and nothing else matters until it passes.
- **S4** Under realistic noise, the reconstruction error and the reported uncertainty are
  **calibrated**: the empirical coverage of the 95 % credible interval is 95 % ± a stated
  tolerance across the benchmark suite. An uncalibrated posterior is worse than no posterior.
- **S5** The identifiability analysis produces an explicit map of the operating space
  partitioned into *identifiable*, *weakly identifiable*, and *non-identifiable* regions,
  with the null-space directions characterised in each.
- **S6** The error budget decomposes total reconstruction error into contributions from
  physics model, discretisation, optical model, detector, calibration, noise, and inversion,
  and the contributions sum to the observed total within a stated tolerance.

### 5.2 Engineering

- **E1** Every solver, instrument, noise model, regulariser and inference engine is a plugin
  discovered at runtime; adding a new one requires no change to the core.
- **E2** A real instrument can replace a virtual one by implementing the same interface,
  with no change anywhere else. This is demonstrated by a mock "hardware" plugin.
- **E3** Every experiment is defined by a single declarative manifest and is bit-for-bit
  reproducible given the manifest, the code version and the seed.
- **E4** Every figure in every report is generated by the pipeline, not by hand, and carries
  embedded provenance metadata.
- **E5** Test coverage ≥ 80 % on the core, with physics correctness tests distinguished from
  software correctness tests.

### 5.3 Communicative

- **C1** A reviewer can ask any of the following and receive a computed answer, on screen,
  within the length of the question:
  - What if one diagnostic fails?
  - How much noise does it tolerate?
  - Which diagnostic contributes most, and how much?
  - Where does it break?
  - Why should I believe the uncertainty?
  - What exactly did you assume?
- **C2** Every claim in the presentation traces to a specific figure, which traces to a
  specific experiment manifest, which traces to a specific commit.

---

## 6. Anti-goals and failure modes we are designing against

| Failure mode | How it looks | Designed-in countermeasure |
|---|---|---|
| **The demo trap** | Impressive UI, one hardcoded scenario, collapses under a follow-up question | Benchmark suite runs *before* the UI exists; the UI only visualises stored artifacts |
| **The inverse-crime trap** | Forward and inverse use the same discretisation and model, so recovery is trivially perfect | Doc 05 §7 mandates deliberate model mismatch: different mesh, different sheath model, different collision set between forward and inverse. Reporting only inverse-crime results is treated as a project defect |
| **The uncalibrated-posterior trap** | Confidence intervals are quoted but never checked | S4 makes coverage a pass/fail gate |
| **The overclaim trap** | "Validated framework" without saying against what | Every claim carries its qualifier: *computationally verified*, *closed-loop validated*, never *experimentally validated* |
| **The reimplementation trap** | Months spent rebuilding FEM/PIC/samplers | Doc 08 §2 has an explicit build/buy table; deviating from it requires an ADR |
| **The parameter-fog trap** | Numbers appear in code with no source | C1 provenance register; `ASSUMED` count is a tracked metric |
| **The scale trap** | Plan assumes compute we do not have | Doc 10 tiers every study; Tier-0 must fit on one A4000 |

---

## 7. Team and ownership

Per the proposal deck, with computational ownership mapped onto the architecture:

| Member | Domain | Primary documents / subsystems |
|---|---|---|
| Nithisha V | Optical diagnostics & spectroscopy | Docs 02, 04 — optical layout, OES/LIF forward models, atomic data |
| Ajayaditya L | Plasma modelling & kinetic simulation | Docs 03, 05 — sheath hierarchy, PIC, inverse physics constraints |
| Denistan B | Experiments, vacuum & lasers | Docs 02, 09 — instrument specification realism, component sourcing, provenance |
| Danushika N | Inverse methods, data & software | Docs 05, 06, 07, 08 — inversion, UQ, V&V, platform |

Shared: docs 00, 01, 10–14.

---

## 8. Definition of done for Phase 0 (this planning phase)

- [ ] All fifteen documents exist at Baseline status.
- [ ] Doc 01 justifies each of the four diagnostics against derived requirements, and
      explicitly rejects at least four alternatives with stated reasons.
- [ ] Doc 09 lists every external data source with licence and citation, with zero
      unresolved `ASSUMED` entries in the *physics constants* category.
- [ ] Doc 10 contains a wall-clock budget per benchmark tier on the A4000.
- [ ] Doc 11 contains acceptance gates that are testable, not aspirational.
- [ ] Every open question is captured either as an ADR or as a risk-register entry — none
      are left implicit in prose.

---

## 9. Open decisions (to be closed by ADR)

| ID | Decision | Status |
|---|---|---|
| ADR-001 | Licence for eventual public release (Apache-2.0 vs BSD-3 vs proprietary-with-open-core) — interacts with the commercialisation posture in doc 12 | Open |
| ADR-002 | Primary autodiff substrate (JAX vs PyTorch vs adjoint-by-hand via FEniCSx/dolfin-adjoint) | Open |
| ADR-003 | PIC engine selection (Smilei vs WarpX vs a purpose-built 1D3V electrostatic kernel for the sheath problem) | Open |
| ADR-004 | Whether the interferometry channel is retained after the doc 01 trade study | Open |
| ADR-005 | Posterior representation on disk (samples vs approximating density vs both) | Open |

---

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Derived from the Techathon 2.0 proposal, the Naari Shakti thematic deck, and the two planning conversations. |
