# Astrophage VPL

**A Virtual Plasma Laboratory for physics-constrained inverse reconstruction of ion energy flux across plasma sheaths.**

Team Astrophage · Techathon 2.0 (Naari Shakti) · Plasma Technologies for Industry & Society · AIC RRCAT π-Hub Foundation, DAE

---

## What this is

This repository is the reference implementation of a **physics-constrained inverse
diagnostics framework** for plasma sheaths. It is not a plasma simulator, and it is not
a digital twin.

The scientific object of the project is a single, precisely posed question:

> Given a set of **non-perturbing optical observables** — optical emission spectra,
> laser-induced fluorescence lineshapes, Thomson scattering spectra, and interferometric
> phase — can the **ion energy flux** delivered to a plasma-facing surface be reconstructed,
> resolved in space and time, with **quantified and honest uncertainty**?

Ion energy flux is not directly measurable. Every instrument that can see it perturbs it;
every instrument that does not perturb it sees only a projection. The framework closes the
gap by treating the problem as a regularised, Bayesian **inverse problem** whose forward
operator is built from first-principles sheath and kinetic physics.

## The closed loop

Everything in this repository exists to support one loop, and to interrogate it from
every angle a reviewer could think of:

```
             TRUE PLASMA STATE  x
                     │                 (known only because we synthesised it)
                     ▼
        ┌────────────────────────┐
        │   Forward physics F₁   │     sheath / fluid / kinetic (PIC) hierarchy
        └────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │  Emission & scattering │     F₂ : collisional-radiative, LIF, Thomson,
        │      response  F₂      │          refractive index
        └────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │   Optical transport    │     F₃ : geometry, apertures, aberration,
        │        F₃              │          transmission, stray light
        └────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │  Detector & digitiser  │     F₄ : QE, shot/read/dark noise, ADC,
        │        F₄              │          blooming, dead pixels, jitter
        └────────────────────────┘
                     ▼
              MEASUREMENTS  y = F(x) + ε        F = F₄∘F₃∘F₂∘F₁
                     │
                     │   ◄── the ground truth x is now HIDDEN
                     ▼
        ┌────────────────────────┐
        │  Physics-constrained   │     regularised / Bayesian inversion
        │    inverse solver      │     with identifiability analysis
        └────────────────────────┘
                     ▼
          RECOVERED STATE  x̂  ,  posterior p(x|y)
                     ▼
          RECOVERED ION ENERGY FLUX  Γ_E(x̂)  ± σ
                     ▼
        ┌────────────────────────┐
        │   Validation engine    │     compare against the hidden truth,
        │  error budget · UQ     │     decompose every error contribution
        └────────────────────────┘
```

The inverse solver never sees the truth. That is the entire point, and it is what makes
the accuracy claims defensible without a laboratory.

## Documentation

The plan is written before the code. Read in order:

| # | Document | What it settles |
|---|----------|-----------------|
| 00 | [Charter](docs/00-charter.md) | Identity, scope, non-goals, hard constraints, success criteria |
| 01 | [Requirements & Trade Study](docs/01-requirements-and-trade-study.md) | What must be observable, and *why these four diagnostics* |
| 02 | [System Design Specification](docs/02-system-design-spec.md) | The complete virtual laboratory: chamber, optics, lasers, detectors, DAQ, timing |
| 03 | [Physics Model Specification](docs/03-physics-models.md) | Sheath hierarchy, kinetics, collisions, atomic data |
| 04 | [Measurement Model Specification](docs/04-measurement-models.md) | Forward operators for OES, LIF, Thomson, interferometry |
| 05 | [Inverse Problem Specification](docs/05-inverse-problem.md) | Formal statement, regularisation, samplers, identifiability |
| 06 | [Uncertainty & Error Budget](docs/06-uncertainty-and-error-budget.md) | Propagation chain, budget decomposition, calibration of the UQ itself |
| 07 | [Verification & Validation](docs/07-verification-and-validation.md) | MMS, convergence, benchmark suite, ablation and robustness matrix |
| 08 | [Software Architecture](docs/08-software-architecture.md) | Packages, API contracts, plugin discovery, data model |
| 09 | [Data, Assets & Provenance](docs/09-data-and-provenance.md) | Every external dataset, its licence, and its citation |
| 10 | [Compute & Infrastructure Plan](docs/10-compute-and-infrastructure.md) | A4000-first budget, scaling path, cost model |
| 11 | [Roadmap & Work Breakdown](docs/11-roadmap-and-wbs.md) | Phases, deliverables, acceptance gates |
| 12 | [Commercialisation](docs/12-commercialisation.md) | Product shape, TRL ladder, IP posture |
| 13 | [Reproducibility & Publication](docs/13-reproducibility-and-publication.md) | Experiment manifests, figure pipeline, archival |
| 14 | [Risk Register](docs/14-risk-register.md) | What can kill this, and the mitigation for each |

Architecture decisions are recorded in [`docs/adr/`](docs/adr/).

## Hard constraints this plan is built under

These are not preferences. Every document is written to respect them.

1. **No physical hardware, no laboratory, no real experimental data.** All measurements are
   synthetic, produced by the forward chain. Every number that enters the forward chain
   traces to a published source, a standard database, or an explicitly declared design
   parameter — never to an unattributed guess.
2. **No unsolved problems.** No new theorems, no open conjectures, no claims of a physics
   breakthrough. Every component is a known, published method. The contribution is the
   *integration, verification, and rigorous validation* of those methods into a coherent
   framework, at an engineering standard that does not currently exist as open software.
3. **Compute: a single NVIDIA RTX A4000 (16 GB, Ampere) at first availability**, with
   better compute assumed to arrive later. The plan is explicitly tiered so that every
   headline result is reachable on the A4000, and the larger compute only *widens* the
   study rather than enabling it.
4. **No hidden assumptions.** Every assumption becomes a named, versioned, logged parameter.
   If a value is a guess, it is tagged as a guess and it appears in the sensitivity study.

## Status

**Phase 1 — Foundation, in progress.** Phase 0 remains the specification; the documents
above are normative and the code is written against them section by section.

| WBS | Item | State |
|---|---|---|
| 1.1 | `vpl-core` — protocols, state model, units, provenance | **Done** |
| 1.2 | Parameter registry + literal-lint rule | **Done** |
| 1.3 | Manifest engine, `vpl run` / `reproduce` / `compare` | Next |
| 1.4 | Storage layer with embedded provenance | **Done** |
| 1.5 | L0 analytic sheath models | **Done** — V-03 passes |
| 1.6 | L1 fluid solver (FEniCSx), V-01 / V-02 | Pending |
| 1.7 | Atomic-data loaders (LXCat, NIST ASD, OpenADAS) | Pending |
| 1.8 | Boltzmann / EEDF integration | Pending |
| 1.9 | CI gates, G-1 gate report | Partial |

Verified on both the development machine (macOS/arm64) and the doc 10 §1 reference
machine (RTX A4000, WSL2 Ubuntu 24.04): 829 tests, `mypy --strict` clean, 98 % coverage,
`ASSUMED` count 0.

### Corrections to the Baseline documents

Implementation has produced three corrections, each recorded as an ADR rather than
silently applied. This is the process working as doc 11 G-1.4 anticipates, not a defect.

- **[ADR-007](docs/adr/ADR-007-child-langmuir-thickness.md)** — doc 01 §2.2 evaluates the
  Child–Langmuir sheath thickness with `λ_D` at the bulk density, but doc 03 §2.3 derives
  the expression by matching to the Bohm flux, which fixes `λ_D` at the sheath edge. The
  self-consistent value is 1.14 mm, not 0.89 mm. `Γ_E` contains no `s`, so **V-03 is
  unaffected** and doc 01 §2.3's transit-time argument survives.
- **ADR-007 §related** — doc 03 §2.1 states `γ_i = 3` in the Bohm speed while doc 03 §2.3's
  own arithmetic uses the cold-ion form, as does doc 01 §2.2.
- **The bias sign** — doc 08 §6's manifest writes `bias: -250.0 V`; doc 03 §2.2 uses `V_w`
  as a positive magnitude. Settled in `vpl.core.state.params`.

## Licence

TBD before any public release — see [Charter §9](docs/00-charter.md). Until then, all
rights reserved by Team Astrophage.
