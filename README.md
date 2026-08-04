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

**Phase 0 — Planning.** The documents above are the deliverable. No production code has
been written yet, by design: the architecture is settled on paper first so that
implementation becomes the act of filling in defined interfaces.

## Licence

TBD before any public release — see [Charter §9](docs/00-charter.md). Until then, all
rights reserved by Team Astrophage.
