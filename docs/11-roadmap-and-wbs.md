# 11 — Roadmap, Work Breakdown and Acceptance Gates

Version 1.0 · Status: **Baseline** · Owner: Team Astrophage

> Instruction: build without restraint on time, pushing complexity as far as it will go while
> staying strictly inside straightforward, published methods. This roadmap is written to that
> instruction. **§9 additionally provides a compressed critical path**, because the planning
> conversations referenced a competition final round, and a plan that ignores a near-term
> deadline it knows about is not being helpful.

---

## 1. Phase structure

| Phase | Name | Output | Gate |
|---|---|---|---|
| **P0** | Planning | Docs 00–14 at Baseline | G-0 |
| **P1** | Foundation | Core protocols, registry, manifest engine, L0/L1 solvers, verified | G-1 |
| **P2** | Forward chain | L2 kinetic, all four instruments, optics, detectors, verified | G-2 |
| **P3** | Inversion | Surrogate, likelihood, engines, first closed-loop recovery | G-3 |
| **P4** | Rigour | UQ, identifiability, error budget, calibration validation | G-4 |
| **P5** | Benchmarks | Full suite, ablations, comparative study | G-5 |
| **P6** | Platform | Interactive interface, report generator, plugin demos | G-6 |
| **P7** | Release | Documentation, public benchmark, paper draft | G-7 |

---

## 2. P1 — Foundation

**Objective:** the skeleton is real and verified before any physics complexity arrives.

| WBS | Task | Owner | Done when |
|---|---|---|---|
| 1.1 | `vpl-core`: protocols, state types, units, provenance | Danushika | `mypy --strict` clean; round-trip tests pass |
| 1.2 | Parameter registry + literal-lint rule | Denistan | CI fails on a numeric literal in physics code |
| 1.3 | Manifest engine (Hydra) + `vpl run`/`reproduce`/`compare` | Danushika | Bit-identical reproduction from archived manifest |
| 1.4 | Storage layer (HDF5/Zarr/Parquet) with embedded provenance | Danushika | Artifact round trip; provenance present |
| 1.5 | L0 analytic sheath models | Ajayaditya | Child–Langmuir reproduces doc 03 §2.3 numbers exactly |
| 1.6 | L1 fluid solver (FEniCSx) | Ajayaditya | V-01, V-02 MMS at design order |
| 1.7 | Atomic-data loaders (LXCat, NIST ASD, OpenADAS) | Nithisha | Version hashes recorded; upstream-change detection works |
| 1.8 | Boltzmann/EEDF integration (BOLSIG+) | Nithisha | Rate coefficients reproduce published Ar values |
| 1.9 | CI: lint, types, tests, coverage, traceability, `ASSUMED` count | Danushika | All gates enforced |

**Gate G-1**
- G-1.1 V-01, V-02 pass at design order ± 0.1
- G-1.2 V-03 Child–Langmuir recovery within 5 %
- G-1.3 Manifest reproduction bit-identical
- G-1.4 **Measured** L1 throughput recorded; doc 03 §4.4 estimates corrected
- G-1.5 `ASSUMED` count in physics constants = 0
- G-1.6 Core coverage ≥ 80 %

---

## 3. P2 — Forward chain

**Objective:** a plasma state becomes realistic synthetic data through a fully verified,
strictly layered chain.

| WBS | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | L2 PIC-MCC kernel (1D3V electrostatic) | Ajayaditya | V-06, V-07, V-09 pass |
| 2.2 | Collision module (null-collision, LXCat sets) | Ajayaditya | CX rate reproduces analytic mean free path |
| 2.3 | Smilei cross-check plugin | Ajayaditya | Independent kinetic implementation agrees within 5 % |
| 2.4 | CR model + escape factors | Nithisha | V-24, V-25, V-26 pass |
| 2.5 | OES forward model + spectrograph | Nithisha | Instrument function matches design |
| 2.6 | LIF forward model (rate eqs, Zeeman, saturation) | Nithisha | V-22, V-23 pass |
| 2.7 | Thomson forward model (Selden, relativistic) | Nithisha | V-20, V-21 pass |
| 2.8 | Interferometry forward model | Denistan | V-29 passes |
| 2.9 | Optical transport (Raysect) | Denistan | V-27 passes |
| 2.10 | Detector chain, 18 noise sources | Denistan | V-28 photon-transfer curve recovers configured gain/read noise |
| 2.11 | Calibration simulation (imperfect, doc 04 §7.3) | Denistan | Estimated ≠ true; residual matches doc 06 §5 |
| 2.12 | Simulated reference instruments (probe, RFEA) | Ajayaditya | Probe perturbation visible in the solution |

**Gate G-2**
- G-2.1 All of V-20…V-30 pass
- G-2.2 Layer-isolation test: `F₄` has no access path to plasma state
- G-2.3 L2 vs Smilei agreement within 5 % on B-01
- G-2.4 Synthetic Thomson data reproduces the doc 02 §7.1 photon budget within 10 %
- G-2.5 Every noise source individually switchable and individually verified

---

## 4. P3 — Inversion

| WBS | Task | Owner | Done when |
|---|---|---|---|
| 3.1 | DS-TRAIN generation (5 000 L2 runs) | Ajayaditya | Complete, provenanced |
| 3.2 | POD reduction + GP surrogate | Danushika | Held-out error meets target |
| 3.3 | Surrogate audit (V-11 / V-48) | Danushika | L2 re-evaluation at posterior modes agrees |
| 3.4 | Per-channel likelihoods (Poisson/Gaussian/correlated) | Danushika | V-46 prior-recovery passes |
| 3.5 | Asynchronous acquisition-window likelihood | Danushika | Phase-binned integration verified |
| 3.6 | Priors + discrepancy field | Danushika | Shrinkage behaves; `τ` identifiable |
| 3.7 | Engines: MAP, Laplace, NUTS, SMC, EnKF, PF | Danushika | V-40…V-45 pass |
| 3.8 | Sealed-truth barrier | Danushika | Truth access from inverse process raises |
| 3.9 | First closed-loop recovery at RP-1 | All | T0 recovers to numerical tolerance |

**Gate G-3**
- G-3.1 **T0 consistency: recovery to numerical tolerance.** *Blocking — nothing proceeds until this passes*
- G-3.2 T1 and T2 both produced and labelled; T1–T2 gap quantified
- G-3.3 Sampler diagnostics clean on B-02
- G-3.4 Truth-leakage test passes
- G-3.5 Measured error budget produced and compared against doc 06 §4 estimates

---

## 5. P4 — Rigour

| WBS | Task | Owner |
|---|---|---|
| 4.1 | Uncertainty propagation across all seven stages | Danushika |
| 4.2 | Error-budget decomposition and attribution | Danushika |
| 4.3 | Coverage test (DS-COVER, 1 000 cases) | Danushika |
| 4.4 | Simulation-based calibration, rank histograms | Danushika |
| 4.5 | Reliability diagrams | Danushika |
| 4.6 | Fisher information, eigen-analysis, CRB comparison | Ajayaditya |
| 4.7 | Profile likelihood | Ajayaditya |
| 4.8 | Sobol global sensitivity | Ajayaditya |
| 4.9 | Per-channel information content (closes ADR-004) | Danushika |
| 4.10 | Model-misspecification detection | Danushika |
| 4.11 | Optimal experiment design | Danushika |

**Gate G-4**
- G-4.1 Coverage of 95 % CI ∈ [0.93, 0.97]
- G-4.2 SBC rank uniformity p > 0.05
- G-4.3 Budget terms reconstruct observed scatter within 25 % (G-V6)
- G-4.4 Identifiability map produced across the envelope
- G-4.5 ADR-004 closed with quantitative evidence

---

## 6. P5 — Benchmarks

| WBS | Task |
|---|---|
| 5.1 | Scenario benchmarks B-01 … B-13 |
| 5.2 | Envelope sweep B-14 (2 000 LHS points) |
| 5.3 | Robustness matrix F-01 … F-19 |
| 5.4 | Comparative study vs simulated probe / RFEA / naive estimate |
| 5.5 | Regression baseline established |
| 5.6 | Failure-boundary characterisation |

**Gate G-5**
- G-5.1 All benchmarks pass their acceptance gates or are documented as characterised failures
- G-5.2 F-15 (physics prior removed) and F-16 (temporal information removed) quantified — **whatever the answer**
- G-5.3 Diagnostic-value matrix complete (F-18, F-19)
- G-5.4 Failure boundary mapped and published as a figure

> **G-5.2 is a commitment to report an inconvenient result if we get one.** If the physics
> prior turns out not to matter much, that is what will be published. Setting that expectation
> now, in writing, is what makes the eventual claim credible.

---

## 7. P6 — Platform, and P7 — Release

| Phase | Tasks |
|---|---|
| **P6** | Interactive interface (doc 08 §11); report generator; plugin demos (Smilei, E-FISH, mock hardware); CLI polish; performance profiling |
| **P7** | Full documentation site; public benchmark release (doc 09 §4.3); paper draft; licence decision (ADR-001); archival DOI |

**Gate G-6:** every UI panel reads only stored artifacts; mock-hardware plugin proves E2;
one-click report reproduces a published figure set.

**Gate G-7:** documentation builds clean; public benchmark downloadable and self-validating;
paper draft complete with every figure traceable to a manifest.

---

## 8. Dependency graph

```
P0 ──► P1 ──► P2 ──► P3 ──► P4 ──► P5 ──► P6 ──► P7
        │      │      │      │      │
        │      │      └──────┴──────┘   P4 and P5 overlap heavily
        │      │
        │      └─► DS-TRAIN generation runs concurrently with P2 completion
        │
        └─► CI, docs and provenance are built in P1 and maintained throughout,
            never retrofitted
```

**Critical path: P1 → P2 (L2 kernel) → DS-TRAIN → P3 (surrogate) → P4.** The 11.5-day
DS-TRAIN campaign (doc 10 §3.2) is the longest single blocking item, so it is started the
moment the L2 kernel passes G-2.1 — before the rest of P2 finishes.

---

## 9. Compressed critical path for a near-term demonstration

> **Assumption flagged:** the planning conversations referred to a competition final round
> approximately one week out. If that deadline is live, the full roadmap above cannot complete
> first. This section defines the minimum subset that produces a **credible and honest**
> demonstration, without misrepresenting maturity. If the deadline has passed or does not
> apply, ignore this section — the main roadmap is unaffected.

**Principle: reduce scope, never reduce honesty.** A smaller result that is correctly labelled
beats a larger result that implies validation it does not have.

**Compute is no longer the binding constraint.** With the PIC on GPU (doc 10 §3.2) the full
programme is ~33 h on the A4000 already in hand. The constraint is now **build time**, so this
list is ordered by what must be *written*, not by what must be *computed*.

| Priority | Item | Why it is on the list |
|---|---|---|
| **1** | **JAX PIC kernel (1D3V electrostatic), GPU** | Everything downstream hangs off this. ~400 lines. Measure throughput immediately (G-1.4) |
| **2** | L0 + L1 forward, RP-1 | Verification anchor and the cheap inversion model |
| **3** | OES + LIF forward models with genuine noise | Two channels demonstrate fusion; add Thomson/interferometry if time allows |
| **4** | Closed-loop T0 + T2 recovery at RP-1 | The core claim, end to end. **T0 is blocking** |
| **5** | Coverage test, 400+ cases | Uncertainty that has been *checked* — the strongest differentiator. **Do not cut** |
| **6** | Ablation: drop each channel, show the CI inflate | The most persuasive 20 seconds available |
| **7** | Comparative figure vs simulated probe / RFEA | Converts "better than probes" into a measurement |
| **8** | Surrogate + full envelope map | Now affordable; include if the build reaches it |
| **9** | Minimal UI over stored artifacts | Makes it tangible without becoming a demo trap |
| — | *Everything else* | Presented as the specified roadmap, backed by these documents |

**If the schedule slips, cut in this order:** 9 → 8 → 7 → 3 (drop to one channel) → 6.
**Never cut 5.**

**What makes this defensible rather than thin:** the planning documents themselves are the
deliverable that most teams will not have. Presenting a modest working system *plus* a
150-page engineering specification that anticipates every question — the LIF tuning-range
limit, the Thomson photon budget, the HeNe rejection, the inverse-crime protocol — demonstrates
exactly the judgement the full build would demonstrate, at a fraction of the elapsed time.

---

## 10. Definition of done, project level

- [ ] Every requirement in doc 01 traced to an implementation and a passing test
- [ ] Every verification test passing at stated tolerance
- [ ] Coverage and SBC gates passing
- [ ] Error budget measured, decomposed, and reconciled with observation
- [ ] Identifiability map published
- [ ] All benchmarks and ablations run; failure boundary characterised
- [ ] `ASSUMED` parameter count = 0 in physics categories
- [ ] Every figure regenerable from an archived manifest
- [ ] Public benchmark released
- [ ] Every claim carries its qualifier: verified / closed-loop validated — **never** experimentally validated

---

## 11. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Eight phases with testable gates; DS-TRAIN identified as the critical-path bottleneck; compressed near-term path added under a flagged assumption. |
