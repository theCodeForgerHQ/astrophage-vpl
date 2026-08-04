# 14 — Risk Register

Version 1.0 · Status: **Baseline** · Owner: Team Astrophage

Scoring: Likelihood (L) and Impact (I) on 1–5; Score = L × I. Anything scoring ≥ 12 requires an
active, funded mitigation with a named owner.

---

## 1. Scientific risks

| ID | Risk | L | I | Score | Mitigation | Owner |
|---|---|---|---|---|---|---|
| **RS-01** | **The inverse problem is not identifiable across much of the envelope** — `Γ_E` cannot be uniquely recovered | 3 | 5 | **15** | This is a *result*, not a failure. Doc 05 §6 maps identifiability explicitly; publishing the boundary is a legitimate and distinctive contribution. The project is designed so that a negative answer is still an output | Ajayaditya |
| **RS-02** | Model-form error dominates and cannot be validated without real data | 4 | 4 | **16** | Doc 06 §6 multi-model inversion bounds it; doc 12 §6 published-data route partially closes it; the limitation is stated in every report | Danushika |
| **RS-03** | LIF tuning range prevents measuring high-energy ions (doc 01 §5.1) | 5 | 3 | **15** | Already identified and mitigated: measure at the sheath edge, propagate through the reconstructed field, budget the model error | Nithisha |
| **RS-04** | Thomson is too photon-starved to be useful at the low end of the envelope | 4 | 3 | 12 | Quantified in doc 02 §7.1; regimes F and G designed around it; the framework operates with reduced channel sets | Nithisha |
| RS-05 | Metastable-fraction systematic in LIF larger than budgeted | 3 | 4 | 12 | Nuisance parameter (doc 05 §2.2); CRDS promoted in doc 06 §4.1 | Nithisha |
| RS-06 | Atomic-data disagreement between LXCat sets exceeds the budgeted term | 3 | 3 | 9 | Three independent sets run; spread reported (doc 09 §2.1) | Nithisha |
| RS-07 | Surrogate error contaminates the posterior | 2 | 4 | 8 | GP predictive variance in the likelihood; emulator audit V-11 | Danushika |
| RS-08 | Cyclo-stationarity assumption fails; phase-locked accumulation invalid | 2 | 4 | 8 | Benchmark B-09 quantifies the bias | Ajayaditya |
| RS-09 | RF posterior is multimodal; NUTS misses modes | 3 | 3 | 9 | SMC / nested sampling available; Q-05 | Danushika |
| RS-10 | Discrepancy field absorbs so much that control parameters become unidentifiable | 3 | 3 | 9 | Profile likelihood on `τ`; basis size tuned; Q-06 | Danushika |

**RS-01 and RS-02 are the two genuine scientific risks**, and both are of a specific kind: they
threaten the *strength of the claim*, not the *validity of the work*. The project is
structured so that discovering "this does not work in region X" is a publishable output rather
than a wasted year. That design choice is the mitigation.

---

## 2. Technical and engineering risks

| ID | Risk | L | I | Score | Mitigation | Owner |
|---|---|---|---|---|---|---|
| **RT-01** | **Inverse crime committed accidentally** — truth leaks into the inversion | 3 | 5 | **15** | Sealed-truth barrier enforced in code (doc 07 §3); mandatory model mismatches (doc 05 §7.1); explicit leakage test G-3.4 | Danushika |
| **RT-02** | **Reported uncertainty is not calibrated** — over-confident posteriors | 3 | 5 | **15** | Coverage and SBC are pass/fail gates (doc 06 §7), not optional checks | Danushika |
| RT-03 | FEniCSx / PETSc / dolfinx installation and version fragility | 4 | 3 | 12 | Containerised from day one; micromamba lock; cold-start test on every release | Danushika |
| RT-04 | L2 PIC slower than the 3-min estimate, breaking the DS-TRAIN budget | 3 | 4 | 12 | Measured at G-1.4 before commitment; contingency in doc 10 §8 reduces DS-TRAIN and inflates budget term 10 explicitly | Ajayaditya |
| RT-05 | Purpose-built PIC kernel has a subtle bug | 3 | 5 | **15** | V-06…V-09 verification; **independent cross-check against Smilei** (doc 03 §9) — two implementations is the mitigation | Ajayaditya |
| RT-06 | Ray tracing too slow for full spectral simulation | 3 | 2 | 6 | Precompute geometric transfer matrices per optical configuration; reuse across runs |Denistan |
| RT-07 | Scope creep into building things the build/buy table says to buy | 4 | 4 | **16** | Doc 08 §2 is binding; deviation requires an ADR with justification | All |
| RT-08 | Documentation and provenance retrofitted rather than built in | 3 | 4 | 12 | Built in P1, enforced in CI thereafter (doc 11 §2) | Danushika |
| RT-09 | GPU non-determinism breaks bit-reproducibility | 3 | 2 | 6 | Deterministic reductions where feasible; disclosed and statistically characterised where not | Danushika |

**RT-07 is scored high deliberately.** The single most likely way this project consumes its
time without producing results is rebuilding solvers that already exist. The build/buy table
is the countermeasure and it is binding rather than advisory.

---

## 3. Project and resource risks

| ID | Risk | L | I | Score | Mitigation | Owner |
|---|---|---|---|---|---|---|
| RP-01 | A4000 unavailable or delayed | 3 | 3 | 9 | Doc 10 §8: CPU-only path extends the programme to ~8 weeks but blocks nothing | Ajayaditya |
| RP-02 | Team bandwidth — four people, most also presenting/writing | 4 | 4 | **16** | Doc 11 §9 compressed critical path; the planning documents are themselves a deliverable that reduces downstream effort | All |
| RP-03 | Competition deadline forces premature demonstration | 4 | 3 | 12 | Doc 11 §9: reduce scope, never reduce honesty. Tier labels prevent misrepresentation | All |
| RP-04 | Storage exhaustion during DS-TRAIN | 2 | 2 | 4 | Raw data is a cache; deletable and regenerable | Danushika |
| RP-05 | Loss of a key contributor mid-project | 2 | 4 | 8 | Documentation-first approach means the specification survives the author | All |

---

## 4. Credibility and communication risks

**These are scored high because they are the risks that actually decide outcomes in a
competitive or funding setting, and because they are entirely within our control.**

| ID | Risk | L | I | Score | Mitigation | Owner |
|---|---|---|---|---|---|---|
| **RC-01** | **Overclaiming validation** — "validated" read as "experimentally validated" | 4 | 5 | **20** | Doc 07 §1 qualifier appears verbatim in every report, slide and abstract. Highest-scoring risk in the register | All |
| **RC-02** | **A reviewer finds a limitation we had not surfaced** | 3 | 5 | **15** | The entire documentation strategy is pre-emptive disclosure: LIF tuning range, Thomson photon budget, HeNe rejection, metastable fraction, electronegative exclusion — all stated before being asked | All |
| RC-03 | Presenting T1 (inverse-crime) results as if they were T2 | 3 | 5 | **15** | Tier labels mandatory on every figure, enforced by the figure engine (doc 13 §3.2) | Danushika |
| RC-04 | Market claims not survivable by a technical audience (etch = electronegative) | 3 | 4 | 12 | Doc 12 §3 qualifies each market explicitly | Team |
| RC-05 | The project reads as a simulation project rather than a diagnostics project | 3 | 4 | 12 | Naming discipline (doc 00 §1.2); the simulation is consistently framed as a subsystem | All |
| RC-06 | Inflated TRL claim | 2 | 5 | 10 | Doc 12 §5 assesses conservatively and states the ceiling | Team |

**RC-01 is the highest-scoring risk in the entire register**, above every scientific and
technical risk. That ordering is intentional. The work can be excellent and still be
discredited by one sentence that claims more than the evidence supports — and unlike the
scientific risks, this one is entirely self-inflicted and entirely preventable.

---

## 5. Risks accepted without mitigation

Stated explicitly, because an unstated accepted risk is indistinguishable from an overlooked
one.

| Risk | Why accepted |
|---|---|
| No experimental validation in scope | Hard constraint (doc 00 C1). Consequence: TRL ceiling of 4, stated everywhere |
| Electronegative plasmas excluded | Doc 03 A8. Extension is straightforward but not attempted in v1 |
| 1-D geometry only | Doc 03 A1. Valid for the central electrode region; edge effects out of scope |
| Single ion species | Doc 03 A3. Ar⁺⁺ available as an option but not the default |
| Detector models from datasheets, not characterised units | Doc 07 §8. Unavoidable without hardware |
| No patent protection pursued | Doc 12 §7. Consistent with the no-novel-methods constraint |

---

## 6. Top risks, ranked

| Rank | ID | Risk | Score |
|---|---|---|---|
| 1 | RC-01 | Overclaiming validation | 20 |
| 2 | RS-02 | Model-form error unvalidatable without real data | 16 |
| 2 | RT-07 | Scope creep into rebuilding existing software | 16 |
| 2 | RP-02 | Team bandwidth | 16 |
| 5 | RS-01 | Identifiability may fail across much of the envelope | 15 |
| 5 | RS-03 | LIF tuning-range limitation | 15 |
| 5 | RT-01 | Accidental inverse crime | 15 |
| 5 | RT-02 | Uncalibrated uncertainty | 15 |
| 5 | RT-05 | Subtle PIC kernel bug | 15 |
| 5 | RC-02 | Unsurfaced limitation found by a reviewer | 15 |
| 5 | RC-03 | Tier mislabelling | 15 |

**Four of the top eleven are communication risks, not technical ones.** For a project whose
technical content is deliberately built from published methods, the differentiator is rigour
and honesty in how the work is presented — and that is where the failure modes concentrate.

---

## 7. Review cadence

| Phase gate | Action |
|---|---|
| Every gate (G-0 … G-7) | Re-score the register; add risks discovered during the phase |
| RS-01 | Re-score after G-4 (identifiability map complete) — likely to move sharply in one direction |
| RT-04 | Re-score after G-1.4 (measured throughput) |
| RC-01 | Reviewed before every external presentation, without exception |

---

## 8. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. 31 risks registered; overclaiming validation scored highest at 20; accepted risks stated explicitly. |
