# 07 — Verification, Validation and Benchmarking

Version 1.0 · Status: **Baseline** · Owner: Danushika N

---

## 1. Verification ≠ validation

The distinction is standard (ASME V&V 20, AIAA G-077) and is routinely conflated. The
framework keeps them structurally separate — different test suites, different reports,
different acceptance gates.

| | Question | Reference | Can we do it? |
|---|---|---|---|
| **Verification** | Are we solving the equations right? | Mathematics: analytic solutions, manufactured solutions, convergence rates, conservation laws | **Yes, fully.** No experiment required |
| **Validation** | Are we solving the right equations? | Reality: experimental data | **Only partially.** We have no experimental data |
| **Closed-loop validation** | Does the inversion recover a known truth from realistic synthetic measurements? | The forward model itself | **Yes, fully.** This is the project's core evidence |

**The honest statement, which appears in every report:**

> The framework is fully *verified*. It is *closed-loop validated* against synthetic ground
> truth under a stated and exhaustively enumerated set of assumptions. It is **not
> experimentally validated**, and no claim of experimental validation is made. Closed-loop
> validation establishes that the inversion is mathematically well-posed and numerically
> accurate given the forward model; it cannot establish that the forward model describes a
> real plasma. That requires laboratory data, and is the purpose of the staged experimental
> roadmap.

Saying this first is worth more than any additional result. A reviewer who extracts this
admission under questioning will discount everything; a reviewer who is told it up front will
credit the rigour.

---

## 2. Verification suite

### 2.1 Code verification

| Class | Tests | Reference |
|---|---|---|
| Physics solvers | V-01 … V-11 | doc 03 §7 |
| Measurement models | V-20 … V-30 | doc 04 §8 |
| Inverse solvers | V-40 … V-48 | §2.2 below |
| Software | unit, property-based, integration | doc 08 §8 |

### 2.2 Inverse-solver verification

| ID | Test | Pass criterion |
|---|---|---|
| V-40 | Linear-Gaussian problem with analytic posterior | Posterior mean and covariance match analytics to 10⁻⁶ |
| V-41 | Recovery of a known posterior by each engine | NUTS, SMC and Laplace agree within MCSE on a Gaussian test case |
| V-42 | Gradient check (autodiff vs finite difference) | Relative error < 10⁻⁶ |
| V-43 | Fisher information vs empirical posterior covariance, Gaussian case | Match within sampling error |
| V-44 | Sampler diagnostics | R̂ < 1.01, ESS > 400 per parameter, zero divergences |
| V-45 | Seed reproducibility | Identical posterior given identical seed |
| V-46 | Prior recovery — run inversion with no data | Posterior = prior to sampling error. *Catches likelihood-normalisation bugs, which are otherwise nearly invisible* |
| V-47 | Simulation-based calibration on a toy problem | Uniform rank histogram (doc 06 §7.2) |
| V-48 | Emulator audit | doc 03 V-11 |

**V-46 is disproportionately valuable.** Running the inversion with the likelihood switched
off must return exactly the prior. A surprising number of Bayesian implementations fail this,
and the failure silently biases every result.

### 2.3 Order-of-accuracy verification

Method of manufactured solutions is applied to every PDE solver: choose an analytic solution,
substitute it into the governing equations, derive the source term that makes it exact, then
verify that the observed convergence rate matches the design order.

Reported as a log–log error-vs-`h` plot with fitted slope. **Design order ± 0.1 is required.**
A solver converging at first order when second was designed has a bug, and MMS is the only
reliable way to find it.

---

## 3. Closed-loop validation protocol

The core experiment, run identically for every benchmark:

```
1. Draw a ground-truth state x* from the scenario definition
2. Run the forward chain F (docs 03, 04) → synthetic measurements y
3. SEAL x*  — an explicit, enforced barrier in the code
4. Run the inversion on y  → posterior p(x|y), p(Γ_E|y)
5. UNSEAL x*  → compute error metrics, coverage, attribution
6. Emit artifacts, figures, provenance
```

**Step 3 is enforced mechanically, not by convention.** The truth object is moved into a
sealed store that the inversion process cannot read; attempting to access it raises. This is
worth the engineering effort, because accidental truth leakage — through a shared config
object, a cached array, a default initialisation — is easy, silent, and invalidates
everything.

---

## 4. Metrics

| Metric | Definition | Applies to |
|---|---|---|
| Relative error | `\|Γ̂_E − Γ_E\| / Γ_E` | scalar flux |
| NRMSE | RMSE normalised by range | profiles |
| Bias | `E[Γ̂_E] − Γ_E` over repeats | systematic error |
| Coverage | fraction of cases with truth inside the 95 % CI | uncertainty calibration |
| SBC rank uniformity | KS statistic on rank histogram | uncertainty calibration |
| Interval score / CRPS | proper scoring rule | joint accuracy + sharpness |
| Wasserstein distance | between true and recovered IEDF | distribution recovery |
| KS distance | between true and recovered IEDF | distribution recovery |
| Condition number | `λ_max/λ_min` of the FIM | identifiability |
| Posterior contraction | `σ_prior / σ_posterior` | how much the data actually taught us |

**CRPS is included deliberately.** Accuracy and sharpness trade off: a wide interval is easy
to make well-covered and useless. A proper scoring rule penalises both miscalibration and
excessive width, and it is the right single number for comparing configurations.

---

## 5. Benchmark suite

### 5.1 Scenario benchmarks

| ID | Scenario | Regime (doc 02 §3.3) | Tests |
|---|---|---|---|
| B-01 | Ideal collisionless DC sheath | A | Baseline; Child–Langmuir recovery |
| B-02 | Reference operating point | B | The headline result |
| B-03 | Collisional sheath | C | **Fluid-vs-kinetic IEDF divergence** — where L1 fails |
| B-04 | RF, transit-resonant | D | Bimodal IEDF; phase-resolved recovery |
| B-05 | RF, high-frequency limit | E | Single-peak IEDF; degenerate phase |
| B-06 | Non-Maxwellian EEDF (bi-Maxwellian) | B + κ | Thomson Maxwellian-fit bias |
| B-07 | Low density | F | **Interferometry blind** — 3-channel operation |
| B-08 | Transient (pulsed bias, ignition, decay) | G | **Thomson blind** — EnKF tracking |
| B-09 | Discharge drift during accumulation | B + drift | Cyclo-stationarity assumption A9 |
| B-10 | Window ageing over a long run | B + drift | Slow calibration drift detection |
| B-11 | Magnetised (50 G) | B + B-field | Zeeman handling (doc 04 §3.3) |
| B-12 | High bias (1000 V) | envelope edge | LIF tuning-range limitation (doc 01 §5.1) |
| B-13 | Xenon working gas | different `m_i` | Generality beyond argon |
| B-14 | Envelope sweep (Latin hypercube, 2000 points) | all | The identifiability map (doc 00 S5) |

### 5.2 Robustness / ablation matrix

Each of the 12 failure modes of doc 02 §13 (F-01 … F-12), plus:

| ID | Ablation |
|---|---|
| F-13 | LIF saturation sweep, `S` = 0.01 → 10 |
| F-14 | Noise scaling sweep, ×0.1 → ×10 |
| F-15 | Physics prior removed (unregularised inversion) |
| F-16 | Temporal information removed (time-averaged data only) |
| F-17 | Discrepancy field removed |
| F-18 | Reduced to a single channel (×4, one per channel) |
| F-19 | Reduced to each pair of channels (×6) |

**F-18 and F-19 together produce the diagnostic-value matrix** — the quantitative answer to
"which diagnostics do I actually need?", which is simultaneously the scientific result and
the commercial product-tiering argument (doc 12).

**F-15 and F-16 are the ablations that justify the project's central claims.** If removing the
physics prior barely changes the result, "physics-constrained" was decoration. If removing
temporal information barely changes it, "time-resolved" was decoration. The framework must be
prepared for either answer — and reporting an honest null result on one of them would be far
better than not testing.

### 5.3 Comparative benchmarks — the persuasive figure

Per doc 01 §4.2, the intrusive diagnostics are retained as *simulated* reference instruments.
For each scenario the framework produces:

| Method | What it is |
|---|---|
| **Ground truth** | Known by construction |
| **Our optical reconstruction** | The framework's answer, with CI |
| **Simulated RFEA** | What a retarding-field analyser would report — *including its own perturbation of the sheath and its own systematics* |
| **Simulated Langmuir probe + Child-law estimate** | The conventional workflow |
| **Naive single-diagnostic estimate** | e.g. `Γ_E = 0.61 n₀ c_s e V_w` from OES-inferred `T_e` alone |

Plotted together against truth. **This is the single most valuable figure the project can
produce**, because it converts the claim "non-intrusive optical inference is better than
conventional practice" from an assertion into a measurement — and because the simulated probe
genuinely perturbs the simulated plasma, which is a comparison no real experiment can make.

---

## 6. Acceptance gates

A benchmark is *passed* only if all apply:

| Gate | Criterion |
|---|---|
| G-V1 | All verification tests pass at their stated tolerance |
| G-V2 | T0 consistency test recovers truth to numerical tolerance |
| G-V3 | Sampler diagnostics clean (R̂ < 1.01, ESS > 400, no divergences) |
| G-V4 | Coverage of the 95 % CI within [0.93, 0.97] |
| G-V5 | SBC rank histogram passes KS uniformity at p > 0.05 |
| G-V6 | Error budget terms sum (in quadrature) to the observed total within 25 % |
| G-V7 | Tier label (T0/T1/T2) present on every reported figure |
| G-V8 | Provenance complete: manifest, commit, seed, environment |

**G-V6 is a strong internal-consistency check** and is easy to overlook. If the decomposed
budget does not reconstruct the observed scatter, either a term is missing or a term is
wrong — and either way the budget is not yet understood.

---

## 7. Continuous validation

| Frequency | Scope | Runtime target |
|---|---|---|
| Every commit | Unit + property tests, V-40…V-48 on toy problems | < 10 min |
| Every PR | Verification suite V-01…V-30 at coarse resolution | < 1 h |
| Nightly | B-01, B-02, B-07 at full resolution | < 8 h |
| Weekly | Full benchmark suite B-01…B-13 | < 48 h |
| On release | Everything including B-14 envelope sweep and all ablations | ~1 week |

Regression detection: every metric is stored per commit; a change beyond its historical
noise band fails the build. **Physics regressions are silent otherwise** — the code runs, the
plots look plausible, and the answer is wrong.

---

## 8. What validation cannot establish

Stated explicitly so it is never implied otherwise:

1. **That the forward model matches a real plasma.** Closed-loop validation is self-referential
   with respect to model form. Mitigated (doc 06 §6) but not eliminated.
2. **That real instruments behave as modelled.** Detector models come from datasheets, not
   from characterisation of specific units.
3. **That the atomic data are correct.** Cross sections and transition probabilities carry
   their own literature uncertainties, propagated but not independently checked.
4. **That the operating envelope covers real devices.** It covers a specified, representative
   range; extrapolation beyond it is unsupported.

Each limitation has a defined closure path in the experimental roadmap (doc 11 §6).

---

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Verification separated from validation; 14 scenario benchmarks, 19 ablations, 8 acceptance gates; sealed-truth protocol specified. |
