# 06 — Uncertainty Propagation and Error Budget

Version 1.0 · Status: **Baseline** · Owner: Danushika N

> A reconstruction without an error bar is an opinion. An error bar that has never been
> checked is a worse opinion, because it looks like evidence. This document specifies both
> the propagation chain and the validation of the propagation itself.

---

## 1. The propagation chain

Uncertainty enters at seven distinct stages and must be carried through all of them.

```
  ┌─ Input uncertainty ──────────── control parameters, gas purity, supply accuracy
  │
  ├─ Physics-model uncertainty ──── which sheath model; assumptions A1–A10 (doc 03 §8)
  │
  ├─ Discretisation uncertainty ─── mesh, timestep, particle count
  │
  ├─ Surrogate uncertainty ──────── L3 emulator error (doc 03 §5.4)
  │
  ├─ Optical-model uncertainty ──── geometry, registration, transmission, PSF
  │
  ├─ Detector uncertainty ───────── 18 noise sources (doc 04 §7.2)
  │
  ├─ Calibration uncertainty ────── the calibration is itself a measurement (doc 02 §11)
  │
  └─► Inversion uncertainty ─────── posterior width, sampler error, regularisation choice
                │
                ▼
          σ(Γ_E)   with the contribution of each stage separately attributable
```

**The requirement is attribution, not just a total.** A single combined uncertainty tells an
engineer nothing about what to improve. A decomposed budget tells them exactly which
component to buy better.

---

## 2. Classification

Following the GUM (Guide to the Expression of Uncertainty in Measurement) convention, which
is what a metrology-literate reviewer will expect:

| Type | Meaning | Treatment |
|---|---|---|
| **Type A** | Evaluated from the statistics of repeated observations | Photon statistics, sampler variance — handled natively by the Bayesian posterior |
| **Type B** | Evaluated by other means (datasheets, literature, judgement) | Calibration, cross sections, `γ_se` — entered as informative priors |
| **Structural** | Model form error, not a parameter | Handled by the discrepancy field (doc 05 §2.3) and by model-averaging (§6) |

The third category is the one usually omitted and is frequently the largest. It is given its
own machinery rather than being absorbed silently.

---

## 3. Methods

| Source class | Propagation method |
|---|---|
| Parametric (Type A + B) | Native — the posterior integrates over them |
| Discretisation | Richardson extrapolation from the convergence study; entered as a bias term with uncertainty |
| Surrogate | GP predictive variance added to the likelihood variance |
| Model form | (a) discrepancy field; (b) inversion under each of L0/L1/L2 physics, spread reported; (c) Bayesian model averaging where evidence is computable |
| Correlated systematics | Full covariance, not diagonal — e.g. a single Rayleigh calibration error affects *every* Thomson point identically and does **not** average down |

**The correlated-systematic point is the one most often botched.** Treating a shared
calibration error as independent per data point makes it appear to shrink as `1/√N` with
accumulation. It does not shrink at all. The framework represents such errors as single
shared nuisance parameters (doc 05 §2.2), which is the structurally correct treatment.

---

## 4. The error budget

Estimated contributions to the relative uncertainty on `Γ_E` at reference point RP-1, Tier T2
(honest configuration, doc 05 §7.2). **These are pre-implementation estimates whose purpose is
to identify the dominant terms; measuring them is Phase 3 gate G-3.2 (doc 11).**

| # | Contribution | Source | σ/Γ_E | Type | Reducible by |
|---|---|---|---|---|---|
| 1 | **Physics model form** (fluid vs kinetic IEDF; assumptions A5, A6) | doc 03 §6, §8 | **8.0 %** | Structural | Using L2 everywhere; better surrogate coverage |
| 2 | **`T_e` inference** (CR model, escape factors, EEDF shape) | doc 04 §2 | **7.5 %** | B + structural | Thomson anchoring; better atomic data |
| 3 | **Thomson absolute calibration** (Rayleigh) | doc 02 §11 | **7.0 %** | B | Better pressure gauge; cross-calibration |
| 4 | **LIF metastable fraction** (nuisance parameter) | doc 02 §5.3 | **5.0 %** | B | Cavity ring-down (Tier 2) |
| 5 | **LIF velocity projection** (15° geometry, 3.86× amplification) | doc 02 §4.2 | **5.0 %** | B | Steeper geometry if optically feasible |
| 6 | **CR-model / escape-factor systematics** | doc 04 §2.3 | **5.0 %** | Structural | Validated CR model; measured metastable density |
| 7 | **Spatial registration** (30 µm on 890 µm sheath) | doc 02 §10.4 | **4.0 %** | B | Better fiducial calibration |
| 8 | **Thomson photon statistics** (3 % at 7 000 shots) | doc 02 §7.1 | **3.0 %** | A | Longer accumulation (∝ √t) |
| 9 | **OES absolute radiometric calibration** | doc 02 §11 | **3.0 %** | B | Better transfer standard |
| 10 | **Surrogate (L3) error** | doc 03 §5.4 | **3.0 %** | Structural | More training runs |
| 11 | **Stray-light background subtraction** | doc 02 §4.3 | **2.0 %** | B | Better baffling |
| 12 | **Discretisation** (mesh, timestep, `N_ppc`) | doc 03 §7 | **1.0 %** | Structural | Refinement (cheap) |
| | **Combined (quadrature)** | | **≈ 17.1 %** | | |
| | **Requirement R-ACC-5** | doc 01 §2.4 | **≤ 20 %** | | ✔ **Meets with 2.9 pp margin** |

### 4.1 What the budget says

Three conclusions follow, and each is actionable:

1. **The budget is dominated by model-form and calibration terms, not by photon statistics.**
   Terms 1, 2, 3 and 6 together account for 74 % of the variance. **Running the experiment
   longer does almost nothing.** This is a counter-intuitive and genuinely useful finding: the
   instinct to improve a measurement by accumulating more data is wrong for this problem.
2. **The single highest-value improvement is a direct metastable-density measurement**
   (terms 4 and partly 2 and 6), which is exactly what cavity ring-down spectroscopy provides.
   The trade study (doc 01 §4.2) deferred CRDS to Tier 2 on cost grounds; the error budget
   now supplies the quantitative case for promoting it. **This is the requirements process
   feeding back on itself, which is what it is for.**
3. **The margin against R-ACC-5 is thin (2.9 percentage points).** Any of terms 1–3 growing by
   50 % would break the requirement. The budget is therefore a live document tracked against
   measurement, not a one-off calculation.

---

## 5. Calibration uncertainty in detail

Calibration is modelled end-to-end (doc 04 §7.3): the framework simulates the *calibration
measurement*, derives an *estimated* response from it, and applies that estimate. The residual
error is therefore emergent rather than asserted.

| Calibration | Chain | Dominant term |
|---|---|---|
| Thomson absolute | Rayleigh cross section (2 %) → gas purity (1 %) → pressure gauge (2 %) → optical stability (3 %) → photon statistics of the calibration itself (5 %) | calibration statistics |
| OES radiometric | NIST lamp scale (2 %) → transfer optics (3 %) → window ageing (4 %) | window ageing |
| LIF frequency | wavemeter (2 MHz) → etalon FSR (0.1 %) | negligible |
| Registration | CMM fiducial (10 µm) → thermal drift (20 µm) | thermal drift |

**Window ageing is a slow, monotonic drift**, which makes it especially dangerous: it produces
a trend in the reconstructed flux that can be mistaken for physics. The framework models it
and benchmark B-10 tests whether the inversion's cross-channel consistency check (doc 05 §8)
detects it.

---

## 6. Model-form uncertainty

The largest single term (8 %) and the hardest to quantify. Three complementary approaches,
all reported:

1. **Multi-model inversion.** Invert the same data under L0, L1 and L2 physics. The spread in
   `Γ_E` is a direct empirical measure of model-form sensitivity.
2. **Discrepancy field.** The `θ_f` term (doc 05 §2.3) absorbs residual structure and inflates
   the posterior accordingly.
3. **Bayesian model averaging.** Where the evidence `p(y|M)` is computable (nested sampling),
   average over models weighted by evidence rather than selecting one.

**The honest position, stated plainly:** model-form uncertainty is the one component that
closed-loop synthetic validation *cannot fully validate*, because the truth was generated by
a model. What the framework can do — and does — is measure the sensitivity of the answer to
*model choice within the hierarchy*, which bounds but does not eliminate the concern. Only
real experimental data closes it. This limitation is stated in every report the framework
generates rather than left for a reviewer to find.

---

## 7. Validating the uncertainty itself

**A posterior that is never checked is decoration.** Doc 00 S4 makes calibration a pass/fail
gate; this section specifies how it is tested.

### 7.1 Coverage test

Over `N = 1000` synthetic cases drawn from the prior:

```
Empirical coverage of the q-credible interval  =  (1/N) Σ 1[ Γ_E^true ∈ CI_q ]
```

Pass criterion: for `q = 0.95`, empirical coverage ∈ **[0.93, 0.97]**.

| Outcome | Interpretation | Action |
|---|---|---|
| Coverage ≈ q | Calibrated ✔ | — |
| Coverage < q | **Over-confident** — the dangerous failure | Investigate: usually unmodelled systematics or an inadequate discrepancy field |
| Coverage > q | Under-confident | Acceptable but wasteful; usually over-inflated priors |

### 7.2 Rank-statistic / simulation-based calibration

A stronger test than coverage alone: for each synthetic case, compute the rank of the true
value within the posterior samples. If the posterior is correctly calibrated, ranks are
**uniformly distributed**. Deviations diagnose the failure mode:

| Rank histogram shape | Diagnosis |
|---|---|
| Uniform | Correct ✔ |
| ∪-shaped | Posteriors too narrow (over-confident) |
| ∩-shaped | Posteriors too wide |
| Sloped | Systematic bias in the mean |

This is the standard SBC procedure and it distinguishes bias from mis-scaled variance, which a
coverage number alone cannot.

### 7.3 Reliability diagram

Plot nominal vs empirical coverage across all quantile levels. A calibrated framework lies on
the diagonal. **This single plot is the most persuasive uncertainty figure the project can
produce**, and it is the direct answer to "why should I trust your error bars?"

---

## 8. Reporting standard

Every reported flux carries:

```
Γ_E  =  6.58  ±  1.13  kW·m⁻²        (17.1 %, 95 % CI [4.41, 8.83])
   tier: T2 (honest)
   dominant contributions: model form 8.0 %, T_e inference 7.5 %, Thomson calibration 7.0 %
   coverage validated: 0.951 over 1000 cases (SBC rank test p = 0.34)
   manifest: 4a7f2e...  commit: 9c1d8b...  seed: 20260804
```

Reporting a bare number is prohibited by the figure pipeline (doc 13) — the metadata travels
with the value, mechanically.

---

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Twelve-term budget estimated at 17.1 %, meeting R-ACC-5 with 2.9 pp margin; identified model form and calibration — not photon statistics — as dominant. |
