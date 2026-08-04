# 05 — Inverse Problem Specification

Version 1.0 · Status: **Baseline** · Owner: Danushika N

> This is the research contribution. Everything in docs 02–04 exists to give this document a
> forward operator worth inverting.

---

## 1. Formal statement

### 1.1 The problem

Given observations `y` from the four channels, recover the plasma state `x` and thence the
ion energy flux `Γ_E`:

```
y  =  F(x)  +  ε           F = F₄ ∘ F₃ ∘ F₂ ∘ F₁       (docs 03, 04)
```

`F` is non-linear, non-injective, and maps a high-dimensional state to a comparatively small
observation vector. A direct inverse `F⁻¹` does not exist. The problem is therefore posed
probabilistically:

```
p(x | y)  ∝  p(y | x) · p(x)
```

and the deliverable is not a number but a **distribution**:

```
p(Γ_E | y)  =  ∫ δ( Γ_E − G(x) ) p(x | y) dx           G = the flux functional (doc 03 §6)
```

reported as a posterior mean with credible intervals.

### 1.2 Why this framing is the right one

Three properties follow from it that a deterministic fit cannot provide:

1. **Non-uniqueness becomes visible rather than hidden.** Where the data do not determine the
   state, the posterior is broad in that direction — the framework says "I don't know" instead
   of returning an arbitrary point from the null space.
2. **Physics enters as a prior, not as a post-hoc filter.** The sheath model is inside `F`,
   so every sample the sampler visits is physically realisable by construction.
3. **The answer carries its own error bar**, and that error bar can itself be validated
   (doc 00 S4) — which is the difference between a diagnostic and a guess.

---

## 2. Parameterisation of the state

The state is parameterised hierarchically. This choice is the most consequential design
decision in the document: too few parameters and the model cannot fit the data (bias); too
many and nothing is identifiable (variance).

### 2.1 Level A — control parameters (`θ_c`, ~8)

Physically meaningful, directly interpretable, and what an operator would actually set.

| Parameter | Symbol | Prior | Units |
|---|---|---|---|
| Bulk electron density | `n₀` | log-uniform [10¹⁵, 10¹⁹] | m⁻³ |
| Electron temperature | `T_e` | log-normal, μ = 3 eV, σ = 0.4 | eV |
| Ion temperature | `T_i` | log-uniform [0.02, 0.5] | eV |
| Neutral pressure | `p` | informative — gauge-measured, σ = 2 % | mTorr |
| Wall bias | `V_w` | informative — supply-measured, σ = 1 % | V |
| RF phase offset | `φ_RF` | uniform [0, 2π) | rad |
| Secondary emission yield | `γ_se` | normal, 0.10 ± 0.03 | — |
| EEDF shape parameter | `κ` | uniform [1, 5] (Maxwellian → Druyvesteyn) | — |

### 2.2 Level B — nuisance parameters (`θ_n`, ~10)

Not of interest, but they must be inferred or the answer is biased. **Enumerating these
honestly is what separates a working inversion from one that quietly absorbs systematics into
the physics.**

| Parameter | Why it must be free |
|---|---|
| Ar⁺ metastable fraction | LIF measures a subpopulation (doc 02 §5.3) |
| OES absolute radiometric scale | 6 % calibration uncertainty (doc 02 §11) |
| Thomson absolute scale (Rayleigh calibration) | 7 % |
| Thomson stray-light pedestal | dominant background systematic |
| Interferometer phase offset | unwrapping reference |
| LIF frequency-axis offset | wavemeter drift |
| Per-channel registration offsets (×4) | 20–100 µm (doc 02 §10.4) |
| Window transmission factors | sputter coating drift |

### 2.3 Level C — profile corrections (`θ_f`, ~20)

The parametric physics model may not capture reality exactly. A **discrepancy field** is added
to the density and potential profiles, expanded in a truncated smooth basis with a strong
shrinkage prior:

```
n_i(z)  =  n_i^model(z; θ_c) · exp( Σ_{j=1..N_b} α_j ψ_j(z) )        α ~ N(0, τ²)
```

with `τ` itself inferred (a hierarchical shrinkage prior). This is the standard
model-discrepancy construction of Kennedy & O'Hagan.

**Why it is included despite the cost.** Without a discrepancy term, any model error is forced
into the physical parameters, producing a confident and wrong answer — the classic
"over-confident and biased" failure of parametric Bayesian inversion. With it, model error
inflates the posterior instead of corrupting the mean. The price is that `τ` and `θ_c` are
partly confounded, which the identifiability analysis (§6) measures explicitly.

---

## 3. The likelihood

### 3.1 Per-channel construction

Channels have genuinely different statistics, and using a Gaussian for all of them would be
wrong in a way that matters.

| Channel | Likelihood | Justification |
|---|---|---|
| **Thomson** | **Poisson** on photoelectron counts | 0.008 pe/channel/shot (doc 02 §7.1). Gaussian is invalid in the counting regime and would badly misstate the uncertainty |
| **OES** | Poisson for weak lines; Gaussian for bright lines above ~100 pe | Explicit switch at a configured threshold |
| **LIF** | Gaussian with heteroscedastic variance | Signal is accumulated; variance is signal-dependent |
| **Interferometry** | Gaussian, with a **correlated** noise covariance | Vibration noise is coloured, not white — a diagonal covariance would overstate the information |
| — | **gated to zero below the detection floor** | doc 01 IF-6 |

### 3.2 The asynchronous structure

Doc 02 §10 established that the channels never measure simultaneously. The likelihood is
therefore written over a shared latent time base, with each observation carrying its own
acquisition window:

```
log p(y | x)  =  Σ_channels Σ_observations  log p( y_{k,i} | ∫_{W_{k,i}} F_k(x(t)) dt )
```

where `W_{k,i}` is the acquisition window (gate width, or accumulation interval) of the
`i`-th observation on channel `k`. For cyclo-stationary RF operation the integral is over the
phase bin rather than over absolute time.

**Getting this right is what makes multi-diagnostic fusion legitimate.** Treating a 3-hour
Thomson accumulation and a 2 ns OES gate as measurements of "the same instant" would be
straightforwardly false, and the resulting agreement or disagreement between channels would
be meaningless.

### 3.3 Robustness to outliers

The nominal likelihood is supplemented by an optional heavy-tailed variant (Student-t, or an
explicit mixture with a broad outlier component). Discrete failures — cosmic rays, fringe
jumps, MCP ion feedback (doc 04 §7.2) — produce occasional wildly wrong data points, and a
Gaussian likelihood will contort the entire fit to accommodate one of them. The outlier
fraction is itself an inferred parameter, and its posterior serves as an **automatic data-quality
flag**.

---

## 4. Priors: where the physics enters

### 4.1 The primary physics constraint is structural

The strongest physics constraint is not a prior term at all — it is that **`F` contains the
sheath physics**. The sampler cannot propose a state that violates Poisson's equation, because
states are generated *by solving* Poisson's equation. This is the precise meaning of
"physics-constrained" and it is worth stating plainly, because "we added a physics loss term"
is a much weaker claim that is often confused with it.

### 4.2 Explicit priors

| Prior | Form | Role |
|---|---|---|
| Bohm criterion | soft penalty on `u_s/c_s < 1` | enforces the sheath-edge condition when the discrepancy field would otherwise violate it |
| Positivity | log-parameterisation of `n`, `T` | structural |
| Profile smoothness | Gaussian-process / Tikhonov on discrepancy coefficients | suppresses non-physical oscillation |
| Quasi-neutrality in the bulk | soft penalty on `(n_i − n_e)/n_i` for `z > 5 z_s` | anchors the boundary condition |
| Instrument priors | informative Gaussians from calibration (doc 02 §11) | nuisance parameters are not free-for-all |

### 4.3 Regularisation choices are parameters, not conventions

The regularisation weight is not hand-tuned. Three options are implemented and compared:

1. **Hierarchical Bayes** — the weight is a hyperparameter with its own prior, inferred jointly. **Default.**
2. **L-curve** — for the MAP-only mode.
3. **Generalised cross-validation** — as an independent check.

Reporting the answer under only one choice would leave a reviewer to wonder how much the
result depends on it. Reporting it under all three, with the spread as a term in the error
budget, closes the question.

---

## 5. Inference engines

All implement one interface (doc 08 §4); the choice is a runtime option, and results from
different engines on the same problem are a consistency check.

| Engine | Use | Cost | Notes |
|---|---|---|---|
| **MAP** (L-BFGS-B / trust-region) | Fast point estimate; initialisation | 10²–10³ evals | Gradients from JAX/adjoint where available |
| **Laplace approximation** | Cheap Gaussian posterior at the MAP | + one Hessian | Valid only when the posterior is near-Gaussian; **its validity is tested, not assumed** |
| **HMC / NUTS** (NumPyro) | Gold-standard posterior | 10⁴–10⁶ evals | Requires differentiable forward model → L3 surrogate |
| **SMC / nested sampling** | Multimodal posteriors; evidence for model comparison | 10⁵–10⁶ | Handles the bimodality that appears in the RF regime |
| **Ensemble Kalman filter** | Time-resolved tracking of transients | 10²–10³ per step | Regime G, where a full posterior per timestep is unaffordable |
| **Particle filter** | Non-Gaussian tracking | 10³–10⁴ per step | Fallback when EnKF's Gaussian assumption fails |

**The Laplace-validity test is a small thing that prevents a large error.** Laplace is
tempting because it is cheap, and it is wrong whenever the posterior is skewed or multimodal —
which, in this problem, it demonstrably is near the identifiability boundary. The framework
runs Laplace and NUTS on a subset of cases and reports the divergence between them; Laplace is
only permitted where that divergence is below a threshold.

---

## 6. Identifiability, information content, and the null space

**This section is what elevates the work from an inversion to an analysis of an inversion.**

### 6.1 Fisher information

```
I(θ)  =  Jᵀ Σ⁻¹ J          J = ∂F/∂θ  (Jacobian, via autodiff or adjoint)
```

The Cramér–Rao bound `Cov(θ̂) ⪰ I⁻¹` gives the best achievable precision for *any* unbiased
estimator. Comparing the actual posterior covariance to `I⁻¹` answers a question that is
rarely asked and always relevant: **is the inversion extracting the information that is
present, or is it leaving some on the table?**

### 6.2 Eigen-analysis and the null space

Eigen-decomposing `I(θ)`:

- **Large eigenvalues** → well-determined parameter combinations;
- **Small eigenvalues** → the near-null space: combinations the data barely constrain;
- **The eigenvectors name the degeneracies.** The expected one, based on the physics, is a
  `n₀`–`T_e` correlation, since `Γ_i ∝ n₀ √T_e` means the data constrain the product far
  better than either factor. Naming such a degeneracy explicitly, and showing which channel
  breaks it, is a genuinely useful scientific statement.

The condition number `λ_max/λ_min` is reported for every operating point, producing the
**identifiability map** required by doc 00 S5: the operating space partitioned into
identifiable / weakly identifiable / non-identifiable regions.

### 6.3 Per-channel information content — the rigorous version of "explainability"

The planning conversations asked for a contribution breakdown ("density 38 %, temperature
22 %…"). The defensible way to produce that is information-theoretic, not ad hoc:

```
Channel k's contribution  =  H( θ | y_{−k} )  −  H( θ | y )
```

the reduction in posterior entropy attributable to channel `k` — equivalently, the mutual
information between `θ` and channel `k`'s data given the others. Computed by:

1. **Leave-one-channel-out** posterior comparison (exact, expensive);
2. **Marginal Fisher information** `log det I(θ)` with and without channel `k` (cheap, and the
   standard D-optimality criterion).

This yields a defensible statement — "LIF supplies 61 % of the information about `⟨E_i⟩` at
RP-1, and its removal inflates the `Γ_E` credible interval by 2.4×" — where an ad hoc
percentage would be indefensible. **It also closes ADR-004 quantitatively**: if
interferometry's marginal `log det I` contribution is negligible across the envelope, the
channel is dropped, and that is a publishable negative result.

### 6.4 Global sensitivity

Fisher information is local. Sobol indices from the polynomial-chaos surrogate (doc 03 §5.3)
give the global picture: which parameters drive the variance in `Γ_E` across the *whole*
envelope, not just at one point. First-order and total-effect indices are both reported —
their difference reveals interaction effects, which is where the interesting physics tends to
hide.

### 6.5 Profile likelihood

For parameters whose posteriors are strongly non-Gaussian, the profile likelihood

```
PL(θ_j)  =  max_{θ_{−j}} log p(y | θ)
```

is computed. A flat profile is the unambiguous signature of structural non-identifiability, and
it distinguishes that condition from mere large-but-finite uncertainty — a distinction the
posterior width alone cannot make.

---

## 7. Avoiding the inverse crime

**An inverse crime is committed when the same model and discretisation generate the data and
perform the inversion.** The result is an artificially perfect recovery that proves nothing.
This is the most common way computational inverse-problem work is invalidated, and it is
guarded against structurally rather than by good intentions.

### 7.1 Mandatory mismatches

| Aspect | Truth generation | Inversion |
|---|---|---|
| Physics level | L2 (PIC-MCC) | L3 surrogate trained on a *differently seeded* L2 ensemble |
| Spatial grid | `Δz = λ_D/2`, graded mesh A | `Δz = λ_D/3`, graded mesh B |
| Time discretisation | `Δt = 0.2/ω_pe` | adaptive BDF2 |
| Collision set | full LXCat set | reduced set (dominant processes only) |
| EEDF | as computed by PIC | κ-distribution parameterisation |
| Calibration | true instrument response | *estimated* response from simulated calibration (doc 04 §7.3) |

### 7.2 The three-tier reporting requirement

Every headline result is reported at three levels, always together:

| Tier | Configuration | What it tests | Expected outcome |
|---|---|---|---|
| **T0 — Consistency** | Same model, no noise | The code is self-consistent | Recovery to numerical tolerance. **Failing T0 means a bug; nothing else is meaningful until it passes.** |
| **T1 — Inverse crime** | Same model, with noise | Statistical performance in the ideal case | Optimistic; the upper bound on achievable accuracy |
| **T2 — Honest** | Mismatched per §7.1, with noise and imperfect calibration | **The real result** | This is the number quoted publicly |

**Reporting T1 as if it were T2 is treated as a project defect**, and the CI enforces that any
figure showing accuracy carries its tier label. The gap between T1 and T2 is itself
informative: it is a direct measure of the framework's sensitivity to model error.

---

## 8. Model misspecification detection

A deployable instrument must know when it is wrong. Three checks run on every inversion:

| Check | Method | Action on failure |
|---|---|---|
| **Residual whiteness** | χ² of standardised residuals; autocorrelation test | Flag: the model does not explain the data structure |
| **Posterior predictive** | Simulate data from the posterior; compare to observed via test statistics | Flag: p-value outside [0.05, 0.95] |
| **Cross-channel consistency** | Invert with each channel subset; compare posteriors | Flag: disjoint credible intervals ⇒ a channel is miscalibrated or the model is wrong |

**Cross-channel consistency is the most operationally valuable of the three.** It is exactly
how a real diagnostic detects that one of its instruments has drifted, and it requires no
ground truth — meaning it works in the field, where the framework is eventually meant to run.

---

## 9. Optimal experiment design

A capability that falls out of §6 almost for free, and that converts the framework from a
results producer into a research assistant.

Given a target uncertainty on `Γ_E`, choose the next measurement to maximise expected
information gain:

```
next  =  argmax_{a ∈ actions}  E_y [ H(θ | y) − H(θ | y, y_a) ]
```

Actions include: another Thomson accumulation, a finer LIF frequency scan, an additional
interferometer chord, a different bias setpoint, a longer OES gate.

Concretely, this answers questions such as *"given 2 hours of machine time, is it better to
improve the Thomson statistics or to scan LIF at a second spatial position?"* — and it can
answer the procurement question *"would buying an E-FISH system (doc 01 §4.2) be worth it?"*
**before** the money is spent, by inserting the proposed channel as a hypothetical and
computing its information gain. That is a directly commercialisable capability.

---

## 10. Outputs

Every inversion emits:

| Output | Contents |
|---|---|
| Posterior samples | `θ_c`, `θ_n`, `θ_f` |
| Derived posterior | `Γ_E(z, t)` with credible intervals; `Γ_i`; `⟨E_i⟩`; IEDF at the wall |
| Diagnostics | R̂, ESS, divergences, trace plots, energy plots |
| Identifiability | FIM eigenvalues/vectors, condition number, CRB comparison, profile likelihoods |
| Information | per-channel entropy reduction, Sobol indices |
| Misspecification | residual χ², posterior-predictive p-values, cross-channel consistency |
| Provenance | manifest hash, code commit, seed, tier label (T0/T1/T2) |

---

## 11. Open questions

| ID | Question | Resolution path |
|---|---|---|
| Q-05 | Is the posterior multimodal in the RF regime (phase-offset degeneracy)? | Nested sampling on regime D; if yes, NUTS alone is insufficient |
| Q-06 | Does the discrepancy field `θ_f` absorb so much that `θ_c` becomes unidentifiable? | Profile likelihood on `τ`; tune basis size `N_b` |
| Q-07 | Is the Laplace approximation adequate anywhere useful? | §5 divergence test across the regime map |
| Q-08 | Can EnKF track regime G given that Thomson is blind there? | Benchmark B-08 |
| Q-09 | What is the emulator-error contribution relative to measurement noise? | Doc 03 §5.4 audit, then doc 06 budget |

---

## 12. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Three-level parameterisation, per-channel likelihoods matched to actual photon statistics, mandatory inverse-crime mismatches, T0/T1/T2 reporting discipline. |
