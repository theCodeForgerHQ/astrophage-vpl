# 03 — Physics Model Specification (the forward operator `F₁`)

Version 1.0 · Status: **Baseline** · Owner: Ajayaditya L

> This document specifies everything that turns a set of control parameters into a plasma
> state `x`. It is the first stage of the forward operator, `F₁`. Nothing here is novel —
> every model is textbook or published, which is a hard constraint (doc 00 C2), and every
> equation is stated so that the implementation has no interpretive freedom.

---

## 1. The model hierarchy

A single interface, four fidelity levels. The inverse solver never learns which one it is
talking to (doc 00 E1).

| Level | Name | Physics | Cost per solve | Role |
|---|---|---|---|---|
| **L0** | Analytic | Child–Langmuir, matrix sheath, Bohm criterion, Lieberman RF sheath | µs | Verification anchor; MCMC proposal; sanity bound |
| **L1** | Fluid | Drift-diffusion ions + Boltzmann electrons + Poisson | ~1 s | Workhorse for sweeps; valid where the IVDF is near-drifting-Maxwellian |
| **L2** | Kinetic | 1D3V electrostatic PIC with Monte-Carlo collisions | ~1–10 min | Ground truth generator; the only level that produces a correct IEDF in the collisional and RF-transit regimes |
| **L3** | Surrogate | GP / neural-operator emulator trained on L2 | ~1 ms | What the sampler actually calls; error is budgeted, not ignored |

**The hierarchy is not decoration.** It exists because of a specific computational fact
established in §7: a single L2 solve is cheap (minutes), but the Bayesian inversion needs
10⁵–10⁶ forward solves, which L2 cannot supply. L3 closes that gap, and L0/L1 provide the
independent checks that keep L3 honest.

**Model-selection discipline.** The level used to *generate* ground truth and the level used
*inside* the inversion must differ. Using L2 for both is an inverse crime (doc 05 §7) and
produces meaningless accuracy. The default configuration is: **truth from L2, inversion via
L3 trained on a differently-seeded L2 ensemble, cross-checked against L1.**

---

## 2. L0 — Analytic sheath models

### 2.1 Bohm criterion and the sheath edge

Ions must enter the sheath at or above the ion-acoustic speed:

```
u_s  ≥  c_s  =  √( e (T_e + γ_i T_i) / m_i )          γ_i = 3 for 1-D adiabatic ions
```

The sheath edge `z_s` is defined by the standard criterion that the quasi-neutrality
parameter departs by a chosen tolerance:

```
z_s  :  (n_i − n_e) / n_i  =  δ ,     δ = 0.01   (a DESIGN parameter, swept)
```

**The sheath edge is a definition, not a physical surface.** Different definitions shift `z_s`
by a fraction of a Debye length and therefore shift the flux. `δ` is registered as a
parameter and appears in the sensitivity study — this is precisely the sort of hidden
convention that doc 00 C4 exists to forbid.

Density at the sheath edge, with `h_l` the standard planar edge-to-centre ratio:

```
n_s  =  h_l · n_0 ,     h_l ≈ 0.86 / √(3 + L/(2λ_i))         (Godyak, low-pressure)
```

### 2.2 Matrix sheath

Uniform ion density, no ion acceleration. Valid only for `V ≪ T_e`; used as a limiting case
and as a numerical initialisation.

```
Φ(z)  =  −V_w (1 − z/s)² ,      s = λ_D √(2 V_w / T_e)
```

### 2.3 Collisionless Child–Langmuir sheath

The high-voltage limit, and the framework's primary analytic verification target:

```
J_i  =  (4/9) ε₀ √(2e/m_i) · V_w^{3/2} / s²
```

Matching `J_i` to the Bohm flux `e n_s c_s` gives the sheath thickness

```
s  =  (√2 / 3) · λ_D · ( 2 V_w / T_e )^{3/4}
```

which is the expression used numerically in doc 01 §2.2. **Recovery of this scaling by the
L1 and L2 solvers in the collisionless, high-bias limit is verification gate V-03
(doc 07).**

Ion energy flux in this limit:

```
Γ_E  =  Γ_i · e V_w  =  h_l n_0 c_s · e V_w
```

At RP-1: `Γ_i = 0.61 × 10¹⁷ × 2691 = 1.64 × 10²⁰ m⁻² s⁻¹`, `⟨E_i⟩ = 250 eV`, giving
**`Γ_E ≈ 6.6 kW·m⁻² = 0.66 W·cm⁻²`**. This is the number every other model must reproduce in
this limit.

### 2.4 Collisional (mobility-limited) sheath

When `s ≳ λ_CX`, ion motion becomes mobility-limited and the current–voltage relation
changes character:

```
J_i  =  (9/8) ε₀ µ_i V_w² / s³                (constant-mobility regime)
J_i  ∝  V_w^{3/2} / s^{5/2}                    (constant-mean-free-path regime, Warren)
```

The framework implements the constant-mean-free-path variant as the default because Ar⁺/Ar
charge exchange has a weakly energy-dependent cross section, making `λ_CX` the better-behaved
constant. Both are available; the choice is a registered parameter.

### 2.5 RF sheath (Lieberman)

For the RF regimes D and E of doc 02 §3.3, the standard collisionless RF sheath model gives
the time-averaged sheath thickness and the ion energy distribution width:

```
ΔE_i  ≈  (2 e V_s / (ω τ_tr)) · ... →  bimodal IEDF with peak separation ΔE
```

with the two peaks separating as `τ_tr / T_RF → 0` and merging as `τ_tr / T_RF → ∞`. At RP-1,
`τ_tr / T_RF = 26/73.7 = 0.35`, placing the reference point **squarely in the partially
resolved regime** where neither limit applies — which is why L2 is required for regime D and
L0 serves only as a bound.

---

## 3. L1 — Fluid model

### 3.1 Governing equations

One-dimensional, in the sheath-normal coordinate `z`, time-dependent.

**Ion continuity:**
```
∂n_i/∂t  +  ∂(n_i u_i)/∂z  =  S_iz  −  L_rec
```

**Ion momentum (drift-diffusion with inertia):**
```
∂(n_i u_i)/∂t  +  ∂(n_i u_i²)/∂z  =  (e n_i / m_i) E  −  (1/m_i) ∂(n_i k T_i)/∂z  −  n_i ν_in u_i
```

**Electrons (inertialess, Boltzmann-distributed):**
```
n_e  =  n_0 exp( e Φ / (k T_e) )
```

**Poisson:**
```
−ε₀ ∂²Φ/∂z²  =  e (n_i − n_e)
```

**Electron energy** (optional, for non-isothermal runs):
```
∂/∂t (3/2 n_e k T_e)  +  ∂/∂z (q_e + 5/2 Γ_e k T_e)  =  −e Γ_e E  −  n_e Σ_j k_j ΔE_j
```

### 3.2 The Boltzmann-electron assumption and where it breaks

`n_e = n_0 exp(eΦ/kT_e)` requires a Maxwellian EEDF and negligible electron inertia. In a
low-pressure sheath both are questionable: the EEDF is typically bi-Maxwellian or Druyvesteyn,
and the high-energy tail — the part that matters for ionisation and for OES line ratios — is
depleted.

**This is handled, not assumed away.** The framework computes the EEDF and the resulting rate
coefficients with a **two-term Boltzmann solver** (BOLSIG+ or its Python equivalent) over the
local reduced field `E/N`, and tabulates:

```
k_iz(E/N),  k_ex,j(E/N),  µ_e(E/N),  D_e(E/N),  ⟨ε⟩(E/N)
```

The Maxwellian assumption is then retained only for the *density* relation, with the
non-Maxwellian tail entering through the rate coefficients. The residual error from this
inconsistency is a budgeted term (doc 06 §4) and is bounded by comparison against L2, which
makes no such assumption.

### 3.3 Boundary conditions

| Boundary | Condition |
|---|---|
| Wall (`z = 0`) | `Φ = V_w`; ion flux absorbed; secondary electron emission with yield `γ_se(E_i, material)`; ion reflection coefficient `R_i` |
| Bulk (`z = L`) | `Φ = 0`; `n_i = n_e = n_0`; ion flux entering at `u = c_s` |

**Secondary electron emission is not optional.** For tungsten at 250 eV Ar⁺ impact,
`γ_se ≈ 0.1`. Secondary electrons are accelerated back through the full sheath potential,
gaining 250 eV, and they modify both the sheath structure and — importantly — the OES
emission profile inside the sheath. Omitting them is a common and significant modelling
error. `γ_se` is drawn from published ion-induced-emission data (doc 09) and its uncertainty
is propagated.

### 3.4 Numerics

| Aspect | Choice | Rationale |
|---|---|---|
| Spatial discretisation | Finite element (FEniCSx), P1/P2 Lagrange | Poisson and the coupled system in one framework; adjoints available via `dolfin-adjoint` |
| Ion advection | Scharfetter–Gummel exponential fitting | Avoids the spurious oscillations that plague centred differences in the strongly-drift-dominated sheath |
| Coupling | Newton on the fully coupled `(n_i, u_i, Φ)` system | The `n_e(Φ)` exponential makes Gummel iteration converge poorly at high bias |
| Time integration | Implicit BDF2, adaptive | The dielectric relaxation time `ε₀/σ` is far shorter than the ion timescale; explicit is unusable |
| Mesh | Graded, refined toward the wall | `λ_D` at the wall is the controlling scale; uniform meshing wastes 90 % of the cells |
| Mesh resolution | `Δz ≤ λ_D / 4` in the sheath | Resolve the Debye scale; verified by mesh-refinement study V-04 |

---

## 4. L2 — Kinetic model (PIC-MCC)

### 4.1 Configuration

**1D3V electrostatic particle-in-cell with Monte-Carlo collisions.** One spatial dimension
(the sheath normal), three velocity components (needed because LIF measures a projected
velocity and Thomson measures the full electron distribution).

Electrostatic rather than electromagnetic: the sheath is far smaller than any electromagnetic
wavelength at 13.56 MHz, and `v/c ≈ 10⁻³`. Electromagnetic PIC would be pure waste.

### 4.2 Algorithm

```
for each timestep:
    1. Deposit charge to grid          (linear / CIC weighting)
    2. Solve Poisson on grid           (direct tridiagonal or FEM)
    3. Interpolate field to particles  (same weighting — required for momentum conservation)
    4. Push particles                  (Boris / leapfrog)
    5. Monte-Carlo collisions          (null-collision method)
    6. Apply boundary conditions       (absorption, SEE, reflection, injection)
```

### 4.3 Stability and accuracy constraints

These are hard constraints, checked at runtime, and violating them is a runtime error rather
than a warning:

```
Δz   ≤  λ_D                   (avoid numerical grid heating)   → Δz = λ_D/2 = 20 µm
Δt   ≤  0.2 / ω_pe            (resolve electron plasma oscillation)
                              ω_pe = 1.78 × 10¹⁰ s⁻¹  →  Δt ≤ 11.2 ps
Δt   ≤  Δz / v_max            (CFL: no particle crosses more than one cell)
N_ppc ≥ 100                   (statistical noise)
```

**Statistical noise** in PIC scales as `N_ppc^(-1/2)`. Because the IEDF at the wall — a
*distribution*, not a moment — is the deliverable, and because distribution tails converge
far more slowly than means, the framework uses `N_ppc = 1000` by default and performs an
explicit `N_ppc` convergence study on the *IEDF shape* (Kolmogorov–Smirnov distance between
successive refinements), not merely on the density profile. Converging the mean while the
tail is still noise is a classic and silent PIC failure.

### 4.4 Estimated cost at RP-1

| Quantity | Value |
|---|---|
| Domain length | 20 mm (≈ 22 × sheath thickness) |
| Cells (`Δz = 20 µm`) | 1 000 |
| Particles (`N_ppc = 1000`, 2 species) | 2 × 10⁶ |
| `Δt` | 11.2 ps |
| Simulated time (10 RF periods) | 737 ns |
| Timesteps | 65 800 |
| Particle-steps | 1.3 × 10¹¹ |
| **Estimated wall clock (16 CPU cores at ~5 × 10⁷ particle-steps/s/core)** | **~3 minutes** |

**A single L2 solve is cheap. An ensemble of them is not.** This asymmetry is the central
computational fact of the project and it is what §1 anticipated: 10⁵ forward evaluations at
3 minutes each is 200 core-years. Hence L3.

> These are *estimates to be measured*, not claims. Phase 1 acceptance gate G-1.4 (doc 11)
> requires the measured throughput to be recorded and this table corrected.

### 4.5 Collision processes

| Process | Reaction | Source |
|---|---|---|
| Electron elastic | e + Ar → e + Ar | LXCat (Phelps / Biagi) |
| Electron excitation | e + Ar → e + Ar* (multiple levels) | LXCat |
| Electron ionisation | e + Ar → 2e + Ar⁺ | LXCat |
| Ion elastic (isotropic) | Ar⁺ + Ar → Ar⁺ + Ar | LXCat (Phelps) |
| Ion charge exchange | Ar⁺ + Ar → Ar + Ar⁺ | LXCat (Phelps) |
| Secondary emission | Ar⁺ + W(surface) → γ_se e⁻ | published ion-induced-emission data |

**Charge exchange is the single most important collision for this project.** A CX event
replaces a fast ion with a slow one at the local potential, which then re-accelerates through
the *remaining* potential drop. The result is the low-energy structure of the IEDF that
dominates the collisional regime, and it is what makes `⟨E_i⟩` differ substantially from
`e V_w`. Getting the CX cross section and its energy dependence right matters more than any
other atomic-data choice.

The **null-collision method** is used so that the collision test is a single uniform random
draw per particle per step regardless of the number of processes — standard, and essential
for performance.

---

## 5. L3 — Surrogate model

### 5.1 Why it is required

Bayesian inversion over ~15 parameters with MCMC needs 10⁵–10⁶ forward evaluations. L2 costs
minutes. L3 must cost milliseconds. This is a standard multi-fidelity UQ construction, not a
research problem (doc 00 C2).

### 5.2 What is emulated

Not the whole solution field — only the quantities the likelihood needs:

```
inputs:   θ = (n_0, T_e, T_i, p, V_w, γ_se, φ_RF, …)      ~15 parameters
outputs:  n_e(z), n_i(z), Φ(z), f_i(v_z; z=0), T_e(z)     projected onto a reduced basis
```

Field outputs are compressed by POD/PCA to ~20 coefficients before emulation, so the
emulator maps ℝ¹⁵ → ℝ~⁸⁰ rather than to a full grid.

### 5.3 Candidate emulators

| Option | Pros | Cons | Disposition |
|---|---|---|---|
| Gaussian process (per POD coefficient) | Native uncertainty; excellent with few hundred training points; interpretable | Scales poorly beyond ~10⁴ training points | **Default** |
| Neural operator (DeepONet / FNO) | Handles large training sets; resolves sharp fronts | Needs 10⁴+ samples; uncertainty requires ensembling | Alternative for the large-compute tier |
| Polynomial chaos | Cheap; analytic Sobol indices | Struggles with the sharp sheath front | Used for global sensitivity only |

**The GP is the default because it reports its own uncertainty.** That uncertainty enters the
likelihood as an additional variance term, so the inversion is automatically more cautious in
regions where the emulator is poorly trained. An emulator without uncertainty would silently
inject bias — the exact failure mode doc 00 §6 lists as the uncalibrated-posterior trap.

### 5.4 Emulator error is budgeted, not assumed small

Mandatory, enforced as a validation gate:

- Held-out test set of ≥ 200 independent L2 runs, never seen in training;
- Reported error metrics per output quantity, including on the IEDF *shape*;
- **Emulator error enters the error budget as its own line item** (doc 06 §4);
- The inversion is re-run at a sample of posterior modes using *full L2* to confirm the
  emulator did not distort the answer — an "emulator audit" (doc 07 V-11).

---

## 6. Computing the quantity of interest

At every level, the same functional is applied so that results are comparable:

```
Γ_E(z, t)  =  ∫ (½ m_i |v|²) v_z f_i(z, v, t) d³v
```

| Level | How `f_i` is obtained |
|---|---|
| L0 | Assumed drifting Maxwellian at `u = √(2eV_w/m_i)`, or the analytic collisional IEDF |
| L1 | Reconstructed as a drifting Maxwellian from the fluid moments `(n_i, u_i, T_i)` — **an approximation, and its error vs L2 is measured and reported** |
| L2 | Directly, by binning the particle population crossing `z` |
| L3 | Emulated IEDF, reconstructed from POD coefficients |

**The L1 reconstruction is a genuine weakness and is treated as one.** A fluid model cannot
represent the bimodal RF IEDF or the collisional low-energy tail; forcing a drifting
Maxwellian onto them produces a systematically wrong `Γ_E`. Benchmark B-03 (doc 07) quantifies
exactly how wrong, across the regime map. **The result — a map of where fluid modelling is
adequate and where kinetics is mandatory — is a publishable output in its own right**, and it
is the sort of finding that comes free from building the hierarchy properly.

---

## 7. Verification strategy for `F₁`

Verification asks whether the equations are solved correctly. It is distinct from validation,
which asks whether the equations describe reality (doc 07 §1).

| ID | Test | Pass criterion |
|---|---|---|
| V-01 | Method of manufactured solutions, Poisson | Observed order = design order ± 0.1 |
| V-02 | MMS, coupled fluid system | Observed order = design order ± 0.15 |
| V-03 | Child–Langmuir limit (L1, L2 → L0) | `s` and `J_i` within 5 % in the collisionless high-bias limit |
| V-04 | Mesh independence | `Γ_E` changes < 1 % on halving `Δz` |
| V-05 | Timestep independence | `Γ_E` changes < 1 % on halving `Δt` |
| V-06 | `N_ppc` convergence (IEDF shape) | KS distance < 0.02 on doubling `N_ppc` |
| V-07 | Energy conservation (PIC, collisionless) | Total energy drift < 0.1 % over the run |
| V-08 | Momentum conservation | Consistent field interpolation/deposition verified |
| V-09 | Bohm criterion satisfied at `z_s` | `u_s / c_s = 1.0 ± 0.05` |
| V-10 | L1 vs L2 agreement in the fluid-valid regime | `Γ_E` within 10 % at low pressure, DC bias |
| V-11 | Emulator audit | L3 posterior modes re-evaluated with L2 agree within emulator-reported uncertainty |

---

## 8. Assumptions register for `F₁`

Every assumption, its justification, where it breaks, and how it is handled. Doc 00 C4 in
practice.

| # | Assumption | Justification | Breaks when | Handling |
|---|---|---|---|---|
| A1 | 1-D planar geometry | Electrode diameter (150 mm) ≫ sheath (0.89 mm) | Near electrode edges | Restrict analysis to the central region; edge effects a separate 2-D study |
| A2 | Electrostatic | `v/c ≈ 10⁻³`; sheath ≪ EM wavelength | Never, in this envelope | — |
| A3 | Single ion species (Ar⁺) | Ar⁺⁺ fraction < 1 % at `T_e` = 3 eV | High `T_e` (> 6 eV) | Ar⁺⁺ channel available as an option; flagged in high-`T_e` runs |
| A4 | Cold, uniform neutral background | Gas heating small at these powers | High power / high pressure | Neutral temperature is a parameter; sensitivity swept |
| A5 | Boltzmann electrons (L1 only) | Standard for low-pressure sheaths | Strong EEDF depletion | L2 makes no such assumption; L1–L2 difference is measured (V-10) |
| A6 | Maxwellian EEDF for rate coefficients | — | Low pressure, high field | **Replaced** by two-term Boltzmann solver output (§3.2) |
| A7 | Constant `γ_se` | Weak energy dependence over the range | Very high bias | Energy-dependent `γ_se(E)` available; uncertainty propagated |
| A8 | No dust, no negative ions | Ar discharge; clean chamber | Electronegative gases (Cl₂, O₂) | Out of scope for v1; noted as a limitation |
| A9 | Cyclo-stationarity in RF | Required for phase-locked accumulation (doc 02 §10.3) | Discharge instability | Drift model + benchmark B-09 quantify the bias |
| A10 | Ion reflection at wall neglected | `R_i` small for Ar⁺ on W at these energies | Grazing incidence, light ions | `R_i` is a parameter, default 0, swept |

**A8 deserves emphasis as a stated limitation.** Real semiconductor etch plasmas are
electronegative, and negative ions change the sheath structure qualitatively. The framework
is specified for electropositive Ar in v1. Claiming etch-reactor applicability without saying
this would be an overclaim; the commercialisation document (doc 12) reflects the honest scope.

---

## 9. Build vs buy

| Component | Decision | Rationale |
|---|---|---|
| Poisson / FEM solver | **Buy** — FEniCSx | Mature, verified, adjoint-capable |
| Mesh generation | **Buy** — Gmsh | — |
| Boltzmann (EEDF) solver | **Buy** — BOLSIG+ / `bolos` | The standard tool; reimplementation is pure risk |
| Cross-section data | **Buy** — LXCat | Curated, cited, versioned |
| PIC engine | **ADR-003, open** | Smilei/WarpX are excellent but are electromagnetic, 3-D-oriented and heavy for a 1D3V electrostatic sheath. A purpose-built 1D3V kernel is ~1500 lines of well-trodden, textbook algorithm and would be far easier to verify and to couple to the surrogate pipeline. **Leaning purpose-built**, with Smilei retained as an independent cross-check for V-10 — which is arguably the stronger scientific position, since it gives two independent kinetic implementations. |
| GP emulator | **Buy** — GPyTorch / scikit-learn | — |
| Sparse linear algebra | **Buy** — PETSc | — |

---

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Four-level hierarchy specified; L2 cost estimated at ~3 min/solve, establishing the ensemble bottleneck that justifies L3. |
