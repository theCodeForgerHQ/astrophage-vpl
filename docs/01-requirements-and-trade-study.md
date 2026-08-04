# 01 — Requirements Derivation and Diagnostic Trade Study

Version 1.0 · Status: **Baseline** · Owner: Team Astrophage

> **Why this document comes first.**
> The natural instinct is to design the system, or to start coding. Both are wrong. If we
> begin by asserting "we use OES, LIF, Thomson and interferometry", the first competent
> question — *"why those four, and why not Stark polarimetry, E-FISH, microwave
> interferometry, or a retarding-field analyser?"* — has no answer except preference.
> This document derives the measurement requirements from the physics of the quantity we
> are trying to recover, and only then selects instruments against those requirements. The
> four diagnostics in the proposal survive this process; two additional ones are promoted
> to the roadmap because the analysis says they should be; several are rejected with
> reasons on the record.

---

## 1. The quantity of interest, stated exactly

### 1.1 Definition

The ion energy flux delivered to a plasma-facing surface, resolved in space and time:

```
Γ_E(z, t)  =  ∫ (½ m_i |v|²) · v_z · f_i(z, v, t) d³v            [W · m⁻²]
```

`f_i` is the ion velocity distribution function (IVDF), `v_z` the velocity component normal
to the surface, `m_i` the ion mass. At the wall (`z = z_w`) this becomes the surface energy
deposition rate that drives sputtering, heating and erosion.

### 1.2 The decomposition that drives everything downstream

Write the flux as a product of a particle flux and a mean energy:

```
Γ_E(z_w, t)  =  Γ_i(z_w, t) · ⟨E_i⟩(z_w, t)
```

where

```
Γ_i  =  ∫ v_z f_i d³v                         ion particle flux        [m⁻² s⁻¹]
⟨E_i⟩ =  Γ_E / Γ_i                             flux-weighted mean ion impact energy   [J]
```

This is not merely algebra. It is the **information decomposition of the problem**: the two
factors are constrained by *different* physics and are therefore observable by *different*
instruments.

### 1.3 What each factor requires

**The particle flux** is set at the sheath edge by the Bohm criterion. For a collisionless
planar sheath,

```
Γ_i  =  n_s · u_s  ,     u_s ≥ c_s = √(e T_e / m_i)  ,     n_s ≈ h_l · n_0
```

so `Γ_i` is determined by **electron temperature** `T_e` and **plasma density at the sheath
edge** `n_s` — both bulk, pre-sheath quantities. Note the striking consequence: *the ion
particle flux is controlled by the electron temperature*, because it is the electrons that
set the ambipolar field that accelerates ions to the Bohm speed.

**The mean impact energy** is set by the sheath potential drop and by what happens to ions
during transit:

```
⟨E_i⟩  =  e·(Φ_s − Φ_w)  +  ⟨E_i⟩_entry  −  ΔE_collisions
```

so `⟨E_i⟩` requires the **sheath potential drop**, the **entry IVDF at the sheath edge**,
and the **collisionality inside the sheath**. If the sheath is collisionless the ion energy
distribution at the wall is a narrow, shifted replica of the entry distribution. If it is
collisional, charge-exchange creates a low-energy tail that can carry a large fraction of
the particle flux at a small fraction of the energy — and the resulting IEDF is bimodal or
broadly structured. The two cases differ by a factor of order unity in `Γ_E` and by an
order of magnitude in the *shape* of the surface energy spectrum, which is what actually
determines sputter yield.

### 1.4 Requirement flow-down, level 0

| ID | Derived necessity | From |
|---|---|---|
| **N1** | Electron temperature `T_e`, spatially resolved through the presheath | §1.3, Bohm flux |
| **N2** | Electron/ion density `n_e ≈ n_i`, spatially resolved, absolute (not relative) | §1.3, Bohm flux |
| **N3** | Ion velocity distribution `f_i(v_z)` at and inside the sheath edge | §1.3, entry IVDF |
| **N4** | Sheath potential profile `Φ(z)`, or an observable that constrains it | §1.3, energy gain |
| **N5** | Sheath spatial extent `s` (to locate `z_s`, and to place all of the above) | all of the above are *position-dependent* |
| **N6** | Neutral gas density `n_g` and species composition | sets collisionality inside the sheath |
| **N7** | Time resolution sufficient to follow the sheath's dynamic response | §2.3 |

**N2 must be absolute.** This is a frequently underestimated requirement. A relative density
profile is insufficient because `Γ_i` scales linearly with `n_s`; a 20 % calibration error
in density is a 20 % error in the answer, propagated directly. This single observation is
what forces an absolutely-calibrating diagnostic into the set (§4.3).

---

## 2. Quantitative requirements from the reference operating point

All requirements below are derived at a **reference operating point** chosen to be
representative of both the semiconductor-etch and the electric-propulsion regimes, and
deliberately placed in the *interesting* part of parameter space where the sheath is neither
fully collisionless nor fully collisional and neither fully static nor fully RF-slaved.

### 2.1 Reference operating point (RP-1)

| Parameter | Symbol | Value | Class |
|---|---|---|---|
| Working gas | — | Argon | DESIGN |
| Neutral pressure | `p` | 5 mTorr (0.667 Pa) | DESIGN |
| Gas temperature | `T_g` | 300 K | DESIGN |
| Bulk electron density | `n_0` | 1 × 10¹⁷ m⁻³ | DESIGN |
| Electron temperature | `T_e` | 3 eV | DESIGN |
| Ion temperature | `T_i` | 0.05 eV (≈ T_g) | DESIGN |
| Applied bias | `V_b` | −250 V (DC) or 13.56 MHz RF | DESIGN |
| Wall material | — | Tungsten | DESIGN |

### 2.2 Derived length scales

```
Debye length            λ_D = √(ε₀ T_e[eV] / (n₀ e))            =  40.7 µm
Bohm speed              c_s = √(e T_e / m_i)                     =  2.69 km/s
Sheath thickness        s ≈ (√2/3) λ_D (2V_b/T_e)^(3/4)          ≈  0.89 mm
   (Child–Langmuir, matched to the Bohm flux at the sheath edge)
Neutral density         n_g = p / (k_B T_g)                      =  1.61 × 10²⁰ m⁻³
CX mean free path       λ_CX = 1/(n_g σ_CX),  σ_CX ≈ 5 × 10⁻¹⁹ m²=  12.4 mm
Collisionality          s / λ_CX                                  =  0.072   (near-collisionless)
```

At 50 mTorr the same formulas give `λ_CX ≈ 1.24 mm` and `s/λ_CX ≈ 0.7` — a genuinely
collisional sheath with a strongly non-Maxwellian, structured IEDF. **The pressure axis of
the operating envelope therefore spans the collisionless-to-collisional transition**, which
is exactly where a framework that only assumes one limit will fail and where a framework
that models both will demonstrate its value.

> `σ_CX` at RP-1 is a `PUBLISHED` value; the actual implementation uses the
> energy-dependent Ar⁺/Ar symmetric charge-exchange cross section from LXCat rather than a
> constant. See doc 09.

### 2.3 Derived time scales

```
RF period (13.56 MHz)                 T_RF     =  73.7 ns
Ion plasma period    ω_pi = √(n e²/(ε₀ m_i)) → f_pi ≈ 10.5 MHz → T_pi ≈ 95 ns
Ion transit time across sheath        τ_tr = s / √(2eV_b/m_i)  ≈  26 ns
Electron plasma period (f_pe ≈ 2.84 GHz)                        ≈  0.35 ns
```

**This is the single most important line in this document:**

```
τ_tr ≈ 26 ns    is comparable to    T_RF ≈ 74 ns
```

The ion transit time and the RF period are the same order. Ions therefore neither follow the
instantaneous field (which would give a wall IEDF tracking the RF waveform) nor see only the
time-average (which would give a single narrow peak). They experience a **partially phase-resolved
field**, producing the characteristic bimodal RF IEDF whose peak separation depends on
`τ_tr / T_RF`. This is a genuinely non-equilibrium, time-dependent phenomenon, it is
well documented in the literature, it is *not* an unsolved problem, and it is invisible to
any time-averaged measurement.

**It is the physical justification for the word "time-resolved" in the proposal title, and
it converts that word from a marketing adjective into a derived requirement.**

### 2.4 Requirements flow-down, level 1 (quantitative)

| ID | Requirement | Value | Derivation |
|---|---|---|---|
| **R-SPAT-1** | Spatial resolution normal to the wall | ≤ 90 µm | resolve `s ≈ 0.89 mm` with ≥ 10 samples |
| **R-SPAT-2** | Measurement volume must reach within | ≤ 100 µm of the wall | IEDF at the surface is the deliverable |
| **R-SPAT-3** | Field of view must extend to | ≥ 10 × s ≈ 10 mm | capture presheath and bulk for boundary conditions |
| **R-TEMP-1** | Temporal resolution | ≤ 5 ns | resolve `T_RF = 73.7 ns` into ≥ 15 phase bins |
| **R-TEMP-2** | Timing jitter between channels | ≤ 1 ns RMS | phase-resolved fusion across instruments is meaningless otherwise |
| **R-TEMP-3** | Slow-transient capability | 10 µs – 100 ms | ignition, mode transitions, instability, shutdown |
| **R-ACC-1** | `n_e` absolute accuracy | ≤ 10 % | propagates 1:1 into `Γ_E` |
| **R-ACC-2** | `T_e` accuracy | ≤ 15 % | enters `Γ_i` as √T_e → 7.5 % in flux |
| **R-ACC-3** | Ion drift velocity accuracy | ≤ 5 % | at the sheath edge, sets the Bohm-criterion check |
| **R-ACC-4** | Velocity resolution of the IVDF | ≤ 300 m/s | ≈ 0.1 c_s; resolves the thermal width `√(eT_i/m_i)` ≈ 350 m/s |
| **R-ACC-5** | Target accuracy on `Γ_E` | ≤ 20 % with calibrated 95 % CI | product of R-ACC-1..3 in quadrature ≈ 14 %, plus model error margin |
| **R-ENV-1** | Operating envelope: `n_e` | 10¹⁵ – 10¹⁹ m⁻³ | spans EP thruster plumes to etch reactors |
| **R-ENV-2** | Operating envelope: `T_e` | 1 – 10 eV | same |
| **R-ENV-3** | Operating envelope: `p` | 0.5 – 100 mTorr | spans collisionless → collisional |
| **R-ENV-4** | Operating envelope: `V_b` | 0 – 1000 V | spans floating to high-bias |
| **R-NON-1** | **All diagnostics must be non-perturbing** | no material object inside the plasma or sheath | the founding premise of the proposal |
| **R-NON-2** | Optical access must be geometrically realisable | ports, working distances, solid angles must close | a design that cannot be built is not a design |

**R-ACC-5 deserves comment.** A 20 % target is chosen deliberately, not conservatively. It
is roughly the level at which a diagnostic becomes decision-useful for thruster lifetime
qualification (where erosion rates scale super-linearly with impact energy) while remaining
achievable given R-ACC-1..3. Claiming 2 % would be indefensible; claiming 50 % would be
useless. The framework's job is to demonstrate *where* 20 % is achievable and where it is not.

---

## 3. Observability analysis: what can and cannot be seen

Before choosing instruments, classify the required quantities by whether any non-perturbing
technique can access them at all.

| Need | Quantity | Directly observable optically? | Notes |
|---|---|---|---|
| N1 | `T_e` | **Partially** | No technique measures `T_e` directly; all infer it from a spectral shape or ratio under a model assumption |
| N2 | `n_e` | **Yes** | Thomson (local), interferometry (line-integrated), both with absolute calibration paths |
| N3 | `f_i(v_z)` | **Yes** | LIF measures the Doppler-resolved IVDF of a specific metastable ion state — the strongest observable in the set |
| N4 | `Φ(z)` | **Weakly** | No standard technique measures potential non-intrusively at this scale; Stark-based E-field methods exist but are demanding |
| N5 | `s` | **Yes** | Sheath edge is visible as an emission discontinuity in gated imaging |
| N6 | `n_g` | **Yes** | TALIF for atoms; pressure gauge + flow model for the fill gas |
| N7 | time | **Yes** | Gated detection and phase-locked acquisition |

**Two entries are the crux of the entire project.**

`T_e` **is not directly measurable.** It is always inferred. OES line-ratio inference
requires a collisional-radiative model and is sensitive to that model's rate coefficients.
Thomson scattering measures the electron velocity distribution directly and therefore
infers `T_e` under only the far weaker assumption of a fitted distribution shape — but it is
photon-starved at 10¹⁷ m⁻³ and requires heroic stray-light rejection.

`Φ(z)` **is essentially not measurable** by routine means. This is the crucial gap. And it is
precisely the gap the physics prior fills: **Poisson's equation links `Φ(z)` to `n_e(z)` and
`n_i(z)`**, both of which *are* observable. This is the concrete, defensible content of the
phrase "physics-constrained inversion":

```
−ε₀ ∇²Φ  =  e (n_i − n_e)            ← this equation is the constraint
```

We do not measure the potential. We measure the densities, and the sheath physics — Poisson
plus the ion momentum equation plus the Boltzmann electron relation — converts them into a
potential profile that is consistent with everything else observed. **The physics prior is
not decoration; it is load-bearing, and it is load-bearing for the one quantity nothing else
can reach.** This is the sentence to say out loud when asked what "physics-constrained"
actually means.

---

## 4. Candidate diagnostic survey

Every technique that could plausibly contribute to N1–N7 without a material probe, assessed
against the requirements of §2.4.

### 4.1 The candidate set

| # | Technique | Primary observable | Class |
|---|---|---|---|
| C1 | Optical emission spectroscopy (OES) + collisional-radiative model | line intensities → `T_e`, excited-state densities | passive |
| C2 | Laser-induced fluorescence (LIF) | Doppler-resolved IVDF of a metastable ion state | active |
| C3 | Incoherent Thomson scattering (ITS) | electron velocity distribution → `T_e`, `n_e` (local, absolute) | active |
| C4 | Laser interferometry (e.g. Mach–Zehnder, 633 nm / 10.6 µm) | line-integrated `n_e` | active |
| C5 | Microwave interferometry (~100 GHz) | line-integrated `n_e` | active |
| C6 | Stark spectroscopy / Stark polarimetry on Rydberg states | **electric field** `E(z)` directly | active |
| C7 | Electric-field-induced second-harmonic generation (E-FISH) | line-integrated `E` field | active |
| C8 | Two-photon absorption LIF (TALIF) | ground-state neutral density `n_g` | active |
| C9 | Gated ICCD emission imaging | sheath edge position `s(t)`, 2-D emission structure | passive |
| C10 | Stark broadening of hydrogenic lines | `n_e` | passive |
| C11 | Langmuir probe | `n_e`, `T_e`, `V_f`, EEDF | **intrusive** |
| C12 | Retarding-field energy analyser (RFEA) | IEDF at the surface — *the target quantity, directly* | **intrusive** |
| C13 | Optical emission tomography / Abel inversion | radially resolved emissivity | passive |
| C14 | Cavity ring-down spectroscopy | absolute metastable densities | active |

### 4.2 Screening against hard requirements

**Eliminated by R-NON-1 (non-perturbing):**

- **C11 Langmuir probe** — a material object inserted into the sheath. It perturbs the
  potential structure it is measuring, draws current, and modifies the local density. This is
  the exact failure the proposal exists to fix.
- **C12 RFEA** — measures the IEDF at the surface *directly*, which is tantalising, but it
  must be physically installed at the wall, replaces the surface it is measuring, and
  cannot survive high-flux environments.

> **However — both are retained as *simulated reference instruments*.** This is a deliberate
> and important design decision. Because the framework knows the ground truth, it can
> simulate what a Langmuir probe and an RFEA *would* have reported (including their own
> perturbation and their own systematic errors) and compare our optical reconstruction
> against the conventional intrusive baseline. This produces the single most persuasive
> figure the project can generate: **optical reconstruction vs. simulated conventional probe
> vs. hidden ground truth, on the same axes.** It answers "why is your approach better?"
> with a plot instead of an assertion. See doc 07 §6.

**Eliminated by the operating envelope:**

- **C10 Stark broadening** — requires `n_e ≳ 10²⁰ m⁻³` for measurable broadening of hydrogenic
  lines. At our envelope maximum of 10¹⁹ m⁻³ the broadening is below the instrument function
  of any realistic spectrometer. Rejected on physics, not on preference.
- **C5 Microwave interferometry** — a ~100 GHz beam has a wavelength of 3 mm and cannot be
  focused below roughly that scale. Against R-SPAT-1 (90 µm) it fails by a factor of ~33. It
  also suffers refraction and cutoff issues. Rejected: it can measure the bulk density but
  contributes nothing at the sheath scale where the answer lives.

**Deferred (technically valid, resource-prohibitive at Tier 0):**

- **C7 E-FISH** — genuinely measures the electric field non-intrusively, and would directly
  supply N4, the one quantity we otherwise infer. It requires a femtosecond laser system,
  it delivers a line-integrated field along the beam (requiring its own inversion), and its
  absolute calibration is non-trivial. **Promoted to the Tier-2 roadmap as the highest-value
  future channel**, and modelled in the framework as an optional plugin so its information
  contribution can be *quantified before anyone buys the laser*. This is itself a
  commercialisable output: the framework can answer "would E-FISH be worth it?" numerically.
- **C6 Stark polarimetry on Rydberg states** — same motivation, same conclusion, different
  hardware. Tier-2.
- **C14 Cavity ring-down** — would give absolute metastable densities and thereby remove the
  largest systematic in the LIF absolute calibration. Tier-2; its absence is instead handled
  as an explicit nuisance parameter in the inversion (doc 05 §4.3).

**Retained as supporting channels:**

- **C8 TALIF** — supplies N6. Modelled, low cost, resolves the collisionality question.
- **C9 Gated ICCD imaging** — supplies N5 and N7 at near-zero marginal cost, since the ICCD
  is already required as the OES and LIF detector.
- **C13 Tomography** — retained as a *method* applied to C1/C9 data rather than as a separate
  instrument.

### 4.3 The four primary channels, and why each is irreplaceable

The survivors are C1–C4. The justification is not that they are the standard four; it is
that each supplies information no other survivor can.

| Channel | Supplies | Why nothing else in the surviving set covers it |
|---|---|---|
| **LIF (C2)** | N3: the IVDF, spatially and phase resolved | The *only* survivor that sees ions at all. Every other channel is an electron or neutral diagnostic. Without LIF, `⟨E_i⟩` rests entirely on the model with no ion-side data to constrain it. **Non-negotiable.** |
| **Thomson (C3)** | N1 + N2: `T_e` and `n_e`, **locally and absolutely** | The only survivor giving a *local, absolutely calibrated* electron measurement with minimal model dependence. It is what makes R-ACC-1 achievable and what anchors the OES collisional-radiative model. |
| **OES (C1)** | N1 + N5 + N7: `T_e` and structure, at high repetition rate | The only *passive* channel. It costs no laser access, can run continuously and fast, and provides the time-resolved backbone that the slow, low-repetition-rate laser diagnostics are interpolated onto. Also the only channel that will survive into a deployable industrial product. |
| **Interferometry (C4)** | N2: line-integrated `n_e`, fast and robust | Redundant with Thomson *by design*. This redundancy is the point: it is a cheap, fast, robustly-calibrated density constraint that cross-checks Thomson's absolute calibration and continues to constrain the inversion when Thomson's low repetition rate leaves temporal gaps. **But see §5.4 — the sensitivity calculation shows this channel is marginal at RP-1 and blind below 3.3 × 10¹⁶ m⁻³, which materially weakens its case.** |

**On the interferometry channel and ADR-004.** Interferometry is the weakest of the four on
pure information content — it is line-integrated where we need local, and it duplicates a
quantity Thomson already measures better. The trade study nevertheless retains it, for three
reasons that must be stated rather than assumed:

1. **Temporal coverage.** Thomson scattering at these densities requires photon accumulation
   over many laser shots; at a realistic 10–100 Hz repetition rate it cannot follow the
   transients of R-TEMP-3. Interferometry runs at kHz–MHz and fills the gap.
2. **Calibration cross-check.** An absolute density measurement with no independent check is
   a single point of failure for R-ACC-1. Two independent absolute paths is basic
   metrological hygiene.
3. **Graceful degradation.** The robustness study (doc 07) asks what happens when each
   channel fails. A framework whose answer collapses when the hardest, most expensive
   diagnostic is unavailable is not deployable. Interferometry is the cheap channel that
   keeps the reconstruction alive.

**The framework will quantify whether this judgement is correct.** The information-content
analysis in doc 05 §6 computes the marginal Fisher information contributed by each channel.
If interferometry's marginal contribution is negligible across the envelope, the analysis
will say so, ADR-004 will close as "drop", and *that will be a legitimate published result* —
"we show the fourth channel is unnecessary" is a stronger finding than silently keeping it.

### 4.4 Rejected-with-reasons summary (the table to have ready)

| Rejected | Reason | Recoverable later? |
|---|---|---|
| Langmuir probe | Violates R-NON-1 — perturbs the sheath it measures | No; retained as simulated baseline for comparison |
| RFEA | Violates R-NON-1 — must replace the surface | No; retained as simulated baseline |
| Microwave interferometry | Fails R-SPAT-1 by ~33× (λ = 3 mm vs 90 µm needed) | No |
| Stark broadening | Requires `n_e ≳ 10²⁰ m⁻³`; envelope max is 10¹⁹ | No |
| E-FISH | Valid and high-value; fs-laser cost, line-integrated, hard absolute calibration | **Yes — Tier 2, plugin already specified** |
| Stark polarimetry | Same class as E-FISH | Yes — Tier 2 |
| Cavity ring-down | Would remove a LIF calibration systematic | Yes — Tier 2; handled as nuisance parameter meanwhile |

---

## 5. Instrument-level requirements derived from the channel selection

These flow into doc 02, where they become the actual optical and detector design.

### 5.1 LIF

| ID | Requirement | Value | Derived from |
|---|---|---|---|
| LIF-1 | Probe transition | Ar II 668.614 nm (3d ⁴F₇/₂ → 4p ⁴D₅/₂), fluorescence at 442.7 nm | standard metastable Ar⁺ scheme; strong, well-characterised |
| LIF-2 | Laser linewidth | ≤ 1 MHz | R-ACC-4: 300 m/s at 668.6 nm ⇒ Doppler shift 449 MHz; need ≥ 100 resolution elements across the profile |
| LIF-3 | Scan range | ± 20 GHz | covers ion energies to ~1 keV: v = 6.9×10⁴ m/s ⇒ 103 GHz… *see note* |
| LIF-4 | Beam waist | ≤ 80 µm in the sheath-normal direction | R-SPAT-1 |
| LIF-5 | Collection solid angle | ≥ 0.05 sr | photon budget, doc 02 §6 |
| LIF-6 | Detection gate | ≤ 5 ns, phase-lockable to the RF | R-TEMP-1 |
| LIF-7 | Wall standoff | beam centre reachable to 100 µm from surface | R-SPAT-2 — drives grazing-incidence geometry and stray-light control |

> **LIF-3 note — an honest constraint that must be surfaced, not buried.** A 250 eV Ar⁺ ion
> travels at 3.47 × 10⁴ m/s, giving a Doppler shift of 51.9 GHz at 668.6 nm — far beyond the
> mode-hop-free tuning range of any single-frequency diode laser (typically 20–40 GHz).
> Full-energy IVDF measurement inside a high-bias sheath by single-photon LIF is therefore
> **not achievable with a conventional scanning setup**, and any proposal claiming otherwise
> is wrong. The framework handles this correctly and explicitly:
> - LIF measures the IVDF at and just inside the **sheath edge**, where ion speeds are
>   ~`c_s`–3`c_s` (2.7–8 km/s ⇒ 4–12 GHz shift) and are comfortably within tuning range;
> - the IVDF *at the wall* is obtained by **propagating** the measured entry distribution
>   through the reconstructed sheath field using the physics model, which is precisely what
>   the physics-constrained inversion is for;
> - the associated model error is a **first-class term in the error budget** (doc 06 §4), not
>   an unstated approximation.
>
> This is exactly the kind of detail that separates a specification from a wish, and it is
> the kind of question a plasma physicist on a judging panel will ask within thirty seconds
> of hearing "LIF measures ion energy". Having the limitation quantified, mitigated and
> budgeted *before* being asked is worth more than any additional feature.

### 5.2 Thomson scattering

| ID | Requirement | Value | Derived from |
|---|---|---|---|
| TS-1 | Laser | Nd:YAG 532 nm, ≥ 0.5 J/pulse, ~8 ns | photon budget at `n_e` = 10¹⁷ m⁻³ (doc 02 §7) |
| TS-2 | Scattering parameter | α = 1/(k λ_D) ≪ 1 ⇒ incoherent regime | at 90° and λ_D = 40.7 µm, α ≈ 0.001 — firmly incoherent |
| TS-3 | Stray-light rejection | ≥ 10⁸ | the dominant engineering difficulty; drives baffles, viewing dumps, triple-grating spectrometer |
| TS-4 | Spatial resolution | ≤ 90 µm along the beam | R-SPAT-1 |
| TS-5 | Repetition rate | 10–100 Hz | laser thermal limit — **this is why interferometry is retained** |
| TS-6 | Accumulation | ~10³ shots per spatial point at RP-1 | photon statistics; drives the phase-locked accumulation strategy |

### 5.3 OES

| ID | Requirement | Value | Derived from |
|---|---|---|---|
| OES-1 | Spectral range | 300–900 nm | Ar I and Ar II line manifold |
| OES-2 | Spectral resolution | ≤ 0.05 nm | separate the line-ratio pairs used by the CR model |
| OES-3 | Absolute radiometric calibration | ≤ 10 % | required for excited-state densities, not just ratios |
| OES-4 | Temporal gating | ≤ 5 ns, phase-locked | R-TEMP-1 |
| OES-5 | Spatial imaging | ≤ 90 µm along sheath normal | R-SPAT-1; drives the imaging-spectrograph configuration |
| OES-6 | Line set | ≥ 6 Ar I + ≥ 4 Ar II lines | over-determination for CR-model inference |

### 5.4 Interferometry

The interferometric phase shift for a probe beam traversing a plasma is

```
Δφ  =  r_e · λ · ∫ n_e dl              r_e = 2.818 × 10⁻¹⁵ m (classical electron radius)
```

Evaluated over a 0.1 m chord across the envelope:

| `n_e` (m⁻³) | Δφ at 633 nm (HeNe) | Δφ at 10.6 µm (CO₂) |
|---|---|---|
| 10¹⁵ | 1.8 × 10⁻⁷ rad | 3.0 × 10⁻⁶ rad |
| **10¹⁷ (RP-1)** | **1.8 × 10⁻⁵ rad** | **3.0 × 10⁻⁴ rad** |
| 10¹⁹ | 1.8 × 10⁻³ rad | 3.0 × 10⁻² rad |

**This calculation overturns the obvious choice.** A HeNe interferometer — the default
instrument most people would specify — produces an 18 µrad phase shift at the reference
operating point. With a realistic heterodyne phase resolution of ~0.1 mrad, its minimum
detectable density over a 0.1 m chord is 5.6 × 10¹⁷ m⁻³: **it is blind at RP-1 by a factor
of about six.** A CO₂ laser at 10.6 µm gains the factor of 16.7 in wavelength and reaches a
floor of 3.3 × 10¹⁶ m⁻³, placing RP-1 roughly 3× above the noise — marginal but usable, and
with the additional advantage that a long wavelength is intrinsically far less sensitive to
mechanical path-length vibration.

| ID | Requirement | Value | Derived from |
|---|---|---|---|
| IF-1 | Configuration | Mach–Zehnder, **heterodyne** | phase-sign recovery and drift immunity; homodyne is inadequate at these phase levels |
| IF-2 | Wavelength | **10.6 µm (CO₂) — mandatory.** 633 nm HeNe **rejected** | the table above: HeNe is below its own detection floor at RP-1 |
| IF-3 | Phase resolution | ≤ 0.1 mrad | gives `n_e` floor of 3.3 × 10¹⁶ m⁻³ over a 0.1 m chord at 10.6 µm |
| IF-4 | Bandwidth | ≥ 1 MHz | R-TEMP-3 and RF-cycle-averaged tracking |
| IF-5 | Chord set | ≥ 8 parallel chords | Abel/tomographic inversion for radial profile |
| IF-6 | **Declared blind region** | `n_e` < 3.3 × 10¹⁶ m⁻³ | the channel contributes *no information* over the lower third of R-ENV-1 and must be modelled as absent there |

**IF-6 is a requirement, not a caveat.** The framework must treat the interferometry channel
as genuinely uninformative below its detection floor rather than as a weak measurement. An
inversion that quietly ingests noise as data will produce confident nonsense at low density.
The channel's likelihood contribution is therefore gated on its own detectability, and the
robustness study (doc 07) includes the low-density regime explicitly as a case where the
diagnostic set is *by design* reduced from four channels to three.

This finding also sharpens ADR-004. The question is no longer "is interferometry worth
keeping?" but "over which part of the operating envelope does it contribute at all?" — a
question the Fisher-information analysis of doc 05 §6 answers quantitatively.

### 5.5 Cross-instrument

| ID | Requirement | Value | Derived from |
|---|---|---|---|
| SYS-1 | Master timing jitter | ≤ 1 ns RMS across all channels | R-TEMP-2 |
| SYS-2 | Common spatial reference frame | ≤ 20 µm registration between channels | fusing channels that disagree about *where* they measured is worse than not fusing |
| SYS-3 | Common absolute calibration chain | traceable, with a single documented reference source | R-ACC-1 |
| SYS-4 | Data model must carry per-sample timestamp, phase bin, and uncertainty | — | the inversion is a joint fit across asynchronous channels; without per-sample time and error, it cannot be posed |

**SYS-1 and SYS-4 are the requirements most often omitted, and they are the ones that make
multi-diagnostic fusion actually work.** The four channels run at wildly different rates —
OES at kHz, interferometry at MHz, LIF at ~kHz with a slow frequency scan, Thomson at tens
of Hz. They are *never* simultaneous. The inversion must therefore be formulated over a
shared latent time base with each observation carrying its own acquisition window, and the
data model must support that from the first line of code. Retrofitting it is not possible.

---

## 6. Requirements traceability

Every requirement above carries an ID. The framework enforces traceability mechanically:

```
Requirement ID
      ↓
Design element (doc 02 §, doc 03 §, …)
      ↓
Implementation (module, class)
      ↓
Verification test (test ID)
      ↓
Validation benchmark (benchmark ID)
      ↓
Figure in the report
```

A requirement with no verification test is a project defect. The traceability matrix is
generated from source annotations rather than maintained by hand — see doc 08 §9.

---

## 7. What this document has established

1. The quantity of interest decomposes into a particle flux and a mean energy, constrained
   by *different* physics and therefore observable by *different* instruments.
2. Seven necessities (N1–N7) follow from that decomposition; quantitative requirements
   follow from a reference operating point using standard formulas, all checkable.
3. The ion transit time and the RF period are comparable at the reference point, which is
   what makes time resolution a derived requirement rather than a slogan.
4. The sheath potential is essentially unobservable, and Poisson's equation is what supplies
   it — this is the precise, defensible meaning of "physics-constrained".
5. Fourteen candidate techniques were screened; four survive as primary channels, two are
   retained as supporting channels, two are promoted to a Tier-2 roadmap, two are kept as
   *simulated intrusive baselines* for comparison, and four are rejected on stated physical
   grounds.
6. A hard limitation of LIF in high-bias sheaths was identified, quantified, and given an
   explicit mitigation with a budgeted error term — rather than being discovered by a
   reviewer.

**Only now is it legitimate to design the system.** That is doc 02.

---

## 8. Open items

| ID | Item | Disposition |
|---|---|---|
| Q-01 | Confirm Ar II 668.614 nm metastable population fraction at RP-1 from CR modelling; if too low, fall back to 611.5 nm scheme | doc 03, then update LIF-1 |
| Q-02 | Quantify interferometry's marginal Fisher information → closes ADR-004 | doc 05 §6 |
| Q-03 | Establish whether Thomson's 10³-shot accumulation is compatible with the transient scenarios, or whether transients are Thomson-blind by construction | doc 07 benchmark B-05 |
| Q-04 | Decide whether the second working gas is Xe (EP-relevant) or a Cl₂/Ar mix (etch-relevant) | doc 09; affects the atomic-data burden substantially |

---

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Requirements derived from first principles at reference point RP-1; fourteen-candidate trade study completed. |
