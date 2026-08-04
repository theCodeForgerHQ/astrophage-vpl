# 04 — Measurement Model Specification (`F₂`, `F₃`, `F₄`)

Version 1.0 · Status: **Baseline** · Owner: Nithisha V

> This document specifies how a plasma state becomes a number in a file. It is the part of
> the project that most student work skips and that most determines whether the accuracy
> claims mean anything. A reconstruction validated against synthetic data that was too clean
> is not validated at all.

---

## 1. The layering discipline

The forward chain is strictly layered. **No module may reach across a boundary.**

```
   x  (plasma state)
   │
   ├─► F₂  EMISSION / SCATTERING RESPONSE
   │      "how many photons, of what wavelength, from where, in what direction"
   │      output: spectral radiance  L(λ, r, Ω, t)   [W m⁻² sr⁻¹ nm⁻¹]
   │
   ├─► F₃  OPTICAL TRANSPORT
   │      "which of those photons reach the detector plane, and where do they land"
   │      output: irradiance at the focal plane  E(λ, x_det, t)   [W m⁻² nm⁻¹]
   │
   ├─► F₄  DETECTION AND DIGITISATION
   │      "what integer does the ADC report"
   │      output: raw frames / traces  [ADU]
   │
   y  (measurement)
```

**The rule that enforces realism: `F₄` never sees the plasma.** The detector model receives
photons and nothing else. It has no access to `n_e`, `T_e`, or any plasma variable. If a
detector module needs a plasma quantity, that is a design error, because a real detector
cannot know one. This single architectural constraint eliminates an entire class of
accidental cheating in which the "measurement" is contaminated by truth.

---

## 2. `F₂` — OES: collisional-radiative emission

### 2.1 Emissivity

The volumetric emission coefficient for a transition `u → l`:

```
ε_ul(r, t)  =  (1 / 4π) · n_u(r, t) · A_ul · h ν_ul            [W m⁻³ sr⁻¹]
```

`A_ul` from NIST ASD (doc 09). The physics problem is `n_u`, the upper-state population.

### 2.2 The collisional-radiative model

Excited-state populations are *not* in local thermodynamic equilibrium at these densities, so
a Boltzmann distribution over levels is wrong. The framework solves a CR model:

```
dn_u/dt  =  Σ_j n_e n_j K_ju   +  Σ_{k>u} n_k A_ku · Λ_ku
            −  n_u ( Σ_j n_e K_uj  +  Σ_{l<u} A_ul Λ_ul  +  ν_wall )   =  0    (quasi-static)
```

with:

| Term | Meaning | Source |
|---|---|---|
| `K_ju(T_e)` | electron-impact excitation/de-excitation rate coefficients | LXCat cross sections integrated over the **BOLSIG+ EEDF**, not a Maxwellian (doc 03 §3.2) |
| `A_ul` | Einstein coefficients | NIST ASD |
| `Λ_ul` | **radiation trapping escape factor** | Holstein–Biberman, computed for the chamber geometry |
| `ν_wall` | wall quenching of metastables | diffusion to wall + surface quenching probability |

### 2.3 Radiation trapping is not optional

The Ar I resonance lines (104.8, 106.7 nm) are strongly self-absorbed, which pumps the
metastable population far above its optically-thin value. Since the 811.53 nm and 763.51 nm
lines used in the `T_e` diagnostic are *metastable-coupled* (doc 02 §6.3), **an optically-thin
CR model would produce systematically wrong line ratios and therefore a systematically wrong
`T_e`** — which propagates straight into `Γ_i` and into the answer.

Escape factors are computed with the Holstein–Biberman formalism for the actual chamber
dimensions and are recomputed as the density profile changes. The uncertainty in the escape
factor is a budgeted error term (doc 06 §4).

This is a good example of the standard the project holds itself to: the easy version of this
model is wrong in a way that is invisible unless you know to look.

### 2.4 Spatial and spectral structure

- Emissivity is evaluated on the plasma grid and integrated along the line of sight in `F₃`.
- Each line carries a **lineshape**: natural (Lorentzian) ⊗ Doppler (Gaussian at `T_g` or
  `T_i`) ⊗ Stark (negligible here, doc 01 §4.2) ⊗ Zeeman (if `B ≠ 0`, §4.5) — combined as a
  Voigt profile.
- **Emission inside the sheath is not zero.** Fast secondary electrons accelerated across the
  sheath (doc 03 §3.3) excite neutrals, producing a characteristic emission structure whose
  spatial profile is itself informative about the field. This is modelled, and it is one of
  the more subtle information channels available.

---

## 3. `F₂` — LIF: rate-equation fluorescence

### 3.1 Model

A three-level rate-equation model for the pump/fluorescence scheme of doc 02 §5.3:

```
dn₂/dt  =  n₁ B₁₂ ρ(ν)  −  n₂ ( B₂₁ ρ(ν) + A₂₁ + A₂₃ + Q )
```

`ρ(ν)` is the spectral energy density of the laser at the ion's Doppler-shifted absorption
frequency, `Q` the collisional quenching rate. Fluorescence signal at 442.7 nm:

```
S(ν_L)  ∝  ∫ n₂(v; ν_L) A₂₃ hν₂₃ dv
```

### 3.2 The Doppler mapping

For an ion with velocity `v`, absorption occurs when

```
ν_L (1 − v·k̂ / c)  =  ν₀           ⇒        v_∥  =  c (1 − ν₀/ν_L)
```

so scanning `ν_L` maps out `f_i(v_∥)` where `v_∥` is along the laser propagation direction.
With the 15° grazing geometry of doc 02 §4.2,

```
v_∥  =  v_z sin(15°)  =  0.259 v_z
```

**and this projection factor must be inverted to recover `v_z`, which amplifies velocity
errors by 3.86×.** This is registered as a systematic and appears in the error budget.

### 3.3 Broadening budget at RP-1

| Mechanism | Width | Note |
|---|---|---|
| Thermal Doppler (`T_i` = 0.05 eV) | ~734 MHz FWHM | the signal we want |
| Natural linewidth | ~5 MHz | negligible |
| Laser linewidth | < 1 MHz | negligible ✔ (doc 01 LIF-2 justified) |
| Pressure broadening | < 10 MHz at 5 mTorr | negligible |
| **Zeeman splitting at 50 G** | **~70 MHz** | **10 % of the Doppler width — NOT negligible** |
| Power broadening | depends on intensity | see §3.4 |

**The Zeeman result matters.** With the optional Helmholtz field on (doc 02 §3.2), Zeeman
splitting is 10 % of the thermal Doppler width. Ignoring it would bias the inferred ion
temperature and distort the IVDF shape. The framework therefore models the full Zeeman
pattern (π and σ components with their polarisation dependence) whenever `B ≠ 0`, and the
LIF collection optics include a polariser so the components can be separated — turning a
nuisance into an additional observable.

### 3.4 Saturation

At high laser intensity the transition saturates, the measured lineshape broadens, and the
signal ceases to be proportional to `n₁`. The saturation parameter

```
S  =  I / I_sat  ,      I_sat  =  2π² h c A₂₁ / (3 λ³)
```

is computed and reported for every run. **The framework deliberately supports operating in
the saturated regime and models the resulting distortion**, because real LIF experiments
often do, and a framework that only handles the linear regime would fail on real data. Doc 07
robustness case F-13 sweeps `S` from 0.01 to 10.

---

## 4. `F₂` — Thomson scattering

### 4.1 Scattering regime

```
α  =  1 / (k λ_D)  ,     k = (4π/λ₀) sin(θ/2)
```

At RP-1, θ = 90°, λ₀ = 532 nm, λ_D = 40.7 µm: **α = 0.0015 ≪ 1** — firmly incoherent. The
scattered spectrum is therefore a direct map of the electron velocity distribution along `k`.

### 4.2 Spectrum

For a Maxwellian EEDF the non-relativistic scattered spectrum is Gaussian with 1/e half-width

```
Δλ_{1/e}  =  (2 λ₀ / c) · √(2 k T_e / m_e) · sin(θ/2)
```

At `T_e` = 3 eV: `Δλ = 2.58 nm`. The 520–545 nm coverage of doc 02 TS-S2 therefore
accommodates `T_e` well beyond 10 eV. ✔ Consistent.

**Relativistic corrections** (Selden's formula) are included: at `T_e` = 10 eV the thermal
velocity is 0.6 % of `c`, producing a small blue-shifted asymmetry. It is a ~1 % effect and
therefore comparable to the 4 % calibration uncertainty — included because it is cheap to
include and because omitting known physics that is larger than a budgeted uncertainty is not
defensible.

**Non-Maxwellian EEDF:** the framework does *not* assume a Maxwellian. It computes the
scattered spectrum from the actual EEDF produced by L2 or by the Boltzmann solver. This
matters: fitting a Maxwellian to a bi-Maxwellian plasma yields a `T_e` that corresponds to
neither population, and this bias is quantified in benchmark B-06.

### 4.3 Photon statistics

Per doc 02 §7.1, the expected signal at RP-1 is **0.008 photoelectrons per spectral channel
per shot**. The model is therefore explicitly Poissonian at the single-photoelectron level;
see §6.3.

---

## 5. `F₂` — Interferometry

### 5.1 Refractive index and phase

```
n_refr  =  √(1 − n_e/n_c)  ≈  1 − n_e/(2 n_c)  ,      n_c = ε₀ m_e ω² / e²
Δφ      =  r_e λ ∫ n_e dl                              r_e = 2.818 × 10⁻¹⁵ m
```

### 5.2 What is modelled beyond the ideal

| Effect | Model |
|---|---|
| Neutral-gas contribution | Ar has a non-zero polarisability; the neutral term partially cancels the electron term and is included |
| Beam refraction | Density gradients bend the beam; ray-traced rather than assumed straight |
| Vibration | Mechanical path-length noise as a 1/f + resonant-peak spectrum |
| Fringe jumps | Phase unwrapping failure under fast transients — a discrete failure mode, modelled |
| Detection floor | Below IF-6 the channel returns *no information*, not a noisy value (doc 01 §5.4) |

Fringe jumps deserve mention: they are a *discrete* failure that produces a plausible-looking
but completely wrong density. Modelling them means the robustness study can ask whether the
inversion detects and rejects such an event — which is exactly the kind of question a
deployable instrument must answer.

---

## 6. `F₃` — Optical transport

### 6.1 Approach

**Ray tracing**, not analytic solid-angle scaling. Every optical chain of doc 02 §5–8 is built
as a sequence of surfaces with real apertures, and rays are traced from the emitting volume
to the focal plane.

| Effect | Included |
|---|---|
| Finite collection solid angle and its spatial variation | ✔ |
| Vignetting at ports and baffles | ✔ |
| Geometric aberration (spherical, coma, astigmatism, field curvature) | ✔ |
| Chromatic aberration | ✔ |
| Surface transmission / reflection vs wavelength and angle | ✔ |
| Window transmission drift (sputter coating) | ✔ (doc 02 §11) |
| Point spread function → spatial resolution degradation | ✔ |
| Depth of field / measurement-volume definition | ✔ |
| Stray light and scattered laser light | ✔ (doc 02 §4.3) |
| Spectrograph instrument function | ✔ (measured-style Voigt, not a delta) |
| Diffraction | Only where it matters — LIF beam waist propagation; geometric elsewhere |

### 6.2 The measurement volume is an integral, not a point

This is the single most commonly mishandled aspect of optical diagnostics of sheaths. The
"measurement at `z = 200 µm`" is really

```
S  =  ∫∫∫ W(r) · ε(r) d³r
```

where `W` is the instrument weighting function set by the beam profile, the collection
aperture and the PSF. In a sheath where `n_e` varies by an order of magnitude over 890 µm, a
75 µm measurement volume averages over a genuinely varying quantity, and reporting the result
as a point value at the volume centre is a **systematic bias, not a resolution limit**.

The framework computes the weighting function explicitly and — critically — **applies the same
weighting inside the inverse model**. This is the correct treatment: rather than deconvolving,
the forward operator used by the inversion includes the instrument's own spatial averaging, so
the comparison is like-for-like. Getting this wrong produces reconstructions that appear
biased near the wall for purely instrumental reasons.

### 6.3 Build vs buy

| Component | Decision |
|---|---|
| Ray tracer | **Buy** — Raysect (designed for plasma diagnostics; has spectral emission primitives) |
| Beam propagation (Gaussian optics) | Buy — standard ABCD / `poppy` where diffraction matters |
| Spectrograph model | Build — thin wrapper computing dispersion, instrument function, imaging mapping |

---

## 7. `F₄` — Detection and digitisation

### 7.1 The chain

```
photons at focal plane
   → photocathode:      Poisson(QE(λ) · N_ph)                     → photoelectrons
   → MCP:               each pe → Pólya-distributed gain          → electron cloud
   → phosphor:          conversion + PSF spread + decay           → photons
   → CCD:               Poisson + CTE + blooming + dark + read    → electrons
   → ADC:               gain, offset, INL, quantisation, clipping → ADU
```

### 7.2 Noise taxonomy

Every source is separately switchable so its contribution can be isolated in the error budget.

| # | Source | Statistics |
|---|---|---|
| N1 | Photon shot noise | Poisson |
| N2 | Photocathode QE fluctuation | Binomial thinning |
| N3 | MCP gain variance | Pólya / negative binomial |
| N4 | MCP ion feedback | Rare, large-amplitude events |
| N5 | Phosphor conversion noise | Poisson + spatial spread |
| N6 | Dark current | Poisson, temperature-dependent |
| N7 | Read noise | Gaussian |
| N8 | Fixed-pattern non-uniformity | Multiplicative map, fixed per device |
| N9 | Dead / hot pixels | Fixed map, clustered |
| N10 | Blooming | Deterministic, threshold-triggered |
| N11 | Cosmic rays | Poisson in time, localised deposit |
| N12 | ADC quantisation | Uniform |
| N13 | ADC non-linearity | Deterministic INL curve |
| N14 | Gate jitter | Gaussian in time → phase smearing |
| N15 | Laser energy jitter (shot-to-shot) | Gaussian, multiplicative |
| N16 | Laser pointing jitter | Gaussian in position → volume overlap variation |
| N17 | Baseline / background drift | Slow random walk |
| N18 | Electromagnetic pickup from RF | Coherent at 13.56 MHz and harmonics |

**N18 is included because it is real and because it is coherent.** RF pickup is not white
noise; it is synchronous with the phase-locked acquisition, which means it does *not* average
away with accumulation. A framework whose noise model is all-Gaussian-all-white will
systematically overestimate the benefit of long accumulations — precisely the error that makes
synthetic validations optimistic.

### 7.3 Calibration is applied, not assumed

The framework simulates the *measured* calibration, not the true one:

```
true instrument response  →  calibration measurement (with its own noise)
                          →  estimated response (biased, uncertain)
                          →  applied by the analysis pipeline
```

The inversion therefore works with data corrected by an **imperfect** calibration, exactly as
in a real experiment. Applying the true calibration would be a form of inverse crime and
would understate the error.

---

## 8. Verification of the measurement models

| ID | Test | Pass criterion |
|---|---|---|
| V-20 | Thomson spectrum vs analytic Gaussian, Maxwellian input | Width matches Selden formula within 0.5 % |
| V-21 | Thomson photon count vs closed-form radiometry | Within 2 % |
| V-22 | LIF lineshape vs analytic Voigt, low saturation | Width within 1 % |
| V-23 | LIF saturation curve vs analytic two-level | Within 2 % |
| V-24 | CR model vs published Ar line ratios | Within published scatter |
| V-25 | CR model reduces to corona at low `n_e`, to LTE at high `n_e` | Both limits recovered |
| V-26 | Escape factor → 1 as `n_g` → 0 | Analytic limit |
| V-27 | Ray trace throughput vs analytic solid angle for an ideal thin lens | Within 1 % |
| V-28 | Detector: mean and variance of the full chain vs analytic photon-transfer curve | Gain and read noise recovered from simulated PTC within 3 % |
| V-29 | Interferometer phase vs analytic `r_e λ ∫n dl` for uniform plasma | Within 0.1 % |
| V-30 | Noise sources switch off cleanly (each in isolation reproduces the noiseless limit) | Exact |

**V-28 is the strongest test of the detector chain**: simulate a photon-transfer curve exactly
as one would measure it on a real camera, fit it, and check that the fitted gain and read
noise match the configured values. If they do not, the detector model is internally
inconsistent.

---

## 9. Interfaces

Every instrument implements the same contract (doc 08 §4):

```python
class Instrument(Protocol):
    def configure(self, cfg: InstrumentConfig) -> None: ...
    def calibrate(self, refs: CalibrationSet) -> Calibration: ...
    def observe(self, state: PlasmaState, t: TimeWindow) -> Measurement: ...
    def forward(self, state: PlasmaState, t: TimeWindow) -> Observable: ...   # noiseless
    def metadata(self) -> InstrumentMetadata: ...
```

`observe` returns raw data with noise and imperfect calibration; `forward` returns the
noiseless expectation used by the likelihood. **Both come from the same code path**, with
noise and calibration error as switchable stages, which guarantees they cannot drift apart —
a class of bug that would silently invalidate every result.

A real instrument implements the same protocol, with `observe` reading hardware and `forward`
unimplemented (raising explicitly). This is how doc 00 E2 is satisfied.

---

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Three-layer forward chain specified with 18 enumerated noise sources and 11 verification tests. |
