# 02 — System Design Specification: The Virtual Laboratory

Version 1.0 · Status: **Baseline** · Owner: Team Astrophage

> **The framing that makes this document worth writing.**
> We have no laboratory and will not build one. Nevertheless this document specifies a
> complete, buildable plasma diagnostic facility down to component part numbers, port
> angles, working distances and trigger cables — as though a funding body had said "build
> it". The reason is not aspiration. It is that **every synthetic measurement the framework
> produces is derived from this design**. A simulated spectrum is only as meaningful as the
> spectrometer it claims to come from. If the optical layout is vague, the noise model is
> arbitrary; if the noise model is arbitrary, the reconstruction accuracy is a fiction; and
> if the accuracy is a fiction, the entire project is a rendering exercise.
>
> Specifying a real facility is what converts "we simulated some noise" into "we simulated
> an Andor iStar 340T at −20 °C with a 2 ns gate behind an f/4 imaging spectrograph".

---

## 1. Design philosophy

### 1.1 Rules the design obeys

| Rule | Consequence |
|---|---|
| **Every component is a real, purchasable part or an explicitly-flagged design placeholder** | The parameter registry (§12) records manufacturer and part number, or marks the entry `DESIGN` |
| **Every geometric quantity is a number, not an adjective** | "the camera views the sheath" is banned; "the collection axis lies at 90° to the laser axis, 340 mm from the measurement volume, f/4" is required |
| **The detector never receives plasma — it receives photons** | The forward chain is strictly layered (doc 04); no module may reach across a layer boundary |
| **Every imperfection is a parameter with a default and a range** | Not "add noise", but a fully enumerated noise model per §9 |
| **If it cannot close geometrically, it is not a design** | Solid angles, working distances and port clearances must be mutually consistent; §5 checks them |

### 1.2 What is deliberately *not* specified

Mechanical engineering (flange bolt patterns, weldments, support structures), electrical
safety, vacuum pump-down sequencing, and interlocks. These matter for construction and are
irrelevant to the forward model. The boundary is: **anything that changes a photon or a
number is specified; anything that only holds the apparatus up is not.**

---

## 2. Coordinate system and reference frame

A single global frame is defined once and used by every module. Registration errors between
channels are among the most damaging and least modelled systematics in multi-diagnostic
fusion (requirement SYS-2), so the frame is normative.

```
        y (vertical)
        │
        │        ┌──────────────────────────────┐
        │        │                              │
        │        │        BULK PLASMA           │
        │        │                              │
        │        │                              │
        │        │  ← presheath →│ sheath │     │
        └────────┼───────────────┼────────┼─────┼──── x
       origin    │               │        │     │
                 └───────────────┴────────┴─────┘
                                 z_s      z_w = 0
                          ↑ z (sheath normal, into plasma)
```

- **Origin** at the geometric centre of the biased electrode surface.
- **z** — normal to the electrode surface, positive *into* the plasma. The wall is at `z = 0`;
  the sheath edge is at `z = z_s > 0`.
- **x, y** — in the electrode plane; **x** is the axis of the Thomson/LIF laser beams.
- All instruments report positions in this frame. Each instrument carries a 6-DOF
  registration transform with its own uncertainty (§10.4).

**Sign convention.** Ion energy flux `Γ_E` is positive when energy flows *toward* the wall,
i.e. in the `−z` direction. Stated explicitly because sign errors in flux quantities are
common, silent and catastrophic.

---

## 3. Vacuum chamber and plasma source

### 3.1 Chamber

| Parameter | Value | Class |
|---|---|---|
| Geometry | Cylindrical, stainless steel 304L | DESIGN |
| Internal diameter | 400 mm | DESIGN |
| Internal height | 300 mm | DESIGN |
| Base pressure | ≤ 1 × 10⁻⁷ Torr | DESIGN |
| Pumping | Turbomolecular 700 l/s + scroll backing | DESIGN |
| Working-gas feed | Mass-flow controller, 0–100 sccm, ±1 % FS | SPEC-class |
| Pressure measurement | Capacitance manometer (10 Torr FS) + ion gauge | SPEC-class |
| Internal surface | Electropolished; wall recombination coefficient a modelled parameter | DESIGN |

**Why the diameter matters to the physics.** 400 mm sets the chord length for
interferometry (§7.4) and therefore the achievable phase shift, and it sets the ratio of
wall area to plasma volume that governs the global particle balance in the fluid model
(doc 03). It is not a cosmetic number.

### 3.2 Plasma source

| Parameter | Value | Class |
|---|---|---|
| Type | Inductively coupled (planar spiral antenna) with independently biased substrate electrode | DESIGN |
| ICP source power | 0–1000 W at 13.56 MHz | DESIGN |
| Bias supply | 0 to −1000 V DC, **or** 13.56 MHz RF with matching network | DESIGN |
| Electrode diameter | 150 mm | DESIGN |
| Electrode material | Tungsten (replaceable: W, Mo, C, Al₂O₃) | DESIGN |
| Magnetic field | Optional Helmholtz pair, 0–100 G axial | DESIGN |

**Why ICP with a separately biased electrode.** This is the decisive architectural choice of
the source. It **decouples plasma production from sheath acceleration**: the ICP sets `n_e`
and `T_e`, the bias independently sets the sheath potential drop. In a capacitively coupled
source the two are locked together, so a bias sweep also changes the density and no clean
sensitivity study is possible. The ICP geometry makes the operating envelope of doc 01 §2.4
a genuine two-dimensional grid rather than a one-dimensional curve through it — which is
exactly what the identifiability mapping (doc 05 §6) needs.

### 3.3 The regime map this source spans

| Regime | `p` (mTorr) | Bias | `s/λ_CX` | Physics character |
|---|---|---|---|---|
| A — collisionless DC | 1 | −250 V DC | 0.014 | textbook Child–Langmuir; the verification anchor |
| B — reference | 5 | −250 V DC | 0.072 | RP-1; weakly collisional |
| C — collisional | 50 | −250 V DC | 0.72 | structured, non-Maxwellian IEDF |
| D — RF, transit-resonant | 5 | 13.56 MHz | 0.072 | bimodal IEDF; `τ_tr ≈ 0.35 T_RF` |
| E — RF, high-frequency limit | 5 | 60 MHz | 0.072 | ions see time-averaged field; single peak |
| F — low density | 5 | −250 V DC, low ICP | 0.072 | **interferometry blind** (doc 01 IF-6); 3-channel operation |
| G — transient | 5 | pulsed bias | varies | **Thomson blind** (§8.3); ignition/decay |

Regimes F and G are included specifically because they are the ones where the diagnostic set
degrades. A benchmark suite that only visits regimes where everything works proves nothing.

---

## 4. Port map and optical access

Optical access is where diagnostic designs actually fail, and it is almost always waved
through in student work. The port map is therefore fully enumerated with angles and clear
apertures.

### 4.1 Port table

| Port | Azimuth | Elevation | Clear aperture | Window | Function |
|---|---|---|---|---|---|
| P1 | 0° | 0° | DN63 (63 mm) | ZnSe, AR @ 10.6 µm, wedged 0.5° | CO₂ interferometer input |
| P2 | 180° | 0° | DN63 | ZnSe, AR @ 10.6 µm, wedged | CO₂ interferometer output |
| P3 | 90° | 0° | DN40 | UV-fused silica, AR 400–700 nm, Brewster | LIF laser input |
| P4 | 270° | 0° | DN40 | UV-fused silica + **beam dump** | LIF laser exit / dump |
| P5 | 45° | 0° | DN100 | UV-fused silica, AR 300–900 nm | LIF & OES collection (imaging) |
| P6 | 135° | 0° | DN63 | UV-fused silica, AR @ 532 nm, Brewster | Thomson laser input |
| P7 | 315° | 0° | DN63 | UV-fused silica + **viewing dump** (Rayleigh horn) | Thomson beam dump |
| P8 | 225° | 0° | DN100 | UV-fused silica, AR 500–560 nm | Thomson collection (90° scattering) |
| P9 | 0° | +60° | DN63 | UV-fused silica | Gated ICCD imaging (sheath edge) |
| P10 | — | top | DN160 | — | ICP antenna / dielectric window |

### 4.2 Geometric consistency check

The design must actually close. Three checks are performed and must pass:

1. **Thomson scattering angle.** Laser in at P6 (135°), collection at P8 (225°) ⇒ scattering
   angle 90°. ✔ Consistent with doc 01 TS-2 (α ≈ 0.0015, incoherent regime).
2. **LIF orthogonality.** Laser along P3→P4 (the x-axis); collection at P5 (45°). The
   component of the ion drift velocity (along `−z`) along the laser propagation axis is
   **zero by construction** — the beam is parallel to the electrode surface. This is
   deliberate and it is a subtlety worth stating: *a beam propagating along x measures the
   x-component of velocity, not the z-component we need.*

   **Resolution:** the LIF beam is introduced at a **grazing angle θ_L = 15° to the
   electrode plane**, so that `v_z` projects onto the laser axis with `sin 15° = 0.259`.
   The measured Doppler shift is then `Δν = v_z sin θ_L / λ`, and the velocity resolution
   requirement of doc 01 R-ACC-4 tightens by a factor of 1/0.259 = 3.86. Re-deriving:
   300 m/s × 0.259 = 77.7 m/s of *projected* velocity ⇒ 116 MHz Doppler shift, still well
   within a 1 MHz-linewidth laser's capability. ✔ Requirement LIF-2 survives with margin.

   The alternative — a beam normal to the surface — is geometrically impossible: it would
   have to pass through the electrode.
3. **Collection cone clearance.** P5 at f/4 over a 340 mm working distance requires an 85 mm
   clear aperture; DN100 provides 100 mm. ✔ Passes with 15 mm margin.

**Check 2 is the kind of thing that is invisible until someone draws the picture.** A
diagnostic geometry in which the instrument is blind to the velocity component of interest
is a complete design failure, and it is a very easy one to specify by accident.

### 4.3 Stray light architecture (Thomson)

Doc 01 TS-3 demands 10⁸ stray-light rejection. This is the hardest engineering requirement
in the entire facility and it is met by a stack of measures, each modelled:

| Measure | Rejection factor | Modelled as |
|---|---|---|
| Brewster-angle entrance window | ~10 | reflection coefficient vs polarisation |
| Baffled entrance snout (3 knife-edge baffles) | ~10² | geometric aperture chain |
| Rayleigh-horn viewing dump | ~10² | residual reflectance parameter |
| Triple-grating spectrometer notch (532 nm) | ~10⁴ | wavelength-dependent transmission |
| Polarisation discrimination | ~5 | Thomson signal is polarised; stray light is less so |
| **Combined** | **~10⁹** | with 10× margin against the 10⁸ requirement |

The residual stray light is *not* set to zero in the model. It is a parameter with a
nominal value and an uncertainty, and it appears in the error budget (doc 06) as a
background-subtraction systematic — because in a real Thomson system it is the dominant one.

---

## 5. Instrument specification: LIF

### 5.1 Optical chain

```
Tunable diode laser  →  optical isolator  →  wavemeter pickoff (λ monitor)
   →  AOM (chopping / timing)  →  beam expander  →  focusing lens (f = 500 mm)
   →  P3 window  →  measurement volume  →  P4 beam dump

Collection: measurement volume →  P5 window  →  f/4 achromat (f = 200 mm)
   →  narrowband filter (442.7 nm, 3 nm FWHM)  →  ICCD or PMT
```

### 5.2 Specification

| ID | Parameter | Value | Class |
|---|---|---|---|
| LIF-L1 | Laser | Tunable ECDL, 668.6 nm, ≥ 20 mW | SPEC (Toptica DL pro class) |
| LIF-L2 | Linewidth | < 1 MHz | SPEC |
| LIF-L3 | Mode-hop-free tuning range | 20 GHz | SPEC |
| LIF-L4 | Wavelength stability | < 2 MHz/min, wavemeter-locked | SPEC |
| LIF-O1 | Beam waist at measurement volume | 75 µm (1/e²) | derived from R-SPAT-1 |
| LIF-O2 | Rayleigh range | 26 mm | consistency: waist is maintained across the sheath |
| LIF-O3 | Grazing angle to electrode | 15.0° | §4.2 check 2 |
| LIF-O4 | Closest approach to wall | 100 µm | R-SPAT-2 |
| LIF-C1 | Collection f-number | f/4 | photon budget |
| LIF-C2 | Collection solid angle | 0.049 sr | = π/(4·f#²) |
| LIF-C3 | Magnification | 1.0 | maps 75 µm volume onto detector pixels |
| LIF-D1 | Detector | Gated ICCD, gate ≥ 2 ns | SPEC (Andor iStar class) |
| LIF-D2 | Filter | 442.7 nm, 3 nm FWHM, OD6 blocking at 668.6 nm | SPEC |

### 5.3 The probe scheme and its assumptions

The Ar II scheme (`3d ⁴F₇/₂ → 4p ⁴D₅/₂` pumped at 668.614 nm, fluorescence observed on
`4p ⁴D₅/₂ → 4s ⁴P₃/₂` at 442.72 nm) is standard, well characterised, and — critically —
**measures only the population in the `3d ⁴F₇/₂ metastable state**, not the full ion
population.

This is the single largest systematic in LIF and it is stated here rather than buried:

```
f_measured(v)  =  [ n_metastable / n_i ] · f_i(v)   ×  (assumption: the metastable
                                                        fraction is velocity-independent)
```

The assumption is *approximately* true and *not exactly* true, because metastable production
and quenching rates depend on the local electron population and on ion transit history
through the sheath. Consequences, all carried explicitly:

- The metastable fraction is a **nuisance parameter** in the inversion, not a constant
  (doc 05 §4.3);
- Its velocity dependence is bounded by a CR-model calculation and enters the error budget
  as a shape systematic (doc 06 §4);
- Cavity ring-down spectroscopy (doc 01 C14) would measure it directly and is the highest-value
  Tier-2 addition for this reason.

**LIF gives a beautifully resolved measurement of a distribution that is not quite the one we
want.** Saying that first is worth more than any additional capability.

---

## 6. Instrument specification: OES

### 6.1 Optical chain

```
Plasma emission  →  P5 window  →  f/4 imaging relay  →  entrance slit (imaging spectrograph)
   →  Czerny–Turner, 750 mm focal length, 1800 gr/mm  →  gated ICCD
```

The spectrograph is used in **imaging mode**: the slit is oriented along the sheath normal
(`z`), so one detector axis is wavelength and the other is position. A single exposure
therefore yields a spatially resolved spectrum through the entire sheath — which is what
makes OES the fast backbone channel.

### 6.2 Specification

| ID | Parameter | Value | Class |
|---|---|---|---|
| OES-S1 | Spectrograph | Czerny–Turner, f = 750 mm, f/9.7 | SPEC (Princeton SP-2750 class) |
| OES-S2 | Grating | 1800 gr/mm, blaze 500 nm | SPEC |
| OES-S3 | Reciprocal linear dispersion | 0.62 nm/mm | derived |
| OES-S4 | Slit width | 20 µm | ⇒ instrument function 0.026 nm FWHM ✔ meets OES-2 (≤ 0.05 nm) |
| OES-S5 | Spatial magnification | 0.5× | 90 µm object → 45 µm at slit → ~3 pixels ✔ meets R-SPAT-1 |
| OES-D1 | Detector | Gated ICCD, 1024 × 1024, 13 µm pixels | SPEC |
| OES-D2 | Gate width | 2 ns minimum | ✔ meets R-TEMP-1 |
| OES-D3 | Photocathode | Gen III, QE ≈ 0.20 at 500 nm | SPEC |
| OES-C1 | Absolute calibration source | NIST-traceable tungsten-halogen + deuterium lamp | SPEC |

### 6.3 Line set

The collisional-radiative inference requires an over-determined line set (doc 01 OES-6).
Selected for: strong emission, minimal blending at 0.026 nm resolution, and differing
excitation-threshold sensitivity so that the ratios actually constrain `T_e`.

| Species | λ (nm) | Upper level | Role |
|---|---|---|---|
| Ar I | 750.39 | 2p₁ | direct excitation dominated — `T_e` sensitive |
| Ar I | 751.47 | 2p₅ | direct excitation dominated |
| Ar I | 811.53 | 2p₉ | metastable-coupled — `n_metastable` sensitive |
| Ar I | 763.51 | 2p₆ | metastable-coupled |
| Ar I | 696.54 | 2p₂ | intermediate |
| Ar I | 706.72 | 2p₃ | intermediate |
| Ar II | 480.60 | — | ion emission — `n_i`, high threshold |
| Ar II | 488.00 | — | ion emission |
| Ar II | 434.81 | — | ion emission |
| Ar II | 476.49 | — | ion emission |

The 750.39/811.53 ratio is the classical `T_e` discriminator precisely because the first is
insensitive and the second is highly sensitive to the metastable population; using both
together separates `T_e` from `n_metastable`, which a single ratio cannot do. Atomic data
provenance for every line is in doc 09.

---

## 7. Instrument specification: Thomson scattering

### 7.1 Photon budget — the calculation that reshapes the architecture

The number of detected Thomson photoelectrons per laser shot from one spatial element:

```
N_pe  =  (E_L / hν) · n_e · L · σ_T · (ΔΩ / 4π) · η
```

with `E_L` = 0.5 J at 532 nm, `L` = 90 µm (the spatial resolution element), `σ_T` =
6.652 × 10⁻²⁹ m², `ΔΩ` = 0.05 sr, and `η` = 0.05 (combined transmission × QE, appropriate
for a triple-grating spectrometer with a Gen III ICCD).

| `n_e` (m⁻³) | `N_pe` per shot (total) | per spectral channel (20 ch) | Shots for 3 % | Shots for 1 % | Wall-clock at 10 Hz, 1 % |
|---|---|---|---|---|---|
| 10¹⁶ | 0.016 | 0.0008 | 69 700 | 627 000 | **17.4 hours** |
| **10¹⁷ (RP-1)** | **0.159** | **0.008** | **6 970** | **62 700** | **1.7 hours** |
| 10¹⁸ | 1.60 | 0.080 | 697 | 6 270 | 10.4 min |
| 10¹⁹ | 16.0 | 0.80 | 70 | 627 | 1.0 min |

**This table has three architectural consequences, and they are not optional.**

1. **Thomson scattering at the reference operating point yields 0.16 photoelectrons per
   shot.** It is a photon-counting experiment, not a signal-averaging one. The detector model
   must therefore be Poissonian at the single-photoelectron level, with photocathode
   statistics and ICCD gain distribution explicitly modelled — a Gaussian read-noise model
   would be qualitatively wrong (doc 04 §6).
2. **Thomson cannot follow transients.** At RP-1 a 3 % measurement takes ~700 s of
   accumulation. Any single-shot or few-shot event is invisible. Phase-resolved operation
   multiplies this by the number of phase bins: 16 bins at 3 % requires ~111 000 shots,
   ≈ 3.1 hours. **Thomson is a steady-state, phase-locked, repetitive-waveform diagnostic
   only.** Regime G in §3.3 is Thomson-blind by construction.
3. **The fast channels are therefore not redundant — they are load-bearing.** This is the
   quantitative justification for retaining OES and interferometry that doc 01 §4.3 argued
   qualitatively. Thomson anchors the absolute calibration during steady operation; OES and
   interferometry carry the time resolution. Neither could do the other's job.

> A project that specified Thomson scattering without computing this number would have built
> a framework whose validation scenarios silently assume an impossible measurement. The
> calculation costs ten minutes and changes the design.

### 7.2 Specification

| ID | Parameter | Value | Class |
|---|---|---|---|
| TS-L1 | Laser | Nd:YAG, 532 nm, 0.5 J, 8 ns, 10 Hz | SPEC (Continuum Powerlite class) |
| TS-L2 | Beam waist at volume | 200 µm | photon budget vs damage threshold |
| TS-L3 | Polarisation | Linear, perpendicular to scattering plane | maximises Thomson cross section |
| TS-O1 | Scattering angle | 90.0° | §4.2 check 1 |
| TS-O2 | Collection f-number | f/4 (ΔΩ = 0.049 sr) | photon budget |
| TS-O3 | Spatial resolution along beam | 90 µm | R-SPAT-1 |
| TS-S1 | Spectrometer | Triple-grating, 532 nm notch, ≥ 10⁴ rejection | SPEC |
| TS-S2 | Spectral coverage | 520–545 nm | covers `T_e` up to 10 eV |
| TS-S3 | Spectral channels | 20 | balances resolution against per-channel counts |
| TS-D1 | Detector | Gated ICCD, photon-counting mode, 5 ns gate | SPEC |
| TS-D2 | Gate timing | Locked to laser Q-switch, ≤ 1 ns jitter | SYS-1 |

### 7.3 Rayleigh calibration

Absolute density calibration is performed by Rayleigh scattering from a known pressure of a
calibration gas, in the same optical configuration. This is modelled as a first-class
procedure, not assumed: the calibration itself has uncertainty (gas purity, pressure gauge
accuracy, Rayleigh cross-section literature value) which propagates into `n_e` and thence
directly into `Γ_E`. See doc 06 §5.

---

## 8. Instrument specification: interferometry

### 8.1 Configuration

Heterodyne Mach–Zehnder at 10.6 µm (doc 01 IF-2, which rejected 633 nm on sensitivity
grounds).

```
CO₂ laser (10.6 µm)  →  beam splitter  ─────────────► reference arm ──┐
                              │                                        ├─► detector (HgCdTe)
                              └─► AOM (40 MHz shift) → P1 → plasma → P2 ┘
```

The 40 MHz acousto-optic frequency shift makes the detection heterodyne, which recovers the
sign of the phase and rejects low-frequency amplitude drift — both essential at the sub-mrad
phase levels this channel operates at.

### 8.2 Specification

| ID | Parameter | Value | Class |
|---|---|---|---|
| IF-L1 | Laser | CO₂, 10.6 µm, 5 W, single-line | SPEC |
| IF-L2 | Heterodyne offset | 40 MHz (AOM) | SPEC |
| IF-D1 | Detector | HgCdTe photovoltaic, ≥ 100 MHz bandwidth | SPEC |
| IF-P1 | Phase resolution | 0.1 mrad | ⇒ `n_e` floor 3.3 × 10¹⁶ m⁻³ over 0.4 m chord: **8.4 × 10¹⁵ m⁻³** |
| IF-P2 | Chord length | 400 mm (full chamber diameter) | §3.1 — the wide chamber buys sensitivity |
| IF-G1 | Chord count | 8, spaced 5 mm in `z` | IF-5 |
| IF-G2 | Vibration isolation | Optical table, floated; common-path reference | mechanical drift is the dominant systematic |

**Note the interaction between §3.1 and IF-P1.** The 400 mm chamber diameter improves the
interferometry detection floor by 4× relative to the 100 mm chord assumed in doc 01 §5.4,
moving it from 3.3 × 10¹⁶ to 8.4 × 10¹⁵ m⁻³. This brings almost the whole of the operating
envelope R-ENV-1 into range and materially strengthens the case for retaining the channel.
**This is a design decision made *because* of a requirements calculation, which is what the
document order is for.**

### 8.3 What interferometry cannot do

Line-integrated along `x`, it has **no intrinsic `z` resolution** — it cannot see the sheath
structure at all. The 8 chords at different heights give a coarse `z` profile of the *bulk*
density, and Abel inversion gives radial structure, but the sheath itself (0.89 mm) is far
below this channel's resolution. Interferometry constrains the *boundary condition* of the
sheath problem, not the sheath. This is the correct and limited role it plays, and the
inversion must not be allowed to over-weight it.

---

## 9. Detector and electronics chain

Doc 01 requires that "no assumption be hidden". The detector is where hidden assumptions
usually live, so the model is fully enumerated. Every parameter below is configurable, has a
default from a real datasheet, and has a range for the sensitivity study.

### 9.1 ICCD model parameters

| Group | Parameters |
|---|---|
| Photocathode | quantum efficiency `QE(λ)`, spatial non-uniformity map, dark emission rate, effective work function |
| MCP | gain (voltage-dependent), gain variance (Furman/pulse-height distribution), ion feedback rate, saturation onset, spatial gain non-uniformity |
| Phosphor | conversion efficiency, decay time constant(s), lateral spread (point-spread function) |
| Gating | gate width, gate rise/fall time, gate-to-trigger delay, gate jitter, irising (spatial gate non-uniformity) |
| CCD | full-well capacity, read noise (e⁻ RMS), dark current vs temperature, charge-transfer efficiency, blooming threshold, dead/hot pixel map |
| ADC | bit depth, gain (e⁻/ADU), offset (bias level), integral non-linearity, quantisation |
| Thermal | sensor temperature, temperature drift, dark-current doubling temperature |

### 9.2 Defaults (Andor iStar 340T class)

| Parameter | Default | Range for sensitivity study |
|---|---|---|
| QE at 500 nm | 0.20 | 0.10 – 0.35 |
| MCP gain | 10³ | 10² – 10⁴ |
| Read noise | 3.5 e⁻ RMS | 2 – 10 |
| Dark current at −20 °C | 0.01 e⁻/pix/s | 0.001 – 0.1 |
| Full well | 200 000 e⁻ | — |
| ADC | 16 bit | 12 – 18 |
| Gate width (min) | 2.0 ns | 1.5 – 100 |
| Gate jitter | 25 ps RMS | 10 – 500 |
| Dead pixel fraction | 10⁻⁴ | 0 – 10⁻² |

### 9.3 Photon-counting regime

Because Thomson operates at ~0.008 photoelectrons per channel per shot (§7.1), the detector
model must be correct in the single-photon limit. Specifically:

- Photoelectron generation is **Poisson**, not Gaussian;
- The MCP gain applied to a *single* photoelectron has a broad, skewed pulse-height
  distribution (modelled as a Pólya/negative-binomial distribution), so the recorded signal
  from one photoelectron is itself a random variable with large relative variance;
- Thresholding to count events introduces detection efficiency < 1 and a false-count rate,
  both of which are parameters.

Getting this wrong is the classic way synthetic Thomson data becomes unrealistically clean.

---

## 10. Timing, synchronisation and acquisition

### 10.1 The core problem

The four channels never measure simultaneously:

| Channel | Native rate | Duty cycle | Latency to result |
|---|---|---|---|
| OES | 1 kHz (gated, phase-locked) | 2 ns gate in 73.7 ns period ⇒ 2.7 % | immediate |
| Interferometry | 1 MHz continuous | 100 % | immediate |
| LIF | 1 kHz, but a full IVDF requires a ~200-point frequency scan | 2 ns gate; ~200 s per full IVDF | ~200 s |
| Thomson | 10 Hz, ~7 000 shots per point | 5 ns gate; ~700 s per point | ~700 s |

**A "measurement" is therefore not a snapshot. It is a set of observations distributed over
minutes, each tagged with an acquisition window and an RF phase bin.** This is the reality
that requirement SYS-4 anticipated and that the data model must express from the outset.

### 10.2 Timing architecture

```
        13.56 MHz RF generator
                 │
                 ▼
        ┌─────────────────┐
        │  Phase-locked   │   master clock, 10 MHz OCXO reference
        │  delay generator│   (Stanford DG645 class, ≤ 25 ps jitter)
        └────────┬────────┘
                 ├──────► OES ICCD gate       (phase bin φ, delay τ_OES)
                 ├──────► LIF ICCD gate       (phase bin φ, delay τ_LIF)
                 ├──────► LIF AOM             (chopper)
                 ├──────► Nd:YAG flashlamp    (10 Hz, free-running)
                 ├──────► Nd:YAG Q-switch     (phase-locked to nearest RF cycle)
                 ├──────► Thomson ICCD gate   (Q-switch + optical delay)
                 └──────► Interferometer DAQ  (continuous, timestamped)
```

### 10.3 Phase-locked accumulation

Steady-state RF operation is treated as **cyclo-stationary**: the plasma state is periodic in
RF phase, so observations at the same phase across different cycles may be accumulated. The
RF period is divided into `N_φ = 16` phase bins of 4.6 ns each (compatible with the 2 ns
minimum gate).

This is what makes Thomson and LIF possible at all in the RF regime, and it carries an
assumption that must be tested, not assumed: **that the discharge is genuinely periodic and
stable over the accumulation time**. Slow drift during a 3-hour Thomson accumulation smears
the phase-resolved result. The framework models this as a slow drift process in the plasma
parameters during accumulation, and benchmark B-09 (doc 07) quantifies the resulting bias.

### 10.4 Registration and its uncertainty

Each instrument has a 6-DOF transform into the global frame with an associated uncertainty:

| Instrument | Position uncertainty (µm) | Angular uncertainty (mrad) |
|---|---|---|
| LIF measurement volume | 20 | 1.0 |
| Thomson measurement volume | 30 | 1.0 |
| OES imaging axis | 25 | 1.5 |
| Interferometer chords | 100 | 0.5 |

Against a sheath thickness of 890 µm, a 30 µm registration error is a 3.4 % position error —
which, in a region where density varies by an order of magnitude over the sheath, is a
significant systematic. Registration uncertainty is therefore a **first-class term in the
error budget** (doc 06 §4), not an afterthought.

---

## 11. Calibration architecture

Every instrument requires calibration, every calibration has uncertainty, and calibration
error is one of the largest contributors to the final flux uncertainty. The calibration chain
is modelled end-to-end.

| Instrument | Calibration | Reference | Dominant uncertainty |
|---|---|---|---|
| OES — wavelength | Hg/Ar pencil lamp | NIST line list | negligible (< 0.005 nm) |
| OES — instrument function | same, fitted Voigt | — | 5 % on width |
| OES — absolute radiometric | NIST-traceable tungsten-halogen lamp | NIST FEL scale | **6 %** (lamp + transfer + window ageing) |
| LIF — frequency axis | wavemeter + Fabry–Pérot etalon | — | 2 MHz ⇒ 1.3 m/s ✔ negligible |
| LIF — absolute density | *not absolutely calibrated* | — | **metastable fraction is a nuisance parameter** |
| Thomson — spectral | Raman scattering in N₂ | Raman cross sections | 4 % |
| Thomson — absolute density | Rayleigh scattering in Ar | Rayleigh cross section | **7 %** |
| Interferometry — phase | AOM offset, known path | — | 3 % (vibration-limited) |
| Geometric registration | machined fiducial target imaged by all channels | CMM measurement | 20–100 µm (§10.4) |

**Window transmission drift** is modelled separately and applies to all optical channels: in
a real plasma facility, sputtered material coats the windows, reducing transmission over
hours to days. Modelled as an exponentially decaying transmission with a configurable time
constant. It affects absolute calibrations but not ratios — which is one more reason the OES
line-ratio approach is retained alongside absolute measurements.

---

## 12. Parameter registry

Every quantity in this document is registered with the following schema, enforced in code
(doc 08 §5):

```yaml
- id: TS-L1.energy
  description: Nd:YAG pulse energy at the measurement volume
  value: 0.5
  units: J
  class: SPEC                    # MEASURED | PUBLISHED | SPEC | DESIGN | ASSUMED
  source: "Continuum Powerlite DLS 8010 datasheet, rev 2019"
  uncertainty: {type: relative, value: 0.05}
  sweep_range: [0.1, 2.0]
  affects: [thomson.photon_budget, thomson.snr]
```

The registry is the single source of truth. **No number appears in code as a literal.** The
count of `ASSUMED`-class entries is a tracked project metric (doc 00 C1) and is reported in
CI.

---

## 13. Degradation and failure modes

Doc 07's robustness matrix draws its scenarios from here. Each is a modelled, switchable
condition — not a hypothetical.

| ID | Failure | Model |
|---|---|---|
| F-01 | Thomson unavailable | channel removed from likelihood |
| F-02 | LIF unavailable | channel removed — the severe case, since it is the only ion diagnostic |
| F-03 | Interferometry below detection floor | channel gated off by IF-6 |
| F-04 | OES absolute calibration lost (ratios only) | radiometric scale becomes a free parameter |
| F-05 | Window transmission degraded 50 % | transmission multiplier on affected channels |
| F-06 | Laser energy drift −20 % over the run | time-varying `E_L` |
| F-07 | LIF wavelength lock lost | frequency-axis offset + drift |
| F-08 | Timing jitter degraded to 5 ns | phase-bin smearing |
| F-09 | Registration error 200 µm | transform perturbation |
| F-10 | 1 % dead pixels in a burst | clustered dead-pixel map |
| F-11 | Stray light 10× above spec | Thomson background pedestal |
| F-12 | Discharge drift during accumulation | slow plasma-parameter walk |

---

## 14. What this document delivers to the rest of the project

- **Doc 03** inherits the plasma source geometry, the regime map (§3.3) and the boundary conditions.
- **Doc 04** inherits the optical chains (§5–8), the detector model (§9) and the calibration chain (§11) as the specification of the forward operators `F₃` and `F₄`.
- **Doc 05** inherits the acquisition reality of §10 — asynchronous, phase-binned, per-sample-uncertainty observations — as the structure of the likelihood.
- **Doc 06** inherits the calibration uncertainties (§11) and registration uncertainties (§10.4) as error-budget terms.
- **Doc 07** inherits §13 as its robustness matrix and §3.3 as its scenario list.
- **Doc 09** inherits §12 as the provenance schema.

---

## 15. Findings that changed the design

Recorded explicitly, because they are the evidence that the specification process did work
rather than merely being performed:

1. **A LIF beam parallel to the electrode is blind to the velocity component of interest** —
   forced the 15° grazing-incidence geometry (§4.2).
2. **HeNe interferometry is below its own detection floor at the reference point** — forced
   10.6 µm CO₂ (doc 01 §5.4).
3. **The 400 mm chamber diameter improves the interferometry floor by 4×** — a chamber
   dimension chosen for a diagnostic reason (§8.2).
4. **Thomson yields 0.16 photoelectrons per shot at RP-1** — forced photon-counting detector
   statistics, and established that Thomson is structurally incapable of following
   transients (§7.1).
5. **LIF measures a metastable subpopulation, not the ion population** — forced a nuisance
   parameter into the inversion rather than an unstated approximation (§5.3).

---

## 16. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Complete facility specification derived from doc 01 requirements. Five design-changing findings recorded in §15. |
