# 09 — Data Sources, Assets and Provenance

Version 1.0 · Status: **Baseline** · Owner: Denistan B

> Because there is no experimental data, **every number in this project must be traceable to
> a citable source**. A synthetic dataset is only as credible as its inputs. This document is
> the register of those inputs, and it is the direct answer to "where did your data come
> from?"

---

## 1. Provenance classes

Every registered parameter (doc 02 §12) carries one of:

| Class | Meaning | Acceptable? |
|---|---|---|
| `MEASURED` | From a standard evaluated database, with version and access date | ✔ |
| `PUBLISHED` | From a specific paper, cited | ✔ |
| `SPEC` | From a real commercial component datasheet, with part number | ✔ |
| `DESIGN` | A free design choice we made — therefore a variable in the sensitivity study | ✔ |
| `ASSUMED` | Could not be sourced | **✗ Defect.** Tracked as a CI metric; must trend to zero |

**Target: zero `ASSUMED` entries in the physics-constants and atomic-data categories before
Phase 2 exit (doc 11).** `DESIGN` entries are unlimited but must all appear in the sensitivity
study — an unswept design choice is a hidden assumption wearing a different label.

---

## 2. Atomic and molecular data

### 2.1 Electron-impact cross sections

| Dataset | Content | Source | Licence / terms |
|---|---|---|---|
| **LXCat — Phelps database** | e + Ar elastic, excitation, ionisation | `lxcat.net` | Free; **citation of the specific database required** |
| **LXCat — Biagi (Magboltz v8.9)** | e + Ar full set, high accuracy | `lxcat.net` | Free; citation required |
| **LXCat — IST-Lisbon** | e + Ar, independently evaluated | `lxcat.net` | Free; citation required |
| **LXCat — Phelps ion database** | Ar⁺ + Ar elastic and **charge exchange** | `lxcat.net` | Free; citation required |

**Three independent electron sets are retained deliberately.** Cross-section databases
disagree, sometimes by tens of percent in the excitation channels that drive the OES
inference. Running the inference under all three and reporting the spread is an honest
measure of atomic-data uncertainty — and it is a term in the error budget (doc 06 §4, term 2)
rather than an unstated risk. Most work picks one set and never mentions the choice.

### 2.2 Spectroscopic data

| Dataset | Content | Source | Licence |
|---|---|---|---|
| **NIST Atomic Spectra Database (ASD) v5.x** | Ar I / Ar II wavelengths, energy levels, `A_ul`, statistical weights | `physics.nist.gov/asd` | US Government — public domain; citation requested |
| **OpenADAS** | ADF11 (ionisation/recombination), ADF15 (photon emissivity coefficients) for Ar | `open.adas.ac.uk` | Free registration; academic use |
| **CHIANTI / ChiantiPy** | Cross-check on level data | `chiantidatabase.org` | Open, citation required |

**Uncertainty on `A_ul`:** NIST assigns accuracy grades (AAA ≤ 0.3 % … E > 50 %). The
framework **ingests the grade and propagates it**. Ar I 811.53 nm carries a much better grade
than several Ar II lines, and weighting lines by their data quality is straightforward,
correct, and almost never done.

### 2.3 Surface data

| Dataset | Content | Source | Class |
|---|---|---|---|
| Ion-induced secondary electron emission | `γ_se(E)` for Ar⁺ on W, Mo, C, Al₂O₃ | Phelps & Petrović (1999) compilation; Raizer | `PUBLISHED` |
| Sputter yields | `Y(E, θ)` for Ar⁺ on W | Yamamura & Tawara (1996) tables; Eckstein | `PUBLISHED` |
| Ion reflection coefficients | `R_N`, `R_E` | Eckstein, TRIM/SDTrimSP tabulations | `PUBLISHED` |
| Metastable wall quenching probability | — | literature range 0.1–1.0 | `DESIGN` (swept) |

Sputter yields are not needed for `Γ_E` itself but are needed to convert flux into the erosion
prediction that makes the result commercially meaningful (doc 12).

### 2.4 Optical and material data

| Dataset | Content | Source |
|---|---|---|
| Refractive indices, dispersion | fused silica, ZnSe, BK7, coatings | `refractiveindex.info` (open) |
| Rayleigh cross sections | Ar, N₂ — for Thomson calibration | Sneep & Ubachs (2005) |
| Raman cross sections | N₂ — for Thomson spectral calibration | Penney et al. |
| Gas polarisability | Ar — interferometry neutral term | NIST Chemistry WebBook |

### 2.5 Fundamental constants

**CODATA 2022**, via `scipy.constants`. Never hand-typed. A mistyped electron mass is a
plausible, undetectable and catastrophic error.

---

## 3. Component specifications

All `SPEC`-class, with manufacturer and part number recorded. Datasheets are archived in
`refs/datasheets/` with retrieval date, because manufacturers revise and withdraw them.

| Component | Representative part | Parameters taken |
|---|---|---|
| Gated ICCD | Andor iStar 340T | QE(λ), gain, read noise, dark current, gate width, jitter |
| Imaging spectrograph | Princeton Instruments SP-2750 | focal length, dispersion, f-number, throughput |
| Nd:YAG laser | Continuum Powerlite DLS 8010 | pulse energy, duration, rep rate, linewidth, jitter |
| Tunable diode laser | Toptica DL pro 670 | power, linewidth, MHF tuning range, drift |
| CO₂ laser | Access Laser L4 | power, stability, linewidth |
| HgCdTe detector | Vigo PVI-4TE | bandwidth, NEP, responsivity |
| Delay generator | Stanford Research DG645 | jitter, resolution, channel count |
| Triple spectrometer | — | rejection, dispersion, throughput |

> **These are *representative* parts chosen to make the model realistic, not a procurement
> list.** Substituting a different vendor changes numbers in the registry and nothing else —
> which is the point of putting them in a registry.

---

## 4. Synthetic dataset generation

### 4.1 What "synthetic data" means here

Not random numbers with noise. The generation chain is:

```
sampled parameters θ*  (Latin hypercube / Sobol over the envelope)
      ↓
L2 kinetic solve                    → ground-truth state x*
      ↓
F₂ · F₃ · F₄  with full noise       → measurements y
      ↓
sealed truth store + measurement store
```

Every dataset is therefore reproducible from `(manifest, commit, seed)` and needs no
distribution — which is fortunate, since the full sweep runs to terabytes.

### 4.2 Planned datasets

| ID | Purpose | Design | Size (est.) |
|---|---|---|---|
| **DS-TRAIN** | L3 surrogate training | 5 000 Sobol points over the envelope, L2 | ~400 GB raw, ~8 GB reduced |
| **DS-TEST** | Surrogate held-out validation | 500 independent points, different seed stream | ~40 GB |
| **DS-BENCH** | Scenario benchmarks B-01…B-13 | 13 scenarios × 200 noise realisations | ~120 GB |
| **DS-COVER** | Coverage / SBC validation | 1 000 prior draws | ~80 GB |
| **DS-ABLATE** | Ablation matrix | 19 ablations × 100 realisations | ~60 GB |
| **DS-ENVELOPE** | Identifiability map (B-14) | 2 000 LHS points | ~160 GB |

**Only reduced artifacts are retained; raw fields are regenerated on demand.** Retention
policy in doc 13 §5.

### 4.3 Public benchmark release

A small curated subset — a few hundred cases with sealed truth published alongside — is
prepared for release as a **public benchmark for plasma-sheath inverse methods**. There is no
standard benchmark for this problem class, and creating one is a low-cost, high-visibility
contribution that costs a few gigabytes and outlives the project.

---

## 5. Licence compatibility

| Source | Terms | Redistribution of derived data? |
|---|---|---|
| LXCat | Free, citation required | Derived rate coefficients: yes with attribution. **Do not redistribute raw tables** |
| NIST ASD | Public domain | Yes |
| OpenADAS | Registration, academic use | **Do not redistribute raw ADF files**; derived quantities acceptable |
| CHIANTI | Open, citation | Yes |
| refractiveindex.info | Public domain / CC0 | Yes |
| Manufacturer datasheets | Copyrighted | **Do not redistribute.** Cite part number and archive locally only |

**This has a concrete architectural consequence.** The repository stores *references and
loaders*, not bulk third-party data. A setup script fetches from the primary source at install
time, records the version hash, and fails loudly if the upstream data has changed — which
preserves reproducibility without redistributing anything we may not.

The OpenADAS and datasheet restrictions matter for the commercialisation posture (doc 12): a
product cannot ship these files, so the data-access layer is designed to be swappable from the
outset.

---

## 6. Citation ledger

`refs/CITATIONS.bib` accumulates every source, and the report generator (doc 13) emits the
bibliography **for the specific sources a given run actually touched** — not a static list.
A run that used only the Phelps set does not cite Biagi.

---

## 7. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Provenance classes defined; atomic, surface, optical and component sources registered with licence terms; six planned datasets specified. |
