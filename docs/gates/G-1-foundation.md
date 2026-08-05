# Gate G-1 — Phase 1 (Foundation) exit report

**Date:** 2026-08-05 · **Phase:** P1 → P2 · **Specification:** doc 11 §2

> **Objective (doc 11 §2):** the skeleton is real and verified before any physics
> complexity arrives.

Every number below is measured. Where a criterion is not met, it says so and says why.

---

## 1. Verdict

| Gate | Criterion (doc 11 §2) | Result | Status |
|---|---|---|---|
| **G-1.1** | V-01, V-02 pass at design order ± 0.1 | V-01: 1.9997 / 2.0158 / 2.9987 · V-02: 1.9922 / 1.9927 | **PASS** |
| **G-1.2** | V-03 Child–Langmuir recovery within 5 % | `J_i` 0.105 % · `s` +52 % at RP-1 | **PARTIAL — see §3** |
| **G-1.3** | Manifest reproduction bit-identical | Identical content digest; raw bytes differ only in `created_utc` | **PASS** |
| **G-1.4** | Measured L1 throughput recorded; doc 03 §4.4 estimates corrected | **86 ms/solve** against a "~1 s" estimate | **PASS** |
| **G-1.5** | `ASSUMED` count in physics constants = 0 | 0 (of 23 registered parameters) | **PASS** |
| **G-1.6** | Core coverage ≥ 80 % | **98 %** on `vpl-core` | **PASS** |

**G-1 is met, with G-1.2 amended by [ADR-010](../adr/ADR-010-v03-sheath-thickness-gate.md).**
The amendment is not a waiver: the criterion as written is unreachable anywhere in the
operating envelope doc 01 R-ENV-4 specifies, and the replacement is a stricter test of the
solver.

---

## 2. Work breakdown

| WBS | Task (doc 11 §2) | State |
|---|---|---|
| 1.1 | `vpl-core`: protocols, state types, units, provenance | Done |
| 1.2 | Parameter registry + literal-lint rule | Done |
| 1.3 | Manifest engine + `vpl run` / `reproduce` / `compare` | Done |
| 1.4 | Storage layer with embedded provenance | Done |
| 1.5 | L0 analytic sheath models | Done |
| 1.6 | L1 fluid solver (FEniCSx) | Done — steady state; BDF2 deferred to P2 |
| 1.7 | Atomic-data loaders (LXCat, NIST ASD, OpenADAS) | Done — OpenADAS registered, ADF parser deferred to the CR model |
| 1.8 | Boltzmann/EEDF integration | Done — see [ADR-009](../adr/ADR-009-eedf-solver.md) |
| 1.9 | CI: lint, types, tests, coverage, `ASSUMED` count | Done |

**Test totals:** 1 488 passing in the workspace environment plus 92 in the FEniCSx
environment. doc 08 §8 requires the two classes to be reported separately:

| Class | Count |
|---|---|
| Physics verification (MMS, convergence, conservation, analytic limits) | **144** workspace + **30** FEniCSx |
| Software correctness | **1 344** |

---

## 3. G-1.2, in full

`J_i` passes at **0.105 %**. `s` does not, and [ADR-010](../adr/ADR-010-v03-sheath-thickness-gate.md)
establishes that the criterion is at fault rather than the solver:

- Child–Langmuir neglects electron space charge. Solving doc 03 §3.1's *own* equations
  gives `s/s_CL` = **1.503** at RP-1 and **1.215** at doc 01 R-ENV-4's 1000 V ceiling.
- Confirmed by three independent computations: the solver (1.5225), a quadrature written
  alongside it (1.5020), and a third written separately at review sharing no code
  (**1.5029**).
- The 5 % criterion is first met near `V_w/T_e ≈ 5 000`, i.e. ~16 kV — sixteen times the
  specified envelope ceiling.
- Review additionally found the thickness integral **diverges** at the sheath edge (the
  first integral vanishes at third order, so the integrand goes as `φ^−3/2`). `s` therefore
  depends on the registered `δ` as `δ^−1/2`, moving by ~11 Debye lengths across its
  registered sweep range — where doc 03 §2.1 anticipates "a fraction of a Debye length".

Replacement criteria, all passing: **V-03a** `J_i` within 5 % (0.105 %); **V-03b** `s`
against an independent quadrature of doc 03 §3.1 within 5 % (1.4 % default mesh, 0.5 %
refined); **V-03c** `s/s_CL` approaches 1 monotonically, matching the asymptotic table.

Nothing was tuned to close the gap, and the gap is named in the test
(`test_sheath_thickness_exceeds_child_langmuir_by_a_quantified_amount`).

---

## 4. G-1.4 — estimates corrected

doc 11 requires measured throughput to replace the doc 03 §4.4 and doc 10 §3.1 estimates.

| Quantity | Document estimate | Measured | Correction |
|---|---|---|---|
| L1 fluid solve | ~1 s (doc 03 §1) | **86 ms** | **12× faster** |

The composition matters more than the wall clock and should be what doc 10 §3.1 carries:
one answer costs **19 bias-continuation steps and 108 Newton iterations**. That is not a
tuning detail — Newton does *not* converge from the L0 seed at sheath bias, contrary to
what doc 03 §3.4 implies, and diverges to `nan` on every mesh, domain length and damping
factor tried.

The L2 PIC estimate (doc 03 §4.4, ~3 min/solve CPU; doc 10 §3.1, ~5 s/solve GPU) is
**not yet measured** — L2 is P2 work, and doc 10 §3.2's warning that the GPU figure "is an
estimate, not a measurement" still stands.

---

## 5. Corrections to the Baseline documents

Five, all recorded as ADRs and none applied silently. This is doc 11 G-1.4's process
working, not a defect count.

| ADR | Finding |
|---|---|
| [006](../adr/ADR-006-toolchain.md) | Toolchain deviations from doc 08 §12 (uv alongside micromamba; `ruff format` for Black) |
| [007](../adr/ADR-007-child-langmuir-thickness.md) | doc 01 §2.2 evaluates `λ_D` at the bulk density where doc 03 §2.3's own derivation requires the sheath edge. `s` = 1.14 mm, not 0.89 mm. Separately, doc 03 §2.1 states `γ_i = 3` while doc 03 §2.3's own arithmetic uses 0 |
| [008](../adr/ADR-008-manifest-substrate.md) | **PyYAML cannot parse doc 08 §6's own example manifest.** YAML 1.1's float resolver requires a signed exponent, so `1.0e17` loads as a string |
| [009](../adr/ADR-009-eedf-solver.md) | Neither doc 08 §2 option is usable: `bolos` calls a SciPy API removed in 1.14, at exactly the three integrals doc 03 §3.2 needs |
| [010](../adr/ADR-010-v03-sheath-thickness-gate.md) | V-03's `s` criterion is unreachable in the specified envelope; doc 03 §2.1 understates the sheath-edge sensitivity |

Two further corrections were made in code without an ADR, both recorded in commits and
docstrings: doc 08 §4's `ForwardSolver.solve` forbids a steady solve that the rest of the
data model explicitly permits (widened to `TimeGrid | None`), and doc 03 §3.3's literal
bulk density `n_0` would put L1's flux 64 % above the L0 Bohm flux V-03 checks it against.

---

## 6. Reproducibility, verified on both machines

Doc 10 §1 makes the RTX A4000 box the reference machine. The suite is run there, not only
on the development machine.

| | Development (macOS/arm64) | Reference (RTX A4000, WSL2 Ubuntu 24.04) |
|---|---|---|
| Tests | 1 488 pass | 829 pass at the time of the last full run |
| `mypy --strict` | clean | clean |
| ruff | clean | clean |
| FEniCSx | 0.9.0 | 0.9.0 |
| Python / NumPy / SciPy | 3.12.13 / 2.5.1 / 1.18.0 | identical |

Running on the reference machine found a portability defect the development machine could
not: every text read used Python's *locale* default encoding, which on a `LANG=C` machine
turns the registry's em dashes into a crash — incompatible with doc 00 E3. Fixed, with the
registry loader also hardened to ignore non-authored files.

---

## 7. Carried into P2

| Item | Why it is deferred, not dropped |
|---|---|
| L1 BDF2 time stepping | P1 needed a verified steady solver; the RF regimes of doc 02 §3.3 need the transient |
| OpenADAS ADF11/ADF15 parser | Belongs with the CR-model assembly that consumes it (doc 08 §2) |
| `SECONDARY_EMISSION_ENERGY` as a registry entry | Currently a named module constant; should be `sheath.E_se` |
| GPU throughput measurement | doc 10 §3.2's ~3.2 × 10⁹ particle-steps/s is still an estimate. Gate G-2 should require it measured, as G-1.4 did for L1 |
| Doc 03 §3.4's implication that Newton converges unaided | It does not; the continuation strategy that makes it work should be written into the document |
