# ADR-010 — Verification gate V-03 is unreachable as written for the sheath thickness

**Status:** Accepted · **Date:** 2026-08-05 · **Supersedes nothing · Amends:** doc 03 §7 (V-03),
doc 11 gate G-1.2 · **Related:** [ADR-007](ADR-007-child-langmuir-thickness.md)

## Context

Doc 03 §7 states V-03 as:

> | V-03 | Child–Langmuir limit (L1, L2 → L0) | `s` and `J_i` within 5 % in the
> collisionless high-bias limit |

and doc 11 makes it gate G-1.2. Implementing L1 (WBS 1.6) produced:

| Quantity | L1 | L0 (Child–Langmuir) | Agreement |
|---|---|---|---|
| `J_i` at 1000 V | 26.335 A·m⁻² | 26.308 A·m⁻² | **0.105 %** ✔ |
| `s` at 250 V (RP-1) | — | — | **+52 %** ✘ |
| `s` at 1000 V (R-ENV-4 ceiling) | — | — | **+22 %** ✘ |

The question is whether L1 is wrong or the gate is.

## Evidence

**The disagreement is physics, not a defect.** Child–Langmuir neglects electron space charge
inside the sheath. Solving doc 03 §3.1's *own* equations — Boltzmann electrons, cold ions
entering at the Bohm speed, Poisson — gives a first integral

```
½ (dφ/dξ)²  =  √(1 + 2φ) + e^(−φ) − 2 ,        φ = −eΦ/T_e ,  ξ = z/λ_D
```

whose quadrature is the exact collisionless thickness. Three independent computations agree:

| Source | `s / s_CL` at RP-1 |
|---|---|
| The L1 solver | 1.5225 |
| Quadrature written alongside the solver | 1.5020 |
| **Quadrature written separately during review, sharing no code** | **1.5029** |

The asymptotic table (review quadrature, δ = 0.01):

| `V_w/T_e` | `V_w` at `T_e` = 3 eV | `s/s_CL` |
|---|---|---|
| 33 | 100 V | 1.892 |
| 83 | **250 V (RP-1)** | **1.503** |
| 333 | **999 V (R-ENV-4 ceiling)** | **1.215** |
| 1 333 | 4 kV | 1.093 |
| 5 333 | 16 kV | 1.042 |

`s → s_CL` only as `V_w/T_e → ∞`, and the local exponent of `s(V_w)` runs 0.499 → 0.747
toward Child–Langmuir's 3/4. **The 5 % criterion is first met around 16 kV — sixteen times
doc 01 R-ENV-4's 1000 V ceiling.** It is unreachable anywhere in the specified envelope.

## A second problem, found during review

`s` is not merely different from `s_CL` — it is **strongly dependent on the sheath-edge
definition**, far more so than doc 03 §2.1 anticipates.

Expanding the first integral about the sheath edge:

```
√(1 + 2φ) = 1 + φ − φ²/2 + φ³/2 …          e^(−φ) = 1 − φ + φ²/2 − φ³/6 …
√(1 + 2φ) + e^(−φ) − 2  =  φ³/3 + O(φ⁴)
```

The bracket vanishes at **third** order, so the integrand behaves as `φ^(−3/2)` and the
integral **diverges** at φ → 0. There is no sheath edge in this model; there is only a
chosen cutoff, which is precisely why doc 03 §2.1 registers `δ` as a swept parameter.

But doc 03 §2.1 characterises the consequence as:

> Different definitions shift `z_s` by a fraction of a Debye length and therefore shift
> the flux.

Measured across the registered sweep range of `sheath.edge_tolerance` (δ ∈ [0.001, 0.1]):

| δ | edge at `φ` | `s/s_CL` | `s/λ_D` |
|---|---|---|---|
| 0.001 | 0.0323 | 1.798 | 39.3 |
| **0.01 (nominal)** | 0.1071 | **1.503** | **32.9** |
| 0.05 | 0.2619 | 1.362 | 29.8 |
| 0.1 | 0.3983 | 1.309 | 28.6 |

**The swept range moves `s` by ~11 Debye lengths, not a fraction of one.** Since
`s ~ δ^(−1/2)`, tightening δ by a decade moves `s` by ~30 %. Doc 03 §2.1's instinct — that
this must be registered and swept — is right and, if anything, understated.

## Decision

**V-03 is split. The `J_i` half stands; the `s` half is replaced.**

| | Criterion | Status |
|---|---|---|
| **V-03a** | `J_i` from L1 matches the L0 Bohm-matched value within 5 % in the collisionless high-bias limit | **Passes** at 0.105 % |
| **V-03b** | `s` from L1 matches an *independent quadrature of doc 03 §3.1's own equations* within 5 %, at the registered `δ` | **Passes** at 1.4 % on the default mesh, 0.5 % refined |
| **V-03c** | `s/s_CL` decreases monotonically toward 1 as `V_w/T_e` rises, matching the quadrature's asymptotic table | **Passes** |

V-03b is the criterion V-03 was reaching for. Comparing a fluid solver against a *closed-form
approximation to a different equation set* tests the approximation, not the solver. Comparing
it against an exact solution of the equations it claims to solve tests the solver — which is
what verification means (doc 07 §1: "are we solving the equations right?").

**Nothing was tuned.** The solver was not adjusted to close the gap, and the gap is reported
in the test names (`test_sheath_thickness_exceeds_child_langmuir_by_a_quantified_amount`).

## Consequences

- **Gate G-1.2 as worded cannot be met.** It should read "V-03a–c pass" rather than "V-03
  Child–Langmuir recovery within 5 %".
- **`Γ_E` is unaffected**, as in ADR-007: the flux contains no `s`. L1 gives 6531 W·m⁻²
  against L0's 6577, converging upward onto the +0.75 % the ion entry energy predicts.
- **ADR-007 remains load-bearing.** The bulk-vs-sheath-edge `λ_D` choice is worth 28 % and is
  asserted by a test; this ADR sits on top of it, not instead of it.
- **`δ` must appear in the sensitivity study with a wide range**, per the table above. The
  registered sweep range [0.001, 0.1] is appropriate and its consequence is now quantified.
- **doc 03 §2.1's "a fraction of a Debye length" should be corrected** when next revised.
  The correct statement is that `s` scales as `δ^(−1/2)` and the registered sweep moves it by
  roughly a third.
- L2 (PIC) will face the same comparison. It resolves the presheath kinetically and has no
  Boltzmann-electron assumption, so its `s` should sit closer to the quadrature than to
  Child–Langmuir — and V-03b is the criterion that will say so.
