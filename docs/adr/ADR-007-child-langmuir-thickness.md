# ADR-007 — Which density sets `λ_D` in the Child–Langmuir sheath thickness

**Status:** Accepted · **Date:** 2026-08-05 · **Blocks:** P1 (V-03), and every downstream
quantity derived from `s`

## Context

Implementing L0 (WBS 1.5) surfaced a disagreement between two Baseline documents that had
not been visible on paper.

Doc 03 §2.3 obtains the collisionless sheath thickness

```
s  =  (√2 / 3) · λ_D · ( 2 V_w / T_e )^{3/4}
```

by explicitly "matching `J_i` to the Bohm flux `e n_s c_s`". Doc 01 §2.2 then evaluates
that expression with `λ_D` at the **bulk** density `n₀ = 10¹⁷ m⁻³`, giving

```
λ_D = 40.7 µm      s ≈ 0.89 mm
```

Carrying the stated matching through symbolically, however, the `λ_D` that appears is the
one at the **sheath-edge** density `n_s = h_l n₀`:

```
s²  =  (4/9) ε₀ √(2e/m_i) V^{3/2} / (e n_s c_s)
    =  (4√2/9) · λ_D(n_s)² · (V_w / T_e)^{3/2}
```

and `√(4√2/9) = 0.79267` matches the document's `(√2/3)·2^{3/4} = 0.79281` exactly, which
confirms the closed form is right and only the density is at issue.

## Evidence

Verified independently of the implementation:

| Quantity | `λ_D` at `n₀` | `λ_D` at `n_s` |
|---|---|---|
| `λ_D` | 40.72 µm | 52.13 µm |
| `s` | **0.8903 mm** (doc 01 §2.2) | **1.140 mm** |
| `J_i` from Child–Langmuir at that `s` | 43.13 A·m⁻² | 26.31 A·m⁻² |
| Bohm flux `e n_s c_s` | 26.31 A·m⁻² | 26.31 A·m⁻² |
| Ratio | **1.639** | **1.000000** |

The doc 01 §2.2 value of `s` returns a Child–Langmuir current 64 % above the Bohm flux it
was supposedly matched to. The sheath-edge value reproduces it to one part in 10¹⁰.

## Decision

**The self-consistent convention is correct: `λ_D` is evaluated at `n_s = h_l n₀`, giving
`s = 1.14 mm` at RP-1.** Doc 01 §2.2's 0.89 mm is a lower bound computed at the bulk
density.

Both are reachable in code and neither is hidden:

- `child_langmuir_thickness(params)` defaults to the doc 01 convention, so the tabulated
  0.89 mm is reproduced and the document's arithmetic is checkable.
- `child_langmuir_thickness(params, h_l=...)` returns the self-consistent 1.14 mm.
- `AnalyticSheathSolver` uses the self-consistent value, because a state whose `J_i` did
  not equal its own Bohm flux would fail V-03 against itself.

## Consequences

Traced through every doc 01 quantity derived from `s`:

| Derived quantity | doc 01 value | Corrected | Does the document's conclusion survive? |
|---|---|---|---|
| Collisionality `s/λ_CX` | 0.072 | 0.092 | **Yes** — still "near-collisionless" |
| Ion transit `τ_tr` (doc 01 §2.3) | 26 ns | 30 ns | — |
| `τ_tr / T_RF` | 0.35 | 0.40 | **Yes** — still squarely in the partially-resolved regime, which is doc 01 §2.3's central argument and the physical justification for "time-resolved" in the project title |
| R-SPAT-1 spatial resolution | ≤ 90 µm | ≤ 114 µm by the same derivation | Requirement **not** relaxed — see below |

**R-SPAT-1 is deliberately not changed.** Doc 01 §2.4 derives "≤ 90 µm" as "resolve
`s ≈ 0.89 mm` with ≥ 10 samples". The same derivation at 1.14 mm gives 114 µm, so the
requirement as written is *stricter* than its own justification now demands. It stays at
90 µm: it constrains the optical resolution of a diagnostic, tightening it costs nothing
here, and loosening a requirement because a corrected calculation permits it is the wrong
direction of travel. A test records the 114 µm coupling so the relationship is on the
record rather than forgotten.

**`Γ_E` is unaffected.** `Γ_E = Γ_i · e V_w = h_l n₀ c_s e V_w` contains no `s`. The V-03
headline number stands at **6.577 kW·m⁻² = 0.658 W·cm⁻²** against doc 03 §2.3's stated
≈ 6.6 kW·m⁻². This is the single most important consequence: the gate that everything else
is checked against did not move.

## Related finding — `γ_i` in the Bohm speed

Surfaced by the same work and recorded here rather than in its own ADR, because the
resolution is identical in kind.

Doc 03 §2.1 states `c_s = √(e(T_e + γ_i T_i)/m_i)` with `γ_i = 3` for 1-D adiabatic ions.
Doc 03 §2.3's *own arithmetic* three subsections later uses `Γ_i = 0.61 × 10¹⁷ × 2691`,
and 2691 m·s⁻¹ is `√(e T_e/m_i)` — the cold-ion form, `γ_i = 0`. Doc 01 §2.2 agrees with
the arithmetic, not the statement. Doc 03 therefore disagrees with itself.

`γ_i` is a registered parameter (`sheath.gamma_i`) with nominal 0.0 and sweep range
[0.0, 3.0]. The nominal reproduces every printed number, which is what the L0 verification
anchor must do; the range makes the choice swept rather than assumed. At RP-1 the two
differ by 2.5 % in `c_s` and move `Γ_E` from 6.58 to 6.74 kW·m⁻² — small enough to sit
unnoticed in a document, large enough to spend an eighth of the R-ACC-5 budget on a
convention.

## Note for docs 01 and 03

Neither document is edited by this ADR; Baseline documents are superseded, not patched.
When they are next revised, doc 01 §2.2 should either state that its `λ_D` is the bulk
value (making `s` an explicit lower bound) or adopt 1.14 mm, and doc 03 should reconcile
§2.1 with §2.3. This is precisely the class of correction gate G-1.4 anticipates.
