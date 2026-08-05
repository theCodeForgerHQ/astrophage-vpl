# ADR-003 — PIC engine: build vs adopt

**Status:** **Accepted (build)** · **Date:** 2026-08-04, decided 2026-08-05 · **Blocks:** P2

## Context

Doc 08 §2 forbids rebuilding mature software. Smilei and WarpX are excellent, verified,
actively developed PIC codes. However both are electromagnetic, 3-D-oriented, and built for
laser-plasma and accelerator physics. The problem here is **1D3V electrostatic** with a
boundary-dominated sheath, secondary emission, and a requirement to emit IEDFs into a
surrogate-training pipeline over thousands of runs.

## Options

| Option | For | Against |
|---|---|---|
| **Smilei** | Verified, documented, MPI+GPU | EM overhead unnecessary; heavy per-run startup for a 5 000-run ensemble; coupling to the pipeline is awkward |
| **WarpX** | Same class; strong GPU support | Same |
| **Build a 1D3V electrostatic kernel** | ~1 500 lines of textbook algorithm; trivial to couple; fast startup; easy to verify exhaustively | It is code we must verify ourselves; violates the spirit of the build/buy rule |

## Decision

**Build**, with Smilei retained as an independent cross-check plugin rather than as the
production engine. Decided 2026-08-05 at the start of P2, because this ADR blocks it.

The lean recorded below is unchanged by anything learned since; what forced the decision was
the schedule. Two additional facts now support it:

- **The GPU is in hand and measured.** RTX A4000, 16 GB, compute capability 8.6, CUDA visible
  from WSL2. A JAX kernel targets it directly; adopting Smilei would mean building and tuning
  an MPI+GPU code inside WSL2 for a 1D3V electrostatic problem it was not written for.
- **ADR-009 and ADR-011 rhyme with this.** The EEDF solver was also a "build a textbook
  algorithm" decision, and it worked — but only once an *external* benchmark was run against
  it, which is what found ADR-011's 52 % error. The lesson transfers directly and is applied
  below.

## Rationale for the lean

The scientific position is arguably *stronger* with two independent kinetic implementations
than with one adopted code: agreement between a purpose-built kernel and Smilei is a far
better verification argument (doc 07 RT-05) than trusting a single implementation. The
algorithm is genuinely textbook — Boris push, CIC weighting, null-collision MCC — and is not
in the category of software the build/buy rule exists to protect.

## Consequences

Accepts risk RT-05 (a subtle PIC bug). The original mitigation was the Smilei cross-check,
and **that cross-check is deferred** — doc 11 §9's compressed critical path does not list
WBS 2.3, and with a two-day build window it cannot be reached. This is a real reduction in
verification strength and is recorded as such rather than quietly dropped.

**What replaces it, so RT-05 is not simply accepted unmitigated.** ADR-011 is the cautionary
case: a solver that agreed with itself 1 488 times and was still 52 % wrong. The kernel is
therefore verified against references it cannot influence:

| Check | What it catches |
|---|---|
| Energy conservation, collisionless (V-07) | The push and the field interpolation disagreeing |
| Two-stream instability growth rate vs the analytic dispersion relation | The field solve and the deposition, together, at the one place theory gives a closed form |
| Bohm criterion at the sheath edge, and `Gamma_E` against the verified L0/L1 chain | The boundary physics — and this is a *cross-model* check, not a self-check |
| `N_ppc` convergence on the IEDF **shape** (KS distance, V-06) | The failure doc 03 §4.3 names: a converged mean over a tail that is still noise |

**Until the Smilei cross-check is built, the kinetic layer carries a single-implementation
risk, and any claim resting on L2 alone must say so.** The claim the project actually needs
from L2 — that inversion against a *different* forward model than the truth (tier T2) still
recovers `Gamma_E` — is weakened but not invalidated by this, because the L0/L1 chain it is
compared against is independently verified.
