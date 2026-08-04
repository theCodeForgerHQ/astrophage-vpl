# ADR-003 — PIC engine: build vs adopt

**Status:** Open (leaning build) · **Date:** 2026-08-04 · **Blocks:** P2

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

Not yet made. **Leaning build**, with Smilei adopted as an **independent cross-check plugin**
rather than as the production engine.

## Rationale for the lean

The scientific position is arguably *stronger* with two independent kinetic implementations
than with one adopted code: agreement between a purpose-built kernel and Smilei is a far
better verification argument (doc 07 RT-05) than trusting a single implementation. The
algorithm is genuinely textbook — Boris push, CIC weighting, null-collision MCC — and is not
in the category of software the build/buy rule exists to protect.

## Consequences

Accepts risk RT-05 (subtle PIC bug) and mitigates it with the cross-check. Requires the Smilei
plugin to be built in P2, not deferred.
