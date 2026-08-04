# ADR-005 — Posterior representation on disk

**Status:** Open · **Date:** 2026-08-04 · **Blocks:** P3

## Context

Each inversion produces 4 chains × 4 000 draws × ~45 parameters, plus derived quantities
(`Γ_E` profiles, IEDFs). Across DS-COVER (1 000), DS-ENVELOPE (2 000), DS-BENCH (2 600) and
DS-ABLATE (1 900), that is ~7 500 posteriors. Doc 10 §7 budgets 46 GB for all retained
artifacts combined.

## Options

| Option | For | Against |
|---|---|---|
| **Full samples** | Lossless; any downstream question answerable | ~7 500 × full chains is the dominant storage term |
| **Summary statistics only** | Tiny | Destroys the ability to recompute derived quantities, correlations, or new metrics later |
| **Thinned samples + summaries** | Compact; retains correlation structure | Thinning loses tail resolution, which matters for coverage |
| **Approximating density (fitted GMM / normalising flow)** | Compact; resamplable | Adds an approximation error into results whose whole point is calibrated uncertainty |
| **Both: thinned samples + full summaries + on-demand regeneration** | Balanced | Regeneration costs compute |

## Decision

Not yet made. **Leaning: thinned samples (retaining ESS ≥ 400 per parameter) + full summary
statistics + the manifest**, with full chains regenerable on demand since runs are
deterministic (doc 13 §2).

## Consequences

Thinning must be ESS-aware rather than fixed-stride, or coverage statistics will be biased.
A fitted approximating density is explicitly rejected for anything feeding the calibration
tests of doc 06 §7 — approximating the posterior and then validating the approximation's
coverage would be circular.
