# ADR-002 — Automatic differentiation substrate

**Status:** Open · **Date:** 2026-08-04 · **Blocks:** P3

## Context

Gradient-based inference (MAP, HMC/NUTS) and the Fisher information analysis (doc 05 §6) both
require derivatives of the forward operator with respect to ~45 parameters. The forward chain
spans FEniCSx (FP64, CPU), a custom PIC kernel, ray tracing and detector models — a
heterogeneous stack that no single AD system covers end to end.

## Options

| Option | For | Against |
|---|---|---|
| **JAX** | Excellent HMC ecosystem (NumPyro); JIT; GPU-native; forward and reverse mode | Cannot differentiate through FEniCSx or the PIC kernel; requires the surrogate boundary |
| **PyTorch** | Mature; good if a neural operator is adopted at Tier 1 | Weaker probabilistic-programming ecosystem than NumPyro |
| **dolfin-adjoint** | Differentiates the FEM solve itself, exactly | Only covers FEniCSx; does not reach the instrument chain |
| **Finite differences** | Works on anything | 45 parameters × forward cost; noisy; unusable for HMC |

## Decision

Not yet made. **Leaning JAX**, with the differentiability boundary placed at the L3 surrogate:
the surrogate is differentiable by construction, so HMC differentiates the *emulator*, not the
PDE. `dolfin-adjoint` is retained for verification of surrogate gradients against exact
adjoints on L1 cases.

## Consequences

Places a hard requirement on the surrogate: it must be accurate in its **gradients**, not only
in its values. A surrogate with small value error and wrong gradients will produce a
confidently wrong posterior. Adds a verification test: surrogate gradient vs `dolfin-adjoint`
exact gradient on L1.
