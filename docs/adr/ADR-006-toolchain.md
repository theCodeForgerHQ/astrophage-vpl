# ADR-006 — Workspace, environment and formatter toolchain

**Status:** Accepted · **Date:** 2026-08-05 · **Blocks:** P1 (everything)

## Context

Doc 08 §12 specifies the infrastructure stack: micromamba for environments, `hatch` per
package for builds, Ruff and Black for lint and format. Doc 08 §2 requires an ADR for any
deviation, and §3 requires that `pip install vpl-core vpl-inverse` work **without** dragging
in FEniCSx and Gmsh — a separation CI must enforce or it decays immediately.

Implementation exposed a tension the plan did not resolve:

1. **Two device targets, one repository.** Development happens on macOS/arm64; the reference
   machine (doc 10 §1) is an RTX A4000 under WSL2 Ubuntu 24.04. The pure-Python packages
   (`vpl-core`, `vpl-inverse`, `vpl-uq`, `vpl-experiment`, `vpl-publish`) must have a fast
   edit-test loop on both. The PDE stack (FEniCSx, PETSc, Gmsh) genuinely is painful under
   pip, which is why doc 08 chose micromamba.
2. **`hatch` is two things.** `hatchling` is a build backend; `hatch` is also an environment
   manager. Doc 08 named the tool without distinguishing, and only the backend role is
   actually required by the package layout.
3. **Ruff now formats.** When doc 08 was written the split was Ruff-lints/Black-formats.
   Running both means two tools that can disagree about the same file.

## Decision

| Concern | Doc 08 §12 | Adopted | Deviation? |
|---|---|---|---|
| Build backend | `hatch` | **`hatchling`** per package | No — this is the backend half of `hatch` |
| Workspace / resolver / lock | (unspecified) | **`uv` workspace** | New — doc 08 left this open |
| Environment, pure-Python packages | micromamba | **`uv venv`** | **Yes** |
| Environment, FEniCSx/PETSc/Gmsh | micromamba | **micromamba (conda-forge)** on the reference machine | No |
| Lint | Ruff | Ruff | No |
| Format | Black | **`ruff format`** | **Yes** |
| Python version | (unspecified) | **3.12**, pinned | New |

`uv` manages the workspace and the pure-Python dependency graph. The heavyweight scientific
stack keeps micromamba exactly as doc 08 intended, and `vpl-physics.fluid` declares those
dependencies as an optional extra so that the doc 08 §3 separation is expressible in metadata
rather than only in prose.

Python 3.12 is pinned because it is what the reference machine ships (WSL2 Ubuntu 24.04) and
because conda-forge FEniCSx builds track it. Pinning avoids a class of "works on the laptop,
fails on the GPU box" defect that would otherwise surface late.

## Consequences

- **Positive.** One lockfile covers the dev loop on both machines; `uv sync` is seconds rather
  than minutes; the package-isolation rule of doc 08 §3 becomes a testable property of the
  dependency metadata rather than a convention.
- **Positive.** A single formatter cannot disagree with itself.
- **Negative.** Two environment managers exist in the project. The boundary is stated once,
  here, and encoded in `packages/vpl-physics/pyproject.toml` extras: anything importing
  `dolfinx`, `petsc4py` or `gmsh` lives behind the `fluid` extra and is skipped by the
  `fenicsx` pytest marker where the stack is absent.
- **Negative.** `ruff format` output differs from Black in a small number of edge cases. This
  is cosmetic and CI enforces one of them, not both.
- **Neutral.** Nothing in the build/buy table of doc 08 §2 changes. No scientific dependency
  is added, removed or substituted by this decision.
