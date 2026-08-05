# Architecture Decision Records

Each ADR records one decision: its context, the options considered, the choice, and the
consequences. An ADR is never edited after acceptance — it is superseded by a new one.

**Rule:** any deviation from the build/buy table (doc 08 §2), or any change to a hard
constraint (doc 00 §4), requires an ADR.

| ID | Decision | Status | Blocks |
|---|---|---|---|
| [ADR-001](ADR-001-licence.md) | Licence for public release | **Open** | P7 release |
| [ADR-002](ADR-002-autodiff-substrate.md) | Automatic differentiation substrate | **Open** | P3 gradient-based inference |
| [ADR-003](ADR-003-pic-engine.md) | PIC engine: build vs adopt | **Open (leaning build)** | P2 kinetic solver |
| [ADR-004](ADR-004-interferometry-channel.md) | Retain or drop the interferometry channel | **Open — decided by data** | P4 information analysis |
| [ADR-005](ADR-005-posterior-storage.md) | Posterior representation on disk | **Open** | P3 storage layer |
| [ADR-006](ADR-006-toolchain.md) | Workspace, environment and formatter toolchain | **Accepted** | P1 (everything) |
| [ADR-007](ADR-007-child-langmuir-thickness.md) | Which density sets lambda_D in the Child-Langmuir thickness | **Accepted** | P1 (V-03), and every quantity derived from s |
| [ADR-008](ADR-008-manifest-substrate.md) | Manifest substrate: OmegaConf now, Hydra at the sweep layer | **Accepted** | P1 (G-1.3) |
| [ADR-009](ADR-009-eedf-solver.md) | EEDF / Boltzmann solver: neither BOLSIG+ nor `bolos` | **Accepted** | P1 (WBS 1.8), and everything downstream of a rate coefficient |
| [ADR-010](ADR-010-v03-sheath-thickness-gate.md) | V-03's sheath-thickness criterion is unreachable as written | **Accepted** | P1 (G-1.2), and L2's V-03 in P2 |
| [ADR-011](ADR-011-effective-momentum-transfer.md) | The two-term solver omitted the inelastic contribution to momentum transfer — found by the first published benchmark | **Accepted** | P1 (WBS 1.8), and every rate coefficient |

## Status values

| Status | Meaning |
|---|---|
| Open | Under consideration; the decision has not been made |
| Accepted | Decided and in force |
| Superseded | Replaced by a later ADR (linked) |
| Rejected | Considered and declined; kept for the record |
