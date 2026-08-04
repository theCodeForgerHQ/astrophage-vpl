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

## Status values

| Status | Meaning |
|---|---|
| Open | Under consideration; the decision has not been made |
| Accepted | Decided and in force |
| Superseded | Replaced by a later ADR (linked) |
| Rejected | Considered and declined; kept for the record |
