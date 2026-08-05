# vpl-validation

Verification, validation and benchmarking machinery — [doc 07](../../docs/07-verification-and-validation.md).

Doc 07 §1 keeps verification ("are we solving the equations right?") structurally separate
from validation ("are we solving the right equations?"). This package holds what both
need and depends on no solver, so nothing being judged can influence the judgement.

| Module | Contents | Specification |
|---|---|---|
| `vpl.validation.convergence` | Observed order of accuracy, Richardson extrapolation, error norms | doc 07 §2.3, doc 06 §3 |
