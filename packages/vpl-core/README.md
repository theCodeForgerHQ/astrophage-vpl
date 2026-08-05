# vpl-core

Protocols, state types, units, parameter registry and provenance for the Astrophage VPL
framework.

This package is the only one every other package depends on, and it depends on nothing
heavier than NumPy and `pint`. Per [doc 08 §1](../../docs/08-software-architecture.md), the
core contains contracts, not implementations: no solver, no instrument and no inference
engine lives here.

| Module | Contents | Specification |
|---|---|---|
| `vpl.core.protocols` | `ForwardSolver`, `Instrument`, `InverseEngine`, `NoiseModel` | doc 08 §4 |
| `vpl.core.state` | `PlasmaState`, `Measurement`, `Posterior`, `Artifact` | doc 08 §7 |
| `vpl.core.units` | `pint`-backed dimensional safety | doc 08 §5 |
| `vpl.core.registry` | entry-point plugin discovery | doc 08 §10 |
| `vpl.core.provenance` | manifest hashing, environment capture | doc 08 §7 |
| `vpl.core.params` | the parameter registry | doc 02 §12 |
