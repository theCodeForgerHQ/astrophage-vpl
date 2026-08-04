# ADR-001 — Licence for public release

**Status:** Open · **Date:** 2026-08-04 · **Blocks:** P7

## Context

Doc 12 §7 argues that adoption, not secrecy, is the moat: an inverse framework nobody can
audit is worth less than one everybody validates against. But the commercialisation model is
open-core, and the licence determines whether a competitor can take the framework, wrap it,
and sell the service tier we intend to sell.

Doc 09 §5 adds a constraint: OpenADAS files and manufacturer datasheets cannot be
redistributed, so the released artifact must work with a fetch-at-install data layer
regardless of licence.

## Options

| Option | For | Against |
|---|---|---|
| **Apache-2.0** | Maximum adoption; explicit patent grant; standard for scientific infrastructure | No copyleft — a competitor may close a derivative |
| **BSD-3-Clause** | Simplest; common in plasma codes | No patent grant |
| **MPL-2.0** | File-level copyleft — improvements to our files return; wrapping is still permitted | Less familiar to scientific users |
| **AGPL-3.0 + commercial dual licence** | Strongest protection of the service tier | Hostile to industrial adoption, which is the customer base |
| **Proprietary with open benchmark only** | Maximum control | Forfeits the credibility argument entirely |

## Decision

Not yet made. **Leaning Apache-2.0** for the framework, with the benchmark suite under CC-BY
and the trained hardware-specific surrogates kept proprietary (doc 12 §7).

## Consequences

To be recorded on acceptance. Note that this decision interacts with the patent posture:
doc 12 §7 recommends against patenting, which removes one argument for AGPL.
