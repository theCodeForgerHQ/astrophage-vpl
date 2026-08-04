# ADR-004 — Retain or drop the interferometry channel

**Status:** Open — to be decided by data · **Date:** 2026-08-04 · **Blocks:** P4

## Context

Doc 01 §4.3 retained interferometry on three qualitative grounds: temporal coverage where
Thomson is too slow, an independent absolute-density cross-check, and graceful degradation.

Two subsequent calculations complicated the picture:

- Doc 01 §5.4: a HeNe interferometer is **below its own detection floor** at RP-1. Only a
  10.6 µm CO₂ system is viable, and even that is only ~3× above its noise floor at a 0.1 m
  chord.
- Doc 02 §8.2: choosing a 400 mm chamber diameter improves the floor 4× to 8.4 × 10¹⁵ m⁻³,
  bringing most of the envelope into range and materially strengthening the case.

## Decision

**Deliberately deferred to measurement.** The per-channel information analysis of doc 05 §6.3
computes interferometry's marginal contribution — `log det I` with and without the channel,
and the posterior entropy reduction — across the operating envelope. The decision rule:

- Retain if the channel contributes materially anywhere in the envelope;
- Drop if its marginal contribution is negligible everywhere.

## Consequences

Either outcome is a publishable result. "We show a fourth diagnostic channel is unnecessary"
is a stronger and more useful finding than silently keeping it, and it directly informs the
product tiering of doc 12 §2.

Closing this ADR is acceptance criterion G-4.5.
