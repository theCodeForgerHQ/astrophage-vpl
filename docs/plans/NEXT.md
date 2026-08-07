# NEXT — task queue

Baseline: `cff2f94`. Context: `docs/plans/2026-08-07-recovery-plan.md`.
All work on `L1 → L0`, calibration ON, `ablate="oes"` unless a task says otherwise. 48 s/run.

Tiers: **O** = orchestrator (Opus, high) · **R** = reviewer (Opus-low / Sonnet-high) · **E** = executor (Sonnet, low–med).

---

## Dependency graph

```
A ──► B ──► C ──► D ──► E
      │
F ────┼──── parallel, independent
G ────┘
R1 ───┐
R2 ───┼──── research, parallel from t=0
R3 ───┘
```

---

## A · Baseline on 10 seeds · 30 min · **E**

Run `L1→L0`, cal ON, `ablate="oes"`, seeds 0–9. Emit JSON per seed: `relative_error`,
`n_0_relative_error`, `T_e_relative_error`, `half_width_fraction`, `truth_within_interval`.
Report mean, SD, coverage count.

- **Pass:** mean < 8 %, |density err| < 2 %, SD < ±4 %
- **Kill:** SD > ±10 % → report, stop, escalate to O
- Needs dolfinx → FEniCSx box

## B · Error-budget audit · 60 min · **E + R**

Per instrument (OES, LIF, Thomson, interferometry), independently:
fix a known plasma → 100 noisy measurements → invert each **with that channel alone** →
measure actual scatter of recovered `n_0`, `T_e` → compare to the uncertainty the channel claims.

- **Output:** one line per channel — `claims ±X%, scatters ±Y%, understating by Y/X`
- **R must independently re-derive ≥1 channel's factor before it is used**
- Local, no dolfinx

## C · Fix the overconfidence · 60 min · **O writes, E runs**

O: add per-channel weights to `vpl.inverse.fusion.JointLikelihood`; `w = 1/(factor from B)²`.
Tests: weight 1.0 is bit-for-bit unchanged; w<1 strictly widens.
E: re-run A's 10 seeds with weights applied.

- **Pass:** coverage 2/10 → 8–9/10
- Mean error should barely move. If it improves a lot, treat as suspect and re-check.

## D · Kill the +8.3 % temperature bias · 45 min · **E**

1. **Cheap hypothesis test first (15 min cap):** run one seed with `truth_eedf=EedfShape.MAXWELLIAN`.
   Bias collapses → confirmed. Bias persists → **stop, report, do not proceed.**
2. If confirmed: let `kappa` be inferred instead of held fixed. Re-run 10 seeds.

- **Pass:** |temp err| < 3 %, total error → 1–2 %
- **Kill:** not confirmed in 15 min → stop; C's result is final

## E · Final run + report · 45 min · **O**

10 seeds, final config, one table, one commit.

| Metric | Now | Target |
|---|---|---|
| Mean error | 4.78 % (3 seeds) | < 3 % (10 seeds) |
| SD | ±2.53 % | < ±2 % |
| Density err | −0.04 % | near 0 |
| Temp err | +8.31 % | < 3 % |
| Coverage | 2/3 | 8–9/10 |

---

## F · Fix interferometry · parallel · **E + R**

`_BulkInterferometer` reads `state.params.n_0` and is blind to the `n_e` field.
Make it integrate the actual field along a chord.
Read `channels_interferometry.py`'s inverse-crime docstring first — sound for L0/L1, breaks for L2.

- **Expect numbers to get worse. That is correct.**

## G · Clean discrepancy sweep · parallel · **E**

5 of 25 fluid solves in the basis sweep did not converge. Make them converge (bounded
search / continuation) or exclude and record the reduced rank.

- **Never** let an unconverged solve into the basis silently

---

## Research · parallel from t=0 · **E (research agents)**

**R1 · Real-world diagnostic error bars.** Published uncertainties for Langmuir probe, LIF,
OES, Thomson density/temperature in low-pressure argon. → Gives 4.78 % a yardstick. **High value.**

**R2 · Probe sheath-expansion correction in practice.** OML / Laframboise / BRL; residual
accuracy of a *corrected* probe. → Decides whether "we beat a probe" survives. **High value.**

**R3 · Multi-diagnostic fusion prior art.** Does anyone fuse 4 channels into a joint
likelihood? Standard remedy when channels are mutually inconsistent (variance inflation,
robust likelihoods, hierarchical models)?

**R4 · BLOCKED — needs human.** Argon `A_ul` for 750.39 nm or 763.51 nm (gate V-24).
Boffard 2004 *J. Phys. D* **37** R143 · Boffard 2010 *PSST* **19** 065001 ·
Zhu & Pu 2010 *J. Phys. D* **43** 403001. All paywalled. One PDF closes it in ~30 min.

---

## Rules for every brief

- No `git add -A`, no commit, no stash. Explicit paths only.
- O re-runs the headline measurement itself before believing it.
- A self-flattering result gets **more** scrutiny, not less.
- Report failures verbatim. An all-green report with no caveats is suspect.
- No number without its configuration and commit.
- Always report density + temperature error next to energy-flux error.
