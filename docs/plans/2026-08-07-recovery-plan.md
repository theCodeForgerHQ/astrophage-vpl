# Recovery plan — get a defensible sub-10 % T2 result in ~4 hours

**Written:** 2026-08-07 · **Baseline commit:** `cff2f94` · **Branch:** `p1-foundation`

This is a self-contained handoff. A new session should be able to start from this file
alone. It states what is measured, what is broken, what to do, and — importantly — what
nobody knows how to do yet.

---

## 0. TL;DR for whoever picks this up

The project already has a **genuine sub-10 % result**: 4.78 % mean energy-flux error with
density recovered to **0.04 %**, using three channels (LIF-led, OES dropped). It is measured
on only 3 seeds and its credible interval covers the truth 2 times in 3.

The work is: confirm it on 10 seeds, find out why every channel understates its own
uncertainty, fix that so coverage becomes honest, then kill a stable +8.3 % temperature bias.

**Do everything on the `L1 → L0` configuration.** It is a legitimate T2 (genuinely
mismatched physics), needs no saved artefact, and runs in **48 seconds**. Every plan that
routed around this configuration's speed — surrogates, GPU work — was solving a problem that
did not exist.

---

## 1. Where the project actually stands

### 1.1 What is verified

| Check | Result |
|---|---|
| Total tests, 6 packages | **2 733 passing**, 0 failures |
| T0 (same model, no noise), analytic | 0.012 % error, covered |
| T0, **fluid model as the inversion** | 0.035 % error, covered |
| T1 (same model + noise) | 0.050 % error |
| `ruff` / `ruff format` / `vpl-lint` | clean as of `cff2f94` |
| `mypy` | 139 files checked, 8 pre-existing errors |

T0 passing at **two independent fidelity levels** is load-bearing: it rules out "the
framework only works because both sides are the same trivial formula", which is the first
objection a knowledgeable examiner raises.

### 1.2 The headline T2 numbers (L1 truth → L0 inversion, four channels, 5 seeds)

| Configuration | Mean error | Spread | Covered |
|---|---|---|---|
| Calibration OFF | 50.6 % | 42–55 % | 0/5 |
| Calibration ON | **29.4 %** | 14–38 % | 0/5 |
| Calibration ON + model discrepancy | 19.6 % | 1.9–33.8 % | 0/5 |

**Calibration OFF is not a real-world configuration** and must not appear in any report. It
means "the software knows its own lens transmission exactly", which no laboratory has ever
been able to say. Only calibration-ON rows correspond to something reproducible in a lab.

### 1.3 The ablation — the most informative table in the project

Same truth, same synthetic data, same seed; only the set of channels changes.
**L1 truth, calibration ON, 3 seeds:**

| Channel removed | Mean error | SD | Density err | Temp err | Interval | Covered |
|---|---|---|---|---|---|---|
| none (all four) | 31.74 % | 5.47 | −34.80 % | +8.05 % | ±0.64 % | 0/3 |
| **OES** | **4.78 %** | 2.53 | **−0.04 %** | +8.31 % | ±5.91 % | **2/3** |
| LIF | 13.81 % | 3.05 | −12.19 % | −4.86 % | ±2.55 % | 0/3 |
| Thomson | 32.36 % | 4.98 | −35.40 % | +8.08 % | ±1.39 % | 0/3 |
| Interferometry | 31.92 % | 5.33 | −34.97 % | +8.06 % | ±1.24 % | 0/3 |

Read three things out of this:

1. **The LIF-led row is genuine, not a cancellation.** Density lands at −0.04 %. The
   remaining 4.78 % is almost entirely the +8.31 % temperature bias propagating through
   `Gamma_E ~ n_0 sqrt(T_e)`: `sqrt(1.083) = 1.041`. The arithmetic closes.
2. **Fusing four channels is worse than three, and worse than either strong channel alone.**
   That is the signature of channels that constrain nearly the same direction, disagree
   slightly, and whose *intersection* amplifies the disagreement.
3. **Removing any channel widens the interval 2–9×.** Adding information makes the posterior
   more confident and *less* correct. Every channel understates its own uncertainty; fusion
   compounds it roughly 50-fold.

### 1.4 Same ablation against an L2 (particle) truth — it disagrees

| Channel removed | Error |
|---|---|
| none | 6.47 % |
| OES | 52.42 % |
| LIF | 0.07 % (covered) |

The two truth models give **opposite** verdicts on which channel is harmful. Neither channel
is reliably right. Do not build a plan on "channel X is broken" — that hypothesis was tested
and failed.

---

## 2. Defects found and their status

| # | Finding | Status |
|---|---|---|
| 1 | The 6.47 % headline was two errors cancelling (+23.7 % density, −32.4 % temperature) | **Understood.** Do not quote 6.47 % |
| 2 | Interferometry reads `state.params.n_0` and is blind to the density field. Halving `n_e` changes its reading by zero bits. Confirmed twice: directly, and by the discrepancy estimator measuring its model error as exactly 0 | **Open** — Block F |
| 3 | Thomson refuses a discrepancy term at RP-1: 13 of 20 channels below the 20-photoelectron floor (counts run 4.7e-8 to 303) | **Documented**, correct behaviour |
| 4 | Thomson and interferometry are inert — removing either moves the answer < 0.5 pp | **Open**, low priority |
| 5 | An L1 Newton solve costs **1.24 s**, not the 65 ms claimed in four places (one annotated "(measured)") | **Corrected in code**, both figures kept |
| 6 | `mypy_path` omitted `vpl-validation`, so the documented mypy command errored out and checked **nothing** | **Fixed** in `cff2f94` |
| 7 | `vpl-lint` FAILED — 3 unregistered literals in `swarm.py` (CI-gating per doc 08 §5) | **Fixed** in `cff2f94` |
| 8 | 5 of 25 fluid solves in the discrepancy sweep did not converge; up to 20 % of the basis may be built from unconverged solutions | **Open** — Block G |
| 9 | Model discrepancy improves accuracy ~10 pp but does **not** fix coverage (0/5). Its error directions are near-orthogonal to the parameter-sensitivity directions | **Understood.** Keep the machinery, stop expecting it to fix coverage |
| 10 | Langmuir probe recovers density 3.3–3.8× too high — **by design**, reproducing the Mott-Smith-Langmuir sheath-expansion systematic. Removing that term recovers the input exactly | **Correct behaviour**, but see §5 |

---

## 3. Agent hierarchy

Three tiers. The orchestrator writes the first module in any new subsystem itself — building
one thing produces the verification strategy the briefs need, and it is cheap insurance
against shipping a plausible-but-wrong result.

| Tier | Model | Role |
|---|---|---|
| **Orchestrator** | Opus, high effort | Plan, write the first module of each new subsystem, design verification, **read every agent report sceptically**, commit |
| **Reviewer** | Opus low / Sonnet high | Independently re-verify an executor's claim before it is believed. Never reviews its own work |
| **Executor** | Sonnet, low–medium | Mechanical implementation and measurement against a tight brief |

**Non-negotiable rules for every brief:**

- Agents must **never** `git add -A`, commit, or stash. Stage explicit paths only.
- The orchestrator re-runs the key measurement itself before believing any headline number.
- A result that flatters the project gets *more* scrutiny, not less. Two findings this
  session (the "26× better than a probe" claim, the "LIF is broken" claim) were both
  self-flattering and both did not survive checking.
- Report failures verbatim. A brief that comes back all-green with no caveats is suspect.
- Never quote a number without the configuration and commit that produced it.

---

## 4. The plan — 4 hours, 6 blocks

Blocks A–E are the critical path. F and G are parallel and independent.

### Block A — Establish the LIF-led baseline properly · 30 min · **Executor (Sonnet-low)**

Three seeds is not a result.

- Run `L1 → L0`, calibration ON, `ablate="oes"`, **seeds 0–9**.
- Emit JSON: error, density error, temperature error, half-width, covered, per seed.
- Report mean, standard deviation, coverage count.

**Success:** mean stays < 8 %, density error stays within ±2 %, SD < ±4 %.
**Kill criterion:** if SD > ±10 %, the 4.78 % was luck. Say so, stop, escalate.

*Runs on the FEniCSx box (needs dolfinx for the L1 truth). ~48 s per seed → ~8 min.*

### Block B — Error-budget audit · 60 min · **Executor (Sonnet-med) + Reviewer**

The root cause of every coverage failure, never measured.

For each of the four instruments independently:
- Fix a known plasma state.
- Generate **100** noisy measurements.
- Invert each one *with that channel alone*.
- Measure the actual scatter of the recovered parameters.
- Compare against the uncertainty that channel's own likelihood claims.

**Output:** one number per channel — *"claims ±1.0 %, actually scatters ±4.2 %, understating
by 4.2×."*

**Reviewer must independently re-derive** at least one channel's factor before it is used.

*Runs locally — no dolfinx needed, these are forward models only.*

### Block C — Correct the overconfidence and re-measure · 60 min · **Orchestrator writes, Executor runs**

Add per-channel likelihood weights to `vpl.inverse.fusion.JointLikelihood`. A Gaussian
log-likelihood scaled by `w` is exactly a Gaussian with variance inflated by `1/w`, so
`w = 1 / (understatement factor)^2` from Block B is principled, not a fudge.

- Orchestrator implements the weighting (small, in `fusion.py`) with tests.
- Executor re-runs Block A's 10 seeds with the weights applied.

**Success:** coverage moves from ~2/10 toward **8–9/10**. Mean error should barely move —
this block buys *honesty*, not accuracy. If the mean improves too, that is a bonus and should
be treated with suspicion until re-checked.

### Block D — Kill the +8.3 % temperature bias · 45 min · **Executor (Sonnet-med)**

The bias is +8.05, +8.06, +8.08, +8.31 across independent seeds. That flatness means
systematic, which means fixable.

**Prime suspect:** the truth is generated with a Druyvesteyn electron-energy distribution and
the inversion assumes Maxwellian. `PlasmaParams` carries a shape parameter `kappa` which the
inversion currently holds fixed.

- First, *test the hypothesis cheaply*: run one seed with the truth EEDF set to Maxwellian
  (`EedfShape.MAXWELLIAN` on the cell). If the temperature bias collapses, the hypothesis is
  confirmed and the fix is worth building. If it does not, **stop and report** — the cause is
  elsewhere and guessing further wastes the remaining time.
- If confirmed: let `kappa` be inferred rather than fixed, re-run 10 seeds.

**Success:** temperature bias < 3 %, total error toward 1–2 %.
**Kill criterion:** hypothesis not confirmed in the first 15 min → stop, report, keep Block C's
result as final.

### Block E — Final run and report · 45 min · **Orchestrator**

10 seeds, final configuration, one table, one commit. Report the trend:

| Metric | Baseline (now) | Target |
|---|---|---|
| Mean error | 4.78 % (3 seeds) | **< 3 %** (10 seeds) |
| SD across seeds | ±2.53 % | **< ±2 %** |
| Density error | −0.04 % | stays near 0 |
| Temperature error | +8.31 % | **< 3 %** |
| Coverage | 2/3 | **8–9/10** |

### Block F — Fix interferometry · parallel · **Executor (Sonnet-med) + Reviewer**

Make `_BulkInterferometer` integrate the actual `n_e` field along a chord instead of reading
`state.params.n_0`. Read `channels_interferometry.py`'s existing docstring first — it argues
at length that reading the parameter is *not* an inverse crime, and that argument is sound
for L0/L1 truths and **breaks for an L2 truth**, which the docstring does not cover.

**Expect the numbers to get worse. That is the correct outcome.**

### Block G — Re-run the discrepancy sweep cleanly · parallel · **Executor (Sonnet-low)**

5 of 25 solves did not converge. Either make them converge (bounded search / continuation) or
exclude them and record the reduced rank. A correction built partly on unconverged solutions
that happens to widen the interval correctly would be the worst possible outcome — right
answer, wrong reason, invisible.

---

## 5. Research tasks — dispatch these to research agents in parallel

These are open questions where the answer is not in the repo. Use web/literature search.

### R1 — What error bars do real plasma diagnostics quote? · **high value**
We have no yardstick. Is 4.78 % on ion energy flux good, ordinary, or poor? Find published
uncertainties for Langmuir probe, LIF, OES and Thomson density/temperature measurements in
low-pressure argon discharges. **Without this, "4.78 %" has no meaning to a judge.**

### R2 — How is sheath-expansion corrected in practice? · **high value**
Our probe comparison is against an *uncorrected* probe, which is not best practice, so
"we beat a probe" is currently unsupported. Find the standard correction (OML, Laframboise,
BRL) and what residual accuracy a corrected probe achieves. This decides whether the
comparison claim survives.

### R3 — Multi-diagnostic fusion in plasma physics
Does anyone fuse OES + LIF + Thomson + interferometry into a joint likelihood? How do they
handle mutually inconsistent channels? Our central finding is that naive fusion is worse than
the best single channel — is that known, and is there a standard remedy (variance inflation,
robust likelihoods, hierarchical models)?

### R4 — Argon transition probabilities (**BLOCKED, needs a human**)
Verification gate V-24 needs an independently-verified `A_ul` for 750.39 nm or 763.51 nm.
NIST ASD's Lines endpoint rejected every query this session; the three primary papers are
paywalled: Boffard 2004 *J. Phys. D* **37** R143; Boffard 2010 *PSST* **19** 065001;
Zhu & Pu 2010 *J. Phys. D* **43** 403001. **One PDF via university access closes this in
~30 min. Indefinite without it.**

---

## 6. Infrastructure

### Local (macOS)
- `uv run pytest|ruff|mypy|vpl-lint`
- **No dolfinx** → no L1 truth, no L1 inversion. L0 and L2-from-artefact only.
- Saved L2 truth (seed 0):
  `/private/tmp/claude-501/.../scratchpad/l2_truth_seed0.npz`

### Remote FEniCSx box (`aiml`, Tailscale, WSL Ubuntu)
Python 3.12.13, dolfinx 0.9.0, 20 cores.
Interpreter: `/home/qernels/miniconda3/envs/vpl-fenicsx/bin/python`

**Inline quoting breaks. Always pipe a script on stdin:**

```bash
cat > /tmp/job.sh <<'EOF'
<bash here>
EOF
timeout 300 sshpass -p '<password>' ssh -o StrictHostKeyChecking=no \
    admin@aiml "wsl -d Ubuntu -- bash -s" < /tmp/job.sh
```

Sync route that works: `COPYFILE_DISABLE=1 tar czf` locally → `scp` to
`admin@aiml:C:\Users\Admin\` → untar inside WSL → `pip install --no-deps -e` each of the six
packages. Never let pip touch numpy/scipy/mpi4py/petsc4py — they are wired to dolfinx.

Long jobs: `nohup timeout <s> <py> <script> > <log> 2>&1 &` then `disown`; poll the log on a
fresh connection. SSH auth intermittently rejects — **retry 2–3 times before concluding
failure.**

### Key entry points
- `vpl.experiment.grid.run_cell(cell, seed=, l2_truth_path=, discrepancy_path=, ablate=)`
- `vpl.experiment.grid.Cell(truth=, inversion=, noise=, imperfect_calibration=,
  calibration_uncertainty=, truth_eedf=, model_discrepancy=)`
- `vpl.experiment.discrepancy_basis.estimate_channel_discrepancy(grid_points=)`

The tier label is **not** a field — it is `tier_of_configuration`'s verdict, and it *raises*
for a mismatched-model run without noise. That is why there is no "T0 for L1-vs-L0" row.

---

## 7. Honesty rules for the final report

1. **Never quote a number without its configuration and commit.** Half this session's
   confusion came from comparing numbers produced by different code states.
2. **Never quote calibration-OFF results.** Not a real-world configuration.
3. **Report density and temperature errors alongside every energy-flux error.** The
   energy-flux number alone hides whether it is genuine or a cancellation. This is exactly
   how the 6.47 % fooled the project.
4. **Coverage is the headline metric, not accuracy.** A wide honest interval beats a narrow
   lying one — doc 00 §5.1 criterion S4.
5. **State that everything is simulation-validated-against-simulation.** Nothing has touched
   a real plasma. That is a fundamental limit of the project as scoped.

---

## 8. Suggested framing for the competition

Do not lead with an accuracy number. Lead with the apparatus:

> We built a framework that catches an inverse problem lying to itself — and here is it
> catching us. It found that our own flattering 6.47 % result was two large errors
> cancelling; that one of our four instruments was reading a configuration parameter instead
> of the plasma; that our type-checker had been checking nothing; and that a documented cost
> model was wrong by 19×. Then it produced a result we can actually defend.

That story survives a follow-up question. A lucky percentage does not.
