# Design — the jury live layer

**Date:** 2026-08-07 · **Status:** approved design, not yet implemented
**Baseline:** `fc9d9a1` (tree dirty: `grid.py`, `map.py`, `random.py`, `test_map.py`)

---

## 1. The problem this solves

The project has no hardware. Every number it reports came out of a computer. The single
question a jury will actually ask is therefore not "is this impressive" but:

> Did you write a program that prints the answer you wanted?

A slide cannot answer that. A demo where the presenter drives cannot fully answer it either,
because the presenter chose the inputs. What answers it is **letting the jury choose the
input and watching the machine work**, with an audit trail that makes the ordering of events
checkable after the fact.

This document specifies a web layer that does that: jurors open a page on their own phones,
choose a seed and a configuration, and watch the closed loop run — truth sealed, forward
chain, blind inversion, then reveal.

The layer adds **no physics**. It is an observation window onto `run_cell`, plus the
bookkeeping that makes what it shows checkable.

## 2. Success criteria

1. A juror with no briefing can submit a run from a phone in under 15 seconds.
2. Every run's truth is a deterministic function of a seed the juror chose.
3. The commitment to the truth is published, timestamped, before the estimate exists.
4. A technical juror can independently verify any run from the tape on their own machine.
5. The science path is provably unchanged: `progress=None` is bit-for-bit inert.
6. Outcomes that are scientific results (missed coverage, no interval, non-convergence) are
   displayed as results, never as errors and never retried.

## 3. Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Audience access | Jury's own phones | They drive; nothing is mediated by the presenter's screen |
| Network | macOS Internet Sharing hotspot + QR to the local IP | No dependence on venue wifi or internet. All frontend assets vendored — no CDN |
| Concurrency | One global FIFO queue, one worker, shared append-only tape | Jurors see each other's runs; no one can be shown a private result |
| Transport | Server-Sent Events | Auto-reconnects on phone wifi; `Last-Event-ID` resumes without gaps |
| Per-run visibility | Staged pipeline + SHA-256 truth commitment | Makes the ordering auditable, not merely narrated |
| Jury controls | seed, channel to ablate, **truth fidelity** (L0 or L1) | Covers every demo act; every combination is valid, so nothing errors on stage |
| Solve execution | Subprocess, not in-process | Crash isolation, real cancellation, hard timeout, and keeps dolfinx/PETSc out of the server |
| Frontend | One HTML page, vanilla JS, no build step | Nothing to rebuild the night before; small payload over a hotspot |

**Rejected:** L2 truth (needs saved PIC artefacts that do not exist in the repo — it would
only raise `TruthArtefactRequiredError`). Precomputed result grids (leads with cached numbers,
which is the thing being disproved). A React/Vite frontend with live charts (adds a build step
and an npm tree for decoration; the event stream already carries what charts would need, so
this stays additive later).

## 4. Architecture

New workspace package `packages/vpl-jury/`. Two narrow additions to existing packages;
nothing else is touched.

### 4.1 Additions to existing packages

**`run_cell(progress=None)`** in `vpl.experiment.grid`. An optional
`Callable[[ProgressEvent], None]`. Defaults to `None` and is bit-for-bit inert when unused —
the same guarantee `channel_weights` already carries, and tested the same way.

The alternative, reimplementing the loop inside the demo layer, is the reimplementation trap
doc 00 §6 and several module docstrings warn against: two copies of the protocol that drift.

**`SealedTruth.commitment()`** in `vpl.validation.sealed`. Returns a SHA-256 hex digest over
a canonical serialisation of the sealed value. This belongs on the seal: it lets the seal
attest to what it holds *without revealing it*, and reading it is exactly the kind of
visible, deliberate act that module is designed around. It must **not** advance the seal's
state machine — computing a commitment is not committing an estimate.

### 4.2 New package layout

```
packages/vpl-jury/src/vpl/jury/
  events.py      wire contract — event dataclasses, canonical JSON, field allowlists
  worker.py      __main__: run request in, NDJSON events out on stdout
  queue.py       FIFO queue, single-worker supervisor, subprocess lifecycle
  tape.py        append-only JSONL, run IDs, replay, truncated-tail tolerance
  server.py      POST /runs · GET /events (SSE) · GET /tape · GET /qr · GET /health
  verify.py      `vpl-jury verify <run-id>`
  preflight.py   startup checks, reported before jurors arrive
  static/        index.html · app.js · app.css — mobile-first, vendored font
packages/vpl-jury/tests/
```

`vpl-jury` depends on `vpl-experiment`, which doc 08 §3 already designates as the only
package permitted to import both `vpl-physics` and `vpl-inverse`. No new dependency
inversion is introduced.

### 4.3 Environment

The server and worker both run in the `vpl-t2` micromamba environment, because the L1 truth
path needs dolfinx. The subprocess is launched with `sys.executable`, so no environment name
is hard-coded. Every test except one L1 smoke test runs in the workspace `.venv` without
dolfinx, using L0/T1 configurations, so CI stays fast.

Environment recipe (verified working on macOS/arm64, 2026-08-07):

```bash
micromamba create -y -n vpl-t2 -c conda-forge python=3.12 "fenics-dolfinx=0.9" \
  petsc4py mpich numpy scipy pytest pint sympy pyyaml h5py zarr pyarrow omegaconf
```

Plus the server's own dependencies: `starlette`, `uvicorn`, `segno` (QR generation).

> The `fenics-dolfinx=0.9` pin is load-bearing. An unpinned install resolves to 0.10.0, under
> which 26 of the 92 L1 verification tests fail and 14 error. This is worth its own note in
> `ci.yml`, which also omits `pyyaml` and the `vpl-core[storage]` extras that the closed loop
> needs.

## 5. Stage events

The jury chooses the **truth fidelity**. The tier is derived from the resulting cell by
`Cell.tier` and displayed as a consequence, never selected: L0 truth against L0 inversion is
T1, L1 truth against L0 inversion is T2.

The event points map onto the step comments already present in `run_cell`:

| # | Event | Emitted at | Payload |
|---|---|---|---|
| 1 | `config_accepted` | after `cell.tier` | tier, cell, seed, ablation, git SHA, dirty flag |
| 2 | `truth_solved` | after step 1 (`_truth_state`) | solver fidelity, wall time. **No truth values.** |
| 3 | `reference_solved` | after the reference-state solve | wall time |
| 4 | `measurements_synthesised` | after step 2 | per-channel summary of what the solver will see, contributing/excluded channel names |
| 5 | `truth_sealed` | after step 3 | **the commitment digest** |
| 6 | `map_progress` | inside the `log_likelihood` closure, throttled | solve index, total-so-far, start index of `n_starts` |
| 7 | `estimate_committed` | after step 4b | `theta_hat`, Γ_E estimate, interval or `None`, MAP converged flag, iterations, distinct modes |
| 8 | `seal_opened` | after step 5 | **truth revealed**, relative error, coverage boolean, digest re-verified |
| 9 | `row` | after step 6 | the full `CellReport` |
| — | `run_failed` / `run_timeout` | supervisor | stderr tail, exit code or timeout |

Event 6 must account for the newly added `n_starts`: with multi-start MAP the solve counter
is not monotonic within a single optimisation, so the event carries the start index and the
UI renders "start 2 of 3 · solve 41".

Events 1–7 carry no truth value. Event 8 is the only one that does.

## 6. Data flow

```
phone ──POST /runs {seed, ablate, tier}──► validate ──► dedupe ──► FIFO queue
                                                                      │
                                          ┌───────────────────────────┘
                                          ▼
                              worker: spawn subprocess
                                          │  NDJSON on stdout
                                          ▼
                       supervisor: stamp seq + server time
                                          │
                              ├──► append to tape (fsync)
                              └──► publish to SSE broker ──► all phones
```

`GET /events` replays the entire tape on connect, then streams live, so a juror joining late
or reopening a locked phone sees full history rather than a blank screen. The SSE event ID is
the sequence number; reconnects resume via `Last-Event-ID`.

## 7. The proof chain

Three independent legs, because a skeptic can attack any one of them.

**Leg 1 — the truth is a function of the seed.** Every stochastic element in `run_cell`
derives from the single recorded seed (doc 00 E3): `_truth_state(seed=)`,
`build_channels(seed=)`, `maximum_a_posteriori(seed=)`. A juror choosing 8231 chose the
truth. No flattering seed can have been pre-selected, and the truth is re-derivable by anyone.

**Leg 2 — ordering is append-only.** The commitment lands at sequence *N*, the estimate at
*M > N*, the reveal at *K > M*, each timestamped and fsync'd, in a log with no rewrite path.
The estimate cannot have been fitted to a known truth; the truth cannot have been swapped
after the estimate.

**Leg 3 — independent recomputation.** `vpl-jury verify <run-id>` re-derives the truth from
the seed in a fresh process and recomputes the digest, comparing it to what was published at
sequence *N*.

### 7.1 Stated limits

Both are displayed in the UI rather than left to be discovered.

- **The commitment is binding, not secret.** Γ_E is a single float; the preimage space is
  searchable. Secrecy is neither achieved nor claimed. Immutability and ordering are.
- **Leg 3 assumes the verifier runs the code the server ran.** The tape therefore records the
  git commit SHA and whether the tree was dirty, and the UI header displays both. A dirty
  tree is reported, not hidden — the same discipline `docs/plans/2026-08-07-results.md`
  already applies to its own measurements.

## 8. Interface

Single scrolling column, mobile-first at 375 px.

```
┌────────────────────────────┐
│ ASTROPHAGE VPL · fc9d9a1 ● │  commit + clean/dirty indicator
├────────────────────────────┤
│ seed [      ] drop [OES ▾] │
│ truth ( )L0  (•)L1  [RUN]  │
│              → tier T2      │  tier is derived, shown not chosen
├────────────────────────────┤
│ ▸ #14 seed 8231 · queued 2 │  live tape, newest first
│ ▾ #13 seed 4417 · T2       │
│   ✓ truth sealed  a3f9c1…  │
│   ⣾ inverting… solve 84    │
│   ✓ estimate  9 184 W/m²   │
│   ✓ revealed  8 513 W/m²   │
│     error 7.9 % · CI 6.4 % │
│     COVERED ✘              │
├────────────────────────────┤
│ 14 runs · mean 5.1 % · 11/14│  scoreboard
└────────────────────────────┘
```

The scoreboard is nearly free given the tape, and it is the quiet win: coverage assembles
toward the documented ~8/10 in front of the jury, from seeds the jury chose.

Units and tier labels are rendered on every accuracy figure. doc 05 §7.2 requires the tier
label on any figure showing accuracy, and a screen is a figure.

## 9. Error handling

### 9.1 Results, not errors

Displayed plainly, never retried, never styled as failure.

| Outcome | Display |
|---|---|
| `interval is None` (non-positive-definite Hessian) | "no interval — Laplace Hessian not positive definite (doc 05 §6 null space, ADR-012)" |
| `truth_within_interval == False` | **COVERED ✘**. Expected roughly 2 in 10 |
| `map_converged == False` | Prominent flag. The results doc warns about converged-looking numbers from runs that half the time did not converge |
| `excluded` non-empty | Lists excluded channels. A four-channel claim with two excluded is a two-channel result (doc 01 IF-6) |

### 9.2 Genuine errors

| Condition | Handling |
|---|---|
| Invalid seed, unknown channel, unknown truth fidelity | `422` before enqueue. `joint.without()` raises `KeyError` on a typo, so names are validated server-side against the known channel set |
| Subprocess crash | `run_failed` event with stderr tail, appended to tape. Visible, not swallowed |
| Overrun | 180 s cap (wide margin over the measured 54 s four-channel case) → SIGTERM, then SIGKILL, `run_timeout` |
| Queue overflow | more than 20 pending → `429` with "queue full, N ahead" |
| Tape write failure | refuse new runs, show banner. An unrecorded run breaks leg 2, so stopping beats running blind |
| dolfinx absent | detected at startup; T2 disabled in the UI with the reason shown, rather than erroring per attempt |

### 9.3 Re-runs are a feature

Duplicate *pending* requests dedupe to one run. A juror may explicitly re-run a *completed*
run, and the UI highlights that the numbers match to every printed digit. Two jurors,
independently, same seed, identical result — the cheapest available answer to "how do I know
this isn't random."

### 9.4 Preflight

On boot, `preflight.py` checks and prints: git SHA and dirty state; dolfinx availability and
which tiers that enables; hotspot IP and QR; tape disk space; and a self-test run at seed 0,
T1, end to end. Failures surface in the green room, not on stage.

## 10. Testing

All of this is **software correctness**, not physics verification. doc 08 §8 requires the two
classes counted separately and the G-1 report reports them separately, so `vpl-jury` must not
inflate the physics-verification count.

### 10.1 The two that matter most

1. **Inertness.** `run_cell(progress=None)` and `run_cell(progress=lambda e: None)` produce a
   bit-for-bit identical `CellReport`. This is what protects the science from the demo layer.
2. **No truth leak before the reveal.** For every event with sequence below the reveal's, the
   serialised payload contains no truth value — enforced by a per-event-type **field
   allowlist**, not a substring scan, so a newly added field cannot quietly leak. Without
   this test one careless addition destroys the demo's integrity while everything still looks
   correct.

### 10.2 The rest

| Area | Tests |
|---|---|
| Proof chain | property-based (hypothesis): `seq(commitment) < seq(estimate) < seq(reveal)` for any config |
| Commitment | known-vector digest; value change ⇒ digest change; identical across processes (no dict-ordering or `PYTHONHASHSEED` dependence); `commitment()` does not advance the seal's state |
| Wire contract | round-trip per event type; golden-file test of the serialised form so a field rename fails in CI, not in the room |
| Worker | crash ⇒ `run_failed` with stderr tail; timeout ⇒ killed with `run_timeout`; **stdout carries only valid NDJSON** — dolfinx's JIT emits `ld: warning` lines, which go to stderr, and one stray stdout write would corrupt the stream |
| Queue | FIFO under concurrent submits; cap ⇒ 429; pending duplicates dedupe; explicit re-run allowed and identical; cancellation kills the subprocess |
| Tape | append-only with no rewrite path; replay reproduces the sequence exactly; `fsync` called; truncated final line tolerated on read |
| Server | validation ⇒ 422; SSE reconnect via `Last-Event-ID` has no gaps and no duplicates; concurrent subscribers receive identical events |
| `verify` | re-derived truth matches taped digest; tampered entry fails; git SHA mismatch warns |

Coverage target ≥ 80 %, per `rules/common/testing.md`.

### 10.3 A deliberate deviation

`rules/web/testing.md` puts visual regression first, with screenshots at four breakpoints.
This design specifies **one** Playwright smoke test at 375 px — load, submit, assert the
reveal renders — and **no screenshot baselines**. The UI will churn until the day before the
presentation and baselines would be pure maintenance cost on a tool whose audience is five
people. Recorded here as a conscious deviation rather than an omission; baselines can be
added if the interface settles.

## 11. Out of scope

- Live physics plots (spectra, posterior contours, convergence curves). The event stream
  already carries the data, so these are a later frontend-only addition.
- L2 truth, for want of PIC artefacts.
- Authentication. The network is a private hotspot with a handful of known people on it.
- Persistence beyond the tape file.
- Any change to the physics, the instruments, the inversion, or the reported numbers.

## 12. Open items for the implementation plan

1. Throttle interval for `map_progress` — must be frequent enough to look alive, rare enough
   not to flood six phones. Start at every 5th solve and tune against a real device.
2. Whether `commitment()` takes an explicit salt parameter. Default no salt, to keep the
   digest independently reproducible from the seed alone.
3. Exact canonical serialisation for the digest (float repr strategy). Must be pinned by the
   known-vector test so it cannot drift.
