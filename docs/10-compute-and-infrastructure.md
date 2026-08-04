# 10 — Compute and Infrastructure Plan

Version 1.0 · Status: **Baseline** · Owner: Ajayaditya L

> Constraint C3 (doc 00): **a single NVIDIA RTX A4000, 16 GB, at first availability**, with
> better compute assumed later. This document establishes that every headline result fits
> inside that constraint, and specifies what larger compute would buy.

---

## 1. The reference machine

### 1.1 GPU — RTX A4000

| Property | Value | Consequence for this project |
|---|---|---|
| Architecture | Ampere GA104 | — |
| VRAM | 16 GB GDDR6 ECC | Caps surrogate training-set size and sampler batch width |
| CUDA cores | 6 144 | — |
| FP32 | ~19.2 TFLOP/s | The useful number |
| TF32 (tensor) | ~76 TFLOP/s | Surrogate training |
| **FP64** | **~0.6 TFLOP/s (1/32 of FP32)** | **Decisive — see §2** |
| Memory bandwidth | 448 GB/s | PIC and stencil work are bandwidth-bound |
| NVLink | **Not supported on A4000** | No multi-GPU scaling path on this card |
| TDP | 140 W, single slot | Can run continuously for weeks — which the plan relies on |

### 1.2 Host

| Component | Assumed | Note |
|---|---|---|
| CPU | 16 cores / 32 threads | The PDE and PIC workhorse |
| RAM | 64 GB | PIC domains and POD reduction fit comfortably |
| Storage | 2 TB NVMe + bulk HDD | DS-TRAIN raw output is ~400 GB (doc 09 §4.2) |

---

## 2. The FP64 finding, and how it shapes the architecture

**A4000 FP64 throughput is 0.6 TFLOP/s. A 16-core server CPU delivers roughly 0.5–1.0
TFLOP/s FP64. The GPU therefore offers no double-precision advantage whatsoever.**

This is not a minor tuning detail — it inverts the naive plan of "put the simulation on the
GPU". The correct allocation is:

| Workload | Precision required | Device | Reason |
|---|---|---|---|
| Poisson / FEM field solve | FP64 | **CPU** | Ill-conditioned; FP32 round-off is comparable to the physical charge separation being resolved |
| Fluid solve (L1) | FP64 | **CPU** | Same |
| PIC field solve | FP64 | **CPU** | Same |
| PIC particle push | FP32 acceptable | **GPU (Tier 1)** | Bandwidth-bound; positions can be FP32 with a fixed-point offset. **Requires the precision-sensitivity test of §5** |
| Surrogate training (GP / NN) | FP32 / TF32 | **GPU** | Native fit |
| MCMC on the surrogate | FP32 | **GPU** | Native fit |
| Ray tracing | FP32 | **GPU** | Native fit |
| Statistical post-processing | FP64 | CPU | Cheap |
| Visualisation | FP32 | GPU | — |

**Mixed-precision PIC (FP32 particles, FP64 fields) is a legitimate and standard optimisation,
but it is not adopted on assertion.** It is gated on a precision-sensitivity test: run the
same case in full FP64 on CPU and in mixed precision on GPU, and require agreement in `Γ_E`
to within 0.5 % and in the IEDF to a KS distance below 0.01. Only then is the fast path used
for production. Anything else is trading correctness for speed without measuring the trade.

---

## 3. Cost model per workload

### 3.1 The dominant cost is the L2 ensemble

| Workload | Unit cost | Count | Total | Device |
|---|---|---|---|---|
| L0 analytic solve | ~10 µs | unlimited | negligible | CPU |
| L1 fluid solve | ~1 s | 10⁴ | ~3 h | CPU (16 cores) |
| **L2 PIC solve** | **~3 min** (doc 03 §4.4) | **5 500** (DS-TRAIN + DS-TEST) | **~275 h ≈ 11.5 days** | **CPU, 16 cores** |
| POD reduction | ~1 h total | 1 | 1 h | CPU |
| GP surrogate training | ~20 min | ~80 outputs, batched | ~2 h | GPU |
| L3 surrogate evaluation | ~1 ms | 10⁶ per inversion | ~15 min per inversion | GPU |
| **Single full inversion (NUTS)** | **~2 min** | — | — | GPU |
| Ray-trace precomputation | ~30 min per optical config | 5 configs | ~3 h | GPU |

### 3.2 Full programme

| Campaign | Composition | Wall clock |
|---|---|---|
| DS-TRAIN + DS-TEST | 5 500 L2 solves | **11.5 days** (CPU, continuous) |
| Surrogate build + audit | training + held-out validation | 1 day |
| DS-BENCH | 13 scenarios × 200 realisations = 2 600 inversions | ~3.6 days |
| DS-COVER | 1 000 inversions + SBC | ~1.4 days |
| DS-ABLATE | 19 × 100 = 1 900 inversions | ~2.6 days |
| DS-ENVELOPE | 2 000 inversions + FIM at each | ~4 days |
| Verification suite | MMS, convergence, conservation | ~1 day |
| **Total, single machine, serial** | | **≈ 25 days of continuous compute** |

**The programme fits on one A4000 plus a 16-core host in under a month of wall clock.** The
CPU-bound L2 ensemble and the GPU-bound inversion campaigns overlap almost perfectly, so
running them concurrently brings the realistic figure closer to **~18 days**.

This is the number that makes doc 00 C3 satisfiable, and it is why the L3 surrogate exists:
without it, DS-ENVELOPE alone would require 2 000 × 10⁶ L2 solves, which is not a large number
of days — it is geological.

### 3.3 Memory feasibility

| Object | Size | Fits in 16 GB? |
|---|---|---|
| PIC particle arrays (2 × 10⁶ particles, 7 floats FP32) | 56 MB | ✔ trivially |
| GP kernel matrix (5 000 × 5 000, FP32) | 100 MB | ✔ |
| GP training, all 80 outputs batched | ~8 GB | ✔ with headroom |
| NUTS state (4 chains × 45 params) | negligible | ✔ |
| Ray-trace scene + spectral buffers | ~2 GB | ✔ |
| POD basis + reduced coefficients (DS-TRAIN) | ~8 GB | ✔ (streamed from disk) |

**Nothing is memory-constrained.** The 16 GB limit does not bind on any planned workload,
which is worth stating because it means better compute buys *throughput*, not *capability*.

---

## 4. Tiering

Every study is defined at three tiers so the plan degrades and extends gracefully.

| Tier | Compute | What changes |
|---|---|---|
| **Tier 0 — A4000** | 1 × A4000 + 16-core CPU | **Every headline result.** DS-TRAIN 5 000 points, DS-ENVELOPE 2 000 points, `N_ppc` = 1 000 |
| **Tier 1 — better single node** | e.g. 1 × A100/L40S + 64-core CPU | DS-TRAIN 20 000 points; `N_ppc` = 10 000; neural-operator surrogate replaces GP; 2-D sheath studies |
| **Tier 2 — cluster / cloud burst** | multi-node MPI | Full 2-D/3-D PIC; electromagnetic effects; full envelope at kinetic fidelity without a surrogate |

**Tier 0 must produce every claim made publicly.** Tiers 1 and 2 widen the study — denser
sweeps, better statistics, higher-dimensional physics — but no headline number depends on
them. A plan whose results require compute that has not arrived is not a plan.

---

## 5. Precision and reproducibility

| Rule | Rationale |
|---|---|
| Physics solvers default to FP64 | Correctness first |
| Any FP32 path requires a documented precision-sensitivity test | §2 |
| Deterministic reductions where feasible | Non-deterministic GPU atomics break bit-reproducibility |
| Seeds are per-stream, not global | A change to the noise model must not perturb the plasma solve's random sequence |
| GPU non-determinism is *disclosed*, not hidden | Where determinism costs too much, the run is tagged `non-deterministic` and reproducibility is stated statistically |

**Per-stream seeding is a small decision with large consequences.** With a single global RNG,
adding one noise source shifts every subsequent random draw, and two runs that should be
comparable are not. Separate streams for plasma initialisation, collisions, photon statistics,
detector noise and sampler proposals make the runs independently perturbable — which the
ablation matrix (doc 07 §5.2) requires.

---

## 6. Job orchestration

Deliberately simple. Distributed schedulers are not needed for a single node, and adopting
one would be complexity without benefit (doc 00 C5).

| Concern | Choice |
|---|---|
| Sweep execution | `vpl sweep <manifest>` → local process pool; SLURM adapter for Tier 2 |
| Queueing | Filesystem-based work queue; resumable |
| Resumability | Every campaign checkpoints per case; interruption loses at most one case |
| Monitoring | MLflow UI + a plain progress log |
| Failure policy | Retry once, then quarantine with full diagnostics; **a quarantined case is never silently dropped from statistics** |

The last point matters: silently discarding failed runs biases every aggregate metric. Failure
counts are reported alongside results.

---

## 7. Storage budget

| Dataset | Raw | Retained (reduced) |
|---|---|---|
| DS-TRAIN | 400 GB | 8 GB |
| DS-TEST | 40 GB | 1 GB |
| DS-BENCH | 120 GB | 12 GB |
| DS-COVER | 80 GB | 4 GB |
| DS-ABLATE | 60 GB | 6 GB |
| DS-ENVELOPE | 160 GB | 15 GB |
| **Total** | **860 GB** | **46 GB** |

Raw data is regenerable from `(manifest, commit, seed)`, so it is treated as a cache. The 2 TB
NVMe holds the working set; only the 46 GB of reduced artifacts is archived long-term (doc 13
§5).

---

## 8. Contingency

| If | Then |
|---|---|
| The A4000 is delayed | Tier 0 minus the GPU is still viable: GP training and NUTS run on CPU at ~10× cost, extending the programme to ~8 weeks. **Nothing is blocked, only slowed** |
| L2 proves slower than the 3 min estimate | Reduce DS-TRAIN to 2 500 points and accept a larger surrogate-error term in the budget (doc 06 §4, term 10); the trade is explicit and measurable |
| Better compute arrives early | Move to Tier 1; re-run DS-TRAIN at higher density and tighten term 10 |
| Storage fills | Raw data is a cache — delete and regenerate |

---

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Established that FP64 on A4000 offers no advantage over CPU, driving the device-allocation policy; full programme costed at ~18–25 days on Tier 0. |
