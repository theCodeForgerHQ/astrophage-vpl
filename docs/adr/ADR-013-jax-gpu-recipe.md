# ADR-013: JAX on the RTX A4000 — the working recipe, and why it took four attempts

**Status:** Accepted
**Date:** 2026-08-06
**Related:** doc 10 §3.2 (GPU throughput), doc 11 WBS 3.1 (DS-TRAIN), ADR-006 (toolchain)

## Context

`jax` on the reference machine's RTX A4000 failed with `CUDA_ERROR_UNKNOWN` on allocations as
small as 2 304 bytes, blocking doc 11 WBS 3.1 (DS-TRAIN: 5 000 L2 runs), and through it the
L3 surrogate, the NUTS engine and the doc 11 §5 envelope sweeps. On CPU those 5 000 runs are
**132 days**, so this was not a performance question but a feasibility one.

Two theories were investigated and **disproved**, and recording them matters because both
were plausible enough to have been accepted:

- *"The GPU or the driver is broken."* Disproved decisively by running PyTorch 2.3.1+cu118 on
  the same card: FP64 alloc + matmul at **249.1 GFLOP/s**. Hardware, driver and WSL2 GPU
  passthrough were all healthy the whole time. The fault was always JAX-side.
- *"Two CUDA runtimes are loaded in one process."* Real (a loader trace showed
  `libcudart.so.12` and `.so.13` both initialising) and worth fixing, but **not the cause** —
  purging the duplicates changed nothing.

## Decision

Use **JAX 0.4.13 with the CUDA 11 wheels**, matching the CUDA 11.8 stack PyTorch already
proves works on this driver, in a dedicated environment.

```
conda create -n jaxcu11test python=3.10
pip install "jax[cuda11_pip]==0.4.13" "jaxlib==0.4.13+cuda11.cudnn86" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "numpy<2" "scipy<1.13"
pip install --force-reinstall "nvidia-cudnn-cu11==8.6.0.163"
pip install "nvidia-cuda-nvrtc-cu11"
ln -sf .../nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2 .../nvidia/cuda_nvrtc/lib/libnvrtc.so
export LD_LIBRARY_PATH=$(find .../site-packages/nvidia -name lib -type d | tr '\n' ':')
```

Verified result:

```
jax 0.4.13 devices: [gpu(id=0)]
OK fp64 alloc+matmul 1000000000.0
FP64   252.2 GFLOP/s
FP32 23451.4 GFLOP/s
```

The FP64 figure independently reproduces PyTorch's 249.1 GFLOP/s on the same card, which is
the cross-check that makes this a measurement rather than a claim.

## Why it took four attempts — a chain, not a bug

Each fix exposed the next failure, and every intermediate state looked like "JAX cannot use
this GPU". Written out because the sequence is the lesson:

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `CUDA_ERROR_UNKNOWN` on a tiny allocation | CUDA 12/13 wheels on this driver | CUDA 11 wheels |
| 2 | `np.issubsctype was removed in NumPy 2.0` | JAX 0.4.13 predates NumPy 2 | `numpy<2` |
| 3 | `FAILED_PRECONDITION: DNN library initialization failed` | `cudnn86` build, but pip resolved cuDNN **9.10** | pin `nvidia-cudnn-cu11==8.6.0.163` |
| 4 | `libnvrtc.so: cannot open shared object file` | package ships `libnvrtc.so.11.2`, loader wants the unversioned name | symlink + `LD_LIBRARY_PATH` |

Failure 3 is the one worth flagging: `jaxlib==0.4.13+cuda11.**cudnn86**` names its cuDNN
requirement in the wheel tag, and pip still installed cuDNN 9.10 beside it without complaint.
Nothing warns; it fails at kernel-compile time with a message that says nothing about
versions.

An earlier attempt was abandoned as "inconclusive" at a 20-minute time box while the 715 MB
cuDNN wheel was still downloading on a ~13 Mbit/s link. That call was correct given the box,
but it is why the budget matters: **the remaining work was four small pins and roughly forty
minutes of download**, and the honest "inconclusive, not a confirmed failure" is what kept
the hypothesis alive to be finished.

## Consequences

- WBS 3.1 (DS-TRAIN) is unblocked in principle, and with it L3, NUTS and the envelope sweeps.
- **The speedup is NOT yet measured, and must not be assumed.** doc 10 §3.2's 3.2e9
  particle-steps/s remains an unverified estimate, and gate G-1.4 requires a measurement to
  replace it. The L2 loop is deliberately *eager* rather than one `lax.scan` — `deposit_cic`
  and `gather_cic` assert every particle is inside the domain, and that assertion reads a
  concrete value, which tracing forbids (see `vpl.physics.kinetic.solver`'s module
  docstring). Only the Poisson solve and the push are jitted. **A working GPU therefore does
  not automatically mean a fast PIC**, and the next step is to measure L2 throughput on this
  device rather than to plan around a number nobody has observed.
- FP64 is ~93x slower than FP32 on this card (252 vs 23 451 GFLOP/s), and
  `vpl.physics.kinetic.precision` argues x64 is a *correctness* requirement rather than a
  preference. Any throughput projection must use the FP64 figure.
- This environment is deliberately separate from `vpl-fenicsx` and `urbantwin`, both of which
  are in active use. Nothing here modifies them.

---

## Addendum, 2026-08-06: the speedup was measured, and there is none

ADR-013 as written said "the speedup is NOT yet measured, and must not be assumed". It has
now been measured, on this device, and the answer changes the plan.

Environment `jaxcu11py311` (Python 3.11.15, jax 0.4.13+cuda11.cudnn86, numpy 1.26.4),
`[gpu(id=0)]` live at 309.9 GFLOP/s FP64. Particle-steps/s, using the solver's own cost
model `2 x n_ppc x n_cells x n_steps`, same configuration on both devices in the same
environment:

| configuration | GPU | CPU (same env) | verdict |
|---|---|---|---|
| 34 cells, 400 ppc, 27 200 particles, 400 steps | 1.293e6 | 3.798e6 | **GPU 2.94x slower** |
| 1000 cells, 1000 ppc, 2 000 000 particles, 100 steps | 8.330e6 | 9.323e6 | **GPU 1.12x slower** |

**There is no speedup at any size tested, and the GPU never wins.** The cause is the one
this module's docstring already documents rather than anything environmental: the loop is
eager by design, because `deposit_cic`/`gather_cic` assert every particle is inside the
domain and that assertion reads a concrete value which tracing forbids. Only the Poisson
solve and the push are jitted, so every step pays a host-device round trip for two kernels
that are a minority of the work. At 27 k particles the round trip dominates outright; at 2 M
it nearly pays for itself and still loses.

The ceiling is structural: **even an infinitely fast device only removes the jitted
quarter.** DS-TRAIN needs the kernel restructured — the in-domain guard replaced by
something traceable so the whole step can live inside one `lax.scan` — not a faster card.
That is a substantially larger piece of work than the environment fix, and it is now the
actual blocker on doc 11 WBS 3.1.

### A discrepancy that must be resolved before any planning number is quoted

The 5.70e7 particle-steps/s figure this project has been planning with **could not be
reproduced on this machine on either device**; the best measured anywhere was 9.32e6, a
factor of 6.1 short. The definitions agree — feeding 5.70e7 into the RP-1 production size
(1.316e11 particle-steps) reproduces the published 132 days exactly — so this is not a
units mismatch. Either that figure came from materially faster hardware, or it is wrong.

Extrapolating from what was actually measured here: **914 days for DS-TRAIN on the GPU,
817 days on this CPU**, against the 132 days on the books. Until the 5.70e7 is traced to a
machine and a configuration, every schedule resting on it is unsound, and gate G-1.4's
requirement to replace the doc 10 §3.2 estimate with a measurement is only half satisfied —
the measurement now exists, and it disagrees with the number in use.

---

## Addendum 2: the bottleneck was never the device — and the CPU wins

A throwaway probe (not committed) reimplemented the inner PIC step — CIC deposit, Thomas
Poisson, CIC gather, leapfrog push, two species — in three variants, timed on both devices
in `jaxcu11py311`. The in-domain assertion was replaced by a **traceable** clamp plus a
deferred violation counter, so the guard's safety is preserved and merely checked after the
loop rather than during it. All three variants agree to 1e-17–1e-20, and the violation
counter was zero in every run.

particle-steps/s (`2 x n_ppc x n_cells x n_steps`):

| variant | device | 27 200p / 34c / 400 steps | 2 000 000p / 1000c / 100 steps |
|---|---|---|---|
| 1 — eager (today's solver) | GPU | 1.460e6 | 3.547e7 |
| 1 — eager | CPU | 5.349e6 | 3.356e7 |
| 2 — whole step jitted | GPU | 1.477e7 | 4.415e7 |
| 2 — whole step jitted | CPU | 1.549e8 | 8.402e7 |
| 3 — `lax.scan` over all steps | GPU | 1.412e9 | 8.785e8 |
| 3 — `lax.scan` over all steps | **CPU** | **9.859e9** | **1.837e9** |

**The CPU beats the GPU in every variant at both sizes, including the fully-compiled
ceiling** — 2.09x at production scale, 7x at the small size. Every previous conclusion in
this ADR about "getting JAX onto the GPU" was solving the wrong problem.

**The gain is the refactor, not the device.** On CPU alone, `lax.scan` is **54.7x** faster
than today's eager loop at production scale. Projected from the measured CPU scan rate:

| configuration | RP-1 solve | DS-TRAIN (5 000 runs) |
|---|---|---|
| today, eager, CPU | 3 921 s | **226.9 days** |
| refactored, `lax.scan`, CPU | **71.6 s** | **4.1 days** |
| refactored, `lax.scan`, GPU | 149.8 s | 8.7 days |

**DS-TRAIN goes from infeasible to a long weekend, on hardware already in hand, with no GPU
involved.** doc 11 WBS 3.1 — and with it L3, NUTS and the doc 11 §5 envelope sweeps — is
blocked by a *code structure*, not by compute.

### What this measurement does not say

The probe omits collisions/MCC, boundary absorption, injection, secondary emission and the
diagnostic bookkeeping — all of which are real work in the production kernel and none of
which trivially fit inside a `scan`. **54.7x is a ceiling, not a promise**; the achievable
figure will be lower, and boundary handling with its fixed-capacity parking scheme is the
part most likely to resist. Even a fifth of it, however, puts DS-TRAIN inside three weeks.

### Consequence for the roadmap

The GPU work recorded above remains correct and was worth doing — it produced a working
environment, a reproducible recipe, and the measurement that showed the device was never the
constraint. But **the project should stop planning around a GPU**. The actionable item is to
make the PIC step traceable (`jax.experimental.checkify`, or the clamp-and-count pattern
this probe validated) so the time loop fits one `lax.scan`, and then run DS-TRAIN on CPU.
