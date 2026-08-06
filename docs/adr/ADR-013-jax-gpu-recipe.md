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
