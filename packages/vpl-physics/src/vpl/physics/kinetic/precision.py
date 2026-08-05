"""Double precision, enabled by the package rather than by the environment.

JAX defaults to **float32**, silently, and for this kernel that is a correctness bug rather
than a performance trade.

The obvious symptom is that 13 of the 21 field tests fail in single precision: the Poisson
solve loses the exactness that makes the parabola test meaningful, and the convergence study
hits its round-off floor before it reaches its asymptotic range.

The symptom that would actually hurt is quieter. doc 03 §4.4 puts a run at 65 800 timesteps.
Position error accumulates every one of them, and in float32 the accumulated phase error over
ten RF periods is comparable to a cell width — so particles arrive at the wall at the wrong
phase, and the IEDF, which is the deliverable, is wrong while every *moment* of it still
looks plausible. doc 03 §4.3 already warns that converging a mean over a noisy tail is a
classic silent PIC failure; this is the same failure with a different cause.

**Why not an environment variable.** ``JAX_ENABLE_X64=1`` in CI works until somebody runs a
script by hand, and then it does not, and nothing says so. Enabling it at import means the
guarantee travels with the code that needs it.

The one cost: this is process-global JAX state. Any other JAX code in the same process gets
float64 too. That is a deliberate trade — for this project, silently *gaining* precision is
harmless and silently losing it is not — and it is stated here rather than discovered.
"""

from __future__ import annotations

import jax

__all__ = ["enable_double_precision", "is_double_precision"]


def enable_double_precision() -> None:
    """Turn on JAX's x64 mode. Idempotent, and called at package import.

    The ``type: ignore`` is JAX's, not ours: ``jax.config.update`` carries no annotations,
    and under ``--strict`` calling it from typed code is an error. Narrowed to the one call
    rather than silenced module-wide.
    """
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


def is_double_precision() -> bool:
    """Whether x64 is currently active — for a runtime guard in the solver."""
    return bool(jax.config.jax_enable_x64)
