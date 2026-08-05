"""Order-of-accuracy verification — doc 07 §2.3, doc 06 §3.

    Method of manufactured solutions is applied to every PDE solver: choose an analytic
    solution, substitute it into the governing equations, derive the source term that
    makes it exact, then verify that the observed convergence rate matches the design
    order. Reported as a log-log error-vs-h plot with fitted slope. **Design order ± 0.1
    is required.** A solver converging at first order when second was designed has a bug,
    and MMS is the only reliable way to find it.

## Why the round-off floor gets its own machinery

Below some mesh size the discretisation error drops beneath accumulated floating-point
error and the sequence stops converging. A study that includes those levels reports an
order far below design — and invites someone to spend a day hunting a bug in a solver
that does not have one.

The naive fix, "only refine three times", trades one silent failure for another: too few
levels and a first-order solver can look second-order by luck. So the study takes as many
levels as it likes, detects where convergence stopped, reports the order over the range
that was still converging, and refuses to hand back a single number when the answer is
"this study cannot tell you".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_ORDER_TOLERANCE",
    "STALL_RATIO",
    "ConvergenceResult",
    "RefinementLevel",
    "assert_design_order",
    "linf_error",
    "observed_order",
    "relative_l2_error",
    "richardson_extrapolate",
    "weighted_l2_error",
]

#: doc 07 §2.3: "Design order ± 0.1 is required." Not a parameter with a convenient
#: default — the default *is* the specification. doc 03 §7 relaxes V-02 to ±0.15, which
#: a caller asks for explicitly so that the looser gate is visible at its call site.
DEFAULT_ORDER_TOLERANCE: Final[float] = 0.1

#: A refinement step is treated as stalled when halving ``h`` fails to reduce the error
#: by at least this factor.
#:
#: A genuinely converging first-order solver gains a factor of two per halving, so this
#: threshold catches "barely moved" without accusing a legitimately slow method. It is
#: intentionally generous: a false stall costs one wasted refinement level, while a
#: missed stall costs a bug hunt in code that is correct.
STALL_RATIO: Final[float] = 1.1


@dataclass(frozen=True, slots=True)
class RefinementLevel:
    """One point of a refinement study: a mesh size and the error at it.

    Attributes:
        h: Characteristic mesh size. For the graded meshes of doc 03 §3.4 this is the
            smallest cell, matching :attr:`SpatialGrid.min_dz` — the doc 03 §4.3
            stability constraints bind there, so the convergence study should measure
            against the same scale they do.
        error: A norm of the difference from the exact or manufactured solution.
    """

    h: float
    error: float

    def __post_init__(self) -> None:
        if self.h <= 0.0:
            raise ValueError(f"mesh size must be positive, got {self.h}")
        if self.error < 0.0:
            raise ValueError(f"an error norm cannot be negative, got {self.error}")


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    """The outcome of a refinement study.

    Attributes:
        levels: Every level supplied, sorted coarse to fine.
        observed_order: Slope of the least-squares fit through all levels.
        asymptotic_order: The same fit restricted to levels that were still converging.
            Equal to :attr:`observed_order` when nothing stalled.
        pairwise_orders: Local order between each successive pair. A global fit can look
            healthy while the finest pair has already fallen off, and only these show it.
        fit_quality: :math:`R^2` of the log-log fit over all levels. Well below 1 means
            the asymptotic regime was never reached, which is itself the finding.
        has_stalled: Whether refinement stopped paying, i.e. the round-off floor.
        stalled_from: The coarsest ``h`` at which it had stalled, or ``None``.
    """

    levels: tuple[RefinementLevel, ...]
    observed_order: float
    asymptotic_order: float
    pairwise_orders: tuple[float, ...]
    fit_quality: float
    has_stalled: bool
    stalled_from: float | None

    def __str__(self) -> str:
        rows = "\n".join(
            f"  h = {level.h:<12.6g} error = {level.error:<14.6e}"
            + ("" if i == 0 else f"  local order = {self.pairwise_orders[i - 1]:.3f}")
            for i, level in enumerate(self.levels)
        )
        stall = (
            f"\n  STALLED from h = {self.stalled_from:.6g} — refinement has reached the "
            f"round-off floor; the asymptotic order below excludes those levels."
            if self.has_stalled
            else ""
        )
        return (
            f"convergence study over {len(self.levels)} levels\n{rows}{stall}\n"
            f"  observed order   = {self.observed_order:.4f}  (R^2 = {self.fit_quality:.5f})\n"
            f"  asymptotic order = {self.asymptotic_order:.4f}"
        )


def _fit_order(levels: tuple[RefinementLevel, ...]) -> tuple[float, float]:
    """Least-squares slope of ``log(error)`` against ``log(h)``, and its :math:`R^2`."""
    log_h = np.log(np.array([level.h for level in levels]))
    log_e = np.log(np.array([level.error for level in levels]))

    slope, intercept = np.polyfit(log_h, log_e, 1)
    residuals = log_e - (slope * log_h + intercept)
    total = log_e - log_e.mean()
    if total.dot(total) == 0.0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - residuals.dot(residuals) / total.dot(total)
    return float(slope), float(r_squared)


def observed_order(levels: list[RefinementLevel]) -> ConvergenceResult:
    """Fit the order of accuracy of a refinement study.

    Levels may be supplied in any order; they are sorted coarse to fine, because a study
    read back from disk or assembled by a sweep has no reason to arrive ordered.

    Raises:
        ValueError: If fewer than two levels are supplied, if any error is zero (the
            log-log fit is undefined), or if two levels share a mesh size — two errors at
            one ``h`` is not a refinement study, and the fit would be vertical.
    """
    if len(levels) < 2:
        raise ValueError(f"a convergence study needs at least two levels, got {len(levels)}")

    ordered = tuple(sorted(levels, key=lambda level: -level.h))
    sizes = [level.h for level in ordered]
    if len(set(sizes)) != len(sizes):
        raise ValueError("refinement levels must have distinct mesh sizes")
    if any(level.error == 0.0 for level in ordered):
        raise ValueError(
            "a zero error norm has no logarithm. An exactly-zero error usually means the "
            "manufactured solution lies in the discrete space, so the study measures "
            "nothing about the order."
        )

    local_orders = tuple(
        float(np.log(a.error / b.error) / np.log(a.h / b.h)) for a, b in pairwise(ordered)
    )

    # A step is stalled when halving h barely moved the error. Once stalled, the finer
    # levels stay stalled: the floor does not lift on further refinement.
    stalled_index: int | None = None
    for index, (a, b) in enumerate(pairwise(ordered)):
        if a.error / b.error < STALL_RATIO:
            stalled_index = index + 1
            break

    slope, quality = _fit_order(ordered)
    converging = ordered[:stalled_index] if stalled_index is not None else ordered
    asymptotic = _fit_order(converging)[0] if len(converging) >= 2 else slope

    return ConvergenceResult(
        levels=ordered,
        observed_order=slope,
        asymptotic_order=asymptotic,
        pairwise_orders=local_orders,
        fit_quality=quality,
        has_stalled=stalled_index is not None,
        stalled_from=ordered[stalled_index].h if stalled_index is not None else None,
    )


def assert_design_order(
    result: ConvergenceResult,
    *,
    design_order: float,
    tolerance: float = DEFAULT_ORDER_TOLERANCE,
) -> None:
    """Fail unless the study converged at its design order — doc 07 §2.3.

    A stalled study fails with the stall named rather than with a low order. "Observed
    1.1, expected 2.0" is a true statement about a study that hit the round-off floor and
    a useless one: it points at the solver when the problem is the study.
    """
    if result.has_stalled:
        raise AssertionError(
            f"the refinement study stalled from h = {result.stalled_from:.6g}: refining "
            f"further stopped reducing the error, which is the round-off floor rather "
            f"than a property of the solver. Over the levels that were still converging "
            f"the order was {result.asymptotic_order:.4f} against a design order of "
            f"{design_order}. Re-run the study without the stalled levels, or in higher "
            f"precision.\n\n{result}"
        )

    if abs(result.observed_order - design_order) > tolerance:
        raise AssertionError(
            f"observed order {result.observed_order:.4f} differs from the design order "
            f"{design_order} by more than the permitted {tolerance}. doc 07 §2.3: a "
            f"solver converging at first order when second was designed has a bug, and "
            f"MMS is the only reliable way to find it.\n\n{result}"
        )


def richardson_extrapolate(coarse: float, fine: float, *, ratio: float, order: float) -> float:
    """Extrapolate two resolutions to zero mesh size.

    doc 06 §3 uses this to turn a convergence study into a discretisation **bias** term
    with an uncertainty, rather than leaving it as an unquantified "we refined until it
    stopped moving". The correction applied, ``(fine - coarse) / (r^p - 1)``, is itself
    the estimated discretisation error of the fine solution.

    Args:
        coarse: The quantity at mesh size ``h``.
        fine: The same quantity at mesh size ``h / ratio``.
        ratio: Refinement ratio, greater than one.
        order: Observed order of accuracy. Use the *observed* order rather than the
            design order — extrapolating with an order the solver did not achieve
            produces a confident and wrong answer.
    """
    if ratio <= 1.0:
        raise ValueError(f"refinement ratio must exceed one, got {ratio}")
    if order <= 0.0:
        raise ValueError(f"order must be positive, got {order}")
    return float(fine + (fine - coarse) / (ratio**order - 1.0))


def _as_pair(
    numeric: ArrayLike, exact: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = np.asarray(numeric, dtype=np.float64)
    b = np.asarray(exact, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: numeric {a.shape} against exact {b.shape}")
    return a, b


def weighted_l2_error(
    numeric: ArrayLike, exact: ArrayLike, *, weights: ArrayLike | None = None
) -> float:
    """Cell-width-weighted discrete :math:`L^2` norm of the difference.

    The weighting is what makes the norm comparable across resolutions. An unweighted
    discrete sum grows with the point count, so refining the mesh would appear to
    *increase* the error and the convergence study would report a negative order on a
    perfect solver. On the graded meshes of doc 03 §3.4 the weights are the cell widths.
    """
    a, b = _as_pair(numeric, exact)
    w = np.ones_like(a) if weights is None else np.abs(np.asarray(weights, dtype=np.float64))
    if w.shape != a.shape:
        raise ValueError(f"shape mismatch: weights {w.shape} against values {a.shape}")
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w)))


def linf_error(numeric: ArrayLike, exact: ArrayLike) -> float:
    """Worst-point difference.

    Reported alongside :func:`weighted_l2_error` and not instead of it. A sheath solver
    can be excellent through the bulk and wrong at the wall, which is the only place the
    answer is wanted; an averaged norm hides exactly that.
    """
    a, b = _as_pair(numeric, exact)
    return float(np.max(np.abs(a - b)))


def relative_l2_error(
    numeric: ArrayLike, exact: ArrayLike, *, weights: ArrayLike | None = None
) -> float:
    """:func:`weighted_l2_error` normalised by the magnitude of the exact solution."""
    a, b = _as_pair(numeric, exact)
    w = np.ones_like(a) if weights is None else np.abs(np.asarray(weights, dtype=np.float64))
    scale = float(np.sqrt(np.sum(w * b**2) / np.sum(w)))
    if scale == 0.0:
        raise ValueError(
            "cannot take a relative error against an all-zero reference; use "
            "weighted_l2_error for an absolute norm"
        )
    return weighted_l2_error(a, b, weights=w) / scale
