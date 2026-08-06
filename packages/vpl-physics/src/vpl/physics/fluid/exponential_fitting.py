"""Scharfetter-Gummel exponential fitting — doc 03 §3.4.

Doc 03 §3.4 specifies the scheme by name and gives the reason in one line:

    Ion advection | Scharfetter-Gummel exponential fitting | Avoids the spurious
    oscillations that plague centred differences in the strongly-drift-dominated sheath

## What exponential fitting is, in one paragraph

For the model problem ``(a n - D n')' = 0`` the exact solution is ``exp(a z / D)``.
Scharfetter & Gummel (1969) built a two-point flux that *reproduces that exponential
exactly at any mesh size*, by weighting the two nodal values with the Bernoulli function
``B(x) = x / (exp(x) - 1)``. In one dimension the same scheme is reachable from the
finite element side: take P1 Galerkin and add an artificial diffusivity, and there is
exactly one choice of that diffusivity — Il'in (1969), Allen & Southwell (1955), and the
"optimal" upwind parameter of Brooks & Hughes (1982) §3.3 — for which the resulting
three-point stencil *is* the Scharfetter-Gummel stencil:

    D_fitted  =  (|a| h / 2) coth( |a| h / (2 D) )

Both forms live here, and the test suite asserts they agree. That matters because the two
limits are easy to confuse in review: as the cell Peclet number vanishes the fitted
diffusivity returns ``D`` and the scheme is second-order; as it diverges the fitted
diffusivity becomes ``|a| h / 2`` and the scheme is first-order upwind. A stabilisation
that sat at the second limit everywhere would be stable, monotone, plausible, and one
order less accurate than doc 03 §7 V-02 requires.

## Why upwind is not "close enough"

Upwinding is what exponential fitting *degenerates to* in the collisionless limit, not a
substitute for it. In the collisional sheath — the regime doc 01 §2.2 puts at
``s / lambda_CX = 0.7`` at 50 mTorr — the cell Peclet number is order one, and there the
two differ: exponential fitting is exact on the local exponential and upwinding smears it
over several cells. Doc 01 §2.2 makes that regime part of the operating envelope, so the
distinction is not academic.

Nothing here imports FEniCSx. The UFL builder that the solvers actually assemble with
lives in :mod:`vpl.physics.fluid.forms`; this module is the mathematics, and it is
testable without a PDE stack.
"""

from __future__ import annotations

from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "COTH_UNITY_ARGUMENT",
    "SERIES_CUTOFF",
    "bernoulli",
    "cell_peclet",
    "fitted_diffusivity",
    "scharfetter_gummel_coefficients",
]

#: A field sampled on a grid, in SI magnitudes.
FloatArray: TypeAlias = NDArray[np.float64]

#: Below this ``|x|`` the Bernoulli function is evaluated from its Taylor series.
#:
#: ``x / expm1(x)`` is well conditioned for small ``x`` — ``expm1`` is exact there — but
#: it is ``0/0`` at the origin, and the origin is reached whenever the field reverses.
#: The series ``1 - x/2 + x^2/12`` agrees with the closed form to better than ``1e-21``
#: at this cutoff, so the switch is invisible.
SERIES_CUTOFF: Final[float] = 1e-5

#: Above this ``x``, ``exp(x)`` overflows a float64 and the Bernoulli function is
#: evaluated as ``x exp(-x)`` instead. ``exp(709.78)`` is the largest finite value.
_OVERFLOW_CUTOFF: Final[float] = 700.0

#: Above this argument ``tanh`` is one to double precision, so ``coth`` is too and the
#: fitted diffusivity is its pure-upwind limit. Guarding here rather than letting the
#: division produce an infinity keeps the expression finite for a vanishing diffusivity.
COTH_UNITY_ARGUMENT: Final[float] = 40.0

#: Smallest diffusivity the Peclet ratio is formed against, to keep the division finite
#: when a caller passes a denormal. Purely a floating-point guard: any diffusivity this
#: small is the collisionless limit, which is handled exactly.
_DIFFUSIVITY_FLOOR: Final[float] = 1e-300


def bernoulli(x: FloatArray) -> FloatArray:
    """``B(x) = x / (exp(x) - 1)``, evaluated without overflow or cancellation.

    The weight function of Scharfetter & Gummel (1969). Three branches, each because the
    naive expression fails there:

    - ``|x| < SERIES_CUTOFF``: the closed form is ``0/0`` at the origin.
    - ``x > 700``: ``exp(x)`` overflows; ``x exp(-x)`` underflows gracefully to zero,
      which is the right answer.
    - otherwise: ``x / expm1(x)``, which is accurate for negative and moderate ``x``
      because ``expm1`` does not cancel against the one.

    Args:
        x: Cell Peclet numbers, or normalised potential differences — the same thing.

    Returns:
        ``B(x)``, of the same shape.
    """
    values = np.asarray(x, dtype=np.float64)
    result = np.empty(values.shape, dtype=np.float64)

    small = np.abs(values) < SERIES_CUTOFF
    large = values > _OVERFLOW_CUTOFF
    ordinary = ~(small | large)

    # Explicit masking rather than np.where: np.where evaluates both branches, so the
    # overflow it is meant to avoid happens anyway and the test configuration
    # (filterwarnings = ["error"]) turns it into a failure.
    xs = values[small]
    result[small] = 1.0 - xs / 2.0 + xs * xs / 12.0
    xl = values[large]
    result[large] = xl * np.exp(-xl)
    xo = values[ordinary]
    result[ordinary] = xo / np.expm1(xo)

    return result


def cell_peclet(*, velocity: float, diffusivity: float, cell_size: float) -> float:
    """``|a| h / (2 D)`` — the number that decides which regime a cell is in.

    Reported rather than merely used: doc 03 §7 V-02 fits a convergence order, and the
    order exponential fitting delivers depends on whether the study is above or below
    unity here. A study that does not state its cell Peclet number cannot be read.

    Returns:
        The cell Peclet number, or ``inf`` in the collisionless limit ``D = 0``.
    """
    _validate(diffusivity=diffusivity, cell_size=cell_size)
    if diffusivity == 0.0:
        return float("inf")
    return abs(velocity) * cell_size / (2.0 * diffusivity)


def fitted_diffusivity(*, velocity: float, diffusivity: float, cell_size: float) -> float:
    """``(|a| h / 2) coth(|a| h / (2 D))`` — Il'in / Allen-Southwell / Brooks-Hughes.

    The **total** diffusivity the P1 operator should carry, physical part included. The
    identity that makes it Scharfetter-Gummel is

        (|a| h / 2) coth(|a| h / (2 D))  =  D B(a h / D)  +  a h / 2

    and the test suite checks it directly, because "we stabilised it somehow" and "we
    implemented the scheme doc 03 §3.4 named" are different claims.

    Args:
        velocity: Advection velocity ``a``. Only its magnitude matters — the fitting is
            symmetric, and the upwind *direction* is carried by the Galerkin advection
            term, not by the diffusivity.
        diffusivity: Physical diffusivity ``D``. Zero is the collisionless limit and is
            handled exactly rather than by a floor.
        cell_size: Cell width ``h``.

    Returns:
        A diffusivity never below ``diffusivity``. A fitted value below the physical one
        would be anti-diffusion — unconditionally unstable, and easy to mistake in a
        coarse study for an unusually accurate scheme.

    Raises:
        ValueError: If ``diffusivity`` is negative or ``cell_size`` is not positive.
    """
    _validate(diffusivity=diffusivity, cell_size=cell_size)

    half_flux = abs(velocity) * cell_size / 2.0
    if half_flux == 0.0:
        return diffusivity
    if diffusivity == 0.0:
        return half_flux

    argument = half_flux / max(diffusivity, _DIFFUSIVITY_FLOOR)
    if argument >= COTH_UNITY_ARGUMENT:
        return half_flux
    return float(half_flux / np.tanh(argument))


def scharfetter_gummel_coefficients(
    *, velocity: float, diffusivity: float, cell_size: float
) -> tuple[float, float]:
    """The two-point flux weights of Scharfetter & Gummel (1969), eq. 20.

    The flux across one cell ``[z_j, z_{j+1}]`` is ``J = left n_j - right n_{j+1}``, with

        left  = (D / h) B(-a h / D)
        right = (D / h) B( a h / D)

    Both are strictly positive at every Peclet number, which is the discrete maximum
    principle: it is what stops the oscillation doc 03 §3.4 chose this scheme to avoid.
    Centred differences lose positivity at cell Peclet one.

    Args:
        velocity: Advection velocity ``a``.
        diffusivity: Physical diffusivity ``D``. Must be positive — the two-point form is
            written as a diffusive flux scaled by ``D / h`` and degenerates without one.
            Use :func:`fitted_diffusivity`, which handles the collisionless limit, for
            the finite element assembly.
        cell_size: Cell width ``h``.

    Returns:
        ``(left, right)``.

    Raises:
        ValueError: If ``diffusivity`` is not positive or ``cell_size`` is not positive.
    """
    _validate(diffusivity=diffusivity, cell_size=cell_size)
    if diffusivity == 0.0:
        raise ValueError(
            "the Scharfetter-Gummel two-point flux needs a positive diffusivity; the "
            "collisionless limit is fitted_diffusivity(diffusivity=0.0), which returns "
            "the upwind value |a| h / 2"
        )

    delta = np.array([velocity * cell_size / diffusivity], dtype=np.float64)
    scale = diffusivity / cell_size
    return float(scale * bernoulli(-delta)[0]), float(scale * bernoulli(delta)[0])


def _validate(*, diffusivity: float, cell_size: float) -> None:
    if diffusivity < 0.0:
        raise ValueError(
            f"diffusivity must be non-negative, got {diffusivity}. A negative one is "
            "anti-diffusion and is unconditionally unstable."
        )
    if cell_size <= 0.0:
        raise ValueError(f"cell_size must be a positive cell width, got {cell_size}")
