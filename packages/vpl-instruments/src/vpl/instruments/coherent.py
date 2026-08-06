"""The rank-``k`` correlated-Gaussian kernel every channel's likelihood scores through.

Doc 05 §3.2 gives each channel its own likelihood term and doc 05 §3.1 gives the channels
genuinely different statistics, so the *families* differ. What does **not** differ, and
must not, is how a **coherent** error is scored once one is present.

## The defect this module exists to make unrepeatable

A coherent error — a radiometric scale, an amplitude scale, a structural model error — is
**one** error applied identically to every channel. Doc 06 §4.1 states the property
directly: a correlated calibration error "affects *every* point identically and does
**not** average down". A likelihood that adds such a term to the *diagonal* scores it as
``n`` independent errors: it averages down as ``1 / sqrt(n)`` in the fit and the resulting
interval is too narrow by that factor. Doc 00 §5.1 criterion S4 states what that costs —
"an uncalibrated posterior is worse than no posterior".

The same defect was found and fixed independently in four places in this project
(interferometry's re-drawn calibration, the OES likelihood's missing calibration term, the
LIF likelihood's diagonal treatment of a shared calibration draw, and the model-discrepancy
term of :mod:`vpl.inverse.discrepancy`). Four independent fixes are four chances to drift.
This module is the single arithmetic both instrument-side fixes now run through, so that
the OES and LIF treatments **cannot** disagree: they are the same nine lines of algebra
called with different diagonals.

## The algebra

A coherent error of per-channel standard deviation ``v`` is a **rank-one** contribution to
the covariance rather than a diagonal one:

    Sigma = D + v v'          (rank one, fully coherent)
    Sigma = D + U' U          (rank k, a basis of k coherent directions)

with ``D`` the channel's own independent variance. Forming and factoring ``Sigma`` densely
would be ``O(n^3)`` per likelihood evaluation — a million entries over a 1024-pixel
detector, on every optimiser step. The Woodbury identity gives the quadratic form and the
log-determinant exactly, in ``O(n k^2)``:

    r' Sigma^-1 r  =  r' D^-1 r  -  (U D^-1 r)' M^-1 (U D^-1 r),    M = I_k + U D^-1 U'
    log|Sigma|     =  log|D| + log|M|

``k`` is one for a single calibration scale and a few tens for an empirical discrepancy
basis, so ``M`` is tiny.

## What the rank-one form buys, in one number

Take a residual that *is* the coherent direction — an observation differing from the
prediction by a pure multiplicative scale, ``r = v = f y``. Writing ``a = v' D^-1 v``:

    diagonal chi^2  =  a                    (grows linearly with the channel count)
    coherent chi^2  =  a / (1 + a)  <  1    (bounded above by one, forever)

The overweighting factor a diagonal treatment applies is therefore exactly ``1 + a``. For
a 201-point LIF scan whose shared 5 % calibration draw was being scored per point, that is
the ~201x figure the finding reports. ``test_calibration_coherence.py`` asserts the
identity ``chi2_diag / chi2_coherent == 1 + chi2_diag`` on both instruments, which is a
statement about the *shape* of the covariance and needs no knowledge of ``D``.

## Layering

This module imports nothing from :mod:`vpl.inverse` and never will — doc 08 §3 has the
inverse layer depending on the instrument layer and not the reverse, and
``test_instrument.py::test_the_instrument_layer_does_not_import_the_inverse_layer``
enforces it against the import graph. Callers pass plain arrays of coherent standard
deviations; this module knows only that they are coherent standard deviations.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "coherent_gaussian_log_prob",
    "fractional_calibration_row",
    "stack_coherent_rows",
]

type FloatArray = NDArray[np.float64]

#: ``2 pi`` in the Gaussian normalisation. Named because doc 08 §5 permits a named
#: constant and not a bare literal, even for one that is not a measurement.
_TWO_PI: Final[float] = 2.0 * math.pi


def fractional_calibration_row(
    prediction: FloatArray, *, relative_uncertainty: float
) -> FloatArray:
    """The coherent standard deviation of a **fractional** calibration error.

    A radiometric or amplitude calibration multiplies the whole prediction by one number,
    so its one-sigma error is ``f`` times the prediction and not a fixed offset: a bright
    pixel is mis-scaled by more absolute radiance than a faint one, by exactly the ratio of
    their brightnesses. That is what makes ``f y_hat`` — rather than ``f`` times anything
    constant — the direction the covariance must be inflated along.

    Returned already flattened to a single ``(1, n)`` basis row, because that is the shape
    :func:`stack_coherent_rows` composes and :func:`coherent_gaussian_log_prob` consumes.

    Args:
        prediction: The predicted values, in the observable's own units.
        relative_uncertainty: The registered one-sigma *relative* uncertainty of the
            calibration — ``OES-C1.radiometric_uncertainty`` (doc 04 §7.3, 6 %) or
            ``LIF.scale_uncertainty`` (doc 06 §4 item 4, 5 %).

    Raises:
        ValueError: If ``relative_uncertainty`` is negative. Zero is permitted and gives a
            null row, which is the honest encoding of a perfectly known scale.
    """
    if relative_uncertainty < 0.0:
        raise ValueError(
            f"a calibration's relative uncertainty is a standard deviation and cannot be "
            f"negative, got {relative_uncertainty}"
        )
    values = np.asarray(prediction, dtype=np.float64)
    return np.asarray(relative_uncertainty * values, dtype=np.float64).reshape(1, -1)


def stack_coherent_rows(
    rows: list[FloatArray], *, expected_shape: tuple[int, ...]
) -> FloatArray | None:
    """Compose zero or more coherent directions into one ``(k, n)`` basis.

    Returns ``None`` for an empty list, which is the caller's signal to take its own
    uncorrelated path unchanged. That ``None`` is load-bearing: it is what makes "no
    coherent term supplied" reproduce the pre-existing likelihood **bit for bit** rather
    than approximately, and therefore what protects the T0 and T1 regression baselines from
    a change that was only ever meant to affect callers that opt in.

    A row already shaped like the prediction is promoted to a single coherent direction; a
    row that is already ``(k, n)`` is taken as a basis whose ``k`` directions the error was
    observed to take. Rank one is the ``k = 1`` case of the same path rather than a second
    implementation of it.

    Args:
        rows: Coherent standard deviations, each either of ``expected_shape`` or already a
            ``(k, n)`` basis, all in the observable's units.
        expected_shape: Shape of the prediction, used to tell a single direction from a
            basis.
    """
    if not rows:
        return None
    flattened: list[FloatArray] = []
    for row in rows:
        promoted = row[None, ...] if row.shape == expected_shape else row
        flattened.append(promoted.reshape(promoted.shape[0], -1))
    return np.asarray(np.vstack(flattened), dtype=np.float64)


def coherent_gaussian_log_prob(
    *, residual: FloatArray, variance: FloatArray, basis: FloatArray
) -> float:
    """``log N(r ; 0, D + U' U)`` by Woodbury — see the module docstring for the identity.

    Everything is flat and in one consistent set of units: the caller converts. OES scores
    in photoelectrons (its diagonal is the Poisson counting variance), LIF scores in the
    observable's own units (its diagonal is the reported per-sample uncertainty squared),
    and neither difference reaches this far.

    Args:
        residual: ``observed - predicted``, flattened, length ``n``.
        variance: The independent per-channel variance ``D``, flattened, length ``n``,
            already floored by the caller if its statistics need a floor.
        basis: ``(k, n)`` coherent standard deviations.

    Raises:
        ValueError: If the basis does not span the same channels as the residual, or if the
            inflated covariance is not positive definite — which for this construction can
            only mean a degenerate basis, since ``D + U'U`` is otherwise positive definite
            by construction.
    """
    flat_residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    flat_variance = np.asarray(variance, dtype=np.float64).reshape(-1)
    if basis.shape[1] != flat_residual.size:
        raise ValueError(
            f"the coherent basis spans {basis.shape[1]} channels but the residual has "
            f"{flat_residual.size}; they describe the same channels."
        )

    # Woodbury for Sigma = D + U^T U, with U the (k, n) basis:
    #   r' Sigma^-1 r = r'D^-1 r - (U D^-1 r)' M^-1 (U D^-1 r),  M = I_k + U D^-1 U'
    #   log|Sigma|    = log|D| + log|M|
    # k is one for a calibration scale and tens for a discrepancy basis, so M is tiny and
    # this stays O(n k^2) against the O(n^3) a dense covariance would cost per evaluation.
    weighted_residual = flat_residual / flat_variance
    scaled_basis = basis / flat_variance
    middle = np.eye(basis.shape[0]) + scaled_basis @ basis.T
    projection = basis @ weighted_residual
    quadratic = float(flat_residual @ weighted_residual) - float(
        projection @ np.linalg.solve(middle, projection)
    )
    sign, log_middle = np.linalg.slogdet(middle)
    if sign <= 0.0:
        raise ValueError(
            "the coherence-inflated covariance is not positive definite; the supplied "
            "basis is degenerate"
        )
    log_det = float(np.sum(np.log(flat_variance))) + float(log_middle)
    return float(-0.5 * quadratic - 0.5 * log_det - 0.5 * flat_residual.size * math.log(_TWO_PI))
