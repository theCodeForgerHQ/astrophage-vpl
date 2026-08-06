"""The identifiability map — doc 00 §5.1 S5, doc 05 §6.

doc 05 §6 opens with "**This section is what elevates the work from an inversion to an
analysis of an inversion**", and doc 00 §5.1 S5 is the corresponding success criterion:

    The identifiability analysis produces an explicit map of the operating space
    partitioned into *identifiable*, *weakly identifiable*, and *non-identifiable*
    regions, with the null-space directions characterised in each.

## What already existed, and what this module adds

:mod:`vpl.inverse.laplace` already eigen-decomposes the Hessian of the negative log
posterior, exposes the eigenvalues ascending as :attr:`~vpl.inverse.laplace.LaplacePosterior.
curvature`, and already *refuses* a posterior whose smallest relative curvature falls below
a floor — naming the offending eigenvector, because which parameter combination is
unconstrained is the informative part. That refusal is a binary non-identifiable verdict at
one point. This module generalises it into doc 00 S5's graded, three-way partition, and adds
the piece :mod:`vpl.inverse.laplace` deliberately does not compute: doc 05 §6.1's
Cramer-Rao bound, `sqrt(diag(I^-1))`.

Kept opaque to the physics on purpose, the same way :mod:`vpl.inverse.laplace` takes an
opaque log-likelihood (doc 08 §3, principle 5 — "layers do not leak"): every function here
takes a plain curvature matrix and a list of parameter names. It does not know what `n_0` or
`T_e` are, only that eigenvector component 0 is called that. This is also why it is
verifiable the only way an eigen-decomposition can be verified honestly — against matrices
whose eigenstructure is written down independently of the code that will decompose them: a
diagonal matrix, a rank-deficient matrix with an exactly known null vector, and doc 05
§6.2's predicted `n_0`-`T_e` degeneracy.

## Fisher information vs. posterior Hessian — the distinction that matters most here

doc 05 §6.1 defines `I(theta) = J^T Sigma^-1 J`: a statement about what *the data* can see.
At the MAP, with a flat prior, `laplace.py`'s posterior Hessian **is** this Fisher
information exactly — the negative log posterior is, up to an additive constant, the
negative log likelihood, so its curvature is the data's curvature.

With an informative prior the two are not the same matrix, and the difference is not a
technicality. doc 05 §2.1 puts a 1 % informative Gaussian prior on the wall bias `V_w` and
a similarly tight one on the neutral pressure `p`; if a direction in parameter space that
the data leaves completely flat happens to align with one of those, the *posterior* Hessian
is well-conditioned there even though nothing was measured. Classifying that posterior
Hessian directly and calling the result "the identifiability analysis" would report the
prior's assumption as the experiment's finding — exactly the "prior does the work, data
gets the credit" failure this module exists to make impossible to produce by accident.

Because the negative log posterior is `-log p(y|theta) - log p(theta)` and differentiation
is linear, the Hessian is additive at any point, in particular at the MAP:

    H_posterior  =  H_likelihood  +  H_prior  =  I(theta)  +  H_prior

so :func:`fisher_information_from_posterior` recovers the data's contribution by
subtraction. :func:`classify_identifiability` and :func:`cramer_rao_bound` must be given
**that** matrix, or an `I(theta)` computed directly (e.g. `J^T Sigma^-1 J`), never a raw
posterior Hessian — the module cannot enforce this by type, so it says so here and at every
call site instead.

## Why the thresholds are round numbers, honestly

:data:`STRONG_IDENTIFIABILITY_FLOOR` and :data:`WEAK_IDENTIFIABILITY_FLOOR` are stated in
terms of the *marginal standard deviation inflation factor* relative to the best-constrained
direction, because that is the quantity a reader can actually feel: a direction with
relative curvature `r` has marginal standard deviation `1/sqrt(r)` times the tightest
direction's. Order-of-magnitude cuts at 10x and 1000x are conventional, not derived from a
decision-theoretic loss function — there is no first-principles argument in this project for
*why* 10x rather than 5x or 30x is the line between "still a measurement" and "getting
vague". They are chosen because they are round in the quantity that matters (the inflation
factor, not the eigenvalue ratio) and because a claim this project does depend on —
"identifiable means constrained to within an order of magnitude of the best-measured
direction" — is one a reader can check against the number without needing the derivation.
If a benchmark ever needs a different cut, it is a one-line change to a named constant, and
every classification made under the old value is still reproducible from :attr:`
ParameterDirection.relative_curvature` alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "STRONG_IDENTIFIABILITY_FLOOR",
    "WEAK_IDENTIFIABILITY_FLOOR",
    "CramerRaoComparison",
    "IdentifiabilityMap",
    "IdentifiabilityVerdict",
    "ParameterDirection",
    "classify_identifiability",
    "compare_to_cramer_rao_bound",
    "cramer_rao_bound",
    "fisher_information_from_posterior",
]

type FloatArray = NDArray[np.float64]

#: Relative curvature (this eigenvalue / the largest eigenvalue in the same
#: decomposition) at or above which a direction is classified
#: :attr:`IdentifiabilityVerdict.IDENTIFIABLE`.
#:
#: `1e-2` because the marginal standard deviation along a direction with relative
#: curvature `r` is `1/sqrt(r)` times the best-constrained direction's, and
#: `1/sqrt(1e-2) = 10`: an error bar an order of magnitude wider than the tightest one is
#: still a number a reader can use. See the module docstring for why this is a chosen
#: convention rather than a derived quantity.
STRONG_IDENTIFIABILITY_FLOOR: Final[float] = 1.0e-2

#: Relative curvature at or above which a direction is at least
#: :attr:`IdentifiabilityVerdict.WEAKLY_IDENTIFIABLE`; below it, :attr:`
#: IdentifiabilityVerdict.NON_IDENTIFIABLE`.
#:
#: `1e-6`, three orders of magnitude below :data:`STRONG_IDENTIFIABILITY_FLOOR`, because
#: `1/sqrt(1e-6) = 1000`: a thousand-fold-wider error bar than the tightest direction is
#: doc 05 §6's null space in every practical sense, whatever its literal (positive)
#: eigenvalue. This is deliberately looser than :mod:`vpl.inverse.laplace`'s
#: `_MINIMUM_RELATIVE_CURVATURE` (`1e-9`), which is a numerical safety floor for inverting
#: a Hessian, not a scientific judgement about interpretability — this module's floor is
#: reached, and a direction is reported as non-identifiable, well before that one would
#: refuse to invert the matrix at all.
WEAK_IDENTIFIABILITY_FLOOR: Final[float] = 1.0e-6

#: Smallest fraction of the dominant |coefficient| a component must reach to appear in a
#: printed parameter combination (:attr:`ParameterDirection.combination`).
#:
#: `0.05`: components below 5 % of the largest one in the same eigenvector are, in every
#: case this project's parameterisation produces, floating-point residue rather than
#: physics — dropping them makes the combination readable without silently hiding a term
#: that matters.
_COMBINATION_DISPLAY_FLOOR: Final[float] = 0.05

#: Decimal places shown for each coefficient in a printed combination. Two, to match the
#: precision doc 05 §6.2's own worked example ("0.71 log n_0 - 0.70 log T_e") is stated to.
_COMBINATION_DECIMALS: Final[int] = 2


class IdentifiabilityVerdict(StrEnum):
    """Which of doc 00 S5's three regions one eigendirection falls in.

    A ``StrEnum`` so the label survives into a manifest or a figure caption without a
    lookup table, matching :class:`~vpl.inverse.laplace.LaplaceValidity`'s reasoning
    (doc 08 §1 principle 4).
    """

    IDENTIFIABLE = "IDENTIFIABLE"
    """Relative curvature at or above :data:`STRONG_IDENTIFIABILITY_FLOOR`: the data (or
    whatever curvature matrix was supplied) constrain this combination to within an order
    of magnitude of the best-measured direction."""

    WEAKLY_IDENTIFIABLE = "WEAKLY_IDENTIFIABLE"
    """Relative curvature between the two floors: constrained, but with a marginal
    standard deviation between 10x and 1000x the tightest direction's — doc 05 §6.2's
    "the near-null space: combinations the data barely constrain"."""

    NON_IDENTIFIABLE = "NON_IDENTIFIABLE"
    """Relative curvature below :data:`WEAK_IDENTIFIABILITY_FLOOR`, or non-positive: doc
    05 §6's null space. :func:`~vpl.inverse.laplace.laplace_posterior` refuses to report a
    covariance here; this module still names the direction rather than discarding it."""


@dataclass(frozen=True, slots=True)
class ParameterDirection:
    """One eigendirection of a curvature matrix, classified and named.

    Attributes:
        eigenvalue: Curvature along :attr:`eigenvector`.
        relative_curvature: ``eigenvalue`` divided by the largest eigenvalue in the same
            decomposition — the quantity the verdict and the two floors are stated in
            terms of.
        eigenvector: Unit vector, in whatever (typically unconstrained/log)
            coordinates the input matrix was expressed in.
        verdict: Which of doc 00 S5's three regions this direction falls in.
        combination: The eigenvector rendered as a readable parameter combination, e.g.
            ``"0.89·log n_0 + 0.45·log T_e"`` — doc 05 §6.2's requirement that "the
            eigenvectors name the degeneracies", not just report their numbers.
    """

    eigenvalue: float
    relative_curvature: float
    eigenvector: FloatArray
    verdict: IdentifiabilityVerdict
    combination: str


@dataclass(frozen=True, slots=True)
class IdentifiabilityMap:
    """doc 00 S5's identifiability map for one operating point.

    Attributes:
        parameter_names: Labels for the coordinate axes, in the order the input matrix
            was given.
        eigenvalues: Ascending, smallest (worst-constrained) first.
        eigenvectors: Columns, matching :attr:`eigenvalues` — ``eigenvectors[:, i]``
            corresponds to ``eigenvalues[i]``.
        directions: One :class:`ParameterDirection` per eigenpair, in the same ascending
            order.
    """

    parameter_names: tuple[str, ...]
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    directions: tuple[ParameterDirection, ...]

    @property
    def condition_number(self) -> float:
        """``lambda_max / lambda_min`` — doc 05 §6.2's "reported for every operating
        point" quantity, in one number."""
        return float(self.eigenvalues[-1] / self.eigenvalues[0])

    @property
    def identifiable(self) -> tuple[ParameterDirection, ...]:
        """Directions verdicted :attr:`IdentifiabilityVerdict.IDENTIFIABLE`, ascending."""
        return tuple(d for d in self.directions if d.verdict is IdentifiabilityVerdict.IDENTIFIABLE)

    @property
    def weakly_identifiable(self) -> tuple[ParameterDirection, ...]:
        """Directions verdicted :attr:`IdentifiabilityVerdict.WEAKLY_IDENTIFIABLE`."""
        return tuple(
            d for d in self.directions if d.verdict is IdentifiabilityVerdict.WEAKLY_IDENTIFIABLE
        )

    @property
    def non_identifiable(self) -> tuple[ParameterDirection, ...]:
        """Directions verdicted :attr:`IdentifiabilityVerdict.NON_IDENTIFIABLE` — doc 05
        §6's null space, characterised rather than discarded."""
        return tuple(
            d for d in self.directions if d.verdict is IdentifiabilityVerdict.NON_IDENTIFIABLE
        )


@dataclass(frozen=True, slots=True)
class CramerRaoComparison:
    """The achieved posterior precision against doc 05 §6.1's Cramer-Rao bound, per
    parameter — "is the inversion extracting the information that is present, or is it
    leaving some on the table?"

    Attributes:
        parameter_names: Labels, matching the coordinate order of the inputs.
        bound: :func:`cramer_rao_bound` applied to the data's Fisher information — the
            best marginal standard deviation *any unbiased estimator using the data
            alone* could achieve.
        achieved: The marginal standard deviation the analysis actually reported for the
            same coordinates — typically :attr:`~vpl.inverse.laplace.LaplacePosterior.
            marginal_std`, which legitimately includes the prior's contribution: the
            bound says what the data alone permit, and the interesting comparison is
            against what the full analysis (data and prior together) returned.
        efficiency: ``bound / achieved``. At most 1 for an unbiased estimator using the
            data alone (the Cramer-Rao inequality). A value above 1 is not automatically a
            bug: an informative prior lets a (biased, shrunk) posterior legitimately beat
            the unbiased-estimator bound, which is the entire point of putting a tight
            prior on a nuisance parameter. It is worth a second look only when the
            achieved value was **not** supposed to carry prior information, or when the
            two matrices were not built in the same coordinates.
    """

    parameter_names: tuple[str, ...]
    bound: FloatArray
    achieved: FloatArray
    efficiency: FloatArray


def classify_identifiability(
    fisher_information: ArrayLike, parameter_names: Sequence[str]
) -> IdentifiabilityMap:
    """Partition parameter space into doc 00 S5's identifiable / weakly identifiable /
    non-identifiable regions.

    Args:
        fisher_information: doc 05 §6.1's ``I(theta) = J^T Sigma^-1 J``, or any curvature
            matrix in the same convention (symmetric, positive semi-definite in theory).
            **Must be the data's information, not a posterior Hessian that includes prior
            curvature** — see :func:`fisher_information_from_posterior` and the module
            docstring. Symmetrised internally before decomposition, the same discipline
            :func:`~vpl.inverse.laplace.laplace_posterior` applies to its Hessian, since
            a matrix assembled from finite differences is symmetric only up to roundoff.
        parameter_names: One label per coordinate, in the matrix's own order. Free-form;
            if a coordinate is a log-transformed parameter, name it accordingly (e.g.
            ``"log n_0"``) so :attr:`ParameterDirection.combination` reads correctly —
            this module does not know the transform, only the label.

    Returns:
        An :class:`IdentifiabilityMap` with one :class:`ParameterDirection` per
        eigenpair, ascending by eigenvalue.

    Raises:
        ValueError: The matrix is not square, ``parameter_names`` does not match its
            dimension, or the largest eigenvalue is not positive (there is then no
            well-constrained direction to measure the others against).
    """
    fisher = _validate_square_symmetric(fisher_information, name="fisher_information")
    dimension = fisher.shape[0]
    if len(parameter_names) != dimension:
        raise ValueError(
            f"{len(parameter_names)} parameter names given for a {dimension}x{dimension} "
            f"matrix; classify_identifiability needs exactly one label per coordinate."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    largest = float(eigenvalues[-1])
    if largest <= 0.0:
        raise ValueError(
            f"the largest eigenvalue is {largest:.3e}, not positive; this matrix has no "
            f"well-constrained direction to measure identifiability against, so it is not "
            f"a usable Fisher information / curvature matrix."
        )

    directions = tuple(
        _direction(eigenvalues[i], eigenvectors[:, i], largest, parameter_names)
        for i in range(dimension)
    )
    return IdentifiabilityMap(
        parameter_names=tuple(parameter_names),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        directions=directions,
    )


def _direction(
    eigenvalue: float,
    eigenvector: FloatArray,
    largest: float,
    parameter_names: Sequence[str],
) -> ParameterDirection:
    relative = float(eigenvalue) / largest
    return ParameterDirection(
        eigenvalue=float(eigenvalue),
        relative_curvature=relative,
        eigenvector=eigenvector,
        verdict=_verdict(relative),
        combination=_describe_combination(eigenvector, parameter_names),
    )


def _verdict(relative_curvature: float) -> IdentifiabilityVerdict:
    if relative_curvature >= STRONG_IDENTIFIABILITY_FLOOR:
        return IdentifiabilityVerdict.IDENTIFIABLE
    if relative_curvature >= WEAK_IDENTIFIABILITY_FLOOR:
        return IdentifiabilityVerdict.WEAKLY_IDENTIFIABLE
    return IdentifiabilityVerdict.NON_IDENTIFIABLE


def _describe_combination(eigenvector: FloatArray, parameter_names: Sequence[str]) -> str:
    """Render an eigenvector as e.g. ``"0.89·log n_0 + 0.45·log T_e"``.

    An eigenvector's overall sign is arbitrary (``v`` and ``-v`` are the same
    eigendirection), which would otherwise make this string non-deterministic across
    runs that differ only in, say, LAPACK version. Fixed here by requiring the
    dominant-magnitude component to be positive, which is a property of the direction
    itself and not of how it happened to be computed.
    """
    dominant_index = int(np.argmax(np.abs(eigenvector)))
    sign = 1.0 if eigenvector[dominant_index] >= 0.0 else -1.0
    normalised = sign * np.asarray(eigenvector, dtype=np.float64)
    scale = float(np.max(np.abs(normalised)))
    if scale == 0.0:
        return "(zero vector)"

    pieces: list[str] = []
    for coefficient, name in zip(normalised, parameter_names, strict=True):
        if abs(coefficient) < _COMBINATION_DISPLAY_FLOOR * scale:
            continue
        magnitude = f"{abs(coefficient):.{_COMBINATION_DECIMALS}f}·{name}"
        if not pieces:
            pieces.append(magnitude if coefficient >= 0.0 else f"-{magnitude}")
        else:
            pieces.append(f"+ {magnitude}" if coefficient >= 0.0 else f"- {magnitude}")
    if not pieces:
        return "(no component above the display floor)"
    return " ".join(pieces)


def fisher_information_from_posterior(
    posterior_hessian: ArrayLike, prior_hessian: ArrayLike
) -> FloatArray:
    """Recover the data-only Fisher information from a Laplace posterior's Hessian.

    See the module docstring's "Fisher information vs. posterior Hessian" section for
    why this subtraction, rather than the posterior Hessian itself, is what doc 05
    §6.1's ``I(theta)`` refers to, and why skipping it can report a prior's assumption as
    the data's finding.

    Args:
        posterior_hessian: The Hessian of the negative log posterior at the MAP, e.g.
            :attr:`~vpl.inverse.laplace.LaplacePosterior.precision`.
        prior_hessian: The Hessian of the negative log prior alone, at the same point and
            in the same coordinates (the same finite-difference machinery applied to only
            the prior term is the intended source).

    Returns:
        ``posterior_hessian - prior_hessian``, symmetrised. May be singular, or even have
        small negative eigenvalues from roundoff, if the prior supplies curvature the
        data does not — :func:`classify_identifiability` treats a non-positive relative
        curvature as :attr:`IdentifiabilityVerdict.NON_IDENTIFIABLE`, which is the
        correct reading: that direction is not identified by the data at all.

    Raises:
        ValueError: The two matrices are not the same shape.
    """
    posterior = _validate_square_symmetric(posterior_hessian, name="posterior_hessian")
    prior = _validate_square_symmetric(prior_hessian, name="prior_hessian")
    if posterior.shape != prior.shape:
        raise ValueError(
            f"posterior_hessian has shape {posterior.shape} but prior_hessian has shape "
            f"{prior.shape}; both must describe the same parameter vector at the same "
            f"point to be subtracted."
        )
    return posterior - prior


def cramer_rao_bound(fisher_information: ArrayLike) -> FloatArray:
    """``sqrt(diag(I^-1))`` — doc 05 §6.1's Cramer-Rao bound, per coordinate.

    Computed from the eigendecomposition rather than ``np.linalg.inv``, so that a
    direction near doc 05 §6's null space reports an honestly infinite bound instead of
    the finite-but-meaningless number a near-singular matrix inverse silently produces —
    the same failure mode :mod:`vpl.inverse.laplace`'s module docstring documents for the
    posterior covariance, applied here to the bound instead of the estimate.

    An eigenvalue is treated as exactly zero, for this purpose, once its relative
    curvature (against the largest eigenvalue in the same decomposition) falls below
    :data:`WEAK_IDENTIFIABILITY_FLOOR` — the same floor :func:`classify_identifiability`
    uses to call the direction non-identifiable, so the two functions agree on which
    directions get a real number and which get infinity.

    Args:
        fisher_information: The data's Fisher information, doc 05 §6.1. Not a posterior
            Hessian with prior curvature folded in — see
            :func:`fisher_information_from_posterior`.

    Returns:
        One bound per input coordinate (not per eigendirection): ``bound[i]`` is
        ``sqrt((I^-1)[i, i])``.

    Raises:
        ValueError: The largest eigenvalue is not positive.
    """
    fisher = _validate_square_symmetric(fisher_information, name="fisher_information")
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    largest = float(eigenvalues[-1])
    if largest <= 0.0:
        raise ValueError(
            f"the largest eigenvalue is {largest:.3e}, not positive; there is no "
            f"well-constrained direction, so no Cramer-Rao bound can be formed."
        )

    floor = WEAK_IDENTIFIABILITY_FLOOR * largest
    identifiable_mask = eigenvalues > floor
    safe_eigenvalues = np.where(identifiable_mask, eigenvalues, 1.0)
    inverse_eigenvalues = np.where(identifiable_mask, 1.0 / safe_eigenvalues, np.inf)

    # A coordinate with an exactly-zero projection onto a non-identifiable direction
    # (component_sq == 0) must contribute exactly 0 to that coordinate's variance, not
    # ``0 * inf = nan``. Machine epsilon, not a literal 0.0 comparison, because a
    # component that is merely floating-point residue from an off-diagonal
    # decomposition must be treated the same way an exact zero is.
    component_sq = eigenvectors**2
    with np.errstate(invalid="ignore"):
        # `0 * inf` raises `invalid value encountered in multiply` even though the
        # result is discarded by the `np.where` below whenever component_sq is
        # negligible; the warning is suppressed only around this one expression, not
        # around the sum, so a genuine nan elsewhere would still surface.
        raw_contribution = component_sq * inverse_eigenvalues[np.newaxis, :]
    contribution = np.where(component_sq > np.finfo(np.float64).eps, raw_contribution, 0.0)
    variances = contribution.sum(axis=1)
    return np.sqrt(variances)


def compare_to_cramer_rao_bound(
    fisher_information: ArrayLike, achieved_std: ArrayLike, parameter_names: Sequence[str]
) -> CramerRaoComparison:
    """Compare an achieved marginal standard deviation to doc 05 §6.1's bound, per
    parameter.

    Args:
        fisher_information: The data-only Fisher information (doc 05 §6.1). See
            :func:`fisher_information_from_posterior` if starting from a Laplace
            posterior with an informative prior.
        achieved_std: The marginal standard deviation the analysis actually reported,
            same order and coordinates as ``fisher_information`` — e.g. :attr:`~vpl.
            inverse.laplace.LaplacePosterior.marginal_std`. This one legitimately
            includes the prior; see :class:`CramerRaoComparison`.
        parameter_names: Labels, matching ``fisher_information``'s coordinate order.

    Returns:
        A :class:`CramerRaoComparison`.

    Raises:
        ValueError: ``achieved_std`` does not match the matrix dimension, or contains a
            non-positive entry (a zero-width posterior claims infinite precision, and
            dividing by it would silently produce an infinite or undefined efficiency).
    """
    bound = cramer_rao_bound(fisher_information)
    achieved = np.asarray(achieved_std, dtype=np.float64)
    if achieved.shape != bound.shape:
        raise ValueError(
            f"achieved_std has shape {achieved.shape} but fisher_information implies "
            f"{bound.shape}; they must describe the same parameter vector."
        )
    if len(parameter_names) != bound.shape[0]:
        raise ValueError(
            f"{len(parameter_names)} parameter names given for {bound.shape[0]} "
            f"coordinates; compare_to_cramer_rao_bound needs exactly one label each."
        )
    if not np.all(achieved > 0.0):
        raise ValueError(
            "every achieved standard deviation must be positive; a zero-width posterior "
            "claims infinite precision, and dividing by it would silently produce an "
            "infinite or undefined efficiency rather than the degenerate input it is."
        )

    efficiency = bound / achieved
    return CramerRaoComparison(
        parameter_names=tuple(parameter_names),
        bound=bound,
        achieved=achieved,
        efficiency=efficiency,
    )


def _validate_square_symmetric(matrix: ArrayLike, *, name: str) -> FloatArray:
    """Shared entry-point validation for every curvature matrix this module accepts."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite entries")
    # Symmetrised rather than rejected if asymmetric: a matrix assembled from finite
    # differences (vpl.inverse.laplace's Hessian, or J^T Sigma^-1 J built in stages) is
    # symmetric only up to roundoff, and eigh reading a single triangle would silently
    # choose one side rather than average them — the same reasoning laplace.py's
    # module-level comment gives for its own Hessian.
    return np.asarray(0.5 * (array + array.T), dtype=np.float64)
