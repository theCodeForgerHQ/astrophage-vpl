"""``InverseEngine`` — and the analysis of the inversion that doc 05 §6 demands.

Doc 08 §4 declares the contract. Doc 05 §5 lists six engines behind it — MAP, Laplace,
NUTS, SMC/nested sampling, EnKF and a particle filter — and states the reason they share
one: "the choice is a runtime option, and results from different engines on the same
problem are a consistency check".

## ``ForwardModel`` is not ``ForwardSolver``

Doc 08 §4 names ``ForwardModel`` in ``fit`` and never defines it, and it is emphatically
not the ``ForwardSolver`` of doc 08 §4's first block. Doc 05 §1.1 defines what the engine
actually samples against::

    y = F(x) + eps        F = F4 o F3 o F2 o F1

``F1`` is the solver. ``F`` is the whole composed chain — plasma state to emission to
optical transport to digitised counts — and doc 04 §1's layering discipline means the
engine must never see the pieces. So :class:`ForwardModel` lives here, in the inverse
package's vocabulary, because it is the inverse problem's view of the forward chain rather
than anything the forward chain knows about itself.

Its three methods are the minimum the specification forces. ``predict`` because doc 05
§3.2's likelihood is written over acquisition windows and needs one prediction per
observation; ``log_likelihood`` because that sum is what the sampler differentiates (doc 05
§5 requires a differentiable ``F``, which is why L3 exists at all); and ``fidelity``
because doc 05 §7.1 makes a *mandatory mismatch* between the level that generated the truth
and the level used inside the inversion, and doc 05 §7.2 requires the resulting tier on
every reported figure. An engine that could not ask which level it was calling could not
label its own output.

## Section 6 is the point

Doc 05 §6 opens: "This section is what elevates the work from an inversion to an analysis
of an inversion." :class:`IdentifiabilityReport` is that section as a type — the Fisher
spectrum, the eigenvectors that "name the degeneracies", the condition number the
identifiability map of doc 00 S5 is coloured by, and the Cramer-Rao comparison that answers
doc 05 §6.1's question: "is the inversion extracting the information that is present, or is
it leaving some on the table?"
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vpl.core.protocols.config import InverseConfig
from vpl.core.protocols.instrument import LogProb
from vpl.core.state import (
    Fidelity,
    MeasurementSet,
    Observable,
    PlasmaParams,
    Posterior,
    SamplerDiagnostics,
)

__all__ = [
    "ForwardModel",
    "Identifiability",
    "IdentifiabilityReport",
    "InverseEngine",
    "classify_by_condition_number",
]


class Identifiability(StrEnum):
    """How well the data determine the parameters at one operating point.

    Doc 05 §6.2 requires "the operating space partitioned into identifiable / weakly
    identifiable / non-identifiable regions", and doc 00 S5 makes that map a success
    criterion. Doc 14 RS-01 is worth reading beside this enum: a large non-identifiable
    region "is a *result*, not a failure", and publishing the boundary is a contribution.
    The type exists so the boundary can be reported, not so it can be avoided.
    """

    IDENTIFIABLE = "identifiable"
    WEAKLY_IDENTIFIABLE = "weakly_identifiable"
    NON_IDENTIFIABLE = "non_identifiable"


def classify_by_condition_number(
    condition_number: float, *, weak_above: float, non_identifiable_above: float
) -> Identifiability:
    """Partition an operating point by its Fisher condition number — doc 05 §6.2.

    The thresholds have **no defaults**, deliberately. Doc 05 §6.2 requires the partition
    but names no numbers for it, and a default here would be this module inventing an
    acceptance criterion the specification declined to state — which doc 00 C4 forbids and
    which would then propagate into every identifiability map without appearing in any
    manifest. They belong in the ``validation:`` block of doc 08 §6, where they are data.

    Args:
        condition_number: ``lambda_max / lambda_min`` of the Fisher information matrix.
        weak_above: Above this, the point is at best weakly identifiable.
        non_identifiable_above: Above this, it is not identifiable at all.

    Returns:
        The doc 05 §6.2 classification.

    Raises:
        ValueError: If the thresholds are not ordered, or if ``condition_number`` is
            below one — which means the caller inverted the ratio, and would classify a
            singular point as pristine.
    """
    if math.isnan(condition_number) or condition_number < 1.0:
        raise ValueError(
            f"condition number must be at least 1, got {condition_number}. It is "
            "lambda_max/lambda_min of a positive semi-definite matrix; anything smaller "
            "means the ratio was taken the wrong way up."
        )
    if not weak_above < non_identifiable_above:
        raise ValueError(
            f"weak_above ({weak_above}) must be below non_identifiable_above "
            f"({non_identifiable_above}); the bands are nested, not disjoint"
        )

    if condition_number > non_identifiable_above:
        return Identifiability.NON_IDENTIFIABLE
    if condition_number > weak_above:
        return Identifiability.WEAKLY_IDENTIFIABLE
    return Identifiability.IDENTIFIABLE


def _frozen_spectrum(values: NDArray[np.float64], *, n_params: int) -> NDArray[np.float64]:
    """Validate a Fisher eigenvalue spectrum and return an immutable copy."""
    array = np.array(values, dtype=np.float64, copy=True)

    if array.ndim != 1 or array.size != n_params:
        raise ValueError(
            f"eigenvalues must be one per parameter; got shape {array.shape} for "
            f"{n_params} parameters"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("eigenvalues must be finite; found nan or inf")
    if np.any(array < 0.0):
        # I = J^T Sigma^-1 J is positive semi-definite by construction (doc 05 §6.1), so
        # a negative eigenvalue is an arithmetic failure and not a weak direction. Reading
        # it as one would report a null space that the data do in fact constrain.
        raise ValueError("eigenvalues must be non-negative; the Fisher matrix is PSD")
    if np.any(np.diff(array) > 0.0):
        # condition_number and weakest_direction both index by position. An unordered
        # spectrum makes both silently name the wrong mode.
        raise ValueError("eigenvalues must be in descending order, largest first")

    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False, slots=True)
class IdentifiabilityReport:
    """The doc 05 §6 analysis at one operating point.

    Attributes:
        at: Where in the envelope this was computed. Fisher information is local
            (doc 05 §6.4), so a report without its operating point is uninterpretable —
            and benchmark B-14 sweeps 2 000 of them into the doc 00 S5 map.
        names: Parameter names, indexing the rows of :attr:`eigenvectors`.
        eigenvalues: Fisher eigenvalues, **descending**. Read-only.
        eigenvectors: Column ``k`` is the eigenvector of ``eigenvalues[k]``, with rows in
            :attr:`names` order. Read-only. Doc 05 §6.2: "The eigenvectors name the
            degeneracies", the expected one being an ``n_0``-``T_e`` correlation, since
            ``Gamma_i ~ n_0 sqrt(T_e)`` constrains the product far better than either
            factor.
        classification: The doc 05 §6.2 verdict. Supplied rather than derived, because
            the thresholds are a manifest choice — see
            :func:`classify_by_condition_number`.
        posterior_variance: Marginal posterior variance per parameter, if the inversion
            has been run. Only needed for :meth:`information_efficiency`, and absent when
            the Fisher analysis is done ahead of the fit for experiment design
            (doc 05 §9).
    """

    at: PlasmaParams
    names: tuple[str, ...]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    classification: Identifiability
    posterior_variance: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        n_params = len(self.names)

        if n_params < 1:
            raise ValueError("an identifiability report needs at least one parameter")
        if len(set(self.names)) != n_params:
            raise ValueError(f"parameter names contain duplicates: {self.names}")

        object.__setattr__(
            self, "eigenvalues", _frozen_spectrum(self.eigenvalues, n_params=n_params)
        )

        vectors = np.array(self.eigenvectors, dtype=np.float64, copy=True)
        if vectors.shape != (n_params, n_params):
            raise ValueError(
                f"eigenvectors must be square and one column per mode; got {vectors.shape} "
                f"for {n_params} parameters"
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("eigenvectors must be finite; found nan or inf")
        vectors.flags.writeable = False
        object.__setattr__(self, "eigenvectors", vectors)

        if self.posterior_variance is not None:
            variance = np.array(self.posterior_variance, dtype=np.float64, copy=True)
            if variance.shape != (n_params,):
                raise ValueError(
                    f"posterior_variance must be one per parameter; got {variance.shape}"
                )
            if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
                raise ValueError("posterior_variance must be finite and positive")
            variance.flags.writeable = False
            object.__setattr__(self, "posterior_variance", variance)

    # ── the doc 05 §6.2 spectrum ────────────────────────────────────────────────

    @property
    def condition_number(self) -> float:
        """``lambda_max / lambda_min`` — what the doc 00 S5 map is coloured by.

        Infinite when the smallest eigenvalue is zero, which is the exact signature of a
        structurally non-identifiable direction (doc 05 §6.5).
        """
        smallest = float(self.eigenvalues[-1])
        if smallest <= 0.0:
            return math.inf
        return float(self.eigenvalues[0]) / smallest

    def weakest_direction(self) -> Mapping[str, float]:
        """Loadings of the least-determined parameter combination — doc 05 §6.2.

        This is the degeneracy: the combination the data barely constrain. Reported as a
        mapping rather than a bare vector so that the statement doc 05 §6.2 asks for —
        naming the degeneracy, and showing which channel breaks it — can be written down
        without the reader having to remember the parameter ordering.
        """
        return dict(zip(self.names, (float(v) for v in self.eigenvectors[:, -1]), strict=True))

    @property
    def cramer_rao_variance(self) -> NDArray[np.float64]:
        """``diag(I^-1)`` — the best variance any unbiased estimator could achieve.

        Doc 05 §6.1: ``Cov(theta_hat) >= I^-1``. Computed from the eigen-decomposition
        rather than by inverting, so that a singular direction gives an infinite bound in
        the parameters it loads instead of raising: an unconstrained direction is a
        *result* (doc 14 RS-01), and a report that could not express it would have to omit
        the very points the identifiability map exists to show.
        """
        weights = np.asarray(self.eigenvectors**2, dtype=np.float64)
        determined = self.eigenvalues > 0.0

        bounded = np.asarray(
            weights[:, determined] @ (1.0 / self.eigenvalues[determined]), dtype=np.float64
        )
        unbounded = np.any(weights[:, ~determined] > 0.0, axis=1)

        return np.asarray(np.where(unbounded, np.inf, bounded), dtype=np.float64)

    def information_efficiency(self) -> NDArray[np.float64]:
        """``CRB / posterior variance``, per parameter — doc 05 §6.1.

        The rigorous form of "is the inversion extracting the information that is present,
        or is it leaving some on the table?". One means the posterior saturates the bound;
        below one means it is wider than the data require and something in the inversion
        is losing information.

        Raises:
            ValueError: If no posterior variance was supplied. The comparison is between
                two numbers and there is no defensible stand-in for the second.
        """
        if self.posterior_variance is None:
            raise ValueError(
                "information_efficiency compares the Cramer-Rao bound against the "
                "posterior_variance this report was not given. Attach the fitted "
                "posterior's marginal variances (doc 05 §6.1)."
            )

        return np.asarray(self.cramer_rao_variance / self.posterior_variance, dtype=np.float64)

    def _same_posterior_variance(self, other: IdentifiabilityReport) -> bool:
        mine, theirs = self.posterior_variance, other.posterior_variance
        if mine is None or theirs is None:
            # A report with the fitted variances attached is not the same report as one
            # computed ahead of the fit for experiment design (doc 05 §9), even at the
            # same operating point: only one of them can answer §6.1's question.
            return mine is None and theirs is None
        return bool(np.array_equal(mine, theirs))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdentifiabilityReport):
            return NotImplemented
        return (
            self.at == other.at
            and self.names == other.names
            and self.classification is other.classification
            and np.array_equal(self.eigenvalues, other.eigenvalues)
            and np.array_equal(self.eigenvectors, other.eigenvectors)
            and self._same_posterior_variance(other)
        )

    def __repr__(self) -> str:
        return (
            f"IdentifiabilityReport({self.classification.value}, "
            f"cond={self.condition_number:.4g}, n_params={len(self.names)})"
        )


@runtime_checkable
class ForwardModel(Protocol):
    """``F = F4 o F3 o F2 o F1`` — the composed map the engine samples against.

    Named but not defined by doc 08 §4; defined here from doc 05 §1.1. See the module
    docstring for why it is not a :class:`~vpl.core.protocols.forward.ForwardSolver` and
    why it lives in this module rather than that one.

    Structural at runtime, nominal at type-check time, as with the other protocols.
    """

    def predict(self, params: PlasmaParams, data: MeasurementSet) -> tuple[Observable, ...]:
        """Noiseless predictions, one per observation in ``data``, in its order.

        ``data`` is passed rather than a bare parameter vector because doc 05 §3.2's
        likelihood integrates each prediction over that observation's own acquisition
        window — the four channels are never simultaneous (doc 02 §10.1), so there is no
        single time at which "the prediction" could be evaluated.
        """
        ...

    def log_likelihood(self, params: PlasmaParams, data: MeasurementSet) -> LogProb:
        """The doc 05 §3.2 sum over channels and observations, in nats."""
        ...

    def fidelity(self) -> Fidelity:
        """Which level of doc 03 §1 this model calls — doc 05 §7.1's mandatory mismatch."""
        ...


@runtime_checkable
class InverseEngine(Protocol):
    """One of the doc 05 §5 engines, behind the doc 08 §4 contract.

    Structural at runtime, nominal at type-check time; see
    :class:`~vpl.core.protocols.forward.ForwardSolver`.
    """

    def configure(self, cfg: InverseConfig) -> None:
        """Apply the ``inverse:`` block of a manifest — doc 08 §6."""
        ...

    def fit(self, data: MeasurementSet, model: ForwardModel) -> Posterior:
        """Recover ``p(x | y)`` — doc 05 §1.1.

        Returns a posterior and never a point estimate, because doc 05 §1.1 says the
        deliverable "is not a number but a distribution". The point-estimate engines of
        doc 05 §5 (MAP, Laplace) supply the equivalent object rather than a different one,
        which is what lets doc 06 §7's coverage gate be applied uniformly.
        """
        ...

    def diagnostics(self) -> SamplerDiagnostics:
        """The sampler's report on its own trustworthiness — doc 05 §10, gate G-V3."""
        ...

    def identifiability(self, at: PlasmaParams) -> IdentifiabilityReport:
        """The doc 05 §6 analysis at one operating point.

        Part of the engine contract rather than a separate tool because the Fisher matrix
        needs the same Jacobian the engine already computes (doc 05 §6.1, "via autodiff or
        adjoint"); splitting it out would mean building ``F`` twice and risking the two
        disagreeing about the model the answer came from.
        """
        ...
