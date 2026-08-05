"""Per-channel log-likelihoods — doc 05 §3.

doc 05 §3.1 opens with the reason this module is not one Gaussian: "Channels have
genuinely different statistics, and using a Gaussian for all of them would be wrong in a
way that matters." The four rows of that table are implemented here, together with the
detection gate of doc 01 IF-6, the asynchronous sum of doc 05 §3.2, and the heavy-tailed
variants of doc 05 §3.3.

| Channel | Family | Why |
|---|---|---|
| Thomson | Poisson | 0.008 pe/channel/shot (doc 02 §7.1); a Gaussian is invalid here |
| OES | Poisson below a threshold, Gaussian above | faint and bright lines in one frame |
| LIF | Gaussian, heteroscedastic | accumulated signal, signal-dependent variance |
| Interferometry | Gaussian, **correlated** | vibration noise is coloured, not white |
| all | gated to zero below the detection floor | doc 01 IF-6 |

## The correlated covariance is the load-bearing part

doc 05 §3.1 says a diagonal covariance "would overstate the information", and doc 06 §3
says the same thing about a different mechanism: "Treating a shared calibration error as
independent per data point makes it appear to shrink as ``1/√N`` with accumulation. It
does not shrink at all." Both are represented, and they are different matrices:

* :func:`coloured_noise_covariance` — an exponential (AR(1)/Ornstein-Uhlenbeck) kernel in
  time; doc 05 §3.1's vibration term. It cuts the *effective* sample count of ``N`` samples
  spaced ``Δt`` apart to ``[N(1-rho) + 2 rho]/(1+rho)`` with ``rho = exp(-Δt/τ)``.
* :func:`shared_systematic_covariance` — a rank-one ``sigma_cal² μμᵀ``; doc 06 §4's terms 3
  (Thomson Rayleigh calibration, 7 %) and 9 (OES radiometric, 3 %). Its contribution to the
  variance of a mean is independent of ``N``, which is the whole point.

They add. A Thomson channel accumulating 7 000 shots has both — term 8's 3 % photon
statistics on the diagonal, term 3's 7 % calibration in the rank-one part — and only the
first shrinks with exposure, which is doc 06 §4.1's counter-intuitive finding that
"running the experiment longer does almost nothing", made structural rather than
remembered.

## What this module deliberately does not know

Nothing here imports :mod:`vpl.physics` or :mod:`vpl.instruments`. A prediction arrives as
an :class:`~vpl.core.state.Observable`; where it came from is not this layer's concern.
doc 08 §3 requires the separation and doc 05 §7.1 is why: the truth generator and the
inversion must differ in physics level, mesh, timestep, collision set and EEDF form, and a
likelihood that could reach for the forward solver would make that mismatch optional.

## Simplifications, and what each one costs

1. **The detection gate acts on the prediction, and is therefore discontinuous in θ.**
   doc 08 §4 defines ``is_informative(state_guess)``, so the gate is a function of the
   model and not of the data — right, because gating on the observed value would condition
   on the data twice and bias the fit. The price is a step as a prediction crosses the
   floor. **Bound:** the step is the gated sample's log-density, which at the floor is by
   construction an SNR-1 sample — under about 1 nat per gated sample. MAP and HMC (doc 05
   §5) see a kink there; SMC and nested sampling do not care. In doc 01 IF-6's actual
   regime the whole channel gates at once, at the edge of the identifiable region.
2. **The asynchronous sum of doc 05 §3.2 is assembled, not integrated.** The
   ``∫_{W_{k,i}} F_k(x(t)) dt`` is performed by whoever produced the
   :class:`~vpl.core.state.Observable`; this module checks that the prediction's window is
   the window the observation was taken over, then sums. **Bound:** none on the sum. The
   check is exact — a mismatched window raises rather than silently comparing a 2 ns gate
   with a 700 s accumulation — but this module cannot verify the integral *inside* it.
3. **The outlier fraction is a field, not yet an inferred coordinate.** doc 05 §3.3 makes
   it "itself an inferred parameter, and its posterior serves as an automatic data-quality
   flag". :class:`MixtureChannel` holds it as a field, so an engine can add it to the
   parameter vector and rebuild the channel per evaluation. **Bound:** the density is
   exactly the §3.3 mixture at any given fraction, so nothing is lost mathematically; the
   plumbing that makes it a coordinate belongs to the joint parameterisation, not here.
4. **Observations within a channel are conditionally independent.** Correlations live
   *within* one :class:`~vpl.core.state.Observable`, through its covariance, and not
   between records or between channels. **Bound:** exact for the vibration term, which is
   confined to one interferometer trace. Not exact for a shared calibration spanning
   several records of one channel — for that, pass those records as a single record with a
   block covariance, which the types permit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

from vpl.core.params import default_registry
from vpl.core.state import InstrumentId, Measurement, MeasurementSet, Observable

__all__ = [
    "OES_GAUSSIAN_SWITCH_COUNTS",
    "ChannelLikelihood",
    "ChannelMismatchError",
    "CorrelatedGaussianChannel",
    "GaussianChannel",
    "MixtureChannel",
    "PoissonChannel",
    "StudentTChannel",
    "SwitchedPoissonGaussianChannel",
    "coloured_noise_covariance",
    "correlated_gaussian_log_likelihood",
    "detection_mask",
    "diagonal_covariance",
    "gaussian_log_likelihood",
    "outlier_mixture_log_likelihood",
    "poisson_log_likelihood",
    "shared_systematic_covariance",
    "student_t_log_likelihood",
    "switched_poisson_gaussian_log_likelihood",
    "total_log_likelihood",
]

_REGISTRY: Final = default_registry()

#: ``log(2π)``, the Gaussian normalisation. Structure, not measurement.
_LOG_TWO_PI: Final[float] = math.log(2.0 * math.pi)

#: doc 05 §3.1: "Gaussian for bright lines above ~100 pe. Explicit switch at a configured
#: threshold." The threshold is configurable per channel; this is the registered default.
OES_GAUSSIAN_SWITCH_COUNTS: Final[float] = float(
    _REGISTRY.value_in("inverse.oes_gaussian_switch", "dimensionless")
)

#: Relative tolerance for the covariance symmetry check.
#:
#: Not a physical magnitude, and named here because doc 08 §5's rule carves out no
#: exception for numbers whose author was confident they did not matter. Zero would be
#: wrong: a covariance assembled as a sum of outer products is symmetric in exact
#: arithmetic and symmetric to a few ulp in floating point, so an exact test would reject
#: matrices that are correct. A few ulp is what this admits and no more.
_SYMMETRY_RTOL: Final[float] = 1e-12

#: Tolerance within which a float count is accepted as an integer. Counts arrive as
#: ``float64`` (see :mod:`vpl.core.state.measurement`), where every integer up to ``2^53``
#: is exact — so anything failing this is a genuinely fractional value and not a rounding
#: artefact of the storage layer.
_COUNT_TOLERANCE: Final[float] = 0.0


class ChannelMismatchError(ValueError):
    """A prediction that cannot be compared with the observation it is paired with.

    Its own type because doc 05 §3.2 makes this a scientific failure rather than a
    programming one: comparing a 2 ns OES gate with a 700 s Thomson accumulation as though
    they were "the same instant" is, in doc 05's words, "straightforwardly false", and the
    agreement it produced would be meaningless.
    """


# ─── array helpers ───────────────────────────────────────────────────────────────


def _paired(
    observed: ArrayLike, predicted: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce an observation and a prediction, insisting they describe the same samples."""
    y = np.asarray(observed, dtype=float)
    mu = np.asarray(predicted, dtype=float)
    if y.shape != mu.shape:
        raise ValueError(
            f"observation and prediction must have the same shape; got {y.shape} and {mu.shape}"
        )
    return y, mu


def _checked_sigma(sigma: ArrayLike, *, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Coerce a per-sample standard deviation.

    The shape is compared exactly and never broadcast — a ``sigma`` that broadcasts is the
    defect :mod:`vpl.core.state.measurement` names: every sample silently inherits one
    error bar, the fit tightens, and nothing says so.
    """
    s = np.asarray(sigma, dtype=float)
    if s.shape != shape:
        raise ValueError(
            f"the uncertainty must have the same shape as the samples it describes; "
            f"got {s.shape} for samples of shape {shape}"
        )
    if not np.all(s > 0.0):
        raise ValueError(
            "every standard deviation must be positive; a zero uncertainty asserts a "
            "measurement with no error, which no channel in doc 02 has"
        )
    return s


# ─── doc 05 §3.1 — the four families ─────────────────────────────────────────────


def poisson_log_likelihood(counts: ArrayLike, expected: ArrayLike) -> float:
    """`log p(k | λ)` summed over samples — doc 05 §3.1, Thomson.

    ``Σ [ k log λ - λ - log k! ]``, which is ``scipy.stats.poisson.logpmf`` written out.

    doc 02 §7.1 puts Thomson at 0.008 photoelectrons per channel per shot. In that regime
    a Gaussian is not a slightly worse approximation, it is a different distribution: it is
    symmetric where the truth is not, it puts mass below zero where no count exists, and it
    would report an uncertainty on a channel that saw nothing as though the channel had
    measured a zero rather than as though it had learned very little.

    ``λ = 0`` is handled explicitly rather than left to ``log 0``: a channel gated off
    below its detection floor legitimately predicts nothing, and ``p(0 | 0) = 1`` is the
    right answer, not a division by zero.

    Args:
        counts: Observed photoelectron counts — non-negative integers held as floats.
        expected: Expected counts ``λ``, non-negative, same shape.

    Returns:
        The summed log-probability mass.
    """
    k, rate = _paired(counts, expected)

    if np.any(k < 0.0):
        raise ValueError("a photoelectron count cannot be negative")
    if np.any(np.abs(k - np.rint(k)) > _COUNT_TOLERANCE):
        raise ValueError(
            "a Poisson channel needs integer counts; a fractional value means something "
            "was subtracted before the likelihood, and doc 04 §7.2's background belongs "
            "in the model rather than in the data"
        )
    if np.any(rate < 0.0):
        raise ValueError("an expected count rate cannot be negative")

    positive = rate > 0.0
    if not np.all(positive):
        # A zero rate is certain to produce zero counts and cannot produce any.
        impossible = (~positive) & (k > 0.0)
        if np.any(impossible):
            return -math.inf

    total = -float(np.sum(rate)) - float(np.sum(gammaln(k + 1.0)))
    total += float(np.sum(k[positive] * np.log(rate[positive])))
    return total


def gaussian_log_likelihood(observed: ArrayLike, predicted: ArrayLike, sigma: ArrayLike) -> float:
    """`log N(y | μ, diag(sigma²))` summed over samples — doc 05 §3.1, LIF.

    ``-0.5 Σ [ (r/sigma)² + log(2πsigma²) ]``.

    Heteroscedastic by construction: ``sigma`` is per sample and never broadcast. doc 05
    §3.1's justification for LIF is that "the signal is accumulated; variance is
    signal-dependent", so one ``sigma`` would be wrong across a 200-point scan whose wings
    and line centre differ by two decades in rate.

    The ``log sigma`` term is not decoration: dropping it turns the likelihood into a
    weighted least squares whose weights the fit may inflate, and the posterior then becomes
    arbitrarily confident by enlarging its own error bars.
    """
    y, mu = _paired(observed, predicted)
    s = _checked_sigma(sigma, shape=y.shape)

    z = (y - mu) / s
    return float(-0.5 * np.sum(z * z) - np.sum(np.log(s)) - 0.5 * y.size * _LOG_TWO_PI)


def correlated_gaussian_log_likelihood(
    observed: ArrayLike, predicted: ArrayLike, covariance: ArrayLike
) -> float:
    """`log N(y | μ, Σ)` with a full covariance — doc 05 §3.1, interferometry.

    ``-0.5 rᵀΣ⁻¹r - 0.5 [ n log(2π) + log det Σ ]``, through a Cholesky factor rather than
    ``inv``: twice as fast, better behaved, and it yields ``log det Σ = 2 Σ log L_ii`` free.

    doc 05 §3.1 requires this for interferometry — "vibration noise is coloured, not white —
    a diagonal covariance would overstate the information" — and doc 06 §3 requires it again
    for a different mechanism: a shared calibration error is perfectly correlated across
    every sample and "does not shrink at all". See :func:`coloured_noise_covariance` and
    :func:`shared_systematic_covariance`.

    Args:
        observed: Observed samples, shape ``(n,)``.
        predicted: Predicted samples, shape ``(n,)``.
        covariance: Symmetric positive-definite ``(n, n)``.

    Raises:
        ValueError: If the covariance is not square, symmetric, correctly sized and
            positive definite. The alternative — silently regularising it — would report an
            uncertainty nobody chose.
    """
    y, mu = _paired(observed, predicted)
    sigma = np.asarray(covariance, dtype=float)

    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"a covariance must be a square matrix; got shape {sigma.shape}")
    if sigma.shape[0] != y.size:
        raise ValueError(
            f"the covariance is {sigma.shape[0]}x{sigma.shape[0]} but there are {y.size} samples"
        )
    if not np.allclose(sigma, sigma.T, rtol=_SYMMETRY_RTOL, atol=0.0):
        raise ValueError("a covariance must be symmetric")

    try:
        factor = cho_factor(sigma, lower=True, check_finite=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"the covariance is not positive definite ({exc}). A singular covariance means "
            "two samples carry identical information, which is a modelling statement and "
            "not something to regularise away silently."
        ) from exc

    residual = (y - mu).ravel()
    quadratic = float(residual @ cho_solve(factor, residual))
    log_det = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    return -0.5 * quadratic - 0.5 * (residual.size * _LOG_TWO_PI + log_det)


def switched_poisson_gaussian_log_likelihood(
    counts: ArrayLike, expected: ArrayLike, *, threshold: float
) -> float:
    """doc 05 §3.1, OES: Poisson for weak lines, Gaussian for bright ones.

    The switch is **per sample**, on the *expected* count. An OES frame holds faint and
    bright lines together (doc 02 §7), so switching per channel would drag the faint lines
    into a regime where the Gaussian is visibly wrong; switching on the *observed* count
    would be worse still, making the family a function of the data.

    Above the threshold the Gaussian used is ``N(k; λ, λ)`` — the Poisson's own variance,
    not a separately supplied one. That is what makes the two families agree at the switch
    rather than merely both being defined there; the residual difference is the Stirling
    remainder, ``≈ 1/(12λ)``, which at the registered threshold of 100 counts is
    ``8.3e-4`` nats.

    Args:
        counts: Observed counts, non-negative integers.
        expected: Expected counts ``λ``, non-negative.
        threshold: Expected count above which the Gaussian is used. Required, because doc
            05 §3.1 calls it "a configured threshold"; :data:`OES_GAUSSIAN_SWITCH_COUNTS`
            is the registered value.
    """
    k, rate = _paired(counts, expected)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"the Poisson/Gaussian switch threshold must be positive, got {threshold}")

    bright = rate >= threshold
    total = poisson_log_likelihood(k[~bright], rate[~bright]) if np.any(~bright) else 0.0
    if np.any(bright):
        total += gaussian_log_likelihood(k[bright], rate[bright], np.sqrt(rate[bright]))
    return total


# ─── doc 05 §3.3 — robustness to outliers ────────────────────────────────────────


def student_t_log_likelihood(
    observed: ArrayLike, predicted: ArrayLike, sigma: ArrayLike, *, dof: float
) -> float:
    """Heavy-tailed Gaussian replacement — doc 05 §3.3.

    doc 05 §3.3: "Discrete failures — cosmic rays, fringe jumps, MCP ion feedback (doc 04
    §7.2) — produce occasional wildly wrong data points, and a Gaussian likelihood will
    contort the entire fit to accommodate one of them."

    The mechanism is the reason this is not cosmetic. A Gaussian's log-density falls as
    ``-z²/2``, so one 40-sigma point costs 800 nats and the fit will move the *entire* model
    to reduce it. A Student-t's falls logarithmically, as ``-((nu+1)/2) log(1 + z²/nu)``, so
    the same point costs about 15 nats and the fit ignores it — which is what a human would
    do on seeing a cosmic ray. ``nu → ∞`` recovers the Gaussian exactly, so ``dof`` is a
    continuous dial rather than a mode switch.
    """
    y, mu = _paired(observed, predicted)
    s = _checked_sigma(sigma, shape=y.shape)
    if not math.isfinite(dof) or dof <= 0.0:
        raise ValueError(f"the degrees of freedom must be positive, got {dof}")

    z = (y - mu) / s
    normaliser = (
        float(gammaln(0.5 * (dof + 1.0)))
        - float(gammaln(0.5 * dof))
        - 0.5 * math.log(dof * math.pi)
    )
    kernel = -0.5 * (dof + 1.0) * np.log1p(z * z / dof)
    return float(y.size * normaliser - np.sum(np.log(s)) + np.sum(kernel))


def outlier_mixture_log_likelihood(
    observed: ArrayLike,
    predicted: ArrayLike,
    sigma: ArrayLike,
    *,
    outlier_fraction: float,
    outlier_scale: float,
) -> float:
    """``(1-ε) N(r; 0, sigma²) + ε N(r; 0, (csigma)²)`` — doc 05 §3.3's explicit mixture.

    The alternative to the Student-t, and the one doc 05 §3.3 prefers operationally: "The
    outlier fraction is itself an inferred parameter, and its posterior serves as an
    **automatic data-quality flag**." A Student-t's ``nu`` has no such reading — a posterior
    on ``ε`` says "3 % of these points are bad", which an operator can act on.

    Evaluated through ``logaddexp`` so that a point deep in the outlier component does not
    underflow the inlier term to zero and take the whole sum to ``-inf``.

    Args:
        outlier_fraction: ``ε`` in ``[0, 1]``.
        outlier_scale: ``c ≥ 1``, the factor by which the outlier component is broader.
    """
    y, mu = _paired(observed, predicted)
    s = _checked_sigma(sigma, shape=y.shape)
    if not 0.0 <= outlier_fraction <= 1.0:
        raise ValueError(f"the outlier fraction must lie in [0, 1], got {outlier_fraction}")
    if not math.isfinite(outlier_scale) or outlier_scale < 1.0:
        raise ValueError(
            f"the outlier component must be broad (scale >= 1), got {outlier_scale}; a "
            "narrow second component turns the mixture into a way of over-fitting"
        )

    residual = y - mu
    inlier = _component_log_density(residual, s) + _log_weight(1.0 - outlier_fraction)
    outlier = _component_log_density(residual, s * outlier_scale) + _log_weight(outlier_fraction)
    return float(np.sum(np.logaddexp(inlier, outlier)))


def _component_log_density(
    residual: NDArray[np.float64], scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-sample ``log N(r; 0, scale²)``."""
    z = residual / scale
    return -0.5 * z * z - np.log(scale) - 0.5 * _LOG_TWO_PI


def _log_weight(weight: float) -> float:
    """``log w``, with ``log 0 = -inf`` rather than a warning."""
    return math.log(weight) if weight > 0.0 else -math.inf


# ─── covariance construction — doc 05 §3.1, doc 06 §3 and §4 ─────────────────────


def diagonal_covariance(sigma: ArrayLike) -> NDArray[np.float64]:
    """``diag(sigma²)`` — the uncorrelated term of doc 06 §4 (photon statistics, term 8)."""
    s = np.asarray(sigma, dtype=float).ravel()
    if not np.all(s > 0.0):
        raise ValueError("every standard deviation must be positive")
    return np.diag(s * s)


def coloured_noise_covariance(
    times: ArrayLike, *, sigma: float, correlation_time: float
) -> NDArray[np.float64]:
    """``Σ_ij = sigma² exp(-|t_i - t_j| / τ)`` — doc 05 §3.1's coloured vibration noise.

    The exponential (Ornstein-Uhlenbeck, or AR(1) when sampled uniformly) kernel: the
    simplest stationary covariance with a finite correlation time, and the one implied by
    treating mechanical path-length drift (doc 01 §4, "mechanical path-length vibration")
    as a damped random walk. It is always positive definite for ``τ > 0``, which a
    hand-assembled correlation matrix generally is not.

    The consequence doc 05 §3.1 cares about is quantitative. For ``N`` samples spaced ``Δt``
    with ``rho = exp(-Δt/τ)``, the Fisher information about a constant offset is
    ``[N(1-rho) + 2 rho]/(1+rho)``, not ``N``: at ``Δt = τ/5`` and ``N = 32`` that is 4.09
    rather than 32. An interferometer trace treated as white noise would report an error bar
    2.8x too small — over-confident in exactly the way doc 06 §7.1 calls "the dangerous
    failure".

    Args:
        times: Sample times. Need not be sorted or uniformly spaced.
        sigma: Marginal standard deviation, the same for every sample.
        correlation_time: ``τ``, in the same units as ``times``.
    """
    t = np.asarray(times, dtype=float).ravel()
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"the marginal standard deviation must be positive, got {sigma}")
    if not math.isfinite(correlation_time) or correlation_time <= 0.0:
        raise ValueError(f"the correlation time must be positive, got {correlation_time}")

    lag = np.abs(t[:, None] - t[None, :])
    kernel: NDArray[np.float64] = (sigma * sigma) * np.exp(-lag / correlation_time)
    return kernel


def shared_systematic_covariance(
    predicted: ArrayLike, *, relative_sigma: float
) -> NDArray[np.float64]:
    """``Σ = sigma_rel² μ μᵀ`` — a calibration error common to every sample, doc 06 §3.

    Rank one, and that is the entire content: one scale error multiplies every sample by the
    same unknown factor. doc 06 §3 names this "the one most often botched", and the symptom
    is specific — the reported uncertainty on an accumulated quantity falls as ``1/√N`` when
    the true one does not fall at all.

    doc 06 §4 lists two instances: term 3, the Thomson Rayleigh calibration at 7 %, and
    term 9, the OES radiometric calibration at 3 %. Between them they are 8.5 % of a 17.1 %
    budget — a quarter of the total variance, entirely in this matrix.

    Add it to a statistical term rather than using it alone: on its own it is singular, which
    :func:`correlated_gaussian_log_likelihood` correctly refuses.

    Args:
        predicted: The predicted signal ``μ``. The error is *multiplicative*, so it scales
            with the prediction and not with the observation — using the observation would
            make the covariance a function of the data.
        relative_sigma: One-sigma fractional calibration uncertainty.
    """
    mu = np.asarray(predicted, dtype=float).ravel()
    if not math.isfinite(relative_sigma) or relative_sigma <= 0.0:
        raise ValueError(
            f"the relative calibration uncertainty must be positive, got {relative_sigma}"
        )
    return (relative_sigma * relative_sigma) * np.outer(mu, mu)


# ─── doc 01 IF-6 — the detection gate ────────────────────────────────────────────


def detection_mask(predicted: ArrayLike, *, floor: float) -> NDArray[np.bool_]:
    """Which samples are above the channel's detection floor — doc 01 IF-6.

    doc 01 IF-6 is emphatic that this is "a requirement, not a caveat": the interferometry
    channel is blind below ``n_e = 3.3e16 m^-3`` and "must be modelled as absent there".
    Ingesting noise as a weak measurement instead "will produce confident nonsense at low
    density".

    The comparison is against the **prediction**, per doc 08 §4's
    ``is_informative(state_guess)``. Gating on the observation would make the likelihood's
    *form* depend on the data, so a downward fluctuation would drop a term and the fit
    would quietly prefer states that produce them.
    """
    mu = np.asarray(predicted, dtype=float)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError(f"a detection floor cannot be negative or non-finite, got {floor}")
    return np.asarray(mu >= floor)


# ─── the channels ────────────────────────────────────────────────────────────────


@runtime_checkable
class ChannelLikelihood(Protocol):
    """One row of doc 05 §3.1: how a channel's observation is scored against a prediction.

    Deliberately narrow: it takes an observation and a prediction and returns a number, and
    cannot see the plasma state, the forward solver or the other channels. doc 08 §1
    principle 5 ("layers do not leak") applied where it matters most — a likelihood that
    could re-evaluate the forward model could commit doc 05 §7's inverse crime unnoticed.
    """

    def log_prob(self, observed: Measurement, predicted: Observable) -> float:
        """The channel's contribution to ``log p(y | x)``."""
        ...


def _gated(
    channel_floor: float | None, observed: Measurement, predicted: Observable
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply the doc 01 IF-6 gate, returning the surviving samples.

    Gated samples are *removed*, not down-weighted. doc 01 IF-6 says "modelled as absent",
    and a term with a large variance is not absent — it still pulls, and its ``log sigma``
    still contributes.
    """
    values = np.asarray(observed.values, dtype=float)
    uncertainty = np.asarray(observed.uncertainty, dtype=float)
    mu = np.asarray(predicted.values, dtype=float)

    if channel_floor is None:
        return values, uncertainty, mu

    keep = detection_mask(mu, floor=channel_floor)
    return values[keep], uncertainty[keep], mu[keep]


def _checked_floor(floor: float | None) -> float | None:
    if floor is None:
        return None
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError(f"a detection floor cannot be negative or non-finite, got {floor}")
    return floor


@dataclass(frozen=True, slots=True)
class PoissonChannel:
    """Thomson — doc 05 §3.1.

    The observation's ``uncertainty`` field is deliberately unused: a Poisson channel's
    uncertainty *is* its rate. Using both would count the photon statistics twice and
    halve the reported error bar for no stated reason.
    """

    detection_floor: float | None = None

    def __post_init__(self) -> None:
        _checked_floor(self.detection_floor)

    def log_prob(self, observed: Measurement, predicted: Observable) -> float:
        counts, _, rate = _gated(self.detection_floor, observed, predicted)
        if counts.size == 0:
            return 0.0
        return poisson_log_likelihood(counts, rate)


@dataclass(frozen=True, slots=True)
class GaussianChannel:
    """LIF — doc 05 §3.1, Gaussian with heteroscedastic variance."""

    detection_floor: float | None = None

    def __post_init__(self) -> None:
        _checked_floor(self.detection_floor)

    def log_prob(self, observed: Measurement, predicted: Observable) -> float:
        values, uncertainty, mu = _gated(self.detection_floor, observed, predicted)
        if values.size == 0:
            return 0.0
        return gaussian_log_likelihood(values, mu, uncertainty)


@dataclass(frozen=True, slots=True)
class CorrelatedGaussianChannel:
    """Interferometry — doc 05 §3.1, Gaussian with a correlated noise covariance.

    The covariance is a field rather than derived from the observation's per-sample
    uncertainty, because the correlation structure is a property of the *apparatus* — the
    table's vibration spectrum, the calibration chain — and not of any one frame. Build it
    from :func:`coloured_noise_covariance` and :func:`shared_systematic_covariance`.

    The detection floor is applied to the covariance as well, by taking the surviving
    sub-block. A sub-block of a positive-definite matrix is positive definite, so gating
    cannot make the covariance invalid.
    """

    covariance: NDArray[np.float64]
    detection_floor: float | None = None

    def __post_init__(self) -> None:
        _checked_floor(self.detection_floor)

    def log_prob(self, observed: Measurement, predicted: Observable) -> float:
        values = np.asarray(observed.values, dtype=float)
        mu = np.asarray(predicted.values, dtype=float)
        if self.covariance.shape != (values.size, values.size):
            raise ChannelMismatchError(
                f"the channel covariance is {self.covariance.shape} but "
                f"{observed.instrument_id!r} reported {values.size} samples"
            )

        keep = (
            detection_mask(mu, floor=self.detection_floor)
            if self.detection_floor is not None
            else np.ones(values.size, dtype=bool)
        )
        if not np.any(keep):
            return 0.0
        return correlated_gaussian_log_likelihood(
            values[keep], mu[keep], self.covariance[np.ix_(keep, keep)]
        )


@dataclass(frozen=True, slots=True)
class SwitchedPoissonGaussianChannel:
    """OES — doc 05 §3.1, Poisson below the threshold and Gaussian above it."""

    threshold: float = OES_GAUSSIAN_SWITCH_COUNTS
    detection_floor: float | None = None

    def __post_init__(self) -> None:
        _checked_floor(self.detection_floor)

    def log_prob(self, observed: Measurement, predicted: Observable) -> float:
        counts, _, rate = _gated(self.detection_floor, observed, predicted)
        if counts.size == 0:
            return 0.0
        return switched_poisson_gaussian_log_likelihood(counts, rate, threshold=self.threshold)


@dataclass(frozen=True, slots=True)
class StudentTChannel:
    """The heavy-tailed variant of doc 05 §3.3, for any Gaussian channel."""

    dof: float
    detection_floor: float | None = None

    def __post_init__(self) -> None:
        _checked_floor(self.detection_floor)

    def log_prob(self, observed: Measurement, predicted: Observable) -> float:
        values, uncertainty, mu = _gated(self.detection_floor, observed, predicted)
        if values.size == 0:
            return 0.0
        return student_t_log_likelihood(values, mu, uncertainty, dof=self.dof)


@dataclass(frozen=True, slots=True)
class MixtureChannel:
    """The explicit outlier mixture of doc 05 §3.3.

    ``outlier_fraction`` is a field so that an engine inferring it (as doc 05 §3.3
    requires) rebuilds the channel per evaluation. That is cheap — a frozen dataclass of two
    floats — and keeps the fraction visible in the object that uses it.
    """

    outlier_fraction: float
    outlier_scale: float
    detection_floor: float | None = None

    def __post_init__(self) -> None:
        _checked_floor(self.detection_floor)

    def log_prob(self, observed: Measurement, predicted: Observable) -> float:
        values, uncertainty, mu = _gated(self.detection_floor, observed, predicted)
        if values.size == 0:
            return 0.0
        return outlier_mixture_log_likelihood(
            values,
            mu,
            uncertainty,
            outlier_fraction=self.outlier_fraction,
            outlier_scale=self.outlier_scale,
        )


# ─── doc 05 §3.2 — the asynchronous sum ──────────────────────────────────────────


def total_log_likelihood(
    measurements: MeasurementSet,
    predictions: Mapping[InstrumentId, Sequence[Observable]],
    channels: Mapping[InstrumentId, ChannelLikelihood],
) -> float:
    """`log p(y | x)` over every channel and every observation — doc 05 §3.2.

        ``log p(y | x) = Σ_channels Σ_observations log p( y_{k,i} | ∫_{W_{k,i}} F_k(x(t)) dt )``

    doc 05 §3.2 calls getting this right "what makes multi-diagnostic fusion legitimate",
    and the reason is the window: the four channels never measure simultaneously (doc 02
    §10.1), so each observation is scored against a prediction integrated over *its own*
    window. This function checks that pairing and refuses to proceed without it — a silent
    mismatch would make cross-channel agreement meaningless, which is worse than an error
    because it looks like a result.

    The sum runs in the deterministic order :class:`MeasurementSet` imposes, so the
    floating-point total is bit-for-bit reproducible (doc 00 E3): addition is not
    associative, and a total depending on manifest traversal order would break
    ``vpl reproduce``.

    Args:
        measurements: The observations, from any mix of channels.
        predictions: Per channel, the noiseless predictions, in the same order as
            ``measurements.by_instrument(channel)``.
        channels: Per channel, the doc 05 §3.1 likelihood family to use.

    Raises:
        ChannelMismatchError: If a channel has observations but no model or no prediction,
            if the counts differ, or if a prediction's instrument, units or acquisition
            window does not match the observation it is paired with.
    """
    total = 0.0
    for instrument_id, records in measurements.grouped_by_instrument().items():
        if instrument_id not in channels:
            raise ChannelMismatchError(
                f"{instrument_id!r} has {len(records)} observation(s) but no likelihood "
                f"model; doc 02 §13 removes a channel by removing its data, not by leaving "
                f"the data unscored"
            )
        if instrument_id not in predictions:
            raise ChannelMismatchError(
                f"{instrument_id!r} has {len(records)} observation(s) but no prediction"
            )

        channel_predictions = predictions[instrument_id]
        if len(channel_predictions) != len(records):
            raise ChannelMismatchError(
                f"{instrument_id!r} has {len(records)} observation(s) but "
                f"{len(channel_predictions)} prediction(s)"
            )

        channel = channels[instrument_id]
        for observed, predicted in zip(records, channel_predictions, strict=True):
            _check_comparable(observed, predicted)
            total += channel.log_prob(observed, predicted)

    return total


def _check_comparable(observed: Measurement, predicted: Observable) -> None:
    """Refuse a pairing doc 05 §3.2 says is not a comparison."""
    if observed.instrument_id != predicted.instrument_id:
        raise ChannelMismatchError(
            f"instrument mismatch: observation from {observed.instrument_id!r} paired with "
            f"a prediction for {predicted.instrument_id!r}"
        )
    if observed.units != predicted.units:
        raise ChannelMismatchError(
            f"{observed.instrument_id!r}: units mismatch — observation in "
            f"{observed.units!r}, prediction in {predicted.units!r}"
        )
    if observed.window != predicted.window:
        raise ChannelMismatchError(
            f"{observed.instrument_id!r}: acquisition window mismatch — the observation "
            f"integrates {observed.window!r} and the prediction integrates "
            f"{predicted.window!r}. doc 05 §3.2: treating these as the same instant would "
            f"be straightforwardly false."
        )
    if observed.shape != predicted.shape:
        raise ChannelMismatchError(
            f"{observed.instrument_id!r}: the observation has shape {observed.shape} and "
            f"the prediction has shape {predicted.shape}"
        )
