"""Priors on the Level A control parameters — doc 05 §2.1 and §4.

doc 05 §4.1 opens by saying what the priors here are *not*: "The strongest physics
constraint is not a prior term at all — it is that `F` contains the sheath physics." This
module holds the rest — the §2.1 table, and the §4.2 soft penalties that anchor conditions
the discrepancy field could otherwise violate.

## The change of variables, and why it has its own test

Every density in the §2.1 table is stated in the parameter's own units. Every inference
engine in doc 05 §5 works in the unconstrained space of :mod:`vpl.inverse.parameters`.
Moving a density between them is not optional and not free:

    ``log p_u(u) = log p_x(x(u)) + log |det ∂x/∂u|``

Omit the second term and nothing breaks. The code runs, the log-density is finite, the
optimiser converges, the sampler mixes, and the answer is wrong — biased towards whichever
end of each transform the map compresses. For the log transforms here that is a systematic
pull towards small ``n_0``, ``T_e``, ``T_i`` and ``p``, of order one prior standard
deviation each, which for ``n_0`` is most of a decade. This is the error the brief singles
out, and :meth:`ControlPrior.log_prob_unconstrained` is the one function that fixes it.

## Why the closed forms are written out rather than delegated to `scipy.stats`

Two reasons, one practical and one methodological. Practically, ``scipy.stats`` frozen
distributions carry per-call argument validation that dominates the cost of a density
evaluated ``10^6`` times (doc 05 §5). Methodologically, doc 05's whole discipline is that
a result checked only against itself is not checked — ADR-011 is the local instance of
that lesson. Writing the density from ``scipy.special`` primitives leaves ``scipy.stats``
free to be an *independent* authority in the tests, which is where it is used and where
the parameterisation errors (``lognorm``'s ``scale`` is a median; ``truncnorm``'s bounds
are standardised) would otherwise hide.

## Simplifications, and what each one costs

1. **The joint prior is a product of independent marginals.** doc 05 §2.1 states eight
   univariate priors and no correlations, so this is faithful to the document — but it is
   not faithful to the apparatus. The gauge pressure and the gas temperature share a
   calibration; ``T_e`` and ``n_0`` are correlated *a posteriori* by construction (doc 05
   §6.2). **Bound:** the prior correlation neglected is that of the gauge chain of doc 06
   §5, which contributes 2 % of a 17.1 % budget (doc 06 §4) — under 1.4 % of the total
   variance, and it enters as a widening rather than a shift.
2. **`gamma_se` is truncated rather than normal.** doc 05 §2.1 says "normal, 0.10 ± 0.03";
   a normal puts mass on negative yields, which the forward model cannot evaluate and
   which is not physical. The prior here is the same normal restricted to the registered
   sweep range of ``sheath.gamma_se_W``, ``[0, 0.3]``. **Bound:** the retained mass is
   ``1 - Φ(-10/3) - Φ(-20/3) = 0.99957094``, so the restriction raises every log-density
   by ``+4.2907e-4`` uniformly and changes no density *ratio* at all. It is not an
   approximation of the document; it is the document's density made proper on the support
   the parameter actually has.
3. **The §4.2 penalty weights are arguments, not defaults.** doc 05 §4.3 is explicit that
   "the regularisation weight is not hand-tuned" and specifies three ways to choose it
   (hierarchical Bayes, L-curve, GCV). Supplying a default here would quietly become a
   fourth. **Bound:** none on correctness; the cost is that a caller must decide, which is
   the intent.
4. **The §4.2 penalties are quadratic hinges, not calibrated log-densities.** They are
   improper — they do not integrate to one over any parameterisation, because the quantity
   they constrain is a functional of the whole state rather than a coordinate.
   **Bound:** they may be used for MAP and for HMC (where only ratios matter and the
   normaliser is constant in ``θ``). They may **not** be used inside an evidence
   computation for the model comparison of doc 05 §5, because the missing normaliser does
   not cancel between models with different penalty weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, ndtr, ndtri

from vpl.core.params import default_registry
from vpl.inverse.parameters import (
    CONTROL_PARAMETERS,
    N_CONTROL,
    TWO_PI,
    ControlParameters,
    log_abs_det_jacobian,
)

__all__ = [
    "ControlPrior",
    "LogNormalPrior",
    "LogUniformPrior",
    "NormalPrior",
    "Prior",
    "TruncatedNormalPrior",
    "UniformPrior",
    "bohm_penalty",
    "default_control_prior",
    "quasineutrality_penalty",
    "smoothness_log_prior",
]

_REGISTRY: Final = default_registry()

#: Relative tolerance for comparing a prior's support with its parameter's.
#:
#: Both come from the same registry entry by different routes, so they agree exactly
#: unless one of them was recomputed — but `2 * pi` reached by two paths need not be the
#: same float, and rejecting a prior for that would be absurd. A few ulp, no more.
_SUPPORT_RTOL: Final[float] = 1e-12

#: ``log(2π)``, the Gaussian normalisation. Structure, not measurement.
_LOG_TWO_PI: Final[float] = math.log(TWO_PI)


# ─── the prior interface ─────────────────────────────────────────────────────────


@runtime_checkable
class Prior(Protocol):
    """A normalised univariate log-density on a stated support.

    ``log_prob`` returns ``-inf`` outside :attr:`support` rather than raising. That is not
    laziness about error handling: a MAP run (doc 05 §5) evaluates the objective at
    trial points a line search proposes, and ``-inf`` is the answer that makes it
    backtrack, where an exception is the answer that makes it stop.
    """

    def log_prob(self, value: float) -> float:
        """Natural log of the density at ``value``; ``-inf`` outside the support."""
        ...

    @property
    def support(self) -> tuple[float, float]:
        """Closed interval outside which the density is zero."""
        ...

    @property
    def median(self) -> float:
        """The 50th percentile — a point that is always strictly inside the support.

        Present on the interface because doc 05 §5 initialises MAP before anything else
        runs, and the obvious initialisation — the RP-1 nominal — is not always usable:
        ``eedf.kappa``'s registered nominal is 1.0, which is the *edge* of doc 05 §2.1's
        ``uniform [1, 5]``, and the edge maps to ``-inf`` under the logit. A prior median
        cannot do that to a caller.
        """
        ...


def _check_positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scale, got {value}")
    return value


def _check_ascending(low: float, high: float) -> None:
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"a prior interval must be finite, got [{low}, {high}]")
    if not low < high:
        raise ValueError(f"a prior interval must be ascending, got [{low}, {high}]")


@dataclass(frozen=True, slots=True)
class NormalPrior:
    """``N(mean, sigma²)`` — doc 05 §2.1's `normal` row, and §4.2's instrument priors.

    Used for ``V_w``, where doc 05 §2.1 says "informative — supply-measured, sigma = 1 %". The
    quantity is signed (RP-1 biases the electrode to -250 V), so a relative uncertainty
    means one percent of the magnitude and the prior stays on the real line.
    """

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        _check_positive("sigma", self.sigma)

    @property
    def support(self) -> tuple[float, float]:
        return (-math.inf, math.inf)

    @property
    def median(self) -> float:
        return self.mean

    def log_prob(self, value: float) -> float:
        z = (value - self.mean) / self.sigma
        return -0.5 * z * z - math.log(self.sigma) - 0.5 * _LOG_TWO_PI


@dataclass(frozen=True, slots=True)
class UniformPrior:
    """``U[low, high]`` — doc 05 §2.1's `uniform` rows: ``phi_RF`` and ``kappa``.

    Uniform is a real statement, not an absence of one: for ``kappa`` it says every EEDF
    shape between Maxwellian and the far side of Druyvesteyn is equally credible before the
    data, which is the honest position given doc 03 §3.2's "the EEDF is typically
    bi-Maxwellian or Druyvesteyn" and no measurement to break the tie.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        _check_ascending(self.low, self.high)

    @property
    def support(self) -> tuple[float, float]:
        return (self.low, self.high)

    @property
    def median(self) -> float:
        return 0.5 * (self.low + self.high)

    def log_prob(self, value: float) -> float:
        if not self.low <= value <= self.high:
            return -math.inf
        return -math.log(self.high - self.low)


@dataclass(frozen=True, slots=True)
class LogUniformPrior:
    """``p(x) ∝ 1/x`` on ``[low, high]`` — doc 05 §2.1's `log-uniform` rows.

    The scale-invariant prior, and the correct one for ``n_0`` and ``T_i``: doc 01 §2.4's
    envelope spans four decades of density, and a uniform prior over that range would put
    99 % of its mass in the top decade before a single photon was collected.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        _check_positive("the lower bound of a log-uniform prior", self.low)
        _check_ascending(self.low, self.high)

    @property
    def support(self) -> tuple[float, float]:
        return (self.low, self.high)

    @property
    def median(self) -> float:
        """The *geometric* mean. A log-uniform prior is uniform in the log, so its
        median is the midpoint there — which is `sqrt(low * high)` and not `(low+high)/2`.
        For `n_0` the two differ by a factor of 50."""
        return math.sqrt(self.low * self.high)

    def log_prob(self, value: float) -> float:
        if not self.low <= value <= self.high:
            return -math.inf
        return -math.log(value) - math.log(math.log(self.high / self.low))


@dataclass(frozen=True, slots=True)
class LogNormalPrior:
    """Log-normal with the given median and log-scale width.

    Parameterised by the **median** rather than by the mean of the log, because that is
    what doc 05 §2.1's "log-normal, μ = 3 eV, sigma = 0.4" means when read against RP-1: the
    3 eV is the registered ``RP1.T_e``, a temperature, and the 0.4 is dimensionless. It is
    also the parameterisation ``scipy.stats.lognorm`` uses under the name ``scale``, which
    is worth stating because supplying ``log(3)`` there instead is a silent factor of
    ``e^3`` and a very believable posterior.

    Also carries the gauge pressure of doc 05 §2.1, where the sigma is relative. A relative
    uncertainty applied on the log scale is a genuine relative uncertainty at every value
    and cannot put probability mass on a negative pressure, which an additive Gaussian
    with a 2 % width would only avoid by being narrow.

    Attributes:
        median: ``exp`` of the mean of the log. The 50th percentile.
        log_sigma: Standard deviation of the log. For small values this is very nearly the
            fractional uncertainty — 2 % is 2.0 % to one part in 300.
    """

    median: float
    log_sigma: float

    def __post_init__(self) -> None:
        _check_positive("median", self.median)
        _check_positive("log_sigma", self.log_sigma)

    # `median` is a field rather than a property here — it is the parameterisation, so
    # there is nothing to derive. That satisfies the `Prior` protocol's read-only
    # `median` member directly, which is the point of parameterising by the median.

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, math.inf)

    def log_prob(self, value: float) -> float:
        if value <= 0.0:
            return -math.inf
        z = (math.log(value) - math.log(self.median)) / self.log_sigma
        return -0.5 * z * z - math.log(value * self.log_sigma) - 0.5 * _LOG_TWO_PI


@dataclass(frozen=True, slots=True)
class TruncatedNormalPrior:
    """``N(mean, sigma²)`` restricted to ``[low, high]`` and renormalised.

    doc 05 §2.1 gives ``gamma_se`` as "normal, 0.10 ± 0.03". A yield is bounded below by
    zero, and the registry bounds ``sheath.gamma_se_W`` above at 0.3, so the density is
    made proper on that interval rather than left to put mass where the forward model
    cannot be evaluated.

    The normalising constant is ``Φ(β) - Φ(alpha)`` with standardised bounds. Computing it as
    a difference of two CDFs loses all precision when both bounds are in the same tail —
    two numbers within ``1e-16`` of each other, subtracted. The branch below evaluates it
    in log space from whichever side is small, which is exact everywhere.
    """

    mean: float
    sigma: float
    low: float
    high: float

    def __post_init__(self) -> None:
        _check_positive("sigma", self.sigma)
        _check_ascending(self.low, self.high)

    @property
    def support(self) -> tuple[float, float]:
        return (self.low, self.high)

    @property
    def median(self) -> float:
        """``F⁻¹(1/2)`` of the truncated density, by inverting the parent CDF.

        Not the parent's mean: truncation is asymmetric in general, and for the
        ``gamma_se`` case (a bound 3.3sigma below the mean and 6.7sigma above it) the median sits
        16 ppm above 0.10. Small here, but the expression is the same one that would be
        badly wrong for a prior truncated close to its mode.
        """
        alpha = float(ndtr((self.low - self.mean) / self.sigma))
        beta = float(ndtr((self.high - self.mean) / self.sigma))
        return self.mean + self.sigma * float(ndtri(alpha + 0.5 * (beta - alpha)))

    @property
    def log_retained_mass(self) -> float:
        """``log(Φ(β) - Φ(alpha))`` — how much of the parent normal survives truncation."""
        alpha = (self.low - self.mean) / self.sigma
        beta = (self.high - self.mean) / self.sigma
        # Work from the tail that is *smaller*, so the `log1p(-exp(delta))` below is a
        # correction rather than a catastrophic cancellation. Mirroring for alpha > 0 uses
        # the symmetry Phi(b) - Phi(a) = Phi(-a) - Phi(-b).
        if alpha > 0.0:
            large, small = float(log_ndtr(-alpha)), float(log_ndtr(-beta))
        else:
            large, small = float(log_ndtr(beta)), float(log_ndtr(alpha))
        return large + math.log1p(-math.exp(small - large))

    def log_prob(self, value: float) -> float:
        if not self.low <= value <= self.high:
            return -math.inf
        z = (value - self.mean) / self.sigma
        parent = -0.5 * z * z - math.log(self.sigma) - 0.5 * _LOG_TWO_PI
        return parent - self.log_retained_mass


# ─── the joint prior on theta_c ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ControlPrior:
    """`p(θ_c)` — the eight marginals of doc 05 §2.1, in table order.

    Attributes:
        marginals: One :class:`Prior` per :data:`CONTROL_PARAMETERS` entry, in the same
            order. Independence is doc 05 §2.1 as written; see the module docstring for
            what that neglects and by how much.
    """

    marginals: tuple[Prior, ...]

    def __post_init__(self) -> None:
        if len(self.marginals) != N_CONTROL:
            raise ValueError(
                f"doc 05 §2.1 has {N_CONTROL} Level A control parameters; "
                f"got {len(self.marginals)} marginals"
            )
        self._check_supports_agree()

    def _check_supports_agree(self) -> None:
        """Reject a marginal whose support differs from the parameter's own.

        A mismatch is not a small error. If the prior's support is wider than the
        parameter's, the joint density does not integrate to one over the space the vector
        can represent — and evidence-based model comparison (doc 05 §5) silently compares
        differently-normalised quantities. If it is narrower, part of the parameter space
        is unreachable for a reason nothing in the output records.
        """
        for spec, marginal in zip(CONTROL_PARAMETERS, self.marginals, strict=True):
            if spec.support is None:
                continue
            low, high = marginal.support
            if not (
                math.isclose(low, spec.support[0], rel_tol=_SUPPORT_RTOL, abs_tol=0.0)
                and math.isclose(high, spec.support[1], rel_tol=_SUPPORT_RTOL, abs_tol=0.0)
            ):
                raise ValueError(
                    f"the prior on {spec.name} has support [{low:g}, {high:g}] but the "
                    f"parameter has support [{spec.support[0]:g}, {spec.support[1]:g}]"
                )

    def marginal(self, name: str) -> Prior:
        """The prior on one named parameter."""
        for spec, prior in zip(CONTROL_PARAMETERS, self.marginals, strict=True):
            if spec.name == name:
                return prior
        raise KeyError(f"{name!r} is not a doc 05 §2.1 Level A control parameter")

    def median(self) -> ControlParameters:
        """The componentwise prior median — a usable, interior initialisation.

        doc 05 §5 makes MAP the initialiser for everything else, so something has to say
        where MAP starts. RP-1 (:meth:`ControlParameters.reference`) is the physically
        meaningful point but is *not* always a legal starting point: ``eedf.kappa``'s
        registered nominal of 1.0 sits exactly on the lower edge of doc 05 §2.1's
        ``uniform [1, 5]``, so its unconstrained image is ``-inf`` and any optimiser
        started there is started nowhere. Every marginal median is strictly interior by
        construction, so this always maps to a finite point in ``R^8``.

        It is not the posterior mode and makes no claim to be. It is the point that
        carries no information beyond the prior, which is the correct place to begin.
        """
        return ControlParameters(
            **{
                spec.name: prior.median
                for spec, prior in zip(CONTROL_PARAMETERS, self.marginals, strict=True)
            }
        )

    def log_prob(self, theta: ControlParameters) -> float:
        """`log p(θ_c)` in the parameters' own units."""
        return float(
            sum(
                prior.log_prob(float(getattr(theta, spec.name)))
                for spec, prior in zip(CONTROL_PARAMETERS, self.marginals, strict=True)
            )
        )

    def log_prob_unconstrained(self, u: ArrayLike) -> float:
        """`log p_u(u)` — the prior as a density on the space an engine works in.

        This is :meth:`log_prob` at ``x(u)`` **plus the log-Jacobian**, and the second term
        is the whole point of the method. See the module docstring for what dropping it
        costs.

        Returns ``-inf`` rather than raising when ``u`` maps outside a bounded support: an
        optimiser must be able to step there and be told to come back.
        """
        vector = np.asarray(u, dtype=float).ravel()
        if vector.size != N_CONTROL:
            raise ValueError(
                f"doc 05 §2.1 has {N_CONTROL} Level A control parameters; got {vector.size}"
            )

        total = 0.0
        for spec, prior, value in zip(CONTROL_PARAMETERS, self.marginals, vector, strict=True):
            total += prior.log_prob(spec.transform.to_constrained(float(value)))
            if total == -math.inf:
                # Short-circuit before the Jacobian: at |u| = inf the Jacobian is also
                # -inf, and -inf + -inf is fine, but -inf + (+inf) is nan and a nan
                # objective silently poisons an L-BFGS-B line search.
                return -math.inf

        return total + log_abs_det_jacobian(vector)


def default_control_prior(
    *,
    gauge_pressure: float | None = None,
    supply_bias: float | None = None,
) -> ControlPrior:
    """doc 05 §2.1's table, with every number read from the parameter registry.

    Two of the eight priors are *informative* in the sense of doc 05 §2.1 — they are
    centred on something an operator measured rather than on a design value. Those are
    arguments here, defaulting to RP-1 so that a test or a design study need not invent a
    reading.

    Args:
        gauge_pressure: The pressure-gauge reading, in mTorr. The prior is log-normal
            about it with the registered relative accuracy of the gauge.
        supply_bias: The supply's reported electrode potential, in volts, signed. The
            prior is normal about it with the registered relative accuracy of the supply.

    Returns:
        A :class:`ControlPrior` whose marginals are, in doc 05 §2.1 table order,
        log-uniform, log-normal, log-uniform, log-normal, normal, uniform,
        truncated-normal and uniform.
    """
    pressure = (
        gauge_pressure
        if gauge_pressure is not None
        else float(_REGISTRY.value_in("RP1.pressure", "mTorr"))
    )
    bias = supply_bias if supply_bias is not None else float(_REGISTRY.value_in("RP1.bias", "V"))

    n_0_low, n_0_high = _support("n_0")
    T_i_low, T_i_high = _support("T_i")
    gamma_low, gamma_high = _support("gamma_se")
    kappa_low, kappa_high = _support("kappa")

    gamma_entry = _REGISTRY["sheath.gamma_se_W"]
    gamma_sigma = gamma_entry.absolute_uncertainty
    if gamma_sigma is None:
        raise ValueError(
            "sheath.gamma_se_W carries no uncertainty, so doc 05 §2.1's "
            "'normal, 0.10 +/- 0.03' cannot be built from the registry"
        )

    supply_relative = _relative_uncertainty("RP1.bias")

    return ControlPrior(
        marginals=(
            LogUniformPrior(low=n_0_low, high=n_0_high),
            LogNormalPrior(
                median=float(_REGISTRY.value_in("RP1.T_e", "eV")),
                log_sigma=float(_REGISTRY.value_in("prior.T_e.log_sigma", "dimensionless")),
            ),
            LogUniformPrior(low=T_i_low, high=T_i_high),
            LogNormalPrior(
                median=pressure,
                log_sigma=float(_REGISTRY.value_in("prior.p.relative_sigma", "dimensionless")),
            ),
            NormalPrior(mean=bias, sigma=supply_relative * abs(bias)),
            UniformPrior(low=0.0, high=TWO_PI),
            TruncatedNormalPrior(
                mean=gamma_entry.value,
                sigma=gamma_sigma,
                low=gamma_low,
                high=gamma_high,
            ),
            UniformPrior(low=kappa_low, high=kappa_high),
        )
    )


def _support(name: str) -> tuple[float, float]:
    """The declared support of a bounded Level A parameter."""
    spec = next(s for s in CONTROL_PARAMETERS if s.name == name)
    if spec.support is None:  # pragma: no cover - the four callers are all bounded
        raise ValueError(f"{name} has no bounded support")
    return spec.support


def _relative_uncertainty(entry_id: str) -> float:
    """A registered relative one-sigma uncertainty, insisting it really is relative."""
    uncertainty = _REGISTRY[entry_id].uncertainty
    if uncertainty is None or uncertainty.kind != "relative":
        raise ValueError(
            f"{entry_id} must carry a relative uncertainty for doc 05 §2.1's "
            f"'informative — supply-measured, sigma = 1 %' to be well posed"
        )
    return uncertainty.value


# ─── doc 05 §4.2 — the explicit physics priors ───────────────────────────────────


def _paired(
    first: ArrayLike, second: ArrayLike, *, names: tuple[str, str]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce two profiles sampled on the same grid, insisting they really are."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"{names[0]} and {names[1]} must be sampled on the same grid; "
            f"got shape {a.shape} and shape {b.shape}"
        )
    return a, b


def _checked_weight(weight: float) -> float:
    """Reject a weight that would reward violating the constraint it penalises."""
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"a penalty weight cannot be negative or non-finite, got {weight}")
    return weight


def bohm_penalty(u_s: ArrayLike, c_s: ArrayLike, *, weight: float) -> float:
    """Soft penalty on ``u_s/c_s < 1`` — doc 05 §4.2.

    The Bohm criterion requires ions to reach the sheath edge at or above the ion sound
    speed. It is a property of the *solution* rather than of any coordinate, so it cannot
    be imposed structurally the way positivity is; doc 05 §4.2 gives its role precisely —
    it "enforces the sheath-edge condition when the discrepancy field would otherwise
    violate it". Without the discrepancy field of §2.3 the forward model satisfies it by
    construction and this term is identically zero.

    One-sided: exceeding the criterion is physical (a pre-sheath can accelerate ions past
    ``c_s``) and is not penalised. The hinge is quadratic so the penalty is ``C¹`` at the
    boundary, which matters because a gradient-based MAP run (doc 05 §5) would otherwise
    see a kink exactly where the constraint activates.

    Args:
        u_s: Ion drift speed at the sheath edge, on some grid. Any consistent units.
        c_s: Ion sound speed on the same grid, in the same units.
        weight: Penalty strength. Required, not defaulted — doc 05 §4.3 makes the
            regularisation weight a parameter chosen by one of three named procedures.

    Returns:
        A non-positive log-prior contribution.
    """
    speed, sound = _paired(u_s, c_s, names=("u_s", "c_s"))
    _checked_weight(weight)
    if np.any(sound <= 0.0):
        raise ValueError("the ion sound speed must be positive everywhere on the grid")

    shortfall = np.maximum(0.0, 1.0 - speed / sound)
    return float(-0.5 * weight * np.sum(shortfall * shortfall))


def quasineutrality_penalty(n_i: ArrayLike, n_e: ArrayLike, *, weight: float) -> float:
    """Soft penalty on ``(n_i - n_e)/n_i`` in the bulk — doc 05 §4.2.

    doc 05 §4.2 scopes this to ``z > 5 z_s``, the region beyond five sheath thicknesses
    where the plasma is quasi-neutral and the boundary condition is anchored. **The scoping
    is the caller's**: this package must not know what a sheath thickness is (doc 08 §3
    keeps the inverse layer ignorant of the forward model), so the caller passes the
    profiles already restricted to the bulk. Passing the whole domain would penalise the
    sheath for being a sheath.

    Args:
        n_i: Ion density on the bulk grid.
        n_e: Electron density on the same grid, in the same units.
        weight: Penalty strength; see :func:`bohm_penalty`.

    Returns:
        A non-positive log-prior contribution.
    """
    ions, electrons = _paired(n_i, n_e, names=("n_i", "n_e"))
    _checked_weight(weight)
    if np.any(ions <= 0.0):
        raise ValueError("the ion density must be positive everywhere on the bulk grid")

    residual = (ions - electrons) / ions
    return float(-0.5 * weight * np.sum(residual * residual))


def smoothness_log_prior(coefficients: ArrayLike, *, tau: float) -> float:
    """``alpha ~ N(0, τ²)`` on the discrepancy coefficients — doc 05 §2.3 and §4.2.

    The shrinkage prior of the Kennedy & O'Hagan construction, which doc 05 §4.2 also
    lists as "profile smoothness | Gaussian-process / Tikhonov on discrepancy
    coefficients | suppresses non-physical oscillation". Written out here because it is
    ordinary and because doc 05 §2.3 makes ``τ`` itself inferred — so this is called with
    a ``τ`` that is a parameter of the fit, not a setting.

    Unlike the two penalties above this one *is* a normalised density, which matters: doc
    05 §11 Q-06 asks whether the discrepancy field absorbs so much that ``θ_c`` becomes
    unidentifiable, and answering it by profile likelihood on ``τ`` requires the ``τ``
    dependence of the normaliser to be present. Dropping the ``-N log τ`` term would make
    the profile monotone in ``τ`` and the answer to Q-06 would be "yes" for the wrong
    reason.

    Args:
        coefficients: The ``alpha_j`` of doc 05 §2.3.
        tau: Shrinkage scale, positive.

    Returns:
        The joint log-density of the coefficients.
    """
    alpha = np.asarray(coefficients, dtype=float).ravel()
    _check_positive("tau", tau)

    quadratic = float(np.sum(alpha * alpha)) / (tau * tau)
    return -0.5 * quadratic - alpha.size * (math.log(tau) + 0.5 * _LOG_TWO_PI)
