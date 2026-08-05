"""The Level A control-parameter vector `θ_c` — doc 05 §2.1.

doc 05 §2 calls the parameterisation "the most consequential design decision in the
document: too few parameters and the model cannot fit the data (bias); too many and
nothing is identifiable (variance)". This module implements the eight parameters of the
§2.1 table and the bijections that carry them onto an unconstrained `R^8`.

## Why an unconstrained space at all

Both consumers need it and neither can be given a box instead:

* **The optimiser.** MAP (doc 05 §5) runs L-BFGS-B. A box constraint at ``n_0 > 0`` puts
  the optimum on the boundary in the low-density corner of doc 01 §2.4's envelope, where
  the gradient is not zero and the Hessian is not the posterior curvature — so the Laplace
  approximation built on it (doc 05 §5) would be wrong in exactly the regime doc 01 IF-6
  already calls hard.
* **The sampler.** HMC integrates Hamilton's equations. A hard wall reflects the
  trajectory, which destroys detailed balance unless handled explicitly; reparameterising
  is the standard and cheaper answer.

The transform choice follows doc 05 §4.2 directly: "Positivity | log-parameterisation of
``n``, ``T`` | structural". Structural means the constraint cannot be violated because
there is no representation of a violation, not that a penalty discourages it.

## The mapping, parameter by parameter

| Parameter | doc 05 §2.1 prior | Transform | Unconstrained support |
|---|---|---|---|
| ``n_0`` | log-uniform [1e15, 1e19] m^-3 | ``log`` | box in log space |
| ``T_e`` | log-normal, 3 eV, sigma = 0.4 | ``log`` | all of R |
| ``T_i`` | log-uniform [0.02, 0.5] eV | ``log`` | box in log space |
| ``p`` | informative, gauge, sigma = 2 % | ``log`` | all of R |
| ``V_w`` | informative, supply, sigma = 1 % | identity | all of R |
| ``phi_RF`` | uniform [0, 2π) rad | logit | all of R |
| ``gamma_se`` | normal, 0.10 ± 0.03 | logit | all of R |
| ``kappa`` | uniform [1, 5] | logit | all of R |

Every number above comes from the parameter registry (doc 08 §5); none is written here.

## Simplifications, and what each one costs

1. **`phi_RF` is mapped with a logit rather than treated as circular.** doc 05 §2.1 gives
   it a uniform prior on ``[0, 2π)``, which is a *circle*: ``φ = 0`` and ``φ = 2π`` are the
   same physical phase. A logit severs them, so a chain cannot cross the branch cut at
   ``φ = 0`` and posterior mass straddling it is reported as two modes rather than one.
   **Bound:** the error is confined to chains whose posterior mass lies within a few
   standard deviations of the cut; for a phase posterior of width ``sigma_phi`` the affected
   prior mass is ``≈ 2sigma_phi/2π``, i.e. under 5 % for the ``sigma_phi ≲ 0.15 rad`` that doc 01
   R-TEMP-1's 15-bin phase resolution implies. It is not confined at all when the phase is
   unidentified, which is precisely open question Q-05 (doc 05 §11) — and Q-05's stated
   resolution path is nested sampling, which does not use this transform.
2. **The log-uniform parameters keep a box in log space.** ``log`` maps ``(0, ∞)`` onto
   ``R``, but doc 05 §2.1's log-uniform priors have *bounded* support, so the transformed
   prior is uniform on an interval of ``R`` and not on all of it.
   **Bound:** exact for MAP, which is given the box by :func:`unconstrained_bounds`. For
   HMC the prior is zero outside the box and the sampler will reject there; the cost is
   entirely in efficiency near a boundary, and it is zero when the posterior is interior —
   which at RP-1 it is, the reference density sitting two decades inside both ends.
3. **The vector is a fixed 8-tuple, not a dynamic schema.** doc 05 §2.2's ten Level B
   nuisance parameters and §2.3's ~20 Level C discrepancy coefficients are not represented
   here. **Bound:** none on `θ_c` itself — the joint parameterisation concatenates blocks —
   but any code that assumes ``N_CONTROL`` is the full dimension of the inference problem
   is wrong by 30 parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, log_expit, logit

from vpl.core.params import default_registry

__all__ = [
    "CONTROL_PARAMETERS",
    "N_CONTROL",
    "ControlParameterSpec",
    "ControlParameters",
    "IdentityTransform",
    "LogTransform",
    "LogitTransform",
    "Transform",
    "control_parameter_names",
    "log_abs_det_jacobian",
    "unconstrained_bounds",
]

_REGISTRY: Final = default_registry()

#: Full turn, in radians. Structure rather than measurement: doc 05 §2.1's ``uniform
#: [0, 2π)`` is the statement that the RF phase is a phase, and there is nowhere to source
#: a value of 2π from.
TWO_PI: Final[float] = 2.0 * math.pi


# ─── transforms ──────────────────────────────────────────────────────────────────


@runtime_checkable
class Transform(Protocol):
    """A smooth bijection between a constrained coordinate and all of ``R``.

    The Jacobian convention is fixed here and used everywhere: ``log_abs_det_jacobian``
    is ``log |dx/du|`` — the *constrained* coordinate differentiated with respect to the
    *unconstrained* one — evaluated at the unconstrained point. That is the direction the
    change-of-variables formula needs,

        ``p_u(u) = p_x(x(u)) · |dx/du|``

    and stating it once in the protocol is cheaper than debugging a sign later. The single
    most common error in this kind of code is applying it inverted, which looks fine at
    ``u = 0`` and is wrong everywhere else.
    """

    def to_constrained(self, u: float) -> float:
        """Map an unconstrained coordinate into the parameter's own support."""
        ...

    def to_unconstrained(self, value: float) -> float:
        """Map a parameter value onto ``R``. Inverse of :meth:`to_constrained`."""
        ...

    def log_abs_det_jacobian(self, u: float) -> float:
        """``log |dx/du|`` at the unconstrained point ``u``."""
        ...


@dataclass(frozen=True, slots=True)
class IdentityTransform:
    """No reparameterisation — for a signed quantity with unbounded support.

    ``V_w`` is the only one. doc 01 §2.1 registers the reference bias as ``-250 V`` and
    doc 01 §2.4's envelope runs to ``-1000 V``, so neither a log (wrong sign) nor a logit
    (a bound the physics does not have) applies. Writing the no-op explicitly keeps every
    parameter on the same interface, so the vector operations need no special case.
    """

    def to_constrained(self, u: float) -> float:
        return u

    def to_unconstrained(self, value: float) -> float:
        return value

    def log_abs_det_jacobian(self, u: float) -> float:  # noqa: ARG002 - protocol shape
        return 0.0


@dataclass(frozen=True, slots=True)
class LogTransform:
    """``x = exp(u)`` — doc 05 §4.2's structural positivity constraint.

    ``log |dx/du| = u`` exactly, which is why this transform is cheap enough that there is
    no argument for enforcing positivity any other way.
    """

    def to_constrained(self, u: float) -> float:
        # `math.exp` rather than `np.exp` for two reasons. It is several times faster on a
        # scalar, which matters in a likelihood evaluated 10^6 times (doc 05 §5); and it
        # signals overflow by raising instead of by emitting a `RuntimeWarning`, which the
        # project's pytest configuration turns into a test failure. Overflow to `+inf` is
        # the correct limit of `exp` and a caller integrating over all of R will reach it.
        try:
            return math.exp(u)
        except OverflowError:
            return math.inf

    def to_unconstrained(self, value: float) -> float:
        if value <= 0.0:
            raise ValueError(f"a log-parameterised quantity must be positive, got {value}")
        return float(np.log(value))

    def log_abs_det_jacobian(self, u: float) -> float:
        return u


@dataclass(frozen=True, slots=True)
class LogitTransform:
    """``x = low + (high - low)·sigma(u)`` — for a parameter bounded on both sides.

    The Jacobian is ``(high - low)·sigma(u)·(1 - sigma(u))``, and it is *not* computed that way.
    ``sigma(u)`` underflows to zero around ``u ≈ -745``, and the product form then returns
    ``log 0 = -inf`` for a quantity whose true value is ``log(high - low) - 745``. An
    unconstrained sampler reaches ``|u| ~ 10²`` routinely during warm-up, so the naive form
    would hand it a ``-inf`` prior in a region that is merely improbable, not impossible,
    and the chain would stick there. Using ``log sigma`` directly

        ``log |dx/du| = log(high - low) + log sigma(u) + log sigma(-u)``

    is exact to machine precision at any ``u`` a float can hold.

    Attributes:
        low: Lower end of the support. Reachable only in the ``u → -∞`` limit.
        high: Upper end of the support.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError(f"a logit interval must be finite, got [{self.low}, {self.high}]")
        if not self.low < self.high:
            raise ValueError(f"a logit interval must be ascending, got [{self.low}, {self.high}]")

    @property
    def width(self) -> float:
        return self.high - self.low

    def to_constrained(self, u: float) -> float:
        return float(self.low + self.width * expit(u))

    def to_unconstrained(self, value: float) -> float:
        if not self.low <= value <= self.high:
            raise ValueError(f"{value} is outside the interval [{self.low}, {self.high}]")
        return float(logit((value - self.low) / self.width))

    def log_abs_det_jacobian(self, u: float) -> float:
        return float(math.log(self.width) + log_expit(u) + log_expit(-u))


# ─── the doc 05 §2.1 table ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ControlParameterSpec:
    """One row of doc 05 §2.1.

    Attributes:
        name: Field name on :class:`ControlParameters`, and the key everything else uses.
        symbol: The document's symbol, for figures and reports.
        units: pint-parseable units of the *constrained* value.
        transform: The bijection onto ``R``.
        support: ``(low, high)`` in constrained space, or ``None`` where the parameter is
            unbounded. This is the support of the doc 05 §2.1 prior, not a numerical
            clamp — a value outside it has zero prior probability, which is a different
            statement from "the solver would struggle there".
        reference_id: Registry id supplying the RP-1 nominal value, where one exists.
        description: What the parameter is, for someone reading a corner plot.
    """

    name: str
    symbol: str
    units: str
    transform: Transform
    support: tuple[float, float] | None
    reference_id: str | None
    description: str


def _sweep_range(entry_id: str) -> tuple[float, float]:
    """The registry's sweep range, which is the doc 05 §2.1 prior support for these rows.

    doc 09 §1 requires every DESIGN entry to declare the range it is swept over, and for
    the Level A parameters that range *is* the operating envelope of doc 01 §2.4 — which
    is what doc 05 §2.1's log-uniform and uniform priors are stated over. Reading it from
    the registry rather than restating it here means the prior and the identifiability map
    (doc 00 S5) cannot be drawn over different boxes.
    """
    entry = _REGISTRY[entry_id]
    if entry.sweep_range is None:
        raise ValueError(f"{entry_id} has no sweep_range, so it cannot bound a doc 05 prior")
    return entry.sweep_range


CONTROL_PARAMETERS: Final[tuple[ControlParameterSpec, ...]] = (
    ControlParameterSpec(
        name="n_0",
        symbol="n₀",
        units="m**-3",
        transform=LogTransform(),
        support=_sweep_range("RP1.n_0"),
        reference_id="RP1.n_0",
        description="Bulk electron density",
    ),
    ControlParameterSpec(
        name="T_e",
        symbol="T_e",
        units="eV",
        transform=LogTransform(),
        support=None,
        reference_id="RP1.T_e",
        description="Electron temperature",
    ),
    ControlParameterSpec(
        name="T_i",
        symbol="T_i",
        units="eV",
        transform=LogTransform(),
        support=_sweep_range("RP1.T_i"),
        reference_id="RP1.T_i",
        description="Ion temperature",
    ),
    ControlParameterSpec(
        name="p",
        symbol="p",
        units="mTorr",
        transform=LogTransform(),
        support=None,
        reference_id="RP1.pressure",
        description="Neutral pressure — gauge-measured",
    ),
    ControlParameterSpec(
        name="V_w",
        symbol="V_w",
        units="V",
        transform=IdentityTransform(),
        support=None,
        reference_id="RP1.bias",
        description="Wall bias — supply-measured",
    ),
    ControlParameterSpec(
        name="phi_RF",
        symbol="φ_RF",
        units="rad",
        transform=LogitTransform(low=0.0, high=TWO_PI),
        support=(0.0, TWO_PI),
        reference_id=None,
        description="RF phase offset",
    ),
    ControlParameterSpec(
        name="gamma_se",
        symbol="gamma_se",
        units="dimensionless",
        transform=LogitTransform(*_sweep_range("sheath.gamma_se_W")),
        support=_sweep_range("sheath.gamma_se_W"),
        reference_id="sheath.gamma_se_W",
        description="Ion-induced secondary electron emission yield",
    ),
    ControlParameterSpec(
        name="kappa",
        symbol="κ",
        units="dimensionless",
        transform=LogitTransform(*_sweep_range("eedf.kappa")),
        support=_sweep_range("eedf.kappa"),
        reference_id="eedf.kappa",
        description="EEDF shape parameter (Maxwellian → Druyvesteyn)",
    ),
)

#: doc 05 §2.1: "Level A — control parameters (θ_c, ~8)".
N_CONTROL: Final[int] = len(CONTROL_PARAMETERS)

#: The RF phase has no registry nominal, because there is nothing to source: doc 05 §2.1
#: gives it a uniform prior over the whole circle, which is the statement that no phase is
#: preferred. The reference vector therefore uses the midpoint of that prior, which is also
#: where the logit sits at ``u = 0`` — so a run started from the reference point starts an
#: optimiser at the origin in this coordinate rather than at a boundary.
_REFERENCE_PHI_RF: Final[float] = TWO_PI / 2.0


def control_parameter_names() -> tuple[str, ...]:
    """Field names in doc 05 §2.1 table order — the order of every array here."""
    return tuple(spec.name for spec in CONTROL_PARAMETERS)


# ─── the vector ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ControlParameters:
    """`θ_c` — the eight Level A parameters of doc 05 §2.1.

    Immutable, because a posterior sample that could be mutated in place is a posterior
    sample that will be, somewhere in a plotting routine, and the artifact written to disk
    would then not be the one that was scored.

    Attributes are in the units of the corresponding :class:`ControlParameterSpec`; they
    are plain floats rather than pint quantities because doc 08 §5 puts dimensional
    quantities at module boundaries and raw magnitudes inside hot loops, and a likelihood
    evaluated ``10^6`` times (doc 05 §5) is the hot loop.
    """

    n_0: float
    T_e: float
    T_i: float
    p: float
    V_w: float
    phi_RF: float
    gamma_se: float
    kappa: float

    def __post_init__(self) -> None:
        for spec in CONTROL_PARAMETERS:
            value = float(getattr(self, spec.name))
            if not math.isfinite(value):
                raise ValueError(f"{spec.name} must be finite, got {value}")
            self._check_support(spec, value)

    @staticmethod
    def _check_support(spec: ControlParameterSpec, value: float) -> None:
        """Reject a value the doc 05 §2.1 prior gives zero probability.

        Checked at construction rather than in the prior, so that a bug producing an
        impossible density surfaces where the value was made instead of as a ``-inf``
        log-posterior an optimiser silently walks away from.

        The bounds are closed. A logit at ``|u| ≳ 745`` returns exactly ``low`` or ``high``
        by underflow, and an optimiser that wandered there would otherwise be met with an
        exception rather than the ``-inf`` prior that correctly tells it to come back.
        """
        if spec.support is not None:
            low, high = spec.support
            if not low <= value <= high:
                raise ValueError(
                    f"{spec.name} = {value:g} is outside its doc 05 §2.1 support "
                    f"[{low:g}, {high:g}] {spec.units}"
                )
        elif isinstance(spec.transform, LogTransform) and value <= 0.0:
            raise ValueError(
                f"{spec.name} = {value:g} is outside its doc 05 §2.1 support "
                f"(0, inf) {spec.units}; doc 05 §4.2 makes positivity structural"
            )

    # ── constructors ────────────────────────────────────────────────────────────

    @classmethod
    def reference(cls) -> ControlParameters:
        """RP-1 — doc 01 §2.1, read from the registry.

        The point every quantitative requirement in the project is derived at.

        **It is not a safe optimiser initialisation.** ``eedf.kappa``'s registered nominal
        is 1.0 — a Maxwellian EEDF — which is exactly the lower edge of doc 05 §2.1's
        ``uniform [1, 5]``, so :meth:`to_unconstrained` maps it to ``-inf``. That is the
        mathematically correct image of a boundary under a logit and not a bug, but an
        engine that starts there starts nowhere. Use ``ControlPrior.median()`` from
        :mod:`vpl.inverse.priors`, which is interior by construction, and keep this for
        what it is: the physical reference point.
        """
        values: dict[str, float] = {}
        for spec in CONTROL_PARAMETERS:
            if spec.reference_id is None:
                values[spec.name] = _REFERENCE_PHI_RF
            else:
                values[spec.name] = float(_REGISTRY.value_in(spec.reference_id, spec.units))
        return cls(**values)

    @classmethod
    def from_array(cls, values: ArrayLike) -> ControlParameters:
        """Build from an 8-vector in doc 05 §2.1 table order."""
        return cls(**dict(zip(control_parameter_names(), _as_vector(values), strict=True)))

    @classmethod
    def from_unconstrained(cls, u: ArrayLike) -> ControlParameters:
        """Build from a point in the unconstrained space of :meth:`to_unconstrained`."""
        vector = _as_vector(u)
        return cls(
            **{
                spec.name: spec.transform.to_constrained(float(value))
                for spec, value in zip(CONTROL_PARAMETERS, vector, strict=True)
            }
        )

    # ── views ───────────────────────────────────────────────────────────────────

    def as_array(self) -> NDArray[np.float64]:
        """The eight values in doc 05 §2.1 table order."""
        return np.array([getattr(self, spec.name) for spec in CONTROL_PARAMETERS], dtype=float)

    def to_dict(self) -> dict[str, float]:
        """Name-to-value mapping, in table order."""
        return {spec.name: float(getattr(self, spec.name)) for spec in CONTROL_PARAMETERS}

    def to_unconstrained(self) -> NDArray[np.float64]:
        """The point in ``R^8`` an optimiser or sampler works in.

        A parameter sitting exactly on a closed support boundary maps to ``±inf``, which is
        the mathematically correct image and not an error. It is also the only way to get
        one, so it is a reliable signal that a fit has run into a prior edge.
        """
        return np.array(
            [
                spec.transform.to_unconstrained(float(getattr(self, spec.name)))
                for spec in CONTROL_PARAMETERS
            ],
            dtype=float,
        )

    def replace(self, **changes: float) -> ControlParameters:
        """A copy with some fields changed — the immutable update of the coding rules."""
        return replace(self, **changes)

    def __repr__(self) -> str:
        body = ", ".join(
            f"{spec.symbol}={getattr(self, spec.name):.4g}" for spec in CONTROL_PARAMETERS
        )
        return f"ControlParameters({body})"


def _as_vector(values: ArrayLike) -> NDArray[np.float64]:
    """Coerce to a finite 8-vector, naming the length mismatch if there is one."""
    vector: NDArray[Any] = np.asarray(values, dtype=float).ravel()
    if vector.size != N_CONTROL:
        raise ValueError(
            f"doc 05 §2.1 has {N_CONTROL} Level A control parameters; got {vector.size} values"
        )
    return vector


# ─── the change of variables ─────────────────────────────────────────────────────


def log_abs_det_jacobian(u: ArrayLike) -> float:
    """``log |det ∂x/∂u|`` for the whole vector, at the unconstrained point ``u``.

    This is the term that makes a prior stated in doc 05 §2.1's units correct in the space
    the sampler works in:

        ``log p_u(u) = log p_x(x(u)) + log |det ∂x/∂u|``

    Omitting it does not make the code fail; it makes the posterior wrong, systematically
    and invisibly, by tilting it towards whichever end of each transform is expanded. For a
    log transform the omission biases every positive quantity low by roughly one prior
    standard deviation, which for ``n_0`` is a factor of a few — and nothing in the output
    says so. :mod:`vpl.inverse.priors` applies it; this is where it is computed.

    The transform acts componentwise, so ``∂x/∂u`` is diagonal and the log-determinant is
    the sum of the component terms. That is asserted by test, not assumed.
    """
    vector = _as_vector(u)
    return float(
        sum(
            spec.transform.log_abs_det_jacobian(float(value))
            for spec, value in zip(CONTROL_PARAMETERS, vector, strict=True)
        )
    )


def unconstrained_bounds() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Box bounds in unconstrained space, for the L-BFGS-B of doc 05 §5.

    Most parameters return ``(-inf, +inf)``: their transform maps all of ``R`` into the
    support, so there is nothing left to constrain. The exceptions are the log-uniform
    ones, whose support is bounded *and* whose transform is a plain ``log`` — for those the
    box survives into log space. Handing an optimiser the box is strictly better than
    letting it discover a ``-inf`` prior by stepping outside: L-BFGS-B projects, whereas a
    line search meeting an infinite objective backtracks to nothing useful.

    Returns:
        ``(lower, upper)``, each of length :data:`N_CONTROL`, in table order.
    """
    lower = np.full(N_CONTROL, -np.inf)
    upper = np.full(N_CONTROL, np.inf)
    for index, spec in enumerate(CONTROL_PARAMETERS):
        if spec.support is None or not isinstance(spec.transform, LogTransform):
            continue
        low, high = spec.support
        lower[index] = spec.transform.to_unconstrained(low)
        upper[index] = spec.transform.to_unconstrained(high)
    return lower, upper
