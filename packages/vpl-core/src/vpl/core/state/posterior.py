"""The result of an inversion — doc 05 §10, doc 07 §2.2, ADR-005.

Doc 05 §1.1 is explicit that the deliverable is "not a number but a distribution", and
doc 06 §8 forbids reporting a bare one: the tier, the credible interval and the sampler
diagnostics travel with the value, mechanically. This module is that object.

Three design constraints come straight from the specification and are worth stating
before the code, because each of them looks like an optional nicety and is not:

1. **The three-level split survives.** Doc 05 §2 parameterises the state as control
   (`theta_c`), nuisance (`theta_n`) and profile-correction (`theta_f`) parameters, and
   doc 05 §10 requires that split in the emitted artifact. Flattening 45 parameters into
   one anonymous array would discard the distinction between "the answer", "the
   systematics we had to infer" and "the model error we absorbed" — which is exactly the
   distinction doc 05 §2.2 says separates a working inversion from one that quietly
   hides its systematics.

2. **The tier is required and never defaults.** Doc 05 §7.2 makes reporting T1 as if it
   were T2 a project *defect*, and G-V7 (doc 07 §6) requires the label on every reported
   figure. A posterior that cannot say which tier produced it is not reportable, so
   :attr:`Posterior.tier` has no default anywhere.

3. **Thinning is ESS-aware, not fixed-stride.** ADR-005's stated consequence is that a
   fixed-stride thin biases coverage statistics — and coverage (doc 06 §7.1, gate G-V4)
   is the evidence for doc 00 S4, the claim that the uncertainty is calibrated. Thinning
   an archive with a stride chosen for convenience would corrupt the one number the
   project exists to defend. :meth:`Posterior.thin` therefore refuses any stride that
   would drop effective sample size below the G-V3 floor, and the posterior it returns
   reports its own *reduced* ESS rather than the pre-thinning one.

Computing R-hat and ESS is not this package's job — doc 08 §3 keeps ``vpl-core`` free of
a sampling stack, and the estimators live with the engines. This module stores those
numbers and gates on them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final, NamedTuple, Self

import numpy as np
from numpy.typing import NDArray

from vpl.core.provenance import Tier

__all__ = [
    "DEFAULT_CREDIBLE_LEVEL",
    "GATE_MAX_DIVERGENCES",
    "GATE_MAX_R_HAT",
    "GATE_MIN_ESS",
    "CredibleInterval",
    "DerivedQuantities",
    "ParameterLevel",
    "Posterior",
    "SamplerDiagnostics",
]

#: Largest permitted potential-scale-reduction factor — gate G-V3, doc 07 §6.
#:
#: The comparison is strict, because doc 07 §6 and test V-44 both write "R-hat < 1.01".
GATE_MAX_R_HAT: Final[float] = 1.01

#: Smallest permitted effective sample size, *per parameter* — gate G-V3, doc 07 §6.
#:
#: Also strict: doc 07 §6 and V-44 write "ESS > 400". ADR-005 writes ">= 400" for the
#: same quantity; the acceptance gate is authoritative and the strict reading is the
#: one that cannot let a marginal run through.
GATE_MIN_ESS: Final[float] = 400.0

#: Permitted divergent transitions — gate G-V3, doc 07 §6, which requires zero.
#:
#: A divergence means the sampler failed to explore a region of the posterior, so the
#: samples are biased in a direction nobody can bound. There is no "acceptable few".
GATE_MAX_DIVERGENCES: Final[int] = 0

#: The credible level every gate and report is stated at.
#:
#: doc 00 S4 makes "the empirical coverage of the 95 % credible interval" a pass/fail
#: criterion, gate G-V4 requires it in [0.93, 0.97], and doc 06 §8's reporting standard
#: prints it. One named constant, so the default interval and the gate that checks it
#: cannot drift apart.
DEFAULT_CREDIBLE_LEVEL: Final[float] = 0.95

#: Quantities that are functions of the samples rather than samples themselves, keyed by
#: name. Doc 05 §10 requires ``Gamma_E(z, t)``, ``Gamma_i``, ``<E_i>`` and the wall IEDF
#: in every emitted posterior. Each array's first two axes are ``(chain, draw)``.
type DerivedQuantities = Mapping[str, NDArray[np.float64]]


class ParameterLevel(StrEnum):
    """Which level of the doc 05 §2 hierarchy a parameter belongs to.

    Deliberately **not** ordered and deliberately not merged. The three levels answer
    different questions — what was measured, what had to be inferred to avoid bias, and
    how much model error was absorbed — and doc 05 §2.3 warns that confounding between
    ``theta_c`` and ``theta_f`` is real and measured (§6). A posterior that could not
    say which level a parameter came from could not support that analysis.
    """

    CONTROL = "theta_c"
    NUISANCE = "theta_n"
    PROFILE = "theta_f"

    @property
    def description(self) -> str:
        return _LEVEL_DESCRIPTIONS[self]


_LEVEL_DESCRIPTIONS: dict[ParameterLevel, str] = {
    ParameterLevel.CONTROL: (
        "Level A control parameters (doc 05 §2.1): physically meaningful, directly "
        "interpretable, and what an operator would actually set."
    ),
    ParameterLevel.NUISANCE: (
        "Level B nuisance parameters (doc 05 §2.2): not of interest, but they must be "
        "inferred or the answer is biased."
    ),
    ParameterLevel.PROFILE: (
        "Level C profile corrections (doc 05 §2.3): coefficients of the discrepancy "
        "field, which inflate the posterior instead of letting model error corrupt it."
    ),
}


class CredibleInterval(NamedTuple):
    """An equal-tailed interval and the level it was computed at.

    The level is part of the value, not context. Doc 06 §8's reporting standard prints
    ``95 % CI [4.41, 8.83]`` and gate G-V4 is stated at 95 % specifically, so an
    interval that travelled without its level could be read against the wrong gate.

    Attributes:
        lower: Lower bound, shaped like the quantity (``()`` for a scalar parameter).
        upper: Upper bound, same shape.
        level: Credible mass between the bounds, in ``(0, 1)``.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    level: float


def _frozen_array(values: NDArray[np.float64], *, name: str) -> NDArray[np.float64]:
    """Validate an array and return an immutable copy of it.

    Copying is not defensive pedantry. A posterior is written to an archive and read by
    people who cannot re-derive it, so an array that aliased the caller's buffer could
    be rewritten after the credible intervals had already been quoted from it.

    Rank is checked by the caller, which knows what the axes mean and can say so.
    """
    array = np.array(values, dtype=np.float64, copy=True)

    if not np.all(np.isfinite(array)):
        # A non-finite sample is a sampler bug, not a wide posterior: the stored
        # position of a chain is always finite, and a nan here would silently poison
        # every mean and quantile computed from it.
        raise ValueError(f"{name} must be finite; found nan or inf")

    array.flags.writeable = False
    return array


def _finite_positive(values: Mapping[str, float], *, name: str) -> Mapping[str, float]:
    """Check a per-parameter diagnostic mapping and freeze a copy of it."""
    for parameter, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite for {parameter!r}, got {value}")
        if value <= 0.0:
            raise ValueError(f"{name} must be positive for {parameter!r}, got {value}")

    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class SamplerDiagnostics:
    """What the sampler reports about its own trustworthiness — doc 05 §10, V-44.

    Gate G-V3 (doc 07 §6) is the only consumer that matters, and it is implemented once,
    in :meth:`gate_failures`. The thresholds are arguments rather than literals so that a
    benchmark can be *stricter* than the gate; passing a weaker one raises, because a
    silently relaxed acceptance criterion is indistinguishable from a passing run in
    every artifact that records only the verdict.

    Attributes:
        r_hat: Potential scale reduction factor per parameter. Read-only.
        ess: Effective sample size per parameter. Read-only. Not bounded above by the
            draw count — NUTS is antithetic and genuinely produces ``ESS > N`` on
            well-behaved targets, so an "ESS cannot exceed the draws" check would reject
            good runs.
        divergences: Divergent transitions over the whole run.
        e_bfmi: Energy Bayesian fraction of missing information, one entry per chain —
            the scalar behind the energy plots of doc 05 §10. Recorded but **not** gated:
            doc 07 §6 states G-V3 in terms of R-hat, ESS and divergences only, and this
            module does not invent an acceptance threshold the specification omits.
    """

    # Excluded from the hash, not from equality. ``MappingProxyType`` is unhashable, and
    # dropping the mappings from equality too would make two diagnostics that disagree
    # about every parameter compare equal — which is worse than a weak hash.
    r_hat: Mapping[str, float] = field(hash=False)
    ess: Mapping[str, float] = field(hash=False)
    divergences: int
    e_bfmi: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if set(self.r_hat) != set(self.ess):
            missing_ess = sorted(set(self.r_hat) - set(self.ess))
            missing_r_hat = sorted(set(self.ess) - set(self.r_hat))
            # G-V3 is stated *per parameter*. A parameter with only half its diagnostics
            # would be gated on the half that happened to be present.
            raise ValueError(
                "every parameter needs both an r_hat and an ess; "
                f"missing ess for {missing_ess}, missing r_hat for {missing_r_hat}"
            )

        if self.divergences < 0:
            raise ValueError(
                f"divergences is a count and cannot be negative, got {self.divergences}"
            )

        for value in self.e_bfmi:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"e_bfmi must be finite and positive, got {value}")

        object.__setattr__(self, "r_hat", _finite_positive(self.r_hat, name="r_hat"))
        object.__setattr__(self, "ess", _finite_positive(self.ess, name="ess"))
        object.__setattr__(self, "e_bfmi", tuple(self.e_bfmi))

    @property
    def parameters(self) -> tuple[str, ...]:
        """The parameters these diagnostics cover, in the order they were supplied."""
        return tuple(self.r_hat)

    @property
    def worst_ess(self) -> float:
        """The smallest effective sample size, which is what G-V3 binds on."""
        return min(self.ess.values())

    def gate_failures(
        self,
        *,
        max_r_hat: float = GATE_MAX_R_HAT,
        min_ess: float = GATE_MIN_ESS,
        max_divergences: int = GATE_MAX_DIVERGENCES,
    ) -> tuple[str, ...]:
        """Every reason this run fails gate G-V3, or ``()`` if it passes.

        Returning the reasons rather than a bare boolean is deliberate: a benchmark that
        fails needs to name the parameter that failed it, and a gate that can only say
        "no" gets worked around instead of fixed.

        Args:
            max_r_hat: Exclusive upper bound on R-hat. May be tightened below
                :data:`GATE_MAX_R_HAT`, never raised above it.
            min_ess: Exclusive lower bound on per-parameter ESS. May be raised above
                :data:`GATE_MIN_ESS`, never lowered below it.
            max_divergences: Inclusive upper bound on divergent transitions. May be
                tightened below :data:`GATE_MAX_DIVERGENCES` — which is already zero.

        Returns:
            Human-readable failure descriptions, in parameter order.

        Raises:
            ValueError: If any threshold is weaker than the corresponding G-V3 value.
        """
        if max_r_hat > GATE_MAX_R_HAT:
            raise ValueError(
                f"max_r_hat={max_r_hat} is weaker than gate G-V3 (doc 07 §6), which requires "
                f"R-hat < {GATE_MAX_R_HAT}. Thresholds may be tightened, never relaxed."
            )
        if min_ess < GATE_MIN_ESS:
            raise ValueError(
                f"min_ess={min_ess} is weaker than gate G-V3 (doc 07 §6), which requires "
                f"ESS > {GATE_MIN_ESS} per parameter. Thresholds may be tightened, never relaxed."
            )
        if max_divergences > GATE_MAX_DIVERGENCES:
            raise ValueError(
                f"max_divergences={max_divergences} is weaker than gate G-V3 (doc 07 §6), which "
                f"requires {GATE_MAX_DIVERGENCES}. Thresholds may be tightened, never relaxed."
            )

        failures = [
            f"R-hat {self.r_hat[parameter]:.4g} for {parameter!r} is not below {max_r_hat}"
            for parameter in self.r_hat
            if not self.r_hat[parameter] < max_r_hat
        ]
        failures.extend(
            f"ESS {self.ess[parameter]:.4g} for {parameter!r} is not above {min_ess}"
            for parameter in self.ess
            if not self.ess[parameter] > min_ess
        )
        if self.divergences > max_divergences:
            failures.append(
                f"{self.divergences} divergent transitions, gate permits {max_divergences}"
            )

        return tuple(failures)

    def is_clean(
        self,
        *,
        max_r_hat: float = GATE_MAX_R_HAT,
        min_ess: float = GATE_MIN_ESS,
        max_divergences: int = GATE_MAX_DIVERGENCES,
    ) -> bool:
        """Whether this run passes gate G-V3 — doc 07 §6, test V-44.

        See :meth:`gate_failures` for the arguments and for why loosening them raises.
        """
        return not self.gate_failures(
            max_r_hat=max_r_hat, min_ess=min_ess, max_divergences=max_divergences
        )

    def __repr__(self) -> str:
        return (
            f"SamplerDiagnostics(n_parameters={len(self.r_hat)}, "
            f"max_r_hat={max(self.r_hat.values()):.4g}, "
            f"min_ess={self.worst_ess:.4g}, divergences={self.divergences})"
        )


def _ess_capped_at(diagnostics: SamplerDiagnostics, n_retained: int) -> SamplerDiagnostics:
    """The same diagnostics with every ESS capped at the retained draw count.

    Thinning by a stride ``s`` leaves draws whose autocorrelation time is ``tau/s`` while
    the count falls by ``s``, so the effective sample size is unchanged until the stride
    reaches ``tau`` and equals the retained count thereafter: ``ESS_thin = min(ESS, N)``.
    That identity is exact for an AR(1) chain and is an *upper bound* in general, which
    is why :meth:`Posterior.with_diagnostics` exists — an ESS recomputed on the thinned
    draws supersedes this estimate.

    R-hat and the divergence count are carried over untouched. Both describe the
    sampling run rather than the retained subset, and ADR-005 keeps the full chains
    regenerable, so they remain true of the thing they describe.
    """
    return SamplerDiagnostics(
        r_hat=dict(diagnostics.r_hat),
        ess={name: min(value, float(n_retained)) for name, value in diagnostics.ess.items()},
        divergences=diagnostics.divergences,
        e_bfmi=diagnostics.e_bfmi,
    )


@dataclass(frozen=True, eq=False, slots=True)
class Posterior:
    """Samples over named parameters, with everything doc 05 §10 requires beside them.

    Shape is ``(n_chains, n_draws, n_params)``. Keeping the chain axis rather than
    pooling it is what makes R-hat, the trace plots of doc 05 §10 and the stride
    arithmetic of :meth:`thin` meaningful — a pooled array cannot tell a poorly mixed
    run from a well mixed one.

    Attributes:
        samples: The draws. Read-only.
        names: Parameter names, in the order of the last sample axis.
        levels: Which doc 05 §2 level each parameter belongs to. Read-only, and required
            to cover every name exactly.
        tier: Which of doc 05 §7.2's three claims these samples support. Never defaulted
            — see the module docstring.
        diagnostics: The sampler's report on itself. Required, because doc 05 §10 emits
            it for every inversion and because ESS-aware thinning is impossible without
            it. Engines that run no Markov chain (MAP, Laplace — doc 05 §5) supply the
            iid equivalent rather than omitting it.
        derived: Quantities that are functions of the samples — ``Gamma_E``, ``Gamma_i``,
            ``<E_i>``, the wall IEDF. Each shares the ``(chain, draw)`` axes so it thins
            with the samples and can be summarised the same way. Read-only.
    """

    samples: NDArray[np.float64]
    names: tuple[str, ...]
    levels: Mapping[str, ParameterLevel]
    tier: Tier
    diagnostics: SamplerDiagnostics
    derived: DerivedQuantities = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _frozen_array(self.samples, name="samples"))
        object.__setattr__(self, "names", tuple(self.names))

        if self.samples.ndim != 3:
            raise ValueError(
                f"samples must be three-dimensional (chain, draw, parameter), "
                f"got shape {self.samples.shape}"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"parameter names contain duplicates: {self.names}")
        if self.samples.shape[2] != len(self.names):
            raise ValueError(
                f"samples has n_params={self.samples.shape[2]} but {len(self.names)} names "
                f"were given: {self.names}"
            )

        named = set(self.names)
        if set(self.levels) != named:
            # Doc 05 §10 requires the theta_c / theta_n / theta_f split in the emitted
            # artifact, so a parameter with no level is a parameter that cannot be
            # written out correctly.
            raise ValueError(
                f"levels must name exactly the sampled parameters; "
                f"missing {sorted(named - set(self.levels))}, "
                f"unexpected {sorted(set(self.levels) - named)}"
            )
        if set(self.diagnostics.parameters) != named:
            raise ValueError(
                f"diagnostics must cover exactly the sampled parameters; "
                f"missing {sorted(named - set(self.diagnostics.parameters))}, "
                f"unexpected {sorted(set(self.diagnostics.parameters) - named)}"
            )
        if self.diagnostics.e_bfmi and len(self.diagnostics.e_bfmi) != self.samples.shape[0]:
            raise ValueError(
                f"diagnostics.e_bfmi has {len(self.diagnostics.e_bfmi)} entries but there are "
                f"{self.samples.shape[0]} chains"
            )

        object.__setattr__(self, "levels", MappingProxyType(dict(self.levels)))
        object.__setattr__(self, "derived", self._frozen_derived())

    def _frozen_derived(self) -> DerivedQuantities:
        draw_axes = self.samples.shape[:2]
        frozen: dict[str, NDArray[np.float64]] = {}

        for name, values in self.derived.items():
            if name in self.names:
                # A derived quantity is a function of the samples, not one of them.
                # Sharing a name would make every lookup ambiguous and would let a
                # reader mistake a computed profile for something the sampler visited.
                raise ValueError(f"derived quantity {name!r} would shadow a sampled parameter")

            array = _frozen_array(values, name=f"derived[{name!r}]")
            if array.ndim < 2 or array.shape[:2] != draw_axes:
                raise ValueError(
                    f"derived quantity {name!r} has draw axes {array.shape[:2]}, expected "
                    f"{draw_axes}: a derived quantity has one value per draw"
                )
            frozen[name] = array

        return MappingProxyType(frozen)

    # ── shape ───────────────────────────────────────────────────────────────────

    @property
    def n_chains(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_draws(self) -> int:
        """Draws retained *per chain*."""
        return int(self.samples.shape[1])

    @property
    def n_params(self) -> int:
        return int(self.samples.shape[2])

    @property
    def n_samples(self) -> int:
        """Total draws across all chains — the pool the quantiles are taken over."""
        return self.n_chains * self.n_draws

    @property
    def derived_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.derived))

    # ── named access ────────────────────────────────────────────────────────────

    def level_of(self, name: str) -> ParameterLevel:
        """Which doc 05 §2 level ``name`` belongs to."""
        return self.levels[name]

    def names_at(self, level: ParameterLevel) -> tuple[str, ...]:
        """The parameters of one level, in sample-axis order."""
        return tuple(name for name in self.names if self.levels[name] is level)

    def samples_for(self, name: str) -> NDArray[np.float64]:
        """The draws of one sampled parameter, shaped ``(n_chains, n_draws)``."""
        try:
            index = self.names.index(name)
        except ValueError:
            hint = " (it is a derived quantity; use derived_for)" if name in self.derived else ""
            raise KeyError(f"{name!r} is not a sampled parameter{hint}") from None

        return self.samples[:, :, index]

    def derived_for(self, name: str) -> NDArray[np.float64]:
        """One derived quantity, shaped ``(n_chains, n_draws, ...)``."""
        if name not in self.derived:
            raise KeyError(f"{name!r} is not a derived quantity of this posterior")
        return self.derived[name]

    def _values_for(self, name: str) -> NDArray[np.float64]:
        """Draws of a sampled parameter or of a derived quantity, whichever it is."""
        if name in self.derived:
            return self.derived[name]
        if name in self.names:
            return self.samples[:, :, self.names.index(name)]

        raise KeyError(
            f"{name!r} is neither a sampled parameter {self.names} nor a derived "
            f"quantity {self.derived_names}"
        )

    # ── summaries ───────────────────────────────────────────────────────────────

    def mean(self, name: str) -> NDArray[np.float64]:
        """Posterior mean of a sampled parameter or derived quantity.

        Returns an array shaped like the quantity itself: ``()`` for a scalar parameter,
        ``(n_z,)`` for a ``Gamma_E`` profile. One return type for both keeps the
        reporting pipeline of doc 06 §8 from needing two code paths.
        """
        return np.asarray(np.mean(self._values_for(name), axis=(0, 1)), dtype=np.float64)

    def credible_interval(
        self, name: str, level: float = DEFAULT_CREDIBLE_LEVEL
    ) -> CredibleInterval:
        """Equal-tailed credible interval — doc 06 §7.1, gate G-V4.

        Equal-tailed rather than highest-density: doc 06 §7.1 defines coverage as
        ``1[truth in CI_q]`` with ``q = 0.95``, and equal-tailed intervals are the ones
        whose coverage the SBC rank statistic of doc 06 §7.2 tests directly. An HPD
        interval is a different object and would silently change what G-V4 measures.

        Args:
            name: A sampled parameter or a derived quantity.
            level: Credible mass, in the open interval ``(0, 1)``. Defaults to the 95 %
                at which doc 06 §7.1 states the coverage gate.

        Returns:
            The bounds and the level they were computed at.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)``.
            KeyError: If ``name`` is neither sampled nor derived.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"credible level must lie in the open interval (0, 1), got {level}")

        values = self._values_for(name)
        pooled = values.reshape(self.n_samples, *values.shape[2:])
        tail = 0.5 * (1.0 - level)
        bounds = np.quantile(pooled, (tail, 1.0 - tail), axis=0)

        return CredibleInterval(
            lower=np.asarray(bounds[0], dtype=np.float64),
            upper=np.asarray(bounds[1], dtype=np.float64),
            level=level,
        )

    # ── ESS-aware thinning (ADR-005) ────────────────────────────────────────────

    def _retained(self, stride: int) -> int:
        """Total draws left after keeping every ``stride``-th draw of every chain."""
        return self.n_chains * -(-self.n_draws // stride)

    def max_safe_stride(self, *, min_ess: float = GATE_MIN_ESS) -> int:
        """The deepest thinning that still holds ``min_ess`` for every parameter.

        This is the ADR-005 archival operation: "thinned samples (retaining ESS >= 400
        per parameter) + full summary statistics + the manifest". Because the retained
        ESS is ``min(ESS, N_retained)`` (see :func:`_ess_capped_at`), the binding
        constraint is simply the retained draw count, and the answer is the largest
        stride whose retained count still clears the floor.

        Args:
            min_ess: Exclusive floor on the retained ESS. Defaults to the G-V3 value.

        Returns:
            The largest usable stride; ``1`` when no thinning is affordable.

        Raises:
            ValueError: If the posterior's ESS is already at or below the floor, in
                which case no stride — not even ``1`` — produces an archivable result.
        """
        self._require_headroom(min_ess)

        # ``_retained`` is non-increasing in the stride, so a bisection is exact.
        required = math.floor(min_ess) + 1
        low, high, best = 1, self.n_draws, 1
        while low <= high:
            middle = (low + high) // 2
            if self._retained(middle) >= required:
                best, low = middle, middle + 1
            else:
                high = middle - 1

        return best

    def thin(self, stride: int, *, min_ess: float = GATE_MIN_ESS) -> Posterior:
        """Keep every ``stride``-th draw of every chain, refusing to break the floor.

        ADR-005's consequence — "thinning must be ESS-aware rather than fixed-stride, or
        coverage statistics will be biased" — is enforced here rather than documented
        here. A stride that would leave any parameter at or below ``min_ess`` raises, and
        the returned posterior reports the ESS it actually has, so a later G-V3 check
        cannot be answered with pre-thinning numbers.

        Lowering ``min_ess`` below :data:`GATE_MIN_ESS` is permitted, and is the only way
        to thin harder. It is not a loophole: the floor is named at the call site, and
        the resulting posterior's own diagnostics then fail :meth:`SamplerDiagnostics.is_clean`,
        so the cost travels with the artifact instead of being lost at the call.

        Args:
            stride: Keep one draw in ``stride``. ``1`` is a no-op.
            min_ess: Exclusive floor on the retained ESS. Defaults to the G-V3 value.

        Returns:
            A new posterior over the retained draws, with derived quantities thinned
            alongside and diagnostics updated.

        Raises:
            ValueError: If ``stride`` is not positive, or if the retained ESS of any
                parameter would fall to or below ``min_ess``.
        """
        if stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride}")

        self._require_headroom(min_ess)

        retained = self._retained(stride)
        if retained <= min_ess:
            raise ValueError(
                f"stride {stride} would retain {retained} draws, dropping ESS to "
                f"{retained} at or below the floor of {min_ess} (ADR-005, gate G-V3). "
                f"The deepest safe stride here is {self.max_safe_stride(min_ess=min_ess)}."
            )

        return Posterior(
            samples=self.samples[:, ::stride, :],
            names=self.names,
            levels=self.levels,
            tier=self.tier,
            diagnostics=_ess_capped_at(self.diagnostics, retained),
            derived={name: values[:, ::stride] for name, values in self.derived.items()},
        )

    def _require_headroom(self, min_ess: float) -> None:
        """Reject a posterior that fails the floor before any thinning is considered."""
        if self.diagnostics.worst_ess <= min_ess:
            raise ValueError(
                f"posterior already has ESS {self.diagnostics.worst_ess:.4g}, at or below the "
                f"floor of {min_ess}; no stride can restore it. Lengthen the chains "
                f"(doc 07 §6, gate G-V3) rather than thinning."
            )

    # ── derivation ──────────────────────────────────────────────────────────────

    def with_diagnostics(self, diagnostics: SamplerDiagnostics) -> Self:
        """Replace the diagnostics, keeping the samples.

        The cap :func:`_ess_capped_at` applies on thinning is an upper bound. When
        ``vpl-uq`` recomputes ESS on the retained draws — the honest number — this is how
        it is installed, without pretending the samples changed.
        """
        return replace(self, diagnostics=diagnostics)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Posterior):
            return NotImplemented

        return (
            self.names == other.names
            and self.tier is other.tier
            and dict(self.levels) == dict(other.levels)
            and self.diagnostics == other.diagnostics
            and np.array_equal(self.samples, other.samples)
            and self.derived_names == other.derived_names
            and all(
                np.array_equal(self.derived[name], other.derived[name])
                for name in self.derived_names
            )
        )

    def __repr__(self) -> str:
        return (
            f"Posterior(n_chains={self.n_chains}, n_draws={self.n_draws}, "
            f"n_params={self.n_params}, tier={self.tier.value}, "
            f"derived={list(self.derived_names)}, min_ess={self.diagnostics.worst_ess:.4g})"
        )
