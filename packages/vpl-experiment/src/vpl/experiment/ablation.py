"""The channel ablation experiment — doc 11 §9 item 6, WBS 4.9.

doc 11 §9 item 6 asks a specific question of the two-channel fusion :mod:`vpl.experiment.
channels` assembles: drop each channel in turn, and show that the credible interval on
``Gamma_E`` inflates and the recovery degrades. :mod:`vpl.inverse.fusion` already makes
"drop a channel" a first-class, non-mutating operation (:meth:`~vpl.inverse.fusion.
JointLikelihood.without`) precisely so this experiment can be built without reaching into
the fusion internals; this module is the harness that runs the MAP recovery once per
configuration and reports the comparison doc 11 §9 item 6 wants.

## Where this sits relative to :mod:`vpl.experiment.closed_loop`

The recovery loop here is the same shape as ``closed_loop._run``'s steps 1-4b — draw a
sealed truth, generate a synthetic measurement, run MAP over the reduced ``(n_0, T_e)``
prior, sample the posterior through ``Gamma_E`` — with two differences that matter:

1. The likelihood is a :class:`~vpl.inverse.fusion.JointLikelihood` over one or two
   channels rather than ``closed_loop``'s single OES term, built by :mod:`vpl.experiment.
   channels`.
2. The likelihood, and therefore the recovery, is re-run once per **ablation
   configuration** against the *same* sealed truth, so the configurations are comparable.

Nothing here re-derives MAP, the reduced prior, the ``Gamma_E`` interval machinery or the
seal/commit discipline: :func:`~vpl.inverse.map.maximum_a_posteriori`, ``closed_loop.
_reduced_prior``, ``closed_loop._theta_from_unconstrained``, ``closed_loop.
_gamma_e_interval`` and :class:`~vpl.validation.sealed.SealedTruth` are reused as-is, so a
change to any of those propagates here rather than drifting apart from it.

## The operating point this experiment must run at, and why

:mod:`vpl.experiment.channels`'s own module docstring records the finding that decides
this: **at RP-1's -250 V wall bias, LIF declares itself blind** (doc 01 §5.1, doc 14
RS-03 — a 20 GHz mode-hop-free laser range is a 25.8 km/s ceiling in ``v_z``, and a 250 V
argon sheath is 34.9 km/s). :class:`~vpl.inverse.fusion.JointLikelihood` then excludes LIF
from the sum entirely (doc 01 IF-6), so "both channels" at RP-1 *is* "OES alone" — dropping
LIF there would change nothing, and an ablation figure built at that bias would show two
identical bars that read as "LIF carries no information", which is the opposite of what
:mod:`vpl.experiment.channels` found. So :func:`run_ablation` runs at
:data:`REACHABLE_BIAS_V`, the same reduced bias ``test_channels.py`` uses (for
comparability — that file is where the value's provenance is worked out in full), and
:func:`run_rp1_baseline` runs the *unmodified* RP-1 configuration as a separate, explicit
datapoint specifically so the blindness in the module docstring above is visible in this
experiment's own output rather than hidden by the choice of operating point.

## What "contributing" and "excluded" mean here, and why both are reported

An ablation configuration built with :meth:`~vpl.inverse.fusion.JointLikelihood.without`
has physically removed a channel: the joint likelihood built from it only ever had one
channel to ask, and :class:`~vpl.inverse.fusion.FusionDetail` reports that channel alone,
with nothing excluded. :func:`run_rp1_baseline`'s configuration is different in kind: it
*asks* for both channels and the fusion layer excludes one anyway, because the channel
reports itself below its detection floor (doc 01 IF-6). Reporting only ``contributing``
would make the two cases look identical; :class:`AblationResult` carries both fields
because doc 01 IF-6 requires "a configuration nominally containing two channels but running
on one" to say so, not just to run correctly.

## What a widened interval does and does not show

A wider credible interval and a larger relative error after dropping a channel is doc 11
§9 item 6's expected qualitative result, and it is the reason the second channel is worth
connecting at all (see ``channels.py``'s account of the ``n_0``-``T_e`` degeneracy a single
channel cannot break). It is not, by itself, evidence that the dropped channel's physics is
correctly modelled — the same caveats ``channels.py`` states about the reconstructed IVDF
and the independence assumption ``fusion.py`` states apply here unchanged, because this
module fuses through exactly those two objects rather than around them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.state import PlasmaState, SpatialGrid, Species
from vpl.core.units import magnitude_in
from vpl.experiment.channels import LIF_CHANNEL, OES_CHANNEL, build_channels
from vpl.experiment.closed_loop import (
    CREDIBLE_LEVEL,
    _argon_ion,
    _draw_true_theta,
    _fixed_spatial_grid,
    _gamma_e_interval,
    _reduced_prior,
    _reference_theta,
    _theta_from_unconstrained,
    _to_plasma_params,
)
from vpl.inverse.fusion import BlindChannelError, JointLikelihood
from vpl.inverse.map import maximum_a_posteriori
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import AnalyticSheathSolver, ion_energy_flux
from vpl.validation.sealed import SealedTruth, Tier

__all__ = [
    "DEFAULT_SEED",
    "REACHABLE_BIAS_V",
    "AblationResult",
    "render_table",
    "run_ablation",
    "run_rp1_baseline",
]

type FloatArray = NDArray[np.float64]

#: Wall bias the main ablation sweep runs at, in volts.
#:
#: **Not RP-1's -250 V.** ``vpl.experiment.channels``'s module docstring: a 250 V argon
#: sheath drives ions to 34.9 km/s, past the 20 GHz laser's 25.8 km/s mode-hop-free ceiling
#: (doc 01 §5.1, doc 14 RS-03), so ``LifInstrument.is_informative`` refuses the channel
#: outright there and every configuration in this sweep would be vacuously testing an
#: excluded channel. -100 V is the same value ``packages/vpl-experiment/tests/
#: test_channels.py`` runs at (its ``_REACHABLE_BIAS_V`` — inside the reachable window with
#: margin), reused verbatim here rather than re-derived so the two files' numbers are
#: directly comparable.
REACHABLE_BIAS_V: Final[float] = -100.0

#: The one recorded seed (doc 00 E3) :func:`run_ablation` and :func:`run_rp1_baseline`
#: default to. Matches ``test_channels.py``'s ``_SEED`` for the same comparability reason
#: as :data:`REACHABLE_BIAS_V`.
DEFAULT_SEED: Final[int] = 7

#: The tier every recovery in this module is committed at. Each configuration's truth is
#: generated by the same ``AnalyticSheathSolver`` (L0) the inversion uses, with noise and
#: imperfect calibration both on (``build_channels``'s defaults) — doc 05 §7.2's "same
#: model, with noise", i.e. T1. This experiment does not vary physics level between truth
#: and inversion; that mismatch is ``closed_loop.run_t2``'s question, not this one's.
_TIER: Final[Tier] = Tier.T1

#: Configuration labels, named so an ablation table and a test assertion cannot drift by
#: one of them being retyped slightly differently.
BASELINE_LABEL: Final[str] = "all channels"
WITHOUT_LIF_LABEL: Final[str] = f"without {LIF_CHANNEL}"
WITHOUT_OES_LABEL: Final[str] = f"without {OES_CHANNEL}"
RP1_LABEL: Final[str] = "RP-1 nominal (both requested)"


@dataclass(frozen=True, slots=True)
class AblationResult:
    """One ablation configuration's recovery, against the shared sealed truth.

    Attributes:
        label: Which configuration this is — see :data:`BASELINE_LABEL` and friends.
        contributing: Channel names that actually formed a likelihood term, evaluated at
            the truth state (doc 01 IF-6's "who is above their detection floor", read off
            :meth:`~vpl.inverse.fusion.JointLikelihood.detail`). For a
            :meth:`~vpl.inverse.fusion.JointLikelihood.without`-built configuration this is
            always every channel the configuration still names; for
            :func:`run_rp1_baseline` it can be fewer than the configuration nominally asks
            for — see the module docstring.
        excluded: Channel names the fusion layer declared blind at the truth state. Empty
            for every configuration in :func:`run_ablation`'s sweep; ``(lif,)`` for
            :func:`run_rp1_baseline`.
        gamma_e_true_w_per_m2: The sealed truth's ``Gamma_E`` — the same value for every
            configuration returned by one call to :func:`run_ablation`, since only the
            likelihood changes between them, not the truth.
        gamma_e_estimate_w_per_m2: ``Gamma_E(theta_hat)`` for this configuration's MAP
            point.
        relative_error: ``|estimate/truth - 1|`` — ``SealedTruth.relative_error``.
        interval_w_per_m2: The ``credible_level`` interval on ``Gamma_E``, or ``None`` when
            the Laplace posterior's Hessian was not positive definite (doc 05 §6's null
            space, ADR-012) — an absence reported rather than a number invented for it.
        interval_width_w_per_m2: ``interval[1] - interval[0]``, or ``None`` alongside it.
        truth_within_interval: Whether the sealed truth fell inside the interval, or
            ``None`` alongside it.
        wall_bias_v: The wall bias this configuration's truth and every trial theta were
            evaluated at.
        seed: The seed everything in this configuration derives from.
    """

    label: str
    contributing: tuple[str, ...]
    excluded: tuple[str, ...]
    gamma_e_true_w_per_m2: float
    gamma_e_estimate_w_per_m2: float
    relative_error: float
    interval_w_per_m2: tuple[float, float] | None
    interval_width_w_per_m2: float | None
    truth_within_interval: bool | None
    wall_bias_v: float
    seed: int


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Everything :func:`_recover` needs, built once and shared across configurations.

    A private bundle rather than a longer parameter list on every call: the same truth,
    grid and reference theta must be identical across a sweep's configurations for them to
    be comparable at all, and a bundle makes "these travel together" the type rather than a
    convention callers have to maintain by hand.
    """

    registry: ParameterRegistry
    species: Species
    solver: AnalyticSheathSolver
    grid: SpatialGrid
    reference: ControlParameters
    truth_state: PlasmaState
    truth_gamma_e: float
    joint: JointLikelihood
    wall_bias_v: float
    seed: int


def _prepare(
    *, seed: int, wall_bias_v: float | None, registry: ParameterRegistry | None
) -> _Prepared:
    """Build the shared truth and the full-channel joint likelihood for one sweep.

    Args:
        wall_bias_v: The bias every trial theta and the truth are evaluated at. ``None``
            means "RP-1's own bias, unmodified" — :func:`run_rp1_baseline`'s case — rather
            than a second, redundant way of spelling the same registry value that
            :func:`~vpl.experiment.closed_loop._reference_theta` already reads.
        registry: Parameter source. ``None`` defaults to
            :func:`~vpl.core.params.default_registry`.
    """
    resolved_registry = registry if registry is not None else default_registry()
    species = _argon_ion(resolved_registry)
    solver = AnalyticSheathSolver()

    # The fixed grid is sized from the *pure* RP-1 reference thickness, exactly as
    # ``test_channels.py`` does — never from the operating bias below, and never from the
    # truth. See closed_loop's own module docstring for the inverse-crime bug this
    # discipline exists to keep unreintroducible.
    rp1_reference = _reference_theta()
    grid = _fixed_spatial_grid(
        solver, reference=rp1_reference, species=species, registry=resolved_registry
    )

    operating_reference = (
        rp1_reference if wall_bias_v is None else rp1_reference.replace(V_w=wall_bias_v)
    )
    resolved_bias = float(operating_reference.V_w)

    reference_params = _to_plasma_params(rp1_reference, species=species, registry=resolved_registry)
    reference_state = solver.solve(reference_params, grid=grid)

    # doc 07 §3 step 1, reused rather than reimplemented: the same draw closed_loop's own
    # T1/T2 truths use, on the same Stream.EXPERIMENT_DESIGN stream, so this experiment's
    # truth is reproducible from `seed` alone exactly the way theirs is.
    true_theta = _draw_true_theta(seed=seed, reference=operating_reference)
    true_params = _to_plasma_params(true_theta, species=species, registry=resolved_registry)
    truth_state = solver.solve(true_params, grid=grid)
    truth_gamma_e = float(
        magnitude_in(ion_energy_flux(true_params, h_l=solver.h_l, gamma_i=solver.gamma_i), "W/m**2")
    )

    channel_set = build_channels(
        reference_state=reference_state, seed=seed, registry=resolved_registry
    )
    observations = channel_set.observe(truth_state)
    joint = channel_set.joint(observations)

    return _Prepared(
        registry=resolved_registry,
        species=species,
        solver=solver,
        grid=grid,
        reference=operating_reference,
        truth_state=truth_state,
        truth_gamma_e=truth_gamma_e,
        joint=joint,
        wall_bias_v=resolved_bias,
        seed=seed,
    )


def _recover(
    prepared: _Prepared, *, joint: JointLikelihood, label: str, credible_level: float
) -> AblationResult:
    """Run MAP and the ``Gamma_E`` interval for one configuration's likelihood.

    ``joint`` is passed separately from ``prepared.joint`` because an ablated
    configuration's likelihood is a *different* object
    (:meth:`~vpl.inverse.fusion.JointLikelihood.without` returns a copy) while everything
    else about the recovery — the truth, the grid, the reference theta, the seed — is
    shared, which is exactly what makes the configurations in one sweep comparable.
    """
    prior = _reduced_prior()

    def log_likelihood(u: FloatArray) -> float:
        theta = _theta_from_unconstrained(u, reference=prepared.reference)
        params = _to_plasma_params(theta, species=prepared.species, registry=prepared.registry)
        state = prepared.solver.solve(params, grid=prepared.grid)
        try:
            return joint.log_prob(state)
        except BlindChannelError:
            # doc 01 IF-6: no channel is above its detection floor at this trial point.
            # Not expected to fire in this module's own sweeps (the wall bias, the one
            # thing `is_informative` is sensitive to here, is fixed across every trial —
            # only n_0, T_e vary), but a trial that reaches it is reported to the optimiser
            # as out-of-support rather than allowed to crash the whole sweep; see
            # `vpl.inverse.map.negative_log_posterior`, which already treats a non-finite
            # likelihood this way.
            return -math.inf

    result = maximum_a_posteriori(log_likelihood=log_likelihood, prior=prior)
    theta_hat = _theta_from_unconstrained(result.unconstrained, reference=prepared.reference)
    params_hat = _to_plasma_params(theta_hat, species=prepared.species, registry=prepared.registry)
    estimate = float(
        magnitude_in(
            ion_energy_flux(params_hat, h_l=prepared.solver.h_l, gamma_i=prepared.solver.gamma_i),
            "W/m**2",
        )
    )

    # The seal/commit discipline, applied per configuration: `prepared.truth_gamma_e` is a
    # plain float already in scope above, but nothing in `log_likelihood` or the MAP call
    # touches it, and wrapping it here — rather than comparing the float directly — is what
    # makes that a checked property of this function rather than a habit.
    sealed = SealedTruth(value=prepared.truth_gamma_e, name="Gamma_E")
    sealed.commit_estimate(estimate, tier=_TIER)

    _, interval = _gamma_e_interval(
        result,
        log_likelihood=log_likelihood,
        prior=prior,
        reference=prepared.reference,
        species=prepared.species,
        registry=prepared.registry,
        inversion_solver=prepared.solver,
        seed=prepared.seed,
        credible_level=credible_level,
    )
    detail = joint.detail(prepared.truth_state)

    width = None if interval is None else float(interval[1] - interval[0])
    within = None if interval is None else bool(interval[0] <= float(sealed.value) <= interval[1])

    return AblationResult(
        label=label,
        contributing=detail.contributing,
        excluded=detail.excluded,
        gamma_e_true_w_per_m2=float(sealed.value),
        gamma_e_estimate_w_per_m2=float(sealed.estimate),
        relative_error=sealed.relative_error,
        interval_w_per_m2=interval,
        interval_width_w_per_m2=width,
        truth_within_interval=within,
        wall_bias_v=prepared.wall_bias_v,
        seed=prepared.seed,
    )


def run_ablation(
    *,
    seed: int = DEFAULT_SEED,
    wall_bias_v: float = REACHABLE_BIAS_V,
    credible_level: float = CREDIBLE_LEVEL,
    registry: ParameterRegistry | None = None,
) -> tuple[AblationResult, ...]:
    """doc 11 §9 item 6's sweep: recover against the full channel set, then each ablation.

    Runs three MAP recoveries against the *same* sealed truth and the *same* synthetic
    measurements — only the likelihood each is scored against changes. The baseline (both
    channels) is always first, so a reader can diff every later row against it directly.

    Args:
        seed: The seed everything (the truth draw, both channels' noise streams, the
            posterior sampler) derives from. Defaults to :data:`DEFAULT_SEED`.
        wall_bias_v: The bias to run at. Defaults to :data:`REACHABLE_BIAS_V`, at which LIF
            is genuinely informative — see the module docstring for why RP-1's own bias
            would make this sweep vacuous, and :func:`run_rp1_baseline` for that
            configuration reported honestly instead.
        credible_level: Central probability mass of the reported ``Gamma_E`` interval.
            Defaults to ``closed_loop.CREDIBLE_LEVEL``, the level doc 11 §9 item 5's
            coverage test uses.
        registry: Parameter source. Defaults to
            :func:`~vpl.core.params.default_registry`.

    Returns:
        ``(baseline, without_lif, without_oes)``, in that order.
    """
    prepared = _prepare(seed=seed, wall_bias_v=wall_bias_v, registry=registry)

    baseline = _recover(
        prepared, joint=prepared.joint, label=BASELINE_LABEL, credible_level=credible_level
    )
    without_lif = _recover(
        prepared,
        joint=prepared.joint.without(LIF_CHANNEL),
        label=WITHOUT_LIF_LABEL,
        credible_level=credible_level,
    )
    without_oes = _recover(
        prepared,
        joint=prepared.joint.without(OES_CHANNEL),
        label=WITHOUT_OES_LABEL,
        credible_level=credible_level,
    )
    return (baseline, without_lif, without_oes)


def run_rp1_baseline(
    *,
    seed: int = DEFAULT_SEED,
    credible_level: float = CREDIBLE_LEVEL,
    registry: ParameterRegistry | None = None,
) -> AblationResult:
    """RP-1's own wall bias, both channels nominally requested, no channel dropped.

    Not an ablation in the doc 11 §9 item 6 sense — see the module docstring. Its purpose
    is narrower and specific: to make ``vpl.experiment.channels``'s blindness finding a
    checked datapoint in *this* module's own output, so that :data:`REACHABLE_BIAS_V`'s
    choice for :func:`run_ablation` reads as a stated experimental design decision rather
    than a silent one. ``result.excluded == (lif,)`` here is the evidence.
    """
    prepared = _prepare(seed=seed, wall_bias_v=None, registry=registry)
    return _recover(prepared, joint=prepared.joint, label=RP1_LABEL, credible_level=credible_level)


def render_table(results: Sequence[AblationResult]) -> str:
    """A plain-text table of an ablation sweep — no plotting dependency, doc 08 §5's spirit
    of a result being readable in a CI log rather than requiring a viewer to open a figure.
    """
    headers = (
        "configuration",
        "contributing",
        "excluded",
        "rel. error",
        "CI width [W/m^2]",
        "covered",
    )
    rows = [
        (
            result.label,
            ",".join(result.contributing) or "-",
            ",".join(result.excluded) or "-",
            f"{result.relative_error:.4f}",
            "n/a"
            if result.interval_width_w_per_m2 is None
            else f"{result.interval_width_w_per_m2:.3e}",
            "n/a" if result.truth_within_interval is None else str(result.truth_within_interval),
        )
        for result in results
    ]
    columns = list(zip(*((headers, *rows)), strict=True)) if rows else [(h,) for h in headers]
    widths = [max(len(cell) for cell in column) for column in columns]

    def _format_row(cells: tuple[str, ...]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)
