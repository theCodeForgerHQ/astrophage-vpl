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

:mod:`vpl.experiment.channels`'s own module docstring recorded the finding that motivated
this: **at RP-1's -250 V wall bias, LIF used to declare itself blind** (doc 01 §5.1, doc 14
RS-03 — a 20 GHz mode-hop-free laser range is a 25.8 km/s ceiling in ``v_z``, and a 250 V
argon sheath drives the free-fall speed *at the wall* to 34.9 km/s, past that ceiling).
:class:`~vpl.inverse.fusion.JointLikelihood` excluded LIF from the sum entirely (doc 01
IF-6), so "both channels" at RP-1 *was* "OES alone" — dropping LIF there would have changed
nothing, and an ablation figure built at that bias alone would have shown two identical
bars that read as "LIF carries no information", which is the opposite of what
:mod:`vpl.experiment.channels` found. That is why :func:`run_ablation` runs its main sweep
at :data:`REACHABLE_BIAS_V` rather than RP-1's own bias, and why :func:`run_rp1_baseline`
exists as a second, separate datapoint at RP-1 itself.

**RP-1's blindness, past tense.** ``vpl-instruments`` has since re-anchored
:meth:`~vpl.instruments.lif.instrument.LifInstrument._expected_ion_speed`'s tuning-range
gate at the sheath-edge Bohm speed (``c_s`` plus a margin of thermal widths) rather than the
free-fall speed across the *whole* sheath drop, and
:meth:`~vpl.instruments.lif.instrument.LifInstrument._resolve_z_index` now defaults the
measurement volume to the sheath edge to match — exactly the refinement
``channels.py``'s own module docstring named as the honest route to a fix ("a refinement of
that gate inside ``vpl-instruments``... not made here because this module does not own that
package"). The consequence for :func:`run_rp1_baseline`: LIF is now informative at RP-1
too, both channels contribute, and that function's row demonstrates a genuine two-channel
fusion at the operating point doc 01 §1.1 actually specifies as the deliverable — a more
useful datapoint than the blindness it used to report, not a less useful one.

**:data:`REACHABLE_BIAS_V` stays as the main sweep's operating point regardless**, for a
reason that has nothing to do with blindness any more: doc 11 §9 item 6's headline numbers
this module has already produced (the seed 7/11/23 table, the seed 23 ~650x interval
inflation, the LIF-likelihood coherent-calibration defect below) were all measured at -100
V, and moving the sweep to RP-1 now would silently invalidate every one of them rather than
extend the experiment — a reader comparing "the sweep" before and after this change would be
comparing two different experiments under one name. Re-pointing :func:`run_ablation` at
RP-1 now that it is reachable there too is a reasonable next step, but it is a deliberate
follow-up with its own fresh seed sweep, not a change folded into this pass.

## Why the LIF measurement volume does not move with theta here

``_resolve_z_index``'s new sheath-edge default (above) is state-dependent: left to resolve
itself, it locates the sheath edge in *whatever* ``PlasmaState`` it is handed, and across
the region a MAP search here actually explores that resolved index changes with the trial
``n_0``/``T_e`` — a **theta-dependent observation location**, the same class of inverse
crime ``closed_loop.py``'s own module docstring documents at length for the spatial grid,
where it let a trial `theta` improve its score by aligning grid points to the truth rather
than by being closer to it, and reported ``converged=True`` while doing so. This module
never hits that failure mode, and not by chance: :func:`~vpl.experiment.channels.
build_channels` always resolves an explicit ``z_index`` before configuring either LIF
instrument (:func:`~vpl.experiment.channels._lif_instrument`'s ``z_index`` argument), so
``settings.z_index is not None`` holds for every state this module's likelihoods ever
score and ``_resolve_z_index`` returns that one fixed index regardless of the state handed
to it. The pin happens once, in :func:`_prepare`, off the reference state — never per
trial — and every configuration in a sweep inherits it unchanged. Anyone tempted to
"simplify" a future version of this module by dropping the explicit index and letting
``LifInstrument`` resolve its own measurement volume per state would reopen exactly this
crime.

## What "contributing" and "excluded" mean here, and why both are reported

An ablation configuration built with :meth:`~vpl.inverse.fusion.JointLikelihood.without`
has physically removed a channel: the joint likelihood built from it only ever had one
channel to ask, and :class:`~vpl.inverse.fusion.FusionDetail` reports that channel alone,
with nothing excluded. :func:`run_rp1_baseline`'s configuration is different in kind: it
*asks* for both channels and lets the fusion layer decide, rather than pre-deciding by
construction — so an instrument reporting itself below its detection floor (doc 01 IF-6)
would show up there as an exclusion rather than an absent row. Both channels currently
contribute at RP-1 (see "RP-1's blindness, past tense" above), so today
``run_rp1_baseline().excluded == ()``; :class:`AblationResult` still carries the field
rather than reporting only ``contributing``, because doc 01 IF-6 requires "a configuration
nominally containing two channels but running on one" to say so *if it ever happens again*,
not just to run correctly while it does not.

## What a widened interval does and does not show

A wider credible interval and a larger relative error after dropping a channel is doc 11
§9 item 6's expected qualitative result, and it is the reason the second channel is worth
connecting at all (see ``channels.py``'s account of the ``n_0``-``T_e`` degeneracy a single
channel cannot break). It is not, by itself, evidence that the dropped channel's physics is
correctly modelled — the same caveats ``channels.py`` states about the reconstructed IVDF
and the independence assumption ``fusion.py`` states apply here unchanged, because this
module fuses through exactly those two objects rather than around them.

**Seed 23 is this sweep's clearest result, and it should be read first.** At
``REACHABLE_BIAS_V`` and ``seed=23``, dropping LIF (``without oes``'s complement — i.e. the
``all channels`` -> ``without lif`` step) inflates the ``Gamma_E`` interval from ~12.9 to
~8449 W/m^2, a ~650x widening, while the two-channel configuration holds it at ~12.9 and
the relative error at 0.008 against ``without lif``'s 0.069. That is the ``n_0``-``T_e``
ridge collapse doc 11 §9 item 3 predicts for an OES-only recovery, caught exactly where the
project's own diagnosis says it should be, and it is the sweep's single most direct
demonstration that connecting LIF was worth doing.

**Seeds 7 and 11 do not show the same direction for the *baseline*, and the cause is
measured, not guessed.** At those seeds the two-channel configuration's relative error and
interval width are both *worse* than one or both single-channel configurations, and
coverage fails for ``all channels`` in every seed tried (7, 11, 23). The cause is a defect
in :meth:`~vpl.instruments.lif.instrument.LifInstrument.likelihood`, not in this module or
in :mod:`vpl.inverse.fusion`: ``LifInstrument.observe`` draws **one** calibration scale and
applies it to every one of the 201 scan points — correct, and doc 06 §4.1's point that a
correlated calibration error "does not average down" — but ``LifInstrument.likelihood``
then divides by a **diagonal** per-point uncertainty with no coherent term, so that single
shared draw is charged as 201 independent residuals rather than one. Measured directly at
``seed=7``'s truth: the per-point residuals from that draw are constant across the scan to
machine precision (ratio standard deviation ~2e-16), the diagonal scoring gives
chi-squared ~3.0e-4, and the same draw scored as the one coherent degree of freedom it
actually is gives chi-squared ~1.5e-6 — an overweighting factor of ~201, matching the scan's
own point count exactly. Rerunning the ``all channels`` recovery at ``seed=7`` with
``build_channels(..., imperfect_calibration=False)`` — i.e. with that one draw removed —
drops the relative error from 0.0094 to 0.00027, confirming the direction: LIF's evidence
is entering the joint sum roughly 201x too strongly, so the fit chases that seed's
particular calibration draw instead of the physics.

This is fixable, and the fix is visible by comparison:
:meth:`~vpl.instruments.oes.instrument.OesInstrument.likelihood` already accepts a
``coherent_discrepancy`` argument that turns a diagonal Gaussian into the rank-one
correlated one a systematic error needs (see its own docstring for why the diagonal form
"average[s] down as ``1/sqrt(n)``" and under-inflates); ``LifInstrument.likelihood`` has no
equivalent parameter. That fix belongs in ``vpl-instruments``, which this module does not
own and does not touch — it is recorded here, at the ablation this defect distorts, rather
than attempted.
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


def without_label(channel: str) -> str:
    """The label for the row that drops ``channel``.

    One function rather than one constant per channel: the set grew from two to four and
    doc 11 §9 item 6 asks for a row per channel, so a per-name constant would have to be
    remembered and added every time a channel is connected — exactly the kind of omission
    that turns "we ablated every channel" into a claim nobody checked.
    """
    return f"without {channel}"


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
        The baseline first, then one leave-one-out row per channel in
        :data:`~vpl.experiment.channels.CHANNEL_NAMES` order.
    """
    prepared = _prepare(seed=seed, wall_bias_v=wall_bias_v, registry=registry)

    # One row per channel rather than the two this function was written with. doc 11 §9
    # item 6 asks for "drop *each* channel, show the CI inflate", and the channel set has
    # since grown from two to four; hard-coding two names would have quietly turned "each"
    # into "the two that existed when this was written" and silently stopped reporting the
    # ablation for Thomson and interferometry — the channels the exercise was for.
    results = [
        _recover(
            prepared, joint=prepared.joint, label=BASELINE_LABEL, credible_level=credible_level
        )
    ]
    results.extend(
        _recover(
            prepared,
            joint=prepared.joint.without(name),
            label=without_label(name),
            credible_level=credible_level,
        )
        for name in prepared.joint.names
    )
    return tuple(results)


def run_rp1_baseline(
    *,
    seed: int = DEFAULT_SEED,
    credible_level: float = CREDIBLE_LEVEL,
    registry: ParameterRegistry | None = None,
) -> AblationResult:
    """RP-1's own wall bias, both channels nominally requested, no channel dropped.

    Not an ablation in the doc 11 §9 item 6 sense — see the module docstring. Originally
    this function existed to make ``vpl.experiment.channels``'s RP-1 blindness finding a
    checked datapoint in *this* module's own output — ``result.excluded == (lif,)`` was the
    evidence, and it was what justified :data:`REACHABLE_BIAS_V` as a stated experimental
    design decision for :func:`run_ablation` rather than a silent one. See the module
    docstring's "RP-1's blindness, past tense" for why that is no longer what this function
    demonstrates: ``vpl-instruments`` has since re-anchored LIF's tuning-range gate at the
    sheath edge, both channels are informative at RP-1 now, and this row instead
    demonstrates a genuine two-channel fusion at the operating point doc 01 §1.1 specifies
    as the actual deliverable. :data:`REACHABLE_BIAS_V` remains :func:`run_ablation`'s
    operating point regardless, for the comparability reason the same module docstring
    section gives.
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
