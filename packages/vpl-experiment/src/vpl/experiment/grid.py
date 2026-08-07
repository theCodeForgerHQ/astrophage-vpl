"""The full tier x fidelity-pair grid, scored one way — doc 05 §7.2, doc 11 §9 items 4-6.

Three closed loops exist in this package and each answers one question well:
:func:`~vpl.experiment.closed_loop.run_t0` / ``run_t1`` / ``run_t2`` (L0 or L1 truth,
L0 inversion, **OES alone**) and :func:`~vpl.experiment.l2_truth.run_l2_to_l1` (L2 truth,
L1 inversion, four channels). They do not answer the question the project actually has to
put on a slide, which is *comparative*: how does the recovery move as the truth model, the
inversion model, the noise, the calibration and the channel set move.

They cannot answer it, because they do not score the same way. ``closed_loop`` fuses one
channel; ``l2_truth`` fuses four. Comparing 35.56 % from the first against 10.38 % from the
second and calling the difference "the fidelity gap" is an error of exactly the kind
ADR-011 was written about — the two numbers differ in the instrument set as well as in the
physics, and nothing in either report says which caused what.

This module is the fix. **One driver, one likelihood, one interval convention, every
cell.** A row differs from another row only in the axes the row names.

## What a cell is, and what the tier label is allowed to be

A :class:`Cell` is five booleans-and-enums: which model made the truth, which model does the
inverting, whether the truth instruments applied their noise models, whether they applied
doc 04 §7.3's *estimated* radiometric response, and whether the inversion-side likelihoods
score their own calibration as the coherent systematic doc 06 §4.1 says it is.

The tier is **not** one of them. It is
:func:`~vpl.validation.sealed.tier_of_configuration`'s verdict on the first three, and that
function refuses some combinations outright — a mismatched-model run without noise is not
T0, not T1 and not T2, and asking for one raises rather than returning a label. That refusal
is load bearing here: it is why the grid has no "T0 for L1-vs-L0" row. Mismatch the physics
and you are at T2 or you are nowhere. So the grid's shape is:

* **T0 and T1** come from the same-model pairs — L0 inverted with L0, L1 inverted with L1.
* **T2** comes from the mismatched pairs — L1 truth / L0 inversion, L2 truth / L0 inversion,
  L2 truth / L1 inversion.

L2 inverted with L2 is absent because it is not affordable rather than not interesting: one
L2 solve is 7 122 s measured (:mod:`vpl.experiment.l2_truth`), and an inversion asks for
thousands.

## The one deliberate departure from ``closed_loop``, stated because it moves T0

``closed_loop._run`` scores T0 with a *continuum* residual rather than with
``OesInstrument.likelihood``, because that likelihood rounds radiance-derived counts to
integers (``np.rint``) and at zero noise the quantisation residual can make the true
generating theta score marginally worse than a nearby impostor. Its reasoning is correct and
is preserved there.

It is **not** reproduced here, and the reason is the whole point of this module: a grid whose
rows are scored by different objectives is not a grid. Every cell below — T0 included — is
scored by the four channels' own likelihoods. The consequence is that this module's T0 is
not ``closed_loop.run_t0``'s number and must not be quoted as if it were; it carries the
quantisation floor that function deliberately steps around. T0's job here is doc 05 §7.2's
job — a blocking bug detector, an error orders of magnitude below any physical effect — and
a quantisation floor does not stop it doing that. The strict continuum T0 remains available
in ``closed_loop`` and both are reported.

## Why the truth for L2 is loaded rather than solved

An L2 truth is a 7 122 s PIC solve and it needs JAX; an L1 inversion needs dolfinx; the two
do not coexist in one environment here. :mod:`vpl.experiment.l2_truth` already solves that
by making the truth travel as a file, and this module inherits the arrangement rather than
inventing a second one — :func:`run_cell` takes a path and refuses an artefact whose seed
disagrees with the run's, because a truth from a different draw silently scored against this
run's instrument realisations would produce a number with no meaning and no symptom.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.random import Stream, generator
from vpl.core.state import Fidelity, PlasmaState, SpatialGrid, Species
from vpl.core.units import magnitude_in
from vpl.experiment.channels import ChannelSet, build_channels
from vpl.experiment.closed_loop import (
    CREDIBLE_LEVEL,
    _argon_ion,
    _draw_true_theta,
    _gamma_e_at_wall,
    _maxwellian_eedf_factory,
    _reduced_prior,
    _reference_theta,
    _resample_onto_grid,
    _t2_truth_eedf_factory,
    _theta_from_unconstrained,
    _to_plasma_params,
)
from vpl.experiment.discrepancy_basis import load_channel_discrepancy
from vpl.experiment.l2_truth import (
    L2_POSTERIOR_SAMPLES,
    L2Truth,
    _L1Forward,
    load_l2_truth,
    observation_grid,
)
from vpl.inverse.fusion import BlindChannelError, JointLikelihood
from vpl.inverse.laplace import (
    LaplacePosterior,
    PosteriorNotPositiveDefiniteError,
    laplace_posterior,
)
from vpl.inverse.map import MapResult, maximum_a_posteriori
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import AnalyticSheathSolver, ion_energy_flux
from vpl.validation.sealed import SealedTruth, Tier, tier_of_configuration

__all__ = [
    "STANDARD_CELLS",
    "Cell",
    "CellReport",
    "EedfShape",
    "Fidelity",
    "InfeasibleCellError",
    "TruthArtefactRequiredError",
    "cell_label",
    "run_cell",
    "run_grid",
]

type FloatArray = NDArray[np.float64]

#: Watt per square metre — the unit ``Gamma_E`` is reported in throughout the project.
_FLUX_UNIT: Final[str] = "W/m**2"

#: Posterior draws pushed through the inversion's ``Gamma_E`` functional when that
#: functional is **closed form** (L0). ``closed_loop``'s own value, reused rather than
#: rechosen so an L0-inversion row here and a ``closed_loop`` row describe intervals built
#: the same way.
_L0_POSTERIOR_SAMPLES: Final[int] = 20000

#: ``run_cell``'s default for :func:`~vpl.inverse.map.maximum_a_posteriori`'s ``n_starts``.
#: **One**, and the reason it is one is a measurement rather than a preference.
#:
#: ## Why multi-start looked right
#:
#: On the OES-alone, matched-physics (L0 truth = L0 inversion) 15-seed cell, single-start
#: landed near truth on 11/14 completed seeds and put a *confident, positive-definite*
#: interval on the wrong mode for 3/14 (n0/T_e ratios 0.221/9.61, 0.728/2.08, 0.227/8.90).
#: Five bounded starts rescued all three — 15/15 near truth, 0 non-positive-definite, 0
#: confidently wrong — and cut OES's measured overconfidence factor from 51.8x/305.5x
#: (n0/T_e) to 16.4x/11.1x. That measurement stands and is still the basis for OES's
#: likelihood weight.
#:
#: ## Why it is nevertheless wrong as a default here
#:
#: Setting this to 5 and re-running the **actual system** — L1 truth, L0 inversion, three
#: channels, ``ablate="oes"``, the configuration the headline number comes from — broke it:
#:
#: * seed 0: ``Gamma_E`` error **10 491 %**, n0 **+1 142 %**, T_e **+7 071 %**
#: * seed 3: ``Gamma_E`` error **3 365 %**, n0 **+323 %**, T_e **+6 502 %**
#: * seed 8: raised ``CoherentScatteringRegimeError`` (alpha=0.144, Thomson's incoherent
#:   regime requires alpha << 1) — a start outside the physics' validity, not merely
#:   outside the fit's basin
#:
#: The same seeds at ``n_starts=1`` give 1.28 % and 11.33 %.
#:
#: Two distinct causes, and neither is fixed by drawing starts more carefully:
#:
#: 1. **A misspecified likelihood has better optima than the truth.** At T2 the inversion
#:    model is *deliberately* wrong, so far from the truth the likelihood surface is not a
#:    noisy version of the right one — it has genuinely higher-scoring modes at absurd
#:    parameter values. Single-start never found them because it started near the reference
#:    and stayed local. Multi-start went looking and succeeded. It optimised the wrong
#:    objective more thoroughly, which is worse than optimising it badly.
#: 2. **Prior-valid is not physics-valid.** ``_bounded_perturbation`` rejects starts outside
#:    the prior's support, which is the only check a dimension-agnostic module can make. It
#:    cannot know that Thomson's spectrum model stops being defined at alpha ~ 0.1.
#:
#: So ``n_starts > 1`` is sound for a single channel against its own generative model, and
#: unsound as a system default at T2. It stays available per-call for diagnostics. Making it
#: safe here needs validity bounds owned by the physics, not by the optimiser — which is
#: real work, not a parameter change.
#:
#: The failure is also a caution about component benchmarks: a 15-seed single-channel test
#: reported "every failure mode eliminated" for a change that is catastrophic on 25 % of
#: system seeds, and it fails *silently* — those runs return ``converged=True`` with no
#: interval, so a mean taken over them would be meaningless rather than obviously wrong.
_DEFAULT_N_STARTS: Final[int] = 1


class EedfShape(Enum):
    """Which electron energy distribution the **truth's measurement** is generated with.

    doc 05 §7.1 lists the EEDF form as one of six mismatch axes, and it is the one most
    easily conflated with the physics-model axis, because both are usually flipped together
    when moving from T1 to T2. Naming it separately is what lets a row hold the solver fixed
    and move only the EEDF — the only way to attribute error to it rather than to the model.

    The inversion side is **always** Maxwellian, at every cell. doc 05 §7.1 asks the truth to
    differ from the inversion, not the inversion to be told how.
    """

    MAXWELLIAN = "maxwellian"
    DRUYVESTEYN = "druyvesteyn"


class InfeasibleCellError(NotImplementedError):
    """A cell this grid cannot run, distinguished from one that failed.

    Raised for L2-as-inversion (thousands of 7 122 s solves) and for any fidelity that has
    no solver here. A distinct type so a sweep can record "not attempted, and why" rather
    than silently omitting a row — an omitted row reads as a row that was never interesting.
    """


class TruthArtefactRequiredError(ValueError):
    """An L2-truth cell was asked for without a usable saved solve.

    Also raised when the artefact's recorded seed disagrees with the run's. That is not
    pedantry: the seed fixes the truth ``theta`` *and* every instrument's noise and
    calibration realisation, so scoring one draw's truth against another draw's instruments
    produces a relative error that means nothing and looks completely ordinary.
    """


@dataclass(frozen=True, slots=True)
class Cell:
    """One configuration of doc 05 §7.1's mismatch axes.

    Attributes:
        truth: Which model generates the sealed state and its ``Gamma_E``.
        inversion: Which model the optimiser runs inside the likelihood.
        noise: Whether the truth instruments apply their noise models — doc 05 §3.1.
        imperfect_calibration: Whether the truth's measurement is generated through doc 04
            §7.3's *estimated* radiometric response rather than the true one.
        calibration_uncertainty: Whether the **inversion-side** likelihoods score their own
            radiometric calibration as one coherent draw across all pixels rather than as N
            independent errors. Orthogonal to the tier — it changes the width of the
            interval, not what the run may be called. Measured on the four-channel L2->L1
            configuration, turning it on moved the error from 36.45 % to 10.38 %, so both
            settings are worth a row and the ``off`` row is the control rather than a
            formality.
        truth_eedf: See :class:`EedfShape`.
    """

    truth: Fidelity
    inversion: Fidelity
    noise: bool
    imperfect_calibration: bool
    calibration_uncertainty: bool
    truth_eedf: EedfShape
    #: Whether the inversion-side likelihoods carry a doc 05 §4 model-discrepancy term.
    #:
    #: Orthogonal to the tier, like ``calibration_uncertainty``: it changes how wide the
    #: interval is, not what the run may be called. Defaults ``False`` so every cell that
    #: existed before this field keeps its exact meaning and its measured numbers.
    #:
    #: Requires a saved basis (:mod:`vpl.experiment.discrepancy_basis`), because estimating
    #: one needs L1 and L1 needs dolfinx.
    model_discrepancy: bool = False

    @property
    def same_model(self) -> bool:
        return self.truth is self.inversion

    @property
    def tier(self) -> Tier:
        """The tier this cell may be reported at — the sealed module's verdict, not a field.

        Raises:
            TierMismatchError: For the combinations doc 05 §7.1 refuses. See the module
                docstring: this is why the grid has no mismatched-model T0 row.
        """
        return tier_of_configuration(
            same_model=self.same_model,
            noise=self.noise,
            imperfect_calibration=self.imperfect_calibration,
        )


def cell_label(cell: Cell) -> str:
    """A stable one-line name for a cell, for logs and for the results table.

    Deterministic and collision-free over the axes, so two runs of the same sweep produce
    rows that can be joined on it.
    """
    calibration = "cal-on" if cell.calibration_uncertainty else "cal-off"
    discrepancy = "disc-on" if cell.model_discrepancy else "disc-off"
    noise = "noise" if cell.noise else "clean"
    response = "est-resp" if cell.imperfect_calibration else "true-resp"
    return (
        f"{cell.truth.value}->{cell.inversion.value}/"
        f"{noise}/{response}/{calibration}/{discrepancy}/{cell.truth_eedf.value}"
    )


@dataclass(frozen=True, slots=True)
class CellReport:
    """doc 05 §10's "every inversion emits" table, for one cell.

    Attributes:
        cell: The configuration this describes, carried so a row cannot drift from its
            label.
        tier: :attr:`Cell.tier`, certified by the sealed module at commit time.
        contributing / excluded: doc 01 IF-6's account of which of the four channels formed
            a likelihood term at the truth state. A four-channel claim with two names in
            ``excluded`` is a two-channel result, and the only way to notice is to carry it.
        half_width_fraction: Half the interval width as a fraction of the estimate — the
            form the project's comparison table is written in.
        truth_within_interval: Read after the seal is committed. The one question that
            separates "a large error, honestly bracketed" from "an overconfident claim".
        interval_w_per_m2: ``None`` when the Laplace Hessian was not positive definite —
            doc 05 §6's null space (ADR-012), reported as an absence rather than replaced by
            a regularised number.
        inversion_solves: How many forward solves the inversion asked for. Bookkeeping at
            L0 and the affordability question at L1, where each is ~65 ms.
    """

    cell: Cell
    tier: Tier
    contributing: tuple[str, ...]
    excluded: tuple[str, ...]
    gamma_e_true_w_per_m2: float
    gamma_e_estimate_w_per_m2: float
    relative_error: float
    n_0_true_per_m3: float
    T_e_true_ev: float
    n_0_hat_per_m3: float
    T_e_hat_ev: float
    interval_w_per_m2: tuple[float, float] | None
    half_width_fraction: float | None
    truth_within_interval: bool | None
    credible_level: float
    seed: int
    map_converged: bool
    map_iterations: int
    inversion_solves: int
    wall_clock_s: float
    #: The Laplace posterior in unconstrained coordinates, or ``None`` alongside a ``None``
    #: interval. Carried rather than discarded because it is what an identifiability or
    #: sensitivity question needs afterwards, and re-deriving it means re-running the cell.
    posterior: LaplacePosterior | None
    #: Channels that actually carried a discrepancy term. Empty when the cell did not ask
    #: for one. Carried because "discrepancy applied" is a claim about *which* channels, and
    #: Thomson is excluded at RP-1 (see `_DISCREPANCY_INELIGIBLE_CHANNELS`) — a row that
    #: reported only a boolean would hide that a quarter of the correction is missing.
    discrepancy_channels: tuple[str, ...] = ()
    #: The channel dropped from the likelihood, or ``None`` for the full set.
    ablated: str | None = None
    #: The likelihood weight actually applied to each contributing channel, positionally
    #: aligned with :attr:`contributing`. All ``1.0`` for an unweighted run. Carried for the
    #: same reason :attr:`ablated` is: a down-weighted run and an unweighted one differ by
    #: nothing visible in the numbers themselves, so without this a weighted result and a
    #: baseline result are indistinguishable after the fact — and the whole point of the
    #: weighting is to be auditable rather than trusted.
    channel_weights: tuple[float, ...] = ()
    #: How many starts :func:`~vpl.inverse.map.maximum_a_posteriori` actually ran from.
    #: ``1`` for every row produced before ``run_cell(n_starts=...)`` existed, so an old row
    #: and a new single-start row remain indistinguishable in every other field, per
    #: :class:`~vpl.inverse.map.MapResult`'s own guarantee.
    map_n_starts: int = 1
    #: How many of those starts converged to a distinct mode — see
    #: :attr:`~vpl.inverse.map.MapResult.n_distinct_modes`. A value ``> 1`` is a per-row,
    #: checkable instance of "this cell's likelihood surface is multimodal", not merely a
    #: claim carried over from a separate audit.
    map_n_distinct_modes: int = 1

    @property
    def label(self) -> str:
        return cell_label(self.cell)


# ── the forward maps ────────────────────────────────────────────────────────────


class _Forward(Protocol):
    """What the driver needs of an inversion model, structurally.

    Two implementations: :class:`_L0Forward` here and
    :class:`~vpl.experiment.l2_truth._L1Forward`, reused rather than reimplemented so that
    an L2->L1 row produced by this module and one produced by ``l2_truth.run_l2_to_l1``
    cannot disagree about what L1 is.
    """

    solves: int

    def state(self, theta: ControlParameters) -> tuple[PlasmaState, PlasmaState]:
        """``(native, observed)`` — the solve on the model's own mesh, and it resampled."""
        ...

    def gamma_e(self, theta: ControlParameters) -> float:
        """``Gamma_E`` at the wall from this model, on the mesh it actually solved."""
        ...


@dataclass(slots=True)
class _L0Forward:
    """L0's closed form, wearing the same interface L1's Newton solve wears.

    ``native`` and ``observed`` are the same object here, and that is not a shortcut being
    hidden: :class:`~vpl.physics.analytic.sheath.AnalyticSheathSolver` evaluates a closed
    form at whatever abscissae it is handed, so solving *on* the fixed observation grid
    introduces no discretisation of its own and there is nothing to resample. The
    theta-dependent-grid inverse crime ``closed_loop``'s module docstring documents at
    length is a hazard of mesh *generators*, and L0 has none.
    """

    solver: AnalyticSheathSolver
    species: Species
    registry: ParameterRegistry
    grid: SpatialGrid
    solves: int = 0

    def state(self, theta: ControlParameters) -> tuple[PlasmaState, PlasmaState]:
        params = _to_plasma_params(theta, species=self.species, registry=self.registry)
        self.solves += 1
        solved = self.solver.solve(params, grid=self.grid)
        return solved, solved

    def gamma_e(self, theta: ControlParameters) -> float:
        params = _to_plasma_params(theta, species=self.species, registry=self.registry)
        return float(
            magnitude_in(
                ion_energy_flux(params, h_l=self.solver.h_l, gamma_i=self.solver.gamma_i),
                _FLUX_UNIT,
            )
        )


def _build_forward(
    fidelity: Fidelity,
    *,
    species: Species,
    registry: ParameterRegistry,
    grid: SpatialGrid,
) -> _Forward:
    if fidelity is Fidelity.L0:
        return _L0Forward(
            solver=AnalyticSheathSolver(), species=species, registry=registry, grid=grid
        )
    if fidelity is Fidelity.L1:
        # Imported at call time for the reason `l2_truth._l1_solver` gives: dolfinx is
        # absent from the machine L2 runs on, and a module-level import would make this
        # file unimportable there — which would take the L2->L0 rows down with it, and
        # those are exactly the rows that machine can produce.
        try:
            from vpl.physics.fluid.sheath import FluidSheathSolver
        except ImportError as exc:
            raise InfeasibleCellError(
                "an L1 inversion needs vpl.physics.fluid.sheath.FluidSheathSolver (doc 03 "
                "§1's L1), which depends on dolfinx (FEniCSx), and dolfinx is not "
                "importable here. Substituting L0 would silently relabel the row while "
                "still reporting an L1 inversion, so this refuses instead."
            ) from exc
        return _L1Forward(solver=FluidSheathSolver(), species=species, registry=registry, grid=grid)
    raise InfeasibleCellError(
        f"{fidelity.value} cannot be an inversion model here. One L2 solve is 7 122 s "
        f"measured (vpl.experiment.l2_truth) and an inversion asks for thousands; the "
        f"configuration is unaffordable rather than unimplemented, and doc 03 §1's L3 "
        f"surrogate — which exists to make it affordable — is not built."
    )


def _posterior_samples(fidelity: Fidelity) -> int:
    """Draws pushed through the interval, sized by what the functional costs.

    L0's ``ion_energy_flux`` is closed form, so 20 000 is free. Every L1 draw is a Newton
    solve, so :data:`~vpl.experiment.l2_truth.L2_POSTERIOR_SAMPLES` applies — see its
    docstring for the Monte-Carlo error that buys.
    """
    return _L0_POSTERIOR_SAMPLES if fidelity is Fidelity.L0 else L2_POSTERIOR_SAMPLES


# ── the truth side ──────────────────────────────────────────────────────────────


def _truth_state(
    cell: Cell,
    *,
    seed: int,
    species: Species,
    registry: ParameterRegistry,
    grid: SpatialGrid,
    l2_truth_path: Path | str | None,
) -> tuple[PlasmaState, float, ControlParameters]:
    """``(state on the observation grid, sealed Gamma_E, the true theta)``.

    The true ``theta`` is ``closed_loop._draw_true_theta``'s at every fidelity, so the L0,
    L1 and L2 rows at one seed describe the *same* plasma seen through three models. That
    is the property that makes the rows subtractable; without it the fidelity gap and the
    draw would be confounded.
    """
    reference = _reference_theta()
    true_theta = _draw_true_theta(seed=seed, reference=reference)
    true_params = _to_plasma_params(true_theta, species=species, registry=registry)

    if cell.truth is Fidelity.L0:
        solver = AnalyticSheathSolver()
        state = solver.solve(true_params, grid=grid)
        return state, _gamma_e_at_wall(solver.flux(state)), true_theta

    if cell.truth is Fidelity.L1:
        try:
            from vpl.physics.fluid.sheath import FluidSheathSolver
        except ImportError as exc:
            raise InfeasibleCellError(
                "an L1 truth needs dolfinx (FEniCSx), which is not importable here."
            ) from exc
        fluid = FluidSheathSolver()
        # Solved unconstrained on L1's own graded mesh and resampled afterwards, for the
        # reason `l2_truth._L1Forward.state` gives: handing the fixed grid to `solve` would
        # relocate the theta-dependent-grid problem into L1's mesh generator. The truth and
        # the inversion are treated identically here, so the two sides differ in physics
        # rather than in numerics-by-accident.
        native = fluid.solve(true_params)
        return _resample_onto_grid(native, grid), _gamma_e_at_wall(fluid.flux(native)), true_theta

    if l2_truth_path is None:
        raise TruthArtefactRequiredError(
            "an L2 truth cell needs `l2_truth_path`. Solving one here would be a 7 122 s "
            "PIC run inside a sweep, and it needs JAX, which does not coexist with dolfinx "
            "in this project's environments — vpl.experiment.l2_truth makes the truth "
            "travel as a file for exactly that reason. Produce one with "
            "`solve_l2_truth`/`save_l2_truth` and pass its path."
        )
    truth: L2Truth = load_l2_truth(l2_truth_path, registry=registry)
    if truth.seed != seed:
        raise TruthArtefactRequiredError(
            f"the L2 artefact at {l2_truth_path} records seed {truth.seed} and this run is "
            f"seed {seed}. The seed fixes the truth theta *and* every instrument's noise "
            f"and calibration realisation, so scoring one draw's truth against another "
            f"draw's instruments gives a relative error that means nothing and looks "
            f"entirely ordinary."
        )
    return truth.state, truth.gamma_e_w_per_m2, true_theta


#: Channels that cannot carry a discrepancy term at this operating point.
#:
#: Thomson only. :meth:`~vpl.instruments.thomson.instrument.ThomsonInstrument.likelihood`
#: switches to a Gaussian when a coherent discrepancy is supplied and refuses below 20
#: expected photoelectrons per channel, because a Poisson has no separate variance slot to
#: inflate and the normal approximation is poor in the tail. Measured on the RP-1 reference:
#: per-channel expected counts run 4.7e-8 to 303, with **13 of 20 channels below the
#: floor**. The 1111-count total is healthy; the line shape concentrates it and starves the
#: wings.
#:
#: So a four-channel discrepancy run corrects three channels. That is a real gap — Thomson's
#: model error goes unmodelled and the interval is correspondingly optimistic — and it is
#: named here and reported on every row rather than discovered later from a raised exception.
_DISCREPANCY_INELIGIBLE_CHANNELS: Final[frozenset[str]] = frozenset({"thomson"})


def _resolve_discrepancy(
    cell: Cell, path: Path | str | None
) -> tuple[dict[str, FloatArray] | None, tuple[str, ...]]:
    """``(mapping for build_channels, the channel names actually corrected)``.

    Returns ``(None, ())`` when the cell does not ask for a discrepancy, so a cell built
    before this axis existed takes a bit-for-bit unchanged path.

    Raises:
        TruthArtefactRequiredError: If a discrepancy is asked for without a saved basis.
            Estimating one needs L1, so this cannot be silently computed on demand
            wherever the inversion happens to be running.
    """
    if not cell.model_discrepancy:
        return None, ()
    if path is None:
        raise TruthArtefactRequiredError(
            "cell.model_discrepancy is set but no discrepancy_path was given. The basis is "
            "an L0-vs-L1 sweep and L1 needs dolfinx, so it cannot be estimated on demand "
            "here — produce one with vpl.experiment.discrepancy_basis."
            "estimate_channel_discrepancy / save_channel_discrepancy and pass its path."
        )
    basis = load_channel_discrepancy(path)
    eligible = {
        name: matrix
        for name, matrix in basis.items()
        if name not in _DISCREPANCY_INELIGIBLE_CHANNELS
    }
    if not eligible:
        raise TruthArtefactRequiredError(
            f"every channel in {path} is ineligible for a discrepancy term at this "
            f"operating point; the correction would be a no-op reported as applied."
        )
    return eligible, tuple(sorted(eligible))


def _truth_eedf_factory(shape: EedfShape) -> object:
    return _maxwellian_eedf_factory if shape is EedfShape.MAXWELLIAN else _t2_truth_eedf_factory


# ── the driver ──────────────────────────────────────────────────────────────────


def run_cell(
    cell: Cell,
    *,
    seed: int = 0,
    l2_truth_path: Path | str | None = None,
    discrepancy_path: Path | str | None = None,
    ablate: str | None = None,
    channel_weights: dict[str, float] | None = None,
    registry: ParameterRegistry | None = None,
    credible_level: float = CREDIBLE_LEVEL,
    verbose: bool = False,
    n_starts: int = _DEFAULT_N_STARTS,
) -> CellReport:
    """doc 07 §3's protocol, steps 1-6, for one cell of the grid, through all four channels.

    Args:
        cell: The configuration. Its :attr:`~Cell.tier` is evaluated **before** any work,
            so a combination doc 05 §7.1 refuses costs nothing to discover.
        seed: The single recorded seed (doc 00 E3) the truth draw and every instrument
            realisation derive from.
        l2_truth_path: Required when ``cell.truth`` is ``L2``; ignored otherwise.
        discrepancy_path: Required when ``cell.model_discrepancy`` is set; ignored
            otherwise. See :func:`_resolve_discrepancy`.
        ablate: Drop one channel by name from the joint likelihood — doc 11 §9 item 6's
            "drop each channel, show the CI inflate". The truth and its measurements are
            **unchanged**: exactly the same sealed state, the same synthetic data, the same
            seed. Only the set of terms summed over moves, which is what makes the
            difference between two rows attributable to the channel rather than to the
            draw. An unknown name raises (``KeyError`` from
            :meth:`~vpl.inverse.fusion.JointLikelihood.without`), because a typo would
            ablate nothing and produce two identical rows that read as "this channel
            carries no information" — the exact opposite of the truth.
        channel_weights: Per-channel likelihood weights, applied after ``ablate``. ``None``
            leaves every weight at 1.0, which is bit-for-bit the unweighted behaviour, so
            every number measured before weights existed stays comparable with every number
            measured after.

            This is the port for an error-budget audit's output. The measured failure at T2
            is that each channel understates its own uncertainty and fusion compounds the
            understatement ~50-fold, so the posterior narrows and stops covering the truth.
            Passing ``w = 1 / (understatement factor)**2`` inflates that channel's effective
            variance back to what it should have claimed — the power likelihood, exact
            algebra for a Gaussian rather than a fudge. See
            :class:`~vpl.inverse.fusion.JointLikelihood` for the derivation and its
            limitations.

            Applied *after* ablation, so weighting a channel that was just ablated raises
            rather than silently doing nothing.
        credible_level: Central mass of the reported interval.
        n_starts: Forwarded to :func:`~vpl.inverse.map.maximum_a_posteriori`. Defaults to
            :data:`_DEFAULT_N_STARTS` (``1``), matching that function's own default, so
            this argument changes nothing unless a caller asks it to.

            **Values above 1 are a single-channel diagnostic, not a system setting.** They
            are measurably harmful at T2 — see the constant's docstring for the run that
            produced ``Gamma_E`` errors of 10 491 % and 3 365 % on two of eight seeds, and
            a ``CoherentScatteringRegimeError`` on a third. The short version: with a
            deliberately mismatched inversion model, the likelihood has genuinely
            higher-scoring optima far from the truth, and a wider search finds them.

            The extra starts derive from this call's own ``seed`` and are bounded to the
            prior's support, so they stay reproducible — but prior-valid is not
            physics-valid, and nothing here checks the latter.

    Raises:
        TierMismatchError: For a configuration that has no tier — see :attr:`Cell.tier`.
        InfeasibleCellError: For a fidelity this environment or this project cannot run.
        TruthArtefactRequiredError: For an L2 cell without a matching saved solve.
    """
    started = time.perf_counter()
    # Evaluated first and deliberately: `tier_of_configuration` refuses mismatched-model
    # runs that skip noise or skip the estimated response, and discovering that after an
    # L1 truth solve would be an avoidable minute — or, on the L2 path, an avoidable load.
    tier = cell.tier
    # Resolved here, beside the tier check and for the same reason: a missing or unusable
    # discrepancy basis is a configuration error, and discovering it *after* an L1 truth
    # solve or an L2 artefact load would be an avoidable wait for an answer that was never
    # going to come.
    discrepancy, corrected_channels = _resolve_discrepancy(cell, discrepancy_path)

    resolved = registry if registry is not None else default_registry()
    species = _argon_ion(resolved)
    reference = _reference_theta()
    grid = observation_grid(resolved)

    # ── step 1: draw and generate the sealed truth ─────────────────────────────
    truth_state, gamma_e_true, true_theta = _truth_state(
        cell,
        seed=seed,
        species=species,
        registry=resolved,
        grid=grid,
        l2_truth_path=l2_truth_path,
    )

    # The reference state that sizes the LIF scan, pins Thomson's measurement volume and
    # anchors the interferometer's chords: L0 at the RP-1 *reference* parameters, never the
    # truth and never a trial. A real instrument's field of view is fixed before the plasma
    # is characterised, and letting any of it follow the truth would be the theta-dependent
    # observation the project has already been bitten by once.
    reference_state = AnalyticSheathSolver().solve(
        _to_plasma_params(reference, species=species, registry=resolved), grid=grid
    )

    # ── step 2: the forward chain -> one synthetic measurement per channel ─────
    channel_set: ChannelSet = build_channels(
        reference_state=reference_state,
        seed=seed,
        registry=resolved,
        noise=cell.noise,
        imperfect_calibration=cell.imperfect_calibration,
        calibration_uncertainty=cell.calibration_uncertainty,
        truth_eedf_factory=_truth_eedf_factory(cell.truth_eedf),  # type: ignore[arg-type]
        discrepancy=discrepancy,
    )
    joint: JointLikelihood = channel_set.joint(channel_set.observe(truth_state))
    if ablate is not None:
        joint = joint.without(ablate)
    if channel_weights:
        joint = joint.with_weights(channel_weights)

    # ── step 3: seal the truth ────────────────────────────────────────────────
    sealed = SealedTruth(value=gamma_e_true, name="Gamma_E")

    # ── step 4: invert ────────────────────────────────────────────────────────
    forward = _build_forward(cell.inversion, species=species, registry=resolved, grid=grid)
    prior = _reduced_prior()

    def log_likelihood(u: FloatArray) -> float:
        theta = _theta_from_unconstrained(u, reference=reference)
        try:
            _, observed = forward.state(theta)
        except (RuntimeError, ValueError):
            # A Newton solve that did not converge, or a mesh that could not be sized, at
            # this trial point: out of support, not a crash. Same treatment
            # `vpl.inverse.map.negative_log_posterior` already gives a non-finite value.
            return -math.inf
        try:
            # `errstate` narrowly, around this one call, and paired with the finiteness
            # check below rather than used to look away. Far out in the tails the OES
            # prediction underflows to zero expected counts and the Gaussian branch divides
            # by it; under this project's `-W error` configuration that ends the run,
            # turning "the optimiser tried an absurd theta" into a crash. What is *not* done
            # is trusting a number that came out of the overflow.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                value = joint.log_prob(observed)
        except BlindChannelError:
            # doc 01 IF-6: nothing is above its detection floor at this theta.
            return -math.inf
        return value if math.isfinite(value) else -math.inf

    result: MapResult = maximum_a_posteriori(
        log_likelihood=log_likelihood, prior=prior, n_starts=n_starts, seed=seed
    )
    theta_hat = _theta_from_unconstrained(result.unconstrained, reference=reference)
    estimate = forward.gamma_e(theta_hat)

    # ── step 4b: the interval, by pushing posterior draws through Gamma_E ──────
    posterior, interval = _gamma_e_interval(
        result,
        log_likelihood=log_likelihood,
        prior=prior,
        reference=reference,
        forward=forward,
        seed=seed,
        credible_level=credible_level,
        samples=_posterior_samples(cell.inversion),
        verbose=verbose,
    )

    # ── step 5: unseal, by committing the estimate at its tier ────────────────
    sealed.commit_estimate(estimate, tier=tier)

    # ── step 6: emit the row ──────────────────────────────────────────────────
    detail = joint.detail(truth_state)
    half_width = None if interval is None else 0.5 * (interval[1] - interval[0])
    return CellReport(
        cell=cell,
        tier=sealed.tier,
        contributing=detail.contributing,
        excluded=detail.excluded,
        channel_weights=detail.weights,
        gamma_e_true_w_per_m2=float(sealed.value),
        gamma_e_estimate_w_per_m2=float(sealed.estimate),
        relative_error=sealed.relative_error,
        n_0_true_per_m3=true_theta.n_0,
        T_e_true_ev=true_theta.T_e,
        n_0_hat_per_m3=theta_hat.n_0,
        T_e_hat_ev=theta_hat.T_e,
        interval_w_per_m2=interval,
        half_width_fraction=None if half_width is None else half_width / estimate,
        truth_within_interval=(
            None if interval is None else bool(interval[0] <= float(sealed.value) <= interval[1])
        ),
        credible_level=credible_level,
        seed=seed,
        map_converged=result.converged,
        map_iterations=result.iterations,
        map_n_starts=result.n_starts,
        map_n_distinct_modes=result.n_distinct_modes,
        inversion_solves=forward.solves,
        wall_clock_s=time.perf_counter() - started,
        posterior=posterior,
        discrepancy_channels=corrected_channels,
        ablated=ablate,
    )


def _gamma_e_interval(
    result: MapResult,
    *,
    log_likelihood: object,
    prior: object,
    reference: ControlParameters,
    forward: _Forward,
    seed: int,
    credible_level: float,
    samples: int,
    verbose: bool,
) -> tuple[LaplacePosterior | None, tuple[float, float] | None]:
    """The credible interval on ``Gamma_E``, through **the inversion's own** functional.

    ``Gamma_E`` is non-linear in several parameters at once (``Gamma_i ~ n_0 sqrt(T_e)``),
    so the exact endpoint transform :mod:`vpl.inverse.laplace` uses for a marginal does not
    apply and a delta-method linearisation would carry an error nobody tracks. Sampling is
    the honest route.

    The draws go through ``forward.gamma_e`` rather than through a fixed closed form, which
    is the difference between describing the model that was used and describing a different
    one. Returns ``(None, None)`` on a non-positive-definite Hessian — doc 05 §6's null
    space (ADR-012), a *result* about the configuration rather than a crash.
    """
    if not 0.0 < credible_level < 1.0:
        raise ValueError(f"credible level must lie strictly in (0, 1), got {credible_level}")

    try:
        posterior = laplace_posterior(
            result.unconstrained,
            log_likelihood=log_likelihood,  # type: ignore[arg-type]
            prior=prior,  # type: ignore[arg-type]
        )
    except PosteriorNotPositiveDefiniteError:
        return None, None

    # Stream.SAMPLER (doc 10 §5) so these draws stay put when anything else in the run
    # consumes a different number of randoms — doc 00 E3's bit-for-bit requirement.
    draws = posterior.sample(generator(seed, Stream.SAMPLER), samples)
    if verbose:
        print(f"pushing {samples} posterior draws through the inversion", flush=True)
    fluxes = np.array(
        [forward.gamma_e(_theta_from_unconstrained(u, reference=reference)) for u in draws],
        dtype=np.float64,
    )
    tail = 0.5 * (1.0 - credible_level)
    low, high = np.quantile(fluxes, [tail, 1.0 - tail])
    return posterior, (float(low), float(high))


# ── the standard grid ───────────────────────────────────────────────────────────


def _same_model(truth: Fidelity, *, noise: bool) -> Cell:
    """A T0 or T1 cell: one model on both sides, Maxwellian on both sides.

    ``imperfect_calibration`` is tied to ``noise`` here, which is ``closed_loop._run``'s own
    default and is what keeps a T0 row meaning "nothing is wrong" rather than "nothing is
    wrong except the lamp".
    """
    return Cell(
        truth=truth,
        inversion=truth,
        noise=noise,
        imperfect_calibration=noise,
        calibration_uncertainty=False,
        truth_eedf=EedfShape.MAXWELLIAN,
    )


def _mismatched(truth: Fidelity, inversion: Fidelity, *, calibration_uncertainty: bool) -> Cell:
    """A T2 cell: all three of doc 05 §7.1's mandatory mismatches, plus the EEDF axis."""
    return Cell(
        truth=truth,
        inversion=inversion,
        noise=True,
        imperfect_calibration=True,
        calibration_uncertainty=calibration_uncertainty,
        truth_eedf=EedfShape.DRUYVESTEYN,
    )


#: Every cell the project can currently produce, in reporting order.
#:
#: The three mismatched pairs each appear twice — once with the inversion asserting its
#: radiometric scale is exactly 1.0, once with it scoring that scale as the coherent 6 %
#: systematic doc 06 §4.1 says it is. The pair is the point: doc 00 §5.1's criterion S4
#: ("an uncalibrated posterior is worse than no posterior") is about the second, and the
#: first is the control that shows what the second cost or bought.
STANDARD_CELLS: Final[tuple[Cell, ...]] = (
    _same_model(Fidelity.L0, noise=False),
    _same_model(Fidelity.L0, noise=True),
    _same_model(Fidelity.L1, noise=False),
    _same_model(Fidelity.L1, noise=True),
    _mismatched(Fidelity.L1, Fidelity.L0, calibration_uncertainty=False),
    _mismatched(Fidelity.L1, Fidelity.L0, calibration_uncertainty=True),
    _mismatched(Fidelity.L2, Fidelity.L0, calibration_uncertainty=False),
    _mismatched(Fidelity.L2, Fidelity.L0, calibration_uncertainty=True),
    _mismatched(Fidelity.L2, Fidelity.L1, calibration_uncertainty=False),
    _mismatched(Fidelity.L2, Fidelity.L1, calibration_uncertainty=True),
)


def run_grid(
    cells: tuple[Cell, ...] = STANDARD_CELLS,
    *,
    seed: int = 0,
    l2_truth_path: Path | str | None = None,
    discrepancy_path: Path | str | None = None,
    registry: ParameterRegistry | None = None,
    credible_level: float = CREDIBLE_LEVEL,
    verbose: bool = True,
) -> tuple[tuple[Cell, CellReport | Exception], ...]:
    """Run every cell, returning a failure **in place of** a row rather than instead of one.

    A sweep that stops at the first infeasible cell reports nothing, and a sweep that drops
    it reports a grid with a hole that reads as a gap in interest rather than a gap in
    capability. So each cell yields either its :class:`CellReport` or the exception that
    prevented it, paired with the cell, and the caller decides what to print.
    """
    rows: list[tuple[Cell, CellReport | Exception]] = []
    for cell in cells:
        label = cell_label(cell)
        if verbose:
            print(f"[grid] {label} ...", flush=True)
        try:
            report = run_cell(
                cell,
                seed=seed,
                l2_truth_path=l2_truth_path,
                discrepancy_path=discrepancy_path,
                registry=registry,
                credible_level=credible_level,
                # Forwarded rather than left at its default: `run_cell`'s own flag drives
                # the within-cell progress line, and an L1 inversion spends minutes pushing
                # posterior draws with nothing printed. A silent sweep is indistinguishable
                # from a hung one.
                verbose=verbose,
            )
        # Recorded, not swallowed — see this function's docstring for why a sweep must not
        # stop at, nor silently drop, a cell it cannot run.
        except Exception as exc:
            if verbose:
                print(f"[grid] {label} -> {type(exc).__name__}: {exc}", flush=True)
            rows.append((cell, exc))
            continue
        if verbose:
            print(
                f"[grid] {label} -> tier={report.tier.name} "
                f"error={report.relative_error:.6f} "
                f"covered={report.truth_within_interval} "
                f"({report.wall_clock_s:.1f} s)",
                flush=True,
            )
        rows.append((cell, report))
    return tuple(rows)
