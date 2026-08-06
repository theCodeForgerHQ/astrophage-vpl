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
    noise = "noise" if cell.noise else "clean"
    response = "est-resp" if cell.imperfect_calibration else "true-resp"
    return (
        f"{cell.truth.value}->{cell.inversion.value}/"
        f"{noise}/{response}/{calibration}/{cell.truth_eedf.value}"
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


def _truth_eedf_factory(shape: EedfShape) -> object:
    return _maxwellian_eedf_factory if shape is EedfShape.MAXWELLIAN else _t2_truth_eedf_factory


# ── the driver ──────────────────────────────────────────────────────────────────


def run_cell(
    cell: Cell,
    *,
    seed: int = 0,
    l2_truth_path: Path | str | None = None,
    registry: ParameterRegistry | None = None,
    credible_level: float = CREDIBLE_LEVEL,
    verbose: bool = False,
) -> CellReport:
    """doc 07 §3's protocol, steps 1-6, for one cell of the grid, through all four channels.

    Args:
        cell: The configuration. Its :attr:`~Cell.tier` is evaluated **before** any work,
            so a combination doc 05 §7.1 refuses costs nothing to discover.
        seed: The single recorded seed (doc 00 E3) the truth draw and every instrument
            realisation derive from.
        l2_truth_path: Required when ``cell.truth`` is ``L2``; ignored otherwise.
        credible_level: Central mass of the reported interval.

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
    )
    joint: JointLikelihood = channel_set.joint(channel_set.observe(truth_state))

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

    result: MapResult = maximum_a_posteriori(log_likelihood=log_likelihood, prior=prior)
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
        inversion_solves=forward.solves,
        wall_clock_s=time.perf_counter() - started,
        posterior=posterior,
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
