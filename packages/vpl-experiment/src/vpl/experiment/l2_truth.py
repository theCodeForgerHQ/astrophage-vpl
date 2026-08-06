"""L2 as the truth, L1 as the inversion — the mismatch doc 05 §7 actually asks for.

:mod:`vpl.experiment.closed_loop`'s T2 generates the truth with L1
(:class:`~vpl.physics.fluid.sheath.FluidSheathSolver`) and inverts with L0
(:class:`~vpl.physics.analytic.sheath.AnalyticSheathSolver`). Measured with four channels
fused and calibration scored coherently, that configuration recovers ``Gamma_E`` to 10.38 %
against a +/-2.29 % interval, and the truth is **not** covered.

That residual has a diagnosis rather than a suspicion. At T1 — same model on both sides,
same four channels — the fit recovers ``n_0`` to 0.115 % and ``T_e`` to 0.008 %. The
parameters are identified; the channels are doing their job. So essentially all of the T2
error is **model error**: L0's collisionless Child-Langmuir closed form being bent to
explain data that a drift-diffusion Newton solve produced. No further channel can fix that,
because no channel is wrong — the forward model is.

The fix is to invert with a better model. But inverting with L1 against an L1 truth is the
inverse crime :mod:`vpl.validation.sealed` and doc 05 §7 exist to prevent, so the truth has
to move up instead:

    truth      = L2, :class:`~vpl.physics.kinetic.solver.Pic1D3VSolver`
    inversion  = L1, :class:`~vpl.physics.fluid.sheath.FluidSheathSolver`

Genuinely mismatched — particle-in-cell against a fluid closure — and a far smaller model
gap than L0-vs-L1, which is the whole point: this is the configuration that asks whether a
*good* forward model, fed by four channels that have already been shown to identify the
parameters, can produce a calibrated interval.

## What this costs, which is much less than "L2 is slow" suggests

doc 03 §4.4 prices a *production* RP-1 PIC run at 1 000 cells and 65 800 steps, and
DS-TRAIN's expense is 5 000 of them. This experiment needs **one**. The inversion then runs
L1 a few thousand times at ~65 ms per Newton solve.

The one solve is taken at the **verification scale** of
``packages/vpl-physics/tests/test_kinetic_solver.py``'s ``_sheath_solver`` — the settings
doc 03 §7's own gates (V-05 timestep independence, V-06 ``N_ppc`` convergence, V-07 energy
conservation, and the L0 cross-check) were measured at — rather than a scale invented here.
Reusing them means the truth's numerical error is already characterised by a suite that
exists, instead of being an open question this module would have to answer for itself. Every
constant below is that suite's, with **one** deliberate exception, stated next.

## The one departure: the domain, and why it is forced rather than chosen

``_sheath_solver`` runs at ``domain_sheaths=1.2``, which puts the bulk boundary a little past
the Child-Langmuir edge and keeps the ion transit time — and therefore the step count —
tractable. That domain is ~0.95 mm at the sealed truth. The closed loop's fixed observation
grid (``closed_loop._fixed_spatial_grid``, two *reference* sheath thicknesses) reaches
2.28 mm. Resampling a 0.95 mm solve onto a 2.28 mm grid would report through 1.3 mm of space
the physics never evaluated — ``numpy.interp`` clamps rather than extrapolates, so it would
do it quietly, which is the same class of failure as the theta-dependent grid
``closed_loop``'s module docstring documents at length.

So :data:`L2_DOMAIN_SHEATHS` is raised to the smallest round value that covers the
observation grid, and :func:`_require_domain_covers_observation_grid` checks the condition
**before** the solve rather than leaving it to ``closed_loop._resample_onto_grid`` to catch
afterwards. That guard is not defensive tidiness: the solve is an hour, and discovering the
mismatch at the end of it is the difference between a cheap failure and a wasted afternoon.

## The second departure: the velocity axis, which the LIF channel forces

An L2 state must carry its ion distribution — :class:`~vpl.core.state.PlasmaState` refuses
to be constructed without one at that fidelity, on the grounds that L2 is the only level
that produces a correct IEDF and a run that reaches the likelihood without it cannot answer
the question it was launched to answer. That is not a formality here: it means the LIF
channel finally sees a **computed** ``f_i`` rather than the drifting Maxwellian
:func:`~vpl.experiment.channels.reconstruct_ivdf` assumes at L0 and L1, which is the case
``channels.py``'s own docstring names as the point where LIF "stops being circular".

The solver's default velocity axis cannot deliver it. At this operating point the PIC's 160
bins span +/-47.1 km/s, i.e. **589 m/s per bin**, while
:func:`~vpl.instruments.lif.scan.fluorescence_response` integrates against a homogeneous
Lorentzian **41.3 m/s** wide in ``v_z`` and refuses any grid coarser than one sample per
width outright — correctly, since the quadrature would step over the resonance and
under-report the signal with no other symptom. So :data:`L2_VELOCITY_BINS` is raised to put
two samples inside that width. Measured: the extra bins cost **1.5 %** of the run's wall
clock, because :meth:`~vpl.physics.kinetic.solver.Pic1D3VSolver._record`'s deposit is
``O(particles)`` and not ``O(bins)``.

**The cost is statistical, not computational, and it is the reason for two rather than
four.** The inversion's own reconstructed axis uses four samples per homogeneous width
(``channels._VELOCITY_SAMPLES_PER_WIDTH``); matching it would halve the macroparticle count
per bin again, and a PIC histogram's shot noise is noise the truth then presents to the
inversion as physics. Two is the compromise: a factor of two of margin over the guard that
would otherwise refuse the run, and no more subdivision of a finite particle count than that
buys.

## The EEDF axis is inherited from T2, not strengthened

The truth's OES measurement is generated with ``closed_loop._t2_truth_eedf_factory``
(Druyvesteyn, ``kappa = 2``) against the inversion's Maxwellian, exactly as ``run_t2`` does
— the factory is passed explicitly, because ``build_channels`` defaults to the *matched*
Maxwellian and a four-channel run that forgot the argument would silently drop one of doc 05
§7.1's mandatory mismatch axes while still certifying at T2. ``closed_loop``'s own docstring
already states why that is a partial exercise of doc 05 §7.1's EEDF axis — both shapes come
from the same one-parameter family — and nothing here improves it. The PIC does not solve
electron kinetics for this level system, so it has no EEDF of its own to offer.

## What is genuinely new relative to `closed_loop.run_t2`

Physics level (L2 against L1 rather than L1 against L0), and the reported ``Gamma_E`` — the
estimate is now L1's enthalpy-flux functional evaluated at the MAP point, not L0's closed
form. Everything else is reused rather than reimplemented: the sealed-truth draw, the fixed
observation grid, the reduced ``(n_0, T_e)`` prior, the MAP engine, the Laplace posterior and
the four-channel assembly all come from :mod:`vpl.experiment.closed_loop`,
:mod:`vpl.experiment.channels` and :mod:`vpl.inverse` unchanged, so a change to any of them
propagates here instead of drifting away from it.

## Why the truth travels through a file

L2 needs JAX (:mod:`vpl.physics.kinetic.precision` turns on x64 at import and the kernel is a
correctness bug without it). L1 needs ``dolfinx``. There is no environment on this project's
machines that has both. :func:`solve_l2_truth` therefore runs where JAX is,
:func:`save_l2_truth` writes the state and its sealed ``Gamma_E``, and
:func:`run_l2_to_l1` reads it where ``dolfinx`` is. :func:`load_l2_truth` re-derives the
truth's parameters from the recorded seed and **refuses a file whose grid is not the fixed
observation grid**, so the artefact cannot be quietly paired with a different run.

### Exactly what crosses that boundary, because a hand-off is a channel

A file between the truth and the inversion is a route by which the inversion could see the
answer, which is the inverse crime doc 05 §7 exists to prevent. So the contents are listed
here and pinned by ``test_the_artefact_carries_exactly_these_keys``, which fails if a future
change adds a key rather than letting one arrive unnoticed:

* ``z_m`` — the observation grid, checked on load to *be* the fixed one.
* ``field_names`` / ``field_units`` / ``field_values`` — ``n_i``, ``n_e``, ``Phi``, ``u_i``,
  ``T_e`` on that grid. This is the state an instrument observes.
* ``v_m_per_s`` / ``f_i`` — the computed ion distribution, likewise observable.
* ``gamma_e_w_per_m2`` — the sealed quantity of interest. It goes straight into a
  :class:`~vpl.validation.sealed.SealedTruth` and is unreadable until an estimate has been
  committed, which is the barrier's whole job.
* ``mean_impact_energy_ev``, ``sheath_thickness_m``, ``energy_drift``, ``wall_clock_s`` —
  diagnostics about the *run*. Reported, never read by the inversion.
* ``seed`` — and this is the one that needs stating plainly.

**The seed reconstitutes the truth's parameters on the inversion machine.**
:func:`load_l2_truth` calls ``closed_loop._draw_true_theta`` to rebuild the
:class:`~vpl.core.state.PlasmaParams` a :class:`~vpl.core.state.PlasmaState` cannot be
constructed without, so ``truth.state.params`` on the dolfinx side really does hold the true
``n_0`` and ``T_e``. Serialising them explicitly instead would be strictly worse — a file
could then disagree with the draw it claims to come from — but either way they are present,
so what matters is where they are *used*:

* **Truth side, legitimately**: :meth:`~vpl.experiment.channels.ChannelSet.observe` generates
  the four synthetic measurements from the truth state, and
  :meth:`~vpl.inverse.fusion.JointLikelihood.detail` answers doc 01 IF-6's "was this channel
  above its floor" at the truth. ``closed_loop._run`` has ``true_params`` in scope for
  exactly the same two purposes; this is not a new exposure.
* **Inversion side, never**: :func:`run_l2_to_l1`'s ``log_likelihood`` builds every trial's
  parameters from ``closed_loop._reference_theta`` and the search vector alone, and
  :class:`_L1Forward` holds the solver, the species, the registry and the observation grid —
  no truth, no truth state, no truth parameters. What the likelihood sees of the truth is
  the :class:`~vpl.core.state.Measurement` objects inside ``joint``, which is data.

The seed is also what ``build_channels`` derives its noise and calibration streams from, so
one recorded number reproduces the truth draw *and* the instrument realisations. That is
``closed_loop``'s own arrangement (doc 00 E3), inherited rather than introduced here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.random import Stream, generator
from vpl.core.state import (
    Fidelity,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    Species,
    VelocityDistribution,
)
from vpl.core.units import magnitude_in
from vpl.experiment.channels import build_channels
from vpl.experiment.closed_loop import (
    CREDIBLE_LEVEL,
    _argon_ion,
    _draw_true_theta,
    _fixed_spatial_grid,
    _gamma_e_at_wall,
    _reduced_prior,
    _reference_theta,
    _resample_onto_grid,
    _t2_truth_eedf_factory,
    _theta_from_unconstrained,
    _to_plasma_params,
)
from vpl.inverse.fusion import BlindChannelError
from vpl.inverse.laplace import (
    LaplacePosterior,
    PosteriorNotPositiveDefiniteError,
    laplace_posterior,
)
from vpl.inverse.map import MapResult, maximum_a_posteriori
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import AnalyticSheathSolver, child_langmuir_thickness
from vpl.validation.sealed import SealedTruth, Tier, tier_of_configuration

if TYPE_CHECKING:
    # Neither import runs at module import time, and both are load-bearing absences rather
    # than tidiness: dolfinx (L1) is missing from the JAX machine and JAX (L2) is missing
    # from the dolfinx machine, so a module-level import of either would make this file
    # unimportable in exactly one of the two environments it has to work in — which is the
    # whole reason the truth travels through a file. Under `from __future__ import
    # annotations` every use below is a string, so neither ever runs for the interpreter.
    from vpl.physics.fluid.sheath import FluidSheathSolver
    from vpl.physics.kinetic.solver import Pic1D3VSolver

__all__ = [
    "L2_TO_L1_TIER",
    "DomainTooShortError",
    "L2ToL1Report",
    "L2Truth",
    "l2_domain_length_m",
    "l2_truth_solver",
    "load_l2_truth",
    "observation_grid",
    "run_l2_to_l1",
    "save_l2_truth",
]

type FloatArray = NDArray[np.float64]


# ── the L2 configuration ────────────────────────────────────────────────────────
#
# Every constant here except L2_DOMAIN_SHEATHS is `_sheath_solver`'s in
# `packages/vpl-physics/tests/test_kinetic_solver.py`. See the module docstring for why
# they are reused rather than chosen, and for the one that is not.

#: Macroparticles per cell per species — doc 03 §4.3's convergence knob, at the value
#: verification gate V-06 demonstrated the wall IEDF to be converged at.
L2_N_PPC: Final[float] = 400.0

#: ``lambda_D / dz``. doc 03 §4.3 bounds this below by 1 and this is the bound itself: a
#: cell wider than a Debye length is a :class:`~vpl.physics.kinetic.constraints.
#: StabilityViolationError`, not a warning.
L2_CELLS_PER_DEBYE: Final[float] = 1.0

#: Ion transit times simulated.
L2_TRANSITS: Final[float] = 6.0

#: Fraction of the run discarded before recording begins. With :data:`L2_TRANSITS` this is
#: three transits of equilibration, which the kinetic suite's own note records as the point
#: where ``Gamma_i``'s overshoot from the uniform load has drained to 0.8 % — against 26 %
#: after 1.2 transits, a quiet failure that leaves the run looking converged.
L2_EQUILIBRATION_FRACTION: Final[float] = 0.5

#: Particle slots as a multiple of the loaded population.
L2_CAPACITY_FACTOR: Final[float] = 1.5

#: Domain length in Child-Langmuir sheath thicknesses. **The one departure from the
#: verification suite**, which runs at 1.2. See the module docstring: 1.2 sheaths is
#: ~0.95 mm at the sealed truth and the fixed observation grid reaches 2.28 mm, so a 1.2
#: sheath solve would be read through 1.3 mm of space it never evaluated. 2.5 is the
#: smallest half-integer that covers it with margin, and :func:`_require_domain_covers_
#: observation_grid` checks the covering rather than trusting this number to stay right if
#: the observation grid ever moves.
L2_DOMAIN_SHEATHS: Final[float] = 2.5

#: Bins on the ``f_i(z, v_z)`` velocity axis. **The second departure** from the
#: verification suite, which leaves this at the solver's default of 160.
#:
#: Derived, not chosen. The axis spans ``+/-1.25 v_max``, measured at 94.2 km/s wide at the
#: sealed truth; :func:`~vpl.instruments.lif.scan.fluorescence_response` refuses a spacing
#: coarser than the 41.3 m/s homogeneous line width, and this experiment asks for two
#: samples inside that width, so ``94231.9 / (41.33 / 2) = 4560`` bins is the floor and this
#: is the next power of two above it. ``test_l2_truth.py`` checks the resulting spacing
#: against the instrument's own guard rather than trusting this arithmetic to stay true.
L2_VELOCITY_BINS: Final[int] = 4608

#: Root seed for the PIC streams. The kinetic suite's own ``_SEED`` (doc 00 E3), reused so
#: that a truth solved here and a gate run there are the same realisation of the same
#: sampler wherever their configurations coincide.
L2_SOLVER_SEED: Final[int] = 20260805

#: The tier this experiment may be reported at, decided by
#: :func:`~vpl.validation.sealed.tier_of_configuration` rather than asserted here. The three
#: booleans: the physics levels differ (L2 truth, L1 inversion), the truth instruments apply
#: their noise models, and the truth's measurement is generated through doc 04 §7.3's
#: *estimated* radiometric response rather than the true one.
L2_TO_L1_TIER: Final[Tier] = tier_of_configuration(
    same_model=False, noise=True, imperfect_calibration=True
)

#: Posterior draws pushed through L1's ``Gamma_E`` functional for the credible interval.
#:
#: ``closed_loop`` uses 20 000, which it can afford because L0's ``ion_energy_flux`` is
#: closed form. Here every draw is a Newton solve. 2 000 puts the Monte-Carlo standard error
#: on a 5 % quantile at about 3 % of the interval's own half-width — an order of magnitude
#: below the model discrepancy the interval is being asked to describe, and stated rather
#: than left for a reader to work out.
#:
#: **The cost claim that used to be here was wrong by a factor of 19, and the correction is
#: kept rather than quietly replaced.** It read "20 000 would be a twenty-minute interval",
#: implying ~65 ms per Newton solve. Measured on the FEniCSx machine at the RP-1 reference,
#: over five ``T_e`` factors spanning the sweep, every one converging: **1.24 s per solve**
#: (0.83-1.41 s). So 2 000 draws is ~41 minutes of interval on top of the inversion, not
#: two, and a full grid row costs about three quarters of an hour.
#:
#: The value is left at 2 000 because the binding error on this interval is not Monte-Carlo:
#: the four-channel T2 grid puts the truth 34 interval half-widths outside, so the interval
#: is wrong by a factor of tens for reasons no number of draws addresses. Cutting draws to
#: buy wall clock would trade a negligible error against a dominant one. It is recorded here
#: so the next person sizing a sweep starts from the measurement rather than from the guess.
L2_POSTERIOR_SAMPLES: Final[int] = 2000

#: Field name carrying the ion density, used only to check that a loaded artefact is a
#: plasma state rather than an arbitrary npz.
_REQUIRED_FIELD: Final[str] = "n_i"

_NPZ_SUFFIX: Final[str] = ".npz"

#: Relative tolerance the loaded grid must match the fixed observation grid to. Tight
#: because both sides are the same deterministic construction from the same registry; this
#: is a corruption check, not a physical tolerance.
_GRID_MATCH_RTOL: Final[float] = 1.0e-12


class DomainTooShortError(ValueError):
    """The L2 domain does not reach as far as the fixed observation grid.

    Its own type rather than a bare :class:`ValueError` because it is the one failure this
    module is arranged to catch *early*: an hour of PIC that ends in a resampling refusal
    costs the afternoon, and a caller that wants to widen the domain and retry needs to be
    able to tell this apart from every other way a solve can be misconfigured.
    """


def observation_grid(registry: ParameterRegistry | None = None) -> SpatialGrid:
    """The closed loop's fixed observation grid — one array of physical positions.

    Delegates to ``closed_loop._fixed_spatial_grid`` rather than rebuilding it, so this
    experiment reads its states at exactly the positions T0, T1 and T2 read theirs at. See
    ``closed_loop``'s module docstring for the inverse crime that fixed grid exists to make
    structurally impossible.
    """
    resolved = registry if registry is not None else default_registry()
    return _fixed_spatial_grid(
        AnalyticSheathSolver(),
        reference=_reference_theta(),
        species=_argon_ion(resolved),
        registry=resolved,
    )


def l2_truth_solver(**overrides: object) -> Pic1D3VSolver:
    """The one PIC configuration this experiment's truth is solved at.

    ``overrides`` exists for the tests that need a deliberately misconfigured solver (a
    domain too short for the observation grid, for instance) without duplicating the seven
    settings that are the point of this function.

    Raises:
        NotImplementedError: If JAX is missing. :mod:`vpl.physics.kinetic.precision` turns
            x64 on at import and states plainly that the kernel is a *correctness* bug in
            float32, so there is nothing to fall back to.
    """
    try:
        from vpl.physics.kinetic.solver import Pic1D3VSolver
    except ImportError as exc:
        raise NotImplementedError(
            "the truth-generating model for this experiment is vpl.physics.kinetic.solver."
            "Pic1D3VSolver (doc 03 §1's L2), which depends on JAX, and JAX is not "
            "installed here. Solve the truth where JAX is and invert against the saved "
            "artefact — see this module's docstring on why the two halves run in different "
            "environments."
        ) from exc

    defaults: dict[str, object] = {
        "n_ppc": L2_N_PPC,
        "cells_per_debye": L2_CELLS_PER_DEBYE,
        "domain_sheaths": L2_DOMAIN_SHEATHS,
        "transits": L2_TRANSITS,
        "equilibration_fraction": L2_EQUILIBRATION_FRACTION,
        "capacity_factor": L2_CAPACITY_FACTOR,
        "n_velocity_bins": L2_VELOCITY_BINS,
        "seed": L2_SOLVER_SEED,
    }
    return Pic1D3VSolver(**{**defaults, **overrides})  # type: ignore[arg-type]


def l2_domain_length_m(params: PlasmaParams, *, solver: Pic1D3VSolver | None = None) -> float:
    """How far into the plasma the PIC domain reaches, in metres.

    The same quantity :meth:`~vpl.physics.kinetic.solver.Pic1D3VSolver._plan` derives —
    ``domain_sheaths`` times doc 03 §2's Child-Langmuir thickness at the sheath-edge density
    ``h_l n_0`` — computed from the public
    :func:`~vpl.physics.analytic.sheath.child_langmuir_thickness` and the solver's own
    ``h_l``, so that a caller can ask what a run will cover without planning one.
    """
    resolved = solver if solver is not None else l2_truth_solver()
    thickness = float(magnitude_in(child_langmuir_thickness(params, h_l=resolved.h_l), "m"))
    return resolved.domain_sheaths * thickness


def _require_domain_covers_observation_grid(
    solver: Pic1D3VSolver, params: PlasmaParams, grid: SpatialGrid
) -> None:
    """Refuse a solve whose result could not honestly be read on ``grid``.

    ``closed_loop._resample_onto_grid`` already refuses this, and for exactly the right
    reason — ``numpy.interp`` clamps past the end of a solve's domain rather than
    extrapolating, so reporting through it is reporting a physical position the physics
    never evaluated. It refuses *after* the solve, which for L2 is an hour. This is the same
    condition asked of the plan.
    """
    reach = l2_domain_length_m(params, solver=solver)
    outermost = float(grid.z_m[-1])
    if reach < outermost:
        raise DomainTooShortError(
            f"the L2 domain reaches z = {reach:.6e} m at {solver.domain_sheaths} Child-"
            f"Langmuir sheath thicknesses, but the fixed observation grid reaches "
            f"z = {outermost:.6e} m. Resampling would clamp to the boundary value across "
            f"the remaining {outermost - reach:.6e} m instead of reporting physics that "
            "was solved there. Raise domain_sheaths."
        )


def _resample_l2_onto_grid(state: PlasmaState, grid: SpatialGrid) -> PlasmaState:
    """``closed_loop._resample_onto_grid``, carrying the ion distribution with it.

    That function cannot be reused for an L2 state, and the obstruction is deliberate on
    its side: it rebuilds the state with ``ion_distribution=None``, and
    :class:`~vpl.core.state.PlasmaState` refuses an L2 state without one. So the scalar
    fields are resampled here the same way — linear interpolation, exact for the
    piecewise-linear fields a converged solve effectively is at the resolution the
    instruments read it at — and ``f_i(z, v_z)`` is interpolated along ``z`` at fixed
    ``v_z``, which is the same operation applied to each velocity column.

    The velocity axis itself is **not** touched. Interpolating a histogram onto a finer
    axis would satisfy the LIF quadrature guard without adding any of the information the
    guard exists to require, which is the "plausible wrong number" failure ADR-011 records;
    the axis is made fine enough in the solver instead (see :data:`L2_VELOCITY_BINS`).

    Raises:
        ValueError: If ``grid`` reaches past the solve's own domain, for the reason
            ``closed_loop._resample_onto_grid``'s docstring gives at length — ``numpy.
            interp`` clamps there rather than extrapolating, so the state would report
            through a position the physics never evaluated.
    """
    distribution = state.ion_distribution
    if distribution is None:
        raise ValueError(
            "this resampler is for L2 states, which carry a computed ion distribution. "
            "Use closed_loop._resample_onto_grid for a state without one."
        )
    if state.time is not None:
        raise ValueError(
            "only steady states are handled here; doc 05 §3.2 forbids fusing channels "
            "whose windows differ by orders of magnitude across a transient."
        )
    if float(grid.z_m[-1]) > float(state.grid.z_m[-1]):
        raise ValueError(
            f"the fixed observation grid reaches z = {float(grid.z_m[-1]):.6e} m, past the "
            f"solve's domain of z = {float(state.grid.z_m[-1]):.6e} m. Resampling past the "
            "edge of a solution would clamp to the boundary value instead of reporting a "
            "physical position the solve evaluated."
        )

    source_z = np.asarray(state.grid.z_m, dtype=np.float64)
    target_z = np.asarray(grid.z_m, dtype=np.float64)
    fields = {
        name: ScalarField(
            name=field.name,
            values=np.interp(target_z, source_z, field.values),
            units=field.units,
            grid=grid,
            time=None,
        )
        for name, field in state.fields.items()
    }
    values = np.asarray(distribution.values, dtype=np.float64)
    resampled = np.stack(
        [np.interp(target_z, source_z, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )
    return PlasmaState(
        params=state.params,
        grid=grid,
        time=None,
        fields=fields,
        ion_distribution=VelocityDistribution(
            grid=grid,
            v_m_per_s=np.asarray(distribution.v_m_per_s, dtype=np.float64),
            values=resampled,
            species=distribution.species,
        ),
        fidelity=state.fidelity,
    )


# ── the truth artefact ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class L2Truth:
    """One L2 solve, reduced to what the L1 inversion needs and what a reader should check.

    Attributes:
        seed: The recorded seed (doc 00 E3) the truth ``theta`` was drawn from. Everything
            else about the truth is a deterministic function of it, which is what lets
            :func:`load_l2_truth` re-derive the parameters instead of serialising them.
        state: The PIC solution, resampled onto the fixed observation grid, carrying the
            computed ``f_i(z, v_z)`` the LIF channel reads — see the module docstring's
            second departure for the velocity resolution that requires.
        gamma_e_w_per_m2: The sealed quantity of interest, **accumulated from every
            macroparticle that crossed the wall** during the diagnostic window rather than
            derived from the gridded state.
            :meth:`~vpl.physics.kinetic.solver.Pic1D3VSolver.flux` refuses to reconstruct it
            from a state, which is why this is carried alongside rather than recomputed.
        mean_impact_energy_ev: ``<E_i>`` from the wall IEDF — doc 01 §1.2's second factor,
            recorded so that a surprising ``Gamma_E`` can be attributed to a flux or an
            energy rather than to "the PIC".
        sheath_thickness_m: Where doc 03 §2.1's ``(n_i - n_e) / n_i`` puts the sheath edge
            in the solved densities. A kinetic sheath should exceed Child-Langmuir; ADR-010
            predicts by less than its Boltzmann-electron quadrature ratio of 1.5029.
        energy_drift: Verification gate V-07's statistic for this run. Not gated here — V-07
            is stated for a closed box and this is an open one with an absorbing wall — but
            recorded, because a truth produced by a run that was haemorrhaging energy is
            worth knowing about before its number is quoted.
        wall_clock_s: What the solve cost, measured.
    """

    seed: int
    state: PlasmaState
    gamma_e_w_per_m2: float
    mean_impact_energy_ev: float
    sheath_thickness_m: float
    energy_drift: float
    wall_clock_s: float


def solve_l2_truth(
    *, seed: int = 0, registry: ParameterRegistry | None = None, verbose: bool = False
) -> L2Truth:
    """Solve the sealed truth once, with L2 — doc 07 §3 step 1 at the top of the hierarchy.

    The ``theta`` is ``closed_loop._draw_true_theta``'s, on ``Stream.EXPERIMENT_DESIGN``,
    unchanged: this experiment's truth is the *same* truth T2 draws at the same seed, which
    is what makes the L2->L1 and L1->L0 rows comparable at all.

    Args:
        seed: The recorded seed the truth ``theta`` is drawn from. Note that the PIC's own
            particle streams derive from :data:`L2_SOLVER_SEED`, not from this: the truth
            state and the Monte-Carlo realisation that resolves it are independent choices,
            and conflating them would make "rerun the same truth with more particles"
            impossible to express.
        registry: Parameter source.
        verbose: Print progress. One solve is an hour; a caller running it from a shell
            deserves to know it started.
    """
    resolved = registry if registry is not None else default_registry()
    species = _argon_ion(resolved)
    grid = observation_grid(resolved)
    true_theta = _draw_true_theta(seed=seed, reference=_reference_theta())
    true_params = _to_plasma_params(true_theta, species=species, registry=resolved)

    solver = l2_truth_solver()
    _require_domain_covers_observation_grid(solver, true_params, grid)

    if verbose:
        print(
            f"L2 truth: n_0 = {true_theta.n_0:.6e} m^-3, T_e = {true_theta.T_e:.6f} eV, "
            f"V_w = {true_theta.V_w} V; domain reaches "
            f"{l2_domain_length_m(true_params, solver=solver):.6e} m against an observation "
            f"grid reaching {float(grid.z_m[-1]):.6e} m",
            flush=True,
        )

    started = time.perf_counter()
    run = solver.run(true_params, None)
    wall_clock = time.perf_counter() - started

    return L2Truth(
        seed=seed,
        state=_resample_l2_onto_grid(run.state, grid),
        gamma_e_w_per_m2=_gamma_e_at_wall(run.flux),
        mean_impact_energy_ev=float(run.iedf.mean_energy_ev),
        sheath_thickness_m=float(run.sheath_thickness_m),
        energy_drift=float(run.diagnostics.energy_drift),
        wall_clock_s=wall_clock,
    )


def _true_params(seed: int, registry: ParameterRegistry) -> tuple[PlasmaParams, Species]:
    species = _argon_ion(registry)
    theta = _draw_true_theta(seed=seed, reference=_reference_theta())
    return _to_plasma_params(theta, species=species, registry=registry), species


def save_l2_truth(truth: L2Truth, path: Path | str) -> Path:
    """Write a truth to a compressed npz so a different environment can invert against it.

    Only the seed, the grid, the field arrays and the scalar diagnostics are stored. The
    ``PlasmaParams`` are **not**: they are a deterministic function of the seed, and
    re-deriving them on load means a file cannot disagree with the draw it claims to come
    from. Serialising them would create exactly that possibility for no benefit.
    """
    destination = Path(path).with_suffix(_NPZ_SUFFIX)
    names = tuple(sorted(truth.state.fields))
    distribution = truth.state.ion_distribution
    if distribution is None:
        raise ValueError(
            "an L2 truth carries a computed ion distribution and this one does not; "
            "writing it would produce an artefact load_l2_truth cannot rebuild a valid "
            "L2 PlasmaState from."
        )
    np.savez_compressed(
        destination,
        seed=np.asarray(truth.seed),
        z_m=np.asarray(truth.state.grid.z_m, dtype=np.float64),
        v_m_per_s=np.asarray(distribution.v_m_per_s, dtype=np.float64),
        f_i=np.asarray(distribution.values, dtype=np.float64),
        field_names=np.asarray(names),
        field_units=np.asarray([truth.state.field(name).units for name in names]),
        field_values=np.stack(
            [np.asarray(truth.state.field(name).values, dtype=np.float64) for name in names]
        ),
        gamma_e_w_per_m2=np.asarray(truth.gamma_e_w_per_m2),
        mean_impact_energy_ev=np.asarray(truth.mean_impact_energy_ev),
        sheath_thickness_m=np.asarray(truth.sheath_thickness_m),
        energy_drift=np.asarray(truth.energy_drift),
        wall_clock_s=np.asarray(truth.wall_clock_s),
    )
    return destination


def load_l2_truth(path: Path | str, *, registry: ParameterRegistry | None = None) -> L2Truth:
    """Read a truth back, re-deriving its parameters from the recorded seed.

    Raises:
        ValueError: If the stored grid is not the fixed observation grid this project reads
            every state on. A truth carrying a different ``z`` array would be compared,
            index by index, against trial predictions taken at different physical positions
            — the aliasing failure ``closed_loop``'s module docstring documents, arriving
            through a file instead of through a solver.
    """
    resolved = registry if registry is not None else default_registry()
    with np.load(Path(path), allow_pickle=False) as handle:
        contents = {key: handle[key] for key in handle.files}

    z_m = np.asarray(contents["z_m"], dtype=np.float64)
    expected = observation_grid(resolved)
    if z_m.shape != expected.z_m.shape or not np.allclose(
        z_m, expected.z_m, rtol=_GRID_MATCH_RTOL, atol=0.0
    ):
        raise ValueError(
            f"the stored truth is on a grid of {z_m.size} points reaching "
            f"{float(z_m[-1]):.6e} m, which is not this project's fixed observation grid "
            f"({expected.n_points} points reaching {float(expected.z_m[-1]):.6e} m). Every "
            "state in the closed loop is read at the same physical positions; a truth on a "
            "different array would be scored against trials taken somewhere else."
        )

    seed = int(contents["seed"])
    params, species = _true_params(seed, resolved)
    names = [str(name) for name in contents["field_names"]]
    units = [str(unit) for unit in contents["field_units"]]
    values = np.asarray(contents["field_values"], dtype=np.float64)
    if _REQUIRED_FIELD not in names:
        raise ValueError(
            f"the stored truth carries fields {names} and no {_REQUIRED_FIELD!r}; that is "
            "not a plasma state this experiment's channels can observe."
        )

    grid = SpatialGrid(z_m=z_m)
    fields = {
        name: ScalarField(name=name, values=values[index], units=unit, grid=grid, time=None)
        for index, (name, unit) in enumerate(zip(names, units, strict=True))
    }
    return L2Truth(
        seed=seed,
        state=PlasmaState(
            params=params,
            grid=grid,
            time=None,
            fields=fields,
            ion_distribution=VelocityDistribution(
                grid=grid,
                v_m_per_s=np.asarray(contents["v_m_per_s"], dtype=np.float64),
                values=np.asarray(contents["f_i"], dtype=np.float64),
                species=species,
            ),
            fidelity=Fidelity.L2,
        ),
        gamma_e_w_per_m2=float(contents["gamma_e_w_per_m2"]),
        mean_impact_energy_ev=float(contents["mean_impact_energy_ev"]),
        sheath_thickness_m=float(contents["sheath_thickness_m"]),
        energy_drift=float(contents["energy_drift"]),
        wall_clock_s=float(contents["wall_clock_s"]),
    )


# ── the L1 inversion ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class L2ToL1Report:
    """One L2-truth / L1-inversion recovery — doc 05 §10's "every inversion emits" table.

    Attributes:
        tier: :data:`L2_TO_L1_TIER`, certified by
            :func:`~vpl.validation.sealed.tier_of_configuration`.
        calibration_uncertainty: Whether the inversion-side likelihoods scored their own
            radiometric calibration as the *coherent* systematic doc 06 §4.1 says it is.
            The switch the four-channel exercise exists to make survivable; see
            :func:`~vpl.experiment.channels.build_channels`.
        contributing / excluded: doc 01 IF-6's account of which channels formed a
            likelihood term at the truth state.
        gamma_e_true_w_per_m2: L2's accumulated wall energy flux.
        gamma_e_estimate_w_per_m2: L1's enthalpy-flux functional at the MAP point.
        interval_w_per_m2: The ``credible_level`` interval, or ``None`` when the Laplace
            Hessian was not positive definite — doc 05 §6's null space (ADR-012), reported
            as an absence rather than replaced by a number.
        half_width_fraction: Half the interval width as a fraction of the estimate, which
            is the form the project's own comparison table is written in.
        truth_within_interval: The question the whole exercise turns on, read after the
            seal is committed.
    """

    tier: Tier
    calibration_uncertainty: bool
    contributing: tuple[str, ...]
    excluded: tuple[str, ...]
    gamma_e_true_w_per_m2: float
    gamma_e_estimate_w_per_m2: float
    relative_error: float
    n_0_hat_per_m3: float
    T_e_hat_ev: float
    interval_w_per_m2: tuple[float, float] | None
    half_width_fraction: float | None
    truth_within_interval: bool | None
    credible_level: float
    seed: int
    map_result: MapResult
    posterior: LaplacePosterior | None
    sealed: SealedTruth
    inversion_solves: int
    wall_clock_s: float


def _l1_solver() -> FluidSheathSolver:
    """L1, imported at call time.

    Deferred rather than hoisted for the same reason ``closed_loop.run_t2`` defers it:
    ``dolfinx`` is absent from the environment L2 runs in, and a module-level import would
    make :func:`solve_l2_truth` unusable there.
    """
    try:
        from vpl.physics.fluid.sheath import FluidSheathSolver
    except ImportError as exc:
        raise NotImplementedError(
            "the inversion model for this experiment is vpl.physics.fluid.sheath."
            "FluidSheathSolver (doc 03 §1's L1), which depends on dolfinx (FEniCSx) for "
            "its finite-element assembly, and dolfinx is not installed here. Inverting "
            "with L0 instead would silently turn this back into closed_loop.run_t2 — the "
            "configuration this experiment exists to move past — while still reporting a "
            "T2 label, so it refuses instead."
        ) from exc
    return FluidSheathSolver()


@dataclass(slots=True)
class _L1Forward:
    """The inversion's forward map, and a count of how often it was asked.

    The count is not bookkeeping for its own sake: L1 is ~65 ms per Newton solve against
    L0's microseconds, so "how many solves did that take" is the difference between this
    experiment being affordable and not, and a reported wall clock with no denominator
    cannot be checked against a future change.
    """

    solver: FluidSheathSolver
    species: Species
    registry: ParameterRegistry
    grid: SpatialGrid
    solves: int = 0

    def state(self, theta: ControlParameters) -> tuple[PlasmaState, PlasmaState]:
        """``(native, observed)`` — the solve on L1's own mesh, and it resampled.

        L1 is solved **unconstrained**, on the graded mesh its Newton solve needs, and the
        result is resampled onto the fixed observation grid afterwards. Passing the fixed
        grid into ``solve`` instead would under- or over-resolve the sheath depending on the
        trial ``theta``, which is the theta-dependent-grid inverse crime ``closed_loop``'s
        module docstring documents, relocated into L1's mesh generator. The truth is treated
        the same way, so the two sides differ in physics and not in numerics-by-accident.
        """
        params = _to_plasma_params(theta, species=self.species, registry=self.registry)
        self.solves += 1
        native = self.solver.solve(params)
        return native, _resample_onto_grid(native, self.grid)

    def gamma_e(self, theta: ControlParameters) -> float:
        """``Gamma_E`` at the wall from L1, on the mesh L1 actually solved.

        Read off the native state rather than the resampled one: the resampling exists for
        the *instrument's* field of view, and evaluating a physical functional on a
        deliberately coarse observation grid would put an interpolation error into the
        reported quantity of interest for no reason.
        """
        native, _ = self.state(theta)
        return _gamma_e_at_wall(self.solver.flux(native))


def run_l2_to_l1(
    truth: L2Truth,
    *,
    calibration_uncertainty: bool,
    credible_level: float = CREDIBLE_LEVEL,
    registry: ParameterRegistry | None = None,
    verbose: bool = False,
) -> L2ToL1Report:
    """Invert an L2 truth with L1, through all four channels — doc 07 §3 steps 2-6.

    Args:
        truth: The sealed L2 solve, from :func:`solve_l2_truth` or :func:`load_l2_truth`.
        calibration_uncertainty: Whether the inversion-side likelihoods score their own
            radiometric calibration coherently. Both settings are worth reporting: doc 00
            §5.1's criterion S4 says an uncalibrated posterior is worse than none, and the
            four-channel work measured that turning this on is what took T2 from 36.45 % to
            10.38 % — so a row with it off is the honest control, not a formality.
        credible_level: Central mass of the reported interval. Defaults to
            ``closed_loop.CREDIBLE_LEVEL``, the level doc 11 §9 item 5's coverage test is
            stated at.

    Raises:
        NotImplementedError: If ``dolfinx`` is missing. See :func:`_l1_solver`.
    """
    started = time.perf_counter()
    solver = _l1_solver()
    resolved = registry if registry is not None else default_registry()
    species = _argon_ion(resolved)
    reference = _reference_theta()
    grid = observation_grid(resolved)

    # The reference state that sizes the LIF scan, pins Thomson's measurement volume and
    # anchors the interferometer's chords. L0 at the RP-1 reference, exactly as
    # `ablation._prepare` builds it — a real instrument's field of view is fixed before the
    # plasma is characterised, so this must not come from the truth or from a trial.
    reference_state = AnalyticSheathSolver().solve(
        _to_plasma_params(reference, species=species, registry=resolved), grid=grid
    )

    channel_set = build_channels(
        reference_state=reference_state,
        seed=truth.seed,
        registry=resolved,
        noise=True,
        imperfect_calibration=True,
        calibration_uncertainty=calibration_uncertainty,
        truth_eedf_factory=_t2_truth_eedf_factory,
    )
    joint = channel_set.joint(channel_set.observe(truth.state))

    sealed = SealedTruth(value=truth.gamma_e_w_per_m2, name="Gamma_E")
    forward = _L1Forward(solver=solver, species=species, registry=resolved, grid=grid)
    prior = _reduced_prior()

    def log_likelihood(u: FloatArray) -> float:
        theta = _theta_from_unconstrained(u, reference=reference)
        try:
            _, observed = forward.state(theta)
        except (RuntimeError, ValueError):
            # A Newton solve that did not converge, or a mesh that could not be sized, at
            # this trial point. Reported to the optimiser as out-of-support rather than
            # allowed to end the run — the same treatment `vpl.inverse.map.
            # negative_log_posterior` already gives a non-finite likelihood.
            return -math.inf
        try:
            # `errstate` narrowly, around this one call, and paired with the finiteness
            # check below rather than used to look away.
            #
            # Far out in the tails of the search the OES prediction underflows to zero
            # expected counts, and `OesInstrument.likelihood`'s Gaussian branch then divides
            # the residual by it. numpy raises that as a RuntimeWarning, which under this
            # project's `-W error` test configuration ends the run — turning "the optimiser
            # tried an absurd theta and should be told so" into a crash. A theta at which
            # the instrument predicts no photons at all is **out of support**, and returning
            # -inf for it is exactly the treatment `vpl.inverse.map.negative_log_posterior`
            # already gives a non-finite likelihood. What is *not* done here is trusting the
            # number that came out of the overflow: anything non-finite is rejected.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                value = joint.log_prob(observed)
        except BlindChannelError:
            # doc 01 IF-6: nothing is above its detection floor here.
            return -math.inf
        return value if math.isfinite(value) else -math.inf

    result = maximum_a_posteriori(log_likelihood=log_likelihood, prior=prior)
    theta_hat = _theta_from_unconstrained(result.unconstrained, reference=reference)
    estimate = forward.gamma_e(theta_hat)

    posterior, interval = _l1_gamma_e_interval(
        result,
        log_likelihood=log_likelihood,
        prior=prior,
        reference=reference,
        forward=forward,
        seed=truth.seed,
        credible_level=credible_level,
        verbose=verbose,
    )

    sealed.commit_estimate(estimate, tier=L2_TO_L1_TIER)
    detail = joint.detail(truth.state)
    half_width = None if interval is None else 0.5 * (interval[1] - interval[0])

    return L2ToL1Report(
        tier=sealed.tier,
        calibration_uncertainty=calibration_uncertainty,
        contributing=detail.contributing,
        excluded=detail.excluded,
        gamma_e_true_w_per_m2=float(sealed.value),
        gamma_e_estimate_w_per_m2=float(sealed.estimate),
        relative_error=sealed.relative_error,
        n_0_hat_per_m3=theta_hat.n_0,
        T_e_hat_ev=theta_hat.T_e,
        interval_w_per_m2=interval,
        half_width_fraction=None if half_width is None else half_width / estimate,
        truth_within_interval=(
            None if interval is None else bool(interval[0] <= float(sealed.value) <= interval[1])
        ),
        credible_level=credible_level,
        seed=truth.seed,
        map_result=result,
        posterior=posterior,
        sealed=sealed,
        inversion_solves=forward.solves,
        wall_clock_s=time.perf_counter() - started,
    )


def _l1_gamma_e_interval(
    result: MapResult,
    *,
    log_likelihood: object,
    prior: object,
    reference: ControlParameters,
    forward: _L1Forward,
    seed: int,
    credible_level: float,
    verbose: bool,
) -> tuple[LaplacePosterior | None, tuple[float, float] | None]:
    """``closed_loop._gamma_e_interval`` with L1's functional in place of L0's.

    Not a call into that function, and the difference is the entire experiment: it pushes
    draws through ``analytic.sheath.ion_energy_flux``, the *inversion's* closed form at L0.
    Here the inversion is L1, so the interval has to be built from L1's own enthalpy flux or
    it would describe a model nobody used. Everything else — the Laplace posterior, the
    sampling rather than a delta-method linearisation, the ``None`` on a non-positive-
    definite Hessian — is that function's reasoning unchanged; see its docstring.
    """
    if not 0.0 < credible_level < 1.0:
        raise ValueError(f"credible level must lie strictly in (0, 1), got {credible_level}")

    try:
        posterior = laplace_posterior(
            result.unconstrained,
            log_likelihood=log_likelihood,  # type: ignore[arg-type]
            prior=prior,
        )
    except PosteriorNotPositiveDefiniteError:
        return None, None

    # Stream.SAMPLER (doc 10 §5), so these draws stay put when anything else in the run
    # consumes a different number of randoms — doc 00 E3's bit-for-bit requirement.
    draws = posterior.sample(generator(seed, Stream.SAMPLER), L2_POSTERIOR_SAMPLES)
    if verbose:
        print(f"pushing {L2_POSTERIOR_SAMPLES} posterior draws through L1", flush=True)
    fluxes = np.array(
        [forward.gamma_e(_theta_from_unconstrained(u, reference=reference)) for u in draws],
        dtype=np.float64,
    )
    tail = 0.5 * (1.0 - credible_level)
    low, high = np.quantile(fluxes, [tail, 1.0 - tail])
    return posterior, (float(low), float(high))
