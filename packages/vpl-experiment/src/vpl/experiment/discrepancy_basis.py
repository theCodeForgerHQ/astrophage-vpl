"""Per-channel model-discrepancy bases, estimated once and carried as a file — doc 05 §4.

## The measurement that makes this necessary

The four-channel T2 grid (:mod:`vpl.experiment.grid`, commit ``0a0c52d``) recovers
``Gamma_E`` from an L2 truth to 6.47 % and reports a 90 % credible interval of +/-0.18 %.
The truth sits **34 interval half-widths** from the estimate. With the coherent calibration
term switched off it is 29.44 % against +/-0.14 % — 290 half-widths. Doc 00 §5.1's criterion
S4 is "an uncalibrated posterior is worse than no posterior", and by that criterion the
current posterior is worse than no posterior.

The cause is not in doubt, which is why a remedy is worth building rather than guessed at.
At T1 — same model on both sides, same four channels — the fit recovers the parameters to a
fraction of a percent. The channels are not the problem and the optimiser is not the
problem. What is left is **model error**: L0's closed-form Child-Langmuir sheath being bent
to explain data a particle-in-cell simulation produced. The likelihood currently describes
photon statistics and vibration and nothing else, so it has no way to express that the
forward model itself is wrong, and the posterior is precise about the wrong number.

Kennedy & O'Hagan (2001) call the missing object model discrepancy, and
:mod:`vpl.inverse.discrepancy` already implements it as a rank-``k`` covariance spanned by
the *observed* directions of model error. What was missing is the estimate itself, for all
four channels rather than for OES alone.

## Why this is a file rather than a function call

``closed_loop.estimate_t2_discrepancy`` exists and estimates the OES discrepancy inline. It
cannot be the whole answer here for a reason that is environmental rather than conceptual:
estimating an L0-vs-L1 discrepancy needs L1, L1 needs dolfinx, and dolfinx does not coexist
with the JAX the L2 truth needs. The L2 truth already travels between those environments as
an npz (:mod:`vpl.experiment.l2_truth`); this module reuses that arrangement rather than
inventing a second one, so an inversion running where L1 is unavailable can still apply a
discrepancy estimated where it was.

## What is estimated, and the inverse crime that is not committed

For each ``theta`` on a **deterministic** grid over the plausible ``(n_0, T_e)`` region:

    d_i(channel) = y_L1(theta_i)[channel] - y_L0(theta_i)[channel]

and the basis is ``{d_i / sqrt(N)}``, so that ``Sigma_disc = basis.T @ basis``.

The grid is deterministic and **not** drawn from a random stream, for the reason
``estimate_t2_discrepancy``'s docstring gives at length: sampling it on
``Stream.EXPERIMENT_DESIGN`` with the run's own seed would reproduce the factors
``closed_loop._draw_true_theta`` uses, and the discrepancy would end up estimated *at the
sealed truth's own parameters*. A grid cannot collide with a draw.

Neither model callable ever sees the truth. That is the structural guarantee
:func:`~vpl.inverse.discrepancy.estimate_empirical_discrepancy` is built around — it takes
two models and no truth argument — and it is inherited here rather than re-argued.

## The optimism this version removes, and the one it does not

``estimate_t2_discrepancy``'s docstring records a stated optimism: at the old T2 the truth
*was* L1, and the discrepancy was estimated against that same L1, so the correction was
matched to the actual misspecification in a way no laboratory could arrange. It names the
honest next step explicitly — "generate the truth with L2 while still estimating the
discrepancy from L0-vs-L1, making the estimate a genuine proxy".

That is exactly the configuration this module is built for. Against an **L2** truth, an
L0-vs-L1 discrepancy is a genuine proxy: it measures how wrong L0 is about a model that is
itself not the truth, which is what a real deployment would have to do.

What remains optimistic, and is not fixed here: reality is not L2 either. A proxy estimated
one rung down the ladder is still a proxy, and this module claims only that the mechanism is
honest, not that the magnitude generalises to a laboratory.

## The instruments are held in their clean configuration on purpose

The channels are built with ``noise=False`` and ``imperfect_calibration=False``, so the
truth-side and inversion-side instruments are the same deterministic map and the difference
between two observations is **only** the difference between the two physics models. Leaving
noise on would fold a Monte-Carlo realisation into a quantity that is supposed to be a
property of the models; leaving the estimated response on would fold in a calibration error
that the likelihood already scores separately, and counting it twice would over-inflate the
posterior in exactly the direction the coherent calibration term already covers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.state import PlasmaParams, PlasmaState, SpatialGrid, Species
from vpl.experiment.channels import CHANNEL_NAMES, build_channels
from vpl.experiment.closed_loop import (
    _argon_ion,
    _reference_theta,
    _resample_onto_grid,
    _to_plasma_params,
)
from vpl.experiment.l2_truth import observation_grid
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import AnalyticSheathSolver

__all__ = [
    "DISCREPANCY_GRID",
    "DISCREPANCY_SPAN",
    "ChannelDiscrepancy",
    "estimate_channel_discrepancy",
    "load_channel_discrepancy",
    "save_channel_discrepancy",
]

type FloatArray = NDArray[np.float64]

#: A per-channel rank-``k`` basis, keyed by the channel names
#: :data:`~vpl.experiment.channels.CHANNEL_NAMES` uses. Each value has shape ``(k, n)`` where
#: ``n`` is that channel's flattened observable length, and ``Sigma_disc = basis.T @ basis``.
type ChannelDiscrepancy = dict[str, FloatArray]


class _UnconstrainedSolver(Protocol):
    """A model that solves on its own mesh, given only parameters.

    Structural rather than an import of :class:`~vpl.physics.fluid.sheath.FluidSheathSolver`
    so that the shape, rank and serialisation plumbing here can be exercised where dolfinx
    is absent, by handing :func:`estimate_channel_discrepancy` a stand-in. A stand-in is for
    plumbing only: a physics stand-in that differs from L0 by a *parameter shift* has its
    error directions parallel to the parameter sensitivities, which is precisely the thing a
    real discrepancy is measured **not** to be parallel to — see
    :class:`~vpl.inverse.discrepancy.EmpiricalDiscrepancy` and the 1.66x that
    non-parallelism cost when it was assumed away.
    """

    def solve(self, params: PlasmaParams) -> PlasmaState: ...


#: Multiplicative span around the reference ``(n_0, T_e)`` the sweep covers.
#:
#: ``closed_loop._DISCREPANCY_SPAN``'s range, reused rather than rechosen so that an OES
#: basis from that function and an OES basis from this one describe the same region. It is
#: wide enough to contain the reduced prior's bulk without reaching parameters at which L1's
#: Newton solve is known to struggle — the sweep is only useful if every point in it
#: converges, and a failed corner would silently shorten the basis.
DISCREPANCY_SPAN: Final[tuple[float, float]] = (0.8, 1.2)

#: Points per axis. The basis has rank ``DISCREPANCY_GRID ** 2``, so this is quadratic in
#: cost and in rank; 5 gives rank 25, which spans the observed error directions while
#: staying at or below every channel's observable length after :func:`_compress_to_rank`
#: re-expresses it. Channel dimensions measured at the reference: OES 11 x 1024, LIF 201,
#: Thomson 20, interferometry **1** — that last one is a single scalar phase, not the 8
#: chords doc 02 §8.2's ladder describes, and it is why the compression rule is min(k, n)
#: rather than anything that subtracts.
DISCREPANCY_GRID: Final[int] = 5

#: npz key prefix for a channel's basis.
_BASIS_KEY: Final[str] = "basis__"

#: npz key holding the channel names, so a loaded file cannot disagree about which channels
#: it describes.
_NAMES_KEY: Final[str] = "channel_names"

_NPZ_SUFFIX: Final[str] = ".npz"


def _sweep_thetas(reference: ControlParameters, grid_points: int) -> tuple[ControlParameters, ...]:
    """The deterministic ``(n_0, T_e)`` grid. See the module docstring for why not a draw."""
    if grid_points < 2:
        raise ValueError(
            f"a discrepancy sweep needs at least 2 points per axis, got {grid_points}; with "
            f"one the basis is a single direction and the estimate carries no information "
            f"about how the model error varies across the region"
        )
    factors = np.linspace(*DISCREPANCY_SPAN, grid_points)
    return tuple(
        reference.replace(n_0=reference.n_0 * fn, T_e=reference.T_e * ft)
        for fn in factors
        for ft in factors
    )


def _compress_to_rank(basis: FloatArray) -> FloatArray:
    """Re-express a ``(k, n)`` basis in at most ``min(k, n)`` rows, **losslessly**.

    ``Sigma_disc = basis.T @ basis`` has rank at most ``min(k, n)``, so when ``k > n`` the
    sweep's rows are necessarily linearly dependent and carry the same covariance in fewer
    directions. An SVD re-expression with the singular values folded back into the retained
    rows reproduces ``basis.T @ basis`` **exactly** — this discards redundancy, not
    information. When ``k <= n`` it is a no-op and the input is returned unchanged.

    ## Why this is compression rather than truncation, corrected from a first attempt

    This function originally kept only ``n - 1`` directions, on the theory that a *full-rank*
    discrepancy covariance would de-weight every direction at once and silently mute the
    channel. That reasoning was wrong twice over, and both errors are worth recording because
    both are tempting.

    First, a full-rank discrepancy is not pathological — it is the normal case. Kennedy &
    O'Hagan model discrepancy with a Gaussian process, whose covariance is full rank by
    construction. Adding it down-weights each direction *in proportion to how wrong the model
    is along it*, which is the intended behaviour, not a failure of it.

    Second, and fatally: doc 02 §8.2's interferometer ladder is 8 chords, but the channel's
    **observable** at this operating point is a single scalar phase — measured, ``(1,)``. A
    keep-``n-1`` rule gives ``min(25, 0) = 0`` rows, an empty basis, and therefore **no
    discrepancy at all** for that channel, applied silently while the run continues to report
    itself as discrepancy-corrected. That is precisely the class of quiet omission the rest of
    this module is built to refuse, reintroduced by the guard meant to prevent it.

    So the rule is ``min(k, n)``: interferometry keeps one direction carrying its scalar model
    error, OES keeps all 25 of a 25-row sweep over 1024 samples, and nothing is dropped
    anywhere.
    """
    k, n = basis.shape
    keep = min(k, n)
    if keep >= k:
        return basis
    # full_matrices=False: only the first min(k, n) singular directions exist anyway, and
    # asking for the full V would allocate an n x n array — 1024 x 1024 on OES, for nothing.
    _, singular_values, vt = np.linalg.svd(basis, full_matrices=False)
    return np.asarray(singular_values[:keep, None] * vt[:keep, :], dtype=np.float64)


def _states(
    theta: ControlParameters,
    *,
    species: Species,
    registry: ParameterRegistry,
    grid: SpatialGrid,
    low_solver: AnalyticSheathSolver,
    high_solver: _UnconstrainedSolver,
) -> tuple[PlasmaState, PlasmaState]:
    """``(L0 state, L1 state)`` at one ``theta``, both on the fixed observation grid.

    L0 is evaluated directly on the observation grid — it is a closed form and has no mesh
    of its own. L1 is solved **unconstrained** on the graded mesh its Newton solve needs and
    resampled afterwards, which is ``l2_truth._L1Forward.state``'s discipline: handing the
    fixed grid to ``solve`` would put a theta-dependent mesh into the estimate, and the
    difference between the two models would then partly be a difference between two
    discretisations.
    """
    params: PlasmaParams = _to_plasma_params(theta, species=species, registry=registry)
    low = low_solver.solve(params, grid=grid)
    native = high_solver.solve(params)
    return low, _resample_onto_grid(native, grid)


def estimate_channel_discrepancy(
    *,
    seed: int = 0,
    registry: ParameterRegistry | None = None,
    high_solver: _UnconstrainedSolver | None = None,
    grid_points: int = DISCREPANCY_GRID,
    verbose: bool = False,
) -> ChannelDiscrepancy:
    """Sweep the grid and collect each channel's model-error directions.

    Args:
        seed: Only the seed the channels are *constructed* with. With ``noise=False`` and
            ``imperfect_calibration=False`` no stream is consumed for a realisation, so this
            does not enter the estimate; it is passed because ``build_channels`` requires
            one and because pinning it keeps the construction reproducible.
        grid_points: Points per axis of the sweep, giving a rank-``grid_points ** 2``
            basis. Defaults to :data:`DISCREPANCY_GRID`. Exposed so a test can exercise the
            machinery cheaply without monkeypatching a module global — a patched global is
            invisible at the call site and outlives the intent of whoever set it.
        high_solver: The higher-fidelity model. Defaults to L1
            (:class:`~vpl.physics.fluid.sheath.FluidSheathSolver`). Injectable so that the
            shape and rank plumbing can be exercised where dolfinx is absent — see
            ``tests/test_discrepancy_basis.py`` — **not** so that a production estimate can
            quietly be taken against something other than L1.

    Raises:
        NotImplementedError: If dolfinx is missing and no ``high_solver`` was supplied.
        ValueError: If any channel produces a non-finite basis, or if the two models
            disagree about a channel's observable shape.
    """
    resolved = registry if registry is not None else default_registry()
    species = _argon_ion(resolved)
    reference = _reference_theta()
    grid = observation_grid(resolved)
    low_solver = AnalyticSheathSolver()

    if high_solver is None:
        try:
            from vpl.physics.fluid.sheath import FluidSheathSolver
        except ImportError as exc:
            raise NotImplementedError(
                "estimating an L0-vs-L1 discrepancy needs vpl.physics.fluid.sheath."
                "FluidSheathSolver, which depends on dolfinx (FEniCSx), and dolfinx is not "
                "importable here. Estimate the basis where it is, save it with "
                "`save_channel_discrepancy`, and load it here — the same arrangement "
                "vpl.experiment.l2_truth uses to move an L2 truth between environments."
            ) from exc
        high_solver = FluidSheathSolver()

    # The reference state that sizes the LIF scan, pins Thomson's measurement volume and
    # anchors the interferometer's chords: L0 at the *reference* parameters, exactly as
    # every other driver builds it. It must not follow the sweep, or each sweep point would
    # be observed by a differently configured instrument and the differences would carry the
    # instrument's motion as well as the models'.
    reference_state = low_solver.solve(
        _to_plasma_params(reference, species=species, registry=resolved), grid=grid
    )
    # Clean configuration — see the module docstring's last section for why both switches
    # are off rather than matching the run being corrected.
    channels = build_channels(
        reference_state=reference_state,
        seed=seed,
        registry=resolved,
        noise=False,
        imperfect_calibration=False,
        calibration_uncertainty=False,
    )

    rows: dict[str, list[FloatArray]] = {name: [] for name in CHANNEL_NAMES}
    thetas = _sweep_thetas(reference, grid_points)
    for index, theta in enumerate(thetas):
        low_state, high_state = _states(
            theta,
            species=species,
            registry=resolved,
            grid=grid,
            low_solver=low_solver,
            high_solver=high_solver,
        )
        low_obs = channels.observe(low_state)
        high_obs = channels.observe(high_state)
        for name in CHANNEL_NAMES:
            low_values = np.asarray(low_obs[name].values, dtype=np.float64).reshape(-1)
            high_values = np.asarray(high_obs[name].values, dtype=np.float64).reshape(-1)
            if low_values.shape != high_values.shape:
                raise ValueError(
                    f"channel {name!r} produced different observable shapes for the two "
                    f"models ({low_values.shape} vs {high_values.shape}); the difference is "
                    f"not a model error, it is a configuration error"
                )
            rows[name].append(high_values - low_values)
        if verbose:
            print(f"[discrepancy] {index + 1}/{len(thetas)}", flush=True)

    scale = math.sqrt(len(thetas))
    basis: ChannelDiscrepancy = {}
    for name in CHANNEL_NAMES:
        stacked = np.asarray(np.stack(rows[name]) / scale, dtype=np.float64)
        if not np.all(np.isfinite(stacked)):
            raise ValueError(
                f"channel {name!r} produced a non-finite discrepancy basis; a NaN here "
                f"propagates into every likelihood evaluation and makes every parameter "
                f"look equally good"
            )
        basis[name] = _compress_to_rank(stacked)
    return basis


def save_channel_discrepancy(basis: Mapping[str, FloatArray], path: Path | str) -> Path:
    """Write the bases to a compressed npz, with the channel names alongside.

    The names are stored rather than inferred from the keys so that a file cannot silently
    describe three channels while a loader assumes four — a missing channel would apply no
    discrepancy to it and look like that channel simply being well modelled.
    """
    destination = Path(path).with_suffix(_NPZ_SUFFIX)
    missing = sorted(set(CHANNEL_NAMES) - set(basis))
    if missing:
        raise ValueError(
            f"no discrepancy basis for {', '.join(missing)}; saving a partial set would "
            f"leave those channels silently uncorrected, which is indistinguishable from "
            f"them being perfectly modelled"
        )
    payload: dict[str, NDArray[np.generic]] = {_NAMES_KEY: np.asarray(CHANNEL_NAMES, dtype=np.str_)}
    for name in CHANNEL_NAMES:
        payload[f"{_BASIS_KEY}{name}"] = np.asarray(basis[name], dtype=np.float64)
    # numpy's stubs type `savez_compressed`'s second positional as the `compress` flag, so
    # the keyword splat trips them; the call itself is the documented `**kwds` form.
    np.savez_compressed(destination, **payload)  # type: ignore[arg-type]
    return destination


def load_channel_discrepancy(path: Path | str) -> ChannelDiscrepancy:
    """Read bases written by :func:`save_channel_discrepancy`.

    Raises:
        ValueError: If the file describes a different channel set than this build expects.
            An artefact estimated before a channel was added would otherwise be applied
            silently to the channels it does know, leaving the new one uncorrected.
    """
    with np.load(Path(path)) as archive:
        if _NAMES_KEY not in archive:
            raise ValueError(
                f"{path} has no {_NAMES_KEY!r} entry, so it was not written by "
                f"save_channel_discrepancy and its keys cannot be trusted to mean what "
                f"this loader would assume"
            )
        stored = tuple(str(name) for name in archive[_NAMES_KEY])
        if stored != tuple(CHANNEL_NAMES):
            raise ValueError(
                f"{path} describes channels {stored} and this build has {CHANNEL_NAMES}. "
                f"Applying it would leave the difference uncorrected while reporting a "
                f"discrepancy-corrected result."
            )
        return {
            name: np.asarray(archive[f"{_BASIS_KEY}{name}"], dtype=np.float64) for name in stored
        }
