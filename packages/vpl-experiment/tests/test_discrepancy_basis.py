"""Per-channel discrepancy bases — doc 05 §4, and the S4 failure that motivates them.

## What this file is guarding

:mod:`vpl.experiment.discrepancy_basis` exists because the four-channel T2 grid recovers
``Gamma_E`` to 6.47 % while reporting a +/-0.18 % interval — the truth sits 34 interval
half-widths away, and doc 00 §5.1's criterion S4 calls that worse than no posterior. The
module estimates the missing object: the covariance of the model error itself.

Its numbers cannot be checked here, because producing one needs L1 and L1 needs dolfinx,
which is absent from most machines this suite runs on. What *can* be checked here — and
what actually goes wrong in a module like this — is everything around the numbers:

* the basis has the shape the instruments' rank-``k`` machinery expects,
* the covariance it implies is symmetric positive semi-definite (a discrepancy that is not
  PSD is not a covariance, and would *sharpen* the likelihood in some direction rather than
  widen it — the exact mean-variance-coupling failure this project has already hit once),
* the redundant directions of an over-determined sweep are compressed away **losslessly**
  rather than truncated — interferometry's observable is a single scalar, so a rule that
  subtracted a direction would leave it with none, and therefore with no discrepancy at all,
  applied silently,
* and the artefact round-trips, refusing anything it cannot honestly interpret.

A stand-in higher-fidelity model is used throughout. That is legitimate for the plumbing and
illegitimate for the physics, and the module's own ``_UnconstrainedSolver`` docstring says
why: a stand-in that differs from L0 by a parameter shift has its error directions parallel
to the parameter sensitivities, which is precisely the property a real discrepancy was
measured **not** to have. So no test here asserts anything about a magnitude.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from vpl.core.state import PlasmaParams, PlasmaState, SpatialGrid
from vpl.core.units import magnitude_in
from vpl.experiment.channels import CHANNEL_NAMES
from vpl.experiment.closed_loop import _reference_theta
from vpl.experiment.discrepancy_basis import (
    DISCREPANCY_GRID,
    DISCREPANCY_SPAN,
    DiscrepancySweepError,
    DiscrepancySweepResult,
    _compress_to_rank,
    estimate_channel_discrepancy,
    load_channel_discrepancy,
    save_channel_discrepancy,
)
from vpl.experiment.l2_truth import observation_grid
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import AnalyticSheathSolver

#: Multiplicative shift the stand-in applies to ``T_e`` before solving. Large enough that the
#: two models disagree well above floating-point noise — a stand-in that happened to agree
#: exactly would produce an all-zero basis and make every assertion below vacuous.
_STANDIN_T_E_FACTOR = 1.10

#: Sweep size used by the cheap tests, in place of :data:`DISCREPANCY_GRID`. The real value
#: costs 25 parameter points times two solves times four channel observations, which is tens
#: of seconds; 2 gives rank 4 and exercises the same code paths.
_CHEAP_GRID = 2


class _ShiftedL0:
    """L0 evaluated at shifted ``T_e``, standing in for a higher-fidelity model.

    Solves on **its own** grid rather than the observation grid, so that
    ``estimate_channel_discrepancy``'s resample step is exercised rather than skipped — that
    step is where an off-by-one or a silent ``numpy.interp`` clamp would live.
    """

    def __init__(self, grid: SpatialGrid) -> None:
        self._solver = AnalyticSheathSolver()
        self._grid = grid

    def solve(self, params: PlasmaParams) -> PlasmaState:
        shifted = params.replace(T_e=params.T_e * _STANDIN_T_E_FACTOR)
        return self._solver.solve(shifted, grid=self._grid)


class _IdenticalL0:
    """L0 on the observation grid — a stand-in that is the *same* model as the low one.

    Used for the degenerate case: two identical models have no discrepancy, and the module
    must produce an all-zero basis rather than a small spurious one. A non-zero basis here
    would mean the estimator is picking up something that is not model error — a stale
    instrument state, a consumed random stream, a resampling artefact.
    """

    def __init__(self, grid: SpatialGrid) -> None:
        self._solver = AnalyticSheathSolver()
        self._grid = grid

    def solve(self, params: PlasmaParams) -> PlasmaState:
        return self._solver.solve(params, grid=self._grid)


@functools.cache
def _cheap_shifted_basis() -> dict[str, np.ndarray]:
    """The cheap-sweep basis against the shifted stand-in, computed once for the whole file.

    Cached because every observation runs all four channels and several tests need the same
    object; recomputing it per test multiplies the suite's cost by the number of tests for
    no additional coverage. Returned mappings are never mutated by the tests below.
    """
    return _estimate(_ShiftedL0(observation_grid()), grid_points=_CHEAP_GRID)


def _estimate(high: object, *, grid_points: int) -> dict[str, np.ndarray]:
    result = estimate_channel_discrepancy(
        seed=0,
        high_solver=high,  # type: ignore[arg-type]
        grid_points=grid_points,
    )
    assert isinstance(result, DiscrepancySweepResult)
    assert not result.failures
    assert result.grid_points_converged == result.grid_points_total == grid_points**2
    return result.basis


class TestTheBasisHasTheShapeTheInstrumentsExpect:
    def test_every_channel_gets_a_basis(self) -> None:
        # A missing channel would apply no discrepancy to it and be indistinguishable from
        # that channel being perfectly modelled — the failure `save_channel_discrepancy`
        # also guards, checked here at the source rather than only at the sink.
        basis = _cheap_shifted_basis()

        assert set(basis) == set(CHANNEL_NAMES)

    def test_each_basis_is_two_dimensional_and_no_wider_than_its_observable(self) -> None:
        # `EmpiricalDiscrepancy` refuses anything that is not (k, n). k <= n is
        # `_compress_to_rank`'s post-condition: rows beyond the observable's dimension are
        # necessarily linearly dependent, so keeping them stores redundancy and widens every
        # downstream Woodbury solve for nothing.
        #
        # k > 0 is the other half, and it is the one that actually bit. Interferometry's
        # observable is a single scalar, so any rule that subtracted a direction would leave
        # it with an empty basis — no discrepancy at all, applied silently, while the run
        # still reported itself discrepancy-corrected.
        basis = _cheap_shifted_basis()

        for name, matrix in basis.items():
            assert matrix.ndim == 2, name
            k, n = matrix.shape
            assert k <= n, f"{name}: rank {k} over {n} dimensions was not compressed"
            assert k > 0, f"{name}: an empty basis applies no discrepancy, silently"

    def test_the_implied_covariance_is_symmetric_positive_semi_definite(self) -> None:
        # The whole point of a discrepancy is to *widen*. A covariance with a negative
        # eigenvalue sharpens the likelihood along that direction instead, which is the
        # mean-variance-coupling failure this project already hit once (the discrepancy that
        # took posterior curvature from 1.4e5 to 2.1e5 — the wrong way).
        basis = _cheap_shifted_basis()

        for name, matrix in basis.items():
            covariance = matrix.T @ matrix
            assert np.allclose(covariance, covariance.T), name
            eigenvalues = np.linalg.eigvalsh(covariance)
            assert eigenvalues.min() > -1e-9 * max(abs(eigenvalues.max()), 1.0), name


class TestTheDegenerateCaseIsExactlyZero:
    def test_two_identical_models_have_no_discrepancy(self) -> None:
        # Not a triviality: the estimator observes each state through the full channel set,
        # and if any of that observation consumed a random stream, carried state between
        # calls, or resampled asymmetrically, this would come back non-zero. The module's
        # claim that `noise=False, imperfect_calibration=False` makes the two observations a
        # deterministic function of the state alone is exactly what this checks.
        basis = _estimate(_IdenticalL0(observation_grid()), grid_points=_CHEAP_GRID)

        for name, matrix in basis.items():
            assert np.all(matrix == 0.0), f"{name}: identical models produced a non-zero basis"

    def test_a_genuinely_different_model_does_not(self) -> None:
        # The vacuity guard for every other test in this file: if the stand-in happened to
        # agree with L0, the assertions above would all hold for the wrong reason.
        basis = _cheap_shifted_basis()

        assert any(np.any(matrix != 0.0) for matrix in basis.values())


class TestCompressionDiscardsRedundancyAndNothingElse:
    """The over-determined case, and the reason it is compression rather than truncation."""

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            ((25, 8), (8, 8)),
            # Interferometry, measured: a single scalar phase against a rank-25 sweep.
            ((25, 1), (1, 1)),
        ],
    )
    def test_an_over_determined_basis_is_re_expressed_at_its_own_rank(
        self, shape: tuple[int, int], expected: tuple[int, int]
    ) -> None:
        rng = np.random.default_rng(0)

        compressed = _compress_to_rank(rng.normal(size=shape))

        assert compressed.shape == expected

    @pytest.mark.parametrize("shape", [(25, 8), (25, 1), (25, 3)])
    def test_the_implied_covariance_is_unchanged_to_floating_point(
        self, shape: tuple[int, int]
    ) -> None:
        # This is the whole claim, and it is what separates this from the first attempt.
        # Compression must reproduce `basis.T @ basis` **exactly**, not approximately: if
        # the singular values were dropped rather than folded back into the retained rows,
        # the covariance would keep its directions and lose its scale — a discrepancy of the
        # right shape and the wrong size, which is the hardest error to notice in a widened
        # interval, because the interval still moves in the right direction.
        rng = np.random.default_rng(1)
        original = rng.normal(size=shape)

        compressed = _compress_to_rank(original)

        assert np.allclose(original.T @ original, compressed.T @ compressed, rtol=1e-12, atol=0.0)

    def test_an_already_thin_basis_is_returned_untouched(self) -> None:
        # OES is 11 x 1024 samples against a rank-25 sweep, so the common case must be a
        # no-op — an SVD that ran anyway would silently re-condition a basis that was fine.
        rng = np.random.default_rng(2)
        thin = rng.normal(size=(4, 100))

        assert _compress_to_rank(thin) is thin

    def test_a_scalar_observable_never_loses_its_only_direction(self) -> None:
        # The bug this class exists for. A keep-(n-1) rule gives min(25, 0) = 0 rows here,
        # which is an empty basis, which is no discrepancy at all — applied silently to the
        # one channel whose observable is a scalar, while the run reports itself corrected.
        rng = np.random.default_rng(3)

        compressed = _compress_to_rank(rng.normal(size=(25, 1)))

        assert compressed.shape[0] == 1
        assert compressed[0, 0] != 0.0


class TestTheArtefactRoundTrips:
    def test_saving_then_loading_reproduces_every_basis_exactly(self, tmp_path: object) -> None:
        # Bit-for-bit, not approximately: a lossy round trip would make a
        # discrepancy-corrected result depend on which machine wrote the file.
        basis = _cheap_shifted_basis()
        path = save_channel_discrepancy(basis, tmp_path / "disc")  # type: ignore[operator]

        loaded = load_channel_discrepancy(path)

        assert set(loaded) == set(basis)
        for name in basis:
            assert np.array_equal(loaded[name], basis[name]), name

    def test_saving_a_partial_set_is_refused(self, tmp_path: object) -> None:
        partial = {CHANNEL_NAMES[0]: np.zeros((2, 5))}

        with pytest.raises(ValueError, match="|".join(CHANNEL_NAMES[1:])):
            save_channel_discrepancy(partial, tmp_path / "partial")  # type: ignore[operator]

    def test_a_file_describing_other_channels_is_refused(self, tmp_path: object) -> None:
        # An artefact estimated before a channel existed would otherwise be applied to the
        # channels it does know, leaving the new one uncorrected while the run still reports
        # itself as discrepancy-corrected.
        path = tmp_path / "stale.npz"  # type: ignore[operator]
        np.savez_compressed(
            path,
            channel_names=np.asarray(("oes", "lif"), dtype=np.str_),
            basis__oes=np.zeros((2, 5)),
            basis__lif=np.zeros((2, 5)),
        )

        with pytest.raises(ValueError, match="oes"):
            load_channel_discrepancy(path)

    def test_a_foreign_npz_is_refused_rather_than_misread(self, tmp_path: object) -> None:
        path = tmp_path / "foreign.npz"  # type: ignore[operator]
        np.savez_compressed(path, something_else=np.zeros(3))

        with pytest.raises(ValueError, match="channel_names"):
            load_channel_discrepancy(path)


class TestTheRealSweepSize:
    @pytest.mark.slow
    def test_the_shipped_sweep_compresses_the_narrow_channel_and_leaves_the_wide_ones(
        self,
    ) -> None:
        # The configuration that actually ships, run at its real setting because the thing
        # being checked *is* the interaction between the sweep size and each channel's
        # dimension — a reduced sweep would check a configuration nobody runs.
        #
        # Measured channel dimensions at the reference: OES 11 x 1024 = 11264, LIF 201,
        # Thomson 20, interferometry 1. So a rank-25 sweep is over-determined for exactly
        # one channel, and that channel must come back with one row rather than zero.
        basis = _estimate(_ShiftedL0(observation_grid()), grid_points=DISCREPANCY_GRID)

        assert DISCREPANCY_GRID**2 == 25
        for name, matrix in basis.items():
            k, n = matrix.shape
            assert k == min(DISCREPANCY_GRID**2, n), f"{name}: {k} rows over {n} dimensions"
            assert k > 0, name
        assert basis["interferometry"].shape == (1, 1)
        assert basis["oes"].shape[0] == DISCREPANCY_GRID**2


class _FailsAtExactTheta:
    """Stands in for a solver that cannot converge at specific nominal grid points.

    Fails only when handed one of ``failing_thetas`` *exactly* — comparison is on the raw
    ``(n_0, T_e)`` magnitudes the sweep computed, which is what a nominal grid point looks
    like. Both axes have to be checked: with a small grid, two different grid points can
    share an ``n_0`` (same row, different ``T_e``), and matching on ``n_0`` alone would
    fail both instead of the one point under test. A retry that nudges ``theta`` even
    fractionally toward the reference no longer matches any entry and succeeds, which is
    the behaviour :func:`_solve_with_bounded_retry` is built to exploit: a point that
    fails at its labelled position may still converge a little way toward the reference.
    """

    def __init__(
        self, grid: SpatialGrid, *, failing_thetas: frozenset[tuple[float, float]]
    ) -> None:
        self._solver = AnalyticSheathSolver()
        self._grid = grid
        self._failing_thetas = failing_thetas

    def solve(self, params: PlasmaParams) -> PlasmaState:
        n_0 = float(magnitude_in(params.n_0, "m**-3"))
        t_e = float(magnitude_in(params.T_e, "eV"))
        if any(abs(n_0 - fn) < 1.0 and abs(t_e - ft) < 1e-9 for fn, ft in self._failing_thetas):
            raise RuntimeError("stand-in Newton failure at this exact theta")
        shifted = params.replace(T_e=params.T_e * _STANDIN_T_E_FACTOR)
        return self._solver.solve(shifted, grid=self._grid)


class _AlwaysFails:
    """A solver that never converges, anywhere — the give-up path, exercised cheaply."""

    def solve(self, params: PlasmaParams) -> PlasmaState:
        raise RuntimeError("stand-in Newton failure, unconditionally")


def _nominal_thetas(*, grid_points: int) -> list[tuple[float, float]]:
    """The exact ``(n_0, T_e)`` pairs :func:`_sweep_thetas` visits, in its own order."""
    reference: ControlParameters = _reference_theta()
    factors = np.linspace(*DISCREPANCY_SPAN, grid_points)
    return [(reference.n_0 * fn, reference.T_e * ft) for fn in factors for ft in factors]


class TestNonConvergenceIsNeverSilent:
    """Recovery plan Block G: a solve that never converged must not enter the basis
    unannounced. Either the sweep raises, naming every failed point, or the caller opted
    into exclusion and the result says explicitly what was left out and how much rank that
    cost — never a basis that looks full when it is not.
    """

    def test_a_fully_converged_sweep_reports_zero_failures_and_full_rank(self) -> None:
        result = estimate_channel_discrepancy(
            seed=0,
            high_solver=_ShiftedL0(observation_grid()),  # type: ignore[arg-type]
            grid_points=_CHEAP_GRID,
        )

        assert result.failures == ()
        assert result.nudged == ()
        assert result.grid_points_converged == result.grid_points_total == _CHEAP_GRID**2
        assert result.rank == _CHEAP_GRID**2

    def test_by_default_a_nonconvergent_point_raises_rather_than_entering_the_basis(
        self,
    ) -> None:
        # The failure mode Block G exists to prevent: without this, a point that never
        # converged could silently sit inside `basis` with nothing distinguishing its row
        # from a converged one.
        grid = observation_grid()
        failing = _nominal_thetas(grid_points=_CHEAP_GRID)[0]
        solver = _FailsAtExactTheta(grid, failing_thetas=frozenset({failing}))

        with pytest.raises(DiscrepancySweepError) as excinfo:
            estimate_channel_discrepancy(
                seed=0,
                high_solver=solver,  # type: ignore[arg-type]
                grid_points=_CHEAP_GRID,
                max_retries=0,  # no nudge can dodge an exact-match failure trivially
            )

        assert excinfo.value.grid_points_total == _CHEAP_GRID**2
        assert len(excinfo.value.failures) == 1
        assert excinfo.value.failures[0].index == 0

    def test_exclude_mode_reports_the_reduced_rank_instead_of_raising(self) -> None:
        grid = observation_grid()
        nominal = _nominal_thetas(grid_points=_CHEAP_GRID)
        failing = frozenset(nominal[:1])
        solver = _FailsAtExactTheta(grid, failing_thetas=failing)

        result = estimate_channel_discrepancy(
            seed=0,
            high_solver=solver,  # type: ignore[arg-type]
            grid_points=_CHEAP_GRID,
            on_nonconvergence="exclude",
            max_retries=0,
        )

        assert len(result.failures) == 1
        assert result.failures[0].index == 0
        assert result.grid_points_converged == _CHEAP_GRID**2 - 1
        assert result.rank == result.grid_points_converged
        for matrix in result.basis.values():
            k, _n = matrix.shape
            assert k <= result.grid_points_converged

    def test_a_point_that_only_converges_after_a_retry_is_recorded_as_nudged(self) -> None:
        # The point fails exactly at its nominal theta but a retry nudges it toward the
        # reference, where the fake no longer matches and succeeds — exercising the
        # "continuation: step in from a converged neighbour" path without dolfinx.
        grid = observation_grid()
        failing = _nominal_thetas(grid_points=_CHEAP_GRID)[0]
        solver = _FailsAtExactTheta(grid, failing_thetas=frozenset({failing}))

        result = estimate_channel_discrepancy(
            seed=0,
            high_solver=solver,  # type: ignore[arg-type]
            grid_points=_CHEAP_GRID,
            max_retries=1,
        )

        assert result.failures == ()
        assert len(result.nudged) == 1
        assert result.nudged[0].index == 0
        assert result.nudged[0].retries == 1
        assert result.nudged[0].solved_theta.n_0 != result.nudged[0].nominal_theta.n_0

    def test_a_solver_that_never_converges_gives_up_after_bounded_retries(self) -> None:
        # Every point fails unconditionally, so this always hits the "nothing left to
        # build a basis from" guard — what is checked here is that giving up happened
        # *after* exhausting the configured retries, not before: DiscrepancySweepError in
        # raise mode (the default) carries that count per failure.
        with pytest.raises(DiscrepancySweepError) as excinfo:
            estimate_channel_discrepancy(
                seed=0,
                high_solver=_AlwaysFails(),  # type: ignore[arg-type]
                grid_points=_CHEAP_GRID,
                max_retries=2,
            )

        assert len(excinfo.value.failures) == _CHEAP_GRID**2
        assert all(f.retries == 2 for f in excinfo.value.failures)

    def test_every_grid_point_failing_refuses_to_return_an_empty_basis(self) -> None:
        with pytest.raises(ValueError, match="nothing left to build a basis from"):
            estimate_channel_discrepancy(
                seed=0,
                high_solver=_AlwaysFails(),  # type: ignore[arg-type]
                grid_points=_CHEAP_GRID,
                on_nonconvergence="exclude",
                max_retries=0,
            )
