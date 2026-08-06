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
from vpl.experiment.channels import CHANNEL_NAMES
from vpl.experiment.discrepancy_basis import (
    DISCREPANCY_GRID,
    _compress_to_rank,
    estimate_channel_discrepancy,
    load_channel_discrepancy,
    save_channel_discrepancy,
)
from vpl.experiment.l2_truth import observation_grid
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
    return estimate_channel_discrepancy(
        seed=0,
        high_solver=high,  # type: ignore[arg-type]
        grid_points=grid_points,
    )


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
