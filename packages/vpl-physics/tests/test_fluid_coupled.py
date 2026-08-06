"""V-02 — manufactured solutions on the coupled fluid system.

doc 03 §7 V-02: *observed order = design order +/- 0.15*. The looser tolerance is the
document's, and it is asked for explicitly at the call site rather than defaulted, so the
weaker gate is visible where it is used.

The system verified here is the one doc 03 §1 names as L1 — "drift-diffusion ions +
Boltzmann electrons + Poisson" — solved by Newton on the coupled residual, which is what
doc 03 §3.4 specifies:

    Coupling | Newton on the fully coupled (n_i, u_i, Phi) system | The n_e(Phi)
    exponential makes Gummel iteration converge poorly at high bias

## How the coupled manufactured problem is built

Both halves come from :mod:`vpl.validation.manufactured`, unmodified:

- ``n*`` and its source from :func:`drift_diffusion_manufactured`;
- ``Phi*`` and its charge density from :func:`poisson_manufactured`.

Poisson is then driven by the *physical* space charge ``e (n_i - n_e(Phi))`` plus the
correction ``rho* - e (n* - n_e(Phi*))`` that makes the pair exact. The continuity
equation carries the doc 03 §3.1 ``S_iz - L_rec`` term with a loss proportional to the
electron density, which vanishes at the exact solution and makes the Jacobian genuinely
two-way coupled — without it the residual would be block-triangular and the Newton
assembly of the off-diagonal block would never be exercised.

**The limitation, stated:** the manufactured coupling into the continuity equation is a
reaction term rather than a field-dependent drift, because a drift that depended on
``Phi*`` would need the continuity source re-derived through the coupled operator, and
``vpl-validation`` does not offer that. A hand-derived correction is exactly the failure
mode :mod:`vpl.validation.manufactured` exists to remove.

## Peclet regime, and why the gate is not run at high Peclet

Exponential fitting is second-order accurate as ``h -> 0`` at fixed diffusivity, because
the artificial diffusion it adds is ``O(h^2)`` once the cell Peclet number is below one.
At *fixed* high cell Peclet it degrades to first-order upwind — that is the price of the
monotonicity doc 03 §3.4 bought it for. A refinement study straddling the two regimes
fits a slope between 1 and 2 and means nothing. The gate therefore runs entirely inside
the asymptotic regime, and the drift-dominated regime is measured separately and
reported rather than gated.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("dolfinx")

from typing import TypeAlias

from vpl.core.constants import ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.state import SpatialGrid
from vpl.core.units import Q_, magnitude_in
from vpl.physics.fluid import (
    Stabilisation,
    solve_drift_diffusion,
    solve_drift_diffusion_poisson,
)
from vpl.validation.convergence import (
    RefinementLevel,
    assert_design_order,
    observed_order,
    weighted_l2_error,
)
from vpl.validation.manufactured import (
    drift_diffusion_manufactured,
    poisson_manufactured,
)

FloatArray: TypeAlias = NDArray[np.float64]

_E_C = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_EPS0 = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

# ── the manufactured problem, at sheath scales ──────────────────────────────────

LENGTH = 1.0e-3
DENSITY = 1.0e17
DRIFT = 2691.0
"""The RP-1 Bohm speed of doc 01 §2.2, so the manufactured advection is the real one."""

T_E_EV = 10.0
POTENTIAL_AMPLITUDE = 6.0
"""One electron temperature of potential swing, which keeps ``exp(e Phi / k T_e)`` in
``[1, e]``. A sheath-sized 250 V swing would make it ``exp(83)``, and a manufactured
problem whose exact solution overflows verifies nothing."""

MODES = 3
CELL_COUNTS = (40, 80, 160, 320)

#: Cell Peclet number ``|u| h / (2 D)`` at the coarsest level. Chosen at one half so the
#: whole study sits in the asymptotic regime — see the module docstring.
COARSE_CELL_PECLET = 0.5
DIFFUSIVITY = DRIFT * (LENGTH / CELL_COUNTS[0]) / (2.0 * COARSE_CELL_PECLET)

#: Drift-dominated variant, reported rather than gated.
DRIFT_DOMINATED_DIFFUSIVITY = DIFFUSIVITY / 100.0

#: Recombination rate for the doc 03 §3.1 ``L_rec`` coupling term, sized so the term is
#: comparable to the divergence of the ion flux rather than a perturbation on it.
COUPLING_RATE = DRIFT / LENGTH


def _uniform(n_cells: int) -> SpatialGrid:
    return SpatialGrid.uniform(length=Q_(LENGTH, "m"), n_points=n_cells + 1)


def _boltzmann(potential: FloatArray) -> FloatArray:
    return np.asarray(DENSITY * np.exp(potential / T_E_EV), dtype=np.float64)


def _manufactured(
    diffusivity: float,
) -> tuple[
    Callable[[FloatArray], FloatArray],
    Callable[[FloatArray], FloatArray],
    dict[str, object],
]:
    """Return ``(n_exact, Phi_exact, kwargs)`` for the coupled manufactured problem."""
    ion = drift_diffusion_manufactured(
        length=LENGTH, density=DENSITY, drift=DRIFT, diffusivity=diffusivity, modes=MODES
    )
    field = poisson_manufactured(length=LENGTH, amplitude=POTENTIAL_AMPLITUDE, modes=MODES)

    def charge_density_source(z: FloatArray) -> FloatArray:
        """``rho* - e (n* - n_e(Phi*))`` — what is left of Poisson's source."""
        return np.asarray(
            field.source(z) - _E_C * (ion.exact(z) - _boltzmann(field.exact(z))),
            dtype=np.float64,
        )

    def electron_reference(z: FloatArray) -> FloatArray:
        return _boltzmann(field.exact(z))

    return (
        ion.exact,
        field.exact,
        {
            "drift": DRIFT,
            "diffusivity": diffusivity,
            "electron_temperature_volts": T_E_EV,
            "reference_density": DENSITY,
            "ion_source": ion.source,
            "charge_density_source": charge_density_source,
            "electron_coupling_rate": COUPLING_RATE,
            "electron_coupling_reference": electron_reference,
            "density_boundary": ion.boundary_values,
            "potential_boundary": field.boundary_values,
        },
    )


def _coupled_study(
    *, diffusivity: float, cell_counts: tuple[int, ...] = CELL_COUNTS
) -> tuple[list[RefinementLevel], list[RefinementLevel], list[int]]:
    n_exact, phi_exact, kwargs = _manufactured(diffusivity)
    density_levels: list[RefinementLevel] = []
    potential_levels: list[RefinementLevel] = []
    iterations: list[int] = []

    for n_cells in cell_counts:
        grid = _uniform(n_cells)
        solution = solve_drift_diffusion_poisson(grid=grid, **kwargs)  # type: ignore[arg-type]
        assert solution.newton.converged
        iterations.append(solution.newton.iterations)

        # Gauss points, not vertices or midpoints — vpl.physics.fluid.forms states why.
        h = float(magnitude_in(grid.min_dz, "m"))
        for field, exact, levels, scale in (
            (solution.density, n_exact, density_levels, DENSITY),
            (solution.potential, phi_exact, potential_levels, 1.0),
        ):
            points = field.quadrature_points
            levels.append(
                RefinementLevel(
                    h=h,
                    error=weighted_l2_error(
                        field.quadrature_values,
                        exact(points),
                        weights=field.quadrature_weights,
                    )
                    / scale,
                )
            )

    return density_levels, potential_levels, iterations


class TestV02CoupledOrderOfAccuracy:
    """doc 03 §7 V-02 / doc 11 G-1.1."""

    @pytest.mark.physics
    @pytest.mark.slow
    def test_both_fields_converge_at_second_order(self) -> None:
        density, potential, iterations = _coupled_study(diffusivity=DIFFUSIVITY)
        density_result = observed_order(density)
        potential_result = observed_order(potential)
        print(
            f"\nV-02 coupled MMS, cell Peclet {COARSE_CELL_PECLET} -> "
            f"{COARSE_CELL_PECLET * CELL_COUNTS[0] / CELL_COUNTS[-1]:.3g}, "
            f"Newton iterations {iterations}"
            f"\n  n_i:\n{density_result}\n  Phi:\n{potential_result}"
        )
        assert_design_order(density_result, design_order=2.0, tolerance=0.15)
        assert_design_order(potential_result, design_order=2.0, tolerance=0.15)

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_drift_dominated_regime_degrades_toward_first_order(self) -> None:
        """Measured and reported, not tuned. See the module docstring.

        Exponential fitting trades accuracy for monotonicity when the cell Peclet number
        is large. The gate above avoids that regime deliberately; this records what
        happens inside it, so that a future change which quietly made the scheme
        first-order everywhere would show up as this number *rising* to meet it.
        """
        density, _, _ = _coupled_study(diffusivity=DRIFT_DOMINATED_DIFFUSIVITY)
        result = observed_order(density)
        peclet = DRIFT * (LENGTH / CELL_COUNTS[0]) / (2.0 * DRIFT_DOMINATED_DIFFUSIVITY)
        print(f"\nV-02 drift-dominated, coarse cell Peclet {peclet:.3g}:\n{result}")
        assert 0.9 <= result.observed_order <= 2.1

    @pytest.mark.physics
    def test_newton_converges_in_a_handful_of_iterations(self) -> None:
        """doc 03 §3.4 chose Newton over Gummel; the claim is that it converges fast."""
        _, _, iterations = _coupled_study(diffusivity=DIFFUSIVITY, cell_counts=(80,))
        assert iterations[0] <= 10


class TestCoupledContract:
    def test_the_boltzmann_relation_holds_on_the_solution(self) -> None:
        _, _, kwargs = _manufactured(DIFFUSIVITY)
        solution = solve_drift_diffusion_poisson(grid=_uniform(64), **kwargs)  # type: ignore[arg-type]
        assert solution.electron_density.cell_values == pytest.approx(
            _boltzmann(solution.potential.cell_values), rel=1e-12
        )

    def test_dirichlet_values_are_imposed_on_both_fields(self) -> None:
        _, _, kwargs = _manufactured(DIFFUSIVITY)
        solution = solve_drift_diffusion_poisson(grid=_uniform(64), **kwargs)  # type: ignore[arg-type]
        density_bc = kwargs["density_boundary"]
        potential_bc = kwargs["potential_boundary"]
        assert isinstance(density_bc, tuple)
        assert isinstance(potential_bc, tuple)
        assert solution.density.nodal[0] == pytest.approx(density_bc[0], rel=1e-10)
        assert solution.density.nodal[-1] == pytest.approx(density_bc[1], rel=1e-10)
        assert solution.potential.nodal[0] == pytest.approx(potential_bc[0], abs=1e-12)
        assert solution.potential.nodal[-1] == pytest.approx(potential_bc[1], abs=1e-12)

    def test_a_non_converging_newton_raises_rather_than_returning_garbage(self) -> None:
        """A silently unconverged nonlinear solve is a plausible-looking wrong answer.

        doc 07 §7 puts it plainly: physics regressions are silent otherwise — the code
        runs, the plots look plausible, and the answer is wrong.
        """
        _, _, kwargs = _manufactured(DIFFUSIVITY)
        with pytest.raises(RuntimeError, match="Newton"):
            solve_drift_diffusion_poisson(
                grid=_uniform(64),
                max_iterations=1,
                **kwargs,  # type: ignore[arg-type]
            )


# ── the claim doc 03 §3.4 makes about centred differences ───────────────────────


class TestDriftDominatedMonotonicity:
    """doc 03 §3.4: centred differences "oscillate in the strongly-drift-dominated sheath".

    The classic one-dimensional test: ``u n' - D n'' = 0`` with ``n(0) = 0``,
    ``n(L) = 1``. The exact solution is a monotone boundary layer at the outflow end and
    lies in ``[0, 1]`` everywhere. At a cell Peclet number well above one, centred
    Galerkin leaves that interval and Scharfetter-Gummel does not.
    """

    LAYER_PECLET = 5.0

    def _layer(self, stabilisation: Stabilisation, n_cells: int = 20) -> FloatArray:
        grid = _uniform(n_cells)
        diffusivity = DRIFT * (LENGTH / n_cells) / (2.0 * self.LAYER_PECLET)
        solution = solve_drift_diffusion(
            grid=grid,
            drift=DRIFT,
            diffusivity=diffusivity,
            source=lambda z: np.zeros_like(z),
            boundary_values=(0.0, 1.0),
            stabilisation=stabilisation,
        )
        return solution.nodal

    @pytest.mark.physics
    def test_centred_galerkin_overshoots(self) -> None:
        values = self._layer(Stabilisation.NONE)
        print(f"\ncentred Galerkin range: [{values.min():.4g}, {values.max():.4g}]")
        assert values.min() < -1e-3 or values.max() > 1.0 + 1e-3

    @pytest.mark.physics
    def test_scharfetter_gummel_stays_inside_the_boundary_data(self) -> None:
        values = self._layer(Stabilisation.EXPONENTIAL_FITTING)
        print(f"\nScharfetter-Gummel range: [{values.min():.4g}, {values.max():.4g}]")
        assert values.min() >= -1e-12
        assert values.max() <= 1.0 + 1e-12

    @pytest.mark.physics
    def test_scharfetter_gummel_is_nodally_exact_on_the_boundary_layer(self) -> None:
        """Exponential fitting solves this problem exactly, at any resolution.

        That is the property that distinguishes it from upwinding, and the reason a
        sheath front can be crossed in a few cells without smearing.
        """
        n_cells = 20
        grid = _uniform(n_cells)
        diffusivity = DRIFT * (LENGTH / n_cells) / (2.0 * self.LAYER_PECLET)
        values = self._layer(Stabilisation.EXPONENTIAL_FITTING, n_cells=n_cells)

        # Written as exp(-Pe (1 - z/L)) rather than the textbook (e^{Pe z/L} - 1) /
        # (e^{Pe} - 1): the latter is e^200 over e^200 here, which is finite but throws
        # away most of its significant figures.
        peclet_global = DRIFT * LENGTH / diffusivity
        exact = (np.exp(-peclet_global * (1.0 - grid.z_m / LENGTH)) - np.exp(-peclet_global)) / (
            1.0 - np.exp(-peclet_global)
        )
        assert values == pytest.approx(exact, abs=1e-9)
