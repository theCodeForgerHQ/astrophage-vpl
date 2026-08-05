"""Manufactured solutions — doc 07 §2.3.

    Choose an analytic solution, substitute it into the governing equations, derive the
    source term that makes it exact, then verify that the observed convergence rate
    matches the design order.

The step everyone gets wrong is the middle one. A hand-derived source term that is subtly
wrong makes a correct solver converge at the wrong order, and the resulting bug hunt goes
looking in the solver. These sources are derived symbolically and then *checked
numerically against the operator they claim to invert*, so a wrong derivation fails here
rather than in the solver it was supposed to verify.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.validation.manufactured import (
    ManufacturedSolution,
    drift_diffusion_manufactured,
    poisson_manufactured,
)


class TestPoissonManufacturedSolution:
    @pytest.mark.physics
    def test_the_source_reproduces_the_operator_applied_to_the_exact_solution(self) -> None:
        # The self-check that makes MMS trustworthy. -eps0 d2phi/dz2 evaluated numerically
        # on the exact solution must equal the source term the symbolic derivation gave.
        # If it does not, the manufactured problem is not the one the solver will be
        # asked to solve, and the convergence study measures nothing.
        mms = poisson_manufactured(length=0.02, amplitude=250.0, modes=3)
        z = np.linspace(0.0, 0.02, 20001)

        phi = mms.exact(z)
        numerical = -mms.coefficient * np.gradient(np.gradient(phi, z), z)

        interior = slice(50, -50)
        np.testing.assert_allclose(
            numerical[interior], mms.source(z)[interior], rtol=1e-4, atol=1e-6
        )

    def test_a_deliberately_wrong_source_fails_the_self_check(self) -> None:
        # Proves the check above can actually fail. A verification harness nobody has
        # seen reject anything is not a harness.
        mms = poisson_manufactured(length=0.02, amplitude=250.0, modes=3)
        z = np.linspace(0.0, 0.02, 20001)

        phi = mms.exact(z)
        numerical = -mms.coefficient * np.gradient(np.gradient(phi, z), z)
        wrong = mms.source(z) * 1.05

        interior = slice(50, -50)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(numerical[interior], wrong[interior], rtol=1e-4)

    def test_the_solution_is_not_representable_in_a_low_order_space(self) -> None:
        # A manufactured solution that a P1 space reproduces exactly gives zero error at
        # every resolution, and the fitted order is then undefined rather than wrong —
        # a study that silently proves nothing. Transcendental content prevents that.
        mms = poisson_manufactured(length=0.02, amplitude=250.0, modes=3)
        z = np.linspace(0.0, 0.02, 9)

        phi = mms.exact(z)
        linear_fit = np.polyval(np.polyfit(z, phi, 1), z)

        assert np.max(np.abs(phi - linear_fit)) > 0.01 * np.max(np.abs(phi))

    def test_boundary_values_come_from_the_exact_solution(self) -> None:
        mms = poisson_manufactured(length=0.02, amplitude=250.0, modes=3)

        assert mms.boundary_values[0] == pytest.approx(float(mms.exact(np.array([0.0]))[0]))
        assert mms.boundary_values[1] == pytest.approx(float(mms.exact(np.array([0.02]))[0]))

    def test_more_modes_make_a_harder_problem(self) -> None:
        # The refinement study needs the solution to have structure the coarse mesh
        # cannot resolve, or the first level is already converged and the fit has no
        # asymptotic range to measure.
        gentle = poisson_manufactured(length=0.02, amplitude=250.0, modes=1)
        harsh = poisson_manufactured(length=0.02, amplitude=250.0, modes=8)
        z = np.linspace(0.0, 0.02, 1001)

        assert np.max(np.abs(harsh.source(z))) > np.max(np.abs(gentle.source(z)))

    def test_rejects_a_non_positive_length(self) -> None:
        with pytest.raises(ValueError, match="length"):
            poisson_manufactured(length=0.0, amplitude=250.0, modes=3)

    def test_rejects_a_non_positive_mode_count(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            poisson_manufactured(length=0.02, amplitude=250.0, modes=0)


class TestDriftDiffusionManufacturedSolution:
    @pytest.mark.physics
    def test_the_source_reproduces_the_operator_applied_to_the_exact_solution(self) -> None:
        # V-02, the coupled system of doc 03 §3.1. Same self-check, one order harder:
        # the operator is d/dz (n u) - d/dz (D dn/dz), so an error in either term shows.
        mms = drift_diffusion_manufactured(
            length=0.02, density=1e17, drift=2.7e3, diffusivity=1.0, modes=3
        )
        z = np.linspace(0.0, 0.02, 20001)

        n = mms.exact(z)
        flux = mms.drift * n - mms.diffusivity * np.gradient(n, z)
        numerical = np.gradient(flux, z)

        interior = slice(50, -50)
        np.testing.assert_allclose(
            numerical[interior], mms.source(z)[interior], rtol=1e-4, atol=1e-3
        )

    def test_the_density_stays_positive_everywhere(self) -> None:
        # A manufactured density that goes negative is not a plasma state, and a solver
        # with a positivity-preserving scheme (doc 03 §3.4, Scharfetter-Gummel) would be
        # penalised for refusing to reproduce something physically impossible.
        mms = drift_diffusion_manufactured(
            length=0.02, density=1e17, drift=2.7e3, diffusivity=1.0, modes=5
        )
        z = np.linspace(0.0, 0.02, 5001)

        assert np.all(mms.exact(z) > 0.0)

    def test_rejects_a_non_positive_diffusivity(self) -> None:
        with pytest.raises(ValueError, match="diffusivity"):
            drift_diffusion_manufactured(
                length=0.02, density=1e17, drift=2.7e3, diffusivity=0.0, modes=3
            )


class TestTheManufacturedSolutionType:
    def test_carries_a_name_and_a_description(self) -> None:
        mms: ManufacturedSolution = poisson_manufactured(length=0.02, amplitude=250.0, modes=3)

        assert mms.name
        assert "Poisson" in mms.description or "poisson" in mms.description

    def test_reports_the_symbolic_forms_it_was_built_from(self) -> None:
        # These go into the V-01 verification report. doc 07 §2.3 wants the manufactured
        # problem stated, not merely asserted to have been used — a reader has to be able
        # to re-derive it.
        mms = poisson_manufactured(length=0.02, amplitude=250.0, modes=3)

        assert "sin" in mms.exact_expression or "cos" in mms.exact_expression
        assert mms.source_expression

    def test_is_immutable(self) -> None:
        mms = poisson_manufactured(length=0.02, amplitude=250.0, modes=3)

        with pytest.raises(AttributeError):
            mms.name = "other"  # type: ignore[misc]

    def test_evaluating_outside_the_domain_is_the_callers_problem_not_a_crash(self) -> None:
        # MMS solutions are analytic expressions; they are defined outside [0, L] and
        # simply stop being the manufactured problem there. Raising would make a solver's
        # ghost cells unnecessarily awkward.
        mms = poisson_manufactured(length=0.02, amplitude=250.0, modes=3)

        assert np.isfinite(mms.exact(np.array([-0.001, 0.021]))).all()
