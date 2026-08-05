"""Scharfetter-Gummel exponential fitting — doc 03 §3.4.

Doc 03 §3.4 chooses exponential fitting for ion advection because "centred differences
oscillate in the strongly-drift-dominated sheath". Two things have to be true for that
choice to be worth anything, and they are tested separately here:

1. The fitted diffusivity **is** Scharfetter-Gummel. The equivalence between the
   exponentially-fitted P1 finite element and the Scharfetter-Gummel finite-volume flux
   in 1-D is a theorem (Il'in 1969; Scharfetter & Gummel 1969; Brooks & Hughes 1982
   §3.3), so it can be asserted rather than assumed. A scheme that merely upwinds would
   pass a stability test and fail this one.
2. The limits are right. ``D_fitted -> D`` as the cell Peclet number vanishes and
   ``D_fitted -> |a| h / 2`` — first-order upwind — as it diverges. Getting the second
   limit wrong gives a scheme that is stable and first-order *everywhere*, which is
   exactly the silent failure the order study of V-02 exists to catch.

The oscillation claim itself is tested in ``test_fluid_coupled.py``, where there is a
solver to oscillate.

The module needs no FEniCSx — the mathematics is pure NumPy — but it is imported from a
package whose ``__init__`` builds on dolfinx, so it skips with the rest.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from vpl.physics.fluid import (
    bernoulli,
    cell_peclet,
    fitted_diffusivity,
    scharfetter_gummel_coefficients,
)

# ── the Bernoulli function ──────────────────────────────────────────────────────


class TestBernoulli:
    """``B(x) = x / (exp(x) - 1)``, the weight Scharfetter-Gummel is built from."""

    def test_value_at_zero_is_one(self) -> None:
        assert bernoulli(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_matches_the_closed_form_away_from_zero(self) -> None:
        x = np.array([-8.0, -1.0, 0.5, 3.0, 12.0])
        assert bernoulli(x) == pytest.approx(x / np.expm1(x))

    def test_is_continuous_through_the_removable_singularity(self) -> None:
        """The naive form is 0/0 at the origin and cancels catastrophically near it.

        A solver whose cell Peclet number passes through zero — which happens wherever
        the field reverses — would otherwise emit a NaN row into the Jacobian.
        """
        x = np.array([-1e-14, -1e-8, 0.0, 1e-8, 1e-14])
        values = bernoulli(x)
        assert np.all(np.isfinite(values))
        assert values == pytest.approx(1.0 - x / 2.0, abs=1e-12)

    def test_does_not_overflow_for_a_large_positive_argument(self) -> None:
        assert bernoulli(np.array([800.0]))[0] == pytest.approx(0.0, abs=1e-300)

    def test_grows_linearly_for_a_large_negative_argument(self) -> None:
        assert bernoulli(np.array([-800.0]))[0] == pytest.approx(800.0, rel=1e-12)

    def test_identity_b_of_minus_x_equals_b_of_x_plus_x(self) -> None:
        """``B(-x) = B(x) + x``, the identity the two-point flux is assembled from."""
        x = np.array([-5.0, -0.3, 0.3, 5.0])
        assert bernoulli(-x) == pytest.approx(bernoulli(x) + x)


# ── the fitted diffusivity ──────────────────────────────────────────────────────


class TestFittedDiffusivity:
    """``D_fitted = (|a| h / 2) coth(|a| h / (2 D))`` — Il'in / Allen-Southwell."""

    def test_reduces_to_the_physical_diffusivity_at_vanishing_peclet(self) -> None:
        assert fitted_diffusivity(velocity=1e-6, diffusivity=1.0, cell_size=1e-3) == pytest.approx(
            1.0, rel=1e-9
        )

    def test_reduces_to_upwind_at_infinite_peclet(self) -> None:
        """``D -> 0`` is the collisionless sheath, and the limit is first-order upwind."""
        assert fitted_diffusivity(velocity=-4.0, diffusivity=0.0, cell_size=0.25) == pytest.approx(
            0.5 * 4.0 * 0.25
        )

    def test_is_never_below_the_physical_diffusivity(self) -> None:
        """Artificial diffusion is added, never subtracted.

        A fitted value below ``D`` would be anti-diffusion, which is unconditionally
        unstable and would look, in a coarse study, like an unusually accurate scheme.
        """
        for peclet in (1e-6, 1e-3, 0.1, 1.0, 10.0, 1e4):
            fitted = fitted_diffusivity(velocity=2.0 * peclet, diffusivity=1.0, cell_size=1.0)
            assert fitted >= 1.0

    def test_is_symmetric_in_the_sign_of_the_velocity(self) -> None:
        forward = fitted_diffusivity(velocity=3.0, diffusivity=0.5, cell_size=0.1)
        reverse = fitted_diffusivity(velocity=-3.0, diffusivity=0.5, cell_size=0.1)
        assert forward == pytest.approx(reverse)

    def test_zero_velocity_leaves_the_diffusivity_untouched(self) -> None:
        assert fitted_diffusivity(velocity=0.0, diffusivity=2.5, cell_size=0.1) == pytest.approx(
            2.5
        )

    def test_rejects_a_negative_diffusivity(self) -> None:
        with pytest.raises(ValueError, match="diffusivity"):
            fitted_diffusivity(velocity=1.0, diffusivity=-1.0, cell_size=0.1)

    def test_rejects_a_non_positive_cell_size(self) -> None:
        with pytest.raises(ValueError, match="cell_size"):
            fitted_diffusivity(velocity=1.0, diffusivity=1.0, cell_size=0.0)


# ── the equivalence that makes this Scharfetter-Gummel ──────────────────────────


class TestScharfetterGummelEquivalence:
    """Exponentially-fitted P1 finite elements equal Scharfetter-Gummel in 1-D.

    Doc 03 §3.4 specifies the scheme *by name*, so the implementation has to be checkable
    against the name rather than against "it is stable".
    """

    @pytest.mark.parametrize("peclet", [-20.0, -3.0, -0.25, 0.0, 0.25, 3.0, 20.0])
    def test_two_point_flux_matches_scharfetter_and_gummel_1969(self, peclet: float) -> None:
        """``J = (D/h) [B(-D) n_j - B(D) n_{j+1}]`` with ``D = a h / D`` the cell Peclet.

        Written directly from Scharfetter & Gummel (1969) eq. 20, with the cell Peclet
        number standing in for their normalised potential difference.
        """
        h, diffusivity = 0.05, 0.7
        velocity = 2.0 * peclet * diffusivity / h
        delta = velocity * h / diffusivity

        left, right = scharfetter_gummel_coefficients(
            velocity=velocity, diffusivity=diffusivity, cell_size=h
        )

        assert left == pytest.approx(diffusivity / h * bernoulli(np.array([-delta]))[0], rel=1e-12)
        assert right == pytest.approx(diffusivity / h * bernoulli(np.array([delta]))[0], rel=1e-12)

    @pytest.mark.parametrize("peclet", [-20.0, -3.0, -0.25, 0.25, 3.0, 20.0])
    def test_fitted_diffusivity_is_the_finite_element_form_of_the_same_scheme(
        self, peclet: float
    ) -> None:
        """``D_fitted = D B(a h / D) + a h / 2``.

        This identity is the whole content of the equivalence: it says the artificial
        diffusion added to the P1 Galerkin operator reproduces, coefficient for
        coefficient, the Scharfetter-Gummel two-point flux. If it fails, the solver is
        stabilised by something that is not the scheme doc 03 §3.4 specified.
        """
        h, diffusivity = 0.05, 0.7
        velocity = 2.0 * peclet * diffusivity / h
        delta = velocity * h / diffusivity

        expected = diffusivity * bernoulli(np.array([delta]))[0] + velocity * h / 2.0
        assert fitted_diffusivity(
            velocity=velocity, diffusivity=diffusivity, cell_size=h
        ) == pytest.approx(expected, rel=1e-12)

    def test_coefficients_never_go_negative_so_the_scheme_is_monotone(self) -> None:
        """Non-negative off-diagonals are the discrete maximum principle.

        Centred differences lose it at cell Peclet one; Scharfetter-Gummel never does,
        and that is the entire reason doc 03 §3.4 chose it. The *downwind* weight
        underflows to exactly zero once the cell Peclet number exceeds about 350 — that
        is the pure-upwind limit arriving, and it is the correct answer rather than a
        loss of positivity.
        """
        for peclet in (-1e3, -50.0, -1.0, 1.0, 50.0, 1e3):
            left, right = scharfetter_gummel_coefficients(
                velocity=2.0 * peclet, diffusivity=1.0, cell_size=1.0
            )
            assert left >= 0.0
            assert right >= 0.0
            assert max(left, right) > 0.0

    def test_the_flux_vanishes_on_the_homogeneous_solution(self) -> None:
        """SG reproduces ``exp(a z / D)`` exactly, at any resolution.

        This is the defining property of exponential fitting and the reason doc 03 §3.4
        specifies it rather than plain upwinding: the discrete flux annihilates the exact
        solution of ``a n - D n' = 0``, so the sharp sheath front is resolved without
        refining onto it. Plain upwinding does not have this property.
        """
        velocity, diffusivity, h = 5.0, 0.1, 0.3
        left, right = scharfetter_gummel_coefficients(
            velocity=velocity, diffusivity=diffusivity, cell_size=h
        )
        n_left = 1.0
        n_right = float(np.exp(velocity * h / diffusivity))

        flux = left * n_left - right * n_right
        assert flux == pytest.approx(0.0, abs=1e-9 * right * n_right)


class TestCellPeclet:
    """``|a| h / (2 D)`` — reported so a convergence study can state its regime."""

    def test_matches_the_definition(self) -> None:
        assert cell_peclet(velocity=-4.0, diffusivity=0.5, cell_size=0.25) == pytest.approx(1.0)

    def test_is_infinite_in_the_collisionless_limit(self) -> None:
        """``D = 0`` is not an error: it is the sheath doc 03 §7 V-03 is stated in."""
        assert cell_peclet(velocity=1.0, diffusivity=0.0, cell_size=1.0) == float("inf")

    def test_rejects_a_negative_diffusivity(self) -> None:
        with pytest.raises(ValueError, match="diffusivity"):
            cell_peclet(velocity=1.0, diffusivity=-1.0, cell_size=1.0)
