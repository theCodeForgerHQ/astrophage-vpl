"""The kappa-distribution EEDF — doc 05 §7.1's "EEDF parameterisation" mismatch axis.

Doc 05 §7.1 lists EEDF parameterisation among the mandatory model mismatches a T2 result
must exercise, and until this module existed :func:`~vpl.physics.eedf.analytic.maxwellian_eedf`
was the only shape available — so that mismatch axis could not be exercised at all.

This is the Summers & Thorne (1991) / Pierrard & Lazar (2010) convention: a power-law tail
in ``eps / ((kappa - 3/2) T_e)`` that recovers the Maxwellian exactly as ``kappa -> inf``,
and whose ``(kappa - 3/2)`` normalisation keeps ``<eps> = 1.5 T_e`` for every valid kappa —
the same invariant :func:`~vpl.physics.eedf.analytic.maxwellian_eedf` carries. Note this is
a *different* kappa from the one :mod:`vpl.physics.eedf.analytic`'s own
:func:`~vpl.physics.eedf.analytic.generalised_eedf` already uses (an exponent inside an
``exp``, Maxwellian at 1 and Druyvesteyn at 2): the two happen to share a symbol and,
coincidentally, doc 05 §2.1's inference target, but they are structurally unrelated
families. See the module docstring of :mod:`vpl.physics.eedf.analytic` for the flag on
what that means for the inference prior.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.physics.eedf.analytic import (
    KAPPA_DISTRIBUTION_MINIMUM_KAPPA,
    kappa_eedf,
    maxwellian_eedf,
)
from vpl.physics.eedf.grid import EnergyGrid


def _fine_grid() -> EnergyGrid:
    """Wide and fine enough that the quadrature error is far below every tolerance here.

    Wider and finer than :mod:`test_eedf_analytic`'s grid of the same name, because a
    kappa distribution's power-law tail decays far more slowly than a Maxwellian's or a
    Druyvesteyn's exponential one: for kappa close to :data:`KAPPA_DISTRIBUTION_MINIMUM_KAPPA`
    the mean-energy integrand falls off close to ``eps**-1``, so a grid sized for the
    exponential families would truncate a non-negligible fraction of the tail and this
    test would be measuring grid truncation rather than the analytic normalisation.
    """
    return EnergyGrid.linear(max_ev=2000.0, n_cells=200000)


class TestTheMaxwellianLimit:
    @pytest.mark.physics
    def test_kappa_to_infinity_recovers_the_maxwellian(self) -> None:
        # The single most valuable property of this family: it is not an independent
        # shape bolted on beside the Maxwellian, it *contains* the Maxwellian as a limit.
        # kappa = 1e8 keeps every gamma-function argument well inside float64 range
        # (gamma overflows around x ~ 171) while still being deep in the asymptotic
        # regime, so the residual is a clean measure of "is this the right limit" rather
        # than of numerical breakdown.
        energy = np.linspace(0.0, 40.0, 400)
        electron_temperature_ev = 3.0

        kappa_shape = kappa_eedf(
            energy, electron_temperature_ev=electron_temperature_ev, kappa=1.0e8
        )
        maxwell = maxwellian_eedf(energy, electron_temperature_ev=electron_temperature_ev)

        np.testing.assert_allclose(kappa_shape, maxwell, rtol=1e-5)

    @pytest.mark.physics
    def test_the_residual_from_the_maxwellian_shrinks_as_kappa_grows(self) -> None:
        energy = np.linspace(0.0, 40.0, 400)
        electron_temperature_ev = 3.0
        maxwell = maxwellian_eedf(energy, electron_temperature_ev=electron_temperature_ev)

        residuals = [
            float(
                np.max(
                    np.abs(
                        kappa_eedf(
                            energy, electron_temperature_ev=electron_temperature_ev, kappa=kappa
                        )
                        - maxwell
                    )
                    / maxwell
                )
            )
            for kappa in (50.0, 5.0e3, 5.0e6)
        ]

        assert residuals == sorted(residuals, reverse=True)


class TestNormalisation:
    # kappa near KAPPA_DISTRIBUTION_MINIMUM_KAPPA is deliberately excluded here: the
    # mean-energy integrand there decays close to eps**-1, which converges but so slowly
    # that no grid of any practical size resolves it to a tight tolerance. That is a
    # property of the physics near the validity bound, not of the implementation — see
    # TestRejections for the bound itself, which is where that regime belongs.
    @pytest.mark.physics
    @pytest.mark.parametrize("kappa", [3.0, 5.0, 10.0, 50.0])
    def test_every_member_of_the_family_integrates_to_one(self, kappa: float) -> None:
        grid = _fine_grid()

        f0 = kappa_eedf(grid.centres_ev, electron_temperature_ev=3.0, kappa=kappa)

        assert grid.moment(f0, 0) == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.physics
    @pytest.mark.parametrize("kappa", [3.0, 5.0, 10.0, 50.0])
    def test_the_mean_energy_is_three_halves_t_e_independent_of_kappa(self, kappa: float) -> None:
        # The whole point of the (kappa - 3/2) normalisation convention: unlike
        # generalised_eedf's family, a kappa distribution *does* have a well-defined
        # temperature, and it is the same T_e that was asked for, for every kappa.
        grid = _fine_grid()
        electron_temperature_ev = 3.0

        f0 = kappa_eedf(
            grid.centres_ev, electron_temperature_ev=electron_temperature_ev, kappa=kappa
        )

        assert grid.moment(f0, 1) == pytest.approx(1.5 * electron_temperature_ev, rel=5e-3)


class TestTheTail:
    @pytest.mark.physics
    def test_the_tail_is_enhanced_relative_to_a_maxwellian_at_the_same_T_e(self) -> None:
        # The entire physical point of a kappa distribution: a power-law tail carries
        # more high-energy electrons than an exponential one at the same T_e, which is
        # what drives the excitation and ionisation rates doc 05 §7.1 cares about.
        energies = np.array([20.0, 25.0, 30.0, 40.0])
        electron_temperature_ev = 3.0

        kappa_shape = kappa_eedf(
            energies, electron_temperature_ev=electron_temperature_ev, kappa=2.0
        )
        maxwell = maxwellian_eedf(energies, electron_temperature_ev=electron_temperature_ev)

        ratio = kappa_shape / maxwell
        assert np.all(ratio > 1.0)
        assert np.all(np.diff(ratio) > 0.0)

    @pytest.mark.physics
    def test_a_smaller_kappa_enhances_the_tail_more(self) -> None:
        # Smaller kappa is a heavier, more power-law-like tail; larger kappa approaches
        # the Maxwellian from above. Far enough into the tail the enhancement shrinks
        # monotonically as kappa grows — "far enough" matters: at a moderate energy the
        # amplitude and the power-law exponent trade off and the ordering is not
        # monotonic, which is why this is checked deep in the tail rather than near the
        # mean energy.
        energy = np.array([1.0e3])
        electron_temperature_ev = 3.0
        maxwell = maxwellian_eedf(energy, electron_temperature_ev=electron_temperature_ev)[0]

        ratios = [
            float(
                kappa_eedf(energy, electron_temperature_ev=electron_temperature_ev, kappa=kappa)[0]
                / maxwell
            )
            for kappa in (1.6, 2.0, 5.0, 50.0)
        ]

        assert ratios == sorted(ratios, reverse=True)


class TestRejections:
    @pytest.mark.parametrize("kappa", [1.5, 1.0, 0.0, -1.0])
    def test_kappa_at_or_below_the_validity_bound_is_refused(self, kappa: float) -> None:
        # kappa = 3/2 is not merely inconvenient, it is where <eps> stops existing: the
        # tail integral integral eps**(3/2) f0 deps diverges. A ValueError here is the
        # correct outcome, not an edge case to special-case away.
        with pytest.raises(ValueError, match="kappa"):
            kappa_eedf(np.array([1.0]), electron_temperature_ev=3.0, kappa=kappa)

    def test_the_validity_bound_is_three_halves(self) -> None:
        assert pytest.approx(1.5) == KAPPA_DISTRIBUTION_MINIMUM_KAPPA

    def test_a_non_positive_electron_temperature_is_refused(self) -> None:
        with pytest.raises(ValueError, match="T_e"):
            kappa_eedf(np.array([1.0]), electron_temperature_ev=0.0, kappa=2.0)

    def test_a_negative_energy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            kappa_eedf(np.array([-1.0]), electron_temperature_ev=3.0, kappa=2.0)

    @pytest.mark.physics
    def test_a_large_kappa_does_not_warn_or_produce_nan(self) -> None:
        # filterwarnings = ["error"] in pyproject.toml turns any RuntimeWarning (gamma
        # overflow, log(0), etc.) into a failure, so this is a numerical-robustness
        # regression test as much as a physics one.
        f0 = kappa_eedf(np.linspace(0.0, 400.0, 4000), electron_temperature_ev=3.0, kappa=1.0e8)

        assert np.all(np.isfinite(f0))
        assert np.all(f0 >= 0.0)
