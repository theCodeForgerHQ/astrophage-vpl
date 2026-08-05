"""Analytic EEDF families — doc 03 §3.2, doc 05 §2.1.

Doc 03 §3.2 names the shapes the low-pressure sheath actually produces: "the EEDF is
typically bi-Maxwellian or Druyvesteyn, and the high-energy tail — the part that matters
for ionisation and for OES line ratios — is depleted." Doc 05 §2.1 makes the shape an
*inferred* parameter, ``kappa``, uniform on [1, 5] and labelled "Maxwellian → Druyvesteyn".

These tests pin the one-parameter family that spans both, and the tail depletion that is
the entire physical point.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.physics.eedf.analytic import (
    DRUYVESTEYN_KAPPA,
    MAXWELLIAN_KAPPA,
    AnalyticEedf,
    druyvesteyn_eedf,
    generalised_eedf,
    maxwellian_eedf,
)
from vpl.physics.eedf.grid import EnergyGrid


def _fine_grid() -> EnergyGrid:
    """Wide and fine enough that the quadrature error is far below every tolerance here."""
    return EnergyGrid.linear(max_ev=200.0, n_cells=20000)


class TestNormalisation:
    @pytest.mark.physics
    @pytest.mark.parametrize("kappa", [1.0, 1.5, 2.0, 3.0, 5.0])
    def test_every_member_of_the_family_integrates_to_one(self, kappa: float) -> None:
        grid = _fine_grid()

        f0 = generalised_eedf(grid.centres_ev, mean_energy_ev=4.5, kappa=kappa)

        assert grid.moment(f0, 0) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.physics
    @pytest.mark.parametrize("kappa", [1.0, 1.5, 2.0, 3.0, 5.0])
    def test_the_mean_energy_is_the_one_that_was_asked_for(self, kappa: float) -> None:
        grid = _fine_grid()

        f0 = generalised_eedf(grid.centres_ev, mean_energy_ev=4.5, kappa=kappa)

        assert grid.moment(f0, 1) == pytest.approx(4.5, rel=1e-6)

    @pytest.mark.physics
    def test_a_maxwellian_carries_three_halves_kT_e(self) -> None:
        grid = _fine_grid()

        f0 = maxwellian_eedf(grid.centres_ev, electron_temperature_ev=3.0)

        assert grid.moment(f0, 1) == pytest.approx(4.5, rel=1e-6)


class TestTheFamily:
    @pytest.mark.physics
    def test_kappa_one_is_the_maxwellian(self) -> None:
        energy = np.linspace(0.0, 40.0, 400)

        family = generalised_eedf(energy, mean_energy_ev=4.5, kappa=MAXWELLIAN_KAPPA)
        maxwell = maxwellian_eedf(energy, electron_temperature_ev=3.0)

        np.testing.assert_allclose(family, maxwell, rtol=1e-12)

    @pytest.mark.physics
    def test_kappa_two_is_the_druyvesteyn(self) -> None:
        energy = np.linspace(0.0, 40.0, 400)

        family = generalised_eedf(energy, mean_energy_ev=4.5, kappa=DRUYVESTEYN_KAPPA)
        druyvesteyn = druyvesteyn_eedf(energy, mean_energy_ev=4.5)

        np.testing.assert_allclose(family, druyvesteyn, rtol=1e-12)

    @pytest.mark.physics
    def test_a_maxwellian_is_exactly_exponential_in_energy(self) -> None:
        # The defining property, and the one the Boltzmann-electron relation of doc 03
        # §3.1 leans on: n_e = n_0 exp(e Phi / k T_e) is this shape and no other.
        energy = np.array([1.0, 4.0])

        f0 = maxwellian_eedf(energy, electron_temperature_ev=3.0)

        assert f0[1] / f0[0] == pytest.approx(np.exp(-3.0 / 3.0))

    @pytest.mark.physics
    def test_the_druyvesteyn_tail_is_depleted_relative_to_a_maxwellian(self) -> None:
        # doc 03 §3.2's stated reason for this whole module: "the high-energy tail — the
        # part that matters for ionisation and for OES line ratios — is depleted."
        # Compared at equal *mean energy*, which is the like-for-like comparison, since
        # that is what a Maxwellian fit to the same plasma would report.
        ionisation_threshold_ev = 15.76

        maxwell = generalised_eedf(
            np.array([ionisation_threshold_ev]), mean_energy_ev=4.5, kappa=MAXWELLIAN_KAPPA
        )
        druyvesteyn = generalised_eedf(
            np.array([ionisation_threshold_ev]), mean_energy_ev=4.5, kappa=DRUYVESTEYN_KAPPA
        )

        # A factor 8.6 at the argon ionisation threshold for a 3 eV plasma, and the
        # depletion deepens with energy — checked below, because a single point could be
        # matched by a rescaling while the shapes stayed the same.
        assert druyvesteyn[0] / maxwell[0] < 0.2

    @pytest.mark.physics
    def test_the_depletion_deepens_with_energy(self) -> None:
        energies = np.array([4.5, 10.0, 15.76, 25.0])

        maxwell = generalised_eedf(energies, mean_energy_ev=4.5, kappa=MAXWELLIAN_KAPPA)
        druyvesteyn = generalised_eedf(energies, mean_energy_ev=4.5, kappa=DRUYVESTEYN_KAPPA)

        ratio = druyvesteyn / maxwell
        assert np.all(np.diff(ratio) < 0.0)
        # Three further orders of magnitude between the threshold and 25 eV.
        assert ratio[-1] / ratio[2] < 1e-2

    @pytest.mark.physics
    def test_higher_kappa_depletes_the_tail_monotonically(self) -> None:
        threshold = np.array([15.76])

        tails = [
            float(generalised_eedf(threshold, mean_energy_ev=4.5, kappa=k)[0])
            for k in (1.0, 1.5, 2.0, 3.0)
        ]

        assert tails == sorted(tails, reverse=True)

    @pytest.mark.physics
    def test_every_family_member_agrees_at_zero_energy_only_by_normalisation(self) -> None:
        # All members are finite and positive at eps = 0; none of them vanishes there.
        for kappa in (1.0, 2.0, 5.0):
            assert generalised_eedf(np.array([0.0]), mean_energy_ev=4.5, kappa=kappa)[0] > 0.0


class TestTheEnum:
    def test_the_named_shapes_carry_their_kappa(self) -> None:
        assert AnalyticEedf.MAXWELLIAN.kappa == MAXWELLIAN_KAPPA
        assert AnalyticEedf.DRUYVESTEYN.kappa == DRUYVESTEYN_KAPPA

    def test_a_named_shape_evaluates_to_its_family_member(self) -> None:
        energy = np.linspace(0.0, 20.0, 100)

        np.testing.assert_allclose(
            AnalyticEedf.DRUYVESTEYN.evaluate(energy, mean_energy_ev=4.5),
            generalised_eedf(energy, mean_energy_ev=4.5, kappa=DRUYVESTEYN_KAPPA),
        )

    def test_the_enum_round_trips_through_its_string(self) -> None:
        assert AnalyticEedf("druyvesteyn") is AnalyticEedf.DRUYVESTEYN


class TestRejections:
    @pytest.mark.parametrize("kappa", [0.0, -1.0])
    def test_a_non_positive_kappa_is_refused(self, kappa: float) -> None:
        with pytest.raises(ValueError, match="kappa"):
            generalised_eedf(np.array([1.0]), mean_energy_ev=4.5, kappa=kappa)

    @pytest.mark.parametrize("mean_energy_ev", [0.0, -2.0])
    def test_a_non_positive_mean_energy_is_refused(self, mean_energy_ev: float) -> None:
        with pytest.raises(ValueError, match="mean energy"):
            generalised_eedf(np.array([1.0]), mean_energy_ev=mean_energy_ev, kappa=1.0)

    def test_a_negative_energy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            generalised_eedf(np.array([-1.0]), mean_energy_ev=4.5, kappa=1.0)

    def test_a_non_positive_electron_temperature_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mean energy"):
            maxwellian_eedf(np.array([1.0]), electron_temperature_ev=0.0)

    @pytest.mark.physics
    def test_a_large_kappa_does_not_overflow_far_out_in_the_tail(self) -> None:
        # exp(-(eps/eps_0)**5) underflows long before the grid ends. Underflow to zero is
        # correct and must not be a warning: pytest turns warnings into errors, and an
        # EEDF that raised on its own tail would be unusable at exactly the energies
        # doc 03 §3.2 cares about.
        f0 = generalised_eedf(np.linspace(0.0, 400.0, 4000), mean_energy_ev=4.5, kappa=5.0)

        assert np.all(np.isfinite(f0))
        assert f0[-1] == 0.0
