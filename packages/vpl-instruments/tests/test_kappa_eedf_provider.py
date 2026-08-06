"""``KappaEedf`` against the :class:`~vpl.instruments.oes.instrument.EedfProvider` contract.

Doc 05 §7.1 lists "EEDF parameterisation" among the mandatory model mismatches a T2 result
must exercise, and :class:`~vpl.instruments.oes.instrument.MaxwellianEedf` was, until this
module, the only :class:`~vpl.instruments.oes.instrument.EedfProvider` available — so that
mismatch axis had nothing to switch to. ``KappaEedf`` is the second one: same protocol,
same grid-normalisation contract, a genuinely different shape underneath (see
:mod:`vpl.physics.eedf.analytic` for the shape itself and the flag on
``PlasmaParams.kappa``'s prior).

These tests mirror what a :class:`~vpl.instruments.oes.instrument.MaxwellianEedf` test file
would check were one to exist: protocol conformance, grid normalisation, per-temperature
caching, construction-time validation and ``__repr__`` — plus one check that is specific to
having *two* providers now, that they disagree at the same ``T_e``.
"""

from __future__ import annotations

import numpy as np
import pytest

from oes_system import energy_grid
from vpl.instruments.oes.instrument import EedfProvider, KappaEedf, MaxwellianEedf
from vpl.physics.eedf.analytic import KAPPA_DISTRIBUTION_MINIMUM_KAPPA
from vpl.physics.eedf.grid import EnergyGrid


@pytest.fixture(scope="module")
def grid() -> EnergyGrid:
    return energy_grid()


class TestProtocolConformance:
    def test_a_kappa_eedf_satisfies_the_protocol_structurally(self, grid: EnergyGrid) -> None:
        provider = KappaEedf(grid=grid, kappa=2.0)

        assert isinstance(provider, EedfProvider)


class TestNormalisation:
    def test_f0_is_normalised_on_the_grid(self, grid: EnergyGrid) -> None:
        provider = KappaEedf(grid=grid, kappa=2.0)

        f0 = provider.f0(electron_temperature_ev=3.0)

        assert grid.moment(f0, 0) == pytest.approx(1.0, rel=1e-9)

    def test_f0_is_the_grid_renormalised_kappa_eedf_shape(self, grid: EnergyGrid) -> None:
        # Same relationship MaxwellianEedf has to maxwellian_eedf: the provider calls the
        # distribution function and then renormalises on the discrete grid, so the two
        # must agree once that renormalisation is applied to both.
        from vpl.physics.eedf.analytic import kappa_eedf

        provider = KappaEedf(grid=grid, kappa=2.0)

        expected = grid.normalise(
            kappa_eedf(grid.centres_ev, electron_temperature_ev=3.0, kappa=2.0)
        )
        np.testing.assert_allclose(provider.f0(electron_temperature_ev=3.0), expected)


class TestCaching:
    def test_f0_is_cached_per_temperature(self, grid: EnergyGrid) -> None:
        provider = KappaEedf(grid=grid, kappa=2.0)

        first = provider.f0(electron_temperature_ev=3.0)
        second = provider.f0(electron_temperature_ev=3.0)

        assert first is second

    def test_different_temperatures_are_not_confused(self, grid: EnergyGrid) -> None:
        provider = KappaEedf(grid=grid, kappa=2.0)

        cool = provider.f0(electron_temperature_ev=2.0)
        hot = provider.f0(electron_temperature_ev=4.0)

        assert not np.allclose(cool, hot)


class TestValidation:
    def test_construction_rejects_kappa_at_or_below_the_validity_bound(
        self, grid: EnergyGrid
    ) -> None:
        # Fail fast, at construction, the same way OesInstrument.centre_wavelength_nm's
        # setter validates eagerly rather than waiting for the first forward() — a bad
        # kappa should be caught at the manifest line that set it, not several calls later.
        with pytest.raises(ValueError, match="kappa"):
            KappaEedf(grid=grid, kappa=KAPPA_DISTRIBUTION_MINIMUM_KAPPA)

    def test_f0_rejects_a_non_positive_electron_temperature(self, grid: EnergyGrid) -> None:
        provider = KappaEedf(grid=grid, kappa=2.0)

        with pytest.raises(ValueError, match="T_e"):
            provider.f0(electron_temperature_ev=0.0)


class TestRepr:
    def test_repr_names_the_class_and_kappa(self, grid: EnergyGrid) -> None:
        provider = KappaEedf(grid=grid, kappa=2.0)

        rendered = repr(provider)

        assert "KappaEedf" in rendered
        assert "2.0" in rendered


class TestGenuinelyDifferentFromTheMaxwellian:
    def test_the_two_providers_disagree_at_the_same_T_e(self, grid: EnergyGrid) -> None:
        # The point of the whole task: doc 05 §7.1's "EEDF parameterisation" mismatch
        # needs a second *shape*, not a relabelled Maxwellian. At the same T_e the two
        # providers must produce measurably different distributions.
        kappa_provider = KappaEedf(grid=grid, kappa=2.0)
        maxwellian_provider = MaxwellianEedf(grid=grid)

        kappa_f0 = kappa_provider.f0(electron_temperature_ev=3.0)
        maxwellian_f0 = maxwellian_provider.f0(electron_temperature_ev=3.0)

        assert not np.allclose(kappa_f0, maxwellian_f0)
