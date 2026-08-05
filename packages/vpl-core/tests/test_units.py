"""Unit-safety layer — doc 08 §5.

The contract this module exists to enforce: dimensional quantities cross module
boundaries, raw arrays live only inside hot loops, and the transition between the two
is an assertion rather than a convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.units import Q_, UREG, DimensionalityError, magnitude_in


class TestSharedRegistry:
    def test_registry_is_a_single_shared_instance(self) -> None:
        # pint quantities built by different registries cannot interoperate, and the
        # resulting error surfaces far from its cause. One registry, imported everywhere.
        import vpl.core.units as units_a
        import vpl.core.units as units_b

        assert units_a.UREG is units_b.UREG

    def test_quantities_from_the_shared_registry_add(self) -> None:
        total = Q_(1.0, "m") + Q_(50.0, "cm")

        assert magnitude_in(total, "m") == pytest.approx(1.5)


class TestProjectSpecificUnits:
    def test_millitorr_converts_to_pascal(self) -> None:
        # doc 01 §2.1: the reference operating point is 5 mTorr = 0.667 Pa.
        assert magnitude_in(Q_(5.0, "mTorr"), "Pa") == pytest.approx(0.6666, rel=1e-3)

    def test_townsend_is_defined_for_reduced_field(self) -> None:
        # doc 03 §3.2 tabulates rate coefficients against E/N, conventionally in
        # townsend. pint does not ship the unit, so the registry must define it.
        assert magnitude_in(Q_(1.0, "Td"), "V * m**2") == pytest.approx(1e-21)

    def test_electronvolt_converts_to_kelvin_as_a_temperature(self) -> None:
        # "T_e = 3 eV" means k_B T_e = 3 eV. Without an explicit conversion path this
        # is the sort of factor-of-11604 error that survives review.
        assert magnitude_in(Q_(1.0, "eV").to("K", "boltzmann"), "K") == pytest.approx(
            11604.518, rel=1e-6
        )


class TestMagnitudeIn:
    def test_converts_and_strips_to_a_plain_float(self) -> None:
        result = magnitude_in(Q_(1.0, "km"), "m")

        assert result == pytest.approx(1000.0)
        assert isinstance(result, float)

    def test_preserves_array_shape(self) -> None:
        z = Q_(np.array([[1.0, 2.0], [3.0, 4.0]]), "mm")

        result = magnitude_in(z, "m")

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, [[1e-3, 2e-3], [3e-3, 4e-3]])

    def test_raises_on_dimensional_mismatch(self) -> None:
        with pytest.raises(DimensionalityError):
            magnitude_in(Q_(1.0, "m"), "s")

    def test_mismatch_message_names_both_dimensionalities(self) -> None:
        # A bare "cannot convert" is not actionable at 2 a.m. The message must say
        # what was supplied and what was wanted.
        with pytest.raises(DimensionalityError, match=r"length.*time|time.*length"):
            magnitude_in(Q_(1.0, "m"), "s")

    def test_rejects_a_bare_number(self) -> None:
        # Silently attaching the expected unit to a bare float is exactly the hidden
        # assumption doc 00 C4 forbids: it would let an unlabelled number through the
        # boundary check that exists to catch it.
        with pytest.raises(DimensionalityError, match="not a dimensional quantity"):
            magnitude_in(2.5, "m")  # type: ignore[arg-type]

    def test_accepts_dimensionless_quantities(self) -> None:
        assert magnitude_in(Q_(0.61, "dimensionless"), "dimensionless") == pytest.approx(0.61)


class TestQuantityConstruction:
    def test_q_builds_on_the_shared_registry(self) -> None:
        assert Q_(1.0, "m")._REGISTRY is UREG
