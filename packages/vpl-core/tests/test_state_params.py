"""Fidelity, species and control parameters — doc 03 §1 and doc 05 §2.1."""

from __future__ import annotations

import pytest

from vpl.core.state import Fidelity, PlasmaParams, Species
from vpl.core.units import Q_, DimensionalityError, magnitude_in


class TestFidelity:
    def test_names_the_four_levels_of_the_hierarchy(self) -> None:
        assert {f.name for f in Fidelity} == {"L0", "L1", "L2", "L3"}

    def test_each_level_carries_its_description(self) -> None:
        assert "analytic" in Fidelity.L0.description.lower()
        assert "fluid" in Fidelity.L1.description.lower()
        assert "kinetic" in Fidelity.L2.description.lower()
        assert "surrogate" in Fidelity.L3.description.lower()

    def test_levels_are_not_ordered(self) -> None:
        # Deliberate. L0 < L1 < L2 is defensible, but L3 is an *emulator of L2* with its
        # own budgeted error (doc 03 §5.4) — it is neither more nor less faithful than
        # what it emulates. Exposing a total order would invite `if fidelity >= L2`
        # checks that silently accept or reject the surrogate for the wrong reason.
        with pytest.raises(TypeError):
            _ = Fidelity.L0 < Fidelity.L2  # type: ignore[operator]

    def test_only_the_kinetic_level_produces_an_iedf_directly(self) -> None:
        # doc 03 §1: L2 is "the only level that produces a correct IEDF in the
        # collisional and RF-transit regimes". L1 reconstructs one from moments and
        # doc 03 §6 calls that a genuine weakness; the flag keeps the distinction
        # machine-checkable rather than a matter of memory.
        assert Fidelity.L2.resolves_distribution is True
        assert Fidelity.L1.resolves_distribution is False
        assert Fidelity.L0.resolves_distribution is False

    def test_the_surrogate_records_what_it_emulates(self) -> None:
        assert Fidelity.L3.emulates is Fidelity.L2
        assert Fidelity.L2.emulates is None

    def test_round_trips_through_its_string_value(self) -> None:
        assert Fidelity("L2") is Fidelity.L2
        assert Fidelity.L2.value == "L2"


class TestSpecies:
    def test_carries_a_name_mass_and_charge(self) -> None:
        argon = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)

        assert argon.name == "Ar+"
        assert magnitude_in(argon.mass, "u") == pytest.approx(39.948)
        assert argon.charge_number == 1

    def test_exposes_mass_in_kilograms_for_hot_loops(self) -> None:
        argon = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)

        assert argon.mass_kg == pytest.approx(6.6335e-26, rel=1e-4)

    def test_rejects_a_non_mass_quantity(self) -> None:
        with pytest.raises(DimensionalityError):
            Species(name="Ar+", mass=Q_(39.948, "m"), charge_number=1)

    def test_rejects_a_non_positive_mass(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Species(name="Ar+", mass=Q_(0.0, "u"), charge_number=1)

    def test_rejects_an_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Species(name="", mass=Q_(39.948, "u"), charge_number=1)

    def test_is_immutable(self) -> None:
        argon = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)

        with pytest.raises(AttributeError):
            argon.name = "Xe+"  # type: ignore[misc]


def _rp1(**overrides: object) -> PlasmaParams:
    """The reference operating point RP-1 of doc 01 §2.1."""
    defaults: dict[str, object] = {
        "species": Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1),
        "n_0": Q_(1e17, "m**-3"),
        "T_e": Q_(3.0, "eV"),
        "T_i": Q_(0.05, "eV"),
        "T_g": Q_(300.0, "K"),
        "pressure": Q_(5.0, "mTorr"),
        "bias": Q_(-250.0, "V"),
        "gamma_se": 0.10,
        "kappa": 1.0,
    }
    return PlasmaParams(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestPlasmaParams:
    def test_holds_the_reference_operating_point(self) -> None:
        params = _rp1()

        assert magnitude_in(params.n_0, "m**-3") == pytest.approx(1e17)
        assert magnitude_in(params.T_e, "eV") == pytest.approx(3.0)
        assert magnitude_in(params.pressure, "mTorr") == pytest.approx(5.0)

    def test_bias_is_signed_and_negative_for_a_biased_electrode(self) -> None:
        # doc 08 §6's manifest writes `bias: {value: -250.0}` while doc 03 §2.2 uses
        # V_w as a positive magnitude inside sqrt(2 V_w / T_e). Both conventions are in
        # the specification, which is exactly the silent sign ambiguity doc 02 §2 warns
        # about. Resolved here: `bias` is the signed applied electrode potential and
        # `bias_magnitude` is the positive sheath potential drop the formulas want.
        params = _rp1()

        assert magnitude_in(params.bias, "V") == pytest.approx(-250.0)
        assert magnitude_in(params.bias_magnitude, "V") == pytest.approx(250.0)

    def test_bias_magnitude_is_positive_for_a_positive_bias_too(self) -> None:
        params = _rp1(bias=Q_(50.0, "V"))

        assert magnitude_in(params.bias_magnitude, "V") == pytest.approx(50.0)

    def test_rejects_a_non_positive_density(self) -> None:
        with pytest.raises(ValueError, match="n_0"):
            _rp1(n_0=Q_(0.0, "m**-3"))

    def test_rejects_a_non_positive_electron_temperature(self) -> None:
        with pytest.raises(ValueError, match="T_e"):
            _rp1(T_e=Q_(-1.0, "eV"))

    def test_rejects_a_non_positive_pressure(self) -> None:
        with pytest.raises(ValueError, match="pressure"):
            _rp1(pressure=Q_(0.0, "mTorr"))

    def test_rejects_a_negative_secondary_emission_yield(self) -> None:
        with pytest.raises(ValueError, match="gamma_se"):
            _rp1(gamma_se=-0.01)

    def test_accepts_a_kappa_outside_its_prior_range(self) -> None:
        # doc 05 §2.1 gives kappa a uniform [1, 5] *prior*. A prior bound is not a
        # physical impossibility, and enforcing it here would make the sampler's own
        # support a hard error rather than a modelling choice. Validation covers what
        # cannot exist, not what is unlikely.
        assert _rp1(kappa=7.5).kappa == pytest.approx(7.5)

    def test_rejects_wrong_dimensions_on_every_quantity(self) -> None:
        with pytest.raises(DimensionalityError):
            _rp1(n_0=Q_(1e17, "m**-2"))
        with pytest.raises(DimensionalityError):
            _rp1(pressure=Q_(5.0, "V"))
        with pytest.raises(DimensionalityError):
            _rp1(bias=Q_(-250.0, "mTorr"))

    def test_accepts_electron_temperature_in_kelvin(self) -> None:
        # "T_e = 3 eV" and "T_e = 34813 K" are the same statement. Refusing one of them
        # would push a unit conversion into every caller, which is where they go wrong.
        params = _rp1(T_e=Q_(34813.6, "K"))

        assert magnitude_in(params.T_e_eV, "eV") == pytest.approx(3.0, rel=1e-4)

    def test_exposes_si_magnitudes_for_hot_loops(self) -> None:
        params = _rp1()

        assert params.n_0_per_m3 == pytest.approx(1e17)
        assert params.T_e_J == pytest.approx(3.0 * 1.602176634e-19, rel=1e-9)

    def test_neutral_density_follows_from_pressure_and_gas_temperature(self) -> None:
        # doc 01 §2.2: n_g = p / (k_B T_g) = 1.61e20 m^-3 at 5 mTorr, 300 K.
        params = _rp1()

        assert magnitude_in(params.n_g, "m**-3") == pytest.approx(1.61e20, rel=2e-3)

    def test_is_immutable(self) -> None:
        params = _rp1()

        with pytest.raises(AttributeError):
            params.n_0 = Q_(1e18, "m**-3")  # type: ignore[misc]

    def test_replace_returns_a_new_object(self) -> None:
        params = _rp1()

        hotter = params.replace(T_e=Q_(5.0, "eV"))

        assert magnitude_in(hotter.T_e, "eV") == pytest.approx(5.0)
        assert magnitude_in(params.T_e, "eV") == pytest.approx(3.0)
