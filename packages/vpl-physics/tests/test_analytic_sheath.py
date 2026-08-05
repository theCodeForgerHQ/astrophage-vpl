"""L0 analytic sheath models — doc 03 §2.

Two kinds of test live here and they are marked differently.

*Physics* tests (``@pytest.mark.physics``) assert numbers that the specification states.
Every derived length scale tabulated in doc 01 §2.2 has one, and the ion energy flux of
doc 03 §2.3 — 6.6 kW·m⁻², the number every other fidelity level must reproduce in the
collisionless high-bias limit — is verification gate V-03. If one of these fails, either
the implementation or the document is wrong, and doc 00 C4 says the resolution goes on
the record rather than into a tuned constant.

*Software* tests assert the contract: units are what they claim, degenerate inputs raise
instead of returning ``inf``, and the assembled state satisfies
:data:`vpl.core.state.REQUIRED_FIELDS`.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from vpl.core.constants import ELEMENTARY_CHARGE
from vpl.core.params import default_registry
from vpl.core.state import Fidelity, PlasmaParams, SpatialGrid, Species
from vpl.core.units import Q_, DimensionalityError, Quantity, magnitude_in
from vpl.physics.analytic.sheath import (
    DEFAULT_EDGE_TO_CENTRE_RATIO,
    GAMMA_I_ADIABATIC,
    GAMMA_I_COLD_ION,
    SIGMA_CX_ARGON,
    AnalyticSheathSolver,
    CollisionalRegime,
    SheathModel,
    bohm_speed,
    child_langmuir_current_density,
    child_langmuir_potential,
    child_langmuir_thickness,
    collisional_current_density,
    collisionality,
    cx_mean_free_path,
    debye_length,
    ion_energy_flux,
    ion_flux,
    matrix_sheath_potential,
    matrix_sheath_thickness,
    matrix_sheath_validity_ratio,
    sheath_edge_density,
)

# ── the reference operating point ───────────────────────────────────────────────

ARGON = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)

#: doc 07 B-13 exercises a different ``m_i`` to prove nothing is argon-specific.
XENON = Species(name="Xe+", mass=Q_(131.293, "u"), charge_number=1)


def _rp1(**overrides: object) -> PlasmaParams:
    """The reference operating point RP-1 of doc 01 §2.1."""
    defaults: dict[str, object] = {
        "species": ARGON,
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


def _m(value: Quantity, units: str) -> float:
    """Scalar magnitude, for readability in the assertions below."""
    return float(magnitude_in(value, units))


# ── doc 01 §2.2, the tabulated length scales ────────────────────────────────────


class TestDerivedLengthScales:
    """Every number doc 01 §2.2 tabulates, reproduced from the documented formula."""

    @pytest.mark.physics
    def test_debye_length_at_rp1_is_40_7_micrometres(self) -> None:
        assert _m(debye_length(_rp1()), "um") == pytest.approx(40.7, rel=1e-2)

    @pytest.mark.physics
    def test_bohm_speed_at_rp1_is_2_69_kilometres_per_second(self) -> None:
        # doc 01 §2.2 quotes c_s = sqrt(e T_e / m_i), i.e. the cold-ion form. doc 03 §2.1
        # writes sqrt(e (T_e + gamma_i T_i) / m_i) with gamma_i = 3. Both are in the
        # Baseline specification; the default reproduces the tabulated number and the
        # discrepancy is stated rather than silently resolved (doc 00 C4).
        assert _m(bohm_speed(_rp1()), "km/s") == pytest.approx(2.69, rel=1e-2)

    @pytest.mark.physics
    def test_child_langmuir_thickness_at_rp1_is_0_89_millimetres(self) -> None:
        assert _m(child_langmuir_thickness(_rp1()), "mm") == pytest.approx(0.89, rel=2e-2)

    @pytest.mark.physics
    def test_neutral_density_at_rp1_is_1_61e20_per_cubic_metre(self) -> None:
        # Owned by PlasmaParams.n_g, but it is a doc 01 §2.2 line and the sheath models
        # consume it through cx_mean_free_path, so it is asserted where it is used.
        assert _m(_rp1().n_g, "m**-3") == pytest.approx(1.61e20, rel=1e-2)

    @pytest.mark.physics
    def test_charge_exchange_mean_free_path_at_rp1_is_12_4_millimetres(self) -> None:
        assert _m(cx_mean_free_path(_rp1(), SIGMA_CX_ARGON), "mm") == pytest.approx(12.4, rel=1e-2)

    @pytest.mark.physics
    def test_collisionality_at_rp1_is_0_072(self) -> None:
        assert collisionality(_rp1(), SIGMA_CX_ARGON) == pytest.approx(0.072, rel=2e-2)

    @pytest.mark.physics
    def test_rp1_is_near_collisionless(self) -> None:
        # The classification doc 01 §2.2 attaches to the number, made machine-checkable.
        assert collisionality(_rp1(), SIGMA_CX_ARGON) < 0.1

    @pytest.mark.physics
    def test_fifty_millitorr_crosses_into_the_collisional_regime(self) -> None:
        # doc 01 §2.2: "At 50 mTorr the same formulas give lambda_CX ~ 1.24 mm and
        # s/lambda_CX ~ 0.7". The pressure axis spanning this transition is the stated
        # justification for modelling both limits.
        collisional = _rp1(pressure=Q_(50.0, "mTorr"))

        assert _m(cx_mean_free_path(collisional, SIGMA_CX_ARGON), "mm") == pytest.approx(
            1.24, rel=1e-2
        )
        assert collisionality(collisional, SIGMA_CX_ARGON) == pytest.approx(0.7, rel=5e-2)


# ── doc 03 §2.3, verification gate V-03 ─────────────────────────────────────────


class TestIonEnergyFlux:
    """V-03. The headline number of the L0 level."""

    @pytest.mark.physics
    def test_ion_energy_flux_at_rp1_is_6_6_kilowatts_per_square_metre(self) -> None:
        # doc 03 §2.3: Gamma_i = 0.61e17 * 2691 = 1.64e20 m^-2 s^-1 at <E_i> = 250 eV,
        # giving Gamma_E ~ 6.6 kW/m^2 = 0.66 W/cm^2. "This is the number every other
        # model must reproduce in this limit."
        assert _m(ion_energy_flux(_rp1()), "kW/m**2") == pytest.approx(6.6, rel=1e-2)

    @pytest.mark.physics
    def test_the_same_flux_in_watts_per_square_centimetre(self) -> None:
        assert _m(ion_energy_flux(_rp1()), "W/cm**2") == pytest.approx(0.66, rel=1e-2)

    @pytest.mark.physics
    def test_ion_particle_flux_at_rp1_is_1_64e20_per_square_metre_per_second(self) -> None:
        assert _m(ion_flux(_rp1()), "m**-2 / s") == pytest.approx(1.64e20, rel=1e-2)

    @pytest.mark.physics
    def test_the_flux_is_the_particle_flux_times_the_energy_per_ion(self) -> None:
        # doc 01 §1.2's decomposition, which is the information decomposition the whole
        # measurement design rests on. It must hold identically, not approximately.
        params = _rp1()
        per_ion = ELEMENTARY_CHARGE * params.bias_magnitude

        assert _m(ion_energy_flux(params), "W/m**2") == pytest.approx(
            _m(ion_flux(params) * per_ion, "W/m**2"), rel=1e-12
        )

    @pytest.mark.physics
    def test_flux_is_positive_toward_the_wall(self) -> None:
        # doc 02 §2: flux toward the wall is positive. The bias is signed and negative;
        # the flux that results from it is not.
        assert _m(ion_energy_flux(_rp1()), "W/m**2") > 0.0
        assert _m(ion_energy_flux(_rp1(bias=Q_(250.0, "V"))), "W/m**2") > 0.0


class TestChildLangmuirSelfConsistency:
    """The matching that produces the doc 03 §2.3 thickness, checked both ways."""

    @pytest.mark.physics
    def test_child_langmuir_current_matches_the_bohm_flux_at_the_matched_thickness(
        self,
    ) -> None:
        # doc 03 §2.3: "Matching J_i to the Bohm flux e n_s c_s gives the sheath
        # thickness s = (sqrt2/3) lambda_D (2 V_w / T_e)^(3/4)". The identity only closes
        # if lambda_D is evaluated at the sheath-edge density, which is what h_l selects.
        params = _rp1()
        h_l = DEFAULT_EDGE_TO_CENTRE_RATIO
        thickness = child_langmuir_thickness(params, h_l=h_l)

        bohm = ELEMENTARY_CHARGE * sheath_edge_density(params, h_l) * bohm_speed(params)

        assert _m(child_langmuir_current_density(params, thickness), "A/m**2") == pytest.approx(
            _m(bohm, "A/m**2"), rel=1e-10
        )

    @pytest.mark.physics
    def test_the_documented_thickness_uses_the_bulk_debye_length(self) -> None:
        # doc 01 §2.2 evaluates lambda_D at n_0, not at n_s, and gets 0.89 mm. The
        # self-consistent matching above needs n_s, giving s / sqrt(h_l) = 1.14 mm. The
        # two differ by 28 % and both are in the Baseline specification; see the module
        # docstring of vpl.physics.analytic.sheath.
        params = _rp1()
        documented = child_langmuir_thickness(params)
        self_consistent = child_langmuir_thickness(params, h_l=DEFAULT_EDGE_TO_CENTRE_RATIO)

        assert _m(documented, "mm") == pytest.approx(0.89, rel=2e-2)
        assert _m(self_consistent, "mm") == pytest.approx(
            _m(documented, "mm") / np.sqrt(DEFAULT_EDGE_TO_CENTRE_RATIO), rel=1e-12
        )


# ── scaling laws ────────────────────────────────────────────────────────────────


class TestScalingLaws:
    """The exponents are the physics; the prefactors are checked above."""

    @pytest.mark.physics
    @given(st.floats(min_value=1.0, max_value=1000.0))
    def test_sheath_thickness_scales_as_bias_to_the_three_quarters(self, volts: float) -> None:
        reference = child_langmuir_thickness(_rp1(bias=Q_(-1.0, "V")))
        scaled = child_langmuir_thickness(_rp1(bias=Q_(-volts, "V")))

        assert _m(scaled, "m") == pytest.approx(_m(reference, "m") * volts**0.75, rel=1e-10)

    @pytest.mark.physics
    @pytest.mark.parametrize("factor", [2.0, 4.0, 10.0])
    def test_child_langmuir_current_scales_as_bias_to_the_three_halves(self, factor: float) -> None:
        thickness = Q_(1.0, "mm")
        base = child_langmuir_current_density(_rp1(bias=Q_(-100.0, "V")), thickness)
        scaled = child_langmuir_current_density(_rp1(bias=Q_(-100.0 * factor, "V")), thickness)

        assert _m(scaled, "A/m**2") == pytest.approx(_m(base, "A/m**2") * factor**1.5, rel=1e-12)

    @pytest.mark.physics
    @pytest.mark.parametrize("factor", [2.0, 4.0])
    def test_child_langmuir_current_scales_as_the_inverse_square_of_thickness(
        self, factor: float
    ) -> None:
        params = _rp1()
        base = child_langmuir_current_density(params, Q_(1.0, "mm"))
        thicker = child_langmuir_current_density(params, Q_(factor, "mm"))

        assert _m(thicker, "A/m**2") == pytest.approx(_m(base, "A/m**2") / factor**2, rel=1e-12)

    @pytest.mark.physics
    @given(st.floats(min_value=1e15, max_value=1e19))
    def test_energy_flux_is_linear_in_bulk_density(self, n_0: float) -> None:
        reference = ion_energy_flux(_rp1(n_0=Q_(1e17, "m**-3")))
        scaled = ion_energy_flux(_rp1(n_0=Q_(n_0, "m**-3")))

        assert _m(scaled, "W/m**2") == pytest.approx(
            _m(reference, "W/m**2") * n_0 / 1e17, rel=1e-10
        )

    @pytest.mark.physics
    @pytest.mark.parametrize("factor", [2.0, 4.0, 9.0])
    def test_energy_flux_scales_as_the_square_root_of_electron_temperature(
        self, factor: float
    ) -> None:
        # Doc 01 R-ACC-2 budgets T_e accuracy at 15 % precisely because it "enters
        # Gamma_i as sqrt(T_e) -> 7.5 % in flux". That claim is this exponent.
        base = ion_energy_flux(_rp1(T_e=Q_(3.0, "eV")))
        hotter = ion_energy_flux(_rp1(T_e=Q_(3.0 * factor, "eV")))

        assert _m(hotter, "W/m**2") == pytest.approx(
            _m(base, "W/m**2") * np.sqrt(factor), rel=1e-12
        )

    @pytest.mark.physics
    def test_debye_length_scales_as_the_inverse_square_root_of_density(self) -> None:
        base = debye_length(_rp1(n_0=Q_(1e17, "m**-3")))
        denser = debye_length(_rp1(n_0=Q_(4e17, "m**-3")))

        assert _m(denser, "m") == pytest.approx(_m(base, "m") / 2.0, rel=1e-12)


# ── the individual models ───────────────────────────────────────────────────────


class TestBohmSpeed:
    @pytest.mark.physics
    def test_the_adiabatic_ion_form_is_larger_than_the_cold_ion_form(self) -> None:
        # doc 03 §2.1 carries gamma_i = 3; doc 01 §2.2 drops the ion term entirely. At
        # RP-1 the difference is 2.5 % because T_i / T_e = 1/60, which is why it can sit
        # unnoticed in the specification.
        params = _rp1()
        cold = _m(bohm_speed(params, gamma_i=GAMMA_I_COLD_ION), "m/s")
        adiabatic = _m(bohm_speed(params, gamma_i=GAMMA_I_ADIABATIC), "m/s")

        assert adiabatic > cold
        assert adiabatic / cold == pytest.approx(np.sqrt(1.0 + 3.0 * 0.05 / 3.0), rel=1e-12)

    def test_the_default_is_the_cold_ion_form_doc_01_tabulates(self) -> None:
        params = _rp1()

        assert _m(bohm_speed(params), "m/s") == pytest.approx(
            _m(bohm_speed(params, gamma_i=GAMMA_I_COLD_ION), "m/s"), rel=1e-15
        )

    @pytest.mark.physics
    def test_a_heavier_ion_is_slower_at_the_same_temperature(self) -> None:
        # doc 07 B-13. c_s ~ 1/sqrt(m_i), so Xe+ enters the sheath at 0.55 x the Ar+ speed.
        argon = _m(bohm_speed(_rp1()), "m/s")
        xenon = _m(bohm_speed(_rp1(species=XENON)), "m/s")

        assert xenon / argon == pytest.approx(np.sqrt(39.948 / 131.293), rel=1e-4)

    def test_rejects_a_negative_adiabatic_index(self) -> None:
        with pytest.raises(ValueError, match="gamma_i"):
            bohm_speed(_rp1(), gamma_i=-1.0)


class TestSheathEdgeDensity:
    @pytest.mark.physics
    def test_the_edge_density_is_the_ratio_times_the_bulk_density(self) -> None:
        params = _rp1()

        assert _m(sheath_edge_density(params, 0.61), "m**-3") == pytest.approx(0.61e17, rel=1e-12)

    def test_a_ratio_of_one_leaves_the_density_unchanged(self) -> None:
        params = _rp1()

        assert _m(sheath_edge_density(params, 1.0), "m**-3") == pytest.approx(1e17, rel=1e-12)

    @pytest.mark.parametrize("h_l", [0.0, -0.1, 1.5])
    def test_rejects_a_ratio_outside_the_physical_range(self, h_l: float) -> None:
        # h_l is the edge-to-centre ratio of a monotonically falling presheath profile,
        # so it lies in (0, 1]. A value above 1 would put more plasma at the sheath edge
        # than in the bulk.
        with pytest.raises(ValueError, match="h_l"):
            sheath_edge_density(_rp1(), h_l)


class TestMatrixSheath:
    @pytest.mark.physics
    def test_matrix_thickness_at_rp1(self) -> None:
        # s = lambda_D sqrt(2 V_w / T_e) = 40.72 um * sqrt(166.7) = 0.526 mm.
        assert _m(matrix_sheath_thickness(_rp1()), "mm") == pytest.approx(0.526, rel=1e-2)

    @pytest.mark.physics
    def test_the_potential_is_minus_the_full_drop_at_the_wall(self) -> None:
        params = _rp1()
        grid = SpatialGrid.from_quantity(Q_(np.array([0.0, 0.5]), "mm"))
        thickness = Q_(1.0, "mm")

        phi = magnitude_in(matrix_sheath_potential(params, grid, thickness), "V")

        assert np.asarray(phi)[0] == pytest.approx(-250.0, rel=1e-12)

    @pytest.mark.physics
    def test_the_potential_follows_the_quadratic_of_doc_03(self) -> None:
        # Phi(z) = -V_w (1 - z/s)^2.
        params = _rp1()
        thickness = Q_(1.0, "mm")
        grid = SpatialGrid.from_quantity(Q_(np.array([0.0, 0.25, 0.5, 1.0]), "mm"))

        phi = np.asarray(magnitude_in(matrix_sheath_potential(params, grid, thickness), "V"))

        np.testing.assert_allclose(phi, [-250.0, -140.625, -62.5, 0.0], rtol=1e-12, atol=1e-12)

    @pytest.mark.physics
    def test_the_potential_is_flat_beyond_the_sheath_edge(self) -> None:
        # The quadratic turns back upward for z > s and would invent a positive potential
        # hill in the bulk. The model's domain ends at the sheath edge, so it is clamped.
        params = _rp1()
        grid = SpatialGrid.from_quantity(Q_(np.array([0.0, 1.0, 2.0, 5.0]), "mm"))

        phi = np.asarray(magnitude_in(matrix_sheath_potential(params, grid, Q_(1.0, "mm")), "V"))

        np.testing.assert_allclose(phi[1:], 0.0, atol=1e-12)

    @pytest.mark.physics
    def test_the_validity_ratio_shows_rp1_is_far_outside_the_matrix_regime(self) -> None:
        # doc 03 §2.2: the matrix sheath is "valid only for V << T_e". At RP-1 the ratio
        # is 250/3 = 83, so the model is used as a limiting case and an initialisation,
        # never as an answer. Reported as a number rather than warned about, because a
        # warning on every call at the project's own reference point is noise.
        assert matrix_sheath_validity_ratio(_rp1()) == pytest.approx(250.0 / 3.0, rel=1e-12)
        assert matrix_sheath_validity_ratio(_rp1()) > 1.0

    def test_rejects_a_non_positive_thickness(self) -> None:
        grid = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=3)

        with pytest.raises(ValueError, match="thickness"):
            matrix_sheath_potential(_rp1(), grid, Q_(0.0, "mm"))


class TestChildLangmuirPotential:
    @pytest.mark.physics
    def test_the_potential_follows_the_four_thirds_power(self) -> None:
        # E ~ z'^(1/3) in the collisionless Child-Langmuir solution of doc 03 §2.3, so
        # Phi ~ z'^(4/3) measured from the sheath edge.
        params = _rp1()
        grid = SpatialGrid.from_quantity(Q_(np.array([0.0, 0.5, 1.0]), "mm"))

        phi = np.asarray(magnitude_in(child_langmuir_potential(params, grid, Q_(1.0, "mm")), "V"))

        np.testing.assert_allclose(
            phi, [-250.0, -250.0 * 0.5 ** (4.0 / 3.0), 0.0], rtol=1e-12, atol=1e-12
        )

    @pytest.mark.physics
    def test_the_child_langmuir_potential_falls_faster_near_the_sheath_edge(self) -> None:
        # Both profiles reach -V_w at the wall and 0 at the sheath edge, but they get
        # there differently, and the difference is the physics. Child-Langmuir has
        # n_i ~ 1/u_i ~ |Phi|^(-1/2), so the ion space charge — and with it the curvature
        # of Phi — piles up at the *sheath edge*, where the ions are still slow. The
        # matrix sheath assumes a uniform n_i and has no such concentration. Hence
        # |Phi_CL| >= |Phi_matrix| everywhere inside: xi^(4/3) >= xi^2 on (0, 1).
        params = _rp1()
        grid = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=11)
        thickness = Q_(1.0, "mm")

        cl = np.asarray(magnitude_in(child_langmuir_potential(params, grid, thickness), "V"))
        matrix = np.asarray(magnitude_in(matrix_sheath_potential(params, grid, thickness), "V"))

        assert np.all(cl[1:-1] < matrix[1:-1])


class TestCollisionalSheath:
    @pytest.mark.physics
    @pytest.mark.parametrize("factor", [2.0, 4.0])
    def test_constant_mean_free_path_current_scales_as_bias_to_the_three_halves(
        self, factor: float
    ) -> None:
        thickness = Q_(1.0, "mm")
        mfp = Q_(10.0, "mm")
        base = collisional_current_density(
            _rp1(bias=Q_(-100.0, "V")), thickness, mean_free_path=mfp
        )
        scaled = collisional_current_density(
            _rp1(bias=Q_(-100.0 * factor, "V")), thickness, mean_free_path=mfp
        )

        assert _m(scaled, "A/m**2") == pytest.approx(_m(base, "A/m**2") * factor**1.5, rel=1e-12)

    @pytest.mark.physics
    @pytest.mark.parametrize("factor", [2.0, 4.0])
    def test_constant_mean_free_path_current_scales_as_thickness_to_the_minus_five_halves(
        self, factor: float
    ) -> None:
        params = _rp1()
        mfp = Q_(10.0, "mm")
        base = collisional_current_density(params, Q_(1.0, "mm"), mean_free_path=mfp)
        thicker = collisional_current_density(params, Q_(factor, "mm"), mean_free_path=mfp)

        assert _m(thicker, "A/m**2") == pytest.approx(_m(base, "A/m**2") / factor**2.5, rel=1e-12)

    @pytest.mark.physics
    def test_constant_mean_free_path_current_scales_as_the_square_root_of_the_mfp(self) -> None:
        params = _rp1()
        thickness = Q_(1.0, "mm")
        base = collisional_current_density(params, thickness, mean_free_path=Q_(1.0, "mm"))
        longer = collisional_current_density(params, thickness, mean_free_path=Q_(4.0, "mm"))

        assert _m(longer, "A/m**2") == pytest.approx(_m(base, "A/m**2") * 2.0, rel=1e-12)

    @pytest.mark.physics
    @pytest.mark.parametrize("factor", [2.0, 4.0])
    def test_constant_mobility_current_scales_as_the_square_of_the_bias(
        self, factor: float
    ) -> None:
        thickness = Q_(1.0, "mm")
        mobility = Q_(0.1, "m**2 / V / s")
        base = collisional_current_density(
            _rp1(bias=Q_(-100.0, "V")),
            thickness,
            regime=CollisionalRegime.CONSTANT_MOBILITY,
            mobility=mobility,
        )
        scaled = collisional_current_density(
            _rp1(bias=Q_(-100.0 * factor, "V")),
            thickness,
            regime=CollisionalRegime.CONSTANT_MOBILITY,
            mobility=mobility,
        )

        assert _m(scaled, "A/m**2") == pytest.approx(_m(base, "A/m**2") * factor**2, rel=1e-12)

    @pytest.mark.physics
    def test_constant_mobility_current_scales_as_the_inverse_cube_of_thickness(self) -> None:
        params = _rp1()
        mobility = Q_(0.1, "m**2 / V / s")
        base = collisional_current_density(
            params, Q_(1.0, "mm"), regime=CollisionalRegime.CONSTANT_MOBILITY, mobility=mobility
        )
        thicker = collisional_current_density(
            params, Q_(2.0, "mm"), regime=CollisionalRegime.CONSTANT_MOBILITY, mobility=mobility
        )

        assert _m(thicker, "A/m**2") == pytest.approx(_m(base, "A/m**2") / 8.0, rel=1e-12)

    @pytest.mark.physics
    def test_the_two_models_agree_to_order_unity_where_the_sheath_is_one_mean_free_path(
        self,
    ) -> None:
        # The collisionless and constant-mean-free-path laws are different limits of the
        # same problem and must meet, to within an order-unity prefactor, at s = lambda.
        # A prefactor slip in either would show up here as orders of magnitude.
        params = _rp1()
        mfp = cx_mean_free_path(params, SIGMA_CX_ARGON)

        collisionless = _m(child_langmuir_current_density(params, mfp), "A/m**2")
        collisional = _m(collisional_current_density(params, mfp, mean_free_path=mfp), "A/m**2")

        assert 0.25 < collisional / collisionless < 4.0

    def test_the_default_regime_is_the_constant_mean_free_path_one(self) -> None:
        # doc 03 §2.4: "The framework implements the constant-mean-free-path variant as
        # the default because Ar+/Ar charge exchange has a weakly energy-dependent cross
        # section, making lambda_CX the better-behaved constant."
        params = _rp1()
        mfp = Q_(10.0, "mm")
        implicit = collisional_current_density(params, Q_(1.0, "mm"), mean_free_path=mfp)
        explicit = collisional_current_density(
            params,
            Q_(1.0, "mm"),
            regime=CollisionalRegime.CONSTANT_MEAN_FREE_PATH,
            mean_free_path=mfp,
        )

        assert _m(implicit, "A/m**2") == pytest.approx(_m(explicit, "A/m**2"), rel=1e-15)

    def test_rejects_the_mean_free_path_regime_without_a_mean_free_path(self) -> None:
        with pytest.raises(ValueError, match="mean_free_path"):
            collisional_current_density(_rp1(), Q_(1.0, "mm"))

    def test_rejects_the_mobility_regime_without_a_mobility(self) -> None:
        with pytest.raises(ValueError, match="mobility"):
            collisional_current_density(
                _rp1(), Q_(1.0, "mm"), regime=CollisionalRegime.CONSTANT_MOBILITY
            )

    def test_rejects_a_non_positive_thickness(self) -> None:
        with pytest.raises(ValueError, match="thickness"):
            collisional_current_density(_rp1(), Q_(0.0, "mm"), mean_free_path=Q_(1.0, "mm"))


class TestChargeExchange:
    @pytest.mark.physics
    def test_the_mean_free_path_is_inversely_proportional_to_pressure(self) -> None:
        base = cx_mean_free_path(_rp1(pressure=Q_(5.0, "mTorr")), SIGMA_CX_ARGON)
        higher = cx_mean_free_path(_rp1(pressure=Q_(50.0, "mTorr")), SIGMA_CX_ARGON)

        assert _m(higher, "m") == pytest.approx(_m(base, "m") / 10.0, rel=1e-12)

    @pytest.mark.physics
    def test_the_mean_free_path_is_inversely_proportional_to_the_cross_section(self) -> None:
        params = _rp1()
        base = cx_mean_free_path(params, Q_(5e-19, "m**2"))
        larger = cx_mean_free_path(params, Q_(1e-18, "m**2"))

        assert _m(larger, "m") == pytest.approx(_m(base, "m") / 2.0, rel=1e-12)

    def test_the_published_cross_section_is_the_doc_01_value(self) -> None:
        # doc 01 §2.2 marks sigma_CX at RP-1 as PUBLISHED and notes the implementation
        # will use the energy-dependent LXCat curve instead. Exposing it as a named
        # constant rather than a default argument keeps that substitution visible.
        assert _m(SIGMA_CX_ARGON, "m**2") == pytest.approx(5e-19, rel=1e-15)

    def test_rejects_a_non_positive_cross_section(self) -> None:
        with pytest.raises(ValueError, match="sigma_cx"):
            cx_mean_free_path(_rp1(), Q_(0.0, "m**2"))

    def test_collisionality_accepts_an_explicit_thickness(self) -> None:
        params = _rp1()
        mfp = cx_mean_free_path(params, SIGMA_CX_ARGON)

        assert collisionality(params, SIGMA_CX_ARGON, thickness=mfp) == pytest.approx(
            1.0, rel=1e-12
        )


# ── the assembled state ─────────────────────────────────────────────────────────


class TestAnalyticSheathSolver:
    def test_reports_the_analytic_fidelity_level(self) -> None:
        assert AnalyticSheathSolver().fidelity() is Fidelity.L0

    def test_produces_a_state_carrying_every_required_field(self) -> None:
        state = AnalyticSheathSolver().solve(_rp1())

        assert set(state.field_names) >= {"n_e", "n_i", "Phi", "T_e"}
        assert state.fidelity is Fidelity.L0

    def test_the_state_carries_no_ion_distribution(self) -> None:
        # doc 03 §6: L0 assumes a drifting Maxwellian rather than computing one, and
        # Fidelity.L0.resolves_distribution is False. Recording the absence is more
        # honest than manufacturing a distribution the level never solved for.
        state = AnalyticSheathSolver().solve(_rp1())

        assert state.ion_distribution is None

    def test_the_state_is_steady(self) -> None:
        assert AnalyticSheathSolver().solve(_rp1()).is_steady

    @pytest.mark.physics
    def test_the_potential_reaches_the_full_bias_at_the_wall(self) -> None:
        state = AnalyticSheathSolver().solve(_rp1())

        phi = state.field("Phi").values
        assert phi[0] == pytest.approx(-250.0, rel=1e-12)

    @pytest.mark.physics
    def test_the_potential_rises_monotonically_away_from_the_wall(self) -> None:
        state = AnalyticSheathSolver().solve(_rp1())

        assert np.all(np.diff(state.field("Phi").values) >= 0.0)

    @pytest.mark.physics
    def test_the_plasma_is_quasineutral_at_and_beyond_the_sheath_edge(self) -> None:
        # The sheath model's outer boundary condition. n_e and n_i must agree there or
        # the model has manufactured space charge in the bulk.
        state = AnalyticSheathSolver().solve(_rp1())

        n_e = state.field("n_e").values
        n_i = state.field("n_i").values
        assert n_e[-1] == pytest.approx(n_i[-1], rel=1e-9)

    @pytest.mark.physics
    def test_electrons_are_depleted_at_the_wall(self) -> None:
        # Boltzmann electrons at eV_w/kT_e = 83 are suppressed by exp(-83).
        state = AnalyticSheathSolver().solve(_rp1())

        n_e = state.field("n_e").values
        assert n_e[0] < 1e-20 * n_e[-1]

    @pytest.mark.physics
    def test_the_ion_density_never_exceeds_the_sheath_edge_density(self) -> None:
        # Ions accelerate monotonically toward the wall and flux is conserved, so their
        # density can only fall.
        state = AnalyticSheathSolver().solve(_rp1())

        n_i = state.field("n_i").values
        assert np.all(np.diff(n_i) >= -1e-6 * n_i.max())
        assert n_i.max() == pytest.approx(n_i[-1], rel=1e-12)

    @pytest.mark.physics
    def test_the_ion_flux_is_conserved_across_the_sheath(self) -> None:
        # n_i(z) u_i(z) = const is the construction; verifying it catches a slip in the
        # energy relation that would otherwise show up only as a wrong wall density.
        solver = AnalyticSheathSolver()
        state = solver.solve(_rp1())

        flux = state.field("n_i").values * state.field("u_i").values
        np.testing.assert_allclose(flux, flux[-1], rtol=1e-9)

    @pytest.mark.physics
    def test_the_solver_flux_matches_the_standalone_functional(self) -> None:
        # doc 03 §6 applies "the same functional" at every level. The solver must not
        # have its own opinion about Gamma_E.
        solver = AnalyticSheathSolver()
        state = solver.solve(_rp1())

        assert _m(solver.flux(state), "kW/m**2") == pytest.approx(6.6, rel=1e-2)

    def test_the_electron_temperature_field_is_uniform_and_in_electronvolts(self) -> None:
        state = AnalyticSheathSolver().solve(_rp1())

        T_e = state.field("T_e")
        assert T_e.units == "eV"
        np.testing.assert_allclose(T_e.values, 3.0, rtol=1e-12)

    def test_accepts_an_explicit_grid(self) -> None:
        grid = SpatialGrid.uniform(length=Q_(2.0, "mm"), n_points=17)

        state = AnalyticSheathSolver().solve(_rp1(), grid=grid)

        assert state.grid == grid

    @pytest.mark.physics
    def test_the_default_grid_satisfies_the_doc_01_resolution_requirements(self) -> None:
        # The *derivations* behind doc 01 R-SPAT-1 and R-SPAT-3: ">= 10 samples" across
        # the sheath, field of view ">= 10 x s". Applied to the solver's own thickness,
        # which is the self-consistent 1.14 mm and not doc 01 §2.2's 0.89 mm — see the
        # test below, and the second discrepancy in the module docstring.
        solver = AnalyticSheathSolver()
        params = _rp1()
        state = solver.solve(params)
        thickness = _m(solver.thickness(params), "m")

        assert _m(state.grid.length, "m") >= 10.0 * thickness
        assert _m(state.grid.min_dz, "m") <= thickness / 10.0

    @pytest.mark.physics
    def test_the_default_grid_is_coarser_than_the_absolute_r_spat_1_figure(self) -> None:
        # A consequence of the Debye-length discrepancy, recorded rather than tuned away.
        # doc 01 R-SPAT-1 quotes "<= 90 um", instantiated from s = 0.89 mm. The solver
        # sizes its grid from the self-consistent s = 1.14 mm and lands at 114 um. The
        # *requirement* (>= 10 samples across the sheath) is met; the *number* doc 01
        # tabulates is not, because that number inherits the smaller s. R-SPAT-1 is a
        # requirement on the diagnostic's optical resolution, not on an L0 mesh, so the
        # right response is to record the coupling — not to raise samples_per_sheath
        # until an unrelated number happens to come out.
        state = AnalyticSheathSolver().solve(_rp1())

        assert _m(state.grid.min_dz, "um") == pytest.approx(114.0, rel=1e-2)
        assert _m(state.grid.min_dz, "um") > 90.0

    @pytest.mark.physics
    def test_the_matrix_model_is_selectable(self) -> None:
        params = _rp1()
        matrix = AnalyticSheathSolver(model=SheathModel.MATRIX).solve(params)
        child = AnalyticSheathSolver(model=SheathModel.CHILD_LANGMUIR).solve(params)

        # Same wall potential, different profile.
        assert matrix.field("Phi").values[0] == pytest.approx(-250.0, rel=1e-12)
        assert not np.allclose(matrix.field("Phi").values, child.field("Phi").values)

    @pytest.mark.physics
    def test_it_solves_a_xenon_discharge(self) -> None:
        # doc 07 B-13. Nothing in the L0 layer may be argon-specific.
        argon = AnalyticSheathSolver().solve(_rp1())
        xenon = AnalyticSheathSolver().solve(_rp1(species=XENON))

        assert xenon.params.species is XENON
        ratio = _m(ion_energy_flux(_rp1(species=XENON)), "W/m**2") / _m(
            ion_energy_flux(_rp1()), "W/m**2"
        )
        assert ratio == pytest.approx(np.sqrt(39.948 / 131.293), rel=1e-4)
        assert argon.field("Phi").values[0] == pytest.approx(
            xenon.field("Phi").values[0], rel=1e-12
        )

    def test_the_solver_is_immutable(self) -> None:
        solver = AnalyticSheathSolver()

        with pytest.raises(AttributeError):
            solver.h_l = 0.5  # type: ignore[misc]

    def test_rejects_an_edge_to_centre_ratio_outside_the_physical_range(self) -> None:
        with pytest.raises(ValueError, match="h_l"):
            AnalyticSheathSolver(h_l=1.5)


# ── degenerate inputs ───────────────────────────────────────────────────────────


class TestZeroBias:
    """What the L0 models do at ``V_w = 0``, and whether that is the right thing."""

    @pytest.mark.physics
    def test_the_sheath_thickness_vanishes(self) -> None:
        # The continuous limit of (2 V_w / T_e)^(3/4). Correct for this model: with no
        # potential drop there is no space-charge region to hold one.
        assert _m(child_langmuir_thickness(_rp1(bias=Q_(0.0, "V"))), "m") == pytest.approx(0.0)

    @pytest.mark.physics
    def test_the_energy_flux_vanishes_and_that_is_a_known_underestimate(self) -> None:
        # Gamma_E = Gamma_i e V_w -> 0. The *physical* flux does not: doc 01 §1.3 writes
        # <E_i> = e(Phi_s - Phi_w) + <E_i>_entry - dE_collisions, and L0 keeps only the
        # first term. Ions still arrive at the Bohm speed carrying ~T_e/2 each, so the
        # L0 flux is a lower bound that is exact at high bias and wrong at zero bias.
        # Asserted, not silently accepted, because doc 00 C4 forbids the silent version.
        params = _rp1(bias=Q_(0.0, "V"))

        assert _m(ion_energy_flux(params), "W/m**2") == pytest.approx(0.0)
        assert _m(ion_flux(params), "m**-2 / s") > 0.0

    def test_the_current_density_refuses_a_zero_thickness_rather_than_returning_infinity(
        self,
    ) -> None:
        # J_i ~ 1/s^2. Returning inf would propagate a plausible-looking number into an
        # error budget; raising names the caller that asked for it.
        with pytest.raises(ValueError, match="thickness"):
            child_langmuir_current_density(_rp1(), Q_(0.0, "m"))

    def test_the_solver_refuses_a_zero_bias_rather_than_building_a_zero_width_grid(self) -> None:
        with pytest.raises(ValueError, match="bias"):
            AnalyticSheathSolver().solve(_rp1(bias=Q_(0.0, "V")))

    def test_the_solver_accepts_a_zero_bias_on_a_caller_supplied_grid(self) -> None:
        # Only the *default* grid needs a sheath thickness to size itself. With a grid in
        # hand the model is still well posed: a flat potential and a uniform plasma.
        grid = SpatialGrid.uniform(length=Q_(1.0, "mm"), n_points=5)

        state = AnalyticSheathSolver().solve(_rp1(bias=Q_(0.0, "V")), grid=grid)

        np.testing.assert_allclose(state.field("Phi").values, 0.0, atol=1e-12)


# ── dimensional contract ────────────────────────────────────────────────────────


class TestDimensionalCorrectness:
    """doc 08 §5: every quantity crossing a module boundary carries its units."""

    @pytest.mark.parametrize(
        ("quantity", "units"),
        [
            ("debye_length", "m"),
            ("bohm_speed", "m/s"),
            ("child_langmuir_thickness", "m"),
            ("matrix_sheath_thickness", "m"),
            ("ion_flux", "m**-2 / s"),
            ("ion_energy_flux", "W/m**2"),
        ],
    )
    def test_scalar_results_convert_to_the_units_they_claim(
        self, quantity: str, units: str
    ) -> None:
        functions = {
            "debye_length": debye_length,
            "bohm_speed": bohm_speed,
            "child_langmuir_thickness": child_langmuir_thickness,
            "matrix_sheath_thickness": matrix_sheath_thickness,
            "ion_flux": ion_flux,
            "ion_energy_flux": ion_energy_flux,
        }

        assert np.isfinite(_m(functions[quantity](_rp1()), units))

    def test_a_length_is_not_silently_usable_as_a_time(self) -> None:
        with pytest.raises(DimensionalityError):
            magnitude_in(debye_length(_rp1()), "s")

    def test_the_current_density_is_a_current_per_area(self) -> None:
        value = child_langmuir_current_density(_rp1(), Q_(1.0, "mm"))

        assert np.isfinite(_m(value, "A/m**2"))
        with pytest.raises(DimensionalityError):
            magnitude_in(value, "W/m**2")

    def test_the_mean_free_path_is_a_length(self) -> None:
        assert np.isfinite(_m(cx_mean_free_path(_rp1(), SIGMA_CX_ARGON), "m"))

    def test_collisionality_is_a_bare_ratio(self) -> None:
        # It is compared against 1 everywhere downstream; a Quantity would invite a
        # dimensionless-but-not-1 unit like percent to creep in.
        assert isinstance(collisionality(_rp1(), SIGMA_CX_ARGON), float)

    def test_rejects_a_cross_section_that_is_not_an_area(self) -> None:
        with pytest.raises(DimensionalityError):
            cx_mean_free_path(_rp1(), Q_(5e-19, "m"))

    def test_rejects_a_thickness_that_is_not_a_length(self) -> None:
        with pytest.raises(DimensionalityError):
            child_langmuir_current_density(_rp1(), Q_(1.0, "s"))

    def test_rejects_a_mobility_that_is_not_a_mobility(self) -> None:
        with pytest.raises(DimensionalityError):
            collisional_current_density(
                _rp1(),
                Q_(1.0, "mm"),
                regime=CollisionalRegime.CONSTANT_MOBILITY,
                mobility=Q_(0.1, "m/s"),
            )


class TestTheRegistryDrivesTheDefaults:
    """doc 08 §5: the registry is "the sole source of numeric defaults".

    These tests exist because this module and the registry could otherwise drift apart
    silently — the module would keep working, the registry would keep loading, and a
    sweep over ``sheath.h_l`` would explore a parameter the solver was not actually
    using. That is the parameter-fog trap of doc 00 §6 in its most deniable form.
    """

    def test_the_edge_to_centre_ratio_comes_from_the_registry(self) -> None:
        assert (
            default_registry().value_in("sheath.h_l", "dimensionless")
            == DEFAULT_EDGE_TO_CENTRE_RATIO
        )

    def test_the_default_adiabatic_index_comes_from_the_registry(self) -> None:
        assert default_registry().value_in("sheath.gamma_i", "dimensionless") == GAMMA_I_COLD_ION

    def test_the_charge_exchange_cross_section_comes_from_the_registry(self) -> None:
        assert magnitude_in(SIGMA_CX_ARGON, "m**2") == default_registry().value_in(
            "collisions.sigma_cx_Ar", "m**2"
        )

    def test_the_adiabatic_alternative_is_inside_the_registered_sweep_range(self) -> None:
        # ADR-007 registers gamma_i with nominal 0 and range [0, 3] precisely so that the
        # doc 03 §2.1 value is swept rather than unreachable. If the range ever stopped
        # covering it, the documented alternative would become impossible to explore and
        # the discrepancy would quietly stop being tested.
        sweep = default_registry()["sheath.gamma_i"].sweep_range

        assert sweep is not None
        assert sweep[0] <= GAMMA_I_ADIABATIC <= sweep[1]
