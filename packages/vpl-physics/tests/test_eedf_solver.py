"""The two-term Boltzmann solve — doc 03 §3.2, doc 11 WBS 1.8.

Doc 11 WBS 1.8's done-when is "rate coefficients reproduce published Ar values". Doc 09
§5 forbids the argon tables from entering the repository, so every fixture here is a
*synthetic* gas — and for the verification that is the stronger position rather than the
weaker one, because for synthetic gases the two-term equation has closed-form solutions
that a tabulated argon set cannot supply:

======================  ======================================================
Limit                   What is checked, exactly
======================  ======================================================
``E/N -> 0``            f0 must relax to a Maxwellian at the **gas** temperature
                        and ``<eps> -> 1.5 kT_g``. Nothing in the discretisation
                        knows this; it falls out of the elastic energy-loss term.
Druyvesteyn             Constant ``sigma_m``, cold gas, no inelastic: the exact
                        solution is ``exp(-(eps/eps_0)**2)`` with
                        ``eps_0 = (E/N)/(sigma_m sqrt(3 m_e/M))``, so shape *and*
                        mean energy both have closed forms.
Constant ``nu``         ``sigma_m ~ eps**(-1/2)`` makes ``mu_e N =
                        gamma/(2 sigma_0 sqrt(eps_ref))`` **independent of the
                        EEDF** — the textbook ``mu = e/(m_e nu)`` (Lieberman &
                        Lichtenberg §5.3). Every EEDF error cancels, so this
                        isolates the transport integral.
Einstein                ``D_e/mu_e = k T_e/e`` for a Maxwellian and any
                        ``sigma_m``.
Growth identity         The dominant eigenvalue of the discrete operator equals
                        the ionisation rate coefficient exactly. This ties the
                        inelastic redistribution to the rate integral: if either
                        is wrong they stop agreeing.
======================  ======================================================

The gases are argon-like where it matters: ``m_e/M`` is argon's and ``sigma_m`` is the
magnitude of argon's momentum-transfer peak, so the fields at which things happen are the
fields at which they happen in argon. The comparison against *tabulated* argon transport
belongs in the benchmark suite, where the cached LXCat data exists; it cannot live in a
unit test without violating doc 09 §5.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy.special import gamma as gamma_function

from vpl.physics.atomic.lxcat import ProcessType
from vpl.physics.eedf.analytic import druyvesteyn_eedf, maxwellian_eedf
from vpl.physics.eedf.grid import EnergyGrid
from vpl.physics.eedf.kinetics import ElectronKinetics, InelasticChannel
from vpl.physics.eedf.solver import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RELATIVE_TOLERANCE,
    GAMMA,
    TOWNSEND_V_M2,
    EedfConvergenceError,
    EedfSolution,
    IonisationSharing,
    TwoTermSolver,
)
from vpl.validation.convergence import RefinementLevel, assert_design_order, observed_order

# ── synthetic gases (doc 09 §5: nothing downloaded) ─────────────────────────────

#: ``m_e / M`` for argon.
ARGON_MASS_RATIO = 1.36e-5

#: The magnitude of argon's momentum-transfer cross section near its peak. Used as a
#: *constant* here, which is what makes the Druyvesteyn form exact; the point of matching
#: argon's magnitude is that the reduced fields at which the closed forms are exercised
#: are then the reduced fields of a real argon discharge.
ARGON_LIKE_SIGMA_M2 = 2e-19

#: 300 K in eV — RP-1's neutral temperature (doc 01 §2.1).
ROOM_TEMPERATURE_EV = 0.025852

ARGON_IONISATION_EV = 15.76
ARGON_FIRST_EXCITATION_EV = 11.5
IONISATION_REACTION = "E + Ar -> E + E + Ar+"


def _constant_kinetics(
    grid: EnergyGrid,
    *,
    sigma_m2: float = ARGON_LIKE_SIGMA_M2,
    mass_ratio: float = ARGON_MASS_RATIO,
    channels: tuple[InelasticChannel, ...] = (),
) -> ElectronKinetics:
    """A hard-sphere gas: ``sigma_m`` constant, so the Druyvesteyn form is exact."""
    return ElectronKinetics(
        grid=grid,
        database="synthetic-constant",
        momentum_transfer_m2=np.full(grid.n_cells, sigma_m2),
        momentum_transfer_edge_m2=np.full(grid.n_cells + 1, sigma_m2),
        mass_ratio=mass_ratio,
        channels=channels,
    )


def _constant_frequency_kinetics(
    grid: EnergyGrid,
    *,
    sigma_ref_m2: float = ARGON_LIKE_SIGMA_M2,
    energy_ref_ev: float = 1.0,
) -> ElectronKinetics:
    """``sigma_m = sigma_0 sqrt(eps_ref/eps)``: a collision frequency independent of energy.

    The value at ``eps = 0`` would diverge and is set to its neighbour. Nothing reads it:
    it multiplies ``eps`` and ``eps**2`` in the two flux coefficients, and the mobility
    sum runs over interior boundaries only.
    """
    edge = grid.boundaries_ev.copy()
    edge[0] = edge[1]
    return ElectronKinetics(
        grid=grid,
        database="synthetic-constant-frequency",
        momentum_transfer_m2=sigma_ref_m2 * np.sqrt(energy_ref_ev / grid.centres_ev),
        momentum_transfer_edge_m2=sigma_ref_m2 * np.sqrt(energy_ref_ev / edge),
        mass_ratio=ARGON_MASS_RATIO,
        channels=(),
    )


def _channel(
    grid: EnergyGrid,
    *,
    threshold_ev: float,
    sigma_m2: float,
    process: ProcessType = ProcessType.EXCITATION,
    reaction: str = "E + Ar -> E + Ar*",
) -> InelasticChannel:
    """A step cross section: zero below threshold, constant above it."""
    return InelasticChannel(
        reaction=reaction,
        process=process,
        threshold_ev=threshold_ev,
        sigma_m2=np.where(grid.centres_ev >= threshold_ev, sigma_m2, 0.0),
    )


def _ionisation(grid: EnergyGrid, *, sigma_m2: float = 3e-21) -> InelasticChannel:
    return _channel(
        grid,
        threshold_ev=ARGON_IONISATION_EV,
        sigma_m2=sigma_m2,
        process=ProcessType.IONIZATION,
        reaction=IONISATION_REACTION,
    )


def _druyvesteyn_scale_ev(
    *, reduced_field_td: float, sigma_m2: float = ARGON_LIKE_SIGMA_M2
) -> float:
    """``eps_0 = (E/N)/(sigma_m sqrt(3 m_e/M))``, from integrating ``d ln f0/d eps = W/D``."""
    return (reduced_field_td * TOWNSEND_V_M2) / (sigma_m2 * np.sqrt(3.0 * ARGON_MASS_RATIO))


def _druyvesteyn_mean_energy_ev(reduced_field_td: float) -> float:
    """``<eps> = eps_0 Gamma(5/4) / Gamma(3/4)`` for ``exp(-(eps/eps_0)**2)``."""
    return _druyvesteyn_scale_ev(reduced_field_td=reduced_field_td) * float(
        gamma_function(1.25) / gamma_function(0.75)
    )


# ── the Maxwellian limit ────────────────────────────────────────────────────────


def _relaxed_solution() -> EedfSolution:
    """Elastic only, vanishing field: the electrons must equilibrate with the gas."""
    grid = EnergyGrid.linear(max_ev=0.6, n_cells=600)
    solver = TwoTermSolver(
        kinetics=_constant_kinetics(grid), gas_temperature_ev=ROOM_TEMPERATURE_EV
    )
    return solver.solve(reduced_field_td=1e-4)


class TestTheMaxwellianLimit:
    """The strongest self-check available without external data.

    Nothing in the discretisation is told about the Maxwellian: it appears only as the
    balance between the elastic drag ``W`` and the thermal part of ``D``.
    """

    @pytest.mark.physics
    def test_the_mean_energy_approaches_three_halves_kT_g(self) -> None:
        assert _relaxed_solution().mean_energy_ev == pytest.approx(
            1.5 * ROOM_TEMPERATURE_EV, rel=1e-4
        )

    @pytest.mark.physics
    def test_the_shape_approaches_a_maxwellian_at_the_gas_temperature(self) -> None:
        solution = _relaxed_solution()
        expected = solution.grid.normalise(
            maxwellian_eedf(solution.grid.centres_ev, electron_temperature_ev=ROOM_TEMPERATURE_EV)
        )

        assert float(np.max(np.abs(solution.f0 - expected)) / expected.max()) < 1e-4

    @pytest.mark.physics
    def test_the_eedf_is_normalised_to_solver_tolerance(self) -> None:
        solution = _relaxed_solution()

        assert solution.grid.moment(solution.f0, 0) == pytest.approx(1.0, rel=1e-14)

    @pytest.mark.physics
    def test_the_eedf_is_everywhere_non_negative(self) -> None:
        assert np.all(_relaxed_solution().f0 >= 0.0)


class TestTheEinsteinRelation:
    """``D_e / mu_e = k T_e / e``. Exact for a Maxwellian and *any* ``sigma_m``."""

    @pytest.mark.physics
    def test_the_characteristic_energy_equals_the_gas_temperature(self) -> None:
        assert _relaxed_solution().characteristic_energy_ev == pytest.approx(
            ROOM_TEMPERATURE_EV, rel=1e-3
        )

    @pytest.mark.physics
    def test_it_holds_for_an_energy_dependent_momentum_transfer_too(self) -> None:
        # The relation is a property of the Maxwellian, not of the cross section, so a
        # solver that only satisfied it for a constant sigma_m would be passing by luck.
        grid = EnergyGrid.linear(max_ev=0.6, n_cells=600)
        solver = TwoTermSolver(
            kinetics=_constant_frequency_kinetics(grid),
            gas_temperature_ev=ROOM_TEMPERATURE_EV,
        )

        solution = solver.solve(reduced_field_td=1e-4)

        assert solution.characteristic_energy_ev == pytest.approx(ROOM_TEMPERATURE_EV, rel=1e-3)

    @pytest.mark.physics
    def test_the_characteristic_energy_is_two_thirds_of_the_mean_energy(self) -> None:
        # True *because* the relaxed distribution is Maxwellian. Where it is not — every
        # field of interest to doc 03 §3.2 — the two differ, and the difference is a
        # measurement of the non-Maxwellian correction rather than an assumption about it.
        solution = _relaxed_solution()

        assert solution.characteristic_energy_ev == pytest.approx(
            solution.effective_temperature_ev, rel=1e-3
        )

    @pytest.mark.physics
    def test_the_two_part_company_once_the_field_is_on(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=600)
        solution = TwoTermSolver(kinetics=_constant_kinetics(grid), gas_temperature_ev=0.0).solve(
            reduced_field_td=30.0
        )

        # A Druyvesteyn has D/mu = <eps> Gamma(3/4)^2 / Gamma(5/4)^2 * ... — the point is
        # only that it is *not* (2/3)<eps>, which is what a Maxwellian analysis assumes.
        assert solution.characteristic_energy_ev != pytest.approx(
            solution.effective_temperature_ev, rel=0.05
        )


# ── the Druyvesteyn limit ───────────────────────────────────────────────────────


def _druyvesteyn_solution(reduced_field_td: float = 30.0, n_cells: int = 1000) -> EedfSolution:
    grid = EnergyGrid.linear(max_ev=200.0, n_cells=n_cells)
    solver = TwoTermSolver(kinetics=_constant_kinetics(grid), gas_temperature_ev=0.0)
    return solver.solve(reduced_field_td=reduced_field_td)


class TestTheDruyvesteynLimit:
    """Constant ``sigma_m``, cold gas, elastic only: shape and mean energy are closed form."""

    @pytest.mark.physics
    def test_the_mean_energy_matches_the_closed_form(self) -> None:
        solution = _druyvesteyn_solution()

        assert solution.mean_energy_ev == pytest.approx(_druyvesteyn_mean_energy_ev(30.0), rel=1e-4)

    @pytest.mark.physics
    def test_the_shape_is_druyvesteyn(self) -> None:
        solution = _druyvesteyn_solution()
        expected = solution.grid.normalise(
            druyvesteyn_eedf(solution.grid.centres_ev, mean_energy_ev=solution.mean_energy_ev)
        )

        assert float(np.max(np.abs(solution.f0 - expected)) / expected.max()) < 1e-4

    @pytest.mark.physics
    def test_the_shape_is_emphatically_not_maxwellian(self) -> None:
        # doc 03 §3.2's whole premise. Compared at equal mean energy, which is the
        # like-for-like comparison because that is what a Maxwellian fit would report.
        solution = _druyvesteyn_solution()
        maxwell = solution.grid.normalise(
            maxwellian_eedf(
                solution.grid.centres_ev,
                electron_temperature_ev=solution.effective_temperature_ev,
            )
        )

        assert float(np.max(np.abs(solution.f0 - maxwell)) / maxwell.max()) > 0.1

    @pytest.mark.physics
    def test_the_scale_energy_is_proportional_to_the_reduced_field(self) -> None:
        # eps_0 ~ (E/N) exactly, so doubling the field must double the mean energy.
        low = _druyvesteyn_solution(reduced_field_td=20.0)
        high = _druyvesteyn_solution(reduced_field_td=40.0)

        assert high.mean_energy_ev / low.mean_energy_ev == pytest.approx(2.0, rel=1e-3)


# ── transport integrals ─────────────────────────────────────────────────────────


class TestReducedMobility:
    """``mu_e N`` against the textbook ``mu = e/(m_e nu)`` at constant collision frequency.

    Lieberman & Lichtenberg, *Principles of Plasma Discharges and Materials Processing*
    §5.3, give ``mu = e/(m_e nu_m)``. With ``sigma_m = sigma_0 sqrt(eps_ref/eps)`` the
    collision frequency ``nu_m = N sigma_0 gamma sqrt(eps_ref)`` is a constant, so

        mu_e N = e/(m_e sigma_0 gamma sqrt(eps_ref)) = gamma/(2 sigma_0 sqrt(eps_ref))

    using ``gamma**2 = 2e/m_e``. The two-term integral then reduces to the normalisation of
    ``f0``, so the result is **independent of the EEDF** and the check isolates the
    transport integral from every other part of the solve.
    """

    @staticmethod
    def _expected(*, sigma_ref_m2: float, energy_ref_ev: float = 1.0) -> float:
        return GAMMA / (2.0 * sigma_ref_m2 * np.sqrt(energy_ref_ev))

    @pytest.mark.physics
    @pytest.mark.parametrize("reduced_field_td", [0.5, 1.5, 3.0])
    def test_the_reduced_mobility_matches_the_closed_form(self, reduced_field_td: float) -> None:
        grid = EnergyGrid.linear(max_ev=120.0, n_cells=600)
        solver = TwoTermSolver(
            kinetics=_constant_frequency_kinetics(grid),
            gas_temperature_ev=ROOM_TEMPERATURE_EV,
        )

        solution = solver.solve(reduced_field_td=reduced_field_td)

        assert solution.reduced_mobility == pytest.approx(
            self._expected(sigma_ref_m2=ARGON_LIKE_SIGMA_M2), rel=1e-12
        )

    @pytest.mark.physics
    def test_the_reduced_mobility_scales_inversely_with_the_cross_section(self) -> None:
        grid = EnergyGrid.linear(max_ev=120.0, n_cells=600)
        solver = TwoTermSolver(
            kinetics=_constant_frequency_kinetics(grid, sigma_ref_m2=4e-19),
            gas_temperature_ev=ROOM_TEMPERATURE_EV,
        )

        solution = solver.solve(reduced_field_td=1.5)

        assert solution.reduced_mobility == pytest.approx(
            self._expected(sigma_ref_m2=4e-19), rel=1e-12
        )

    def test_the_reduced_mobility_is_positive(self) -> None:
        # Sign convention: the magnitude is returned. Electrons drift against E, and a
        # signed mobility would have every downstream expression carrying a minus sign
        # that half the call sites would forget.
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=400)
        solver = TwoTermSolver(kinetics=_constant_kinetics(grid))

        assert solver.solve(reduced_field_td=10.0).reduced_mobility > 0.0

    @pytest.mark.physics
    def test_the_drift_velocity_is_mobility_times_field(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=400)
        solution = TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(reduced_field_td=10.0)

        assert solution.drift_velocity_m_per_s == pytest.approx(
            solution.reduced_mobility * 10.0 * TOWNSEND_V_M2
        )


# ── rate coefficients ───────────────────────────────────────────────────────────


def _ionising_solver(grid: EnergyGrid, *, sigma_m2: float = 3e-21) -> TwoTermSolver:
    return TwoTermSolver(
        kinetics=_constant_kinetics(grid, channels=(_ionisation(grid, sigma_m2=sigma_m2),)),
        gas_temperature_ev=ROOM_TEMPERATURE_EV,
    )


class TestRateCoefficients:
    @pytest.mark.physics
    def test_the_ionisation_rate_coefficient_increases_with_the_reduced_field(self) -> None:
        grid = EnergyGrid.linear(max_ev=400.0, n_cells=800)
        solver = _ionising_solver(grid)

        rates = [
            solver.solve(reduced_field_td=field).ionisation_rate_coefficient
            for field in (5.0, 10.0, 20.0, 40.0, 80.0)
        ]

        assert all(later > earlier for earlier, later in pairwise(rates))

    @pytest.mark.physics
    def test_a_channel_whose_threshold_is_off_the_grid_has_exactly_zero_rate(self) -> None:
        grid = EnergyGrid.linear(max_ev=10.0, n_cells=200)
        solver = _ionising_solver(grid)

        assert solver.solve(reduced_field_td=2.0).ionisation_rate_coefficient == 0.0

    @pytest.mark.physics
    def test_the_growth_rate_equals_the_ionisation_rate_coefficient(self) -> None:
        # An exact discrete identity: summing the operator's rows leaves only the net
        # particle creation, which is the ionisation rate integral. If the inelastic
        # redistribution and the rate integral disagree, this is where it shows.
        grid = EnergyGrid.linear(max_ev=400.0, n_cells=800)

        solution = _ionising_solver(grid).solve(reduced_field_td=40.0)

        assert solution.net_ionisation_frequency == pytest.approx(
            solution.ionisation_rate_coefficient, rel=1e-9
        )

    @pytest.mark.physics
    def test_the_growth_model_omission_is_reported_and_small(self) -> None:
        # The solver drops Hagelaar & Pitchford's growth correction to sigma_m. The
        # omission is bounded rather than assumed small: this ratio is what bounds it.
        grid = EnergyGrid.linear(max_ev=400.0, n_cells=800)

        assert _ionising_solver(grid).solve(reduced_field_td=40.0).growth_correction < 1e-3

    @pytest.mark.physics
    def test_excitation_cools_the_electrons(self) -> None:
        # An inelastic channel is an energy sink; adding one at fixed E/N must lower
        # <eps>. This is doc 03 §3.2's depleted tail, measured.
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=600)
        bare = TwoTermSolver(kinetics=_constant_kinetics(grid))
        cooled = TwoTermSolver(
            kinetics=_constant_kinetics(
                grid,
                channels=(_channel(grid, threshold_ev=ARGON_FIRST_EXCITATION_EV, sigma_m2=1e-20),),
            )
        )

        assert (
            cooled.solve(reduced_field_td=20.0).mean_energy_ev
            < bare.solve(reduced_field_td=20.0).mean_energy_ev
        )

    def test_rate_coefficients_are_keyed_by_reaction(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=300)
        solver = TwoTermSolver(
            kinetics=_constant_kinetics(
                grid,
                channels=(_channel(grid, threshold_ev=ARGON_FIRST_EXCITATION_EV, sigma_m2=1e-20),),
            )
        )

        rates = solver.solve(reduced_field_td=20.0).rate_coefficients

        assert set(rates) == {"E + Ar -> E + Ar*"}

    def test_a_gas_with_no_ionisation_channel_reports_zero(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=300)
        solver = TwoTermSolver(kinetics=_constant_kinetics(grid))

        assert solver.solve(reduced_field_td=20.0).ionisation_rate_coefficient == 0.0

    @pytest.mark.physics
    def test_excitation_rates_are_reported_per_channel(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=600)
        solver = TwoTermSolver(
            kinetics=_constant_kinetics(
                grid,
                channels=(
                    _channel(grid, threshold_ev=11.5, sigma_m2=1e-20, reaction="low"),
                    _channel(grid, threshold_ev=13.0, sigma_m2=1e-20, reaction="high"),
                ),
            )
        )

        rates = solver.solve(reduced_field_td=20.0).excitation_rate_coefficients

        assert set(rates) == {"low", "high"}
        # The higher threshold samples a more depleted part of the tail, and on a
        # Druyvesteyn tail 1.5 eV of threshold is three orders of magnitude.
        assert rates["high"] < rates["low"]

    def test_the_ionisation_channel_is_excluded_from_the_excitation_rates(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=300)
        solver = TwoTermSolver(
            kinetics=_constant_kinetics(
                grid,
                channels=(
                    _channel(grid, threshold_ev=11.5, sigma_m2=1e-20),
                    _ionisation(grid),
                ),
            )
        )

        solution = solver.solve(reduced_field_td=20.0)

        assert IONISATION_REACTION not in solution.excitation_rate_coefficients
        assert solution.ionisation_rate_coefficient > 0.0


class TestIonisationSharing:
    """Where the two electrons appear is a real modelling choice, and it is measurable."""

    @staticmethod
    def _solve(sharing: IonisationSharing) -> EedfSolution:
        grid = EnergyGrid.linear(max_ev=300.0, n_cells=600)
        return TwoTermSolver(
            kinetics=_constant_kinetics(grid, channels=(_ionisation(grid, sigma_m2=3e-20),)),
            ionisation_sharing=sharing,
        ).solve(reduced_field_td=60.0)

    @pytest.mark.physics
    @pytest.mark.parametrize("sharing", list(IonisationSharing))
    def test_every_sharing_rule_conserves_the_normalisation(
        self, sharing: IonisationSharing
    ) -> None:
        solution = self._solve(sharing)

        assert solution.grid.moment(solution.f0, 0) == pytest.approx(1.0, rel=1e-14)

    @pytest.mark.physics
    def test_equal_sharing_puts_more_electrons_in_the_lowest_cell(self) -> None:
        # Both electrons leave with half the excess energy, and because the ionising
        # population sits just above threshold that half is close to zero — so equal
        # sharing feeds the bottom of the grid *two* slow electrons where one-takes-all
        # feeds it one slow and one fast. The two rules therefore bracket the low-energy
        # population, which is what makes the choice worth exposing rather than fixing.
        one = self._solve(IonisationSharing.ONE_TAKES_ALL)
        equal = self._solve(IonisationSharing.EQUAL_SHARING)

        assert equal.f0[0] > one.f0[0]
        assert equal.mean_energy_ev != pytest.approx(one.mean_energy_ev, rel=1e-4)


# ── the distribution itself, not only its moments ───────────────────────────────


class TestTheDistributionIsAvailable:
    """doc 04 §4.2: the Thomson spectrum is computed "from the actual EEDF"."""

    @pytest.mark.physics
    def test_the_energy_distribution_integrates_to_one_over_energy(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=600)
        solution = TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(reduced_field_td=10.0)

        # F(eps) = f0 sqrt(eps) is the energy distribution proper; integral F deps = 1.
        assert float(np.sum(solution.energy_distribution * grid.widths_ev)) == pytest.approx(
            1.0, rel=1e-3
        )

    @pytest.mark.physics
    def test_the_tail_is_depleted_relative_to_a_maxwellian_of_the_same_mean_energy(
        self,
    ) -> None:
        # doc 03 §3.2: "the high-energy tail — the part that matters for ionisation and
        # for OES line ratios — is depleted." Quantified here rather than asserted.
        grid = EnergyGrid.linear(max_ev=60.0, n_cells=1200)
        solution = TwoTermSolver(kinetics=_constant_kinetics(grid), gas_temperature_ev=0.0).solve(
            reduced_field_td=5.0
        )

        maxwell = grid.normalise(
            maxwellian_eedf(
                grid.centres_ev, electron_temperature_ev=solution.effective_temperature_ev
            )
        )
        above = grid.centres_ev >= ARGON_IONISATION_EV
        maxwell_tail = float(np.dot(maxwell[above], grid.cell_masses[above]))

        # Not a marginal effect: four orders of magnitude at the argon ionisation
        # threshold, for a distribution a Maxwellian analysis would call 1.9 eV.
        assert solution.tail_fraction_above(ARGON_IONISATION_EV) < 1e-3 * maxwell_tail

    def test_a_tail_fraction_of_zero_energy_is_the_whole_population(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=200)
        solution = TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(reduced_field_td=10.0)

        assert solution.tail_fraction_above(0.0) == pytest.approx(1.0, rel=1e-12)

    def test_the_solution_names_its_database_and_field(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=200)
        solution = TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(reduced_field_td=10.0)

        assert solution.database == "synthetic-constant"
        assert solution.reduced_field_td == pytest.approx(10.0)
        assert "10" in repr(solution)


# ── convergence and refusal ─────────────────────────────────────────────────────


class TestConvergence:
    @pytest.mark.physics
    def test_the_mean_energy_converges_at_second_order_in_the_cell_width(self) -> None:
        # The exponential scheme evaluates its coefficients at cell boundaries, which on a
        # uniform grid are the exact midpoints between adjacent cell centres, so the
        # discretisation is second order. Measured against the Druyvesteyn closed form,
        # not against a fine-grid solution, so the study cannot converge to its own error.
        exact = _druyvesteyn_mean_energy_ev(30.0)

        levels = [
            RefinementLevel(
                h=200.0 / n_cells,
                error=abs(_druyvesteyn_solution(n_cells=n_cells).mean_energy_ev - exact) / exact,
            )
            for n_cells in (100, 200, 400, 800)
        ]

        assert_design_order(observed_order(levels), design_order=2.0)

    @pytest.mark.physics
    def test_a_graded_grid_also_converges_at_second_order(self) -> None:
        # The expectation is that giving up the exact-midpoint property of a uniform grid
        # costs an order. It does not, and the docstring of vpl.physics.eedf.grid says so
        # only because this measures it.
        exact = _druyvesteyn_mean_energy_ev(30.0)

        levels = []
        for n_cells in (100, 200, 400, 800):
            grid = EnergyGrid.quadratic(max_ev=200.0, n_cells=n_cells)
            mean_energy = (
                TwoTermSolver(kinetics=_constant_kinetics(grid), gas_temperature_ev=0.0)
                .solve(reduced_field_td=30.0)
                .mean_energy_ev
            )
            levels.append(
                RefinementLevel(h=200.0 / n_cells, error=abs(mean_energy - exact) / exact)
            )

        assert_design_order(observed_order(levels), design_order=2.0)

    def test_a_solve_that_will_not_converge_raises(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=400)
        solver = TwoTermSolver(kinetics=_constant_kinetics(grid), max_iterations=1)

        with pytest.raises(EedfConvergenceError, match="did not converge"):
            solver.solve(reduced_field_td=20.0)

    def test_the_iteration_count_and_residual_are_reported(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=200)

        solution = TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(reduced_field_td=20.0)

        assert 0 < solution.iterations <= DEFAULT_MAX_ITERATIONS
        assert solution.residual < DEFAULT_RELATIVE_TOLERANCE

    def test_a_warm_start_reaches_the_same_answer_in_fewer_iterations(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=400)
        solver = TwoTermSolver(kinetics=_constant_kinetics(grid))

        cold = solver.solve(reduced_field_td=20.0)
        warm = solver.solve(reduced_field_td=20.0, initial=cold.f0)

        np.testing.assert_allclose(warm.f0, cold.f0, rtol=1e-9)
        assert warm.iterations < cold.iterations


class TestRejections:
    @pytest.mark.parametrize("reduced_field_td", [0.0, -1.0])
    def test_a_non_positive_reduced_field_is_refused(self, reduced_field_td: float) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=100)

        with pytest.raises(ValueError, match="reduced field"):
            TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(
                reduced_field_td=reduced_field_td
            )

    def test_a_negative_gas_temperature_is_refused(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=100)

        with pytest.raises(ValueError, match="gas temperature"):
            TwoTermSolver(kinetics=_constant_kinetics(grid), gas_temperature_ev=-1.0)

    def test_a_non_positive_tolerance_is_refused(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=100)

        with pytest.raises(ValueError, match="tolerance"):
            TwoTermSolver(kinetics=_constant_kinetics(grid), relative_tolerance=0.0)

    def test_a_non_positive_iteration_budget_is_refused(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=100)

        with pytest.raises(ValueError, match="iteration"):
            TwoTermSolver(kinetics=_constant_kinetics(grid), max_iterations=0)

    def test_an_initial_guess_of_the_wrong_length_is_refused(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=100)

        with pytest.raises(ValueError, match="cell"):
            TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(
                reduced_field_td=20.0, initial=np.ones(7)
            )

    def test_a_tail_fraction_below_zero_energy_is_refused(self) -> None:
        grid = EnergyGrid.linear(max_ev=100.0, n_cells=100)
        solution = TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(reduced_field_td=20.0)

        with pytest.raises(ValueError, match="negative"):
            solution.tail_fraction_above(-1.0)


class TestTheGridTruncationCheck:
    """The upper boundary condition is zero flux at ``eps_max``. It has to be earned."""

    @pytest.mark.physics
    def test_a_grid_too_short_for_the_field_is_reported(self) -> None:
        # Silent truncation is the classic way a Boltzmann solve produces a plausible and
        # entirely wrong answer: the result is still smooth, normalised and positive.
        grid = EnergyGrid.linear(max_ev=2.0, n_cells=200)
        solver = TwoTermSolver(kinetics=_constant_kinetics(grid), gas_temperature_ev=0.0)

        assert not solver.solve(reduced_field_td=10.0).is_grid_long_enough

    @pytest.mark.physics
    def test_an_adequate_grid_reports_that_it_is_adequate(self) -> None:
        grid = EnergyGrid.linear(max_ev=200.0, n_cells=600)
        solver = TwoTermSolver(kinetics=_constant_kinetics(grid), gas_temperature_ev=0.0)

        assert solver.solve(reduced_field_td=10.0).is_grid_long_enough


class TestDegenerateStartingPoints:
    @pytest.mark.physics
    def test_a_grid_far_hotter_than_the_default_guess_still_starts(self) -> None:
        # The default initial guess is a 1 eV Maxwellian. On a grid whose first cell
        # centre is tens of keV that underflows to zero everywhere, and normalising it
        # would raise. The fallback is a uniform distribution — a poor guess, but a
        # non-degenerate one, and the converged answer does not depend on it.
        grid = EnergyGrid.linear(max_ev=1e5, n_cells=1)

        solution = TwoTermSolver(kinetics=_constant_kinetics(grid)).solve(reduced_field_td=10.0)

        assert solution.mean_energy_ev > 0.0

    def test_the_growth_correction_of_a_collisionless_solution_is_zero(self) -> None:
        # Unreachable through the solver, because a zero momentum-transfer cross section
        # is refused at assembly. Constructed directly so that the guard is exercised
        # rather than left as an unreached branch that might be wrong.
        grid = EnergyGrid.linear(max_ev=10.0, n_cells=4)
        solution = EedfSolution(
            grid=grid,
            f0=grid.normalise(np.ones(grid.n_cells)),
            reduced_field_td=1.0,
            database="synthetic",
            mean_energy_ev=1.0,
            reduced_mobility=1.0,
            reduced_diffusion=1.0,
            rate_coefficients={},
            net_ionisation_frequency=0.0,
            momentum_transfer_frequency=0.0,
            iterations=1,
            residual=0.0,
        )

        assert solution.growth_correction == 0.0
