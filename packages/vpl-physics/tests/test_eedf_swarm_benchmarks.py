"""The EEDF solver against published values — doc 11 WBS 1.8's acceptance criterion.

WBS 1.8 requires rate coefficients to **reproduce published values**. Every other test of
this solver is internal: analytic limits it was built to reach, and convergence of its own
discretisation. Those catch a solver that disagrees with itself. They cannot catch one
that is self-consistently wrong, and ADR-009 made that risk real by replacing a
community-standard code with a solver written here.

The reference values live in :mod:`vpl.validation.swarm`, in the package that depends on
no solver, so nothing being judged influences the judgement.

## The pass criterion, stated before the numbers were looked at

Our solver is two-term. The published tables carry both a two-term column, ``MB(2)``, and
a converged ten-term column, ``MB(10)``, and the gap between them is the error of the
two-term *approximation* — physics we have deliberately adopted, not a defect.

So the criterion is:

    the disagreement between this solver and MB(2) must be **smaller than the
    disagreement between MB(2) and MB(10)** at the same reduced field.

That is a principled gate rather than a fitted tolerance. It says the implementation error
must be smaller than the physics approximation it implements — if it is not, the solver is
contributing more error than the model it is solving, and the two-term result cannot be
attributed to the two-term expansion. It also tightens automatically at low field, where
the two-term approximation is nearly exact and there is nowhere for an implementation
error to hide.

An absolute floor of 0.5 % is allowed alongside it, because at 1 Td the two columns agree
to 0.3 % and no finite grid will beat that.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.physics.atomic.lxcat import ProcessType
from vpl.physics.eedf.grid import EnergyGrid
from vpl.physics.eedf.kinetics import ElectronKinetics, InelasticChannel
from vpl.physics.eedf.solver import IonisationSharing, TwoTermSolver
from vpl.validation.swarm import (
    LUCAS_SAELEE,
    REID_RAMP,
    ModelChannel,
    ModelGas,
    SwarmBenchmark,
    benchmarks_for,
)

#: The absolute floor described in the module docstring. Below this, the two published
#: columns agree so closely that the comparison is measuring their grids, not ours.
ABSOLUTE_FLOOR = 0.005


def kinetics_from_model_gas(gas: ModelGas, grid: EnergyGrid) -> ElectronKinetics:
    """Sample a synthetic model gas onto the solver's grid.

    The cross sections are evaluated functionally at the grid points rather than
    interpolated from a table, matching how Flynn et al drive their codes. Interpolating
    here would put an interpolation error inside the thing being verified.

    The elastic cross section of the Lucas-Saelee gas diverges as ``eps -> 0``, so the
    zero-energy boundary is set to its neighbour. Nothing reads it: it multiplies ``eps``
    and ``eps**2`` in the two flux coefficients, and the mobility sum runs over interior
    boundaries only.
    """
    edges = grid.boundaries_ev.copy()
    edges[0] = edges[1]
    return ElectronKinetics(
        grid=grid,
        database=gas.key,
        momentum_transfer_m2=gas.elastic_mtcs_m2(grid.centres_ev),
        momentum_transfer_edge_m2=gas.elastic_mtcs_m2(edges),
        mass_ratio=gas.mass_ratio,
        channels=tuple(
            InelasticChannel(
                reaction=channel.reaction,
                process=(
                    ProcessType.IONIZATION if channel.is_ionisation else ProcessType.EXCITATION
                ),
                threshold_ev=channel.threshold_ev,
                sigma_m2=channel.sigma_m2(grid.centres_ev),
                sigma_edge_m2=channel.sigma_m2(grid.boundaries_ev),
            )
            for channel in gas.channels
        ),
    )


def _relative(computed: float, published: float) -> float:
    return abs(computed / published - 1.0)


def _reid_grid() -> EnergyGrid:
    """Linear to 12 eV. The ramp threshold at 0.2 eV needs the low end resolved."""
    return EnergyGrid.linear(max_ev=12.0, n_cells=3000)


def _lucas_saelee_grid() -> EnergyGrid:
    """Quadratic to 300 eV: the ionisation threshold is 15.6 eV and the tail runs far."""
    return EnergyGrid.quadratic(max_ev=300.0, n_cells=2000)


def _solve(gas: ModelGas, grid: EnergyGrid, field_td: float, **kwargs: object) -> object:
    solver = TwoTermSolver(
        kinetics=kinetics_from_model_gas(gas, grid),
        gas_temperature_ev=gas.gas_temperature_ev,
        **kwargs,  # type: ignore[arg-type]
    )
    return solver.solve(reduced_field_td=field_td)


def _paired(gas: str, field_td: float) -> tuple[SwarmBenchmark, SwarmBenchmark]:
    """The two-term and ten-term published rows at one field, from the same source."""
    two = next(
        b for b in benchmarks_for(gas, terms=2, source="flynn") if b.reduced_field_td == field_td
    )
    ten = next(
        b for b in benchmarks_for(gas, terms=10, source="flynn") if b.reduced_field_td == field_td
    )
    return two, ten


REID_FIELDS = [1.0, 12.0, 24.0]


class TestTheReidRampGas:
    """Conservative model gas — Reid (1979), and Flynn et al (2024) Table 1."""

    @pytest.mark.physics
    @pytest.mark.parametrize("field_td", REID_FIELDS)
    def test_the_mean_energy_matches_the_published_two_term_value(self, field_td: float) -> None:
        two, ten = _paired("reid-ramp", field_td)
        solution = _solve(REID_RAMP, _reid_grid(), field_td)

        ours = _relative(solution.mean_energy_ev, two.mean_energy_ev)  # type: ignore[attr-defined]
        approximation = _relative(two.mean_energy_ev, ten.mean_energy_ev)

        assert ours < max(approximation, ABSOLUTE_FLOOR), (
            f"at {field_td} Td this solver differs from MB(2) by {ours:.2%}, which is not "
            f"smaller than the {approximation:.2%} the two-term approximation itself costs"
        )

    @pytest.mark.physics
    @pytest.mark.parametrize("field_td", REID_FIELDS)
    def test_the_drift_velocity_matches_the_published_two_term_value(self, field_td: float) -> None:
        two, ten = _paired("reid-ramp", field_td)
        solution = _solve(REID_RAMP, _reid_grid(), field_td)

        ours = _relative(solution.drift_velocity_m_per_s, two.drift_velocity_m_per_s)  # type: ignore[attr-defined]
        approximation = _relative(two.drift_velocity_m_per_s, ten.drift_velocity_m_per_s)

        assert ours < max(approximation, ABSOLUTE_FLOOR), (
            f"at {field_td} Td this solver differs from MB(2) by {ours:.2%}, which is not "
            f"smaller than the {approximation:.2%} the two-term approximation itself costs"
        )

    @pytest.mark.physics
    def test_the_whole_of_reids_own_table_is_reproduced(self) -> None:
        # Reid tabulates eight fields from 1 to 40 Td, four more than Flynn. Checking the
        # trend across all of them catches an error that happens to be small at the three
        # fields Flynn reports. Reid quotes his own Boltzmann uncertainty as 0.2%, so 2%
        # here is loose against his numbers and tight against a wrong solver.
        grid = _reid_grid()
        worst = 0.0
        for point in benchmarks_for("reid-ramp", terms=2, source="reid"):
            solution = _solve(REID_RAMP, grid, point.reduced_field_td)
            worst = max(worst, _relative(solution.mean_energy_ev, point.mean_energy_ev))  # type: ignore[attr-defined]

        assert worst < 0.02

    @pytest.mark.physics
    def test_the_transverse_diffusion_matches_the_published_value(self) -> None:
        # N*D_T is the one quantity all three sources report independently, and the one
        # that confirmed the unit reading of Reid's table. It is also the moment most
        # sensitive to the shape of f0 rather than its width.
        two, ten = _paired("reid-ramp", 1.0)
        solution = _solve(REID_RAMP, _reid_grid(), 1.0)

        ours = _relative(solution.reduced_diffusion, two.reduced_transverse_diffusion)  # type: ignore[attr-defined]
        approximation = _relative(
            two.reduced_transverse_diffusion, ten.reduced_transverse_diffusion
        )

        assert ours < max(approximation, ABSOLUTE_FLOOR)

    def test_the_gas_is_conservative_so_no_electrons_are_created(self) -> None:
        solution = _solve(REID_RAMP, _reid_grid(), 24.0)

        assert solution.ionisation_rate_coefficient == 0.0  # type: ignore[attr-defined]


class TestTheLucasSaeleeGas:
    """Non-conservative model gas — the only published check on a *rate coefficient*."""

    @pytest.mark.physics
    def test_the_ionisation_rate_coefficient_matches_the_published_value(self) -> None:
        # The reason this gas is here at all. doc 11 WBS 1.8 asks for rate coefficients
        # against published values; the Reid gas has no ionisation channel to check.
        two, ten = _paired("lucas-saelee", 30.0)
        solution = _solve(LUCAS_SAELEE, _lucas_saelee_grid(), 30.0)

        assert two.ionisation_rate_m3_per_s is not None
        assert ten.ionisation_rate_m3_per_s is not None
        ours = _relative(solution.ionisation_rate_coefficient, two.ionisation_rate_m3_per_s)  # type: ignore[attr-defined]

        assert ours < 0.05, (
            f"k_iz differs from the published two-term value by {ours:.2%}; "
            f"MB(2) and MB(10) differ by "
            f"{_relative(two.ionisation_rate_m3_per_s, ten.ionisation_rate_m3_per_s):.2%}"
        )

    @pytest.mark.physics
    def test_the_mean_energy_matches_the_published_two_term_value(self) -> None:
        two, _ = _paired("lucas-saelee", 30.0)
        solution = _solve(LUCAS_SAELEE, _lucas_saelee_grid(), 30.0)

        assert _relative(solution.mean_energy_ev, two.mean_energy_ev) < 0.05  # type: ignore[attr-defined]

    @pytest.mark.physics
    def test_the_drift_velocity_matches_the_published_two_term_value(self) -> None:
        two, _ = _paired("lucas-saelee", 30.0)
        solution = _solve(LUCAS_SAELEE, _lucas_saelee_grid(), 30.0)

        assert _relative(solution.drift_velocity_m_per_s, two.drift_velocity_m_per_s) < 0.05  # type: ignore[attr-defined]

    @pytest.mark.physics
    def test_the_energy_sharing_rule_is_reported_rather_than_chosen_silently(self) -> None:
        # ADR-009 exposes the sharing rule because the published sources do not state
        # theirs. The two rules bracket any real sharing distribution, so the published
        # value must lie between them — if it does not, the disagreement is not about
        # sharing and blaming it on sharing would be a rationalisation.
        two, _ = _paired("lucas-saelee", 30.0)
        grid = _lucas_saelee_grid()

        rates = {
            rule: _solve(  # type: ignore[attr-defined]
                LUCAS_SAELEE, grid, 30.0, ionisation_sharing=rule
            ).ionisation_rate_coefficient
            for rule in IonisationSharing
        }

        assert two.ionisation_rate_m3_per_s is not None
        assert min(rates.values()) <= two.ionisation_rate_m3_per_s * 1.05
        assert max(rates.values()) >= two.ionisation_rate_m3_per_s * 0.95


class TestTheBenchmarkHarnessItself:
    """A benchmark nobody has seen reject anything is not a benchmark."""

    @pytest.mark.physics
    def test_an_elastic_only_momentum_transfer_fails_the_gate(self) -> None:
        # The regression guard for the defect this benchmark found. Reid eq. (4) makes
        # the momentum-transfer cross section the sum of the elastic one and every
        # inelastic one; using the elastic alone put the drift velocity 52 % high at
        # 24 Td while every internal test stayed green.
        #
        # Reconstructed here by hand rather than by a flag on the solver: a switch that
        # can restore the wrong physics is a switch somebody can set.
        grid = _reid_grid()
        correct = kinetics_from_model_gas(REID_RAMP, grid)
        elastic_only = ElectronKinetics(
            grid=grid,
            database=correct.database,
            momentum_transfer_m2=correct.momentum_transfer_m2,
            momentum_transfer_edge_m2=correct.momentum_transfer_edge_m2,
            mass_ratio=correct.mass_ratio,
            channels=(),
        )
        assert np.any(correct.effective_momentum_transfer_m2 > correct.momentum_transfer_m2), (
            "the ramp channel must actually contribute, or this control proves nothing"
        )

        two, _ = _paired("reid-ramp", 24.0)
        crippled = TwoTermSolver(kinetics=elastic_only, gas_temperature_ev=0.0).solve(
            reduced_field_td=24.0
        )

        assert _relative(crippled.drift_velocity_m_per_s, two.drift_velocity_m_per_s) > 0.05

    @pytest.mark.physics
    def test_a_wrong_ramp_slope_fails_the_gate(self) -> None:
        # A second, independent perturbation. Reid studied k from 1 to 50 A^2/eV and
        # tabulated how much the transport coefficients move with it, so a 20 % error in
        # k is a perturbation the published table itself says should be visible.
        #
        # The mass ratio, by contrast, is *not* a usable control here: at 24 Td the Reid
        # gas loses essentially all its energy inelastically, so doubling m_e/M moves the
        # mean energy by under 0.01 %. That is physics worth knowing rather than a
        # weakness in the benchmark, and it is why this control perturbs the ramp.
        (original,) = REID_RAMP.channels

        def steeper(energy_ev: np.ndarray) -> np.ndarray:
            return 1.2 * original.sigma_m2(energy_ev)

        wrong = ModelGas(
            key=REID_RAMP.key,
            name=REID_RAMP.name,
            citation=REID_RAMP.citation,
            mass_ratio=REID_RAMP.mass_ratio,
            gas_temperature_ev=REID_RAMP.gas_temperature_ev,
            elastic_mtcs_m2=REID_RAMP.elastic_mtcs_m2,
            channels=(
                ModelChannel(
                    reaction=original.reaction,
                    is_ionisation=False,
                    threshold_ev=original.threshold_ev,
                    sigma_m2=steeper,
                ),
            ),
        )
        two, _ = _paired("reid-ramp", 24.0)
        solution = _solve(wrong, _reid_grid(), 24.0)

        assert _relative(solution.mean_energy_ev, two.mean_energy_ev) > 0.05  # type: ignore[attr-defined]

    def test_the_model_gas_is_sampled_onto_the_grid_without_interpolation(self) -> None:
        grid = _reid_grid()

        kinetics = kinetics_from_model_gas(REID_RAMP, grid)

        np.testing.assert_allclose(kinetics.momentum_transfer_m2, 6.0e-20)
        (channel,) = kinetics.channels
        np.testing.assert_allclose(
            channel.sigma_m2, 10.0e-20 * np.clip(grid.centres_ev - 0.2, 0.0, None)
        )
