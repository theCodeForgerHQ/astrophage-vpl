"""Published swarm benchmarks — the external half of doc 11 WBS 1.8.

WBS 1.8's acceptance criterion is that rate coefficients *reproduce published values*.
Everything the EEDF solver is currently checked against is internal: analytic limits it
was built to reach, and convergence of its own discretisation. Those catch a solver that
disagrees with itself. They cannot catch one that is self-consistently wrong, and
ADR-009 makes that risk concrete by replacing a community-standard code with a solver
written here.

This module holds the outside opinion. It is deliberately in the package that depends on
no solver, for the reason ``convergence.py`` states: nothing being judged can influence
the judgement.

The two model gases are the standard ones for this purpose. They are *synthetic* — chosen
so that every code solves an identical problem with no atomic-data uncertainty in the way,
which is what makes a disagreement attributable to the solver rather than to the database.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.validation.swarm import (
    LUCAS_SAELEE,
    REID_RAMP,
    ModelGas,
    SwarmBenchmark,
    benchmarks_for,
    published_swarm_benchmarks,
)


class TestTheReidRampModelGas:
    """Reid (1979) §4a(ii), the "ramp" model with ``k = 10 A^2 eV^-1``."""

    def test_the_elastic_cross_section_is_energy_independent(self) -> None:
        energy = np.array([0.01, 0.2, 1.0, 5.0, 20.0])

        sigma = REID_RAMP.elastic_mtcs_m2(energy)

        np.testing.assert_allclose(sigma, 6.0e-20)

    def test_the_inelastic_cross_section_is_zero_below_threshold(self) -> None:
        # Reid eq. (23a). Exactly zero, not merely small: the threshold is the whole
        # point of the model, and a solver that leaks excitation below it would still
        # pass a tolerance-based check.
        (channel,) = REID_RAMP.channels

        assert np.all(channel.sigma_m2(np.array([0.0, 0.1, 0.1999])) == 0.0)

    def test_the_inelastic_cross_section_ramps_linearly_above_threshold(self) -> None:
        # Reid eq. (23b): sigma_i = k (eps - eps_i), k = 10 A^2 / eV, eps_i = 0.2 eV.
        (channel,) = REID_RAMP.channels

        sigma = channel.sigma_m2(np.array([0.2, 1.2, 2.2]))

        np.testing.assert_allclose(sigma, [0.0, 10.0e-20, 20.0e-20])

    def test_the_single_channel_is_an_excitation_not_an_ionisation(self) -> None:
        # Reid's ramp gas is conservative — electron number is fixed. A solver told
        # otherwise would grow the population and report a spurious growth rate.
        (channel,) = REID_RAMP.channels

        assert channel.is_ionisation is False

    def test_the_mass_ratio_comes_from_four_amu_and_codata(self) -> None:
        # Reid states the molecular weight as 4.0 a.m.u. rather than a mass ratio, so
        # the ratio is derived from it through the same CODATA the solver uses. A
        # hand-computed constant here would drift from the physics it is compared with.
        assert REID_RAMP.mass_ratio == pytest.approx(1.3714e-4, rel=1e-3)

    def test_the_gas_is_at_absolute_zero(self) -> None:
        # Reid: "The gas temperature was taken to be zero, thus avoiding the need" to
        # model superelastic collisions. Not room temperature.
        assert REID_RAMP.gas_temperature_ev == 0.0


class TestTheLucasSaeleeModelGas:
    """Lucas & Saelee (1975), as restated by Flynn et al (2024) eq. (28)."""

    def test_the_elastic_cross_section_falls_as_one_over_root_energy(self) -> None:
        sigma = LUCAS_SAELEE.elastic_mtcs_m2(np.array([1.0, 4.0, 16.0]))

        np.testing.assert_allclose(sigma, [4.0e-20, 2.0e-20, 1.0e-20])

    def test_the_ionisation_cross_section_ramps_from_its_threshold(self) -> None:
        (channel,) = LUCAS_SAELEE.channels

        sigma = channel.sigma_m2(np.array([15.6, 25.6, 35.6]))

        np.testing.assert_allclose(sigma, [0.0, 1.0e-20, 2.0e-20])

    def test_the_single_channel_is_an_ionisation(self) -> None:
        # This is what makes LS the harder benchmark: the swarm is non-conservative, so
        # it tests the ionisation term the Reid gas cannot reach at all.
        (channel,) = LUCAS_SAELEE.channels

        assert channel.is_ionisation is True
        assert channel.threshold_ev == 15.6

    def test_the_mass_ratio_is_the_stated_one_thousandth(self) -> None:
        # LS fixes m_e/M = 1e-3 outright. It is not a real gas and there is no amu to
        # derive it from; deriving one would be inventing a number the model does not
        # have.
        assert LUCAS_SAELEE.mass_ratio == 1e-3


class TestTheModelGasType:
    def test_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            REID_RAMP.name = "other"  # type: ignore[misc]

    def test_carries_a_citation_naming_the_primary_source(self) -> None:
        # doc 09 §1 classes these PUBLISHED, and a published value whose paper cannot be
        # named is indistinguishable from one somebody remembered.
        assert "Reid" in REID_RAMP.citation
        assert "1979" in REID_RAMP.citation

    def test_cross_sections_are_evaluated_functionally_not_interpolated(self) -> None:
        # Flynn et al evaluate the model cross sections "functionally at all electron
        # energies with no interpolation error". Sampling a table here would put an
        # interpolation error inside the reference the solver is judged against.
        gas: ModelGas = REID_RAMP
        fine = np.linspace(0.2, 5.0, 4001)

        (channel,) = gas.channels
        np.testing.assert_allclose(
            channel.sigma_m2(fine), 10.0e-20 * (fine - 0.2), rtol=0.0, atol=0.0
        )


class TestThePublishedValues:
    def test_reid_and_flynn_agree_on_the_two_term_answer(self) -> None:
        # The strongest evidence these numbers are right. Reid's own Boltzmann column
        # (1979) and MultiBolt's two-term mode (2024) are independent implementations
        # 45 years apart. Where both report a field, they must agree — if they do not,
        # the transcription is wrong and no solver result means anything.
        by_field = {
            source: {
                b.reduced_field_td: b for b in benchmarks_for("reid-ramp", terms=2, source=source)
            }
            for source in ("reid", "flynn")
        }
        reid, flynn = by_field["reid"], by_field["flynn"]

        shared = sorted(set(reid) & set(flynn))
        assert shared, "the two sources must overlap or neither can check the other"

        for field in shared:
            assert reid[field].mean_energy_ev == pytest.approx(
                flynn[field].mean_energy_ev, rel=2e-3
            ), f"sources disagree on <eps> at {field} Td"
            assert reid[field].drift_velocity_m_per_s == pytest.approx(
                flynn[field].drift_velocity_m_per_s, rel=2e-3
            ), f"sources disagree on W at {field} Td"

    def test_the_two_term_and_multi_term_answers_differ_where_they_should(self) -> None:
        # The two-term approximation is not exact, and the benchmark is only meaningful
        # if it records by how much. At 1 Td the swarm is nearly isotropic and the two
        # agree; by 24 Td they must not, or the multi-term column adds nothing.
        two = {b.reduced_field_td: b for b in benchmarks_for("reid-ramp", terms=2, source="flynn")}
        ten = {b.reduced_field_td: b for b in benchmarks_for("reid-ramp", terms=10, source="flynn")}

        gentle = abs(two[1.0].drift_velocity_m_per_s / ten[1.0].drift_velocity_m_per_s - 1.0)
        harsh = abs(two[24.0].drift_velocity_m_per_s / ten[24.0].drift_velocity_m_per_s - 1.0)

        assert gentle < 0.005
        assert harsh > 0.02

    def test_the_lucas_saelee_point_carries_an_ionisation_rate(self) -> None:
        # The reason LS is in here at all. doc 11 WBS 1.8 asks for *rate coefficients*
        # against published values, and the Reid gas has no ionisation channel.
        (point,) = benchmarks_for("lucas-saelee", terms=2, source="flynn")

        assert point.reduced_field_td == 30.0
        assert point.ionisation_rate_m3_per_s is not None
        assert point.ionisation_rate_m3_per_s == pytest.approx(2.7230e-16, rel=1e-6)

    def test_the_reid_gas_never_carries_an_ionisation_rate(self) -> None:
        for point in benchmarks_for("reid-ramp"):
            assert point.ionisation_rate_m3_per_s is None

    def test_every_published_value_names_its_source_and_term_count(self) -> None:
        # A benchmark that cannot say whether it is the two-term or the converged answer
        # is unusable: our solver is two-term, and judging it against a multi-term
        # column would report a physics approximation as a solver defect.
        for point in published_swarm_benchmarks():
            assert point.source
            assert point.citation
            assert point.terms in (2, 10)

    def test_the_table_covers_the_field_range_reid_tabulated(self) -> None:
        fields = {b.reduced_field_td for b in benchmarks_for("reid-ramp", source="reid")}

        assert fields == {1.0, 5.0, 10.0, 15.0, 20.0, 24.0, 30.0, 40.0}

    def test_values_are_positive_and_finite(self) -> None:
        for point in published_swarm_benchmarks():
            assert point.mean_energy_ev > 0.0
            assert point.drift_velocity_m_per_s > 0.0

    def test_asking_for_an_unknown_gas_raises_rather_than_returning_nothing(self) -> None:
        # An empty tuple would make a benchmark test vacuously pass, which is the one
        # failure mode a verification harness must not have.
        with pytest.raises(KeyError, match="argon"):
            benchmarks_for("argon")


class TestTheBenchmarkType:
    def test_is_immutable(self) -> None:
        point: SwarmBenchmark = published_swarm_benchmarks()[0]

        with pytest.raises(AttributeError):
            point.mean_energy_ev = 1.0  # type: ignore[misc]

    def test_is_printable_with_its_field_and_source(self) -> None:
        rendered = repr(published_swarm_benchmarks()[0])

        assert "Td" in rendered
