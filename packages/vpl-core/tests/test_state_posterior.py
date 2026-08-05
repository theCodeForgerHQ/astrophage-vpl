"""Posterior and sampler diagnostics — doc 05 §10, doc 07 §2.2 and ADR-005.

The deliverable of an inversion is a distribution (doc 05 §1.1), and doc 06 §8 forbids
reporting a bare number: the tier, the credible interval and the sampler diagnostics
travel with the value. These tests hold the type to that, and in particular to the
ADR-005 consequence that thinning must be ESS-aware rather than fixed-stride — a
fixed-stride thin silently biases the coverage statistics of doc 06 §7.1, which are the
evidence for doc 00 S4.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

from vpl.core.provenance import Tier
from vpl.core.state import (
    GATE_MAX_DIVERGENCES,
    GATE_MAX_R_HAT,
    GATE_MIN_ESS,
    CredibleInterval,
    ParameterLevel,
    Posterior,
    SamplerDiagnostics,
)

# A miniature version of the doc 05 §2 split: two control parameters, one nuisance
# parameter and one discrepancy-field coefficient.
_LEVELS: Mapping[str, ParameterLevel] = {
    "n_0": ParameterLevel.CONTROL,
    "T_e": ParameterLevel.CONTROL,
    "oes_scale": ParameterLevel.NUISANCE,
    "alpha_1": ParameterLevel.PROFILE,
}
_NAMES: tuple[str, ...] = ("n_0", "T_e", "oes_scale", "alpha_1")


def _diagnostics(
    names: tuple[str, ...] = _NAMES,
    *,
    r_hat: float = 1.0,
    ess: float = 1000.0,
    divergences: int = 0,
    e_bfmi: tuple[float, ...] = (),
) -> SamplerDiagnostics:
    return SamplerDiagnostics(
        r_hat=dict.fromkeys(names, r_hat),
        ess=dict.fromkeys(names, ess),
        divergences=divergences,
        e_bfmi=e_bfmi,
    )


def _samples(n_chains: int, n_draws: int, n_params: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(20260804)
    return rng.standard_normal((n_chains, n_draws, n_params))


def _posterior(
    *,
    n_chains: int = 4,
    n_draws: int = 8,
    ess: float = 1000.0,
    r_hat: float = 1.0,
    divergences: int = 0,
    tier: Tier = Tier.T2,
    derived: Mapping[str, NDArray[np.float64]] | None = None,
) -> Posterior:
    return Posterior(
        samples=_samples(n_chains, n_draws, len(_NAMES)),
        names=_NAMES,
        levels=_LEVELS,
        tier=tier,
        diagnostics=_diagnostics(r_hat=r_hat, ess=ess, divergences=divergences),
        derived={} if derived is None else derived,
    )


class TestGateConstants:
    def test_the_constants_are_the_g_v3_thresholds_of_doc_07_section_6(self) -> None:
        assert GATE_MAX_R_HAT == 1.01
        assert GATE_MIN_ESS == 400.0
        assert GATE_MAX_DIVERGENCES == 0


class TestSamplerDiagnosticsConstruction:
    def test_rejects_a_parameter_with_an_r_hat_but_no_ess(self) -> None:
        # G-V3 requires ESS *per parameter*; a missing entry would let a parameter
        # through the gate without ever being tested.
        with pytest.raises(ValueError, match="ess"):
            SamplerDiagnostics(r_hat={"n_0": 1.0, "T_e": 1.0}, ess={"n_0": 900.0}, divergences=0)

    def test_rejects_a_negative_divergence_count(self) -> None:
        with pytest.raises(ValueError, match="divergences"):
            SamplerDiagnostics(r_hat={"n_0": 1.0}, ess={"n_0": 900.0}, divergences=-1)

    def test_rejects_a_non_finite_r_hat(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            SamplerDiagnostics(r_hat={"n_0": np.inf}, ess={"n_0": 900.0}, divergences=0)

    def test_rejects_a_non_positive_ess(self) -> None:
        with pytest.raises(ValueError, match="ess"):
            SamplerDiagnostics(r_hat={"n_0": 1.0}, ess={"n_0": 0.0}, divergences=0)

    def test_rejects_a_non_positive_e_bfmi(self) -> None:
        with pytest.raises(ValueError, match="e_bfmi"):
            SamplerDiagnostics(
                r_hat={"n_0": 1.0}, ess={"n_0": 900.0}, divergences=0, e_bfmi=(0.9, 0.0)
            )

    def test_repr_names_the_worst_value_of_each_gated_quantity(self) -> None:
        text = repr(_diagnostics(r_hat=1.004, ess=612.0, divergences=3))

        assert "1.004" in text
        assert "612" in text
        assert "3" in text

    def test_reports_the_parameters_it_covers(self) -> None:
        assert _diagnostics().parameters == _NAMES

    def test_the_mappings_are_copies_the_caller_cannot_edit(self) -> None:
        r_hat = {"n_0": 1.0}
        diagnostics = SamplerDiagnostics(r_hat=r_hat, ess={"n_0": 900.0}, divergences=0)

        r_hat["n_0"] = 9.9

        assert diagnostics.r_hat["n_0"] == 1.0
        with pytest.raises(TypeError):
            diagnostics.r_hat["n_0"] = 9.9  # type: ignore[index]


class TestSamplerDiagnosticsGate:
    def test_a_clean_run_passes_gate_g_v3(self) -> None:
        assert _diagnostics(r_hat=1.005, ess=900.0, divergences=0).is_clean()

    def test_an_r_hat_exactly_at_the_threshold_fails_because_the_gate_is_strict(self) -> None:
        # doc 07 §6 writes "R-hat < 1.01", not "<=".
        assert not _diagnostics(r_hat=GATE_MAX_R_HAT).is_clean()

    def test_an_ess_exactly_at_the_floor_fails_because_the_gate_is_strict(self) -> None:
        # doc 07 §6 and V-44 both write "ESS > 400". ADR-005 writes ">= 400"; the gate
        # is authoritative and the strict form is the safe reading of the two.
        assert not _diagnostics(ess=GATE_MIN_ESS).is_clean()

    def test_one_parameter_below_the_floor_fails_the_whole_gate(self) -> None:
        diagnostics = SamplerDiagnostics(
            r_hat=dict.fromkeys(_NAMES, 1.0),
            ess={**dict.fromkeys(_NAMES, 900.0), "alpha_1": 12.0},
            divergences=0,
        )

        assert not diagnostics.is_clean()

    def test_a_single_divergence_fails(self) -> None:
        assert not _diagnostics(divergences=1).is_clean()

    def test_failures_name_the_offending_parameter_and_its_value(self) -> None:
        failures = _diagnostics(("n_0",), ess=12.0).gate_failures()

        assert len(failures) == 1
        assert "n_0" in failures[0]
        assert "12" in failures[0]

    def test_a_clean_run_reports_no_failures(self) -> None:
        assert _diagnostics().gate_failures() == ()

    def test_a_benchmark_may_tighten_the_thresholds(self) -> None:
        diagnostics = _diagnostics(r_hat=1.005, ess=900.0)

        assert diagnostics.is_clean(max_r_hat=1.01, min_ess=800.0)
        assert not diagnostics.is_clean(max_r_hat=1.001, min_ess=800.0)
        assert not diagnostics.is_clean(min_ess=1200.0)

    def test_a_loosened_r_hat_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="G-V3"):
            _diagnostics().is_clean(max_r_hat=1.05)

    def test_a_loosened_ess_floor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="G-V3"):
            _diagnostics().is_clean(min_ess=100.0)

    def test_a_loosened_divergence_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="G-V3"):
            _diagnostics().is_clean(max_divergences=3)


class TestPosteriorConstruction:
    def test_records_the_three_level_split_of_doc_05_section_2(self) -> None:
        posterior = _posterior()

        assert posterior.names_at(ParameterLevel.CONTROL) == ("n_0", "T_e")
        assert posterior.names_at(ParameterLevel.NUISANCE) == ("oes_scale",)
        assert posterior.names_at(ParameterLevel.PROFILE) == ("alpha_1",)
        assert posterior.level_of("alpha_1") is ParameterLevel.PROFILE

    def test_each_level_describes_the_doc_05_section_2_role_it_plays(self) -> None:
        assert "Level A" in ParameterLevel.CONTROL.description
        assert "Level B" in ParameterLevel.NUISANCE.description
        assert "discrepancy field" in ParameterLevel.PROFILE.description

    def test_reports_its_shape(self) -> None:
        posterior = _posterior(n_chains=4, n_draws=8)

        assert posterior.n_chains == 4
        assert posterior.n_draws == 8
        assert posterior.n_params == 4
        assert posterior.n_samples == 32

    def test_requires_a_tier_because_doc_05_section_7_2_forbids_defaulting_it(self) -> None:
        with pytest.raises(TypeError):
            Posterior(  # type: ignore[call-arg]
                samples=_samples(2, 4, 4),
                names=_NAMES,
                levels=_LEVELS,
                diagnostics=_diagnostics(),
            )

    def test_rejects_a_level_map_that_omits_a_parameter(self) -> None:
        with pytest.raises(ValueError, match="levels"):
            Posterior(
                samples=_samples(2, 4, 4),
                names=_NAMES,
                levels={"n_0": ParameterLevel.CONTROL},
                tier=Tier.T2,
                diagnostics=_diagnostics(),
            )

    def test_rejects_diagnostics_that_omit_a_parameter(self) -> None:
        with pytest.raises(ValueError, match="diagnostics"):
            Posterior(
                samples=_samples(2, 4, 4),
                names=_NAMES,
                levels=_LEVELS,
                tier=Tier.T2,
                diagnostics=_diagnostics(("n_0", "T_e", "oes_scale")),
            )

    def test_rejects_samples_whose_last_axis_does_not_match_the_names(self) -> None:
        with pytest.raises(ValueError, match="n_params"):
            Posterior(
                samples=_samples(2, 4, 3),
                names=_NAMES,
                levels=_LEVELS,
                tier=Tier.T2,
                diagnostics=_diagnostics(),
            )

    def test_rejects_a_sample_array_that_is_not_chains_by_draws_by_parameters(self) -> None:
        with pytest.raises(ValueError, match="three-dimensional"):
            Posterior(
                samples=np.zeros((4, 4)),
                names=("n_0", "T_e", "oes_scale", "alpha_1"),
                levels=_LEVELS,
                tier=Tier.T2,
                diagnostics=_diagnostics(),
            )

    def test_rejects_duplicate_parameter_names(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            Posterior(
                samples=_samples(2, 4, 4),
                names=("n_0", "n_0", "oes_scale", "alpha_1"),
                levels=_LEVELS,
                tier=Tier.T2,
                diagnostics=_diagnostics(),
            )

    def test_rejects_non_finite_samples(self) -> None:
        samples = _samples(2, 4, 4)
        samples[0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="finite"):
            Posterior(
                samples=samples,
                names=_NAMES,
                levels=_LEVELS,
                tier=Tier.T2,
                diagnostics=_diagnostics(),
            )

    def test_rejects_an_e_bfmi_that_does_not_cover_every_chain(self) -> None:
        with pytest.raises(ValueError, match="e_bfmi"):
            Posterior(
                samples=_samples(4, 4, 4),
                names=_NAMES,
                levels=_LEVELS,
                tier=Tier.T2,
                diagnostics=_diagnostics(e_bfmi=(0.9, 0.8)),
            )


class TestPosteriorDerivedQuantities:
    def test_carries_derived_quantities_alongside_the_samples(self) -> None:
        # doc 05 §10 requires Gamma_E(z, t), Gamma_i, <E_i> and the wall IEDF, none of
        # which are sampled parameters: they are functions of the samples.
        gamma_e = np.ones((4, 8, 16))
        posterior = _posterior(derived={"Gamma_E": gamma_e, "E_i_mean": np.ones((4, 8))})

        assert posterior.derived_names == ("E_i_mean", "Gamma_E")
        assert posterior.derived_for("Gamma_E").shape == (4, 8, 16)

    def test_derived_for_refuses_a_sampled_parameter_name(self) -> None:
        with pytest.raises(KeyError, match="T_e"):
            _posterior().derived_for("T_e")

    def test_rejects_a_derived_quantity_that_shadows_a_parameter_name(self) -> None:
        with pytest.raises(ValueError, match="shadow"):
            _posterior(derived={"T_e": np.ones((4, 8))})

    def test_rejects_a_derived_quantity_whose_draw_axes_do_not_match(self) -> None:
        # A derived quantity is a function of the samples, so it has one value per
        # draw. Anything else means it was computed against different chains.
        with pytest.raises(ValueError, match="Gamma_E"):
            _posterior(derived={"Gamma_E": np.ones((4, 7, 16))})


class TestPosteriorImmutability:
    def test_the_sample_array_cannot_be_written(self) -> None:
        posterior = _posterior()

        with pytest.raises(ValueError, match="read-only"):
            posterior.samples[0, 0, 0] = 5.0

    def test_construction_copies_the_caller_samples(self) -> None:
        samples = _samples(2, 4, 4)
        posterior = Posterior(
            samples=samples,
            names=_NAMES,
            levels=_LEVELS,
            tier=Tier.T2,
            diagnostics=_diagnostics(),
        )
        original = float(samples[0, 0, 0])

        samples[0, 0, 0] = 99.0

        assert float(posterior.samples[0, 0, 0]) == original

    def test_derived_arrays_cannot_be_written(self) -> None:
        posterior = _posterior(derived={"Gamma_E": np.ones((4, 8, 3))})

        with pytest.raises(ValueError, match="read-only"):
            posterior.derived_for("Gamma_E")[0, 0, 0] = 5.0

    def test_the_derived_mapping_cannot_be_extended(self) -> None:
        posterior = _posterior(derived={"Gamma_E": np.ones((4, 8, 3))})

        with pytest.raises(TypeError):
            posterior.derived["IEDF"] = np.ones((4, 8, 3))  # type: ignore[index]

    def test_the_fields_cannot_be_rebound(self) -> None:
        posterior = _posterior()

        with pytest.raises(AttributeError):
            posterior.tier = Tier.T1  # type: ignore[misc]


class TestPosteriorSummaries:
    def _ramped(self) -> Posterior:
        """A posterior whose ``n_0`` draws are exactly 1..100, for checkable quantiles."""
        samples = np.zeros((2, 50, len(_NAMES)))
        samples[:, :, 0] = np.arange(1.0, 101.0).reshape(2, 50)
        return Posterior(
            samples=samples,
            names=_NAMES,
            levels=_LEVELS,
            tier=Tier.T2,
            diagnostics=_diagnostics(),
        )

    def test_samples_for_returns_one_parameters_chains(self) -> None:
        posterior = self._ramped()

        assert posterior.samples_for("n_0").shape == (2, 50)
        np.testing.assert_allclose(posterior.samples_for("n_0")[0], np.arange(1.0, 51.0))

    def test_mean_pools_chains_and_draws(self) -> None:
        assert float(self._ramped().mean("n_0")) == pytest.approx(50.5)

    def test_the_mean_of_a_scalar_parameter_is_a_scalar(self) -> None:
        assert self._ramped().mean("n_0").shape == ()

    def test_the_mean_of_a_derived_profile_keeps_its_trailing_shape(self) -> None:
        profile = np.tile(np.arange(16.0), (4, 8, 1))
        posterior = _posterior(derived={"Gamma_E": profile})

        assert posterior.mean("Gamma_E").shape == (16,)
        np.testing.assert_allclose(posterior.mean("Gamma_E"), np.arange(16.0))

    def test_credible_interval_is_equal_tailed(self) -> None:
        # Equal-tailed at 95%: the 2.5% and 97.5% quantiles of 1..100.
        interval = self._ramped().credible_interval("n_0", 0.95)

        assert float(interval.lower) == pytest.approx(3.475)
        assert float(interval.upper) == pytest.approx(97.525)

    def test_credible_interval_carries_the_level_it_was_computed_at(self) -> None:
        # doc 06 §8: the metadata travels with the value, mechanically.
        interval = self._ramped().credible_interval("n_0", 0.5)

        assert isinstance(interval, CredibleInterval)
        assert interval.level == 0.5
        assert float(interval.lower) == pytest.approx(25.75)
        assert float(interval.upper) == pytest.approx(75.25)

    def test_credible_interval_defaults_to_the_ninety_five_percent_level(self) -> None:
        # doc 06 §7.1 makes 95% the level the coverage gate G-V4 is stated at.
        assert self._ramped().credible_interval("n_0").level == 0.95

    def test_credible_interval_of_a_derived_profile_is_computed_per_point(self) -> None:
        profile = np.tile(np.arange(16.0), (4, 8, 1))
        interval = _posterior(derived={"Gamma_E": profile}).credible_interval("Gamma_E")

        assert interval.lower.shape == (16,)
        np.testing.assert_allclose(interval.lower, np.arange(16.0))

    def test_rejects_a_credible_level_outside_the_open_unit_interval(self) -> None:
        posterior = self._ramped()

        with pytest.raises(ValueError, match="level"):
            posterior.credible_interval("n_0", 1.0)
        with pytest.raises(ValueError, match="level"):
            posterior.credible_interval("n_0", 0.0)

    def test_an_unknown_name_is_a_key_error(self) -> None:
        with pytest.raises(KeyError, match="V_w"):
            self._ramped().mean("V_w")

    def test_samples_for_refuses_a_derived_name(self) -> None:
        # Derived quantities are not samples; conflating them would let a caller treat
        # a function of the posterior as a sampled parameter.
        posterior = _posterior(derived={"Gamma_E": np.ones((4, 8, 3))})

        with pytest.raises(KeyError, match="Gamma_E"):
            posterior.samples_for("Gamma_E")


class TestEssAwareThinning:
    def _run(self, *, ess: float = 1600.0, n_draws: int = 2000) -> Posterior:
        return _posterior(
            n_chains=4,
            n_draws=n_draws,
            ess=ess,
            derived={"Gamma_E": np.ones((4, n_draws, 3))},
        )

    def test_thinning_keeps_every_kth_draw_of_every_chain(self) -> None:
        posterior = self._run()
        thinned = posterior.thin(19)

        assert thinned.n_chains == 4
        assert thinned.n_draws == 106
        np.testing.assert_allclose(thinned.samples, posterior.samples[:, ::19, :])

    def test_thinning_thins_derived_quantities_alongside_the_samples(self) -> None:
        # If the derived quantities were not thinned with the samples, the draw index
        # would no longer line up and every derived credible interval would be wrong.
        thinned = self._run().thin(19)

        assert thinned.derived_for("Gamma_E").shape == (4, 106, 3)

    def test_the_thinned_posterior_reports_its_reduced_ess(self) -> None:
        # ADR-005: thinning must be ESS-aware. A thinned posterior that kept the
        # pre-thinning ESS would let downstream gates pass on numbers it no longer has.
        thinned = self._run().thin(19)

        assert thinned.diagnostics.ess["n_0"] == pytest.approx(424.0)

    def test_thinning_never_inflates_ess_above_the_original(self) -> None:
        thinned = self._run(ess=600.0).thin(1)

        assert thinned.diagnostics.ess["n_0"] == pytest.approx(600.0)

    def test_thinning_preserves_r_hat_and_the_divergence_count(self) -> None:
        # Divergences are a property of the sampling run, not of the retained draws;
        # ADR-005 keeps the full chains regenerable, so the count stays true of them.
        posterior = _posterior(n_chains=4, n_draws=2000, ess=1600.0, r_hat=1.004)
        thinned = posterior.thin(19)

        assert thinned.diagnostics.r_hat["n_0"] == pytest.approx(1.004)
        assert thinned.diagnostics.divergences == 0

    def test_thinning_preserves_the_tier(self) -> None:
        assert self._run().thin(19).tier is Tier.T2

    def test_max_safe_stride_is_the_largest_stride_that_holds_the_g_v3_floor(self) -> None:
        # 4 chains x 2000 draws: stride 19 retains 4 x 106 = 424 > 400; stride 20
        # retains exactly 400, which the strict gate rejects.
        assert self._run().max_safe_stride() == 19

    def test_the_maximum_safe_stride_yields_a_posterior_that_passes_the_gate(self) -> None:
        posterior = self._run()

        assert posterior.thin(posterior.max_safe_stride()).diagnostics.is_clean()

    def test_a_stride_one_beyond_the_maximum_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ESS"):
            self._run().thin(20)

    def test_refuses_to_thin_a_posterior_that_already_fails_the_floor(self) -> None:
        # No stride can rescue a chain that never reached the floor, so even a no-op
        # thin is refused rather than blessing the samples as archival.
        with pytest.raises(ValueError, match="ESS"):
            self._run(ess=300.0).thin(1)

    def test_max_safe_stride_refuses_a_posterior_that_already_fails_the_floor(self) -> None:
        with pytest.raises(ValueError, match="ESS"):
            self._run(ess=300.0).max_safe_stride()

    def test_an_explicitly_lowered_floor_permits_deeper_thinning(self) -> None:
        # The escape hatch is not silent: the floor is named at the call site and the
        # thinned posterior's own diagnostics then fail G-V3.
        thinned = self._run().thin(100, min_ess=50.0)

        assert thinned.n_draws == 20
        assert thinned.diagnostics.ess["n_0"] == pytest.approx(80.0)
        assert not thinned.diagnostics.is_clean()

    def test_a_tightened_floor_shortens_the_safe_stride(self) -> None:
        assert self._run().max_safe_stride(min_ess=800.0) == 9

    def test_rejects_a_non_positive_stride(self) -> None:
        with pytest.raises(ValueError, match="stride"):
            self._run().thin(0)

    def test_thinning_by_one_is_a_no_op(self) -> None:
        posterior = self._run()

        assert posterior.thin(1) == posterior


class TestPosteriorProvenanceAndEquality:
    def test_the_tier_travels_with_the_posterior(self) -> None:
        assert _posterior(tier=Tier.T1).tier is Tier.T1

    def test_equality_compares_samples_names_levels_and_tier(self) -> None:
        a = _posterior()
        b = _posterior()
        c = _posterior(tier=Tier.T1)

        assert a == b
        assert a != c
        assert a != "not a posterior"

    def test_with_diagnostics_replaces_them_without_touching_the_samples(self) -> None:
        # doc 05 §10 emits diagnostics per inversion, but ESS recomputed on the thinned
        # draws by vpl-uq supersedes the bound thinning uses; installing it must not
        # be an excuse to rebuild the posterior.
        posterior = _posterior()
        replaced = posterior.with_diagnostics(_diagnostics(ess=777.0))

        assert replaced.diagnostics.ess["n_0"] == pytest.approx(777.0)
        np.testing.assert_allclose(replaced.samples, posterior.samples)
        assert posterior.diagnostics.ess["n_0"] == pytest.approx(1000.0)

    def test_with_diagnostics_rejects_a_set_that_omits_a_parameter(self) -> None:
        with pytest.raises(ValueError, match="diagnostics"):
            _posterior().with_diagnostics(_diagnostics(("n_0",)))

    def test_repr_names_the_shape_and_the_tier(self) -> None:
        text = repr(_posterior(n_chains=4, n_draws=8))

        assert "4" in text
        assert "T2" in text
