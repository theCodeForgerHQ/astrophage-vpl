"""What :mod:`vpl.experiment.channels_interferometry` must be true for — doc 04 §5.1,
doc 05 §3.2, doc 02 §8.3, doc 11 §9 item 3.

This channel was reframed from an 8-chord sheath sampler to a single bulk-density line
integral (see the module docstring under test for the full finding and the reasoning).
Four things this file exists to pin down, each a claim the module docstring states in
prose:

1. The observable tracks ``n_0`` and is blind to the sheath's own shape —
   ``TestObservableIsABulkMeasurementNotASheathSample`` and
   ``TestDensitySensitivityAndTemperatureBlindness``.
2. The doc 01 IF-6 detection floor still gates the channel off at low density —
   ``TestDetectionFloorStillGates`` and
   ``TestFusionIntegration.test_a_plasma_below_the_detection_floor_is_excluded_by_name_not_
   scored_as_noise``.
3. The chord geometry — now a single fixed path length rather than an 8-element ladder —
   still does not move with theta — ``TestChordGeometryDoesNotDependOnTheta``.
4. ``calibration_uncertainty`` (Task 3's amendment) still reaches the instrument through
   :class:`~vpl.inverse.fusion.JointLikelihood`'s fixed two-positional-argument call shape
   — ``TestCalibrationUncertaintyAdapter``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.state import Measurement, PlasmaState
from vpl.core.units import magnitude_in
from vpl.experiment.channels_interferometry import (
    INTERFEROMETRY_CHANNEL,
    InterferometryChannel,
    build_interferometry_channel,
    interferometry_acquisition_window,
    interferometry_calibration_set,
)
from vpl.experiment.closed_loop import (
    _argon_ion,
    _fixed_spatial_grid,
    _reduced_prior,
    _reference_theta,
    _to_plasma_params,
)
from vpl.instruments.interferometry.phase import CHAMBER_DIAMETER_M
from vpl.inverse.fusion import BlindChannelError, Channel, JointLikelihood
from vpl.inverse.parameters import ControlParameters
from vpl.inverse.priors import LogNormalPrior, LogUniformPrior
from vpl.physics.analytic.sheath import AnalyticSheathSolver

_SEED = 13

#: The wall bias the reachability/sensitivity tests below run at. Not RP-1's -250 V for the
#: same reason `test_channels.py`'s `_REACHABLE_BIAS_V` exists — this file has no LIF tuning
#: range to worry about, but running every test at one consistent, documented operating
#: point (rather than RP-1 in some and this bias in others) is what keeps the numbers in
#: this file's docstrings comparable to each other.
_OPERATING_BIAS_V = -100.0

#: A wall bias far from ``_OPERATING_BIAS_V`` (RP-1's own reference point), used only to
#: force a materially different sheath structure while holding ``n_0`` fixed — see
#: ``TestObservableIsABulkMeasurementNotASheathSample``.
_ALTERNATE_BIAS_V = -250.0


def _solver() -> AnalyticSheathSolver:
    return AnalyticSheathSolver()


def _state(theta: ControlParameters) -> PlasmaState:
    registry = default_registry()
    species = _argon_ion(registry)
    solver = _solver()
    grid = _fixed_spatial_grid(
        solver, reference=_reference_theta(), species=species, registry=registry
    )
    params = _to_plasma_params(theta, species=species, registry=registry)
    return solver.solve(params, grid=grid)


def _reference_state() -> PlasmaState:
    return _state(_reference_theta())


def _operating_theta() -> ControlParameters:
    return _reference_theta().replace(V_w=_OPERATING_BIAS_V)


def _map_search_bounds() -> tuple[float, float, float, float]:
    """``(n_0 low, n_0 high, T_e median, T_e log_sigma)`` — ``_reduced_prior``'s own
    doc 05 §2.1 marginal bounds, which is the region a MAP search actually explores.

    ``_reduced_prior().n_0``/``.T_e`` are typed as the abstract
    :class:`~vpl.inverse.priors.Prior`; the ``isinstance`` checks narrow them to the
    concrete family ``default_control_prior`` is documented to use for these two
    parameters, so this file reads their bounds without a ``cast``.
    """
    prior = _reduced_prior()
    n0, t_e = prior.n_0, prior.T_e
    assert isinstance(n0, LogUniformPrior)
    assert isinstance(t_e, LogNormalPrior)
    return n0.low, n0.high, t_e.median, t_e.log_sigma


@pytest.fixture
def channel() -> InterferometryChannel:
    return build_interferometry_channel(
        reference_state=_reference_state(), seed=_SEED, registry=default_registry()
    )


def _noiseless_channel() -> InterferometryChannel:
    """A channel with no truth-side noise and the true calibration — so the truth's own
    ``observe()`` reproduces ``forward()`` exactly and a sensitivity comparison is not
    contaminated by a random noise draw."""
    return build_interferometry_channel(
        reference_state=_reference_state(),
        seed=_SEED,
        registry=default_registry(),
        noise=False,
        imperfect_calibration=False,
    )


class TestTheStandaloneHelpers:
    def test_interferometry_calibration_set_has_the_phase_scale_standard(self) -> None:
        # Reads the registry rather than pinning a number. IF.phase_scale_uncertainty is
        # ASSUMED-class (0.03) with a stated retirement path; this test follows it
        # wherever it goes rather than hardcoding a value that could silently drift out of
        # sync with the registry.
        registry = default_registry()
        refs = interferometry_calibration_set(registry)
        standard = refs.for_quantity("interferometer_phase_scale")

        assert standard.relative_uncertainty == pytest.approx(
            registry.value_in("IF.phase_scale_uncertainty", "dimensionless")
        )

    def test_interferometry_acquisition_window_matches_the_registered_vibration_reference_window(
        self,
    ) -> None:
        registry = default_registry()
        window = interferometry_acquisition_window(registry)
        expected_s = float(registry.value_in("IF.vibration_reference_window_s", "s"))

        assert float(magnitude_in(window.duration, "s")) == pytest.approx(expected_s)
        assert float(magnitude_in(window.start, "s")) == 0.0


class TestBuildDefaults:
    def test_chord_length_m_is_the_doc_02_8_2_chamber_diameter(
        self, channel: InterferometryChannel
    ) -> None:
        # doc 02 §8.2 IF-P2: the chord length is the chamber diameter, a fixed machine
        # dimension, not a fitted parameter — see the module docstring under test. This is
        # the same `CHAMBER_DIAMETER_M` the (unedited) physics module already names.
        assert channel.chord_length_m == pytest.approx(CHAMBER_DIAMETER_M)

    def test_start_z_m_is_accepted_but_no_longer_moves_anything(self) -> None:
        # `start_z_m` has no chord ladder left to anchor (see the module docstring's
        # reframing) but is kept in `build_interferometry_channel`'s signature because
        # `vpl.experiment.channels.build_channels` calls it by keyword and is out of scope
        # for this change. This is the checked claim that supplying a value does nothing,
        # rather than a comment asserting it.
        default = build_interferometry_channel(
            reference_state=_reference_state(), seed=_SEED, registry=default_registry()
        )
        moved = build_interferometry_channel(
            reference_state=_reference_state(),
            seed=_SEED,
            registry=default_registry(),
            start_z_m=0.02,
        )
        state = _state(_operating_theta())

        assert moved.chord_length_m == pytest.approx(default.chord_length_m)
        np.testing.assert_array_equal(
            moved.inversion.forward(state, moved.window).values,
            default.inversion.forward(state, default.window).values,
        )

    def test_the_window_matches_the_registered_vibration_reference_window(
        self, channel: InterferometryChannel
    ) -> None:
        registry = default_registry()
        expected_s = float(registry.value_in("IF.vibration_reference_window_s", "s"))

        assert float(magnitude_in(channel.window.duration, "s")) == pytest.approx(expected_s)
        assert float(magnitude_in(channel.window.start, "s")) == 0.0

    def test_truth_and_inversion_are_separate_instrument_objects(
        self, channel: InterferometryChannel
    ) -> None:
        # doc 04 §9: sharing one object would silently apply the drawn calibration error to
        # both sides and cancel it out of the residual entirely.
        assert channel.truth is not channel.inversion.instrument

    def test_the_channel_name_matches_what_the_instruments_stamp_on_their_output(
        self, channel: InterferometryChannel
    ) -> None:
        observed = channel.truth.observe(_reference_state(), channel.window)

        assert observed.instrument_id == INTERFEROMETRY_CHANNEL


class TestChordGeometryDoesNotDependOnTheta:
    """The trap the module docstring names: the old 8-chord ladder integrated along fixed
    positions rather than sampling one node, so the theta-dependent-observation inverse
    crime ``closed_loop.py`` documents for the spatial grid, and
    ``vpl.experiment.channels`` documents for LIF's default ``z_index``, had to be ruled
    out here by reading the code rather than assumed away by analogy. The reframing to a
    single bulk line integral does not relax that requirement — the chord length is still
    fixed machine geometry (doc 02 §8.2 IF-P2), and this test is the checked guard that it
    stays that way rather than starting to depend on the trial state's own fields.

    Ruled out structurally: :attr:`InterferometryChannel.chord_length_m` is a resolved
    Python float that never reads ``state`` at all, so there is nothing here for a trial
    ``theta`` to move. This test evaluates ``forward`` at several materially different
    thetas spanning ``_reduced_prior``'s full range and checks the geometry is bit-for-bit
    identical before and after.
    """

    def test_chord_length_is_identical_before_and_after_forward_across_the_map_search_region(
        self, channel: InterferometryChannel
    ) -> None:
        n0_low, n0_high, te_median, te_log_sigma = _map_search_bounds()
        reference = _reference_theta()
        thetas = [
            reference,
            reference.replace(n_0=n0_low),
            reference.replace(n_0=n0_high),
            reference.replace(T_e=te_median * math.exp(3.0 * te_log_sigma)),
            reference.replace(T_e=te_median * math.exp(-3.0 * te_log_sigma)),
        ]

        before = channel.inversion.instrument.chord_length_m
        for theta in thetas:
            channel.inversion.forward(_state(theta), channel.window)
        after = channel.inversion.instrument.chord_length_m

        assert before == after


class TestObservableIsABulkMeasurementNotASheathSample:
    """The finding that motivated the reframing: against the closed loop's 0-2.28 mm
    observation grid, the old 8-chord-at-5-mm ladder put seven of its eight chords past the
    grid's edge, where ``np.interp`` clamped them to the outermost node's density — "eight
    independent line integrals" degenerating into one interior reading plus seven repeats
    of the same clamped value. That pathology cannot recur here: the observable below never
    reads the spatial grid, the density field, or the chord ladder at all. It is
    ``r_e * lambda * n_0 * L`` — the doc 04 §5.1 phase evaluated at the bulk density the
    solver treats as the sheath problem's own boundary condition (doc 02 §8.3), over the
    fixed chord length. Both properties below are checked directly rather than argued from
    the formula.
    """

    def test_forward_returns_one_value_not_eight_clamped_chords(
        self, channel: InterferometryChannel
    ) -> None:
        state = _state(_operating_theta())

        predicted = channel.inversion.forward(state, channel.window)

        assert predicted.values.shape == (1,)

    def test_the_observable_is_unchanged_across_wildly_different_sheath_profiles_at_fixed_n_0(
        self,
    ) -> None:
        # Same n_0, very different bias -> very different sheath thickness and z-profile
        # shape (the presheath solved by AnalyticSheathSolver looks nothing alike at -100 V
        # vs -250 V), but the bulk boundary condition n_0 is untouched by V_w. A channel
        # that still depended on the sheath's own shape would move here; this one must not.
        noiseless = _noiseless_channel()
        near_wall_theta = _operating_theta()
        far_wall_theta = near_wall_theta.replace(V_w=_ALTERNATE_BIAS_V)
        assert near_wall_theta.n_0 == far_wall_theta.n_0

        near_state = _state(near_wall_theta)
        far_state = _state(far_wall_theta)
        # Sanity check that the two states really do have different sheath structure, so
        # an unchanged observable below is evidence of insensitivity and not a fixture bug.
        assert not np.allclose(near_state.field("n_e").values, far_state.field("n_e").values)

        near_phase = noiseless.inversion.forward(near_state, noiseless.window)
        far_phase = noiseless.inversion.forward(far_state, noiseless.window)

        np.testing.assert_array_equal(near_phase.values, far_phase.values)


class TestForwardNeverRaisesAcrossTheMapSearchRegion:
    """``JointLikelihood.detail`` only calls ``is_informative`` (a density-floor gate)
    before calling ``forward`` — an exception from ``forward`` crashes the optimiser rather
    than being caught anywhere. Swept across ``_reduced_prior``'s full ``n_0`` range and
    several points of ``T_e``'s log-normal spread, which is not a smaller region than doc
    05 §6's identifiability question is actually asked over.
    """

    def test_forward_and_observe_stay_finite_across_the_prior(
        self, channel: InterferometryChannel
    ) -> None:
        n0_low, n0_high, te_median, te_log_sigma = _map_search_bounds()
        reference = _reference_theta()
        n0_grid = np.geomspace(n0_low, n0_high, 5)
        te_grid = te_median * np.exp(np.linspace(-3.0, 3.0, 5) * te_log_sigma)

        for n0 in n0_grid:
            for te in te_grid:
                theta = reference.replace(n_0=float(n0), T_e=float(te))
                state = _state(theta)

                predicted = channel.inversion.forward(state, channel.window)
                observed = channel.truth.observe(state, channel.window)

                assert np.all(np.isfinite(predicted.values))
                assert np.all(np.isfinite(observed.values))


class TestDetectionFloorStillGates:
    """doc 01 IF-6: reframing what the channel measures must not remove the requirement
    that it declares itself blind below its floor. ``detection_floor_n_e_per_m3`` is the
    same (unedited) formula as before, evaluated over the same chord length, so the floor
    value is unchanged — this is the direct, single-channel check;
    ``TestFusionIntegration`` below checks the same gate through the full fused likelihood.
    """

    def test_is_informative_is_false_below_the_prior_lower_bound(
        self, channel: InterferometryChannel
    ) -> None:
        n0_low, _n0_high, _te_median, _te_log_sigma = _map_search_bounds()
        blind_state = _state(_operating_theta().replace(n_0=n0_low))

        assert channel.inversion.is_informative(blind_state.params) is False

    def test_is_informative_is_true_at_the_reference_density(
        self, channel: InterferometryChannel
    ) -> None:
        reference_state = _reference_state()

        assert channel.inversion.is_informative(reference_state.params) is True


class TestDensitySensitivityAndTemperatureBlindness:
    """doc 04 §5.1: the phase depends on ``n_e`` alone, and under the bulk reframing that
    is ``n_0`` alone — ``n_g`` (the neutral correction, doc 04 §5.2) depends on pressure and
    ``T_g``, never on ``T_e``. Measured directly, not assumed: a 6 % change in ``n_0``
    moves this channel's log-likelihood by a materially non-zero amount, and a comparable
    change in ``T_e`` moves it by exactly zero, bit for bit, because ``_predict`` never
    reads ``T_e`` at all — a stronger, more direct claim than the old chord ladder could
    make (there, chord 0's genuine T_e-via-presheath-fraction dependence was real, just
    small; here there is no code path from ``T_e`` to the prediction whatsoever).
    """

    def test_the_likelihood_moves_for_a_six_percent_change_in_n_0(self) -> None:
        noiseless = _noiseless_channel()
        theta = _operating_theta()
        truth_state = _state(theta)
        observed = noiseless.truth.observe(truth_state, noiseless.window)
        ll_truth = noiseless.inversion.likelihood(
            observed, noiseless.inversion.forward(truth_state, noiseless.window)
        )

        denser_state = _state(theta.replace(n_0=theta.n_0 * 1.06))
        ll_denser = noiseless.inversion.likelihood(
            observed, noiseless.inversion.forward(denser_state, noiseless.window)
        )

        # Observed directly: ~0.26 log-probability units at this operating point.
        assert ll_truth - ll_denser > 0.1

    def test_the_likelihood_is_bit_for_bit_unmoved_by_a_comparable_change_in_t_e(self) -> None:
        noiseless = _noiseless_channel()
        theta = _operating_theta()
        truth_state = _state(theta)
        observed = noiseless.truth.observe(truth_state, noiseless.window)
        ll_truth = noiseless.inversion.likelihood(
            observed, noiseless.inversion.forward(truth_state, noiseless.window)
        )

        hotter_state = _state(theta.replace(T_e=theta.T_e * 1.06))
        ll_hotter = noiseless.inversion.likelihood(
            observed, noiseless.inversion.forward(hotter_state, noiseless.window)
        )

        assert ll_truth == ll_hotter


class TestCalibrationUncertaintyAdapter:
    """Task 3's amendment: ``calibration_uncertainty`` must reach the underlying
    instrument's ``likelihood`` through ``JointLikelihood``'s fixed ``likelihood(obs,
    pred)`` call shape.
    """

    def test_default_channel_is_bit_for_bit_identical_to_the_raw_instrument_with_the_flag_off(
        self,
    ) -> None:
        noiseless = _noiseless_channel()
        state = _state(_operating_theta())
        observed = noiseless.truth.observe(state, noiseless.window)
        predicted = noiseless.inversion.forward(state, noiseless.window)

        via_adapter = noiseless.inversion.likelihood(observed, predicted)
        via_raw = noiseless.inversion.instrument.likelihood(
            observed, predicted, calibration_uncertainty=False
        )

        assert via_adapter == via_raw

    def test_calibration_uncertainty_true_reaches_the_instrument_through_the_adapter(
        self,
    ) -> None:
        with_coherence = build_interferometry_channel(
            reference_state=_reference_state(),
            seed=_SEED,
            registry=default_registry(),
            noise=True,
            imperfect_calibration=True,
            calibration_uncertainty=True,
        )
        state = _state(_operating_theta())
        observed = with_coherence.truth.observe(state, with_coherence.window)
        predicted = with_coherence.inversion.forward(state, with_coherence.window)

        via_adapter = with_coherence.inversion.likelihood(observed, predicted)
        via_raw_off = with_coherence.inversion.instrument.likelihood(
            observed, predicted, calibration_uncertainty=False
        )
        via_raw_on = with_coherence.inversion.instrument.likelihood(
            observed, predicted, calibration_uncertainty=True
        )

        assert via_adapter == via_raw_on
        assert via_adapter != via_raw_off

    def test_requesting_the_coherent_term_never_raises_even_though_use_true_calibration_also_ran(
        self,
    ) -> None:
        # build_interferometry_channel() always calls calibrate() on the inversion
        # instrument (so the standard's registered uncertainty is on record) *and*
        # use_true_calibration() (so its predictions stay exact). This is the check that
        # the combination actually works end to end.
        with_coherence = build_interferometry_channel(
            reference_state=_reference_state(),
            seed=_SEED,
            registry=default_registry(),
            calibration_uncertainty=True,
        )
        state = _state(_operating_theta())
        observed = with_coherence.truth.observe(state, with_coherence.window)
        predicted = with_coherence.inversion.forward(state, with_coherence.window)

        result = with_coherence.inversion.likelihood(observed, predicted)

        assert math.isfinite(result)

    def test_the_wrapped_instrument_is_reachable_for_metadata_and_chord_length(
        self, channel: InterferometryChannel
    ) -> None:
        assert channel.inversion.instrument.chord_length_m == pytest.approx(CHAMBER_DIAMETER_M)
        assert channel.inversion.instrument.metadata().instrument_id == INTERFEROMETRY_CHANNEL


class TestFusionIntegration:
    """The adapter must satisfy :mod:`vpl.inverse.fusion`'s structural instrument protocol
    directly — this channel needs no reconstructed velocity distribution, so the only
    reason it carries an adapter at all is Task 3's ``calibration_uncertainty`` flag.
    """

    def test_the_channel_scores_through_joint_likelihood(
        self, channel: InterferometryChannel
    ) -> None:
        truth_state = _state(_operating_theta())
        observed = channel.truth.observe(truth_state, channel.window)

        joint = JointLikelihood(
            (
                Channel(
                    name=INTERFEROMETRY_CHANNEL,
                    instrument=channel.inversion,
                    observation=observed,
                    window=channel.window,
                ),
            )
        )

        detail = joint.detail(truth_state)

        assert detail.contributing == (INTERFEROMETRY_CHANNEL,)
        assert math.isfinite(detail.log_prob)

    def test_a_plasma_below_the_detection_floor_is_excluded_by_name_not_scored_as_noise(
        self, channel: InterferometryChannel
    ) -> None:
        # doc 01 IF-6: a blind channel contributes no term rather than a weak one. The
        # doc 05 §2.1 prior's own lower support bound (1e15 m**-3) already sits below the
        # doc 02 §8.2 detection floor (~8.4e15 m**-3), so it doubles as a blind point that
        # is still a value ControlParameters.replace will accept.
        truth_state = _state(_operating_theta())
        observed = channel.truth.observe(truth_state, channel.window)
        n0_low, _n0_high, _te_median, _te_log_sigma = _map_search_bounds()
        blind_state = _state(_operating_theta().replace(n_0=n0_low))

        joint = JointLikelihood(
            (
                Channel(
                    name=INTERFEROMETRY_CHANNEL,
                    instrument=channel.inversion,
                    observation=observed,
                    window=channel.window,
                ),
            )
        )

        with pytest.raises(BlindChannelError, match="no channel is above its detection floor"):
            joint.detail(blind_state)


class TestCoherentDiscrepancyAdapter:
    """doc 05 §4's ``coherent_discrepancy`` term, pinned onto
    ``_InterferometryWithOptionalCoherence`` the same way ``calibration_uncertainty`` is —
    :class:`~vpl.inverse.fusion.Channel` calls ``instrument.likelihood(obs, pred)`` with no
    keyword arguments, so the discrepancy has nowhere to travel except as constructor
    state.
    """

    def test_default_is_bit_for_bit_identical_to_no_discrepancy(self) -> None:
        no_discrepancy = build_interferometry_channel(
            reference_state=_reference_state(), seed=_SEED, registry=default_registry()
        )
        explicit_none = build_interferometry_channel(
            reference_state=_reference_state(),
            seed=_SEED,
            registry=default_registry(),
            discrepancy=None,
        )
        state = _state(_operating_theta())
        observed = no_discrepancy.truth.observe(state, no_discrepancy.window)
        predicted = no_discrepancy.inversion.forward(state, no_discrepancy.window)

        assert no_discrepancy.inversion.likelihood(
            observed, predicted
        ) == explicit_none.inversion.likelihood(observed, predicted)

    def test_a_discrepancy_reaches_the_instrument_through_the_adapter(self) -> None:
        state = _state(_operating_theta())
        no_discrepancy = build_interferometry_channel(
            reference_state=_reference_state(), seed=_SEED, registry=default_registry()
        )
        observed = no_discrepancy.truth.observe(state, no_discrepancy.window)
        predicted = no_discrepancy.inversion.forward(state, no_discrepancy.window)
        discrepancy_value = 0.1 * np.abs(np.asarray(predicted.values)) + 1.0e-8

        with_discrepancy = build_interferometry_channel(
            reference_state=_reference_state(),
            seed=_SEED,
            registry=default_registry(),
            discrepancy=discrepancy_value,
        )

        via_adapter = with_discrepancy.inversion.likelihood(observed, predicted)
        via_raw_off = with_discrepancy.inversion.instrument.likelihood(
            observed, predicted, coherent_discrepancy=None
        )
        via_raw_on = with_discrepancy.inversion.instrument.likelihood(
            observed, predicted, coherent_discrepancy=discrepancy_value
        )

        assert via_adapter == via_raw_on
        assert via_adapter != via_raw_off

    def test_a_discrepancy_widens_the_interval_through_the_adapter(self) -> None:
        """doc 00 §5.1 S4's point, reproduced through the adapter: a discrepancy pinned at
        build time must lower the curvature of the log-likelihood, never raise it."""
        state = _state(_operating_theta())
        no_discrepancy = build_interferometry_channel(
            reference_state=_reference_state(), seed=_SEED, registry=default_registry()
        )
        observed = no_discrepancy.truth.observe(state, no_discrepancy.window)
        predicted = no_discrepancy.inversion.forward(state, no_discrepancy.window)
        discrepancy_value = 0.1 * np.abs(np.asarray(predicted.values)) + 1.0e-8

        with_discrepancy = build_interferometry_channel(
            reference_state=_reference_state(),
            seed=_SEED,
            registry=default_registry(),
            discrepancy=discrepancy_value,
        )

        def curvature(built: InterferometryChannel) -> float:
            step = 1.0e-4

            def score(n0_factor: float) -> float:
                perturbed_theta = _operating_theta().replace(n_0=_operating_theta().n_0 * n0_factor)
                perturbed_pred = built.inversion.forward(_state(perturbed_theta), built.window)
                return built.inversion.likelihood(observed, perturbed_pred)

            centre = score(1.0)
            up = score(1.0 + step)
            down = score(1.0 - step)
            return -(up - 2.0 * centre + down) / step**2

        without = curvature(no_discrepancy)
        with_term = curvature(with_discrepancy)

        assert without > 0.0
        assert with_term > 0.0
        assert with_term < without

    def test_matches_a_dense_covariance_via_scipy(self) -> None:
        """The direct algebra check, at ``n = 1`` (the reframed channel's own scale): a
        single-sample instance of the same rank-``k``-plus-vibration composition
        ``InterferometryInstrument.likelihood``'s own test module checks at ``n = 8``."""
        from scipy.stats import multivariate_normal

        from vpl.instruments.interferometry import noise as if_noise

        state = _state(_operating_theta())
        no_discrepancy = build_interferometry_channel(
            reference_state=_reference_state(), seed=_SEED, registry=default_registry()
        )
        predicted = no_discrepancy.inversion.forward(state, no_discrepancy.window)
        n = predicted.n_samples
        assert n == 1

        rng = np.random.default_rng(20260807)
        discrepancy_value = np.abs(rng.normal(scale=2.0e-5, size=n)) + 1.0e-8
        with_discrepancy = build_interferometry_channel(
            reference_state=_reference_state(),
            seed=_SEED,
            registry=default_registry(),
            discrepancy=discrepancy_value,
        )
        residual = rng.normal(scale=1.0e-4, size=n)
        observed = Measurement(
            instrument_id=predicted.instrument_id,
            values=np.asarray(predicted.values) + residual,
            uncertainty=np.ones(n),
            units=predicted.units,
            window=predicted.window,
        )

        result = with_discrepancy.inversion.likelihood(observed, predicted)

        registry = default_registry()
        sigma_independent = if_noise.independent_phase_std_rad(registry=registry)
        sigma_common = if_noise.vibration_phase_std_rad(
            with_discrepancy.window.duration_s, registry=registry
        )
        d = np.full(n, sigma_independent**2)
        v = np.full(n, sigma_common)
        dense_covariance = (
            np.diag(d) + np.outer(v, v) + np.outer(discrepancy_value, discrepancy_value)
        )

        expected = multivariate_normal.logpdf(
            np.asarray(observed.values) - np.asarray(predicted.values),
            mean=np.zeros(n),
            cov=dense_covariance,
        )

        assert result == pytest.approx(expected, rel=1.0e-8)
