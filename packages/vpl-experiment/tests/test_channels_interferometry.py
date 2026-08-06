"""What :mod:`vpl.experiment.channels_interferometry` must be true for — doc 04 §5.1,
doc 05 §3.2, doc 11 §9 item 3.

Three things this file exists to pin down, each a finding the module docstring states in
prose:

1. The chord ladder does not move with theta — ``TestChordGeometryDoesNotDependOnTheta`` is
   the guard the module docstring names.
2. The closed loop's fixed observation grid is far shorter than the chord ladder's 35 mm
   span — ``TestGridDomainVsChordSpanFinding`` pins the measured extent so a later change to
   either number cannot silently un-measure it.
3. ``calibration_uncertainty`` (Task 3's amendment) reaches
   :meth:`~vpl.instruments.interferometry.instrument.InterferometryInstrument.likelihood`
   through :class:`~vpl.inverse.fusion.JointLikelihood`'s fixed two-positional-argument call
   shape — ``TestCalibrationUncertaintyAdapter``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.state import PlasmaState
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
from vpl.instruments.interferometry.phase import CHORD_SPACING_M, N_CHORDS
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
        # Reads the registry rather than pinning a number. The previous version asserted a
        # literal 0.005, which was an invented value — six times tighter than doc 02 §11's
        # only figure, and tighter in the direction that flattered this channel. Pinning a
        # literal here is what let that survive; the entry is now ASSUMED in the register
        # with a retirement path, and this test follows it wherever it goes.
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
    def test_start_z_m_defaults_to_the_wall(self, channel: InterferometryChannel) -> None:
        assert channel.start_z_m == 0.0

    def test_start_z_m_is_configurable_and_reflected_in_the_resolved_field(self) -> None:
        moved = build_interferometry_channel(
            reference_state=_reference_state(),
            seed=_SEED,
            registry=default_registry(),
            start_z_m=0.01,
        )

        assert moved.start_z_m == pytest.approx(0.01)
        np.testing.assert_allclose(
            moved.inversion.instrument.chord_positions_m(),
            0.01 + CHORD_SPACING_M * np.arange(N_CHORDS),
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
    """The trap this module's docstring names: interferometry integrates along chords
    rather than sampling one node, so the same class of theta-dependent-observation inverse
    crime ``closed_loop.py`` documents for the spatial grid, and
    ``vpl.experiment.channels`` documents for LIF's default ``z_index``, had to be ruled out
    here by reading the code rather than assumed away by analogy.

    It is ruled out structurally: ``InterferometryInstrument.chord_positions_m`` takes no
    ``state`` argument at all, so there is nothing here for a trial ``theta`` to move. This
    test evaluates ``forward`` at several materially different thetas spanning
    ``_reduced_prior``'s full range and checks the chord ladder is bit-for-bit identical
    before and after — so this is a checked property of the code, not a claim resting on
    having read it once.
    """

    def test_chord_positions_are_identical_before_and_after_forward_across_the_map_search_region(
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

        before = channel.inversion.instrument.chord_positions_m().copy()
        for theta in thetas:
            channel.inversion.forward(_state(theta), channel.window)
        after = channel.inversion.instrument.chord_positions_m()

        np.testing.assert_array_equal(before, after)


class TestGridDomainVsChordSpanFinding:
    """Measured finding, pinned as a regression check — see the module docstring's "grid
    domain" section. ``start_z_m`` is deliberately not moved and the chord spacing is
    deliberately not shrunk to make this look better; IF-G1's 5 mm spacing is doc 02 §8.2's
    hardware specification, not a knob this module owns.
    """

    def test_the_fixed_observation_grid_is_shorter_than_the_full_chord_ladder(self) -> None:
        registry = default_registry()
        species = _argon_ion(registry)
        solver = _solver()
        grid = _fixed_spatial_grid(
            solver, reference=_reference_theta(), species=species, registry=registry
        )
        chord_z = CHORD_SPACING_M * np.arange(N_CHORDS)

        assert grid.z_m[-1] < chord_z[-1]
        inside = int(np.sum(chord_z <= grid.z_m[-1]))
        assert inside == 1


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


class TestDensitySensitivityAndTemperatureBlindness:
    """doc 04 §5.1: the phase depends on ``n_e`` alone. Measured directly, not assumed: a
    6 % change in ``n_0`` moves this channel's log-likelihood by O(1); a comparable change
    in ``T_e`` moves it by less than the double-precision noise floor of the sum itself —
    against this grid (see ``TestGridDomainVsChordSpanFinding``), the seven clamped chords
    report a presheath density this L0 solver sets from ``n_0`` alone (the Bohm-criterion
    presheath fraction), and the one interior chord's genuine ``T_e`` dependence lives at an
    absolute density many orders of magnitude below the bulk value it is compared against.
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

        # Observed directly: ~0.92 log-probability units at this operating point.
        assert ll_truth - ll_denser > 0.1

    def test_the_likelihood_is_essentially_unmoved_by_a_comparable_change_in_t_e(self) -> None:
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

        assert abs(ll_truth - ll_hotter) < 1.0e-6


class TestCalibrationUncertaintyAdapter:
    """Task 3's amendment: ``calibration_uncertainty`` must reach
    ``InterferometryInstrument.likelihood`` through ``JointLikelihood``'s fixed
    ``likelihood(obs, pred)`` call shape, via ``_InterferometryWithOptionalCoherence``.
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
        # use_true_calibration() (so its predictions stay exact) — see the module
        # docstring's "Scoring the calibration coherently" section for why both are
        # necessary. This is the check that the combination actually works end to end.
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

    def test_the_wrapped_instrument_is_reachable_for_metadata_and_chord_positions(
        self, channel: InterferometryChannel
    ) -> None:
        assert channel.inversion.instrument.chord_positions_m().size == N_CHORDS
        assert channel.inversion.instrument.metadata().instrument_id == INTERFEROMETRY_CHANNEL


class TestFusionIntegration:
    """The adapter must satisfy :mod:`vpl.inverse.fusion`'s structural instrument protocol
    directly — unlike LIF, this channel needs no reconstructed velocity distribution, so the
    only reason it carries an adapter at all is Task 3's ``calibration_uncertainty`` flag.
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
