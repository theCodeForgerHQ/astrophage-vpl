"""Coverage for the Thomson scattering channel builder — doc 05 §3.2, doc 11 §9 item 3.

The reason this channel is being connected at all: with honest 6 % OES calibration
uncertainty, a single optical channel cannot separate `n_0` from `T_e` — overall
brightness carries all the density information, so "the plasma is denser" and "my lamp
reads bright" are the same data. `TestWhatThomsonMeasuresThatBrightnessAlone
Cannot` is the test that decides whether connecting Thomson is worth doing, mirroring
`test_channels.py`'s `TestWhatLifAddsThatOesCannotSee` for the same reason: it is not a
nice-to-have, it is the check that the whole exercise has a point.

`TestTheMeasurementVolumeIsPinned` is the guard against the class of inverse crime
`channels.py`'s module docstring documents for LIF and `closed_loop.py`'s documents for the
spatial grid: a measurement location that moves with the trial parameters lets an optimiser
improve its score by relocating the instrument rather than by finding the right physics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.core.state import PlasmaState, ScalarField
from vpl.core.units import magnitude_in
from vpl.experiment.channels_thomson import (
    THOMSON_CHANNEL,
    ThomsonChannel,
    build_thomson_channel,
    thomson_acquisition_window,
    thomson_calibration_set,
)
from vpl.experiment.closed_loop import (
    _argon_ion,
    _fixed_spatial_grid,
    _reference_theta,
    _to_plasma_params,
)
from vpl.instruments.thomson import photons
from vpl.inverse.fusion import Channel, JointLikelihood
from vpl.inverse.parameters import ControlParameters
from vpl.physics.analytic.sheath import AnalyticSheathSolver

#: The single recorded seed (doc 00 E3) this file's channel fixture derives its streams
#: from. Not `test_channels.py`'s `_SEED = 7`: a different value demonstrates nothing
#: here depends on that file's particular draw.
_SEED = 11


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


@pytest.fixture
def reference_state() -> PlasmaState:
    return _reference_state()


@pytest.fixture
def channel(reference_state: PlasmaState) -> ThomsonChannel:
    return build_thomson_channel(
        reference_state=reference_state, seed=_SEED, registry=default_registry()
    )


@pytest.fixture
def noiseless_channel(reference_state: PlasmaState) -> ThomsonChannel:
    # A deterministic truth side, used wherever a test wants to attribute a log-likelihood
    # change to the physics being varied rather than to a particular photon-shot draw.
    return build_thomson_channel(
        reference_state=reference_state, seed=_SEED, registry=default_registry(), noise=False
    )


def _joint_from(channel: ThomsonChannel, truth_state: PlasmaState) -> JointLikelihood:
    observation = channel.truth.observe(truth_state, channel.window)
    return JointLikelihood(
        (
            Channel(
                name=THOMSON_CHANNEL,
                instrument=channel.inversion,
                observation=observation,
                window=channel.window,
            ),
        )
    )


class TestThomsonCalibrationSet:
    """doc 02 §7.3's one standard."""

    def test_it_carries_the_rayleigh_standard_with_the_packages_own_uncertainty(self) -> None:
        calibration_set = thomson_calibration_set()

        standard = calibration_set.for_quantity(photons.RAYLEIGH_CALIBRATION_QUANTITY)

        assert standard.relative_uncertainty == photons.RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY
        assert standard.name == "Rayleigh scattering from a known neutral fill"


class _FakeTimeResolvedField:
    """A field double exposing only `is_steady` — see `fusion.py`'s own doubles for why a
    local stand-in is used here instead of constructing a real time-resolved `PlasmaState`
    (which would need a full field set and a time grid this test has no other use for)."""

    is_steady = False


class _FakeGrid:
    n_points = 5


class _FakeTimeResolvedState:
    grid = _FakeGrid()

    def field(self, name: str) -> _FakeTimeResolvedField:
        return _FakeTimeResolvedField()


class TestThomsonAcquisitionWindow:
    """doc 02 §7.1's photon budget, sized from the reference state and never the truth's."""

    def test_it_matches_the_photon_budget_formula_at_the_pinned_index(
        self, reference_state: PlasmaState
    ) -> None:
        # Independent of `channels_thomson`'s own private helpers: this recomputes the
        # expected duration directly from the documented default index
        # (`grid.n_points - 1`) and `photons.required_accumulation_s`, so the test cannot
        # pass merely because it calls the same private function the implementation does.
        pinned_index = reference_state.grid.n_points - 1
        density_m3 = float(reference_state.field("n_e").values[pinned_index])
        expected_s = photons.required_accumulation_s(
            electron_density_m3=density_m3,
            target_relative_uncertainty=photons.DEFAULT_TARGET_RELATIVE_UNCERTAINTY,
        )

        window = thomson_acquisition_window(reference_state)

        assert float(magnitude_in(window.duration, "s")) == pytest.approx(expected_s, rel=1e-12)
        assert float(magnitude_in(window.start, "s")) == 0.0

    def test_it_is_the_same_order_of_magnitude_as_doc_02_7_1s_worked_example(
        self, reference_state: PlasmaState
    ) -> None:
        # doc 02 §7.1 quotes "~700 s" for a 3 % point at n_e = 1e17 m^-3 exactly. The
        # pinned index sits in the quasineutral presheath, where n_e is somewhat below the
        # bulk RP1.n_0, so the reproduced figure is longer than 700 s but on the same
        # physical scale rather than, say, an order of magnitude off from a units error.
        window = thomson_acquisition_window(reference_state)

        duration_s = float(magnitude_in(window.duration, "s"))

        assert 300.0 < duration_s < 3000.0

    def test_it_is_sized_from_the_reference_state_never_the_state_under_test(self) -> None:
        reference = _reference_state()
        theta = _reference_theta()
        denser = _state(theta.replace(n_0=theta.n_0 * 4.0))

        reference_window = thomson_acquisition_window(reference)
        denser_window = thomson_acquisition_window(denser)

        # photoelectrons_per_shot is linear in n_e (doc 02 §7.1), so a state that is 4x
        # denser at the pinned index needs a quarter of the accumulation for the same
        # target precision -- *if* the window were sized from that state. It is not: the
        # two windows below are each computed straight from the state passed to
        # `thomson_acquisition_window`, so this is a property of the function, not of
        # `build_thomson_channel`'s discipline of always calling it on the reference.
        ratio = float(magnitude_in(reference_window.duration, "s")) / float(
            magnitude_in(denser_window.duration, "s")
        )
        assert ratio == pytest.approx(4.0, rel=1e-6)

    def test_it_refuses_a_time_resolved_reference_state(self) -> None:
        with pytest.raises(ValueError, match="must be steady"):
            thomson_acquisition_window(_FakeTimeResolvedState())  # type: ignore[arg-type]


class TestBuildThomsonChannel:
    """doc 04 §9: truth-side and inversion-side instruments are separate objects."""

    def test_truth_and_inversion_are_separate_instrument_objects(
        self, channel: ThomsonChannel
    ) -> None:
        assert channel.truth is not channel.inversion

    def test_both_sides_stamp_the_channel_name_the_module_exports(
        self, channel: ThomsonChannel
    ) -> None:
        assert channel.truth.instrument_id == THOMSON_CHANNEL
        assert channel.inversion.instrument_id == THOMSON_CHANNEL

    def test_z_index_defaults_to_the_outermost_grid_point(
        self, channel: ThomsonChannel, reference_state: PlasmaState
    ) -> None:
        assert channel.z_index == reference_state.grid.n_points - 1

    def test_an_explicit_z_index_overrides_the_default(self, reference_state: PlasmaState) -> None:
        overridden = build_thomson_channel(
            reference_state=reference_state,
            seed=_SEED,
            registry=default_registry(),
            z_index=0,
        )

        assert overridden.z_index == 0

    def test_forward_agrees_bit_for_bit_between_truth_and_inversion(
        self, channel: ThomsonChannel, reference_state: PlasmaState
    ) -> None:
        # doc 04 §9's shared code path: `forward()` always takes the noiseless, true-
        # calibration branch regardless of how either instrument was configured, which is
        # what keeps a calibration error confined to `observe()` and out of every
        # prediction the inversion evaluates. If a future refactor let `forward()` read
        # `self._applied_state`, this test is what would catch the calibration error
        # leaking into the trial-side predictions fusion actually scores against.
        truth_prediction = channel.truth.forward(reference_state, channel.window)
        inversion_prediction = channel.inversion.forward(reference_state, channel.window)

        np.testing.assert_array_equal(truth_prediction.values, inversion_prediction.values)

    def test_with_perfect_calibration_the_truth_observation_matches_its_own_prediction(
        self, reference_state: PlasmaState
    ) -> None:
        # The doc 00 §5.1 S3 consistency discipline, at channel-builder granularity: with
        # noise off and the true calibration applied, `observe()` must reduce to exactly
        # `forward()` -- the same "both come from the same code path" contract the
        # instrument module docstring states, exercised end to end through the builder.
        perfect = build_thomson_channel(
            reference_state=reference_state,
            seed=_SEED,
            registry=default_registry(),
            noise=False,
            imperfect_calibration=False,
        )

        observation = perfect.truth.observe(reference_state, perfect.window)
        prediction = perfect.truth.forward(reference_state, perfect.window)

        np.testing.assert_allclose(observation.values, prediction.values, rtol=1e-12)

    def test_the_joint_likelihood_contains_the_channel_by_name(
        self, channel: ThomsonChannel, reference_state: PlasmaState
    ) -> None:
        joint = _joint_from(channel, reference_state)

        assert joint.names == (THOMSON_CHANNEL,)
        assert math.isfinite(joint.log_prob(reference_state))


class TestTheMeasurementVolumeIsPinned:
    """The guard against the LIF-shaped inverse crime `channels.py` documents at length.

    Unlike `LifInstrument`, `ThomsonInstrument.configure` has no per-state dynamic
    resolver today -- its own default (`_DEFAULT_Z_INDEX = -1`) is a fixed array position,
    not a physics-resolved node that moves with `theta`. The risk this guards against is
    the *calling* discipline rather than a bug already present in the instrument: relying
    on an unstated instrument default for where a manifest-level measurement is taken is
    the same class of implicit-location risk, and this test is what stops
    `build_thomson_channel` from ever being "simplified" into passing no `z_index` at all,
    or into resolving one from a trial state instead of the reference.
    """

    def test_the_configured_index_is_a_fixed_explicit_int_across_materially_different_thetas(
        self, channel: ThomsonChannel, reference_state: PlasmaState
    ) -> None:
        theta = _reference_theta()
        # Three materially different trial states -- the reference and two perturbations
        # in `n_0` and `T_e` together -- evaluated through the *same* built channel, the
        # way a MAP search would reuse one channel object across many trial thetas.
        # Chosen to stay inside the region this module's own robustness sweep found the
        # fixed reference-sized window can actually see (see this module's final report:
        # `forward` goes blind for some combinations well inside the doc 05 §2.1 prior's
        # nominal range, a separate, already-reported finding this test does not re-probe).
        trial_states = (
            reference_state,
            _state(theta.replace(n_0=theta.n_0 * 1.5, T_e=theta.T_e * 2.0)),
            _state(theta.replace(n_0=theta.n_0 / 1.5, T_e=theta.T_e / 1.5)),
        )

        # `_settings` is only ever assigned inside `configure()`, which `build_thomson_
        # channel` calls exactly once per instrument; `forward()` never calls it again.
        # Capturing the object here and asserting identity after every trial is what would
        # fail if a future change re-resolved it per state.
        truth_settings = channel.truth._settings
        inversion_settings = channel.inversion._settings
        assert truth_settings is not None
        assert truth_settings.z_index == channel.z_index
        assert inversion_settings is not None
        assert inversion_settings.z_index == channel.z_index

        for trial_state in trial_states:
            channel.inversion.forward(trial_state, channel.window)
            assert channel.truth._settings is truth_settings
            assert channel.inversion._settings is inversion_settings
            assert channel.inversion._settings.z_index == channel.z_index

    def test_the_prediction_responds_to_n_e_at_the_pinned_index_and_not_elsewhere(
        self, channel: ThomsonChannel, reference_state: PlasmaState
    ) -> None:
        # The independent physical check the guard needs: perturbing `n_e` only at the
        # pinned index must move the prediction, and perturbing it only at a different
        # index -- the kind of change a theta-dependent relocation of the measurement
        # volume would actually see -- must not. This is what would fail if the pin were
        # removed and the instrument fell back to resolving a location from the state.
        pinned = channel.z_index
        other = 0 if pinned != 0 else reference_state.grid.n_points - 1
        assert other != pinned

        n_e_field = reference_state.field("n_e")
        baseline = channel.inversion.forward(reference_state, channel.window)

        perturbed_at_pin = np.array(n_e_field.values, dtype=np.float64)
        perturbed_at_pin[pinned] *= 2.0
        state_perturbed_at_pin = _with_replaced_field(
            reference_state, name="n_e", values=perturbed_at_pin
        )
        prediction_at_pin = channel.inversion.forward(state_perturbed_at_pin, channel.window)

        perturbed_elsewhere = np.array(n_e_field.values, dtype=np.float64)
        perturbed_elsewhere[other] *= 2.0
        state_perturbed_elsewhere = _with_replaced_field(
            reference_state, name="n_e", values=perturbed_elsewhere
        )
        prediction_elsewhere = channel.inversion.forward(state_perturbed_elsewhere, channel.window)

        assert not np.array_equal(baseline.values, prediction_at_pin.values)
        np.testing.assert_array_equal(baseline.values, prediction_elsewhere.values)


def _with_replaced_field(state: PlasmaState, *, name: str, values: np.ndarray) -> PlasmaState:
    """A copy of `state` with one field's values swapped -- the minimal edit needed for
    `TestTheMeasurementVolumeIsPinned`'s independent localisation check."""
    original = state.field(name)
    fields = dict(state.fields)
    fields[name] = ScalarField(
        name=name, values=values, units=original.units, grid=original.grid, time=original.time
    )
    return PlasmaState(
        params=state.params,
        grid=state.grid,
        time=state.time,
        fields=fields,
        ion_distribution=state.ion_distribution,
        fidelity=state.fidelity,
    )


class TestWhatThomsonMeasuresThatBrightnessAloneCannot:
    """The test that decides whether connecting Thomson is worth doing at all.

    Doc 02 §7.1: the integrated scattered intensity gives `n_e` and the spectral width
    gives `T_e`, from two different physical mechanisms -- neither dependent on an
    absolute brightness calibration the way OES's single line-ratio channel is. If Thomson
    is not sensitive to both independently, and does not resolve the `n_0 sqrt(T_e)`
    degeneracy direction OES cannot, connecting it does not fix anything.
    """

    def test_the_likelihood_responds_to_electron_temperature(
        self, noiseless_channel: ThomsonChannel
    ) -> None:
        theta = _reference_theta()
        joint = _joint_from(noiseless_channel, _state(theta))

        at_truth = joint.log_prob(_state(theta))
        hotter = joint.log_prob(_state(theta.replace(T_e=theta.T_e * 1.5)))
        colder = joint.log_prob(_state(theta.replace(T_e=theta.T_e / 1.5)))

        # Measured (noise off, seed 11): delta ~ 30.9 hotter, ~50.8 colder. A wide margin
        # below both keeps this a regression check rather than a re-assertion of the exact
        # photon-statistics draw.
        assert at_truth - hotter > 10.0
        assert at_truth - colder > 10.0

    def test_the_likelihood_responds_to_bulk_density(
        self, noiseless_channel: ThomsonChannel
    ) -> None:
        theta = _reference_theta()
        joint = _joint_from(noiseless_channel, _state(theta))

        at_truth = joint.log_prob(_state(theta))
        denser = joint.log_prob(_state(theta.replace(n_0=theta.n_0 * 1.5)))
        sparser = joint.log_prob(_state(theta.replace(n_0=theta.n_0 / 1.5)))

        # Measured (noise off, seed 11): delta ~ 167.1 denser, ~18.1 sparser.
        assert at_truth - denser > 10.0
        assert at_truth - sparser > 10.0

    def test_the_likelihood_constrains_the_n0_te_degeneracy_direction(
        self, noiseless_channel: ThomsonChannel
    ) -> None:
        # Walks the ridge `n_0 -> k n_0`, `T_e -> T_e / k^2` that leaves `Gamma_i ~
        # n_0 sqrt(T_e)` exactly fixed -- the direction a single brightness-only channel
        # cannot resolve (module docstring). Thomson separates the two because the
        # integrated intensity carries `n_e` alone and the spectral width carries `T_e`
        # alone (doc 04 §4), so it should mind moving along this ridge.
        theta = _reference_theta()
        joint = _joint_from(noiseless_channel, _state(theta))

        k = 1.05
        at_truth = joint.log_prob(_state(theta))
        along_ridge = joint.log_prob(_state(theta.replace(n_0=theta.n_0 * k, T_e=theta.T_e / k**2)))

        # Measured (noise off, seed 11): delta ~ 12.3 at k = 1.05.
        assert at_truth - along_ridge > 1.0
