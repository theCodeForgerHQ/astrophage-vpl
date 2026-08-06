"""``vpl.instruments.interferometry.noise`` — doc 04 §5.2, doc 02 §8.1-8.2.

Plain module-level helpers, not a shared ``conftest.py`` — see the note at the top of
``oes_system.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vpl.instruments.interferometry.noise import (
    FRINGE_JUMP_MAGNITUDE_RAD,
    independent_phase_std_rad,
    sample_fringe_jumps,
    vibration_phase_std_rad,
)
from vpl.instruments.interferometry.phase import PHASE_RESOLUTION_RAD


class TestVibrationPhaseStd:
    def test_is_positive_for_a_representative_window(self) -> None:
        assert vibration_phase_std_rad(1.0e-3) > 0.0

    def test_is_below_the_full_phase_resolution_budget(self) -> None:
        # The vibration term is only part of the doc 02 IF-P1 noise budget - see
        # TestIndependentPhaseStd for the rest of it.
        assert vibration_phase_std_rad(1.0e-3) < PHASE_RESOLUTION_RAD

    def test_a_longer_window_sees_more_of_the_1_over_f_tail(self) -> None:
        # A longer acquisition resolves lower frequencies (f_min = 1/duration), which is
        # exactly the slow-drift content a 1/f spectrum concentrates its power in. Doc 02
        # IF-G2 calls mechanical drift "the dominant systematic" for the reason that a
        # longer measurement accrues more of it, not less.
        short = vibration_phase_std_rad(1.0e-4)
        long = vibration_phase_std_rad(1.0e-1)

        assert long > short

    def test_returns_zero_when_the_window_is_too_short_to_resolve_any_vibration(self) -> None:
        # A gate far shorter than one period of the fastest modelled vibration frequency
        # cannot see it at all; the model must not extrapolate noise power from nothing.
        assert vibration_phase_std_rad(1.0e-9) == 0.0

    def test_refuses_a_nonpositive_duration(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            vibration_phase_std_rad(0.0)


class TestIndependentPhaseStd:
    def test_is_positive(self) -> None:
        assert independent_phase_std_rad() > 0.0

    def test_combines_with_the_vibration_term_to_the_calibrated_phase_resolution(self) -> None:
        # By construction (see the module docstring): at the reference window the two
        # variances sum to doc 02 IF-P1's quoted 0.1 mrad.
        reference_window_s = 1.0e-3
        total_variance = (
            vibration_phase_std_rad(reference_window_s) ** 2 + independent_phase_std_rad() ** 2
        )

        assert math.sqrt(total_variance) == pytest.approx(PHASE_RESOLUTION_RAD, rel=1.0e-9)


class TestFringeJumps:
    def test_a_certain_rate_produces_a_jump_of_exactly_two_pi_on_every_chord(self) -> None:
        rng = np.random.default_rng(0)

        jumps = sample_fringe_jumps(rng, n_chords=8, rate=1.0)

        assert jumps.shape == (8,)
        np.testing.assert_allclose(jumps, FRINGE_JUMP_MAGNITUDE_RAD)
        assert pytest.approx(2.0 * math.pi) == FRINGE_JUMP_MAGNITUDE_RAD

    def test_a_zero_rate_produces_no_jumps(self) -> None:
        rng = np.random.default_rng(0)

        jumps = sample_fringe_jumps(rng, n_chords=8, rate=0.0)

        np.testing.assert_array_equal(jumps, np.zeros(8))

    def test_refuses_a_rate_outside_zero_one(self) -> None:
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="probability"):
            sample_fringe_jumps(rng, n_chords=8, rate=1.5)
        with pytest.raises(ValueError, match="probability"):
            sample_fringe_jumps(rng, n_chords=8, rate=-0.1)

    def test_is_deterministic_for_a_seeded_generator(self) -> None:
        first = sample_fringe_jumps(np.random.default_rng(42), n_chords=8, rate=0.5)
        second = sample_fringe_jumps(np.random.default_rng(42), n_chords=8, rate=0.5)

        np.testing.assert_array_equal(first, second)
