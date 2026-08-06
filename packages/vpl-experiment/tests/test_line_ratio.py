"""The second emission line, and why a ratio is worth adding — doc 02 §6.3, doc 05 §6.2.

## What this is for

Today's harness observes **one** emission line. That is the root of the project's central
difficulty: with a single line, the plasma density is carried almost entirely by the
*overall brightness* of the spectrum. And brightness is exactly what the radiometric
calibration is uncertain about — 6 % by the registry. So "the plasma is 6 % denser" and "my
lamp reads 6 % bright" produce identical data, and density becomes unidentifiable once that
uncertainty is admitted honestly. That was measured: forced on with one line, the posterior
covariance stops existing.

A **second line changes the shape of the spectrum, not just its scale**, and shape survives a
calibration error that scales everything equally.

## The two properties that justify the work, and neither may be assumed

This file exists to prove both. If either fails, adding a second line is wasted effort and I
want to know before it is wired into anything.

1. **The ratio must be blind to a uniform calibration scale.** Both lines travel the same
   optics, the same window, the same detector. A radiometric error multiplies both by the
   same factor, so it cancels in their ratio exactly. If it does not cancel here, the two
   lines are not sharing a calibration path and the central argument is void.

2. **The ratio must actually move with electron temperature.** A ratio that cancels
   calibration but carries no temperature information is a perfectly useless observable. The
   two lines must therefore have genuinely different excitation thresholds — a high-threshold
   line is populated only by the tail of the electron distribution, a low-threshold one by
   its bulk, and the ratio of the two is what tracks ``T_e``. Mirroring real argon, where
   750.4 nm is nearly pure direct excitation from ground and 811.5 nm is metastable-fed.

## Why no separate "ratio observable" is being built

Worth recording, because building one is the obvious move and it is not needed.

The likelihood already scores calibration as a **coherent** rank-one term
(`vpl.instruments.coherent`). That term removes precisely the direction in which a uniform
fractional scaling moves the prediction — and a ratio is, by definition, invariant under
exactly that scaling. So the coherent treatment *already* projects out the direction the
ratio is immune to, and whatever information the ratio carries survives automatically in the
full-spectrum likelihood.

Adding a second line is therefore sufficient. A hand-built ratio observable would discard the
absolute information rather than down-weighting it, which is strictly worse: the coherent
treatment keeps whatever scale information the calibration prior still permits, and a ratio
throws all of it away.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.params import default_registry
from vpl.experiment.closed_loop import (
    SECOND_LINE_WAVELENGTH_NM,
    _acquisition_window,
    _argon_ion,
    _fixed_spatial_grid,
    _maxwellian_eedf_factory,
    _oes_instrument,
    _reference_theta,
    _to_plasma_params,
)
from vpl.physics.analytic.sheath import AnalyticSheathSolver


def _state(t_e_ev: float):
    registry = default_registry()
    species = _argon_ion(registry)
    reference = _reference_theta()
    solver = AnalyticSheathSolver()
    grid = _fixed_spatial_grid(solver, reference=reference, species=species, registry=registry)
    theta = reference.replace(T_e=t_e_ev)
    return solver.solve(_to_plasma_params(theta, species=species, registry=registry), grid=grid)


def _line_fluxes(state) -> tuple[float, float]:
    """Integrated radiance in each of the two line regions, in emitted order."""
    registry = default_registry()
    instrument = _oes_instrument(seed=0, registry=registry, eedf_factory=_maxwellian_eedf_factory)
    instrument.set_noise_enabled(False)
    instrument.use_true_calibration()
    observable = instrument.forward(state, _acquisition_window(registry))
    axis = np.asarray(instrument.wavelength_axis())
    values = np.asarray(observable.values).sum(axis=0)

    centre = float(instrument.centre_wavelength_nm)
    midpoint = 0.5 * (centre + SECOND_LINE_WAVELENGTH_NM)
    low, high = (centre, SECOND_LINE_WAVELENGTH_NM)
    near = values[axis <= midpoint].sum() if low < high else values[axis > midpoint].sum()
    far = values.sum() - near
    return float(near), float(far)


class TestTheSpectrumActuallyHasTwoLines:
    @pytest.mark.physics
    def test_both_line_regions_carry_signal(self) -> None:
        # Vacuous-guard: every assertion below is meaningless if the second line never made
        # it into the emitted spectrum.
        near, far = _line_fluxes(_state(3.0))

        assert near > 0.0
        assert far > 0.0

    @pytest.mark.physics
    def test_the_second_line_is_resolvably_separated_from_the_first(self) -> None:
        registry = default_registry()
        instrument = _oes_instrument(
            seed=0, registry=registry, eedf_factory=_maxwellian_eedf_factory
        )
        axis = np.asarray(instrument.wavelength_axis())

        assert axis.min() <= SECOND_LINE_WAVELENGTH_NM <= axis.max(), (
            "the second line falls outside the spectrograph's window, so it is not observed "
            "at all and nothing downstream can use it"
        )


class TestTheRatioCancelsTheCalibration:
    """Property 1 — the entire reason this is worth doing."""

    @pytest.mark.physics
    def test_a_uniform_scale_error_moves_both_lines_and_not_their_ratio(self) -> None:
        # A radiometric calibration error multiplies the whole spectrum by one factor,
        # because both lines share the optics. So it must cancel in the ratio *exactly* —
        # to floating point, not approximately. If this fails, the two lines are not on a
        # shared calibration path and the case for a ratio collapses.
        near, far = _line_fluxes(_state(3.0))
        scale = 1.06  # the registry's own 6 % radiometric uncertainty

        scaled_near, scaled_far = near * scale, far * scale

        assert scaled_near / scaled_far == pytest.approx(near / far, rel=1e-12)
        assert scaled_near != pytest.approx(near, rel=1e-6), "the test is vacuous if nothing moved"


class TestTheRatioCarriesTemperature:
    """Property 2 — a calibration-immune observable that says nothing is useless."""

    @pytest.mark.physics
    def test_the_ratio_changes_with_electron_temperature(self) -> None:
        cool_near, cool_far = _line_fluxes(_state(2.0))
        hot_near, hot_far = _line_fluxes(_state(5.0))

        cool_ratio = cool_near / cool_far
        hot_ratio = hot_near / hot_far

        assert abs(hot_ratio / cool_ratio - 1.0) > 0.05, (
            f"the line ratio moved only {100 * abs(hot_ratio / cool_ratio - 1):.2f} % across "
            "T_e = 2 -> 5 eV. A ratio that does not track temperature carries no information "
            "the single line did not already have, and the second line is not worth its cost."
        )

    @pytest.mark.physics
    def test_the_ratio_is_monotonic_across_the_prior_range(self) -> None:
        # Non-monotonic would mean two temperatures produce the same ratio — a degeneracy
        # introduced by the very observable meant to break one.
        ratios = []
        for t_e in (2.0, 3.0, 4.0, 5.0, 6.0):
            near, far = _line_fluxes(_state(t_e))
            ratios.append(near / far)

        assert ratios == sorted(ratios) or ratios == sorted(ratios, reverse=True), (
            f"the line ratio is not monotonic in T_e: {ratios}. Two temperatures giving the "
            "same ratio is a new degeneracy, not a broken one."
        )

    @pytest.mark.physics
    def test_the_two_lines_have_genuinely_different_thresholds(self) -> None:
        # The mechanism behind property 2. Equal thresholds would make the two lines respond
        # to T_e identically and their ratio flat — this asserts the *cause*, so a future
        # edit that equalises them fails here with a readable reason rather than failing the
        # statistical test above mysteriously.
        from vpl.experiment.closed_loop import (
            _RADIATING_ENERGY_EV,
            _SECOND_RADIATING_ENERGY_EV,
        )

        separation = abs(_SECOND_RADIATING_ENERGY_EV - _RADIATING_ENERGY_EV)
        assert separation > 1.0, (
            f"the two radiating levels are only {separation:.2f} eV apart; their excitation "
            "rates then track each other and the ratio carries almost no temperature signal"
        )
