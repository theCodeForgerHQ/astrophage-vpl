"""Observation types — doc 01 SYS-4 and the acquisition reality of doc 02 §10.

Doc 02 §10.1 is blunt that a "measurement" is not a snapshot: it is a set of observations
spread over minutes, each tagged with an acquisition window and an RF phase bin. These
tests hold the data model to that, because doc 01 SYS-4 says that without per-sample time
and error the joint inversion of doc 05 §3.2 cannot be posed at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    Measurement,
    MeasurementSet,
    Observable,
    PhaseGrid,
    TimeGrid,
    WindowMode,
)
from vpl.core.units import Q_, DimensionalityError, magnitude_in


def _phase_grid() -> PhaseGrid:
    """The doc 02 §10.3 binning: 16 bins of 4.6 ns across the 73.7 ns RF period."""
    return PhaseGrid(n_bins=16, period=Q_(73.7, "ns"))


def _gate() -> AcquisitionWindow:
    """The doc 02 §10.1 OES gate: 2 ns, on the shared latent time base."""
    return AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(2.0, "ns"))


def _measurement(
    instrument_id: str = "oes",
    *,
    values: tuple[float, ...] = (10.0, 20.0),
    uncertainty: tuple[float, ...] = (1.0, 2.0),
    units: str = "count",
    window: AcquisitionWindow | None = None,
    calibration: CalibrationState = CalibrationState.ESTIMATED,
) -> Measurement:
    return Measurement(
        instrument_id=instrument_id,
        values=np.array(values),
        uncertainty=np.array(uncertainty),
        units=units,
        window=window if window is not None else _gate(),
        calibration=calibration,
    )


class TestAbsoluteAcquisitionWindow:
    def test_reports_the_start_and_duration_it_was_given(self) -> None:
        window = AcquisitionWindow.absolute(start=Q_(1.5, "s"), duration=Q_(2.0, "ns"))

        assert magnitude_in(window.start, "s") == pytest.approx(1.5)
        assert magnitude_in(window.duration, "ns") == pytest.approx(2.0)

    def test_is_not_phase_locked(self) -> None:
        window = _gate()

        assert window.mode is WindowMode.ABSOLUTE
        assert window.is_absolute
        assert not window.is_phase_locked

    def test_stops_one_duration_after_it_starts(self) -> None:
        window = AcquisitionWindow.absolute(start=Q_(10.0, "s"), duration=Q_(2.0, "s"))

        assert magnitude_in(window.stop, "s") == pytest.approx(12.0)

    def test_a_seven_hundred_second_accumulation_is_still_one_window(self) -> None:
        # doc 02 §10.1: a Thomson point is ~7000 shots over ~700 s. Doc 05 §3.2 warns
        # that treating that and a 2 ns gate as "the same instant" is straightforwardly
        # false, so the long integral has to be representable as itself.
        window = AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(700.0, "s"))

        assert magnitude_in(window.duration, "s") == pytest.approx(700.0)
        assert window.duty_cycle == pytest.approx(1.0)

    def test_can_span_a_solver_time_grid(self) -> None:
        grid = TimeGrid.uniform(duration=Q_(73.7, "ns"), n_points=17)

        window = AcquisitionWindow.spanning(grid)

        assert magnitude_in(window.start, "ns") == pytest.approx(0.0)
        assert magnitude_in(window.duration, "ns") == pytest.approx(73.7)

    def test_exposes_seconds_for_hot_loops(self) -> None:
        window = AcquisitionWindow.absolute(start=Q_(1.0, "ms"), duration=Q_(2.0, "ns"))

        assert window.start_s == pytest.approx(1e-3)
        assert window.duration_s == pytest.approx(2e-9)


class TestPhaseLockedAcquisitionWindow:
    def test_carries_the_bin_index_it_was_accumulated_into(self) -> None:
        window = AcquisitionWindow.phase_locked(grid=_phase_grid(), bin_index=5)

        assert window.mode is WindowMode.PHASE_LOCKED
        assert window.is_phase_locked
        assert window.phase_bin == 5

    def test_reports_the_phase_at_the_centre_of_its_bin(self) -> None:
        grid = PhaseGrid(n_bins=4, period=Q_(73.7, "ns"))

        window = AcquisitionWindow.phase_locked(grid=grid, bin_index=1)

        assert window.phase_centre_rad == pytest.approx(3.0 * np.pi / 4.0)

    def test_the_gate_defaults_to_the_full_bin_width(self) -> None:
        window = AcquisitionWindow.phase_locked(grid=_phase_grid(), bin_index=0)

        assert magnitude_in(window.duration, "ns") == pytest.approx(73.7 / 16)

    def test_duty_cycle_is_the_gate_as_a_fraction_of_the_rf_period(self) -> None:
        # doc 02 §10.1 tabulates OES as "2 ns gate in 73.7 ns period => 2.7 %".
        window = AcquisitionWindow.phase_locked(grid=_phase_grid(), bin_index=0, gate=Q_(2.0, "ns"))

        assert window.duty_cycle == pytest.approx(0.027, abs=1e-3)

    def test_an_explicit_accumulation_sets_the_wall_clock_span(self) -> None:
        # doc 02 §10.1: Thomson accumulates ~700 s of wall clock into a 5 ns gate. The
        # gate is what the doc 05 §3.2 integral runs over; the 700 s is what the doc 02
        # §10.3 drift assumption has to survive, so both have to be recorded.
        window = AcquisitionWindow.phase_locked(
            grid=_phase_grid(),
            bin_index=3,
            gate=Q_(4.0, "ns"),
            accumulation=Q_(700.0, "s"),
        )

        assert magnitude_in(window.duration, "ns") == pytest.approx(4.0)
        assert magnitude_in(window.wall_clock_span, "s") == pytest.approx(700.0)
        assert magnitude_in(window.stop, "s") == pytest.approx(700.0)

    def test_rejects_a_bin_index_past_the_end_of_its_grid(self) -> None:
        with pytest.raises(ValueError, match="outside the 16-bin grid"):
            AcquisitionWindow.phase_locked(grid=_phase_grid(), bin_index=16)

    def test_rejects_a_negative_bin_index(self) -> None:
        # Negative indices would silently wrap under Python indexing, putting the
        # observation in a bin nobody chose.
        with pytest.raises(ValueError, match="outside the 16-bin grid"):
            AcquisitionWindow.phase_locked(grid=_phase_grid(), bin_index=-1)

    def test_rejects_a_gate_wider_than_the_phase_bin(self) -> None:
        # doc 02 §10.3 sizes the bins to stay "compatible with the 2 ns minimum gate".
        # A gate straddling two bins smears exactly the phase resolution it exists for.
        with pytest.raises(ValueError, match="wider than the phase bin"):
            AcquisitionWindow.phase_locked(grid=_phase_grid(), bin_index=0, gate=Q_(10.0, "ns"))

    def test_rejects_an_accumulation_shorter_than_its_own_gate(self) -> None:
        with pytest.raises(ValueError, match="shorter than"):
            AcquisitionWindow.phase_locked(
                grid=_phase_grid(),
                bin_index=0,
                gate=Q_(4.0, "ns"),
                accumulation=Q_(1.0, "ns"),
            )

    def test_rejects_a_non_positive_accumulation(self) -> None:
        with pytest.raises(ValueError, match="accumulation must be positive"):
            AcquisitionWindow.phase_locked(
                grid=_phase_grid(), bin_index=0, accumulation=Q_(0.0, "s")
            )

    def test_names_its_bin_when_printed(self) -> None:
        # The repr is what appears in a failed-assertion diff, and "AcquisitionWindow(...)"
        # with no bin would make a phase-registration bug unreadable at exactly the moment
        # it needs reading.
        window = AcquisitionWindow.phase_locked(grid=_phase_grid(), bin_index=5)

        assert "phase bin 5/16" in repr(window)
        assert "absolute" in repr(_gate())


class TestAcquisitionWindowInvariants:
    def test_rejects_a_zero_duration(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(0.0, "s"))

    def test_rejects_a_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(-1.0, "ns"))

    def test_rejects_a_duration_that_is_not_a_time(self) -> None:
        with pytest.raises(DimensionalityError):
            AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(2.0, "m"))

    def test_rejects_a_start_that_is_not_a_time(self) -> None:
        with pytest.raises(DimensionalityError):
            AcquisitionWindow.absolute(start=Q_(0.0, "m"), duration=Q_(2.0, "ns"))

    def test_allows_a_pre_trigger_start(self) -> None:
        # doc 02 §10.2: a pre-trigger baseline window is a legitimate acquisition, so
        # t = 0 carries no privileged meaning on the shared latent time base.
        window = AcquisitionWindow.absolute(start=Q_(-5.0, "ns"), duration=Q_(2.0, "ns"))

        assert window.start_s == pytest.approx(-5e-9)

    def test_rejects_a_phase_bin_without_a_grid_to_index(self) -> None:
        with pytest.raises(ValueError, match="both a phase grid and a bin index"):
            AcquisitionWindow(start=Q_(0.0, "s"), duration=Q_(2.0, "ns"), phase_bin=3)

    def test_rejects_a_phase_grid_without_a_bin(self) -> None:
        with pytest.raises(ValueError, match="both a phase grid and a bin index"):
            AcquisitionWindow(start=Q_(0.0, "s"), duration=Q_(2.0, "ns"), phase_grid=_phase_grid())

    def test_asking_an_absolute_window_for_its_phase_raises(self) -> None:
        # Returning None here would be worse: a caller that forgot to check would feed
        # a phase of zero into an accumulation and never see it.
        with pytest.raises(ValueError, match="no phase bin"):
            _ = _gate().phase_centre_rad

    def test_cannot_be_rebound_after_construction(self) -> None:
        window = _gate()

        with pytest.raises(AttributeError):
            window.duration = Q_(1.0, "s")  # type: ignore[misc]


class TestCalibrationState:
    def test_the_estimated_calibration_is_the_honest_one(self) -> None:
        # doc 04 §7.3: the pipeline works with data corrected by an *imperfect*
        # calibration, exactly as in a real experiment.
        assert not CalibrationState.ESTIMATED.is_inverse_crime

    def test_applying_the_true_calibration_is_an_inverse_crime(self) -> None:
        # doc 04 §7.3 names it: applying the true calibration "would be a form of
        # inverse crime and would understate the error". Machine-checkable, not a matter
        # of whoever is reading the code remembering it.
        assert CalibrationState.TRUE.is_inverse_crime


class TestObservable:
    def test_carries_values_units_window_and_instrument(self) -> None:
        observable = Observable(
            instrument_id="thomson",
            values=np.array([1.0, 2.0, 3.0]),
            units="count",
            window=_gate(),
        )

        assert observable.instrument_id == "thomson"
        np.testing.assert_allclose(observable.values, [1.0, 2.0, 3.0])
        assert observable.units == "count"
        assert observable.window == _gate()

    def test_exposes_its_values_as_a_dimensional_quantity(self) -> None:
        observable = Observable(
            instrument_id="interferometry",
            values=np.array([1.0, 2.0]),
            units="rad",
            window=_gate(),
        )

        np.testing.assert_allclose(magnitude_in(observable.quantity, "rad"), [1.0, 2.0])

    def test_reports_its_sample_count_and_shape(self) -> None:
        observable = Observable(
            instrument_id="oes",
            values=np.zeros((4, 8)),
            units="count",
            window=_gate(),
        )

        assert observable.n_samples == 32
        assert observable.shape == (4, 8)

    def test_rejects_an_empty_instrument_id(self) -> None:
        with pytest.raises(ValueError, match="instrument id"):
            Observable(instrument_id="  ", values=np.array([1.0]), units="count", window=_gate())

    def test_rejects_units_the_registry_cannot_parse(self) -> None:
        # An artifact whose units string is not a unit is an artifact that cannot be
        # read back (doc 08 §7), and the failure would surface at analysis time.
        with pytest.raises(ValueError, match="not a unit"):
            Observable(
                instrument_id="oes",
                values=np.array([1.0]),
                units="widgets",
                window=_gate(),
            )

    def test_rejects_non_finite_values(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Observable(
                instrument_id="oes",
                values=np.array([1.0, np.nan]),
                units="count",
                window=_gate(),
            )

    def test_rejects_an_empty_value_array(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            Observable(instrument_id="oes", values=np.array([]), units="count", window=_gate())

    def test_rejects_a_bare_scalar(self) -> None:
        # Even a single-channel photodiode reading is one *sample*, not a scalar; keeping
        # the rank uniform is what lets the likelihood iterate without a shape branch.
        with pytest.raises(ValueError, match="got a scalar"):
            Observable(instrument_id="oes", values=np.array(1.0), units="count", window=_gate())

    def test_names_its_channel_when_printed(self) -> None:
        observable = Observable(
            instrument_id="thomson", values=np.array([1.0]), units="count", window=_gate()
        )

        assert "'thomson'" in repr(observable)

    def test_copies_the_caller_array(self) -> None:
        source = np.array([1.0, 2.0])
        observable = Observable(instrument_id="oes", values=source, units="count", window=_gate())

        source[0] = 99.0

        np.testing.assert_allclose(observable.values, [1.0, 2.0])

    def test_the_underlying_array_cannot_be_written(self) -> None:
        observable = Observable(
            instrument_id="oes", values=np.array([1.0]), units="count", window=_gate()
        )

        with pytest.raises(ValueError, match="read-only"):
            observable.values[0] = 5.0

    def test_equality_compares_values_units_and_window(self) -> None:
        def build(units: str) -> Observable:
            return Observable(
                instrument_id="oes",
                values=np.array([1.0, 2.0]),
                units=units,
                window=_gate(),
            )

        assert build("count") == build("count")
        assert build("count") != build("rad")
        assert build("count") != "not an observable"


class TestMeasurement:
    def test_carries_a_per_sample_uncertainty(self) -> None:
        # doc 01 SYS-4: without per-sample error the joint fit cannot be posed.
        measurement = _measurement(values=(10.0, 20.0), uncertainty=(1.0, 2.0))

        np.testing.assert_allclose(measurement.uncertainty, [1.0, 2.0])

    def test_exposes_values_as_a_dimensional_quantity(self) -> None:
        measurement = _measurement(units="rad", values=(0.5, 1.5))

        np.testing.assert_allclose(magnitude_in(measurement.quantity, "rad"), [0.5, 1.5])

    def test_reports_its_sample_count_and_shape(self) -> None:
        measurement = _measurement(values=(1.0, 2.0), uncertainty=(0.1, 0.2))

        assert measurement.n_samples == 2
        assert measurement.shape == (2,)

    def test_names_its_channel_and_calibration_when_printed(self) -> None:
        assert "'oes'" in repr(_measurement())
        assert "estimated" in repr(_measurement())

    def test_exposes_uncertainty_as_a_dimensional_quantity(self) -> None:
        measurement = _measurement(units="rad", uncertainty=(0.1, 0.2))

        np.testing.assert_allclose(
            magnitude_in(measurement.uncertainty_quantity, "rad"), [0.1, 0.2]
        )

    def test_defaults_to_the_estimated_calibration(self) -> None:
        # doc 04 §7.3: the pipeline is never handed the true calibration by default,
        # because that is an inverse crime that understates the error.
        measurement = Measurement(
            instrument_id="oes",
            values=np.array([1.0]),
            uncertainty=np.array([0.1]),
            units="count",
            window=_gate(),
        )

        assert measurement.calibration is CalibrationState.ESTIMATED
        assert not measurement.is_inverse_crime

    def test_can_record_that_the_true_calibration_was_applied(self) -> None:
        measurement = _measurement(calibration=CalibrationState.TRUE)

        assert measurement.is_inverse_crime

    def test_rejects_an_uncertainty_of_a_different_shape(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            Measurement(
                instrument_id="oes",
                values=np.array([1.0, 2.0, 3.0]),
                uncertainty=np.array([0.1, 0.2]),
                units="count",
                window=_gate(),
            )

    def test_rejects_a_scalar_uncertainty_that_would_silently_broadcast(self) -> None:
        # A uncertainty that broadcasts is the defect that produces a confident wrong
        # posterior: every sample inherits one channel's error bar without complaint.
        with pytest.raises(ValueError, match="same shape"):
            Measurement(
                instrument_id="oes",
                values=np.array([1.0, 2.0, 3.0]),
                uncertainty=np.array(0.1),
                units="count",
                window=_gate(),
            )

    def test_rejects_a_negative_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            _measurement(values=(1.0, 2.0), uncertainty=(0.1, -0.2))

    def test_rejects_a_non_finite_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _measurement(values=(1.0, 2.0), uncertainty=(0.1, np.inf))

    def test_allows_a_zero_uncertainty(self) -> None:
        # A gated-off channel (doc 01 IF-6) legitimately reports zero; rejecting it here
        # would push the special case into every instrument.
        measurement = _measurement(values=(0.0, 0.0), uncertainty=(0.0, 0.0))

        np.testing.assert_allclose(measurement.uncertainty, [0.0, 0.0])

    def test_the_uncertainty_array_cannot_be_written(self) -> None:
        measurement = _measurement()

        with pytest.raises(ValueError, match="read-only"):
            measurement.uncertainty[0] = 5.0

    def test_copies_the_caller_uncertainty_array(self) -> None:
        source = np.array([1.0, 2.0])
        measurement = Measurement(
            instrument_id="oes",
            values=np.array([10.0, 20.0]),
            uncertainty=source,
            units="count",
            window=_gate(),
        )

        source[0] = 99.0

        np.testing.assert_allclose(measurement.uncertainty, [1.0, 2.0])

    def test_drops_to_an_observable_for_comparison_with_the_forward_model(self) -> None:
        # doc 04 §9: `observe` and `forward` come from the same code path, so the noisy
        # result must be reducible to the shape the noiseless one has.
        measurement = _measurement()

        observable = measurement.as_observable()

        assert isinstance(observable, Observable)
        assert observable.instrument_id == measurement.instrument_id
        assert observable.window == measurement.window
        np.testing.assert_allclose(observable.values, measurement.values)


class TestMeasurementSetOrdering:
    def test_iterates_in_an_order_fixed_by_content_not_by_insertion(self) -> None:
        # doc 00 E3: a reduction over a set that iterated in insertion order would not
        # be bit-for-bit reproducible across processes.
        oes = _measurement("oes")
        thomson = _measurement("thomson")

        forwards = MeasurementSet.of(oes, thomson)
        backwards = MeasurementSet.of(thomson, oes)

        assert [m.instrument_id for m in forwards] == ["oes", "thomson"]
        assert [m.instrument_id for m in backwards] == ["oes", "thomson"]

    def test_orders_one_instrument_by_acquisition_start(self) -> None:
        late = _measurement(
            "thomson",
            window=AcquisitionWindow.absolute(start=Q_(700.0, "s"), duration=Q_(5.0, "ns")),
        )
        early = _measurement(
            "thomson",
            window=AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(5.0, "ns")),
        )

        ordered = MeasurementSet.of(late, early)

        assert [m.window.start_s for m in ordered] == [0.0, 700.0]

    def test_orders_simultaneous_observations_by_phase_bin(self) -> None:
        def at_bin(index: int) -> Measurement:
            return _measurement(
                "lif",
                window=AcquisitionWindow.phase_locked(grid=_phase_grid(), bin_index=index),
            )

        ordered = MeasurementSet.of(at_bin(7), at_bin(1), at_bin(4))

        assert [m.window.phase_bin for m in ordered] == [1, 4, 7]


class TestMeasurementSetContents:
    def test_reports_how_many_records_it_holds(self) -> None:
        assert len(MeasurementSet.of(_measurement("oes"), _measurement("lif"))) == 2

    def test_counts_every_sample_not_every_record(self) -> None:
        # The doc 05 §3.2 likelihood sums over observations, not over records: one LIF
        # record is a whole 200-point frequency scan (doc 02 §10.1).
        wide = _measurement("lif", values=(1.0, 2.0, 3.0), uncertainty=(1.0, 1.0, 1.0))

        assert MeasurementSet.of(_measurement("oes"), wide).n_observations == 5

    def test_lists_its_instrument_ids_once_each_in_order(self) -> None:
        measurements = MeasurementSet.of(
            _measurement("thomson"), _measurement("oes"), _measurement("oes")
        )

        assert measurements.instrument_ids == ("oes", "thomson")

    def test_groups_records_by_instrument(self) -> None:
        grouped = MeasurementSet.of(
            _measurement("thomson"), _measurement("oes"), _measurement("oes")
        ).grouped_by_instrument()

        assert list(grouped) == ["oes", "thomson"]
        assert len(grouped["oes"]) == 2
        assert len(grouped["thomson"]) == 1

    def test_selects_the_records_of_one_channel(self) -> None:
        measurements = MeasurementSet.of(_measurement("oes"), _measurement("lif"))

        assert len(measurements.by_instrument("oes")) == 1

    def test_selecting_an_absent_channel_gives_nothing(self) -> None:
        measurements = MeasurementSet.of(_measurement("oes"))

        assert measurements.by_instrument("thomson") == ()

    def test_dropping_a_channel_leaves_the_others_untouched(self) -> None:
        # doc 02 §13 F-02: "LIF unavailable — channel removed from the likelihood".
        measurements = MeasurementSet.of(_measurement("oes"), _measurement("lif"))

        remaining = measurements.without_instrument("lif")

        assert remaining.instrument_ids == ("oes",)
        assert measurements.instrument_ids == ("lif", "oes")

    def test_can_be_empty_once_every_channel_has_been_dropped(self) -> None:
        # The degraded configurations of doc 02 §13 are reached by filtering, so an
        # empty intermediate set is legitimate and must not raise.
        empty = MeasurementSet.of(_measurement("oes")).without_instrument("oes")

        assert len(empty) == 0
        assert empty.n_observations == 0
        assert list(empty) == []

    def test_adding_a_record_returns_a_new_set(self) -> None:
        original = MeasurementSet.of(_measurement("oes"))

        extended = original.with_measurements(_measurement("lif"))

        assert len(original) == 1
        assert extended.instrument_ids == ("lif", "oes")

    def test_can_be_built_from_a_sequence(self) -> None:
        measurements = MeasurementSet.from_iterable([_measurement("thomson"), _measurement("oes")])

        assert measurements.instrument_ids == ("oes", "thomson")

    def test_equality_ignores_the_order_the_records_arrived_in(self) -> None:
        # The direct consequence of ordering by content (doc 00 E3): two runs that emitted
        # the same observations in different orders produced the same dataset.
        oes, lif = _measurement("oes"), _measurement("lif")

        assert MeasurementSet.of(oes, lif) == MeasurementSet.of(lif, oes)
        assert MeasurementSet.of(oes) != MeasurementSet.of(oes, lif)
        assert MeasurementSet.of(oes) != "not a measurement set"

    def test_summarises_its_channels_when_printed(self) -> None:
        measurements = MeasurementSet.of(_measurement("oes"), _measurement("lif"))

        assert "2 records" in repr(measurements)
        assert "lif, oes" in repr(measurements)
