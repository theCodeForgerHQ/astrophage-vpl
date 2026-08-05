"""Observation types — doc 01 SYS-4, doc 02 §10 and doc 04 §9.

Doc 02 §10.1 states the constraint that shapes everything here: the four channels never
measure simultaneously, so **a "measurement" is not a snapshot — it is a set of
observations distributed over minutes, each tagged with an acquisition window and an RF
phase bin**. Doc 01 SYS-4 turns that into a requirement on the data model, and gives the
reason bluntly: the inversion is a joint fit across asynchronous channels, and *without
per-sample time and error it cannot be posed*.

The likelihood of doc 05 §3.2 is written over a shared latent time base:

    log p(y | x)  =  Σ_channels Σ_observations  log p( y_{k,i} | ∫_{W_{k,i}} F_k(x(t)) dt )

`W_{k,i}` is an :class:`AcquisitionWindow`. It comes in two kinds, and doc 05 §3.2 names
both: an absolute-time integral (a 2 ns ICCD gate, a 700 s Thomson accumulation) and, for
cyclo-stationary RF operation, an integral over an RF phase bin rather than over absolute
time (doc 02 §10.3).

Two observation types, not one, because doc 04 §9 draws the line there:
:class:`Observable` is what ``Instrument.forward`` returns — the noiseless expectation the
likelihood compares against. :class:`Measurement` is what ``Instrument.observe`` returns —
the same thing with noise, a per-sample uncertainty, and an *imperfect* calibration
applied. Doc 04 §7.3 is emphatic that the pipeline works with the estimated calibration and
never the true one, so :class:`CalibrationState` records which was used and names the
alternative for what it is.

## Why the two types are not one type with an optional field

An ``Observable`` with ``uncertainty=None`` would let a noiseless prediction flow into a
likelihood term that then divides by ``None``-turned-zero, or worse, skips the term. The
type split makes ``forward`` and ``observe`` distinguishable at the signature, which is
what doc 04 §9 asks for. Their shared validation is factored into module-level helpers
rather than into a base class: dataclass inheritance with ``slots`` and defaulted fields is
fragile, and composition (``measurement.observable.values``) reads badly at every call site
in a likelihood loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from vpl.core.state.grid import PhaseGrid, TimeGrid
from vpl.core.units import Q_, UREG, Quantity, magnitude_in

__all__ = [
    "AcquisitionWindow",
    "CalibrationState",
    "InstrumentId",
    "Measurement",
    "MeasurementSet",
    "Observable",
    "WindowMode",
]

#: How an instrument identifies itself in an artifact group (doc 08 §7: "one group per
#: instrument"). A bare string rather than an enum, because doc 00 E1 requires new
#: instruments to be plugins discovered at runtime — a closed enum would need editing in
#: the core every time one is added, which is the coupling that requirement forbids.
type InstrumentId = str

#: The origin of the shared latent time base of doc 05 §3.2. A phase-locked window is
#: defined by its bin, not by when the accumulation happened to begin, so it defaults its
#: start here rather than demanding a timestamp the caller does not have.
_ORIGIN: Quantity = Q_(0.0, "s")

#: Sort position for an observation with no phase bin. Ordering is a bookkeeping concern,
#: so this is a sentinel index and not a physical quantity.
_UNBINNED = -1


def _validated_instrument_id(instrument_id: InstrumentId) -> InstrumentId:
    """Reject an identifier that cannot name an artifact group."""
    if not instrument_id.strip():
        raise ValueError("instrument id must not be empty")
    return instrument_id


def _validated_units(units: str) -> str:
    """Reject a units string the shared registry cannot parse.

    Units are stored as a string rather than folded into the values because doc 08 §7
    writes measurements to HDF5, where the units live in an attribute. Checking the string
    here rather than at read-back means the failure surfaces where the mistake was made,
    not hours later during analysis.
    """
    try:
        UREG.Unit(units)
    except Exception as exc:  # pragma: no cover - re-raised with context below
        raise ValueError(f"{units!r} is not a unit this registry knows: {exc}") from exc
    return units


def _frozen_samples(values: NDArray[np.float64], *, name: str) -> NDArray[np.float64]:
    """Validate a sample array and return an immutable copy of it.

    The copy is not defensive pedantry. The caller's buffer is theirs to reuse for the next
    frame, and an observation aliasing it would silently change under a posterior that had
    already been computed from it.

    Values are stored as ``float64`` even for the Poisson channels of doc 05 §3.1, whose
    photoelectron counts are integers: the count is exact in ``float64`` far beyond any
    achievable exposure, and one dtype keeps the likelihood free of dtype branching.
    """
    array = np.array(values, dtype=np.float64, copy=True)

    if array.ndim < 1:
        raise ValueError(f"{name} must be an array of samples, got a scalar")
    if array.size < 1:
        raise ValueError(f"{name} needs at least one sample, got an empty array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite; found nan or inf")

    array.flags.writeable = False
    return array


def _frozen_uncertainty(
    uncertainty: NDArray[np.float64], *, shape: tuple[int, ...]
) -> NDArray[np.float64]:
    """Validate a per-sample uncertainty against the values it describes.

    The shape is compared exactly rather than broadcast. An uncertainty that broadcasts is
    the defect that produces a confident wrong posterior: every sample silently inherits
    one channel's error bar, the fit tightens, and nothing in the output says so.

    Zero is allowed — a channel gated off below its detection floor (doc 01 IF-6) reports
    it legitimately, and rejecting it here would push the special case into every
    instrument. Negative is not: doc 01 SYS-4's "uncertainty" is a standard deviation.
    """
    array = np.array(uncertainty, dtype=np.float64, copy=True)

    if array.shape != shape:
        raise ValueError(
            f"uncertainty must have the same shape as the values it describes; "
            f"got {array.shape} for values of shape {shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("uncertainty must be finite; found nan or inf")
    if np.any(array < 0.0):
        raise ValueError("uncertainty is a standard deviation and cannot be negative")

    array.flags.writeable = False
    return array


class WindowMode(Enum):
    """Which of the two integrals of doc 05 §3.2 an acquisition window denotes."""

    ABSOLUTE = "absolute"
    PHASE_LOCKED = "phase_locked"


class CalibrationState(Enum):
    """Which calibration was applied to a measurement — doc 04 §7.3.

    The framework simulates the *measured* calibration, not the true one: a calibration
    measurement has its own noise, so the estimated response is biased and uncertain, and
    the analysis pipeline is handed that. Doc 04 §7.3 names the alternative — "applying the
    true calibration would be a form of inverse crime and would understate the error" — so
    the flag exists to make that machine-checkable rather than a matter of whoever is
    reading the code remembering it.

    ``TRUE`` is not forbidden: doc 07's verification work legitimately needs a run with the
    crime committed on purpose, as the reference the honest run is compared against. It is
    simply never the default.
    """

    ESTIMATED = "estimated"
    TRUE = "true"

    @property
    def is_inverse_crime(self) -> bool:
        """Whether using this calibration understates the error budget."""
        return self is CalibrationState.TRUE


@dataclass(frozen=True, slots=True)
class AcquisitionWindow:
    """The interval one observation integrates over — ``W_{k,i}`` of doc 05 §3.2.

    Both modes of doc 05 §3.2 live in one type rather than in a union, because the
    likelihood sums over windows of both kinds uniformly and every term needs
    :attr:`duration`. A union would force an ``isinstance`` branch at each of those sites;
    the cost of the single type is two fields that are ``None`` in absolute mode, which
    :meth:`__post_init__` cross-checks so the two cannot be half-set.

    Prefer the :meth:`absolute`, :meth:`phase_locked` and :meth:`spanning` constructors:
    they make the mode explicit at the call site.

    Attributes:
        start: When acquisition began, on the shared latent time base of doc 05 §3.2.
            Negative is legal — doc 02 §10.2 treats a pre-trigger baseline as a legitimate
            acquisition, so ``t = 0`` carries no privileged meaning.
        duration: The **integration support**: the gate width, or the length of a
            continuous accumulation interval. This is what the doc 05 §3.2 integral runs
            over, and it is not the same as the wall-clock span of a phase-locked
            accumulation — see :attr:`wall_clock_span`.
        phase_grid: The doc 02 §10.3 binning, for a cyclo-stationary window; ``None``
            otherwise.
        phase_bin: Index into ``phase_grid``; ``None`` otherwise.
        accumulation: Wall-clock span over which repeated cycles were summed, if that
            differs from ``duration``. Doc 02 §10.3 warns that slow drift over a long
            accumulation smears the phase-resolved result and that benchmark B-09
            quantifies the bias — which needs the real elapsed time, not the gate width.
    """

    start: Quantity
    duration: Quantity
    phase_grid: PhaseGrid | None = None
    phase_bin: int | None = None
    accumulation: Quantity | None = None

    def __post_init__(self) -> None:
        # Dimensionality assertion only: any instant on the latent time base is reachable.
        magnitude_in(self.start, "s")

        if float(magnitude_in(self.duration, "s")) <= 0.0:
            raise ValueError(f"duration must be positive, got {self.duration}")

        if (self.phase_grid is None) != (self.phase_bin is None):
            raise ValueError(
                "a phase-locked window needs both a phase grid and a bin index; "
                f"got grid={self.phase_grid!r} and bin={self.phase_bin!r}"
            )

        if self.phase_grid is not None and self.phase_bin is not None:
            self._check_phase_bin(self.phase_grid, self.phase_bin)

        if self.accumulation is not None:
            self._check_accumulation(self.accumulation)

    def _check_phase_bin(self, grid: PhaseGrid, index: int) -> None:
        """Reject a bin index the grid does not have.

        Negative indices are rejected rather than wrapped: Python would silently accept
        ``-1`` as the last bin, putting the observation in a bin nobody chose. Phase
        *values* wrap (that is what :meth:`PhaseGrid.bin_index` is for); phase *indices*
        do not.
        """
        if not 0 <= index < grid.n_bins:
            raise ValueError(f"phase bin {index} is outside the {grid.n_bins}-bin grid")

        bin_width_s = float(magnitude_in(grid.bin_width, "s"))
        if float(magnitude_in(self.duration, "s")) > bin_width_s:
            raise ValueError(
                f"gate {self.duration} is wider than the phase bin it sits in "
                f"({grid.bin_width}); doc 02 §10.3 sizes the bins to hold the gate, and a "
                "gate straddling two bins smears the phase resolution it exists for"
            )

    def _check_accumulation(self, accumulation: Quantity) -> None:
        accumulated_s = float(magnitude_in(accumulation, "s"))
        if accumulated_s <= 0.0:
            raise ValueError(f"accumulation must be positive, got {accumulation}")
        if accumulated_s < float(magnitude_in(self.duration, "s")):
            raise ValueError(
                f"accumulation {accumulation} is shorter than the {self.duration} gate it "
                "accumulates"
            )

    # ── constructors ────────────────────────────────────────────────────────────

    @classmethod
    def absolute(cls, *, start: Quantity, duration: Quantity) -> AcquisitionWindow:
        """A window on absolute time: a gate, or a continuous accumulation interval.

        Doc 02 §10.1 spans nine orders of magnitude here — a 2 ns OES gate and a 700 s
        Thomson point are both this kind of window.
        """
        return cls(start=start, duration=duration)

    @classmethod
    def phase_locked(
        cls,
        *,
        grid: PhaseGrid,
        bin_index: int,
        gate: Quantity | None = None,
        start: Quantity = _ORIGIN,
        accumulation: Quantity | None = None,
    ) -> AcquisitionWindow:
        """A cyclo-stationary window: one RF phase bin — doc 02 §10.3.

        Steady-state RF operation is treated as periodic in phase, so observations at the
        same phase across different cycles may be accumulated. Doc 02 §10.3 notes this is
        "what makes Thomson and LIF possible at all in the RF regime".

        Args:
            grid: The phase binning being used.
            bin_index: Which bin this observation was accumulated into.
            gate: The detector gate width. Defaults to the full bin width, which is the
                widest gate that does not straddle bins.
            start: When the accumulation began, for provenance and for the drift model.
            accumulation: Wall-clock span of the accumulation, if it matters.
        """
        return cls(
            start=start,
            duration=gate if gate is not None else grid.bin_width,
            phase_grid=grid,
            phase_bin=bin_index,
            accumulation=accumulation,
        )

    @classmethod
    def spanning(cls, grid: TimeGrid) -> AcquisitionWindow:
        """An absolute window covering the whole extent of a solver time grid.

        The common case where an instrument integrates over exactly the interval a solver
        produced, with no separate gate.
        """
        return cls.absolute(start=Q_(float(grid.t_s[0]), "s"), duration=grid.duration)

    # ── views ───────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> WindowMode:
        return WindowMode.PHASE_LOCKED if self.phase_grid is not None else WindowMode.ABSOLUTE

    @property
    def is_phase_locked(self) -> bool:
        return self.phase_grid is not None

    @property
    def is_absolute(self) -> bool:
        return self.phase_grid is None

    @property
    def phase_centre_rad(self) -> float:
        """Phase at the centre of this window's bin, in radians.

        Raises rather than returning ``None`` for an absolute window. A caller that forgot
        to check would feed a phase of zero into an accumulation and never see it — the
        class of silent error doc 02 §2 calls "common, silent and catastrophic".
        """
        if self.phase_grid is None or self.phase_bin is None:
            raise ValueError("an absolute window has no phase bin, so it has no phase")
        return float(self.phase_grid.centres_rad[self.phase_bin])

    @property
    def wall_clock_span(self) -> Quantity:
        """How much real time elapsed acquiring this observation.

        Distinct from :attr:`duration`, which is the integration support. For a
        phase-locked window with no explicit ``accumulation`` this is one RF period: the
        gate opens once per cycle, so a single period is the span the gate is a fraction
        of, and that is the fraction doc 02 §10.1 tabulates as the duty cycle.
        """
        if self.accumulation is not None:
            return self.accumulation
        if self.phase_grid is not None:
            return self.phase_grid.period
        return self.duration

    @property
    def stop(self) -> Quantity:
        """When acquisition finished, on the shared latent time base."""
        return Q_(self.start_s + float(magnitude_in(self.wall_clock_span, "s")), "s")

    @property
    def duty_cycle(self) -> float:
        """Integration support as a fraction of elapsed time — doc 02 §10.1.

        For the OES channel this reproduces the tabulated "2 ns gate in 73.7 ns period
        => 2.7 %". It is the factor by which a phase-locked channel's photon budget is
        reduced relative to a continuous one, so it belongs with the window rather than
        being recomputed wherever an SNR estimate is needed.
        """
        return self.duration_s / float(magnitude_in(self.wall_clock_span, "s"))

    # ── SI magnitudes for hot loops (doc 08 §5) ─────────────────────────────────

    @property
    def start_s(self) -> float:
        return float(magnitude_in(self.start, "s"))

    @property
    def duration_s(self) -> float:
        return float(magnitude_in(self.duration, "s"))

    def __repr__(self) -> str:
        if self.phase_grid is not None:
            return (
                f"AcquisitionWindow(phase bin {self.phase_bin}/{self.phase_grid.n_bins}, "
                f"gate={self.duration:.3g~P})"
            )
        return (
            f"AcquisitionWindow(absolute, start={self.start:.4g~P}, duration={self.duration:.3g~P})"
        )


@dataclass(frozen=True, eq=False, slots=True)
class Observable:
    """The noiseless prediction returned by ``Instrument.forward`` — doc 04 §9.

    This is the right-hand side of the doc 05 §3.2 likelihood: the expectation
    ``∫_{W} F_k(x(t)) dt`` that an observation is compared against. It carries no
    uncertainty because it has none — the statistics belong to the observation, not to the
    prediction.

    Attributes:
        instrument_id: Which channel produced this. Names the HDF5 group of doc 08 §7.
        values: Predicted samples. Read-only; see :func:`_frozen_samples`.
        units: Units of ``values``, as a registry-parseable string.
        window: The acquisition window the prediction was integrated over.
    """

    instrument_id: InstrumentId
    values: NDArray[np.float64]
    units: str
    window: AcquisitionWindow

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _validated_instrument_id(self.instrument_id))
        object.__setattr__(self, "units", _validated_units(self.units))
        object.__setattr__(self, "values", _frozen_samples(self.values, name="values"))

    @property
    def quantity(self) -> Quantity:
        """Values as a dimensional quantity, for module boundaries (doc 08 §5)."""
        return Q_(self.values, self.units)

    @property
    def n_samples(self) -> int:
        return int(self.values.size)

    @property
    def shape(self) -> tuple[int, ...]:
        return cast("tuple[int, ...]", self.values.shape)

    def __eq__(self, other: object) -> bool:
        """Compare the stored representation, not the physical value.

        Two observables differing only by a unit prefix are *not* equal here. Making them
        equal would mean converting on every comparison, and an artifact round-trip that
        silently changed units would then pass its own regression test.
        """
        if not isinstance(other, Observable):
            return NotImplemented
        return (
            self.instrument_id == other.instrument_id
            and self.units == other.units
            and self.window == other.window
            and np.array_equal(self.values, other.values)
        )

    def __repr__(self) -> str:
        return (
            f"Observable({self.instrument_id!r}, n={self.n_samples}, "
            f"units={self.units!r}, {self.window!r})"
        )


@dataclass(frozen=True, eq=False, slots=True)
class Measurement:
    """A noisy observation returned by ``Instrument.observe`` — doc 04 §9, doc 01 SYS-4.

    Everything an :class:`Observable` carries, plus the two things doc 01 SYS-4 says the
    inversion cannot be posed without: a per-sample uncertainty, and — through
    :attr:`window` — a timestamp and phase bin. The third addition is
    :attr:`calibration`, which records whether doc 04 §7.3 was honoured.

    Attributes:
        instrument_id: Which channel produced this.
        values: Observed samples, calibration already applied. Read-only.
        uncertainty: Per-sample standard deviation, in the same units and of exactly the
            same shape as ``values``. Doc 05 §3.1 gives the channels genuinely different
            statistics — Poisson for Thomson, Gaussian for interferometry — so this is the
            first moment of the noise, not the whole noise model; the likelihood family
            belongs to the channel.
        units: Units of ``values`` and ``uncertainty``.
        window: When and over what interval this was acquired.
        calibration: Which calibration produced ``values``. Defaults to
            :attr:`CalibrationState.ESTIMATED`, because doc 04 §7.3 requires the pipeline
            to work with the imperfect calibration a real experiment would have.
    """

    instrument_id: InstrumentId
    values: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    units: str
    window: AcquisitionWindow
    calibration: CalibrationState = CalibrationState.ESTIMATED

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", _validated_instrument_id(self.instrument_id))
        object.__setattr__(self, "units", _validated_units(self.units))
        object.__setattr__(self, "values", _frozen_samples(self.values, name="values"))
        object.__setattr__(
            self, "uncertainty", _frozen_uncertainty(self.uncertainty, shape=self.values.shape)
        )

    @property
    def quantity(self) -> Quantity:
        return Q_(self.values, self.units)

    @property
    def uncertainty_quantity(self) -> Quantity:
        return Q_(self.uncertainty, self.units)

    @property
    def n_samples(self) -> int:
        return int(self.values.size)

    @property
    def shape(self) -> tuple[int, ...]:
        return cast("tuple[int, ...]", self.values.shape)

    @property
    def is_inverse_crime(self) -> bool:
        """Whether this measurement was corrected with the true calibration — doc 04 §7.3."""
        return self.calibration.is_inverse_crime

    def as_observable(self) -> Observable:
        """Drop the noise metadata, leaving the shape ``Instrument.forward`` returns.

        Doc 04 §9 requires ``observe`` and ``forward`` to come from the same code path so
        they cannot drift apart. This is the projection that lets the two be compared
        directly in the tests that hold them to it.
        """
        return Observable(
            instrument_id=self.instrument_id,
            values=self.values,
            units=self.units,
            window=self.window,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Measurement):
            return NotImplemented
        return (
            self.instrument_id == other.instrument_id
            and self.units == other.units
            and self.window == other.window
            and self.calibration is other.calibration
            and np.array_equal(self.values, other.values)
            and np.array_equal(self.uncertainty, other.uncertainty)
        )

    def __repr__(self) -> str:
        return (
            f"Measurement({self.instrument_id!r}, n={self.n_samples}, "
            f"units={self.units!r}, {self.window!r}, {self.calibration.value})"
        )


def _ordering_key(measurement: Measurement) -> tuple[InstrumentId, float, int]:
    """Total order over observations, derived entirely from their content.

    doc 00 E3 requires bit-for-bit reproducibility. A set that iterated in insertion order
    would make any floating-point reduction over it — a summed log-likelihood, most
    obviously — depend on how a manifest happened to be walked, and floating-point addition
    is not associative. Sorting on ``(channel, when, phase bin)`` is also the order a
    reviewer reads the acquisition in.

    Exact ties are left in the order the caller supplied. That order is itself
    deterministic, because a :class:`MeasurementSet` is built from an ordered sequence and
    never from a hash-ordered container.
    """
    window = measurement.window
    phase = window.phase_bin if window.phase_bin is not None else _UNBINNED
    return (measurement.instrument_id, window.start_s, phase)


@dataclass(frozen=True, eq=False, slots=True)
class MeasurementSet:
    """Observations from several channels at different times — the doc 05 §3.2 sum.

    This is what a "measurement" of the sheath actually is, per doc 02 §10.1: not a
    snapshot but a collection spread over minutes. The outer sum of the doc 05 §3.2
    likelihood runs over channels and the inner one over that channel's observations, which
    is what :meth:`grouped_by_instrument` exists to serve.

    Attributes:
        measurements: The observations, re-sorted at construction into the deterministic
            order of :func:`_ordering_key`. Read-only.
    """

    measurements: tuple[Measurement, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "measurements", tuple(sorted(self.measurements, key=_ordering_key))
        )

    @classmethod
    def of(cls, *measurements: Measurement) -> MeasurementSet:
        """Build from positional observations, for call sites that have them to hand."""
        return cls(measurements=measurements)

    @classmethod
    def from_iterable(cls, measurements: Iterable[Measurement]) -> MeasurementSet:
        """Build from any ordered iterable.

        Deliberately named for what it takes. An unordered container passed here still
        produces a deterministic set — sorting sees to that — but exact ties would resolve
        by the container's iteration order, so the name is a reminder to pass a sequence.
        """
        return cls(measurements=tuple(measurements))

    # ── the doc 05 §3.2 sum ─────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Measurement]:
        return iter(self.measurements)

    def __len__(self) -> int:
        """How many observation *records* this holds. See :attr:`n_observations`."""
        return len(self.measurements)

    @property
    def n_observations(self) -> int:
        """Total number of individual samples across every record.

        Not the same as ``len(self)``, and it is this count the doc 05 §3.2 likelihood sums
        over: one LIF record is an entire ~200-point frequency scan (doc 02 §10.1), and
        reporting it as a single observation would misstate the information content of the
        dataset by more than two orders of magnitude.
        """
        return sum(measurement.n_samples for measurement in self.measurements)

    @property
    def instrument_ids(self) -> tuple[InstrumentId, ...]:
        """Every channel present, once each, in deterministic order."""
        # The records are already sorted by instrument id, so first-seen order is sorted
        # order; `dict.fromkeys` deduplicates without a set, whose iteration order would
        # reintroduce exactly the non-determinism doc 00 E3 forbids.
        return tuple(dict.fromkeys(m.instrument_id for m in self.measurements))

    def by_instrument(self, instrument_id: InstrumentId) -> tuple[Measurement, ...]:
        """Records from one channel, in order. Empty if the channel is absent."""
        return tuple(m for m in self.measurements if m.instrument_id == instrument_id)

    def grouped_by_instrument(self) -> dict[InstrumentId, tuple[Measurement, ...]]:
        """Records keyed by channel — the outer sum of doc 05 §3.2.

        The returned mapping iterates in sorted-instrument order, because it is built in
        that order and Python dictionaries preserve insertion order.
        """
        return {name: self.by_instrument(name) for name in self.instrument_ids}

    # ── degraded configurations (doc 02 §13) ────────────────────────────────────

    def without_instrument(self, instrument_id: InstrumentId) -> MeasurementSet:
        """A new set with one channel removed — doc 02 §13 F-01 and F-02.

        An empty result is legitimate and does not raise: the robustness matrix is explored
        by filtering, and the caller that assembles the likelihood is where "no data" has
        to be handled.
        """
        return MeasurementSet(
            measurements=tuple(m for m in self.measurements if m.instrument_id != instrument_id)
        )

    def with_measurements(self, *measurements: Measurement) -> MeasurementSet:
        """A new set with further observations merged in and re-sorted."""
        return MeasurementSet(measurements=self.measurements + measurements)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MeasurementSet):
            return NotImplemented
        return self.measurements == other.measurements

    def __repr__(self) -> str:
        channels = ", ".join(self.instrument_ids)
        return (
            f"MeasurementSet({len(self)} records, {self.n_observations} observations, "
            f"channels=[{channels}])"
        )
