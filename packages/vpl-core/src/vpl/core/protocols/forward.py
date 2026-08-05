"""``ForwardSolver`` — ``F1``, and the quantity the project exists to compute.

Doc 08 §4 declares the contract; doc 03 §1 declares that four fidelity levels implement it
and that "the inverse solver never learns which one it is talking to". That promise is what
makes :class:`IonEnergyFlux` the interesting type here: L0's analytic sheath, L1's fluid
moments, L2's binned particles and L3's emulated POD coefficients all reduce to the same
object, and doc 03 §6 requires the *same functional* be applied at every level "so that
results are comparable".

## The sign convention, carried in the names

Doc 02 §2 fixes it and calls sign errors in flux quantities "common, silent and
catastrophic": the wall is at ``z = 0``, ``z`` is positive into the plasma, so an ion that
reaches the wall has ``v_z < 0``. :class:`~vpl.core.state.VelocityDistribution` already
absorbs the resulting minus sign in a named method,
``particle_flux_toward_wall_per_m2_s``, rather than leaving it inline somewhere
downstream. The fields here are named the same way for the same reason, and a reader who
finds a bare ``Gamma_E`` anywhere else in the framework should treat it as a defect.

## Why both factors are stored and not just the product

Doc 01 §1.2 decomposes ``Gamma_E = Gamma_i * <E_i>`` and says why in as many words: "This
is not merely algebra. It is the **information decomposition of the problem**: the two
factors are constrained by *different* physics and are therefore observable by *different*
instruments." ``Gamma_i`` follows from the Bohm criterion and so from ``T_e`` and ``n_s``;
``<E_i>`` follows from the sheath potential drop and the in-sheath collisionality. A type
that stored only the product would discard the split the whole diagnostic set is designed
around, and doc 05 §10 requires all three in the emitted posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vpl.core.protocols.config import SolverConfig
from vpl.core.protocols.metadata import SolverMetadata
from vpl.core.state import Fidelity, PlasmaParams, PlasmaState, Species, TimeGrid
from vpl.core.units import Q_, Quantity, magnitude_in

__all__ = [
    "CostBasis",
    "CostEstimate",
    "Device",
    "ForwardSolver",
    "IonEnergyFlux",
]

#: Units of ``Gamma_E``, doc 01 §1.1.
_ENERGY_FLUX_UNITS = "W/m**2"

#: Units of ``Gamma_i``, doc 01 §1.2.
_PARTICLE_FLUX_UNITS = "1/(m**2*s)"


def _frozen_flux(values: NDArray[np.float64], *, name: str) -> NDArray[np.float64]:
    """Validate one flux moment and return an immutable copy of it."""
    array = np.array(values, dtype=np.float64, copy=True)

    if not np.all(np.isfinite(array)):
        # An infinite flux is an arithmetic failure, not a large one. Whatever produced
        # it is wrong, and every aggregate taken afterwards inherits the error silently.
        raise ValueError(f"{name} must be finite; found nan or inf")

    array.flags.writeable = False
    return array


class Device(StrEnum):
    """Where a workload runs — the device column of doc 10 §3.

    Not decoration: doc 00 C3 records that FP64 throughput on the A4000 is 1/32 of FP32,
    so "double precision runs on CPU; the GPU is used in FP32/TF32 for sampling, autodiff
    and ray tracing only". A cost that did not say which device it was measured on could
    not be compared against the doc 10 §3.2 budget it is meant to feed.
    """

    CPU = "cpu"
    GPU = "gpu"


class CostBasis(StrEnum):
    """Whether a cost was predicted or observed — doc 10 §3.2, gate G-1.4.

    Doc 10 §3.2 flags its own GPU throughput as "an **estimate, not a measurement**" and
    gate G-1.4 (doc 11 §2) requires the measured figure to be recorded and the table
    corrected. Two values in one field is the cheapest way to make that gate checkable:
    a schedule built entirely from ``ESTIMATED`` costs after G-1.4 has passed is a
    schedule nobody re-derived.
    """

    ESTIMATED = "estimated"
    MEASURED = "measured"


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """What one solve costs — doc 10 §3, consumed by the doc 10 §6 work queue.

    Prefer the :meth:`estimated` and :meth:`measured` constructors: they make the basis
    explicit at the call site, which is the entire point of recording it.

    Attributes:
        wall_clock: Time for **one** solve. Doc 10 §3.1 costs workloads per unit and
            multiplies by a case count; :meth:`scaled` is that multiplication.
        device: Where it runs.
        basis: Predicted or observed. Never defaulted — see :class:`CostBasis`.
        source: Where the number came from: a document section for an estimate, a run
            identity for a measurement. Required, so that a figure a reviewer disputes
            can be traced to whatever produced it.
        peak_memory: Peak device memory, or ``None`` if not characterised. Doc 10 §3.3
            checks every workload against the 16 GB of doc 00 C3.
    """

    wall_clock: Quantity
    device: Device
    basis: CostBasis
    source: str
    peak_memory: Quantity | None = None

    def __post_init__(self) -> None:
        seconds = float(magnitude_in(self.wall_clock, "s"))
        if seconds <= 0.0:
            raise ValueError(f"wall_clock must be positive, got {self.wall_clock}")

        if not self.source.strip():
            raise ValueError(
                "a cost needs a source. doc 10 §3.2's own throughput figure is disputed "
                "in the document that states it, and a number with no provenance cannot "
                "be corrected at gate G-1.4 because nobody knows what it replaced."
            )

        if self.peak_memory is not None and float(magnitude_in(self.peak_memory, "byte")) <= 0.0:
            raise ValueError(f"peak_memory must be positive, got {self.peak_memory}")

    @classmethod
    def estimated(
        cls,
        *,
        wall_clock: Quantity,
        device: Device,
        source: str,
        peak_memory: Quantity | None = None,
    ) -> CostEstimate:
        """A predicted cost, to be superseded by a measurement at gate G-1.4."""
        return cls(
            wall_clock=wall_clock,
            device=device,
            basis=CostBasis.ESTIMATED,
            source=source,
            peak_memory=peak_memory,
        )

    @classmethod
    def measured(
        cls,
        *,
        wall_clock: Quantity,
        device: Device,
        source: str,
        peak_memory: Quantity | None = None,
    ) -> CostEstimate:
        """An observed cost. ``source`` should identify the run it was observed in."""
        return cls(
            wall_clock=wall_clock,
            device=device,
            basis=CostBasis.MEASURED,
            source=source,
            peak_memory=peak_memory,
        )

    @property
    def is_measured(self) -> bool:
        """Whether gate G-1.4 has been satisfied for this workload."""
        return self.basis is CostBasis.MEASURED

    @property
    def wall_clock_s(self) -> float:
        return float(magnitude_in(self.wall_clock, "s"))

    @property
    def peak_memory_bytes(self) -> float | None:
        if self.peak_memory is None:
            return None
        return float(magnitude_in(self.peak_memory, "byte"))

    def scaled(self, count: int) -> CostEstimate:
        """The campaign total for ``count`` cases — the doc 10 §3.1 arithmetic.

        ``peak_memory`` is carried over unchanged, not multiplied. Doc 10 §6 runs a
        campaign through a local process pool over a resumable serial queue, so 5 500
        cases do not need 5 500 times the VRAM; scaling it would fail the doc 10 §3.3
        feasibility check against doc 00 C3's 16 GB for no reason.

        The basis is preserved. A total built from an estimate is an estimate.
        """
        if count < 1:
            raise ValueError(f"case count must be at least one, got {count}")

        return replace(self, wall_clock=self.wall_clock * count)

    def __repr__(self) -> str:
        memory = "" if self.peak_memory is None else f", peak={self.peak_memory:.3g~P}"
        return (
            f"CostEstimate({self.wall_clock:.3g~P} on {self.device.value}, "
            f"{self.basis.value}{memory})"
        )


@dataclass(frozen=True, eq=False, slots=True)
class IonEnergyFlux:
    """``Gamma_E`` at one position, with the doc 01 §1.2 decomposition beside it.

    Values are indexed by time and only by time: the position is fixed by the
    ``ForwardSolver.flux(state, z)`` call that produced this. A steady solve gives
    zero-dimensional arrays; a time-resolved one gives ``(n_t,)``, matching the
    steady/time-dependent convention of :class:`~vpl.core.state.ScalarField`.

    Attributes:
        position: Where the doc 03 §6 functional was evaluated. ``z = 0`` is the wall.
        species: Whose flux this is. The energy moment weights by ``m_i``, so a flux
            without its species cannot be checked or converted.
        energy_flux_toward_wall_watt_per_m2: ``Gamma_E``. **Positive means energy arriving at
            the wall** (doc 02 §2). Read-only.
        particle_flux_toward_wall_per_m2_s: ``Gamma_i``, same sign convention. Read-only.
        fidelity: Which level of the doc 03 §1 hierarchy produced it. Duplicated from the
            solver deliberately: benchmark B-03 plots L1 against L2 to quantify exactly
            how wrong the fluid IEDF reconstruction is, and by the time the artifact is
            read the solver is gone.
        time: The instants the values are indexed by, or ``None`` for a steady flux.
    """

    position: Quantity
    species: Species
    energy_flux_toward_wall_watt_per_m2: NDArray[np.float64]
    particle_flux_toward_wall_per_m2_s: NDArray[np.float64]
    fidelity: Fidelity
    time: TimeGrid | None = None

    def __post_init__(self) -> None:
        # Dimensionality assertion only: any position in the domain is reachable.
        magnitude_in(self.position, "m")

        object.__setattr__(
            self,
            "energy_flux_toward_wall_watt_per_m2",
            _frozen_flux(self.energy_flux_toward_wall_watt_per_m2, name="Gamma_E"),
        )
        object.__setattr__(
            self,
            "particle_flux_toward_wall_per_m2_s",
            _frozen_flux(self.particle_flux_toward_wall_per_m2_s, name="Gamma_i"),
        )

        if (
            self.energy_flux_toward_wall_watt_per_m2.shape
            != self.particle_flux_toward_wall_per_m2_s.shape
        ):
            # The two are moments of one distribution over one weight. Different shapes
            # means they came from different states, and their ratio — <E_i>, the
            # headline number — would be meaningless.
            raise ValueError(
                f"Gamma_E and Gamma_i must have the same shape; got "
                f"{self.energy_flux_toward_wall_watt_per_m2.shape} and "
                f"{self.particle_flux_toward_wall_per_m2_s.shape}"
            )

        expected = () if self.time is None else (self.time.n_points,)
        if self.energy_flux_toward_wall_watt_per_m2.shape != expected:
            raise ValueError(
                f"flux has shape {self.energy_flux_toward_wall_watt_per_m2.shape}, expected "
                f"{expected}: the position is fixed, so a flux is indexed by time alone"
            )

    # ── the two factors of doc 01 §1.2 ──────────────────────────────────────────

    @property
    def energy_flux(self) -> Quantity:
        """``Gamma_E`` as a dimensional quantity, for module boundaries (doc 08 §5)."""
        return Q_(self.energy_flux_toward_wall_watt_per_m2, _ENERGY_FLUX_UNITS)

    @property
    def particle_flux(self) -> Quantity:
        """``Gamma_i`` as a dimensional quantity."""
        return Q_(self.particle_flux_toward_wall_per_m2_s, _PARTICLE_FLUX_UNITS)

    @property
    def mean_impact_energy_joules(self) -> NDArray[np.float64]:
        """``<E_i> = Gamma_E / Gamma_i`` — doc 01 §1.2, in joules.

        Instants where no ions arrive are ``nan``, not zero. Zero would be a claim — "the
        ions that arrived carried no energy" — and there were no ions; a NaN says so and
        propagates loudly, where a zero would quietly drag a time-average down. This is a
        deliberate departure from
        :meth:`~vpl.core.state.VelocityDistribution.mean_velocity_m_per_s`, which returns
        zero for an empty cell because zero velocity *is* the right answer there.
        """
        numerator = self.energy_flux_toward_wall_watt_per_m2
        denominator = self.particle_flux_toward_wall_per_m2_s
        # ``where=`` rather than a post-hoc mask: numpy would otherwise emit a divide
        # warning, and the test configuration turns warnings into errors.
        return np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan, dtype=np.float64),
            where=denominator != 0.0,
        )

    @property
    def mean_impact_energy(self) -> Quantity:
        """``<E_i>`` as a dimensional quantity."""
        return Q_(self.mean_impact_energy_joules, "J")

    # ── views ───────────────────────────────────────────────────────────────────

    @property
    def is_toward_wall(self) -> bool:
        """Whether energy is arriving at the wall at every instant — doc 02 §2.

        Not an invariant, and not validated as one. ``Gamma_E`` and ``Gamma_i`` weight
        ``v_z`` by different non-negative functions, so a fast outgoing population over a
        slow incoming one can genuinely give them opposite signs; and a transient
        (doc 07 B-08) can reverse either. This reports the sign rather than asserting it.
        """
        return bool(np.all(self.energy_flux_toward_wall_watt_per_m2 > 0.0))

    @property
    def is_steady(self) -> bool:
        return self.time is None

    @property
    def n_times(self) -> int:
        """How many instants this flux covers. ``1`` for a steady flux."""
        return 1 if self.time is None else self.time.n_points

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IonEnergyFlux):
            return NotImplemented
        return (
            self.species == other.species
            and self.fidelity is other.fidelity
            and self.time == other.time
            and bool(self.position == other.position)
            and np.array_equal(
                self.energy_flux_toward_wall_watt_per_m2, other.energy_flux_toward_wall_watt_per_m2
            )
            and np.array_equal(
                self.particle_flux_toward_wall_per_m2_s,
                other.particle_flux_toward_wall_per_m2_s,
            )
        )

    def __repr__(self) -> str:
        when = "steady" if self.time is None else f"{self.n_times} times"
        gamma_e = cast("float", np.mean(self.energy_flux_toward_wall_watt_per_m2))
        return (
            f"IonEnergyFlux({self.species.name} at z={self.position:.3g~P}, "
            f"{self.fidelity.name}, {when}, mean Gamma_E={gamma_e:.4g} W/m^2)"
        )


@runtime_checkable
class ForwardSolver(Protocol):
    """``F1`` — a plasma state and its ion energy flux, at one fidelity level.

    Declared exactly as doc 08 §4 declares it. Four levels implement it (doc 03 §1) and
    the inverse solver never learns which (doc 00 E1); a plugin supplies its own through
    the ``vpl.solvers`` entry-point group (doc 08 §10).

    **Checking is structural at runtime and nominal at type-check time**, and the
    difference matters. ``isinstance(plugin, ForwardSolver)`` verifies that the six method
    *names* are present — which is worth having, because it turns a broken plugin into a
    loud failure at load time instead of an ``AttributeError`` an hour into a sweep — but
    it says nothing about signatures. The signatures are mypy's job, run against the
    plugin's own source under ``--strict``. A plugin that passes ``isinstance`` and fails
    mypy is a real and expected outcome, not a contradiction.

    ``configure`` returns ``None`` and mutates, which is the one place this codebase
    departs from immutability. Doc 08 §4 declares it that way because a solver is built by
    the plugin loader (doc 08 §10) and configured by the manifest engine (doc 08 §6) at two
    different moments; the alternative — construction from configuration — would put the
    manifest schema into every plugin's ``__init__``.
    """

    def configure(self, cfg: SolverConfig) -> None:
        """Apply the ``forward:`` block of a manifest — doc 08 §6."""
        ...

    def solve(self, params: PlasmaParams, t: TimeGrid) -> PlasmaState:
        """Advance the plasma over ``t`` and return the resulting state."""
        ...

    def flux(self, state: PlasmaState, z: float) -> IonEnergyFlux:
        """Apply the doc 03 §6 flux functional at position ``z``, in metres.

        A bare float rather than a :class:`~vpl.core.units.Quantity`, as doc 08 §4 writes
        it: this is called inside the sampler's inner loop, and doc 08 §5 puts raw SI
        magnitudes there deliberately. The units contract is asserted at the boundaries
        either side.
        """
        ...

    def fidelity(self) -> Fidelity:
        """Which level of the doc 03 §1 hierarchy this is: L0, L1, L2 or L3."""
        ...

    def cost_estimate(self, cfg: SolverConfig) -> CostEstimate:
        """What one solve under ``cfg`` will cost — doc 10 §3, for the doc 10 §6 queue."""
        ...

    def metadata(self) -> SolverMetadata:
        """Name, version and citations — doc 00 C2, doc 08 §7."""
        ...
