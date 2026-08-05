"""Monte-Carlo collisions — doc 03 §4.2 step 5, doc 03 §4.5.

doc 03 §4.5 specifies the null-collision method and gives its reason:

    The **null-collision method** is used so that the collision test is a single uniform
    random draw per particle per step regardless of the number of processes — standard,
    and essential for performance.

That is implemented literally, and the "single draw" is single including the choice of
*which* process fired. The trick is that a uniform ``R`` conditioned on ``R < P_null`` has
``R / P_null`` uniform on ``[0, 1)`` exactly, so one variate does both jobs with no
approximation. The post-collision kinematics — scattering angles, target velocities — draw
separately; they are not the collision test.

## Why a ceiling frequency, and what happens above it

The method pretends the total collision frequency is a constant ``nu_max`` everywhere,
collides a fixed fraction ``1 - exp(-nu_max dt)`` of the particles, and then rejects the
fraction ``1 - nu(E)/nu_max`` of those as *null* collisions that do nothing. The rejection
restores the real energy dependence, so ``nu_max`` must genuinely bound ``nu(E)``.

If it does not, nothing complains. A particle whose real ``nu`` exceeds ``nu_max`` collides
at ``nu_max`` instead, the run finishes, and it reports an IEDF that is too collisionless by
an amount nothing in the output advertises — the failure shape doc 03 §4.3 describes for
grid heating. So :class:`NullCollisionSampler` takes an explicit energy ceiling, computes
``nu_max`` as the **exact** maximum of ``n_gas sigma_tot(E) v(E)`` over ``[0, ceiling]``, and
raises :class:`EnergyCeilingExceededError` above it. Exact rather than scanned because
``sigma`` is piecewise linear and ``sigma(E) sqrt(E)`` can be stationary *inside* a tabulated
interval: a scan over the table's own nodes returns a bound that is too small precisely where
the cross section falls fastest.

## Charge exchange

doc 03 §4.5: "Charge exchange is the single most important collision for this project." It
is implemented as what it physically is — an identity swap. The ion leaves with the velocity
of the neutral it struck and the neutral leaves with the ion's, so the pair's momentum and
energy are unchanged to the bit. The surviving ion is therefore cold *and* correlated with
its partner; an implementation that drew a fresh thermal velocity instead would produce an
identical-looking IEDF while violating momentum conservation in every event.

"At the local potential" needs no code. The new slow ion sits where the collision happened
and re-accelerates through whatever potential drop remains between there and the wall. That
is the mechanism that fills the low-energy IEDF and makes ``<E_i>`` differ from ``e V_w``;
the module's only job is to make the ion genuinely slow and genuinely there.

## Where this runs, and why it is NumPy

The push is JAX and stays JAX. This step is not, and cannot usefully be: ionisation makes
the output shape depend on the data, and doc 10 §5's reproducibility contract is expressed
through NumPy's per-stream :class:`~numpy.random.Generator`. It converts at its boundary and
returns JAX arrays, so the result feeds straight back into
:func:`~vpl.physics.kinetic.push.push_leapfrog` at an ``O(n)`` cost small beside the field
solve.

## Simplifications, and what each one costs

Stated here rather than discovered, per doc 00 C4.

1. **The background gas is a fixed Maxwellian reservoir**, neither depleted by ionisation
   nor heated by the collisions it absorbs. At doc 01 §2.1's RP-1 the neutral density is
   1.6e20 m^-3 against a plasma density of 1e17 m^-3, so the ionisation degree is 6e-4 and
   depletion over a 737 ns run is bounded by it; gas heating is doc 03 §8 A4.
2. **Electrons see a stationary target.** The neglected term is
   ``v_th,gas / v_e = sqrt((m_e T_g)/(M T_e)) = 3.4e-4`` at RP-1, four orders below every
   other error here.
3. **Electron elastic scattering is isotropic, sampled from the momentum-transfer cross
   section.** For isotropic scattering ``sigma_m == sigma_total``, so the momentum-transfer
   *rate* — which sets electron mobility and therefore doc 03 §3.2's transport coefficients
   — is exact by construction, while the *number* of elastic events is overestimated by
   ``sigma_total/sigma_m`` (≲2 for argon over 1-100 eV). An elastic electron-argon collision
   moves at most ``4 m_e/M = 5.5e-5`` of the electron's energy, so the EEDF consequence is
   bounded by that product. Vahedi-Surendra anisotropic sampling is a one-function change.
4. **Ionisation shares the excess energy equally.** The measured differential
   (Opal-Peterson, ``B ~ 10 eV`` for argon) favours low secondary energies. Energy is
   conserved exactly either way, so the error is in the secondary *spectrum*; at
   ``T_e = 3 eV`` the rate is dominated by electrons within a few eV of the 15.76 eV
   threshold, where the excess to share is small and the two nearly coincide.
5. **Single ionisation only, and no Coulomb collisions.** Ar²⁺ is negligible at these
   energies, and the electron-ion Coulomb frequency is orders below the electron-neutral
   one at a 6e-4 ionisation degree.
6. **The per-step Bernoulli test is first order in ``dt``.** The real-collision probability
   is ``(1 - exp(-nu_max dt)) nu/nu_max = nu dt (1 - nu_max dt/2 + ...)``, so a run
   under-collides by ``nu_max dt / 2`` — 6e-5 at doc 03 §4.3's 11.2 ps timestep and RP-1's
   ``nu_max ~ 1e8 s^-1``, far below the ``N_ppc = 1000`` noise floor. A bias rather than
   noise, so it is quoted rather than averaged away.

Secondary electron emission — the last row of doc 03 §4.5's table — is a *surface* process
and belongs to the boundary module, not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray

from vpl.core.constants import BOLTZMANN, ELEMENTARY_CHARGE
from vpl.core.random import Stream, generator
from vpl.core.units import magnitude_in
from vpl.physics.atomic.interpolation import (
    DEFAULT_ABOVE,
    DEFAULT_BELOW,
    ExtrapolationPolicy,
    interpolate_cross_section,
)
from vpl.physics.atomic.lxcat import CrossSection, ProcessType
from vpl.physics.kinetic.precision import enable_double_precision
from vpl.physics.kinetic.push import Species

enable_double_precision()

__all__ = [
    "BackgroundGas",
    "CollisionKind",
    "CollisionOutcome",
    "EnergyCeilingExceededError",
    "NullCollisionSampler",
    "Process",
    "collision_generator",
    "neutral_density_m3",
]

type FloatArray = NDArray[np.float64]

_E: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_K_B: Final[float] = float(magnitude_in(BOLTZMANN, "J/K"))

#: Sentinel in :attr:`CollisionOutcome.process_index` for a particle that did not undergo a
#: real collision — either it was never a candidate, or its candidacy was a null collision.
#: The two are deliberately indistinguishable downstream: a null collision *is* nothing
#: happening, and a code that treated it as an event would collide at ``nu_max``.
NO_COLLISION: Final[int] = -1

#: How far the summed per-process frequency may exceed ``nu_max`` before the sampler
#: refuses to continue. The bound is exact in exact arithmetic — ``nu_max`` is the maximum
#: of the same piecewise-linear function that is evaluated per particle — so anything
#: beyond round-off means the two disagree about the cross section and the rejection step
#: would silently truncate.
FREQUENCY_BOUND_TOLERANCE: Final[float] = 1e-9

#: Where ``d/dE [(a + bE) sqrt(E)] = 0`` puts the stationary point: ``E* = -a / (3b)``.
_STATIONARY_POINT_FACTOR: Final[float] = 3.0

#: Relative slack on the energy ceiling. A particle launched at exactly the ceiling energy
#: round-trips through ``v = sqrt(2E/m)`` and back with a relative error of order 1e-16, and
#: refusing that would make the ceiling exclusive by accident. Orders tighter than any
#: physical excess, so a particle genuinely off the end of the model is still caught.
CEILING_TOLERANCE: Final[float] = 1e-9


class EnergyCeilingExceededError(RuntimeError):
    """A particle reached an energy above the one ``nu_max`` was computed for.

    A distinct type, and a ``RuntimeError`` rather than a ``ValueError``, for the same
    reason :class:`~vpl.physics.kinetic.constraints.StabilityViolationError` is: an
    ensemble driver needs to tell "this configuration ran off the end of its collision
    model" apart from "this configuration is malformed".
    """


class CollisionKind(StrEnum):
    """What a process *does*, as against what it is called in an LXCat file.

    :class:`~vpl.physics.atomic.lxcat.ProcessType` names the block in the export;
    ``ELASTIC`` there covers both the electron and the ion channel, and the kinematics are
    completely different. This enum is the kinematic identity, and a manifest names it —
    doc 08 §1 principle 4.
    """

    #: ``e + Ar -> e + Ar``. Isotropic, ``2 m/M`` energy loss.
    ELECTRON_ELASTIC = "electron_elastic"
    #: ``e + Ar -> e + Ar*``. Threshold energy removed, direction randomised.
    ELECTRON_EXCITATION = "electron_excitation"
    #: ``e + Ar -> 2e + Ar+``. Emits one electron and one ion.
    ELECTRON_IONISATION = "electron_ionisation"
    #: ``Ar+ + Ar -> Ar+ + Ar``. Isotropic in the centre-of-mass frame.
    ION_ELASTIC = "ion_elastic"
    #: ``Ar+ + Ar -> Ar + Ar+``. doc 03 §4.5's most consequential channel.
    ION_CHARGE_EXCHANGE = "ion_charge_exchange"

    @property
    def is_electron_process(self) -> bool:
        """Whether the projectile is an electron.

        This decides whether the target is sampled from the gas Maxwellian or taken at
        rest — simplification 2 of the module docstring.
        """
        return self in _ELECTRON_KINDS


_ELECTRON_KINDS: Final[frozenset[CollisionKind]] = frozenset(
    {
        CollisionKind.ELECTRON_ELASTIC,
        CollisionKind.ELECTRON_EXCITATION,
        CollisionKind.ELECTRON_IONISATION,
    }
)

#: Which LXCat blocks may back which kinematics. Checked, because the one genuinely
#: dangerous confusion here — an ion elastic section driving electron kinematics — produces
#: a run rather than an error.
_COMPATIBLE_SECTIONS: Final[Mapping[CollisionKind, frozenset[ProcessType]]] = MappingProxyType(
    {
        CollisionKind.ELECTRON_ELASTIC: frozenset({ProcessType.ELASTIC, ProcessType.EFFECTIVE}),
        CollisionKind.ELECTRON_EXCITATION: frozenset({ProcessType.EXCITATION}),
        CollisionKind.ELECTRON_IONISATION: frozenset({ProcessType.IONIZATION}),
        CollisionKind.ION_ELASTIC: frozenset({ProcessType.ELASTIC, ProcessType.EFFECTIVE}),
        CollisionKind.ION_CHARGE_EXCHANGE: frozenset({ProcessType.CHARGE_EXCHANGE}),
    }
)


def collision_generator(root_seed: int) -> np.random.Generator:
    """The collision stream's generator — doc 10 §5.

    A named function rather than a call to :func:`vpl.core.random.generator` at each site,
    so that :attr:`~vpl.core.random.Stream.COLLISIONS` cannot be mistyped into some other
    stream: the run would still finish and still be reproducible, and the ablation matrix
    of doc 07 §5.2 would quietly be measuring the random number generator.
    """
    return generator(root_seed, Stream.COLLISIONS)


def neutral_density_m3(*, pressure_pa: float, temperature_k: float) -> float:
    """``n = p / (k_B T)`` — the reservoir density from doc 01 §2.1's pressure and ``T_g``.

    At RP-1, 5 mTorr and 300 K give 1.61e20 m^-3: three orders above the plasma density,
    which is what makes the fixed-reservoir assumption of simplification 1 defensible.
    """
    if pressure_pa <= 0.0:
        raise ValueError(f"pressure must be positive, got {pressure_pa}")
    if temperature_k <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature_k}")
    return pressure_pa / (_K_B * temperature_k)


@dataclass(frozen=True, slots=True)
class BackgroundGas:
    """The neutral reservoir the projectiles collide with.

    Attributes:
        mass_kg: Mass of one neutral. Equal to the ion mass for the symmetric argon case,
            and kept separate anyway so the centre-of-mass kinematics stay general.
        density_m3: Number density, from :func:`neutral_density_m3` or a manifest.
        temperature_k: Gas temperature, in kelvin because that is the unit doc 01 §2.1
            registers ``RP1.T_g`` in.
    """

    mass_kg: float
    density_m3: float
    temperature_k: float

    def __post_init__(self) -> None:
        if not self.mass_kg > 0.0:
            raise ValueError(f"neutral mass must be positive, got {self.mass_kg}")
        if not self.density_m3 > 0.0:
            raise ValueError(f"neutral density must be positive, got {self.density_m3}")
        if not self.temperature_k > 0.0:
            raise ValueError(f"gas temperature must be positive, got {self.temperature_k}")

    @property
    def thermal_speed_m_per_s(self) -> float:
        """``sqrt(k_B T / m)`` — the width of *one* Cartesian velocity component.

        Not the mean speed and not the most probable speed. This is the standard deviation
        the target velocities are drawn with, and quoting it as anything else is how a
        Maxwellian acquires a stray ``sqrt(2)``.
        """
        return math.sqrt(_K_B * self.temperature_k / self.mass_kg)


@dataclass(frozen=True, slots=True)
class Process:
    """One collision channel: a tabulated cross section plus what to do when it fires.

    Attributes:
        kind: The kinematics.
        section: The tabulated ``sigma(E)``, as parsed by
            :func:`~vpl.physics.atomic.lxcat.parse_lxcat`.
    """

    kind: CollisionKind
    section: CrossSection

    def __post_init__(self) -> None:
        allowed = _COMPATIBLE_SECTIONS[self.kind]
        if self.section.process not in allowed:
            raise ValueError(
                f"{self.kind} cannot be driven by a {self.section.process} cross section; "
                f"expected one of {', '.join(sorted(p.value for p in allowed))}. The "
                f"kinematics and the table would disagree about what the collision is, and "
                f"the run would finish."
            )

    @property
    def threshold_ev(self) -> float | None:
        """The inelastic threshold, or ``None`` where the process has none."""
        return self.section.threshold_ev

    @property
    def label(self) -> str:
        """The reaction as published — the channel's identity in a report."""
        return self.section.reaction


@dataclass(frozen=True, slots=True)
class CollisionOutcome:
    """What one call to :meth:`NullCollisionSampler.collide` did.

    Attributes:
        velocities_m_per_s: Post-collision velocities, shape ``(n, 3)``. Particles that did
            not collide are returned unchanged.
        process_index: Which process fired for each particle, or :data:`NO_COLLISION`.
        target_velocities_m_per_s: The neutral each particle actually struck, shape
            ``(n, 3)``, zero where nothing happened. Carried because the conservation
            statement for charge exchange and for ion elastic scattering is about the
            *pair*, and a caller — or a test — that cannot see the partner cannot check it.
        new_electron_positions_m: Where each ionisation-born electron appeared.
        new_electron_velocities_m_per_s: Its velocity, shape ``(k, 3)``.
        new_ion_positions_m: Where each ionisation-born ion appeared — the same places.
        new_ion_velocities_m_per_s: Its velocity, drawn from the gas Maxwellian, because
            the ion is the struck neutral and inherits its motion.
        counts_by_process: Events per process, aligned with
            :attr:`NullCollisionSampler.processes`. The diagnostic doc 07 needs to show a
            collision model was actually exercised rather than merely configured.
    """

    velocities_m_per_s: Array
    process_index: Array
    target_velocities_m_per_s: Array
    new_electron_positions_m: Array
    new_electron_velocities_m_per_s: Array
    new_ion_positions_m: Array
    new_ion_velocities_m_per_s: Array
    counts_by_process: tuple[int, ...]

    @property
    def n_collisions(self) -> int:
        """Real collisions this step. Null collisions are not events and are not counted."""
        return sum(self.counts_by_process)


@dataclass(frozen=True, eq=False)
class NullCollisionSampler:
    """doc 03 §4.5's null-collision method, for one species.

    One sampler serves one species: the kinematics of an electron hitting an argon atom and
    of an argon ion hitting one share no formula, and a sampler that held both would have
    to branch on the projectile inside the hot loop for no benefit.

    Attributes:
        species: The projectile. Supplies the mass; the charge is the push's business.
        gas: The neutral reservoir.
        processes: The channels, in the order :attr:`CollisionOutcome.counts_by_process`
            reports them.
        energy_ceiling_ev: The highest energy the run may reach. ``nu_max`` is the exact
            maximum of the total collision frequency over ``[0, energy_ceiling_ev]``, and a
            particle above it is a :class:`EnergyCeilingExceededError`. Set it from the
            applied bias — an ion falling through the whole sheath at doc 01 §2.1's -250 V
            arrives with 250 eV — plus a margin.
        below: Extrapolation policy below a table's first point. Defaults to the atomic
            layer's, which holds the first value.
        above: Policy above the last point. Defaults to the atomic layer's, which refuses —
            so a ceiling beyond the data is an error at construction, not a silent
            extrapolation into a rate integral.
        nu_max_per_s: The computed ceiling frequency. Derived, not configured.
    """

    species: Species
    gas: BackgroundGas
    processes: tuple[Process, ...]
    energy_ceiling_ev: float
    below: ExtrapolationPolicy = DEFAULT_BELOW
    above: ExtrapolationPolicy = DEFAULT_ABOVE
    nu_max_per_s: float = dataclass_field(init=False)
    is_electron_species: bool = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not self.processes:
            raise ValueError(
                "a collision sampler needs at least one process; an empty set would run "
                "and would be indistinguishable from a collisionless configuration"
            )
        if not self.energy_ceiling_ev > 0.0:
            raise ValueError(f"energy ceiling must be positive, got {self.energy_ceiling_ev}")

        families = {process.kind.is_electron_process for process in self.processes}
        if len(families) != 1:
            raise ValueError(
                "a sampler serves one species: every process must be an electron process "
                "or every one must be an ion process. Mixing them would apply electron "
                "kinematics to ions, which is a wrong answer rather than a crash."
            )
        object.__setattr__(self, "is_electron_species", families.pop())
        object.__setattr__(self, "nu_max_per_s", self._maximum_frequency())

    # ── the ceiling frequency ───────────────────────────────────────────────────

    def _maximum_frequency(self) -> float:
        """The exact maximum of ``n_gas sigma_tot(E) v(E)`` over ``[0, ceiling]``.

        Exact, not scanned. Every cross section is piecewise linear on the union of the
        tabulated energies, so the total is too; and on a segment where ``sigma = a + bE``
        the product ``sigma(E) sqrt(E)`` is stationary at ``E* = -a/(3b)``, which for a
        falling cross section lies *inside* the segment. Taking the maximum over the
        tabulated nodes alone would return a bound that is too small exactly where the
        cross section falls fastest — and a ``nu_max`` that is too small does not announce
        itself, it under-collides.
        """
        grid = self._union_grid()
        sigma_total = np.zeros_like(grid)
        for process in self.processes:
            sigma_total += interpolate_cross_section(
                process.section, grid, below=self.below, above=self.above
            )

        best = float((sigma_total * self._speed(grid)).max())

        lower, upper = grid[:-1], grid[1:]
        slope = (sigma_total[1:] - sigma_total[:-1]) / (upper - lower)
        intercept = sigma_total[:-1] - slope * lower
        with np.errstate(divide="ignore", invalid="ignore"):
            stationary = -intercept / (_STATIONARY_POINT_FACTOR * slope)
        inside = np.isfinite(stationary) & (stationary > lower) & (stationary < upper)
        if np.any(inside):
            energies = stationary[inside]
            sigma_star = intercept[inside] + slope[inside] * energies
            best = max(best, float((sigma_star * self._speed(energies)).max()))

        return self.gas.density_m3 * best

    def _union_grid(self) -> FloatArray:
        """Every tabulated energy in range, plus the two endpoints.

        The union is what makes the piecewise-linear argument above hold for the *sum*:
        each section is linear between its own nodes, so the sum is linear between the
        nodes of all of them.
        """
        pieces = [np.asarray([0.0, self.energy_ceiling_ev], dtype=np.float64)]
        for process in self.processes:
            energies = np.asarray(process.section.energy_ev, dtype=np.float64)
            pieces.append(energies[energies < self.energy_ceiling_ev])
            threshold = process.threshold_ev
            if threshold is not None and threshold < self.energy_ceiling_ev:
                # The threshold is a kink in sigma even when it is not a tabulated point.
                pieces.append(np.asarray([threshold], dtype=np.float64))
        return np.unique(np.concatenate(pieces))

    def _speed(self, energy_ev: FloatArray) -> FloatArray:
        """``v = sqrt(2E/m)``, the speed that appears in ``nu = n sigma v``."""
        return np.sqrt(2.0 * energy_ev * _E / self.species.mass_kg)

    # ── frequencies, for the sampler and for anyone verifying it ────────────────

    def frequencies_per_s(self, energy_ev: ArrayLike) -> FloatArray:
        """``nu_k(E) = n_gas sigma_k(E) v(E)`` for every channel, shape ``(n_proc, n)``.

        Public because it is the closed form the sampled rate has to reproduce, and a
        verification that had to reach into a private helper to state its expectation would
        be testing the implementation rather than the physics (ADR-011).
        """
        energies = np.atleast_1d(np.asarray(energy_ev, dtype=np.float64))
        speeds = self._speed(energies)
        rows = [
            self.gas.density_m3
            * interpolate_cross_section(
                process.section, energies, below=self.below, above=self.above
            )
            * speeds
            for process in self.processes
        ]
        return np.vstack(rows)

    def total_frequency_per_s(self, energy_ev: ArrayLike) -> FloatArray:
        """``nu(E) = sum_k nu_k(E)``. Bounded above by :attr:`nu_max_per_s` by construction."""
        return np.asarray(self.frequencies_per_s(energy_ev).sum(axis=0), dtype=np.float64)

    def null_probability(self, dt_s: float) -> float:
        """``P = 1 - exp(-nu_max dt)`` — the fraction of particles tested each step.

        ``expm1`` rather than ``1 - exp``: at doc 03 §4.3's 11.2 ps timestep the exponent is
        of order 1e-3 and the naive form loses three significant figures to cancellation.
        """
        if not dt_s > 0.0:
            raise ValueError(f"timestep must be positive, got {dt_s}")
        return -math.expm1(-self.nu_max_per_s * dt_s)

    # ── the step ────────────────────────────────────────────────────────────────

    def collide(
        self,
        rng: np.random.Generator,
        *,
        positions_m: ArrayLike,
        velocities_m_per_s: ArrayLike,
        dt_s: float,
    ) -> CollisionOutcome:
        """One collision step — doc 03 §4.2 step 5.

        Args:
            rng: The collision stream's generator, from :func:`collision_generator`. Passed
                in rather than owned: a sampler holding its own would make the run's random
                sequence depend on how many sampler objects were constructed.
            positions_m: Particle positions, shape ``(n,)``. Read only — collisions do not
                move particles — but needed so ionisation products are born in place.
            velocities_m_per_s: Shape ``(n, 3)``.
            dt_s: The timestep.

        Raises:
            ValueError: On a malformed shape or a non-positive timestep.
            EnergyCeilingExceededError: If a particle's collision energy exceeds
                :attr:`energy_ceiling_ev`, where ``nu_max`` would no longer bound ``nu``.
        """
        positions = np.asarray(positions_m, dtype=np.float64)
        velocities = np.array(velocities_m_per_s, dtype=np.float64, copy=True)
        self._check_shapes(positions, velocities)
        probability = self.null_probability(dt_s)

        # Up front, on every particle, and not only on the ones the draw selects. A ceiling
        # violation checked on candidates alone would surface after a random number of
        # steps, or not at all in a short run, which is the difference between a bug report
        # and a wrong paper.
        self._check_ceiling(self._energy_ev(velocities))

        n = positions.shape[0]
        process_index = np.full(n, NO_COLLISION, dtype=np.int64)
        targets = np.zeros((n, 3), dtype=np.float64)

        # THE draw. One uniform per particle per step, doc 03 §4.5, and it does the
        # process selection too: conditioned on r < P, the ratio r/P is uniform on [0, 1)
        # exactly, so no second variate is needed and none is spent.
        # The ``asarray`` is for the type checker, not the arithmetic: NumPy's stub for
        # ``Generator.random`` collapses to the scalar overload under --strict.
        draw: FloatArray = np.asarray(rng.random(n), dtype=np.float64)
        candidates = np.flatnonzero(draw < probability)

        if candidates.size:
            self._select(rng, velocities, targets, process_index, candidates, draw, probability)

        born = _Products()
        for index, process in enumerate(self.processes):
            selected = np.flatnonzero(process_index == index)
            if selected.size:
                self._apply(rng, process, velocities, targets, positions, selected, born)

        counts = np.bincount(
            process_index[process_index >= 0], minlength=len(self.processes)
        ).astype(int)
        return CollisionOutcome(
            velocities_m_per_s=jnp.asarray(velocities),
            process_index=jnp.asarray(process_index),
            target_velocities_m_per_s=jnp.asarray(targets),
            new_electron_positions_m=jnp.asarray(_stack(born.electron_positions, columns=1)),
            new_electron_velocities_m_per_s=jnp.asarray(
                _stack(born.electron_velocities, columns=3)
            ),
            new_ion_positions_m=jnp.asarray(_stack(born.ion_positions, columns=1)),
            new_ion_velocities_m_per_s=jnp.asarray(_stack(born.ion_velocities, columns=3)),
            counts_by_process=tuple(int(value) for value in counts),
        )

    def _check_shapes(self, positions: FloatArray, velocities: FloatArray) -> None:
        if positions.ndim != 1:
            raise ValueError(f"positions must have shape (n,), got {positions.shape}")
        if velocities.ndim != 2 or velocities.shape[1] != 3:
            raise ValueError(
                f"velocities must have shape (n, 3) for a 1D3V model, got {velocities.shape}"
            )
        if velocities.shape[0] != positions.shape[0]:
            raise ValueError(
                f"velocities shape {velocities.shape} does not match positions shape "
                f"{positions.shape}"
            )

    def _select(
        self,
        rng: np.random.Generator,
        velocities: FloatArray,
        targets: FloatArray,
        process_index: NDArray[np.int64],
        candidates: NDArray[np.intp],
        draw: FloatArray,
        probability: float,
    ) -> None:
        """Turn candidates into real collisions, or into null collisions that do nothing.

        The target is drawn *here*, for candidates only. Drawing one for every particle
        would be three Gaussians per particle per step to serve the ~0.1 % of them that a
        typical ``nu_max dt`` selects.
        """
        if not self.is_electron_species:
            targets[candidates] = rng.normal(
                scale=self.gas.thermal_speed_m_per_s, size=(candidates.size, 3)
            )

        # The lookup energy is the *relative* one: the cross-section tables are quoted
        # against the energy of the projectile in the target's rest frame, and for an ion
        # near the Bohm speed the target's thermal motion is not a small correction to it.
        # It can exceed the projectile's own energy, hence the second ceiling check.
        relative = velocities[candidates] - targets[candidates]
        relative_speed = np.linalg.norm(relative, axis=1)
        energy_ev = 0.5 * self.species.mass_kg * relative_speed**2 / _E
        self._check_ceiling(energy_ev)
        energy_ev = np.minimum(energy_ev, self.energy_ceiling_ev)

        frequencies = self.frequencies_per_s(energy_ev)
        edges = np.cumsum(frequencies, axis=0) / self.nu_max_per_s
        if float(edges[-1].max()) > 1.0 + FREQUENCY_BOUND_TOLERANCE:
            raise EnergyCeilingExceededError(
                f"the summed collision frequency reached {float(edges[-1].max()):.6f} of "
                f"nu_max = {self.nu_max_per_s:.6e} /s, so the rejection step would truncate "
                f"it and the run would under-collide. nu_max and the per-particle "
                f"evaluation disagree about the cross section."
            )

        # r/P is the uniform variate; the process is the first whose cumulative share it
        # falls under. Landing past the last edge is a null collision — the whole point of
        # the method — and leaves the particle exactly as it was.
        selector = draw[candidates] / probability
        chosen = (selector[None, :] >= edges).sum(axis=0)
        real = chosen < len(self.processes)
        process_index[candidates[real]] = chosen[real]

    def _check_ceiling(self, energy_ev: FloatArray) -> None:
        if energy_ev.size == 0:
            return
        highest = float(energy_ev.max())
        if highest > self.energy_ceiling_ev * (1.0 + CEILING_TOLERANCE):
            raise EnergyCeilingExceededError(
                f"a particle reached {highest:.4g} eV, above the {self.energy_ceiling_ev:.4g} "
                f"eV ceiling nu_max was computed for. Above it nu(E) may exceed nu_max, the "
                f"scheme collides at nu_max instead, and the run reports an IEDF that is too "
                f"collisionless with nothing in the output to say so. Raise the ceiling to "
                f"cover the full sheath potential drop."
            )

    # ── kinematics ──────────────────────────────────────────────────────────────

    def _apply(
        self,
        rng: np.random.Generator,
        process: Process,
        velocities: FloatArray,
        targets: FloatArray,
        positions: FloatArray,
        selected: NDArray[np.intp],
        born: _Products,
    ) -> None:
        """Dispatch to the kinematics for one channel."""
        match process.kind:
            case CollisionKind.ELECTRON_ELASTIC:
                self._electron_elastic(rng, velocities, selected)
            case CollisionKind.ELECTRON_EXCITATION:
                self._electron_inelastic(rng, velocities, selected, process.threshold_ev or 0.0)
            case CollisionKind.ELECTRON_IONISATION:
                self._electron_ionisation(
                    rng, velocities, positions, selected, process.threshold_ev or 0.0, born
                )
            case CollisionKind.ION_CHARGE_EXCHANGE:
                velocities[selected] = targets[selected]
            case CollisionKind.ION_ELASTIC:
                self._ion_elastic(rng, velocities, targets, selected)

    def _electron_elastic(
        self, rng: np.random.Generator, velocities: FloatArray, selected: NDArray[np.intp]
    ) -> None:
        """Isotropic scattering with the ``2 m/M (1 - cos chi)`` energy loss.

        The outgoing direction is drawn isotropically and ``cos chi`` is then *read off* it
        rather than sampled separately and used to rotate. The two are the same
        distribution, and this way the deflection angle and the energy loss cannot drift
        apart — which is the failure that makes a code conserve momentum and not energy.
        """
        incoming = velocities[selected]
        speed = np.linalg.norm(incoming, axis=1)
        direction = _isotropic(rng, selected.size)

        safe = np.where(speed > 0.0, speed, 1.0)
        cos_chi = np.sum(incoming * direction, axis=1) / safe
        ratio = self.species.mass_kg / self.gas.mass_kg
        retained = 1.0 - 2.0 * ratio * (1.0 - cos_chi)
        velocities[selected] = (speed * np.sqrt(retained))[:, None] * direction

    def _electron_inelastic(
        self,
        rng: np.random.Generator,
        velocities: FloatArray,
        selected: NDArray[np.intp],
        threshold_ev: float,
    ) -> None:
        """Excitation: pay the threshold, then scatter isotropically on what is left."""
        energy_ev = self._energy_ev(velocities[selected])
        remaining = np.maximum(energy_ev - threshold_ev, 0.0)
        speed = self._speed(remaining)
        velocities[selected] = speed[:, None] * _isotropic(rng, selected.size)

    def _electron_ionisation(
        self,
        rng: np.random.Generator,
        velocities: FloatArray,
        positions: FloatArray,
        selected: NDArray[np.intp],
        threshold_ev: float,
        born: _Products,
    ) -> None:
        """One electron in, two electrons and one ion out — doc 03 §4.5.

        The excess is shared equally (simplification 4). Both electrons are scattered
        isotropically and independently, and the ion is born out of the *neutral*, so it
        carries the gas temperature rather than a share of the electron's energy — which is
        what makes ionisation a source of cold ions and not of hot ones.
        """
        energy_ev = self._energy_ev(velocities[selected])
        share = np.maximum(energy_ev - threshold_ev, 0.0) / 2.0
        speed = self._speed(share)

        velocities[selected] = speed[:, None] * _isotropic(rng, selected.size)
        born.electron_positions.append(positions[selected])
        born.electron_velocities.append(speed[:, None] * _isotropic(rng, selected.size))
        born.ion_positions.append(positions[selected])
        born.ion_velocities.append(
            rng.normal(scale=self.gas.thermal_speed_m_per_s, size=(selected.size, 3))
        )

    def _ion_elastic(
        self,
        rng: np.random.Generator,
        velocities: FloatArray,
        targets: FloatArray,
        selected: NDArray[np.intp],
    ) -> None:
        """Isotropic in the centre-of-mass frame — doc 03 §4.5's "Ion elastic (isotropic)".

        Written with the general mass ratio rather than the equal-mass shortcut. Argon on
        argon makes ``m_i = m_n`` and the two coincide, but doc 01 §2.4's envelope includes
        xenon (benchmark B-13) and the shortcut would then be wrong in a way that still
        conserves momentum, so nothing downstream would catch it.
        """
        incoming = velocities[selected]
        neutral = targets[selected]
        total_mass = self.species.mass_kg + self.gas.mass_kg
        centre_of_mass = (self.species.mass_kg * incoming + self.gas.mass_kg * neutral) / total_mass
        relative_speed = np.linalg.norm(incoming - neutral, axis=1)

        scattered = (self.gas.mass_kg / total_mass) * relative_speed
        velocities[selected] = centre_of_mass + scattered[:, None] * _isotropic(rng, selected.size)

    def _energy_ev(self, velocities: FloatArray) -> FloatArray:
        return np.asarray(
            0.5 * self.species.mass_kg * np.sum(velocities**2, axis=1) / _E, dtype=np.float64
        )


def _isotropic(rng: np.random.Generator, n: int) -> FloatArray:
    """``n`` unit vectors uniform on the sphere.

    ``cos theta`` uniform on ``[-1, 1]`` and ``phi`` uniform on ``[0, 2 pi)``. Sampling
    ``theta`` uniformly instead is the classic error and it clusters directions at the
    poles — which in a 1D3V code means clustering them along ``z``, the one axis every
    observable in this project is projected onto.
    """
    cos_theta = 2.0 * rng.random(n) - 1.0
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = 2.0 * np.pi * rng.random(n)
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=1)


@dataclass(slots=True)
class _Products:
    """Particles born during a step, held as per-channel chunks until the arrays are built.

    Chunks rather than a growing array: ionisation fires for well under a percent of the
    electrons, and reallocating per event would cost more than the channel does.
    """

    electron_positions: list[FloatArray] = dataclass_field(default_factory=list)
    electron_velocities: list[FloatArray] = dataclass_field(default_factory=list)
    ion_positions: list[FloatArray] = dataclass_field(default_factory=list)
    ion_velocities: list[FloatArray] = dataclass_field(default_factory=list)


def _stack(pieces: list[FloatArray], *, columns: int) -> FloatArray:
    """Join the chunks, keeping the shape right when nothing was born.

    ``(0, 3)`` and not ``(0,)``: a solver concatenating an empty velocity array onto its
    particle list would otherwise fail as a broadcast error several steps later.
    """
    if not pieces:
        return np.empty(0 if columns == 1 else (0, columns), dtype=np.float64)
    return np.concatenate(pieces, axis=0)
