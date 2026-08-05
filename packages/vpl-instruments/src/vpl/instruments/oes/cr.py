"""The collisional-radiative model — doc 04 §2.2.

Doc 04 §2.2 states the problem and why it is not a Boltzmann distribution over levels:

    Excited-state populations are *not* in local thermodynamic equilibrium at these
    densities, so a Boltzmann distribution over levels is wrong. The framework solves a CR
    model:

        dn_u/dt  =  sum_j n_e n_j K_ju  +  sum_(k>u) n_k A_ku Lambda_ku
                    -  n_u ( sum_j n_e K_uj  +  sum_(l<u) A_ul Lambda_ul  +  nu_wall ) = 0

This module is that equation, assembled as a linear system in the excited-state densities
and solved. The rate coefficients come from :mod:`vpl.instruments.oes.levels`, which
integrates LXCat cross sections over the EEDF the Boltzmann solver produced — doc 04 §2.2's
"not a Maxwellian" — and the escape factors from :mod:`vpl.instruments.oes.escape`.

## The linear system

The ground state is held fixed at ``n_g`` and is not an unknown. That is the standard
closure and it is exact to the excited fraction, which at the doc 01 §2.1 operating point
is below 1e-4. It makes excitation out of the ground state a *source* rather than a matrix
element, so the system is

    M n  +  s  =  0 ,      s_u = n_e n_g K_gu

with ``M`` carrying, for every pair of excited levels, the collisional exchange in both
directions and the radiative cascade downwards; and on its diagonal every loss channel out
of the level. ``n = solve(-M, s)``.

The matrix is small — one row per excited level, so ten or twenty — and dense. There is no
case for anything cleverer: doc 08 §1 principle 7 asks for small focused modules, and a
sparse solver here would be complexity bought for nothing.

## Radiation trapping makes it non-linear, and that is not avoidable

Doc 04 §2.3 requires escape factors "computed for the chamber geometry" and "recomputed as
the density profile changes". The absorber for a trapped line is the *lower level of that
line*, which for the Ar I resonance lines is the ground state — fixed, so the escape factor
is a constant — but for any line terminating on a metastable is one of the unknowns.

Where that happens the solve is a fixed-point iteration: solve with the current escape
factors, recompute them from the populations that came out, repeat. It converges quickly
because ``Lambda`` depends on ``n_l`` only through ``ln(tau_0)`` once the line is thick,
but it does not converge unconditionally, and :class:`CrConvergenceError` is raised rather
than a best-effort answer returned — a partly converged CR solution is a plausible set of
line ratios, and doc 02 §6.3 sends those straight into the ``T_e`` inference.

Where no line is trapped the loop runs exactly once and reports one iteration, so the
common case pays nothing.

## What is deliberately not modelled

Named here rather than discovered later (doc 00 C4). See also
:mod:`vpl.instruments.oes.levels`, which lists the omissions in the atomic data itself, and
:mod:`vpl.instruments.oes.escape`, which lists those in the trapping.

- **Zero-dimensional.** One call is one point. Doc 04 §2.4 evaluates the emissivity "on the
  plasma grid", which is done by calling this per grid point; the levels do not diffuse
  between points. For the metastables that is the questionable one: their diffusion length
  at 5 mTorr is centimetres, comparable to the chamber, so a metastable density computed
  point-by-point in a **sheath** — where the gradient scale is 890 um — is wrong. The wall
  term is the only concession to transport, and it is a lumped one. This is the largest
  approximation in the module and it bounds where its output can be trusted: bulk plasma,
  not the sheath interior.
- **Quasi-static.** ``dn_u/dt = 0``, as doc 04 §2.2 writes it. Valid while the level
  lifetimes are short against the driving timescale. For the radiating levels (~30 ns) that
  holds against the 73.7 ns RF period only marginally, and for **metastables it does not
  hold at all** — a 1e-4 s metastable lifetime against a 7.4e-8 s period means the
  metastable population is phase-*averaged*, not phase-resolved. A phase-resolved OES
  measurement of a metastable-coupled line (doc 02 §6.3's 811.53 nm) therefore sees a
  constant metastable background modulating a varying excitation rate, and this model does
  not reproduce that structure.
- **No ionisation from excited states.** Doc 04 §2.2's equation has no ionisation term and
  neither does this. Stepwise ionisation through the metastables is a real loss channel at
  RP-1 and its omission over-estimates the metastable density; the rate coefficient for
  Ar(1s5) ionisation at ``T_e`` = 3 eV is of order 1e-14 m^3/s, giving ~1e3 /s at
  ``n_e`` = 1e17 m^-3 against a ~1e4 /s wall term — so of order **10 %** on the metastable
  density at RP-1, growing linearly with ``n_e``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.instruments.oes.escape import escape_factor
from vpl.instruments.oes.levels import LevelSystem
from vpl.physics.eedf.grid import EnergyGrid

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_RELATIVE_TOLERANCE",
    "CollisionalRadiativeModel",
    "CrConvergenceError",
    "LevelPopulations",
]

type FloatArray = NDArray[np.float64]

#: Convergence criterion on the escape-factor fixed point: the largest relative change in
#: any escape factor between successive iterations.
DEFAULT_RELATIVE_TOLERANCE: Final[float] = 1e-10

#: Iteration budget for that fixed point. Generous relative to the two or three iterations
#: a converging case takes, because a case that needs many is a case worth seeing fail
#: loudly rather than one worth cutting short.
DEFAULT_MAX_ITERATIONS: Final[int] = 100


class CrConvergenceError(RuntimeError):
    """The escape-factor fixed point did not converge.

    An error and not a flag, for :class:`~vpl.physics.eedf.solver.EedfConvergenceError`'s
    reason: a partly converged CR solution is a smooth, positive, plausible set of level
    populations, and doc 02 §6.3 turns those into a ``T_e``.
    """


@dataclass(frozen=True, slots=True, eq=False)
class LevelPopulations:
    """The solved excited-state densities, and how they were obtained.

    Attributes:
        system: The level system these belong to.
        densities_per_m3: Density of every level including the ground state, keyed by
            label. Read-only.
        escape_factors: The ``Lambda_ul`` actually used, keyed by ``(upper, lower)``.
            Reported rather than discarded because doc 06 §4 budgets the escape-factor
            uncertainty, and a term whose value is not in the artifact cannot be budgeted
            after the fact.
        iterations: Fixed-point iterations taken. One where nothing is trapped.
        residual: The final relative change in the escape factors.
    """

    system: LevelSystem
    densities_per_m3: Mapping[str, float]
    escape_factors: Mapping[tuple[str, str], float]
    iterations: int
    residual: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "densities_per_m3", MappingProxyType(dict(self.densities_per_m3)))
        object.__setattr__(self, "escape_factors", MappingProxyType(dict(self.escape_factors)))

    def __getitem__(self, label: str) -> float:
        try:
            return self.densities_per_m3[label]
        except KeyError:
            raise KeyError(
                f"no level {label!r} in this solution; it has {', '.join(self.labels)}"
            ) from None

    @property
    def labels(self) -> tuple[str, ...]:
        return self.system.labels

    def __iter__(self) -> Iterator[str]:
        return iter(self.labels)

    def fraction_of_ground(self, label: str) -> float:
        """``n_u / n_g`` — the number doc 04 §2.2's LTE limit is stated in."""
        return self[label] / self[self.system.ground.label]

    def __repr__(self) -> str:
        excited = ", ".join(
            f"{level.label}={self[level.label]:.3g}" for level in self.system.excited
        )
        return f"LevelPopulations({excited}, {self.iterations} iterations)"


@dataclass(frozen=True, slots=True)
class CollisionalRadiativeModel:
    """Doc 04 §2.2's quasi-static balance, for one level system on one energy grid.

    Attributes:
        system: The levels and channels.
        grid: The energy grid the cross sections and the EEDF share.
        wall_loss_per_s: ``nu_wall`` per level. Absent means zero. Doc 04 §2.2 describes it
            as "diffusion to wall + surface quenching probability"; both are chamber
            properties, so the number is supplied rather than derived here.
        trapping_path_length_m: The path the escape factors are computed over — doc 02
            §3.1's chamber dimension. Required if any radiative channel carries trapping
            data, and rejected as meaningless if none does.
        relative_tolerance: Convergence criterion on the escape-factor fixed point.
        max_iterations: Its budget.
    """

    system: LevelSystem
    grid: EnergyGrid
    wall_loss_per_s: Mapping[str, float] = field(default_factory=dict)
    trapping_path_length_m: float | None = None
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    def __post_init__(self) -> None:
        for label, loss in self.wall_loss_per_s.items():
            if label not in self.system.labels:
                raise ValueError(
                    f"wall loss given for {label!r} but there is no level of that name; "
                    f"the system has {', '.join(self.system.labels)}"
                )
            if loss < 0.0:
                raise ValueError(f"{label}: a wall loss rate cannot be negative, got {loss}")

        for impact in self.system.electron_impact:
            if impact.sigma_m2.size != self.grid.n_cells:
                raise ValueError(
                    f"{impact.lower} -> {impact.upper}: cross section has "
                    f"{impact.sigma_m2.size} points but the grid has {self.grid.n_cells} cells"
                )

        if self.system.has_trapping and self.trapping_path_length_m is None:
            raise ValueError(
                "a trapped radiative channel needs a trapping path length — doc 04 §2.3 "
                "computes escape factors 'for the chamber geometry', and there is no "
                "defensible default for a chamber dimension"
            )
        if self.trapping_path_length_m is not None and not self.trapping_path_length_m > 0.0:
            raise ValueError(
                f"the trapping path length must be positive, got {self.trapping_path_length_m}"
            )
        if not self.relative_tolerance > 0.0:
            raise ValueError(
                f"the convergence tolerance must be positive, got {self.relative_tolerance}"
            )
        if self.max_iterations < 1:
            raise ValueError(f"the iteration budget must be at least 1, got {self.max_iterations}")

        object.__setattr__(self, "wall_loss_per_s", MappingProxyType(dict(self.wall_loss_per_s)))

    # ── the solve ───────────────────────────────────────────────────────────────

    def solve(
        self,
        *,
        electron_density_per_m3: float,
        ground_density_per_m3: float,
        f0: ArrayLike,
    ) -> LevelPopulations:
        """Solve doc 04 §2.2's balance for the excited-state densities.

        Args:
            electron_density_per_m3: ``n_e`` at this point.
            ground_density_per_m3: ``n_g``, held fixed. See the module docstring.
            f0: The isotropic EEDF at the grid cell centres, in the
                ``integral f0 sqrt(eps) deps = 1`` normalisation that
                :class:`~vpl.physics.eedf.solver.EedfSolution` uses. **Not** a Maxwellian
                unless the caller means it to be.

        Returns:
            Every level's density, the escape factors used, and the iteration record.

        Raises:
            ValueError: If a density is negative or ``f0`` is the wrong length.
            CrConvergenceError: If the escape-factor fixed point did not converge.
            numpy.linalg.LinAlgError: If the balance matrix is singular, which means some
                level has no loss channel at all — at ``n_e = 0`` with no wall term, every
                metastable is such a level.
        """
        n_e = float(electron_density_per_m3)
        n_g = float(ground_density_per_m3)
        for name, value in (("the electron density", n_e), ("the ground-state density", n_g)):
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative, got {value}")

        distribution = np.asarray(f0, dtype=np.float64)
        if distribution.shape != (self.grid.n_cells,):
            raise ValueError(
                f"a distribution must carry one value per cell: expected shape "
                f"({self.grid.n_cells},), got {distribution.shape}"
            )

        rates = self._rate_coefficients(distribution)
        escape = {channel.key: 1.0 for channel in self.system.radiative}

        densities = self._balance(n_e=n_e, n_g=n_g, rates=rates, escape=escape)
        if not self.system.has_trapping:
            return LevelPopulations(
                system=self.system,
                densities_per_m3=densities,
                escape_factors=escape,
                iterations=1,
                residual=0.0,
            )

        iterations = 1
        residual = np.inf
        for _ in range(self.max_iterations):
            updated = self._escape_factors(densities, ground_density_per_m3=n_g)
            residual = max(
                abs(updated[key] - escape[key]) / max(escape[key], updated[key]) for key in updated
            )
            escape = updated
            densities = self._balance(n_e=n_e, n_g=n_g, rates=rates, escape=escape)
            if residual < self.relative_tolerance:
                break
            iterations += 1
        else:
            raise CrConvergenceError(
                f"the escape-factor fixed point did not converge: relative change "
                f"{residual:.3g} after {iterations} iterations, tolerance "
                f"{self.relative_tolerance:g}. A line whose absorbing level is itself "
                "strongly trapped is the usual cause."
            )

        return LevelPopulations(
            system=self.system,
            densities_per_m3=densities,
            escape_factors=escape,
            iterations=iterations,
            residual=float(residual),
        )

    # ── assembly ────────────────────────────────────────────────────────────────

    def _rate_coefficients(self, f0: FloatArray) -> dict[tuple[str, str], tuple[float, float]]:
        """``(K_lu, K_ul)`` per channel, keyed by ``(lower, upper)``.

        Computed once per solve and reused across every fixed-point iteration: the escape
        factors change, the EEDF does not, and re-integrating the cross sections inside
        the loop would be the dominant cost for no reason.
        """
        weights = self.system.degeneracies()
        return {
            (impact.lower, impact.upper): (
                impact.excitation_rate_coefficient(self.grid, f0),
                impact.de_excitation_rate_coefficient(
                    self.grid,
                    f0,
                    g_lower=weights[impact.lower],
                    g_upper=weights[impact.upper],
                ),
            )
            for impact in self.system.electron_impact
        }

    def _balance(
        self,
        *,
        n_e: float,
        n_g: float,
        rates: Mapping[tuple[str, str], tuple[float, float]],
        escape: Mapping[tuple[str, str], float],
    ) -> dict[str, float]:
        """Assemble ``M n + s = 0`` over the excited levels and solve it."""
        excited = self.system.excited
        index = {level.label: position for position, level in enumerate(excited)}
        ground = self.system.ground.label

        size = len(excited)
        matrix = np.zeros((size, size), dtype=np.float64)
        source = np.zeros(size, dtype=np.float64)

        for (lower, upper), (excitation, de_excitation) in rates.items():
            upper_index = index[upper]
            if lower == ground:
                # Excitation out of the fixed ground state is a source; de-excitation back
                # into it is a loss with no matrix partner, because the ground state is not
                # an unknown.
                source[upper_index] += n_e * n_g * excitation
                matrix[upper_index, upper_index] -= n_e * de_excitation
                continue

            lower_index = index[lower]
            matrix[upper_index, lower_index] += n_e * excitation
            matrix[lower_index, lower_index] -= n_e * excitation
            matrix[lower_index, upper_index] += n_e * de_excitation
            matrix[upper_index, upper_index] -= n_e * de_excitation

        for channel in self.system.radiative:
            rate = channel.a_ul_per_s * escape[channel.key]
            upper_index = index[channel.upper]
            matrix[upper_index, upper_index] -= rate
            if channel.lower != ground:
                matrix[index[channel.lower], upper_index] += rate

        for label, loss in self.wall_loss_per_s.items():
            if label != ground:
                matrix[index[label], index[label]] -= loss

        solution = np.linalg.solve(-matrix, source)
        return {ground: n_g} | {
            level.label: float(solution[index[level.label]]) for level in excited
        }

    def _escape_factors(
        self, densities: Mapping[str, float], *, ground_density_per_m3: float
    ) -> dict[tuple[str, str], float]:
        """``Lambda_ul`` for every channel, at the current absorber densities."""
        path = self.trapping_path_length_m
        if path is None:  # pragma: no cover - __post_init__ guarantees this
            raise AssertionError("a trapped system reached solve() with no path length")
        factors: dict[tuple[str, str], float] = {}
        for channel in self.system.radiative:
            if channel.trapping is None:
                factors[channel.key] = 1.0
                continue
            absorber = (
                ground_density_per_m3
                if channel.lower == self.system.ground.label
                else densities[channel.lower]
            )
            factors[channel.key] = escape_factor(
                channel.trapping.optical_depth(lower_density_per_m3=absorber, path_length_m=path),
                shape=channel.trapping.shape,
            )
        return factors

    def __repr__(self) -> str:
        return (
            f"CollisionalRadiativeModel({self.system!r}, {self.grid.n_cells} energy cells, "
            f"{len(self.wall_loss_per_s)} wall terms)"
        )
