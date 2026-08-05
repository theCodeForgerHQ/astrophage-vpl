"""The level system a CR model is assembled from — doc 04 §2.2.

Doc 04 §2.2's balance equation needs four kinds of thing, and this module is the three of
them that are *data*::

    dn_u/dt  =  sum_j n_e n_j K_ju  +  sum_(k>u) n_k A_ku Lambda_ku
                -  n_u ( sum_j n_e K_uj  +  sum_(l<u) A_ul Lambda_ul  +  nu_wall )  =  0

:class:`Level` carries the energies and statistical weights, :class:`ElectronImpactChannel`
the ``K``, :class:`RadiativeChannel` the ``A`` and its escape factor. The fourth —
``nu_wall`` — is a property of the chamber and not of the atom, so it is supplied to
:class:`~vpl.instruments.oes.cr.CollisionalRadiativeModel` and is not here.

## The superelastic rates, and why they are integrals rather than a formula

Doc 04 §2.2 is explicit that the rate coefficients come from "LXCat cross sections
integrated over the **BOLSIG+ EEDF**, not a Maxwellian". That constraint has a consequence
most CR models quietly duck: the *de-excitation* coefficient ``K_ul`` cannot be obtained
from ``K_lu`` by the Maxwellian detailed-balance formula
``K_ul = (g_l/g_u) exp(dE/T_e) K_lu``, because there is no ``T_e``.

What is legitimate for any distribution is the Klein-Rosseland relation between the *cross
sections*, which is a statement about the collision itself::

    g_l eps sigma_lu(eps)  =  g_u eps' sigma_ul(eps')  ,   eps' = eps - dE

Substituting it into the rate integral ``k = gamma integral eps sigma f0 deps`` and
changing variable to ``eps'`` gives

    K_ul  =  (g_l/g_u) gamma integral_0^inf (eps' + dE) sigma_lu(eps' + dE) f0(eps') deps'

which is what :meth:`ElectronImpactChannel.de_excitation_rate_coefficient` evaluates. Two
properties of that form are worth stating because they are why it is written this way:

1. ``f0`` is sampled at the **grid centres**, unshifted. Nothing interpolates the
   distribution, which is the object with hundreds of decades of dynamic range; only the
   cross section is interpolated, and that is smooth and bounded.
2. On a uniform grid whose cell width divides ``dE``, the shift is an exact index offset,
   so handing this a Maxwellian reproduces the detailed-balance formula to floating-point
   rounding. That identity is the strongest verification available for the whole module
   and ``test_cr.py`` asserts it.

The quadrature is character-for-character the one
:class:`~vpl.physics.eedf.solver.TwoTermSolver` uses for its own rate coefficients, and
``GAMMA`` is imported from it rather than recomputed. A CR model whose rate integral
disagreed with the Boltzmann solver's would be inconsistent with the EEDF it was handed.

## What is deliberately not modelled

- **No ionisation or recombination.** Doc 04 §2.2's equation has neither, and the level
  system is a single ionisation stage. The consequence is that the Ar II lines of doc 02
  §6.3 cannot be produced by this model at all; they need a second stage and a Saha-like
  coupling, which is not implemented.
- **No heavy-particle collisions.** Ar(1s) + Ar mixing and two-body/three-body quenching
  of the metastables are omitted. At 5 mTorr the neutral density is 1.6e20 m^-3 and the
  1s-mixing rate coefficient is of order 1e-18 m^3/s, giving ~1e2 /s against the ~1e4 /s
  wall term — a **~1 % effect at RP-1**, rising linearly with pressure, so at the 100 mTorr
  top of the doc 01 §2.4 envelope it is of order 20 % and this omission stops being safe.
- **No photo-ionisation, no radiative recombination, no dielectronic capture.**
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.instruments.oes.escape import TrappedLine
from vpl.physics.eedf.grid import EnergyGrid
from vpl.physics.eedf.solver import GAMMA

__all__ = [
    "THRESHOLD_TOLERANCE_EV",
    "ElectronImpactChannel",
    "Level",
    "LevelSystem",
    "RadiativeChannel",
]

type FloatArray = NDArray[np.float64]

#: How far a channel's declared threshold may sit from the level-energy difference it is
#: supposed to equal. Tight, because the two come from the same NIST level table and a
#: disagreement means one of them was transcribed from somewhere else — the exact failure
#: doc 09 §2.2's ``h nu = E_k - E_i`` check catches for the radiative data.
THRESHOLD_TOLERANCE_EV: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class Level:
    """One atomic level of one ionisation stage.

    Attributes:
        label: How every channel refers to this level. Free text and Paschen notation is
            fine (``"1s5"``, ``"2p9"``); it is a key, not a spectroscopic assertion.
        energy_ev: Excitation energy above the ground state. Zero for the ground state.
        degeneracy: Statistical weight ``g``. The ``g_u`` and ``g_l`` of every
            detailed-balance relation in the package.
        is_metastable: Whether this level is metastable. Carried for reporting and for the
            wall-quenching term of doc 04 §2.2 — a metastable is the only level for which
            diffusion to the wall competes with radiative decay, because it has none.
    """

    label: str
    energy_ev: float
    degeneracy: int
    is_metastable: bool = False

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("a level needs a label")
        if self.energy_ev < 0.0:
            raise ValueError(
                f"{self.label}: level energies are measured above the ground state and "
                f"cannot be negative, got {self.energy_ev} eV"
            )
        if self.degeneracy < 1:
            raise ValueError(
                f"{self.label}: a statistical weight counts degenerate states and is at "
                f"least 1, got {self.degeneracy}"
            )

    def boltzmann_ratio_to_ground(self, *, electron_temperature_ev: float, g_ground: int) -> float:
        """``(g_u / g_0) exp(-E_u / T_e)`` — the LTE population ratio.

        Only meaningful where the EEDF is Maxwellian. It is here as the closed form
        doc 04 §8 V-25's high-density limit is checked against, and for nothing else.
        """
        if not electron_temperature_ev > 0.0:
            raise ValueError(f"T_e must be positive, got {electron_temperature_ev} eV")
        return (self.degeneracy / g_ground) * float(
            np.exp(-self.energy_ev / electron_temperature_ev)
        )

    def __repr__(self) -> str:
        kind = ", metastable" if self.is_metastable else ""
        return f"Level({self.label!r}, {self.energy_ev} eV, g={self.degeneracy}{kind})"


@dataclass(frozen=True, slots=True, eq=False)
class ElectronImpactChannel:
    """One electron-impact excitation channel, and its superelastic partner.

    The two directions are one object because they are one cross section. Storing them
    separately would let a level system be assembled in which the de-excitation of a
    transition disagreed with its excitation, and the LTE limit would then fail for a
    reason no test could localise.

    Attributes:
        lower: Label of the lower level.
        upper: Label of the upper level.
        threshold_ev: ``E_u - E_l``.
        sigma_m2: ``sigma_lu`` at the energy-grid cell centres. Zero below threshold.
        shifted_sigma_m2: ``sigma_lu(eps + dE)`` at the same cell centres, and zero where
            that lands above the grid. Precomputed because it is the whole content of the
            superelastic integral and because getting the top-of-grid truncation right is
            what makes the detailed-balance identity exact rather than approximate.
    """

    lower: str
    upper: str
    threshold_ev: float
    sigma_m2: FloatArray
    shifted_sigma_m2: FloatArray

    def __post_init__(self) -> None:
        if self.lower == self.upper:
            raise ValueError(f"{self.lower}: a channel connects two different levels")
        if not self.threshold_ev > 0.0:
            raise ValueError(
                f"{self.lower} -> {self.upper}: the threshold must be positive, got "
                f"{self.threshold_ev} eV"
            )
        sampled = (("sigma_m2", self.sigma_m2), ("shifted_sigma_m2", self.shifted_sigma_m2))
        for name, values in sampled:
            if values.ndim != 1 or values.size != self.sigma_m2.size:
                raise ValueError(f"{name} must be one value per energy-grid cell")
            if np.any(values < 0.0) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite and non-negative")

    # ── construction ────────────────────────────────────────────────────────────

    @classmethod
    def from_cross_section(
        cls,
        grid: EnergyGrid,
        *,
        lower: str,
        upper: str,
        threshold_ev: float,
        sigma_m2: ArrayLike,
    ) -> Self:
        """Build from ``sigma_lu`` sampled at the grid cell centres.

        The shifted cross section is obtained by linear interpolation *of the tabulated
        values on this grid*, not by re-evaluating the source. That is deliberate: it
        makes the superelastic integral use exactly the same numbers as the excitation
        integral, so the discrete detailed-balance identity holds, and it makes the
        top-of-grid truncation symmetric between the two directions. Re-sampling the
        original cross section above the last cell centre would add terms to the
        superelastic sum that the excitation sum has no counterpart for.
        """
        centres = grid.centres_ev
        sigma = np.asarray(sigma_m2, dtype=np.float64)
        if sigma.shape != (grid.n_cells,):
            raise ValueError(
                f"a cross section must carry one value per cell: expected shape "
                f"({grid.n_cells},), got {sigma.shape}"
            )
        shifted = np.interp(centres + threshold_ev, centres, sigma, left=0.0, right=0.0)
        return cls(
            lower=lower,
            upper=upper,
            threshold_ev=threshold_ev,
            sigma_m2=sigma,
            shifted_sigma_m2=np.asarray(shifted),
        )

    @classmethod
    def from_sampler(
        cls,
        grid: EnergyGrid,
        *,
        lower: str,
        upper: str,
        threshold_ev: float,
        sampler: Callable[[FloatArray], FloatArray],
    ) -> Self:
        """Build from anything that evaluates ``sigma_lu`` on an array of energies.

        The interpolators of :mod:`vpl.physics.atomic.interpolation` satisfy this, which
        is how an LXCat set reaches a CR model without this module knowing about LXCat.
        """
        return cls.from_cross_section(
            grid,
            lower=lower,
            upper=upper,
            threshold_ev=threshold_ev,
            sigma_m2=sampler(grid.centres_ev),
        )

    # ── the rate coefficients ───────────────────────────────────────────────────

    def excitation_rate_coefficient(self, grid: EnergyGrid, f0: ArrayLike) -> float:
        """``K_lu = gamma integral eps sigma(eps) f0(eps) deps``, in m^3/s.

        The quadrature is :class:`~vpl.physics.eedf.solver.TwoTermSolver`'s, so a rate
        computed here and a rate computed there from the same cross section and the same
        ``f0`` agree exactly rather than nearly.
        """
        distribution = self._checked(grid, f0)
        weight = self.sigma_m2 * grid.centres_ev * grid.widths_ev
        return GAMMA * float(np.dot(weight, distribution))

    def de_excitation_rate_coefficient(
        self, grid: EnergyGrid, f0: ArrayLike, *, g_lower: int, g_upper: int
    ) -> float:
        """``K_ul`` by Klein-Rosseland over the actual EEDF — see the module docstring.

        Args:
            grid: The energy grid ``f0`` and the cross sections live on.
            f0: The isotropic EEDF at cell centres, in the
                ``integral f0 sqrt(eps) deps = 1`` normalisation.
            g_lower: Statistical weight of the lower level.
            g_upper: Statistical weight of the upper level.

        Returns:
            The superelastic rate coefficient, in m^3/s.
        """
        if g_lower < 1 or g_upper < 1:
            raise ValueError(f"statistical weights are at least 1, got {g_lower} and {g_upper}")
        distribution = self._checked(grid, f0)
        weight = self.shifted_sigma_m2 * (grid.centres_ev + self.threshold_ev) * grid.widths_ev
        return (g_lower / g_upper) * GAMMA * float(np.dot(weight, distribution))

    def _checked(self, grid: EnergyGrid, f0: ArrayLike) -> FloatArray:
        values = np.asarray(f0, dtype=np.float64)
        if values.shape != (grid.n_cells,):
            raise ValueError(
                f"a distribution must carry one value per cell: expected shape "
                f"({grid.n_cells},), got {values.shape}"
            )
        if values.size != self.sigma_m2.size:
            raise ValueError(
                f"{self.lower} -> {self.upper}: cross section has {self.sigma_m2.size} "
                f"points but the grid has {grid.n_cells} cells"
            )
        return values

    def __repr__(self) -> str:
        return (
            f"ElectronImpactChannel({self.lower!r} -> {self.upper!r}, "
            f"threshold={self.threshold_ev} eV)"
        )


@dataclass(frozen=True, slots=True)
class RadiativeChannel:
    """One spontaneous-emission channel, with its optional radiation trapping.

    Attributes:
        upper: Label of the emitting level.
        lower: Label of the terminating level.
        a_ul_per_s: Einstein A coefficient — doc 09 §2.2, NIST ASD.
        wavelength_nm: Transition wavelength, for the emissivity of doc 04 §2.1.
        trapping: The line's absorption data, or ``None`` for a line taken as optically
            thin. ``None`` is a *statement*, not a default: doc 04 §2.3 says an
            optically-thin CR model produces a systematically wrong ``T_e``, so a channel
            left untrapped should be one whose opacity has been checked and found small.
    """

    upper: str
    lower: str
    a_ul_per_s: float
    wavelength_nm: float
    trapping: TrappedLine | None = None

    def __post_init__(self) -> None:
        if self.upper == self.lower:
            raise ValueError(f"{self.upper}: a transition connects two different levels")
        if not self.a_ul_per_s > 0.0:
            raise ValueError(
                f"{self.upper} -> {self.lower}: A_ul must be positive, got {self.a_ul_per_s} /s. "
                "A transition with no transition probability is a level pair, not a line."
            )
        if not self.wavelength_nm > 0.0:
            raise ValueError(
                f"{self.upper} -> {self.lower}: wavelength must be positive, got "
                f"{self.wavelength_nm} nm"
            )

    @property
    def key(self) -> tuple[str, str]:
        """``(upper, lower)`` — how an escape factor is keyed."""
        return (self.upper, self.lower)

    @property
    def is_trapped(self) -> bool:
        return self.trapping is not None

    def __repr__(self) -> str:
        trapped = ", trapped" if self.is_trapped else ""
        return (
            f"RadiativeChannel({self.upper!r} -> {self.lower!r}, {self.wavelength_nm} nm, "
            f"A_ul={self.a_ul_per_s:.3g} /s{trapped})"
        )


@dataclass(frozen=True, slots=True, eq=False)
class LevelSystem:
    """Levels plus the channels connecting them — the input to a CR model.

    Attributes:
        levels: In ascending energy, ground state first. The order is the order the
            CR matrix is assembled in, so it is fixed here rather than left to whatever
            order a loader produced (doc 00 E3).
        electron_impact: The ``K`` channels.
        radiative: The ``A`` channels.
    """

    levels: tuple[Level, ...]
    electron_impact: tuple[ElectronImpactChannel, ...]
    radiative: tuple[RadiativeChannel, ...]

    def __post_init__(self) -> None:
        self._check_levels()
        self._check_channels()

    def _check_levels(self) -> None:
        if len(self.levels) < 2:
            raise ValueError(
                f"a level system needs a ground state and at least one excited level, got "
                f"{len(self.levels)}"
            )
        labels = [level.label for level in self.levels]
        if len(set(labels)) != len(labels):
            raise ValueError(f"level labels must be unique, got {labels}")
        if self.levels[0].energy_ev != 0.0:
            raise ValueError(
                f"the first level is the ground state and must sit at zero energy, got "
                f"{self.levels[0].energy_ev} eV. Level energies are measured from it."
            )
        energies = [level.energy_ev for level in self.levels]
        if any(b < a for a, b in itertools.pairwise(energies)):
            raise ValueError(
                f"levels must be given in ascending energy, got {energies}. The CR matrix "
                "is assembled in this order and doc 00 E3 requires it to be fixed."
            )

    def _check_channels(self) -> None:
        known = {level.label: level for level in self.levels}
        for impact in self.electron_impact:
            for label in (impact.lower, impact.upper):
                if label not in known:
                    raise ValueError(
                        f"electron-impact channel references no level {label!r}; "
                        f"the system has {', '.join(known)}"
                    )
            gap = known[impact.upper].energy_ev - known[impact.lower].energy_ev
            if abs(gap - impact.threshold_ev) > THRESHOLD_TOLERANCE_EV:
                raise ValueError(
                    f"{impact.lower} -> {impact.upper}: declared threshold "
                    f"{impact.threshold_ev} eV disagrees with the level gap {gap} eV. Both "
                    "come from the same level table, so one of them was transcribed from "
                    "somewhere else."
                )
        for radiative in self.radiative:
            for label in (radiative.lower, radiative.upper):
                if label not in known:
                    raise ValueError(
                        f"radiative channel references no level {label!r}; "
                        f"the system has {', '.join(known)}"
                    )
            if known[radiative.upper].energy_ev <= known[radiative.lower].energy_ev:
                raise ValueError(
                    f"{radiative.upper} -> {radiative.lower}: emission requires the upper "
                    "level to lie above the lower one"
                )

    # ── access ──────────────────────────────────────────────────────────────────

    @property
    def ground(self) -> Level:
        return self.levels[0]

    @property
    def excited(self) -> tuple[Level, ...]:
        """Every level except the ground state — the unknowns of the CR solve."""
        return self.levels[1:]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(level.label for level in self.levels)

    def level(self, label: str) -> Level:
        for candidate in self.levels:
            if candidate.label == label:
                return candidate
        raise KeyError(f"no level {label!r}; the system has {', '.join(self.labels)}")

    def degeneracies(self) -> dict[str, int]:
        return {level.label: level.degeneracy for level in self.levels}

    @property
    def has_trapping(self) -> bool:
        return any(channel.is_trapped for channel in self.radiative)

    def radiative_from(self, label: str) -> tuple[RadiativeChannel, ...]:
        """Every channel this level decays through — the ``sum_(l<u) A_ul`` of doc 04 §2.2."""
        return tuple(channel for channel in self.radiative if channel.upper == label)

    def __iter__(self) -> Iterator[Level]:
        return iter(self.levels)

    def __len__(self) -> int:
        return len(self.levels)

    def __repr__(self) -> str:
        trapped = sum(1 for c in self.radiative if c.is_trapped)
        return (
            f"LevelSystem({len(self.levels)} levels, {len(self.electron_impact)} impact, "
            f"{len(self.radiative)} radiative, {trapped} trapped)"
        )
