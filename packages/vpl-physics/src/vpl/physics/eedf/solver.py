"""The two-term Boltzmann solve — doc 03 §3.2, doc 11 WBS 1.8.

Doc 03 §3.2 requires the framework to compute "the EEDF and the resulting rate
coefficients with a **two-term Boltzmann solver** ... over the local reduced field
``E/N``". Doc 08 §2's build/buy table names BOLSIG+ or ``bolos``; neither is usable here,
and ADR-009 records why and what was done instead. What follows is the two-term expansion
as Hagelaar and Pitchford publish it (*Plasma Sources Sci. Technol.* **14** (2005) 722),
which doc 00 C2 permits because it is textbook rather than novel.

## The equations, as implemented

Expand the velocity distribution in Legendre polynomials of the field direction and keep
two terms, ``f(v) = f0(eps) + cos(theta) f1(eps)``. For a steady, spatially uniform,
DC-driven electron swarm this collapses to a convection-diffusion equation in energy for
``f0`` alone, with the anisotropic part eliminated. Writing energies in eV, dropping the
gas density ``N`` throughout (so time is reduced time ``N t`` and every output is a
reduced quantity), and with ``gamma = sqrt(2 e / m_e)``::

    d/deps [ W(eps) f0 - D(eps) df0/deps ]  =  S_inelastic(eps)

    W(eps) = -gamma eps**2 sigma_eps(eps)                            (elastic drag, down)
    D(eps) = (gamma/3) (E/N)**2 eps / sigma_m(eps)                   (field heating, up)
             + gamma kT_g eps**2 sigma_eps(eps)                      (thermal diffusion)
    sigma_eps = 2 (m_e/M) sigma_m                                    (energy-loss section)

Two limits fall straight out of it and are the verification anchors of
``test_eedf_solver.py``:

- ``E/N -> 0``: the flux vanishes, so ``W f0 = D df0/deps``, giving
  ``df0/deps = -f0/kT_g`` — a **Maxwellian at the gas temperature**, whatever ``sigma_m``
  is.
- ``T_g = 0`` with constant ``sigma_m``: ``df0/deps = -3 (2m_e/M) sigma_m**2 eps / (E/N)**2``,
  giving ``f0 ~ exp(-(eps/eps_0)**2)`` with ``eps_0 = (E/N) / (sigma_m sqrt(3 m_e/M))`` —
  the **Druyvesteyn** distribution of doc 03 §3.2, with a closed-form scale.

The transport coefficients and rate coefficients are the standard moments::

    mu_e N  = -(gamma/3) integral (eps/sigma_m) (df0/deps) deps      [1 / (m V s)]
    D_e N   =  (gamma/3) integral (eps/sigma_m) f0 deps              [1 / (m s)]
    <eps>   =  integral eps**(3/2) f0 deps                           [eV]
    k       =  gamma integral eps sigma(eps) f0 deps                 [m^3 / s]

## Discretisation: the exponential scheme, and why

Finite volume on the grid of :mod:`vpl.physics.eedf.grid`, with the energy-space flux
evaluated at cell boundaries by the exponential (Scharfetter-Gummel) scheme — the same
device doc 03 §3.4 mandates for ion advection in L1, for the same reason. Solving
``W f - D f' = F`` exactly across one cell interval gives

    F_j = a0_j f_(j-1) + a1_j f_j ,
    a0 = W / (1 - exp(-z)) ,  a1 = W / (1 - exp(z)) ,  z = W h / D

which is exact for an exponential ``f0`` and degenerates gracefully to pure upwinding when
the drag dominates and to a centred difference when diffusion does. A centred difference
would oscillate wherever ``|z| > 2``, which in the tail is everywhere.

The scheme's accuracy is better than it looks: on a uniform grid the cell boundary is the
exact midpoint of the two adjacent cell centres, so accumulating ``z`` along the grid is a
midpoint rule for ``integral (W/D) deps`` and the scheme reproduces both analytic limits
above to second order. That is measured, not asserted — see the convergence study in
``test_eedf_solver.py``.

## Inelastic collisions: conservative redistribution

An electron at ``eps`` that excites a level of threshold ``u`` reappears at ``eps - u``. So
a cell ``[b_j, b_(j+1)]`` empties at rate ``gamma sigma_k(eps_j) eps_j d_j f_j`` and the
same electrons reappear spread over ``[b_j - u, b_(j+1) - u]``, which in general straddles
several cells. They are distributed by **overlap length**, which makes the operator
conserve particles exactly rather than to the accuracy of an interpolation — and exact
conservation is what lets the normalisation of ``f0`` be a check rather than a hope.

Ionisation additionally *creates* an electron, and where that electron appears is a real
modelling choice rather than a detail; see :class:`IonisationSharing`.

## The solve: inverse iteration, and what the eigenvalue means

Without ionisation the discrete operator conserves particles, so the steady state is its
null vector. With ionisation it does not, and there is no exact steady state: the electron
density grows. Writing ``f0 ~ exp(nu_bar N t)`` turns the problem into the generalised
eigenproblem ``M f = nu_bar diag(mass) f``, whose dominant eigenpair is what this solver
computes by inverse iteration with a growing shift. ``nu_bar`` is then the net reduced
ionisation frequency, and summing the rows of ``M`` shows it must equal the ionisation
rate coefficient exactly — an identity the tests check, because it is violated by any
inconsistency between the redistribution and the rate integral.

## What is deliberately not modelled

Stated here rather than discovered later (doc 00 C4):

- **No growth-model correction to** ``sigma_m``. Hagelaar and Pitchford's temporal growth
  model also replaces ``sigma_m`` by ``sigma_m + nu_bar / (gamma sqrt(eps))`` in the
  ``f1`` relation. That correction is of order ``nu_bar / (N gamma sqrt(eps) sigma_m)``,
  the ratio of the ionisation frequency to the momentum-transfer frequency. It is omitted,
  and the omission is **bounded rather than assumed small**:
  :attr:`EedfSolution.growth_correction` reports the ratio for every solve. On the
  argon-like synthetic gas of the tests it measures 5e-11 at 5 Td, 3e-5 at 20 Td, 1.5e-4 at
  40 Td and 5.6e-4 at 80 Td — small in the range doc 01 §2.1 operates in, and *growing*, so
  a run far up the ``E/N`` axis should read the number rather than trust this sentence.
- **No superelastic collisions.** Recovering energy from excited states needs the excited
  populations, which is doc 04 §2.2's CR model — a downstream consumer of this module, not
  an input to it. Valid while the metastable fraction is small; doc 05 §2.2 makes that
  fraction an inferred nuisance parameter, so the assumption is at least visible.
- **No electron-electron collisions.** Negligible at the ionisation degrees of doc 01
  §2.1 (``n_e/n_g ~ 10^-6``), where they are the standard omission.
- **No magnetic field, no anisotropic scattering beyond momentum transfer.** The former is
  outside doc 03's scope; the latter is what the momentum-transfer cross section *is*.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.core.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE
from vpl.core.params import default_registry
from vpl.core.units import Q_, magnitude_in
from vpl.physics.eedf.analytic import maxwellian_eedf
from vpl.physics.eedf.grid import EnergyGrid
from vpl.physics.eedf.kinetics import ElectronKinetics, InelasticChannel

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_RELATIVE_TOLERANCE",
    "GAMMA",
    "GRID_TRUNCATION_TOLERANCE",
    "ROOM_TEMPERATURE_EV",
    "TOWNSEND_V_M2",
    "EedfConvergenceError",
    "EedfSolution",
    "IonisationSharing",
    "TwoTermSolver",
]

type FloatArray = NDArray[np.float64]

# ── constants ───────────────────────────────────────────────────────────────────
#
# Unwrapped from vpl.core.constants exactly once, here. doc 00 C1 forbids a numeric
# literal with physical meaning anywhere in physics code; doc 08 §5 puts bare arrays in
# hot loops and quantities at module boundaries. Both hold if the unwrapping happens at
# import and nowhere else.

_REGISTRY: Final = default_registry()

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E_KG: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))
_K_B_J_PER_K: Final[float] = float(magnitude_in(BOLTZMANN, "J/K"))

#: ``gamma = sqrt(2 e / m_e)``, in m s^-1 eV^-1/2 — the speed of a 1 eV electron divided by
#: ``sqrt(1 eV)``. Every integral in this module carries it, because energies are in eV and
#: speeds in m/s and this is the conversion between them.
GAMMA: Final[float] = float(np.sqrt(2.0 * _E_C / _M_E_KG))

#: One townsend in SI, from the unit :mod:`vpl.core.units` defines for exactly this
#: purpose. Read from the registry rather than written as ``1e-21`` so that the definition
#: lives in one place.
TOWNSEND_V_M2: Final[float] = float(magnitude_in(Q_(1.0, "Td"), "V * m**2"))

#: The neutral temperature of reference point RP-1 (doc 01 §2.1), as an energy. The gas
#: temperature is what the EEDF relaxes to at vanishing field, so a default of zero would
#: make the ``E/N -> 0`` limit unphysical rather than merely approximate.
ROOM_TEMPERATURE_EV: Final[float] = _K_B_J_PER_K * float(_REGISTRY.value_in("RP1.T_g", "K")) / _E_C

#: Convergence criterion: the change in ``f0`` between successive iterations, measured in
#: the same ``integral . sqrt(eps) deps`` norm the distribution is normalised in. Both
#: iterates are normalised, so this is a fraction of the population that moved.
DEFAULT_RELATIVE_TOLERANCE: Final[float] = 1e-10

#: Iteration budget. Generous: a solve that needs many is a solve whose grid or operating
#: point is unusual, and failing it early would hide that behind a convergence error.
DEFAULT_MAX_ITERATIONS: Final[int] = 400

#: Population fraction allowed in the top tenth of the energy grid before
#: :attr:`EedfSolution.is_grid_long_enough` reports the grid as too short. The upper
#: boundary condition is zero flux at ``eps_max``; if an appreciable population is still
#: there, that condition is fiction and every moment inherits the error.
GRID_TRUNCATION_TOLERANCE: Final[float] = 1e-6

#: Fraction of the grid, measured from the top, that the truncation check looks at.
_TRUNCATION_WINDOW: Final[float] = 0.1

#: Initial shift for the inverse iteration, in units of the inverse of the fastest
#: collisional rate in the problem. Purely algorithmic: it changes how many iterations are
#: needed and never the converged answer, which is the fixed point of an equation the shift
#: does not appear in. Measured over 1 to 1e10 on the elastic, ionising and thermalised
#: cases, the converged mean energy is identical to every digit printed and only the
#: iteration count moves; 1e4 is near the middle of the flat region for all three.
_INITIAL_SHIFT: Final[float] = 1e4

#: Factor the shift grows by each accepted iteration. Larger shifts separate the dominant
#: eigenvector faster; the growth stops when the linear system starts losing positivity.
_SHIFT_GROWTH: Final[float] = 4.0

#: Factor the shift shrinks by when an iterate comes back non-positive, which is how an
#: over-large shift announces that the linear system has become too ill-conditioned.
_SHIFT_REDUCTION: Final[float] = 8.0

#: Largest total growth of the shift, relative to its initial value.
_MAX_SHIFT_GROWTH: Final[float] = 1e14

#: Negative values below this fraction of the peak are round-off in the far tail of a
#: distribution spanning hundreds of decades, and are clipped. Anything larger is a real
#: loss of positivity and reduces the shift instead.
_NEGATIVITY_TOLERANCE: Final[float] = 1e-10

#: Mean energy of the default initial guess, in eV. Only a starting point; the converged
#: answer does not depend on it, and :meth:`TwoTermSolver.solve` takes an explicit one.
_INITIAL_MEAN_ENERGY_EV: Final[float] = 1.5

#: Electrons produced per ionisation event, counting the scattered one.
_ELECTRONS_PER_IONISATION: Final[int] = 2


class EedfConvergenceError(RuntimeError):
    """The inverse iteration did not reach the requested tolerance.

    Deliberately an error rather than a returned flag with a best-effort EEDF. A partly
    converged distribution is a plausible-looking array that produces plausible-looking
    rate coefficients, and doc 03 §3.2 sends those straight into the fluid model.
    """


class IonisationSharing(StrEnum):
    """Where the two electrons leaving an ionising collision appear in energy.

    Not a formality. An ionisation at ``eps`` leaves ``eps - u`` to be divided between the
    scattered and the ejected electron, and the division sets the low-energy population,
    which sets the mobility. BOLSIG+ offers the same choice; the framework makes it a
    manifest field (doc 08 §1 principle 4) so the ablation of doc 07 can sweep it rather
    than inheriting whatever the default was.
    """

    #: The scattered electron keeps all of ``eps - u``; the ejected electron appears at
    #: zero energy. The usual default, and exact in the limit of a strongly forward-peaked
    #: ionisation differential cross section near threshold.
    ONE_TAKES_ALL = "one_takes_all"

    #: Both electrons leave with ``(eps - u)/2``. The opposite extreme, and the one that
    #: brackets the first: any real sharing distribution lies between them.
    EQUAL_SHARING = "equal_sharing"


# ── the assembled operator ──────────────────────────────────────────────────────


def _flux_coefficients(
    *, drift: FloatArray, diffusion: FloatArray, spacing: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """``a0``, ``a1`` of the exponential scheme at the interior cell boundaries.

    ``F_j = a0_j f_(j-1) + a1_j f_j`` is the exact flux across ``[c_(j-1), c_j]`` for
    constant ``W`` and ``D``. Both asymptotes and the ``z -> 0`` limit are taken
    explicitly rather than left to floating point: ``exp(-z)`` overflows for ``z`` below
    about ``-709``, and ``1 - exp(0)`` is a division by zero. Either would arrive as a nan
    inside a rate coefficient.
    """
    z = np.clip(drift * spacing / diffusion, -np.log(np.finfo(np.float64).max), None)
    z = np.clip(z, None, np.log(np.finfo(np.float64).max))

    negligible = np.abs(z) < np.sqrt(np.finfo(np.float64).eps)
    # Where z is negligible the scheme degenerates to a centred difference, F = D (f_(j-1)
    # - f_j)/h. The 1.0 keeps the division defined at the points that branch takes.
    lower = np.where(negligible, 1.0, -np.expm1(-z))
    upper = np.where(negligible, 1.0, -np.expm1(z))
    centred = diffusion / spacing

    return (
        np.asarray(np.where(negligible, centred, drift / lower)),
        np.asarray(np.where(negligible, -centred, drift / upper)),
    )


def _redistribution(grid: EnergyGrid, *, lower_ev: FloatArray, upper_ev: FloatArray) -> FloatArray:
    """Conservative overlap weights taking each source cell onto the destination cells.

    ``weights[i, j]`` is the fraction of the electrons leaving cell ``j`` that land in cell
    ``i``, given that they arrive spread uniformly over ``[lower[j], upper[j]]``. Columns
    sum to one exactly, which is what makes the inelastic operator conserve particles to
    machine precision rather than to the accuracy of an interpolation.

    A degenerate destination interval — which happens only for cells whose cross section
    is already zero — is sent entirely to the lowest cell, so that no column ever sums to
    something other than one.
    """
    left = grid.boundaries_ev[:-1, np.newaxis]
    right = grid.boundaries_ev[1:, np.newaxis]

    overlap = np.clip(upper_ev[np.newaxis, :], left, right) - np.clip(
        lower_ev[np.newaxis, :], left, right
    )
    span = upper_ev - lower_ev
    weights = overlap / np.where(span > 0.0, span, 1.0)
    weights[0, span <= 0.0] = 1.0
    return np.asarray(weights)


@dataclass(frozen=True, slots=True, eq=False)
class EedfSolution:
    """One converged EEDF and everything doc 03 §3.2 tabulates from it.

    Attributes:
        grid: The energy grid ``f0`` is sampled on.
        f0: The isotropic part of the distribution at cell centres, normalised so that
            ``integral f0 sqrt(eps) deps = 1``. **This is the EEDF itself, not a summary of
            it** — doc 04 §4.2 computes the Thomson spectrum from it and states explicitly
            that the framework "does not assume a Maxwellian".
        reduced_field_td: The ``E/N`` this was solved at, in townsend.
        database: Which of doc 09 §2.1's electron sets produced it.
        mean_energy_ev: ``<eps>``. The doc 03 §3.2 output; also ``1.5 k T_e`` if and only
            if the distribution happens to be Maxwellian, which is the whole point.
        reduced_mobility: ``mu_e N``, in ``1 / (m V s)``. Positive: the magnitude.
        reduced_diffusion: ``D_e N``, in ``1 / (m s)``.
        rate_coefficients: ``k`` in m^3/s, keyed by reaction — ``k_iz`` and every
            ``k_ex,j`` of doc 03 §3.2, and the input to doc 04 §2.2's ``K_ju``.
        net_ionisation_frequency: The dominant eigenvalue, in m^3/s. Equal to the total
            ionisation rate coefficient by an exact discrete identity.
        momentum_transfer_frequency: ``gamma <eps sigma_m>``, in m^3/s. Carried only so
            that :attr:`growth_correction` can bound the growth-model omission.
        iterations: How many inverse iterations were taken.
        residual: The final iterate-to-iterate change, in the normalisation norm.
    """

    grid: EnergyGrid
    f0: FloatArray
    reduced_field_td: float
    database: str
    mean_energy_ev: float
    reduced_mobility: float
    reduced_diffusion: float
    rate_coefficients: Mapping[str, float]
    net_ionisation_frequency: float
    momentum_transfer_frequency: float
    iterations: int
    residual: float
    ionisation_reactions: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rate_coefficients", MappingProxyType(dict(self.rate_coefficients))
        )

    # ── derived transport quantities ────────────────────────────────────────────

    @property
    def effective_temperature_ev(self) -> float:
        """``(2/3) <eps>`` — the temperature a Maxwellian of this mean energy would have.

        A *convenience*, and one to use carefully: doc 04 §4.2 warns that "fitting a
        Maxwellian to a bi-Maxwellian plasma yields a ``T_e`` that corresponds to neither
        population". This number is that fit, and it is exactly as meaningful as the
        distribution is Maxwellian.
        """
        return (2.0 / 3.0) * self.mean_energy_ev

    @property
    def characteristic_energy_ev(self) -> float:
        """``D_e / mu_e``, the Townsend characteristic energy, in eV.

        Equals ``k T_e / e`` when the distribution is Maxwellian — the Einstein relation —
        and departs from :attr:`effective_temperature_ev` by exactly the amount the
        distribution is not one. That difference is a measurement of the doc 03 §3.2
        non-Maxwellian correction rather than an assumption about it.
        """
        return self.reduced_diffusion / self.reduced_mobility

    @property
    def drift_velocity_m_per_s(self) -> float:
        """``mu_e E``, the magnitude of the electron drift velocity."""
        return self.reduced_mobility * self.reduced_field_td * TOWNSEND_V_M2

    @property
    def ionisation_rate_coefficient(self) -> float:
        """``k_iz`` — every ionisation channel summed. Zero when there is none."""
        return float(sum(self.rate_coefficients[r] for r in self.ionisation_reactions))

    @property
    def excitation_rate_coefficients(self) -> Mapping[str, float]:
        """``k_ex,j`` per channel — the doc 04 §2.2 CR-model input."""
        return MappingProxyType(
            {
                reaction: value
                for reaction, value in self.rate_coefficients.items()
                if reaction not in self.ionisation_reactions
            }
        )

    # ── the distribution itself ─────────────────────────────────────────────────

    @property
    def energy_distribution(self) -> FloatArray:
        """``F(eps) = f0 sqrt(eps)``, normalised so that ``integral F deps = 1``.

        The EEDF in the convention that integrates against ``deps`` alone. :attr:`f0` is
        the electron energy *probability* function; this is the electron energy
        *distribution* function, and confusing the two is a factor of ``sqrt(eps)`` in
        every line ratio.
        """
        return np.asarray(self.f0 * np.sqrt(self.grid.centres_ev))

    def tail_fraction_above(self, energy_ev: float) -> float:
        """The fraction of electrons above ``energy_ev``.

        The number doc 03 §3.2 is about: "the high-energy tail — the part that matters for
        ionisation and for OES line ratios — is depleted". Compare it against the same
        fraction for a Maxwellian of the same mean energy to see by how much.
        """
        if energy_ev < 0.0:
            raise ValueError(f"an energy threshold cannot be negative, got {energy_ev}")
        above = self.grid.centres_ev >= energy_ev
        return float(np.dot(self.f0[above], self.grid.cell_masses[above]))

    @property
    def is_grid_long_enough(self) -> bool:
        """Whether the zero-flux condition at the top of the grid is defensible.

        Returned rather than warned about, and checked rather than assumed: the upper
        boundary condition is that no electrons leave the top of the grid, and a grid too
        short for its field violates that silently. The result is still a smooth,
        normalised, entirely wrong distribution — which is the classic way a Boltzmann
        solve produces a plausible answer.
        """
        window = (1.0 - _TRUNCATION_WINDOW) * self.grid.max_ev
        return self.tail_fraction_above(window) < GRID_TRUNCATION_TOLERANCE

    @property
    def growth_correction(self) -> float:
        """``nu_bar / nu_m`` — the size of the growth-model term this solver omits.

        See the module docstring. Hagelaar and Pitchford's temporal growth model corrects
        ``sigma_m`` by this ratio; omitting it is defensible exactly while this number is
        small, so it is reported rather than argued about.
        """
        if self.momentum_transfer_frequency <= 0.0:
            return 0.0
        return self.net_ionisation_frequency / self.momentum_transfer_frequency

    def __repr__(self) -> str:
        return (
            f"EedfSolution({self.database!r}, E/N = {self.reduced_field_td:g} Td, "
            f"<eps> = {self.mean_energy_ev:.4g} eV, {self.iterations} iterations)"
        )


@dataclass(frozen=True, slots=True)
class TwoTermSolver:
    """The doc 03 §3.2 solver, over one gas on one energy grid.

    Attributes:
        kinetics: The cross sections, already on the grid.
        gas_temperature_ev: ``k T_g / e``. Defaults to RP-1's 300 K
            (:data:`ROOM_TEMPERATURE_EV`). Zero is allowed and gives the cold-gas limit in
            which the Druyvesteyn form is exact.
        ionisation_sharing: Where the products of an ionising collision appear.
        relative_tolerance: Convergence criterion on the iterate-to-iterate change.
        max_iterations: Iteration budget before :class:`EedfConvergenceError`.
    """

    kinetics: ElectronKinetics
    gas_temperature_ev: float = ROOM_TEMPERATURE_EV
    ionisation_sharing: IonisationSharing = IonisationSharing.ONE_TAKES_ALL
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    def __post_init__(self) -> None:
        if self.gas_temperature_ev < 0.0:
            raise ValueError(
                f"the gas temperature cannot be negative, got {self.gas_temperature_ev} eV"
            )
        if not self.relative_tolerance > 0.0:
            raise ValueError(
                f"the convergence tolerance must be positive, got {self.relative_tolerance}"
            )
        if self.max_iterations < 1:
            raise ValueError(f"the iteration budget must be at least 1, got {self.max_iterations}")

    @property
    def grid(self) -> EnergyGrid:
        return self.kinetics.grid

    # ── operator assembly ───────────────────────────────────────────────────────

    def _elastic_operator(self, reduced_field_si: float) -> FloatArray:
        """The tridiagonal part: elastic drag, field heating and thermal diffusion."""
        grid = self.grid
        interior = grid.boundaries_ev[1:-1]
        sigma_m = self.kinetics.momentum_transfer_edge_m2[1:-1]
        sigma_eps = 2.0 * self.kinetics.mass_ratio * sigma_m

        drift = -GAMMA * interior**2 * sigma_eps
        diffusion = (GAMMA / 3.0) * reduced_field_si**2 * interior / sigma_m + (
            GAMMA * self.gas_temperature_ev * interior**2 * sigma_eps
        )
        a0, a1 = _flux_coefficients(
            drift=drift, diffusion=diffusion, spacing=grid.centre_spacing_ev
        )

        n_cells = grid.n_cells
        operator = np.zeros((n_cells, n_cells), dtype=np.float64)
        index = np.arange(n_cells - 1)
        # dP_j/dt = -(F_(j+1) - F_j): the flux at an interior boundary leaves the cell
        # below it and enters the cell above it, with equal and opposite sign. That is
        # what makes the column sums vanish and the operator conserve particles.
        operator[index, index] -= a0
        operator[index, index + 1] -= a1
        operator[index + 1, index] += a0
        operator[index + 1, index + 1] += a1
        return operator

    def _loss_rate(self, channel: InelasticChannel) -> FloatArray:
        """``gamma sigma_k eps d_j`` — the cell-integrated collision rate, per unit ``f0``."""
        grid = self.grid
        return np.asarray(GAMMA * channel.sigma_m2 * grid.centres_ev * grid.widths_ev)

    def _inelastic_operator(self) -> FloatArray:
        """Loss from each cell and conservative redistribution of the products.

        Independent of ``E/N``: the cross sections and thresholds do not depend on the
        field, only the population that samples them does.
        """
        grid = self.grid
        operator = np.zeros((grid.n_cells, grid.n_cells), dtype=np.float64)
        diagonal = np.arange(grid.n_cells)

        for channel in self.kinetics.channels:
            loss = self._loss_rate(channel)
            operator[diagonal, diagonal] -= loss

            shifted_lower = np.maximum(grid.boundaries_ev[:-1] - channel.threshold_ev, 0.0)
            shifted_upper = np.maximum(grid.boundaries_ev[1:] - channel.threshold_ev, 0.0)

            if channel.is_ionisation and self.ionisation_sharing is IonisationSharing.EQUAL_SHARING:
                weights = _redistribution(
                    grid, lower_ev=0.5 * shifted_lower, upper_ev=0.5 * shifted_upper
                )
                operator += _ELECTRONS_PER_IONISATION * weights * loss
                continue

            weights = _redistribution(grid, lower_ev=shifted_lower, upper_ev=shifted_upper)
            operator += weights * loss
            if channel.is_ionisation:
                # One-takes-all: the ejected electron appears at zero energy.
                operator[0, :] += loss

        return operator

    # ── the solve ───────────────────────────────────────────────────────────────

    def _initial_guess(self, initial: ArrayLike | None) -> FloatArray:
        if initial is None:
            mean = max(_INITIAL_MEAN_ENERGY_EV, 1.5 * self.gas_temperature_ev)
            guess = maxwellian_eedf(
                self.grid.centres_ev, electron_temperature_ev=(2.0 / 3.0) * mean
            )
            # A grid far colder or hotter than the guess can underflow it entirely; a
            # uniform distribution is a poor but non-degenerate fallback.
            if not np.any(guess > 0.0):
                guess = np.ones(self.grid.n_cells, dtype=np.float64)
            return self.grid.normalise(guess)
        return self.grid.normalise(np.asarray(initial, dtype=np.float64))

    def solve(self, reduced_field_td: float, *, initial: ArrayLike | None = None) -> EedfSolution:
        """Solve for ``f0`` at one reduced field.

        Args:
            reduced_field_td: ``E/N`` in townsend. Strictly positive: the zero-field limit
                is the Maxwellian at the gas temperature and is reached by taking a small
                value, not by passing zero, because at exactly zero the drift and the
                diffusion both vanish and the equation is degenerate.
            initial: A starting ``f0``, one value per cell. Normally the solution at the
                previous field of a sweep; the converged answer does not depend on it.

        Returns:
            The converged EEDF and its moments.

        Raises:
            ValueError: If the field is not positive or the initial guess is the wrong
                length.
            EedfConvergenceError: If the iteration did not reach
                :attr:`relative_tolerance` within :attr:`max_iterations`.
        """
        if not reduced_field_td > 0.0:
            raise ValueError(
                f"the reduced field must be strictly positive, got {reduced_field_td} Td. "
                "The zero-field limit is approached, not evaluated: at E/N = 0 the "
                "energy-space equation has no field-heating term to balance the drag."
            )

        grid = self.grid
        operator = self._elastic_operator(reduced_field_td * TOWNSEND_V_M2)
        operator += self._inelastic_operator()

        mass = grid.cell_masses
        f0 = self._initial_guess(initial)

        scale = float(np.max(np.abs(np.diag(operator)) / mass))
        shift = _INITIAL_SHIFT / scale if scale > 0.0 else _INITIAL_SHIFT
        ceiling = shift * _MAX_SHIFT_GROWTH

        iterations = 0
        residual = np.inf
        for _ in range(self.max_iterations):
            iterations += 1
            candidate = np.linalg.solve(np.diag(mass) - shift * operator, mass * f0)

            peak = float(np.max(candidate)) if candidate.size else 0.0
            if (
                not np.all(np.isfinite(candidate))
                or peak <= 0.0
                or bool(np.any(candidate < -_NEGATIVITY_TOLERANCE * peak))
            ):
                shift /= _SHIFT_REDUCTION
                continue

            candidate = grid.normalise(np.maximum(candidate, 0.0))
            residual = grid.moment(np.abs(candidate - f0), 0)
            f0 = candidate
            if residual < self.relative_tolerance:
                break
            shift = min(shift * _SHIFT_GROWTH, ceiling)
        else:
            raise EedfConvergenceError(
                f"the two-term solve did not converge at E/N = {reduced_field_td} Td: "
                f"residual {residual:.3g} after {iterations} iterations, tolerance "
                f"{self.relative_tolerance:g}. A grid too short for the field is the "
                "usual cause; check EedfSolution.is_grid_long_enough on a coarser run."
            )

        return self._assemble(
            f0=f0,
            operator=operator,
            reduced_field_td=reduced_field_td,
            iterations=iterations,
            residual=residual,
        )

    # ── moments ─────────────────────────────────────────────────────────────────

    def _assemble(
        self,
        *,
        f0: FloatArray,
        operator: FloatArray,
        reduced_field_td: float,
        iterations: int,
        residual: float,
    ) -> EedfSolution:
        grid = self.grid
        rates = {
            channel.reaction: float(np.dot(self._loss_rate(channel), f0))
            for channel in self.kinetics.channels
        }
        return EedfSolution(
            grid=grid,
            f0=f0,
            reduced_field_td=reduced_field_td,
            database=self.kinetics.database,
            mean_energy_ev=grid.moment(f0, 1),
            reduced_mobility=self._reduced_mobility(f0),
            reduced_diffusion=self._reduced_diffusion(f0),
            rate_coefficients=rates,
            # Summing every row of the operator leaves only the net particle creation,
            # which is the ionisation rate integral. Computed this way rather than from
            # the rates so that the identity the tests check is not a tautology.
            net_ionisation_frequency=float(np.sum(operator @ f0)),
            momentum_transfer_frequency=float(
                GAMMA
                * np.dot(self.kinetics.momentum_transfer_m2 * grid.centres_ev * grid.widths_ev, f0)
            ),
            iterations=iterations,
            residual=residual,
            ionisation_reactions=frozenset(c.reaction for c in self.kinetics.ionisation),
        )

    def _reduced_mobility(self, f0: FloatArray) -> float:
        """``-(gamma/3) integral (eps/sigma_m) df0/deps deps``, in ``1 / (m V s)``.

        For a piecewise-constant ``f0`` the derivative is a sum of jumps at the interior
        cell boundaries, so the integral is exactly this sum — no quadrature error, and
        the closed-form constant-collision-frequency result comes out to machine
        precision.
        """
        grid = self.grid
        interior = grid.boundaries_ev[1:-1]
        weight = interior / self.kinetics.momentum_transfer_edge_m2[1:-1]
        return float(-(GAMMA / 3.0) * np.dot(weight, np.diff(f0)))

    def _reduced_diffusion(self, f0: FloatArray) -> float:
        """``(gamma/3) integral (eps/sigma_m) f0 deps``, in ``1 / (m s)``."""
        grid = self.grid
        weight = grid.centres_ev / self.kinetics.momentum_transfer_m2
        return float((GAMMA / 3.0) * np.dot(weight * grid.widths_ev, f0))
