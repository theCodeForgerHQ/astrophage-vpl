"""Tabulation over ``E/N``, and the three-database spread — doc 03 §3.2, doc 09 §2.1.

Doc 03 §3.2 states the deliverable of this package exactly::

    k_iz(E/N),  k_ex,j(E/N),  mu_e(E/N),  D_e(E/N),  <eps>(E/N)

:class:`RateTable` is that, as one object, keyed so that doc 04 §2.2's CR model can look a
channel up by the reaction it is named by rather than by a position in a list.

## Why the spread is a returned number

Doc 09 §2.1 keeps three independent electron databases and says why:

    Running the inference under all three and reporting the spread is an honest measure of
    atomic-data uncertainty — and it is a term in the error budget (doc 06 §4, term 2)
    rather than an unstated risk. Most work picks one set and never mentions the choice.

Doc 06 §4 term 2 budgets 7.5 % for ``T_e`` inference, of which the atomic-data spread is a
component. A spread that has to be assembled by hand from three separate runs gets
assembled once, by whoever writes the paper, and then never again; so
:func:`tabulate_each_electron_set` is one call and :meth:`DatabaseSpread.relative_spread`
returns an array rather than a plot.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.physics.atomic.dataset import ElectronDatabase
from vpl.physics.atomic.interpolation import ExtrapolationPolicy
from vpl.physics.atomic.lxcat import CrossSectionSet
from vpl.physics.eedf.grid import EnergyGrid
from vpl.physics.eedf.kinetics import DEFAULT_ABOVE_GRID, DEFAULT_BELOW_GRID, kinetics_from_set
from vpl.physics.eedf.solver import (
    ROOM_TEMPERATURE_EV,
    IonisationSharing,
    TwoTermSolver,
)

__all__ = [
    "DatabaseSpread",
    "ElectronSetSource",
    "RateTable",
    "TabulatedQuantity",
    "tabulate",
    "tabulate_each_electron_set",
    "tabulate_electron_sets",
]

FloatArray: TypeAlias = NDArray[np.float64]


class TabulatedQuantity(StrEnum):
    """The columns of doc 03 §3.2's table, addressable by name.

    A ``StrEnum`` so that a manifest can ask for a spread on ``ionisation_rate`` and get a
    checked member back (doc 08 §1 principle 4), and so that
    :meth:`DatabaseSpread.summary` can iterate the quantities rather than repeat them.
    """

    MEAN_ENERGY = "mean_energy"
    REDUCED_MOBILITY = "reduced_mobility"
    REDUCED_DIFFUSION = "reduced_diffusion"
    IONISATION_RATE = "ionisation_rate"


class ElectronSetSource(Protocol):
    """What :func:`tabulate_each_electron_set` needs of an atomic-data store.

    Structural rather than nominal, because doc 09 §5 says the data-access layer is
    "designed to be swappable from the outset": a product build that fetches from a
    licensed mirror implements this and nothing else changes.
    :class:`~vpl.physics.atomic.store.AtomicDataStore` satisfies it already.
    """

    def each_electron_set(self) -> Iterator[tuple[ElectronDatabase, CrossSectionSet]]: ...


def _locked(values: ArrayLike) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, eq=False)
class RateTable:
    """Doc 03 §3.2's table for one gas and one database.

    Attributes:
        database: Which of doc 09 §2.1's electron sets produced it.
        reduced_field_td: The ``E/N`` axis, in townsend, strictly increasing.
        mean_energy_ev: ``<eps>(E/N)``.
        reduced_mobility: ``mu_e N (E/N)``, in ``1 / (m V s)``.
        reduced_diffusion: ``D_e N (E/N)``, in ``1 / (m s)``.
        rate_coefficients: ``k(E/N)`` in m^3/s per reaction — every channel, ionising or
            not.
        ionisation_reactions: Which of those keys are ionisation, so that ``k_iz`` and the
            ``k_ex,j`` can be separated without re-reading the cross sections.
        iterations: Inverse iterations spent at each field. Reported because a sweep whose
            iteration count climbs is a sweep running out of grid.
    """

    database: str
    reduced_field_td: FloatArray
    mean_energy_ev: FloatArray
    reduced_mobility: FloatArray
    reduced_diffusion: FloatArray
    rate_coefficients: Mapping[str, FloatArray]
    ionisation_reactions: frozenset[str]
    iterations: NDArray[np.int64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reduced_field_td", _locked(self.reduced_field_td))
        object.__setattr__(
            self,
            "rate_coefficients",
            MappingProxyType({k: _locked(v) for k, v in self.rate_coefficients.items()}),
        )

    def __len__(self) -> int:
        return int(self.reduced_field_td.size)

    @property
    def ionisation(self) -> FloatArray:
        """``k_iz(E/N)`` — every ionisation channel summed. Zeros when there is none."""
        total = np.zeros(len(self), dtype=np.float64)
        for reaction in sorted(self.ionisation_reactions):
            total = total + self.rate_coefficients[reaction]
        return total

    @property
    def excitations(self) -> Mapping[str, FloatArray]:
        """``k_ex,j(E/N)`` per channel — doc 04 §2.2's ``K_ju``, before the CR model."""
        return MappingProxyType(
            {k: v for k, v in self.rate_coefficients.items() if k not in self.ionisation_reactions}
        )

    @property
    def characteristic_energy_ev(self) -> FloatArray:
        """``D_e / mu_e (E/N)``. Equals ``k T_e / e`` only where the EEDF is Maxwellian."""
        return np.asarray(self.reduced_diffusion / self.reduced_mobility)

    def column(self, quantity: TabulatedQuantity) -> FloatArray:
        """One column by its manifest name."""
        if quantity is TabulatedQuantity.MEAN_ENERGY:
            return self.mean_energy_ev
        if quantity is TabulatedQuantity.REDUCED_MOBILITY:
            return self.reduced_mobility
        if quantity is TabulatedQuantity.REDUCED_DIFFUSION:
            return self.reduced_diffusion
        return self.ionisation

    def __repr__(self) -> str:
        return (
            f"RateTable({self.database!r}, {len(self)} points, "
            f"{self.reduced_field_td[0]:g} to {self.reduced_field_td[-1]:g} Td, "
            f"{len(self.rate_coefficients)} channels)"
        )


def tabulate(solver: TwoTermSolver, reduced_fields_td: ArrayLike) -> RateTable:
    """Solve at every ``E/N`` and assemble doc 03 §3.2's table.

    Each solve warm-starts from the previous field's EEDF, so that successive points
    approach the same fixed point along the same path rather than each finding their own
    iterate of it — which matters because the fluid model of doc 03 §3.1 interpolates in
    this table and a table with iterate-level jitter in it is not smooth.

    **It is not a speed optimisation, and measurement says so.** Over a twenty-point
    log-spaced sweep the warm-started run took 129 inverse iterations against 130 cold.
    The cost of a solve is set by the shift ramp of
    :class:`~vpl.physics.eedf.solver.TwoTermSolver`, not by the quality of the starting
    guess, and a docstring claiming otherwise would be a plausible untested assertion of
    exactly the kind doc 00 §6 warns about.

    Args:
        solver: The configured solver, carrying the gas and the grid.
        reduced_fields_td: Strictly increasing reduced fields, in townsend.

    Returns:
        The table.

    Raises:
        ValueError: If the fields are empty or not strictly increasing.
        ~vpl.physics.eedf.solver.EedfConvergenceError: Propagated from any point that did
            not converge, naming the field it failed at.
    """
    fields = np.asarray(reduced_fields_td, dtype=np.float64)
    if fields.ndim != 1 or fields.size == 0:
        raise ValueError("a rate table needs at least one reduced field")
    if fields.size > 1 and not np.all(np.diff(fields) > 0.0):
        raise ValueError(
            "the reduced fields must be strictly increasing; the sweep warm-starts each "
            "solve from the previous one, which only helps if they are ordered"
        )

    reactions = tuple(c.reaction for c in solver.kinetics.channels)
    rates: dict[str, list[float]] = {reaction: [] for reaction in reactions}
    mean_energy: list[float] = []
    mobility: list[float] = []
    diffusion: list[float] = []
    iterations: list[int] = []

    previous: FloatArray | None = None
    for reduced_field_td in fields:
        solution = solver.solve(float(reduced_field_td), initial=previous)
        previous = solution.f0
        mean_energy.append(solution.mean_energy_ev)
        mobility.append(solution.reduced_mobility)
        diffusion.append(solution.reduced_diffusion)
        iterations.append(solution.iterations)
        for reaction in reactions:
            rates[reaction].append(solution.rate_coefficients[reaction])

    return RateTable(
        database=solver.kinetics.database,
        reduced_field_td=fields,
        mean_energy_ev=np.asarray(mean_energy, dtype=np.float64),
        reduced_mobility=np.asarray(mobility, dtype=np.float64),
        reduced_diffusion=np.asarray(diffusion, dtype=np.float64),
        rate_coefficients={k: np.asarray(v, dtype=np.float64) for k, v in rates.items()},
        ionisation_reactions=frozenset(c.reaction for c in solver.kinetics.ionisation),
        iterations=np.asarray(iterations, dtype=np.int64),
    )


# ── the three-database spread — doc 09 §2.1, doc 06 §4 term 2 ───────────────────


@dataclass(frozen=True, slots=True, eq=False)
class DatabaseSpread:
    """One rate table per electron database, and the disagreement between them.

    Attributes:
        tables: Keyed by :class:`~vpl.physics.atomic.dataset.ElectronDatabase`. Every
            table must share the same ``E/N`` axis, because a spread between tables
            evaluated at different fields is not a spread.
    """

    tables: Mapping[ElectronDatabase, RateTable]

    def __post_init__(self) -> None:
        if not self.tables:
            raise ValueError("a spread needs at least one database")
        axes = [table.reduced_field_td for table in self.tables.values()]
        if any(not np.array_equal(axis, axes[0]) for axis in axes[1:]):
            raise ValueError(
                "every table in a spread must share the same reduced-field axis; "
                "comparing tables evaluated at different E/N is not a comparison"
            )
        object.__setattr__(self, "tables", MappingProxyType(dict(self.tables)))

    @property
    def reduced_field_td(self) -> FloatArray:
        """The shared ``E/N`` axis."""
        return next(iter(self.tables.values())).reduced_field_td

    def relative_spread(self, quantity: TabulatedQuantity) -> FloatArray:
        """``(max - min) / mean`` across the databases, at every ``E/N``.

        The doc 06 §4 term-2 contribution, as a function of the operating point rather
        than as a single remembered percentage. Zero where the databases agree, which is
        the honest answer and not a degenerate one.
        """
        columns = np.vstack([table.column(quantity) for table in self.tables.values()])
        mean = np.mean(columns, axis=0)
        span = np.max(columns, axis=0) - np.min(columns, axis=0)
        return np.asarray(np.where(mean != 0.0, span / np.where(mean != 0.0, mean, 1.0), 0.0))

    def max_relative_spread(self, quantity: TabulatedQuantity) -> float:
        """The worst case over the swept range — the number an error budget wants."""
        return float(np.max(self.relative_spread(quantity)))

    def summary(self) -> str:
        """A report of the spread, for doc 09 §6's per-run provenance record."""
        lines = [
            f"Atomic-data spread over {len(self.tables)} electron databases "
            f"(doc 09 §2.1; doc 06 §4 term 2), "
            f"E/N = {self.reduced_field_td[0]:g} to {self.reduced_field_td[-1]:g} Td",
            "  databases: " + ", ".join(sorted(db.value for db in self.tables)),
        ]
        lines.extend(
            f"  {quantity.value:<20} max relative spread {self.max_relative_spread(quantity):.3%}"
            for quantity in TabulatedQuantity
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DatabaseSpread({len(self.tables)} databases, {len(self.reduced_field_td)} points)"


def tabulate_electron_sets(
    sets: Iterable[tuple[ElectronDatabase, CrossSectionSet]],
    *,
    reduced_fields_td: ArrayLike,
    grid: EnergyGrid,
    mass_ratio: float | None = None,
    gas_temperature_ev: float = ROOM_TEMPERATURE_EV,
    ionisation_sharing: IonisationSharing = IonisationSharing.ONE_TAKES_ALL,
    below: ExtrapolationPolicy = DEFAULT_BELOW_GRID,
    above: ExtrapolationPolicy = DEFAULT_ABOVE_GRID,
) -> DatabaseSpread:
    """Run the doc 03 §3.2 tabulation once per electron database.

    Every solver setting except the cross sections is held fixed, which is the point: the
    spread that comes out is attributable to the databases and to nothing else.

    Args:
        sets: ``(database, cross sections)`` pairs, normally from
            :meth:`~vpl.physics.atomic.store.AtomicDataStore.each_electron_set`.
        reduced_fields_td: The shared ``E/N`` axis.
        grid: The shared energy grid.
        mass_ratio: ``m_e / M``, if the databases' own values are not to be used.
        gas_temperature_ev: ``k T_g / e``.
        ionisation_sharing: See :class:`~vpl.physics.eedf.solver.IonisationSharing`.
        below: Extrapolation policy below the tabulated range.
        above: Extrapolation policy above it.

    Returns:
        One table per database, with the spread between them.

    Raises:
        ValueError: If ``sets`` is empty.
    """
    tables: dict[ElectronDatabase, RateTable] = {}
    for database, sections in sets:
        solver = TwoTermSolver(
            kinetics=kinetics_from_set(
                sections, grid, mass_ratio=mass_ratio, below=below, above=above
            ),
            gas_temperature_ev=gas_temperature_ev,
            ionisation_sharing=ionisation_sharing,
        )
        tables[database] = tabulate(solver, reduced_fields_td)

    if not tables:
        raise ValueError(
            "the sweep produced no tables: at least one database is needed. doc 09 §2.1 "
            "keeps three, and the spread between them is a budgeted error term."
        )
    return DatabaseSpread(tables=tables)


def tabulate_each_electron_set(
    store: ElectronSetSource,
    *,
    reduced_fields_td: ArrayLike,
    grid: EnergyGrid,
    mass_ratio: float | None = None,
    gas_temperature_ev: float = ROOM_TEMPERATURE_EV,
    ionisation_sharing: IonisationSharing = IonisationSharing.ONE_TAKES_ALL,
    below: ExtrapolationPolicy = DEFAULT_BELOW_GRID,
    above: ExtrapolationPolicy = DEFAULT_ABOVE_GRID,
) -> DatabaseSpread:
    """The doc 09 §2.1 sweep, driven straight off an atomic-data store.

    One call, so that "run this under every electron set" is not three hand-written call
    sites that quietly fall out of step. The store records which datasets were touched, so
    the bibliography of doc 09 §6 comes out of the same run.
    """
    return tabulate_electron_sets(
        store.each_electron_set(),
        reduced_fields_td=reduced_fields_td,
        grid=grid,
        mass_ratio=mass_ratio,
        gas_temperature_ev=gas_temperature_ev,
        ionisation_sharing=ionisation_sharing,
        below=below,
        above=above,
    )
