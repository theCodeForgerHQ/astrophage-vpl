"""Cross sections to solver input — doc 03 §3.2, §4.5, doc 09 §2.1.

The two-term solve needs three things from the atomic layer and nothing else:

1. the momentum-transfer cross section on the energy grid,
2. the mass ratio ``m_e / M`` that sets the elastic energy loss per collision,
3. one channel per inelastic process, with its threshold.

This module is that join. It reads nothing from disk: the input is a parsed
:class:`~vpl.physics.atomic.lxcat.CrossSectionSet`, which
:class:`~vpl.physics.atomic.store.AtomicDataStore` produces after verifying it against the
version lock (doc 09 §5).

## Why the momentum-transfer cross section is stored twice

Once at cell centres and once at cell boundaries, because the two transport integrals of
doc 03 §3.2 are naturally taken at different places::

    D_e N   = (gamma/3) integral (eps / sigma_m) f0 deps        -> f0 lives at centres
    mu_e N  = -(gamma/3) integral (eps / sigma_m) (df0/deps) deps -> df0 lives at boundaries

For a piecewise-constant ``f0`` the derivative is a sum of jumps at the interior cell
boundaries, so the mobility integral is a boundary sum and the diffusion integral is a
cell sum. Averaging one set of values to produce the other would introduce an
interpolation error into whichever integral got the averaged values — and it is
unnecessary, because both come from the same tabulated curve by the same interpolator.
The flux coefficients of the exponential scheme also live at boundaries.

## The extrapolation policy is decided here, and the decision is to refuse

:mod:`vpl.physics.atomic.interpolation` defaults to raising above the tabulated range, and
this module keeps that default rather than choosing something quieter. The reasoning is
doc 03 §4.5's: a rate coefficient is an integral over the cross section, so a value
invented off the end of the table is indistinguishable from data once it is inside the
integral. In practice the default costs nothing — LXCat argon runs to 10^3-10^4 eV and the
grids here reach tens of eV — and when it does fire it means the grid is longer than the
data, which is a configuration error worth a stack trace rather than a silent
extrapolation into the exact energy range doc 03 §3.2 says matters most.

Below the table the default is the atomic layer's :data:`~vpl.physics.atomic.interpolation.
DEFAULT_BELOW`, holding the first tabulated value. That is not a free choice either: the
grid starts at exactly zero and every real momentum-transfer table starts above it, so
*some* statement is required, and holding the first value is the mildest one available.
Both are keyword arguments so that a manifest can record what was used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.physics.atomic.interpolation import (
    DEFAULT_ABOVE,
    DEFAULT_BELOW,
    ExtrapolationPolicy,
    interpolate_cross_section,
)
from vpl.physics.atomic.lxcat import CrossSectionSet, ProcessType
from vpl.physics.eedf.grid import EnergyGrid

__all__ = [
    "DEFAULT_ABOVE_GRID",
    "DEFAULT_BELOW_GRID",
    "ElectronKinetics",
    "InelasticChannel",
    "kinetics_from_set",
]

type FloatArray = NDArray[np.float64]

#: What to do above the tabulated range. See the module docstring: refusing.
DEFAULT_ABOVE_GRID: ExtrapolationPolicy = DEFAULT_ABOVE

#: What to do below it, where no threshold already fixes the answer.
DEFAULT_BELOW_GRID: ExtrapolationPolicy = DEFAULT_BELOW


def _finite_positive(values: ArrayLike, *, what: str, size: int, label: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != (size,):
        raise ValueError(
            f"{what} must carry one value per {label}: expected ({size},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} must be finite; found nan or inf")
    if np.any(array <= 0.0):
        raise ValueError(
            f"{what} must be strictly positive everywhere. It appears in a denominator in "
            "both transport integrals of doc 03 §3.2, so a zero would return inf and the "
            "rate table would carry it as a number."
        )
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, eq=False)
class InelasticChannel:
    """One inelastic process on the energy grid — a row of doc 03 §4.5's table.

    Attributes:
        reaction: The channel's identity, as
            :attr:`~vpl.physics.atomic.lxcat.CrossSection.reaction` spells it. This is the
            key of the rate table doc 03 §3.2 asks for, and the key doc 04 §2.2's CR model
            looks a rate coefficient up by.
        process: Excitation or ionisation. Ionisation is not merely an energy sink — it
            adds an electron, and where that electron appears is a modelling choice.
        threshold_ev: The energy an electron loses in one such collision.
        sigma_m2: The cross section at the grid's cell centres. Exactly zero below the
            threshold, which is physics rather than policy.
        sigma_edge_m2: The same cross section at the cell boundaries. It exists for the
            same reason the elastic pair does — see the module docstring — and because
            an inelastic collision transfers momentum as well as energy, so this channel
            contributes to :attr:`ElectronKinetics.effective_momentum_transfer_edge_m2`
            and thence to the flux terms and the mobility sum, both of which are
            evaluated at boundaries.
    """

    reaction: str
    process: ProcessType
    threshold_ev: float
    sigma_m2: FloatArray
    sigma_edge_m2: FloatArray

    def __post_init__(self) -> None:
        if not self.threshold_ev > 0.0:
            raise ValueError(
                f"an inelastic channel needs a positive threshold energy, got "
                f"{self.threshold_ev} for {self.reaction!r}"
            )
        for attribute in ("sigma_m2", "sigma_edge_m2"):
            sigma = np.array(getattr(self, attribute), dtype=np.float64, copy=True)
            if sigma.ndim != 1:
                raise ValueError(
                    f"a channel cross section must be one-dimensional, got {sigma.shape}"
                )
            if not np.all(np.isfinite(sigma)):
                raise ValueError(f"the {self.reaction!r} cross section must be finite")
            if np.any(sigma < 0.0):
                raise ValueError(f"the {self.reaction!r} cross section has a negative value")
            sigma.flags.writeable = False
            object.__setattr__(self, attribute, sigma)

    @property
    def is_ionisation(self) -> bool:
        """Whether this channel creates an electron as well as consuming energy."""
        return self.process is ProcessType.IONIZATION

    def __repr__(self) -> str:
        return f"InelasticChannel({self.reaction!r}, {self.threshold_ev:g} eV)"


@dataclass(frozen=True, slots=True, eq=False)
class ElectronKinetics:
    """Everything the two-term solve needs about one gas, on one grid.

    Attributes:
        grid: The energy grid every array below is sampled on.
        database: Which of doc 09 §2.1's electron sets this came from. Carried because the
            spread between them is a budgeted error term (doc 06 §4 term 2), and a rate
            coefficient that cannot say which set produced it cannot contribute to it.
        momentum_transfer_m2: ``sigma_m`` at cell centres.
        momentum_transfer_edge_m2: ``sigma_m`` at cell boundaries. See the module
            docstring for why both exist.
        mass_ratio: ``m_e / M``. The elastic energy loss per collision is ``2 m_e / M``
            times the electron energy, and it is the only thing that stops the field from
            heating the electrons without limit.
        channels: The inelastic processes, in the order the database lists them.
    """

    grid: EnergyGrid
    database: str
    momentum_transfer_m2: FloatArray
    momentum_transfer_edge_m2: FloatArray
    mass_ratio: float
    channels: tuple[InelasticChannel, ...]

    def __post_init__(self) -> None:
        centres = _finite_positive(
            self.momentum_transfer_m2,
            what="the momentum-transfer cross section",
            size=self.grid.n_cells,
            label="cell",
        )
        edges = _finite_positive(
            self.momentum_transfer_edge_m2,
            what="the momentum-transfer cross section at the cell boundaries",
            size=self.grid.n_cells + 1,
            label="cell boundary",
        )
        if not 0.0 < self.mass_ratio < 1.0:
            raise ValueError(
                f"the mass ratio m_e / M must lie in (0, 1), got {self.mass_ratio}. It is "
                "the electron mass over the *target* mass; for argon it is 1.36e-5."
            )
        for channel in self.channels:
            if channel.sigma_m2.shape != (self.grid.n_cells,):
                raise ValueError(
                    f"channel {channel.reaction!r} carries {channel.sigma_m2.shape} values "
                    f"but the grid has {self.grid.n_cells} cells"
                )
            if channel.sigma_edge_m2.shape != (self.grid.n_cells + 1,):
                raise ValueError(
                    f"channel {channel.reaction!r} carries {channel.sigma_edge_m2.shape} "
                    f"values at the cell boundaries but the grid has "
                    f"{self.grid.n_cells + 1} boundaries"
                )
        object.__setattr__(self, "momentum_transfer_m2", centres)
        object.__setattr__(self, "momentum_transfer_edge_m2", edges)

    @property
    def effective_momentum_transfer_m2(self) -> FloatArray:
        """Reid (1979) eq. (4): ``sigma_m,e + sum_k sigma_k``, at cell centres.

        An inelastic collision randomises the electron's direction just as an elastic one
        does, so it contributes to momentum transfer whatever it does to energy. Reid
        derives the momentum equation with both present and shows it keeps the elastic
        form only if ``sigma_m,e`` is replaced by this sum (for isotropic inelastic
        scattering, which is what both model gases and most LXCat sets assume).

        He is unusually direct about the consequence, and it is worth quoting because the
        substitution is easy to make in the wrong direction:

            if equation (5b) is used to determine ``f(eps)``, the momentum transfer cross
            section that appears in the equation is the *total* cross section for
            momentum transfer and not the cross section for momentum transfer in elastic
            collisions.

        Using the elastic cross section alone makes the electrons too mobile and, through
        the field-heating term, too hot. The error is invisible when the inelastic cross
        sections are small next to the elastic one — which is the usual case, and why it
        is worth a named property rather than an inline sum.
        """
        total = np.array(self.momentum_transfer_m2, dtype=np.float64, copy=True)
        for channel in self.channels:
            total += channel.sigma_m2
        return total

    @property
    def effective_momentum_transfer_edge_m2(self) -> FloatArray:
        """The same sum at the cell boundaries, where the flux terms are evaluated."""
        total = np.array(self.momentum_transfer_edge_m2, dtype=np.float64, copy=True)
        for channel in self.channels:
            total += channel.sigma_edge_m2
        return total

    @property
    def ionisation(self) -> tuple[InelasticChannel, ...]:
        """The channels that create an electron — doc 03 §4.5's ``e + Ar -> 2e + Ar+``."""
        return tuple(c for c in self.channels if c.is_ionisation)

    @property
    def excitations(self) -> tuple[InelasticChannel, ...]:
        """The channels that only remove energy — the ``k_ex,j`` of doc 03 §3.2."""
        return tuple(c for c in self.channels if not c.is_ionisation)

    def __repr__(self) -> str:
        plural = "" if len(self.channels) == 1 else "s"
        return (
            f"ElectronKinetics({self.database!r}, {self.grid.n_cells} cells to "
            f"{self.grid.max_ev:g} eV, {len(self.channels)} channel{plural})"
        )


def kinetics_from_set(
    sections: CrossSectionSet,
    grid: EnergyGrid,
    *,
    mass_ratio: float | None = None,
    below: ExtrapolationPolicy = DEFAULT_BELOW_GRID,
    above: ExtrapolationPolicy = DEFAULT_ABOVE_GRID,
) -> ElectronKinetics:
    """Put one electron database on one energy grid.

    Args:
        sections: A parsed LXCat electron set, from
            :meth:`~vpl.physics.atomic.store.AtomicDataStore.electron_cross_sections`.
        grid: Where to sample.
        mass_ratio: ``m_e / M``. Defaults to the value the momentum-transfer block
            publishes — LXCat states it as the bare number under the reaction. Supplying
            it explicitly overrides that, which is what an ablation over the target mass
            needs.
        below: Extrapolation policy below the tabulated range. See the module docstring.
        above: Extrapolation policy above it. Defaults to refusing.

    Returns:
        The assembled kinetics.

    Raises:
        LookupError: If the set publishes neither ``ELASTIC`` nor ``EFFECTIVE``, so there
            is no momentum-transfer channel to give the solver.
        ValueError: If no mass ratio is available from either source.
        ~vpl.physics.atomic.interpolation.OutsideTabulatedRangeError: If the grid reaches
            past a table and ``above`` refuses.
    """
    try:
        transfer = sections.momentum_transfer()
    except LookupError as exc:
        raise LookupError(
            f"database {sections.database!r} has no momentum-transfer channel, so the "
            f"two-term solve has no elastic collision operator: {exc}"
        ) from exc

    ratio = transfer.mass_ratio if mass_ratio is None else mass_ratio
    if ratio is None:
        raise ValueError(
            f"database {sections.database!r} publishes no mass ratio on its "
            f"{transfer.process} block and none was supplied. The elastic energy loss is "
            "2 m_e / M per collision and there is no defensible default for it — a wrong "
            "one rescales the whole EEDF."
        )

    def sampled(energy_ev: FloatArray) -> FloatArray:
        return interpolate_cross_section(transfer, energy_ev, below=below, above=above)

    channels = tuple(
        InelasticChannel(
            reaction=section.reaction,
            process=section.process,
            # A threshold process always carries one; the parser refuses otherwise.
            threshold_ev=float(section.threshold_ev or 0.0),
            sigma_m2=interpolate_cross_section(section, grid.centres_ev, below=below, above=above),
            # Also at the boundaries: this channel contributes to the effective
            # momentum-transfer cross section of Reid eq. (4), which the flux terms and
            # the mobility sum both evaluate there.
            sigma_edge_m2=interpolate_cross_section(
                section, grid.boundaries_ev, below=below, above=above
            ),
        )
        for section in sections
        if section.process.has_threshold
    )

    return ElectronKinetics(
        grid=grid,
        database=sections.database,
        momentum_transfer_m2=sampled(grid.centres_ev),
        momentum_transfer_edge_m2=sampled(grid.boundaries_ev),
        mass_ratio=float(ratio),
        channels=channels,
    )
