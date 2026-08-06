"""Putting a tabulated cross section on an arbitrary energy grid — doc 03 §3.2, §4.5.

Doc 03 §3.2 has a two-term Boltzmann solver consume ``sigma(E)`` on whatever grid it
chose, and tabulate the resulting rate coefficients against ``E/N``. Doc 03 §4.5 says
that getting the charge-exchange cross section "and its energy dependence right matters
more than any other atomic-data choice". This module is where those two statements meet,
and the meeting point is the *tail*.

## Why the tail is the whole problem

A rate coefficient is ``k = <sigma v> = integral sigma(E) f(E) sqrt(E) dE``. Once a value
is inside that integral there is nothing left to distinguish a tabulated point from an
invented one. So an extrapolation off the end of the table does not announce itself as an
extrapolation; it announces itself as a rate coefficient, several modules later, in a
posterior.

Three plausible things one can do above the last tabulated point, and each is wrong in a
different direction:

- **Zero** deletes the channel. Ionisation stops above the last point, so the
  high-``E/N`` end of the doc 03 §3.2 table is wrong and monotonically so.
- **Constant** holds a falling cross section flat. Electron-impact ionisation falls as
  roughly ``1/E`` above its peak, so holding it constant overestimates the tail by the
  ratio of the energies.
- **A power law** continues the last decade's slope. Correct in form for the asymptotics
  of both ionisation (Bethe, ``~ln E / E``) and momentum transfer, and wrong wherever the
  real curve turns over just past the table.

None of the three is defensible as a silent default, so the default is the fourth option:
**refuse**. In practice this costs nothing — LXCat argon tables run to 10^3 to 10^4 eV, and the
EEDF at ``T_e = 3 eV`` (doc 01 §2.1) has nothing left up there — and it converts a silent
tail error into a loud one at the call site that asked for it.

## Below the table is different, and mostly not a choice

Where the process has a threshold, ``sigma = 0`` below it is *physics*, not policy: an
inelastic process cannot occur below threshold. That zero is applied unconditionally and
before any policy is consulted, which is what lets a caller ask for a grid starting at
0 eV without having to opt out of a policy that does not apply to it. Where there is no
threshold — momentum transfer, charge exchange — the default holds the first tabulated
value, which is the mildest of the available statements: charge exchange rises slowly and
monotonically as ``E -> 0`` and is emphatically not zero there.

## Interpolation inside the table is linear in both axes

Not log-log, which is the more usual choice for a quantity spanning decades, and the
reason is concrete: cross-section tables carry **exact zeros** at and below threshold, and
``log 0`` does not exist. Working around that with a floor would put an invented smallest
cross section into the one region where the true value is known exactly. Linear
interpolation also invents no structure between tabulated points, which is the
conservative failure mode for something about to be integrated, and it is what BOLSIG+
does with the same tables — so the framework and the solver it feeds agree by
construction rather than by luck.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.physics.atomic.lxcat import CrossSection, CrossSectionSet

__all__ = [
    "DEFAULT_ABOVE",
    "DEFAULT_BELOW",
    "ExtrapolationPolicy",
    "OutsideTabulatedRangeError",
    "interpolate_cross_section",
    "interpolate_set",
]

FloatArray: TypeAlias = NDArray[np.float64]


class OutsideTabulatedRangeError(ValueError):
    """A grid reached past the table and the policy did not permit an answer."""


class ExtrapolationPolicy(StrEnum):
    """What to do off the end of a tabulated cross section.

    A ``StrEnum`` because this is a modelling choice and doc 00 C4 makes every modelling
    choice a manifest field: ``cross_sections.above: power_law`` round-trips into a
    checked member and back out for the provenance record, and the ablation matrix of
    doc 07 can sweep it.
    """

    #: Refuse. The default above the table; see the module docstring.
    RAISE = "raise"

    #: The channel does not exist out there.
    ZERO = "zero"

    #: Hold the nearest tabulated value. The default below the table.
    CONSTANT = "constant"

    #: Continue the slope of the two nearest tabulated points in log-log.
    POWER_LAW = "power_law"


#: Below the tabulated range, where no threshold applies. Charge exchange and momentum
#: transfer are both finite and slowly varying as ``E -> 0``, so holding the first value
#: is the mildest defensible statement; zero would delete a channel that is real there.
DEFAULT_BELOW: Final[ExtrapolationPolicy] = ExtrapolationPolicy.CONSTANT

#: Above the tabulated range. See the module docstring: every silent option is wrong in a
#: different direction and none of them show up in the answer, so this one is loud.
DEFAULT_ABOVE: Final[ExtrapolationPolicy] = ExtrapolationPolicy.RAISE

_BELOW: Final[str] = "below"
_ABOVE: Final[str] = "above"


def _checked_grid(energy_ev: ArrayLike) -> FloatArray:
    """Validate the requested grid before it can produce a plausible-looking nan."""
    grid = np.asarray(energy_ev, dtype=np.float64)
    if grid.ndim > 1:
        raise ValueError(f"energy grid must be scalar or one-dimensional, got shape {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("energy grid must be finite; found nan or inf")
    if np.any(grid < 0.0):
        raise ValueError("energy grid contains a negative energy")
    return grid


def _log_log_slope(
    *, e_1: float, s_1: float, e_2: float, s_2: float, section: CrossSection, side: str
) -> float:
    """The exponent ``p`` in ``sigma ~ E^p`` through two tabulated points.

    The energies are safe to take a logarithm of without a guard: a table is strictly
    increasing and its energies are non-negative, so ``e_2 > 0`` always, and the only way
    ``e_1`` can be zero is if it is the first point — in which case nothing lies below it
    to extrapolate to, because the requested grid is validated non-negative.
    """
    if s_1 <= 0.0 or s_2 <= 0.0:
        raise OutsideTabulatedRangeError(
            f"{section.database} {section.process}: cannot continue a power law {side} the "
            f"table because a nearest tabulated cross section is zero. A zero is exact "
            f"data, not a small number — pick another policy rather than a floor."
        )
    return float(np.log(s_2 / s_1) / np.log(e_2 / e_1))


def _extrapolate(
    *,
    policy: ExtrapolationPolicy,
    section: CrossSection,
    energies: FloatArray,
    side: str,
) -> FloatArray:
    """Values for grid points that fall outside the tabulated range on one side."""
    lowest, highest = section.energy_range_ev

    if policy is ExtrapolationPolicy.RAISE:
        extreme = float(energies.min()) if side == _BELOW else float(energies.max())
        raise OutsideTabulatedRangeError(
            f"{section.database} {section.process} ({section.reaction}) is tabulated over "
            f"[{lowest:g}, {highest:g}] eV, and {energies.size} requested energies lie "
            f"{side} it (extreme: {extreme:g} eV). "
            f"Pass {side}=... to choose an extrapolation; doc 03 §4.5 makes this the most "
            "consequential atomic-data choice in the project, so it is not made for you."
        )

    if policy is ExtrapolationPolicy.ZERO:
        return np.zeros_like(energies)

    anchor_energy, anchor_sigma = (
        (section.energy_ev[0], section.sigma_m2[0])
        if side == _BELOW
        else (section.energy_ev[-1], section.sigma_m2[-1])
    )

    if policy is ExtrapolationPolicy.CONSTANT:
        return np.full_like(energies, float(anchor_sigma))

    first, second = (0, 1) if side == _BELOW else (-2, -1)
    exponent = _log_log_slope(
        e_1=float(section.energy_ev[first]),
        s_1=float(section.sigma_m2[first]),
        e_2=float(section.energy_ev[second]),
        s_2=float(section.sigma_m2[second]),
        section=section,
        side=side,
    )
    return np.asarray(
        float(anchor_sigma) * (energies / float(anchor_energy)) ** exponent, dtype=np.float64
    )


def interpolate_cross_section(
    section: CrossSection,
    energy_ev: ArrayLike,
    *,
    below: ExtrapolationPolicy = DEFAULT_BELOW,
    above: ExtrapolationPolicy = DEFAULT_ABOVE,
) -> FloatArray:
    """``sigma(E)`` on an arbitrary energy grid.

    Args:
        section: The tabulated cross section.
        energy_ev: Energies in eV. Scalar or one-dimensional, finite and non-negative.
        below: What to do below the first tabulated energy. Consulted only where the
            answer is not already fixed by a threshold — see :data:`DEFAULT_BELOW`.
        above: What to do above the last tabulated energy. See :data:`DEFAULT_ABOVE`;
            the default is to refuse.

    Returns:
        A writeable array shaped like ``energy_ev``. Never negative, and exactly zero
        below an inelastic process's threshold.

    Raises:
        ValueError: If the grid is not finite, not non-negative, or not at most
            one-dimensional.
        OutsideTabulatedRangeError: If the grid leaves the table where the chosen policy
            refuses, or where a power law would need the logarithm of a tabulated zero.
    """
    grid = _checked_grid(energy_ev)
    flat = np.atleast_1d(grid)
    lowest, highest = section.energy_range_ev

    result = np.asarray(np.interp(flat, section.energy_ev, section.sigma_m2), dtype=np.float64)

    # Physics before policy. The table normally *starts* at the threshold, so nearly
    # every sub-threshold point is also below the tabulated range — and a caller asking
    # for a grid from 0 eV should not have to opt out of an extrapolation policy to be
    # told the exact answer.
    known_zero = (
        np.zeros(flat.shape, dtype=np.bool_)
        if section.threshold_ev is None
        else flat < section.threshold_ev
    )

    under = (flat < lowest) & ~known_zero
    over = flat > highest

    if np.any(under):
        result[under] = _extrapolate(
            policy=below, section=section, energies=flat[under], side=_BELOW
        )
    if np.any(over):
        result[over] = _extrapolate(policy=above, section=section, energies=flat[over], side=_ABOVE)
    result[known_zero] = 0.0

    return result.reshape(grid.shape)


def interpolate_set(
    sections: CrossSectionSet,
    energy_ev: ArrayLike,
    *,
    below: ExtrapolationPolicy = DEFAULT_BELOW,
    above: ExtrapolationPolicy = DEFAULT_ABOVE,
) -> Mapping[str, FloatArray]:
    """Every channel of one database on one grid — the handoff to doc 03 §3.2.

    Keyed by :attr:`CrossSection.reaction`, because that is the channel's identity in
    doc 03 §4.5's table and in any report that has to say which process a rate
    coefficient belongs to. A duplicate key raises rather than overwriting: two blocks
    claiming the same reaction means the export lists a channel twice, and silently
    keeping the last one would drop a real excitation channel out of the CR model.

    Args:
        sections: A parsed database.
        energy_ev: The grid, shared by every channel.
        below: See :func:`interpolate_cross_section`.
        above: See :func:`interpolate_cross_section`.

    Returns:
        A read-only mapping from reaction to ``sigma(E)`` on the grid.
    """
    values: dict[str, FloatArray] = {}
    for section in sections:
        key = section.reaction
        if key in values:
            raise ValueError(
                f"database {sections.database!r} lists the reaction {key!r} twice; one of "
                "the two would be silently discarded"
            )
        values[key] = interpolate_cross_section(section, energy_ev, below=below, above=above)
    return MappingProxyType(values)
