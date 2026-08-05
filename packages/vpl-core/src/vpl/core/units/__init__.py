"""Dimensional safety — doc 08 §5.

Every physical quantity crossing a module boundary is a :class:`Quantity`. Raw arrays
appear only inside hot loops, and the transition between the two is
:func:`magnitude_in`, which asserts the unit contract instead of assuming it.

The reason this layer is worth its cost: doc 02 §2 notes that a sign error in a flux
quantity is "common, silent and catastrophic". A unit error is the same failure with a
larger multiplier. Both are cheap to prevent at the boundary and expensive to find in a
posterior.
"""

from __future__ import annotations

import numpy as np
import pint
from numpy.typing import NDArray

__all__ = [
    "Q_",
    "UREG",
    "DimensionalityError",
    "Magnitude",
    "Quantity",
    "magnitude_in",
]

#: What a quantity's magnitude may be: a plain float for a scalar, an array for a field.
type Magnitude = float | NDArray[np.float64]

#: The single shared unit registry.
#:
#: pint quantities built by *different* registries raise on any binary operation, and the
#: error surfaces far from the import that caused it. One registry, imported everywhere,
#: and registered as pint's application registry so third-party code finds the same one.
UREG: pint.UnitRegistry[Magnitude] = pint.UnitRegistry(autoconvert_offset_to_baseunit=False)

# Units the project needs that pint does not ship.
#
# The townsend is the conventional unit of reduced electric field E/N, against which
# doc 03 section 3.2 tabulates every rate coefficient the fluid model consumes.
UREG.define("townsend = 1e-21 * volt * meter**2 = Td")

# pint ships `torr` but will not accept an SI prefix on it, so `mTorr` does not parse.
# That is the unit every pressure in docs 01 and 02 is quoted in, so it is defined here
# rather than rewriting the documents into a unit no plasma physicist uses.
UREG.define("millitorr = 1e-3 * torr = mTorr = mtorr = milliTorr")

pint.set_application_registry(UREG)  # type: ignore[no-untyped-call]

#: A dimensional quantity drawn from :data:`UREG`.
type Quantity = pint.Quantity[Magnitude]

#: Quantity constructor bound to :data:`UREG`. ``Q_(5.0, "mTorr")``.
Q_ = UREG.Quantity


class DimensionalityError(TypeError):
    """A value crossing a module boundary was not in the units the boundary requires.

    Deliberately distinct from :class:`pint.DimensionalityError` so that callers can
    catch *this project's* boundary violations without also catching every dimensional
    slip inside pint itself. The originating pint error is always chained.
    """


def magnitude_in(value: Quantity, units: str) -> Magnitude:
    """Convert ``value`` to ``units`` and strip them, raising if it cannot be done.

    This is the entry half of the doc 08 section 5 contract: hot loops receive bare
    arrays, but only after the caller has proved they mean what the loop assumes.

    Args:
        value: A dimensional quantity from :data:`UREG`.
        units: The units the caller requires, as a pint-parseable string.

    Returns:
        The magnitude of ``value`` expressed in ``units``.

    Raises:
        DimensionalityError: If ``value`` is not a quantity at all, or is a quantity
            whose dimensionality cannot be converted to ``units``.
    """
    if not isinstance(value, UREG.Quantity):
        raise DimensionalityError(
            f"expected a dimensional quantity in {units!r}, got "
            f"{type(value).__name__} {value!r} - not a dimensional quantity. "
            f"Wrap it where it was created: Q_(value, {units!r})."
        )

    try:
        # ``m_as`` converts and strips in one step, so no converted-but-unasserted
        # quantity ever exists.
        return value.m_as(units)
    except pint.DimensionalityError as exc:
        raise DimensionalityError(
            f"cannot supply this value where {units!r} was required: {exc}"
        ) from exc
