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

from typing import Any, TypeAlias, cast, overload

import numpy as np
import pint
from numpy.typing import NDArray

__all__ = [
    "Q_",
    "UREG",
    "ArrayQuantity",
    "DimensionalityError",
    "Magnitude",
    "Quantity",
    "ScalarQuantity",
    "magnitude_in",
]

#: What a quantity's magnitude may be: a plain float for a scalar, an array for a field.
Magnitude: TypeAlias = float | NDArray[np.float64]

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

#: A dimensional quantity drawn from :data:`UREG`, of unspecified magnitude type.
#:
#: Deliberately ``pint.Quantity[Any]`` rather than ``pint.Quantity[float | NDArray]``.
#: pint's ``Quantity`` is *invariant* in its magnitude, so the union form makes
#: ``Quantity[float]`` unassignable to it — and every parameter position in the project
#: would then reject the scalar quantities that are the common case. The rule that falls
#: out is the usual one: **accept broadly, return precisely.** Parameters take
#: :data:`Quantity`; returns are :data:`ScalarQuantity` or :data:`ArrayQuantity` wherever
#: which one it is is a fact about the quantity rather than about one caller.
Quantity: TypeAlias = pint.Quantity[Any]

#: A quantity whose magnitude is known to be a scalar.
#:
#: Use this in a signature wherever the scalar-ness is a fact about the quantity rather
#: than about one caller — a sheath thickness is always one number. It lets
#: :func:`magnitude_in` return a plain ``float`` instead of a union the caller then has
#: to narrow, and hand-narrowing at every call site is how a union type stops being
#: checked at all.
ScalarQuantity: TypeAlias = pint.Quantity[float]

#: A quantity whose magnitude is known to be an array — a field, a grid, an axis.
ArrayQuantity: TypeAlias = pint.Quantity[NDArray[np.float64]]

#: Quantity constructor bound to :data:`UREG`. ``Q_(5.0, "mTorr")``.
Q_ = UREG.Quantity


class DimensionalityError(TypeError):
    """A value crossing a module boundary was not in the units the boundary requires.

    Deliberately distinct from :class:`pint.DimensionalityError` so that callers can
    catch *this project's* boundary violations without also catching every dimensional
    slip inside pint itself. The originating pint error is always chained.
    """


@overload
def magnitude_in(value: ScalarQuantity, units: str) -> float: ...


@overload
def magnitude_in(value: ArrayQuantity, units: str) -> NDArray[np.float64]: ...


@overload
def magnitude_in(value: Quantity, units: str) -> Magnitude: ...


def magnitude_in(value: Quantity, units: str) -> Magnitude:
    """Convert ``value`` to ``units`` and strip them, raising if it cannot be done.

    This is the entry half of the doc 08 section 5 contract: hot loops receive bare
    arrays, but only after the caller has proved they mean what the loop assumes.

    The overloads carry the magnitude's own type through, so a caller who passed a
    scalar gets a ``float`` back and one who passed an array gets an array. Without
    them every call site would face ``float | NDArray`` and would either index into a
    possible float or narrow it by hand — and hand-narrowing at a hundred call sites is
    how a union type stops being checked at all.

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
        # quantity ever exists. The cast is needed because `Quantity` is
        # `pint.Quantity[Any]` in parameter position (see its docstring), so pint can
        # only promise `Any` back; the overloads above are what give callers a real type.
        return cast("Magnitude", value.m_as(units))
    except pint.DimensionalityError as exc:
        raise DimensionalityError(
            f"cannot supply this value where {units!r} was required: {exc}"
        ) from exc
