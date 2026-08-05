"""The registered-parameter schema — doc 02 §12, doc 09 §1.

doc 02 §12 fixes the shape of an entry and doc 09 §1 fixes what each provenance class
means. Both are enforced here, at load time, because a registry that accepts a
malformed entry is a registry that will hand a wrong number to a solver later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pint

from vpl.core.units import Q_, UREG, DimensionalityError, Quantity

__all__ = ["Parameter", "ProvenanceClass", "Uncertainty"]


class ProvenanceClass(StrEnum):
    """Where a number came from — doc 09 §1.

    The classes are not decoration. doc 00 C1 makes an unsourced value a *defect* and
    tracks the count as a project metric, which only works if every entry has to declare
    which kind of thing it is.
    """

    #: From a standard evaluated database, with version and access date.
    MEASURED = "MEASURED"
    #: From a specific paper, cited.
    PUBLISHED = "PUBLISHED"
    #: From a real commercial component datasheet, with part number.
    SPEC = "SPEC"
    #: A free design choice. Therefore a variable in the sensitivity study.
    DESIGN = "DESIGN"
    #: Could not be sourced. A defect; the count trends to zero.
    ASSUMED = "ASSUMED"


#: Classes that assert traceability and must therefore name what they trace to.
_SOURCED_CLASSES = frozenset(
    {ProvenanceClass.MEASURED, ProvenanceClass.PUBLISHED, ProvenanceClass.SPEC}
)

type UncertaintyKind = Literal["relative", "absolute"]

_UNCERTAINTY_KINDS: frozenset[str] = frozenset({"relative", "absolute"})


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """A one-sigma uncertainty, either fractional or in the parameter's own units."""

    kind: UncertaintyKind
    value: float

    def __post_init__(self) -> None:
        if self.kind not in _UNCERTAINTY_KINDS:
            raise ValueError(
                f"unknown uncertainty type {self.kind!r}; expected one of "
                f"{', '.join(sorted(_UNCERTAINTY_KINDS))}"
            )
        if self.value < 0.0:
            raise ValueError(f"uncertainty cannot be negative, got {self.value}")


@dataclass(frozen=True, slots=True)
class Parameter:
    """One registered quantity.

    Attributes:
        id: Dotted identifier, e.g. ``TS-L1.energy``. Unique across the whole registry.
        description: What the number is, in words, for someone who did not write it.
        value: The nominal value, in :attr:`units`.
        units: pint-parseable units.
        provenance_class: doc 09 §1 class.
        source: Citation, part number, or the rationale for a design choice.
        uncertainty: One-sigma uncertainty, or ``None`` where it is not yet characterised.
        sweep_range: ``(low, high)`` for the sensitivity study. Mandatory for ``DESIGN``.
        affects: Downstream quantities this feeds, for impact analysis.
        category: Coarse grouping used by the CI metric — ``physics``, ``atomic``,
            ``instrument``, ``operating_point``, ``numerics``.
        defined_in: The file this was loaded from. When a value looks wrong the first
            question is always "where is this defined", and grepping an installed wheel
            is not an answer.
    """

    id: str
    description: str
    value: float
    units: str
    provenance_class: ProvenanceClass
    source: str
    uncertainty: Uncertainty | None
    sweep_range: tuple[float, float] | None
    affects: tuple[str, ...]
    category: str
    defined_in: Path | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("a registered parameter needs an id")
        if not self.description.strip():
            raise ValueError(
                f"{self.id}: description is required — it is what makes the "
                "registry readable by someone who did not write the entry"
            )

        try:
            UREG.Unit(self.units)
        except (pint.UndefinedUnitError, TypeError, ValueError) as exc:
            raise DimensionalityError(f"{self.id}: unrecognised units {self.units!r}") from exc

        if self.provenance_class in _SOURCED_CLASSES and not self.source.strip():
            raise ValueError(
                f"{self.id}: class {self.provenance_class} asserts traceability but names no "
                "source. An entry claiming SPEC with no datasheet is an ASSUMED value "
                "wearing a better label (doc 09 §1)."
            )

        if self.provenance_class is ProvenanceClass.DESIGN and self.sweep_range is None:
            raise ValueError(
                f"{self.id}: a DESIGN parameter must declare a sweep_range. doc 09 §1 is "
                "explicit that an unswept design choice is a hidden assumption wearing a "
                "different label."
            )

        if self.sweep_range is not None:
            low, high = self.sweep_range
            if not low < high:
                raise ValueError(f"{self.id}: sweep_range must be ascending, got [{low}, {high}]")
            if not low <= self.value <= high:
                raise ValueError(
                    f"{self.id}: sweep_range [{low}, {high}] does not bracket the nominal "
                    f"value {self.value}. One of the two is wrong, and the sensitivity "
                    "study would explore somewhere the model never runs."
                )

    @property
    def quantity(self) -> Quantity:
        """The nominal value as a dimensional quantity."""
        return Q_(self.value, self.units)

    @property
    def is_defect(self) -> bool:
        """Whether this entry counts against the doc 00 C1 metric."""
        return self.provenance_class is ProvenanceClass.ASSUMED

    @property
    def absolute_uncertainty(self) -> float | None:
        """One-sigma uncertainty in the parameter's own units, whichever way it was given."""
        if self.uncertainty is None:
            return None
        if self.uncertainty.kind == "relative":
            return abs(self.value) * self.uncertainty.value
        return self.uncertainty.value

    def __repr__(self) -> str:
        return (
            f"Parameter({self.id!r}, {self.value:g} {self.units}, "
            f"{self.provenance_class}, category={self.category!r})"
        )
