"""Ion and neutral species.

Masses are *supplied*, never defaulted. Doc 09 §2.5 requires every constant to trace to
an evaluated source, and doc 00 C1 makes an unsourced value a defect, so this type takes
its mass from the parameter registry rather than carrying a table of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpl.core.units import Quantity, magnitude_in

__all__ = ["Species"]


@dataclass(frozen=True, slots=True)
class Species:
    """A charged or neutral species participating in the sheath.

    Attributes:
        name: Human-readable label, e.g. ``"Ar+"``. Appears in artifact metadata.
        mass: Rest mass, as a dimensional quantity.
        charge_number: Charge in units of the elementary charge. ``0`` for a neutral,
            ``-1`` for an electron, ``+1`` for a singly-ionised atom.
    """

    name: str
    mass: Quantity
    charge_number: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("species name must not be empty")
        if magnitude_in(self.mass, "kg") <= 0.0:
            raise ValueError(f"{self.name}: mass must be positive, got {self.mass}")

    @property
    def mass_kg(self) -> float:
        """Rest mass in kilograms, for hot loops."""
        return float(magnitude_in(self.mass, "kg"))

    @property
    def is_neutral(self) -> bool:
        return self.charge_number == 0

    def __repr__(self) -> str:
        return f"Species({self.name!r}, mass={self.mass:~P}, Z={self.charge_number:+d})"
