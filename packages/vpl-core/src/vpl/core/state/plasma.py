"""PlasmaState — what every level of the doc 03 §1 hierarchy returns.

doc 00 E1 promises that the inverse solver never learns which fidelity level produced
the state it is looking at. That promise only holds if all four levels return the same
*shape of object*, with the same required contents, validated the same way. This type is
where that is enforced.

The required field set is doc 03 §5.2's: ``n_e``, ``n_i``, ``Phi``, ``T_e``. Anything
else a solver computes — ``T_i``, ``u_i``, metastable populations, the EEDF — is
optional and carried alongside.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from vpl.core.state.fidelity import Fidelity
from vpl.core.state.fields import ScalarField, VelocityDistribution
from vpl.core.state.grid import SpatialGrid, TimeGrid
from vpl.core.state.params import PlasmaParams
from vpl.core.units import UREG, DimensionalityError

__all__ = ["REQUIRED_FIELDS", "PlasmaState"]

#: The fields doc 03 §5.2 names as the emulated set, with the dimensionality each must
#: have. A potential expressed in metres type-checks perfectly and is nonsense; catching
#: it at construction is the difference between a loud error and a silent 1e19 in a
#: synthetic spectrum three layers downstream.
REQUIRED_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        "n_e": "m**-3",
        "n_i": "m**-3",
        "Phi": "V",
        "T_e": "eV",
    }
)


@dataclass(frozen=True, eq=False, slots=True)
class PlasmaState:
    """A plasma state on a grid, at a stated fidelity.

    Attributes:
        params: The control parameters that produced this state.
        grid: The spatial grid every field shares.
        time: The time grid every field shares, or ``None`` for a steady state.
        fields: Named scalar fields. Must contain at least :data:`REQUIRED_FIELDS`.
        ion_distribution: ``f_i(z, v_z)``, or ``None`` where the level does not resolve
            one. doc 03 §6 is explicit that L1 reconstructs a drifting Maxwellian from
            its moments and that this is a genuine weakness; recording the absence is
            more honest than manufacturing a distribution the solver never computed.
        fidelity: Which level of the doc 03 §1 hierarchy produced this.
    """

    params: PlasmaParams
    grid: SpatialGrid
    time: TimeGrid | None
    fields: Mapping[str, ScalarField]
    ion_distribution: VelocityDistribution | None
    fidelity: Fidelity

    def __post_init__(self) -> None:
        # Copy before freezing: the caller's dict is theirs to keep mutating, and a
        # state that aliased it would change contents after every validation had passed.
        frozen = MappingProxyType(dict(self.fields))
        object.__setattr__(self, "fields", frozen)

        self._check_fields_are_self_consistent()
        self._check_required_fields_present_and_dimensional()
        self._check_distribution()

    # ── validation ──────────────────────────────────────────────────────────────

    def _check_fields_are_self_consistent(self) -> None:
        for key, field in self.fields.items():
            if field.name != key:
                raise ValueError(
                    f"field keyed as {key!r} calls itself {field.name!r}. The key and the "
                    "field's own name must agree, or every downstream error message and "
                    "artifact group refers to the wrong quantity."
                )
            if field.grid != self.grid:
                raise ValueError(
                    f"field {key!r} is on a different grid from the state "
                    f"({field.grid.n_points} points vs {self.grid.n_points})"
                )
            if field.time != self.time:
                expected = "steady" if self.time is None else f"{self.time.n_points} times"
                actual = "steady" if field.time is None else f"{field.time.n_points} times"
                raise ValueError(
                    f"field {key!r} is {actual} but the state is {expected}; "
                    "a state cannot mix steady and time-dependent fields"
                )

    def _check_required_fields_present_and_dimensional(self) -> None:
        missing = sorted(set(REQUIRED_FIELDS) - set(self.fields))
        if missing:
            raise ValueError(
                f"state is missing required field(s) {', '.join(missing)}. "
                f"doc 03 §5.2 requires {', '.join(sorted(REQUIRED_FIELDS))}."
            )

        for name, expected_units in REQUIRED_FIELDS.items():
            actual = UREG.Unit(self.fields[name].units)
            if actual.dimensionality != UREG.Unit(expected_units).dimensionality:
                raise DimensionalityError(
                    f"field {name!r} has units {self.fields[name].units!r} "
                    f"({actual.dimensionality}), but must be compatible with "
                    f"{expected_units!r} ({UREG.Unit(expected_units).dimensionality})"
                )

    def _check_distribution(self) -> None:
        if self.fidelity.resolves_distribution and self.ion_distribution is None:
            raise ValueError(
                f"a {self.fidelity.name} state must carry its ion distribution. "
                "doc 03 §1 calls L2 the only level that produces a correct IEDF, and a "
                "run that reaches the likelihood without one cannot answer the question "
                "it was launched to answer."
            )
        if self.ion_distribution is None:
            return
        if self.ion_distribution.grid != self.grid:
            raise ValueError("the ion distribution is on a different grid from the state")
        if self.ion_distribution.species != self.params.species:
            raise ValueError(
                f"ion distribution species mismatch: the distribution is for "
                f"{self.ion_distribution.species.name} but the parameters describe "
                f"{self.params.species.name}"
            )

    # ── access ──────────────────────────────────────────────────────────────────

    def field(self, name: str) -> ScalarField:
        """One field by name, with an error that says what *is* available.

        A forward model asking for a field the solver did not emit is a wiring error,
        and a bare ``KeyError`` does not say whether the solver or the instrument is
        the one at fault.
        """
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(
                f"no field {name!r} in this {self.fidelity.name} state; "
                f"available: {', '.join(self.field_names)}"
            ) from None

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in a deterministic order.

        Sorted, not insertion-ordered. doc 00 E3 promises bit-for-bit reproducibility,
        and anything that iterates the fields and reduces over them — a hash, an
        artifact write, a diff between two runs — must see the same order in every
        process rather than whatever order the solver happened to fill them in.
        """
        return tuple(sorted(self.fields))

    @property
    def is_steady(self) -> bool:
        return self.time is None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlasmaState):
            return NotImplemented
        return (
            self.fidelity is other.fidelity
            and self.params == other.params
            and self.grid == other.grid
            and self.time == other.time
            and dict(self.fields) == dict(other.fields)
            and self.ion_distribution == other.ion_distribution
        )

    def __repr__(self) -> str:
        when = "steady" if self.time is None else f"{self.time.n_points} times"
        ivdf = "with IVDF" if self.ion_distribution is not None else "no IVDF"
        return (
            f"PlasmaState({self.fidelity.name}, {self.grid.n_points} points, {when}, "
            f"{len(self.fields)} fields, {ivdf})"
        )
