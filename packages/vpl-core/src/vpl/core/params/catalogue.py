"""Loading and querying the parameter registry — doc 02 §12, doc 08 §5.

    The registry is the single source of truth. No number appears in code as a literal.

The loader is strict on purpose. Every rule it enforces corresponds to a way a registry
stops being a source of truth: a duplicate id makes the answer depend on file ordering,
an unrecognised key makes a parameter silently not apply, a missing source turns an
assumption into an assertion. All of them are cheap to catch at load and expensive to
find in a posterior.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from vpl.core.params.schema import Parameter, ProvenanceClass, Uncertainty
from vpl.core.units import Magnitude, Quantity, magnitude_in

__all__ = ["DEFAULT_REGISTRY_DIR", "ParameterRegistry", "default_registry"]

#: Where the registry files that ship with the package live.
DEFAULT_REGISTRY_DIR = Path(__file__).parent / "data"

#: Similarity threshold for the "did you mean" hint on an unknown parameter id.
#:
#: An ergonomic knob, not a physical quantity — but it is named because doc 08 §5's rule
#: does not carve out exceptions for numbers whose author was confident they did not
#: matter, and that is the point of the rule.
_SUGGESTION_CUTOFF = 0.6

#: How many suggestions to offer. More than a handful stops being a hint.
_MAX_SUGGESTIONS = 3

_REQUIRED_KEYS = frozenset({"id", "description", "value", "units", "class", "category"})
_OPTIONAL_KEYS = frozenset({"source", "uncertainty", "sweep_range", "affects"})
_KNOWN_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS


def _parse_uncertainty(raw: object, *, entry_id: str) -> Uncertainty | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{entry_id}: uncertainty must be a mapping, got {type(raw).__name__}")
    missing = {"type", "value"} - set(raw)
    if missing:
        raise ValueError(f"{entry_id}: uncertainty is missing {', '.join(sorted(missing))}")
    return Uncertainty(kind=raw["type"], value=float(raw["value"]))


def _parse_sweep_range(raw: object, *, entry_id: str) -> tuple[float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != 2:
        raise ValueError(f"{entry_id}: sweep_range must be a two-element list [low, high]")
    return (float(raw[0]), float(raw[1]))


def _parse_entry(raw: Mapping[str, Any], *, path: Path) -> Parameter:
    entry_id = str(raw.get("id", "<no id>"))

    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        # A typo'd key that is silently ignored is a parameter that silently does not
        # apply — the doc 00 §6 parameter-fog trap wearing a YAML costume.
        raise ValueError(
            f"{entry_id} in {path.name}: unrecognised key(s) {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(_KNOWN_KEYS))}."
        )

    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(
            f"{entry_id} in {path.name}: missing required key(s) {', '.join(sorted(missing))}"
        )

    class_name = str(raw["class"])
    try:
        provenance = ProvenanceClass(class_name)
    except ValueError as exc:
        raise ValueError(
            f"{entry_id} in {path.name}: unknown provenance class {class_name!r}; "
            f"expected one of {', '.join(c.value for c in ProvenanceClass)}"
        ) from exc

    return Parameter(
        id=entry_id,
        description=str(raw["description"]),
        value=float(raw["value"]),
        units=str(raw["units"]),
        provenance_class=provenance,
        source=str(raw.get("source", "")),
        uncertainty=_parse_uncertainty(raw.get("uncertainty"), entry_id=entry_id),
        sweep_range=_parse_sweep_range(raw.get("sweep_range"), entry_id=entry_id),
        affects=tuple(str(a) for a in raw.get("affects", ())),
        category=str(raw["category"]),
        defined_in=path,
    )


class ParameterRegistry:
    """An immutable collection of registered parameters, keyed by id."""

    __slots__ = ("_entries",)

    def __init__(self, entries: Mapping[str, Parameter]) -> None:
        self._entries: Mapping[str, Parameter] = MappingProxyType(dict(entries))

    @classmethod
    def load(cls, paths: Sequence[Path]) -> ParameterRegistry:
        """Load and merge registry files.

        Merging is strict: a duplicate id raises rather than overriding. Last-one-wins
        would make the answer depend on file iteration order, which doc 00 E3 forbids,
        and the losing entry would never announce itself.
        """
        entries: dict[str, Parameter] = {}
        for path in paths:
            document = yaml.safe_load(path.read_text())
            if document is None:
                continue
            if not isinstance(document, list):
                raise ValueError(
                    f"{path.name}: a registry file must contain a list of entries, "
                    f"got {type(document).__name__}"
                )
            for raw in document:
                entry = _parse_entry(raw, path=path)
                if entry.id in entries:
                    previous = entries[entry.id].defined_in
                    raise ValueError(
                        f"duplicate parameter id {entry.id!r}: defined in "
                        f"{previous.name if previous else '<unknown>'} and again in "
                        f"{path.name}. Ids must be unique across the whole registry."
                    )
                entries[entry.id] = entry
        return cls(entries)

    # ── access ──────────────────────────────────────────────────────────────────

    @property
    def entries(self) -> Mapping[str, Parameter]:
        """The underlying mapping, read-only."""
        return self._entries

    def __getitem__(self, entry_id: str) -> Parameter:
        try:
            return self._entries[entry_id]
        except KeyError:
            near = difflib.get_close_matches(
                entry_id, self._entries, n=_MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
            )
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise KeyError(f"no registered parameter {entry_id!r}.{hint}") from None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Parameter]:
        """Iterate entries in id order.

        Sorted, not insertion-ordered: doc 00 E3 requires that anything hashing or
        writing the registry sees the same order in every process.
        """
        return (self._entries[key] for key in sorted(self._entries))

    def __contains__(self, entry_id: object) -> bool:
        return entry_id in self._entries

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def quantity(self, entry_id: str) -> Quantity:
        return self[entry_id].quantity

    def value_in(self, entry_id: str, units: str) -> Magnitude:
        """The nominal value converted to ``units`` and stripped.

        This is what physics code calls, so it performs the doc 08 §5 boundary
        assertion on the caller's behalf rather than trusting them to remember.
        """
        return magnitude_in(self[entry_id].quantity, units)

    # ── the doc 00 C1 metric ────────────────────────────────────────────────────

    def count_by_class(self) -> Mapping[ProvenanceClass, int]:
        """Entry count per provenance class, including zeros.

        Zeros are included deliberately: a metric that omits absent classes cannot
        distinguish "no ASSUMED entries" from "nobody looked".
        """
        counts = dict.fromkeys(ProvenanceClass, 0)
        for entry in self._entries.values():
            counts[entry.provenance_class] += 1
        return MappingProxyType(counts)

    @property
    def assumed_count(self) -> int:
        """The doc 00 C1 tracked metric. Its increase fails the build."""
        return sum(1 for entry in self._entries.values() if entry.is_defect)

    def defects(self, categories: frozenset[str] | None = None) -> tuple[Parameter, ...]:
        """ASSUMED-class entries, optionally restricted to some categories."""
        return tuple(
            entry
            for entry in self
            if entry.is_defect and (categories is None or entry.category in categories)
        )

    def __repr__(self) -> str:
        counts = self.count_by_class()
        summary = ", ".join(f"{c.value}={counts[c]}" for c in ProvenanceClass if counts[c])
        return f"ParameterRegistry({len(self)} entries: {summary})"


@lru_cache(maxsize=1)
def default_registry() -> ParameterRegistry:
    """The registry that ships with the package.

    Memoised: doc 08 §5 calls the registry "the sole source of numeric defaults", and a
    source of truth that is re-parsed per call can drift between callers within one run
    if a file changes underneath them.
    """
    return ParameterRegistry.load(sorted(DEFAULT_REGISTRY_DIR.glob("*.yaml")))
