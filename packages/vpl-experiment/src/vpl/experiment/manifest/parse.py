"""Reading a manifest block strictly — doc 08 §6, doc 08 §1 principle 4.

Configuration is data, so every mistake in a manifest is a *data* error and has to be
reported as one: naming the block, the key, what was found and what was expected. A
stack trace pointing into a loader tells the person who mistyped ``descrition`` nothing
they can act on.

## Why unknown keys raise

The rule is the one :mod:`vpl.core.params.catalogue` already enforces on the registry:
**a typo'd key that is silently ignored is a setting that silently does not apply.** The
manifest is worse than the registry in this respect, because doc 08 §6 makes it the only
place an experiment is specified — a silently dropped ``enabled: false`` is a channel that
was supposed to be ablated and was not, and the ablation matrix of doc 07 §5.2 then
measures nothing while reporting a number.

The suggestion machinery exists for the same reason it does in the registry: the failure
has to be cheaper to fix than to work around, or people work around it.

## Why a bare number is refused where a quantity belongs

doc 08 §6 writes every physical quantity as ``{value: ..., units: ...}`` and doc 08 §5
forbids an undimensioned magnitude from crossing a module boundary. ``pressure: 5.0`` is
ambiguous between mTorr and Pa by four orders of magnitude, and the run would complete.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import pint

from vpl.core.provenance import ManifestValue
from vpl.core.units import Q_, UREG, Quantity

__all__ = [
    "UNITS_KEY",
    "VALUE_KEY",
    "UnknownKeyError",
    "block",
    "check_keys",
    "flag",
    "frozen",
    "integer",
    "member",
    "number",
    "plain",
    "quantity",
    "required",
    "strings",
    "text",
]

#: Similarity threshold for the "did you mean" hint. Matches the registry loader's, so a
#: user who has learned what a near-miss suggestion looks like in one place recognises it
#: in the other.
_SUGGESTION_CUTOFF: Final[float] = 0.6

#: More than a handful of suggestions stops being a hint.
_MAX_SUGGESTIONS: Final[int] = 3

#: The two keys doc 08 §6 writes a dimensional quantity with.
VALUE_KEY: Final[str] = "value"
UNITS_KEY: Final[str] = "units"


class UnknownKeyError(ValueError):
    """A manifest set a key nothing reads.

    A subclass of :class:`ValueError` because that is what the rest of the framework
    raises for input it cannot accept, so a caller that already handles a bad manifest
    keeps working while one that wants to offer an interactive correction (doc 08 §11's
    scenario designer) can name this case specifically.
    """


def _suggest(key: str, known: Sequence[str]) -> str:
    near = difflib.get_close_matches(key, known, n=_MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF)
    return f" Did you mean: {', '.join(near)}?" if near else ""


def block(value: object, *, where: str) -> Mapping[str, ManifestValue]:
    """Narrow a manifest value to a block, naming it if it is not one."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{where}: expected a mapping of keys to values, got {type(value).__name__} {value!r}"
        )

    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{where}: manifest keys are strings, got {key!r}")

    return {str(key): entry for key, entry in value.items()}


def check_keys(
    data: Mapping[str, ManifestValue],
    *,
    required_keys: Sequence[str],
    optional_keys: Sequence[str] = (),
    where: str,
) -> None:
    """Refuse a block that sets a key nothing reads, or omits one something needs."""
    known = (*required_keys, *optional_keys)

    unknown = sorted(set(data) - set(known))
    if unknown:
        first = unknown[0]
        raise UnknownKeyError(
            f"{where}: unrecognised key(s) {', '.join(unknown)}.{_suggest(first, known)} "
            f"This block reads: {', '.join(sorted(known))}. doc 08 §6 makes the manifest "
            "the only place an experiment is specified, so a key nothing reads is a "
            "setting that silently does not apply."
        )

    missing = sorted(set(required_keys) - set(data))
    if missing:
        raise ValueError(f"{where}: missing required key(s) {', '.join(missing)}")


def required(data: Mapping[str, ManifestValue], key: str, *, where: str) -> ManifestValue:
    """The value at ``key``, which :func:`check_keys` has already proved is present."""
    if key not in data:
        raise ValueError(f"{where}: missing required key {key}")
    return data[key]


def text(value: object, *, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{where}: expected a string, got {type(value).__name__} {value!r}")
    return value


def flag(value: object, *, where: str) -> bool:
    """A boolean, never coerced from a number.

    ``enabled: 1`` reads as ``True`` under coercion, and the ablation matrix of
    doc 07 §5.2 depends on the switches meaning exactly what the manifest wrote.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{where}: expected true or false, got {type(value).__name__} {value!r}")
    return value


def integer(value: object, *, where: str) -> int:
    """An integer. ``bool`` is a subclass of ``int`` and is refused here."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an integer, got {type(value).__name__} {value!r}")
    return value


def number(value: object, *, where: str) -> float:
    """A number, accepting an integer. YAML writes ``4000`` and ``1.0e17`` alike."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{where}: expected a number, got {type(value).__name__} {value!r}")
    return float(value)


def member[E: StrEnum](value: object, enum_cls: type[E], *, where: str) -> E:
    """Coerce to a member of ``enum_cls``, listing the alternatives on failure.

    A boolean gets its own message. doc 08 §6 writes ``calibration: estimated`` and
    comments "NOT 'true'"; had it written the alternative, YAML would have produced the
    *boolean* ``True`` rather than the mode name, and a message about an unknown mode
    would send the reader looking for a mode that is in fact spelled correctly.
    """
    permitted = ", ".join(item.value for item in enum_cls)

    if isinstance(value, bool):
        raise ValueError(
            f"{where}: YAML read {str(value).lower()} as a boolean, not as a mode name; "
            f'quote it — {where.rsplit(".", 1)[-1]}: "{str(value).lower()}" — if you '
            f"meant one of [{permitted}]."
        )

    try:
        return enum_cls(text(value, where=where))
    except ValueError as exc:
        raise ValueError(f"{where}: must be one of [{permitted}], got {value!r}") from exc


def strings(value: object, *, where: str) -> tuple[str, ...]:
    """A sequence of strings, in the order the manifest wrote them.

    Order is preserved and is part of the manifest's identity: doc 08 §6 orders
    instruments and noise sources meaningfully, and :func:`vpl.core.provenance
    .manifest_sha256` hashes sequence order for exactly that reason.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{where}: expected a list, got {type(value).__name__} {value!r}")
    return tuple(text(item, where=f"{where}[{index}]") for index, item in enumerate(value))


def quantity(value: object, *, where: str) -> Quantity:
    """A dimensional quantity written the way doc 08 §6 writes one.

    Raises:
        TypeError: If it is not a ``{value, units}`` mapping — most often a bare number,
            which has no units to check and would silently mean whichever unit the reader
            assumed.
        ValueError: If the units are not ones the shared registry knows.
    """
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{where}: a physical quantity is written as "
            f"{{{VALUE_KEY}: ..., {UNITS_KEY}: ...}} (doc 08 §6), got "
            f"{type(value).__name__} {value!r}. A bare magnitude carries no units, and "
            "doc 08 §5 does not let one cross a module boundary."
        )

    fields = block(value, where=where)
    check_keys(fields, required_keys=(VALUE_KEY, UNITS_KEY), where=where)

    units = text(required(fields, UNITS_KEY, where=where), where=f"{where}.{UNITS_KEY}")
    try:
        UREG.Unit(units)
    except (pint.UndefinedUnitError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: unrecognised units {units!r}: {exc}") from exc

    magnitude = number(required(fields, VALUE_KEY, where=where), where=f"{where}.{VALUE_KEY}")
    return Q_(magnitude, units)


def frozen(value: ManifestValue, *, where: str) -> ManifestValue:
    """Deep-freeze a loaded document.

    Mappings become read-only views and sequences become tuples, so a solver handed a
    block cannot edit the manifest the run is meant to be reproducible from. Insertion
    order is preserved: it is irrelevant to the digest, which sorts keys, and it is what a
    reader sees when the archived manifest is printed back.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                text(key, where=where): frozen(entry, where=f"{where}.{key}" if where else str(key))
                for key, entry in value.items()
            }
        )

    if isinstance(value, Sequence):
        return tuple(frozen(item, where=f"{where}[{index}]") for index, item in enumerate(value))

    raise TypeError(
        f"{where}: {type(value).__name__} is not a value a manifest can hold. "
        "doc 08 §6 manifests are YAML: scalars, sequences and mappings only."
    )


def plain(value: ManifestValue) -> ManifestValue:
    """Undo :func:`frozen` into the plain containers JSON can serialise.

    :func:`vpl.core.provenance.manifest_sha256` is JSON-based and a ``MappingProxyType``
    is not JSON-serialisable, so the digest is taken over this form. It is also what a
    caller gets from :meth:`~vpl.experiment.Manifest.as_document`, which is why it is a
    copy: a caller that edits what it was handed must not be editing the manifest.
    """
    if isinstance(value, Mapping):
        return {str(key): plain(entry) for key, entry in value.items()}
    if not isinstance(value, str) and isinstance(value, Sequence):
        return [plain(item) for item in value]
    return value
