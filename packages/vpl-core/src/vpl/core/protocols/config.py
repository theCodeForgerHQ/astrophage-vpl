"""Configuration blocks handed to ``configure`` — doc 08 §4, §6.

**This is a placeholder with a deliberate seam, and it should be read as one.**

Doc 08 §1 principle 4 says configuration is data, not code, and doc 08 §6 specifies one
declarative manifest per experiment, loaded through Hydra + OmegaConf (doc 08 §2, "buy").
That engine does not exist yet — it is WBS 1.3 in doc 11 §2 — and inventing a rich
configuration schema here would be building the thing the build/buy table says to buy,
which doc 14 RT-07 scores as the most likely way this project wastes its time.

So this module is the thinnest honest thing: an immutable, typed view over one block of a
loaded manifest, with accessors that fail loudly on the wrong type. **The seam** is
:attr:`Config.values`. When the manifest engine lands, it will construct these from
resolved OmegaConf nodes and nothing downstream changes, because a solver only ever sees
``cfg.require_int("n_ppc")`` and never how the mapping was produced.

What is *not* here, on purpose: interpolation (``dz: lambda_D/2`` is stored as the string
the manifest wrote, for the physics package to resolve against the parameter registry),
defaults, schema validation, and merging. All four belong to Hydra.

## Why three types and not one

``ForwardSolver.configure`` takes a ``SolverConfig`` and ``Instrument.configure`` takes an
``InstrumentConfig``. They are separate classes rather than one class with a label so that
passing the instrument block of a manifest to a solver is a **type error** rather than a
runtime ``KeyError`` several stages into a sweep. The checking here is nominal, not
structural — the three types are structurally identical and that is exactly why the
distinction has to be by name.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["Config", "ConfigValue", "InstrumentConfig", "InverseConfig", "SolverConfig"]

#: Anything a manifest block may hold once loaded from YAML — doc 08 §6.
type ConfigValue = (
    str | int | float | bool | Sequence[ConfigValue] | Mapping[str, ConfigValue] | None
)


def _frozen(value: object, *, path: str) -> ConfigValue:
    """Deep-freeze one manifest value, rejecting anything YAML could not have produced.

    Mappings are sorted on the way in. Doc 00 E3 promises bit-for-bit reproducibility, and
    anything that iterates a config and reduces over it — a provenance hash, most
    obviously — must see one order regardless of how the loader walked the file.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if isinstance(value, Mapping):
        frozen: dict[str, ConfigValue] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{path}: manifest keys are strings, got {key!r}")
            frozen[key] = _frozen(value[key], path=f"{path}.{key}" if path else key)
        return MappingProxyType(frozen)

    if isinstance(value, Sequence):
        # Tuples, so that a solver handed a list of chords cannot append to the manifest
        # the run is meant to be reproducible from.
        return tuple(_frozen(item, path=f"{path}[{index}]") for index, item in enumerate(value))

    raise TypeError(
        f"{path}: {type(value).__name__} is not a value a manifest can hold. "
        "doc 08 §6 manifests are YAML: scalars, sequences and mappings only."
    )


@dataclass(frozen=True, slots=True)
class Config:
    """One immutable block of a loaded manifest.

    Attributes:
        values: The block's contents, deep-frozen and key-sorted at construction. This is
            the seam the manifest engine will fill; see the module docstring.
    """

    values: Mapping[str, ConfigValue]

    def __post_init__(self) -> None:
        # Copy before freezing: the caller's dict is theirs to keep mutating, and a
        # config aliasing it would change under a run that had already been provenanced.
        frozen = _frozen(dict(self.values), path="")
        object.__setattr__(self, "values", frozen)

    # ── reading ─────────────────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> ConfigValue:
        try:
            return self.values[key]
        except KeyError:
            raise KeyError(self._missing(key)) from None

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue:
        """The value at ``key``, or ``default`` if the manifest did not set it."""
        return self.values.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def _missing(self, key: str) -> str:
        present = ", ".join(self.values) or "nothing"
        return (
            f"{type(self).__name__} has no {key!r}. This block sets: {present}. "
            "The manifest and the code that reads it disagree (doc 08 §6)."
        )

    def _wrong_type(self, key: str, value: object, expected: str) -> TypeError:
        return TypeError(
            f"{type(self).__name__}[{key!r}] is {type(value).__name__} {value!r}, "
            f"expected {expected}"
        )

    # ── typed accessors ─────────────────────────────────────────────────────────

    def require_str(self, key: str) -> str:
        """The string at ``key``, or a failure naming the key and what was found."""
        value = self[key]
        if not isinstance(value, str):
            raise self._wrong_type(key, value, "a string")
        return value

    def require_bool(self, key: str) -> bool:
        """The boolean at ``key``.

        A number is refused rather than coerced: ``enabled: 1`` reads as ``True`` under
        coercion, and the noise-ablation matrix of doc 07 §5.2 depends on the switches
        meaning exactly what the manifest wrote.
        """
        value = self[key]
        if not isinstance(value, bool):
            raise self._wrong_type(key, value, "a boolean")
        return value

    def require_int(self, key: str) -> int:
        """The integer at ``key``.

        ``bool`` is a subclass of ``int``, so an ``enabled: true`` landing in a count
        field would otherwise read as ``1`` and run a sweep of one case.
        """
        value = self[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._wrong_type(key, value, "an integer")
        return value

    def require_float(self, key: str) -> float:
        """The number at ``key``, as a float.

        An integer is accepted. YAML writes ``4000`` and ``1.0e17`` for quantities of the
        same kind, and refusing the first would push a ``float()`` call into every
        ``configure`` in the framework.
        """
        value = self[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._wrong_type(key, value, "a number")
        return float(value)

    def section(self, key: str) -> Config:
        """A nested block, e.g. ``mesh`` inside ``forward`` (doc 08 §6).

        Returns a plain :class:`Config` rather than this class: a sub-block of a solver
        configuration is not itself a solver configuration, and typing it as one would
        let it be passed to ``configure``.
        """
        value = self[key]
        if not isinstance(value, Mapping):
            raise self._wrong_type(key, value, "a nested block")
        return Config(values=value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self.values) or 'empty'})"


class SolverConfig(Config):
    """The ``forward:`` block of a manifest — what ``ForwardSolver.configure`` takes."""

    __slots__ = ()


class InstrumentConfig(Config):
    """One entry of the ``instruments:`` list — what ``Instrument.configure`` takes."""

    __slots__ = ()


class InverseConfig(Config):
    """The ``inverse:`` block — what ``InverseEngine.configure`` takes."""

    __slots__ = ()
