"""LXCat cross-section tables — doc 09 §2.1, doc 03 §4.5.

LXCat publishes evaluated electron- and ion-impact cross sections as a plain-text export:
a header naming the database and the citation it requires, then one block per process.
This module turns that text into typed objects. It does **not** ship any of it — doc 09
§5 forbids redistributing the raw tables, so what the repository contains is this parser
and the version lock of :mod:`vpl.physics.atomic.dataset`, never a data file.

## Why a parser at all, when doc 08 §2 says "buy"

Doc 08 §2 buys the *data* from LXCat and buys the *Boltzmann solver* from BOLSIG+. It
does not name a library that reads the export format into anything this project can hold,
and the format is a documented, stable, twenty-line grammar. Parsing it is not a
reimplementation of anything.

## What the validation is for

Every check below corresponds to a misreading that produces a plausible number rather
than an error:

- **The column units.** An Angstrom-squared table read as m² is twenty orders of
  magnitude small, and the only symptom is a rate coefficient of zero.
- **The threshold's source.** The bare number under the reaction is a *mass ratio* for
  momentum transfer and a *threshold energy* for an inelastic process. Swapping them puts
  a 1.4e-5 eV threshold on the elastic channel, or a 11.5 mass ratio on nothing.
- **The reactant separator.** ``E + E + Ar+`` split on ``"+"`` loses the ion. The
  separator is a plus surrounded by whitespace.
- **The process label.** LXCat states the process twice, as the block keyword and as the
  tail of the ``PROCESS:`` line. When they disagree the file has been edited by hand.
- **Sub-threshold cross sections.** An inelastic process cannot occur below its
  threshold. A table that says otherwise has had its threshold misread.

## Charge exchange

Doc 03 §4.5: "Getting the CX cross section and its energy dependence right matters more
than any other atomic-data choice." ``CHARGE EXCHANGE`` is therefore a first-class
keyword with its own accessor, rather than something a caller filters for by string — a
parser that did not recognise the keyword would drop the block and leave behind a set
that looked complete.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "REQUIRED_ELECTRON_PROCESSES",
    "REQUIRED_ION_PROCESSES",
    "THRESHOLD_CONSISTENCY_TOLERANCE",
    "CrossSection",
    "CrossSectionSet",
    "LxcatParseError",
    "ProcessType",
    "parse_lxcat",
]

#: A tabulated column, in SI magnitudes.
type FloatArray = NDArray[np.float64]

#: Relative agreement demanded between the two places LXCat states a threshold — the
#: bare number under the reaction and the ``E = ... eV`` of the ``PARAM.:`` line. Files
#: quote both to at least four significant figures, so anything looser than this would
#: accept a genuinely edited file and anything tighter would trip on the rounding.
THRESHOLD_CONSISTENCY_TOLERANCE: Final[float] = 1e-3

#: The fewest points a table can have and still be interpolable.
_MINIMUM_TABLE_POINTS: Final[int] = 2

#: How many whitespace-separated numbers a data row carries.
_COLUMNS_PER_ROW: Final[int] = 2

#: The shortest run of dashes LXCat uses to delimit a data block.
_MINIMUM_RULER_LENGTH: Final[int] = 3

_ARROW: Final[str] = "->"

#: Reactants and products are separated by a plus **surrounded by whitespace**. The bare
#: character also spells the charge of an ion.
_SPECIES_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"\s+\+\s+")

#: An all-capitals header field at the start of a line. Deliberately strict: a lowercase
#: continuation line carrying a colon (a journal reference, say) must not be mistaken for
#: a new field and silently discard the line it belongs to.
_FIELD: Final[re.Pattern[str]] = re.compile(r"^(?P<key>[A-Z][A-Z0-9_.\- ]*?)\s*:\s*(?P<value>.*)$")

#: The threshold as ``PARAM.:`` states it.
_PARAM_THRESHOLD: Final[re.Pattern[str]] = re.compile(r"\bE\s*=\s*([0-9.eE+-]+)\s*eV")

#: The declared column units. Both are checked; see the module docstring.
_COLUMNS: Final[re.Pattern[str]] = re.compile(
    r"^\s*Energy\s*\((?P<energy>[^)]*)\)\s*\|\s*Cross\s+section\s*\((?P<sigma>[^)]*)\)\s*$",
    re.IGNORECASE,
)

_ENERGY_UNIT: Final[str] = "ev"
_CROSS_SECTION_UNITS: Final[frozenset[str]] = frozenset({"m2", "m^2", "m**2"})

_DATABASE_KEY: Final[str] = "DATABASE"
_REFERENCE_KEY: Final[str] = "HOW TO REFERENCE"
_PROCESS_KEY: Final[str] = "PROCESS"
_SPECIES_KEY: Final[str] = "SPECIES"
_COLUMNS_KEY: Final[str] = "COLUMNS"
_PARAM_KEYS: Final[tuple[str, ...]] = ("PARAM.", "PARAM")


class LxcatParseError(ValueError):
    """An LXCat export could not be read, or could be read two ways."""


class ProcessType(StrEnum):
    """The collision processes doc 03 §4.5 tabulates, as LXCat spells them.

    A ``StrEnum`` because manifests are data (doc 08 §1 principle 4): an ablation that
    switches off a channel names it as a string and gets back a checked member.
    """

    #: Ion elastic scattering, and electron elastic where a database publishes it.
    ELASTIC = "ELASTIC"

    #: Momentum-transfer cross section. **Not interchangeable with ELASTIC** outside a
    #: two-term Boltzmann solve: it is the elastic cross section weighted by
    #: ``1 - cos(theta)``, so a Monte-Carlo collision routine that sampled it as an
    #: elastic cross section would get both the rate and the scattering angle wrong.
    #: LXCat's Phelps argon set publishes this rather than ELASTIC.
    EFFECTIVE = "EFFECTIVE"

    #: Electron-impact excitation. One block per upper level.
    EXCITATION = "EXCITATION"

    #: Electron-impact ionisation.
    IONIZATION = "IONIZATION"

    #: Electron attachment. Absent from argon; present in any electronegative admixture,
    #: and cheaper to recognise than to discover missing later.
    ATTACHMENT = "ATTACHMENT"

    #: ``Ar+ + Ar -> Ar + Ar+``. Doc 03 §4.5: "the single most important collision for
    #: this project".
    CHARGE_EXCHANGE = "CHARGE EXCHANGE"

    @property
    def has_threshold(self) -> bool:
        """Whether the process has an energetic threshold below which it cannot occur.

        Only the inelastic electron processes do. Symmetric resonant charge exchange is
        exothermic to zeroth order, and momentum transfer has no threshold at all — a
        threshold read in for either would zero the cross section over exactly the
        low-energy range that produces the IEDF structure of doc 03 §4.5.
        """
        return self in _THRESHOLD_PROCESSES

    @property
    def is_momentum_transfer(self) -> bool:
        """Whether this is the channel doc 03 §4.5 calls "elastic", in either spelling."""
        return self in _MOMENTUM_TRANSFER_PROCESSES


_THRESHOLD_PROCESSES: Final[frozenset[ProcessType]] = frozenset(
    {ProcessType.EXCITATION, ProcessType.IONIZATION}
)

_MOMENTUM_TRANSFER_PROCESSES: Final[frozenset[ProcessType]] = frozenset(
    {ProcessType.ELASTIC, ProcessType.EFFECTIVE}
)

#: What doc 03 §4.5 requires of an electron database.
#:
#: Momentum transfer is deliberately absent: doc 03 §4.5 asks for "electron elastic" and
#: the databases disagree about whether that is published as ELASTIC or as EFFECTIVE, so
#: the requirement is expressed by :meth:`CrossSectionSet.momentum_transfer` instead of
#: by a membership test that Phelps would fail and Biagi would pass.
REQUIRED_ELECTRON_PROCESSES: Final[frozenset[ProcessType]] = frozenset(
    {ProcessType.EXCITATION, ProcessType.IONIZATION}
)

#: What doc 03 §4.5 requires of the ion database.
REQUIRED_ION_PROCESSES: Final[frozenset[ProcessType]] = frozenset(
    {ProcessType.ELASTIC, ProcessType.CHARGE_EXCHANGE}
)

#: The tail of a ``PROCESS:`` line, mapped to the keyword it must agree with. Both
#: spellings of "ionisation" appear in the wild.
_PROCESS_LABELS: Final[Mapping[str, ProcessType]] = MappingProxyType(
    {
        "ELASTIC": ProcessType.ELASTIC,
        "EFFECTIVE": ProcessType.EFFECTIVE,
        "EXCITATION": ProcessType.EXCITATION,
        "IONIZATION": ProcessType.IONIZATION,
        "IONISATION": ProcessType.IONIZATION,
        "ATTACHMENT": ProcessType.ATTACHMENT,
        "CHARGE EXCHANGE": ProcessType.CHARGE_EXCHANGE,
    }
)


# ── the data ────────────────────────────────────────────────────────────────────


def _tabulated(values: NDArray[np.float64], *, what: str) -> FloatArray:
    """Copy a column, check it, and lock it against writes.

    The same treatment :class:`vpl.core.state.ScalarField` gives a field, for the same
    reason: a cross section is loaded once and read by every rate integral in the run,
    so an in-place edit anywhere would silently change results everywhere else.
    """
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} must be finite; found nan or inf")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, eq=False)
class CrossSection:
    """One tabulated ``sigma(E)`` for one process.

    Attributes:
        process: Which of doc 03 §4.5's channels this is.
        database: The LXCat database it came from. Carried on every section because the
            spread between databases is a budgeted error term (doc 06 §4 term 2), and a
            number that cannot say which set produced it cannot contribute to it.
        projectile: The incident species, from ``SPECIES:``.
        target: The struck species.
        reactants: The left-hand side of the reaction, as the file writes it. Kept
            alongside ``projectile``/``target`` because ``SPECIES: e / Ar`` and
            ``PROCESS: E + Ar -> ...`` do not spell the electron the same way, and the
            reaction string is what identifies a channel in a report.
        products: The right-hand side of the reaction.
        threshold_ev: The energetic threshold, for the inelastic processes that have one.
        mass_ratio: ``m/M``, for the momentum-transfer processes that publish it.
        energy_ev: Strictly increasing energy axis. Read-only.
        sigma_m2: Cross section on that axis. Read-only.
        parameters: The block's header fields, verbatim, for the provenance record.
    """

    process: ProcessType
    database: str
    projectile: str
    target: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    threshold_ev: float | None
    mass_ratio: float | None
    energy_ev: FloatArray
    sigma_m2: FloatArray
    parameters: Mapping[str, str]

    def __post_init__(self) -> None:
        energy = _tabulated(self.energy_ev, what="energy axis")
        sigma = _tabulated(self.sigma_m2, what="cross section")

        if energy.size != sigma.size:
            raise ValueError(
                f"energy and cross-section columns must be the same length, got "
                f"{energy.size} and {sigma.size}"
            )
        if energy.size < _MINIMUM_TABLE_POINTS:
            raise ValueError(
                f"a cross section needs at least two points to be interpolable, got {energy.size}"
            )
        if np.any(energy < 0.0):
            raise ValueError("energy axis contains a negative energy")
        if not np.all(np.diff(energy) > 0.0):
            raise ValueError(
                "energy axis must be strictly increasing; a repeated or reversed point "
                "makes the interpolation ambiguous"
            )
        if np.any(sigma < 0.0):
            raise ValueError("cross section contains a negative value")
        if not self.reactants or not self.products:
            raise ValueError("a reaction needs at least one reactant and one product")

        self._check_threshold(energy, sigma)

        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(self, "sigma_m2", sigma)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def _check_threshold(self, energy: FloatArray, sigma: FloatArray) -> None:
        """Reject a threshold that contradicts the process or the table."""
        if self.threshold_ev is None:
            if self.process.has_threshold:
                raise ValueError(
                    f"{self.process} needs a threshold energy; without one the rate "
                    "coefficient would be finite where it must be exactly zero"
                )
            return

        if self.threshold_ev < 0.0:
            raise ValueError(f"threshold energy cannot be negative, got {self.threshold_ev}")

        below = energy < self.threshold_ev
        if np.any(sigma[below] > 0.0):
            raise ValueError(
                f"{self.process} has a non-zero cross section below its {self.threshold_ev} eV "
                "threshold. An inelastic process cannot occur below threshold, so the "
                "threshold has been misread."
            )

    @property
    def n_points(self) -> int:
        return int(self.energy_ev.size)

    @property
    def reaction(self) -> str:
        """The reaction as the published equation, reassembled from what was parsed.

        This is the channel's identity in a report and in
        :func:`~vpl.physics.atomic.interpolation.interpolate_set`, so it is rebuilt from
        the parsed sides rather than kept as the raw line: a round trip that changes the
        string would mean the parse changed the reaction.
        """
        return f"{' + '.join(self.reactants)} {_ARROW} {' + '.join(self.products)}"

    @property
    def energy_range_ev(self) -> tuple[float, float]:
        """``(lowest, highest)`` tabulated energy."""
        return float(self.energy_ev[0]), float(self.energy_ev[-1])

    @property
    def peak_sigma_m2(self) -> float:
        return float(self.sigma_m2.max())

    def __repr__(self) -> str:
        return (
            f"CrossSection({self.process}, {self.reaction!r}, {self.database!r}, "
            f"{self.n_points} points)"
        )


@dataclass(frozen=True, slots=True, eq=False)
class CrossSectionSet:
    """Every process one LXCat database publishes for one gas.

    Attributes:
        database: The database name from the export header.
        sections: The parsed blocks, in file order.
        reference: The ``HOW TO REFERENCE:`` wording the database asks to be cited with.
            doc 09 §2.1 makes citation of the *specific* database a condition of use, and
            the required wording ships inside the export — reconstructing it later from
            memory is how a citation ledger stops being traceable.
    """

    database: str
    sections: tuple[CrossSection, ...]
    reference: str

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError(
                f"database {self.database!r} yielded at least one block but no cross "
                "sections; an empty set would be indistinguishable from a complete one"
            )

    def __len__(self) -> int:
        return len(self.sections)

    def __iter__(self) -> Iterator[CrossSection]:
        return iter(self.sections)

    def __getitem__(self, index: int) -> CrossSection:
        return self.sections[index]

    def process_types(self) -> frozenset[ProcessType]:
        return frozenset(section.process for section in self.sections)

    def of_type(self, process: ProcessType) -> tuple[CrossSection, ...]:
        return tuple(section for section in self.sections if section.process is process)

    def require(self, process: ProcessType) -> CrossSection:
        """The single section of that process, or an error naming how many there were.

        Never "the first one". An argon set has one ionisation channel but many
        excitation channels, and silently returning the first would make a rate
        coefficient depend on the order blocks happen to appear in the file.
        """
        found = self.of_type(process)
        if len(found) != 1:
            raise LookupError(
                f"database {self.database!r} has {len(found)} {process} sections, expected "
                f"exactly one. Available: "
                f"{', '.join(sorted(p.value for p in self.process_types()))}."
            )
        return found[0]

    def charge_exchange(self) -> CrossSection:
        """``Ar+ + Ar -> Ar + Ar+`` — doc 03 §4.5's most consequential channel."""
        return self.require(ProcessType.CHARGE_EXCHANGE)

    def excitations(self) -> tuple[CrossSection, ...]:
        """Every excitation channel, in file order — the input to doc 04 §2.2's ``K_ju``."""
        return self.of_type(ProcessType.EXCITATION)

    def momentum_transfer(self) -> CrossSection:
        """The elastic channel, whichever of the two spellings this database uses.

        Phelps publishes argon as ``EFFECTIVE``, Biagi as ``ELASTIC``. Doc 03 §4.5 asks
        for "electron elastic" and means whichever one is there; forcing the caller to
        know which would make swapping databases — the whole point of doc 09 §2.1 —
        a code change.
        """
        for process in (ProcessType.ELASTIC, ProcessType.EFFECTIVE):
            found = self.of_type(process)
            if found:
                return self.require(process)
        raise LookupError(
            f"database {self.database!r} publishes neither ELASTIC nor EFFECTIVE; there "
            "is no momentum-transfer channel to give the Boltzmann solver."
        )

    def missing_processes(self, required: frozenset[ProcessType]) -> tuple[ProcessType, ...]:
        """Which of ``required`` this set does not carry, in a stable order.

        Returned rather than raised: a caller assembling doc 03 §4.5's collision list
        wants to report everything that is absent at once, not the first thing.
        """
        return tuple(sorted(required - self.process_types(), key=lambda p: p.value))

    def __repr__(self) -> str:
        processes = ", ".join(sorted(p.value for p in self.process_types()))
        return f"CrossSectionSet({self.database!r}, {len(self)} sections: {processes})"


# ── parsing ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, eq=False)
class _Block:
    """One ``KEYWORD ... ----- data ----- `` block, before it is interpreted."""

    process: ProcessType
    parameter: float | None
    fields: Mapping[str, str]
    energy_ev: FloatArray
    sigma_m2: FloatArray
    line: int


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _is_ruler(text: str) -> bool:
    return len(text) >= _MINIMUM_RULER_LENGTH and set(text) == {"-"}


def _keyword(text: str) -> ProcessType | None:
    try:
        return ProcessType(text.upper())
    except ValueError:
        return None


def _scan_header(lines: list[str]) -> Mapping[str, str]:
    """The export's top-level fields, first occurrence winning.

    Read in a pass of its own so that a file with no ``DATABASE:`` is reported as such
    rather than as whatever the first malformed block happens to be — the missing
    database is the actionable fact, and it is the one that decides whether the data can
    be cited at all.
    """
    header: dict[str, str] = {}
    for raw in lines:
        match = _FIELD.match(raw.strip())
        if match is not None:
            header.setdefault(match["key"].strip(), match["value"].strip())
    return header


def _parse_data_row(text: str, *, line: int) -> tuple[float, float]:
    cells = text.split()
    if len(cells) != _COLUMNS_PER_ROW:
        raise LxcatParseError(
            f"line {line}: a data row must be two numbers (energy, cross section), got "
            f"{len(cells)}: {text!r}"
        )
    energy, sigma = (_as_float(cell) for cell in cells)
    if energy is None or sigma is None:
        raise LxcatParseError(f"line {line}: data row is not a pair of numbers: {text!r}")
    return energy, sigma


def _parse_block(
    lines: list[str], index: int, *, process: ProcessType, start: int
) -> tuple[_Block, int]:
    """Read one block, returning it and the index just past its closing ruler."""
    total = len(lines)
    while index < total and not lines[index].strip():
        index += 1
    if index >= total:
        raise LxcatParseError(f"line {start}: {process} block ends before its reaction line")
    # The reaction shorthand under the keyword is a display label; the authoritative
    # reactants and products come from the PROCESS: field below.
    index += 1

    parameter: float | None = None
    fields: dict[str, str] = {}
    last_key: str | None = None
    energy: list[float] = []
    sigma: list[float] = []
    in_table = False

    while index < total:
        raw = lines[index]
        stripped = raw.strip()
        line = index + 1
        index += 1

        if not stripped:
            continue

        if _is_ruler(stripped):
            if not in_table:
                in_table = True
                continue
            return (
                _Block(
                    process=process,
                    parameter=parameter,
                    fields=fields,
                    energy_ev=np.asarray(energy, dtype=np.float64),
                    sigma_m2=np.asarray(sigma, dtype=np.float64),
                    line=start,
                ),
                index,
            )

        if in_table:
            row_energy, row_sigma = _parse_data_row(stripped, line=line)
            energy.append(row_energy)
            sigma.append(row_sigma)
            continue

        if parameter is None and not fields:
            bare = _as_float(stripped)
            if bare is not None:
                parameter = bare
                continue

        match = _FIELD.match(stripped)
        if match is not None:
            last_key = match["key"].strip()
            fields[last_key] = match["value"].strip()
            continue

        if last_key is None:
            raise LxcatParseError(
                f"line {line}: {stripped!r} is neither a header field nor a continuation "
                f"of one, in the {process} block beginning at line {start}"
            )
        fields[last_key] = f"{fields[last_key]} {stripped}".strip()

    raise LxcatParseError(
        f"line {start}: the {process} block has no closing ruler; its data table runs to "
        "the end of the file"
    )


def _split_process(
    value: str, *, process: ProcessType, line: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reactants and products from a ``PROCESS:`` line, with the label cross-checked."""
    body, comma, tail = value.rpartition(",")
    if comma:
        labelled = _PROCESS_LABELS.get(tail.strip().upper())
        if labelled is not None:
            if labelled is not process:
                raise LxcatParseError(
                    f"line {line}: block keyword {process} but the PROCESS line is "
                    f"labelled {labelled}. The file states the process twice and the two "
                    "disagree."
                )
            value = body

    left, arrow, right = value.partition(_ARROW)
    if not arrow:
        raise LxcatParseError(f"line {line}: PROCESS line has no {_ARROW!r}: {value!r}")

    return (
        tuple(_SPECIES_SEPARATOR.split(left.strip())),
        tuple(_SPECIES_SEPARATOR.split(right.strip())),
    )


def _threshold_and_mass_ratio(
    block: _Block, fields: Mapping[str, str]
) -> tuple[float | None, float | None]:
    """Interpret the bare number under the reaction, per the process it belongs to."""
    declared: float | None = None
    for key in _PARAM_KEYS:
        param = fields.get(key)
        if param is not None:
            match = _PARAM_THRESHOLD.search(param)
            if match is not None:
                declared = float(match[1])
            break

    if not block.process.has_threshold:
        # The bare number is m/M for momentum transfer and absent otherwise.
        return None, block.parameter

    stated_twice = block.parameter is not None and declared is not None
    if stated_twice and not math.isclose(
        block.parameter or 0.0,
        declared or 0.0,
        rel_tol=THRESHOLD_CONSISTENCY_TOLERANCE,
        abs_tol=0.0,
    ):
        raise LxcatParseError(
            f"line {block.line}: the block states a threshold of {block.parameter} eV "
            f"and its PARAM. line states {declared} eV. Picking either would be a coin "
            "flip on the rate coefficient."
        )

    threshold = block.parameter if block.parameter is not None else declared
    if threshold is None:
        raise LxcatParseError(
            f"line {block.line}: {block.process} block states no threshold energy, in "
            "either the bare parameter line or PARAM.:"
        )
    return threshold, None


def _check_columns(fields: Mapping[str, str], *, line: int) -> None:
    """Reject a block whose declared column units are not eV and m².

    Not pedantry: LXCat exports Angstroms squared as readily as square metres, and a
    table twenty orders of magnitude small produces a rate coefficient of zero rather
    than an error.
    """
    declared = fields.get(_COLUMNS_KEY)
    if declared is None:
        raise LxcatParseError(
            f"line {line}: block has no COLUMNS: line, so the units of its table are "
            "undeclared and cannot be assumed"
        )
    match = _COLUMNS.match(declared)
    if match is None:
        raise LxcatParseError(f"line {line}: unrecognised COLUMNS: declaration {declared!r}")
    if match["energy"].strip().lower() != _ENERGY_UNIT:
        raise LxcatParseError(
            f"line {line}: COLUMNS declares the energy in {match['energy']!r}, expected eV"
        )
    if match["sigma"].strip().lower() not in _CROSS_SECTION_UNITS:
        raise LxcatParseError(
            f"line {line}: COLUMNS declares the cross section in {match['sigma']!r}, expected m2"
        )


def _build(block: _Block, *, database: str) -> CrossSection:
    fields = block.fields
    _check_columns(fields, line=block.line)

    process_line = fields.get(_PROCESS_KEY)
    if process_line is None:
        raise LxcatParseError(
            f"line {block.line}: block has no PROCESS: line, so its reactants and products "
            "are unknown"
        )
    reactants, products = _split_process(process_line, process=block.process, line=block.line)

    species = fields.get(_SPECIES_KEY, "")
    projectile, _, target = (part.strip() for part in species.partition("/"))
    if not projectile or not target:
        # Older exports omit SPECIES:. The reaction says the same thing, less directly.
        projectile = reactants[0]
        target = reactants[1] if len(reactants) > 1 else reactants[0]

    threshold, mass_ratio = _threshold_and_mass_ratio(block, fields)

    try:
        return CrossSection(
            process=block.process,
            database=database,
            projectile=projectile,
            target=target,
            reactants=reactants,
            products=products,
            threshold_ev=threshold,
            mass_ratio=mass_ratio,
            energy_ev=block.energy_ev,
            sigma_m2=block.sigma_m2,
            parameters=fields,
        )
    except ValueError as exc:
        raise LxcatParseError(f"line {block.line}: {exc}") from exc


def parse_lxcat(text: str, *, database: str | None = None) -> CrossSectionSet:
    """Parse an LXCat plain-text export.

    Args:
        text: The export as downloaded. Line endings are irrelevant.
        database: Overrides the ``DATABASE:`` header. Supplied by
            :class:`~vpl.physics.atomic.store.AtomicDataStore` so that the name in the
            provenance record is the registered dataset's, not whatever wording the
            export happened to use — and so that an export saved without its header is
            still loadable rather than a dead end.

    Returns:
        Every block in the file, in file order.

    Raises:
        LxcatParseError: If the export is unreadable, or readable two ways. The module
            docstring lists what is checked and why each check exists.
    """
    lines = text.splitlines()
    header = _scan_header(lines)

    name = database if database is not None else header.get(_DATABASE_KEY, "")
    if not name:
        raise LxcatParseError(
            "export has no DATABASE: header and no database= was given. doc 09 §2.1 makes "
            "citation of the specific database a condition of use, so a set that cannot "
            "name itself cannot be used."
        )

    blocks: list[_Block] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        start = index + 1
        index += 1
        if not stripped or stripped.startswith("*"):
            continue
        process = _keyword(stripped)
        if process is not None:
            block, index = _parse_block(lines, index, process=process, start=start)
            blocks.append(block)

    if not blocks:
        raise LxcatParseError(
            f"database {name!r}: the export contains no cross-section blocks. Expected one "
            f"of {', '.join(sorted(p.value for p in ProcessType))} on a line of its own."
        )

    return CrossSectionSet(
        database=name,
        sections=tuple(_build(block, database=name) for block in blocks),
        reference=header.get(_REFERENCE_KEY, ""),
    )
