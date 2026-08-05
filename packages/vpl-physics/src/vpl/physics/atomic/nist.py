"""NIST ASD line data — doc 09 §2.2, doc 02 §6.3, doc 04 §2.

Doc 04 §2.1 writes the volumetric emission coefficient as
``eps_ul = n_u A_ul h nu_ul / 4pi``, and doc 04 §2.2 lists what the collisional-radiative
model needs on top of the upper-state population: ``A_ul`` and the level data that fixes
``nu_ul`` and the statistical weights. This module supplies exactly that, for the ten
lines doc 02 §6.3 selected.

## The grade is not optional

Doc 09 §2.2:

    NIST assigns accuracy grades (AAA <= 0.3 % ... E > 50 %). The framework **ingests the
    grade and propagates it**. Ar I 811.53 nm carries a much better grade than several
    Ar II lines, and weighting lines by their data quality is straightforward, correct,
    and almost never done.

So :class:`Transition` has no default for :attr:`Transition.accuracy` and no way to omit
it: a transition probability that arrived without its grade would be indistinguishable
from one whose grade was AAA, and the doc 06 §4 term 2 uncertainty on the ``T_e``
inference would silently lose its atomic-data contribution. :meth:`LineList.weights` is
the payoff — the inverse-variance weighting the passage describes, refusing to run on a
grade E line unless the caller states what "> 50 %" is worth.

## What is deliberately not here

**Escape factors.** Doc 04 §2.2 needs ``Lambda_ul`` and doc 04 §2.3 is emphatic that an
optically-thin CR model gives a systematically wrong ``T_e``. But Holstein-Biberman
escape factors are a function of the *chamber geometry and density profile*, recomputed
as the profile changes — they are not atomic data and they do not come from a database.
Putting them here would put a chamber dimension inside a data loader.

**Level populations.** ``n_u`` is the CR model's unknown, which doc 08 §2 lists as Build.

## What is checked at load

The one cross-check worth its cost is ``h nu = E_k - E_i``. A row where the two disagree
has had a column misread — an Angstrom wavelength taken as nanometres, a cm⁻¹ level taken
as eV, two rows spliced by a broken delimiter — and every one of those produces a
plausible emissivity rather than an error. The tolerance is 1 %, which is loose enough to
pass the ~0.03 % offset every real ASD row above 200 nm carries (air wavelengths against
vacuum level energies) and tight enough that no unit confusion survives it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.constants import PLANCK, SPEED_OF_LIGHT
from vpl.core.params import Uncertainty
from vpl.core.units import magnitude_in

__all__ = [
    "DOC_02_LINE_SET",
    "LINE_MATCH_TOLERANCE_NM",
    "WAVELENGTH_ENERGY_TOLERANCE",
    "AccuracyGrade",
    "LineList",
    "LineSpec",
    "NistParseError",
    "Transition",
    "parse_nist_asd",
]

type FloatArray = NDArray[np.float64]

#: ``h c`` in eV nm — 1239.84. From CODATA via :mod:`vpl.core.constants`, never typed in:
#: doc 09 §2.5 makes a hand-entered fundamental constant "a plausible, undetectable and
#: catastrophic error", and this one converts every wavelength in the project.
_HC_EV_NM: Final[float] = float(magnitude_in(PLANCK * SPEED_OF_LIGHT, "eV * nm"))

#: How far a tabulated ``h nu`` may sit from ``E_k - E_i`` before the row is rejected.
#: See the module docstring: 0.03 % is the air-vacuum offset of a real row, and anything
#: that survives 1 % is not a unit confusion.
WAVELENGTH_ENERGY_TOLERANCE: Final[float] = 0.01

#: How far an ASD wavelength may sit from the one doc 02 §6.3 tabulates and still be the
#: same line. Doc 02 §6.3 quotes two decimals and ASD carries four, so the tolerance has
#: to absorb 0.014 nm of rounding on the worst line in the set (Ar II 488.00). The
#: closest pair in the set is 1.08 nm apart, so there is two orders of magnitude of
#: headroom before this could match the wrong line.
LINE_MATCH_TOLERANCE_NM: Final[float] = 0.05

#: Spectrum number to spectroscopic numeral. Ar I is the neutral, Ar II the singly
#: charged ion — the two doc 02 §6.3 uses, with room for the rest.
_ROMAN_NUMERALS: Final[tuple[str, ...]] = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")


class NistParseError(ValueError):
    """An ASD export could not be read, or could be read two ways."""


class AccuracyGrade(StrEnum):
    """NIST ASD's accuracy grade for a transition probability — doc 09 §2.2.

    Declared best-first, so iteration order is quality order.
    """

    AAA = "AAA"
    AA = "AA"
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C_PLUS = "C+"
    C = "C"
    D_PLUS = "D+"
    D = "D"
    #: "> 50 %". An open-ended bound, not a value — see :attr:`is_bounded`.
    E = "E"

    @property
    def relative_uncertainty(self) -> float:
        """The NIST bound, as a one-sigma relative uncertainty.

        Treating a stated *bound* as a one-sigma value is conservative for every grade
        except E, where it is not conservative at all — hence :attr:`is_bounded`.
        """
        return _GRADE_UNCERTAINTY[self]

    @property
    def is_bounded(self) -> bool:
        """Whether :attr:`relative_uncertainty` is an upper bound or merely a floor.

        ``False`` only for grade E, whose NIST definition is "> 50 %" and therefore has
        no upper end. Reporting 50 % for it would give the least trustworthy data in the
        set a finite, respectable weight.
        """
        return self is not AccuracyGrade.E

    def as_uncertainty(self) -> Uncertainty:
        """The grade as the project's one-sigma type — doc 02 §12.

        Deliberately :class:`vpl.core.params.Uncertainty` rather than a second spelling
        of the same idea: an uncertainty that could not be compared with a registered
        parameter's would have to be converted somewhere, and that conversion is where
        the relative/absolute distinction gets lost.
        """
        return Uncertainty(kind="relative", value=self.relative_uncertainty)


#: doc 09 §2.2's grade table, verbatim. Each value is the NIST bound for that grade.
_GRADE_UNCERTAINTY: Final[Mapping[AccuracyGrade, float]] = MappingProxyType(
    {
        AccuracyGrade.AAA: 0.003,
        AccuracyGrade.AA: 0.01,
        AccuracyGrade.A_PLUS: 0.02,
        AccuracyGrade.A: 0.03,
        AccuracyGrade.B_PLUS: 0.07,
        AccuracyGrade.B: 0.10,
        AccuracyGrade.C_PLUS: 0.18,
        AccuracyGrade.C: 0.25,
        AccuracyGrade.D_PLUS: 0.40,
        AccuracyGrade.D: 0.50,
        AccuracyGrade.E: 0.50,
    }
)


# ── the data ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Transition:
    """One radiative transition, with the grade that qualifies its ``A_ul``.

    Attributes:
        element: Chemical symbol, e.g. ``"Ar"``.
        spectrum: ASD's ``sp_num``: 1 for the neutral, 2 for the singly charged ion.
        wavelength_nm: The transition wavelength as ASD reports it — air above 200 nm,
            which is why the energy check below is not exact.
        a_ul_per_s: The Einstein A coefficient of doc 04 §2.1.
        accuracy: The NIST grade. **Required**; see the module docstring.
        e_lower_ev: Lower level energy.
        e_upper_ev: Upper level energy.
        g_lower: Lower level statistical weight.
        g_upper: Upper level statistical weight — the ``g_u`` of the CR model's
            detailed-balance relations (doc 04 §2.2).
        upper_level: ASD's configuration/term/J label for the upper level. Free text:
            doc 02 §6.3 names its levels in Paschen notation, which ASD does not use, so
            this identifies the level for a reader rather than for a matcher.
    """

    element: str
    spectrum: int
    wavelength_nm: float
    a_ul_per_s: float
    accuracy: AccuracyGrade
    e_lower_ev: float
    e_upper_ev: float
    g_lower: int
    g_upper: int
    upper_level: str

    def __post_init__(self) -> None:
        if not self.element.strip():
            raise ValueError("a transition needs an element symbol")
        if not 1 <= self.spectrum <= len(_ROMAN_NUMERALS):
            raise ValueError(f"spectrum number {self.spectrum} is outside I..{_ROMAN_NUMERALS[-1]}")
        if self.wavelength_nm <= 0.0:
            raise ValueError(f"wavelength must be positive, got {self.wavelength_nm} nm")
        if self.a_ul_per_s <= 0.0:
            raise ValueError(
                f"A_ul must be positive, got {self.a_ul_per_s} /s. A transition with no "
                "transition probability is a level pair, not a line."
            )
        if self.g_lower < 1 or self.g_upper < 1:
            raise ValueError(
                f"a statistical weight counts degenerate states and is at least 1, got "
                f"g_lower={self.g_lower}, g_upper={self.g_upper}"
            )
        if self.e_upper_ev <= self.e_lower_ev:
            raise ValueError(
                f"the upper level must lie above the lower level, got "
                f"{self.e_upper_ev} eV and {self.e_lower_ev} eV"
            )

        mismatch = abs(self.photon_energy_ev - self.energy_gap_ev) / self.photon_energy_ev
        if mismatch > WAVELENGTH_ENERGY_TOLERANCE:
            raise ValueError(
                f"{self.species} {self.wavelength_nm} nm: the photon energy "
                f"{self.photon_energy_ev:.6g} eV disagrees with the level gap "
                f"{self.energy_gap_ev:.6g} eV by {mismatch:.1%}. A column has been "
                "misread — Angstroms for nanometres, or cm^-1 for eV."
            )

    @property
    def species(self) -> str:
        """The doc 02 §6.3 spelling, e.g. ``"Ar I"``."""
        return f"{self.element} {_ROMAN_NUMERALS[self.spectrum - 1]}"

    @property
    def species_wavelength(self) -> tuple[str, float]:
        """The pair that identifies a line in a report or a table index."""
        return self.species, self.wavelength_nm

    @property
    def energy_gap_ev(self) -> float:
        """``E_k - E_i`` from the tabulated levels."""
        return self.e_upper_ev - self.e_lower_ev

    @property
    def photon_energy_ev(self) -> float:
        """``h nu = h c / lambda`` — the ``h nu_ul`` of doc 04 §2.1."""
        return _HC_EV_NM / self.wavelength_nm

    @property
    def uncertainty(self) -> Uncertainty:
        """The one-sigma uncertainty on ``A_ul``, from the grade."""
        return self.accuracy.as_uncertainty()

    @property
    def a_ul_uncertainty_per_s(self) -> float:
        """The same uncertainty in the units of ``A_ul`` itself."""
        return self.a_ul_per_s * self.accuracy.relative_uncertainty

    def __repr__(self) -> str:
        return (
            f"Transition({self.species!r}, {self.wavelength_nm} nm, "
            f"A_ul={self.a_ul_per_s:.3g} /s, {self.accuracy})"
        )


@dataclass(frozen=True, slots=True)
class LineSpec:
    """A line the specification asks for, before anything has been loaded.

    Attributes:
        species: ``"Ar I"`` or ``"Ar II"``.
        wavelength_nm: As doc 02 §6.3 tabulates it, to two decimals.
        paschen_level: The upper level in the Paschen notation doc 02 §6.3 uses, or
            ``None`` where it gives none. Documentation, not a matching key: ASD labels
            levels by configuration and term, and translating between the two notations
            is a lookup this layer has no business inventing.
        role: Why doc 02 §6.3 selected the line. Carried because it is what decides
            whether a missing line is survivable: losing an intermediate line costs
            precision, losing 811.53 costs the ``n_metastable`` handle entirely.
    """

    species: str
    wavelength_nm: float
    paschen_level: str | None
    role: str


#: The over-determined line set of doc 02 §6.3, in the order that document tabulates it.
#:
#: doc 02 §6.3 chose these for strong emission, minimal blending at 0.026 nm resolution,
#: and differing excitation-threshold sensitivity — "so that the ratios actually
#: constrain ``T_e``". The 750.39/811.53 pair is the classical discriminator: the first
#: is insensitive and the second highly sensitive to the metastable population, and using
#: both separates ``T_e`` from ``n_metastable`` in a way one ratio cannot.
DOC_02_LINE_SET: Final[tuple[LineSpec, ...]] = (
    LineSpec("Ar I", 750.39, "2p1", "direct excitation dominated"),
    LineSpec("Ar I", 751.47, "2p5", "direct excitation dominated"),
    LineSpec("Ar I", 811.53, "2p9", "metastable-coupled"),
    LineSpec("Ar I", 763.51, "2p6", "metastable-coupled"),
    LineSpec("Ar I", 696.54, "2p2", "intermediate"),
    LineSpec("Ar I", 706.72, "2p3", "intermediate"),
    LineSpec("Ar II", 480.60, None, "ion emission"),
    LineSpec("Ar II", 488.00, None, "ion emission"),
    LineSpec("Ar II", 434.81, None, "ion emission"),
    LineSpec("Ar II", 476.49, None, "ion emission"),
)


@dataclass(frozen=True, slots=True)
class LineList:
    """Every transition one ASD query returned.

    Attributes:
        transitions: The rows that carried a transition probability.
        rows_without_a_ul: How many rows did not. An ASD "Lines" query returns level
            pairs with no ``A_ki``; they are not transitions in the sense doc 04 §2.1
            needs, so they are dropped — but the count is kept, because a silently
            shortened table and a genuinely short one look identical.
    """

    transitions: tuple[Transition, ...]
    rows_without_a_ul: int

    def __post_init__(self) -> None:
        if self.rows_without_a_ul < 0:
            raise ValueError(f"rows_without_a_ul is a count, got {self.rows_without_a_ul}")

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions)

    def __getitem__(self, index: int) -> Transition:
        return self.transitions[index]

    def of_species(self, species: str) -> tuple[Transition, ...]:
        return tuple(line for line in self.transitions if line.species == species)

    def near(
        self,
        species: str,
        wavelength_nm: float,
        *,
        tolerance_nm: float = LINE_MATCH_TOLERANCE_NM,
    ) -> tuple[Transition, ...]:
        """Every transition of that species within ``tolerance_nm`` of that wavelength.

        The species is part of the query, not a filter applied afterwards. Ar I and Ar II
        lines answer different questions in doc 02 §6.3 — the neutral lines constrain
        ``T_e``, the ion lines ``n_i`` — so a wavelength match in the wrong ionisation
        stage is not a near miss, it is a different measurement.
        """
        return tuple(
            line
            for line in self.transitions
            if line.species == species and abs(line.wavelength_nm - wavelength_nm) <= tolerance_nm
        )

    def match(self, spec: LineSpec, *, tolerance_nm: float = LINE_MATCH_TOLERANCE_NM) -> Transition:
        """The one transition answering a specification line.

        Raises:
            LookupError: If nothing matches, or if more than one does. Two ASD rows
                inside the tolerance means the line is blended at this resolution, which
                is precisely what doc 02 §6.3 selected the set to avoid; picking the
                first would silently halve or double the modelled emissivity.
        """
        candidates = self.near(spec.species, spec.wavelength_nm, tolerance_nm=tolerance_nm)
        if not candidates:
            raise LookupError(
                f"no {spec.species} transition within {tolerance_nm} nm of "
                f"{spec.wavelength_nm} nm ({spec.role}). doc 02 §6.3 requires it."
            )
        if len(candidates) > 1:
            found = ", ".join(f"{line.wavelength_nm} nm" for line in candidates)
            raise LookupError(
                f"{spec.species} {spec.wavelength_nm} nm is ambiguous within "
                f"{tolerance_nm} nm: {found}"
            )
        return candidates[0]

    def missing(
        self,
        specs: Sequence[LineSpec] = DOC_02_LINE_SET,
        *,
        tolerance_nm: float = LINE_MATCH_TOLERANCE_NM,
    ) -> tuple[LineSpec, ...]:
        """Which specification lines this list cannot answer.

        Returned rather than raised, so that a report can name every gap at once. Doc 01
        OES-6 wants the set over-determined; knowing *which* lines are absent is what
        decides whether it still is.
        """
        return tuple(
            spec
            for spec in specs
            if not self.near(spec.species, spec.wavelength_nm, tolerance_nm=tolerance_nm)
        )

    def weights(self, *, unbounded_relative_uncertainty: float | None = None) -> FloatArray:
        """Inverse-variance weights from the accuracy grades — doc 09 §2.2.

        ``w_i ~ 1 / sigma_i^2``, normalised to sum to one. This is the weighting doc 09
        §2.2 calls "straightforward, correct, and almost never done", and it is why the
        grade is ingested rather than discarded at load.

        Args:
            unbounded_relative_uncertainty: What to use for a grade E line, whose NIST
                definition ("> 50 %") has no upper end. Required if any line carries one;
                there is no defensible default, and substituting 0.5 would hand the worst
                data in the set the same weight as a grade D line.

        Raises:
            ValueError: If the list is empty, or if a grade E line is present and no
                bound was supplied for it.
        """
        if not self.transitions:
            raise ValueError("an empty line list has no weights to distribute")

        relative: list[float] = []
        for line in self.transitions:
            if line.accuracy.is_bounded:
                relative.append(line.accuracy.relative_uncertainty)
                continue
            if unbounded_relative_uncertainty is None:
                raise ValueError(
                    f"{line.species} {line.wavelength_nm} nm carries grade E, which NIST "
                    "defines as '> 50 %' — a floor, not a value. Pass "
                    "unbounded_relative_uncertainty= to state what it is worth, or drop "
                    "the line."
                )
            relative.append(unbounded_relative_uncertainty)

        inverse_variance = 1.0 / np.square(np.asarray(relative, dtype=np.float64))
        return np.asarray(inverse_variance / inverse_variance.sum(), dtype=np.float64)

    def __repr__(self) -> str:
        return f"LineList({len(self)} transitions, {self.rows_without_a_ul} without A_ul)"


# ── parsing ─────────────────────────────────────────────────────────────────────

#: Which wavelength column to prefer, best first. Observed before Ritz because doc 02 §11
#: calibrates the wavelength axis against a NIST line list and the observed value is what
#: a spectrometer would see; air before vacuum for the same reason.
_WAVELENGTH_COLUMNS: Final[tuple[str, ...]] = (
    "obs_wl_air",
    "obs_wl_vac",
    "ritz_wl_air",
    "ritz_wl_vac",
)

#: ``(column, expected units)``. ``None`` means the column carries no units.
_REQUIRED_COLUMNS: Final[tuple[tuple[str, str | None], ...]] = (
    ("element", None),
    ("sp_num", None),
    ("Aki", "s^-1"),
    ("Acc", None),
    ("Ei", "eV"),
    ("Ek", "eV"),
    ("g_i", None),
    ("g_k", None),
)

#: Assembled, in order, into :attr:`Transition.upper_level`.
_UPPER_LEVEL_COLUMNS: Final[tuple[str, ...]] = ("conf_k", "term_k", "J_k")

_WAVELENGTH_UNITS: Final[str] = "nm"

#: Characters ASD uses to qualify a number: brackets for a predicted value, a question
#: mark for a questionable one. Stripped so the value is readable; the qualification is
#: not modelled, which is a known simplification rather than an oversight.
_QUALIFIERS: Final[str] = "[]()?"

_DELIMITERS: Final[tuple[str, ...]] = ("\t", "|")

#: Characters that make up a rule between ASD's header and its rows.
_RULE_CHARACTERS: Final[frozenset[str]] = frozenset("-+| ")


def _clean(cell: str) -> str:
    """Undo ASD's spreadsheet-safe quoting.

    The "tab-delimited with = prefix" export writes ``="750.3869"`` so that a spreadsheet
    does not reinterpret the cell. Left in place, every numeric column fails to parse and
    the whole table looks corrupt rather than quoted.
    """
    return cell.strip().lstrip("=").strip().strip('"').strip()


def _split_name(header_cell: str) -> tuple[str, str | None]:
    """``"Ei(eV)"`` into ``("Ei", "eV")``."""
    name, paren, rest = header_cell.partition("(")
    if not paren:
        return name.strip(), None
    return name.strip(), rest.rstrip(")").strip()


def _normalised_unit(unit: str) -> str:
    return unit.replace("^", "").replace("**", "").strip().lower()


def _is_rule(text: str) -> bool:
    return bool(text) and set(text) <= _RULE_CHARACTERS and "-" in text


def _resolve_columns(header: str, delimiter: str) -> tuple[Mapping[str, int], int]:
    """Locate every column the loader needs, checking the units it declares."""
    names = [_split_name(_clean(cell)) for cell in header.split(delimiter)]
    positions = {name: index for index, (name, _) in enumerate(names)}
    units = dict(names)

    columns: dict[str, int] = {}

    wavelength = next((name for name in _WAVELENGTH_COLUMNS if name in positions), None)
    if wavelength is None:
        raise NistParseError(
            f"table has no wavelength column; expected one of {', '.join(_WAVELENGTH_COLUMNS)}"
        )
    wavelength_unit = units[wavelength]
    if wavelength_unit is None or _normalised_unit(wavelength_unit) != _WAVELENGTH_UNITS:
        raise NistParseError(
            f"wavelength column {wavelength!r} is declared in {wavelength_unit!r}, "
            "expected nm. ASD exports Angstroms just as readily."
        )
    columns["wavelength"] = positions[wavelength]

    for name, expected in _REQUIRED_COLUMNS:
        if name not in positions:
            raise NistParseError(f"table has no {name!r} column")
        declared = units[name]
        if expected is not None and (
            declared is None or _normalised_unit(declared) != _normalised_unit(expected)
        ):
            raise NistParseError(f"column {name} is declared in {declared!r}, expected {expected}")
        columns[name] = positions[name]

    for name in _UPPER_LEVEL_COLUMNS:
        if name in positions:
            columns[name] = positions[name]

    return MappingProxyType(columns), max(columns.values())


def _number(cells: Sequence[str], index: int, *, name: str, line: int) -> float:
    text = _clean(cells[index]).strip(_QUALIFIERS)
    try:
        return float(text)
    except ValueError as exc:
        raise NistParseError(f"row {line}: {name} is not a number: {text!r}") from exc


def _grade(text: str, *, line: int) -> AccuracyGrade:
    if not text:
        raise NistParseError(
            f"row {line}: a transition probability with no accuracy grade. doc 09 §2.2 "
            "requires the grade to be ingested and propagated, and an ungraded A_ul is "
            "indistinguishable from a AAA one."
        )
    try:
        return AccuracyGrade(text)
    except ValueError as exc:
        raise NistParseError(
            f"row {line}: unrecognised accuracy grade {text!r}; expected one of "
            f"{', '.join(g.value for g in AccuracyGrade)}"
        ) from exc


def parse_nist_asd(text: str) -> LineList:
    """Parse a NIST ASD "Lines" export.

    Handles the tab- and pipe-delimited forms and the ``="..."`` spreadsheet-safe
    quoting, which are the shapes the ASD download page produces.

    Args:
        text: The export as downloaded.

    Returns:
        Every row that carried a transition probability, plus a count of those that did
        not.

    Raises:
        NistParseError: If the header is absent, a needed column is missing or declared
            in the wrong units, a row is short or unparseable, or a transition
            probability arrives without an accuracy grade.
    """
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, raw in enumerate(lines)
            if raw.strip() and not _is_rule(raw.strip()) and any(d in raw for d in _DELIMITERS)
        ),
        None,
    )
    if header_index is None:
        raise NistParseError(
            "no header row found. Expected a tab- or pipe-delimited ASD 'Lines' export."
        )

    header = lines[header_index]
    delimiter = next(d for d in _DELIMITERS if d in header)
    columns, widest = _resolve_columns(header, delimiter)

    transitions: list[Transition] = []
    without_a_ul = 0

    for offset, raw in enumerate(lines[header_index + 1 :], start=header_index + 2):
        stripped = raw.strip()
        if not stripped or _is_rule(stripped):
            continue

        cells = raw.split(delimiter)
        if len(cells) <= widest:
            raise NistParseError(
                f"row {offset}: {len(cells)} cells, expected at least {widest + 1}"
            )

        if not _clean(cells[columns["Aki"]]):
            # A level pair, not a line. Counted rather than silently forgotten.
            without_a_ul += 1
            continue

        transitions.append(_transition(cells, columns=columns, line=offset))

    if not transitions and without_a_ul == 0:
        raise NistParseError("the export has a header but no rows")

    return LineList(transitions=tuple(transitions), rows_without_a_ul=without_a_ul)


def _transition(cells: Sequence[str], *, columns: Mapping[str, int], line: int) -> Transition:
    """One row, with every failure naming the row it came from."""
    spectrum = _number(cells, columns["sp_num"], name="sp_num", line=line)
    label = " ".join(
        _clean(cells[columns[name]])
        for name in _UPPER_LEVEL_COLUMNS
        if name in columns and _clean(cells[columns[name]])
    )

    try:
        return Transition(
            element=_clean(cells[columns["element"]]),
            spectrum=int(spectrum),
            wavelength_nm=_number(cells, columns["wavelength"], name="wavelength", line=line),
            a_ul_per_s=_number(cells, columns["Aki"], name="Aki", line=line),
            accuracy=_grade(_clean(cells[columns["Acc"]]), line=line),
            e_lower_ev=_number(cells, columns["Ei"], name="Ei", line=line),
            e_upper_ev=_number(cells, columns["Ek"], name="Ek", line=line),
            g_lower=int(_number(cells, columns["g_i"], name="g_i", line=line)),
            g_upper=int(_number(cells, columns["g_k"], name="g_k", line=line)),
            upper_level=label,
        )
    except NistParseError:
        # Already names its row; re-wrapping would name it twice.
        raise
    except ValueError as exc:
        raise NistParseError(f"row {line}: {exc}") from exc
