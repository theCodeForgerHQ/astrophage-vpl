"""Published argon spectroscopic data — doc 04 §8 V-24, the one CR-model verification
item nothing in this project has ever satisfied.

Doc 04 §8's table gives V-24 as "CR model vs published Ar line ratios", pass criterion
"within published scatter". Every other CR-model check
(:mod:`vpl.instruments.oes.cr`'s own module docstring, and ``test_cr.py``'s Klein-Rosseland
identity and V-25/V-26 limits) is internal: an analytic identity the *model* was built to
reach. ``vpl-instruments/tests/oes_system.py`` says outright that its level system is
"argon-shaped, not argon", and the package README lists V-24 as the unverified line in its
table. This module is the external anchor, transcribed exactly as
:mod:`vpl.validation.swarm` transcribes the published swarm benchmarks: real values, a
full citation each, no bulk third-party data (doc 09 §5).

## Why this holds atomic data rather than a solver

This package's own ``pyproject.toml`` states the rule ``vpl.validation.swarm`` follows:
"depends on no solver, so that a solver cannot influence the thing that judges it." That
principle is honoured here too — this module imports nothing from ``vpl-physics`` or
``vpl-instruments``. It holds only data: NIST level energies and a published accuracy
figure, both plain facts about argon rather than anything computed by a solver. The
*comparison* against :class:`~vpl.instruments.oes.cr.CollisionalRadiativeModel` therefore
lives in ``test_line_ratios.py``, not here — the same relationship
``vpl-physics/tests/test_eedf_swarm_benchmarks.py`` has to ``vpl.validation.swarm``: the
package with the solver imports the pure data, not the reverse. (That test file is a
workspace member of the same ``uv`` monorepo, so the import resolves without this package
declaring a new dependency; whether it *should* declare one is a call for whoever owns
this package's ``pyproject.toml``, not made here.)

## What is transcribed, and what is not

1. **NIST ASD level energies and degeneracies** for the Ar I "1s" (Paschen, ``3p^5.4s``)
   and "2p" (Paschen, ``3p^5.4p``) manifolds that the three classic diagnostic lines
   (750.39, 763.51, 811.53 nm) connect. Fetched live from
   ``physics.nist.gov/cgi-bin/ASD/energy1.pl`` while this module was written. Real argon,
   for the first time anywhere a CR model is exercised in this project.
2. **Zheng et al (2020)**, arXiv:2010.10714 — the accuracy of a real, published,
   18-level Ar collisional-radiative model against an independent Langmuir probe. This is
   the closest genuinely obtainable stand-in for doc 04 §8's "published scatter": not a
   line-ratio value itself, but a real published error bound on a real argon CR
   technique's ``T_e``.

**Not transcribed: any Einstein ``A_ul`` coefficient**, and therefore no literal
line-intensity-ratio-vs-``T_e`` table. Computing either needs data this session could not
independently verify — see ``test_line_ratios.py``'s module docstring for exactly which
sources were tried and why each was unusable (two paywalled review papers, one paywalled
topical review, and a NIST ASD "Lines" endpoint that returned "Invalid Input" for every
query attempted). A remembered ``A_ul`` is not a substitute for a verified one, and doc 00
C4 asks for a gap to be stated rather than discovered later — this is that statement, and
``test_line_ratios.py::TestTheV24Comparison`` is where it becomes a test result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final

import yaml

from vpl.core.constants import PLANCK, SPEED_OF_LIGHT
from vpl.core.units import magnitude_in

__all__ = [
    "HC_EV_NM",
    "ArgonLevel",
    "SpectroscopicAccuracyBenchmark",
    "argon_level",
    "published_accuracy_benchmarks",
    "published_argon_levels",
    "transition_wavelength_nm",
]

#: Where the transcribed tables live — kept as data so a reviewer can diff it against the
#: cited NIST query and paper without reading Python, exactly as
#: ``vpl.validation.swarm``'s ``_DATA_FILE`` does.
_DATA_FILE: Final[Path] = Path(__file__).with_name("data") / "argon-line-ratios.yaml"

#: ``h c`` in eV nm, from CODATA via :mod:`vpl.core.constants` rather than typed in — doc
#: 09 §2.5 makes a hand-entered fundamental constant "a plausible, undetectable and
#: catastrophic error". Duplicated from (rather than imported from)
#: :mod:`vpl.physics.atomic.nist`'s identical ``_HC_EV_NM``: that module lives in
#: vpl-physics, and this package depends on no solver package — see the module
#: docstring. The duplication is one float, derived the same way in both places from the
#: same CODATA release, so it cannot drift into a different *value*, only a different
#: import path.
HC_EV_NM: Final[float] = float(magnitude_in(PLANCK * SPEED_OF_LIGHT, "eV * nm"))


@dataclass(frozen=True, slots=True)
class ArgonLevel:
    """One real Ar I atomic level, transcribed from a live NIST ASD query.

    Attributes:
        label: Paschen-notation key (``"1s5"``, ``"2p9"``), or ``"ground"``. Matches
            :class:`vpl.instruments.oes.levels.Level.label`'s free-text convention, so a
            level here can be handed straight to a real
            :class:`~vpl.instruments.oes.levels.LevelSystem`.
        term: NIST's configuration/term label, e.g. ``"3p5(2P3/2)4s 2[3/2]"``.
        j: Total angular momentum. Carried because ``degeneracy`` is checked against it
            (``g = 2J + 1``) rather than trusted outright — see
            ``test_line_ratios.py::test_every_degeneracy_is_two_j_plus_one``.
        energy_ev: Vacuum level energy above the ground state, as NIST ASD reports it.
        degeneracy: Statistical weight ``g``.
        role: What this level is for — ``"metastable"``, ``"resonant"``, ``"ground"``, or
            (for a 2p level) which diagnostic line it is the upper level of.
        citation: Key into the data file's ``citations`` block.
    """

    label: str
    term: str
    j: float
    energy_ev: float
    degeneracy: int
    role: str
    citation: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("a level needs a label")
        if self.energy_ev < 0.0:
            raise ValueError(f"{self.label}: energy cannot be negative, got {self.energy_ev} eV")
        if self.degeneracy < 1:
            raise ValueError(f"{self.label}: degeneracy is at least 1, got {self.degeneracy}")
        if self.j < 0.0:
            raise ValueError(f"{self.label}: J cannot be negative, got {self.j}")
        if not self.citation:
            raise ValueError(f"{self.label}: a published level needs a citation (doc 09 §1)")

    def __repr__(self) -> str:
        return f"ArgonLevel({self.label!r}, {self.energy_ev} eV, g={self.degeneracy})"


@dataclass(frozen=True, slots=True)
class SpectroscopicAccuracyBenchmark:
    """One published accuracy figure for a real argon CR/OES ``T_e``, ``n_e`` technique.

    Attributes:
        source: Short key for the paper.
        citation: The full reference.
        calibration: Which condition Zheng et al (2020) used as the method's calibration
            point — free text, since it names an RF frequency and power rather than a
            single number.
        gas_pressure_pa: The fixed discharge condition the row was measured at.
        max_n_e_error: Largest relative error between the CRM+OES-derived ``n_e`` and an
            independent Langmuir-probe measurement, over the power sweep.
        avg_n_e_error: The average of the same.
        max_t_e_error: As ``max_n_e_error``, for ``T_e``.
        avg_t_e_error: As ``avg_n_e_error``, for ``T_e``.
        figure: Which figure of the paper the row was read from.
    """

    source: str
    citation: str
    calibration: str
    gas_pressure_pa: float
    max_n_e_error: float
    avg_n_e_error: float
    max_t_e_error: float
    avg_t_e_error: float
    figure: str

    def __post_init__(self) -> None:
        if not self.citation:
            raise ValueError("a published benchmark needs a citation (doc 09 §1)")
        for name, value in (
            ("max_n_e_error", self.max_n_e_error),
            ("avg_n_e_error", self.avg_n_e_error),
            ("max_t_e_error", self.max_t_e_error),
            ("avg_t_e_error", self.avg_t_e_error),
        ):
            if value < 0.0:
                raise ValueError(f"{name} is a relative error and cannot be negative, got {value}")

    def __repr__(self) -> str:
        return f"SpectroscopicAccuracyBenchmark({self.calibration}, {self.source})"


def _read_data_file() -> dict[str, Any]:
    """Load the transcribed tables, UTF-8 explicitly — see ``vpl.validation.swarm``'s
    identical function for why the encoding is not decoration."""
    try:
        text = _DATA_FILE.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{_DATA_FILE} is not valid UTF-8: {error}") from error
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{_DATA_FILE} must contain a mapping, got {type(loaded).__name__}")
    return loaded


@cache
def published_argon_levels() -> tuple[ArgonLevel, ...]:
    """Every transcribed NIST ASD level, in file order (ascending energy)."""
    document = _read_data_file()
    citations = document["citations"]
    rows: Sequence[Mapping[str, Any]] = document["levels"]

    parsed: list[ArgonLevel] = []
    for row in rows:
        key = str(row["citation"])
        if key not in citations:
            raise KeyError(f"level {row['label']!r} cites {key!r}, which is not in 'citations'")
        parsed.append(
            ArgonLevel(
                label=str(row["label"]),
                term=str(row["term"]),
                j=float(row["j"]),
                energy_ev=float(row["energy_ev"]),
                degeneracy=int(row["degeneracy"]),
                role=str(row["role"]),
                citation=str(citations[key]).strip(),
            )
        )
    return tuple(parsed)


def argon_level(label: str) -> ArgonLevel:
    """The transcribed level named ``label``.

    Raises:
        KeyError: If no such level is transcribed. Returning ``None`` would let a
            benchmark silently compare against nothing — the same reasoning
            ``vpl.validation.swarm.model_gas`` gives for its own lookup.
    """
    for level in published_argon_levels():
        if level.label == label:
            return level
    known = ", ".join(level.label for level in published_argon_levels())
    raise KeyError(f"no transcribed argon level {label!r}; known levels are {known}")


def transition_wavelength_nm(upper: ArgonLevel, lower: ArgonLevel) -> float:
    """``lambda = h c / (E_upper - E_lower)``, from transcribed level energies alone.

    This is the one cross-check available on the transcription without an ``A_ul``: the
    result should land within a few tenths of a nanometre of a line's well-known
    position, the size of the air/vacuum offset a real ASD row carries
    (:mod:`vpl.physics.atomic.nist`'s module docstring documents the same offset for its
    own NIST rows). Agreement to many more digits than that would indicate the "known
    position" being checked against was derived from these same numbers rather than
    independently — this function does not embed one, precisely so that
    ``test_line_ratios.py`` can compare its output against an externally-known value
    without circularity.

    Raises:
        ValueError: If ``upper`` does not lie above ``lower`` — emission is only defined
            one way.
    """
    if upper.energy_ev <= lower.energy_ev:
        raise ValueError(
            f"{upper.label} does not lie above {lower.label} "
            f"({upper.energy_ev} eV vs {lower.energy_ev} eV); a transition wavelength "
            "needs the upper level strictly above the lower one"
        )
    return HC_EV_NM / (upper.energy_ev - lower.energy_ev)


@cache
def published_accuracy_benchmarks() -> tuple[SpectroscopicAccuracyBenchmark, ...]:
    """Every transcribed row of Zheng et al (2020) §4.1, in file order."""
    document = _read_data_file()
    citations = document["citations"]
    rows: Sequence[Mapping[str, Any]] = document["accuracy_benchmarks"]

    parsed: list[SpectroscopicAccuracyBenchmark] = []
    for row in rows:
        key = str(row["citation"])
        if key not in citations:
            raise KeyError(f"benchmark {row['calibration']!r} cites {key!r}, not in 'citations'")
        parsed.append(
            SpectroscopicAccuracyBenchmark(
                source=str(row["source"]),
                citation=str(citations[key]).strip(),
                calibration=str(row["calibration"]),
                gas_pressure_pa=float(row["gas_pressure_pa"]),
                max_n_e_error=float(row["max_n_e_error"]),
                avg_n_e_error=float(row["avg_n_e_error"]),
                max_t_e_error=float(row["max_t_e_error"]),
                avg_t_e_error=float(row["avg_t_e_error"]),
                figure=str(row["figure"]),
            )
        )
    return tuple(parsed)
