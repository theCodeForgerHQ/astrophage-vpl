"""Published electron-swarm benchmarks — the external half of doc 11 WBS 1.8.

WBS 1.8's acceptance criterion is that rate coefficients **reproduce published values**.
Everything else the EEDF solver is checked against is internal: analytic limits it was
built to reach (Maxwellian, Druyvesteyn, the Einstein relation) and the convergence of
its own discretisation. Those catch a solver that disagrees with itself. They cannot
catch one that is *self-consistently wrong*, and ADR-009 makes that risk concrete by
replacing a community-standard code with a solver written here.

This module holds the outside opinion. It lives in the package that depends on no
solver, for the reason :mod:`vpl.validation.convergence` states: nothing being judged
can influence the judgement.

## Why synthetic gases

Neither model is argon. That is the point. A model gas has no atomic-data uncertainty in
it, so every code in the literature solves an identical problem, and a disagreement is
attributable to the solver rather than to whichever LXCat set each author happened to
download. Doc 07 §1 draws exactly this line: verification asks whether we are solving the
equations right, and a benchmark contaminated by data uncertainty cannot answer it.

- **Reid ramp** (Reid 1979) — conservative: one constant elastic cross section and one
  linearly ramping excitation above 0.2 eV. It stresses the inelastic operator near
  threshold, where the ratio of inelastic to elastic collisions is large and the two-term
  approximation is known to strain.
- **Lucas-Saelee** (Lucas & Saelee 1975) — non-conservative: elastic plus *ionisation*.
  It is the only one of the two that exercises the ionisation term, and therefore the
  only one that can check a rate coefficient, which is what WBS 1.8 actually asks for.

## The two-term column is the one that matters

Our solver is two-term. Flynn et al report MultiBolt in both two-term, ``MB(2)``, and
ten-term, ``MB(10)``, modes, and both are carried here. Judging a two-term solver against
a converged multi-term column would report a known physics approximation as a solver
defect; judging it against ``MB(2)`` tests the implementation. The gap between the two
columns is then a *measurement* of the two-term error rather than an assumption that it
is small — the same discipline ADR-009 applies to the growth-model correction.

Reid's own Boltzmann column is likewise a conventional two-term solution, so the two
sources are independent two-term implementations 45 years apart. That they agree to four
digits is the strongest available evidence that the transcription in
``data/swarm-benchmarks.yaml`` is right, and it is asserted by a test.

## A limitation, stated rather than buried

The **Lucas-Saelee primary paper was not obtainable** — it is paywalled and no open copy
could be found. The model-gas definition used here is Flynn et al (2024) eq. (28), which
restates it. The Reid model, by contrast, is taken from the primary source, which is
freely available from CSIRO Publishing and was read directly. Doc 00 C4 asks for
simplifications to be stated where they are made rather than discovered later; this is
one, and it means an error in Flynn's restatement of LS would propagate here undetected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from numpy.typing import NDArray

from vpl.core.constants import ATOMIC_MASS, ELECTRON_MASS
from vpl.core.units import magnitude_in

__all__ = [
    "LUCAS_SAELEE",
    "REID_RAMP",
    "ModelChannel",
    "ModelGas",
    "SwarmBenchmark",
    "benchmarks_for",
    "model_gas",
    "published_swarm_benchmarks",
]

type FloatArray = NDArray[np.float64]
type CrossSection = Callable[[FloatArray], FloatArray]

#: Where the transcribed tables live. Kept as data so a reviewer can diff it against the
#: cited table without reading Python.
_DATA_FILE: Final[Path] = Path(__file__).with_name("data") / "swarm-benchmarks.yaml"

#: 1 Angstrom^2 in m^2. Both papers quote cross sections in units of 10^-20 m^2.
_ANGSTROM_SQUARED: Final[float] = 1.0e-20


def _mass_ratio_from_amu(mass_amu: float) -> float:
    """``m_e / M`` from a molecular weight, through the same CODATA the solver uses.

    Reid states a molecular weight, not a mass ratio. Deriving the ratio here rather
    than writing a constant keeps it tied to :mod:`vpl.core.constants`; a hand-computed
    number would drift from the physics it is being compared against.
    """
    electron_kg = float(magnitude_in(ELECTRON_MASS, "kg"))
    amu_kg = float(magnitude_in(ATOMIC_MASS, "kg"))
    return electron_kg / (mass_amu * amu_kg)


@dataclass(frozen=True, slots=True)
class ModelChannel:
    """One inelastic process of a model gas, as an exact function of energy.

    Attributes:
        reaction: The channel's identity, in the form the solver keys rate coefficients
            by.
        is_ionisation: Whether the channel creates an electron. This is what separates a
            conservative model gas from a non-conservative one, and it changes which
            operator the solver exercises.
        threshold_ev: The energy lost in one such collision.
        sigma_m2: The cross section, evaluated functionally. See :class:`ModelGas` for
            why it is not a table.
    """

    reaction: str
    is_ionisation: bool
    threshold_ev: float
    sigma_m2: CrossSection


@dataclass(frozen=True, slots=True)
class ModelGas:
    """A synthetic gas whose cross sections are exact formulae.

    The cross sections are callables rather than tabulated arrays deliberately. Flynn et
    al evaluate them "functionally at all electron energies with no interpolation error",
    and sampling a table here would put an interpolation error *inside* the reference the
    solver is judged against — the benchmark would then measure the interpolator.

    Attributes:
        key: Short identifier, matching the ``gas`` field of the published values.
        name: Human-readable name for reports.
        citation: The paper the model is defined in.
        mass_ratio: ``m_e / M``.
        gas_temperature_ev: Both models fix the gas at 0 K, which removes superelastic
            collisions from the comparison entirely.
        elastic_mtcs_m2: The elastic momentum-transfer cross section.
        channels: The inelastic processes.
    """

    key: str
    name: str
    citation: str
    mass_ratio: float
    gas_temperature_ev: float
    elastic_mtcs_m2: CrossSection
    channels: tuple[ModelChannel, ...]


def _reid_elastic(energy_ev: FloatArray) -> FloatArray:
    """Reid §4a(i): ``sigma_e = sigma_m,e = 6.0 A^2``, independent of energy."""
    return np.full_like(np.asarray(energy_ev, dtype=np.float64), 6.0 * _ANGSTROM_SQUARED)


def _reid_ramp(energy_ev: FloatArray) -> FloatArray:
    """Reid eq. (23): zero below 0.2 eV, then ``k (eps - eps_i)`` with ``k = 10 A^2/eV``.

    The clip at zero is the model's definition, not a numerical guard — the threshold is
    the entire physical content of the gas.
    """
    energy = np.asarray(energy_ev, dtype=np.float64)
    return 10.0 * _ANGSTROM_SQUARED * np.clip(energy - 0.2, 0.0, None)


def _lucas_saelee_elastic(energy_ev: FloatArray) -> FloatArray:
    """Flynn eq. (28a): ``sigma_m = 4 eps^-1/2 A^2``, which diverges as ``eps -> 0``."""
    energy = np.asarray(energy_ev, dtype=np.float64)
    return 4.0 * _ANGSTROM_SQUARED / np.sqrt(energy)


def _lucas_saelee_ionisation(energy_ev: FloatArray) -> FloatArray:
    """Flynn eq. (28b): ``sigma_iz = 0.1 (eps - 15.6) A^2`` above threshold."""
    energy = np.asarray(energy_ev, dtype=np.float64)
    return 0.1 * _ANGSTROM_SQUARED * np.clip(energy - 15.6, 0.0, None)


REID_RAMP: Final[ModelGas] = ModelGas(
    key="reid-ramp",
    name="Reid ramp model gas",
    citation=(
        "I D Reid, Aust. J. Phys. 32 (1979) 231, section 4a(ii), k = 10 A^2 eV^-1; "
        "restated as Flynn et al (2024) eq. (27)"
    ),
    mass_ratio=_mass_ratio_from_amu(4.0),
    gas_temperature_ev=0.0,
    elastic_mtcs_m2=_reid_elastic,
    channels=(
        ModelChannel(
            reaction="e + M -> e + M*",
            is_ionisation=False,
            threshold_ev=0.2,
            sigma_m2=_reid_ramp,
        ),
    ),
)
"""Conservative model gas: constant elastic, linearly ramping excitation above 0.2 eV."""


LUCAS_SAELEE: Final[ModelGas] = ModelGas(
    key="lucas-saelee",
    name="Lucas-Saelee ionisation model gas",
    citation=(
        "J Lucas and H T Saelee, J. Phys. D 8 (1975) 640, as restated by "
        "Flynn et al (2024) eq. (28) with F = 1; primary source not consulted"
    ),
    # Stated outright as 1e-3 by Flynn eq. (28). LS is not a real gas and has no
    # molecular weight to derive a ratio from; deriving one would invent a number the
    # model does not have.
    mass_ratio=1.0e-3,
    gas_temperature_ev=0.0,
    elastic_mtcs_m2=_lucas_saelee_elastic,
    channels=(
        ModelChannel(
            reaction="e + M -> 2e + M+",
            is_ionisation=True,
            threshold_ev=15.6,
            sigma_m2=_lucas_saelee_ionisation,
        ),
    ),
)
"""Non-conservative model gas: the only one here that exercises the ionisation term."""


_MODEL_GASES: Final[Mapping[str, ModelGas]] = {
    REID_RAMP.key: REID_RAMP,
    LUCAS_SAELEE.key: LUCAS_SAELEE,
}


def model_gas(key: str) -> ModelGas:
    """The model gas registered under ``key``.

    Raises:
        KeyError: If no such gas is defined. Returning ``None`` would let a benchmark
            test skip silently, which is the one failure mode a verification harness
            must not have.
    """
    try:
        return _MODEL_GASES[key]
    except KeyError:
        known = ", ".join(sorted(_MODEL_GASES))
        raise KeyError(f"no model gas {key!r}; known gases are {known}") from None


@dataclass(frozen=True, slots=True)
class SwarmBenchmark:
    """One published row: a gas, a reduced field, and the transport coefficients.

    Attributes:
        gas: Which model gas, matching :attr:`ModelGas.key`.
        source: Short key for the paper the *values* came from, distinct from the paper
            the *model* is defined in.
        citation: The full reference.
        terms: How many terms in the spherical-harmonic expansion produced the value.
            Load-bearing: our solver is two-term, and comparing it against a converged
            ten-term column would report a physics approximation as a defect.
        reduced_field_td: ``E/N`` in townsend.
        mean_energy_ev: ``<eps>``.
        drift_velocity_m_per_s: The flux drift velocity ``W_F``.
        reduced_transverse_diffusion: ``N D_T`` in ``1 / (m s)``.
        ionisation_rate_m3_per_s: ``k_iz``, where the gas has an ionisation channel.
            ``None`` for a conservative gas, which is a statement about the model rather
            than a missing measurement.
    """

    gas: str
    source: str
    citation: str
    terms: int
    reduced_field_td: float
    mean_energy_ev: float
    drift_velocity_m_per_s: float
    reduced_transverse_diffusion: float
    ionisation_rate_m3_per_s: float | None = None

    def __repr__(self) -> str:
        return (
            f"SwarmBenchmark({self.gas}, {self.reduced_field_td:g} Td, "
            f"{self.terms}-term, {self.source})"
        )


def _read_data_file() -> dict[str, Any]:
    """Load the transcribed tables, UTF-8 explicitly.

    The explicit encoding is not decoration. The reference machine of doc 10 §1 runs
    with a ``LANG=C`` locale, where Python's default would turn the em dashes in these
    citations into a crash — a portability defect that only a run there exposed.
    """
    try:
        text = _DATA_FILE.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{_DATA_FILE} is not valid UTF-8: {error}") from error
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{_DATA_FILE} must contain a mapping, got {type(loaded).__name__}")
    return loaded


@cache
def published_swarm_benchmarks() -> tuple[SwarmBenchmark, ...]:
    """Every transcribed row, in file order.

    Numbers are coerced with :func:`float` rather than trusted. ADR-008 records why: YAML
    1.1's float resolver requires a signed exponent, so an unsigned ``1.0e17`` loads as a
    *string*. The data file writes signed exponents throughout, and this coercion is the
    belt to that braces — a benchmark silently comparing against a string would fail in a
    way that looks like a solver bug.
    """
    document = _read_data_file()
    citations = document["citations"]
    rows: Sequence[Mapping[str, Any]] = document["benchmarks"]

    parsed: list[SwarmBenchmark] = []
    for row in rows:
        rate = row.get("ionisation_rate_m3_per_s")
        parsed.append(
            SwarmBenchmark(
                gas=str(row["gas"]),
                source=str(row["source"]),
                citation=str(citations[row["citation"]]).strip(),
                terms=int(row["terms"]),
                reduced_field_td=float(row["reduced_field_td"]),
                mean_energy_ev=float(row["mean_energy_ev"]),
                drift_velocity_m_per_s=float(row["drift_velocity_m_per_s"]),
                reduced_transverse_diffusion=float(row["reduced_transverse_diffusion"]),
                ionisation_rate_m3_per_s=None if rate is None else float(rate),
            )
        )
    return tuple(parsed)


def benchmarks_for(
    gas: str, *, terms: int | None = None, source: str | None = None
) -> tuple[SwarmBenchmark, ...]:
    """The published rows for one gas, optionally narrowed to a term count or source.

    Raises:
        KeyError: If ``gas`` names no registered model gas. An empty tuple would make a
            benchmark test vacuously pass.
    """
    model_gas(gas)
    selected = [
        point
        for point in published_swarm_benchmarks()
        if point.gas == gas
        and (terms is None or point.terms == terms)
        and (source is None or point.source == source)
    ]
    return tuple(selected)
