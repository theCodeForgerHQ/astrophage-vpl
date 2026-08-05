"""``manifest.plasma`` → ``PlasmaParams`` — doc 08 §6, doc 05 §2.1, doc 08 §5.

doc 08 §6's plasma block writes five keys: ``gas``, ``pressure``, ``n0``, ``Te`` and
``bias``. :class:`~vpl.core.state.PlasmaParams` needs eleven. The other six are not
invented here — doc 08 §5 makes the parameter registry "the sole source of numeric
defaults", so each one resolves from a *named registry entry* and the manifest never
restates a registered value.

## Why every field records its source

doc 00 C4 forbids hidden assumptions. "T_i defaulted" and "T_i = 0.05 eV from
``RP1.T_i``, class DESIGN, swept over [0.02, 0.5]" are very different statements, and only
the second survives someone asking six months later why the ion temperature was what it
was. :class:`ResolvedPlasma` therefore carries a source string per field, and the run
record writes it out (doc 13 §2).

The consequence worth stating: **the registry defaults are RP-1's**, and a manifest that
is not at the reference operating point and cares about ``T_i`` or ``T_g`` has to say so.
That is visible rather than silent precisely because the source is recorded — a run whose
record says ``registry:RP1.T_g`` at 50 mTorr is one a reader can question.

## ``kappa``, and why it is the one convention rather than a registry lookup

doc 05 §2.1 gives the EEDF shape parameter a uniform ``[1, 5]`` *prior*. A prior is not a
nominal, so there is nothing in the registry to read, and inventing an entry would assert
a sourced value where doc 05 §2.1 deliberately declined to state one. The default is
therefore a stated modelling convention — the Maxwellian limit — named as such, so the
run record distinguishes it from the registry lookups beside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from vpl.core.params import ParameterRegistry, default_registry
from vpl.core.state import PlasmaParams, Species
from vpl.core.units import Quantity
from vpl.experiment.manifest.schema import BiasMode, PlasmaSpec

__all__ = ["MAXWELLIAN_KAPPA", "ResolvedPlasma", "resolve_plasma"]

#: The EEDF shape parameter at the Maxwellian limit — doc 05 §2.1's ``kappa = 1``.
#:
#: Named rather than inlined because doc 08 §5's rule does not carve out an exception for
#: a number whose author was confident it did not matter, and because a reader of the run
#: record has to be able to find where ``convention:maxwellian-eedf`` is defined.
MAXWELLIAN_KAPPA: Final[float] = 1.0

#: Bias phase when the manifest does not set one. Zero *is* the definition of the phase
#: origin rather than a measured value, which is why its source is recorded as a
#: convention and not as a registry lookup.
_PHASE_ORIGIN_RAD: Final[float] = 0.0

#: How the manifest's ``gas`` names a species, and where its mass comes from.
#:
#: Categorical, so it is code and not a registry entry — :mod:`vpl.core.params` is
#: explicit that the registry holds numbers and that "argon" has no uncertainty, no sweep
#: range and no units. The *mass* is a number and does come from the registry.
_GAS_SPECIES: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "argon": ("Ar+", "species.Ar.mass"),
        "xenon": ("Xe+", "species.Xe.mass"),
    }
)

#: Charge number of a singly-ionised species. An integer count, not a magnitude.
_SINGLY_IONISED: Final[int] = 1

#: Registry entry each unstated control parameter resolves from.
_REGISTRY_DEFAULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "T_i": "RP1.T_i",
        "T_g": "RP1.T_g",
        # doc 03 §3.3: not optional. The entry is tungsten-specific because that is the
        # electrode doc 02 §3 specifies; a manifest for another wall material states its
        # own yield until the registry grows an entry per material.
        "gamma_se": "sheath.gamma_se_W",
        "rf_frequency": "RP1.rf_frequency",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedPlasma:
    """Control parameters, beside where each of them came from.

    Attributes:
        params: The doc 05 §2.1 control parameters, ready for a solver.
        sources: One entry per field of :class:`~vpl.core.state.PlasmaParams`, naming
            ``"manifest"``, ``"registry:<id>"`` or ``"convention:<name>"``. Written into
            the run record (doc 13 §2) so that no default is invisible (doc 00 C4).
    """

    params: PlasmaParams
    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))


def _species(gas: str, registry: ParameterRegistry) -> tuple[Species, str]:
    """The ion species a gas name denotes, and the registry entry its mass came from."""
    entry = _GAS_SPECIES.get(gas.strip().lower())
    if entry is None:
        known = ", ".join(sorted(_GAS_SPECIES))
        raise ValueError(
            f"plasma.gas: no species is registered for {gas!r}. Known gases: {known}. "
            "A gas becomes available by adding its mass to the parameter registry "
            "(doc 08 §5) and naming it here — never by a mass literal in a manifest."
        )

    name, mass_id = entry
    return (
        Species(name=name, mass=registry[mass_id].quantity, charge_number=_SINGLY_IONISED),
        f"registry:{mass_id}",
    )


def _resolved(
    stated: Quantity | None, *, field: str, registry: ParameterRegistry
) -> tuple[Quantity, str]:
    """A quantity the manifest may have stated, else the registered one."""
    if stated is not None:
        return stated, "manifest"

    entry_id = _REGISTRY_DEFAULTS[field]
    return registry[entry_id].quantity, f"registry:{entry_id}"


def resolve_plasma(
    plasma: PlasmaSpec, *, registry: ParameterRegistry | None = None
) -> ResolvedPlasma:
    """Build the doc 05 §2.1 control parameters from a manifest's plasma block.

    Args:
        plasma: The parsed ``plasma:`` block.
        registry: The parameter registry to read defaults from. Defaults to the one that
            ships with :mod:`vpl.core.params`; taken as an argument so that a sensitivity
            study can resolve the same manifest against a perturbed registry without
            editing the manifest.

    Returns:
        The parameters and, for every one of them, where it came from.

    Raises:
        ValueError: If the gas is not one the registry has a mass for, or if a value is
            outside what :class:`~vpl.core.state.PlasmaParams` accepts as physical.
    """
    catalogue = default_registry() if registry is None else registry
    sources: dict[str, str] = {}

    species, sources["species"] = _species(plasma.gas, catalogue)

    T_i, sources["T_i"] = _resolved(plasma.T_i, field="T_i", registry=catalogue)
    T_g, sources["T_g"] = _resolved(plasma.T_g, field="T_g", registry=catalogue)

    if plasma.gamma_se is None:
        entry_id = _REGISTRY_DEFAULTS["gamma_se"]
        gamma_se = float(catalogue[entry_id].value)
        sources["gamma_se"] = f"registry:{entry_id}"
    else:
        gamma_se = plasma.gamma_se
        sources["gamma_se"] = "manifest"

    if plasma.kappa is None:
        kappa = MAXWELLIAN_KAPPA
        sources["kappa"] = "convention:maxwellian-eedf"
    else:
        kappa = plasma.kappa
        sources["kappa"] = "manifest"

    rf_frequency: Quantity | None = None
    if plasma.bias.mode is BiasMode.RF:
        rf_frequency, sources["rf_frequency"] = _resolved(
            plasma.bias.frequency, field="rf_frequency", registry=catalogue
        )
    else:
        # Absent means DC (doc 02 §3.3). Recorded as a manifest statement, because the
        # manifest did state it: `mode: dc` is a choice, not an omission.
        sources["rf_frequency"] = "manifest"

    if plasma.bias.phase is None:
        rf_phase = _PHASE_ORIGIN_RAD
        sources["rf_phase"] = "convention:phase-origin"
    else:
        rf_phase = plasma.bias.phase
        sources["rf_phase"] = "manifest"

    # The four the manifest is required to state. Listed rather than inferred, so that a
    # field added to PlasmaParams without a source here fails the run record's own check.
    for stated_field in ("n_0", "T_e", "pressure", "bias"):
        sources[stated_field] = "manifest"

    params = PlasmaParams(
        species=species,
        n_0=plasma.n_0,
        T_e=plasma.T_e,
        T_i=T_i,
        T_g=T_g,
        pressure=plasma.pressure,
        bias=plasma.bias.value,
        gamma_se=gamma_se,
        kappa=kappa,
        rf_frequency=rf_frequency,
        rf_phase=rf_phase,
    )
    return ResolvedPlasma(params=params, sources=sources)
