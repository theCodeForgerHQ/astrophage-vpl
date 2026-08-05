"""Control parameters — the ``theta_c`` of doc 05 §2.1.

These are the physically meaningful, directly interpretable quantities an operator would
actually set. Nuisance parameters (doc 05 §2.2) and the discrepancy field (doc 05 §2.3)
belong to the inverse package, not here: the forward chain takes a physical state, and
"OES absolute radiometric scale" is not one.

## The sign convention, resolved

Doc 08 §6's manifest writes ``bias: {mode: dc, value: -250.0, units: V}`` while
doc 03 §2.2 uses ``V_w`` as a positive magnitude inside ``sqrt(2 V_w / T_e)``. Both
appear in the Baseline specification, which is exactly the silent sign ambiguity that
doc 02 §2 calls "common, silent and catastrophic".

Resolved here, once, in code:

- :attr:`PlasmaParams.bias` is the **signed applied electrode potential** relative to
  the plasma reference. It is negative for a biased electrode, matching the manifest.
- :attr:`PlasmaParams.bias_magnitude` is the **positive sheath potential drop** that
  every doc 03 formula wants.

Nothing else in the framework is permitted to take the absolute value of a bias inline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from vpl.core.constants import BOLTZMANN
from vpl.core.state.species import Species
from vpl.core.units import Q_, Quantity, magnitude_in

__all__ = ["PlasmaParams"]


def _as_kelvin(temperature: Quantity, *, name: str) -> Quantity:
    """Accept a temperature stated either as energy (eV) or as temperature (K).

    "T_e = 3 eV" and "T_e = 34813 K" are the same statement. Refusing one of them would
    push a k_B conversion into every caller, and that is where such conversions go wrong.
    """
    try:
        return temperature.to("K", "boltzmann")
    except Exception as exc:  # pragma: no cover - re-raised with context below
        raise ValueError(f"{name}: expected a temperature or an energy, got {temperature}") from exc


@dataclass(frozen=True, slots=True)
class PlasmaParams:
    """The eight control parameters of doc 05 §2.1, plus the species they act on.

    Validation here covers what **cannot exist** — a negative density, a zero pressure —
    and deliberately not what is merely unlikely. Doc 05 §2.1 gives ``kappa`` a uniform
    ``[1, 5]`` *prior*; a prior bound is a modelling choice, and enforcing it as a hard
    error would turn the sampler's own support into a crash.

    Attributes:
        species: The ion species. Its mass sets ``c_s`` and therefore the Bohm flux.
        n_0: Bulk plasma density.
        T_e: Electron temperature, as energy or as temperature.
        T_i: Ion temperature, as energy or as temperature.
        T_g: Neutral gas temperature.
        pressure: Neutral fill pressure.
        bias: Signed applied electrode potential. See the module docstring.
        gamma_se: Ion-induced secondary electron emission yield. doc 03 §3.3 is
            explicit that this is not optional: at 250 eV Ar+ on tungsten it is ~0.1,
            the secondaries re-cross the full sheath, and omitting them is a common and
            significant modelling error.
        kappa: EEDF shape parameter, Maxwellian through Druyvesteyn.
        rf_frequency: Bias drive frequency, or ``None`` for DC. doc 02 §3.3 spans DC,
            13.56 MHz and 60 MHz.
        rf_phase: Phase offset of the bias waveform, radians.
    """

    species: Species
    n_0: Quantity
    T_e: Quantity
    T_i: Quantity
    T_g: Quantity
    pressure: Quantity
    bias: Quantity
    gamma_se: float
    kappa: float
    rf_frequency: Quantity | None = None
    rf_phase: float = 0.0

    def __post_init__(self) -> None:
        if magnitude_in(self.n_0, "m**-3") <= 0.0:
            raise ValueError(f"n_0 must be positive, got {self.n_0}")
        if magnitude_in(_as_kelvin(self.T_e, name="T_e"), "K") <= 0.0:
            raise ValueError(f"T_e must be positive, got {self.T_e}")
        if magnitude_in(_as_kelvin(self.T_i, name="T_i"), "K") <= 0.0:
            raise ValueError(f"T_i must be positive, got {self.T_i}")
        if magnitude_in(_as_kelvin(self.T_g, name="T_g"), "K") <= 0.0:
            raise ValueError(f"T_g must be positive, got {self.T_g}")
        if magnitude_in(self.pressure, "Pa") <= 0.0:
            raise ValueError(f"pressure must be positive, got {self.pressure}")

        # Dimensionality assertion only: any potential is physically reachable.
        magnitude_in(self.bias, "V")

        if self.gamma_se < 0.0:
            raise ValueError(f"gamma_se is a yield and cannot be negative, got {self.gamma_se}")
        if self.rf_frequency is not None and magnitude_in(self.rf_frequency, "Hz") <= 0.0:
            raise ValueError(
                f"rf_frequency must be positive; use None for DC bias, got {self.rf_frequency}"
            )

    # ── derived quantities ──────────────────────────────────────────────────────

    @property
    def bias_magnitude(self) -> Quantity:
        """The positive sheath potential drop the doc 03 formulas take."""
        return Q_(abs(self.bias_volts), "V")

    @property
    def is_rf(self) -> bool:
        return self.rf_frequency is not None

    @property
    def n_g(self) -> Quantity:
        """Neutral density from the ideal gas law, ``p / (k_B T_g)`` — doc 01 §2.2."""
        return cast("Quantity", self.pressure / (BOLTZMANN * _as_kelvin(self.T_g, name="T_g")))

    @property
    def T_e_eV(self) -> Quantity:
        """Electron temperature expressed as an energy."""
        return self.T_e.to("eV", "boltzmann")

    @property
    def T_i_eV(self) -> Quantity:
        """Ion temperature expressed as an energy."""
        return self.T_i.to("eV", "boltzmann")

    # ── SI magnitudes for hot loops (doc 08 §5) ─────────────────────────────────

    @property
    def n_0_per_m3(self) -> float:
        return float(magnitude_in(self.n_0, "m**-3"))

    @property
    def T_e_J(self) -> float:
        return float(magnitude_in(self.T_e_eV, "J"))

    @property
    def T_i_J(self) -> float:
        return float(magnitude_in(self.T_i_eV, "J"))

    @property
    def bias_volts(self) -> float:
        """Signed applied potential in volts."""
        return float(magnitude_in(self.bias, "V"))

    @property
    def n_g_per_m3(self) -> float:
        return float(magnitude_in(self.n_g, "m**-3"))

    def replace(self, **changes: object) -> PlasmaParams:
        """Return a new parameter set with ``changes`` applied.

        Every validation re-runs, so a sweep cannot walk a parameter out of physical
        range one step at a time.
        """
        return replace(self, **changes)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        drive = f"{self.rf_frequency:.4g~P}" if self.rf_frequency is not None else "DC"
        return (
            f"PlasmaParams({self.species.name}, n_0={self.n_0:.3g~P}, "
            f"T_e={self.T_e_eV:.3g~P}, p={self.pressure.to('mTorr'):.3g~P}, "
            f"bias={self.bias:.4g~P}, drive={drive})"
        )
