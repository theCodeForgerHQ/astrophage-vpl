"""The doc 03 §4.3 stability constraints, as runtime errors.

doc 03 §4.3 is unusually prescriptive about enforcement:

    These are hard constraints, checked at runtime, and violating them is a runtime error
    rather than a warning.

That wording is doing real work, so it is honoured literally. Every one of these constraints
has the same character: **violating it produces output.** A grid-heated run yields a density
profile, a sheath and an IEDF; they are simply wrong, by an amount nothing in the result
advertises. A warning in a log is not a control for that, because the ensemble runs of doc 11
WBS 3.1 are 5 000 solves and nobody reads 5 000 logs.

The four constraints and what each one's violation actually does:

- ``dz <= lambda_D`` — numerical grid heating. Energy flows from field to particles with
  no physics doing it, and the plasma warms until ``lambda_D`` grows to match the cell.
- ``dt <= 0.2 / omega_pe`` — the electron plasma oscillation is under-resolved and the
  leapfrog goes unstable. This one at least announces itself, eventually, as a blow-up.
- ``dt <= dz / v_max`` — a particle crosses more than one cell per step, so it deposits
  into cells whose field it never felt.
- ``N_ppc >= 100`` — statistical noise. doc 03 §4.3 defaults to 1 000 because the
  deliverable is the IEDF *tail*, which converges far more slowly than any moment.

Nothing here is a magic number: every threshold comes from the parameter registry or from
CODATA, per doc 08 §5.
"""

from __future__ import annotations

import math
from typing import Final

from vpl.core.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.params import default_registry
from vpl.core.units import magnitude_in
from vpl.physics.kinetic.grid import UniformGrid

__all__ = [
    "StabilityViolationError",
    "check_stability",
    "debye_length_m",
    "electron_plasma_frequency_rad_per_s",
]

_E: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E: Final[float] = float(magnitude_in(ELECTRON_MASS, "kg"))
_EPS0: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))
_K_B: Final[float] = float(magnitude_in(BOLTZMANN, "J/K"))

_REGISTRY = default_registry()

#: ``dt <= TIMESTEP_SAFETY / omega_pe``. doc 03 §4.3 gives 0.2; the registry owns it so the
#: sensitivity study of doc 06 can sweep it rather than a reader trusting it.
TIMESTEP_SAFETY: Final[float] = float(_REGISTRY.value_in("pic.timestep_safety", "dimensionless"))

#: doc 03 §4.3's statistical-noise floor. The registry default is 1 000, ten times this;
#: this is the point below which the run is refused outright rather than merely coarse.
MINIMUM_PARTICLES_PER_CELL: Final[float] = 100.0


class StabilityViolationError(RuntimeError):
    """A doc 03 §4.3 constraint was violated.

    A distinct exception type rather than ``ValueError`` because doc 03 §4.3 asks for a
    *runtime* error, and because an ensemble driver needs to tell "this configuration is
    unstable" apart from "this configuration is malformed" — the first is a point to skip
    and record, the second is a bug.
    """


def debye_length_m(*, density_m3: float, electron_temperature_ev: float) -> float:
    """``lambda_D = sqrt(eps0 T_e / (n e^2))`` with ``T_e`` in eV.

    At doc 01 §2.1's RP-1 — ``n = 1e17 m^-3``, ``T_e = 3 eV`` — this is 40.7 um, and
    doc 03 §4.3 takes ``dz = lambda_D / 2 = 20 um`` from it.
    """
    if density_m3 <= 0.0:
        raise ValueError(f"density must be positive, got {density_m3}")
    if electron_temperature_ev <= 0.0:
        raise ValueError(f"electron temperature must be positive, got {electron_temperature_ev}")
    return math.sqrt(_EPS0 * electron_temperature_ev / (density_m3 * _E))


def electron_plasma_frequency_rad_per_s(*, density_m3: float) -> float:
    """``omega_pe = sqrt(n e^2 / (eps0 m_e))``.

    At RP-1 this is 1.78e10 rad/s, which is where doc 03 §4.3's 11.2 ps timestep ceiling
    comes from. It is also the frequency the cold-plasma-oscillation verification test
    measures — the one closed form that pins the field solve and the push *together*.
    """
    if density_m3 <= 0.0:
        raise ValueError(f"density must be positive, got {density_m3}")
    return math.sqrt(density_m3 * _E**2 / (_EPS0 * _M_E))


def check_stability(
    grid: UniformGrid,
    *,
    dt_s: float,
    density_m3: float,
    electron_temperature_ev: float,
    n_ppc: float,
    v_max_m_per_s: float,
) -> None:
    """Check all four doc 03 §4.3 constraints, raising on the first violation.

    Raises:
        StabilityViolationError: If any constraint fails. The message names the offending value,
            the permitted bound and the consequence, because the reader of a failed ensemble
            point is usually not the person who configured it.
    """
    lambda_d = debye_length_m(
        density_m3=density_m3, electron_temperature_ev=electron_temperature_ev
    )
    if grid.dz_m > lambda_d:
        raise StabilityViolationError(
            f"cell width dz = {grid.dz_m:.3e} m exceeds the Debye length "
            f"lambda_D = {lambda_d:.3e} m (doc 03 §4.3). The run would suffer numerical "
            f"grid heating: energy enters the particles from the field with no physics "
            f"doing it. Use at least {math.ceil(grid.length_m / lambda_d)} cells."
        )

    omega_pe = electron_plasma_frequency_rad_per_s(density_m3=density_m3)
    dt_ceiling = TIMESTEP_SAFETY / omega_pe
    if dt_s > dt_ceiling:
        raise StabilityViolationError(
            f"timestep dt = {dt_s:.3e} s exceeds {TIMESTEP_SAFETY} / omega_pe = "
            f"{dt_ceiling:.3e} s (doc 03 §4.3), so the electron plasma frequency "
            f"omega_pe = {omega_pe:.3e} rad/s is under-resolved and the leapfrog is unstable."
        )

    cfl_ceiling = grid.dz_m / v_max_m_per_s
    if dt_s > cfl_ceiling:
        raise StabilityViolationError(
            f"timestep dt = {dt_s:.3e} s violates the CFL condition dz / v_max = "
            f"{cfl_ceiling:.3e} s (doc 03 §4.3). A particle at v_max = "
            f"{v_max_m_per_s:.3e} m/s would cross more than one cell per step and deposit "
            f"into cells whose field it never felt."
        )

    if n_ppc < MINIMUM_PARTICLES_PER_CELL:
        raise StabilityViolationError(
            f"{n_ppc:g} particles per cell is below the floor of "
            f"{MINIMUM_PARTICLES_PER_CELL:g} (doc 03 §4.3). Statistical noise scales as "
            f"N_ppc^(-1/2) and the deliverable is the IEDF tail, which converges far more "
            f"slowly than any moment."
        )
