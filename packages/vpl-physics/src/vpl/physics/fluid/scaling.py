"""The sheath scaling — what makes the L1 Newton system solvable at all.

In SI the three L1 unknowns span twenty orders of magnitude: ``n_i ~ 1e17 m^-3``,
``u_i ~ 1e4 m/s``, ``Phi ~ 1e2 V``, with a Poisson residual whose Laplacian coefficient is
``8.85e-12``. A Newton increment norm over that vector means nothing, and the iteration
converges the largest block while reporting success for all of them.

Referred to the sheath's own scales — the sheath-edge density, the Bohm speed, the
electron temperature as a potential, and the Debye length — every unknown becomes ``O(1)``
and the system reduces to

    ``(nu w)'         = sigma``
    ``(nu w^2)'       = -Theta nu phi' - gamma_i tau Theta nu' - nu w |w| / lambda_mfp``
    ``-lambda_D^2 phi'' = nu - exp(phi) - nu_se(phi)``

which is the classical collisionless fluid sheath (Lieberman & Lichtenberg §6.3) with the
doc 03 §3.1 collision and source terms retained.

## The one place doc 03 disagrees with itself, and how it is closed

ADR-007's second finding is that doc 03 §2.1 states ``c_s = sqrt(e(T_e + gamma_i T_i)/m_i)``
with ``gamma_i = 3`` while doc 03 §2.3's own arithmetic uses ``gamma_i = 0``, and that the
registry nominal is 0 because that is what reproduces every printed number. Doc 03 §3.1
then writes the ion pressure term as ``∂(n_i k T_i)/∂z`` — an *isothermal* closure, which
is ``gamma_i = 1``.

Left alone, that combination is not merely inconsistent, it is **ill-posed**. The critical
speed of the fluid system with an isothermal ion pressure is ``sqrt((T_e + T_i)/m_i)``,
which at RP-1 exceeds the ``gamma_i = 0`` Bohm speed by 0.8 %. Injecting ions at ``c_s``
at the bulk boundary — doc 03 §3.3, verbatim — then injects them *subsonically*, one
characteristic points back out of the domain, and the two inflow conditions doc 03 §3.3
specifies over-determine the problem.

:attr:`SheathScaling.pressure_coefficient` closes it by using the **same** ``gamma_i`` in
the pressure closure as in the Bohm speed, which is what an adiabatic index means. The
critical speed is then exactly ``c_s`` for every ``gamma_i``, the Bohm criterion becomes
the model's own sonic condition rather than an external assertion about it, and doc 03
§3.3's inflow condition is exactly marginal. See
:data:`~vpl.physics.fluid.sheath.DEFAULT_BOHM_MARGIN` for what is then done about
"exactly marginal".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from vpl.core.constants import ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.state import PlasmaParams
from vpl.core.units import Q_, Quantity, magnitude_in
from vpl.physics.analytic.sheath import bohm_speed

__all__ = ["SheathScaling"]

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_EPS0_F_PER_M: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))


@dataclass(frozen=True, slots=True)
class SheathScaling:
    """The reference values every L1 unknown is measured against.

    Attributes:
        density: ``n_s = h_l n_0``, the sheath-edge density. The Boltzmann relation is
            referenced here rather than to ``n_0``; see
            :func:`vpl.physics.analytic.sheath.AnalyticSheathSolver.solve` for why using
            ``n_0`` double-counts the presheath drop that ``h_l`` already encodes.
        speed: ``c_s``, the Bohm speed, at the caller's ``gamma_i``.
        potential: ``k T_e / e`` in volts.
        length: ``lambda_D`` at ``n_s`` — the controlling mesh scale of doc 03 §3.4.
        temperature_ratio: ``tau = T_i / T_e``.
        bohm_index: ``gamma_i``, the ion adiabatic index. See the module docstring.
        ion_mass: ``m_i`` in kg, needed for the doc 03 §6 energy flux.
    """

    density: float
    speed: float
    potential: float
    length: float
    temperature_ratio: float
    bohm_index: float
    ion_mass: float

    @classmethod
    def from_params(cls, params: PlasmaParams, *, h_l: float, gamma_i: float) -> SheathScaling:
        """Build the scaling for one operating point.

        Args:
            params: Control parameters.
            h_l: Planar edge-to-centre density ratio, a DESIGN parameter. See
                :data:`vpl.physics.analytic.sheath.DEFAULT_EDGE_TO_CENTRE_RATIO`.
            gamma_i: Ion adiabatic index, used in *both* the Bohm speed and the pressure
                closure. See the module docstring.
        """
        if not 0.0 < h_l <= 1.0:
            raise ValueError(f"h_l must lie in (0, 1], got {h_l}")
        if gamma_i < 0.0:
            raise ValueError(f"gamma_i cannot be negative, got {gamma_i}")

        density = h_l * params.n_0_per_m3
        potential = params.T_e_J / _E_C
        return cls(
            density=density,
            speed=float(magnitude_in(bohm_speed(params, gamma_i=gamma_i), "m/s")),
            potential=potential,
            length=float(np.sqrt(_EPS0_F_PER_M * potential / (density * _E_C))),
            temperature_ratio=params.T_i_J / params.T_e_J,
            bohm_index=gamma_i,
            ion_mass=params.species.mass_kg,
        )

    @property
    def sonic_factor(self) -> float:
        """``Theta = T_e / (T_e + gamma_i T_i) = e T_e / (m_i c_s^2)``.

        The coefficient the electric force carries once the momentum equation is measured
        in units of ``c_s^2``. One when ``gamma_i = 0``.
        """
        return 1.0 / (1.0 + self.bohm_index * self.temperature_ratio)

    @property
    def pressure_coefficient(self) -> float:
        """``gamma_i tau Theta`` — the ion pressure term in the scaled momentum equation.

        Zero for cold ions, which is the registry nominal. Together with
        :attr:`sonic_factor` it makes the critical speed exactly one; see the module
        docstring.
        """
        return self.bohm_index * self.temperature_ratio * self.sonic_factor

    @property
    def flux(self) -> float:
        """``n_s c_s`` in m^-2 s^-1 — the Bohm flux, and the unit of ion flux."""
        return self.density * self.speed

    @property
    def debye_length(self) -> Quantity:
        """``lambda_D`` at the sheath-edge density, as a dimensional quantity."""
        return Q_(self.length, "m")

    def normalised_bias(self, params: PlasmaParams) -> float:
        """``V_w / T_e``, positive — the sheath drop in units of electron temperature.

        83 at RP-1, which doc 03 §2.2 notes is far outside the matrix sheath's stated
        ``V << T_e`` validity and squarely inside Child-Langmuir's.
        """
        return float(magnitude_in(params.bias_magnitude, "V")) / self.potential

    def __repr__(self) -> str:
        return (
            f"SheathScaling(n_s={self.density:.4g} m^-3, c_s={self.speed:.4g} m/s, "
            f"T_e={self.potential:.4g} V, lambda_D={self.length:.4g} m)"
        )
