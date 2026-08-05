"""The coupled drift-diffusion / Boltzmann / Poisson system — doc 03 §1's L1.

    ``d/dz (u n_i - D dn_i/dz)  =  S_iz - L_rec``
    ``n_e  =  n_s exp(e Phi / k T_e)``
    ``-eps0 d2Phi/dz2  =  e (n_i - n_e)``

Solved as one Newton system, which doc 03 §3.4 requires by name:

    Coupling | Newton on the fully coupled ``(n_i, u_i, Phi)`` system | The ``n_e(Phi)``
    exponential makes Gummel iteration converge poorly at high bias

This is the system doc 03 §7 V-02 gates. The inertial three-field sheath, which needs the
ion momentum equation and is what doc 03 §7 V-03 gates, is
:mod:`vpl.physics.fluid.sheath`; it shares this module's mesh, stabilisation and Newton
machinery but not its transport closure.

## Scaling, and why it is not optional

The three unknowns of a sheath span twenty orders of magnitude in SI: ``n_i ~ 1e17``,
``Phi ~ 1e2``, and a Poisson residual whose Laplacian term is ``1e-11``. A Newton
iteration whose increment norm is dominated by one block converges that block and reports
success while another is still wrong. Both unknowns are therefore solved in units of
their own scale — density in units of the Boltzmann reference density, potential in units
of the electron temperature — so that "the increment is small" means the same thing for
both. The Poisson operator then carries a single dimensional coefficient, the Debye
length squared, which is the correct statement of what makes a sheath thin.

## The Boltzmann assumption travels with an A5 warning

doc 03 §8 A5 registers ``n_e = n_0 exp(e Phi / k T_e)`` as an assumption that breaks under
"strong EEDF depletion", and doc 03 §3.2 is blunt that in a low-pressure sheath the EEDF
"is typically bi-Maxwellian or Druyvesteyn, and the high-energy tail [...] is depleted".
L1 makes the assumption; L2 does not; V-10 measures the difference. Nothing here is
entitled to forget that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem

from vpl.core.constants import ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from vpl.core.state import SpatialGrid
from vpl.core.units import magnitude_in
from vpl.physics.fluid.drift_diffusion import Stabilisation
from vpl.physics.fluid.forms import (
    DEFAULT_NEWTON_RTOL,
    DEFAULT_SOURCE_DEGREE_OFFSET,
    FieldSolution,
    NewtonReport,
    SourceTerm,
    dirichlet_on_component,
    interpolate_callable,
    run_newton,
    sample,
    ufl_fitted_diffusivity,
)
from vpl.physics.fluid.mesh import interval_mesh

__all__ = ["CoupledSolution", "solve_drift_diffusion_poisson"]

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_EPS0_F_PER_M: Final[float] = float(magnitude_in(VACUUM_PERMITTIVITY, "F/m"))

#: Default Newton iteration cap. Generous: doc 03 §3.4's claim is that Newton converges
#: where Gummel does not, and a cap tight enough to be the binding constraint would turn
#: a hard problem into a failed one without saying which.
DEFAULT_MAX_ITERATIONS: Final[int] = 50


@dataclass(frozen=True, slots=True)
class CoupledSolution:
    """What one coupled solve produced.

    Attributes:
        density: ``n_i`` in m^-3.
        potential: ``Phi`` in volts.
        electron_density: ``n_e`` in m^-3, from the Boltzmann relation. Evaluated *from*
            the potential rather than interpolated from nodal electron densities: the
            relation is algebraic and exact, so ``n_e`` at a cell midpoint is the
            exponential of the potential there and not the average of its neighbours. The
            two differ at ``O(h^2)``, and the exact one is the one that belongs in the
            space-charge term.
        newton: How the nonlinear solve went.
    """

    density: FieldSolution
    potential: FieldSolution
    electron_density: FieldSolution
    newton: NewtonReport


def solve_drift_diffusion_poisson(
    *,
    grid: SpatialGrid,
    drift: float,
    diffusivity: float,
    electron_temperature_volts: float,
    reference_density: float,
    ion_source: SourceTerm,
    charge_density_source: SourceTerm,
    density_boundary: tuple[float, float],
    potential_boundary: tuple[float, float],
    electron_coupling_rate: float = 0.0,
    electron_coupling_reference: SourceTerm | None = None,
    stabilisation: Stabilisation = Stabilisation.EXPONENTIAL_FITTING,
    source_degree_offset: int = DEFAULT_SOURCE_DEGREE_OFFSET,
    rtol: float = DEFAULT_NEWTON_RTOL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> CoupledSolution:
    """Newton on the coupled ``(n_i, Phi)`` system, P1 on both.

    Args:
        grid: Where to solve. The wall is at ``z = 0`` (doc 02 §2).
        drift: Ion advection velocity ``u`` in m/s, constant.
        diffusivity: Ion diffusivity ``D`` in m^2/s, positive.
        electron_temperature_volts: ``k T_e / e``, i.e. ``T_e`` in eV read as a potential.
        reference_density: ``n_s``, the density at which ``Phi = 0`` in the Boltzmann
            relation. doc 03 §3.1 writes ``n_0``; :mod:`vpl.physics.analytic.sheath`
            explains at length why the sheath-edge density is the self-consistent choice
            and why using ``n_0`` double-counts the presheath drop that ``h_l`` encodes.
        ion_source: ``S_iz - L_rec`` in m^-3 s^-1.
        charge_density_source: Any charge density beyond ``e (n_i - n_e)``, in C/m^3.
            Zero for a physical sheath; the method of manufactured solutions puts the
            term that makes its chosen pair exact here.
        density_boundary: ``(n_i(0), n_i(L))`` in m^-3.
        potential_boundary: ``(Phi(0), Phi(L))`` in volts.
        electron_coupling_rate: Rate in s^-1 of a loss term proportional to the electron
            density — the electron-dependent half of doc 03 §3.1's ``L_rec``. Zero by
            default. Non-zero, it makes the Jacobian genuinely two-way coupled, which is
            what a manufactured solution needs in order to verify the off-diagonal block
            rather than a block-triangular system that Gummel iteration would also solve.
        electron_coupling_reference: The electron density the loss term relaxes toward,
            in m^-3. Required when ``electron_coupling_rate`` is non-zero.
        stabilisation: See :class:`~vpl.physics.fluid.drift_diffusion.Stabilisation`.
        source_degree_offset: How far above P1 the sources are represented.
        rtol: Relative tolerance on the Newton increment.
        max_iterations: Newton iteration cap.

    Returns:
        The two fields, the Boltzmann electron density, and the Newton report.

    Raises:
        ValueError: If ``diffusivity`` or ``reference_density`` is not positive, if
            ``electron_temperature_volts`` is not positive, or if a coupling rate was
            given without a reference.
        RuntimeError: If Newton does not converge.
    """
    _validate(
        diffusivity=diffusivity,
        reference_density=reference_density,
        electron_temperature_volts=electron_temperature_volts,
        electron_coupling_rate=electron_coupling_rate,
        electron_coupling_reference=electron_coupling_reference,
    )

    domain = interval_mesh(grid)
    element = basix.ufl.element("Lagrange", "interval", 1)
    space = fem.functionspace(domain, basix.ufl.mixed_element([element, element]))

    unknown = fem.Function(space)
    scaled_density, scaled_potential = ufl.split(unknown)
    density_test, potential_test = ufl.TestFunctions(space)

    source_degree = 1 + source_degree_offset
    length = float(grid.z_m[-1])

    def _scaled(term: SourceTerm, factor: float) -> fem.Function:
        return interpolate_callable(
            domain, lambda z: np.asarray(term(z), dtype=np.float64) * factor, degree=source_degree
        )

    ion = _scaled(ion_source, 1.0 / reference_density)
    charge = _scaled(charge_density_source, 1.0 / (_E_C * reference_density))
    electrons = ufl.exp(scaled_potential)

    total_diffusivity = (
        ufl.as_ufl(diffusivity)
        if stabilisation is Stabilisation.NONE
        else ufl_fitted_diffusivity(
            velocity=ufl.as_ufl(drift),
            diffusivity=diffusivity,
            cell_size=ufl.CellDiameter(domain),
        )
    )

    continuity = (
        total_diffusivity * ufl.grad(scaled_density)[0] * ufl.grad(density_test)[0]
        - drift * scaled_density * ufl.grad(density_test)[0]
        - ion * density_test
    ) * ufl.dx

    if electron_coupling_reference is not None and electron_coupling_rate != 0.0:
        reference = _scaled(electron_coupling_reference, 1.0 / reference_density)
        continuity += electron_coupling_rate * (electrons - reference) * density_test * ufl.dx

    # The residual blocks are rescaled onto a common size so that "the Newton increment
    # is small" means the same thing in both. See the module docstring.
    continuity_scale = length / max(abs(drift), diffusivity / length)

    debye_squared = _EPS0_F_PER_M * electron_temperature_volts / (_E_C * reference_density)
    poisson = (
        debye_squared * ufl.grad(scaled_potential)[0] * ufl.grad(potential_test)[0]
        - (scaled_density - electrons + charge) * potential_test
    ) * ufl.dx

    boundary = [
        dirichlet_on_component(
            space, 0, position=0.0, value=density_boundary[0] / reference_density
        ),
        dirichlet_on_component(
            space, 0, position=length, value=density_boundary[1] / reference_density
        ),
        dirichlet_on_component(
            space, 1, position=0.0, value=potential_boundary[0] / electron_temperature_volts
        ),
        dirichlet_on_component(
            space, 1, position=length, value=potential_boundary[1] / electron_temperature_volts
        ),
    ]

    report = run_newton(
        continuity_scale * continuity + poisson,
        unknown,
        bcs=boundary,
        rtol=rtol,
        max_iterations=max_iterations,
        what="the drift-diffusion / Poisson system",
    )

    density = _rescale(sample(unknown.sub(0).collapse(), grid), reference_density)
    potential = _rescale(sample(unknown.sub(1).collapse(), grid), electron_temperature_volts)
    return CoupledSolution(
        density=density,
        potential=potential,
        electron_density=_boltzmann(potential, reference_density, electron_temperature_volts),
        newton=report,
    )


def _validate(
    *,
    diffusivity: float,
    reference_density: float,
    electron_temperature_volts: float,
    electron_coupling_rate: float,
    electron_coupling_reference: SourceTerm | None,
) -> None:
    if diffusivity <= 0.0:
        raise ValueError(f"diffusivity must be positive, got {diffusivity}")
    if reference_density <= 0.0:
        raise ValueError(f"reference_density must be positive, got {reference_density}")
    if electron_temperature_volts <= 0.0:
        raise ValueError(
            f"electron_temperature_volts must be positive, got {electron_temperature_volts}"
        )
    if electron_coupling_rate != 0.0 and electron_coupling_reference is None:
        raise ValueError(
            "electron_coupling_rate was given without electron_coupling_reference; the "
            "loss term needs a density to relax toward, and defaulting it to zero would "
            "silently add a sink the caller did not ask for"
        )


def _rescale(solution: FieldSolution, factor: float) -> FieldSolution:
    """Put a dimensionless solved field back onto its physical scale."""
    return FieldSolution(
        grid=solution.grid,
        nodal=solution.nodal * factor,
        cell_values=solution.cell_values * factor,
        function=solution.function,
        scale=factor,
    )


def _boltzmann(
    potential: FieldSolution, reference_density: float, electron_temperature_volts: float
) -> FieldSolution:
    """``n_e = n_s exp(e Phi / k T_e)`` evaluated on the potential's own samples."""
    return FieldSolution(
        grid=potential.grid,
        nodal=reference_density * np.exp(potential.nodal / electron_temperature_volts),
        cell_values=reference_density * np.exp(potential.cell_values / electron_temperature_volts),
        function=potential.function,
        scale=potential.scale,
    )
