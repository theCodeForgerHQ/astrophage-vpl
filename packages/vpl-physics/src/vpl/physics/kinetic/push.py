"""The particle push — doc 03 §4.2 step 4.

doc 03 §4.2 says "Boris / leapfrog". **Leapfrog is implemented, and that is the whole of
Boris for this problem**: the Boris algorithm's distinguishing feature is its exact rotation
of the velocity about a magnetic field, and doc 03 §4.1 specifies an *electrostatic* model
with no ``B``. With ``B = 0`` the Boris rotation is the identity and the two coincide
exactly, so implementing Boris here would be a rotation by zero, dressed up.

**If a magnetic field is ever added** — doc 04 §3 mentions Zeeman splitting for LIF, which
would imply one — this is the function to replace, and the replacement is the standard Boris
half-acceleration/rotation/half-acceleration. It is called out here so the substitution is a
known edit rather than a discovery.

## Why the transverse components exist at all

The field is one-dimensional, so ``v_x`` and ``v_y`` are untouched here and change only
through collisions. They are carried anyway because doc 03 §4.1 requires it: LIF measures a
*projected* velocity along the laser beam and Thomson scattering measures the full electron
distribution, so a 1D1V code could not produce either observable. The cost is two-thirds of
the velocity memory and none of the compute.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from vpl.physics.kinetic.precision import enable_double_precision

enable_double_precision()

__all__ = ["Species", "push_leapfrog"]


@dataclass(frozen=True, slots=True)
class Species:
    """One particle species and the constants the push needs from it.

    Attributes:
        name: Identifier, used as a key in the emitted state and in diagnostics.
        charge_c: Signed charge of a *physical* particle. Signed, not magnitude — the sign
            is what makes electrons and ions move in opposite directions in the sheath, and
            a magnitude here with the sign applied at the call site is how that gets lost.
        mass_kg: Mass of a physical particle.
        weight: Physical particles represented by one computational particle. This is what
            makes 1e17 m^-3 tractable with 1e6 markers.
    """

    name: str
    charge_c: float
    mass_kg: float
    weight: float

    def __post_init__(self) -> None:
        if not self.mass_kg > 0.0:
            raise ValueError(f"species {self.name!r} needs a positive mass, got {self.mass_kg}")
        if not self.weight > 0.0:
            raise ValueError(f"species {self.name!r} needs a positive weight, got {self.weight}")

    @property
    def charge_to_mass(self) -> float:
        """``q/m``, signed. The only combination the push actually uses."""
        return self.charge_c / self.mass_kg


def push_leapfrog(
    species: Species,
    *,
    positions_m: Array,
    velocities_m_per_s: Array,
    e_field_v_per_m: Array,
    dt_s: float,
) -> tuple[Array, Array]:
    """Advance velocity then position by one step — the leapfrog of doc 03 §4.2 step 4.

    Velocity first, then position with the *updated* velocity. That ordering is what makes
    the scheme time-reversible and symplectic, which is why the energy error stays bounded
    over 65 800 steps instead of drifting; updating position first with the old velocity is
    forward Euler wearing a leapfrog's clothes, and it heats.

    Args:
        species: Supplies ``q/m``.
        positions_m: Particle positions, shape ``(n,)``.
        velocities_m_per_s: Particle velocities, shape ``(n, 3)`` — the "3V" of 1D3V.
        e_field_v_per_m: Field already gathered to each particle, shape ``(n,)``. Only the
            ``z`` component exists in 1-D electrostatics.
        dt_s: Timestep, already checked against doc 03 §4.3's constraints by the caller.

    Returns:
        The advanced ``(positions, velocities)``.
    """
    if velocities_m_per_s.ndim != 2 or velocities_m_per_s.shape[1] != 3:
        raise ValueError(
            f"velocities must have shape (n, 3) for a 1D3V model, got {velocities_m_per_s.shape}"
        )
    if e_field_v_per_m.shape != positions_m.shape:
        raise ValueError(
            f"field shape {e_field_v_per_m.shape} does not match positions shape "
            f"{positions_m.shape}"
        )
    if velocities_m_per_s.shape[0] != positions_m.shape[0]:
        raise ValueError(
            f"velocities shape {velocities_m_per_s.shape} does not match positions shape "
            f"{positions_m.shape}"
        )

    dvz = species.charge_to_mass * e_field_v_per_m * dt_s
    velocities = velocities_m_per_s.at[:, 2].add(dvz)
    positions = positions_m + velocities[:, 2] * dt_s
    return positions, velocities
