"""The model hierarchy — doc 03 §1.

One interface, four fidelity levels. The inverse solver never learns which one it is
talking to (doc 00 E1), so this enum exists for provenance and for verification gates,
not for dispatch.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Fidelity"]


class Fidelity(Enum):
    """Which level of the doc 03 §1 hierarchy produced a state.

    Deliberately **not** ordered. ``L0 < L1 < L2`` is defensible, but L3 is an *emulator
    of L2* carrying its own budgeted error (doc 03 §5.4) — it is neither more nor less
    faithful than the thing it emulates. A total order would invite ``if fidelity >= L2``
    checks that accept or reject the surrogate for the wrong reason, and the surrogate is
    precisely what the sampler actually calls.
    """

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]

    @property
    def resolves_distribution(self) -> bool:
        """Whether this level produces the ion distribution function directly.

        doc 03 §1 calls L2 "the only level that produces a correct IEDF in the
        collisional and RF-transit regimes". L1 reconstructs one by forcing a drifting
        Maxwellian onto its moments, which doc 03 §6 calls a genuine weakness and treats
        as one. Keeping the distinction as a flag makes it machine-checkable instead of
        a matter of whoever is reading the code remembering it.

        L3 inherits ``True`` because it emulates the L2 distribution through its POD
        basis — with the emulator error budgeted separately (doc 06 §4 term 10).
        """
        return self in (Fidelity.L2, Fidelity.L3)

    @property
    def emulates(self) -> Fidelity | None:
        """The level this one is a surrogate for, if any."""
        return Fidelity.L2 if self is Fidelity.L3 else None


_DESCRIPTIONS: dict[Fidelity, str] = {
    Fidelity.L0: (
        "Analytic: Child-Langmuir, matrix sheath, Bohm criterion, Lieberman RF sheath. "
        "Verification anchor, MCMC proposal and sanity bound."
    ),
    Fidelity.L1: (
        "Fluid: drift-diffusion ions, Boltzmann electrons, Poisson. Workhorse for "
        "sweeps where the IVDF is near-drifting-Maxwellian."
    ),
    Fidelity.L2: (
        "Kinetic: 1D3V electrostatic PIC with Monte-Carlo collisions. Ground-truth "
        "generator and the only level with a correct IEDF in the collisional and "
        "RF-transit regimes."
    ),
    Fidelity.L3: (
        "Surrogate: GP or neural-operator emulator trained on L2. What the sampler "
        "actually calls; its error is budgeted, not ignored."
    ),
}
