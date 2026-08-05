"""The EEDF layer — doc 03 §3.2, doc 03 §8 A6, doc 11 WBS 1.8.

Doc 03 §8 assumption A6 is "Maxwellian EEDF for rate coefficients", and its Handling
column reads, in full: **Replaced by two-term Boltzmann solver output (§3.2)**. This
package is that replacement.

Doc 03 §3.2 states what it must produce::

    k_iz(E/N),  k_ex,j(E/N),  mu_e(E/N),  D_e(E/N),  <eps>(E/N)

and doc 04 §4.2 adds that the EEDF itself has to survive, not only its moments: the
Thomson spectrum is computed "from the actual EEDF", because "fitting a Maxwellian to a
bi-Maxwellian plasma yields a ``T_e`` that corresponds to neither population".

## Where to start

- :class:`EnergyGrid` — the finite-volume energy grid everything lives on.
- :func:`kinetics_from_set` — a parsed LXCat database, on that grid.
- :class:`TwoTermSolver` — the solve. :class:`EedfSolution` carries ``f0`` and its moments.
- :func:`tabulate` — the doc 03 §3.2 table over ``E/N``.
- :func:`tabulate_each_electron_set` — the same, under all three electron databases of
  doc 09 §2.1, with the spread between them as a returned number rather than a caveat.
- :func:`generalised_eedf` — Maxwellian and Druyvesteyn, for comparison and for the
  ``kappa`` of doc 05 §2.1. The solver reaches both from the collision physics; they are
  not inputs to it.

## Build versus buy

Doc 08 §2's table says "EEDF / Boltzmann solver | Buy | BOLSIG+ or ``bolos``". Neither is
usable in this project, and **ADR-009** records what was tried, what was measured, and why
the two-term expansion of Hagelaar & Pitchford (2005) is implemented here instead. Doc 00
C2 forbids unsolved problems, not textbook ones; the deviation from the buy table is the
thing that needed arguing, and it is argued in the ADR rather than assumed here.
"""

from vpl.physics.eedf.analytic import (
    DRUYVESTEYN_KAPPA,
    MAXWELLIAN_KAPPA,
    AnalyticEedf,
    druyvesteyn_eedf,
    generalised_eedf,
    maxwellian_eedf,
)
from vpl.physics.eedf.grid import EnergyGrid
from vpl.physics.eedf.kinetics import (
    DEFAULT_ABOVE_GRID,
    DEFAULT_BELOW_GRID,
    ElectronKinetics,
    InelasticChannel,
    kinetics_from_set,
)
from vpl.physics.eedf.solver import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RELATIVE_TOLERANCE,
    GAMMA,
    GRID_TRUNCATION_TOLERANCE,
    ROOM_TEMPERATURE_EV,
    TOWNSEND_V_M2,
    EedfConvergenceError,
    EedfSolution,
    IonisationSharing,
    TwoTermSolver,
)
from vpl.physics.eedf.tabulate import (
    DatabaseSpread,
    ElectronSetSource,
    RateTable,
    TabulatedQuantity,
    tabulate,
    tabulate_each_electron_set,
    tabulate_electron_sets,
)

__all__ = [
    "DEFAULT_ABOVE_GRID",
    "DEFAULT_BELOW_GRID",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_RELATIVE_TOLERANCE",
    "DRUYVESTEYN_KAPPA",
    "GAMMA",
    "GRID_TRUNCATION_TOLERANCE",
    "MAXWELLIAN_KAPPA",
    "ROOM_TEMPERATURE_EV",
    "TOWNSEND_V_M2",
    "AnalyticEedf",
    "DatabaseSpread",
    "EedfConvergenceError",
    "EedfSolution",
    "ElectronKinetics",
    "ElectronSetSource",
    "EnergyGrid",
    "InelasticChannel",
    "IonisationSharing",
    "RateTable",
    "TabulatedQuantity",
    "TwoTermSolver",
    "druyvesteyn_eedf",
    "generalised_eedf",
    "kinetics_from_set",
    "maxwellian_eedf",
    "tabulate",
    "tabulate_each_electron_set",
    "tabulate_electron_sets",
]
