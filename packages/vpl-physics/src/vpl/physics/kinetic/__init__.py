"""L2 — the kinetic layer, 1D3V electrostatic PIC-MCC. doc 03 §4, ADR-003.

ADR-003 decides *build* rather than adopt: Smilei and WarpX are electromagnetic and
3-D-oriented, and this problem is 1D3V electrostatic with a boundary-dominated sheath and a
requirement to emit IEDFs into a surrogate-training pipeline over thousands of runs.

ADR-003 also records what that costs. The Smilei cross-check it named as the mitigation for
risk RT-05 is **deferred**, so this layer carries a single-implementation risk. What replaces
it is verification against references the kernel cannot influence — see ADR-003's consequences
table, and note that ADR-011 is why the distinction is taken seriously here.
"""

from vpl.physics.kinetic.precision import enable_double_precision, is_double_precision

# Before anything else. JAX defaults to float32 and this kernel is not correct in it —
# see precision.py for why that is a correctness statement rather than a preference.
enable_double_precision()

from vpl.physics.kinetic.fields import (  # noqa: E402  — must follow the x64 switch
    deposit_cic,
    gather_cic,
    solve_poisson_dirichlet,
)
from vpl.physics.kinetic.grid import UniformGrid  # noqa: E402

__all__ = [
    "UniformGrid",
    "deposit_cic",
    "enable_double_precision",
    "gather_cic",
    "is_double_precision",
    "solve_poisson_dirichlet",
]
