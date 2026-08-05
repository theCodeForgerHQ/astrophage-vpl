"""L0 — analytic sheath models. Doc 03 §2.

Verification anchor, MCMC proposal and sanity bound (doc 03 §1). Verification gate V-03
(doc 07) checks L1 and L2 against what this package computes.
"""

from vpl.physics.analytic.sheath import (
    DEFAULT_DOMAIN_SHEATHS,
    DEFAULT_EDGE_TO_CENTRE_RATIO,
    DEFAULT_SAMPLES_PER_SHEATH,
    GAMMA_I_ADIABATIC,
    GAMMA_I_COLD_ION,
    SIGMA_CX_ARGON,
    AnalyticSheathSolver,
    CollisionalRegime,
    SheathModel,
    bohm_speed,
    child_langmuir_current_density,
    child_langmuir_potential,
    child_langmuir_thickness,
    collisional_current_density,
    collisionality,
    cx_mean_free_path,
    debye_length,
    ion_energy_flux,
    ion_flux,
    matrix_sheath_potential,
    matrix_sheath_thickness,
    matrix_sheath_validity_ratio,
    sheath_edge_density,
)

__all__ = [
    "DEFAULT_DOMAIN_SHEATHS",
    "DEFAULT_EDGE_TO_CENTRE_RATIO",
    "DEFAULT_SAMPLES_PER_SHEATH",
    "GAMMA_I_ADIABATIC",
    "GAMMA_I_COLD_ION",
    "SIGMA_CX_ARGON",
    "AnalyticSheathSolver",
    "CollisionalRegime",
    "SheathModel",
    "bohm_speed",
    "child_langmuir_current_density",
    "child_langmuir_potential",
    "child_langmuir_thickness",
    "collisional_current_density",
    "collisionality",
    "cx_mean_free_path",
    "debye_length",
    "ion_energy_flux",
    "ion_flux",
    "matrix_sheath_potential",
    "matrix_sheath_thickness",
    "matrix_sheath_validity_ratio",
    "sheath_edge_density",
]
