"""The inverse problem — doc 05.

Three modules, in the order the posterior is assembled:

* :mod:`vpl.inverse.parameters` — the Level A control-parameter vector `θ_c` of doc 05
  §2.1, and the bijections onto the unconstrained `R^8` every engine of doc 05 §5 works in.
* :mod:`vpl.inverse.priors` — the §2.1 prior table, the log-Jacobian that makes it a
  correct density in that unconstrained space, and the §4.2 physics penalties.
* :mod:`vpl.inverse.likelihood` — the §3.1 per-channel families, the §3.2 asynchronous
  sum over acquisition windows, and the §3.3 heavy-tailed variants.

**This package does not import `vpl.physics` or `vpl.instruments`, and must not.** doc 08
§3 keeps the inverse layer ignorant of any particular forward model, and doc 08 §8 makes
the isolation an import-graph test. doc 05 §7.1 is the reason it matters: the truth
generator and the inversion are required to differ in physics level, spatial grid, time
discretisation, collision set, EEDF parameterisation and calibration. An inverse package
that could reach for the solver next door would make that mismatch a matter of discipline
rather than of structure, and doc 05 §7 is explicit that the inverse crime is "guarded
against structurally rather than by good intentions".
"""

from vpl.inverse.likelihood import (
    OES_GAUSSIAN_SWITCH_COUNTS,
    ChannelLikelihood,
    ChannelMismatchError,
    CorrelatedGaussianChannel,
    GaussianChannel,
    MixtureChannel,
    PoissonChannel,
    StudentTChannel,
    SwitchedPoissonGaussianChannel,
    coloured_noise_covariance,
    correlated_gaussian_log_likelihood,
    detection_mask,
    diagonal_covariance,
    gaussian_log_likelihood,
    outlier_mixture_log_likelihood,
    poisson_log_likelihood,
    shared_systematic_covariance,
    student_t_log_likelihood,
    switched_poisson_gaussian_log_likelihood,
    total_log_likelihood,
)
from vpl.inverse.parameters import (
    CONTROL_PARAMETERS,
    N_CONTROL,
    ControlParameters,
    ControlParameterSpec,
    IdentityTransform,
    LogitTransform,
    LogTransform,
    Transform,
    control_parameter_names,
    log_abs_det_jacobian,
    unconstrained_bounds,
)
from vpl.inverse.priors import (
    ControlPrior,
    LogNormalPrior,
    LogUniformPrior,
    NormalPrior,
    Prior,
    TruncatedNormalPrior,
    UniformPrior,
    bohm_penalty,
    default_control_prior,
    quasineutrality_penalty,
    smoothness_log_prior,
)

__all__ = [
    "CONTROL_PARAMETERS",
    "N_CONTROL",
    "OES_GAUSSIAN_SWITCH_COUNTS",
    "ChannelLikelihood",
    "ChannelMismatchError",
    "ControlParameterSpec",
    "ControlParameters",
    "ControlPrior",
    "CorrelatedGaussianChannel",
    "GaussianChannel",
    "IdentityTransform",
    "LogNormalPrior",
    "LogTransform",
    "LogUniformPrior",
    "LogitTransform",
    "MixtureChannel",
    "NormalPrior",
    "PoissonChannel",
    "Prior",
    "StudentTChannel",
    "SwitchedPoissonGaussianChannel",
    "Transform",
    "TruncatedNormalPrior",
    "UniformPrior",
    "bohm_penalty",
    "coloured_noise_covariance",
    "control_parameter_names",
    "correlated_gaussian_log_likelihood",
    "default_control_prior",
    "detection_mask",
    "diagonal_covariance",
    "gaussian_log_likelihood",
    "log_abs_det_jacobian",
    "outlier_mixture_log_likelihood",
    "poisson_log_likelihood",
    "quasineutrality_penalty",
    "shared_systematic_covariance",
    "smoothness_log_prior",
    "student_t_log_likelihood",
    "switched_poisson_gaussian_log_likelihood",
    "total_log_likelihood",
    "unconstrained_bounds",
]
