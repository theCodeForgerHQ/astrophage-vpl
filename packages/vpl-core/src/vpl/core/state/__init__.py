"""The data model — doc 08 §7.

Types in this package describe *what exists*, not what is done to it. A solver returns a
:class:`PlasmaState`; an instrument returns a :class:`Measurement`; an engine returns a
:class:`Posterior`. None of them know how the others are computed, which is what lets the
inverse solver stay ignorant of which fidelity level produced its forward model
(doc 00 E1).
"""

from vpl.core.state.fidelity import Fidelity
from vpl.core.state.fields import ScalarField, VelocityDistribution
from vpl.core.state.grid import PhaseGrid, SpatialGrid, TimeGrid
from vpl.core.state.measurement import (
    AcquisitionWindow,
    CalibrationState,
    InstrumentId,
    Measurement,
    MeasurementSet,
    Observable,
    WindowMode,
)
from vpl.core.state.params import PlasmaParams
from vpl.core.state.plasma import REQUIRED_FIELDS, PlasmaState
from vpl.core.state.posterior import (
    GATE_MAX_DIVERGENCES,
    GATE_MAX_R_HAT,
    GATE_MIN_ESS,
    CredibleInterval,
    DerivedQuantities,
    ParameterLevel,
    Posterior,
    SamplerDiagnostics,
)
from vpl.core.state.species import Species

__all__ = [
    "GATE_MAX_DIVERGENCES",
    "GATE_MAX_R_HAT",
    "GATE_MIN_ESS",
    "REQUIRED_FIELDS",
    "AcquisitionWindow",
    "CalibrationState",
    "CredibleInterval",
    "DerivedQuantities",
    "Fidelity",
    "InstrumentId",
    "Measurement",
    "MeasurementSet",
    "Observable",
    "ParameterLevel",
    "PhaseGrid",
    "PlasmaParams",
    "PlasmaState",
    "Posterior",
    "SamplerDiagnostics",
    "ScalarField",
    "SpatialGrid",
    "Species",
    "TimeGrid",
    "VelocityDistribution",
    "WindowMode",
]
