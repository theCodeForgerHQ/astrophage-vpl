"""The data model — doc 08 §7.

Types in this package describe *what exists*, not what is done to it. A solver returns a
:class:`PlasmaState`; an instrument returns a :class:`Measurement`; an engine returns a
:class:`Posterior`. None of them know how the others are computed, which is what lets the
inverse solver stay ignorant of which fidelity level produced its forward model
(doc 00 E1).
"""

from vpl.core.state.grid import PhaseGrid, SpatialGrid, TimeGrid

__all__ = ["PhaseGrid", "SpatialGrid", "TimeGrid"]
