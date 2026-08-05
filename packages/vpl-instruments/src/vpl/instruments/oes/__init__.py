"""Optical emission spectroscopy — doc 04 §2 and §5, doc 02 §6.

Doc 02 §6.1 calls OES "the fast backbone channel": a 750 mm imaging spectrograph looking
along the sheath normal through a gated ICCD, so one exposure gives a spatially resolved
spectrum through the whole sheath.

The chain, in the order it runs:

1. :mod:`~vpl.instruments.oes.levels` — the level system and the EEDF-integrated rate
   coefficients, including the superelastic ones by Klein-Rosseland.
2. :mod:`~vpl.instruments.oes.escape` — the Holstein-Biberman escape factors of doc 04
   §2.3, without which the metastable-coupled lines of doc 02 §6.3 give the wrong ``T_e``.
3. :mod:`~vpl.instruments.oes.cr` — the quasi-static collisional-radiative solve for the
   upper-state populations.
4. :mod:`~vpl.instruments.oes.emissivity` — doc 04 §2.1's
   ``eps_ul = n_u A_ul h nu_ul / 4pi``, and the chord integral that turns it into a
   radiance.
5. :mod:`~vpl.instruments.oes.lineshape` and
   :mod:`~vpl.instruments.oes.spectrograph` — the profile each line carries and the
   instrument that smears it.
6. :mod:`~vpl.instruments.oes.instrument` — :class:`OesInstrument`, which is the doc 08 §4
   ``Instrument`` contract over all of the above.

Every module names its own simplifications in its docstring; the package README indexes
them. The one worth repeating here: **doc 04 §8 V-24, "CR model vs published Ar line
ratios", is not satisfied.** Nothing in this package has been compared against a real argon
measurement or a published argon CR calculation.
"""

from vpl.instruments.oes.cr import (
    CollisionalRadiativeModel,
    CrConvergenceError,
    LevelPopulations,
)
from vpl.instruments.oes.emissivity import (
    LineEmission,
    chord_radiance,
    emission_spectrum,
    line_emissivity,
)
from vpl.instruments.oes.escape import (
    LineProfileShape,
    TrappedLine,
    escape_factor,
    line_centre_optical_depth,
    slab_escape_probability,
)
from vpl.instruments.oes.instrument import MaxwellianEedf, OesInstrument
from vpl.instruments.oes.levels import (
    ElectronImpactChannel,
    Level,
    LevelSystem,
    RadiativeChannel,
)
from vpl.instruments.oes.lineshape import (
    doppler_fwhm_nm,
    natural_fwhm_nm,
    voigt_fwhm_nm,
    voigt_profile,
)
from vpl.instruments.oes.spectrograph import Grating, Spectrograph

__all__ = [
    "CollisionalRadiativeModel",
    "CrConvergenceError",
    "ElectronImpactChannel",
    "Grating",
    "Level",
    "LevelPopulations",
    "LevelSystem",
    "LineEmission",
    "LineProfileShape",
    "MaxwellianEedf",
    "OesInstrument",
    "RadiativeChannel",
    "Spectrograph",
    "TrappedLine",
    "chord_radiance",
    "doppler_fwhm_nm",
    "emission_spectrum",
    "escape_factor",
    "line_centre_optical_depth",
    "line_emissivity",
    "natural_fwhm_nm",
    "slab_escape_probability",
    "voigt_fwhm_nm",
    "voigt_profile",
]
