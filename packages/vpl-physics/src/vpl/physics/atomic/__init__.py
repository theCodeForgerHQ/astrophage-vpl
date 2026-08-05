"""The atomic-data layer — doc 09 §2, §5, §6; doc 03 §4.5; doc 04 §2.

Doc 09 §5 fixes what this package is:

    The repository stores *references and loaders*, not bulk third-party data.

So there is no data here. There is a register of the datasets doc 09 §2 names, a version
lock that fails loudly when upstream changes, parsers for the two formats the project
consumes, and one interpolator with an explicit extrapolation policy — because doc 03
§4.5 makes the energy dependence of the charge-exchange cross section the most
consequential atomic-data choice in the project, and a silent extrapolation off the end
of a table is indistinguishable from data once it is inside a rate integral.

Start at :class:`AtomicDataStore`.
"""

from vpl.physics.atomic.dataset import (
    DATASETS,
    LOCK_FILENAME,
    LXCAT_LICENCE,
    NIST_LICENCE,
    OPENADAS_LICENCE,
    BundledDataError,
    DatasetId,
    DatasetLock,
    DatasetNotRecordedError,
    DatasetSpec,
    DatasetVersion,
    ElectronDatabase,
    Licence,
    LockIntegrityError,
    UpstreamDataChangedError,
    bibliography,
    checked_cache_root,
    fetch_dataset,
)
from vpl.physics.atomic.interpolation import (
    DEFAULT_ABOVE,
    DEFAULT_BELOW,
    ExtrapolationPolicy,
    OutsideTabulatedRangeError,
    interpolate_cross_section,
    interpolate_set,
)
from vpl.physics.atomic.lxcat import (
    REQUIRED_ELECTRON_PROCESSES,
    REQUIRED_ION_PROCESSES,
    THRESHOLD_CONSISTENCY_TOLERANCE,
    CrossSection,
    CrossSectionSet,
    LxcatParseError,
    ProcessType,
    parse_lxcat,
)
from vpl.physics.atomic.nist import (
    DOC_02_LINE_SET,
    LINE_MATCH_TOLERANCE_NM,
    WAVELENGTH_ENERGY_TOLERANCE,
    AccuracyGrade,
    LineList,
    LineSpec,
    NistParseError,
    Transition,
    parse_nist_asd,
)
from vpl.physics.atomic.store import AtomicDataStore

__all__ = [
    "DATASETS",
    "DEFAULT_ABOVE",
    "DEFAULT_BELOW",
    "DOC_02_LINE_SET",
    "LINE_MATCH_TOLERANCE_NM",
    "LOCK_FILENAME",
    "LXCAT_LICENCE",
    "NIST_LICENCE",
    "OPENADAS_LICENCE",
    "REQUIRED_ELECTRON_PROCESSES",
    "REQUIRED_ION_PROCESSES",
    "THRESHOLD_CONSISTENCY_TOLERANCE",
    "WAVELENGTH_ENERGY_TOLERANCE",
    "AccuracyGrade",
    "AtomicDataStore",
    "BundledDataError",
    "CrossSection",
    "CrossSectionSet",
    "DatasetId",
    "DatasetLock",
    "DatasetNotRecordedError",
    "DatasetSpec",
    "DatasetVersion",
    "ElectronDatabase",
    "ExtrapolationPolicy",
    "Licence",
    "LineList",
    "LineSpec",
    "LockIntegrityError",
    "LxcatParseError",
    "NistParseError",
    "OutsideTabulatedRangeError",
    "ProcessType",
    "Transition",
    "UpstreamDataChangedError",
    "bibliography",
    "checked_cache_root",
    "fetch_dataset",
    "interpolate_cross_section",
    "interpolate_set",
    "parse_lxcat",
    "parse_nist_asd",
]
