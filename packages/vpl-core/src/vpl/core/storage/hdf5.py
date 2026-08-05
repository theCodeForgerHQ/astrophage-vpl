"""HDF5 primitives shared by the state and measurement writers — doc 08 §7.

doc 08 §7 puts fields and measurements in HDF5 "chunked, compressed", and is blunt about
the alternative: *"Never CSV for simulation output. No dtype, no units, no metadata, no
compression, no chunking — every one of which the project needs."* This module is the
place those five properties are actually supplied, once, so that no writer has to
remember them.

## The compression is lossless, deliberately

gzip with the shuffle pre-filter is a byte rearrangement plus DEFLATE: it reproduces the
input exactly. HDF5 also offers ``scaleoffset``, which compresses floats far harder by
discarding mantissa bits. It is not used anywhere in this package and must not be: doc 00
E3 promises bit-for-bit reproducibility, and a comparison against an artifact that lost
three bits cannot distinguish "the solver changed" from "the writer rounded".

## Units live in an attribute, next to the data

The state model already stores units as registry-parseable strings
(:class:`~vpl.core.state.fields.ScalarField`), and that string is written verbatim into a
``units`` attribute — the same convention netCDF and CF use, so an external reader finds
what it expects. The string is *not* re-parsed here; the constructor on the way back in
parses it through the shared registry and raises on anything it does not recognise, which
puts the failure at the boundary where it can still be explained.
"""

from __future__ import annotations

from typing import Any, Final

import h5py
import numpy as np
from numpy.typing import NDArray

from vpl.core.units import Q_, Quantity

__all__ = [
    "COMPRESSION",
    "COMPRESSION_LEVEL",
    "UNITS_ATTRIBUTE",
    "checked_group_name",
    "group_attributes",
    "read_array",
    "read_optional_quantity",
    "read_quantity",
    "read_units",
    "write_array",
    "write_quantity",
]

#: Lossless codec. See the module docstring for why nothing lossy appears here.
COMPRESSION: Final[str] = "gzip"

#: gzip level. Four is the knee of the curve for float64 field data: most of the ratio of
#: level 9 at a small fraction of the write cost, which matters when doc 10 §7 budgets
#: 860 GB of raw fields regenerated on demand rather than stored.
COMPRESSION_LEVEL: Final[int] = 4

#: Attribute carrying a dataset's units, following the netCDF/CF convention.
UNITS_ATTRIBUTE: Final[str] = "units"


def checked_group_name(name: str, *, what: str) -> str:
    """Reject a name HDF5 would read as a path rather than as a name.

    HDF5 treats ``/`` as a group separator, so an instrument or field called ``oes/iccd``
    would silently nest itself one level deeper than anyone asked and read back under a
    different name. Both are plugin-supplied (doc 08 §10), so neither is under this
    package's control and neither can be assumed well-behaved.
    """
    if "/" in name or not name.strip():
        raise ValueError(
            f"{what} {name!r} cannot name an HDF5 group: names must be non-empty and must "
            "not contain '/', which HDF5 reads as a path separator."
        )
    return name


def write_array(
    group: h5py.Group,
    name: str,
    values: NDArray[np.float64],
    *,
    units: str,
) -> None:
    """Write a float64 array chunked, compressed and carrying its units.

    Chunking is left to h5py's heuristic. It is deterministic given the shape and dtype,
    so two runs that produced the same array produce the same layout — and choosing a
    chunk shape by hand here would be guessing at access patterns this module cannot see.
    """
    dataset = group.create_dataset(
        name,
        data=np.asarray(values, dtype=np.float64),
        chunks=True,
        compression=COMPRESSION,
        compression_opts=COMPRESSION_LEVEL,
        shuffle=True,
    )
    dataset.attrs[UNITS_ATTRIBUTE] = units


def read_array(group: h5py.Group, name: str) -> NDArray[np.float64]:
    """Read a float64 array back exactly as it was written."""
    return np.asarray(group[name][...], dtype=np.float64)


def read_units(group: h5py.Group, name: str) -> str:
    """The units attribute of a dataset, as the registry-parseable string it was."""
    return str(group[name].attrs[UNITS_ATTRIBUTE])


def write_quantity(group: h5py.Group, name: str, value: Quantity) -> None:
    """Write a scalar dimensional quantity as a dataset plus its units.

    The magnitude is stored **in the units it was supplied in**, not converted to SI.
    Converting would introduce a rounding step between ``Q_(2.0, "ns")`` and what comes
    back, and doc 00 E3's bit-for-bit promise leaves no room for one. It also keeps the
    file readable: an operating point archived as ``5.0 mTorr`` is the number in the
    manifest (doc 08 §6), not ``0.6666...`` pascals.

    Scalar datasets are written uncompressed and unchunked. HDF5 cannot chunk a rank-zero
    dataset, and there would be nothing to gain from it if it could.
    """
    magnitude = np.asarray(value.magnitude, dtype=np.float64)
    if magnitude.ndim != 0:
        raise ValueError(f"{name} must be a scalar quantity, got shape {magnitude.shape}")

    dataset = group.create_dataset(name, data=magnitude)
    dataset.attrs[UNITS_ATTRIBUTE] = str(value.units)


def read_quantity(group: h5py.Group, name: str) -> Quantity:
    """Read back a scalar quantity written by :func:`write_quantity`."""
    magnitude = float(np.asarray(group[name][()], dtype=np.float64))
    return Q_(magnitude, read_units(group, name))


def read_optional_quantity(group: h5py.Group, name: str) -> Quantity | None:
    """The quantity, or ``None`` when the dataset is absent.

    Absence is meaningful rather than defaulted: ``rf_frequency=None`` is a DC bias
    (doc 02 §3.3), and substituting a zero would turn a DC run into an RF run with an
    impossible drive — which :class:`~vpl.core.state.params.PlasmaParams` rejects, but
    only after the artifact had already claimed it.
    """
    if name not in group:
        return None
    return read_quantity(group, name)


def group_attributes(group: h5py.Group) -> dict[str, Any]:
    """A plain dictionary of a group's attributes, for the metadata decoders."""
    return dict(group.attrs)
