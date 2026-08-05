"""Plasma state artifacts — doc 08 §7, "Plasma state fields | HDF5 (chunked, compressed)".

A written state is everything the inverse chain needs to be re-run against, and doc 13 §5
treats raw fields as a 90-day cache that is *regenerated* rather than kept. That is only
affordable because regeneration can be checked: a state read back from disk must equal the
state that was written, exactly, or the comparison against the regenerated cache is not a
comparison at all. So nothing here converts, rounds or normalises — magnitudes are stored
in the units they arrived in and read back unchanged.

## What is not in the file

Nothing third-party. doc 09 §5 forbids redistributing LXCat and OpenADAS raw tables, and
this layer never sees one: it writes the *state a solver produced* — grids, fields, an ion
distribution and the control parameters — not the atomic data that went into producing it.
Where that data matters it is referenced by hash and citation from the manifest (doc 13 §2
``data_versions``), and the artifact inherits the reference through
:attr:`~vpl.core.provenance.Provenance.manifest_sha256` rather than by carrying a copy.
doc 09 §5's architectural consequence — "the repository stores *references and loaders*,
not bulk third-party data" — is honoured by the loader, and this module is simply never in
a position to break it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

import h5py

from vpl.core.provenance import Provenance
from vpl.core.state.fidelity import Fidelity
from vpl.core.state.fields import ScalarField, VelocityDistribution
from vpl.core.state.grid import SpatialGrid, TimeGrid
from vpl.core.state.params import PlasmaParams
from vpl.core.state.plasma import PlasmaState
from vpl.core.state.species import Species
from vpl.core.storage.hdf5 import (
    checked_group_name,
    group_attributes,
    read_array,
    read_optional_quantity,
    read_quantity,
    read_units,
    write_array,
    write_quantity,
)
from vpl.core.storage.metadata import (
    ArtifactKind,
    artifact_attributes,
    check_artifact_kind,
    provenance_from_attributes,
    require_provenance,
)

__all__ = ["ArchivedState", "read_plasma_state", "write_plasma_state"]

_FIELDS_GROUP: Final[str] = "fields"
_GRID_GROUP: Final[str] = "grid"
_PARAMS_GROUP: Final[str] = "params"
_SPECIES_GROUP: Final[str] = "species"
_DISTRIBUTION_GROUP: Final[str] = "ion_distribution"

_Z_DATASET: Final[str] = "z"
_T_DATASET: Final[str] = "t"
_VELOCITY_DATASET: Final[str] = "v"
_DISTRIBUTION_DATASET: Final[str] = "f"

_FIDELITY_ATTRIBUTE: Final[str] = "fidelity"

_LENGTH_UNITS: Final[str] = "m"
_TIME_UNITS: Final[str] = "s"
_VELOCITY_UNITS: Final[str] = "m / s"

#: Units of the phase-space density ``f_i(z, v_z)``.
#:
#: Not carried by :class:`~vpl.core.state.fields.VelocityDistribution`, which stores the
#: array alone, but implied exactly by the normalisation doc 03 §5.2 states — ``int f
#: dv_z = n`` in ``m^-3`` over a velocity axis in ``m/s``. Written so that a reader
#: outside this package is not left to infer it.
_DISTRIBUTION_UNITS: Final[str] = "m**-4 * s"


class ArchivedState(NamedTuple):
    """A plasma state read back from disk, beside the provenance it was written with."""

    state: PlasmaState
    provenance: Provenance


def _write_params(group: h5py.Group, params: PlasmaParams) -> None:
    """Write the doc 05 §2.1 control parameters.

    Written and read field by field, in the declaration order of
    :class:`~vpl.core.state.params.PlasmaParams`, rather than through a loop over a name
    table. The two functions are then mirror images, and a control parameter added to one
    and forgotten in the other is visible side by side instead of hidden in a tuple.

    The dimensionless parameters — a secondary-emission yield, an EEDF shape exponent and
    a phase in radians — are attributes rather than datasets because they carry no units
    to attach, and inventing a units string for them would be metadata the state model
    does not have.
    """
    write_quantity(group, "n_0", params.n_0)
    write_quantity(group, "T_e", params.T_e)
    write_quantity(group, "T_i", params.T_i)
    write_quantity(group, "T_g", params.T_g)
    write_quantity(group, "pressure", params.pressure)
    write_quantity(group, "bias", params.bias)
    if params.rf_frequency is not None:
        # Absent means DC (doc 02 §3.3). A zero would be an RF run with an impossible
        # drive, which PlasmaParams rejects — but only after the artifact claimed it.
        write_quantity(group, "rf_frequency", params.rf_frequency)

    group.attrs["gamma_se"] = float(params.gamma_se)
    group.attrs["kappa"] = float(params.kappa)
    group.attrs["rf_phase"] = float(params.rf_phase)

    species = group.create_group(_SPECIES_GROUP)
    species.attrs["name"] = params.species.name
    species.attrs["charge_number"] = params.species.charge_number
    write_quantity(species, "mass", params.species.mass)


def _read_params(group: h5py.Group) -> PlasmaParams:
    """The mirror of :func:`_write_params`."""
    species_group = group[_SPECIES_GROUP]

    return PlasmaParams(
        species=Species(
            name=str(species_group.attrs["name"]),
            mass=read_quantity(species_group, "mass"),
            charge_number=int(species_group.attrs["charge_number"]),
        ),
        n_0=read_quantity(group, "n_0"),
        T_e=read_quantity(group, "T_e"),
        T_i=read_quantity(group, "T_i"),
        T_g=read_quantity(group, "T_g"),
        pressure=read_quantity(group, "pressure"),
        bias=read_quantity(group, "bias"),
        gamma_se=float(group.attrs["gamma_se"]),
        kappa=float(group.attrs["kappa"]),
        rf_frequency=read_optional_quantity(group, "rf_frequency"),
        rf_phase=float(group.attrs["rf_phase"]),
    )


def _write_distribution(root: h5py.Group, distribution: VelocityDistribution) -> None:
    group = root.create_group(_DISTRIBUTION_GROUP)
    write_array(group, _VELOCITY_DATASET, distribution.v_m_per_s, units=_VELOCITY_UNITS)
    write_array(group, _DISTRIBUTION_DATASET, distribution.values, units=_DISTRIBUTION_UNITS)
    group.attrs["species"] = distribution.species.name


def _read_distribution(
    root: h5py.Group, *, grid: SpatialGrid, species: Species
) -> VelocityDistribution | None:
    if _DISTRIBUTION_GROUP not in root:
        return None

    group = root[_DISTRIBUTION_GROUP]
    return VelocityDistribution(
        grid=grid,
        v_m_per_s=read_array(group, _VELOCITY_DATASET),
        values=read_array(group, _DISTRIBUTION_DATASET),
        species=species,
    )


def write_plasma_state(path: Path, state: PlasmaState, *, provenance: Provenance) -> Path:
    """Write a plasma state to a chunked, compressed HDF5 file — doc 08 §7.

    The provenance is checked *before* the file is created. A half-written artifact that a
    later read treats as real is worse than no artifact: it is a result whose provenance
    is absent for a reason nobody recorded.

    Args:
        path: Destination file. Overwritten if it exists.
        state: The state to archive. Fields are written in the deterministic order of
            :attr:`~vpl.core.state.plasma.PlasmaState.field_names`, so two runs that
            produced the same state produce the same file.
        provenance: The doc 08 §7 record. Required — see :class:`MissingProvenanceError`.

    Returns:
        The path written.

    Raises:
        MissingProvenanceError: If ``provenance`` is absent.
        ValueError: If a field name cannot be used as an HDF5 group name.
    """
    record = require_provenance(provenance)
    for name in state.field_names:
        checked_group_name(name, what="field")

    with h5py.File(path, "w") as handle:
        handle.attrs.update(artifact_attributes(ArtifactKind.PLASMA_STATE, record))
        handle.attrs[_FIDELITY_ATTRIBUTE] = state.fidelity.value

        grid = handle.create_group(_GRID_GROUP)
        write_array(grid, _Z_DATASET, state.grid.z_m, units=_LENGTH_UNITS)
        if state.time is not None:
            write_array(grid, _T_DATASET, state.time.t_s, units=_TIME_UNITS)

        fields = handle.create_group(_FIELDS_GROUP)
        for name in state.field_names:
            field = state.field(name)
            write_array(fields, name, field.values, units=field.units)

        _write_params(handle.create_group(_PARAMS_GROUP), state.params)

        if state.ion_distribution is not None:
            _write_distribution(handle, state.ion_distribution)

    return path


def read_plasma_state(path: Path) -> ArchivedState:
    """Read back a state written by :func:`write_plasma_state`.

    Every value passes back through the state model's own constructors, so the checks that
    applied when the state was produced apply again on the way in. That is the point of
    reconstructing rather than returning raw arrays: an artifact hand-edited into an
    inconsistent state fails here, loudly, instead of at the likelihood.

    Raises:
        MissingProvenanceError: If the doc 08 §7 provenance block is incomplete.
        ValueError: If the file holds a different artifact, or a state the model rejects.
    """
    with h5py.File(path, "r") as handle:
        attributes = group_attributes(handle)
        check_artifact_kind(attributes, ArtifactKind.PLASMA_STATE)
        record = provenance_from_attributes(attributes)

        grid_group = handle[_GRID_GROUP]
        grid = SpatialGrid(z_m=read_array(grid_group, _Z_DATASET))
        time = (
            TimeGrid(t_s=read_array(grid_group, _T_DATASET)) if _T_DATASET in grid_group else None
        )

        field_group = handle[_FIELDS_GROUP]
        fields = {
            name: ScalarField(
                name=name,
                values=read_array(field_group, name),
                units=read_units(field_group, name),
                grid=grid,
                time=time,
            )
            for name in sorted(field_group)
        }

        params = _read_params(handle[_PARAMS_GROUP])
        distribution = _read_distribution(handle, grid=grid, species=params.species)
        fidelity = Fidelity(str(handle.attrs[_FIDELITY_ATTRIBUTE]))

    state = PlasmaState(
        params=params,
        grid=grid,
        time=time,
        fields=fields,
        ion_distribution=distribution,
        fidelity=fidelity,
    )
    return ArchivedState(state=state, provenance=record)
