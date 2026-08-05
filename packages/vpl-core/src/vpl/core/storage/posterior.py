"""Posterior artifacts — doc 08 §7 ("Zarr, chunked, cloud-ready") and ADR-005.

ADR-005 leans to **thinned samples (retaining ESS >= 400 per parameter) + full summary
statistics + the manifest**, with full chains regenerable on demand because runs are
deterministic (doc 13 §2). This module implements exactly that, and its two load-bearing
decisions both come from ADR-005's stated consequences.

**The thinning is the posterior's own.** ADR-005: *"Thinning must be ESS-aware rather than
fixed-stride, or coverage statistics will be biased."* Coverage is gate G-V4 and the
evidence for doc 00 S4 — the claim that the uncertainty is calibrated — so a stride chosen
here for file size would corrupt the one number the project exists to defend. The stride
therefore comes from :meth:`~vpl.core.state.posterior.Posterior.max_safe_stride` and the
thinning from :meth:`~vpl.core.state.posterior.Posterior.thin`, which already refuse to
break the floor. This module grows no second, differently-biased copy of that rule.

**The summaries are computed before the thinning.** ADR-005 keeps full summaries precisely
because thinning loses tail resolution. Summaries recomputed from the retained draws would
reproduce the loss they exist to compensate for — and would agree with the archive closely
enough that nothing would ever flag it.

## Why Zarr, and what is stored where

doc 08 §7 chose Zarr for chains that are "large, appended incrementally". The sample array
is chunked one chain per chunk along the draw axis, so a chain can be written or read
without touching the others.

Non-scalar attributes are JSON strings, the same encoding
:mod:`vpl.core.storage.metadata` uses for HDF5. Zarr attributes are natively JSON and
could hold nested objects directly; storing them the same way as everywhere else means one
decoder for all three of doc 08 §7's formats, and a provenance block that is
character-for-character identical whichever format an artifact was written in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import numpy as np
import zarr
from numpy.typing import NDArray

from vpl.core.provenance import Provenance, Tier
from vpl.core.state.posterior import (
    DEFAULT_CREDIBLE_LEVEL,
    GATE_MIN_ESS,
    CredibleInterval,
    ParameterLevel,
    Posterior,
    SamplerDiagnostics,
)
from vpl.core.storage.metadata import (
    ArtifactKind,
    artifact_attributes,
    check_artifact_kind,
    provenance_from_attributes,
    require_provenance,
)

__all__ = [
    "ArchivedPosterior",
    "PosteriorSummary",
    "read_posterior",
    "write_posterior",
]

_SAMPLES_ARRAY: Final[str] = "samples"
_DERIVED_GROUP: Final[str] = "derived"
_SUMMARY_GROUP: Final[str] = "summary"
_MEAN_GROUP: Final[str] = "mean"
_LOWER_GROUP: Final[str] = "lower"
_UPPER_GROUP: Final[str] = "upper"

_NAMES_ATTRIBUTE: Final[str] = "parameter_names"
_LEVELS_ATTRIBUTE: Final[str] = "parameter_levels"
_TIER_ATTRIBUTE: Final[str] = "tier"
_STRIDE_ATTRIBUTE: Final[str] = "thinning_stride"
_FULL_DRAWS_ATTRIBUTE: Final[str] = "full_n_draws"
_CREDIBLE_LEVEL_ATTRIBUTE: Final[str] = "credible_level"
_THINNED_DIAGNOSTICS_ATTRIBUTE: Final[str] = "thinned_diagnostics"
_FULL_DIAGNOSTICS_ATTRIBUTE: Final[str] = "full_diagnostics"

#: Draws per chunk along the draw axis.
#:
#: One chain per chunk is what makes ADR-005's "appended incrementally" and doc 08 §7's
#: "cloud-ready" true at the same time: a chunk is a contiguous slice of one chain, so a
#: reader that wants chain 2 fetches only chain 2. The draw span is a compromise between
#: object count and per-object size, not a physical quantity.
DRAW_CHUNK: Final[int] = 512


@dataclass(frozen=True, eq=False, slots=True)
class PosteriorSummary:
    """Statistics of the **unthinned** chains — the second half of ADR-005.

    Kept beside the thinned draws because thinning loses tail resolution, and coverage
    (doc 06 §7.1, gate G-V4) is a tail question. A summary recomputed from the retained
    draws would inherit exactly the loss it is there to compensate for.

    Attributes:
        mean: Posterior mean per sampled parameter and per derived quantity, shaped like
            the quantity itself. Read-only.
        credible: Equal-tailed credible interval per quantity, at the level doc 06 §7.1
            states gate G-V4 in. Read-only.
        diagnostics: The sampler's report on the **full** run. The archived posterior
            reports the ESS it actually retains, which is the honest number to gate on;
            this is what still says whether the chains were long enough to begin with.
        n_draws: Draws per chain before thinning, so the discarded fraction is recoverable.
    """

    mean: Mapping[str, NDArray[np.float64]]
    credible: Mapping[str, CredibleInterval]
    diagnostics: SamplerDiagnostics
    n_draws: int

    def __post_init__(self) -> None:
        if set(self.mean) != set(self.credible):
            raise ValueError(
                "every summarised quantity needs both a mean and a credible interval; "
                f"means cover {sorted(self.mean)}, intervals cover {sorted(self.credible)}"
            )
        object.__setattr__(self, "mean", MappingProxyType(dict(self.mean)))
        object.__setattr__(self, "credible", MappingProxyType(dict(self.credible)))

    @property
    def quantities(self) -> tuple[str, ...]:
        """Everything summarised, in a deterministic order."""
        return tuple(sorted(self.mean))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PosteriorSummary):
            return NotImplemented
        return (
            self.diagnostics == other.diagnostics
            and self.n_draws == other.n_draws
            and self.quantities == other.quantities
            and all(
                np.array_equal(self.mean[name], other.mean[name])
                and np.array_equal(self.credible[name].lower, other.credible[name].lower)
                and np.array_equal(self.credible[name].upper, other.credible[name].upper)
                and self.credible[name].level == other.credible[name].level
                for name in self.quantities
            )
        )

    def __repr__(self) -> str:
        return (
            f"PosteriorSummary(n_quantities={len(self.quantities)}, "
            f"n_draws={self.n_draws}, min_ess={self.diagnostics.worst_ess:.4g})"
        )


class ArchivedPosterior(NamedTuple):
    """What ADR-005 puts on disk: thinned draws, full summaries and the provenance.

    Attributes:
        posterior: The retained draws, reporting the ESS they actually have.
        summary: Statistics of the full chains, computed before thinning.
        stride: The ESS-aware stride that was applied. ``1`` means nothing was discarded.
        provenance: The doc 08 §7 record the artifact was written with.
    """

    posterior: Posterior
    summary: PosteriorSummary
    stride: int
    provenance: Provenance


def _diagnostics_payload(diagnostics: SamplerDiagnostics) -> dict[str, Any]:
    return {
        "r_hat": dict(diagnostics.r_hat),
        "ess": dict(diagnostics.ess),
        "divergences": diagnostics.divergences,
        "e_bfmi": list(diagnostics.e_bfmi),
    }


def _diagnostics_from_payload(payload: Mapping[str, Any]) -> SamplerDiagnostics:
    return SamplerDiagnostics(
        r_hat={str(name): float(value) for name, value in payload["r_hat"].items()},
        ess={str(name): float(value) for name, value in payload["ess"].items()},
        divergences=int(payload["divergences"]),
        e_bfmi=tuple(float(value) for value in payload["e_bfmi"]),
    )


def _summarise(posterior: Posterior, *, level: float) -> PosteriorSummary:
    """Mean and credible interval for every sampled parameter and derived quantity."""
    quantities = (*posterior.names, *posterior.derived_names)
    return PosteriorSummary(
        mean={name: posterior.mean(name) for name in quantities},
        credible={name: posterior.credible_interval(name, level) for name in quantities},
        diagnostics=posterior.diagnostics,
        n_draws=posterior.n_draws,
    )


def _subgroup(group: zarr.Group, name: str) -> zarr.Group:
    """A child group, or a refusal.

    Zarr's ``__getitem__`` cannot say whether a member is a group or an array, so the
    check happens here rather than as an unchecked cast. A store whose ``summary`` is an
    array is a store this package did not write, and guessing at its layout is how a
    plausible but wrong number reaches a figure.
    """
    member = group[name]
    if not isinstance(member, zarr.Group):
        raise ValueError(f"{name!r} is not a group in this posterior store")
    return member


def _member_array(group: zarr.Group, name: str) -> zarr.Array[Any]:
    """A child array, or a refusal. See :func:`_subgroup`."""
    member = group[name]
    if not isinstance(member, zarr.Array):
        raise ValueError(f"{name!r} is not an array in this posterior store")
    return member


def _write_array(group: zarr.Group, name: str, values: NDArray[np.float64]) -> None:
    """Store an array whole. Summaries are small; only the draws are worth chunking."""
    array = group.create_array(name, shape=values.shape, chunks=values.shape, dtype="float64")
    array[...] = values


def _read_array(group: zarr.Group, name: str) -> NDArray[np.float64]:
    return np.asarray(_member_array(group, name)[...], dtype=np.float64)


def _write_summary(root: zarr.Group, summary: PosteriorSummary) -> None:
    group = root.create_group(_SUMMARY_GROUP)
    means = group.create_group(_MEAN_GROUP)
    lowers = group.create_group(_LOWER_GROUP)
    uppers = group.create_group(_UPPER_GROUP)

    for name in summary.quantities:
        _write_array(means, name, summary.mean[name])
        _write_array(lowers, name, summary.credible[name].lower)
        _write_array(uppers, name, summary.credible[name].upper)


def _read_summary(root: zarr.Group, *, level: float, n_draws: int) -> PosteriorSummary:
    group = _subgroup(root, _SUMMARY_GROUP)
    means = _subgroup(group, _MEAN_GROUP)
    lowers = _subgroup(group, _LOWER_GROUP)
    uppers = _subgroup(group, _UPPER_GROUP)
    names = sorted(means.array_keys())

    return PosteriorSummary(
        mean={name: _read_array(means, name) for name in names},
        credible={
            name: CredibleInterval(
                lower=_read_array(lowers, name),
                upper=_read_array(uppers, name),
                level=level,
            )
            for name in names
        },
        diagnostics=_diagnostics_from_payload(
            json.loads(str(root.attrs[_FULL_DIAGNOSTICS_ATTRIBUTE]))
        ),
        n_draws=n_draws,
    )


def write_posterior(
    path: Path,
    posterior: Posterior,
    *,
    provenance: Provenance,
    min_ess: float = GATE_MIN_ESS,
    credible_level: float = DEFAULT_CREDIBLE_LEVEL,
) -> Path:
    """Archive a posterior as ADR-005 describes: thinned draws plus full summaries.

    Args:
        path: Destination store. Created as a directory; overwritten if it exists.
        posterior: The full chains. Not modified.
        provenance: The doc 08 §7 record. Required.
        min_ess: Floor the retained effective sample size must stay above. Defaults to
            gate G-V3's value (doc 07 §6). Lowering it thins harder and is not a loophole:
            the archived posterior then reports the reduced ESS and fails its own G-V3
            check, so the cost travels with the artifact.
        credible_level: Level the archived intervals are computed at. Defaults to the 95 %
            gate G-V4 is stated at.

    Returns:
        The path written.

    Raises:
        MissingProvenanceError: If ``provenance`` is absent.
        ValueError: If the posterior's ESS is already at or below ``min_ess``, in which
            case no stride can produce an archivable result and the chains need
            lengthening (doc 07 §6) rather than thinning.
    """
    record = require_provenance(provenance)

    # Both raise on a posterior that never cleared the floor. Nothing is created on disk
    # first, so a refused posterior leaves no store for a later read to misinterpret.
    stride = posterior.max_safe_stride(min_ess=min_ess)
    thinned = posterior.thin(stride, min_ess=min_ess)
    summary = _summarise(posterior, level=credible_level)

    root = zarr.open_group(store=path, mode="w")
    root.attrs.update(artifact_attributes(ArtifactKind.POSTERIOR, record))
    root.attrs.update(
        {
            _NAMES_ATTRIBUTE: json.dumps(list(posterior.names)),
            _LEVELS_ATTRIBUTE: json.dumps(
                {name: posterior.levels[name].value for name in posterior.names}
            ),
            _TIER_ATTRIBUTE: posterior.tier.value,
            _STRIDE_ATTRIBUTE: stride,
            _FULL_DRAWS_ATTRIBUTE: posterior.n_draws,
            _CREDIBLE_LEVEL_ATTRIBUTE: credible_level,
            _FULL_DIAGNOSTICS_ATTRIBUTE: json.dumps(_diagnostics_payload(posterior.diagnostics)),
            _THINNED_DIAGNOSTICS_ATTRIBUTE: json.dumps(_diagnostics_payload(thinned.diagnostics)),
        }
    )

    samples = root.create_array(
        _SAMPLES_ARRAY,
        shape=thinned.samples.shape,
        chunks=(1, min(thinned.n_draws, DRAW_CHUNK), thinned.n_params),
        dtype="float64",
    )
    samples[...] = thinned.samples

    derived = root.create_group(_DERIVED_GROUP)
    for name in thinned.derived_names:
        _write_array(derived, name, thinned.derived_for(name))

    _write_summary(root, summary)

    return path


def read_posterior(path: Path) -> ArchivedPosterior:
    """Read back an archive written by :func:`write_posterior`.

    The posterior returned is the **thinned** one, reporting the effective sample size it
    actually retains — so a gate G-V3 check on an archived run cannot be answered with
    pre-thinning numbers. The full-chain figures are in :attr:`ArchivedPosterior.summary`.

    Raises:
        MissingProvenanceError: If the doc 08 §7 provenance block is incomplete.
        ValueError: If the store holds a different artifact, or a posterior the model
            rejects.
    """
    root = zarr.open_group(store=path, mode="r")
    attributes = dict(root.attrs)
    check_artifact_kind(attributes, ArtifactKind.POSTERIOR)
    provenance = provenance_from_attributes(attributes)

    names = tuple(json.loads(str(attributes[_NAMES_ATTRIBUTE])))
    levels = {
        str(name): ParameterLevel(value)
        for name, value in json.loads(str(attributes[_LEVELS_ATTRIBUTE])).items()
    }
    derived_group = _subgroup(root, _DERIVED_GROUP)

    posterior = Posterior(
        samples=_read_array(root, _SAMPLES_ARRAY),
        names=names,
        levels=levels,
        tier=Tier(str(attributes[_TIER_ATTRIBUTE])),
        diagnostics=_diagnostics_from_payload(
            json.loads(str(attributes[_THINNED_DIAGNOSTICS_ATTRIBUTE]))
        ),
        derived={
            name: _read_array(derived_group, name) for name in sorted(derived_group.array_keys())
        },
    )

    return ArchivedPosterior(
        posterior=posterior,
        summary=_read_summary(
            root,
            level=float(attributes[_CREDIBLE_LEVEL_ATTRIBUTE]),  # type: ignore[arg-type]
            n_draws=int(attributes[_FULL_DRAWS_ATTRIBUTE]),  # type: ignore[arg-type]
        ),
        stride=int(attributes[_STRIDE_ATTRIBUTE]),  # type: ignore[arg-type]
        provenance=provenance,
    )
