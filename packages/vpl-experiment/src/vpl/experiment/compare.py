"""`vpl compare` — a structured diff of two runs (doc 08 §6).

Three questions, asked in the order a reader asks them:

1. **What did I change?** Every manifest key that differs, addressed by its dotted path,
   with both sides. A textual diff of two YAML files answers this badly: it reports a
   reordered block as a change and a semantically identical one as different.
2. **What did it do to the answer?** Every metric that differs, with the absolute and the
   relative change. doc 07 §7 gates metrics against their own history; this is the same
   comparison between two points rather than along one.
3. **Did the science actually change?** One content digest per run, over the artifacts
   with the whole doc 08 §7 provenance block set aside — because two different runs
   necessarily disagree about their commit and their manifest digest, and that says
   nothing about whether the arrays agree.

Provenance differences are reported separately from manifest differences, and
``created_utc`` is excluded from them: two runs are always at different times, and
reporting that as a difference would bury the ones that matter.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

from vpl.core.provenance import ManifestValue
from vpl.core.storage import read_metrics
from vpl.experiment.digest import (
    EXECUTION_ONLY_FIELDS,
    PROVENANCE_ONLY_FIELDS,
    run_content_digest,
)
from vpl.experiment.run.engine import METRICS_FILENAME
from vpl.experiment.run.store import ARTIFACTS_DIRNAME, RunDirectory, RunStore

__all__ = ["MetricDifference", "RunComparison", "ValueDifference", "compare_runs"]

#: Sentinel for "this key is not in that manifest at all". ``None`` cannot serve, because
#: ``None`` is a value a manifest can legitimately hold.
_ABSENT: Final[object] = object()


@dataclass(frozen=True, slots=True)
class ValueDifference:
    """One keyed difference between two documents.

    Attributes:
        path: Dotted address, with list elements indexed — ``plasma.Te.value``,
            ``outputs.artifacts[0]``.
        left: The value in the first run, or ``None`` where the key is absent.
        right: The value in the second run, or ``None`` where the key is absent.
        left_present: Whether ``left`` is a value or an absence. Carried explicitly so
            that "set to null" and "not set" stay distinguishable.
        right_present: The same, for the second run.
    """

    path: str
    left: ManifestValue
    right: ManifestValue
    left_present: bool = True
    right_present: bool = True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "left": self.left,
            "right": self.right,
            "left_present": self.left_present,
            "right_present": self.right_present,
        }

    def render(self) -> str:
        left = repr(self.left) if self.left_present else "<absent>"
        right = repr(self.right) if self.right_present else "<absent>"
        return f"  {self.path}: {left} -> {right}"


@dataclass(frozen=True, slots=True)
class MetricDifference:
    """One metric's movement between two runs — doc 07 §7's comparison, between points.

    Attributes:
        name: The metric.
        left: Its value in the first run, or ``None`` if that run did not measure it.
        right: Its value in the second run, or ``None``.
        units: The units both sides carry, or the one side that has them.
    """

    name: str
    left: float | None
    right: float | None
    units: str

    @property
    def absolute(self) -> float | None:
        if self.left is None or self.right is None:
            return None
        return self.right - self.left

    @property
    def relative(self) -> float | None:
        """``(right - left) / |left|``, or ``None`` where it is not defined.

        ``None`` rather than infinity when the baseline is zero. An infinity in a
        comparison table reads as a computed result; a blank reads as what it is.
        """
        if self.left is None or self.right is None or self.left == 0.0:
            return None
        return (self.right - self.left) / abs(self.left)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "left": self.left,
            "right": self.right,
            "units": self.units,
            "absolute": self.absolute,
            "relative": self.relative,
        }

    def render(self) -> str:
        left = "<absent>" if self.left is None else f"{self.left:.6g}"
        right = "<absent>" if self.right is None else f"{self.right:.6g}"
        change = "" if self.relative is None else f"  ({self.relative:+.3%})"
        return f"  {self.name} [{self.units}]: {left} -> {right}{change}"


@dataclass(frozen=True, slots=True)
class RunComparison:
    """What ``vpl compare`` found."""

    left: str
    right: str
    manifest_differences: tuple[ValueDifference, ...]
    provenance_differences: tuple[ValueDifference, ...]
    metric_differences: tuple[MetricDifference, ...]
    content_identical: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "manifest_differences": [d.to_mapping() for d in self.manifest_differences],
            "provenance_differences": [d.to_mapping() for d in self.provenance_differences],
            "metric_differences": [d.to_mapping() for d in self.metric_differences],
            "content_identical": self.content_identical,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True)

    def render(self) -> str:
        lines = [f"{self.left}  ->  {self.right}", ""]

        for heading, differences in (
            ("manifest", self.manifest_differences),
            ("provenance", self.provenance_differences),
        ):
            lines.append(f"{heading}: {len(differences)} difference(s)")
            lines.extend(difference.render() for difference in differences)
            lines.append("")

        lines.append(f"metrics: {len(self.metric_differences)} difference(s)")
        lines.extend(difference.render() for difference in self.metric_differences)
        lines.append("")
        verdict = "identical" if self.content_identical else "different"
        lines.append(f"artifact content: {verdict}")
        return "\n".join(lines)


def _flatten(value: ManifestValue, prefix: str = "") -> Iterator[tuple[str, ManifestValue]]:
    """Every leaf of a document, addressed by a dotted path.

    Lists are indexed rather than compared whole, so that changing one instrument in a
    four-channel manifest reports that instrument and not the list.
    """
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _flatten(value[key], f"{prefix}.{key}" if prefix else str(key))
        return
    if not isinstance(value, str) and isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
        return
    yield prefix, value


def _diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[ValueDifference, ...]:
    """Keyed differences between two flat mappings."""
    differences: list[ValueDifference] = []
    for path in sorted(set(left) | set(right)):
        first = left.get(path, _ABSENT)
        second = right.get(path, _ABSENT)
        if first == second:
            continue
        differences.append(
            ValueDifference(
                path=path,
                left=None if first is _ABSENT else first,
                right=None if second is _ABSENT else second,
                left_present=first is not _ABSENT,
                right_present=second is not _ABSENT,
            )
        )
    return tuple(differences)


def _metric_values(run: RunDirectory) -> dict[str, tuple[float, str]]:
    path = run.artifacts_path / METRICS_FILENAME
    if not path.is_file():
        return {}
    return {metric.name: (metric.value, metric.units) for metric in read_metrics(path).metrics}


def _metric_diff(
    left: Mapping[str, tuple[float, str]], right: Mapping[str, tuple[float, str]]
) -> tuple[MetricDifference, ...]:
    differences: list[MetricDifference] = []
    for name in sorted(set(left) | set(right)):
        first = left.get(name)
        second = right.get(name)
        if first is not None and second is not None and first == second:
            continue
        units = (first or second or (0.0, ""))[1]
        differences.append(
            MetricDifference(
                name=name,
                left=None if first is None else first[0],
                right=None if second is None else second[0],
                units=units,
            )
        )
    return tuple(differences)


def compare_runs(store: RunStore, left_id: str, right_id: str) -> RunComparison:
    """Compare two archived runs — doc 08 §6's ``vpl compare``.

    Args:
        store: The store holding both runs.
        left_id: The baseline run's identity, or an unambiguous prefix.
        right_id: The other run's.

    Returns:
        The manifest, provenance and metric differences, and whether the stored arrays
        agree.

    Raises:
        RunNotFoundError: If either identity names nothing or is ambiguous.
    """
    left = store.resolve(left_id)
    right = store.resolve(right_id)

    manifest_differences = _diff(
        dict(_flatten(left.read_manifest().as_document())),
        dict(_flatten(right.read_manifest().as_document())),
    )

    # ``created_utc`` is dropped: two runs are always at different times, and reporting
    # that alongside a changed seed would bury the one that matters.
    left_provenance = {
        key: value
        for key, value in left.read_provenance().to_dict().items()
        if key not in EXECUTION_ONLY_FIELDS
    }
    right_provenance = {
        key: value
        for key, value in right.read_provenance().to_dict().items()
        if key not in EXECUTION_ONLY_FIELDS
    }

    left_content = run_content_digest(
        left.path / ARTIFACTS_DIRNAME, excluded=PROVENANCE_ONLY_FIELDS
    )
    right_content = run_content_digest(
        right.path / ARTIFACTS_DIRNAME, excluded=PROVENANCE_ONLY_FIELDS
    )

    return RunComparison(
        left=left.id,
        right=right.id,
        manifest_differences=manifest_differences,
        provenance_differences=_diff(left_provenance, right_provenance),
        metric_differences=_metric_diff(_metric_values(left), _metric_values(right)),
        content_identical=left_content == right_content,
    )
