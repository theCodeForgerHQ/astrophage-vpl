"""Loading a manifest file — doc 08 §6, doc 08 §2.

The substrate is **OmegaConf, with Hydra reserved for the doc 10 §6 sweep layer**. That is
a deviation from doc 08 §2's "Configuration | Buy | Hydra + OmegaConf" and is recorded in
``docs/adr/ADR-008-manifest-substrate.md``. The short form: doc 08 §6's own usage is
``vpl run experiments/b02.yaml`` — a path to one self-contained file — and Hydra's model is
config-group composition rooted at a search path, with an opinionated working directory and
its own ``outputs/`` run tree. Two run-directory managers in one framework is a conflict
doc 13 §5 does not need, and none of what Hydra adds is used by "one file, one experiment".

What OmegaConf earns on its own is not decoration:

- **It reads doc 08 §6's own numbers.** PyYAML implements YAML 1.1, whose float resolver
  requires a signed exponent, so ``n0: {value: 1.0e17}`` — written exactly that way in
  doc 08 §6 — loads as the *string* ``"1.0e17"``. OmegaConf resolves it as a float. A
  manifest language that could not read its own specification's example would be a poor
  one, and the alternative was to rewrite the document.
- **Interpolation**, so a manifest can state a value once and reference it.
- **Dotted overrides**, which is the seam ``vpl sweep`` will drive (doc 10 §6). An override
  changes the document and therefore the digest, which is what keeps a swept case as
  reproducible as a hand-written one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from vpl.core.provenance import ManifestValue
from vpl.experiment.manifest.schema import Manifest, manifest_from_document

__all__ = ["load_manifest"]


def _checked_overrides(overrides: Sequence[str]) -> list[str]:
    """Reject an override that is not a ``key=value`` pair.

    OmegaConf's own message for this names neither the offending item nor the command that
    supplied it, and a mistyped sweep argument that silently did nothing would produce a
    campaign of identical cases.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"override {item!r} is not a key=value pair. Overrides are dotted paths "
                "into the manifest, e.g. experiment.seed=7 or forward.n_ppc=2000."
            )
    return list(overrides)


def load_manifest(path: Path, *, overrides: Sequence[str] = ()) -> Manifest:
    """Load, override, resolve and validate a manifest file.

    Args:
        path: The manifest. doc 08 §6 takes a direct path, not a config-group name.
        overrides: Dotted ``key=value`` pairs merged over the file. They become part of
            the manifest and therefore part of its digest — a swept case is a different
            experiment, and doc 00 E3 requires it to be identified as one.

    Returns:
        The validated manifest, carrying the merged document it was built from.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
        ValueError: If the document is not a mapping, an override is malformed, or the
            manifest fails validation.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no manifest at {path}")

    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise ValueError(
            f"{path.name}: a manifest is a mapping of blocks (experiment, plasma, "
            f"forward, ...), got {type(loaded).__name__}"
        )

    merged: DictConfig = loaded
    checked = _checked_overrides(overrides)
    if checked:
        combined = OmegaConf.merge(loaded, OmegaConf.from_dotlist(checked))
        if not isinstance(combined, DictConfig):  # pragma: no cover - merge keeps the kind
            raise ValueError(f"{path.name}: overrides produced something that is not a manifest")
        merged = combined

    # ``resolve=True`` collapses interpolations, so the archived document is the one that
    # ran rather than one that would have to be re-resolved to be understood.
    # ``throw_on_missing`` refuses a `???` placeholder: doc 08 §6 has no notion of a
    # manifest that is only partly specified, and a missing value reaching a solver as
    # OmegaConf's sentinel would be a run configured by a placeholder.
    container = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):  # pragma: no cover - DictConfig yields a dict
        raise ValueError(f"{path.name}: a manifest is a mapping, got {type(container).__name__}")

    document: dict[str, ManifestValue] = {str(key): value for key, value in container.items()}
    return manifest_from_document(document)
