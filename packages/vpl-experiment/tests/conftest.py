"""Shared fixtures for the manifest engine tests.

Everything shared lives here rather than in an importable helper module: the test
directory is not a package, and a cross-module import between test files works only by
accident of ``sys.path``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf

from vpl.experiment import Manifest, RunStore, load_manifest

#: The manifests that ship with the package. They are the fixtures deliberately: a test
#: that used its own private copy would keep passing while the shipped example rotted.
_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

_RUNNABLE = _EXAMPLES / "l0-child-langmuir-rp1.yaml"
_DOCUMENTED = _EXAMPLES / "b02-reference-operating-point.yaml"


@pytest.fixture
def runnable_path() -> Path:
    """The L0 example — the smallest manifest that runs today."""
    return _RUNNABLE


@pytest.fixture
def documented_path() -> Path:
    """doc 08 §6's own manifest, reproduced verbatim."""
    return _DOCUMENTED


def _document(path: Path) -> dict[str, Any]:
    """Load through OmegaConf, as the engine does.

    Not through ``yaml.safe_load``: PyYAML implements YAML 1.1, whose float resolver
    requires a signed exponent, so it reads doc 08 §6's own ``1.0e17`` as the *string*
    ``"1.0e17"``. A fixture that loaded the shipped example differently from the loader
    under test would be testing the fixture. See ADR-008.
    """
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(loaded, dict)
    return dict(loaded)


@pytest.fixture
def runnable_document() -> dict[str, Any]:
    """The L0 example as a plain structure, for tests that mutate one key."""
    return _document(_RUNNABLE)


@pytest.fixture
def documented_document() -> dict[str, Any]:
    return _document(_DOCUMENTED)


@pytest.fixture
def runnable_manifest() -> Manifest:
    return load_manifest(_RUNNABLE)


@pytest.fixture
def documented_manifest() -> Manifest:
    return load_manifest(_DOCUMENTED)


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


@pytest.fixture
def write_manifest() -> Callable[[Path, Mapping[str, Any]], Path]:
    """Write a manifest document to a path and return it."""

    def _write(path: Path, document: Mapping[str, Any]) -> Path:
        path.write_text(yaml.safe_dump(dict(document), sort_keys=False), encoding="utf-8")
        return path

    return _write
