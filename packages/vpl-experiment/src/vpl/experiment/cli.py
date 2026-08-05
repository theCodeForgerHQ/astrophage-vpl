"""The `vpl` command line — doc 08 §6, verbatim.

    vpl run experiments/b02-reference-operating-point.yaml
    vpl reproduce <run-id>          # re-executes bit-for-bit from the archived manifest
    vpl compare <run-id-a> <run-id-b>

``argparse``, not a CLI framework: three commands with a handful of options each is not a
dependency's worth of problem, and :mod:`vpl.core.lint` already ships a console script the
same way.

## Errors are reported, not raised

A manifest with a typo, a run identity that does not exist and a stage that is not built
yet are all *user-facing* outcomes. Each prints its message and exits non-zero; the
traceback stays out of the way. Anything else propagates, because an unexpected exception
is a bug and a bug's traceback is the useful part.

**Exit codes.** ``0`` success; ``1`` a failure the message explains — including a
reproduction that came out different, which is a result rather than an error; ``2`` a
usage error, which is argparse's own convention.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from vpl.core.registry import PluginError
from vpl.experiment.compare import compare_runs
from vpl.experiment.digest import UnknownArtifactFormatError
from vpl.experiment.manifest import load_manifest
from vpl.experiment.run import RunNotFoundError, RunStore, StageNotImplementedError, execute
from vpl.experiment.run.reproduce import reproduce

__all__ = ["DEFAULT_STORE", "STORE_VARIABLE", "main"]

#: Where runs go when nothing says otherwise.
DEFAULT_STORE: Final[Path] = Path("runs")

#: Environment variable overriding :data:`DEFAULT_STORE`.
STORE_VARIABLE: Final[str] = "VPL_RUN_STORE"

_EXIT_OK: Final[int] = 0
_EXIT_FAILED: Final[int] = 1
_EXIT_USAGE: Final[int] = 2

#: The failures that are outcomes rather than bugs. Each already carries a message
#: written for the person who caused it, so the handler prints it and nothing else.
_REPORTED: Final[tuple[type[Exception], ...]] = (
    FileExistsError,
    FileNotFoundError,
    PluginError,
    RunNotFoundError,
    StageNotImplementedError,
    TypeError,
    UnknownArtifactFormatError,
    ValueError,
)


def _store(namespace: argparse.Namespace) -> RunStore:
    if namespace.store is not None:
        return RunStore(Path(namespace.store))
    return RunStore(Path(os.environ.get(STORE_VARIABLE, DEFAULT_STORE)))


def _add_store_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store",
        default=None,
        help=f"run store directory (default: ${STORE_VARIABLE} or ./{DEFAULT_STORE})",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vpl",
        description=(
            "Astrophage VPL — run an experiment manifest, reproduce an archived run, or "
            "compare two runs (doc 08 §6)."
        ),
    )
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="execute a manifest")
    run.add_argument("manifest", type=Path, help="path to the manifest (doc 08 §6)")
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override merged over the manifest; changes its digest",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing run of the same manifest",
    )
    _add_store_option(run)

    reproduce_command = commands.add_parser(
        "reproduce", help="re-execute an archived run and verify it (gate G-1.3)"
    )
    reproduce_command.add_argument("run_id", help="run identity, or an unambiguous prefix")
    _add_store_option(reproduce_command)

    compare = commands.add_parser("compare", help="diff two runs")
    compare.add_argument("left", help="baseline run identity")
    compare.add_argument("right", help="the other run identity")
    compare.add_argument("--json", action="store_true", help="emit machine-readable output")
    _add_store_option(compare)

    return parser


def _run(namespace: argparse.Namespace) -> int:
    manifest = load_manifest(Path(namespace.manifest), overrides=tuple(namespace.overrides))
    run = execute(manifest, store=_store(namespace), force=namespace.force)
    record = run.read_record()

    print(run.id)
    print(f"  tier         {record.tier.value}")
    print(f"  manifest     {record.manifest_sha256}")
    print(f"  commit       {record.git_commit}{' (dirty)' if record.git_dirty else ''}")
    print(f"  artifacts    {', '.join(sorted(record.artifacts)) or 'none'}")
    print(f"  directory    {run.path}")
    return _EXIT_OK


def _reproduce(namespace: argparse.Namespace) -> int:
    result = reproduce(_store(namespace), namespace.run_id)
    print(result.summary())
    for name, (left, right) in sorted(result.artifact_digests.items()):
        mark = "==" if left == right else "!="
        print(f"  {name}: {left} {mark} {right}")
    return _EXIT_OK if result.is_identical else _EXIT_FAILED


def _compare(namespace: argparse.Namespace) -> int:
    comparison = compare_runs(_store(namespace), namespace.left, namespace.right)
    print(comparison.to_json() if namespace.json else comparison.render())
    return _EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a ``vpl`` invocation.

    Returns a process exit code rather than calling :func:`sys.exit`, so that doc 08 §11's
    interactive backend can call exactly the same operations without a command line
    around them.
    """
    parser = _parser()
    namespace = parser.parse_args(argv)

    handlers = {"run": _run, "reproduce": _reproduce, "compare": _compare}
    handler = handlers.get(namespace.command)
    if handler is None:
        parser.print_usage(file=sys.stderr)
        print("vpl: choose one of: run, reproduce, compare", file=sys.stderr)
        return _EXIT_USAGE

    try:
        return handler(namespace)
    except _REPORTED as exc:
        print(f"vpl {namespace.command}: {exc}", file=sys.stderr)
        return _EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    raise SystemExit(main())
