"""The literal-lint rule — doc 08 §5, doc 00 C1.

    A lint rule fails CI on any numeric literal in physics code outside a registry file
    or a named constant.

Without this the parameter registry is a convention rather than a control. doc 00 §6
names the failure it prevents — the parameter-fog trap, "numbers appear in code with no
source" — and a registry that physics code is free to bypass does not prevent it. It only
provides a nicer home for the numbers somebody remembered to move.

## What counts as a physical magnitude

The rule has to separate *measurement* from *algebra*, because doc 03 §2 is full of both:

```
J_i  =  (4/9) ε₀ √(2e/m_i) · V_w^{3/2} / s²        ← 4/9, 2, 3/2 are algebra
n_s  =  0.61 · n_0                                  ← 0.61 is a measurement
```

The discriminator used here is representability as a ratio of small integers. ``4/9``,
``0.5``, ``1.5``, ``9/8`` and ``5/2`` are structure: they carry no units, they have no
uncertainty, and there is nowhere to source them from because they are not sourced from
anywhere. ``0.61``, ``5e-19`` and ``39.948`` are not expressible that way, and every one
of them is a number somebody measured.

Integers are always allowed. Array indices, exponents, dimension counts and loop bounds
are structure, and flagging them would generate enough noise that the rule would be
switched off — which is the only way a check like this really fails.

## The escape hatch, and why it is the right one

doc 08 §5 permits a **named constant**, and this implementation permits exactly that: a
module-level assignment to an ``UPPER_SNAKE_CASE`` name. That is the correct escape
because it makes the value greppable, documentable and reviewable in one place. It is
deliberately *not* a magic comment: a ``# noqa``-style suppression can be applied where
the number sits, which is precisely where it should not stay.

Note that the framework's own practice is stricter than the rule. ``vpl.physics.analytic``
does not merely name its constants, it reads them from the registry, so the two cannot
diverge. The lint is the floor.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

__all__ = [
    "MAX_STRUCTURAL_TERM",
    "LiteralFinding",
    "check_source",
    "check_tree",
    "main",
]

#: Largest numerator and denominator a float literal may have, in lowest terms, and still
#: count as algebra rather than measurement.
#:
#: Sixteen covers every coefficient printed in doc 03 §2 — 1/2, 2/3, 3/4, 4/9, 9/8, 3/2,
#: 5/2, 5/3 — and excludes 0.61 (61/100), which is the value the rule most needs to catch.
#: Raising it towards 100 would readmit exactly the class of two-decimal empirical
#: constant this exists to find.
#:
#: The **numerator** bound matters as much as the denominator, and for a reason that is
#: not obvious: ``1e17`` is a float whose denominator is 1. Bounding only the denominator
#: would classify the reference density of doc 01 §2.1 as algebra and wave it straight
#: through — the single most important number in the project, missed by the rule written
#: to catch it.
MAX_STRUCTURAL_TERM: Final[int] = 16

#: Directory names whose contents are exempt.
_EXEMPT_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        # doc 08 §5 exempts "a registry file" by name; that is where the numbers live.
        "data",
        # A test asserting Gamma_E == 6.577 kW/m^2 must contain the number it checks.
        # Routing it through the registry would have the test and the code read the same
        # value from the same place, where they could agree while both were wrong.
        "tests",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }
)


@dataclass(frozen=True, slots=True)
class LiteralFinding:
    """One numeric literal that should have come from the registry."""

    path: Path
    line: int
    column: int
    value: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.column}: numeric literal {self.value} in "
            f"physics code. doc 08 §5 requires it to come from the parameter registry, "
            f"or to be a named module-level constant."
        )


def _is_structural(value: float) -> bool:
    """Whether a float literal is algebra rather than a measurement.

    Integers are handled separately and unconditionally allowed: an integer cannot carry
    a decimal measurement, and indices, exponents and dimension counts are structure.
    Flagging them would generate enough noise that the rule would be switched off, which
    is the only way a check like this really fails.
    """
    if not math.isfinite(value):
        return False
    ratio = Fraction(value).limit_denominator(MAX_STRUCTURAL_TERM)
    if ratio != Fraction(value):
        return False
    return abs(ratio.numerator) <= MAX_STRUCTURAL_TERM


def _is_named_constant_target(node: ast.AST) -> bool:
    """Whether an assignment binds an ``UPPER_SNAKE_CASE`` name."""
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    names = [t.id for t in targets if isinstance(t, ast.Name)]
    return bool(names) and all(n.isupper() or n.upper() == n for n in names)


class _LiteralVisitor(ast.NodeVisitor):
    """Walks a module, collecting numeric literals that are neither algebra nor named."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[LiteralFinding] = []
        self._depth = 0

    # Module-level assignments to an UPPER_SNAKE name are the doc 08 §5 escape. The whole
    # right-hand side is exempt, including composite forms such as
    # `SWEEP_RANGE = (0.61, 0.86)` and `COEFF = 4.0 / 9.0 * 0.61`, because the value is
    # named and reviewable wherever it is assembled.
    def visit_Assign(self, node: ast.Assign) -> None:
        if self._depth == 0 and _is_named_constant_target(node):
            return
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._depth == 0 and _is_named_constant_target(node):
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._nested(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._nested(node)

    def _nested(self, node: ast.AST) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_Constant(self, node: ast.Constant) -> None:
        # bool is a subclass of int and must be excluded before the numeric check, or
        # `True` is read as the integer 1.
        if isinstance(node.value, bool) or not isinstance(node.value, float | int):
            return
        if isinstance(node.value, int) or _is_structural(node.value):
            return
        self.findings.append(
            LiteralFinding(
                path=self.path,
                line=node.lineno,
                column=node.col_offset + 1,
                value=repr(node.value),
            )
        )


def check_source(source: str, *, path: Path) -> list[LiteralFinding]:
    """Findings for one module's source text.

    Raises:
        SyntaxError: If ``source`` cannot be parsed. A file the checker cannot read is a
            file it cannot vouch for, and skipping it silently would let an unparseable
            module carry any number at all.
    """
    visitor = _LiteralVisitor(path)
    visitor.visit(ast.parse(source, filename=str(path)))
    return visitor.findings


def _python_sources(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("."):
            # AppleDouble sidecars and editor swap files are not source. See the note in
            # vpl.core.params.catalogue: this was found on the reference machine, not here.
            continue
        if _EXEMPT_DIRECTORIES.isdisjoint(path.relative_to(root).parts):
            yield path


def check_tree(root: Path) -> list[LiteralFinding]:
    """Findings across every non-exempt Python source under ``root``.

    Paths in the findings are relative to ``root``, so a diagnostic is the same whichever
    machine produced it — which matters because these end up in CI output that a reader
    has to match against their own checkout.
    """
    findings: list[LiteralFinding] = []
    for path in _python_sources(root):
        relative = path.relative_to(root)
        findings.extend(check_source(path.read_text(encoding="utf-8"), path=relative))
    return findings


# ── the CI gate ─────────────────────────────────────────────────────────────────


def _report_assumed_count(baseline_path: Path) -> int:
    """Print the doc 00 C1 metric and return a process exit code.

    doc 00 C1 says the ASSUMED count "is a tracked project metric and is reported in
    CI", and doc 09 §1 sets its target to zero. Both halves matter: the count is printed
    on *every* run, not only when it fails, because a metric you only see when it breaks
    is a metric nobody is tracking.

    An absent baseline is read as zero rather than as "no opinion". doc 09 §1's target is
    zero, so the strict reading is also the correct default; the alternative would let a
    deleted baseline file silently disable the gate.
    """
    from vpl.core.params import default_registry

    registry = default_registry()
    counts = registry.count_by_class()
    current = registry.assumed_count

    print("parameter registry:")
    for provenance_class, count in sorted(counts.items(), key=lambda kv: kv[0].value):
        print(f"  {provenance_class.value:<10} {count}")

    baseline = 0
    if baseline_path.is_file():
        baseline = int(json.loads(baseline_path.read_text(encoding="utf-8"))["assumed_count"])

    if current > baseline:
        print(
            f"\nFAIL: ASSUMED count {current} exceeds the committed baseline {baseline}. "
            f"doc 00 C1 makes every ASSUMED value a defect and its increase a build "
            f"failure. Source the value, or raise the baseline deliberately in "
            f"{baseline_path}."
        )
        for defect in registry.defects():
            print(f"  {defect.id} ({defect.category}) — {defect.description}")
        return 1

    if current < baseline:
        print(
            f"\nASSUMED count {current} is below the baseline {baseline}. Lower "
            f"{baseline_path} to {current} to lock the improvement in."
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the doc 08 §5 literal rule and report the doc 00 C1 metric."""
    parser = argparse.ArgumentParser(
        prog="vpl-lint",
        description=(
            "Fail on numeric literals in physics code that should have come from the "
            "parameter registry (doc 08 §5), and report the ASSUMED-class count "
            "(doc 00 C1)."
        ),
    )
    parser.add_argument("roots", nargs="+", type=Path, help="package roots to scan")
    parser.add_argument(
        "--assumed-baseline",
        type=Path,
        default=Path("metrics/assumed-baseline.json"),
        help="committed baseline for the ASSUMED count; absent is read as zero",
    )
    args = parser.parse_args(argv)

    findings: list[LiteralFinding] = []
    for root in args.roots:
        findings.extend(check_tree(root))

    for finding in findings:
        print(finding)
    if findings:
        print(f"\nFAIL: {len(findings)} unregistered numeric literal(s).")

    metric_code = _report_assumed_count(args.assumed_baseline)
    return 1 if findings else metric_code


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    raise SystemExit(main())
