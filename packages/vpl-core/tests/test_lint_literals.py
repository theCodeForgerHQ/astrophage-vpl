"""The literal-lint rule — doc 08 §5, doc 00 C1.

    A lint rule fails CI on any numeric literal in physics code outside a registry file
    or a named constant. The ASSUMED-class count is emitted as a CI metric and its
    increase fails the build.

Without this, the parameter registry is a convention. doc 00 §6 names the failure mode:
the parameter-fog trap, where numbers appear in code with no source. A registry that
physics code is free to bypass does not prevent it; it only provides a nicer place for
the numbers someone remembered to move.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vpl.core.lint import LiteralFinding, check_source, check_tree, main


def _find(source: str, *, path: str = "sheath.py") -> list[LiteralFinding]:
    return check_source(source, path=Path(path))


class TestWhatIsFlagged:
    def test_flags_a_physical_magnitude_inside_a_function(self) -> None:
        findings = _find("def f(n):\n    return n * 0.61\n")

        assert len(findings) == 1
        assert findings[0].value == "0.61"
        assert findings[0].line == 2

    def test_flags_a_float_whose_value_happens_to_be_an_integer(self) -> None:
        # The regression that motivated the numerator bound. 1e17 is a float with
        # denominator 1, so a denominator-only rule classifies the reference density of
        # doc 01 §2.1 as algebra and passes it — the most important number in the
        # project, missed by the check written to catch it.
        findings = _find("def f():\n    return 1e17\n")

        assert [f.value for f in findings] == ["1e+17"]

    def test_flags_a_large_round_float(self) -> None:
        assert len(_find("def f():\n    return 6600.0\n")) == 1

    def test_flags_a_value_in_scientific_notation(self) -> None:
        # 5e-19 m^2 is a cross section. There is no arrangement of small integers that
        # produces it, which is exactly what makes it a measurement rather than algebra.
        findings = _find("def f():\n    return 5e-19\n")

        assert [f.value for f in findings] == ["5e-19"]

    def test_flags_a_lowercase_module_level_assignment(self) -> None:
        # doc 08 §5 permits a *named constant*. A lowercase module-level binding is a
        # variable that happens to be at module scope, and treating it as a constant
        # would make the rule trivially evadable.
        findings = _find("sigma_cx = 5.0e-19\n")

        assert len(findings) == 1

    def test_flags_a_literal_buried_in_a_keyword_argument(self) -> None:
        findings = _find("def f(solver):\n    return solver.run(h_l=0.61)\n")

        assert [f.value for f in findings] == ["0.61"]

    def test_flags_each_offending_literal_separately(self) -> None:
        findings = _find("def f():\n    return 0.61 * 39.948\n")

        assert {f.value for f in findings} == {"0.61", "39.948"}

    def test_reports_file_line_and_column(self) -> None:
        findings = _find("def f():\n    return 0.61\n", path="packages/x/sheath.py")

        assert findings[0].path == Path("packages/x/sheath.py")
        assert findings[0].line == 2
        assert findings[0].column > 0


class TestWhatIsAllowed:
    def test_allows_integers(self) -> None:
        # Array indices, exponents, dimension counts and loop bounds are structure, not
        # measurement. Flagging them would produce so much noise that the rule would be
        # switched off, which is the only way this check actually fails.
        assert _find("def f(a):\n    return a[3] ** 2 + len(a) * 100\n") == []

    def test_allows_simple_rational_coefficients(self) -> None:
        # 4/9, sqrt(2)/3, 9/8, 5/2 are the coefficients printed in doc 03 §2. They are
        # algebra, they carry no units, and there is nowhere to source them from because
        # they are not measurements.
        assert _find("def f(x):\n    return 0.5 * x + 0.75 * x + 1.5 * x + 1.125 * x\n") == []

    def test_allows_a_named_module_level_constant(self) -> None:
        # This is the escape doc 08 §5 names, and it is the right one: the value is now
        # greppable, documentable and reviewable in one place.
        assert _find("EDGE_TO_CENTRE_RATIO = 0.61\n") == []

    def test_allows_an_annotated_named_constant(self) -> None:
        assert _find("SIGMA_CX: Final[float] = 5.0e-19\n") == []

    def test_allows_a_named_constant_built_from_an_expression(self) -> None:
        assert _find("SHEATH_COEFFICIENT = 4.0 / 9.0 * 0.61\n") == []

    def test_allows_a_tuple_of_named_constants(self) -> None:
        assert _find("SWEEP_RANGE = (0.61, 0.86)\n") == []

    def test_allows_zero_and_one_anywhere(self) -> None:
        assert _find("def f(x):\n    return 0.0 if x else 1.0\n") == []

    def test_allows_negative_rationals(self) -> None:
        assert _find("def f(x):\n    return -0.5 * x\n") == []

    def test_does_not_flag_string_or_boolean_literals(self) -> None:
        assert _find("def f():\n    return ('argon', True, None)\n") == []


class TestScope:
    def test_skips_test_files(self, tmp_path: Path) -> None:
        # A test asserting Gamma_E == 6.577 kW/m^2 must contain the number it is
        # checking. Routing that through the registry would mean the test and the code
        # read the same value from the same place and could agree while both are wrong.
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_sheath.py").write_text("def test_x():\n    assert 0.61\n")

        assert check_tree(tmp_path) == []

    def test_skips_the_registry_data_itself(self, tmp_path: Path) -> None:
        # doc 08 §5 exempts "a registry file" by name — that is where the numbers live.
        data = tmp_path / "src" / "vpl" / "core" / "params" / "data"
        data.mkdir(parents=True)
        (data / "physics.yaml").write_text("- id: x\n  value: 0.61\n")

        assert check_tree(tmp_path) == []

    def test_scans_python_sources_under_the_given_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "vpl" / "physics"
        src.mkdir(parents=True)
        (src / "sheath.py").write_text("def f():\n    return 0.61\n")

        findings = check_tree(tmp_path)

        assert len(findings) == 1
        assert findings[0].path.name == "sheath.py"

    def test_reports_paths_relative_to_the_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def f():\n    return 0.61\n")

        assert check_tree(tmp_path)[0].path == Path("src/a.py")

    def test_a_syntax_error_is_reported_rather_than_swallowed(self, tmp_path: Path) -> None:
        # A file the checker cannot parse is a file it cannot vouch for. Skipping it
        # silently would let an unparseable module carry any number at all.
        (tmp_path / "broken.py").write_text("def f(:\n")

        with pytest.raises(SyntaxError):
            check_tree(tmp_path)


class TestTheRuleAgainstTheRealCodebase:
    def test_the_shipped_physics_package_is_clean(self) -> None:
        # The rule is worthless if it has never been run against the code it governs.
        root = Path(__file__).resolve().parents[2] / "vpl-physics"

        findings = check_tree(root)

        assert findings == [], "\n".join(str(f) for f in findings)

    def test_the_core_package_is_clean(self) -> None:
        root = Path(__file__).resolve().parents[1]

        findings = check_tree(root)

        assert findings == [], "\n".join(str(f) for f in findings)


class TestFindingFormatting:
    def test_a_finding_prints_as_a_locatable_diagnostic(self) -> None:
        finding = _find("def f():\n    return 0.61\n", path="a/b.py")[0]

        rendered = str(finding)

        assert "a/b.py" in rendered
        assert "2" in rendered
        assert "0.61" in rendered


class TestTheCommandLineGate:
    """What CI actually runs — doc 08 §5 and doc 00 C1."""

    def test_a_clean_tree_exits_zero(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("SPEED = 0.61\n")
        empty_registry_dir = tmp_path / "registry"
        empty_registry_dir.mkdir()

        assert (
            main(
                [
                    str(tmp_path),
                    "--assumed-baseline",
                    str(tmp_path / "none.json"),
                    "--registry-dir",
                    str(empty_registry_dir),
                ]
            )
            == 0
        )

    def test_a_finding_exits_nonzero_and_prints_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "a.py").write_text("def f():\n    return 0.61\n")

        code = main([str(tmp_path), "--assumed-baseline", str(tmp_path / "none.json")])

        assert code != 0
        assert "0.61" in capsys.readouterr().out

    def test_the_assumed_count_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # doc 00 C1: the ASSUMED count "is a tracked project metric and is reported in
        # CI". Reported means printed on every run, not only when it fails — a metric
        # you only see when it breaks is a metric nobody is tracking.
        main([str(tmp_path), "--assumed-baseline", str(tmp_path / "none.json")])

        assert "ASSUMED" in capsys.readouterr().out

    def test_an_increase_over_the_baseline_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # doc 00 C1: "its increase fails the build". The baseline is committed, so
        # adding an unsourced value is a deliberate, reviewable act rather than a drift.
        baseline = tmp_path / "baseline.json"
        baseline.write_text('{"assumed_count": -1}')

        code = main([str(tmp_path), "--assumed-baseline", str(baseline)])

        assert code != 0
        assert "baseline" in capsys.readouterr().out.lower()

    def test_a_decrease_below_the_baseline_passes_and_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Driving the count down is the stated goal (doc 09 §1). It must not fail the
        # build, and it should tell you to update the baseline so the gain is locked in.
        baseline = tmp_path / "baseline.json"
        baseline.write_text('{"assumed_count": 99}')

        code = main([str(tmp_path), "--assumed-baseline", str(baseline)])

        assert code == 0
        assert "baseline" in capsys.readouterr().out.lower()

    def test_a_missing_baseline_is_treated_as_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The strictest reading, and the right default: doc 09 §1's target is zero, so an
        # absent baseline must not silently mean "anything goes". Uses an empty isolated
        # registry (see test_a_clean_tree_exits_zero) rather than the shipped one, so this
        # test asserts the gate's own zero-baseline logic and not "the shipped registry
        # currently happens to have no ASSUMED entries".
        empty_registry_dir = tmp_path / "registry"
        empty_registry_dir.mkdir()

        assert (
            main(
                [
                    str(tmp_path),
                    "--assumed-baseline",
                    str(tmp_path / "absent.json"),
                    "--registry-dir",
                    str(empty_registry_dir),
                ]
            )
            == 0
        )
