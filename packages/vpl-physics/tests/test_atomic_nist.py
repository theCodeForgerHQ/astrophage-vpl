"""NIST ASD line data — doc 09 §2.2, doc 02 §6.3, doc 04 §2.

NIST ASD is US Government work and therefore public domain (doc 09 §5), so unlike the
LXCat fixtures these *could* legally be a committed extract. They are still synthetic,
for the same reason the LXCat ones are: a generated table with arithmetically exact
level energies tests the parser harder than a real one, and doc 09 §5's architectural
rule — the repository stores references and loaders, not bulk third-party data — is
worth applying uniformly rather than per-licence.

The one exception is the Ar I 750.39 nm row, whose energies are the real NIST values
against an **air** wavelength. It is there to prove that the parser's energy/wavelength
consistency check tolerates the 0.03 % air-vacuum offset instead of rejecting every real
row above 200 nm.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.params import Uncertainty
from vpl.physics.atomic.nist import (
    DOC_02_LINE_SET,
    LINE_MATCH_TOLERANCE_NM,
    AccuracyGrade,
    LineList,
    LineSpec,
    NistParseError,
    Transition,
    parse_nist_asd,
)

#: h c in eV nm. Written out independently of the value the module under test computes
#: from CODATA, so that the energy/wavelength consistency check is not tested against
#: its own arithmetic.
HC_EV_NM = 1239.841984

_HEADER = (
    "element\tsp_num\tobs_wl_air(nm)\tritz_wl_air(nm)\tAki(s^-1)\tAcc\tEi(eV)\tEk(eV)\t"
    "conf_i\tterm_i\tJ_i\tconf_k\tterm_k\tJ_k\tg_i\tg_k\tType"
)

#: ``(spectrum, wavelength_nm, A_ul, grade, E_lower_eV)`` for the doc 02 §6.3 set.
#: The upper level is derived so the row is exactly self-consistent; see the docstring.
_ROWS = (
    (1, 751.4652, 4.02e7, "AA", 11.623587),
    (1, 811.5311, 3.31e7, "AA", 11.548357),
    (1, 763.5106, 2.45e7, "AA", 11.548357),
    (1, 696.5431, 6.39e6, "A", 11.548357),
    (1, 706.7218, 3.80e6, "A", 11.548357),
    (2, 480.6021, 7.80e7, "B+", 16.643989),
    (2, 487.9864, 8.23e7, "B+", 16.643989),
    (2, 434.8064, 1.17e8, "B", 16.643989),
    (2, 476.4865, 6.40e7, "C", 16.643989),
)


def _row(
    *,
    spectrum: int,
    wavelength_nm: float,
    a_ul: str,
    grade: str,
    e_lower: float,
    e_upper: str | None = None,
) -> str:
    upper = f"{e_lower + HC_EV_NM / wavelength_nm:.6f}" if e_upper is None else e_upper
    return "\t".join(
        [
            "Ar",
            str(spectrum),
            f"{wavelength_nm:.4f}",
            f"{wavelength_nm:.4f}",
            a_ul,
            grade,
            f"{e_lower:.6f}",
            upper,
            "3s23p6",
            "1S",
            "0",
            "3s23p5(2P*)4p",
            "2[1/2]*",
            "1",
            "1",
            "3",
            "",
        ]
    )


def _asd_table(extra: tuple[str, ...] = ()) -> str:
    # The real Ar I 750.39 nm row: an air wavelength against vacuum level energies.
    real = _row(
        spectrum=1,
        wavelength_nm=750.3869,
        a_ul="4.45e+07",
        grade="AA",
        e_lower=11.828071,
        e_upper="13.479770",
    )
    generated = [
        _row(
            spectrum=spectrum,
            wavelength_nm=wavelength_nm,
            a_ul=f"{a_ul:.2e}",
            grade=grade,
            e_lower=e_lower,
        )
        for spectrum, wavelength_nm, a_ul, grade, e_lower in _ROWS
    ]
    return "\n".join([_HEADER, real, *generated, *extra]) + "\n"


def _transition(**overrides: object) -> Transition:
    defaults: dict[str, object] = {
        "element": "Ar",
        "spectrum": 1,
        "wavelength_nm": 750.3869,
        "a_ul_per_s": 4.45e7,
        "accuracy": AccuracyGrade.AA,
        "e_lower_ev": 11.828071,
        "e_upper_ev": 13.479770,
        "g_lower": 1,
        "g_upper": 3,
        "upper_level": "3s23p5(2P*)4p 2[1/2]* 1",
    }
    return Transition(**{**defaults, **overrides})  # type: ignore[arg-type]


# ── the accuracy grade, which doc 09 §2.2 makes mandatory ───────────────────────


class TestAccuracyGrade:
    """doc 09 §2.2: "the framework ingests the grade and propagates it"."""

    @pytest.mark.parametrize(
        ("grade", "relative"),
        [
            (AccuracyGrade.AAA, 0.003),
            (AccuracyGrade.AA, 0.01),
            (AccuracyGrade.A_PLUS, 0.02),
            (AccuracyGrade.A, 0.03),
            (AccuracyGrade.B_PLUS, 0.07),
            (AccuracyGrade.B, 0.10),
            (AccuracyGrade.C_PLUS, 0.18),
            (AccuracyGrade.C, 0.25),
            (AccuracyGrade.D_PLUS, 0.40),
            (AccuracyGrade.D, 0.50),
        ],
    )
    def test_each_grade_maps_to_the_nist_bound(self, grade: AccuracyGrade, relative: float) -> None:
        assert grade.relative_uncertainty == pytest.approx(relative)

    def test_grade_e_is_a_lower_bound_rather_than_a_value(self) -> None:
        # doc 09 §2.2 gives grade E as "> 50 %", which has no upper end. Reporting 50 %
        # for it would understate the uncertainty of the worst-known data in the set.
        assert AccuracyGrade.E.relative_uncertainty == pytest.approx(0.50)
        assert not AccuracyGrade.E.is_bounded
        assert AccuracyGrade.AAA.is_bounded

    def test_the_grades_are_ordered_from_best_to_worst(self) -> None:
        grades = list(AccuracyGrade)
        uncertainties = [g.relative_uncertainty for g in grades]

        assert uncertainties == sorted(uncertainties)

    def test_a_grade_becomes_the_registry_uncertainty_type(self) -> None:
        # vpl.core.params.Uncertainty is the project's one-sigma type (doc 02 §12); a
        # second spelling of the same idea in this package would be a fork of it.
        assert AccuracyGrade.AA.as_uncertainty() == Uncertainty(kind="relative", value=0.01)

    def test_the_manifest_spelling_round_trips(self) -> None:
        assert AccuracyGrade("A+") is AccuracyGrade.A_PLUS

    def test_an_unknown_grade_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="A\\+\\+"):
            AccuracyGrade("A++")


# ── parsing ─────────────────────────────────────────────────────────────────────


class TestParsingAnAsdTable:
    def test_every_data_row_becomes_a_transition(self) -> None:
        assert len(parse_nist_asd(_asd_table())) == len(_ROWS) + 1

    def test_the_species_label_is_the_doc_02_spelling(self) -> None:
        lines = parse_nist_asd(_asd_table())

        assert {t.species for t in lines} == {"Ar I", "Ar II"}

    def test_the_wavelength_transition_probability_and_grade_are_read(self) -> None:
        line = parse_nist_asd(_asd_table()).match(DOC_02_LINE_SET[0])

        assert line.wavelength_nm == pytest.approx(750.3869)
        assert line.a_ul_per_s == pytest.approx(4.45e7)
        assert line.accuracy is AccuracyGrade.AA

    def test_the_statistical_weights_are_read(self) -> None:
        line = parse_nist_asd(_asd_table()).match(DOC_02_LINE_SET[0])

        assert line.g_lower == 1
        assert line.g_upper == 3

    def test_the_level_energies_are_read(self) -> None:
        line = parse_nist_asd(_asd_table()).match(DOC_02_LINE_SET[0])

        assert line.e_lower_ev == pytest.approx(11.828071)
        assert line.e_upper_ev == pytest.approx(13.479770)

    def test_the_upper_level_label_is_assembled_from_the_configuration_columns(self) -> None:
        line = parse_nist_asd(_asd_table()).match(DOC_02_LINE_SET[0])

        assert line.upper_level == "3s23p5(2P*)4p 2[1/2]* 1"

    def test_a_pipe_delimited_export_parses_identically(self) -> None:
        piped = _asd_table().replace("\t", "|")

        assert len(parse_nist_asd(piped)) == len(_ROWS) + 1

    def test_spreadsheet_safe_quoting_is_stripped(self) -> None:
        # ASD's "tab-delimited, with = prefix" mode writes ="750.3869" so that a
        # spreadsheet does not reinterpret the cell. Left in place, float() fails and the
        # whole table looks unparseable.
        quoted = "\n".join(
            "\t".join(f'="{cell}"' for cell in row.split("\t")) for row in _asd_table().splitlines()
        )

        assert len(parse_nist_asd(quoted)) == len(_ROWS) + 1

    def test_blank_and_ruled_lines_are_ignored(self) -> None:
        ruled = _asd_table(extra=("", "-" * 40, ""))

        assert len(parse_nist_asd(ruled)) == len(_ROWS) + 1


class TestATransitionProbabilityNeverArrivesWithoutItsGrade:
    """doc 09 §2.2, made structural rather than procedural."""

    def test_a_row_with_a_transition_probability_and_no_grade_is_rejected(self) -> None:
        orphan = _row(spectrum=1, wavelength_nm=700.0, a_ul="1.00e+07", grade="", e_lower=11.5)

        with pytest.raises(NistParseError, match="accuracy"):
            parse_nist_asd(_asd_table(extra=(orphan,)))

    def test_a_row_with_an_unrecognised_grade_is_rejected(self) -> None:
        orphan = _row(spectrum=1, wavelength_nm=700.0, a_ul="1.00e+07", grade="Z", e_lower=11.5)

        with pytest.raises(NistParseError, match="Z"):
            parse_nist_asd(_asd_table(extra=(orphan,)))

    def test_the_type_cannot_be_built_without_a_grade(self) -> None:
        with pytest.raises(TypeError):
            Transition(  # type: ignore[call-arg]
                element="Ar",
                spectrum=1,
                wavelength_nm=750.3869,
                a_ul_per_s=4.45e7,
                e_lower_ev=11.828071,
                e_upper_ev=13.479770,
                g_lower=1,
                g_upper=3,
                upper_level="",
            )


class TestRowsWithoutATransitionProbability:
    """An ASD query returns levels with no ``A_ki``. They are counted, not dropped."""

    def test_such_rows_do_not_become_transitions(self) -> None:
        blank = _row(spectrum=1, wavelength_nm=700.0, a_ul="", grade="", e_lower=11.5)

        assert len(parse_nist_asd(_asd_table(extra=(blank,)))) == len(_ROWS) + 1

    def test_such_rows_are_counted_so_the_omission_is_visible(self) -> None:
        blank = _row(spectrum=1, wavelength_nm=700.0, a_ul="", grade="", e_lower=11.5)

        assert parse_nist_asd(_asd_table(extra=(blank,))).rows_without_a_ul == 1

    def test_the_count_appears_in_the_repr(self) -> None:
        blank = _row(spectrum=1, wavelength_nm=700.0, a_ul="", grade="", e_lower=11.5)

        assert "1 without" in repr(parse_nist_asd(_asd_table(extra=(blank,))))


# ── the doc 02 §6.3 line set ────────────────────────────────────────────────────


class TestTheDoc02LineSet:
    def test_the_specification_lists_ten_lines(self) -> None:
        assert len(DOC_02_LINE_SET) == 10

    def test_it_is_six_ar_one_lines_and_four_ar_two_lines(self) -> None:
        assert sum(spec.species == "Ar I" for spec in DOC_02_LINE_SET) == 6
        assert sum(spec.species == "Ar II" for spec in DOC_02_LINE_SET) == 4

    def test_the_te_discriminator_pair_is_present(self) -> None:
        # doc 02 §6.3: "The 750.39/811.53 ratio is the classical T_e discriminator."
        wavelengths = {spec.wavelength_nm for spec in DOC_02_LINE_SET}

        assert {750.39, 811.53} <= wavelengths

    def test_a_complete_table_leaves_nothing_missing(self) -> None:
        assert parse_nist_asd(_asd_table()).missing() == ()

    def test_an_incomplete_table_names_what_is_missing(self) -> None:
        without_811 = "\n".join(
            row for row in _asd_table().splitlines() if not row.startswith("Ar\t1\t811")
        )

        missing = parse_nist_asd(without_811).missing()

        assert len(missing) == 1
        assert missing[0].wavelength_nm == pytest.approx(811.53)

    def test_every_specification_line_carries_the_role_doc_02_gives_it(self) -> None:
        roles = {spec.role for spec in DOC_02_LINE_SET}

        assert "metastable-coupled" in roles
        assert "direct excitation dominated" in roles


class TestMatching:
    def test_a_line_is_matched_within_the_calibration_tolerance(self) -> None:
        # doc 02 §6.3 quotes wavelengths to 0.01 nm; ASD carries four decimals. The
        # tolerance has to absorb that rounding without reaching a neighbouring line.
        lines = parse_nist_asd(_asd_table())

        assert lines.match(DOC_02_LINE_SET[0]).wavelength_nm == pytest.approx(750.3869)

    def test_the_tolerance_is_far_smaller_than_the_nearest_line_separation(self) -> None:
        # 750.39 and 751.47 are 1.08 nm apart, the closest pair in the set. A tolerance
        # anywhere near that would let one line match the other's specification.
        assert LINE_MATCH_TOLERANCE_NM < 0.1

    def test_an_unmatched_line_raises_rather_than_returning_nothing(self) -> None:
        without_811 = "\n".join(
            row for row in _asd_table().splitlines() if not row.startswith("Ar\t1\t811")
        )

        with pytest.raises(LookupError, match=r"811\.53"):
            parse_nist_asd(without_811).match(DOC_02_LINE_SET[2])

    def test_an_ambiguous_match_raises_rather_than_picking_one(self) -> None:
        # Two ASD rows inside the tolerance means the line is blended at this resolution,
        # which doc 02 §6.3 selected the set to avoid. Picking the first would silently
        # halve or double the modelled emissivity.
        twin = _row(spectrum=1, wavelength_nm=750.4000, a_ul="1.00e+07", grade="B", e_lower=11.6)

        with pytest.raises(LookupError, match="ambiguous"):
            parse_nist_asd(_asd_table(extra=(twin,))).match(DOC_02_LINE_SET[0])

    def test_a_wavelength_match_in_the_wrong_ionisation_stage_is_not_a_match(self) -> None:
        # Ar I and Ar II lines are different physics: doc 02 §6.3 uses the ion lines for
        # n_i and the neutral lines for T_e. Matching on wavelength alone would let a
        # mislabelled row cross the two.
        lines = parse_nist_asd(_asd_table())
        wrong_stage = LineSpec(
            species="Ar II", wavelength_nm=750.39, paschen_level=None, role="test"
        )

        with pytest.raises(LookupError, match="Ar II"):
            lines.match(wrong_stage)

    def test_near_returns_every_candidate_in_the_window(self) -> None:
        lines = parse_nist_asd(_asd_table())

        assert len(lines.near("Ar I", 750.39, tolerance_nm=2.0)) == 2

    def test_of_species_partitions_the_list(self) -> None:
        lines = parse_nist_asd(_asd_table())

        assert len(lines.of_species("Ar I")) == 6
        assert len(lines.of_species("Ar II")) == 4


# ── propagating the grade ───────────────────────────────────────────────────────


class TestUncertaintyPropagation:
    def test_a_transition_reports_its_one_sigma_uncertainty(self) -> None:
        line = _transition()

        assert line.uncertainty == Uncertainty(kind="relative", value=0.01)
        assert line.a_ul_uncertainty_per_s == pytest.approx(4.45e5)

    @pytest.mark.physics
    def test_a_better_graded_line_is_weighted_more_heavily(self) -> None:
        # doc 09 §2.2: "Ar I 811.53 nm carries a much better grade than several Ar II
        # lines, and weighting lines by their data quality is straightforward, correct,
        # and almost never done." This is that weighting.
        lines = parse_nist_asd(_asd_table())
        weights = dict(zip([t.species_wavelength for t in lines], lines.weights(), strict=True))

        assert weights[("Ar I", 811.5311)] > weights[("Ar II", 476.4865)]

    def test_the_weights_are_normalised(self) -> None:
        assert parse_nist_asd(_asd_table()).weights().sum() == pytest.approx(1.0)

    def test_the_weights_go_as_the_inverse_square_of_the_relative_uncertainty(self) -> None:
        lines = LineList(
            transitions=(
                _transition(accuracy=AccuracyGrade.AA),
                _transition(wavelength_nm=751.0, accuracy=AccuracyGrade.B),
            ),
            rows_without_a_ul=0,
        )

        weights = lines.weights()

        assert weights[0] / weights[1] == pytest.approx((0.10 / 0.01) ** 2)

    def test_an_unbounded_grade_refuses_to_be_weighted_silently(self) -> None:
        # Grade E means "> 50 %". Treating it as exactly 50 % would give the least
        # trustworthy line in the set a finite, respectable weight.
        lines = LineList(transitions=(_transition(accuracy=AccuracyGrade.E),), rows_without_a_ul=0)

        with pytest.raises(ValueError, match="E"):
            lines.weights()

    def test_an_unbounded_grade_can_be_weighted_with_an_explicit_bound(self) -> None:
        lines = LineList(transitions=(_transition(accuracy=AccuracyGrade.E),), rows_without_a_ul=0)

        assert lines.weights(unbounded_relative_uncertainty=2.0) == pytest.approx(np.array([1.0]))


# ── validation ──────────────────────────────────────────────────────────────────


class TestTransitionValidation:
    def test_rejects_a_non_positive_transition_probability(self) -> None:
        with pytest.raises(ValueError, match="A_ul"):
            _transition(a_ul_per_s=0.0)

    def test_rejects_a_non_positive_wavelength(self) -> None:
        with pytest.raises(ValueError, match="wavelength"):
            _transition(wavelength_nm=0.0)

    def test_rejects_a_statistical_weight_below_one(self) -> None:
        with pytest.raises(ValueError, match="statistical weight"):
            _transition(g_upper=0)

    def test_rejects_an_upper_level_below_the_lower_level(self) -> None:
        with pytest.raises(ValueError, match="above"):
            _transition(e_upper_ev=1.0)

    @pytest.mark.physics
    def test_rejects_energies_that_disagree_with_the_wavelength(self) -> None:
        # h nu = E_k - E_i. A row where they disagree has had a column misread — an
        # Angstrom wavelength, a cm^-1 level, or two rows spliced together — and the
        # emissivity of doc 04 §2.1 would be wrong by whatever the mismatch is.
        with pytest.raises(ValueError, match="photon energy"):
            _transition(e_upper_ev=11.828071 + 2.0)

    @pytest.mark.physics
    def test_accepts_the_air_vacuum_offset_of_a_real_row(self) -> None:
        # The real Ar I 750.39 nm row is 0.035 % inconsistent because ASD quotes an air
        # wavelength against vacuum level energies. A check tight enough to reject that
        # would reject every line in the doc 02 set.
        line = _transition()

        assert line.photon_energy_ev == pytest.approx(line.energy_gap_ev, rel=1e-3)

    def test_rejects_a_transition_with_no_element(self) -> None:
        with pytest.raises(ValueError, match="element"):
            _transition(element=" ")

    def test_rejects_an_ionisation_stage_it_has_no_numeral_for(self) -> None:
        with pytest.raises(ValueError, match="spectrum"):
            _transition(spectrum=99)

    def test_a_row_failing_a_physics_check_names_the_row_it_came_from(self) -> None:
        # The validation lives on the type, but a table is read row by row, and "the
        # photon energy disagrees" with no row number is not actionable on a 400-line
        # export.
        inconsistent = _row(
            spectrum=1,
            wavelength_nm=700.0,
            a_ul="1.00e+07",
            grade="AA",
            e_lower=11.5,
            e_upper="99.0",
        )

        with pytest.raises(NistParseError, match="row 12"):
            parse_nist_asd(_asd_table(extra=(inconsistent,)))

    def test_the_transition_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            _transition().a_ul_per_s = 1.0  # type: ignore[misc]

    def test_the_repr_names_the_species_wavelength_and_grade(self) -> None:
        text = repr(_transition())

        assert "Ar I" in text
        assert "750" in text
        assert "AA" in text


class TestTheParserFailsLoudly:
    def test_rejects_a_table_with_no_header(self) -> None:
        with pytest.raises(NistParseError, match="header"):
            parse_nist_asd("")

    def test_rejects_a_table_with_no_data_rows(self) -> None:
        with pytest.raises(NistParseError, match="no rows"):
            parse_nist_asd(_HEADER + "\n")

    def test_rejects_a_missing_required_column(self) -> None:
        without_g = _asd_table().replace("\tg_k\t", "\tspare\t")

        with pytest.raises(NistParseError, match="g_k"):
            parse_nist_asd(without_g)

    def test_rejects_a_wavelength_column_in_the_wrong_units(self) -> None:
        # ASD will export Angstroms just as happily as nanometres.
        angstroms = _asd_table().replace("obs_wl_air(nm)", "obs_wl_air(A)")

        with pytest.raises(NistParseError, match="wavelength"):
            parse_nist_asd(angstroms)

    def test_rejects_a_table_with_no_wavelength_column_at_all(self) -> None:
        headerless = (
            _asd_table().replace("obs_wl_air(nm)", "spare1").replace("ritz_wl_air(nm)", "spare2")
        )

        with pytest.raises(NistParseError, match="wavelength column"):
            parse_nist_asd(headerless)

    def test_an_export_without_the_optional_level_labels_still_parses(self) -> None:
        # ASD's "Lines" query can be asked for transition data without level
        # configurations. The label is documentation, so its absence is not an error.
        minimal = _asd_table().replace("conf_k", "spare_a").replace("term_k", "spare_b")

        assert parse_nist_asd(minimal)[0].upper_level == "1"

    def test_rejects_a_level_energy_column_in_the_wrong_units(self) -> None:
        wavenumbers = _asd_table().replace("Ei(eV)", "Ei(cm-1)")

        with pytest.raises(NistParseError, match="Ei"):
            parse_nist_asd(wavenumbers)

    def test_rejects_a_row_with_too_few_cells(self) -> None:
        with pytest.raises(NistParseError, match="cells"):
            parse_nist_asd(_asd_table(extra=("Ar\t1\t700.0",)))

    def test_rejects_a_non_numeric_cell(self) -> None:
        broken = _row(spectrum=1, wavelength_nm=700.0, a_ul="strong", grade="AA", e_lower=11.5)

        with pytest.raises(NistParseError, match="number"):
            parse_nist_asd(_asd_table(extra=(broken,)))

    def test_names_the_row_that_failed(self) -> None:
        broken = _row(spectrum=1, wavelength_nm=700.0, a_ul="strong", grade="AA", e_lower=11.5)

        with pytest.raises(NistParseError, match="row 12"):
            parse_nist_asd(_asd_table(extra=(broken,)))


class TestTheListContract:
    def test_an_empty_list_is_permitted_but_reports_itself(self) -> None:
        empty = LineList(transitions=(), rows_without_a_ul=0)

        assert len(empty) == 0
        assert empty.missing() == DOC_02_LINE_SET

    def test_an_empty_list_has_no_weights_to_distribute(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            LineList(transitions=(), rows_without_a_ul=0).weights()

    def test_a_negative_skipped_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="rows_without_a_ul"):
            LineList(transitions=(), rows_without_a_ul=-1)

    def test_the_list_is_iterable_and_indexable(self) -> None:
        lines = parse_nist_asd(_asd_table())

        assert lines[0].wavelength_nm == pytest.approx(750.3869)
        assert len(list(lines)) == len(lines)
