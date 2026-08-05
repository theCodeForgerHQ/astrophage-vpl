"""The LXCat export parser — doc 09 §2.1, doc 03 §4.5.

Every fixture in this file is **synthetic**. Doc 09 §5 forbids redistributing raw LXCat
tables, so the test suite cannot check the parser against a downloaded file without
committing one. The blocks below are hand-written in the documented export format and
carry made-up numbers chosen to be arithmetically checkable — which is the stronger test
anyway: a real table would let a parser that silently dropped its last row still pass.

The *shape* of the format is what is under test, and the parts of it that matter are the
ones where a plausible misreading is silent: which column is the cross section, where the
threshold comes from, and whether a process the specification requires is present at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.physics.atomic.lxcat import (
    REQUIRED_ELECTRON_PROCESSES,
    REQUIRED_ION_PROCESSES,
    CrossSection,
    CrossSectionSet,
    LxcatParseError,
    ProcessType,
    parse_lxcat,
)

# ── fixtures in the documented export format ────────────────────────────────────

ELECTRON_EXPORT = """\
LXCat, www.lxcat.net
Generated on 05 Aug 2026. All rights reserved.

RECOMMENDED REFERENCE FORMAT
- Phelps database, www.lxcat.net, retrieved on August 5, 2026.

************************************************************************

DATABASE:         Phelps
PERMLINK:         www.lxcat.net/Phelps
DESCRIPTION:      A compilation of atomic and molecular data.
CONTACT:          leanne.pitchford@lxcat.net
HOW TO REFERENCE: Phelps database, www.lxcat.net, retrieved on August 5, 2026.
************************************************************************

*********************************** Ar ***********************************

EFFECTIVE
Ar
 1.360000e-5
SPECIES: e / Ar
PROCESS: E + Ar -> E + Ar, Effective
PARAM.:  m/M = 0.0000136, complete set
COMMENT: EFFECTIVE MOMENTUM TRANSFER CROSS SECTION.
UPDATED: 2011-06-06 11:19:52
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 0.000000e+0	1.000000e-20
 1.000000e+0	2.000000e-20
 1.000000e+1	4.000000e-20
 1.000000e+2	8.000000e-21
-----------------------------

EXCITATION
Ar -> Ar*(11.5eV)
 1.150000e+1
SPECIES: e / Ar
PROCESS: E + Ar -> E + Ar*(11.5eV), Excitation
PARAM.:  E = 11.5 eV, complete set
COMMENT: metastable, lumped.
UPDATED: 2011-06-06 11:19:52
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.150000e+1	0.000000e+0
 1.500000e+1	1.000000e-21
 2.000000e+1	2.000000e-21
 1.000000e+2	5.000000e-22
-----------------------------

IONIZATION
Ar -> Ar^+
 1.580000e+1
SPECIES: e / Ar
PROCESS: E + Ar -> E + E + Ar+, Ionization
PARAM.:  E = 15.8 eV, complete set
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.580000e+1	0.000000e+0
 2.000000e+1	1.000000e-21
 1.000000e+2	3.000000e-20
 1.000000e+3	8.000000e-21
-----------------------------
"""

ION_EXPORT = """\
LXCat, www.lxcat.net

DATABASE:         Phelps
HOW TO REFERENCE: Phelps database, www.lxcat.net, retrieved on August 5, 2026.

ELASTIC
Ar+ -> Ar+
SPECIES: Ar+ / Ar
PROCESS: Ar+ + Ar -> Ar+ + Ar, Elastic
COMMENT: isotropic part.
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.000000e-2	1.000000e-18
 1.000000e+0	4.000000e-19
 1.000000e+3	1.000000e-19
-----------------------------

CHARGE EXCHANGE
Ar+ -> Ar
SPECIES: Ar+ / Ar
PROCESS: Ar+ + Ar -> Ar + Ar+, Charge exchange
COMMENT: symmetric resonant charge transfer.
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.000000e-2	8.000000e-19
 1.000000e+0	6.000000e-19
 1.000000e+3	3.000000e-19
-----------------------------
"""


def _minimal_block(
    *,
    keyword: str = "IONIZATION",
    reaction: str = "Ar -> Ar^+",
    parameter: str = " 1.580000e+1",
    process: str = "PROCESS: E + Ar -> E + E + Ar+, Ionization",
    species: str = "SPECIES: e / Ar",
    param_line: str = "PARAM.:  E = 15.8 eV, complete set",
    columns: str = "COLUMNS: Energy (eV) | Cross section (m2)",
    rows: str = " 1.580000e+1\t0.000000e+0\n 2.000000e+1\t1.000000e-21",
    closing: str = "-----------------------------",
) -> str:
    """One block, with each part substitutable so a single defect can be isolated."""
    parts = [
        "DATABASE: Phelps",
        "",
        keyword,
        reaction,
        *([parameter] if parameter else []),
        species,
        process,
        param_line,
        columns,
        "-----------------------------",
        rows,
        closing,
        "",
    ]
    return "\n".join(parts)


# ── the electron set ────────────────────────────────────────────────────────────


class TestParsingAnElectronSet:
    def test_the_database_name_comes_from_the_export_header(self) -> None:
        assert parse_lxcat(ELECTRON_EXPORT).database == "Phelps"

    def test_every_block_becomes_a_cross_section(self) -> None:
        assert len(parse_lxcat(ELECTRON_EXPORT)) == 3

    def test_the_process_types_are_the_block_keywords(self) -> None:
        found = parse_lxcat(ELECTRON_EXPORT).process_types()

        assert found == frozenset(
            {ProcessType.EFFECTIVE, ProcessType.EXCITATION, ProcessType.IONIZATION}
        )

    def test_the_reactants_are_split_without_breaking_a_charged_product(self) -> None:
        # "E + E + Ar+" naively split on "+" yields ['E', 'E', 'Ar', ''] and loses the
        # ion. The separator is a plus *surrounded by whitespace*, not any plus.
        ionization = parse_lxcat(ELECTRON_EXPORT).require(ProcessType.IONIZATION)

        assert ionization.products == ("E", "E", "Ar+")

    def test_the_projectile_and_target_come_from_the_species_line(self) -> None:
        ionization = parse_lxcat(ELECTRON_EXPORT).require(ProcessType.IONIZATION)

        assert ionization.projectile == "e"
        assert ionization.target == "Ar"

    def test_the_threshold_is_read_for_an_inelastic_process(self) -> None:
        excitation = parse_lxcat(ELECTRON_EXPORT).require(ProcessType.EXCITATION)

        assert excitation.threshold_ev == pytest.approx(11.5, rel=1e-12)

    def test_an_elastic_block_carries_a_mass_ratio_and_no_threshold(self) -> None:
        # The bare number under the reaction means different things per process type:
        # m/M for momentum transfer, a threshold energy for an inelastic process. Reading
        # one as the other would put a 1.4e-5 eV threshold on the elastic channel.
        effective = parse_lxcat(ELECTRON_EXPORT).require(ProcessType.EFFECTIVE)

        assert effective.threshold_ev is None
        assert effective.mass_ratio == pytest.approx(1.36e-5, rel=1e-12)

    def test_the_table_is_read_as_energy_then_cross_section(self) -> None:
        effective = parse_lxcat(ELECTRON_EXPORT).require(ProcessType.EFFECTIVE)

        np.testing.assert_allclose(effective.energy_ev, [0.0, 1.0, 10.0, 100.0])
        np.testing.assert_allclose(effective.sigma_m2, [1e-20, 2e-20, 4e-20, 8e-21])

    def test_the_header_fields_are_kept_for_the_provenance_record(self) -> None:
        excitation = parse_lxcat(ELECTRON_EXPORT).require(ProcessType.EXCITATION)

        assert excitation.parameters["COMMENT"] == "metastable, lumped."
        assert excitation.parameters["UPDATED"] == "2011-06-06 11:19:52"

    def test_the_how_to_reference_line_is_retained(self) -> None:
        # doc 09 §2.1: LXCat is free but "citation of the specific database is required",
        # and the required wording is in the export itself. Dropping it would leave the
        # citation ledger of doc 09 §6 to be reconstructed from memory.
        assert "Phelps database" in parse_lxcat(ELECTRON_EXPORT).reference

    def test_the_reaction_reads_back_as_the_published_equation(self) -> None:
        ionization = parse_lxcat(ELECTRON_EXPORT).require(ProcessType.IONIZATION)

        assert ionization.reaction == "E + Ar -> E + E + Ar+"

    def test_the_energy_range_is_reported(self) -> None:
        ionization = parse_lxcat(ELECTRON_EXPORT).require(ProcessType.IONIZATION)

        assert ionization.energy_range_ev == pytest.approx((15.8, 1000.0))

    def test_a_carriage_return_line_ending_parses_identically(self) -> None:
        # Files fetched on one machine and read on another. A stray '\r' left on the
        # last column would make every cross section unparseable, or worse, parseable.
        crlf = parse_lxcat(ELECTRON_EXPORT.replace("\n", "\r\n"))

        assert len(crlf) == 3
        np.testing.assert_allclose(
            crlf.require(ProcessType.IONIZATION).sigma_m2,
            parse_lxcat(ELECTRON_EXPORT).require(ProcessType.IONIZATION).sigma_m2,
        )


# ── the ion set, which doc 03 §4.5 singles out ──────────────────────────────────


class TestParsingAnIonSet:
    def test_charge_exchange_is_recognised_as_a_two_word_keyword(self) -> None:
        # doc 03 §4.5: "Getting the CX cross section and its energy dependence right
        # matters more than any other atomic-data choice." A parser that did not know
        # the keyword would drop the block and leave a set that looked complete.
        assert parse_lxcat(ION_EXPORT).charge_exchange().process is ProcessType.CHARGE_EXCHANGE

    def test_the_ion_is_the_projectile_and_the_neutral_is_the_target(self) -> None:
        cx = parse_lxcat(ION_EXPORT).charge_exchange()

        assert cx.projectile == "Ar+"
        assert cx.target == "Ar"

    def test_charge_exchange_has_no_threshold(self) -> None:
        # Symmetric resonant transfer is exothermic to zeroth order; a threshold read in
        # here would zero the cross section over exactly the low-energy range that
        # produces the IEDF structure of doc 03 §4.5.
        assert parse_lxcat(ION_EXPORT).charge_exchange().threshold_ev is None

    def test_a_block_without_a_bare_parameter_line_still_parses(self) -> None:
        assert parse_lxcat(ION_EXPORT).require(ProcessType.ELASTIC).mass_ratio is None

    def test_the_ion_set_covers_the_processes_doc_03_requires(self) -> None:
        assert parse_lxcat(ION_EXPORT).missing_processes(REQUIRED_ION_PROCESSES) == ()

    def test_an_electron_set_covers_the_electron_processes_doc_03_requires(self) -> None:
        assert parse_lxcat(ELECTRON_EXPORT).missing_processes(REQUIRED_ELECTRON_PROCESSES) == ()

    def test_a_missing_process_is_named_rather_than_silently_absent(self) -> None:
        electron = parse_lxcat(ELECTRON_EXPORT)

        assert electron.missing_processes(REQUIRED_ION_PROCESSES) == (
            ProcessType.CHARGE_EXCHANGE,
            ProcessType.ELASTIC,
        )


# ── selection ───────────────────────────────────────────────────────────────────


class TestSelectingSections:
    def test_of_type_returns_every_matching_section(self) -> None:
        assert len(parse_lxcat(ELECTRON_EXPORT).of_type(ProcessType.EXCITATION)) == 1

    def test_of_type_returns_nothing_for_an_absent_process(self) -> None:
        assert parse_lxcat(ELECTRON_EXPORT).of_type(ProcessType.ATTACHMENT) == ()

    def test_require_raises_when_the_process_is_absent(self) -> None:
        with pytest.raises(LookupError, match="ATTACHMENT"):
            parse_lxcat(ELECTRON_EXPORT).require(ProcessType.ATTACHMENT)

    def test_require_raises_when_the_process_is_ambiguous(self) -> None:
        # An argon set has one ionisation channel but many excitation channels. Silently
        # returning the first would make the rate coefficient depend on file ordering.
        doubled = _minimal_block() + _minimal_block()

        with pytest.raises(LookupError, match="2"):
            parse_lxcat(doubled).require(ProcessType.IONIZATION)

    def test_charge_exchange_raises_on_an_electron_set(self) -> None:
        with pytest.raises(LookupError, match="CHARGE EXCHANGE"):
            parse_lxcat(ELECTRON_EXPORT).charge_exchange()

    def test_excitations_are_returned_in_file_order(self) -> None:
        assert len(parse_lxcat(ELECTRON_EXPORT).excitations()) == 1

    def test_a_set_is_indexable_and_iterable(self) -> None:
        electron = parse_lxcat(ELECTRON_EXPORT)

        assert electron[0].process is ProcessType.EFFECTIVE
        assert [s.process for s in electron][-1] is ProcessType.IONIZATION

    def test_momentum_transfer_accepts_either_spelling(self) -> None:
        # LXCat's Phelps argon set publishes EFFECTIVE (momentum transfer); Biagi
        # publishes ELASTIC. Doc 03 §4.5 asks for "electron elastic" and means whichever
        # of the two the database supplies.
        assert parse_lxcat(ELECTRON_EXPORT).momentum_transfer().process is ProcessType.EFFECTIVE
        assert parse_lxcat(ION_EXPORT).momentum_transfer().process is ProcessType.ELASTIC


# ── failing loudly ──────────────────────────────────────────────────────────────


class TestTheParserFailsLoudly:
    """Every silent misreading this format admits, turned into an exception."""

    def test_rejects_an_export_with_no_database_and_no_override(self) -> None:
        with pytest.raises(LxcatParseError, match="DATABASE"):
            parse_lxcat("ELASTIC\nAr\n")

    def test_an_explicit_database_name_supplies_a_missing_header(self) -> None:
        text = _minimal_block().replace("DATABASE: Phelps\n", "")

        assert parse_lxcat(text, database="Biagi").database == "Biagi"

    def test_an_explicit_database_name_overrides_the_header(self) -> None:
        assert parse_lxcat(ELECTRON_EXPORT, database="IST-Lisbon").database == "IST-Lisbon"

    def test_rejects_an_export_containing_no_blocks(self) -> None:
        with pytest.raises(LxcatParseError, match="no cross-section"):
            parse_lxcat("DATABASE: Phelps\nDESCRIPTION: nothing here.\n")

    def test_rejects_a_block_with_no_process_line(self) -> None:
        with pytest.raises(LxcatParseError, match="PROCESS"):
            parse_lxcat(_minimal_block(process="COMMENT: none"))

    def test_rejects_an_unterminated_data_block(self) -> None:
        with pytest.raises(LxcatParseError, match="closing"):
            parse_lxcat(_minimal_block(closing=""))

    def test_rejects_a_data_row_that_is_not_two_numbers(self) -> None:
        with pytest.raises(LxcatParseError, match="two"):
            parse_lxcat(_minimal_block(rows=" 1.0\t2.0\t3.0"))

    def test_rejects_a_data_row_that_is_not_numeric(self) -> None:
        with pytest.raises(LxcatParseError, match="number"):
            parse_lxcat(_minimal_block(rows=" 1.0\tn/a"))

    def test_rejects_a_table_with_one_point(self) -> None:
        with pytest.raises(LxcatParseError, match="two points"):
            parse_lxcat(_minimal_block(rows=" 1.580000e+1\t0.0"))

    def test_rejects_a_cross_section_column_in_the_wrong_units(self) -> None:
        # An Angstrom-squared table read as m^2 is wrong by 20 orders of magnitude, and
        # nothing downstream would look obviously wrong — the rate coefficient would just
        # be zero.
        with pytest.raises(LxcatParseError, match="COLUMNS"):
            parse_lxcat(_minimal_block(columns="COLUMNS: Energy (eV) | Cross section (A2)"))

    def test_rejects_an_energy_column_in_the_wrong_units(self) -> None:
        with pytest.raises(LxcatParseError, match="COLUMNS"):
            parse_lxcat(_minimal_block(columns="COLUMNS: Energy (K) | Cross section (m2)"))

    def test_rejects_a_missing_columns_line(self) -> None:
        with pytest.raises(LxcatParseError, match="COLUMNS"):
            parse_lxcat(_minimal_block(columns="COMMENT: no columns declared"))

    def test_rejects_an_inelastic_process_with_no_threshold(self) -> None:
        with pytest.raises(LxcatParseError, match="threshold"):
            parse_lxcat(_minimal_block(parameter="", param_line="COMMENT: none"))

    def test_rejects_a_threshold_that_disagrees_with_the_param_line(self) -> None:
        # LXCat states the threshold twice. When the two disagree the file has been
        # hand-edited, and picking either silently is a coin flip on the rate coefficient.
        with pytest.raises(LxcatParseError, match="threshold"):
            parse_lxcat(_minimal_block(parameter=" 1.100000e+1"))

    def test_rejects_a_process_label_that_contradicts_the_block_keyword(self) -> None:
        with pytest.raises(LxcatParseError, match="EXCITATION"):
            parse_lxcat(_minimal_block(process="PROCESS: E + Ar -> E + Ar*(11.5eV), Excitation"))

    def test_rejects_a_continuation_line_with_no_field_to_continue(self) -> None:
        with pytest.raises(LxcatParseError, match="header"):
            parse_lxcat(_minimal_block(reaction="Ar -> Ar^+\nstray text"))

    def test_rejects_a_columns_line_it_cannot_parse_at_all(self) -> None:
        with pytest.raises(LxcatParseError, match="COLUMNS"):
            parse_lxcat(_minimal_block(columns="COLUMNS: energy and cross section"))

    def test_rejects_a_block_that_ends_before_its_reaction_line(self) -> None:
        with pytest.raises(LxcatParseError, match="reaction"):
            parse_lxcat("DATABASE: Phelps\n\nIONIZATION\n")


class TestFormatTolerances:
    """Shapes of the export that are legal but easy to mishandle."""

    def test_a_wrapped_header_field_is_joined_to_the_line_it_continues(self) -> None:
        # LXCat wraps long COMMENT fields. A continuation dropped on the floor loses
        # whatever caveat the compiler thought worth writing down.
        text = _minimal_block(param_line="COMMENT: first part\n  second part")

        section = parse_lxcat(text).require(ProcessType.IONIZATION)

        assert section.parameters["COMMENT"] == "first part second part"

    def test_an_export_without_a_species_line_falls_back_to_the_reaction(self) -> None:
        # Older exports omit SPECIES:. The PROCESS line says the same thing less
        # directly, and refusing the file would make an archived download unloadable.
        section = parse_lxcat(_minimal_block(species="COMMENT: no species line")).require(
            ProcessType.IONIZATION
        )

        assert section.projectile == "E"
        assert section.target == "Ar"

    def test_a_set_with_no_elastic_or_effective_channel_says_so(self) -> None:
        excitation_only = _minimal_block(
            keyword="EXCITATION",
            process="PROCESS: E + Ar -> E + Ar*(15.8eV), Excitation",
        )

        with pytest.raises(LookupError, match="momentum-transfer"):
            parse_lxcat(excitation_only).momentum_transfer()

    def test_a_blank_line_between_the_keyword_and_the_reaction_is_skipped(self) -> None:
        text = _minimal_block(reaction="\nAr -> Ar^+")

        assert parse_lxcat(text).require(ProcessType.IONIZATION).threshold_ev == pytest.approx(15.8)

    def test_the_peak_cross_section_is_reported(self) -> None:
        assert parse_lxcat(ELECTRON_EXPORT).require(
            ProcessType.IONIZATION
        ).peak_sigma_m2 == pytest.approx(3e-20)


class TestCrossSectionValidation:
    """Checks that belong to the data, not to the file format."""

    def _section(self, **overrides: object) -> CrossSection:
        defaults: dict[str, object] = {
            "process": ProcessType.IONIZATION,
            "database": "Phelps",
            "projectile": "e",
            "target": "Ar",
            "reactants": ("E", "Ar"),
            "products": ("E", "E", "Ar+"),
            "threshold_ev": 15.8,
            "mass_ratio": None,
            "energy_ev": np.array([15.8, 20.0, 100.0]),
            "sigma_m2": np.array([0.0, 1e-21, 3e-20]),
            "parameters": {},
        }
        return CrossSection(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_the_arrays_are_copied_from_the_caller(self) -> None:
        energy = np.array([15.8, 20.0, 100.0])
        section = self._section(energy_ev=energy)
        energy[0] = 1.0

        assert section.energy_ev[0] == pytest.approx(15.8)

    def test_the_arrays_are_write_locked(self) -> None:
        section = self._section()

        with pytest.raises(ValueError, match="read-only"):
            section.sigma_m2[0] = 1.0

    def test_rejects_a_non_monotonic_energy_axis(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            self._section(energy_ev=np.array([20.0, 15.8, 100.0]))

    def test_rejects_a_negative_cross_section(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            self._section(sigma_m2=np.array([-1e-21, 1e-21, 3e-20]))

    def test_rejects_a_negative_energy(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            self._section(energy_ev=np.array([-1.0, 20.0, 100.0]))

    def test_rejects_mismatched_column_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            self._section(sigma_m2=np.array([0.0, 1e-21]))

    def test_rejects_a_non_finite_entry(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            self._section(sigma_m2=np.array([0.0, np.inf, 3e-20]))

    def test_rejects_a_missing_threshold_on_an_inelastic_process(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            self._section(threshold_ev=None)

    def test_rejects_a_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            self._section(threshold_ev=-1.0)

    @pytest.mark.physics
    def test_rejects_a_non_zero_cross_section_below_the_threshold(self) -> None:
        # An inelastic process cannot occur below its threshold. A table that says
        # otherwise has had its threshold misread, and the resulting rate coefficient
        # would be finite where it must be exactly zero.
        with pytest.raises(ValueError, match="below"):
            self._section(threshold_ev=30.0)

    def test_rejects_a_reaction_with_an_empty_side(self) -> None:
        with pytest.raises(ValueError, match="reactant"):
            self._section(reactants=())

    def test_rejects_a_two_dimensional_table(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            self._section(energy_ev=np.zeros((2, 3)), sigma_m2=np.zeros((2, 3)))

    def test_the_section_is_immutable(self) -> None:
        section = self._section()

        with pytest.raises(AttributeError):
            section.threshold_ev = 1.0  # type: ignore[misc]

    def test_the_repr_names_the_reaction_and_the_database(self) -> None:
        text = repr(self._section())

        assert "Phelps" in text
        assert "Ar+" in text

    def test_the_parameters_mapping_is_read_only(self) -> None:
        section = self._section(parameters={"COMMENT": "x"})

        with pytest.raises(TypeError):
            section.parameters["COMMENT"] = "y"  # type: ignore[index]


class TestProcessType:
    @pytest.mark.parametrize(
        ("process", "expected"),
        [
            (ProcessType.EXCITATION, True),
            (ProcessType.IONIZATION, True),
            (ProcessType.ELASTIC, False),
            (ProcessType.EFFECTIVE, False),
            (ProcessType.CHARGE_EXCHANGE, False),
            (ProcessType.ATTACHMENT, False),
        ],
    )
    def test_only_the_inelastic_processes_carry_a_threshold(
        self, process: ProcessType, expected: bool
    ) -> None:
        assert process.has_threshold is expected

    def test_the_members_round_trip_through_their_manifest_spelling(self) -> None:
        assert ProcessType("CHARGE EXCHANGE") is ProcessType.CHARGE_EXCHANGE

    def test_momentum_transfer_is_either_elastic_or_effective(self) -> None:
        assert ProcessType.EFFECTIVE.is_momentum_transfer
        assert ProcessType.ELASTIC.is_momentum_transfer
        assert not ProcessType.IONIZATION.is_momentum_transfer


class TestTheSetContract:
    def test_an_empty_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CrossSectionSet(database="Phelps", sections=(), reference="")

    def test_a_set_reports_its_length(self) -> None:
        assert len(parse_lxcat(ION_EXPORT)) == 2

    def test_the_repr_names_the_database_and_the_processes(self) -> None:
        text = repr(parse_lxcat(ION_EXPORT))

        assert "Phelps" in text
        assert "2" in text
