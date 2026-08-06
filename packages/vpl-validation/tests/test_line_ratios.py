"""Argon spectroscopic data — doc 04 §8 V-24, "CR model vs published Ar line ratios".

V-24 is the one item in doc 04 §8's table with no test anywhere in the project. Every
other CR-model check — ``test_cr.py``'s Klein-Rosseland identity, the corona and LTE
limits of V-25, the escape-factor limit of V-26 — is an analytic identity the *model* was
built to satisfy. None of them touches argon: ``vpl-instruments/tests/oes_system.py``
says so explicitly ("The level system here is **not argon**"), and the package README
lists V-24 as the one unverified line in its table.

This module is the external half doc 09 §1 calls PUBLISHED, in the style of
``vpl.validation.swarm``: transcribed values with a full citation, not vendored bulk data
(doc 09 §5). Two independent things are transcribed here, both real and both
independently checkable against their source:

1. **NIST ASD level energies and degeneracies** for the Ar I ``1s`` (Paschen, ``3p^5 4s``)
   and ``2p`` (Paschen, ``3p^5 4p``) manifolds — fetched live from
   ``physics.nist.gov/cgi-bin/ASD/energy1.pl`` during this work. These are *real* argon
   atomic structure, unlike every level system elsewhere in this project.
2. **Zheng, Wu, Cao, Zhang and Huang (2020)**, arXiv:2010.10714 — the accuracy (vs an
   independent Langmuir probe) of an 18-level Ar collisional-radiative model driven by
   real OES line intensities. Read directly from the paper's §4.1, not a search snippet.

## What could not be obtained, and is not faked

A literal published table of *line-intensity-ratio vs T_e* — the "811.53/763.51 nm vs
T_e" curve doc 04 §8 names — could not be transcribed. Three primary sources were
checked and are all paywalled with no accessible full text this session:

* J B Boffard, C C Lin and C A DeJoseph, "Application of excitation cross sections to
  optical plasma diagnostics", J. Phys. D 37 (2004) R143-R161.
* J B Boffard, R O Jung, C C Lin and A E Wendt, Plasma Sources Sci. Technol. 19 (2010)
  065001.
* X M Zhu and Y K Pu, "Optical emission spectroscopy in low-temperature plasmas
  containing argon and nitrogen...", J. Phys. D 43 (2010) 403001.

NIST ASD's own "Lines" query (which would supply the ``A_ul`` values needed to compute a
ratio from the levels below) returned "Invalid Input" for every parameter combination
tried against ``physics.nist.gov/cgi-bin/ASD/lines1.pl`` in this session — only the
"Levels" endpoint (``energy1.pl``) could be queried successfully. Rather than substitute
a remembered ``A_ul`` for an unverified one, none is transcribed. ``TestTheV24Comparison``
below is where this gap becomes a test result rather than a silent omission.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.instruments.oes import (
    CollisionalRadiativeModel,
    ElectronImpactChannel,
    Level,
    LevelSystem,
    RadiativeChannel,
)
from vpl.physics.eedf.analytic import maxwellian_eedf
from vpl.physics.eedf.grid import EnergyGrid
from vpl.validation.line_ratios import (
    ArgonLevel,
    SpectroscopicAccuracyBenchmark,
    argon_level,
    published_accuracy_benchmarks,
    published_argon_levels,
    transition_wavelength_nm,
)

# ── the real argon levels ────────────────────────────────────────────────────────


class TestPublishedArgonLevels:
    def test_the_ground_state_sits_at_zero(self) -> None:
        assert argon_level("ground").energy_ev == 0.0
        assert argon_level("ground").degeneracy == 1

    def test_the_four_1s_levels_are_present_in_ascending_energy(self) -> None:
        # The textbook ordering: 1s5 (metastable) < 1s4 (resonant) < 1s3 (metastable)
        # < 1s2 (resonant). Getting this order wrong is the kind of transcription error
        # doc 09 §1 exists to catch before it reaches a CR model.
        energies = [argon_level(label).energy_ev for label in ("1s5", "1s4", "1s3", "1s2")]

        assert energies == sorted(energies)

    def test_the_metastables_are_flagged_by_role(self) -> None:
        assert argon_level("1s5").role == "metastable"
        assert argon_level("1s3").role == "metastable"

    def test_every_degeneracy_is_two_j_plus_one(self) -> None:
        # g = 2J + 1 is a physics identity, not a transcribed number. Checking it against
        # the independently-transcribed J and g columns is a check on the transcription
        # itself, in the spirit of the h*nu = E_k - E_i check vpl.physics.atomic.nist
        # applies to NIST "Lines" rows.
        for level in published_argon_levels():
            assert level.degeneracy == pytest.approx(2 * level.j + 1)

    def test_asking_for_an_unknown_level_raises_rather_than_returning_nothing(self) -> None:
        with pytest.raises(KeyError, match="2p3"):
            argon_level("2p3")

    def test_every_level_carries_a_citation(self) -> None:
        for level in published_argon_levels():
            assert level.citation
            assert "NIST" in level.citation

    def test_the_type_is_immutable(self) -> None:
        level: ArgonLevel = argon_level("1s5")
        with pytest.raises(AttributeError):
            level.energy_ev = 0.0  # type: ignore[misc]


class TestTransitionWavelengthsReproduceTheKnownLines:
    """``lambda = hc / (E_upper - E_lower)`` from the transcribed levels alone.

    This is the one place independent verification is possible without an ``A_ul``: the
    well-known positions of the three classic Ar I diagnostic lines (811.53, 763.51,
    750.39 nm) are reproduced purely from the NIST *level* energies transcribed above,
    with no line-list data at all. Agreement to a few tenths of a nanometre is exactly
    the size of the air/vacuum offset ``vpl.physics.atomic.nist`` documents for a real
    ASD row — not agreement to 8 decimal places, which would be suspicious.
    """

    def test_2p9_to_1s5_lands_near_811_53_nm(self) -> None:
        wavelength = transition_wavelength_nm(argon_level("2p9"), argon_level("1s5"))
        assert wavelength == pytest.approx(811.53, abs=0.5)

    def test_2p6_to_1s5_lands_near_763_51_nm(self) -> None:
        wavelength = transition_wavelength_nm(argon_level("2p6"), argon_level("1s5"))
        assert wavelength == pytest.approx(763.51, abs=0.5)

    def test_2p1_to_1s2_lands_near_750_39_nm(self) -> None:
        wavelength = transition_wavelength_nm(argon_level("2p1"), argon_level("1s2"))
        assert wavelength == pytest.approx(750.39, abs=0.5)

    def test_a_downward_pair_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="above"):
            transition_wavelength_nm(argon_level("1s5"), argon_level("2p9"))


# ── the accuracy benchmark ───────────────────────────────────────────────────────


class TestPublishedAccuracyBenchmarks:
    def test_three_calibration_points_are_transcribed(self) -> None:
        assert len(published_accuracy_benchmarks()) == 3

    def test_every_row_names_its_citation_and_calibration(self) -> None:
        for row in published_accuracy_benchmarks():
            assert row.citation
            assert "Zheng" in row.citation
            assert row.calibration

    def test_the_maximum_error_is_never_smaller_than_the_average(self) -> None:
        # A sanity check on the transcription itself: Zheng et al report both a maximum
        # and an average error over their power sweep, and the maximum cannot be less
        # than the average of the same set.
        for row in published_accuracy_benchmarks():
            assert row.max_t_e_error >= row.avg_t_e_error
            assert row.max_n_e_error >= row.avg_n_e_error

    def test_the_type_is_immutable(self) -> None:
        row: SpectroscopicAccuracyBenchmark = published_accuracy_benchmarks()[0]
        with pytest.raises(AttributeError):
            row.avg_t_e_error = 0.0  # type: ignore[misc]


# ── V-24 itself ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def grid() -> EnergyGrid:
    return EnergyGrid.linear(max_ev=60.0, n_cells=600)


def _born_like_cross_section(
    energy_ev: np.ndarray, *, threshold_ev: float, peak_m2: float
) -> np.ndarray:
    """A threshold-correct, monotone cross-section shape.

    Synthetic, exactly as ``vpl-instruments/tests/oes_system.py`` uses for its own
    "argon-shaped, not argon" system, and for the same reason: doc 09 §5 keeps bulk
    cross-section tables out of the repository, so nothing here is claimed to be a real
    argon cross section. What *is* real, for the first time in this project's CR-model
    tests, is the level structure it excites.
    """
    above = energy_ev > threshold_ev
    return np.where(above, peak_m2 * (1.0 - threshold_ev / np.where(above, energy_ev, 1.0)), 0.0)


@pytest.fixture(scope="module")
def real_argon_partial_system(grid: EnergyGrid) -> LevelSystem:
    """Ground, 1s5 and 2p9 — three *real* NIST energies, still-synthetic cross sections.

    This is as far as a real argon level system can be assembled from what this research
    session could independently verify: two transitions' worth of level structure
    (``g -> 1s5`` populates the metastable, ``1s5 -> 2p9`` is the metastable-coupled
    excitation route doc 04 §2.3 names for the 811.53 nm line) and one radiative channel,
    whose ``A_ul`` is *not* independently re-verified this session — see the module
    docstring and ``TestTheV24Comparison`` below.
    """
    ground = argon_level("ground")
    metastable = argon_level("1s5")
    upper = argon_level("2p9")

    g_to_m_threshold = metastable.energy_ev - ground.energy_ev
    m_to_u_threshold = upper.energy_ev - metastable.energy_ev

    levels = (
        Level(label="ground", energy_ev=0.0, degeneracy=ground.degeneracy),
        Level(
            label="1s5",
            energy_ev=metastable.energy_ev,
            degeneracy=metastable.degeneracy,
            is_metastable=True,
        ),
        Level(label="2p9", energy_ev=upper.energy_ev, degeneracy=upper.degeneracy),
    )
    electron_impact = (
        ElectronImpactChannel.from_sampler(
            grid,
            lower="ground",
            upper="1s5",
            threshold_ev=g_to_m_threshold,
            sampler=lambda e: _born_like_cross_section(
                e, threshold_ev=g_to_m_threshold, peak_m2=3.0e-21
            ),
        ),
        ElectronImpactChannel.from_sampler(
            grid,
            lower="1s5",
            upper="2p9",
            threshold_ev=m_to_u_threshold,
            sampler=lambda e: _born_like_cross_section(
                e, threshold_ev=m_to_u_threshold, peak_m2=5.0e-20
            ),
        ),
    )
    radiative = (
        RadiativeChannel(
            upper="2p9",
            lower="1s5",
            # 3.31e7 /s is the widely-published Ar I 2p9 -> 1s5 Einstein coefficient
            # (already used for this same transition in
            # vpl-instruments/tests/oes_system.py). It is used here as an *inherited*
            # textbook value, not one this session re-verified against a live NIST ASD
            # "Lines" query — that query returned "Invalid Input" for every parameter
            # combination tried. See the module docstring.
            a_ul_per_s=3.31e7,
            wavelength_nm=transition_wavelength_nm(upper, metastable),
        ),
    )
    return LevelSystem(levels=levels, electron_impact=electron_impact, radiative=radiative)


class TestTheV24Comparison:
    """doc 04 §8 V-24, attempted honestly.

    This is not a skip. The model is actually run, on real NIST argon level energies, at
    an electron temperature inside the range Zheng et al (2020) report for the discharge
    their CRM was validated against. The run succeeds and produces a positive population
    for the real 811.53 nm upper level — proof that a real-argon CR run is now possible
    for the first time in this project. What blocks V-24 is the next step: turning that
    single line into a *ratio*, which needs a second real diagnostic line's Einstein
    coefficient (750.39 or 763.51 nm), and none could be independently verified this
    session (see the module docstring). Without a second line there is no dimensionless
    ratio to compare against any published value, so the comparison the doc 04 §8 pass
    criterion asks for ("within published scatter") cannot be completed, and this is
    reported as an ``xfail`` rather than papered over with a fabricated second
    coefficient.
    """

    @pytest.mark.physics
    def test_the_real_argon_system_solves_and_the_gap_that_remains(
        self, grid: EnergyGrid, real_argon_partial_system: LevelSystem
    ) -> None:
        # T_e = 3.5 eV sits inside the 2.4-8.1 eV range Zheng et al (2020) report across
        # their triple-frequency Ar CCP conditions (Figs 2(b), 3(b), 4(b), 5(b)).
        t_e_ev = 3.5
        f0 = grid.normalise(maxwellian_eedf(grid.centres_ev, electron_temperature_ev=t_e_ev))
        # n_g = 1.6e20 m^-3 and n_e = 1e17 m^-3 match the reference argon density and a
        # mid-envelope electron density already used throughout
        # vpl-instruments/tests/test_cr.py, so this run sits in the same regime the
        # rest of the CR-model test suite exercises.
        model = CollisionalRadiativeModel(
            system=real_argon_partial_system, grid=grid, wall_loss_per_s={"1s5": 1.0e4}
        )

        populations = model.solve(
            electron_density_per_m3=1.0e17, ground_density_per_m3=1.6e20, f0=f0
        )

        # The real, independently-anchored half of this test: a CR run at genuine argon
        # level energies produces a physically sensible (positive, finite) population for
        # the real 811.53 nm upper level.
        assert populations["2p9"] > 0.0
        assert np.isfinite(populations["2p9"])
        line_811_intensity = populations["2p9"] * 3.31e7  # proportional to eps_ul

        # The classic diagnostic Zheng et al (and Boffard, and Zhu & Pu) actually report
        # is a *ratio* of two lines — 750.39/811.53 or 763.51/811.53 nm — because a
        # single line's absolute intensity depends on n_g and the optical calibration in
        # a way a ratio does not. Forming it needs a second A_ul this session could not
        # verify (module docstring). No fabricated value is substituted.
        missing_companion_lines = ("2p1 -> 1s2 (750.39 nm)", "2p6 -> 1s5 (763.51 nm)")
        assert line_811_intensity > 0.0  # the half of the ratio that *can* be computed

        pytest.xfail(
            "doc 04 §8 V-24 is not closed by this test. A CR run on real NIST argon "
            f"level energies now works (see the assertions above), but forming either "
            f"published ratio needs an independently-verified A_ul for one of "
            f"{missing_companion_lines}, and none could be obtained: NIST ASD's "
            "'Lines' endpoint returned 'Invalid Input' for every query tried this "
            "session, and the three primary papers with tabulated ratio-vs-T_e data "
            "(Boffard 2004 J. Phys. D 37 R143; Boffard 2010 PSST 19 065001; Zhu & Pu "
            "2010 J. Phys. D 43 403001) are paywalled with no accessible full text. "
            "Closing V-24 needs either a working NIST ASD 'Lines' fetch or a "
            "transcribed A_ul table from one of those three papers."
        )
