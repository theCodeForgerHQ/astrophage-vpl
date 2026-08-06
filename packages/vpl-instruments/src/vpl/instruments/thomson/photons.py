"""The photon budget, accumulation and stray-light background — doc 02 §7.1, §4.3, §11.

## The number that reshapes everything downstream (doc 02 §7.1)

    N_pe = (E_L / h*nu) . n_e . L . sigma_T . (Delta_Omega / 4*pi) . eta

At the reference operating point (RP-1, ``n_e`` = 1e17 m^-3) this evaluates to **0.1595
photoelectrons per shot** — doc 02 §7.1's table quotes 0.159 to three figures. It is
linear in ``n_e`` (:func:`photoelectrons_per_shot` is tested at all four of doc 02 §7.1's
table rows), which is the whole reason the *integrated* signal gives an absolute density:
sum enough shots and the total photoelectron count is a direct, calibratable proxy for
``n_e`` (Rayleigh calibration, §5 below, fixes the constant of proportionality).

**This is a single-photoelectron counting experiment, not a signal-averaging one** (doc 02
§7.1 consequence 1). :func:`relative_statistical_uncertainty` and
:func:`required_accumulation_s` make that arithmetic explicit: reaching 3 % on the
integrated RP-1 signal takes 6970 shots, ~697 s at the TS-L1 10 Hz repetition rate — the
"~700 s" doc 02 §7.1 and doc 06 §4 term 8 both quote.

## Thomson cannot follow transients (doc 02 §7.1 consequence 2)

Because reaching a *useful* count takes hundreds of seconds, a single shot or a few shots
carry essentially no information, and the accumulation must be phase-locked and repeated
over many RF cycles to resolve anything phase-dependent — multiplying the required shots
by the number of phase bins (doc 02 §10.3's ``N_phi`` = 16): "16 bins at 3 % requires
~111 000 shots, ~3.1 hours" (doc 02 §7.1). :func:`required_shots` and
:func:`required_accumulation_s` both take an ``n_phase_bins`` argument for exactly this,
and :data:`MINIMUM_EXPECTED_TOTAL_PHOTOELECTRONS` gives the absolute floor below which an
accumulation is refused outright rather than silently returned as a very noisy number —
see :mod:`vpl.instruments.thomson.instrument` for where that refusal happens (the
blindness has to be visible in the API, not left as a comment doc 07's robustness study
would never exercise).

## Stray light is not zero (doc 02 §4.3)

TS-3 requires >= 1e8 stray-light rejection. Doc 02 §4.3 enumerates the rejection stack
(:data:`BREWSTER_WINDOW_REJECTION` through :data:`POLARISATION_DISCRIMINATION_REJECTION`)
and is explicit that what gets through is not modelled as zero: "It is a parameter with a
nominal value and an uncertainty, and it appears in the error budget as a
background-subtraction systematic — because in a real Thomson system it is the dominant
one." :func:`combined_rejection_factor` computes the doc 02 §4.3 stack and
:func:`stray_light_spectrum_pe_per_shot` gives the residual pedestal that
:mod:`vpl.instruments.thomson.instrument` adds into the raw (pre-subtraction) signal.

**Honest flag on the pedestal's absolute size.** Doc 02 §4.3 states that a nominal value
and uncertainty exist; it does not give the pre-rejection baseline needed to compute one
from first principles (how much of a 0.5 J pulse geometrically couples into the collection
solid angle via window/wall scattering before any of the five rejection measures act is
not specified anywhere in docs 01-09, and inventing a coupling-fraction number to derive
one would be exactly the "parameter-fog" doc 00 warns against).
:data:`STRAY_LIGHT_PEDESTAL_PE_PER_SHOT` is instead pinned to the same order of magnitude
as the genuine signal at RP-1 — consistent
with doc 02 §4.3 calling it "the dominant" background-subtraction systematic, i.e. large
enough to matter rather than negligible — and the uncertainty on its *subtraction*
(:data:`STRAY_LIGHT_BACKGROUND_RELATIVE_UNCERTAINTY`) is doc 06 §4 term 11's booked 2 %.
This is a documented modelling choice, not a doc-sourced number; see the WBS 2.7 handoff
report.

## Rayleigh calibration uncertainty chain (doc 02 §7.3, §11; doc 06 §5)

Doc 06 §5 gives the chain explicitly: "Rayleigh cross section (2 %) -> gas purity (1 %) ->
pressure gauge (2 %) -> optical stability (3 %) -> photon statistics of the calibration
itself (5 %)". :data:`RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY` combines the five named
terms in quadrature (~6.56 %) rather than hard-coding doc 06 §4 term 3's rounded booked
figure of 7.0 %, so the budget stays auditable against its named sources — the largest
single term is the calibration's own photon statistics (5 %), which is the same
photon-starvation problem the measurement itself has, showing up again in how it is
calibrated.

Doc 06 §4.1 is explicit that this is a **correlated** systematic: "a single Rayleigh
calibration error affects *every* Thomson point identically and does **not** average
down." :mod:`vpl.instruments.thomson.instrument` therefore draws the calibration scale
once, in ``calibrate()``, on the :attr:`~vpl.core.random.Stream.CALIBRATION` stream, and
keeps it — never re-draws it per observation.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from vpl.core.constants import CLASSICAL_ELECTRON_RADIUS, PLANCK, SPEED_OF_LIGHT
from vpl.core.units import magnitude_in
from vpl.instruments.thomson.spectrum import LASER_WAVELENGTH_M, LASER_WAVELENGTH_NM

__all__ = [
    "BAFFLED_SNOUT_REJECTION",
    "BREWSTER_WINDOW_REJECTION",
    "CALIBRATION_PHOTON_STATISTICS_UNCERTAINTY",
    "COLLECTION_SOLID_ANGLE_SR",
    "COMBINED_TRANSMISSION_QE",
    "DEFAULT_TARGET_RELATIVE_UNCERTAINTY",
    "GAS_PURITY_UNCERTAINTY",
    "LASER_PULSE_ENERGY_J",
    "MINIMUM_EXPECTED_TOTAL_PHOTOELECTRONS",
    "OPTICAL_STABILITY_UNCERTAINTY",
    "POLARISATION_DISCRIMINATION_REJECTION",
    "PRESSURE_GAUGE_UNCERTAINTY",
    "RAYLEIGH_CALIBRATION_QUANTITY",
    "RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY",
    "RAYLEIGH_CROSS_SECTION_UNCERTAINTY",
    "RAYLEIGH_HORN_REJECTION",
    "REFERENCE_OPERATING_DENSITY_M3",
    "REPETITION_RATE_HZ",
    "SPATIAL_RESOLUTION_M",
    "STRAY_LIGHT_BACKGROUND_RELATIVE_UNCERTAINTY",
    "STRAY_LIGHT_PEDESTAL_PE_PER_SHOT",
    "STRAY_LIGHT_REJECTION_MARGIN",
    "STRAY_LIGHT_REJECTION_REQUIREMENT",
    "THOMSON_CROSS_SECTION_M2",
    "TRIPLE_GRATING_NOTCH_REJECTION",
    "combined_rejection_factor",
    "is_blind",
    "photoelectrons_per_shot",
    "relative_statistical_uncertainty",
    "required_accumulation_s",
    "required_shots",
    "shots_in_window",
    "stray_light_spectrum_pe_per_shot",
    "total_photoelectrons",
]

type FloatArray = NDArray[np.float64]

# ── the photon budget — doc 02 §7.1, TS-L1, TS-O2, TS-O3 ────────────────────────────

#: ``E_L`` — TS-L1.
LASER_PULSE_ENERGY_J: Final[float] = 0.5

#: ``L`` — the doc 02 §7.1 spatial resolution element along the beam, TS-O3.
SPATIAL_RESOLUTION_M: Final[float] = 90.0e-6

#: ``Delta_Omega`` as doc 02 §7.1's formula states it (the f/4 collection of TS-O2 rounds
#: to 0.049 sr; §7.1 uses 0.05 sr in the worked calculation this module reproduces).
COLLECTION_SOLID_ANGLE_SR: Final[float] = 0.05

#: ``eta`` — combined transmission x QE, doc 02 §7.1: "appropriate for a triple-grating
#: spectrometer with a Gen III ICCD".
COMBINED_TRANSMISSION_QE: Final[float] = 0.05

#: TS-L1: 10 Hz repetition rate.
REPETITION_RATE_HZ: Final[float] = 10.0

_CLASSICAL_ELECTRON_RADIUS_M: Final[float] = float(magnitude_in(CLASSICAL_ELECTRON_RADIUS, "m"))

#: ``sigma_T = (8/3) pi r_e^2``, the classical Thomson cross section, computed rather than
#: quoted so it stays tied to :data:`~vpl.core.constants.CLASSICAL_ELECTRON_RADIUS`'s
#: CODATA provenance. Evaluates to 6.652e-29 m^2, matching doc 02 §7.1's quoted value.
THOMSON_CROSS_SECTION_M2: Final[float] = (8.0 / 3.0) * math.pi * _CLASSICAL_ELECTRON_RADIUS_M**2

_PLANCK_J_S: Final[float] = float(magnitude_in(PLANCK, "J*s"))
_SPEED_OF_LIGHT_M_PER_S: Final[float] = float(magnitude_in(SPEED_OF_LIGHT, "m/s"))

#: ``h*nu`` at the TS-L1 laser line.
_PHOTON_ENERGY_J: Final[float] = _PLANCK_J_S * _SPEED_OF_LIGHT_M_PER_S / LASER_WAVELENGTH_M


def photoelectrons_per_shot(electron_density_m3: float) -> float:
    """``N_pe = (E_L/h*nu) . n_e . L . sigma_T . (Delta_Omega/4*pi) . eta`` — doc 02 §7.1.

    Linear in ``n_e``; reproduces doc 02 §7.1's table to three figures at each of its four
    rows (1e16 -> 0.016, 1e17 -> 0.159, 1e18 -> 1.60, 1e19 -> 16.0).
    """
    if electron_density_m3 < 0.0:
        raise ValueError(f"electron_density_m3 cannot be negative, got {electron_density_m3}")
    photons_per_pulse = LASER_PULSE_ENERGY_J / _PHOTON_ENERGY_J
    return (
        photons_per_pulse
        * electron_density_m3
        * SPATIAL_RESOLUTION_M
        * THOMSON_CROSS_SECTION_M2
        * (COLLECTION_SOLID_ANGLE_SR / (4.0 * math.pi))
        * COMBINED_TRANSMISSION_QE
    )


# ── accumulation and the "cannot follow transients" floor — doc 02 §7.1 consequence 2 ──

#: RP-1 — doc 01 §2.1 / doc 02 §3.3 regime B, the reference operating point every
#: doc 02 §7.1 worked number in this module is checked against.
REFERENCE_OPERATING_DENSITY_M3: Final[float] = 1.0e17

#: The doc 02 §7.1 table's standard benchmark: "shots for 3 %".
DEFAULT_TARGET_RELATIVE_UNCERTAINTY: Final[float] = 0.03

#: Below one total expected photoelectron across the whole spectrum, the Poisson process
#: this channel is (doc 02 §7.1 consequence 1) reports exactly zero counts with
#: probability ``exp(-1) ~ 37 %` and cannot be distinguished from a dark frame the rest of
#: the time either — doc 02 §7.1 consequence 2's "any single-shot or few-shot event is
#: invisible", pinned to the smallest accumulation this module will treat as an
#: observation rather than as nothing.
MINIMUM_EXPECTED_TOTAL_PHOTOELECTRONS: Final[float] = 1.0


def shots_in_window(duration_s: float) -> float:
    """Laser shots fired over ``duration_s`` at the TS-L1 10 Hz repetition rate."""
    if duration_s < 0.0:
        raise ValueError(f"duration_s cannot be negative, got {duration_s}")
    return duration_s * REPETITION_RATE_HZ


def total_photoelectrons(*, electron_density_m3: float, duration_s: float) -> float:
    """The doc 02 §7.1 "N_pe per shot (total)" summed over every shot in the window."""
    return photoelectrons_per_shot(electron_density_m3) * shots_in_window(duration_s)


def relative_statistical_uncertainty(*, electron_density_m3: float, duration_s: float) -> float:
    """``1/sqrt(N_pe)`` on the integrated signal — doc 06 §4 term 8's "3 % at 7000 shots".

    At RP-1 with a 696.7 s window this evaluates to ~0.03, matching doc 02 §7.1's "shots
    for 3 %" row and doc 06 §4 term 8's booked figure.
    """
    total = total_photoelectrons(electron_density_m3=electron_density_m3, duration_s=duration_s)
    if total <= 0.0:
        return math.inf
    return 1.0 / math.sqrt(total)


def required_shots(
    *,
    electron_density_m3: float,
    target_relative_uncertainty: float = DEFAULT_TARGET_RELATIVE_UNCERTAINTY,
    n_phase_bins: float = 1.0,
) -> float:
    """Laser shots needed to reach ``target_relative_uncertainty`` on the integrated signal.

    Args:
        electron_density_m3: The local ``n_e`` the photon budget is evaluated at.
        target_relative_uncertainty: The doc 02 §7.1 table's "3 %" or "1 %" columns, or
            any other target.
        n_phase_bins: Doc 02 §7.1 consequence 2: a phase-resolved accumulation shares its
            shots across this many RF phase bins, so reaching the same per-bin count needs
            this many times the unbinned shot count. ``1.0`` (the default) is the
            unbinned, absolute-window case.

    Raises:
        ValueError: If ``electron_density_m3`` is not positive (there is then no photon
            budget to invert) or ``target_relative_uncertainty`` is not positive.
    """
    if target_relative_uncertainty <= 0.0:
        raise ValueError(
            f"target_relative_uncertainty must be positive, got {target_relative_uncertainty}"
        )
    per_shot = photoelectrons_per_shot(electron_density_m3)
    if per_shot <= 0.0:
        raise ValueError(f"electron_density_m3 must be positive, got {electron_density_m3}")
    required_total_pe = 1.0 / target_relative_uncertainty**2
    return (required_total_pe / per_shot) * n_phase_bins


def required_accumulation_s(
    *,
    electron_density_m3: float,
    target_relative_uncertainty: float = DEFAULT_TARGET_RELATIVE_UNCERTAINTY,
    n_phase_bins: float = 1.0,
) -> float:
    """``required_shots(...) / REPETITION_RATE_HZ`` — the doc 02 §7.1 "wall-clock" column.

    At RP-1, 3 % and ``n_phase_bins=1`` this evaluates to ~696.7 s ("~700 s", doc 02 §7.1
    and doc 06 §4 term 8). At ``n_phase_bins=16`` (doc 02 §10.3's phase binning) it
    evaluates to ~11 147 s (~3.1 hours), matching doc 02 §7.1 consequence 2's "16 bins at
    3 % requires ~111 000 shots, ~3.1 hours".
    """
    shots = required_shots(
        electron_density_m3=electron_density_m3,
        target_relative_uncertainty=target_relative_uncertainty,
        n_phase_bins=n_phase_bins,
    )
    return shots / REPETITION_RATE_HZ


def is_blind(*, electron_density_m3: float, shots: float) -> bool:
    """Whether ``shots`` at ``electron_density_m3`` fall below the doc 02 §7.1 floor.

    ``True`` means the expected total photoelectron count is below
    :data:`MINIMUM_EXPECTED_TOTAL_PHOTOELECTRONS` — the regime doc 02 §7.1 consequence 2
    calls "invisible", not merely noisy.
    """
    return (
        photoelectrons_per_shot(electron_density_m3) * shots < MINIMUM_EXPECTED_TOTAL_PHOTOELECTRONS
    )


# ── stray light — doc 02 §4.3, TS-3 ─────────────────────────────────────────────────

#: TS-3's requirement.
STRAY_LIGHT_REJECTION_REQUIREMENT: Final[float] = 1.0e8

#: Doc 02 §4.3: "with 10x margin against the 1e8 requirement".
STRAY_LIGHT_REJECTION_MARGIN: Final[float] = 10.0

#: Brewster-angle entrance window — doc 02 §4.3.
BREWSTER_WINDOW_REJECTION: Final[float] = 10.0
#: Baffled entrance snout (3 knife-edge baffles) — doc 02 §4.3.
BAFFLED_SNOUT_REJECTION: Final[float] = 1.0e2
#: Rayleigh-horn viewing dump — doc 02 §4.3.
RAYLEIGH_HORN_REJECTION: Final[float] = 1.0e2
#: Triple-grating spectrometer notch at 532 nm — doc 02 §4.3, TS-S1.
TRIPLE_GRATING_NOTCH_REJECTION: Final[float] = 1.0e4
#: Polarisation discrimination — doc 02 §4.3: "Thomson signal is polarised; stray light
#: is less so".
POLARISATION_DISCRIMINATION_REJECTION: Final[float] = 5.0


def combined_rejection_factor() -> float:
    """The doc 02 §4.3 rejection stack multiplied together.

    Evaluates to ~5e9 against the TS-3 requirement of 1e8 — comfortably past doc 02 §4.3's
    "with 10x margin" (the individual factors are quoted as order-of-magnitude "~"
    figures, so the product's precise value is not meant to be read to more than one
    significant figure; what matters, and what is tested, is that it clears the
    requirement with margin at least as large as documented).
    """
    return (
        BREWSTER_WINDOW_REJECTION
        * BAFFLED_SNOUT_REJECTION
        * RAYLEIGH_HORN_REJECTION
        * TRIPLE_GRATING_NOTCH_REJECTION
        * POLARISATION_DISCRIMINATION_REJECTION
    )


#: Doc 06 §4 term 11: the background-subtraction systematic on Gamma_E. Used here as the
#: relative uncertainty on the *subtraction* of :data:`STRAY_LIGHT_PEDESTAL_PE_PER_SHOT`
#: (see the module docstring's honest flag on this figure's provenance).
STRAY_LIGHT_BACKGROUND_RELATIVE_UNCERTAINTY: Final[float] = 0.02

#: The residual stray-light pedestal at RP-1, pinned to the RP-1 signal level itself — see
#: the module docstring. Independent of ``n_e`` in :func:`stray_light_spectrum_pe_per_shot`
#: (unlike the genuine Thomson signal): stray light comes from the laser pulse scattering
#: off windows and chamber walls, not from the plasma, so it does not scale with density —
#: which is exactly why it is proportionally worse at low ``n_e`` than at RP-1 and above.
STRAY_LIGHT_PEDESTAL_PE_PER_SHOT: Final[float] = photoelectrons_per_shot(
    REFERENCE_OPERATING_DENSITY_M3
)


def stray_light_spectrum_pe_per_shot(axis_nm: FloatArray, *, notch_width_nm: float) -> FloatArray:
    """Per-channel residual stray-light photoelectrons, one shot's worth — doc 02 §4.3.

    Modelled as a Gaussian pedestal centred on
    :data:`~vpl.instruments.thomson.spectrum.LASER_WAVELENGTH_NM` rather than a flat
    background across every TS-S3 channel: the dominant leakage path is
    the triple-grating notch's imperfect rejection *at* 532 nm (TS-S1), so what gets
    through is concentrated near the laser line rather than spread evenly across the 25 nm
    TS-S2 window.

    Args:
        axis_nm: Detector channel centre wavelengths, nm.
        notch_width_nm: Characteristic width of the residual notch leakage. The caller
            (:mod:`vpl.instruments.thomson.instrument`) passes the detector's own channel
            spacing, since doc 02 does not specify a notch leakage width independently and
            the channel spacing is the finest scale the detector can resolve regardless.
    """
    offset = np.asarray(axis_nm, dtype=np.float64) - LASER_WAVELENGTH_NM
    shape = np.exp(-((offset / notch_width_nm) ** 2))
    return STRAY_LIGHT_PEDESTAL_PE_PER_SHOT * shape


# ── Rayleigh calibration uncertainty chain — doc 02 §7.3, §11; doc 06 §5 ────────────

#: The quantity name :meth:`~vpl.core.protocols.CalibrationSet.for_quantity` looks up for
#: the Rayleigh standard — doc 02 §11's "Rayleigh scattering in Ar" row.
RAYLEIGH_CALIBRATION_QUANTITY: Final[str] = "rayleigh_scattering"

#: Doc 06 §5 chain, term 1: Rayleigh cross-section literature value.
RAYLEIGH_CROSS_SECTION_UNCERTAINTY: Final[float] = 0.02
#: Doc 06 §5 chain, term 2: calibration-gas purity.
GAS_PURITY_UNCERTAINTY: Final[float] = 0.01
#: Doc 06 §5 chain, term 3: pressure gauge accuracy.
PRESSURE_GAUGE_UNCERTAINTY: Final[float] = 0.02
#: Doc 06 §5 chain, term 4: optical stability of the calibration setup.
OPTICAL_STABILITY_UNCERTAINTY: Final[float] = 0.03
#: Doc 06 §5 chain, term 5, and the dominant one: photon statistics of the calibration
#: measurement itself — the same photon-starvation problem the science measurement has.
CALIBRATION_PHOTON_STATISTICS_UNCERTAINTY: Final[float] = 0.05

#: The five doc 06 §5 terms combined in quadrature (~6.56 %); doc 06 §4 term 3 books the
#: rounded figure as 7.0 %. Computed rather than hard-coded so the chain stays auditable.
RAYLEIGH_CALIBRATION_RELATIVE_UNCERTAINTY: Final[float] = math.sqrt(
    RAYLEIGH_CROSS_SECTION_UNCERTAINTY**2
    + GAS_PURITY_UNCERTAINTY**2
    + PRESSURE_GAUGE_UNCERTAINTY**2
    + OPTICAL_STABILITY_UNCERTAINTY**2
    + CALIBRATION_PHOTON_STATISTICS_UNCERTAINTY**2
)
