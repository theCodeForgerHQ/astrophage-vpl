"""``OesInstrument`` against the doc 08 §4 contract — doc 04 §9.

The two contract properties doc 04 §9 calls out by name, and which only a test can hold:

* ``forward`` and ``observe`` come from **one code path**. Asserted by switching the noise
  and the calibration off and requiring the two to agree bit for bit.
* ``forward`` returns the *noiseless* expectation. Asserted by averaging many ``observe``
  draws onto it.

Plus the doc 08 §1 principle 5 layering check: this package must not reach into
``vpl.inverse``, which is asserted against the import graph rather than left to review.
"""

from __future__ import annotations

import numpy as np
import pytest

from oes_system import energy_grid, plain_system
from vpl.core.protocols.config import InstrumentConfig
from vpl.core.protocols.instrument import (
    CalibrationReference,
    CalibrationSet,
    Instrument,
)
from vpl.core.state import (
    AcquisitionWindow,
    CalibrationState,
    Fidelity,
    Measurement,
    Observable,
    PlasmaParams,
    PlasmaState,
    ScalarField,
    SpatialGrid,
    Species,
)
from vpl.core.units import Q_
from vpl.instruments.oes.cr import CollisionalRadiativeModel
from vpl.instruments.oes.instrument import MaxwellianEedf, OesInstrument
from vpl.instruments.oes.spectrograph import Spectrograph
from vpl.physics.eedf.grid import EnergyGrid

ROOT_SEED = 20260805
N_POINTS = 8


@pytest.fixture(scope="module")
def grid() -> EnergyGrid:
    return energy_grid()


def build_instrument(grid: EnergyGrid, **overrides: object) -> OesInstrument:
    model = CollisionalRadiativeModel(
        system=plain_system(grid), grid=grid, wall_loss_per_s={"m": 1.0e4}
    )
    kwargs: dict[str, object] = {
        "model": model,
        "spectrograph": Spectrograph.from_registry(),
        "eedf": MaxwellianEedf(grid=grid),
        "centre_wavelength_nm": 811.53,
        "root_seed": ROOT_SEED,
    }
    kwargs.update(overrides)
    return OesInstrument(**kwargs)  # type: ignore[arg-type]


def plasma_state(*, n_0_per_m3: float = 1e17, t_e_ev: float = 3.0) -> PlasmaState:
    spatial = SpatialGrid.uniform(length=Q_(2.0, "mm"), n_points=N_POINTS)
    params = PlasmaParams(
        species=Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1),
        n_0=Q_(n_0_per_m3, "m**-3"),
        T_e=Q_(t_e_ev, "eV"),
        T_i=Q_(0.05, "eV"),
        T_g=Q_(300.0, "K"),
        pressure=Q_(5.0, "mTorr"),
        bias=Q_(-250.0, "V"),
        gamma_se=0.1,
        kappa=1.0,
    )
    # A crude sheath: density rising from the wall, T_e flat. Nothing here is a physics
    # claim; the instrument is being tested, not the sheath.
    profile = n_0_per_m3 * (0.1 + 0.9 * np.linspace(0.0, 1.0, N_POINTS))
    fields = {
        "n_e": ScalarField(name="n_e", values=profile, units="m**-3", grid=spatial, time=None),
        "n_i": ScalarField(name="n_i", values=profile, units="m**-3", grid=spatial, time=None),
        "Phi": ScalarField(
            name="Phi",
            values=np.linspace(-250.0, 0.0, N_POINTS),
            units="V",
            grid=spatial,
            time=None,
        ),
        "T_e": ScalarField(
            name="T_e",
            values=np.full(N_POINTS, t_e_ev),
            units="eV",
            grid=spatial,
            time=None,
        ),
    }
    return PlasmaState(
        params=params,
        grid=spatial,
        time=None,
        fields=fields,
        ion_distribution=None,
        fidelity=Fidelity.L0,
    )


def window() -> AcquisitionWindow:
    return AcquisitionWindow.absolute(start=Q_(0.0, "s"), duration=Q_(2.0, "ns"))


def references() -> CalibrationSet:
    return CalibrationSet.of(
        CalibrationReference(
            name="NIST FEL tungsten-halogen lamp",
            quantity="absolute_radiometric",
            value=Q_(1.0, "dimensionless"),
            relative_uncertainty=0.06,
            traceable_to="NIST FEL scale",
        )
    )


# ── the doc 08 §4 contract ──────────────────────────────────────────────────────


def test_the_instrument_satisfies_the_protocol(grid: EnergyGrid) -> None:
    assert isinstance(build_instrument(grid), Instrument)


def test_metadata_carries_a_citation_and_a_detection_floor(grid: EnergyGrid) -> None:
    metadata = build_instrument(grid).metadata()
    assert metadata.instrument_id == "oes"
    assert metadata.citations
    assert metadata.detection_floor.quantity == "n_0"


def test_configure_applies_a_manifest_block(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    instrument.configure(
        InstrumentConfig(values={"centre_wavelength_nm": 750.39, "chord_length_m": 0.2})
    )
    assert instrument.centre_wavelength_nm == 750.39
    assert instrument.chord_length_m == 0.2


def test_configure_refuses_a_wavelength_the_grating_cannot_reach(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    with pytest.raises(ValueError, match="cannot diffract"):
        instrument.configure(InstrumentConfig(values={"centre_wavelength_nm": 2000.0}))


# ── forward ─────────────────────────────────────────────────────────────────────


@pytest.mark.physics
def test_forward_returns_one_spectrum_per_grid_point(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    observable = instrument.forward(plasma_state(), window())
    assert isinstance(observable, Observable)
    assert observable.shape == (N_POINTS, instrument.spectrograph.n_pixels)
    assert observable.units == "W / m**2 / sr / nm"
    assert np.all(observable.values >= 0.0)


@pytest.mark.physics
def test_forward_is_deterministic(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    state, acquisition = plasma_state(), window()
    assert np.array_equal(
        instrument.forward(state, acquisition).values,
        instrument.forward(state, acquisition).values,
    )


@pytest.mark.physics
def test_brighter_plasma_gives_more_signal(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    dim = instrument.forward(plasma_state(n_0_per_m3=1e16), window()).values
    bright = instrument.forward(plasma_state(n_0_per_m3=1e17), window()).values
    assert float(bright.sum()) > float(dim.sum())


@pytest.mark.physics
def test_the_line_sits_where_the_wavelength_axis_says_it_does(grid: EnergyGrid) -> None:
    """The 811.53 nm analogue must land on the pixel nearest 811.53 nm."""
    instrument = build_instrument(grid)
    spectrum = instrument.forward(plasma_state(), window()).values[-1]
    axis = instrument.wavelength_axis()
    assert axis[int(np.argmax(spectrum))] == pytest.approx(811.53, abs=0.01)


# ── observe, and doc 04 §9's shared code path ───────────────────────────────────


@pytest.mark.physics
def test_observe_and_forward_agree_exactly_with_noise_and_calibration_off(
    grid: EnergyGrid,
) -> None:
    """Doc 04 §9: "**Both come from the same code path**".

    With the switchable stages off, the two must be *identical*, not merely close. Any
    difference at all means there are two code paths, which is the bug class doc 04 §9
    says "would silently invalidate every result".
    """
    instrument = build_instrument(grid)
    instrument.set_noise_enabled(False)
    instrument.calibrate(references())
    instrument.use_true_calibration()

    state, acquisition = plasma_state(), window()
    assert np.array_equal(
        instrument.observe(state, acquisition).values,
        instrument.forward(state, acquisition).values,
    )


@pytest.mark.physics
def test_observe_is_noisy_and_carries_a_per_sample_uncertainty(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    instrument.calibrate(references())
    measurement = instrument.observe(plasma_state(), window())

    assert isinstance(measurement, Measurement)
    assert measurement.calibration is CalibrationState.ESTIMATED
    assert measurement.shape == (N_POINTS, instrument.spectrograph.n_pixels)
    assert not np.array_equal(
        measurement.values, instrument.forward(plasma_state(), window()).values
    )
    assert np.all(measurement.uncertainty >= 0.0)
    assert np.any(measurement.uncertainty > 0.0)


@pytest.mark.physics
@pytest.mark.slow
def test_the_noise_averages_down_onto_the_noiseless_forward(grid: EnergyGrid) -> None:
    """Doc 04 §9: ``forward`` is the noiseless *expectation* of ``observe``.

    Averaged with the calibration held true, so that what is being tested is the photon
    statistics and not the calibration bias — which by doc 04 §7.3 is drawn once per
    instrument and does *not* average down, and is the reason the two stages are separate.

    Accumulated, because a single 2 ns gate on this plasma delivers **0.97 photoelectrons
    in the brightest pixel** — a number worth knowing, and the reason doc 02 §10.3
    accumulates over RF cycles at all. At one count per pixel the sample mean of 120 draws
    has a 9 % standard error and the test would be measuring its own statistics.
    """
    instrument = build_instrument(grid, accumulations=2000)
    instrument.calibrate(references())
    instrument.use_true_calibration()

    state, acquisition = plasma_state(), window()
    expected = instrument.forward(state, acquisition).values
    draws = np.stack([instrument.observe(state, acquisition).values for _ in range(120)])
    mean = draws.mean(axis=0)

    bright = expected > 0.05 * expected.max()
    assert np.allclose(mean[bright], expected[bright], rtol=0.05)


@pytest.mark.physics
def test_observe_is_reproducible_from_the_seed(grid: EnergyGrid) -> None:
    """Doc 00 E3. Two instruments with the same root seed draw the same noise."""
    state, acquisition = plasma_state(), window()
    first, second = build_instrument(grid), build_instrument(grid)
    for instrument in (first, second):
        instrument.calibrate(references())
    assert np.array_equal(
        first.observe(state, acquisition).values, second.observe(state, acquisition).values
    )


# ── calibration — doc 04 §7.3 ───────────────────────────────────────────────────


def test_calibration_is_estimated_and_carries_the_doc_02_uncertainty(
    grid: EnergyGrid,
) -> None:
    calibration = build_instrument(grid).calibrate(references())
    assert calibration.state is CalibrationState.ESTIMATED
    assert calibration.is_inverse_crime is False
    assert calibration.relative_uncertainty["radiance_scale"] == pytest.approx(0.06)
    assert calibration.coefficients["radiance_scale"] != 1.0


def test_calibration_needs_a_radiometric_standard(grid: EnergyGrid) -> None:
    empty = CalibrationSet(references=())
    with pytest.raises(KeyError, match="absolute_radiometric"):
        build_instrument(grid).calibrate(empty)


def test_observing_before_calibrating_is_refused(grid: EnergyGrid) -> None:
    with pytest.raises(RuntimeError, match="calibrate"):
        build_instrument(grid).observe(plasma_state(), window())


# ── the likelihood and the detection gate ───────────────────────────────────────


@pytest.mark.physics
def test_the_likelihood_peaks_at_the_truth(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    instrument.calibrate(references())
    state, acquisition = plasma_state(), window()
    measurement = instrument.observe(state, acquisition)

    truth = instrument.likelihood(measurement, instrument.forward(state, acquisition))
    wrong = instrument.likelihood(
        measurement, instrument.forward(plasma_state(t_e_ev=6.0), acquisition)
    )
    assert truth > wrong


@pytest.mark.physics
def test_the_likelihood_is_finite_and_negative(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    instrument.calibrate(references())
    state, acquisition = plasma_state(), window()
    value = instrument.likelihood(
        instrument.observe(state, acquisition), instrument.forward(state, acquisition)
    )
    assert np.isfinite(value)


def test_the_likelihood_refuses_a_mismatched_prediction(grid: EnergyGrid) -> None:
    instrument = build_instrument(grid)
    instrument.calibrate(references())
    state, acquisition = plasma_state(), window()
    measurement = instrument.observe(state, acquisition)
    truncated = Observable(
        instrument_id="oes",
        values=np.ones(3),
        units="W / m**2 / sr / nm",
        window=acquisition,
    )
    with pytest.raises(ValueError, match="same shape"):
        instrument.likelihood(measurement, truncated)


def test_the_detection_floor_gates_the_channel_off(grid: EnergyGrid) -> None:
    """doc 01 IF-6 applied to OES: below the floor the channel is absent, not weak."""
    instrument = build_instrument(grid)
    assert instrument.is_informative(plasma_state(n_0_per_m3=1e17).params)
    assert not instrument.is_informative(plasma_state(n_0_per_m3=1e13).params)


# ── doc 08 §1 principle 5 ───────────────────────────────────────────────────────


def test_the_instrument_layer_does_not_import_the_inverse_layer() -> None:
    """ "Layers do not leak" — doc 08 §1 principle 5, doc 04 §1.

    An instrument reads a state and produces an observable. If it could see the inverse
    problem it could be tuned to it, and the closed-loop validation of doc 07 §1 would be
    measuring the tuning rather than the physics.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "vpl" / "instruments"
    forbidden = {"vpl.inverse", "vpl.uq", "vpl.validation"}
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders.extend(
                f"{path.name}: {name}"
                for name in names
                if any(name == bad or name.startswith(f"{bad}.") for bad in forbidden)
            )
    assert offenders == []
