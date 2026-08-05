"""Radiation trapping — doc 04 §2.3, verification item V-26 and the thick-line limits.

Three anchors here are not self-referential, which matters because an escape factor is a
quantity nobody tabulates and every internal consistency check would pass on a wrong one:

* :func:`test_monochromatic_escape_matches_exponential_integral` checks the closed form
  the module uses against a brute-force double quadrature over emitter position and
  emission angle written here, from the definition, sharing no code with it.
* :func:`test_lorentz_escape_reaches_its_analytic_thick_limit` checks the optically-thick
  constant ``4 / (3 sqrt(pi))``, derived in the module docstring, to eight digits.
* :func:`test_doppler_escape_converges_to_its_analytic_thick_limit` checks the
  corresponding Doppler limit, which is approached only logarithmically and is therefore
  asserted as a convergence rather than as an equality.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from scipy import special

from vpl.instruments.oes import escape as escape_module
from vpl.instruments.oes.escape import (
    LineProfileShape,
    TrappedLine,
    escape_factor,
    line_centre_optical_depth,
    slab_escape_probability,
)

# ── the independent reference ───────────────────────────────────────────────────


def _escape_probability_by_quadrature(
    tau_0: float, *, n_depth: int = 4001, n_mu: int = 2001
) -> float:
    """Mean single-flight escape probability of a uniform slab, from the definition.

    A photon born at optical depth ``t`` in a slab of total optical thickness ``tau_0``
    and emitted isotropically escapes with probability
    ``(1/2) integral_0^1 [exp(-t/mu) + exp(-(tau_0 - t)/mu)] dmu``; averaging over ``t``
    gives the slab mean. Written out longhand on purpose — it shares no code with
    :func:`slab_escape_probability`, which is the point of the comparison.
    """
    depth = np.linspace(0.0, tau_0, n_depth)
    mu = np.linspace(1.0 / n_mu, 1.0, n_mu)
    outward = np.exp(-depth[:, None] / mu[None, :])
    inward = np.exp(-(tau_0 - depth)[:, None] / mu[None, :])
    per_depth = 0.5 * np.trapezoid(outward + inward, mu, axis=1)
    return float(np.trapezoid(per_depth, depth) / tau_0)


# ── V-26: the analytic thin limit ───────────────────────────────────────────────


@pytest.mark.physics
@pytest.mark.parametrize("shape", list(LineProfileShape))
def test_escape_factor_tends_to_one_as_the_gas_thins(shape: LineProfileShape) -> None:
    """doc 04 §8 V-26: escape factor -> 1 as n_g -> 0."""
    assert escape_factor(1e-12, shape=shape) == pytest.approx(1.0, abs=1e-9)
    assert escape_factor(1e-6, shape=shape) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.physics
def test_escape_factor_is_exactly_one_at_zero_opacity() -> None:
    for shape in LineProfileShape:
        assert escape_factor(0.0, shape=shape) == 1.0


# ── the external check on the escape probability ────────────────────────────────


@pytest.mark.physics
@pytest.mark.parametrize("tau_0", [0.05, 0.5, 2.0, 10.0])
def test_monochromatic_escape_matches_exponential_integral(tau_0: float) -> None:
    """The closed form against a brute-force position/angle quadrature."""
    assert slab_escape_probability(tau_0) == pytest.approx(
        _escape_probability_by_quadrature(tau_0), rel=2e-4
    )


@pytest.mark.physics
def test_slab_escape_probability_is_the_published_closed_form() -> None:
    """``P = (1 - 2 E_3(tau)) / (2 tau)``, with E_3 from SciPy rather than from us."""
    tau = np.array([0.1, 1.0, 5.0, 50.0])
    expected = (1.0 - 2.0 * special.expn(3, tau)) / (2.0 * tau)
    assert slab_escape_probability(tau) == pytest.approx(expected, rel=1e-10)


@pytest.mark.physics
def test_slab_escape_probability_is_accurate_where_the_closed_form_cancels() -> None:
    """Below ~1e-6 the closed form is ``1 - 1`` in double precision; the series is not."""
    tau = 1e-12
    assert slab_escape_probability(tau) == pytest.approx(1.0, abs=1e-10)
    # The leading correction is -(tau/2)(psi(3) - ln tau); check the sign and the size.
    small = 1e-6
    predicted = 1.0 - 0.5 * small * (float(special.digamma(3)) - np.log(small))
    assert slab_escape_probability(small) == pytest.approx(predicted, rel=1e-9)


# ── the published asymptotics ───────────────────────────────────────────────────


@pytest.mark.physics
def test_lorentz_escape_reaches_its_analytic_thick_limit() -> None:
    """``Lambda sqrt(tau_0) -> 4 / (3 sqrt(pi))`` — derived in the module docstring.

    Eight digits, at three opacities two decades apart. This is the strongest check in
    the module: an error anywhere in the profile normalisation, the substitution, the
    panelling or the closed form for ``P`` moves this constant, and there is nothing
    inside the implementation that could conspire to produce it.
    """
    limit = 4.0 / (3.0 * np.sqrt(np.pi))
    for tau_0 in (1e8, 1e10, 1e12):
        measured = escape_factor(tau_0, shape=LineProfileShape.LORENTZ) * np.sqrt(tau_0)
        assert measured == pytest.approx(limit, rel=1e-7)


@pytest.mark.physics
def test_doppler_escape_converges_to_its_analytic_thick_limit() -> None:
    """``Lambda -> sqrt(ln tau_0) / (tau_0 sqrt(pi))``, approached as ``1 + O(1/ln tau_0)``.

    The approach is logarithmically slow, so what is asserted is that the ratio is
    converging monotonically towards one from above and is within a few percent of it by
    ``tau_0 = 1e12`` — which is the honest statement of a limit that is not reachable in
    double precision.
    """
    ratios = [
        escape_factor(tau_0, shape=LineProfileShape.DOPPLER)
        * tau_0
        * np.sqrt(np.pi)
        / np.sqrt(np.log(tau_0))
        for tau_0 in (1e4, 1e6, 1e8, 1e10, 1e12)
    ]
    assert all(r > 1.0 for r in ratios)
    assert np.all(np.diff(ratios) < 0.0)
    assert ratios[-1] == pytest.approx(1.0, rel=0.03)


@pytest.mark.physics
@pytest.mark.parametrize("shape", [LineProfileShape.DOPPLER, LineProfileShape.LORENTZ])
def test_the_profile_quadrature_is_converged(shape: LineProfileShape) -> None:
    """Doubling the nodes per panel must not move the answer.

    The panelled rule is the one place a smooth, plausible and entirely wrong number can
    come out — see :func:`vpl.instruments.oes.escape._panelled_rule`. Refuting that here
    is cheaper than discovering it in a line ratio.
    """
    reference = {tau: escape_factor(tau, shape=shape) for tau in (1.0, 1e3, 1e6, 1e9)}
    with mock.patch.object(escape_module, "PANEL_NODES", 2 * escape_module.PANEL_NODES):
        for tau, coarse in reference.items():
            assert escape_factor(tau, shape=shape) == pytest.approx(coarse, rel=1e-10)


@pytest.mark.physics
def test_wings_help_so_a_broadened_line_escapes_more_easily() -> None:
    """At equal line-centre opacity, a profile with wings traps less."""
    tau_0 = 1e3
    monochromatic = escape_factor(tau_0, shape=LineProfileShape.MONOCHROMATIC)
    doppler = escape_factor(tau_0, shape=LineProfileShape.DOPPLER)
    lorentz = escape_factor(tau_0, shape=LineProfileShape.LORENTZ)
    assert monochromatic < doppler < lorentz


# ── shape and contract ──────────────────────────────────────────────────────────


@pytest.mark.physics
@pytest.mark.parametrize("shape", list(LineProfileShape))
def test_escape_factor_decreases_monotonically_with_opacity(shape: LineProfileShape) -> None:
    opacities = np.logspace(-3.0, 5.0, 40)
    factors = np.array([escape_factor(float(t), shape=shape) for t in opacities])
    assert np.all(np.diff(factors) < 0.0)
    assert np.all((factors > 0.0) & (factors <= 1.0))


def test_escape_factor_rejects_a_negative_opacity() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        escape_factor(-1.0, shape=LineProfileShape.DOPPLER)


def test_slab_escape_probability_rejects_a_negative_optical_depth() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        slab_escape_probability(np.array([-0.1]))


# ── the opacity itself ──────────────────────────────────────────────────────────


def _resonance_line() -> TrappedLine:
    """Ar I 106.67 nm, one of the two resonance lines doc 04 §2.3 names.

    ``A_ul`` and the statistical weights are NIST ASD values for the 3p6 -> 3p5(2P3/2)4s
    transition; the width is the 300 K Doppler FWHM for argon.
    """
    return TrappedLine(
        wavelength_nm=106.67,
        a_ul_per_s=1.19e8,
        g_upper=3,
        g_lower=1,
        shape=LineProfileShape.DOPPLER,
        profile_fwhm_nm=2.09e-4,
    )


@pytest.mark.physics
def test_line_centre_optical_depth_is_linear_in_absorber_density_and_path() -> None:
    def tau(*, density: float, path: float) -> float:
        return line_centre_optical_depth(
            wavelength_nm=106.67,
            a_ul_per_s=1.19e8,
            g_upper=3,
            g_lower=1,
            profile_fwhm_nm=2.09e-4,
            shape=LineProfileShape.DOPPLER,
            lower_density_per_m3=density,
            path_length_m=path,
        )

    base = tau(density=1e20, path=0.4)
    assert tau(density=2e20, path=0.4) == pytest.approx(2.0 * base)
    assert tau(density=1e20, path=0.8) == pytest.approx(2.0 * base)


@pytest.mark.physics
def test_line_centre_optical_depth_is_undefined_for_a_monochromatic_line() -> None:
    with pytest.raises(ValueError, match="monochromatic"):
        line_centre_optical_depth(
            wavelength_nm=106.67,
            a_ul_per_s=1.19e8,
            g_upper=3,
            g_lower=1,
            profile_fwhm_nm=2.09e-4,
            shape=LineProfileShape.MONOCHROMATIC,
            lower_density_per_m3=1e20,
            path_length_m=0.4,
        )


@pytest.mark.physics
def test_ar_resonance_line_is_strongly_trapped_at_the_reference_point() -> None:
    """doc 04 §2.3: the 106.7 nm Ar I resonance line is "strongly self-absorbed".

    A ground-state density of 1.61e20 m^-3 — 5 mTorr at 300 K, RP-1 — over the 400 mm
    chamber has to come out optically thick by three orders of magnitude. This is the
    only test here with real atomic numbers in it, and it is the one that would catch a
    factor of 8 pi or a nanometre-for-metre slip that every dimensionless test passes.
    """
    line = _resonance_line()
    tau = line.optical_depth(lower_density_per_m3=1.61e20, path_length_m=0.4)
    assert 1e3 < tau < 1e4
    assert line.escape_factor(lower_density_per_m3=1.61e20, path_length_m=0.4) < 1e-3


@pytest.mark.physics
def test_a_near_infrared_line_on_a_thin_lower_level_is_essentially_free() -> None:
    """The 811.53 nm line terminates on the 1s5 metastable, not on the ground state.

    At a metastable density four orders below the neutral density the line is optically
    thin — which is why doc 02 §6.3 can use it — but only just: raising the metastable
    density to 1e15 m^-3 already brings the opacity to order unity, so ``Lambda_ul`` for
    this line is a function of the state and not a constant.
    """
    line = TrappedLine(
        wavelength_nm=811.53,
        a_ul_per_s=3.31e7,
        g_upper=7,
        g_lower=5,
        shape=LineProfileShape.DOPPLER,
        profile_fwhm_nm=1.59e-3,
    )
    assert line.escape_factor(lower_density_per_m3=1e12, path_length_m=0.4) > 0.99
    assert line.escape_factor(lower_density_per_m3=1e15, path_length_m=0.4) < 0.9


def test_trapped_line_rejects_a_non_positive_width() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrappedLine(
            wavelength_nm=811.53,
            a_ul_per_s=3.31e7,
            g_upper=7,
            g_lower=5,
            shape=LineProfileShape.DOPPLER,
            profile_fwhm_nm=0.0,
        )
