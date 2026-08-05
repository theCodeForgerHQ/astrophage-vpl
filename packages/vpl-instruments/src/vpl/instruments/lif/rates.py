"""The LIF rate equations and saturation — doc 04 §3.1 and §3.4.

Doc 04 §3.1 writes the model this module implements::

    dn_2/dt  =  n_1 B_12 rho(nu)  -  n_2 ( B_21 rho(nu) + A_21 + A_23 + Q )

and doc 04 §3.4 states the requirement that shapes it:

    **The framework deliberately supports operating in the saturated regime and models
    the resulting distortion**, because real LIF experiments often do, and a framework
    that only handles the linear regime would fail on real data.

## Where saturation actually has to be applied, and the error that is easy to make

The tempting implementation computes a Doppler-broadened absorption profile and multiplies
it by one scalar saturation factor. That is wrong, and wrong in a way that looks right: a
scalar factor changes the amplitude and leaves the **width** alone, so the fitted ``T_i``
comes out correct and the model appears to work.

Saturation is a property of a *velocity class*, not of a line. An ion detuned into
resonance sees the full laser intensity and saturates; an ion two Doppler widths away sees
almost nothing and stays linear. The excited fraction is therefore evaluated
detuning-by-detuning, before any integral over the distribution::

    n_2/n (delta)  =  [g_2/(g_1+g_2)] . S L(delta) / (1 + S L(delta))

with ``L`` the peak-normalised homogeneous response and ``S = I/I_sat`` its on-resonance
value. Two consequences fall straight out and are both verified analytically in
``test_lif_rates.py``:

- ``S -> 0``: the fraction is ``[g_2/(g_1+g_2)] S L(delta)`` — linear in laser power, and
  the lineshape is exactly the unbroadened homogeneous one.
- ``S`` large: the half-maximum moves to ``L = 1/(2+S)``, so the observed width is
  ``Gamma sqrt(1+S)``. That is the doc 04 §3.4 "measured lineshape broadens", and it
  emerges from the algebra rather than being pasted on afterwards.

## The closure, and its bound

Doc 04 §3.1's is a three-level scheme: level 2 decays to level 3, not back to level 1.
Taken literally over infinite time the pump would empty level 1 entirely. The steady state
here closes the system instead — ``n_1 + n_2 = n`` — which is the standard LIF treatment
and assumes level 3 refills level 1 fast compared with the pump-out rate.

**The bound.** In the weak-pump limit the closure is exact to ``O(S)``: nothing is
optically pumped out because almost nothing is excited. It degrades as ``S`` grows, and it
degrades in a known direction — a genuinely open system would show *less* signal than this
model at large ``S``, never more. So the modelled saturation curve is an upper bound on the
open-system one, and the doc 04 V-23 comparison against "analytic two-level" is a
comparison this closure is the right one for.

## A convention that will surprise a reader

Doc 04 §3.4 defines ``I_sat = 2 pi^2 h c A_21 / (3 lambda^3)``. The form more common in the
literature is ``pi h c A_21 / (3 lambda^3)``, smaller by ``2 pi``. The specification's is
used, unmodified, because it is the specification; a saturation parameter quoted here is
therefore ``2 pi`` times smaller than one computed with the textbook convention for the
same beam. Doc 07 F-13's "sweeps ``S`` from 0.01 to 10" is read in the doc's convention.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "lorentzian_response",
    "power_broadened_fwhm",
    "pump_rate_from_saturation",
    "saturation_ceiling",
    "steady_state_excited_fraction",
    "upper_state_rate",
]

type FloatArray = NDArray[np.float64]


def _checked_degeneracies(lower: int, upper: int) -> tuple[float, float]:
    if lower <= 0 or upper <= 0:
        raise ValueError(
            f"degeneracies must be positive, got g_1 = {lower} and g_2 = {upper}. "
            "A zero would make the statistical ceiling of the saturated transition "
            "either zero or one, and both are wrong in a way nothing downstream checks."
        )
    return float(lower), float(upper)


def lorentzian_response(detuning_hz: FloatArray, *, fwhm_hz: float) -> FloatArray:
    """Peak-normalised homogeneous response, ``1 / (1 + (2 delta / Gamma)^2)``.

    Normalised to **unit peak** rather than unit area, on purpose. The saturation
    parameter ``S`` is defined on resonance, so the quantity multiplying it must be one at
    line centre; an area-normalised profile would fold the width into ``S`` and make the
    saturated limit depend on the linewidth, which it does not.
    """
    if fwhm_hz <= 0.0:
        raise ValueError(f"homogeneous FWHM must be positive, got {fwhm_hz}")

    scaled = 2.0 * np.asarray(detuning_hz, dtype=np.float64) / fwhm_hz
    return np.asarray(1.0 / (1.0 + scaled**2), dtype=np.float64)


def saturation_ceiling(*, lower_degeneracy: int, upper_degeneracy: int) -> float:
    """``g_2 / (g_1 + g_2)`` — the excited fraction an infinite pump cannot exceed.

    The statistical limit of a closed two-level system: stimulated emission catches up
    with absorption once the populations reach their degeneracy ratio. Named because a
    model that saturates at 1 instead has stimulated emission missing from its rate
    equation, and the difference only shows up at large ``S``.
    """
    g_lower, g_upper = _checked_degeneracies(lower_degeneracy, upper_degeneracy)
    return g_upper / (g_lower + g_upper)


def steady_state_excited_fraction(
    *,
    saturation: float,
    response: FloatArray,
    lower_degeneracy: int,
    upper_degeneracy: int,
) -> FloatArray:
    """``n_2 / n`` at each detuning — the steady state of doc 04 §3.1's rate equation.

    Args:
        saturation: ``S = I / I_sat`` on resonance, doc 04 §3.4.
        response: Peak-normalised homogeneous response at each detuning, from
            :func:`lorentzian_response` or a sum of Zeeman-shifted copies of it.
        lower_degeneracy: ``g_1 = 2 J_1 + 1``.
        upper_degeneracy: ``g_2 = 2 J_2 + 1``.

    Returns:
        The excited fraction, elementwise. Between zero and
        :func:`saturation_ceiling`, strictly.
    """
    if saturation < 0.0:
        raise ValueError(f"saturation parameter cannot be negative, got {saturation}")

    values = np.asarray(response, dtype=np.float64)
    if np.any(values < 0.0):
        raise ValueError("homogeneous response cannot be negative")

    ceiling = saturation_ceiling(
        lower_degeneracy=lower_degeneracy, upper_degeneracy=upper_degeneracy
    )
    local = saturation * values
    return np.asarray(ceiling * local / (1.0 + local), dtype=np.float64)


def pump_rate_from_saturation(
    *,
    saturation: float,
    relaxation_rate: float,
    lower_degeneracy: int,
    upper_degeneracy: int,
) -> float:
    """``B_12 rho`` corresponding to a given ``S`` — the bridge between the two forms.

    Doc 04 §3.1 parameterises the pump by ``B_12 rho(nu)``; doc 04 §3.4 parameterises it by
    ``S = I/I_sat``. They describe the same field, and the identification that makes the
    two-level steady state come out as ``[g_2/(g_1+g_2)] S/(1+S)`` is

        ``S = B_12 rho (1 + g_1/g_2) / (A_total + Q)``.

    Written here once rather than inline, because it is the only place the two
    parameterisations touch and a factor lost here would move the whole saturation curve
    without changing its shape — invisible to every test that only checks limits.
    """
    if relaxation_rate <= 0.0:
        raise ValueError(f"relaxation rate must be positive, got {relaxation_rate}")

    g_lower, g_upper = _checked_degeneracies(lower_degeneracy, upper_degeneracy)
    return saturation * relaxation_rate / (1.0 + g_lower / g_upper)


def upper_state_rate(
    *,
    n_lower: float,
    n_upper: float,
    pump_rate: float,
    relaxation_rate: float,
    lower_degeneracy: int,
    upper_degeneracy: int,
) -> float:
    """``dn_2/dt`` of doc 04 §3.1, evaluated as written.

    Kept as an explicit function even though the framework only ever uses its steady
    state. It is what ``test_lif_rates.py`` integrates numerically to check that steady
    state, and an algebra slip in :func:`steady_state_excited_fraction` is otherwise
    undetectable — the wrong closed form is still smooth, still monotone in ``S``, and
    still saturates.

    ``B_21 rho = (g_1/g_2) B_12 rho`` by the Einstein relation, which is where the
    degeneracies enter.
    """
    g_lower, g_upper = _checked_degeneracies(lower_degeneracy, upper_degeneracy)
    stimulated = pump_rate * g_lower / g_upper
    return n_lower * pump_rate - n_upper * (stimulated + relaxation_rate)


def power_broadened_fwhm(*, fwhm_hz: float, saturation: float) -> float:
    """``Gamma sqrt(1 + S)`` — the observed homogeneous width under saturation.

    Reported rather than used: the broadening is already present in whatever
    :func:`steady_state_excited_fraction` returns, because it is a consequence of the
    nonlinearity and not a separate stage. This exists so a run can *state* its power
    broadening alongside the ``S`` doc 04 §3.4 requires it to report, and so that
    ``test_lif_rates.py`` can check the stated value against the width it measures off the
    computed profile.
    """
    if fwhm_hz <= 0.0:
        raise ValueError(f"homogeneous FWHM must be positive, got {fwhm_hz}")
    if saturation < 0.0:
        raise ValueError(f"saturation parameter cannot be negative, got {saturation}")

    return fwhm_hz * math.sqrt(1.0 + saturation)
