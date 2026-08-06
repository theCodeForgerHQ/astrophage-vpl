"""The naive single-probe analysis — what a real experimentalist does with an I-V curve.

Split from :mod:`vpl.instruments.probe.langmuir.physics` (the generative model) so that
the comparison this package exists for is visible as a seam in the code, not just in
prose: :func:`estimate_from_iv_curve` is the *only* thing in this package that decides
what ``T_e`` and ``n_e`` a probe reports, and it never sees which EEDF shape or sheath
model produced the curve it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vpl.core.constants import ELEMENTARY_CHARGE
from vpl.core.units import magnitude_in
from vpl.instruments.probe.langmuir.physics import bohm_speed_m_per_s
from vpl.physics.analytic.sheath import DEFAULT_EDGE_TO_CENTRE_RATIO

__all__ = ["LangmuirEstimate", "estimate_from_iv_curve"]

type FloatArray = NDArray[np.float64]

_E_C: Final[float] = float(magnitude_in(ELEMENTARY_CHARGE, "C"))

#: Fraction of the (sorted) sweep used to estimate the ion-saturation current — the most
#: negative points, where the electron branch has underflowed.
_ION_SAT_FRACTION_OF_SWEEP: Final[float] = 0.1

#: Lower and upper bounds, as a fraction of the recovered electron-saturation current, of
#: the window the log-slope fit uses. Excludes the ion-saturation-dominated points below
#: and the already-saturated points above, the same practical window a real analysis uses.
#: The lower bound is not arbitrary: below ~10-15 % of the electron-saturation current the
#: constant-ion-current subtraction of step 2 below is no longer a good approximation at
#: the default probe geometry (the sheath-expanded ion current at the bottom of the sweep
#: is itself a comparable fraction of the electron-saturation current there), and fitting
#: through that region biases even the Maxwellian recovery. 0.2-0.4 is clear of it.
_FIT_WINDOW_LOWER_FRACTION: Final[float] = 0.2
_FIT_WINDOW_UPPER_FRACTION: Final[float] = 0.4


@dataclass(frozen=True, slots=True)
class LangmuirEstimate:
    """What :func:`estimate_from_iv_curve` extracts — the naive single-probe result.

    Attributes:
        electron_temperature_ev: From the log-slope of the exponential region.
        electron_density_m3: From ``I_sat`` at the *bare* probe area.
        ion_saturation_current_a: The ``I_sat`` the estimate was derived from.
    """

    electron_temperature_ev: float
    electron_density_m3: float
    ion_saturation_current_a: float

    def __repr__(self) -> str:
        return (
            f"LangmuirEstimate(T_e={self.electron_temperature_ev:.4g} eV, "
            f"n_e={self.electron_density_m3:.4g} m^-3, "
            f"I_sat={self.ion_saturation_current_a:.4g} A)"
        )


def estimate_from_iv_curve(
    voltage_v: ArrayLike,
    current_a: ArrayLike,
    *,
    ion_mass_kg: float,
    probe_area_m2: float,
    edge_to_centre_ratio: float = DEFAULT_EDGE_TO_CENTRE_RATIO,
) -> LangmuirEstimate:
    """The naive analysis a real experimentalist runs on a measured I-V curve.

    Not tied to :class:`~vpl.instruments.probe.langmuir.instrument.LangmuirProbe` or to
    any simulated state: this operates on raw voltage/current arrays exactly as they
    would come off real hardware, which is the point doc 00 E2 makes about the
    ``Instrument`` contract generally — the comparison this module exists for is against
    what an experimentalist actually does with the data, not against the instrument's
    raw signal.

    The procedure (Langmuir 1923; Merlino 2007 §II):

    1. **Ion saturation.** Average the most negative
       :data:`_ION_SAT_FRACTION_OF_SWEEP` of the (sorted) sweep, where the electron
       branch has underflowed to nothing. This reads off whatever the true ion current
       is *there* — sheath-expanded, if the generative model included it — with no
       correction applied, exactly as a real analysis that has not modelled sheath
       expansion would.
    2. **Subtract a constant ion current** from the whole curve to isolate the electron
       branch. The standard simplified subtraction (Merlino 2007 §II.B): real practice
       also linearly extrapolates the ion branch, and the constant-subtraction here is
       the more conservative (larger-residual) of the two only in the saturated region,
       where doc 00 C4 would rather the offset were visible than hidden.
    3. **Fit ``ln(I_e)`` vs ``V``** by ordinary least squares over the window
       :data:`_FIT_WINDOW_LOWER_FRACTION` to :data:`_FIT_WINDOW_UPPER_FRACTION` of the
       recovered electron-saturation current — below is dominated by the ion-subtraction
       residual, above is already saturated. ``T_e = 1 / slope``.
    4. **``n_e`` from ``I_sat``,** using the *bare* probe area and the fitted ``T_e``
       (through ``c_s``). No sheath-expansion correction — that correction requires
       knowing the true sheath law, which a real analysis of a real probe does not have.

    Args:
        voltage_v: Swept probe bias, any order.
        current_a: Collected current at each voltage, electron-positive convention.
        ion_mass_kg: Ion mass, for the Bohm speed in the density conversion.
        probe_area_m2: The bare (not sheath-expanded) collecting area.
        edge_to_centre_ratio: The presheath density ratio ``h_l`` assumed by the analysis.

    Returns:
        The recovered :class:`LangmuirEstimate`.

    Raises:
        ValueError: If the arrays disagree in shape, if the ion-saturation region does
            not carry a net negative current, or if too few points fall in the fit
            window to determine a slope.
    """
    voltages = np.asarray(voltage_v, dtype=np.float64)
    currents = np.asarray(current_a, dtype=np.float64)
    if voltages.shape != currents.shape:
        raise ValueError(
            f"voltage and current arrays must have the same shape, got {voltages.shape} "
            f"and {currents.shape}"
        )

    order = np.argsort(voltages)
    voltages = voltages[order]
    currents = currents[order]

    n_ion_points = max(2, round(_ION_SAT_FRACTION_OF_SWEEP * voltages.size))
    i_sat = -float(np.mean(currents[:n_ion_points]))
    if not i_sat > 0.0:
        raise ValueError(
            "the most negative part of this sweep does not carry a net negative "
            "current, so no ion-saturation current can be read off it; widen the sweep "
            "toward more negative bias"
        )

    electron_current = currents + i_sat
    positive = electron_current > 0.0
    if not np.any(positive):
        raise ValueError(
            "no positive electron current recovered after subtracting I_sat; widen the "
            "sweep past the plasma potential"
        )
    peak = float(np.max(electron_current[positive]))

    window = (
        positive
        & (electron_current >= _FIT_WINDOW_LOWER_FRACTION * peak)
        & (electron_current <= _FIT_WINDOW_UPPER_FRACTION * peak)
    )
    if np.count_nonzero(window) < 2:
        raise ValueError(
            "too few points fall inside the exponential fit window "
            f"[{_FIT_WINDOW_LOWER_FRACTION}, {_FIT_WINDOW_UPPER_FRACTION}] of the "
            "recovered electron-saturation current; the sweep does not resolve the "
            "exponential region"
        )

    slope, _intercept = np.polyfit(voltages[window], np.log(electron_current[window]), 1)
    if not slope > 0.0:
        raise ValueError(
            f"the fitted log-slope is not positive ({slope!r}); a Langmuir probe's "
            "electron current must rise with bias in the exponential region"
        )
    t_e_est = 1.0 / float(slope)

    c_s = bohm_speed_m_per_s(electron_temperature_ev=t_e_est, ion_mass_kg=ion_mass_kg)
    n_e_est = i_sat / (_E_C * edge_to_centre_ratio * c_s * probe_area_m2)

    return LangmuirEstimate(
        electron_temperature_ev=t_e_est, electron_density_m3=n_e_est, ion_saturation_current_a=i_sat
    )
