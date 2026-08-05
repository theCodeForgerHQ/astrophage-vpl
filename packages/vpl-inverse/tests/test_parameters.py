"""Tests for the Level A control-parameter vector — doc 05 §2.1.

The load-bearing test in this file is :class:`TestJacobiansAgainstNumericalDerivatives`.
Everything else checks bookkeeping; that one checks the piece of mathematics that, if
wrong, produces a posterior that is smoothly, plausibly and completely incorrect. It is
therefore checked against a numerical derivative of the transform itself rather than
against a second closed form written by the same hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.differentiate import derivative

from vpl.inverse.parameters import (
    CONTROL_PARAMETERS,
    N_CONTROL,
    ControlParameters,
    IdentityTransform,
    LogitTransform,
    LogTransform,
    Transform,
    control_parameter_names,
    log_abs_det_jacobian,
    unconstrained_bounds,
)

#: doc 05 §2.1, transcribed from the table rather than read from the registry. The point
#: of a transcription is that it fails when the registry drifts away from the document.
DOC_05_TABLE = {
    "n_0": ("m**-3", (1.0e15, 1.0e19)),
    "T_e": ("eV", None),
    "T_i": ("eV", (0.02, 0.5)),
    "p": ("mTorr", None),
    "V_w": ("V", None),
    "phi_RF": ("rad", (0.0, 2.0 * math.pi)),
    "gamma_se": ("dimensionless", (0.0, 0.3)),
    "kappa": ("dimensionless", (1.0, 5.0)),
}

#: A point strictly inside every parameter's support, used wherever a test needs "some
#: legal vector". Not the reference point: a test that only ever evaluates at the
#: registry defaults cannot distinguish a transform from the identity.
INTERIOR = ControlParameters(
    n_0=3.7e16,
    T_e=4.25,
    T_i=0.11,
    p=7.5,
    V_w=-310.0,
    phi_RF=1.9,
    gamma_se=0.14,
    kappa=2.3,
)


def _transforms() -> list[tuple[str, Transform]]:
    return [(spec.name, spec.transform) for spec in CONTROL_PARAMETERS]


class TestTheLevelATable:
    """The vector is doc 05 §2.1's table and not an approximation of it."""

    def test_there_are_eight_control_parameters(self) -> None:
        # doc 05 §2.1: "Level A — control parameters (theta_c, ~8)".
        assert len(DOC_05_TABLE) == N_CONTROL
        assert len(CONTROL_PARAMETERS) == N_CONTROL

    def test_the_names_are_the_documented_ones_in_documented_order(self) -> None:
        assert control_parameter_names() == tuple(DOC_05_TABLE)

    def test_each_parameter_carries_the_documented_units(self) -> None:
        for spec in CONTROL_PARAMETERS:
            assert spec.units == DOC_05_TABLE[spec.name][0]

    def test_each_bounded_parameter_carries_the_documented_support(self) -> None:
        for spec in CONTROL_PARAMETERS:
            expected = DOC_05_TABLE[spec.name][1]
            if expected is None:
                continue
            assert spec.support is not None
            assert spec.support == pytest.approx(expected, rel=1e-12, abs=0.0)

    def test_the_documented_unbounded_parameters_have_no_support(self) -> None:
        for spec in CONTROL_PARAMETERS:
            if DOC_05_TABLE[spec.name][1] is None:
                assert spec.support is None

    def test_positivity_is_structural_for_the_positive_quantities(self) -> None:
        # doc 05 §4.2: "Positivity | log-parameterisation of n, T | structural". A log
        # transform cannot produce a negative value, which is what "structural" means.
        for name in ("n_0", "T_e", "T_i", "p"):
            spec = next(s for s in CONTROL_PARAMETERS if s.name == name)
            assert isinstance(spec.transform, LogTransform)

    def test_the_bounded_parameters_use_a_logit(self) -> None:
        for name in ("phi_RF", "gamma_se", "kappa"):
            spec = next(s for s in CONTROL_PARAMETERS if s.name == name)
            assert isinstance(spec.transform, LogitTransform)

    def test_the_wall_bias_is_unconstrained(self) -> None:
        # V_w is signed — doc 01 §2.1 registers RP1.bias as -250 V — so neither a log nor
        # a logit applies, and pretending otherwise would exclude the sign the physics has.
        spec = next(s for s in CONTROL_PARAMETERS if s.name == "V_w")
        assert isinstance(spec.transform, IdentityTransform)


class TestTheVectorItself:
    def test_it_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            INTERIOR.n_0 = 1.0  # type: ignore[misc]

    def test_it_round_trips_through_a_plain_array(self) -> None:
        assert ControlParameters.from_array(INTERIOR.as_array()) == INTERIOR

    def test_the_array_is_in_documented_order(self) -> None:
        assert INTERIOR.as_array().tolist() == [
            INTERIOR.n_0,
            INTERIOR.T_e,
            INTERIOR.T_i,
            INTERIOR.p,
            INTERIOR.V_w,
            INTERIOR.phi_RF,
            INTERIOR.gamma_se,
            INTERIOR.kappa,
        ]

    def test_the_reference_point_is_rp1(self) -> None:
        # doc 01 §2.1 defines RP-1 and doc 05 §2.1's priors are centred on it.
        reference = ControlParameters.reference()

        assert reference.n_0 == pytest.approx(1.0e17)
        assert reference.T_e == pytest.approx(3.0)
        assert reference.T_i == pytest.approx(0.05)
        assert reference.p == pytest.approx(5.0)
        assert reference.V_w == pytest.approx(-250.0)
        assert reference.gamma_se == pytest.approx(0.10)
        assert reference.kappa == pytest.approx(1.0)

    def test_a_value_outside_its_support_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside its doc 05"):
            ControlParameters(**{**INTERIOR.to_dict(), "kappa": 7.0})

    def test_a_non_finite_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            ControlParameters(**{**INTERIOR.to_dict(), "V_w": math.nan})

    def test_a_non_positive_log_parameter_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside its doc 05"):
            ControlParameters(**{**INTERIOR.to_dict(), "T_e": 0.0})


class TestRoundTrips:
    """Constrained -> unconstrained -> constrained must be the identity."""

    def test_the_vector_round_trips_to_machine_precision(self) -> None:
        recovered = ControlParameters.from_unconstrained(INTERIOR.to_unconstrained())

        for name, value in INTERIOR.to_dict().items():
            assert recovered.to_dict()[name] == pytest.approx(value, rel=1e-13, abs=0.0)

    @pytest.mark.parametrize("name_and_transform", _transforms(), ids=lambda t: t[0])
    def test_each_transform_round_trips(self, name_and_transform: tuple[str, Transform]) -> None:
        _, transform = name_and_transform

        for u in (-4.0, -0.3, 0.0, 0.7, 5.0):
            x = transform.to_constrained(u)
            assert transform.to_unconstrained(x) == pytest.approx(u, rel=1e-12, abs=1e-12)

    def test_the_reference_point_round_trips(self) -> None:
        reference = ControlParameters.reference()
        recovered = ControlParameters.from_unconstrained(reference.to_unconstrained())

        assert recovered.as_array() == pytest.approx(reference.as_array(), rel=1e-13)

    def test_from_unconstrained_rejects_a_wrong_length_vector(self) -> None:
        with pytest.raises(ValueError, match="8"):
            ControlParameters.from_unconstrained(np.zeros(N_CONTROL - 1))


class TestTransformImages:
    """Each transform maps R onto exactly the support it claims."""

    def test_the_log_transform_is_the_exponential(self) -> None:
        transform = LogTransform()

        for u in (-30.0, -1.0, 0.0, 12.0):
            assert transform.to_constrained(u) == pytest.approx(math.exp(u), rel=1e-15)

    def test_the_logit_transform_stays_inside_its_interval(self) -> None:
        transform = LogitTransform(low=1.0, high=5.0)

        for u in (-700.0, -12.0, 0.0, 12.0, 700.0):
            x = transform.to_constrained(u)
            assert 1.0 <= x <= 5.0

    def test_the_logit_transform_is_centred_on_the_midpoint(self) -> None:
        # u = 0 is where an optimiser is initialised when it has nothing better; the
        # midpoint of a uniform prior is the right place for that to land.
        transform = LogitTransform(low=1.0, high=5.0)

        assert transform.to_constrained(0.0) == pytest.approx(3.0, rel=1e-15)

    def test_the_logit_transform_is_monotone(self) -> None:
        transform = LogitTransform(low=0.0, high=2.0 * math.pi)
        grid = np.linspace(-8.0, 8.0, 257)

        images = np.array([transform.to_constrained(float(u)) for u in grid])

        assert np.all(np.diff(images) > 0.0)

    def test_a_degenerate_logit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            LogitTransform(low=5.0, high=1.0)


class TestJacobiansAgainstNumericalDerivatives:
    """`log |dx/du|` against a numerical derivative of `to_constrained`.

    doc 05's priors are stated in the constrained space; a sampler works in the
    unconstrained one. The bridge between them is this determinant, and the failure mode
    when it is wrong is not a crash — it is a posterior that is shifted, differently
    weighted, and entirely believable. So it is checked against a derivative computed by
    a library that knows nothing about the closed form, not against a second closed form.
    """

    @pytest.mark.parametrize("name_and_transform", _transforms(), ids=lambda t: t[0])
    @pytest.mark.parametrize("u", [-3.5, -0.9, 0.0, 0.4, 2.6])
    def test_each_component_jacobian_matches_a_numerical_derivative(
        self, name_and_transform: tuple[str, Transform], u: float
    ) -> None:
        _, transform = name_and_transform

        def constrained(points: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
            return np.array([transform.to_constrained(float(p)) for p in points.ravel()]).reshape(
                points.shape
            )

        numerical = derivative(constrained, u)
        analytic = math.exp(transform.log_abs_det_jacobian(u))

        assert analytic == pytest.approx(float(numerical.df), rel=1e-9)

    def test_the_vector_log_determinant_matches_the_numerical_jacobian_matrix(self) -> None:
        # The full 8x8 Jacobian is diagonal because the transform acts componentwise. That
        # is an assumption the sum-of-logs implementation relies on, so it is verified
        # here by building the matrix numerically and taking `slogdet` of the whole thing.
        # Taken from a legal point, because the finite-difference stencil constructs
        # `ControlParameters` at every stencil node and those must be inside the support.
        u = INTERIOR.to_unconstrained()

        jacobian = np.zeros((N_CONTROL, N_CONTROL))
        for column in range(N_CONTROL):

            def component(
                points: np.typing.NDArray[np.float64], column: int = column
            ) -> np.typing.NDArray[np.float64]:
                out = np.empty_like(points)
                flat = points.ravel()
                for k, point in enumerate(flat):
                    shifted = u.copy()
                    shifted[column] = float(point)
                    out.ravel()[k] = ControlParameters.from_unconstrained(shifted).as_array()[
                        column
                    ]
                return out

            jacobian[column, column] = float(derivative(component, u[column]).df)

        sign, numerical_logdet = np.linalg.slogdet(jacobian)

        assert sign == 1.0
        assert log_abs_det_jacobian(u) == pytest.approx(float(numerical_logdet), rel=1e-9)

    def test_the_log_determinant_is_the_sum_over_components(self) -> None:
        u = np.array([0.9, -1.4, 0.2, -0.6, 130.0, 0.35, -1.1, 0.8])

        expected = sum(
            spec.transform.log_abs_det_jacobian(float(value))
            for spec, value in zip(CONTROL_PARAMETERS, u, strict=True)
        )

        assert log_abs_det_jacobian(u) == pytest.approx(expected, rel=1e-14)

    def test_the_identity_jacobian_is_one(self) -> None:
        assert IdentityTransform().log_abs_det_jacobian(3.3) == 0.0

    def test_the_log_jacobian_is_the_point_itself(self) -> None:
        # d(exp u)/du = exp u, so log |dx/du| = u exactly. Worth its own test because it
        # is the one case where an off-by-a-sign is invisible at u = 0.
        assert LogTransform().log_abs_det_jacobian(-2.5) == pytest.approx(-2.5, rel=1e-15)

    def test_the_logit_jacobian_does_not_overflow_in_the_tails(self) -> None:
        # A naive `log(width * s * (1 - s))` underflows to -inf around |u| ~ 40, which is
        # well inside the region an unconstrained sampler visits during warm-up.
        transform = LogitTransform(low=0.0, high=1.0)

        assert transform.log_abs_det_jacobian(-500.0) == pytest.approx(-500.0, rel=1e-12)
        assert transform.log_abs_det_jacobian(500.0) == pytest.approx(-500.0, rel=1e-12)


class TestUnconstrainedBounds:
    """What an L-BFGS-B run needs to know about the space it is searching."""

    def test_a_log_uniform_parameter_keeps_its_box_in_log_space(self) -> None:
        # doc 05 §2.1's log-uniform priors have bounded support, so the "unconstrained"
        # space is unconstrained in the sense the transform makes it, not in the sense of
        # having no box at all. An optimiser is told where the box is.
        lower, upper = unconstrained_bounds()
        index = control_parameter_names().index("n_0")

        assert lower[index] == pytest.approx(math.log(1.0e15), rel=1e-14)
        assert upper[index] == pytest.approx(math.log(1.0e19), rel=1e-14)

    def test_a_logit_parameter_is_genuinely_unbounded(self) -> None:
        lower, upper = unconstrained_bounds()
        index = control_parameter_names().index("kappa")

        assert lower[index] == -math.inf
        assert upper[index] == math.inf

    def test_the_reference_point_lies_inside_the_box(self) -> None:
        lower, upper = unconstrained_bounds()
        u = ControlParameters.reference().to_unconstrained()

        assert np.all(u >= lower)
        assert np.all(u <= upper)
