"""Tests for signed-magnitude input linear cost in the discrete QP."""

from __future__ import annotations

import numpy as np
import pytest

from mbc.control import (
    StandardLinearDiscreteOCP,
    infer_signed_magnitude_input_indices,
    InputLinearCostMode,
)
from tests.ocp_fixtures import ScalarDiscretePlant


class BidirectionalPlant(ScalarDiscretePlant):
    """Scalar plant with signed input bounds."""

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-2.0]), np.array([2.0])


class TestInferSignedMagnitudeIndices:
    def test_spans_zero(self):
        idx = infer_signed_magnitude_input_indices(
            np.array([-1.0, 0.0, -2.0]),
            np.array([1.0, 5.0, -1.0]),
        )
        np.testing.assert_array_equal(idx, [0])

    def test_one_sided_excluded(self):
        idx = infer_signed_magnitude_input_indices(
            np.array([0.0]), np.array([3.0]),
        )
        assert idx.size == 0


class TestSignedMagnitudeLinearCost:
    def test_auto_selects_slack_for_bidirectional_input(self):
        model = BidirectionalPlant()
        ocp = StandardLinearDiscreteOCP(
            model, N=4, Q=np.eye(1) * 0.01, R=np.eye(1) * 0.01,
        )
        ocp.set_disturbance_profile(np.zeros(4))
        price = np.array([1.0, 1.0, 1.0, 1.0])
        ocp.set_input_linear_cost_profile(price.reshape(-1, 1))

        U_pos, _ = ocp.solve(x0=[0.0], x_ref=[0.0])
        U_neg, _ = ocp.solve(x0=[0.0], x_ref=[-3.0])

        assert U_pos.reshape(-1)[0] >= -1e-6
        assert U_neg.reshape(-1)[0] <= 1e-6

    def test_direct_penalty_on_one_sided_input(self):
        model = ScalarDiscretePlant()
        ocp = StandardLinearDiscreteOCP(
            model, N=3, Q=np.eye(1) * 0.01, R=np.eye(1) * 0.01,
        )
        ocp.set_disturbance_profile(np.zeros(3))
        ocp.set_input_linear_cost_profile(np.array([2.0, 2.0, 2.0]).reshape(-1, 1))
        U, _ = ocp.solve(x0=[0.0], x_ref=[5.0])
        assert np.all(U.reshape(-1) >= -1e-6)

    def test_explicit_slack_indices(self):
        model = ScalarDiscretePlant()
        ocp = StandardLinearDiscreteOCP(
            model, N=2, Q=np.eye(1) * 0.01, R=np.eye(1) * 0.01,
        )
        ocp.set_disturbance_profile(np.zeros(2))
        ocp.set_input_linear_cost_profile(
            np.array([1.0, 1.0]).reshape(-1, 1),
            signed_magnitude_input_indices=np.array([0]),
        )
        U_hi, _ = ocp.solve(x0=[0.0], x_ref=[4.0])
        U_lo, _ = ocp.solve(x0=[0.0], x_ref=[-4.0])
        assert U_hi.reshape(-1)[0] >= -1e-6
        assert U_lo.reshape(-1)[0] <= 1e-6

    def test_asymmetric_slack_coefficients(self):
        model = BidirectionalPlant()
        ocp = StandardLinearDiscreteOCP(
            model, N=2, Q=np.eye(1) * 0.01, R=np.eye(1) * 0.01,
        )
        ocp.set_disturbance_profile(np.zeros(2))
        ocp.set_input_linear_cost_profile(
            np.ones((2, 1)),
            signed_magnitude_input_indices=np.array([0]),
            positive_slack_coefficient_profile=np.array([[0.1], [0.1]]),
            negative_slack_coefficient_profile=np.array([[5.0], [5.0]]),
        )
        U, _ = ocp.solve(x0=[0.0], x_ref=[-2.0])
        assert U.reshape(-1)[0] >= -0.05

    def test_slack_with_soft_output_constraints(self):
        """Input slacks must not break soft-output slack columns in the QP."""
        model = BidirectionalPlant()
        ocp = StandardLinearDiscreteOCP(
            model, N=4, Q=np.eye(1) * 10.0, R=np.eye(1) * 0.01,
            z_offset=0.5, rho=1e4,
        )
        ocp.set_input_linear_cost_profile(
            np.full((4, 1), 0.05),
            signed_magnitude_input_indices=np.array([0]),
        )
        U, _ = ocp.solve(x0=[0.0], x_ref=[5.0], D=np.zeros(0))
        assert U.reshape(-1)[0] > 0.5
        u_min, u_max = model.u_bounds
        assert U.reshape(-1)[0] <= u_max[0] + 1e-5
