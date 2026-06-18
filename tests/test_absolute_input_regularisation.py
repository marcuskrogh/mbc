"""Tests for absolute (not deviation) input regularisation in deviation QPs."""

from __future__ import annotations

import numpy as np
import pytest

from mbc.control import (
    StandardLinearDiscreteOCP,
    absolute_quadratic_input_regularisation_linear_term,
)
from tests.ocp_fixtures import ScalarDiscretePlant


class DeviationScalarPlant(ScalarDiscretePlant):
    """Scalar plant whose QP decision variable is a deviation input δu."""

    def __init__(self, u_ss: float = 0.0) -> None:
        super().__init__()
        self.u_ss = float(u_ss)

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        u_min_abs, u_max_abs = super().u_bounds
        return u_min_abs - self.u_ss, u_max_abs - self.u_ss


class TestAbsoluteQuadraticRegularisation:
    def test_linear_term_matches_expanded_quadratic(self):
        R = np.array([[2.0]])
        u_eq = np.array([0.5])
        f = absolute_quadratic_input_regularisation_linear_term(
            R, u_eq, N=3, nu=1, r_scales=np.ones(3),
        )
        np.testing.assert_allclose(f, [2.0, 2.0, 2.0])

    def test_deviation_only_penalises_toward_equilibrium(self):
        """Without input_equilibrium, R drives δu → 0 (u_abs → u_ss)."""
        u_ss = 0.6
        model = DeviationScalarPlant(u_ss=u_ss)
        ocp = StandardLinearDiscreteOCP(
            model, N=4, Q=np.eye(1) * 0.01, R=np.eye(1) * 5.0,
        )
        U_dev, _ = ocp.solve(x0=[0.0], x_ref=[0.0], D=np.zeros(0))
        u_abs = float(U_dev.reshape(-1)[0] + u_ss)
        assert abs(u_abs - u_ss) < 0.15

    def test_absolute_penalises_toward_zero(self):
        """With input_equilibrium, R drives u_abs toward 0 rather than u_ss."""
        u_ss = 0.6
        model = DeviationScalarPlant(u_ss=u_ss)
        base = dict(model=model, N=4, Q=np.eye(1) * 0.01, R=np.eye(1) * 5.0)
        ocp_dev = StandardLinearDiscreteOCP(**base)
        U_dev_only, _ = ocp_dev.solve(x0=[0.0], x_ref=[0.0], D=np.zeros(0))
        u_abs_dev = float(U_dev_only.reshape(-1)[0] + u_ss)

        ocp_abs = StandardLinearDiscreteOCP(**base)
        ocp_abs.set_input_equilibrium(np.array([u_ss]))
        U_abs, _ = ocp_abs.solve(x0=[0.0], x_ref=[0.0], D=np.zeros(0))
        u_abs_reg = float(U_abs.reshape(-1)[0] + u_ss)

        assert abs(u_abs_dev - u_ss) < 0.15
        assert abs(u_abs_reg) < abs(u_abs_dev)

    @pytest.mark.parametrize("formulation", ["condensed", "sparse"])
    def test_formulations_agree(self, formulation):
        u_ss = 0.4
        model = DeviationScalarPlant(u_ss=u_ss)
        kwargs = dict(
            model=model, N=3, Q=np.eye(1) * 0.01, R=np.eye(1) * 3.0,
            formulation=formulation,
        )
        ocp_dev = StandardLinearDiscreteOCP(**kwargs)
        U_dev, _ = ocp_dev.solve(x0=[0.0], x_ref=[0.0], D=np.zeros(0))
        u_abs_dev = float(U_dev.reshape(-1)[0] + u_ss)

        ocp_abs = StandardLinearDiscreteOCP(**kwargs)
        ocp_abs.set_input_equilibrium(np.array([u_ss]))
        U, _ = ocp_abs.solve(x0=[0.0], x_ref=[0.0], D=np.zeros(0))
        u_abs_reg = float(U.reshape(-1)[0] + u_ss)
        assert abs(u_abs_reg) < abs(u_abs_dev)


class TestAbsoluteSignedMagnitudeLinearCost:
    def test_slack_links_absolute_input(self):
        """Signed-magnitude slacks enforce u_abs = s − t, not δu = s − t."""
        u_ss = 0.5
        model = DeviationScalarPlant(u_ss=u_ss)

        ocp = StandardLinearDiscreteOCP(
            model, N=2, Q=np.eye(1) * 0.01, R=np.eye(1) * 0.01,
        )
        ocp.set_input_equilibrium(np.array([u_ss]))
        ocp.set_input_linear_cost_profile(
            np.array([2.0, 2.0]).reshape(-1, 1),
            signed_magnitude_input_indices=np.array([0]),
        )
        U_dev, _ = ocp.solve(x0=[0.0], x_ref=[0.0], D=np.zeros(0))
        u_abs = float(U_dev.reshape(-1)[0] + u_ss)
        assert abs(u_abs) < abs(u_ss) * 0.5
