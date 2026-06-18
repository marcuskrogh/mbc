"""
Dedicated tests for optimal control problem formulations.

Uses lightweight scalar LTI fixtures from :mod:`tests.ocp_fixtures` and
existing nonlinear CD models from :mod:`tests.test_mpc`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbc.control import (
    StandardLinearDiscreteOCP,
    StandardLinearContinuousDiscreteOCP,
    StandardLinearizedContinuousDiscreteOCP,
    GeneralContinuousOCP,
    StandardContinuousOCP,
)
from tests.ocp_fixtures import ScalarDiscretePlant, ScalarCDPlant


@pytest.fixture
def scalar_disc():
    return ScalarDiscretePlant()


@pytest.fixture
def scalar_cd():
    return ScalarCDPlant()


class TestStandardLinearDiscreteOCP:
    def test_setpoint_tracking(self, scalar_disc):
        model = scalar_disc
        ocp = StandardLinearDiscreteOCP(model, N=5, Q=np.eye(1), R=np.eye(1) * 0.1)
        D = np.zeros(5)
        U, X = ocp.solve(x0=[0.0], D=D, x_ref=[2.0])
        assert U.shape == (5,)
        assert X.shape == (5,)
        assert U[0] > 0.0

    def test_hard_input_bounds(self, scalar_disc):
        model = scalar_disc
        ocp = StandardLinearDiscreteOCP(model, N=3, Q=np.eye(1), R=np.eye(1) * 0.01)
        D = np.zeros(3)
        U, _ = ocp.solve(x0=[0.0], D=D, x_ref=[10.0])
        u_min, u_max = model.u_bounds
        assert np.all(U.reshape(3, 1) >= u_min - 1e-6)
        assert np.all(U.reshape(3, 1) <= u_max + 1e-6)

    def test_soft_output_constraint(self, scalar_disc):
        ocp = StandardLinearDiscreteOCP(
            model=scalar_disc, N=4, Q=np.eye(1), R=np.eye(1) * 0.1,
            y_offset=0.5, rho=1e4,
        )
        U, X = ocp.solve(x0=[0.0], D=np.zeros(4), x_ref=[1.0])
        z = np.array(X).reshape(-1, 1)
        assert z.max() <= 1.0 + 0.5 + 0.1

    def test_rom_penalty(self, scalar_disc):
        ocp_smooth = StandardLinearDiscreteOCP(
            model=scalar_disc, N=5, Q=np.eye(1), R=np.eye(1) * 0.01, S=np.eye(1) * 1.0,
        )
        ocp_plain = StandardLinearDiscreteOCP(
            model=scalar_disc, N=5, Q=np.eye(1), R=np.eye(1) * 0.01,
        )
        D = np.zeros(5)
        U_s, _ = ocp_smooth.solve([0.0], D, [2.0], u_prev=[0.5])
        U_p, _ = ocp_plain.solve([0.0], D, [2.0], u_prev=[0.5])
        assert np.std(U_s) <= np.std(U_p) + 1e-6

    def test_hard_rom_limits(self, scalar_disc):
        ocp = StandardLinearDiscreteOCP(
            model=scalar_disc, N=4, Q=np.eye(1), R=np.eye(1) * 0.01,
            du_min=np.array([-0.2]), du_max=np.array([0.2]),
        )
        U, _ = ocp.solve(x0=[0.0], D=np.zeros(4), x_ref=[5.0], u_prev=[0.0])
        U_mat = U.reshape(4, 1)
        du0 = U_mat[0, 0] - 0.0
        assert -0.2 - 1e-6 <= du0 <= 0.2 + 1e-6
        for k in range(1, 4):
            du = U_mat[k, 0] - U_mat[k - 1, 0]
            assert -0.2 - 1e-6 <= du <= 0.2 + 1e-6


class TestStandardLinearContinuousDiscreteOCP:
    def test_zoh_tracking(self, scalar_cd):
        ocp = StandardLinearContinuousDiscreteOCP(
            scalar_cd, N=4, Q=np.eye(1), R=np.eye(1) * 0.1,
        )
        U, X = ocp.solve(x0=[0.0], D=np.zeros(4), x_ref=[1.5])
        assert U[0] > 0.0
        assert X.shape == (4,)


class TestGeneralContinuousOCP:
    def test_tracking_and_bounds(self, scalar_cd):
        model = scalar_cd.nonlinear_model
        ocp = StandardContinuousOCP(
            model, N=3, dt=1.0, Q_z=np.eye(1), z_ref=np.array([1.0]),
            u_min=np.array([-2.0]), u_max=np.array([2.0]),
            n_steps=2,
        )
        u_opt, cost, info = ocp.solve(
            x0=np.array([0.0]),
            d_trajectory=np.zeros((3, model.nd)),
        )
        assert u_opt.shape == (3, model.nu)
        assert np.all(u_opt >= -2.0 - 1e-5)
        assert np.all(u_opt <= 2.0 + 1e-5)
        assert "X" in info
