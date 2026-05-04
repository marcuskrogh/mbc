"""
Tests for MPC controllers and Optimal Control Problems.

Covers the following classes:

Linear (discrete-time):
  - OptimalControlProblem     (mbc.control)
  - MPCController             (mbc.control)

Linear continuous-discrete:
  - CDOptimalControlProblem   (mbc.control)
  - CDMPCController           (mbc.control)

Nonlinear continuous-discrete:
  - CDTrackingOptimalControlProblem (mbc.control)
  - EconomicOptimalControlProblem   (mbc.control)
  - CDNMPCController                (mbc.control)
"""

from __future__ import annotations

import numpy as np
import pytest
from cvxopt import matrix

from mbc.models import (
    LinearDiscreteModel,
    LinearContinuousDiscreteModel,
    ContinuousDiscreteModel,
)
from mbc.estimation import KalmanFilter, CDKalmanFilter, ContinuousDiscreteEKF
from mbc.control import (
    OptimalControlProblem,
    MPCController,
    CDOptimalControlProblem,
    CDMPCController,
    CDTrackingOptimalControlProblem,
    EconomicOptimalControlProblem,
    CDNMPCController,
)


# ── Concrete model fixtures ───────────────────────────────────────────────────


class DoubleIntegrator(LinearDiscreteModel):
    """
    Discrete-time double integrator (position + velocity), dt=1.

        x[k+1] = [[1, 1], [0, 1]] x[k] + [[0.5], [1.0]] u[k]
        ym[k]  = [1, 0] x[k] + v[k]
    """

    def __init__(self, x0=None):
        self._x = list(x0) if x0 is not None else [0.0, 0.0]

    @property
    def nx(self): return 2

    @property
    def nu(self): return 1

    @property
    def nd(self): return 1

    @property
    def Ad(self): return np.array([[1.0, 1.0], [0.0, 1.0]])

    @property
    def Bd(self): return np.array([[0.5], [1.0]])

    @property
    def Ed(self): return np.zeros((2, 1))

    @property
    def Cm(self): return np.array([[1.0, 0.0]])

    @property
    def Qd(self): return np.eye(2) * 1e-4

    @property
    def Rm(self): return np.array([[0.01]])

    @property
    def x(self): return list(self._x)

    @x.setter
    def x(self, val): self._x = list(val)

    @property
    def x_ref(self): return np.array([5.0, 0.0])

    @property
    def u_bounds(self): return np.array([-5.0]), np.array([5.0])


class SimpleLinearCD(LinearContinuousDiscreteModel):
    """
    First-order lag:  dx/dt = -x + u,  y = x,  dt = 1.0.
    """

    def __init__(self, x0=None):
        self._x = list(x0) if x0 is not None else [0.0]

    @property
    def nx(self): return 1

    @property
    def nu(self): return 1

    @property
    def nd(self): return 1

    @property
    def A(self): return np.array([[-1.0]])

    @property
    def B(self): return np.array([[1.0]])

    @property
    def E(self): return np.array([[0.0]])

    @property
    def G(self): return np.array([[0.1]])

    @property
    def Cm(self): return np.array([[1.0]])

    @property
    def Rm(self): return np.array([[0.01]])

    @property
    def dt(self): return 1.0

    @property
    def x(self): return list(self._x)

    @x.setter
    def x(self, val): self._x = list(val)

    @property
    def x_ref(self): return np.array([2.0])

    @property
    def u_bounds(self): return np.array([-5.0]), np.array([5.0])


class ScalarNonlinear(ContinuousDiscreteModel):
    """
    Scalar nonlinear system:  dx/dt = -x + u,  y = x,  z = x.
    Same as linear but implemented nonlinearly to test the NLP path.
    """

    @property
    def nx(self): return 1

    @property
    def nu(self): return 1

    @property
    def nd(self): return 1

    @property
    def nw(self): return 1

    @property
    def nym(self): return 1

    @property
    def nz(self): return 1

    @property
    def Rm(self): return np.array([[0.01]])

    def f(self, x, u, d, p, t):
        return np.array([-x[0] + u[0]])

    def sigma(self, x, u, d, p, t):
        return np.array([[0.1]])

    def hm(self, x, u, d, p, t=0.0):
        return np.array([x[0]])

    def g(self, x, u, d, p, t):
        return np.array([x[0]])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cvx(arr: np.ndarray) -> matrix:
    """Convert numpy array to cvxopt column vector."""
    arr = np.asarray(arr, dtype=float).ravel()
    return matrix(arr.tolist(), (len(arr), 1), tc="d")


def _np(m: matrix) -> np.ndarray:
    """Convert cvxopt column vector to 1-D numpy array."""
    return np.array(list(m), dtype=float)


# ── Tests: OptimalControlProblem ─────────────────────────────────────────────


class TestOptimalControlProblem:
    """Tests for the linear discrete-time OCP (QP solver)."""

    def _make_ocp(self, N=10, **kw):
        model = DoubleIntegrator()
        Q = matrix(np.eye(1))
        R = matrix(np.eye(1) * 0.1)
        return OptimalControlProblem(model, N=N, Q=Q, R=R, y_offset=20.0, **kw)

    def test_solve_returns_correct_shapes(self):
        ocp = self._make_ocp(N=5)
        model = DoubleIntegrator()
        x0 = _cvx(np.array([0.0, 0.0]))
        D = matrix(0.0, (5 * model.nd, 1))
        x_ref = _cvx(model.x_ref)
        U, X = ocp.solve(x0, D, x_ref)
        assert U.size == (5 * model.nu, 1), f"U shape {U.size}"
        assert X.size == (5 * model.nx, 1), f"X shape {X.size}"

    def test_solve_numpy_inputs_accepted(self):
        """solve() accepts numpy arrays for x0 and x_ref."""
        ocp = self._make_ocp(N=5)
        model = DoubleIntegrator()
        x0 = np.array([0.0, 0.0])
        D = matrix(0.0, (5 * model.nd, 1))
        x_ref = model.x_ref
        U, X = ocp.solve(x0, D, x_ref)
        assert U.size == (5 * model.nu, 1)

    def test_solve_drives_toward_reference(self):
        """The optimal trajectory should approach x_ref."""
        ocp = self._make_ocp(N=20)
        model = DoubleIntegrator()
        x0 = np.array([0.0, 0.0])
        x_ref = model.x_ref
        D = matrix(0.0, (20 * model.nd, 1))
        _, X = ocp.solve(x0, D, x_ref)
        X_np = _np(X).reshape(20, model.nx)
        # Position at end of horizon should be closer to reference than start
        dist_start = abs(x0[0] - x_ref[0])
        dist_end = abs(X_np[-1, 0] - x_ref[0])
        assert dist_end < dist_start, (
            f"Final position {X_np[-1, 0]:.3f} not closer to ref {x_ref[0]:.1f} "
            f"than initial {x0[0]:.1f}"
        )

    def test_input_bounds_respected(self):
        """Optimal inputs must lie within u_bounds."""
        ocp = self._make_ocp(N=10)
        model = DoubleIntegrator()
        x0 = np.array([0.0, 0.0])
        D = matrix(0.0, (10 * model.nd, 1))
        x_ref = model.x_ref
        U, _ = ocp.solve(x0, D, x_ref)
        u_vals = _np(U)
        u_min, u_max = model.u_bounds
        assert np.all(u_vals >= u_min[0] - 1e-6)
        assert np.all(u_vals <= u_max[0] + 1e-6)

    def test_rate_penalty_reduces_variation(self):
        """Adding rate-of-movement penalty S should reduce input variation."""
        model = DoubleIntegrator()
        Q = matrix(np.eye(1))
        R = matrix(np.eye(1) * 0.01)
        N = 10
        x0 = np.array([0.0, 0.0])
        D = matrix(0.0, (N * model.nd, 1))
        x_ref = model.x_ref

        ocp_no_rom = OptimalControlProblem(model, N=N, Q=Q, R=R, y_offset=20.0)
        ocp_rom = OptimalControlProblem(
            model, N=N, Q=Q, R=R, y_offset=20.0,
            S=matrix(np.eye(1) * 10.0),
        )
        U_no_rom, _ = ocp_no_rom.solve(x0, D, x_ref)
        U_rom, _ = ocp_rom.solve(x0, D, x_ref)

        u_no_rom = _np(U_no_rom).reshape(N, 1)
        u_rom = _np(U_rom).reshape(N, 1)
        var_no_rom = float(np.var(np.diff(u_no_rom, axis=0)))
        var_rom = float(np.var(np.diff(u_rom, axis=0)))
        assert var_rom <= var_no_rom + 1e-6, (
            f"ROM variance {var_rom:.4f} not ≤ no-ROM variance {var_no_rom:.4f}"
        )


# ── Tests: MPCController ──────────────────────────────────────────────────────


class TestMPCController:
    """Tests for the discrete-time MPCController (KF + OCP)."""

    def _make_ctrl(self, N=10):
        model = DoubleIntegrator()
        Q_kf = matrix(np.eye(2) * 1e-4)
        R_kf = matrix(np.eye(1) * 0.01)
        kf = KalmanFilter(model, Q=Q_kf, R=R_kf)
        Q_ocp = matrix(np.eye(1))
        R_ocp = matrix(np.eye(1) * 0.1)
        ocp = OptimalControlProblem(model, N=N, Q=Q_ocp, R=R_ocp, y_offset=20.0)
        ctrl = MPCController(model, estimator=kf, ocp=ocp)
        return ctrl, model

    def test_step_returns_correct_shapes(self):
        ctrl, model = self._make_ctrl(N=5)
        y = _cvx(np.array([0.5]))
        D = matrix(0.0, (5 * model.nd, 1))
        u, U_seq, X_seq = ctrl.step(y, D)
        assert u.size == (model.nu, 1)
        assert U_seq.size == (5 * model.nu, 1)
        assert X_seq.size == (5 * model.nx, 1)

    def test_step_input_within_bounds(self):
        ctrl, model = self._make_ctrl(N=10)
        y = _cvx(np.array([0.0]))
        D = matrix(0.0, (10 * model.nd, 1))
        u, _, _ = ctrl.step(y, D)
        u_val = float(list(u)[0])
        u_min, u_max = model.u_bounds
        assert u_val >= u_min[0] - 1e-6
        assert u_val <= u_max[0] + 1e-6

    def test_repeated_steps_do_not_crash(self):
        ctrl, model = self._make_ctrl(N=5)
        D = matrix(0.0, (5 * model.nd, 1))
        for k in range(10):
            y = _cvx(np.array([0.0 + 0.1 * k]))
            u, _, _ = ctrl.step(y, D)
            assert u.size == (model.nu, 1)

    def test_closed_loop_drives_toward_reference(self):
        """Running MPC for many steps should bring the output near x_ref[0]."""
        # Use a stable scalar model for reliable closed-loop behavior
        class ScalarLinearDiscrete(LinearDiscreteModel):
            """Stable scalar system: x[k+1] = 0.8 x[k] + 0.2 u[k], y = x."""
            @property
            def nx(self): return 1
            @property
            def nu(self): return 1
            @property
            def nd(self): return 1
            @property
            def Ad(self): return np.array([[0.8]])
            @property
            def Bd(self): return np.array([[0.2]])
            @property
            def Ed(self): return np.zeros((1, 1))
            @property
            def Cm(self): return np.eye(1)
            @property
            def Qd(self): return np.eye(1) * 1e-4
            @property
            def Rm(self): return np.array([[0.01]])
            @property
            def x(self): return [0.0]
            @x.setter
            def x(self, v): pass
            @property
            def x_ref(self): return np.array([3.0])
            @property
            def u_bounds(self): return np.array([-5.0]), np.array([5.0])

        model = ScalarLinearDiscrete()
        Q_kf = matrix(np.eye(1) * 1e-4)
        R_kf = matrix(np.eye(1) * 0.01)
        kf = KalmanFilter(model, Q=Q_kf, R=R_kf)
        Q_ocp = matrix(np.eye(1) * 5.0)
        R_ocp = matrix(np.eye(1) * 0.1)
        ocp = OptimalControlProblem(model, N=10, Q=Q_ocp, R=R_ocp, y_offset=20.0)
        ctrl = MPCController(model, estimator=kf, ocp=ocp)

        x = np.array([0.0])
        Ad = model.Ad
        Bd = model.Bd
        x_ref = model.x_ref
        N_steps = 30
        for _ in range(N_steps):
            y = _cvx(model.Cm @ x)
            D = matrix(0.0, (10 * model.nd, 1))
            u, _, _ = ctrl.step(y, D)
            u_np = np.array(list(u)).ravel()
            x = Ad @ x + Bd @ u_np

        # Position should be close to reference
        assert abs(x[0] - x_ref[0]) < 0.5, (
            f"Position {x[0]:.3f} not close to reference {x_ref[0]:.1f} after "
            f"{N_steps} steps"
        )


# ── Tests: CDOptimalControlProblem ────────────────────────────────────────────


class TestCDOptimalControlProblem:
    """Tests for CDOptimalControlProblem (linear CD system, QP via ZOH)."""

    def _make_ocp(self, N=10):
        model = SimpleLinearCD()
        Q = matrix(np.eye(1))
        R = matrix(np.eye(1) * 0.1)
        return CDOptimalControlProblem(model, N=N, Q=Q, R=R, y_offset=10.0), model

    def test_solve_returns_correct_shapes(self):
        ocp, model = self._make_ocp(N=5)
        x0 = np.array([0.0])
        D = matrix(0.0, (5 * model.nd, 1))
        x_ref = matrix(model.x_ref, (model.nx, 1))
        U, X = ocp.solve(x0, D, x_ref)
        assert U.size == (5 * model.nu, 1)
        assert X.size == (5 * model.nx, 1)

    def test_solve_input_within_bounds(self):
        ocp, model = self._make_ocp(N=8)
        x0 = np.array([0.0])
        D = matrix(0.0, (8 * model.nd, 1))
        x_ref = matrix(model.x_ref, (model.nx, 1))
        U, _ = ocp.solve(x0, D, x_ref)
        u_vals = _np(U)
        u_min, u_max = model.u_bounds
        assert np.all(u_vals >= u_min[0] - 1e-6)
        assert np.all(u_vals <= u_max[0] + 1e-6)

    def test_solve_drives_toward_reference(self):
        ocp, model = self._make_ocp(N=20)
        x0 = np.array([0.0])
        D = matrix(0.0, (20 * model.nd, 1))
        x_ref = matrix(model.x_ref, (model.nx, 1))
        _, X = ocp.solve(x0, D, x_ref)
        X_np = _np(X)
        # Final predicted state should be closer to reference than initial
        ref_val = model.x_ref[0]
        dist_start = abs(x0[0] - ref_val)
        dist_end = abs(X_np[-1] - ref_val)
        assert dist_end < dist_start, (
            f"Final state {X_np[-1]:.3f} not closer to ref {ref_val:.1f}"
        )

    def test_numpy_x0_accepted(self):
        """solve() must accept a numpy x0 (not only cvxopt)."""
        ocp, model = self._make_ocp(N=5)
        x0 = np.array([0.5])   # numpy, not cvxopt
        D = matrix(0.0, (5 * model.nd, 1))
        x_ref = matrix(model.x_ref, (model.nx, 1))
        U, X = ocp.solve(x0, D, x_ref)
        assert U.size == (5 * model.nu, 1)


# ── Tests: CDMPCController ────────────────────────────────────────────────────


class TestCDMPCController:
    """Tests for the linear CD-MPC controller (CDKalmanFilter + CDOptOCP)."""

    def _make_ctrl(self, N=10):
        model = SimpleLinearCD()
        kf = CDKalmanFilter(model, n_steps=10)
        Q = matrix(np.eye(1))
        R = matrix(np.eye(1) * 0.1)
        ocp = CDOptimalControlProblem(model, N=N, Q=Q, R=R, y_offset=10.0)
        ctrl = CDMPCController(model, estimator=kf, ocp=ocp)
        return ctrl, model

    def test_step_returns_correct_shapes(self):
        ctrl, model = self._make_ctrl(N=5)
        y = _cvx(np.array([0.5]))
        D = matrix(0.0, (5 * model.nd, 1))
        u, U_seq, X_seq = ctrl.step(y, D)
        assert u.size == (model.nu, 1)
        assert U_seq.size == (5 * model.nu, 1)
        assert X_seq.size == (5 * model.nx, 1)

    def test_step_input_within_bounds(self):
        ctrl, model = self._make_ctrl(N=10)
        y = _cvx(np.array([0.0]))
        D = matrix(0.0, (10 * model.nd, 1))
        u, _, _ = ctrl.step(y, D)
        u_val = float(list(u)[0])
        u_min, u_max = model.u_bounds
        assert u_val >= u_min[0] - 1e-6
        assert u_val <= u_max[0] + 1e-6

    def test_repeated_steps_do_not_crash(self):
        ctrl, model = self._make_ctrl(N=5)
        D = matrix(0.0, (5 * model.nd, 1))
        for k in range(8):
            y = _cvx(np.array([float(k) * 0.1]))
            u, _, _ = ctrl.step(y, D)
            assert u.size == (model.nu, 1)

    def test_closed_loop_drives_toward_reference(self):
        """CD-MPC should drive the system output toward x_ref."""
        model = SimpleLinearCD(x0=[0.0])
        kf = CDKalmanFilter(model, n_steps=10)
        Q = matrix(np.eye(1) * 5.0)
        R = matrix(np.eye(1) * 0.01)
        ocp = CDOptimalControlProblem(model, N=20, Q=Q, R=R, y_offset=10.0)
        ctrl = CDMPCController(model, estimator=kf, ocp=ocp)

        from mbc._utils import _zoh_full
        Ad, Bd, _ = _zoh_full(model.A, model.B, model.E, model.dt)
        x = np.array([0.0])
        x_ref = model.x_ref
        N_steps = 30
        for _ in range(N_steps):
            y = _cvx(model.Cm @ x)
            D = matrix(0.0, (20 * model.nd, 1))
            u, _, _ = ctrl.step(y, D)
            u_np = np.array(list(u)).ravel()
            x = Ad @ x + Bd @ u_np

        assert abs(x[0] - x_ref[0]) < 1.0, (
            f"State {x[0]:.3f} not close to reference {x_ref[0]:.1f}"
        )


# ── Tests: CDTrackingOptimalControlProblem ────────────────────────────────────


class TestCDTrackingOptimalControlProblem:
    """Tests for the nonlinear tracking OCP (SLSQP NLP solver)."""

    def _make_ocp(self, N=5, **kw):
        model = ScalarNonlinear()
        Q = np.eye(1)
        R = np.eye(1) * 0.1
        return CDTrackingOptimalControlProblem(
            model, N=N, Q=Q, R=R,
            z_ref=np.array([2.0]),
            u_min=np.array([-3.0]),
            u_max=np.array([3.0]),
            dt=1.0,
            **kw,
        ), model

    def test_solve_returns_correct_shapes(self):
        ocp, model = self._make_ocp(N=5)
        x0 = np.array([0.0])
        d_traj = np.zeros((5, model.nd))
        u_opt, cost = ocp.solve(x0, d_traj)
        assert u_opt.shape == (5, model.nu)
        assert np.isfinite(cost)

    def test_step_returns_first_action(self):
        ocp, model = self._make_ocp(N=5)
        x0 = np.array([0.0])
        d_traj = np.zeros((5, model.nd))
        u0 = ocp.step(x0, d_traj)
        assert u0.shape == (model.nu,)

    def test_input_bounds_respected(self):
        ocp, model = self._make_ocp(N=5)
        x0 = np.array([0.0])
        d_traj = np.zeros((5, model.nd))
        u_opt, _ = ocp.solve(x0, d_traj)
        assert np.all(u_opt >= -3.0 - 1e-6)
        assert np.all(u_opt <= 3.0 + 1e-6)

    def test_drives_toward_reference(self):
        """NLP tracking OCP should push the output toward z_ref=2."""
        ocp, model = self._make_ocp(N=10)
        x0 = np.array([0.0])
        d_traj = np.zeros((10, model.nd))
        u_opt, _ = ocp.solve(x0, d_traj)
        # Simulate forward with optimal u
        x = x0.copy()
        p = np.array([])
        for k in range(10):
            x = x + model.f(x, u_opt[k], d_traj[k], p, float(k)) * 1.0
        z_ref = np.array([2.0])
        assert abs(x[0] - z_ref[0]) < abs(x0[0] - z_ref[0]), (
            f"Final state {x[0]:.3f} not closer to ref {z_ref[0]:.1f} "
            f"than initial {x0[0]:.1f}"
        )

    def test_rom_penalty_reduces_variation(self):
        """Rate-of-movement penalty S should reduce input variation."""
        model = ScalarNonlinear()
        N = 8
        Q = np.eye(1)
        R = np.eye(1) * 0.01
        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))

        ocp_no_rom = CDTrackingOptimalControlProblem(
            model, N=N, Q=Q, R=R, z_ref=np.array([2.0]),
            u_min=np.array([-5.0]), u_max=np.array([5.0]), dt=1.0,
        )
        ocp_rom = CDTrackingOptimalControlProblem(
            model, N=N, Q=Q, R=R, z_ref=np.array([2.0]),
            S=np.eye(1) * 10.0,
            u_min=np.array([-5.0]), u_max=np.array([5.0]), dt=1.0,
        )
        u_no_rom, _ = ocp_no_rom.solve(x0, d_traj)
        u_rom, _ = ocp_rom.solve(x0, d_traj)
        var_no_rom = float(np.var(np.diff(u_no_rom, axis=0)))
        var_rom = float(np.var(np.diff(u_rom, axis=0)))
        assert var_rom <= var_no_rom + 1e-6

    def test_soft_state_constraint(self):
        """Soft state constraints with high rho should penalise violations."""
        model = ScalarNonlinear()
        N = 5
        Q = np.eye(1) * 0.0  # no tracking cost — only state penalty
        R = np.eye(1) * 0.01
        x0 = np.array([3.0])
        d_traj = np.zeros((N, model.nd))

        # Without state constraint, the unconstrained optimal might let x stay
        # With soft state constraint x_max = 2, input should push x down
        ocp = CDTrackingOptimalControlProblem(
            model, N=N, Q=Q, R=R, z_ref=np.array([3.0]),
            x_max=np.array([2.0]), rho_x=1e3,
            u_min=np.array([-5.0]), u_max=np.array([5.0]), dt=1.0,
        )
        u_opt, _ = ocp.solve(x0, d_traj)
        # First input should be negative (pushing x down toward constraint)
        assert u_opt[0, 0] < 0.0 + 1e-3, (
            f"Expected negative u[0] with soft upper state constraint, got {u_opt[0, 0]:.3f}"
        )


# ── Tests: EconomicOptimalControlProblem ─────────────────────────────────────


class TestEconomicOptimalControlProblem:
    """Tests for the economic nonlinear OCP."""

    def _make_model_ocp(self, N=5, **kw):
        model = ScalarNonlinear()

        # Economic cost: minimise -x (maximise state = maximise x toward x=2)
        def lagrange(x, u, d):
            return -float(x[0]) + 0.5 * float(u[0] ** 2)

        ocp = EconomicOptimalControlProblem(
            model, N=N,
            lagrange=lagrange,
            u_min=np.array([-3.0]),
            u_max=np.array([3.0]),
            dt=1.0,
            **kw,
        )
        return ocp, model

    def test_solve_returns_correct_shapes(self):
        ocp, model = self._make_model_ocp(N=5)
        x0 = np.array([0.0])
        d_traj = np.zeros((5, model.nd))
        u_opt, cost = ocp.solve(x0, d_traj)
        assert u_opt.shape == (5, model.nu)
        assert np.isfinite(cost)

    def test_step_returns_first_action(self):
        ocp, model = self._make_model_ocp(N=5)
        x0 = np.array([0.0])
        d_traj = np.zeros((5, model.nd))
        u0 = ocp.step(x0, d_traj)
        assert u0.shape == (model.nu,)

    def test_input_bounds_respected(self):
        ocp, model = self._make_model_ocp(N=5)
        x0 = np.array([0.0])
        d_traj = np.zeros((5, model.nd))
        u_opt, _ = ocp.solve(x0, d_traj)
        assert np.all(u_opt >= -3.0 - 1e-6)
        assert np.all(u_opt <= 3.0 + 1e-6)

    def test_warm_start_preserves_feasibility(self):
        """Providing a warm start should not raise and should return valid result."""
        ocp, model = self._make_model_ocp(N=5)
        x0 = np.array([0.0])
        d_traj = np.zeros((5, model.nd))
        u_prev = np.ones((5, 1)) * 0.5
        u_opt, cost = ocp.solve(x0, d_traj, u_prev=u_prev)
        assert u_opt.shape == (5, model.nu)
        assert np.isfinite(cost)

    def test_mayer_term_is_included(self):
        """A terminal (Mayer) cost should influence the solution."""
        model = ScalarNonlinear()
        N = 5

        # Without terminal cost
        def lag(x, u, d):
            return 0.1 * float(u[0] ** 2)

        ocp_no_mayer = EconomicOptimalControlProblem(
            model, N=N, lagrange=lag, dt=1.0,
        )

        # With Mayer term penalising terminal state x[N] far from 1
        def mayer(x):
            return 100.0 * float((x[0] - 1.0) ** 2)

        ocp_mayer = EconomicOptimalControlProblem(
            model, N=N, lagrange=lag, mayer=mayer, dt=1.0,
        )

        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))
        u_no, _ = ocp_no_mayer.solve(x0, d_traj)
        u_mayer, _ = ocp_mayer.solve(x0, d_traj)

        # Simulate both and compare terminal states
        p = np.array([])
        x_no = x0.copy()
        x_with = x0.copy()
        for k in range(N):
            x_no = x_no + model.f(x_no, u_no[k], d_traj[k], p, float(k))
            x_with = x_with + model.f(x_with, u_mayer[k], d_traj[k], p, float(k))
        # Mayer controller should end closer to 1.0
        assert abs(x_with[0] - 1.0) < abs(x_no[0] - 1.0) + 0.1

    def test_soft_output_constraint(self):
        """z_min soft constraint should push controlled output upward."""
        model = ScalarNonlinear()
        N = 5

        def lag(x, u, d):
            return 0.01 * float(u[0] ** 2)

        ocp = EconomicOptimalControlProblem(
            model, N=N, lagrange=lag,
            z_min=np.array([1.5]),
            rho_z=1e3,
            u_min=np.array([-5.0]),
            u_max=np.array([5.0]),
            dt=1.0,
        )
        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))
        u_opt, _ = ocp.solve(x0, d_traj)
        # First input should be positive (pushing x toward z_min)
        assert u_opt[0, 0] > 0.0 - 1e-3


# ── Tests: CDNMPCController ───────────────────────────────────────────────────


class TestCDNMPCController:
    """Tests for the generic CD-NMPC controller (nonlinear estimator + OCP)."""

    def _make_ctrl(self, N=5, ocp_cls="tracking"):
        model = ScalarNonlinear()
        x0 = np.array([0.0])
        P0 = np.eye(1)
        ekf = ContinuousDiscreteEKF(model, x0, P0, dt=1.0)

        if ocp_cls == "tracking":
            ocp = CDTrackingOptimalControlProblem(
                model, N=N, Q=np.eye(1), R=np.eye(1) * 0.1,
                z_ref=np.array([2.0]),
                u_min=np.array([-3.0]),
                u_max=np.array([3.0]),
                dt=1.0,
            )
        else:
            def lag(x, u, d):
                return -float(x[0]) + 0.1 * float(u[0] ** 2)
            ocp = EconomicOptimalControlProblem(
                model, N=N, lagrange=lag,
                u_min=np.array([-3.0]),
                u_max=np.array([3.0]),
                dt=1.0,
            )
        ctrl = CDNMPCController(estimator=ekf, ocp=ocp)
        return ctrl, model

    def test_step_returns_correct_shape_tracking(self):
        ctrl, model = self._make_ctrl(ocp_cls="tracking")
        ym = np.array([0.5])
        d_traj = np.zeros((5, model.nd))
        u = ctrl.step(ym, d_traj, p=None, t=0.0)
        assert u.shape == (model.nu,)

    def test_step_returns_correct_shape_economic(self):
        ctrl, model = self._make_ctrl(ocp_cls="economic")
        ym = np.array([0.5])
        d_traj = np.zeros((5, model.nd))
        u = ctrl.step(ym, d_traj, p=None, t=0.0)
        assert u.shape == (model.nu,)

    def test_repeated_steps_do_not_crash(self):
        ctrl, model = self._make_ctrl()
        d_traj = np.zeros((5, model.nd))
        for k in range(8):
            ym = np.array([0.1 * k])
            u = ctrl.step(ym, d_traj, p=None, t=float(k))
            assert u.shape == (model.nu,), f"Bad u shape at step {k}: {u.shape}"

    def test_closed_loop_tracking(self):
        """CDNMPCController with tracking OCP should drive state toward z_ref."""
        model = ScalarNonlinear()
        x0 = np.array([0.0])
        P0 = np.eye(1)
        ekf = ContinuousDiscreteEKF(model, x0.copy(), P0, dt=1.0)
        ocp = CDTrackingOptimalControlProblem(
            model, N=10, Q=np.eye(1) * 5.0, R=np.eye(1) * 0.01,
            z_ref=np.array([2.0]),
            u_min=np.array([-5.0]),
            u_max=np.array([5.0]),
            dt=1.0,
        )
        ctrl = CDNMPCController(estimator=ekf, ocp=ocp)

        x = x0.copy()
        p = np.array([])
        dt = 1.0
        z_ref = 2.0
        N_steps = 20
        for k in range(N_steps):
            ym = model.hm(x, np.zeros(1), np.zeros(1), p, float(k))
            d_traj = np.zeros((10, model.nd))
            u = ctrl.step(ym, d_traj, p=None, t=float(k))
            # Simple Euler step (mean dynamics, no noise)
            x = x + model.f(x, u, np.zeros(model.nd), p, float(k)) * dt

        assert abs(x[0] - z_ref) < 1.0, (
            f"State {x[0]:.3f} not close to z_ref={z_ref:.1f} after {N_steps} steps"
        )

    def test_input_within_bounds_after_steps(self):
        """All applied inputs must respect u_min / u_max."""
        ctrl, model = self._make_ctrl()
        d_traj = np.zeros((5, model.nd))
        for k in range(10):
            ym = np.array([float(k) * 0.2])
            u = ctrl.step(ym, d_traj, p=None, t=float(k))
            assert np.all(u >= -3.0 - 1e-6), f"u={u} below u_min at step {k}"
            assert np.all(u <= 3.0 + 1e-6), f"u={u} above u_max at step {k}"
