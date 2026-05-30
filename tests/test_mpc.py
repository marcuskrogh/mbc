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


def matrix(data, size=None, tc=None):
    """numpy-backed stand-in for the former cvxopt ``matrix`` constructor.

    Supports the two call patterns used in this test module:
    ``matrix(array)`` (cost matrices) and ``matrix(value_or_seq, (rows, cols))``
    (column vectors / filled matrices).
    """
    arr = np.asarray(data, dtype=float)
    if size is None:
        return arr
    rows, cols = size
    if arr.ndim == 0:
        return np.full((rows, cols), float(arr))
    return arr.reshape(rows, cols)

from mbc.models import (
    DiscreteLinearSDE,
    ContinuousDiscreteLinearSDE,
    ContinuousDiscreteSDE,
    ContinuousDiscreteSDAE,
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
    CDLinearizedMPCController,
    linearize_cd_model,
    discretize_cd_linearization,
    NLPScalingPolicy,
    ScipyNLPBackend,
)


# ── Concrete model fixtures ───────────────────────────────────────────────────


class DoubleIntegrator(DiscreteLinearSDE):
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
    def Ts(self): return 1.0

    @property
    def x(self): return list(self._x)

    @x.setter
    def x(self, val): self._x = list(val)

    @property
    def x_ref(self): return np.array([5.0, 0.0])

    @property
    def u_bounds(self): return np.array([-5.0]), np.array([5.0])


class SimpleLinearCD(ContinuousDiscreteLinearSDE):
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
    def Ts(self): return 1.0

    @property
    def x(self): return list(self._x)

    @x.setter
    def x(self, val): self._x = list(val)

    @property
    def x_ref(self): return np.array([2.0])

    @property
    def u_bounds(self): return np.array([-5.0]), np.array([5.0])


class ScalarNonlinear(ContinuousDiscreteSDE):
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

    def gm(self, x, u, d, p, t):
        return np.array([x[0]])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cvx(arr: np.ndarray) -> np.ndarray:
    """Return a 1-D numpy array (controllers accept array-like inputs)."""
    return np.asarray(arr, dtype=float).ravel()


def _np(m) -> np.ndarray:
    """Coerce a solver output to a 1-D numpy array."""
    return np.asarray(m, dtype=float).ravel()


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
        assert U.shape == (5 * model.nu,), f"U shape {U.size}"
        assert X.shape == (5 * model.nx,), f"X shape {X.size}"

    def test_solve_numpy_inputs_accepted(self):
        """solve() accepts numpy arrays for x0 and x_ref."""
        ocp = self._make_ocp(N=5)
        model = DoubleIntegrator()
        x0 = np.array([0.0, 0.0])
        D = matrix(0.0, (5 * model.nd, 1))
        x_ref = model.x_ref
        U, X = ocp.solve(x0, D, x_ref)
        assert U.shape == (5 * model.nu,)

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
        # Qd, Rm read directly from the model (DiscreteLinearSDE
        # provides them as abstract properties).
        kf = KalmanFilter(model)
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
        assert u.shape == (model.nu,)
        assert U_seq.shape == (5 * model.nu,)
        assert X_seq.shape == (5 * model.nx,)

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
            assert u.shape == (model.nu,)

    def test_closed_loop_drives_toward_reference(self):
        """Running MPC for many steps should bring the output near x_ref[0]."""
        # Use a stable scalar model for reliable closed-loop behavior
        class ScalarLinearDiscrete(DiscreteLinearSDE):
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
        kf = KalmanFilter(model)
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
        assert U.shape == (5 * model.nu,)
        assert X.shape == (5 * model.nx,)

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
        assert U.shape == (5 * model.nu,)


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
        assert u.shape == (model.nu,)
        assert U_seq.shape == (5 * model.nu,)
        assert X_seq.shape == (5 * model.nx,)

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
            assert u.shape == (model.nu,)

    def test_closed_loop_drives_toward_reference(self):
        """CD-MPC should drive the system output toward x_ref."""
        model = SimpleLinearCD(x0=[0.0])
        kf = CDKalmanFilter(model, n_steps=10)
        Q = matrix(np.eye(1) * 5.0)
        R = matrix(np.eye(1) * 0.01)
        ocp = CDOptimalControlProblem(model, N=20, Q=Q, R=R, y_offset=10.0)
        ctrl = CDMPCController(model, estimator=kf, ocp=ocp)

        from mbc._utils import _zoh_full
        Ad, Bd, _ = _zoh_full(model.A, model.B, model.E, model.Ts)
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
        u_opt, cost, _ = ocp.solve(x0, d_traj)
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
        u_opt, _, _ = ocp.solve(x0, d_traj)
        assert np.all(u_opt >= -3.0 - 1e-6)
        assert np.all(u_opt <= 3.0 + 1e-6)

    def test_drives_toward_reference(self):
        """NLP tracking OCP should push the output toward z_ref=2."""
        ocp, model = self._make_ocp(N=10)
        x0 = np.array([0.0])
        d_traj = np.zeros((10, model.nd))
        u_opt, _, _ = ocp.solve(x0, d_traj)
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
        u_no_rom, _, _ = ocp_no_rom.solve(x0, d_traj)
        u_rom, _, _ = ocp_rom.solve(x0, d_traj)
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
        u_opt, _, _ = ocp.solve(x0, d_traj)
        # First input should be negative (pushing x down toward constraint)
        assert u_opt[0, 0] < 0.0 + 1e-3, (
            f"Expected negative u[0] with soft upper state constraint, got {u_opt[0, 0]:.3f}"
        )


# ── Tests: EconomicOptimalControlProblem ─────────────────────────────────────


class TestEconomicOptimalControlProblem:
    """Tests for the economic nonlinear OCP."""

    def _make_model_ocp(self, N=5, **kw):
        model = ScalarNonlinear()

        # Economic cost: minimise −x + 0.5 u² (i.e. push x toward larger values
        # while paying for input).  ControlToolbox §EMPC Lagrange signature is
        # ``l(t, x, y, u, theta)``.
        def lagrange(t, x, y, u, theta):
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
        u_opt, cost, _ = ocp.solve(x0, d_traj)
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
        u_opt, _, _ = ocp.solve(x0, d_traj)
        assert np.all(u_opt >= -3.0 - 1e-6)
        assert np.all(u_opt <= 3.0 + 1e-6)

    def test_warm_start_preserves_feasibility(self):
        """Providing a warm start should not raise and should return valid result."""
        ocp, model = self._make_model_ocp(N=5)
        x0 = np.array([0.0])
        d_traj = np.zeros((5, model.nd))
        u_prev = np.ones((5, 1)) * 0.5
        u_opt, cost, _ = ocp.solve(x0, d_traj, u_prev=u_prev)
        assert u_opt.shape == (5, model.nu)
        assert np.isfinite(cost)

    def test_mayer_term_is_included(self):
        """A terminal (Mayer) cost should influence the solution."""
        model = ScalarNonlinear()
        N = 5

        # Lagrange: ControlToolbox §EMPC signature ``l(t, x, y, u, theta)``.
        def lag(t, x, y, u, theta):
            return 0.1 * float(u[0] ** 2)

        ocp_no_mayer = EconomicOptimalControlProblem(
            model, N=N, lagrange=lag, dt=1.0,
        )

        # Mayer: ControlToolbox §EMPC signature ``l_hat(x, y, theta)``.
        def mayer(x, y, theta):
            return 100.0 * float((x[0] - 1.0) ** 2)

        ocp_mayer = EconomicOptimalControlProblem(
            model, N=N, lagrange=lag, mayer=mayer, dt=1.0,
        )

        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))
        u_no, _, _ = ocp_no_mayer.solve(x0, d_traj)
        u_mayer, _, _ = ocp_mayer.solve(x0, d_traj)

        # Simulate both forward and compare terminal states
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

        def lag(t, x, y, u, theta):
            return 0.01 * float(u[0] ** 2)

        ocp = EconomicOptimalControlProblem(
            model, N=N, lagrange=lag,
            z_min=np.array([1.5]),
            rho_z_2=1e3,
            u_min=np.array([-5.0]),
            u_max=np.array([5.0]),
            dt=1.0,
        )
        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))
        u_opt, _, _ = ocp.solve(x0, d_traj)
        # First input should be positive (pushing x toward z_min)
        assert u_opt[0, 0] > 0.0 - 1e-3

    def test_soft_output_constraints_share_one_slack_per_output(self):
        """Soft z bounds should use one shared slack block per output."""
        model = ScalarNonlinear()
        N = 4
        ocp = EconomicOptimalControlProblem(
            model,
            N=N,
            z_min=np.array([-0.5]),
            z_max=np.array([0.5]),
            rho_z_2=1e3,
            u_min=np.array([-5.0]),
            u_max=np.array([5.0]),
            dt=1.0,
        )
        L = ocp._layout
        assert L.pz_size == (L.M + 1) * model.nz
        assert not hasattr(L, "pz_hi_size")
        expected_total = (
            L.u_size + L.x_size + L.y_size
            + L.px_lo_size + L.px_hi_size + L.pz_size
        )
        assert L.total == expected_total

        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))
        u_opt, cost, _ = ocp.solve(x0, d_traj)
        assert u_opt.shape == (N, model.nu)
        assert np.isfinite(cost)

    def test_solver_backend_swap_with_reserved_scipy_key(self):
        """Selecting solver='scipy' should run via backend wrapper."""
        model = ScalarNonlinear()
        N = 5
        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))

        ocp = EconomicOptimalControlProblem(
            model,
            N=N,
            Q_z=np.array([[1.0]]),
            z_ref=np.array([2.0]),
            u_min=np.array([-3.0]),
            u_max=np.array([3.0]),
            solver="scipy",
            solver_options={"method": "SLSQP", "maxiter": 80},
            dt=1.0,
        )
        u_opt, cost, info = ocp.solve(x0, d_traj)
        assert u_opt.shape == (N, model.nu)
        assert np.isfinite(cost)
        assert info["result"].success

    def test_solver_backend_swap_with_custom_backend_object(self):
        """A pluggable backend object should be accepted by EOCP."""
        model = ScalarNonlinear()
        N = 4
        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))
        backend = ScipyNLPBackend(method="SLSQP", options={"maxiter": 80})
        ocp = EconomicOptimalControlProblem(
            model,
            N=N,
            Q_z=np.array([[1.0]]),
            z_ref=np.array([1.0]),
            u_min=np.array([-2.0]),
            u_max=np.array([2.0]),
            solver=backend,
            dt=1.0,
        )
        u_opt, cost, info = ocp.solve(x0, d_traj)
        assert u_opt.shape == (N, model.nu)
        assert np.isfinite(cost)
        assert info["result"].success

    def test_solver_scaling_policy_is_supported(self):
        """Scaling controls should be accepted and produce a valid NLP solve."""
        model = ScalarNonlinear()
        N = 5
        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))
        scaling = NLPScalingPolicy(
            objective_scale=0.5,
            variable_scale=10.0,
            constraint_scale=2.0,
        )
        ocp = EconomicOptimalControlProblem(
            model,
            N=N,
            Q_z=np.array([[1.0]]),
            z_ref=np.array([2.0]),
            u_min=np.array([-3.0]),
            u_max=np.array([3.0]),
            solver="SLSQP",
            solver_options={"maxiter": 80},
            solver_scaling=scaling,
            dt=1.0,
        )
        u_opt, cost, info = ocp.solve(x0, d_traj)
        assert u_opt.shape == (N, model.nu)
        assert np.isfinite(cost)
        assert info["result"].success

    def test_ipopt_backend_if_available(self):
        """IPOPT backend should solve the SDE EOCP when cyipopt is installed."""
        pytest.importorskip("cyipopt")
        model = ScalarNonlinear()
        N = 4
        x0 = np.array([0.0])
        d_traj = np.zeros((N, model.nd))
        ocp = EconomicOptimalControlProblem(
            model,
            N=N,
            Q_z=np.array([[1.0]]),
            z_ref=np.array([1.5]),
            u_min=np.array([-2.0]),
            u_max=np.array([2.0]),
            solver="ipopt",
            solver_options={"max_iter": 100},
            dt=1.0,
        )
        u_opt, cost, info = ocp.solve(x0, d_traj)
        assert u_opt.shape == (N, model.nu)
        assert np.isfinite(cost)
        assert info["result"].success


# ── Tests: EconomicOptimalControlProblem on SDAE plant ──────────────────────


class _IsomerisationReactor(ContinuousDiscreteSDAE):
    """
    Minimal SDAE plant for testing the EOCP's SDAE code path.

    Differential state : x = [C_tot]      total concentration (mol/L)
    Algebraic state    : y = [C_A]        species-A concentration (mol/L)
    Algebraic constraint: (K_eq + 1) * C_A − C_tot = 0
    Drift              : dC_tot/dt = u * (C_feed − C_tot)
    Output             : z = C_A
    """

    _K_eq = 3.0
    _C_feed = 5.0

    @property
    def nx(self): return 1
    @property
    def ny(self): return 1
    @property
    def nu(self): return 1
    @property
    def nd(self): return 0
    @property
    def nw(self): return 1
    @property
    def nym(self): return 1
    @property
    def nz(self): return 1
    @property
    def Rm(self): return np.array([[0.01]])

    def f(self, x, y, u, d, p, t):
        return np.array([u[0] * (self._C_feed - x[0])])

    def sigma(self, x, y, u, d, p, t):
        return np.array([[0.02]])

    def g(self, x, y, u, d, p, t):
        return np.array([(self._K_eq + 1.0) * y[0] - x[0]])

    def gm(self, x, y, u, d, p, t):
        return np.array([y[0]])

    def hm(self, x, y, u, d, p, t):
        return np.array([y[0]])


class TestEconomicOCPOnSDAE:
    """The EOCP's direct-simultaneous formulation must satisfy g(x, y, p) = 0
    to machine precision at every sub-step (ControlToolbox §EMPC)."""

    def _make_ocp(self, N=5, n_steps=2):
        model = _IsomerisationReactor()
        ocp = EconomicOptimalControlProblem(
            model, N=N,
            Q_z=np.array([[1.0]]), z_ref=np.array([1.5]),
            u_min=np.array([0.0]), u_max=np.array([1.0]),
            n_steps=n_steps, dt=1.0,
        )
        return ocp, model

    def test_solve_returns_X_and_Y(self):
        ocp, model = self._make_ocp(N=5, n_steps=2)
        x0 = np.array([4.0])               # consistent with C_A = 1.0
        d_traj = np.zeros((5, 0))
        u_opt, cost, info = ocp.solve(x0, d_traj)
        M = 5 * 2
        assert u_opt.shape == (5, model.nu)
        assert info["X"].shape == (M + 1, model.nx)
        assert info["Y"].shape == (M + 1, model.ny)
        assert np.isfinite(cost)

    def test_algebraic_constraint_satisfied(self):
        """g(x_n, y_n, p) = 0 must hold at every sub-step of the optimal trajectory."""
        ocp, model = self._make_ocp(N=5, n_steps=2)
        x0 = np.array([4.0])
        d_traj = np.zeros((5, 0))
        _, _, info = ocp.solve(x0, d_traj)
        X, Y = info["X"], info["Y"]
        zeros_u = np.zeros(model.nu)
        zeros_d = np.zeros(0)
        empty_p = np.array([], dtype=float)
        for n in range(X.shape[0]):
            g_val = model.g(X[n], Y[n], zeros_u, zeros_d, empty_p, 0.0)
            assert np.allclose(g_val, 0.0, atol=1e-6), (
                f"Algebraic constraint violated at sub-step {n}: g = {g_val}"
            )

    def test_tracking_drives_y_toward_zref(self):
        """With Q_z=1 and z_ref=1.5, the optimal y_M should be closer to 1.5
        than the initial y_0 = 1.0."""
        ocp, model = self._make_ocp(N=8, n_steps=2)
        x0 = np.array([4.0])               # → y_0 = 1.0 by constraint
        d_traj = np.zeros((8, 0))
        _, _, info = ocp.solve(x0, d_traj)
        y_initial = info["Y"][0, 0]
        y_final = info["Y"][-1, 0]
        assert abs(y_final - 1.5) < abs(y_initial - 1.5), (
            f"Tracking failed: y_final={y_final:.4f}, y_initial={y_initial:.4f}"
        )

    def test_ipopt_backend_if_available(self):
        """IPOPT backend should solve SDAE EOCP when cyipopt is installed."""
        pytest.importorskip("cyipopt")
        ocp, model = self._make_ocp(N=4, n_steps=2)
        # Rebuild with explicit IPOPT backend selector.
        ocp = EconomicOptimalControlProblem(
            model, N=4,
            Q_z=np.array([[1.0]]), z_ref=np.array([1.5]),
            u_min=np.array([0.0]), u_max=np.array([1.0]),
            n_steps=2, dt=1.0,
            solver="ipopt",
            solver_options={"max_iter": 120},
        )
        x0 = np.array([4.0])
        d_traj = np.zeros((4, 0))
        u_opt, cost, info = ocp.solve(x0, d_traj)
        assert u_opt.shape == (4, model.nu)
        assert np.isfinite(cost)
        assert info["result"].success


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
            def lag(t, x, y, u, theta):
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


# ── Tests: Analytical Jacobians ───────────────────────────────────────────────


class TestAnalyticalJacobians:
    """
    Verify that the analytical constraint and objective Jacobians match the
    finite-difference Jacobians produced by the NLP itself.
    """

    def _make_sde_ocp(self):
        """Small SDE tracking OCP with ROM and soft-z constraints."""
        model = ScalarNonlinear()
        return EconomicOptimalControlProblem(
            model,
            N=2,
            Q_z=np.eye(1) * 2.0,
            z_ref=np.array([1.5]),
            Q_du=np.eye(1) * 0.5,
            p_u_eco=np.array([0.1]),
            du_min=np.array([-2.0]),
            du_max=np.array([2.0]),
            z_min=np.array([0.0]),
            z_max=np.array([3.0]),
            rho_z_2=500.0,
            n_steps=2,
            dt=1.0,
        )

    def _fd_jac(self, fun, z, h=1e-5):
        """Forward finite-difference Jacobian of fun(z)."""
        f0 = np.asarray(fun(z), dtype=float)
        J = np.zeros((f0.size, z.size))
        for i in range(z.size):
            ze = z.copy()
            ze[i] += h
            J[:, i] = (np.asarray(fun(ze), dtype=float) - f0) / h
        return J

    def _setup_ocp_state(self, ocp):
        """Return (z0, x_hat, d_traj, u_prev_0, p_theta, t0) for testing."""
        L = ocp._layout
        x0 = np.array([0.3])
        d_traj = np.zeros((ocp._N, ocp._nd))
        u_prev_0 = np.zeros(ocp._nu)
        p_theta = np.array([])
        t0 = 0.0
        z0 = ocp._build_initial_guess(x0, None, None, None)
        return z0, x0, d_traj, u_prev_0, p_theta, t0

    def test_equality_jac_matches_fd(self):
        ocp = self._make_sde_ocp()
        z0, x_hat, d_traj, _, p_theta, t0 = self._setup_ocp_state(ocp)

        fun = lambda z: ocp._equality_constraints(z, x_hat, d_traj, p_theta, t0)
        J_ana = ocp._equality_constraint_jac(z0, x_hat, d_traj, p_theta, t0)
        J_fd = self._fd_jac(fun, z0)

        assert J_ana.shape == J_fd.shape, f"shape mismatch: {J_ana.shape} vs {J_fd.shape}"
        np.testing.assert_allclose(J_ana, J_fd, atol=1e-4, rtol=1e-4,
                                   err_msg="Equality constraint Jacobian mismatch")

    def test_inequality_jac_matches_fd(self):
        ocp = self._make_sde_ocp()
        z0, _, d_traj, u_prev_0, p_theta, t0 = self._setup_ocp_state(ocp)

        fun = lambda z: ocp._inequality_constraints(z, u_prev_0, d_traj, p_theta, t0)
        J_ana = ocp._inequality_constraint_jac(z0, u_prev_0, d_traj, p_theta, t0)
        J_fd = self._fd_jac(fun, z0)

        assert J_ana.shape == J_fd.shape, f"shape mismatch: {J_ana.shape} vs {J_fd.shape}"
        np.testing.assert_allclose(J_ana, J_fd, atol=1e-4, rtol=1e-4,
                                   err_msg="Inequality constraint Jacobian mismatch")

    def test_objective_jac_matches_fd_when_all_terms_analytical(self):
        """For a pure tracking OCP (no user lagrange/mayer), objective grad is analytical."""
        model = ScalarNonlinear()
        ocp = EconomicOptimalControlProblem(
            model,
            N=2,
            Q_z=np.eye(1) * 2.0,
            z_ref=np.array([1.5]),
            Q_du=np.eye(1) * 0.5,
            p_u_eco=np.array([0.1]),
            n_steps=2,
            dt=1.0,
        )
        z0, x_hat, d_traj, u_prev_0, p_theta, t0 = self._setup_ocp_state(ocp)

        assert ocp._can_use_analytical_objective_jac(), (
            "Expected analytical objective Jacobian to be available"
        )

        fun = lambda z: np.array([ocp._objective(z, x_hat, d_traj, u_prev_0, p_theta, t0)])
        grad_ana = ocp._objective_jac(z0, d_traj, u_prev_0, p_theta, t0)
        grad_fd = self._fd_jac(fun, z0).ravel()

        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4, rtol=1e-4,
                                   err_msg="Objective gradient mismatch")

    def test_cdtracking_ocp_has_analytical_jac(self):
        """CDTrackingOptimalControlProblem always provides analytical lagrange/mayer Jacs."""
        model = ScalarNonlinear()
        ocp_wrapper = CDTrackingOptimalControlProblem(
            model, N=3,
            Q=np.eye(1) * 2.0,
            R=np.eye(1) * 0.1,
            P=np.eye(1) * 5.0,
            z_ref=np.array([1.5]),
            dt=1.0,
        )
        eocp = ocp_wrapper._eocp
        assert eocp._can_use_analytical_objective_jac(), (
            "CDTrackingOCP should provide analytical Jacobians for all objective terms"
        )

    def test_cdtracking_ocp_objective_jac_matches_fd(self):
        """CDTrackingOptimalControlProblem objective gradient matches finite differences."""
        model = ScalarNonlinear()
        ocp_wrapper = CDTrackingOptimalControlProblem(
            model, N=2,
            Q=np.eye(1) * 2.0,
            R=np.eye(1) * 0.1,
            P=np.eye(1) * 5.0,
            z_ref=np.array([1.5]),
            dt=1.0,
            n_steps=2,
        )
        ocp = ocp_wrapper._eocp
        x0 = np.array([0.5])
        d_traj = np.zeros((2, model.nd))
        u_prev_0 = np.zeros(1)
        p_theta = np.array([])
        t0 = 0.0
        z0 = ocp._build_initial_guess(x0, None, None, None)

        fun = lambda z: np.array([ocp._objective(z, x0, d_traj, u_prev_0, p_theta, t0)])
        grad_ana = ocp._objective_jac(z0, d_traj, u_prev_0, p_theta, t0)
        grad_fd = self._fd_jac(fun, z0).ravel()

        np.testing.assert_allclose(grad_ana, grad_fd, atol=1e-4, rtol=1e-4,
                                   err_msg="CDTracking objective gradient mismatch")

    def test_solve_produces_same_result_with_analytical_jac(self):
        """Solving with analytical Jacobians yields the same optimum as without."""
        model = ScalarNonlinear()

        # Without analytical Jacobians (no lagrange_jac)
        ocp_nograd = EconomicOptimalControlProblem(
            model, N=3,
            lagrange=lambda t, x, y, u, p: float(u @ u),
            Q_z=np.eye(1) * 2.0,
            z_ref=np.array([1.5]),
            n_steps=3,
            dt=1.0,
        )

        # With analytical Jacobians
        ocp_grad = EconomicOptimalControlProblem(
            model, N=3,
            lagrange=lambda t, x, y, u, p: float(u @ u),
            lagrange_jac=lambda t, x, y, u, p: (
                np.zeros_like(x), np.zeros_like(y), 2.0 * u
            ),
            Q_z=np.eye(1) * 2.0,
            z_ref=np.array([1.5]),
            n_steps=3,
            dt=1.0,
        )

        x0 = np.array([0.0])
        d_traj = np.zeros((3, model.nd))

        U_nograd, cost_nograd, _ = ocp_nograd.solve(x0, d_traj)
        U_grad, cost_grad, _ = ocp_grad.solve(x0, d_traj)

        np.testing.assert_allclose(U_nograd, U_grad, atol=1e-4,
                                   err_msg="Optimal inputs differ with/without analytical Jac")
        assert abs(cost_nograd - cost_grad) < 1e-4, (
            f"Cost differs: {cost_nograd:.6f} vs {cost_grad:.6f}"
        )


# ── Tests: CDLinearizedMPCController and linearisation utilities ─────────────────


class _ScalarBoundedNonlinear(ContinuousDiscreteSDE):
    """Scalar nonlinear model with bounded input for linearised-MPC tests."""

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
        return np.array([-x[0] + 0.2 * x[0] * x[0] + u[0] + 0.5 * d[0]])

    def sigma(self, x, u, d, p, t):
        return np.array([[0.1]])

    def hm(self, x, u, d, p, t=0.0):
        return np.array([x[0]])

    def gm(self, x, u, d, p, t):
        return np.array([x[0]])


class _DummyEstimator2:
    """Estimator stub returning (x_hat, P) with standard step signature."""

    def __init__(self, x0):
        self._x = np.asarray(x0, dtype=float)

    def step(self, y, u, d, p, t):
        self._x = np.asarray(y, dtype=float).copy()
        return self._x.copy(), np.eye(self._x.shape[0])


class _DummyEstimator3:
    """Estimator stub returning (x_hat, y_hat, P) to mimic DAE-EKF output."""

    def __init__(self, x0):
        self._x = np.asarray(x0, dtype=float)

    def step(self, y, u, d, p, t):
        self._x = np.asarray(y, dtype=float).copy()
        return self._x.copy(), self._x.copy(), np.eye(self._x.shape[0])


class TestCDLinearizationUtilities:
    def test_linearize_dimensions(self):
        model = _ScalarBoundedNonlinear()
        lin = linearize_cd_model(
            model=model,
            x_ss=np.array([0.2]),
            u_ss=np.array([0.1]),
            d_ss=np.array([0.3]),
            p=np.array([]),
            t=0.0,
        )
        assert lin["A"].shape == (1, 1)
        assert lin["B"].shape == (1, 1)
        assert lin["E"].shape == (1, 1)
        assert lin["Cm"].shape == (1, 1)
        assert lin["Cz"].shape == (1, 1)
        assert lin["G"].shape == (1, 1)

    def test_linearize_matches_known_linear_model(self):
        model = SimpleLinearCD()
        lin = linearize_cd_model(
            model=model,
            x_ss=np.array([1.2]),
            u_ss=np.array([0.7]),
            d_ss=np.array([0.0]),
            p=np.array([]),
            t=0.0,
        )
        np.testing.assert_allclose(lin["A"], model.A)
        np.testing.assert_allclose(lin["B"], model.B)
        np.testing.assert_allclose(lin["E"], model.E)
        np.testing.assert_allclose(lin["Cm"], model.Cm)
        np.testing.assert_allclose(lin["Cz"], model.Cz)

    def test_discretize_matches_zoh_for_known_linear_model(self):
        model = SimpleLinearCD()
        lin = linearize_cd_model(
            model=model,
            x_ss=np.array([0.0]),
            u_ss=np.array([0.0]),
            d_ss=np.array([0.0]),
            p=np.array([]),
            t=0.0,
        )
        disc = discretize_cd_linearization(lin, dt=model.Ts)
        from mbc._utils import _zoh_full
        Ad_ref, Bd_ref, Ed_ref = _zoh_full(model.A, model.B, model.E, model.Ts)
        np.testing.assert_allclose(disc["Ad"], Ad_ref, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(disc["Bd"], Bd_ref, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(disc["Ed"], Ed_ref, atol=1e-10, rtol=1e-10)


class TestCDLinearizedMPCController:
    def _make_ctrl(self, estimator, x_ref=np.array([2.0])):
        model = _ScalarBoundedNonlinear()
        Q = matrix(np.eye(1) * 5.0)
        R = matrix(np.eye(1) * 0.05)
        ctrl = CDLinearizedMPCController(
            model=model,
            estimator=estimator,
            N=8,
            Q=Q,
            R=R,
            dt=1.0,
            u_min=np.array([-1.0]),
            u_max=np.array([1.0]),
            x_ref=x_ref,
            y_offset=10.0,
        )
        return ctrl, model

    def test_step_returns_absolute_action_and_sequences(self):
        est = _DummyEstimator2([0.0])
        ctrl, model = self._make_ctrl(estimator=est)
        u, U, X = ctrl.step(y=np.array([0.0]), d=np.array([0.0]), p=np.array([]), t=0.0)
        assert u.shape == (model.nu,)
        assert U.shape == (8, model.nu)
        assert X.shape == (8, model.nx)

    def test_step_respects_absolute_bounds(self):
        est = _DummyEstimator2([0.0])
        ctrl, _ = self._make_ctrl(estimator=est, x_ref=np.array([10.0]))
        u, _, _ = ctrl.step(y=np.array([0.0]), d=np.array([0.0]), p=np.array([]), t=0.0)
        assert np.all(u >= -1.0 - 1e-8)
        assert np.all(u <= 1.0 + 1e-8)

    def test_relinearizes_each_step(self):
        x0 = np.array([0.0])
        P0 = np.eye(1)
        model = _ScalarBoundedNonlinear()
        ekf = ContinuousDiscreteEKF(model, x0=x0, P0=P0, dt=1.0)
        Q = matrix(np.eye(1) * 2.0)
        R = matrix(np.eye(1) * 0.1)
        ctrl = CDLinearizedMPCController(
            model=model, estimator=ekf, N=5, Q=Q, R=R, dt=1.0,
            u_min=np.array([-2.0]), u_max=np.array([2.0]), x_ref=np.array([1.0]), y_offset=10.0,
        )

        ctrl.step(y=np.array([0.0]), d=np.array([0.0]), p=np.array([]), t=0.0)
        Ad_0 = ctrl._lin_model.Ad.copy()
        ctrl.step(y=np.array([1.0]), d=np.array([0.5]), p=np.array([]), t=1.0)
        Ad_1 = ctrl._lin_model.Ad.copy()

        assert not np.allclose(Ad_0, Ad_1), "Expected re-linearization to update local Ad"

    def test_disturbance_hold_assumption_uses_zero_deviation_trajectory(self):
        est = _DummyEstimator2([0.0])
        ctrl, model = self._make_ctrl(estimator=est)
        ctrl.step(y=np.array([0.0]), d=np.array([0.7]), p=np.array([]), t=0.0)
        D_dev = ctrl.last_disturbance_deviation_trajectory
        assert D_dev.shape == (8, model.nd)
        assert np.allclose(D_dev, 0.0)

    def test_closed_loop_moves_toward_reference(self):
        model = _ScalarBoundedNonlinear()
        x0 = np.array([0.0])
        P0 = np.eye(1)
        ekf = ContinuousDiscreteEKF(model, x0=x0.copy(), P0=P0, dt=1.0, n_steps=8)

        Q = matrix(np.eye(1) * 8.0)
        R = matrix(np.eye(1) * 0.05)
        ctrl = CDLinearizedMPCController(
            model=model, estimator=ekf, N=10, Q=Q, R=R, dt=1.0,
            u_min=np.array([-2.0]), u_max=np.array([2.0]), x_ref=np.array([2.0]), y_offset=10.0,
        )

        x = x0.copy()
        p = np.array([])
        for k in range(15):
            y = model.hm(x, np.zeros(model.nu), np.zeros(model.nd), p, float(k))
            u, _, _ = ctrl.step(y=y, d=np.zeros(model.nd), p=p, t=float(k))
            x = x + model.f(x, u, np.zeros(model.nd), p, float(k)) * 1.0

        assert abs(x[0] - 2.0) < 1.0

    def test_estimator_return_tuple_compatibility_regression(self):
        ctrl2, _ = self._make_ctrl(estimator=_DummyEstimator2([0.0]))
        u2, _, _ = ctrl2.step(y=np.array([0.2]), d=np.array([0.0]), p=np.array([]), t=0.0)

        ctrl3, _ = self._make_ctrl(estimator=_DummyEstimator3([0.0]))
        u3, _, _ = ctrl3.step(y=np.array([0.2]), d=np.array([0.0]), p=np.array([]), t=0.0)

        assert u2.shape == (1,)
        assert u3.shape == (1,)


# ── Tests: condensed vs sparse formulation equivalence ───────────────────────


class TestOCPFormulationEquivalence:
    """The condensed and sparse QP formulations must give the same optimum."""

    # Pin the exact HiGHS active-set backend so the test validates the
    # formulation *math* (OSQP's first-order tolerance is checked separately).
    @pytest.mark.parametrize("N", [3, 10, 30])
    @pytest.mark.parametrize("use_rom", [False, True])
    def test_discrete_condensed_equals_sparse(self, N, use_rom):
        model = DoubleIntegrator()
        kw = dict(model=model, N=N, Q=np.eye(1), R=np.eye(1) * 0.1, y_offset=2.0,
                  solver="highs")
        if use_rom:
            kw["S"] = np.eye(1) * 5.0
        ocp_c = OptimalControlProblem(formulation="condensed", **kw)
        ocp_s = OptimalControlProblem(formulation="sparse", **kw)

        x0 = np.array([0.0, 0.0])
        D = np.zeros(N * model.nd)
        x_ref = model.x_ref
        Uc, Xc = ocp_c.solve(x0, D, x_ref, u_prev=np.array([0.5]))
        Us, Xs = ocp_s.solve(x0, D, x_ref, u_prev=np.array([0.5]))
        np.testing.assert_allclose(Uc, Us, atol=1e-5)
        np.testing.assert_allclose(Xc, Xs, atol=1e-5)

    def test_cd_condensed_equals_sparse(self):
        model = SimpleLinearCD()
        N = 12
        kw = dict(model=model, N=N, Q=np.eye(1), R=np.eye(1) * 0.1, y_offset=10.0,
                  solver="highs")
        ocp_c = CDOptimalControlProblem(formulation="condensed", **kw)
        ocp_s = CDOptimalControlProblem(formulation="sparse", **kw)
        x0 = np.array([0.0])
        D = np.zeros(N * model.nd)
        x_ref = model.x_ref
        Uc, Xc = ocp_c.solve(x0, D, x_ref)
        Us, Xs = ocp_s.solve(x0, D, x_ref)
        np.testing.assert_allclose(Uc, Us, atol=1e-5)
        np.testing.assert_allclose(Xc, Xs, atol=1e-5)

    def test_all_backends_agree_on_optimum(self):
        """HiGHS and OSQP (with their auto formulations) reach the same U*."""
        model = DoubleIntegrator()
        N = 15
        x0 = np.array([0.0, 0.0])
        D = np.zeros(N * model.nd)
        x_ref = model.x_ref
        kw = dict(model=model, N=N, Q=np.eye(1), R=np.eye(1) * 0.1, y_offset=2.0)
        U_ref, _ = OptimalControlProblem(solver="highs", formulation="condensed",
                                         **kw).solve(x0, D, x_ref)
        for solver in ("highs", "osqp"):
            U, _ = OptimalControlProblem(solver=solver, **kw).solve(x0, D, x_ref)
            np.testing.assert_allclose(U, U_ref, atol=1e-4,
                                       err_msg=f"{solver} disagrees with reference")

    def test_auto_is_backend_aware(self):
        """``auto`` → sparse for OSQP (banded-exploiting), condensed for HiGHS."""
        model = DoubleIntegrator()
        kw = dict(Q=np.eye(1), R=np.eye(1) * 0.1, y_offset=2.0)
        for N in (5, 40, 100):
            assert OptimalControlProblem(
                model, N=N, solver="osqp", **kw)._resolve_formulation() == "sparse"
            assert OptimalControlProblem(
                model, N=N, solver="highs", **kw)._resolve_formulation() == "condensed"

    def test_explicit_formulation_overrides_auto(self):
        model = DoubleIntegrator()
        ocp = OptimalControlProblem(
            model, N=5, Q=np.eye(1), R=np.eye(1), formulation="sparse"
        )
        assert ocp._resolve_formulation() == "sparse"

    def test_invalid_formulation_raises(self):
        model = DoubleIntegrator()
        with pytest.raises(ValueError):
            OptimalControlProblem(model, N=5, Q=np.eye(1), R=np.eye(1),
                                  formulation="banana")


# ── Tests: warm-starting the receding-horizon loop ───────────────────────────


class TestWarmStartMPC:
    """Warm-starting must not change the closed-loop trajectory."""

    def _run(self, warm_start, N=10, n_steps=12, formulation="auto"):
        model = DoubleIntegrator()
        kf = KalmanFilter(model)
        ocp = OptimalControlProblem(
            model, N=N, Q=np.eye(1), R=np.eye(1) * 0.1, y_offset=20.0,
            formulation=formulation,
        )
        ctrl = MPCController(model, estimator=kf, ocp=ocp, warm_start=warm_start)
        us = []
        for k in range(n_steps):
            y = np.array([0.1 * k])
            D = np.zeros(N * model.nd)
            u, _, _ = ctrl.step(y, D)
            us.append(np.asarray(u, dtype=float).copy())
        return np.array(us)

    def test_warm_vs_cold_same_inputs(self):
        u_cold = self._run(warm_start=False)
        u_warm = self._run(warm_start=True)
        np.testing.assert_allclose(u_warm, u_cold, atol=1e-5)

    def test_warm_start_with_sparse_formulation(self):
        u_cold = self._run(warm_start=False, formulation="sparse")
        u_warm = self._run(warm_start=True, formulation="sparse")
        np.testing.assert_allclose(u_warm, u_cold, atol=1e-5)

    def test_cd_warm_vs_cold_same_inputs(self):
        def run(warm):
            model = SimpleLinearCD()
            kf = CDKalmanFilter(model, n_steps=10)
            ocp = CDOptimalControlProblem(
                model, N=10, Q=np.eye(1), R=np.eye(1) * 0.1, y_offset=10.0
            )
            ctrl = CDMPCController(model, estimator=kf, ocp=ocp, warm_start=warm)
            us = []
            for k in range(10):
                u, _, _ = ctrl.step(np.array([0.1 * k]), np.zeros(10 * model.nd))
                us.append(float(np.asarray(u).ravel()[0]))
            return np.array(us)
        np.testing.assert_allclose(run(True), run(False), atol=1e-5)
