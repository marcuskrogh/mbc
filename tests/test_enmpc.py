"""
Tests for EconomicOptimalControlProblem and CDNMPCController.

Uses a simple scalar linear system:

    dx/dt = -x + u,   y = x,   output = x + u

as the test model.  The economic stage cost is  l_e(x, u, d) = x^2 + 0.1*u^2
(standard LQR-style cost).

Test coverage
-------------
- ``output()`` default (falls back to ``h``) and custom override.
- ``n_out`` property.
- Soft state and output constraint penalties: zero when feasible, positive
  when violated, and quadratically scaling with violation magnitude.
- ``EconomicOptimalControlProblem.solve`` returns a finite cost and a
  (N, nu) array.
- ``CDNMPCController.step`` returns a scalar input.
- Backward-compat: ``EconomicNMPCController`` is the same class as
  ``CDNMPCController``.
"""

from __future__ import annotations

import numpy as np
import pytest

import mbc
from mbc.models import ContinuousDiscreteModel
from mbc.control import EconomicOptimalControlProblem, CDNMPCController, EconomicNMPCController


# ── Minimal CD model ──────────────────────────────────────────────────────────


class ScalarModel(ContinuousDiscreteModel):
    """dx/dt = -x + u,  y = x,  output = x + u  (measurement ≠ output)."""

    dt = 0.1

    @property
    def nx(self) -> int:
        return 1

    @property
    def nu(self) -> int:
        return 1

    @property
    def nd(self) -> int:
        return 0

    @property
    def ny(self) -> int:
        return 1

    @property
    def nw(self) -> int:
        return 1

    @property
    def n_out(self) -> int:
        return 1

    @property
    def Q_c(self) -> np.ndarray:
        return np.array([[0.01]])

    @property
    def R(self) -> np.ndarray:
        return np.array([[0.1]])

    def f(self, x, u, d, p, t):
        return np.array([-x[0] + u[0]])

    def g(self, x, u, d, p, t):
        return np.array([[0.1]])

    def h(self, x, u, d, p):
        return x.copy()

    def output(self, x, u, d, p):
        """Controlled output: sum of state and input."""
        return np.array([x[0] + u[0]])


class ScalarModelDefaultOutput(ContinuousDiscreteModel):
    """Same dynamics but relies on the default output = h fallback."""

    dt = 0.1

    @property
    def nx(self) -> int:
        return 1

    @property
    def nu(self) -> int:
        return 1

    @property
    def nd(self) -> int:
        return 0

    @property
    def ny(self) -> int:
        return 1

    @property
    def nw(self) -> int:
        return 1

    @property
    def Q_c(self) -> np.ndarray:
        return np.array([[0.01]])

    @property
    def R(self) -> np.ndarray:
        return np.array([[0.1]])

    def f(self, x, u, d, p, t):
        return np.array([-x[0] + u[0]])

    def g(self, x, u, d, p, t):
        return np.array([[0.1]])

    def h(self, x, u, d, p):
        return x.copy()


# ── Minimal stub estimator ────────────────────────────────────────────────────


class TrivialEstimator:
    """Estimator that returns the measurement as the state estimate."""

    def step(self, y, u, d, p, t):
        return np.asarray(y, dtype=float).ravel(), np.eye(len(np.asarray(y).ravel()))


# ── Model fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def model():
    return ScalarModel()


@pytest.fixture
def model_default():
    return ScalarModelDefaultOutput()


# ── output() and n_out ────────────────────────────────────────────────────────


def test_output_custom_override(model):
    x = np.array([1.0])
    u = np.array([0.5])
    d = np.zeros(0)
    p = np.zeros(0)
    z = model.output(x, u, d, p)
    assert z.shape == (1,)
    np.testing.assert_allclose(z, np.array([1.5]))


def test_output_default_falls_back_to_h(model_default):
    x = np.array([2.0])
    u = np.array([1.0])
    d = np.zeros(0)
    p = np.zeros(0)
    # default output should equal h(x) = x
    np.testing.assert_array_equal(model_default.output(x, u, d, p), model_default.h(x, u, d, p))


def test_n_out_custom(model):
    assert model.n_out == 1


def test_n_out_default(model_default):
    assert model_default.n_out == model_default.ny


# ── Soft constraint penalties ─────────────────────────────────────────────────


def _make_ocp(model, **kwargs):
    return EconomicOptimalControlProblem(
        model,
        N=3,
        stage_cost=lambda x, u, d: float(x[0] ** 2 + 0.1 * u[0] ** 2),
        **kwargs,
    )


def test_no_soft_constraints_zero_penalty(model):
    ocp = _make_ocp(model)
    p = np.zeros(0)
    d = np.zeros(0)
    x = np.array([5.0])
    u = np.array([5.0])
    penalty = ocp._soft_penalty(x, u, d, p)
    assert penalty == 0.0


def test_state_lb_no_violation_no_penalty(model):
    ocp = _make_ocp(model, state_lb=np.array([-10.0]))
    p = np.zeros(0)
    d = np.zeros(0)
    x = np.array([1.0])
    u = np.array([0.0])
    assert ocp._soft_penalty(x, u, d, p) == 0.0


def test_state_lb_violation_positive_penalty(model):
    ocp = _make_ocp(model, state_lb=np.array([2.0]), state_weight=1.0)
    p = np.zeros(0)
    d = np.zeros(0)
    x = np.array([1.0])   # 1.0 < lb=2.0 → violation = 1.0
    u = np.array([0.0])
    penalty = ocp._soft_penalty(x, u, d, p)
    expected = 0.5 * 1.0 * 1.0**2
    np.testing.assert_allclose(penalty, expected)


def test_state_ub_violation_positive_penalty(model):
    ocp = _make_ocp(model, state_ub=np.array([0.5]), state_weight=2.0)
    p = np.zeros(0)
    d = np.zeros(0)
    x = np.array([1.0])   # 1.0 > ub=0.5 → violation = 0.5
    u = np.array([0.0])
    penalty = ocp._soft_penalty(x, u, d, p)
    expected = 0.5 * 2.0 * 0.5**2
    np.testing.assert_allclose(penalty, expected)


def test_output_lb_violation_positive_penalty(model):
    # output = x + u;  with x=0.1, u=0.1, output=0.2
    # lb = 0.5 → violation = 0.5 - 0.2 = 0.3
    ocp = _make_ocp(model, output_lb=np.array([0.5]), output_weight=4.0)
    p = np.zeros(0)
    d = np.zeros(0)
    x = np.array([0.1])
    u = np.array([0.1])
    penalty = ocp._soft_penalty(x, u, d, p)
    expected = 0.5 * 4.0 * 0.3**2
    np.testing.assert_allclose(penalty, expected, rtol=1e-10)


def test_output_ub_violation_positive_penalty(model):
    # output = x + u = 1.0 + 1.0 = 2.0;  ub = 1.5 → violation = 0.5
    ocp = _make_ocp(model, output_ub=np.array([1.5]), output_weight=1.0)
    p = np.zeros(0)
    d = np.zeros(0)
    x = np.array([1.0])
    u = np.array([1.0])
    penalty = ocp._soft_penalty(x, u, d, p)
    expected = 0.5 * 1.0 * 0.5**2
    np.testing.assert_allclose(penalty, expected, rtol=1e-10)


def test_soft_penalty_scales_quadratically(model):
    ocp = _make_ocp(model, state_lb=np.array([2.0]), state_weight=1.0)
    p = np.zeros(0)
    d = np.zeros(0)
    u = np.array([0.0])

    x1 = np.array([1.0])   # violation = 1
    x2 = np.array([0.0])   # violation = 2

    pen1 = ocp._soft_penalty(x1, u, d, p)
    pen2 = ocp._soft_penalty(x2, u, d, p)
    np.testing.assert_allclose(pen2 / pen1, 4.0, rtol=1e-10)


# ── EconomicOptimalControlProblem.solve ───────────────────────────────────────


def test_solve_returns_correct_shapes(model):
    N = 5
    ocp = EconomicOptimalControlProblem(
        model, N=N,
        stage_cost=lambda x, u, d: float(x[0] ** 2 + 0.1 * u[0] ** 2),
    )
    x0 = np.array([1.0])
    d_traj = np.zeros((N, 0))
    u_opt, cost = ocp.solve(x0, d_traj)
    assert u_opt.shape == (N, 1)
    assert np.isfinite(cost)


def test_solve_cost_increases_with_soft_constraint(model):
    """Adding an active soft output lower-bound must increase the cost."""
    N = 3
    x0 = np.array([0.1])
    d_traj = np.zeros((N, 0))

    ocp_free = EconomicOptimalControlProblem(
        model, N=N,
        stage_cost=lambda x, u, d: float(x[0] ** 2 + 0.1 * u[0] ** 2),
    )
    ocp_constrained = EconomicOptimalControlProblem(
        model, N=N,
        stage_cost=lambda x, u, d: float(x[0] ** 2 + 0.1 * u[0] ** 2),
        output_lb=np.array([1.0]),    # output = x + u; forces u ≳ 1
        output_weight=100.0,
    )
    _, cost_free = ocp_free.solve(x0, d_traj)
    _, cost_constrained = ocp_constrained.solve(x0, d_traj)
    assert cost_constrained > cost_free


# ── CDNMPCController ──────────────────────────────────────────────────────────


def test_cdnmpc_step_returns_correct_shape(model):
    N = 3
    ocp = EconomicOptimalControlProblem(
        model, N=N,
        stage_cost=lambda x, u, d: float(x[0] ** 2 + 0.1 * u[0] ** 2),
    )
    estimator = TrivialEstimator()
    ctrl = CDNMPCController(model, estimator, ocp)

    y = np.array([0.5])
    d_traj = np.zeros((N, 0))
    u = ctrl.step(y, d_traj)
    assert u.shape == (1,)
    assert np.isfinite(u[0])


def test_cdnmpc_multiple_steps(model):
    N = 3
    ocp = EconomicOptimalControlProblem(
        model, N=N,
        stage_cost=lambda x, u, d: float(x[0] ** 2 + 0.1 * u[0] ** 2),
    )
    ctrl = CDNMPCController(model, TrivialEstimator(), ocp)
    d_traj = np.zeros((N, 0))
    for _ in range(5):
        y = np.random.randn(1)
        u = ctrl.step(y, d_traj)
        assert np.isfinite(u[0])


# ── Backward-compat alias ─────────────────────────────────────────────────────


def test_economic_nmpc_controller_is_alias():
    assert EconomicNMPCController is CDNMPCController


def test_top_level_exports():
    assert hasattr(mbc, "CDNMPCController")
    assert hasattr(mbc, "EconomicNMPCController")
    assert mbc.CDNMPCController is mbc.EconomicNMPCController
