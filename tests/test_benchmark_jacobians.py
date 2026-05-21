"""
Benchmark tests for analytical Jacobian efficiency in nonlinear CD systems.

These tests verify that the analytical constraint and objective Jacobians
added to ``EconomicOptimalControlProblem`` and
``CDTrackingOptimalControlProblem`` produce a measurable reduction in the
number of NLP function evaluations (``nfev``) compared to the numerical
finite-difference baseline.

The key insight: scipy's SLSQP must estimate the constraint Jacobian by
finite differences when ``jac`` is not provided.  Each FD column requires
one extra function evaluation, so for ``n`` decision variables the FD
baseline performs O(n) extra ``nfev`` per iteration.  Analytical Jacobians
eliminate this overhead entirely.

Background
----------
The analytical Jacobians were introduced in:

* ``EconomicOptimalControlProblem._equality_constraint_jac``   — dynamics
* ``EconomicOptimalControlProblem._inequality_constraint_jac`` — ROM/soft constraints
* ``EconomicOptimalControlProblem._objective_jac``             — tracking + penalties
* ``CDTrackingOptimalControlProblem``                          — lagrange + mayer Jacs

All tests compare ``nfev`` between two modes:

* **analytical** — the default; ``NLPConstraint.jac`` and
  ``NLPProblem.objective_jac`` are populated.
* **numerical FD** — a thin backend wrapper strips all ``jac`` fields before
  the solve so scipy falls back to finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbc.control import (
    CDTrackingOptimalControlProblem,
    EconomicOptimalControlProblem,
)
from mbc.control.nlp_solver import NLPConstraint, NLPProblem, NLPResult, ScipyNLPBackend
from mbc.models import ContinuousDiscreteDAEModel, ContinuousDiscreteModel


# ── Helper: strip analytical Jacobians ───────────────────────────────────────


class _StrippedJacBackend:
    """Wraps a backend and removes all Jacobian callables from NLPProblem."""

    def __init__(self, inner: ScipyNLPBackend) -> None:
        self._inner = inner

    def solve(self, problem: NLPProblem) -> NLPResult:
        stripped = NLPProblem(
            objective=problem.objective,
            objective_jac=None,
            x0=problem.x0,
            lb=problem.lb,
            ub=problem.ub,
            constraints=tuple(
                NLPConstraint(kind=c.kind, fun=c.fun, jac=None)
                for c in problem.constraints
            ),
        )
        return self._inner.solve(stripped)


# ── Benchmark models ──────────────────────────────────────────────────────────


class _ScalarNonlinear(ContinuousDiscreteModel):
    """SDE: dx = (−0.2 x + tanh(u)) dt + 0.05 dw."""

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
        return np.array([-0.2 * x[0] + np.tanh(u[0])])

    def sigma(self, x, u, d, p, t):
        return np.array([[0.05]])

    def hm(self, x, u, d, p, t):
        return np.array([x[0]])

    def gm(self, x, u, d, p, t):
        return np.array([x[0]])


class _IsomerisationReactor(ContinuousDiscreteDAEModel):
    """SDAE: isomerisation reactor with equilibrium algebraic constraint."""

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


# ── Shared fixture helpers ────────────────────────────────────────────────────


def _make_sde_eocp(N: int, n_steps: int, strip_jac: bool) -> EconomicOptimalControlProblem:
    model = _ScalarNonlinear()
    ocp = EconomicOptimalControlProblem(
        model, N,
        Q_z=np.eye(1) * 2.0,
        z_ref=np.array([1.5]),
        Q_du=np.eye(1) * 0.5,
        p_u_eco=np.array([0.1]),
        du_min=np.array([-2.0]),
        du_max=np.array([2.0]),
        n_steps=n_steps,
        dt=1.0,
        u_min=np.array([-3.0]),
        u_max=np.array([3.0]),
        solver_options={"maxiter": 300},
    )
    if strip_jac:
        ocp._solver_backend = _StrippedJacBackend(ocp._solver_backend)
    return ocp


def _make_sdae_eocp(N: int, n_steps: int, strip_jac: bool) -> EconomicOptimalControlProblem:
    model = _IsomerisationReactor()
    ocp = EconomicOptimalControlProblem(
        model, N,
        Q_z=np.eye(1) * 2.0,
        z_ref=np.array([1.5]),
        Q_du=np.eye(1) * 0.5,
        n_steps=n_steps,
        dt=1.0,
        u_min=np.array([0.1]),
        u_max=np.array([2.0]),
        solver_options={"maxiter": 300},
    )
    if strip_jac:
        ocp._solver_backend = _StrippedJacBackend(ocp._solver_backend)
    return ocp


def _make_cdtracking(N: int, n_steps: int, strip_jac: bool) -> CDTrackingOptimalControlProblem:
    model = _ScalarNonlinear()
    ocp = CDTrackingOptimalControlProblem(
        model, N,
        Q=np.eye(1) * 2.0,
        R=np.eye(1) * 0.1,
        P=np.eye(1) * 5.0,
        z_ref=np.array([1.5]),
        n_steps=n_steps,
        dt=1.0,
        u_min=np.array([-3.0]),
        u_max=np.array([3.0]),
        solver_options={"maxiter": 300},
    )
    if strip_jac:
        ocp._eocp._solver_backend = _StrippedJacBackend(ocp._eocp._solver_backend)
    return ocp


# ── Tests: analytical Jacobians are wired in ─────────────────────────────────


class TestAnalyticalJacobiansAreWired:
    """Verify that analytical Jacobians are passed to the NLP solver."""

    def test_eocp_equality_constraint_has_jac(self):
        """NLPConstraint for dynamics must include a jac callable."""
        from mbc.control.nlp_solver import NLPProblem as _NLPProblem

        captured: list[_NLPProblem] = []

        class _CapturingBackend:
            def solve(self, problem):
                captured.append(problem)
                # Satisfy the return contract
                from mbc.control.nlp_solver import NLPResult as _NLPResult
                return _NLPResult(
                    x=problem.x0,
                    fun=0.0,
                    success=True,
                    status=0,
                    message="captured",
                    nit=0, nfev=1, njev=0,
                )

        model = _ScalarNonlinear()
        ocp = EconomicOptimalControlProblem(
            model, N=3, Q_z=np.eye(1), z_ref=np.array([1.0]),
            n_steps=2, dt=1.0,
        )
        ocp._solver_backend = _CapturingBackend()
        try:
            ocp.solve(np.array([0.0]), np.zeros((3, 1)))
        except Exception:
            pass

        assert captured, "No NLPProblem was captured by the backend"
        prob = captured[0]
        # Equality constraint (dynamics) must have a jac
        eq_cons = [c for c in prob.constraints if c.kind == "eq"]
        assert eq_cons, "No equality constraint found"
        assert eq_cons[0].jac is not None, (
            "Equality constraint Jacobian is None — analytical jac not wired in"
        )

    def test_eocp_objective_has_analytical_jac_for_pure_tracking(self):
        """EOCP with only Q_z tracking must provide an objective Jacobian."""
        model = _ScalarNonlinear()
        ocp = EconomicOptimalControlProblem(
            model, N=3, Q_z=np.eye(1) * 2.0, z_ref=np.array([1.5]),
            n_steps=2, dt=1.0,
        )
        assert ocp._can_use_analytical_objective_jac(), (
            "Pure tracking EOCP should provide analytical objective Jacobian"
        )

    def test_cdtracking_always_has_analytical_objective_jac(self):
        """CDTrackingOptimalControlProblem always provides analytical Jacs."""
        model = _ScalarNonlinear()
        ocp = CDTrackingOptimalControlProblem(
            model, N=3,
            Q=np.eye(1) * 2.0, R=np.eye(1) * 0.1, P=np.eye(1) * 5.0,
            z_ref=np.array([1.5]), dt=1.0,
        )
        assert ocp._eocp._can_use_analytical_objective_jac(), (
            "CDTrackingOCP should always provide analytical objective Jacobian"
        )

    def test_eocp_with_user_lagrange_no_jac_disables_analytical_obj_grad(self):
        """If lagrange_jac is not provided, analytical obj grad must be disabled."""
        model = _ScalarNonlinear()
        ocp = EconomicOptimalControlProblem(
            model, N=3,
            lagrange=lambda t, x, y, u, p: float(u @ u),
            n_steps=2, dt=1.0,
        )
        assert not ocp._can_use_analytical_objective_jac(), (
            "EOCP with lagrange but no lagrange_jac must disable analytical obj grad"
        )

    def test_eocp_with_user_lagrange_and_jac_enables_analytical_obj_grad(self):
        """If both lagrange and lagrange_jac are provided, analytical grad is enabled."""
        model = _ScalarNonlinear()
        ocp = EconomicOptimalControlProblem(
            model, N=3,
            lagrange=lambda t, x, y, u, p: float(u @ u),
            lagrange_jac=lambda t, x, y, u, p: (np.zeros_like(x), np.zeros_like(y), 2.0 * u),
            n_steps=2, dt=1.0,
        )
        assert ocp._can_use_analytical_objective_jac(), (
            "EOCP with lagrange + lagrange_jac must enable analytical obj grad"
        )


# ── Tests: efficiency gains (nfev reduction) ─────────────────────────────────


class TestNfevReduction:
    """
    Verify that analytical Jacobians reduce the number of function evaluations.

    Each test runs the same NLP problem twice — once with analytical Jacobians
    (the current default) and once with a numerical FD baseline (Jacobians
    stripped).  The analytical variant must use significantly fewer ``nfev``.

    Conservative threshold: ``nfev_ratio ≥ 3`` (empirically we see 10–50+x).
    """

    _NFEV_REDUCTION_THRESHOLD = 3.0  # conservative; practice shows >>10x

    def _compare(self, ocp_ana, ocp_num, x0, d_traj):
        _, cost_ana, info_ana = ocp_ana.solve(x0, d_traj)
        _, cost_num, info_num = ocp_num.solve(x0, d_traj)
        r_ana = info_ana["result"]
        r_num = info_num["result"]
        return r_ana, r_num, cost_ana, cost_num

    def test_sde_eocp_nfev_reduction(self):
        """SDE EOCP analytical Jacobians reduce nfev by at least 3x."""
        N, n_steps = 5, 4
        ocp_ana = _make_sde_eocp(N, n_steps, strip_jac=False)
        ocp_num = _make_sde_eocp(N, n_steps, strip_jac=True)
        x0 = np.array([0.0])
        d_traj = np.zeros((N, 1))
        r_ana, r_num, cost_ana, cost_num = self._compare(ocp_ana, ocp_num, x0, d_traj)

        assert r_ana.success, f"Analytical solve failed: {r_ana.message}"
        assert r_num.success, f"Numerical solve failed: {r_num.message}"
        assert r_ana.nfev > 0, "nfev must be positive"
        nfev_ratio = r_num.nfev / r_ana.nfev
        assert nfev_ratio >= self._NFEV_REDUCTION_THRESHOLD, (
            f"nfev reduction too small: {r_num.nfev} / {r_ana.nfev} = {nfev_ratio:.1f}x "
            f"(expected ≥ {self._NFEV_REDUCTION_THRESHOLD}x)"
        )
        assert abs(cost_ana - cost_num) < 1e-3, (
            f"Cost diverged: analytical={cost_ana:.6f}, numerical={cost_num:.6f}"
        )

    def test_sdae_eocp_nfev_reduction(self):
        """SDAE EOCP analytical Jacobians (including algebraic rows) reduce nfev by at least 3x."""
        N, n_steps = 4, 3
        ocp_ana = _make_sdae_eocp(N, n_steps, strip_jac=False)
        ocp_num = _make_sdae_eocp(N, n_steps, strip_jac=True)
        x0 = np.array([4.0])
        d_traj = np.zeros((N, 0))
        r_ana, r_num, cost_ana, cost_num = self._compare(ocp_ana, ocp_num, x0, d_traj)

        assert r_ana.success, f"Analytical solve failed: {r_ana.message}"
        assert r_num.success, f"Numerical solve failed: {r_num.message}"
        nfev_ratio = r_num.nfev / r_ana.nfev
        assert nfev_ratio >= self._NFEV_REDUCTION_THRESHOLD, (
            f"SDAE nfev reduction too small: {r_num.nfev} / {r_ana.nfev} = {nfev_ratio:.1f}x"
        )
        assert abs(cost_ana - cost_num) < 1e-3, (
            f"SDAE cost diverged: analytical={cost_ana:.6f}, numerical={cost_num:.6f}"
        )

    def test_cdtracking_nfev_reduction(self):
        """CDTracking analytical Jacobians (lagrange + mayer + constraints) reduce nfev by ≥ 3x."""
        N, n_steps = 5, 4
        ocp_ana = _make_cdtracking(N, n_steps, strip_jac=False)
        ocp_num = _make_cdtracking(N, n_steps, strip_jac=True)
        x0 = np.array([0.0])
        d_traj = np.zeros((N, 1))
        r_ana, r_num, cost_ana, cost_num = self._compare(ocp_ana, ocp_num, x0, d_traj)

        assert r_ana.success, f"Analytical solve failed: {r_ana.message}"
        assert r_num.success, f"Numerical solve failed: {r_num.message}"
        nfev_ratio = r_num.nfev / r_ana.nfev
        assert nfev_ratio >= self._NFEV_REDUCTION_THRESHOLD, (
            f"CDTracking nfev reduction too small: {r_num.nfev} / {r_ana.nfev} = {nfev_ratio:.1f}x"
        )
        assert abs(cost_ana - cost_num) < 1e-3, (
            f"CDTracking cost diverged: analytical={cost_ana:.6f}, numerical={cost_num:.6f}"
        )

    def test_nfev_reduction_scales_with_decision_variable_count(self):
        """
        Confirm that the nfev reduction ratio grows with the number of decision
        variables (finite-difference cost is O(n) per iteration).
        """
        x0 = np.array([0.0])
        ratios = {}
        for N in (3, 10):
            n_steps = 4
            ocp_ana = _make_sde_eocp(N, n_steps, strip_jac=False)
            ocp_num = _make_sde_eocp(N, n_steps, strip_jac=True)
            d_traj = np.zeros((N, 1))
            r_ana, r_num, _, _ = self._compare(ocp_ana, ocp_num, x0, d_traj)
            ratios[N] = r_num.nfev / max(r_ana.nfev, 1)

        assert ratios[10] > ratios[3], (
            f"Expected nfev ratio to grow with N: "
            f"ratio(N=3)={ratios[3]:.1f}, ratio(N=10)={ratios[10]:.1f}"
        )


# ── Tests: solution equivalence ───────────────────────────────────────────────


class TestSolutionEquivalence:
    """
    Verify that analytical and numerical Jacobian variants converge to the
    same optimal solution and cost (same problem, different Jacobian sources).
    """

    _COST_TOL = 1e-3
    _INPUT_TOL = 1e-3

    def test_sde_eocp_same_optimal_cost(self):
        N, n_steps = 5, 3
        ocp_ana = _make_sde_eocp(N, n_steps, strip_jac=False)
        ocp_num = _make_sde_eocp(N, n_steps, strip_jac=True)
        x0, d_traj = np.array([0.0]), np.zeros((N, 1))
        U_ana, cost_ana, _ = ocp_ana.solve(x0, d_traj)
        U_num, cost_num, _ = ocp_num.solve(x0, d_traj)
        assert abs(cost_ana - cost_num) < self._COST_TOL, (
            f"Cost differs: {cost_ana:.6f} vs {cost_num:.6f}"
        )
        np.testing.assert_allclose(U_ana, U_num, atol=self._INPUT_TOL,
                                   err_msg="Optimal inputs differ")

    def test_sdae_eocp_same_optimal_cost(self):
        N, n_steps = 4, 3
        ocp_ana = _make_sdae_eocp(N, n_steps, strip_jac=False)
        ocp_num = _make_sdae_eocp(N, n_steps, strip_jac=True)
        x0, d_traj = np.array([4.0]), np.zeros((N, 0))
        U_ana, cost_ana, _ = ocp_ana.solve(x0, d_traj)
        U_num, cost_num, _ = ocp_num.solve(x0, d_traj)
        assert abs(cost_ana - cost_num) < self._COST_TOL, (
            f"SDAE cost differs: {cost_ana:.6f} vs {cost_num:.6f}"
        )
        np.testing.assert_allclose(U_ana, U_num, atol=self._INPUT_TOL,
                                   err_msg="SDAE optimal inputs differ")

    def test_cdtracking_same_optimal_cost(self):
        N, n_steps = 5, 3
        ocp_ana = _make_cdtracking(N, n_steps, strip_jac=False)
        ocp_num = _make_cdtracking(N, n_steps, strip_jac=True)
        x0, d_traj = np.array([0.0]), np.zeros((N, 1))
        U_ana, cost_ana, _ = ocp_ana.solve(x0, d_traj)
        U_num, cost_num, _ = ocp_num.solve(x0, d_traj)
        assert abs(cost_ana - cost_num) < self._COST_TOL, (
            f"CDTracking cost differs: {cost_ana:.6f} vs {cost_num:.6f}"
        )
        np.testing.assert_allclose(U_ana, U_num, atol=self._INPUT_TOL,
                                   err_msg="CDTracking optimal inputs differ")
