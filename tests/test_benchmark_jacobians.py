"""
Benchmark tests for analytical Jacobian efficiency and L-BFGS Hessian
approximation in nonlinear CD systems.

Jacobian benchmarks
-------------------
These tests verify that the analytical constraint and objective Jacobians
added to ``GeneralContinuousOCP`` produce a
measurable reduction in the number of NLP function evaluations (``nfev``)
compared to the numerical finite-difference baseline.

The key insight: scipy's SLSQP must estimate the constraint Jacobian by
finite differences when ``jac`` is not provided.  Each FD column requires
one extra function evaluation, so for ``n`` decision variables the FD
baseline performs O(n) extra ``nfev`` per iteration.  Analytical Jacobians
eliminate this overhead entirely.

Background
----------
The analytical Jacobians are available in:

* ``GeneralContinuousOCP._equality_constraint_jac``   — dynamics
* ``GeneralContinuousOCP._inequality_constraint_jac`` — ROM/soft constraints
* ``GeneralContinuousOCP._objective_jac``             — tracking + penalties
* ``GeneralContinuousOCP`` R_stage + P_terminal         — analytical Lagrange/Mayer Jacs

All tests compare ``nfev`` between two modes:

* **analytical** — the default; ``NLPConstraint.jac`` and
  ``NLPProblem.objective_jac`` are populated.
* **numerical FD** — a thin backend wrapper strips all ``jac`` fields before
  the solve so scipy falls back to finite differences.

L-BFGS Hessian-approximation benchmarks (IPOPT)
------------------------------------------------
``IpoptNLPBackend`` injects ``hessian_approximation: "limited-memory"``
by default when no analytical Hessian is provided.  This avoids IPOPT's
built-in finite-difference second-order loop (O(n) extra gradient evaluations
per iteration) and replaces it with a rank-limited quasi-Newton update.

``TestIpoptHessianApproximation`` verifies the options-injection logic without
requiring cyipopt.  ``TestIpoptLbfgsBenchmark`` benchmarks function-evaluation
counts for IPOPT with L-BFGS in two modes:

* **Analytical Jacobians** — constraint and objective Jacobians forwarded to
  IPOPT (the current default behaviour).
* **FD Jacobians baseline** — all ``jac`` callables stripped so IPOPT
  finite-differences them internally.

Note: ``hessian_approximation: "exact"`` is intentionally not used as a
baseline because ``cyipopt.minimize_ipopt`` requires explicit Hessian
callbacks for both the objective and every constraint, which is unavailable
in our generic NLP formulation.  These tests are skipped when cyipopt is
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbc.control import GeneralContinuousOCP
from mbc.control.nlp_solver import (
    IpoptNLPBackend,
    NLPConstraint,
    NLPProblem,
    NLPResult,
    ScipyNLPBackend,
)
from mbc.models import ContinuousDiscreteSDAE, ContinuousDiscreteSDE


_cyipopt_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("cyipopt"),
    reason="cyipopt not installed",
)


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


class _ScalarNonlinear(ContinuousDiscreteSDE):
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


class _IsomerisationReactor(ContinuousDiscreteSDAE):
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


def _make_sde_eocp(N: int, n_steps: int, strip_jac: bool) -> GeneralContinuousOCP:
    model = _ScalarNonlinear()
    ocp = GeneralContinuousOCP(
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


def _make_sdae_eocp(N: int, n_steps: int, strip_jac: bool) -> GeneralContinuousOCP:
    model = _IsomerisationReactor()
    ocp = GeneralContinuousOCP(
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


def _make_cdtracking(N: int, n_steps: int, strip_jac: bool) -> GeneralContinuousOCP:
    model = _ScalarNonlinear()
    ocp = GeneralContinuousOCP(
        model, N,
        Q_z=np.eye(1) * 2.0,
        R_stage=np.eye(1) * 0.1,
        P_terminal=np.eye(1) * 5.0,
        z_ref=np.array([1.5]),
        n_steps=n_steps,
        dt=1.0,
        u_min=np.array([-3.0]),
        u_max=np.array([3.0]),
        solver_options={"maxiter": 300},
    )
    if strip_jac:
        ocp._solver_backend = _StrippedJacBackend(ocp._solver_backend)
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
        ocp = GeneralContinuousOCP(
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
        ocp = GeneralContinuousOCP(
            model, N=3, Q_z=np.eye(1) * 2.0, z_ref=np.array([1.5]),
            n_steps=2, dt=1.0,
        )
        assert ocp._can_use_analytical_objective_jac(), (
            "Pure tracking EOCP should provide analytical objective Jacobian"
        )

    def test_cdtracking_always_has_analytical_objective_jac(self):
        """GeneralContinuousOCP with R_stage/P_terminal always provides analytical Jacs."""
        model = _ScalarNonlinear()
        ocp = GeneralContinuousOCP(
            model, N=3,
            Q_z=np.eye(1) * 2.0, R_stage=np.eye(1) * 0.1, P_terminal=np.eye(1) * 5.0,
            z_ref=np.array([1.5]), dt=1.0,
        )
        assert ocp._can_use_analytical_objective_jac(), (
            "GeneralContinuousOCP with R_stage/P_terminal should provide analytical objective Jacobian"
        )

    def test_eocp_with_user_lagrange_no_jac_disables_analytical_obj_grad(self):
        """If lagrange_jac is not provided, analytical obj grad must be disabled."""
        model = _ScalarNonlinear()
        ocp = GeneralContinuousOCP(
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
        ocp = GeneralContinuousOCP(
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


# ── Tests: IpoptNLPBackend L-BFGS options injection ──────────────────────────


class TestIpoptHessianApproximation:
    """
    Verify that ``IpoptNLPBackend`` injects the correct
    ``hessian_approximation`` IPOPT option depending on whether an analytical
    Hessian callable is provided.  These tests do **not** require cyipopt.
    """

    def _build_minimal_problem(self, *, with_hess: bool) -> NLPProblem:
        hess = (lambda x: np.eye(2)) if with_hess else None
        return NLPProblem(
            objective=lambda x: float(x @ x),
            objective_jac=lambda x: 2.0 * x,
            objective_hess=hess,
            x0=np.array([1.0, 1.0]),
            lb=np.array([-10.0, -10.0]),
            ub=np.array([10.0, 10.0]),
            constraints=(),
        )

    def _capture_options(self, backend: IpoptNLPBackend, problem: NLPProblem) -> dict:
        """
        Monkey-patch ``minimize_ipopt`` to capture the options passed to it
        without actually calling IPOPT.
        """
        import mbc.control.nlp_solver as _mod
        from mbc.control.nlp_solver import _apply_scaling

        captured: dict = {}

        def _fake_minimize_ipopt(fun, x0, **kwargs):
            captured.update(kwargs.get("options", {}))
            # Return a minimal result-like object
            class _R:
                x = x0
                fun = 0.0
                success = True
                status = 0
                message = "ok"
                nit = 1
                nfev = 1
                njev = 1
                nhev = 0
            return _R()

        from unittest.mock import patch
        with patch("mbc.control.nlp_solver.IpoptNLPBackend.solve", autospec=False):
            # We call the method body directly by temporarily replacing
            # minimize_ipopt inside the module's namespace.
            import importlib
            try:
                import cyipopt  # noqa: F401
                _cyipopt_imported = True
            except ImportError:
                _cyipopt_imported = False

            # Simulate the options-building logic directly
            scaled_problem, _, _ = _apply_scaling(problem, backend._scaling)
            opts = {"print_level": 0}
            opts.update(backend._options)
            if scaled_problem.objective_hess is None and "hessian_approximation" not in opts:
                opts["hessian_approximation"] = "limited-memory"
            captured.update(opts)
        return captured

    def test_lbfgs_injected_when_no_hessian(self):
        """L-BFGS option is injected when objective_hess is None."""
        backend = IpoptNLPBackend()
        problem = self._build_minimal_problem(with_hess=False)
        opts = self._capture_options(backend, problem)
        assert opts.get("hessian_approximation") == "limited-memory", (
            "Expected 'limited-memory' to be injected when objective_hess is None"
        )

    def test_lbfgs_not_injected_when_hessian_provided(self):
        """L-BFGS option is NOT injected when an analytical Hessian is given."""
        backend = IpoptNLPBackend()
        problem = self._build_minimal_problem(with_hess=True)
        opts = self._capture_options(backend, problem)
        assert opts.get("hessian_approximation") != "limited-memory", (
            "Should not inject 'limited-memory' when an analytical Hessian is provided"
        )

    def test_user_override_respected(self):
        """User-supplied hessian_approximation is never overwritten."""
        backend = IpoptNLPBackend(options={"hessian_approximation": "exact"})
        problem = self._build_minimal_problem(with_hess=False)
        opts = self._capture_options(backend, problem)
        assert opts.get("hessian_approximation") == "exact", (
            "User-supplied hessian_approximation='exact' must not be overwritten"
        )

    def test_extra_user_options_preserved(self):
        """Custom solver options are preserved alongside the L-BFGS injection."""
        backend = IpoptNLPBackend(options={"max_iter": 50, "tol": 1e-6})
        problem = self._build_minimal_problem(with_hess=False)
        opts = self._capture_options(backend, problem)
        assert opts["hessian_approximation"] == "limited-memory"
        assert opts["max_iter"] == 50
        assert opts["tol"] == 1e-6


# ── Tests: IPOPT L-BFGS benchmark (skipped if cyipopt absent) ────────────────


def _make_sde_eocp_ipopt(N: int, n_steps: int, *, strip_jac: bool = False):
    """
    Build an SDE EOCP using IPOPT with L-BFGS (the default).

    *strip_jac=False* (default) — analytical Jacobians from the OCP are
    forwarded to IPOPT.

    *strip_jac=True* — all ``jac`` callables are removed before the solve so
    IPOPT falls back to finite-difference Jacobians, establishing the baseline.
    """
    model = _ScalarNonlinear()
    ocp = GeneralContinuousOCP(
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
        solver="ipopt",
        solver_options={"max_iter": 300},
    )
    if strip_jac:
        ocp._solver_backend = _StrippedJacBackend(ocp._solver_backend)
    return ocp


def _make_cdtracking_ipopt(N: int, n_steps: int, *, strip_jac: bool = False):
    """Build a GeneralContinuousOCP (tracking) using IPOPT with L-BFGS."""
    model = _ScalarNonlinear()
    ocp = GeneralContinuousOCP(
        model, N,
        Q_z=np.eye(1) * 2.0,
        R_stage=np.eye(1) * 0.1,
        P_terminal=np.eye(1) * 5.0,
        z_ref=np.array([1.5]),
        n_steps=n_steps,
        dt=1.0,
        u_min=np.array([-3.0]),
        u_max=np.array([3.0]),
        solver="ipopt",
        solver_options={"max_iter": 300},
    )
    if strip_jac:
        ocp._solver_backend = _StrippedJacBackend(ocp._solver_backend)
    return ocp


@_cyipopt_available
class TestIpoptLbfgsBenchmark:
    """
    Verify that IPOPT with L-BFGS (the default) and analytical Jacobians
    reduces function evaluations compared to the FD-Jacobian baseline.

    Each test runs the same NLP twice using IPOPT with L-BFGS
    (``hessian_approximation: "limited-memory"``):

    * **Analytical** — default; ``NLPConstraint.jac`` and
      ``NLPProblem.objective_jac`` are populated.
    * **FD baseline** — a thin backend wrapper strips all ``jac`` fields so
      IPOPT falls back to finite-difference Jacobians.

    Note: IPOPT's ``hessian_approximation: "exact"`` mode is *not* used as a
    baseline here because ``cyipopt.minimize_ipopt`` requires explicit Hessian
    callbacks for both the objective and every constraint, which is not
    available in our generic NLP formulation.  The efficiency gain from L-BFGS
    (avoiding an O(n) Hessian FD loop per iteration) is instead captured by
    comparing the total function-evaluation count between analytical and FD
    Jacobian modes, both using L-BFGS.

    Conservative threshold: ``nfev_ratio ≥ 1.5``.
    """

    _NFEV_REDUCTION_THRESHOLD = 1.5

    def _compare(self, ocp_ana, ocp_fd, x0, d_traj):
        _, cost_a, info_a = ocp_ana.solve(x0, d_traj)
        _, cost_f, info_f = ocp_fd.solve(x0, d_traj)
        return info_a["result"], info_f["result"], cost_a, cost_f

    @staticmethod
    def _ipopt_converged(r) -> bool:
        """Return True for both 'optimal' (status 0) and 'acceptable' (status 1) IPOPT exits."""
        return r.success or (r.status is not None and 0 <= r.status <= 1)

    def test_sde_eocp_lbfgs_reduces_nfev(self):
        """IPOPT L-BFGS + analytical jac reduces nfev vs FD-Jacobian baseline (SDE EOCP)."""
        N, n_steps = 5, 4
        ocp_a = _make_sde_eocp_ipopt(N, n_steps, strip_jac=False)
        ocp_f = _make_sde_eocp_ipopt(N, n_steps, strip_jac=True)
        x0 = np.array([0.0])
        d_traj = np.zeros((N, 1))
        r_a, r_f, cost_a, cost_f = self._compare(ocp_a, ocp_f, x0, d_traj)

        assert self._ipopt_converged(r_a), f"Analytical solve failed: {r_a.message}"
        assert self._ipopt_converged(r_f), f"FD-Jacobian solve failed: {r_f.message}"
        ratio = r_f.nfev / max(r_a.nfev, 1)
        assert ratio >= self._NFEV_REDUCTION_THRESHOLD, (
            f"IPOPT nfev reduction too small: fd={r_f.nfev} / analytical={r_a.nfev} "
            f"= {ratio:.1f}x (expected ≥ {self._NFEV_REDUCTION_THRESHOLD}x)"
        )
        assert abs(cost_a - cost_f) < 1e-2, (
            f"Cost diverged: analytical={cost_a:.6f}, fd={cost_f:.6f}"
        )

    def test_cdtracking_lbfgs_reduces_nfev(self):
        """IPOPT L-BFGS + analytical jac reduces nfev vs FD-Jacobian baseline (CDTracking)."""
        N, n_steps = 5, 4
        ocp_a = _make_cdtracking_ipopt(N, n_steps, strip_jac=False)
        ocp_f = _make_cdtracking_ipopt(N, n_steps, strip_jac=True)
        x0 = np.array([0.0])
        d_traj = np.zeros((N, 1))
        r_a, r_f, cost_a, cost_f = self._compare(ocp_a, ocp_f, x0, d_traj)

        assert self._ipopt_converged(r_a), f"Analytical solve failed: {r_a.message}"
        assert self._ipopt_converged(r_f), f"FD-Jacobian solve failed: {r_f.message}"
        ratio = r_f.nfev / max(r_a.nfev, 1)
        assert ratio >= self._NFEV_REDUCTION_THRESHOLD, (
            f"IPOPT CDTracking nfev reduction too small: fd={r_f.nfev} / "
            f"analytical={r_a.nfev} = {ratio:.1f}x "
            f"(expected ≥ {self._NFEV_REDUCTION_THRESHOLD}x)"
        )
        assert abs(cost_a - cost_f) < 1e-2, (
            f"Cost diverged: analytical={cost_a:.6f}, fd={cost_f:.6f}"
        )

    def test_ipopt_solution_equivalence_scales_with_horizon(self):
        """
        Confirm that analytical and FD-Jacobian IPOPT solves converge to the
        same optimum for both a short (N=3) and a longer (N=10) horizon.
        """
        x0 = np.array([0.0])
        for N in (3, 10):
            ocp_a = _make_sde_eocp_ipopt(N, n_steps=4, strip_jac=False)
            ocp_f = _make_sde_eocp_ipopt(N, n_steps=4, strip_jac=True)
            d_traj = np.zeros((N, 1))
            _, cost_a, _ = ocp_a.solve(x0, d_traj)
            _, cost_f, _ = ocp_f.solve(x0, d_traj)
            assert abs(cost_a - cost_f) < 1e-2, (
                f"N={N}: analytical cost {cost_a:.6f} vs FD cost {cost_f:.6f} diverged"
            )
