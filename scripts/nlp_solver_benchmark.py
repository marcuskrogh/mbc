"""
Benchmark nonlinear EOCP solver backends on representative SDE/SDAE models.

Usage:
    python scripts/nlp_solver_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mbc.ocp import ContinuousNonlinearOCP
from mbc.ocp import NLPProblem, ScipyNLPBackend
from mbc.models import ContinuousDiscreteSDAE, ContinuousDiscreteSDE

SAFE_DIVISOR_FLOOR = 1e-12  # Fallback floor to avoid division-by-zero in ratio calculations.


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, SAFE_DIVISOR_FLOOR))


@dataclass
class RunStats:
    success: bool
    elapsed_s: float
    num_iterations: int | None
    cost: float


class ScalarNonlinear(ContinuousDiscreteSDE):
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
        return np.array([-0.2 * x[0] + np.tanh(u[0]) + 0.1 * (d[0] if d.size else 0.0)])

    def sigma(self, x, u, d, p, t):
        return np.array([[0.05]])

    def hm(self, x, u, d, p, t):
        return np.array([x[0]])

    def gm(self, x, u, d, p, t):
        return np.array([x[0]])


class IsomerisationReactor(ContinuousDiscreteSDAE):
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


def _normalize_solver_name(solver: str) -> str:
    return solver.lower().strip()


def _solve_once(solver: str, model, N: int, n_steps: int, dt: float) -> RunStats:
    d_traj = np.zeros((N, model.nd))
    x0 = np.array([0.0]) if not isinstance(model, IsomerisationReactor) else np.array([4.0])
    options = {"maxiter": 150} if _normalize_solver_name(solver) != "ipopt" else {"max_iter": 150}
    ocp = ContinuousNonlinearOCP(
        model,
        N=N,
        Q_z=np.array([[1.0]]),
        z_ref=np.array([1.5]),
        u_min=np.array([-3.0]) if model.nu == 1 else None,
        u_max=np.array([3.0]) if model.nu == 1 else None,
        n_steps=n_steps,
        solver=solver,
        solver_options=options,
        dt=dt,
    )
    tic = time.perf_counter()
    _, cost, info = ocp.solve(x0, d_traj)
    elapsed = time.perf_counter() - tic
    result = info["result"]
    num_iterations = int(result.nit) if getattr(result, "nit", None) is not None else None
    return RunStats(
        success=bool(result.success),
        elapsed_s=float(elapsed),
        num_iterations=num_iterations,
        cost=float(cost),
    )


def _aggregate(stats: list[RunStats]) -> dict:
    elapsed = np.array([s.elapsed_s for s in stats], dtype=float)
    nit = np.array(
        [float(s.num_iterations) if s.num_iterations is not None else np.nan for s in stats],
        dtype=float,
    )
    success = np.array([1.0 if s.success else 0.0 for s in stats], dtype=float)
    return {
        "runs": len(stats),
        "success_rate": float(success.mean()),
        "median_time_s": float(np.median(elapsed)),
        "median_nit": float(np.nanmedian(nit)),
        "p90_time_s": float(np.quantile(elapsed, 0.9)),
    }


def _scaling_slope(horizons: list[int], times: list[float]) -> float:
    x = np.log(np.asarray(horizons, dtype=float))
    y = np.log(np.asarray(times, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def run_benchmark(
    solvers: list[str],
    horizons: list[int],
    repeats: int = 3,
    n_steps: int = 2,
    dt: float = 1.0,
) -> dict:
    model_cases = {
        "sde": ScalarNonlinear(),
        "sdae": IsomerisationReactor(),
    }
    out: dict[str, dict] = {}

    for case_name, model in model_cases.items():
        out[case_name] = {}
        for solver in solvers:
            per_horizon: dict[int, dict] = {}
            for N in horizons:
                stats: list[RunStats] = []
                for _ in range(repeats):
                    try:
                        stats.append(_solve_once(solver, model, N=N, n_steps=n_steps, dt=dt))
                    except Exception as exc:
                        per_horizon[N] = {"error": str(exc)}
                        stats = []
                        break
                if stats:
                    per_horizon[N] = _aggregate(stats)
            out[case_name][solver] = {"by_horizon": per_horizon}

    # Provisional acceptance criteria for automated comparisons. These thresholds
    # are intentionally conservative defaults and can be tightened/relaxed after
    # collecting solver-specific benchmark history in CI or local studies.
    # Candidate should be at least 20% faster median solve-time, 10% fewer median
    # iterations, and no worse than baseline success-rate by more than 2 points.
    criteria = {
        "baseline_solver": "SLSQP",
        "min_speedup_ratio": 1.20,  # baseline_time / candidate_time
        "min_iteration_improvement_ratio": 1.10,  # baseline_nit / candidate_nit
        "max_success_drop_pp": 0.02,
    }
    out["acceptance_criteria"] = criteria

    for case_name in model_cases:
        base = out[case_name].get("SLSQP", {}).get("by_horizon", {})
        base_times = []
        base_h = []
        for N in horizons:
            if N in base and "median_time_s" in base[N]:
                base_times.append(base[N]["median_time_s"])
                base_h.append(N)
        if len(base_h) >= 2:
            out[case_name]["SLSQP"]["scaling_slope"] = _scaling_slope(base_h, base_times)

        for solver in solvers:
            if solver == "SLSQP":
                continue
            cand = out[case_name].get(solver, {}).get("by_horizon", {})
            verdict = {"eligible": False}
            ratios_speed = []
            iteration_improvement_ratios = []
            success_drops = []
            for N in horizons:
                if (
                    N in base and N in cand
                    and "median_time_s" in base[N]
                    and "median_time_s" in cand[N]
                    and "median_nit" in base[N]
                    and "median_nit" in cand[N]
                ):
                    ratios_speed.append(_safe_ratio(base[N]["median_time_s"], cand[N]["median_time_s"]))
                    iteration_improvement_ratios.append(
                        _safe_ratio(base[N]["median_nit"], cand[N]["median_nit"])
                    )
                    success_drops.append(base[N]["success_rate"] - cand[N]["success_rate"])
            if ratios_speed:
                verdict = {
                    "eligible": bool(
                        np.median(ratios_speed) >= criteria["min_speedup_ratio"]
                        and np.median(iteration_improvement_ratios) >= criteria["min_iteration_improvement_ratio"]
                        and np.max(success_drops) <= criteria["max_success_drop_pp"]
                    ),
                    "median_speedup_ratio": float(np.median(ratios_speed)),
                    "median_iteration_improvement_ratio": float(np.median(iteration_improvement_ratios)),
                    "max_success_drop_pp": float(np.max(success_drops)),
                }
            out[case_name][solver]["acceptance_check_vs_SLSQP"] = verdict

            cand_times = []
            cand_h = []
            for N in horizons:
                if N in cand and "median_time_s" in cand[N]:
                    cand_times.append(cand[N]["median_time_s"])
                    cand_h.append(N)
            if len(cand_h) >= 2:
                out[case_name][solver]["scaling_slope"] = _scaling_slope(cand_h, cand_times)

    return out


def run_gradient_efficiency_example(
    *,
    n_vars: int = 60,
    repeats: int = 3,
) -> dict:
    """
    Compare SciPy/SLSQP solve efficiency with analytical vs numerical gradients.

    Uses a diagonal quadratic objective:
        f(x) = 0.5 * x^T Q x + c^T x
        grad f(x) = Q x + c
    """
    rng = np.random.default_rng(1)
    q_diag = 1.0 + rng.random(n_vars)
    c = rng.normal(size=n_vars)
    x0 = rng.normal(size=n_vars)

    def objective(x: np.ndarray) -> float:
        return float(0.5 * np.sum(q_diag * x * x) + np.dot(c, x))

    def objective_jac(x: np.ndarray) -> np.ndarray:
        return q_diag * x + c

    base_problem = dict(
        objective=objective,
        x0=x0,
        lb=np.full(n_vars, -10.0),
        ub=np.full(n_vars, 10.0),
        constraints=tuple(),
    )

    def _solve(with_jac: bool) -> dict:
        backend = ScipyNLPBackend(method="SLSQP", options={"maxiter": 300, "ftol": 1e-9})
        elapsed = []
        nfev = []
        njev = []
        nit = []
        for _ in range(repeats):
            problem = NLPProblem(
                **base_problem,
                objective_jac=objective_jac if with_jac else None,
            )
            tic = time.perf_counter()
            result = backend.solve(problem)
            elapsed.append(time.perf_counter() - tic)
            nfev.append(float(result.nfev or np.nan))
            njev.append(float(result.njev or np.nan))
            nit.append(float(result.nit or np.nan))
        return {
            "median_time_s": float(np.median(elapsed)),
            "median_nfev": float(np.nanmedian(np.asarray(nfev, dtype=float))),
            "median_njev": float(np.nanmedian(np.asarray(njev, dtype=float))),
            "median_nit": float(np.nanmedian(np.asarray(nit, dtype=float))),
        }

    analytical = _solve(with_jac=True)
    numerical = _solve(with_jac=False)
    return {
        "problem": {"n_vars": n_vars, "repeats": repeats, "solver": "SLSQP"},
        "analytical_gradient": analytical,
        "numerical_gradient": numerical,
        "ratios": {
            "time_speedup": _safe_ratio(
                numerical["median_time_s"], analytical["median_time_s"]
            ),
            "nfev_reduction_ratio": _safe_ratio(
                numerical["median_nfev"], analytical["median_nfev"]
            ),
        },
    }


def run_hessian_efficiency_example(
    *,
    n_vars: int = 16,
    repeats: int = 2,
) -> dict:
    """
    Compare trust-constr solve efficiency with analytical vs numerical Hessians.

    Uses a diagonal quadratic objective:
        f(x) = 0.5 * x^T Q x + c^T x
        grad f(x) = Q x + c
        hess f(x) = diag(Q)
    """
    rng = np.random.default_rng(2)
    q_diag = 1.0 + rng.random(n_vars)
    c = rng.normal(size=n_vars)
    x0 = rng.normal(size=n_vars)

    def objective(x: np.ndarray) -> float:
        return float(0.5 * np.sum(q_diag * x * x) + np.dot(c, x))

    def objective_jac(x: np.ndarray) -> np.ndarray:
        return q_diag * x + c

    def objective_hess(_: np.ndarray) -> np.ndarray:
        return np.diag(q_diag)

    def finite_difference_hessian(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        h = np.zeros((n_vars, n_vars), dtype=float)
        f0 = objective(x)
        for i in range(n_vars):
            dx = np.zeros_like(x)
            dx[i] = eps
            h[i, i] = (objective(x + dx) - 2.0 * f0 + objective(x - dx)) / (eps * eps)
        for i in range(n_vars):
            dxi = np.zeros_like(x)
            dxi[i] = eps
            for j in range(i + 1, n_vars):
                dxj = np.zeros_like(x)
                dxj[j] = eps
                val = (
                    objective(x + dxi + dxj)
                    - objective(x + dxi - dxj)
                    - objective(x - dxi + dxj)
                    + objective(x - dxi - dxj)
                ) / (4.0 * eps * eps)
                h[i, j] = val
                h[j, i] = val
        return h

    base_problem = dict(
        objective=objective,
        objective_jac=objective_jac,
        x0=x0,
        lb=np.full(n_vars, -10.0),
        ub=np.full(n_vars, 10.0),
        constraints=tuple(),
    )

    def _solve(with_analytical_hess: bool) -> dict:
        backend = ScipyNLPBackend(
            method="trust-constr",
            options={"maxiter": 200, "gtol": 1e-9, "xtol": 1e-9},
        )
        elapsed = []
        nfev = []
        njev = []
        nhev = []
        nit = []
        for _ in range(repeats):
            problem = NLPProblem(
                **base_problem,
                objective_hess=objective_hess if with_analytical_hess else finite_difference_hessian,
            )
            tic = time.perf_counter()
            result = backend.solve(problem)
            elapsed.append(time.perf_counter() - tic)
            nfev.append(float(result.nfev or np.nan))
            njev.append(float(result.njev or np.nan))
            nhev.append(float(result.nhev or np.nan))
            nit.append(float(result.nit or np.nan))
        return {
            "median_time_s": float(np.median(elapsed)),
            "median_nfev": float(np.nanmedian(np.asarray(nfev, dtype=float))),
            "median_njev": float(np.nanmedian(np.asarray(njev, dtype=float))),
            "median_nhev": float(np.nanmedian(np.asarray(nhev, dtype=float))),
            "median_nit": float(np.nanmedian(np.asarray(nit, dtype=float))),
        }

    analytical = _solve(with_analytical_hess=True)
    numerical = _solve(with_analytical_hess=False)
    return {
        "problem": {"n_vars": n_vars, "repeats": repeats, "solver": "trust-constr"},
        "analytical_hessian": analytical,
        "numerical_hessian": numerical,
        "numerical_hessian_method": "finite_difference_callback",
        "ratios": {
            "time_speedup": _safe_ratio(
                numerical["median_time_s"], analytical["median_time_s"]
            ),
            "nhev_reduction_ratio": _safe_ratio(
                numerical["median_nhev"], analytical["median_nhev"]
            ),
        },
    }


if __name__ == "__main__":
    report = {
        "ocp_solver_benchmark": run_benchmark(
            solvers=["SLSQP", "ipopt"],
            horizons=[5, 10, 20],
            repeats=3,
            n_steps=2,
        ),
        "gradient_efficiency_example": run_gradient_efficiency_example(
            n_vars=60,
            repeats=3,
        ),
        "hessian_efficiency_example": run_hessian_efficiency_example(
            n_vars=16,
            repeats=2,
        ),
    }
    print(json.dumps(report, indent=2))
