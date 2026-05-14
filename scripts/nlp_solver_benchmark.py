"""
Benchmark nonlinear EOCP solver backends on representative SDE/SDAE models.

Usage:
    python /home/runner/work/mbc/mbc/scripts/nlp_solver_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mbc.control import EconomicOptimalControlProblem
from mbc.models import ContinuousDiscreteDAEModel, ContinuousDiscreteModel


@dataclass
class RunStats:
    success: bool
    elapsed_s: float
    nit: float
    cost: float


class ScalarNonlinear(ContinuousDiscreteModel):
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


class IsomerisationReactor(ContinuousDiscreteDAEModel):
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


def _safe_solver_name(solver: str) -> str:
    return solver.lower().strip()


def _solve_once(solver: str, model, N: int, n_steps: int, dt: float) -> RunStats:
    d_traj = np.zeros((N, model.nd))
    x0 = np.array([0.0]) if not isinstance(model, IsomerisationReactor) else np.array([4.0])
    options = {"maxiter": 150} if _safe_solver_name(solver) != "ipopt" else {"max_iter": 150}
    ocp = EconomicOptimalControlProblem(
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
    nit = float(result.nit) if getattr(result, "nit", None) is not None else np.nan
    return RunStats(success=bool(result.success), elapsed_s=float(elapsed), nit=nit, cost=float(cost))


def _aggregate(stats: list[RunStats]) -> dict:
    elapsed = np.array([s.elapsed_s for s in stats], dtype=float)
    nit = np.array([s.nit for s in stats], dtype=float)
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

    # Acceptance criteria definition:
    # candidate should be at least 20% faster median solve-time, 10% fewer median
    # iterations, and no worse than baseline success-rate by more than 2 pp.
    criteria = {
        "baseline_solver": "SLSQP",
        "min_speedup_ratio": 1.20,   # baseline_time / candidate_time
        "max_iteration_ratio": 0.90, # candidate_nit / baseline_nit
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
            ratios_nit = []
            success_drops = []
            for N in horizons:
                if (
                    N in base and N in cand
                    and "median_time_s" in base[N]
                    and "median_time_s" in cand[N]
                    and "median_nit" in base[N]
                    and "median_nit" in cand[N]
                ):
                    ratios_speed.append(base[N]["median_time_s"] / max(cand[N]["median_time_s"], 1e-12))
                    ratios_nit.append(cand[N]["median_nit"] / max(base[N]["median_nit"], 1e-12))
                    success_drops.append(base[N]["success_rate"] - cand[N]["success_rate"])
            if ratios_speed:
                verdict = {
                    "eligible": bool(
                        np.median(ratios_speed) >= criteria["min_speedup_ratio"]
                        and np.median(ratios_nit) <= criteria["max_iteration_ratio"]
                        and np.max(success_drops) <= criteria["max_success_drop_pp"]
                    ),
                    "median_speedup_ratio": float(np.median(ratios_speed)),
                    "median_iteration_ratio": float(np.median(ratios_nit)),
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


if __name__ == "__main__":
    np.random.seed(0)
    report = run_benchmark(
        solvers=["SLSQP", "ipopt"],
        horizons=[5, 10, 20],
        repeats=3,
        n_steps=2,
    )
    print(json.dumps(report, indent=2))
