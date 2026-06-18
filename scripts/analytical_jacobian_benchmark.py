"""
Benchmark: Analytical vs Numerical Jacobians for Continuous-Discrete NLP.

Measures the numerical efficiency gain from the analytical constraint and
objective Jacobians provided by ``GeneralContinuousOCP``.

Metrics
-------
* **nfev**  — number of objective/constraint function evaluations (lower is
  better; finite-difference Jacobians inflate this O(n) per iteration).
* **nit**   — number of solver iterations.
* **time_s** — wall-clock solve time.

Usage
-----
    python scripts/analytical_jacobian_benchmark.py
    python scripts/analytical_jacobian_benchmark.py --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mbc.control import GeneralContinuousOCP
from mbc.control.nlp_solver import NLPConstraint, NLPProblem, NLPResult, ScipyNLPBackend
from mbc.models import ContinuousDiscreteSDAE, ContinuousDiscreteSDE


# ── Helper: strip all Jacobians from an NLPProblem ───────────────────────────


class _StrippedJacBackend:
    """
    Thin wrapper around a backend that removes all analytical Jacobians from
    the NLPProblem before forwarding to the inner solver.

    This replicates the behaviour *before* the analytical-Jacobian
    implementation and serves as the numerical-FD baseline.
    """

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


# ── Benchmark models ─────────────────────────────────────────────────────────


class ScalarNonlinear(ContinuousDiscreteSDE):
    """
    Scalar SDE model:  dx = (−0.2 x + tanh(u)) dt + 0.05 dw.

    Nonlinear dynamics make the analytical Jacobians non-trivial.
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
        return np.array([-0.2 * x[0] + np.tanh(u[0])])

    def sigma(self, x, u, d, p, t):
        return np.array([[0.05]])

    def hm(self, x, u, d, p, t):
        return np.array([x[0]])

    def gm(self, x, u, d, p, t):
        return np.array([x[0]])


class IsomerisationReactor(ContinuousDiscreteSDAE):
    """
    Scalar SDAE model (isomerisation reactor, equilibrium algebraic constraint).

    DAE algebraic state ``y`` creates additional Jacobian blocks, making the
    analytical Jacobian benefit even more pronounced.
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


# ── Per-solve measurements ────────────────────────────────────────────────────


@dataclass
class _SolveRecord:
    nfev: int
    njev: int
    nit: int
    time_s: float
    cost: float
    success: bool
    n_decision_vars: int


def _solve_eocp(
    model: ContinuousDiscreteSDE,
    N: int,
    n_steps: int,
    *,
    strip_jac: bool = False,
) -> _SolveRecord:
    """Run a single EOCP solve and return metrics."""
    ocp = GeneralContinuousOCP(
        model,
        N,
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

    x0 = (
        np.array([0.0])
        if not isinstance(model, IsomerisationReactor)
        else np.array([4.0])
    )
    d_traj = np.zeros((N, model.nd))
    t0 = time.perf_counter()
    _, cost, info = ocp.solve(x0, d_traj)
    elapsed = time.perf_counter() - t0
    r = info["result"]
    return _SolveRecord(
        nfev=int(r.nfev or 0),
        njev=int(r.njev or 0),
        nit=int(r.nit or 0),
        time_s=float(elapsed),
        cost=float(cost),
        success=bool(r.success),
        n_decision_vars=int(ocp._layout.total),
    )


def _solve_cdtracking(
    model: ContinuousDiscreteSDE,
    N: int,
    n_steps: int,
    *,
    strip_jac: bool = False,
) -> _SolveRecord:
    """Run a single GeneralContinuousOCP (tracking) solve and return metrics."""
    ocp = GeneralContinuousOCP(
        model,
        N,
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

    x0 = np.array([0.0])
    d_traj = np.zeros((N, model.nd))
    t0 = time.perf_counter()
    _, cost, info = ocp.solve(x0, d_traj)
    elapsed = time.perf_counter() - t0
    r = info["result"]
    return _SolveRecord(
        nfev=int(r.nfev or 0),
        njev=int(r.njev or 0),
        nit=int(r.nit or 0),
        time_s=float(elapsed),
        cost=float(cost),
        success=bool(r.success),
        n_decision_vars=int(ocp._layout.total),
    )


# ── Aggregation ───────────────────────────────────────────────────────────────


def _aggregate(records: list[_SolveRecord]) -> dict:
    nfev = np.array([r.nfev for r in records], dtype=float)
    njev = np.array([r.njev for r in records], dtype=float)
    nit = np.array([r.nit for r in records], dtype=float)
    ts = np.array([r.time_s for r in records], dtype=float)
    success = np.mean([r.success for r in records])
    return {
        "median_nfev": float(np.median(nfev)),
        "median_njev": float(np.median(njev)),
        "median_nit": float(np.median(nit)),
        "median_time_s": float(np.median(ts)),
        "p90_time_s": float(np.percentile(ts, 90)),
        "success_rate": float(success),
        "n_decision_vars": records[0].n_decision_vars,
    }


# ── Main benchmark routine ────────────────────────────────────────────────────


def run_benchmark(
    horizons: list[int] = (3, 5, 10),
    n_steps: int = 5,
    repeats: int = 3,
) -> dict:
    """
    Run the analytical-vs-numerical Jacobian benchmark.

    Parameters
    ----------
    horizons : list[int]
        Prediction horizon values ``N`` to sweep over.
    n_steps : int
        Sub-steps per control interval (determines number of decision variables).
    repeats : int
        Number of repeated solves per configuration (median reported).

    Returns
    -------
    dict
        Nested result dict with keys ``cases``, ``summary``.
    """
    _SOLVER_LABEL = {True: "numerical_fd", False: "analytical"}

    cases: dict[str, dict] = {}

    configs: list[tuple[str, Callable, object]] = [
        ("sde_eocp", _solve_eocp, ScalarNonlinear()),
        ("sdae_eocp", _solve_eocp, IsomerisationReactor()),
        ("sde_cdtracking", _solve_cdtracking, ScalarNonlinear()),
    ]

    for case_name, solve_fn, model in configs:
        cases[case_name] = {}
        for N in horizons:
            cases[case_name][N] = {}
            for strip in (False, True):
                recs: list[_SolveRecord] = []
                for _ in range(repeats):
                    try:
                        recs.append(solve_fn(model, N, n_steps, strip_jac=strip))
                    except Exception as exc:
                        cases[case_name][N][_SOLVER_LABEL[strip]] = {"error": str(exc)}
                        break
                if recs:
                    cases[case_name][N][_SOLVER_LABEL[strip]] = _aggregate(recs)

    # ── Compute ratios ────────────────────────────────────────────────────────
    summary: dict[str, dict] = {}
    for case_name in cases:
        summary[case_name] = {}
        for N in horizons:
            ana = cases[case_name][N].get("analytical", {})
            num = cases[case_name][N].get("numerical_fd", {})
            if "median_nfev" in ana and "median_nfev" in num:
                nfev_ratio = num["median_nfev"] / max(ana["median_nfev"], 1)
                time_speedup = num["median_time_s"] / max(ana["median_time_s"], 1e-12)
                summary[case_name][N] = {
                    "n_decision_vars": ana["n_decision_vars"],
                    "nfev_reduction_ratio": round(nfev_ratio, 2),
                    "time_speedup": round(time_speedup, 2),
                    "nit_analytical": ana["median_nit"],
                    "nit_numerical": num["median_nit"],
                    "nfev_analytical": ana["median_nfev"],
                    "nfev_numerical": num["median_nfev"],
                    "success_analytical": ana["success_rate"],
                    "success_numerical": num["success_rate"],
                }

    return {"cases": cases, "summary": summary}


# ── ASCII table printer ───────────────────────────────────────────────────────


def _print_table(summary: dict, title: str, horizons: list[int]) -> None:
    print()
    print(f"  {title}")
    print("  " + "─" * 78)
    hdr = (
        f"  {'N':>4}  {'vars':>6}  {'nfev(ana)':>10}  {'nfev(num)':>10}  "
        f"{'nfev ↓':>8}  {'speedup':>8}  {'nit_eq':>7}"
    )
    print(hdr)
    print("  " + "─" * 78)
    for N in horizons:
        d = summary.get(N)
        if d is None:
            continue
        nit_eq = "yes" if d["nit_analytical"] == d["nit_numerical"] else "NO"
        print(
            f"  {N:>4}  {d['n_decision_vars']:>6}  "
            f"{d['nfev_analytical']:>10.0f}  {d['nfev_numerical']:>10.0f}  "
            f"{d['nfev_reduction_ratio']:>7.1f}x  {d['time_speedup']:>7.2f}x  "
            f"{nit_eq:>7}"
        )
    print("  " + "─" * 78)


def print_report(result: dict) -> None:
    """Print a human-readable benchmark report."""
    horizons = sorted({N for v in result["summary"].values() for N in v})
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Analytical Jacobian Efficiency Benchmark (SLSQP / scipy)          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(
        "  nfev ↓ = numerical_nfev / analytical_nfev (how many times more function")
    print(
        "           evaluations the numerical baseline needs)."
    )
    print(
        "  speedup = numerical solve time / analytical solve time."
    )
    print(
        "  nit_eq = 'yes' if both variants converge in the same number of iterations."
    )

    titles = {
        "sde_eocp": "EOCP SDE — ScalarNonlinear  (dx = −0.2x + tanh(u))",
        "sdae_eocp": "EOCP SDAE — IsomerisationReactor  (differential + algebraic)",
        "sde_cdtracking": "GeneralContinuousOCP tracking — ScalarNonlinear  (includes analytical R_stage+P_terminal Jac)",
    }
    for key, title in titles.items():
        _print_table(result["summary"].get(key, {}), title, horizons)

    print()
    print("  Legend: n_steps=5 sub-steps per control interval.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analytical Jacobian efficiency benchmark")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 20],
                        help="Prediction horizon values to sweep (default: 3 5 10 20)")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Sub-steps per control interval (default: 5)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeated solves per configuration (default: 3)")
    args = parser.parse_args()

    result = run_benchmark(
        horizons=args.horizons,
        n_steps=args.n_steps,
        repeats=args.repeats,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
