"""
Benchmark the linear-MPC QP backends and formulations.

Compares the QP backends (HiGHS, OSQP) crossed with the two
``DiscreteLinearOCP`` build strategies — the dense *condensed*
(state-eliminated) form and the sparse *simultaneous* form with banded
dynamics equalities — across a range of prediction horizons, and checks that
they agree to solver tolerance.

Findings:
- **OSQP + sparse** is the fastest combination and scales ~linearly in the
  horizon (it exploits the banded KKT structure) — this is the default
  (`solver="osqp"`, `formulation="auto"` → sparse).
- HiGHS solves the small dense *condensed* QP fast (active-set, exact) but its
  solver does not exploit the banded structure, so HiGHS + sparse is slow at
  long horizons; `formulation="auto"` therefore pairs HiGHS with condensed.
- OSQP + condensed is *not* recommended for long horizons: the dense condensed
  Hessian is ill-conditioned for OSQP's first-order method.

Usage:
    python scripts/qp_formulation_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mbc.control import DiscreteLinearOCP
from mbc.models import DiscreteLinearSDE


class _MassChain(DiscreteLinearSDE):
    """Discrete-time chain of ``m`` masses — a non-trivial multivariable plant."""

    def __init__(self, m: int = 4) -> None:
        self._m = m
        n = 2 * m
        A = np.eye(n)
        dt = 0.1
        for i in range(m):
            A[2 * i, 2 * i + 1] = dt
        # simple spring-damper coupling
        for i in range(m):
            k = 1.0
            c = 0.2
            A[2 * i + 1, 2 * i] += -k * dt
            A[2 * i + 1, 2 * i + 1] += -c * dt
            if i + 1 < m:
                A[2 * i + 1, 2 * (i + 1)] += k * dt
        self._A = A
        B = np.zeros((n, m))
        for i in range(m):
            B[2 * i + 1, i] = dt
        self._B = B
        self._n = n

    @property
    def nx(self): return self._n
    @property
    def nu(self): return self._m
    @property
    def nd(self): return 1
    @property
    def Ad(self): return self._A
    @property
    def Bd(self): return self._B
    @property
    def Ed(self): return np.zeros((self._n, 1))
    @property
    def Cm(self): return np.eye(self._n)
    @property
    def Qd(self): return np.eye(self._n) * 1e-4
    @property
    def Rm(self): return np.eye(self._n) * 1e-2
    @property
    def x(self): return [0.0] * self._n
    @x.setter
    def x(self, v): pass
    @property
    def x_ref(self): return np.ones(self._n) * 0.5
    @property
    def u_bounds(self): return -np.ones(self._m), np.ones(self._m)


def _time_solve(ocp, x0, D, x_ref, repeats=5):
    best = float("inf")
    U = X = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        U, X = ocp.solve(x0, D, x_ref)
        best = min(best, time.perf_counter() - t0)
    return best, U, X


def main() -> None:
    model = _MassChain(m=4)
    x0 = np.zeros(model.nx)
    x_ref = model.x_ref

    combos = [
        ("highs", "condensed"),
        ("highs", "sparse"),
        ("osqp", "condensed"),
        ("osqp", "sparse"),
    ]
    header = "  ".join(f"{s}/{f}".rjust(14) for s, f in combos)
    print(f"{'N':>4} | {header}   | {'max|dU|':>9}")
    print("-" * (8 + len(header) + 14))
    for N in [5, 10, 20, 40, 80]:
        D = np.zeros(N * model.nd)
        times = []
        Us = []
        for solver, form in combos:
            ocp = DiscreteLinearOCP(
                model, N=N, Q=np.eye(model.nx), R=np.eye(model.nu) * 0.1,
                y_offset=5.0, solver=solver, formulation=form,
            )
            t, U, _ = _time_solve(ocp, x0, D, x_ref)
            times.append(t * 1e3)
            Us.append(U)
        dev = max(float(np.max(np.abs(U - Us[0]))) for U in Us[1:])
        cells = "  ".join(f"{t:14.2f}" for t in times)
        print(f"{N:>4} | {cells}   | {dev:>9.2e}")


if __name__ == "__main__":
    main()
