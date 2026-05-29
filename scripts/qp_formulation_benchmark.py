"""
Benchmark the linear-MPC QP formulations and warm-starting.

Compares the two ``OptimalControlProblem`` build strategies — the dense
*condensed* (state-eliminated) form and the sparse *simultaneous* form with
banded dynamics equalities — across a range of prediction horizons, and
checks that they agree to solver tolerance.  Also times a receding-horizon
loop with and without warm-starting.

Finding (with the default HiGHS backend): the condensed form is faster — its
active-set QP solver does not exploit the banded KKT structure, so the small
dense condensed QP beats the larger simultaneous QP, and primal warm starts
are ignored.  The sparse form and warm-starting are therefore opt-in and are
intended for a banded-exploiting backend (e.g. OSQP / a Riccati solver),
where the simultaneous structure and warm starts pay off for long horizons.

Usage:
    python scripts/qp_formulation_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mbc.control import MPCController, OptimalControlProblem
from mbc.estimation import KalmanFilter
from mbc.models import LinearDiscreteModel


class _MassChain(LinearDiscreteModel):
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

    print(f"{'N':>4} | {'condensed (ms)':>15} | {'sparse (ms)':>12} | {'max|dU|':>10}")
    print("-" * 52)
    for N in [5, 10, 20, 40, 80]:
        D = np.zeros(N * model.nd)
        kw = dict(model=model, N=N, Q=np.eye(model.nx), R=np.eye(model.nu) * 0.1,
                  y_offset=5.0)
        oc = OptimalControlProblem(formulation="condensed", **kw)
        os_ = OptimalControlProblem(formulation="sparse", **kw)
        tc, Uc, _ = _time_solve(oc, x0, D, x_ref)
        ts, Us, _ = _time_solve(os_, x0, D, x_ref)
        d = float(np.max(np.abs(Uc - Us)))
        print(f"{N:>4} | {tc * 1e3:>15.2f} | {ts * 1e3:>12.2f} | {d:>10.2e}")

    # Warm-start vs cold-start over a receding-horizon loop.
    print("\nReceding-horizon loop (N=40, 40 steps):")
    for warm in (False, True):
        kf = KalmanFilter(model)
        ocp = OptimalControlProblem(
            model, N=40, Q=np.eye(model.nx), R=np.eye(model.nu) * 0.1, y_offset=5.0,
        )
        ctrl = MPCController(model, estimator=kf, ocp=ocp, warm_start=warm)
        t0 = time.perf_counter()
        for k in range(40):
            ctrl.step(np.zeros(model.nx), np.zeros(40 * model.nd))
        dt = time.perf_counter() - t0
        print(f"  warm_start={str(warm):>5}:  {dt * 1e3:8.1f} ms total")


if __name__ == "__main__":
    main()
