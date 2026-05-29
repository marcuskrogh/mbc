"""Backend interface and HiGHS implementation for convex QP solves.

The linear (and successive-linearisation / continuous-discrete) MPC problems
in :mod:`mbc.control` reduce to a single finite-horizon convex **quadratic
program**

    min_x   ½ xᵀ P x + qᵀ x
    s.t.    G x ≤ h            (general inequalities)
            A x = b            (general equalities)
            lb ≤ x ≤ ub        (variable box)

This module mirrors the pluggable-backend pattern of
:mod:`mbc.control.nlp_solver`: a :class:`QPSolverBackend` protocol with a
normalised :class:`QPProblem` / :class:`QPResult` contract, and a default
:class:`HighsQPBackend` built on the MIT-licensed `HiGHS <https://highs.dev>`_
solver (via the ``highspy`` wheel).  HiGHS is a high-performance LP/MIP/QP
solver with no copyleft obligations, which keeps the toolbox cleanly
MIT-licensed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class QPProblem:
    """Convex QP in the standard form ``min ½ xᵀP x + qᵀx`` s.t. constraints.

    Parameters
    ----------
    P : (n, n) ndarray
        Symmetric positive-semidefinite Hessian of the objective.
    q : (n,) ndarray
        Linear term of the objective.
    lb, ub : (n,) ndarray
        Variable lower/upper bounds.  Use ``±np.inf`` for free variables.
    G : (m, n) ndarray or None
        Inequality matrix for ``G x ≤ h``.  ``None`` → no inequalities.
    h : (m,) ndarray or None
        Inequality right-hand side.
    A : (p, n) ndarray or None
        Equality matrix for ``A x = b``.  ``None`` → no equalities.
    b : (p,) ndarray or None
        Equality right-hand side.
    """

    P: np.ndarray
    q: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    G: np.ndarray | None = None
    h: np.ndarray | None = None
    A: np.ndarray | None = None
    b: np.ndarray | None = None


@dataclass(frozen=True)
class QPResult:
    """Normalised QP backend result."""

    x: np.ndarray
    obj: float
    success: bool
    status: str
    raw: Any = None


class QPSolverBackend(Protocol):
    """Protocol for swappable convex-QP solvers."""

    def solve(self, problem: QPProblem) -> QPResult:
        """Solve a convex QP and return a normalised result."""


def _stack_constraints(
    problem: QPProblem,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack inequality and equality rows into a single constraint system.

    Returns ``(A_con, row_lower, row_upper)`` with the row-bound convention
    used by HiGHS: inequality ``G x ≤ h`` becomes ``-inf ≤ G x ≤ h`` and
    equality ``A x = b`` becomes ``b ≤ A x ≤ b``.
    """
    rows: list[np.ndarray] = []
    low: list[np.ndarray] = []
    upp: list[np.ndarray] = []

    if problem.G is not None and problem.G.size:
        G = np.asarray(problem.G, dtype=float).reshape(-1, n)
        h = np.asarray(problem.h, dtype=float).reshape(-1)
        rows.append(G)
        low.append(np.full(G.shape[0], -np.inf))
        upp.append(h)

    if problem.A is not None and problem.A.size:
        A = np.asarray(problem.A, dtype=float).reshape(-1, n)
        b = np.asarray(problem.b, dtype=float).reshape(-1)
        rows.append(A)
        low.append(b)
        upp.append(b)

    if rows:
        return np.vstack(rows), np.concatenate(low), np.concatenate(upp)
    return np.zeros((0, n)), np.zeros(0), np.zeros(0)


class HighsQPBackend:
    """Convex-QP backend built on the MIT-licensed HiGHS solver (``highspy``).

    Parameters
    ----------
    options : dict, optional
        HiGHS option overrides forwarded via ``setOptionValue`` (e.g.
        ``{"presolve": "on", "time_limit": 10.0}``).  Solver output is
        silenced by default.
    """

    def __init__(self, *, options: dict[str, Any] | None = None) -> None:
        self._options = dict(options) if options is not None else {}

    def solve(self, problem: QPProblem) -> QPResult:
        try:
            import highspy
        except ImportError as exc:  # pragma: no cover - core dependency
            raise RuntimeError(
                "HiGHS QP backend requested but 'highspy' is not available. "
                "Install it with 'pip install highspy'."
            ) from exc

        from scipy.sparse import csc_matrix

        P = np.asarray(problem.P, dtype=float)
        q = np.asarray(problem.q, dtype=float).reshape(-1)
        n = q.shape[0]
        lb = np.asarray(problem.lb, dtype=float).reshape(-1)
        ub = np.asarray(problem.ub, dtype=float).reshape(-1)

        A_con, row_lower, row_upper = _stack_constraints(problem, n)
        m = A_con.shape[0]

        inf = highspy.kHighsInf
        lb = np.where(np.isneginf(lb), -inf, lb)
        ub = np.where(np.isposinf(ub), inf, ub)
        row_lower = np.where(np.isneginf(row_lower), -inf, row_lower)
        row_upper = np.where(np.isposinf(row_upper), inf, row_upper)

        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = m
        lp.col_cost_ = q
        lp.col_lower_ = lb
        lp.col_upper_ = ub
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper

        # Constraint matrix in compressed-sparse-column form.
        A_csc = csc_matrix(A_con) if m else csc_matrix((0, n))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = n
        lp.a_matrix_.num_row_ = m
        lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
        lp.a_matrix_.value_ = A_csc.data.astype(float)

        # Hessian Q (objective ½ xᵀQx): HiGHS stores the lower triangle in CSC.
        P_tri = csc_matrix(np.tril(P))
        hessian = highspy.HighsHessian()
        hessian.dim_ = n
        hessian.format_ = highspy.HessianFormat.kTriangular
        hessian.start_ = P_tri.indptr.astype(np.int32)
        hessian.index_ = P_tri.indices.astype(np.int32)
        hessian.value_ = P_tri.data.astype(float)

        model = highspy.HighsModel()
        model.lp_ = lp
        model.hessian_ = hessian

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        for key, value in self._options.items():
            h.setOptionValue(key, value)

        h.passModel(model)
        run_status = h.run()

        model_status = h.getModelStatus()
        status_str = h.modelStatusToString(model_status)
        optimal = model_status == highspy.HighsModelStatus.kOptimal

        solution = h.getSolution()
        x = np.asarray(solution.col_value, dtype=float).reshape(-1)
        if x.shape[0] != n:
            x = np.zeros(n)
        obj = float(0.5 * x @ P @ x + q @ x) if x.shape[0] == n else float("nan")

        return QPResult(
            x=x,
            obj=obj,
            success=bool(optimal and run_status == highspy.HighsStatus.kOk),
            status=str(status_str),
            raw=solution,
        )


def make_qp_backend(
    solver: str | QPSolverBackend = "highs",
    *,
    solver_options: dict[str, Any] | None = None,
) -> QPSolverBackend:
    """Create a QP backend from a solver spec or return the provided backend."""
    if hasattr(solver, "solve") and not isinstance(solver, str):
        return solver

    if not isinstance(solver, str):
        raise TypeError("solver must be a string key or a QP backend object.")

    key = solver.lower()
    if key in {"highs", "highspy"}:
        return HighsQPBackend(options=solver_options)

    raise ValueError(
        f"Unknown QP solver '{solver}'. Supported: 'highs'. "
        "Alternatively pass a QPSolverBackend instance."
    )
