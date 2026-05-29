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
    warm_start: np.ndarray | None = None
    """Optional primal warm-start point ``x`` (length ``n``).

    Used as the initial iterate when the backend supports it.  A warm start
    never changes the optimum — at worst it is ignored — so callers may
    always supply the previous solution in a receding-horizon loop.
    """


@dataclass(frozen=True)
class QPResult:
    """Normalised QP backend result."""

    x: np.ndarray
    obj: float
    success: bool
    status: str
    iterations: int | None = None
    raw: Any = None


class QPSolverBackend(Protocol):
    """Protocol for swappable convex-QP solvers."""

    def solve(self, problem: QPProblem) -> QPResult:
        """Solve a convex QP and return a normalised result."""


def _stack_constraints(problem: QPProblem, n: int):
    """Stack inequality and equality rows into a single sparse constraint system.

    Accepts dense ndarrays or ``scipy.sparse`` matrices for ``G``/``A`` and
    returns ``(A_con_csc, row_lower, row_upper)`` with the row-bound
    convention used by HiGHS: inequality ``G x ≤ h`` becomes
    ``-inf ≤ G x ≤ h`` and equality ``A x = b`` becomes ``b ≤ A x ≤ b``.
    The constraint matrix is built sparsely so the simultaneous (banded)
    formulation never materialises a dense block.
    """
    import scipy.sparse as sp

    rows = []
    low: list[np.ndarray] = []
    upp: list[np.ndarray] = []

    def _nrows(M) -> int:
        return M.shape[0] if sp.issparse(M) else np.asarray(M).reshape(-1, n).shape[0]

    if problem.G is not None and getattr(problem.G, "shape", (0,))[0]:
        G = sp.csr_matrix(problem.G) if not sp.issparse(problem.G) else problem.G.tocsr()
        m = _nrows(G)
        rows.append(G)
        low.append(np.full(m, -np.inf))
        upp.append(np.asarray(problem.h, dtype=float).reshape(-1))

    if problem.A is not None and getattr(problem.A, "shape", (0,))[0]:
        A = sp.csr_matrix(problem.A) if not sp.issparse(problem.A) else problem.A.tocsr()
        b = np.asarray(problem.b, dtype=float).reshape(-1)
        rows.append(A)
        low.append(b)
        upp.append(b)

    if rows:
        return sp.vstack(rows, format="csc"), np.concatenate(low), np.concatenate(upp)
    return sp.csc_matrix((0, n)), np.zeros(0), np.zeros(0)


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

        import scipy.sparse as sp

        P = problem.P if sp.issparse(problem.P) else np.asarray(problem.P, dtype=float)
        q = np.asarray(problem.q, dtype=float).reshape(-1)
        n = q.shape[0]
        lb = np.asarray(problem.lb, dtype=float).reshape(-1)
        ub = np.asarray(problem.ub, dtype=float).reshape(-1)

        A_csc, row_lower, row_upper = _stack_constraints(problem, n)
        m = A_csc.shape[0]

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
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = n
        lp.a_matrix_.num_row_ = m
        lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
        lp.a_matrix_.value_ = A_csc.data.astype(float)

        # Hessian Q (objective ½ xᵀQx): HiGHS stores the lower triangle in CSC.
        P_csc = P.tocsc() if sp.issparse(P) else sp.csc_matrix(P)
        P_tri = sp.tril(P_csc).tocsc()
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

        # Optional primal warm start.  HiGHS accepts a starting point via
        # setSolution; a warm start never alters the optimum (it is only an
        # initial iterate), so it is safe to pass best-effort.
        if problem.warm_start is not None:
            ws = np.asarray(problem.warm_start, dtype=float).reshape(-1)
            if ws.shape[0] == n:
                try:
                    start = highspy.HighsSolution()
                    start.col_value = ws.tolist()
                    h.setSolution(start)
                except Exception:  # pragma: no cover - version-dependent API
                    pass

        run_status = h.run()

        model_status = h.getModelStatus()
        status_str = h.modelStatusToString(model_status)
        optimal = model_status == highspy.HighsModelStatus.kOptimal

        try:
            info = h.getInfo()
            iterations = int(getattr(info, "qp_iteration_count", 0)) or None
        except Exception:  # pragma: no cover - version-dependent API
            iterations = None

        solution = h.getSolution()
        x = np.asarray(solution.col_value, dtype=float).reshape(-1)
        if x.shape[0] != n:
            x = np.zeros(n)
        Px = np.asarray(P @ x, dtype=float).reshape(-1)
        obj = float(0.5 * x @ Px + q @ x)

        return QPResult(
            x=x,
            obj=obj,
            success=bool(optimal and run_status == highspy.HighsStatus.kOk),
            status=str(status_str),
            iterations=iterations,
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
