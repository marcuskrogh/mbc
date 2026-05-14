"""Backend interface and wrappers for nonlinear NLP solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class NLPScalingPolicy:
    """Backend-agnostic scaling controls for NLP solve variables/functions."""

    objective_scale: float = 1.0
    variable_scale: float | np.ndarray | None = None
    constraint_scale: float | np.ndarray | None = None


@dataclass(frozen=True)
class NLPConstraint:
    """NLP constraint with scipy-style convention."""

    kind: Literal["eq", "ineq"]
    fun: Callable[[np.ndarray], np.ndarray]
    jac: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass(frozen=True)
class NLPProblem:
    """NLP problem definition passed to a backend."""

    objective: Callable[[np.ndarray], float]
    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    constraints: tuple[NLPConstraint, ...]


@dataclass(frozen=True)
class NLPResult:
    """Normalized backend result object."""

    x: np.ndarray
    fun: float
    success: bool
    status: int | None
    message: str
    nit: int | None = None
    nfev: int | None = None
    njev: int | None = None
    raw: Any = None


class NLPSolverBackend(Protocol):
    """Protocol for swappable NLP solvers."""

    def solve(self, problem: NLPProblem) -> NLPResult:
        """Solve an NLP and return a normalized result."""


def _coerce_scale_vector(
    scale: float | np.ndarray | None,
    size: int,
    *,
    default: float = 1.0,
) -> np.ndarray:
    if scale is None:
        vec = np.full(size, float(default), dtype=float)
    elif np.isscalar(scale):
        vec = np.full(size, float(scale), dtype=float)
    else:
        vec = np.asarray(scale, dtype=float).reshape(-1)
        if vec.size != size:
            raise ValueError(f"Scale vector must have size {size}; got {vec.size}.")
    if np.any(vec <= 0.0):
        raise ValueError("Scale entries must be strictly positive.")
    return vec


def _apply_scaling(problem: NLPProblem, scaling: NLPScalingPolicy | None) -> tuple[
    NLPProblem,
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    if scaling is None:
        def identity(v: np.ndarray) -> np.ndarray:
            return np.asarray(v, dtype=float)

        return problem, identity, identity

    obj_scale = float(scaling.objective_scale)
    if obj_scale <= 0.0:
        raise ValueError("objective_scale must be strictly positive.")

    var_scale = _coerce_scale_vector(scaling.variable_scale, problem.x0.size, default=1.0)
    inv_var_scale = 1.0 / var_scale

    def to_unscaled(y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) * inv_var_scale

    def to_scaled(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) * var_scale

    def objective_scaled(y: np.ndarray) -> float:
        return float(problem.objective(to_unscaled(y)) * obj_scale)

    con_scale_raw = scaling.constraint_scale

    def scale_constraint(c_val: np.ndarray) -> np.ndarray:
        arr = np.asarray(c_val, dtype=float)
        if con_scale_raw is None:
            return arr
        if np.isscalar(con_scale_raw):
            s = float(con_scale_raw)
            if s <= 0.0:
                raise ValueError("constraint_scale must be strictly positive.")
            return arr * s
        s_vec = np.asarray(con_scale_raw, dtype=float).reshape(-1)
        if np.any(s_vec <= 0.0):
            raise ValueError("constraint_scale entries must be strictly positive.")
        return arr * s_vec

    constraints_scaled = []
    for con in problem.constraints:
        constraints_scaled.append(
            NLPConstraint(
                kind=con.kind,
                fun=lambda y, _f=con.fun: scale_constraint(_f(to_unscaled(y))),
                jac=None,
            )
        )

    lb_scaled = np.asarray(problem.lb, dtype=float) * var_scale
    ub_scaled = np.asarray(problem.ub, dtype=float) * var_scale
    x0_scaled = to_scaled(problem.x0)
    scaled_problem = NLPProblem(
        objective=objective_scaled,
        x0=x0_scaled,
        lb=lb_scaled,
        ub=ub_scaled,
        constraints=tuple(constraints_scaled),
    )
    return scaled_problem, to_unscaled, to_scaled


class ScipyNLPBackend:
    """SciPy ``minimize`` NLP backend."""

    def __init__(
        self,
        *,
        method: str = "SLSQP",
        options: dict[str, Any] | None = None,
        scaling: NLPScalingPolicy | None = None,
    ) -> None:
        self._method = method
        self._options = dict(options) if options is not None else None
        self._scaling = scaling

    def solve(self, problem: NLPProblem) -> NLPResult:
        from scipy.optimize import Bounds, minimize

        scaled_problem, to_unscaled, _ = _apply_scaling(problem, self._scaling)
        constraints = []
        for con in scaled_problem.constraints:
            constraints.append({"type": con.kind, "fun": con.fun})

        result = minimize(
            scaled_problem.objective,
            scaled_problem.x0,
            method=self._method,
            bounds=Bounds(scaled_problem.lb, scaled_problem.ub),
            constraints=constraints,
            options=self._options,
        )
        x = to_unscaled(np.asarray(result.x, dtype=float))
        return NLPResult(
            x=x,
            fun=float(problem.objective(x)),
            success=bool(result.success),
            status=getattr(result, "status", None),
            message=str(getattr(result, "message", "")),
            nit=getattr(result, "nit", None),
            nfev=getattr(result, "nfev", None),
            njev=getattr(result, "njev", None),
            raw=result,
        )


class IpoptNLPBackend:
    """IPOPT backend via ``cyipopt.minimize_ipopt``."""

    def __init__(
        self,
        *,
        options: dict[str, Any] | None = None,
        scaling: NLPScalingPolicy | None = None,
    ) -> None:
        self._options = dict(options) if options is not None else {}
        self._scaling = scaling

    def solve(self, problem: NLPProblem) -> NLPResult:
        try:
            from cyipopt import minimize_ipopt
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "IPOPT backend requested but cyipopt is not available. "
                "Install optional dependency 'mbc[ipopt]'."
            ) from exc

        from scipy.optimize import Bounds

        scaled_problem, to_unscaled, _ = _apply_scaling(problem, self._scaling)
        constraints = []
        for con in scaled_problem.constraints:
            constraints.append({"type": con.kind, "fun": con.fun})

        options = {"print_level": 0}
        options.update(self._options)
        result = minimize_ipopt(
            scaled_problem.objective,
            scaled_problem.x0,
            bounds=Bounds(scaled_problem.lb, scaled_problem.ub),
            constraints=constraints,
            options=options,
        )
        x = to_unscaled(np.asarray(result.x, dtype=float))
        return NLPResult(
            x=x,
            fun=float(problem.objective(x)),
            success=bool(result.success),
            status=getattr(result, "status", None),
            message=str(getattr(result, "message", "")),
            nit=getattr(result, "nit", None),
            nfev=getattr(result, "nfev", None),
            njev=getattr(result, "njev", None),
            raw=result,
        )


def make_nlp_backend(
    solver: str | NLPSolverBackend,
    *,
    solver_options: dict[str, Any] | None = None,
    scaling: NLPScalingPolicy | dict[str, Any] | None = None,
) -> NLPSolverBackend:
    """Create an NLP backend from a solver spec or return the provided backend."""
    if isinstance(scaling, dict):
        scaling = NLPScalingPolicy(**scaling)

    if hasattr(solver, "solve") and not isinstance(solver, str):
        return solver

    if not isinstance(solver, str):
        raise TypeError("solver must be a string key/method or an NLP backend object.")

    key = solver.lower()
    options = dict(solver_options) if solver_options is not None else None

    if key in {"ipopt", "cyipopt"}:
        return IpoptNLPBackend(options=options, scaling=scaling)

    if key in {"scipy", "scipy-minimize"}:
        method = "SLSQP"
        if options is not None and "method" in options:
            method = str(options.pop("method"))
        return ScipyNLPBackend(method=method, options=options, scaling=scaling)

    # Backwards-compatible path: any non-reserved string is treated
    # as a scipy.optimize.minimize method name (e.g. "SLSQP", "trust-constr").
    return ScipyNLPBackend(method=solver, options=options, scaling=scaling)
