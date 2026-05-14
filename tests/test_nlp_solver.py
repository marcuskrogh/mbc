from __future__ import annotations

import numpy as np

from mbc.control import (
    NLPConstraint,
    NLPProblem,
    NLPScalingPolicy,
    ScipyNLPBackend,
)


def _quadratic_objective(x: np.ndarray) -> float:
    return float((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)


def _quadratic_objective_jac(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] + 2.0)])


def _equality_constraint(x: np.ndarray) -> np.ndarray:
    return np.array([x[0] + x[1]])


def _equality_constraint_jac(_: np.ndarray) -> np.ndarray:
    return np.array([[1.0, 1.0]])


def _make_problem() -> NLPProblem:
    return NLPProblem(
        objective=_quadratic_objective,
        objective_jac=_quadratic_objective_jac,
        x0=np.array([0.0, 0.0]),
        lb=np.array([-10.0, -10.0]),
        ub=np.array([10.0, 10.0]),
        constraints=(
            NLPConstraint(
                kind="eq",
                fun=_equality_constraint,
                jac=_equality_constraint_jac,
            ),
        ),
    )


def test_scipy_backend_uses_analytical_derivatives():
    backend = ScipyNLPBackend(method="SLSQP", options={"maxiter": 200})
    result = backend.solve(_make_problem())

    assert result.success
    assert np.allclose(result.x, np.array([1.5, -1.5]), atol=1e-6)
    assert np.isclose(result.fun, 0.5, atol=1e-6)
    assert result.njev is not None and result.njev > 0


def test_scipy_backend_analytical_derivatives_with_scaling():
    scaling = NLPScalingPolicy(
        objective_scale=0.5,
        variable_scale=np.array([4.0, 0.5]),
        constraint_scale=3.0,
    )
    backend = ScipyNLPBackend(method="SLSQP", options={"maxiter": 200}, scaling=scaling)
    result = backend.solve(_make_problem())

    assert result.success
    assert np.allclose(result.x, np.array([1.5, -1.5]), atol=1e-6)
    assert np.isclose(result.fun, 0.5, atol=1e-6)
