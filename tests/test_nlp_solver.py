from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from mbc.control import (
    IpoptNLPBackend,
    NLPConstraint,
    NLPProblem,
    NLPScalingPolicy,
    ScipyNLPBackend,
)
from mbc.control.nlp_solver import make_nlp_backend


def _quadratic_objective(x: np.ndarray) -> float:
    return float((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)


def _quadratic_objective_jac(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] + 2.0)])


def _quadratic_objective_hess(_: np.ndarray) -> np.ndarray:
    return 2.0 * np.eye(2)


def _equality_constraint(x: np.ndarray) -> np.ndarray:
    return np.array([x[0] + x[1]])


def _equality_constraint_jac(_: np.ndarray) -> np.ndarray:
    return np.array([[1.0, 1.0]])


def _make_problem() -> NLPProblem:
    return NLPProblem(
        objective=_quadratic_objective,
        objective_jac=_quadratic_objective_jac,
        objective_hess=_quadratic_objective_hess,
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


def _make_problem_without_derivatives() -> NLPProblem:
    return NLPProblem(
        objective=_quadratic_objective,
        x0=np.array([0.0, 0.0]),
        lb=np.array([-10.0, -10.0]),
        ub=np.array([10.0, 10.0]),
        constraints=(
            NLPConstraint(
                kind="eq",
                fun=_equality_constraint,
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


def test_scipy_backend_uses_analytical_hessian_when_supported():
    backend = ScipyNLPBackend(method="trust-constr", options={"maxiter": 200})
    result = backend.solve(_make_problem())

    assert result.success
    assert np.allclose(result.x, np.array([1.5, -1.5]), atol=1e-6)
    assert np.isclose(result.fun, 0.5, atol=1e-6)
    assert result.nhev is not None and result.nhev > 0


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


def test_scipy_backend_analytical_hessian_with_scaling():
    scaling = NLPScalingPolicy(
        objective_scale=0.5,
        variable_scale=np.array([4.0, 0.5]),
        constraint_scale=3.0,
    )
    backend = ScipyNLPBackend(method="trust-constr", options={"maxiter": 200}, scaling=scaling)
    result = backend.solve(_make_problem())

    assert result.success
    assert np.allclose(result.x, np.array([1.5, -1.5]), atol=1e-6)
    assert np.isclose(result.fun, 0.5, atol=1e-6)
    assert result.nhev is not None and result.nhev > 0


def test_scipy_backend_falls_back_to_numerical_derivatives_for_slsqp():
    backend = ScipyNLPBackend(method="SLSQP", options={"maxiter": 200})
    result = backend.solve(_make_problem_without_derivatives())

    assert result.success
    assert np.allclose(result.x, np.array([1.5, -1.5]), atol=1e-6)
    assert np.isclose(result.fun, 0.5, atol=1e-6)
    assert result.nfev is not None and result.nfev > 0


def test_scipy_backend_falls_back_to_numerical_derivatives_for_trust_constr():
    backend = ScipyNLPBackend(method="trust-constr", options={"maxiter": 200})
    result = backend.solve(_make_problem_without_derivatives())

    assert result.success
    assert np.allclose(result.x, np.array([1.5, -1.5]), atol=1e-6)
    assert np.isclose(result.fun, 0.5, atol=1e-6)
    assert result.nfev is not None and result.nfev > 0


def test_make_nlp_backend_solver_selection_keys_and_aliases():
    ipopt_backend = make_nlp_backend("ipopt")
    ipopt_alias_backend = make_nlp_backend("cyipopt")
    scipy_backend = make_nlp_backend("scipy")
    scipy_alias_backend = make_nlp_backend("scipy-minimize", solver_options={"method": "trust-constr"})
    scipy_method_backend = make_nlp_backend("SLSQP")

    assert isinstance(ipopt_backend, IpoptNLPBackend)
    assert isinstance(ipopt_alias_backend, IpoptNLPBackend)
    assert isinstance(scipy_backend, ScipyNLPBackend)
    assert isinstance(scipy_alias_backend, ScipyNLPBackend)
    assert isinstance(scipy_method_backend, ScipyNLPBackend)
    assert scipy_backend._method == "SLSQP"
    assert scipy_alias_backend._method == "trust-constr"
    assert scipy_method_backend._method == "SLSQP"


def test_ipopt_backend_passes_no_analytical_derivatives_when_not_supplied(monkeypatch):
    recorded: dict[str, object] = {}

    def fake_minimize_ipopt(fun, x0, **kwargs):
        recorded["kwargs"] = kwargs
        return SimpleNamespace(
            x=np.array([1.5, -1.5]),
            success=True,
            status=0,
            message="ok",
            nit=1,
            nfev=1,
            njev=0,
            nhev=0,
        )

    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(minimize_ipopt=fake_minimize_ipopt),
    )

    backend = IpoptNLPBackend(options={"max_iter": 100})
    result = backend.solve(_make_problem_without_derivatives())

    assert result.success
    kwargs = recorded["kwargs"]
    assert "jac" not in kwargs
    assert "hess" not in kwargs


def test_ipopt_backend_passes_analytical_derivatives_when_supplied(monkeypatch):
    recorded: dict[str, object] = {}

    def fake_minimize_ipopt(fun, x0, **kwargs):
        recorded["kwargs"] = kwargs
        return SimpleNamespace(
            x=np.array([1.5, -1.5]),
            success=True,
            status=0,
            message="ok",
            nit=1,
            nfev=1,
            njev=1,
            nhev=1,
        )

    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(minimize_ipopt=fake_minimize_ipopt),
    )

    backend = IpoptNLPBackend(options={"max_iter": 100})
    result = backend.solve(_make_problem())

    assert result.success
    kwargs = recorded["kwargs"]
    assert "jac" in kwargs
    assert "hess" in kwargs
