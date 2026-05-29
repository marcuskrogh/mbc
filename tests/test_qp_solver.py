"""Tests for the convex-QP backend (HiGHS) used by the linear MPC stack."""

from __future__ import annotations

import numpy as np
import pytest

from mbc.control import (
    QPProblem,
    QPResult,
    HighsQPBackend,
    make_qp_backend,
)


def _backend():
    return make_qp_backend("highs")


# Reference QP:  min ½xᵀPx + qᵀx,  P = 2I, q = [-2, -6]
#   unconstrained optimum:           x* = [1, 3]
#   with x1 + x2 ≤ 2 (active):        x* = [0, 2]
_P = np.array([[2.0, 0.0], [0.0, 2.0]])
_Q = np.array([-2.0, -6.0])


class TestHighsQPBackend:
    def test_unconstrained(self):
        res = _backend().solve(
            QPProblem(P=_P, q=_Q, lb=np.full(2, -np.inf), ub=np.full(2, np.inf))
        )
        assert res.success
        np.testing.assert_allclose(res.x, [1.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(res.obj, 0.5 * res.x @ _P @ res.x + _Q @ res.x)

    def test_inequality_constraint_active(self):
        res = _backend().solve(
            QPProblem(
                P=_P, q=_Q,
                lb=np.zeros(2), ub=np.full(2, 10.0),
                G=np.array([[1.0, 1.0]]), h=np.array([2.0]),
            )
        )
        assert res.success
        np.testing.assert_allclose(res.x, [0.0, 2.0], atol=1e-6)

    def test_equality_constraint(self):
        res = _backend().solve(
            QPProblem(
                P=_P, q=_Q,
                lb=np.full(2, -np.inf), ub=np.full(2, np.inf),
                A=np.array([[1.0, 1.0]]), b=np.array([2.0]),
            )
        )
        assert res.success
        np.testing.assert_allclose(res.x, [0.0, 2.0], atol=1e-6)

    def test_variable_bounds_respected(self):
        res = _backend().solve(
            QPProblem(P=_P, q=_Q, lb=np.array([1.5, 1.5]), ub=np.array([10.0, 10.0]))
        )
        assert res.success
        assert np.all(res.x >= 1.5 - 1e-6)
        # Unconstrained x1*=1 is below its lower bound → clamps to 1.5.
        np.testing.assert_allclose(res.x[0], 1.5, atol=1e-6)

    def test_iterations_field_present(self):
        res = _backend().solve(
            QPProblem(P=_P, q=_Q, lb=np.full(2, -np.inf), ub=np.full(2, np.inf))
        )
        assert isinstance(res, QPResult)
        assert res.iterations is None or isinstance(res.iterations, int)


class TestWarmStart:
    def test_warm_start_does_not_change_optimum(self):
        """A warm start (even a deliberately wrong one) must not move x*."""
        backend = _backend()
        prob = QPProblem(
            P=_P, q=_Q, lb=np.zeros(2), ub=np.full(2, 10.0),
            G=np.array([[1.0, 1.0]]), h=np.array([2.0]),
        )
        cold = backend.solve(prob)
        warm_good = backend.solve(
            QPProblem(**{**prob.__dict__, "warm_start": cold.x.copy()})
        )
        warm_bad = backend.solve(
            QPProblem(**{**prob.__dict__, "warm_start": np.array([9.0, 9.0])})
        )
        np.testing.assert_allclose(warm_good.x, cold.x, atol=1e-6)
        np.testing.assert_allclose(warm_bad.x, cold.x, atol=1e-6)

    def test_wrong_size_warm_start_ignored(self):
        backend = _backend()
        prob = QPProblem(
            P=_P, q=_Q, lb=np.full(2, -np.inf), ub=np.full(2, np.inf),
            warm_start=np.array([0.0, 0.0, 0.0]),  # wrong length
        )
        res = backend.solve(prob)
        assert res.success
        np.testing.assert_allclose(res.x, [1.0, 3.0], atol=1e-6)


class TestBackendFactory:
    def test_passthrough_instance(self):
        b = HighsQPBackend()
        assert make_qp_backend(b) is b

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError):
            make_qp_backend("nonsense-solver")

    def test_non_string_non_backend_raises(self):
        with pytest.raises(TypeError):
            make_qp_backend(42)
