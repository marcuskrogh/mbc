"""Tests for the convex-QP backends (HiGHS, OSQP) used by the linear MPC stack."""

from __future__ import annotations

import numpy as np
import pytest

from mbc.control import (
    QPProblem,
    QPResult,
    HighsQPBackend,
    OSQPBackend,
    make_qp_backend,
)

try:
    import osqp as _osqp_mod  # noqa: F401
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False

# Backends under test and a solution tolerance appropriate to each
# (HiGHS active-set is essentially exact; OSQP is a first-order/ADMM solver).
BACKENDS = [("highs", 1e-6)] + ([("osqp", 1e-4)] if _OSQP_AVAILABLE else [])


# Reference QP:  min ½xᵀPx + qᵀx,  P = 2I, q = [-2, -6]
#   unconstrained optimum:           x* = [1, 3]
#   with x1 + x2 ≤ 2 (active):        x* = [0, 2]
_P = np.array([[2.0, 0.0], [0.0, 2.0]])
_Q = np.array([-2.0, -6.0])


@pytest.mark.parametrize("solver, atol", BACKENDS)
class TestQPBackends:
    def test_unconstrained(self, solver, atol):
        res = make_qp_backend(solver).solve(
            QPProblem(P=_P, q=_Q, lb=np.full(2, -np.inf), ub=np.full(2, np.inf))
        )
        assert res.success
        np.testing.assert_allclose(res.x, [1.0, 3.0], atol=atol)

    def test_inequality_constraint_active(self, solver, atol):
        res = make_qp_backend(solver).solve(
            QPProblem(
                P=_P, q=_Q, lb=np.zeros(2), ub=np.full(2, 10.0),
                G=np.array([[1.0, 1.0]]), h=np.array([2.0]),
            )
        )
        assert res.success
        np.testing.assert_allclose(res.x, [0.0, 2.0], atol=atol)

    def test_equality_constraint(self, solver, atol):
        res = make_qp_backend(solver).solve(
            QPProblem(
                P=_P, q=_Q, lb=np.full(2, -np.inf), ub=np.full(2, np.inf),
                A=np.array([[1.0, 1.0]]), b=np.array([2.0]),
            )
        )
        assert res.success
        np.testing.assert_allclose(res.x, [0.0, 2.0], atol=atol)

    def test_variable_bounds_respected(self, solver, atol):
        res = make_qp_backend(solver).solve(
            QPProblem(P=_P, q=_Q, lb=np.array([1.5, 1.5]), ub=np.array([10.0, 10.0]))
        )
        assert res.success
        assert np.all(res.x >= 1.5 - 1e-4)
        np.testing.assert_allclose(res.x[0], 1.5, atol=atol)

    def test_iterations_field_present(self, solver, atol):
        res = make_qp_backend(solver).solve(
            QPProblem(P=_P, q=_Q, lb=np.full(2, -np.inf), ub=np.full(2, np.inf))
        )
        assert isinstance(res, QPResult)
        assert res.iterations is None or isinstance(res.iterations, int)

    def test_sparse_inputs_accepted(self, solver, atol):
        import scipy.sparse as sp
        res = make_qp_backend(solver).solve(
            QPProblem(
                P=sp.csc_matrix(_P), q=_Q, lb=np.zeros(2), ub=np.full(2, 10.0),
                G=sp.csc_matrix(np.array([[1.0, 1.0]])), h=np.array([2.0]),
            )
        )
        assert res.success
        np.testing.assert_allclose(res.x, [0.0, 2.0], atol=atol)


@pytest.mark.parametrize("solver, atol", BACKENDS)
class TestWarmStart:
    def test_warm_start_does_not_change_optimum(self, solver, atol):
        """A warm start (even a wrong one) must not move x*."""
        backend = make_qp_backend(solver)
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
        np.testing.assert_allclose(warm_good.x, cold.x, atol=atol)
        np.testing.assert_allclose(warm_bad.x, cold.x, atol=atol)

    def test_wrong_size_warm_start_ignored(self, solver, atol):
        res = make_qp_backend(solver).solve(
            QPProblem(
                P=_P, q=_Q, lb=np.full(2, -np.inf), ub=np.full(2, np.inf),
                warm_start=np.array([0.0, 0.0, 0.0]),  # wrong length
            )
        )
        assert res.success
        np.testing.assert_allclose(res.x, [1.0, 3.0], atol=atol)


class TestBackendFactory:
    def test_passthrough_instance(self):
        b = HighsQPBackend()
        assert make_qp_backend(b) is b

    def test_highs_key(self):
        assert isinstance(make_qp_backend("highs"), HighsQPBackend)

    @pytest.mark.skipif(not _OSQP_AVAILABLE, reason="osqp not installed")
    def test_osqp_key(self):
        assert isinstance(make_qp_backend("osqp"), OSQPBackend)

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError):
            make_qp_backend("nonsense-solver")

    def test_non_string_non_backend_raises(self):
        with pytest.raises(TypeError):
            make_qp_backend(42)
