"""
Optimal Control Problem for linear continuous-discrete systems.

``CDOptimalControlProblem`` is a thin, typed wrapper around
``OptimalControlProblem`` that accepts a ``LinearContinuousDiscreteModel``
and documents the continuous-discrete problem statement.

Problem formulation (M.Sc. thesis, Ch. 5)
------------------------------------------
Given the continuous-discrete model

    dx = (A_c x + B_c u + E_c d) dt + G dw
    y[k] = C x[k] + v[k]

the controller operates on the ZOH-discretised prediction model

    x[k+1] = A_d x[k] + B_d u[k] + E_d d[k]

where (A_d, B_d, E_d) = ``model.discretize(d)``.

The receding-horizon quadratic program over horizon N is:

    min_{U}  J(U) = Σₖ₌₀ᴺ⁻¹ [ ‖y[k+1] − r‖²_Q  +  ‖u[k]‖²_R
                               + ‖Δu[k]‖²_S ]
                   + ‖y[N] − r‖²_P
                   + ρ Σₖ₌₀ᴺ⁻¹ ‖ε[k+1]‖²

    s.t.  x[k+1] = A_d x[k] + B_d u[k] + E_d d[k]
          u_min ≤ u[k] ≤ u_max                        (hard input box)
          y_min − ε[k+1] ≤ y[k+1] ≤ y_max + ε[k+1]   (soft output box)
          ε[k+1] ≥ 0

The lifted (batch) QP is solved using ``cvxopt.solvers.qp``.

Notation
---------
    nx  – state dimension          x ∈ ℝⁿˣ
    nu  – input dimension          u ∈ ℝⁿᵘ
    nd  – disturbance dimension    d ∈ ℝⁿᵈ
    ny  – output dimension         y ∈ ℝⁿʸ
    N   – prediction horizon
    Ψ   – free-response matrix     Ψ ∈ ℝᴺⁿˣˣⁿˣ
    Γ   – forced-response matrix   Γ ∈ ℝᴺⁿˣˣᴺⁿᵘ
    Λ   – disturbance matrix       Λ ∈ ℝᴺⁿˣˣᴺⁿᵈ
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from cvxopt import matrix

from .ocp import OptimalControlProblem

if TYPE_CHECKING:
    from ..models import LinearContinuousDiscreteModel


class _CDModelAdapter:
    """
    Thin adapter that wraps a ``LinearContinuousDiscreteModel`` and exposes
    the cvxopt-compatible interface expected by ``OptimalControlProblem``.

    ``OptimalControlProblem.solve`` accesses:
      - ``model.n_x``, ``model.n_u``, ``model.n_d``  (underscore form, int)
      - ``model.C``         (cvxopt matrix, for output prediction)
      - ``model.u_bounds``  (tuple of cvxopt column vectors)
      - ``model.discretize(d)``  (returns cvxopt matrices)

    This adapter satisfies all four requirements by delegating to the
    appropriate numpy → cvxopt conversion properties on the CD model.
    """

    def __init__(self, model: "LinearContinuousDiscreteModel") -> None:
        self._m = model

    # Dimensions — underscore form as expected by OptimalControlProblem
    @property
    def n_x(self) -> int:
        return self._m.nx

    @property
    def n_u(self) -> int:
        return self._m.nu

    @property
    def n_d(self) -> int:
        return self._m.nd

    # cvxopt-format matrices
    @property
    def C(self) -> matrix:
        return self._m.C_cvx

    @property
    def u_bounds(self) -> Tuple[matrix, matrix]:
        return self._m.u_bounds_cvx

    # Discretisation — delegates directly (already returns cvxopt)
    def discretize(self, d: matrix) -> Tuple[matrix, matrix, matrix]:
        return self._m.discretize(d)


class CDOptimalControlProblem(OptimalControlProblem):
    """
    Receding-horizon QP for a linear continuous-discrete system.

    Inherits the full QP formulation and solver from
    ``OptimalControlProblem``.  The only difference is the model type:
    ``LinearContinuousDiscreteModel.discretize(d)`` provides ZOH-discretised
    matrices (A_d, B_d, E_d) in cvxopt format, compatible with the inherited
    ``solve`` method.

    A ``_CDModelAdapter`` is used internally to translate the numpy-based
    CD model interface (``C`` as ndarray, ``u_bounds`` as ndarray tuples)
    into the cvxopt-compatible interface required by
    ``OptimalControlProblem.solve``.  The original CD model is also stored
    as ``self._cd_model`` for direct access.

    Parameters
    ----------
    model : LinearContinuousDiscreteModel
        Plant model.  Must implement ``nx``, ``nu``, ``nd``, ``C``,
        ``u_bounds``, and ``discretize(d)``.
    N : int
        Prediction horizon (number of sampling intervals).
    Q : cvxopt.matrix (ny, ny)
        Stage output tracking cost  ‖y − r‖²_Q.
    R : cvxopt.matrix (nu, nu)
        Stage input cost  ‖u‖²_R.
    P : cvxopt.matrix (ny, ny), optional
        Terminal output tracking cost.  Default: Q.
    S : cvxopt.matrix (nu, nu), optional
        Input rate-of-movement cost  ‖Δu‖²_S.  ``None`` → disabled.
    rho : float, optional
        Penalty weight on soft output constraint violation.  Default: 1e4.
    y_offset : float, optional
        Symmetric half-width δ of the soft output constraint band
        ``[r − δ,  r + δ]``.  Default: 2.0.

    Notes
    -----
    ``CDOptimalControlProblem.solve`` has the same signature as
    ``OptimalControlProblem.solve`` — see that class for full documentation.
    """

    def __init__(
        self,
        model: "LinearContinuousDiscreteModel",
        N: int,
        Q: matrix,
        R: matrix,
        P: matrix | None = None,
        S: matrix | None = None,
        rho: float = 1e4,
        y_offset: float = 2.0,
    ) -> None:
        # Store the original CD model for direct access (e.g. x_ref_cvx)
        self._cd_model = model

        # Pass an adapter to the parent so that OptimalControlProblem.solve
        # receives the cvxopt-compatible interface it expects.
        super().__init__(
            model=_CDModelAdapter(model),  # type: ignore[arg-type]
            N=N,
            Q=Q,
            R=R,
            P=P,
            S=S,
            rho=rho,
            y_offset=y_offset,
        )
