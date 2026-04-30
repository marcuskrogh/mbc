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
    n   – state dimension          x ∈ ℝⁿ
    m   – input dimension          u ∈ ℝᵐ
    p   – disturbance dimension    d ∈ ℝᵖ
    l   – output dimension         y ∈ ℝˡ
    N   – prediction horizon
    Ψ   – free-response matrix     Ψ ∈ ℝᴺⁿˣⁿ
    Γ   – forced-response matrix   Γ ∈ ℝᴺⁿˣᴺᵐ
    Λ   – disturbance matrix       Λ ∈ ℝᴺⁿˣᴺᵖ
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cvxopt import matrix

from .ocp import OptimalControlProblem

if TYPE_CHECKING:
    from ..models import LinearContinuousDiscreteModel


class CDOptimalControlProblem(OptimalControlProblem):
    """
    Receding-horizon QP for a linear continuous-discrete system.

    Inherits the full QP formulation and solver from
    ``OptimalControlProblem``.  The only difference is the model type:
    ``LinearContinuousDiscreteModel.discretize(d)`` provides ZOH-discretised
    matrices (A_d, B_d, E_d) in cvxopt format, compatible with the inherited
    ``solve`` method.

    Parameters
    ----------
    model : LinearContinuousDiscreteModel
        Plant model.  Must implement ``n_x``, ``n_u``, ``n_d``, ``C``,
        ``u_bounds``, and ``discretize(d)``.
    N : int
        Prediction horizon (number of sampling intervals).
    Q : cvxopt.matrix (l, l)
        Stage output tracking cost  ‖y − r‖²_Q.
    R : cvxopt.matrix (m, m)
        Stage input cost  ‖u‖²_R.
    P : cvxopt.matrix (l, l), optional
        Terminal output tracking cost.  Default: Q.
    S : cvxopt.matrix (m, m), optional
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
        super().__init__(
            model=model,   # type: ignore[arg-type]  — duck-type compatible
            N=N,
            Q=Q,
            R=R,
            P=P,
            S=S,
            rho=rho,
            y_offset=y_offset,
        )
