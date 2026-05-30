"""
Backward-compatibility shim: re-exports from ``continuous_linear_ocp``.

``CDOptimalControlProblem`` is an alias for :class:`ContinuousLinearOCP`.
``CDTrackingOptimalControlProblem`` remains as a thin tracking-friendly
wrapper around :class:`~mbc.control.ContinuousOCP`.

New code should use :class:`ContinuousLinearOCP` and :class:`ContinuousOCP`
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .continuous_linear_ocp import ContinuousLinearOCP, _CDModelAdapter
from .continuous_ocp import ContinuousOCP
from .nlp_solver import NLPScalingPolicy, NLPSolverBackend

if TYPE_CHECKING:
    from ..models import ContinuousDiscreteSDE

# Backward-compatible alias
CDOptimalControlProblem = ContinuousLinearOCP


class CDTrackingOptimalControlProblem:
    """
    Tracking OCP for continuous-discrete nonlinear systems — thin wrapper
    around :class:`~mbc.control.ContinuousOCP` that exposes a
    quadratic-tracking-friendly constructor (``Q``, ``R``, ``P``, ``S``, ``c_u``).

    .. deprecated::
        Use :class:`~mbc.control.ContinuousOCP` directly with ``Q_z``,
        ``R_stage``, ``P_terminal``, and ``Q_du`` parameters instead.
    """

    def __init__(
        self,
        model: "ContinuousDiscreteSDE",
        N: int,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray | None = None,
        S: np.ndarray | None = None,
        c_u: np.ndarray | None = None,
        z_ref: np.ndarray | None = None,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        du_min: np.ndarray | None = None,
        du_max: np.ndarray | None = None,
        x_min: np.ndarray | None = None,
        x_max: np.ndarray | None = None,
        rho_x: float = 1e4,
        z_min: np.ndarray | None = None,
        z_max: np.ndarray | None = None,
        rho_z: float = 1e4,
        n_steps: int = 10,
        solver: str | NLPSolverBackend = "SLSQP",
        solver_options: dict | None = None,
        solver_scaling: NLPScalingPolicy | dict | None = None,
        dt: float | None = None,
    ) -> None:
        Q_arr = np.asarray(Q, dtype=float)
        R_arr = np.asarray(R, dtype=float)
        z_ref_arr = (
            np.asarray(z_ref, dtype=float)
            if z_ref is not None
            else np.zeros(model.nz)
        )

        self._eocp = ContinuousOCP(
            model,
            N,
            Q_z=Q_arr,
            z_ref=z_ref_arr,
            R_stage=R_arr,
            P_terminal=np.asarray(P, dtype=float) if P is not None else None,
            Q_du=np.asarray(S, dtype=float) if S is not None else None,
            p_u_eco=np.asarray(c_u, dtype=float) if c_u is not None else None,
            u_min=u_min,
            u_max=u_max,
            du_min=du_min,
            du_max=du_max,
            x_min=x_min,
            x_max=x_max,
            rho_x_2=rho_x,
            z_min=z_min,
            z_max=z_max,
            rho_z_2=rho_z,
            n_steps=n_steps,
            solver=solver,
            solver_options=solver_options,
            solver_scaling=solver_scaling,
            dt=dt,
        )

    @property
    def N(self) -> int:
        """Prediction horizon (number of control intervals)."""
        return self._eocp.N

    @property
    def nu(self) -> int:
        """Input dimension."""
        return self._eocp.nu

    def solve(
        self,
        x0: np.ndarray,
        d_trajectory: np.ndarray,
        u_prev: np.ndarray | None = None,
        x_prev: np.ndarray | None = None,
        y_prev: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t0: float = 0.0,
    ) -> tuple[np.ndarray, float, dict]:
        """
        Solve the tracking OCP from initial state ``x0``.

        Returns
        -------
        u_opt : (N, nu) ndarray  — optimal input sequence.
        cost  : float            — optimal NLP objective value.
        info  : dict             — ``{"X", "Y", "result"}`` for warm-starting.
        """
        return self._eocp.solve(
            x0, d_trajectory,
            u_prev=u_prev, x_prev=x_prev, y_prev=y_prev,
            p=p, t0=t0,
        )

    def step(
        self,
        x0: np.ndarray,
        d_trajectory: np.ndarray,
        u_prev: np.ndarray | None = None,
        x_prev: np.ndarray | None = None,
        y_prev: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t0: float = 0.0,
    ) -> np.ndarray:
        """Solve and return only the first optimal control action."""
        return self._eocp.step(
            x0, d_trajectory,
            u_prev=u_prev, x_prev=x_prev, y_prev=y_prev,
            p=p, t0=t0,
        )
