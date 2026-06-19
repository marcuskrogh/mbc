"""
Continuous-discrete linearised receding-horizon QP
(``ContinuousLinearisedOCP``).

Extends :class:`ContinuousLinearOCP` with an explicit steady-state operating
point.  The OCP is formulated and solved in **deviation coordinates**
(``δx = x − x_s``, ``δu = u − u_s``, ``δd = d − d_s``), and the
``solve`` interface accepts and returns **absolute** coordinates so that
the same closed-loop code works with either linearised or non-linearised
linear models.

The steady-state operating point is read from the
:class:`~mbc.models.ContinuousDiscreteLinearisedSDE` model via its
``x_s``, ``u_s``, ``d_s``, and the ``*_dev`` / ``*_abs`` helpers.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from .continuous_linear_ocp import StandardLinearContinuousDiscreteOCP
from .qp_solver import QPSolverBackend

if TYPE_CHECKING:
    from ..models import ContinuousDiscreteLinearisedSDE


def _deviation_input_bound_profiles(
    *,
    N: int,
    nu: int,
    u_s: np.ndarray,
    u_min_abs: np.ndarray,
    u_max_abs: np.ndarray,
    input_min_profile: np.ndarray | None,
    input_max_profile: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert absolute input limits to deviation limits for δu = u − u_s.

    Horizon profiles, when supplied, are interpreted as absolute limits and
    shifted by the operating-point input ``u_s``.
    """
    u_s = np.asarray(u_s, dtype=float).reshape(nu)
    u_ss_row = u_s.reshape(1, -1)
    if input_min_profile is not None and input_max_profile is not None:
        u_min_dev = np.asarray(input_min_profile, dtype=float).reshape(N, nu) - u_ss_row
        u_max_dev = np.asarray(input_max_profile, dtype=float).reshape(N, nu) - u_ss_row
        return u_min_dev, u_max_dev
    u_min_abs = np.asarray(u_min_abs, dtype=float).reshape(nu)
    u_max_abs = np.asarray(u_max_abs, dtype=float).reshape(nu)
    return (
        np.tile((u_min_abs - u_s).reshape(1, -1), (N, 1)),
        np.tile((u_max_abs - u_s).reshape(1, -1), (N, 1)),
    )


class StandardLinearizedContinuousDiscreteOCP(StandardLinearContinuousDiscreteOCP):
    """
    Receding-horizon QP for a linearised continuous-discrete system, with
    automatic steady-state coordinate shifting.

    The QP is built and solved in deviation space; ``solve`` transparently
    converts incoming absolute coordinates to deviations, calls the parent
    solver, and converts the results back to absolute space.

    Parameters
    ----------
    model : ContinuousDiscreteLinearisedSDE
        Linearised continuous-discrete plant with steady-state operating point.
    N : int
        Prediction horizon (number of sampling intervals).
    Q : array-like
        Stage tracking cost in deviation space.  Accepts scalar, ``(N,)``
        per-step scalars, ``(N, nz)`` per-step diagonal vectors, or ``(nz, nz)``
        constant matrix.
    R : array-like
        Stage input cost on **absolute** inputs.  Same four forms as ``Q`` with
        ``nu`` replacing ``nz``.
    P : array-like, optional
        Terminal tracking cost.  Accepts scalar, ``(nz,)`` diagonal, or
        ``(nz, nz)`` matrix.  Default: last step's ``Q`` matrix.
    S : array-like, optional
        Input rate-of-movement cost.  Same four forms as ``R``.  ``None`` → disabled.
    du_min, du_max : (nu,) array-like, optional
        Hard input rate-of-movement box ``du_min ≤ Δu ≤ du_max``.
        ``None`` disables the corresponding bound.
    rho : float, (N,) or (N, nz) array-like, optional
        Quadratic penalty on the soft-output slack variable ``ε``.  Default: 1e4.
    rho_lin : float, (N,) or (N, nz) array-like, optional
        Linear (L1-style) penalty on ``ε``.  Default: 0.0.
    z_offset : float, (N,) or (N, nz) array-like, optional
        Symmetric half-width δ of the soft output constraint band.  Default: 2.0.
    solver : str or QPSolverBackend, optional
        Convex-QP backend.  Default: ``"highs"``.
    solver_options : dict, optional
        Forwarded to the QP backend.
    formulation : {"auto", "condensed", "sparse"}, optional
        QP construction strategy.  Default: ``"auto"``.

    Notes
    -----
    ``solve`` accepts and returns values in **absolute** coordinate space.
    The warm-start dict ``{"U", "X"}`` must also contain absolute-space
    values (as returned by a previous ``solve`` call).
    """

    def __init__(
        self,
        model: "ContinuousDiscreteLinearisedSDE",
        N: int,
        Q: Any,
        R: Any,
        P: Any | None = None,
        S: Any | None = None,
        du_min: Any | None = None,
        du_max: Any | None = None,
        rho: float = 1e4,
        rho_lin: float = 0.0,
        z_offset: float = 2.0,
        solver: str | QPSolverBackend = "highs",
        solver_options: dict[str, Any] | None = None,
        formulation: str = "auto",
    ) -> None:
        super().__init__(
            model=model,
            N=N,
            Q=Q,
            R=R,
            P=P,
            S=S,
            du_min=du_min,
            du_max=du_max,
            rho=rho,
            rho_lin=rho_lin,
            z_offset=z_offset,
            solver=solver,
            solver_options=solver_options,
            formulation=formulation,
        )

    def solve(
        self,
        x0: Any,
        D: Any,
        x_ref: Any,
        u_prev: Any | None = None,
        warm_start: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the QP starting from absolute state estimate ``x0``.

        Shifts all inputs to deviation coordinates, solves the parent QP,
        and shifts the outputs back to absolute coordinates.

        Parameters
        ----------
        x0    : (nx,) array-like — current absolute state estimate.
        D     : (N · nd,) array-like — stacked absolute disturbance forecast.
        x_ref : (nx,) array-like — absolute state reference.
        u_prev : (nu,) array-like, optional
            Previously-applied absolute input (used with rate-of-movement penalty).
        warm_start : dict, optional
            ``{"U": (N·nu,), "X": (N·nx,)}`` warm-start in **absolute** space
            (as returned by a previous ``solve`` call).

        Returns
        -------
        U : (N · nu,) ndarray — optimal absolute input sequence.
        X : (N · nx,) ndarray — predicted absolute state trajectory.
        """
        model = self._cd_model
        x_s = model.x_s
        u_s = model.u_s
        d_s = model.d_s

        nx = model.nx
        nu = model.nu
        nd = model.nd
        N = self._N

        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        x_ref_arr = np.asarray(x_ref, dtype=float).reshape(-1)
        D_arr = np.asarray(D, dtype=float).reshape(-1) if D is not None else np.zeros(N * nd)

        # Shift to deviation space
        delta_x0 = x0_arr - x_s
        delta_x_ref = x_ref_arr - x_s
        delta_D = (D_arr.reshape(N, nd) - d_s).reshape(-1)
        delta_u_prev = (
            np.asarray(u_prev, dtype=float).reshape(-1) - u_s
            if u_prev is not None
            else None
        )

        # Convert warm start from absolute to deviation space
        ws_dev: dict[str, np.ndarray] | None = None
        if warm_start is not None:
            ws_dev = {}
            U_ws = warm_start.get("U")
            X_ws = warm_start.get("X")
            if U_ws is not None:
                ws_dev["U"] = (np.asarray(U_ws, dtype=float).reshape(N, nu) - u_s).reshape(-1)
            if X_ws is not None:
                ws_dev["X"] = (np.asarray(X_ws, dtype=float).reshape(N, nx) - x_s).reshape(-1)

        # Solve in deviation space with absolute input regularisation.
        prof = self._horizon_profile
        u_min_abs, u_max_abs = self._cd_model.u_bounds
        saved_min = prof.input_min_profile
        saved_max = prof.input_max_profile
        prof.input_min_profile, prof.input_max_profile = _deviation_input_bound_profiles(
            N=N,
            nu=nu,
            u_s=u_s,
            u_min_abs=u_min_abs,
            u_max_abs=u_max_abs,
            input_min_profile=saved_min,
            input_max_profile=saved_max,
        )
        saved_ue = prof.input_equilibrium
        prof.input_equilibrium = u_s
        try:
            U_dev, X_dev = super().solve(
                delta_x0, delta_D, delta_x_ref, delta_u_prev, ws_dev
            )
        finally:
            prof.input_min_profile = saved_min
            prof.input_max_profile = saved_max
            prof.input_equilibrium = saved_ue

        # Convert back to absolute space
        U_abs = (U_dev.reshape(N, nu) + u_s).reshape(-1)
        X_abs = (X_dev.reshape(N, nx) + x_s).reshape(-1)

        return U_abs, X_abs
