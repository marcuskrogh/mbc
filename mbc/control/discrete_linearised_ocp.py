"""
Discrete-time linearised receding-horizon QP (``DiscreteLinearisedOCP``).

Extends :class:`DiscreteLinearOCP` with an explicit steady-state operating
point.  The OCP is formulated and solved in **deviation coordinates**
(``δx = x − x_s``, ``δu = u − u_s``, ``δd = d − d_s``), and the
``solve`` interface accepts and returns **absolute** coordinates so that
the same closed-loop code works with either linearised or non-linearised
discrete-time linear models.

The steady-state operating point is read from the
:class:`~mbc.models.DiscreteLinearisedSDE` model via its ``x_s``, ``u_s``,
``d_s``, and the ``*_dev`` / ``*_abs`` helpers.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from .discrete_linear_ocp import StandardLinearDiscreteOCP
from .qp_solver import QPSolverBackend

if TYPE_CHECKING:
    from ..models import DiscreteLinearisedSDE


class StandardLinearisedDiscreteOCP(StandardLinearDiscreteOCP):
    """
    Receding-horizon QP for a linearised discrete-time system, with
    automatic steady-state coordinate shifting.

    The QP is built and solved in deviation space; ``solve`` transparently
    converts incoming absolute coordinates to deviations, calls the parent
    solver, and converts the results back to absolute space.

    The model's system matrices (``Ad``, ``Bd``, ``Ed``, ``Cz``) already act
    on deviation variables, so no additional transformation of the matrices is
    required — only the initial condition, disturbance trajectory, reference,
    and previous input are shifted.

    Parameters
    ----------
    model : DiscreteLinearisedSDE
        Linearised discrete-time plant with steady-state operating point.
    N : int
        Prediction horizon (number of control intervals).
    Q : (nz, nz) array-like
        Stage output tracking cost  ‖δz − δz_ref‖²_Q  (deviation space).
    R : (nu, nu) array-like
        Stage input cost  ``‖u‖²_R``  on **absolute** inputs.  In the deviation
        QP this is implemented via :attr:`horizon_profile.input_equilibrium`
        (set automatically from ``u_s``).
    P : (nz, nz) array-like, optional
        Terminal output tracking cost.  Default: Q.
    S : (nu, nu) array-like, optional
        Input rate-of-movement cost  ‖Δ(δu)‖²_S.  ``None`` → disabled.
    rho : float, optional
        Penalty weight on soft output constraint violation.  Default: 1e4.
    y_offset : float, optional
        Symmetric half-width δ of the soft output constraint band.
        Default: 2.0.
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
            Previously-applied absolute input (used with rate-of-movement
            penalty).
        warm_start : dict, optional
            ``{"U": (N·nu,), "X": (N·nx,)}`` warm-start in **absolute** space
            (as returned by a previous ``solve`` call).

        Returns
        -------
        U : (N · nu,) ndarray — optimal absolute input sequence.
        X : (N · nx,) ndarray — predicted absolute state trajectory.
        """
        model = self._model
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
        saved_ue = self._horizon_profile.input_equilibrium
        self._horizon_profile.input_equilibrium = u_s
        try:
            U_dev, X_dev = super().solve(
                delta_x0, delta_D, delta_x_ref, delta_u_prev, ws_dev
            )
        finally:
            self._horizon_profile.input_equilibrium = saved_ue

        # Convert back to absolute space
        U_abs = (U_dev.reshape(N, nu) + u_s).reshape(-1)
        X_abs = (X_dev.reshape(N, nx) + x_s).reshape(-1)

        return U_abs, X_abs
