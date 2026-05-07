"""
Model Predictive Controller for linear discrete-time systems.

Composes a :class:`~mbc.estimation.KalmanFilter` and an
:class:`OptimalControlProblem` for any
:class:`~mbc.models.LinearDiscreteModel` and implements the receding-horizon
policy described in ControlToolbox §EMPC — *ENMPC Algorithm*, specialised
to the linear case.

At each measurement time t_k:

    1. **Measure**   ym[k]                                  (passed to ``step``)
    2. **Estimate**  x̂[k|k] = κ(x̂[k-1|k-1], u[k-1], d[k-1], ym[k])
                                                            (estimator.step)
    3. **Optimise**  U* = λ(x̂[k|k], …)                       (ocp.solve)
    4. **Apply**     u[k] = U*[0:nu]                          (returned to caller)
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import numpy as np
from cvxopt import matrix

from .._utils import _zeros, _np_to_cvx
from ..estimation import KalmanFilter
from .ocp import OptimalControlProblem

if TYPE_CHECKING:
    from ..models import LinearDiscreteModel


class MPCController:
    """
    Model predictive controller for a linear discrete-time plant.

    Combines a :class:`~mbc.estimation.KalmanFilter` (estimator) and an
    :class:`OptimalControlProblem` (OCP) into a closed-loop controller.

    The previously-applied input ``u_{k-1}`` and disturbance ``d_{k-1}``
    are tracked internally so that the estimator's predict step has the
    correct ZOH inputs over the just-completed interval.

    Parameters
    ----------
    model     : LinearDiscreteModel
    estimator : KalmanFilter
    ocp       : OptimalControlProblem
    """

    def __init__(
        self,
        model: "LinearDiscreteModel",
        estimator: KalmanFilter,
        ocp: OptimalControlProblem,
    ) -> None:
        self._model = model
        self._estimator = estimator
        self._ocp = ocp
        # Previous applied (u, d) — used by the estimator's predict step.
        self._u_prev_np: np.ndarray = np.zeros(model.nu)
        self._d_prev_np: np.ndarray = np.zeros(model.nd)

    def step(
        self,
        ym: matrix,
        D: matrix,
    ) -> Tuple[matrix, matrix, matrix]:
        """
        Execute one closed-loop MPC step.

        Parameters
        ----------
        ym : (nym, 1) cvxopt column  — measurement ``ym[k]``.
        D  : (N · nd, 1) cvxopt column  — stacked disturbance forecast
             ``[d[k]; d[k+1]; …; d[k + N − 1]]``.

        Returns
        -------
        u     : (nu, 1) cvxopt column — optimal input ``u_k``.
        U_seq : (N · nu, 1) full optimal input sequence.
        X_seq : (N · nx, 1) predicted state trajectory.
        """
        nu = self._model.nu
        nd = self._model.nd

        # Step 2: estimate using the previously-applied (u, d)
        ym_np = np.array(list(ym), dtype=float)
        x_hat_np, _ = self._estimator.step(
            ym_np, self._u_prev_np, self._d_prev_np,
        )

        # Step 3: optimise (OCP returns cvxopt columns)
        x_hat_cvx = _np_to_cvx(x_hat_np.reshape(-1, 1))
        x_ref_cvx = _np_to_cvx(
            np.asarray(self._model.x_ref, dtype=float).reshape(-1, 1)
        )
        U_seq, X_seq = self._ocp.solve(
            x_hat_cvx, D, x_ref_cvx, u_prev=matrix(self._u_prev_np.tolist(), (nu, 1)),
        )

        # Step 4: extract the first action; cache (u_now, d_now) for next step
        u = U_seq[:nu]
        u_now_np = np.array(list(u), dtype=float)
        d_now_np = np.array(list(D[:nd]), dtype=float)
        self._u_prev_np = u_now_np
        self._d_prev_np = d_now_np

        return u, U_seq, X_seq
