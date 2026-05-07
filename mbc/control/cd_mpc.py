"""
Model Predictive Controller for linear continuous-discrete systems.

:class:`CDMPCController` composes a :class:`~mbc.estimation.CDKalmanFilter`
and a :class:`~mbc.control.CDOptimalControlProblem` and implements the
receding-horizon policy described in ControlToolbox §EMPC —
*ENMPC Algorithm*, specialised to the linear continuous-discrete case.

At each measurement time t_k:

    1. **Measure**   ym[k]                                  (passed to ``step``)
    2. **Estimate**  x̂[k|k] = κ(x̂[k-1|k-1], u[k-1], d[k-1], ym[k])
                                                            (estimator.step,
                                                             continuous ODE
                                                             integration)
    3. **Optimise**  U* = λ(x̂[k|k], …)                       (ocp.solve, ZOH-QP)
    4. **Apply**     u[k] = U*[0:nu]                          (returned to caller)

The estimator integrates the continuous-time matrices ``A``, ``B``, ``E``
directly via ODE integration; the OCP uses ZOH-discretised matrices
``(Ad, Bd, Ed)`` computed once at construction time inside
:class:`CDOptimalControlProblem`.
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import numpy as np
from cvxopt import matrix

from .._utils import _zeros, _np_to_cvx
from ..estimation.cd_kalman import CDKalmanFilter
from .cd_ocp import CDOptimalControlProblem

if TYPE_CHECKING:
    from ..models import LinearContinuousDiscreteModel


class CDMPCController:
    """
    MPC controller for a linear continuous-discrete plant.

    Composes a :class:`~mbc.estimation.CDKalmanFilter` and a
    :class:`~mbc.control.CDOptimalControlProblem` into a single
    receding-horizon controller.  The previously-applied ``(u, d)`` are
    tracked internally so that the estimator's predict step has the
    correct ZOH inputs over the just-completed interval.

    Parameters
    ----------
    model     : LinearContinuousDiscreteModel
        Plant model providing ``nu``, ``nd``, ``x_ref``, ``discretize``,
        and ``discretize_noise``.
    estimator : CDKalmanFilter
        State estimator (continuous ODE integration internally).
    ocp       : CDOptimalControlProblem
        Optimal control problem (lifted-batch QP on ZOH-discretised matrices).
    """

    def __init__(
        self,
        model: "LinearContinuousDiscreteModel",
        estimator: CDKalmanFilter,
        ocp: CDOptimalControlProblem,
    ) -> None:
        self._model = model
        self._estimator = estimator
        self._ocp = ocp
        self._u_prev_np: np.ndarray = np.zeros(model.nu)
        self._d_prev_np: np.ndarray = np.zeros(model.nd)

    def step(
        self,
        ym: matrix,
        D: matrix,
    ) -> Tuple[matrix, matrix, matrix]:
        """
        Execute one closed-loop CD-MPC step.

        Parameters
        ----------
        ym : (nym, 1) cvxopt column — measurement ``ym[k]``.
        D  : (N · nd, 1) cvxopt column — stacked disturbance forecast
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

        # Step 3: optimise
        x_hat_cvx = _np_to_cvx(x_hat_np.reshape(-1, 1))
        x_ref_cvx = _np_to_cvx(
            np.asarray(self._model.x_ref, dtype=float).reshape(-1, 1)
        )
        U_seq, X_seq = self._ocp.solve(
            x_hat_cvx, D, x_ref_cvx, u_prev=matrix(self._u_prev_np.tolist(), (nu, 1)),
        )

        # Step 4: cache (u_now, d_now) for next step
        u = U_seq[:nu]
        u_now_np = np.array(list(u), dtype=float)
        d_now_np = np.array(list(D[:nd]), dtype=float)
        self._u_prev_np = u_now_np
        self._d_prev_np = d_now_np

        return u, U_seq, X_seq
