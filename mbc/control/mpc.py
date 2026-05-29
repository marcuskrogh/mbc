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

from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from .._utils import _any_to_np1d
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
        ym: Any,
        D: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute one closed-loop MPC step.

        Parameters
        ----------
        ym : (nym,) array-like  — measurement ``ym[k]``.
        D  : (N · nd,) array-like  — stacked disturbance forecast
             ``[d[k]; d[k+1]; …; d[k + N − 1]]``.

        Returns
        -------
        u     : (nu,) ndarray — optimal input ``u_k``.
        U_seq : (N · nu,) ndarray — full optimal input sequence.
        X_seq : (N · nx,) ndarray — predicted state trajectory.
        """
        nu = self._model.nu
        nd = self._model.nd

        # Step 2: estimate using the previously-applied (u, d)
        ym_np = _any_to_np1d(ym)
        x_hat_np, _ = self._estimator.step(
            ym_np, self._u_prev_np, self._d_prev_np,
        )

        # Step 3: optimise (OCP returns numpy 1-D arrays)
        D_np = _any_to_np1d(D)
        x_ref_np = np.asarray(self._model.x_ref, dtype=float).reshape(-1)
        U_seq, X_seq = self._ocp.solve(
            x_hat_np, D_np, x_ref_np, u_prev=self._u_prev_np,
        )

        # Step 4: extract the first action; cache (u_now, d_now) for next step
        u = U_seq[:nu]
        self._u_prev_np = np.asarray(u, dtype=float).copy()
        self._d_prev_np = D_np[:nd].copy()

        return u, U_seq, X_seq
