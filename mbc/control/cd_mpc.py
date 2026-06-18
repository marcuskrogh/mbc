"""
Model Predictive Controller for linear continuous-discrete systems.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from .._utils import _any_to_np1d
from ..estimation.continuous_discrete_linear_kf import ContinuousDiscreteLinearKF
from ._base import ModelPredictiveController
from .continuous_linear_ocp import StandardLinearContinuousDiscreteOCP
from .discrete_linear_ocp import _shift_warm_start

if TYPE_CHECKING:
    from ..models import ContinuousDiscreteLinearSDE


class LinearContinuousMPC(ModelPredictiveController):
    """Abstract MPC for linear CD plant + CD estimator + discrete-time OCP."""

    @abstractmethod
    def step(
        self,
        ym: Any,
        D: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute one closed-loop MPC step."""


class StandardLinearContinuousMPC(LinearContinuousMPC):
    """Standard MPC for linear continuous-discrete plants."""

    def __init__(
        self,
        model: "ContinuousDiscreteLinearSDE",
        estimator: ContinuousDiscreteLinearKF,
        ocp: StandardLinearContinuousDiscreteOCP,
        warm_start: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self._estimator = estimator
        self._ocp = ocp
        self._bind_ocp(ocp)
        self._warm_start = bool(warm_start)
        self._u_prev_np: np.ndarray = np.zeros(model.nu)
        self._d_prev_np: np.ndarray = np.zeros(model.nd)
        self._prev_U: np.ndarray | None = None
        self._prev_X: np.ndarray | None = None

    def step(
        self,
        ym: Any,
        D: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nu = self._model.nu
        nd = self._model.nd

        ym_np = _any_to_np1d(ym)
        x_hat_np, _ = self._estimator.step(
            ym_np, self._u_prev_np, self._d_prev_np,
        )

        if D is not None:
            self.set_disturbance_profile(np.asarray(D, dtype=float))

        warm = None
        if self._warm_start and self._prev_U is not None:
            warm = _shift_warm_start(
                self._prev_U, self._prev_X, nu, self._model.nx
            )

        U_seq, X_seq = self._ocp.solve(
            x_hat_np, u_prev=self._u_prev_np, warm_start=warm,
        )

        u = U_seq[:nu]
        D_np = self._horizon_profile.disturbance_profile
        self._u_prev_np = np.asarray(u, dtype=float).copy()
        self._d_prev_np = _any_to_np1d(D_np)[:nd].copy()
        self._prev_U, self._prev_X = U_seq, X_seq

        return u, U_seq, X_seq
