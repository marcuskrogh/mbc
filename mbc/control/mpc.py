"""
Model Predictive Controller for linear discrete-time systems.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from .._utils import _any_to_np1d
from ..estimation import DiscreteLinearKF
from ._base import ModelPredictiveController
from .discrete_linear_ocp import StandardLinearDiscreteOCP, _shift_warm_start

if TYPE_CHECKING:
    from ..models import DiscreteLinearSDE


class LinearDiscreteMPC(ModelPredictiveController):
    """Abstract MPC for linear discrete-time plant + estimator + discrete OCP."""

    @abstractmethod
    def compute(
        self,
        ym: Any,
        D: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and return the optimal closed-loop MPC action."""

    @abstractmethod
    def propagate(
        self,
        ym: Any,
        D: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the estimator without solving the OCP; return ``(x_hat, P)``."""


class StandardLinearDiscreteMPC(LinearDiscreteMPC):
    """
    Standard MPC for a linear discrete-time plant.

    Combines :class:`~mbc.estimation.DiscreteLinearKF` and
    :class:`StandardLinearDiscreteOCP`.  Horizon-varying quantities are
    configured on the controller (shared with the OCP) before :meth:`compute`.
    """

    def __init__(
        self,
        model: "DiscreteLinearSDE",
        estimator: DiscreteLinearKF,
        ocp: StandardLinearDiscreteOCP,
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

    def compute(
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

    def propagate(
        self,
        ym: Any,
        D: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the estimator without solving the OCP; return ``(x_hat, P)``.

        Use this when the controller is switched off but state tracking must
        continue.  The last applied input is held constant for the prediction
        step; no new control action is produced and warm-start sequences are
        not updated.

        Parameters
        ----------
        ym : (nym,) array-like   — current measurement.
        D  : array-like, optional — disturbance; updates the stored disturbance
             for the *next* call if provided.

        Returns
        -------
        x_hat : (nx,) filtered state estimate.
        P     : (nx, nx) state error covariance.
        """
        nd = self._model.nd
        ym_np = _any_to_np1d(ym)
        x_hat, P = self._estimator.step(ym_np, self._u_prev_np, self._d_prev_np)
        if D is not None:
            self._d_prev_np = _any_to_np1d(D)[:nd].copy()
        return np.asarray(x_hat, dtype=float), P
