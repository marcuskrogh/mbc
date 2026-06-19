"""
Successive-linearisation MPC for nonlinear discrete-time models.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, TYPE_CHECKING

import numpy as np

from .._utils import _fd_jacobian
from ..estimation import DiscreteLinearKF
from ._base import ModelPredictiveController
from .discrete_linear_ocp import StandardLinearDiscreteOCP

if TYPE_CHECKING:
    pass


def linearize_discrete_model(
    model: Any,
    x_ss: np.ndarray,
    u_ss: np.ndarray,
    d_ss: np.ndarray,
) -> dict[str, np.ndarray]:
    """Linearise a nonlinear discrete-time model at (x_ss, u_ss, d_ss)."""
    x_ss = np.asarray(x_ss, dtype=float)
    u_ss = np.asarray(u_ss, dtype=float)
    d_ss = np.asarray(d_ss, dtype=float)

    Ad = _fd_jacobian(lambda x: model.f(x, u_ss, d_ss), x_ss)
    Bd = _fd_jacobian(lambda u: model.f(x_ss, u, d_ss), u_ss)
    if model.nd > 0:
        Ed = _fd_jacobian(lambda d: model.f(x_ss, u_ss, d), d_ss)
    else:
        Ed = np.zeros((model.nx, 0))
    Cm = _fd_jacobian(lambda x: model.hm(x, u_ss, d_ss), x_ss)
    Cz = _fd_jacobian(lambda x: model.gm(x, u_ss, d_ss), x_ss)

    return {"Ad": Ad, "Bd": Bd, "Ed": Ed, "Cm": Cm, "Cz": Cz}


class _MutableDiscreteLinearSDE:
    """Minimal mutable discrete linear model for deviation MPC."""

    def __init__(self, nx: int, nu: int, nd: int, u_min: np.ndarray, u_max: np.ndarray) -> None:
        self._nx, self._nu, self._nd = nx, nu, nd
        self._Ad = np.eye(nx)
        self._Bd = np.zeros((nx, nu))
        self._Ed = np.zeros((nx, nd))
        self._Cm = np.zeros((1, nx))
        self._Cz = np.zeros((1, nx))
        self._u_min = np.asarray(u_min, dtype=float)
        self._u_max = np.asarray(u_max, dtype=float)
        self._x_ref = np.zeros(nx)
        self.x_ss = np.zeros(nx)
        self.u_ss = np.zeros(nu)
        self.d_ss = np.zeros(nd)

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._nu

    @property
    def nd(self) -> int:
        return self._nd

    @property
    def Ad(self) -> np.ndarray:
        return self._Ad

    @property
    def Bd(self) -> np.ndarray:
        return self._Bd

    @property
    def Ed(self) -> np.ndarray:
        return self._Ed

    @property
    def Cm(self) -> np.ndarray:
        return self._Cm

    @property
    def Cz(self) -> np.ndarray:
        return self._Cz

    @property
    def x_ref(self) -> np.ndarray:
        return self._x_ref

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._u_min - self.u_ss, self._u_max - self.u_ss


class LinearisedDiscreteMPC(ModelPredictiveController):
    """
    Abstract MPC with mixed nonlinear estimation and linearised control.

    State estimation runs on the full nonlinear discrete-time plant; the OCP
    linearises at each sample and solves a discrete-time QP in deviation
    coordinates.
    """

    @abstractmethod
    def compute(self, ym: Any, d: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and return the optimal closed-loop MPC action."""

    @abstractmethod
    def propagate(self, ym: Any, d: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run the estimator without solving the OCP; return ``(x_hat, P)``."""


class StandardLinearisedDiscreteMPC(LinearisedDiscreteMPC):
    """Standard linearised discrete-time MPC using :class:`StandardLinearDiscreteOCP`."""

    def __init__(
        self,
        model: Any,
        estimator: DiscreteLinearKF,
        N: int,
        Q: Any,
        R: Any,
        u_min: np.ndarray,
        u_max: np.ndarray,
        x_ref: np.ndarray | None = None,
        **ocp_kwargs: Any,
    ) -> None:
        super().__init__()
        self._model = model
        self._estimator = estimator
        self._N = int(N)
        self._x_ref = np.zeros(model.nx) if x_ref is None else np.asarray(x_ref, dtype=float)
        self._lin = _MutableDiscreteLinearSDE(model.nx, model.nu, model.nd, u_min, u_max)
        self._ocp = StandardLinearDiscreteOCP(model=self._lin, N=N, Q=Q, R=R, **ocp_kwargs)
        self._bind_ocp(self._ocp)
        self._u_prev = np.zeros(model.nu)
        self._d_prev = np.zeros(model.nd)

    def compute(self, ym: Any, d: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_now = (
            np.zeros(self._model.nd) if d is None
            else np.asarray(d, dtype=float).reshape(self._model.nd)
        )
        x_hat, _ = self._estimator.step(ym, self._u_prev, self._d_prev)
        x_hat = np.asarray(x_hat, dtype=float).reshape(self._model.nx)

        lp = self._horizon_profile.linearisation_point
        x_ss = lp.x if lp is not None else x_hat
        u_ss = lp.u if lp is not None else self._u_prev
        d_ss = lp.d if lp is not None else d_now

        lin = linearize_discrete_model(self._model, x_ss, u_ss, d_ss)
        self._lin._Ad = lin["Ad"]
        self._lin._Bd = lin["Bd"]
        self._lin._Ed = lin["Ed"]
        self._lin._Cm = lin["Cm"]
        self._lin._Cz = lin["Cz"]
        self._lin.x_ss = x_ss
        self._lin.u_ss = u_ss
        self._lin.d_ss = d_ss
        self._lin._x_ref = self._x_ref - x_ss

        self.set_disturbance_profile(np.zeros(self._N * self._model.nd))
        saved_ue = self._horizon_profile.input_equilibrium
        self._horizon_profile.input_equilibrium = u_ss
        try:
            U_dev, X_dev = self._ocp.solve(
                np.zeros(self._model.nx), u_prev=self._u_prev - u_ss,
            )
        finally:
            self._horizon_profile.input_equilibrium = saved_ue
        U = U_dev.reshape(self._N, self._model.nu) + u_ss
        X = X_dev.reshape(self._N, self._model.nx) + x_ss
        u = U[0].copy()
        self._u_prev = u
        self._d_prev = d_now
        return u, U.reshape(-1), X.reshape(-1)

    def propagate(self, ym: Any, d: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the estimator without solving the OCP; return ``(x_hat, P)``.

        Use this when the controller is switched off but state tracking must
        continue.  The last applied input is held constant for the prediction
        step; no new control action is produced.

        Parameters
        ----------
        ym : (nym,) array-like   — current measurement.
        d  : (nd,) array-like, optional — current disturbance; updates the
             stored disturbance for the next call if provided.

        Returns
        -------
        x_hat : (nx,) filtered state estimate.
        P     : (nx, nx) state error covariance.
        """
        d_now = (
            np.zeros(self._model.nd) if d is None
            else np.asarray(d, dtype=float).reshape(self._model.nd)
        )
        x_hat, P = self._estimator.step(ym, self._u_prev, self._d_prev)
        self._d_prev = d_now.copy()
        return np.asarray(x_hat, dtype=float), P
