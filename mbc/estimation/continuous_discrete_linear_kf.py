"""
Continuous-Discrete Linear Kalman Filter (``ContinuousDiscreteLinearKF``).

Linear specialisation of the continuous-discrete EKF (ControlToolbox §SDE —
*CD-EKF*) where ``A`` and ``Cm`` are the constant drift and measurement
Jacobians.

Model
-----
    dx(t)  = (A x + B u + E d) dt + G dw(t),     dw ~ N(0, I dt)
    ym[k]  = Cm x[k] + v[k],                       v ~ N(0, Rm)

Time update over ``[t_{k-1}, t_k]``
-----------------------------------
Forward-Euler integration of the state ODE and Lyapunov-type covariance ODE
with ``n_steps`` sub-steps of size ``h = Ts / n_steps``:

    dx̂/dt(t) = A x̂(t) + B u + E d
    dP/dt(t) = A P(t) + P(t) Aᵀ + G Gᵀ

Inputs and disturbances are zero-order hold over each sampling interval.

Measurement update at ``t_k`` (Joseph form)
-------------------------------------------
    e_k = ym_k − Cm x̂_{k|k-1}
    R_e = Cm P_{k|k-1} Cmᵀ + Rm
    K_k = P_{k|k-1} Cmᵀ R_e⁻¹

    x̂_{k|k} = x̂_{k|k-1} + K_k e_k
    P_{k|k} = (I − K_k Cm) P_{k|k-1} (I − K_k Cm)ᵀ + K_k Rm K_kᵀ

Missing observations are handled by the optional ``mask`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from .._utils import _any_to_np1d, _any_to_np2d
from ._base import ContinuousDiscreteEstimator, EstimatorParams

if TYPE_CHECKING:
    from ..models import ContinuousDiscreteLinearSDE


# ── Parameter structure ───────────────────────────────────────────────────────


@dataclass
class ContinuousDiscreteLinearKFParams(EstimatorParams):
    """
    Algorithm parameters for :class:`ContinuousDiscreteLinearKF`.

    Parameters
    ----------
    n_steps : int
        Number of Forward-Euler sub-steps per sampling interval.  Default: 10.
    """
    n_steps: int = 10


# ── Estimator ─────────────────────────────────────────────────────────────────


class ContinuousDiscreteLinearKF(ContinuousDiscreteEstimator):
    """
    Continuous-discrete Kalman filter for a linear continuous-discrete
    plant (linear specialisation of :class:`~.continuous_discrete_ekf.ContinuousDiscreteEKF`).

    The filter integrates the state ODE and Lyapunov-type covariance ODE
    continuously over each sampling interval using Forward-Euler sub-steps;
    ``Rm`` is read directly from ``model.Rm``.

    Parameters
    ----------
    model : ContinuousDiscreteLinearSDE
        Linear continuous-discrete plant providing ``A``, ``B``, ``E``,
        ``G``, ``Cm``, ``Rm``, ``Ts``, ``nx``, ``nu``, ``nd``.
    x0 : (nx,) ndarray, optional
        Initial state estimate ``x̂_{0|0}``.  Defaults to ``np.zeros(nx)``.
    P0 : (nx, nx) ndarray, optional
        Initial state error covariance ``P_{0|0}``.  Defaults to ``I_{nx}``.
    params : ContinuousDiscreteLinearKFParams, optional
        Algorithm parameter struct.  Pass to control ``n_steps``.
    """

    def __init__(
        self,
        model: "ContinuousDiscreteLinearSDE",
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        params: ContinuousDiscreteLinearKFParams | None = None,
    ) -> None:
        if params is None:
            params = ContinuousDiscreteLinearKFParams()

        self._model = model
        nx = model.nx

        # Cache continuous-time matrices; G Gᵀ is the noise intensity.
        self._A_c: np.ndarray = np.asarray(model.A, dtype=float)
        self._B_c: np.ndarray = np.asarray(model.B, dtype=float)
        self._E_c: np.ndarray = np.asarray(model.E, dtype=float)
        G = np.asarray(model.G, dtype=float)
        self._GGT: np.ndarray = G @ G.T

        self._Ts: float = float(model.Ts)
        self._n_steps: int = int(params.n_steps)
        self._h: float = self._Ts / self._n_steps

        self._x: np.ndarray = (
            np.asarray(x0, dtype=float).copy() if x0 is not None
            else np.zeros(nx)
        )
        self._P: np.ndarray = (
            _any_to_np2d(P0).copy() if P0 is not None else np.eye(nx)
        )
        self._last_innovation: Optional[np.ndarray] = None

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def x_hat(self) -> np.ndarray:
        """Current state estimate x̂ ∈ ℝⁿˣ (copy)."""
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        """Current state error covariance P ∈ ℝⁿˣˣⁿˣ (copy)."""
        return self._P.copy()

    @property
    def last_innovation(self) -> Optional[List[float]]:
        """Most recent innovation ``e_k = ym_k − Cm x̂_{k|k-1}``."""
        if self._last_innovation is None:
            return None
        return [float(v) for v in self._last_innovation]

    # ── Filter steps ─────────────────────────────────────────────────────────

    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p=None,
        t: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update — Forward-Euler integration of the state and
        Lyapunov-type covariance ODEs over one sampling interval (ZOH).

            dx̂/dt = A x̂ + B u + E d
            dP/dt = A P + P Aᵀ + G Gᵀ

        Parameters
        ----------
        u : (nu,) ndarray  — input applied (ZOH) over the just-completed interval.
        d : (nd,) ndarray  — disturbance over the same interval.
        p : ignored         — accepted for interface uniformity.
        t : ignored         — accepted for interface uniformity (LTI).

        Returns
        -------
        x_pred : (nx,) predicted state estimate x̂_{k|k-1}.
        P_pred : (nx, nx) predicted covariance P_{k|k-1}.
        """
        u_np = _any_to_np1d(u)
        d_np = _any_to_np1d(d)

        x = self._x.copy()
        P = self._P.copy()
        h = self._h
        A = self._A_c
        Bu = self._B_c @ u_np
        Ed = self._E_c @ d_np
        GGT = self._GGT

        for _ in range(self._n_steps):
            x_dot = A @ x + Bu + Ed
            P_dot = A @ P + P @ A.T + GGT
            x = x + h * x_dot
            P = P + h * P_dot
        P = 0.5 * (P + P.T)

        self._x = x
        self._P = P
        return x.copy(), P.copy()

    def update(
        self,
        ym: np.ndarray,
        u: np.ndarray | None = None,
        d: np.ndarray | None = None,
        p: np.ndarray | None = None,
        mask: list[bool] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update at ``t_k`` (Joseph form).

        Parameters
        ----------
        ym : (nym,) ndarray
            Measurement at time ``t_k``.
        u, d, p : ignored — accepted for interface uniformity (linear model
            has no direct feedthrough in the measurement equation).
        mask : (nym,) bool ndarray or list, optional
            Active-channel mask; see :meth:`DiscreteLinearKF.update`.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.
        """
        model = self._model
        nx = model.nx
        Cm = model.Cm
        Rm = model.Rm
        ym_np = _any_to_np1d(ym)

        if mask is not None:
            active = np.where(np.asarray(mask, dtype=bool))[0]
            if len(active) == 0:
                return self._x.copy(), self._P.copy()
            Cm = Cm[active, :]
            Rm = Rm[np.ix_(active, active)]
            ym_np = ym_np[active]

        e = ym_np - Cm @ self._x
        R_e = Cm @ self._P @ Cm.T + Rm

        Kt = np.linalg.solve(R_e, Cm @ self._P)
        K = Kt.T

        x_new = self._x + K @ e

        IKC = np.eye(nx) - K @ Cm
        P_new = IKC @ self._P @ IKC.T + K @ Rm @ K.T
        P_new = 0.5 * (P_new + P_new.T)

        self._last_innovation = e.copy()
        self._x = x_new
        self._P = P_new
        return x_new.copy(), P_new.copy()

    def step(
        self,
        ym: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None = None,
        t: float | None = None,
        mask: list[bool] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combined time + measurement update.

        Parameters
        ----------
        ym : (nym,) ndarray  — measurement at time ``t_k``.
        u  : (nu,) ndarray   — input applied over the previous interval.
        d  : (nd,) ndarray   — disturbance over the previous interval.
        p  : ignored          — accepted for interface uniformity.
        t  : ignored          — accepted for interface uniformity.
        mask : (nym,) bool ndarray, optional — see :meth:`update`.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.
        """
        self.predict(u, d)
        return self.update(ym, mask=mask)
