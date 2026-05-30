"""
Discrete-time Linear Kalman Filter (``DiscreteLinearKF``).

Implements the standard discrete-time Kalman filter for a linear
discrete-time stochastic system

    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
    ym[k]  = Cm x[k] + v[k],                            v[k] ~ N(0, Rm)

The filter notation matches the continuous-discrete state-estimation
documents (ControlToolbox §SDE / §SDAE state estimation): innovation
``e_k``, innovation covariance ``R_e``, Kalman gain ``K``, measurement
output matrix ``Cm``, process-noise covariance ``Qd``, measurement-noise
covariance ``Rm``.

Time update over ``[t_{k-1}, t_k]``
-----------------------------------
    x̂_{k|k-1}  = Ad x̂_{k-1|k-1} + Bd u[k−1] + Ed d[k−1] + offset(d)
    P_{k|k-1}  = Ad P_{k-1|k-1} Adᵀ + Gd Qd Gdᵀ

Inputs and disturbances are zero-order hold over each interval.

Measurement update at ``t_k`` (Joseph form)
-------------------------------------------
    ŷ^m_{k|k-1} = Cm x̂_{k|k-1}
    e_k         = ym_k − ŷ^m_{k|k-1}
    R_e         = Cm P_{k|k-1} Cmᵀ + Rm
    K_k         = P_{k|k-1} Cmᵀ R_e⁻¹

    x̂_{k|k}    = x̂_{k|k-1} + K_k e_k
    P_{k|k}    = (I − K_k Cm) P_{k|k-1} (I − K_k Cm)ᵀ + K_k Rm K_kᵀ

The Joseph stabilising form preserves symmetry and positive semi-definiteness
in finite-precision arithmetic.

Missing observations are handled by the optional ``mask`` argument of
:meth:`update` / :meth:`step`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from .._utils import _any_to_np1d, _any_to_np2d
from ._base import DiscreteEstimator, EstimatorParams

if TYPE_CHECKING:
    from ..models import DiscreteLinearSDE


# ── Parameter structure ───────────────────────────────────────────────────────


@dataclass
class DiscreteLinearKFParams(EstimatorParams):
    """
    Algorithm parameters for :class:`DiscreteLinearKF`.

    The discrete-time linear Kalman filter has no algorithm-specific
    hyper-parameters beyond the model and initial conditions supplied to the
    constructor; this class exists for consistency with the unified
    :class:`~._base.EstimatorParams` hierarchy.
    """


# ── Estimator ─────────────────────────────────────────────────────────────────


class DiscreteLinearKF(DiscreteEstimator):
    """
    Discrete-time Kalman filter with Joseph-stabilised covariance update.

    The filter reads ``Qd``, ``Rm`` and ``Gd`` directly from the supplied
    ``DiscreteLinearSDE`` model; no tuning parameters are required beyond
    the initial conditions.

    Parameters
    ----------
    model : DiscreteLinearSDE
        Plant model providing ``Ad``, ``Bd``, ``Ed``, ``Gd``, ``Cm``,
        ``Qd``, ``Rm`` and ``predict_offset``.
    x0 : (nx,) ndarray, optional
        Initial state estimate ``x̂_{0|0}``.  Defaults to ``np.zeros(nx)``.
    P0 : (nx, nx) ndarray, optional
        Initial state error covariance ``P_{0|0}``.  Defaults to ``I_{nx}``.
    params : DiscreteLinearKFParams, optional
        Algorithm parameter struct.  Accepted for API uniformity; unused.
    """

    def __init__(
        self,
        model: "DiscreteLinearSDE",
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        params: DiscreteLinearKFParams | None = None,
    ) -> None:
        self._model = model
        nx = model.nx

        self._x: np.ndarray = (
            np.asarray(x0, dtype=float).copy() if x0 is not None
            else np.zeros(nx)
        )
        self._P: np.ndarray = (
            _any_to_np2d(P0).copy() if P0 is not None
            else np.eye(nx)
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
        """
        Most recent innovation ``e_k = ym_k − Cm x̂_{k|k-1}`` as a plain
        Python list, or ``None`` before the first measurement update.
        """
        if self._last_innovation is None:
            return None
        return [float(v) for v in self._last_innovation]

    # ── Filter steps ─────────────────────────────────────────────────────────

    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update: propagate state and covariance one sampling interval.

            x̂_{k|k-1} = Ad x̂ + Bd u + Ed d + offset(d)
            P_{k|k-1} = Ad P Adᵀ + Gd Qd Gdᵀ

        Parameters
        ----------
        u : (nu,) ndarray  — input applied (ZOH) over the just-completed interval.
        d : (nd,) ndarray  — disturbance over the same interval.

        Returns
        -------
        x_pred : (nx,) predicted state estimate x̂_{k|k-1}.
        P_pred : (nx, nx) predicted covariance P_{k|k-1}.
        """
        model = self._model
        u_np = _any_to_np1d(u)
        d_np = _any_to_np1d(d)

        x_pred = (
            model.Ad @ self._x
            + model.Bd @ u_np
            + model.Ed @ d_np
            + model.predict_offset(d_np)
        )
        P_pred = model.Ad @ self._P @ model.Ad.T + model.Gd @ model.Qd @ model.Gd.T

        self._x = x_pred
        self._P = P_pred
        return x_pred.copy(), P_pred.copy()

    def update(
        self,
        ym: np.ndarray,
        mask: list[bool] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update (Joseph stabilising form).

            e_k    = ym_k − Cm x̂_{k|k-1}
            R_e    = Cm P_{k|k-1} Cmᵀ + Rm
            K_k    = P_{k|k-1} Cmᵀ R_e⁻¹
            x̂_{k|k} = x̂_{k|k-1} + K_k e_k
            P_{k|k} = (I − K_k Cm) P_{k|k-1} (I − K_k Cm)ᵀ + K_k Rm K_kᵀ

        Parameters
        ----------
        ym : (nym,) ndarray
            Measurement vector at time ``t_k``.
        mask : (nym,) bool ndarray or list, optional
            When provided only outputs where ``mask[i]`` is ``True`` are
            used.  If every entry is ``False`` the update is skipped.
            ``None`` (default) uses all outputs.

        Returns
        -------
        x_hat : (nx,) corrected state estimate x̂_{k|k}.
        P     : (nx, nx) corrected covariance P_{k|k}.
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
        p=None,
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
