"""
Continuous-Discrete Extended Kalman Filter (``ContinuousDiscreteEKF``).

(ControlToolbox §State Estimation for Nonlinear SDE Systems —
*Continuous-Discrete Extended Kalman Filter*)

Model
-----
    dx(t)  = f(x, u, d, p, t) dt + sigma(x, u, d, p, t) dw(t),  dw ~ N(0, I dt)
    ym(tk) = hm(x(tk), u, d, p) + v(tk),                         v  ~ N(0, Rm)

Time update over ``[t_k, t_{k+1}]``
------------------------------------
The mean evolves as the expectation of the SDE (diffusion contributes zero mean):

    dx̂_k/dt(t) = f(x̂_k(t), u, d, p, t),    x̂_k(t_k) = x̂_{k|k}.

Both mean and covariance are integrated with ``n_steps`` sub-steps of size
``h = Ts / n_steps``.  Two schemes are available:

Explicit Euler  (``scheme="euler"``, default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A_j     = ∂f/∂x (x̂_j, u, d, p, t_j)
    σ_j     = sigma (x̂_j, u, d, p, t_j)
    x̂_{j+1} = x̂_j + h f(x̂_j, u, d, p, t_j)
    P_{j+1} = P_j + h (A_j P_j + P_j A_jᵀ + σ_j σ_jᵀ)
    P_{j+1} ← ½(P_{j+1} + P_{j+1}ᵀ)                     (symmetrise)

Implicit Euler  (``scheme="implicit-euler"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
L-stable; suitable for stiff drift dynamics.

  1. Newton solve:  x_{j+1} − x_j − h f(x_{j+1}, u, d, p, t_{j+1}) = 0
  2. Sensitivity:   Φ = (I − h A_{j+1})⁻¹
  3. Covariance:    τ = P_j + h σ_j σ_jᵀ
                    P_{j+1} = Φ τ Φᵀ
                    P_{j+1} ← ½(P_{j+1} + P_{j+1}ᵀ)      (symmetrise)

Measurement update at ``t_k`` (Joseph form)
-------------------------------------------
    e_k   = ym_k − hm(x̂_{k|k-1}, u, d, p)
    C_k   = ∂hm/∂x (x̂_{k|k-1}, u, d, p)
    R_e,k = C_k P_{k|k-1} C_kᵀ + Rm
    K_k   = P_{k|k-1} C_kᵀ R_e,k⁻¹

    x̂_{k|k} = x̂_{k|k-1} + K_k e_k
    P_{k|k} = (I − K_k C_k) P_{k|k-1} (I − K_k C_k)ᵀ + K_k Rm K_kᵀ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import ContinuousDiscreteSDE
from .._utils import _newton_solve
from ._base import ContinuousDiscreteEstimator, EstimatorParams

_VALID_SCHEMES = ("euler", "implicit-euler")


# ── Parameter structure ───────────────────────────────────────────────────────


@dataclass
class ContinuousDiscreteEKFParams(EstimatorParams):
    """
    Algorithm parameters for :class:`ContinuousDiscreteEKF`.

    Parameters
    ----------
    n_steps : int
        Number of integration sub-steps per sampling interval.  Default: 10.
    scheme : {"euler", "implicit-euler"}
        Propagation scheme.  ``"euler"`` is explicit and cheap; use
        ``"implicit-euler"`` for stiff drift dynamics.  Default: ``"euler"``.
    newton_tol : float
        Convergence tolerance for the implicit-Euler Newton solver.
        Ignored when ``scheme="euler"``.  Default: 1e-10.
    newton_max_iter : int
        Maximum Newton iterations per implicit sub-step.
        Ignored when ``scheme="euler"``.  Default: 50.
    """
    n_steps: int = 10
    scheme: str = "euler"
    newton_tol: float = 1e-10
    newton_max_iter: int = 50


# ── Estimator ─────────────────────────────────────────────────────────────────


class ContinuousDiscreteEKF(ContinuousDiscreteEstimator):
    """
    Continuous-Discrete Extended Kalman Filter for SDE systems
    (ControlToolbox §SDE State Estimation — *CD-EKF*).

    Parameters
    ----------
    model : ContinuousDiscreteSDE
        Nonlinear continuous-discrete SDE system providing ``f``, ``sigma``,
        ``hm``, ``Rm``, and Jacobians ``dfdx`` and ``dhmdx``.  Must expose a
        ``Ts`` property giving the measurement sampling interval (seconds).
    x0 : (nx,) ndarray
        Initial state estimate x̂_{0|0}.
    P0 : (nx, nx) ndarray
        Initial state covariance P_{0|0}.
    params : ContinuousDiscreteEKFParams, optional
        Algorithm parameter struct.  Pass to control ``n_steps``, ``scheme``,
        and Newton solver settings.
    """

    def __init__(
        self,
        model: ContinuousDiscreteSDE,
        x0: np.ndarray,
        P0: np.ndarray,
        params: ContinuousDiscreteEKFParams | None = None,
    ) -> None:
        if params is None:
            params = ContinuousDiscreteEKFParams()
        if params.n_steps < 1:
            raise ValueError(
                f"n_steps must be a positive integer, got {params.n_steps!r}."
            )
        if params.scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"scheme must be one of {_VALID_SCHEMES!r}, got {params.scheme!r}."
            )

        self._model = model
        self._x: np.ndarray = np.array(x0, dtype=float)
        self._P: np.ndarray = np.array(P0, dtype=float)
        self._Ts: float = float(model.Ts)
        self._n_steps: int = int(params.n_steps)
        self._h: float = self._Ts / self._n_steps
        self._scheme: str = params.scheme
        self._newton_tol: float = params.newton_tol
        self._newton_max_iter: int = params.newton_max_iter

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def x_hat(self) -> np.ndarray:
        """Current state estimate x̂ ∈ ℝⁿˣ (copy)."""
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        """Current state covariance P ∈ ℝⁿˣˣⁿˣ (copy)."""
        return self._P.copy()

    # ── Filter steps ──────────────────────────────────────────────────────

    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update: integrate the mean ODE and covariance from ``t`` to
        ``t + Ts`` using the configured ``scheme``.

        Parameters
        ----------
        u : (nu,) ndarray       — input applied (ZOH) over [t, t+Ts].
        d : (nd,) ndarray       — disturbance over [t, t+Ts].
        p : (np,) ndarray       — parameter vector.
        t : float               — current time.

        Returns
        -------
        x_pred : (nx,) predicted state estimate.
        P_pred : (nx, nx) predicted covariance.
        """
        if self._scheme == "euler":
            return self._predict_euler(u, d, p, t)
        return self._predict_implicit_euler(u, d, p, t)

    def _predict_euler(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self._x.copy()
        P = self._P.copy()
        h = self._h
        model = self._model

        t_j = t
        for _ in range(self._n_steps):
            A_j = model.dfdx(x, u, d, p, t_j)
            sigma_j = model.sigma(x, u, d, p, t_j)
            f_j = model.f(x, u, d, p, t_j)

            P_dot = A_j @ P + P @ A_j.T + sigma_j @ sigma_j.T
            x = x + h * f_j
            P = P + h * P_dot
            P = (P + P.T) * 0.5
            t_j += h

        self._x = x
        self._P = P
        return x.copy(), P.copy()

    def _predict_implicit_euler(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self._x.copy()
        P = self._P.copy()
        h = self._h
        model = self._model
        nx = x.shape[0]
        I_nx = np.eye(nx)

        t_n = t
        for _ in range(self._n_steps):
            t_next = t_n + h
            x_n = x.copy()
            sigma_n = model.sigma(x_n, u, d, p, t_n)

            x_rhs = x_n

            def residual(xk: np.ndarray) -> np.ndarray:
                return xk - x_rhs - h * model.f(xk, u, d, p, t_next)

            def jacobian(xk: np.ndarray) -> np.ndarray:
                return I_nx - h * model.dfdx(xk, u, d, p, t_next)

            x = _newton_solve(
                residual, jacobian, x_n,
                tol=self._newton_tol,
                max_iter=self._newton_max_iter,
            )

            M = jacobian(x)
            Phi = np.linalg.solve(M, I_nx)

            tau = P + h * sigma_n @ sigma_n.T
            P = Phi @ tau @ Phi.T
            P = (P + P.T) * 0.5

            t_n = t_next

        self._x = x
        self._P = P
        return x.copy(), P.copy()

    def update(
        self,
        ym: np.ndarray,
        u: np.ndarray | None,
        d: np.ndarray | None,
        p: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update (Joseph stabilising form).

        Parameters
        ----------
        ym   : (nym,) ndarray  — measurement vector.
        u    : (nu,) ndarray   — input at measurement time.
        d    : (nd,) ndarray   — disturbance at measurement time.
        p    : (np,) ndarray   — parameter vector.
        mask : (nym,) bool ndarray, optional — active-channel mask.

        Returns
        -------
        x_hat : (nx,) corrected state estimate x̂_{k|k}.
        P     : (nx, nx) corrected covariance P_{k|k}.
        """
        x = self._x
        P = self._P
        nx = x.shape[0]
        R = self._model.Rm

        C = self._model.dhmdx(x, u, d, p, 0.0)
        y_hat = self._model.hm(x, u, d, p, 0.0)

        if mask is not None:
            active = np.where(mask)[0]
            if len(active) == 0:
                return x.copy(), P.copy()
            C = C[active, :]
            y_hat = y_hat[active]
            y_sub = ym[active]
            R_sub = R[np.ix_(active, active)]
        else:
            y_sub = ym
            R_sub = R

        R_e = C @ P @ C.T + R_sub
        Kt = np.linalg.solve(R_e, C @ P)
        K = Kt.T

        e = y_sub - y_hat
        x_new = x + K @ e

        IKC = np.eye(nx) - K @ C
        P_new = IKC @ P @ IKC.T + K @ R_sub @ K.T
        P_new = (P_new + P_new.T) * 0.5

        self._x = x_new
        self._P = P_new
        return x_new.copy(), P_new.copy()

    def step(
        self,
        ym: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combined time + measurement update."""
        self.predict(u, d, p, t)
        return self.update(ym, u, d, p, mask=mask)
