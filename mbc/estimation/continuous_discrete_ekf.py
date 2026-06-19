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
``h = Ts / n_steps``.  Two schemes are available, selected via
:class:`~mbc.estimation.IntegrationScheme`:

Explicit Euler  (``IntegrationScheme.EXPLICIT_EULER``, default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A_j     = ∂f/∂x (x̂_j, u, d, p, t_j)
    σ_j     = sigma (x̂_j, u, d, p, t_j)
    x̂_{j+1} = x̂_j + h f(x̂_j, u, d, p, t_j)
    P_{j+1} = P_j + h (A_j P_j + P_j A_jᵀ + σ_j σ_jᵀ)
    P_{j+1} ← ½(P_{j+1} + P_{j+1}ᵀ)                     (symmetrise)

Implicit Euler  (``IntegrationScheme.IMPLICIT_EULER``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
from ._base import ContinuousDiscreteEstimator, EstimatorParams, IntegrationScheme


# ── Moment-propagation step kernels ───────────────────────────────────────────


class _EulerMomentStep:
    """
    Explicit Euler sub-step for the EKF moment ODE ``(x̂, P)``.

    Evaluates ``f``, ``dfdx``, ``sigma`` at the current sub-step and
    advances both state mean and covariance by one forward Euler step.
    """

    def __init__(self, model: ContinuousDiscreteSDE) -> None:
        self._m = model

    def __call__(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        A = self._m.dfdx(x, u, d, p, t)
        S = self._m.sigma(x, u, d, p, t)
        x_next = x + h * self._m.f(x, u, d, p, t)
        P_dot = A @ P + P @ A.T + S @ S.T
        P_next = P + h * P_dot
        P_next = (P_next + P_next.T) * 0.5
        return x_next, P_next


class _ImplicitEulerMomentStep:
    """
    Implicit-Euler sub-step for the EKF moment ODE ``(x̂, P)``.

    The state mean is advanced by an implicit Newton solve; the resulting
    sensitivity ``Φ = (I − h A_{n+1})⁻¹`` is used to propagate the
    covariance, guaranteeing positive-definiteness.
    """

    def __init__(
        self,
        model: ContinuousDiscreteSDE,
        newton_tol: float,
        newton_max_iter: int,
    ) -> None:
        self._m = model
        self._tol = newton_tol
        self._mi = newton_max_iter

    def __call__(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        nx = x.shape[0]
        I_nx = np.eye(nx)
        t1 = t + h

        # Implicit drift solve
        S_n = self._m.sigma(x, u, d, p, t)

        def residual(xk: np.ndarray) -> np.ndarray:
            return xk - x - h * self._m.f(xk, u, d, p, t1)

        def jacobian(xk: np.ndarray) -> np.ndarray:
            return I_nx - h * self._m.dfdx(xk, u, d, p, t1)

        x_next = _newton_solve(residual, jacobian, x.copy(), self._tol, self._mi)

        # Sensitivity matrix Φ = (I − h A_{n+1})⁻¹
        M = jacobian(x_next)
        Phi = np.linalg.solve(M, I_nx)

        # Covariance: P_{n+1} = Φ (P_n + h σ_n σ_nᵀ) Φᵀ
        tau = P + h * S_n @ S_n.T
        P_next = Phi @ tau @ Phi.T
        P_next = (P_next + P_next.T) * 0.5
        return x_next, P_next


# ── Parameter structure ───────────────────────────────────────────────────────


@dataclass
class ContinuousDiscreteEKFParams(EstimatorParams):
    """
    Algorithm parameters for :class:`ContinuousDiscreteEKF`.

    Parameters
    ----------
    n_steps : int
        Number of integration sub-steps per sampling interval.  Default: 10.
    scheme : IntegrationScheme
        Propagation scheme.  :attr:`~IntegrationScheme.EXPLICIT_EULER` is explicit and
        cheap; use :attr:`~IntegrationScheme.IMPLICIT_EULER` for stiff drift
        dynamics.  Default: ``IntegrationScheme.EXPLICIT_EULER``.
    newton_tol : float
        Convergence tolerance for the implicit-Euler Newton solver.
        Ignored when ``scheme=IntegrationScheme.EXPLICIT_EULER``.  Default: 1e-10.
    newton_max_iter : int
        Maximum Newton iterations per implicit sub-step.
        Ignored when ``scheme=IntegrationScheme.EXPLICIT_EULER``.  Default: 50.
    """
    n_steps: int = 10
    scheme: IntegrationScheme = IntegrationScheme.EXPLICIT_EULER
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
        if not isinstance(params.scheme, IntegrationScheme):
            raise TypeError(
                f"scheme must be an IntegrationScheme member, got {params.scheme!r}."
            )

        self._model = model
        self._x: np.ndarray = np.array(x0, dtype=float)
        self._P: np.ndarray = np.array(P0, dtype=float)
        self._Ts: float = float(model.Ts)
        self._n_steps: int = int(params.n_steps)
        self._h: float = self._Ts / self._n_steps

        if params.scheme is IntegrationScheme.EXPLICIT_EULER:
            self._moment_step = _EulerMomentStep(model)
        else:
            self._moment_step = _ImplicitEulerMomentStep(
                model, params.newton_tol, params.newton_max_iter
            )

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
        ``t + Ts`` using the configured scheme.

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
        x = self._x.copy()
        P = self._P.copy()
        h = self._h
        t_j = t
        for _ in range(self._n_steps):
            x, P = self._moment_step(x, P, u, d, p, t_j, h)
            t_j += h
        self._x = x
        self._P = P
        return x.copy(), P.copy()

    def predict_for(
        self,
        dt: float,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update over an arbitrary interval ``dt``.

        Keeps the same nominal sub-step size ``h = Ts / n_steps`` and runs
        ``max(1, round(dt / h))`` sub-steps so per-step accuracy is unchanged
        regardless of whether ``dt`` is shorter or longer than ``Ts``.

        Parameters
        ----------
        dt : float              — integration duration in seconds.
        u  : (nu,) ndarray      — input (ZOH) over the interval.
        d  : (nd,) ndarray      — disturbance over the interval.
        p  : (np,) ndarray      — parameter vector.
        t  : float              — start time of the interval.
        """
        n_steps = max(1, round(dt / self._h))
        h = dt / n_steps

        x = self._x.copy()
        P = self._P.copy()
        t_j = t
        for _ in range(n_steps):
            x, P = self._moment_step(x, P, u, d, p, t_j, h)
            t_j += h
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
