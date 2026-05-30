"""
Continuous-Discrete EKF for SDAE Systems (``ContinuousDiscreteDAEEKF``).

(ControlToolbox §State Estimation for Nonlinear SDAE Systems —
*Continuous-Discrete Extended Kalman Filter for SDAEs*)

Model
-----
    dx(t) = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),  dw ~ N(0, I dt)
    0     = g(x, y, u, d, p, t)
    ym(tk) = hm(x(tk), y(tk), u, d, p) + v(tk),   v ~ N(0, Rm)

The algebraic variables y are *never* added to the differential state vector.
The implicit function theorem expresses y as an implicit function of x and
propagates the resulting sensitivities; the state covariance P remains nx × nx.

Initialisation
--------------
The initial ŷ_{0|0} satisfies g(x̂_{0|0}, ŷ_{0|0}, …) = 0 (Newton solve).
The initial algebraic covariance is

    P_{y,0|0} = Φ_{yx} P_{0|0} Φ_{yx}ᵀ,    (∂g/∂y) Φ_{yx} = −∂g/∂x.

Time update — implicit Euler with sensitivity propagation
---------------------------------------------------------
At each sub-step n:

1. Newton solve for (x_{n+1}, y_{n+1}):

       R(x_{n+1}, y_{n+1}) = [ x_{n+1} − x_n − f h;  g ] = 0.

2. One-step sensitivities from the residual Jacobian:

       [ I−(∂f/∂x)h  −(∂f/∂y)h ] [ Φ_{xx} ] = [ I ]
       [ ∂g/∂x        ∂g/∂y    ] [ Φ_{yx} ]   [ 0 ]

3. Covariance sub-step (left rectangular rule):

       τ = P_n + sigma sigmaᵀ h,    P_{n+1} = Φ_{xx} τ Φ_{xx}ᵀ

4. Algebraic covariance via implicit function theorem at (x_{n+1}, y_{n+1}).

Measurement update — total-derivative C
---------------------------------------
    C_k = ∂hm/∂x + (∂hm/∂y)(∂y/∂x),   with (∂g/∂y)(∂y/∂x) = −(∂g/∂x)
    e_k = ym_k − hm(x̂_{k|k-1}, ŷ_{k|k-1}, …)
    R_e = C_k P C_kᵀ + Rm,   K = P C_kᵀ R_e⁻¹

    x̂_{k|k} = x̂_{k|k-1} + K e_k                            (Joseph form)
    ŷ_{k|k} — Newton-projected onto g = 0 at x̂_{k|k}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import ContinuousDiscreteSDAE
from .._utils import _newton_solve
from ._base import ContinuousDiscreteDAEEstimator, EstimatorParams


# ── Parameter structure ───────────────────────────────────────────────────────


@dataclass
class ContinuousDiscreteDAEEKFParams(EstimatorParams):
    """
    Algorithm parameters for :class:`ContinuousDiscreteDAEEKF`.

    Parameters
    ----------
    n_steps : int
        Implicit-Euler sub-steps per measurement interval.  Default: 10.
    newton_tol : float
        Newton convergence tolerance for the implicit sub-step and the
        algebraic projections.  Default: 1e-10.
    newton_max_iter : int
        Maximum Newton iterations per solve.  Default: 50.
    """
    n_steps: int = 10
    newton_tol: float = 1e-10
    newton_max_iter: int = 50


# ── Estimator ─────────────────────────────────────────────────────────────────


class ContinuousDiscreteDAEEKF(ContinuousDiscreteDAEEstimator):
    """
    Continuous-Discrete EKF for SDAE systems
    (ControlToolbox §SDAE State Estimation — *CD-EKF for SDAEs*).

    Parameters
    ----------
    model : ContinuousDiscreteSDAE
        Nonlinear SDAE system.  Must implement ``f``, ``sigma``, ``g``,
        ``hm``, ``Rm`` plus Jacobians ``dfdx``, ``dfdy``, ``dgdx``, ``dgdy``,
        ``dhmdx``, ``dhmdy``.
    x0 : (nx,) ndarray
        Initial differential state estimate x̂_{0|0}.
    y0 : (ny,) ndarray
        Initial algebraic state guess (projected onto g = 0 by Newton iteration).
    P0 : (nx, nx) ndarray
        Initial state covariance P_{0|0}.
    Ts : float
        Measurement sampling interval (seconds).
    params : ContinuousDiscreteDAEEKFParams, optional
        Algorithm parameter struct.
    """

    def __init__(
        self,
        model: ContinuousDiscreteSDAE,
        x0: np.ndarray,
        y0: np.ndarray,
        P0: np.ndarray,
        Ts: float,
        params: ContinuousDiscreteDAEEKFParams | None = None,
    ) -> None:
        if params is None:
            params = ContinuousDiscreteDAEEKFParams()

        self._model = model
        self._Ts = float(Ts)
        self._n_steps = int(params.n_steps)
        self._h_sub = self._Ts / self._n_steps
        self._newton_tol = params.newton_tol
        self._newton_max_iter = int(params.newton_max_iter)

        self._x = np.array(x0, dtype=float)
        self._P = np.array(P0, dtype=float)
        self._nx = self._x.shape[0]
        self._ny = model.ny

        # Project initial y onto the constraint manifold
        self._y = self._consistent_y(
            self._x, np.array(y0, dtype=float),
            np.zeros(model.nu), np.zeros(model.nd),
            np.array([], dtype=float), 0.0,
        )
        # Initial algebraic covariance via implicit function theorem
        self._Py = self._compute_Py(
            self._x, self._y,
            np.zeros(model.nu), np.zeros(model.nd),
            np.array([], dtype=float), 0.0, P=self._P,
        )

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def x_hat(self) -> np.ndarray:
        """Current differential state estimate x̂ ∈ ℝⁿˣ (copy)."""
        return self._x.copy()

    @property
    def y_hat(self) -> np.ndarray:
        """Current algebraic state estimate ŷ ∈ ℝⁿʸ (copy)."""
        return self._y.copy()

    @property
    def P(self) -> np.ndarray:
        """Current differential state covariance P ∈ ℝⁿˣˣⁿˣ (copy)."""
        return self._P.copy()

    @property
    def Py(self) -> np.ndarray:
        """Algebraic-variable covariance P_y ∈ ℝⁿʸˣⁿʸ (copy)."""
        return self._Py.copy()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _consistent_y(
        self,
        x: np.ndarray,
        y_init: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Solve g(x, y, u, d, p, t) = 0 for y by Newton iteration."""
        return _newton_solve(
            residual=lambda y_: self._model.g(x, y_, u, d, p, t),
            jacobian=lambda y_: self._model.dgdy(x, y_, u, d, p, t),
            x0=y_init,
            tol=self._newton_tol,
            max_iter=self._newton_max_iter,
        )

    def _compute_Py(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
        P: np.ndarray,
    ) -> np.ndarray:
        """P_y = Φ_{yx} P Φ_{yx}ᵀ via the implicit function theorem."""
        dgdx = self._model.dgdx(x, y, u, d, p, t)
        dgdy = self._model.dgdy(x, y, u, d, p, t)
        Phi_yx = np.linalg.solve(dgdy, -dgdx)
        Py = Phi_yx @ P @ Phi_yx.T
        return 0.5 * (Py + Py.T)

    # ── Filter steps ──────────────────────────────────────────────────────

    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Time update via implicit-Euler / sensitivity propagation.

        Parameters
        ----------
        u : (nu,) ndarray  — input applied (ZOH) over [t, t+Ts].
        d : (nd,) ndarray  — disturbance over [t, t+Ts].
        p : (np,) ndarray  — parameter vector.
        t : float          — current time.

        Returns
        -------
        x_pred : (nx,) predicted differential state.
        y_pred : (ny,) consistent algebraic state.
        P_pred : (nx, nx) predicted covariance.
        """
        model = self._model
        nx = self._nx
        ny = self._ny
        h = self._h_sub

        x_n = self._x.copy()
        y_n = self._y.copy()
        P_n = self._P.copy()
        t_n = t

        I_xx = np.eye(nx)
        rhs_selector = np.concatenate([I_xx, np.zeros((ny, nx))], axis=0)  # (nx+ny, nx)

        for _ in range(self._n_steps):
            t_next = t_n + h
            rhs_x = x_n.copy()

            def residual(z: np.ndarray) -> np.ndarray:
                xn = z[:nx]
                yn = z[nx:]
                return np.concatenate([
                    xn - rhs_x - model.f(xn, yn, u, d, p, t_next) * h,
                    model.g(xn, yn, u, d, p, t_next),
                ])

            def jacobian(z: np.ndarray) -> np.ndarray:
                xn = z[:nx]
                yn = z[nx:]
                top = np.concatenate([
                    I_xx - model.dfdx(xn, yn, u, d, p, t_next) * h,
                    -model.dfdy(xn, yn, u, d, p, t_next) * h,
                ], axis=1)
                bot = np.concatenate([
                    model.dgdx(xn, yn, u, d, p, t_next),
                    model.dgdy(xn, yn, u, d, p, t_next),
                ], axis=1)
                return np.concatenate([top, bot], axis=0)

            z0 = np.concatenate([x_n, y_n])
            z_next = _newton_solve(
                residual, jacobian, z0,
                tol=self._newton_tol, max_iter=self._newton_max_iter,
            )
            x_next = z_next[:nx]
            y_next = z_next[nx:]

            J = jacobian(z_next)
            Phi_block = np.linalg.solve(J, rhs_selector)
            Phi_xx = Phi_block[:nx, :]

            sigma_n = model.sigma(x_n, y_n, u, d, p, t_n)
            tau = P_n + sigma_n @ sigma_n.T * h
            P_next = Phi_xx @ tau @ Phi_xx.T
            P_next = 0.5 * (P_next + P_next.T)

            x_n = x_next
            y_n = y_next
            P_n = P_next
            t_n = t_next

        self._x = x_n
        self._y = y_n
        self._P = P_n
        self._Py = self._compute_Py(x_n, y_n, u, d, p, t_n, P=P_n)
        return x_n.copy(), y_n.copy(), P_n.copy()

    def update(
        self,
        ym: np.ndarray,
        u: np.ndarray | None,
        d: np.ndarray | None,
        p: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Measurement update with total-derivative C and Joseph form.

        Parameters
        ----------
        ym   : (nym,) ndarray  — measurement ym_k.
        u    : (nu,) ndarray   — input at measurement time.
        d    : (nd,) ndarray   — disturbance at measurement time.
        p    : (np,) ndarray   — parameter vector.
        mask : (nym,) bool ndarray, optional — active-channel mask.

        Returns
        -------
        x_hat : (nx,) corrected differential state estimate.
        y_hat : (ny,) consistent algebraic state.
        P     : (nx, nx) corrected covariance.
        """
        model = self._model
        nx = self._nx
        x = self._x
        y_alg = self._y
        P = self._P
        R = model.Rm

        dhmdx = model.dhmdx(x, y_alg, u, d, p, 0.0)
        dhmdy = model.dhmdy(x, y_alg, u, d, p, 0.0)
        dgdx  = model.dgdx(x, y_alg, u, d, p, 0.0)
        dgdy  = model.dgdy(x, y_alg, u, d, p, 0.0)
        Phi_yx = np.linalg.solve(dgdy, -dgdx)
        C = dhmdx + dhmdy @ Phi_yx

        y_hat = model.hm(x, y_alg, u, d, p, 0.0)

        if mask is not None:
            active = np.where(mask)[0]
            if len(active) == 0:
                return x.copy(), y_alg.copy(), P.copy()
            C_sub      = C[active, :]
            y_hat_sub  = y_hat[active]
            y_obs_sub  = ym[active]
            R_sub      = R[np.ix_(active, active)]
        else:
            C_sub     = C
            y_hat_sub = y_hat
            y_obs_sub = ym
            R_sub     = R

        R_e = C_sub @ P @ C_sub.T + R_sub
        Kt = np.linalg.solve(R_e, C_sub @ P)
        K = Kt.T

        e = y_obs_sub - y_hat_sub
        x_new = x + K @ e

        IKC = np.eye(nx) - K @ C_sub
        P_new = IKC @ P @ IKC.T + K @ R_sub @ K.T
        P_new = 0.5 * (P_new + P_new.T)

        y_new = self._consistent_y(x_new, y_alg, u, d, p, 0.0)

        self._x = x_new
        self._y = y_new
        self._P = P_new
        self._Py = self._compute_Py(x_new, y_new, u, d, p, 0.0, P=P_new)
        return x_new.copy(), y_new.copy(), P_new.copy()

    def step(
        self,
        ym: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combined time + measurement update."""
        self.predict(u, d, p, t)
        return self.update(ym, u, d, p, mask=mask)
