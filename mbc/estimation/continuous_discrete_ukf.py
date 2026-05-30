"""
Continuous-Discrete Unscented Kalman Filter (``ContinuousDiscreteUKF``).

(ControlToolbox §State Estimation for Nonlinear SDE Systems —
*Continuous-Discrete Unscented Kalman Filter*)

The CD-UKF represents the state distribution by deterministically sampled
sigma points (the unscented transform).  No Jacobians are required.

Time update — augmented dimension n̄ = nx + nω
----------------------------------------------
Tuning parameters α ∈ ]0, 1], κ ≥ 0, β ≥ 0 (β = 2 optimal for Gaussian):

    c̄ = α²(n̄ + κ),    λ̄ = α²(n̄ + κ) − n̄

Two sigma-point sets are built from (x̂_{k|k}, P_{k|k}):

Deterministic state set (2 nx + 1 points, captures state covariance)
    χ^{(0)}      = x̂_{k|k}
    χ^{(i)}      = x̂_{k|k} + √c̄ (√P)_i,   i = 1, …, nx
    χ^{(nx+i)}   = x̂_{k|k} − √c̄ (√P)_i,   i = 1, …, nx

Stochastic noise sigma set (2 nω points, all placed at the mean) with
deterministic Wiener increments that produce total increment √(c̄ Ts) e_i:

    Δω^{(2nx+i)}    = +√(c̄ Ts) e_i / n_steps,   i = 1, …, nω   (per sub-step)
    Δω^{(2nx+nω+i)} = −√(c̄ Ts) e_i / n_steps,   i = 1, …, nω

The state set is propagated via the drift ODE (explicit Euler, sub-step h);
the noise set is propagated via the full SDE with its structured increments.

Predicted mean and covariance are the Wm/Wc-weighted statistics over all
2 n̄ + 1 sigma points.

Measurement update — state dimension nx
---------------------------------------
A new 2 nx + 1 sigma-point set is generated from (x̂_{k|k-1}, P_{k|k-1})
using state-only tuning parameters (dimension nx):

    z^{m,(i)} = hm(χ^{(i)}, u, d, p),
    ŷ^m_{k|k-1} = Σ W_m^{(i)} z^{m,(i)},
    R_zz = Σ W_c^{(i)} (z − ŷ^m)ᵀ,   R_e = R_zz + Rm,
    R_xy = Σ W_c^{(i)} (χ − x̂)(z − ŷ^m)ᵀ,
    K = R_xy R_e⁻¹,
    x̂_{k|k} = x̂_{k|k-1} + K (ym_k − ŷ^m_{k|k-1}),
    P_{k|k} = P_{k|k-1} − K R_e Kᵀ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import ContinuousDiscreteSDE
from .._utils import _cholesky_psd
from ._base import ContinuousDiscreteEstimator, EstimatorParams


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sigma_weights(n: int, alpha: float, beta: float, kappa: float):
    """Compute scaled unscented sigma-point weights for dimension n."""
    c = alpha ** 2 * (n + kappa)
    lam = c - n
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wm[0] = lam / (n + lam)
    Wc = Wm.copy()
    Wc[0] = lam / (n + lam) + (1.0 - alpha ** 2 + beta)
    return c, lam, Wm, Wc


# ── Parameter structure ───────────────────────────────────────────────────────


@dataclass
class ContinuousDiscreteUKFParams(EstimatorParams):
    """
    Algorithm parameters for :class:`ContinuousDiscreteUKF`.

    Parameters
    ----------
    n_steps : int
        Number of explicit-Euler integration sub-steps per measurement
        interval.  Default: 10.
    alpha : float
        Sigma-point spread parameter α ∈ ]0, 1].  Default: 1.0.
    beta : float
        Distribution parameter (β = 2 is optimal for a Gaussian).
        Default: 2.0.
    kappa : float
        Secondary spread parameter κ ≥ 0.  Default: 0.0.
    """
    n_steps: int = 10
    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0


# ── Estimator ─────────────────────────────────────────────────────────────────


class ContinuousDiscreteUKF(ContinuousDiscreteEstimator):
    """
    Continuous-Discrete Unscented Kalman Filter for SDE systems
    (ControlToolbox §SDE State Estimation — *CD-UKF*).

    Parameters
    ----------
    model : ContinuousDiscreteSDE
        Nonlinear continuous-discrete SDE system.  Must expose a ``Ts``
        property giving the measurement sampling interval (seconds).
    x0 : (nx,) ndarray
        Initial state estimate x̂_{0|0}.
    P0 : (nx, nx) ndarray
        Initial state covariance P_{0|0}.
    params : ContinuousDiscreteUKFParams, optional
        Algorithm parameter struct.  Pass to control integration steps and
        unscented-transform tuning scalars.
    """

    def __init__(
        self,
        model: ContinuousDiscreteSDE,
        x0: np.ndarray,
        P0: np.ndarray,
        params: ContinuousDiscreteUKFParams | None = None,
    ) -> None:
        if params is None:
            params = ContinuousDiscreteUKFParams()

        self._model = model
        self._x = np.array(x0, dtype=float)
        self._P = np.array(P0, dtype=float)
        self._Ts = float(model.Ts)
        self._n_steps = int(params.n_steps)
        self._h_sub = self._Ts / self._n_steps
        self._alpha = float(params.alpha)
        self._beta = float(params.beta)
        self._kappa = float(params.kappa)

        self._nx = self._x.shape[0]
        self._nw = model.nw

        # Augmented (time-update) weights: dimension nx + nw
        n_bar = self._nx + self._nw
        self._n_bar = n_bar
        c_bar, lam_bar, Wm_bar, Wc_bar = _sigma_weights(
            n_bar, self._alpha, self._beta, self._kappa
        )
        self._c_bar = c_bar
        self._lam_bar = lam_bar
        self._Wm_bar = Wm_bar
        self._Wc_bar = Wc_bar

        # State-only (measurement-update) weights: dimension nx
        c_x, lam_x, Wm_x, Wc_x = _sigma_weights(
            self._nx, self._alpha, self._beta, self._kappa
        )
        self._c_x = c_x
        self._lam_x = lam_x
        self._Wm_x = Wm_x
        self._Wc_x = Wc_x

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
        Time update via the augmented sigma-point scheme.

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
        model = self._model
        nx = self._nx
        nw = self._nw
        n_bar = self._n_bar
        h = self._h_sub

        # Deterministic state sigma points (2 nx + 1)
        L = _cholesky_psd(self._P)
        sqrt_c = np.sqrt(self._c_bar)
        det_sigma = np.empty((2 * nx + 1, nx))
        det_sigma[0] = self._x
        for i in range(nx):
            det_sigma[i + 1]      = self._x + sqrt_c * L[:, i]
            det_sigma[nx + i + 1] = self._x - sqrt_c * L[:, i]

        # Stochastic noise sigma set (2 nw points, all placed at mean)
        # Total Wiener increment √(c̄ Ts) e_i split deterministically over n_steps.
        stoch_sigma = np.tile(self._x, (2 * nw, 1))
        I_nw = np.eye(nw)
        per_step_pos = np.sqrt(self._c_bar * self._Ts) / self._n_steps * I_nw
        per_step_neg = -per_step_pos
        stoch_d_omega = np.concatenate([per_step_pos, per_step_neg], axis=0)  # (2nw, nw)

        # Propagate
        t_j = t
        for _ in range(self._n_steps):
            for i in range(2 * nx + 1):
                f_i = model.f(det_sigma[i], u, d, p, t_j)
                det_sigma[i] = det_sigma[i] + h * f_i

            for i in range(2 * nw):
                xi = stoch_sigma[i]
                f_i = model.f(xi, u, d, p, t_j)
                sigma_i = model.sigma(xi, u, d, p, t_j)
                stoch_sigma[i] = xi + h * f_i + sigma_i @ stoch_d_omega[i]

            t_j += h

        # Reconstruct mean and covariance over all 2 n̄ + 1 sigma points
        all_sigma = np.concatenate([det_sigma, stoch_sigma], axis=0)

        x_pred = np.einsum("i,ij->j", self._Wm_bar, all_sigma)
        P_pred = np.zeros((nx, nx))
        for i in range(2 * n_bar + 1):
            diff = all_sigma[i] - x_pred
            P_pred += self._Wc_bar[i] * np.outer(diff, diff)
        P_pred = 0.5 * (P_pred + P_pred.T)

        self._x = x_pred
        self._P = P_pred
        return x_pred.copy(), P_pred.copy()

    def update(
        self,
        ym: np.ndarray,
        u: np.ndarray | None,
        d: np.ndarray | None,
        p: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update via the standard (state-only) unscented transform.

        Parameters
        ----------
        ym   : (nym,) ndarray  — measurement vector.
        u    : (nu,) ndarray   — input at measurement time.
        d    : (nd,) ndarray   — disturbance at measurement time.
        p    : (np,) ndarray   — parameter vector.
        mask : (nym,) bool ndarray, optional — active-channel mask.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.
        """
        model = self._model
        nx = self._nx
        R = model.Rm

        L = _cholesky_psd(self._P)
        sqrt_c = np.sqrt(self._c_x)
        sigma = np.empty((2 * nx + 1, nx))
        sigma[0] = self._x
        for i in range(nx):
            sigma[i + 1]      = self._x + sqrt_c * L[:, i]
            sigma[nx + i + 1] = self._x - sqrt_c * L[:, i]

        Z = np.array([
            model.hm(sigma[i], u, d, p, 0.0) for i in range(2 * nx + 1)
        ])

        if mask is not None:
            active = np.where(mask)[0]
            if len(active) == 0:
                return self._x.copy(), self._P.copy()
            Z = Z[:, active]
            y_sub = ym[active]
            R_sub = R[np.ix_(active, active)]
        else:
            y_sub = ym
            R_sub = R

        y_pred = np.einsum("i,ij->j", self._Wm_x, Z)

        ny_act = y_pred.shape[0]
        R_zz = np.zeros((ny_act, ny_act))
        R_xy = np.zeros((nx, ny_act))
        for i in range(2 * nx + 1):
            dy = Z[i] - y_pred
            dx = sigma[i] - self._x
            R_zz += self._Wc_x[i] * np.outer(dy, dy)
            R_xy += self._Wc_x[i] * np.outer(dx, dy)
        R_e = R_zz + R_sub

        K = np.linalg.solve(R_e.T, R_xy.T).T

        innov = y_sub - y_pred
        x_new = self._x + K @ innov
        P_new = self._P - K @ R_e @ K.T
        P_new = 0.5 * (P_new + P_new.T)

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
