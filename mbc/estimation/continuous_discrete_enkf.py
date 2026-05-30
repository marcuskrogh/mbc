"""
Continuous-Discrete Ensemble Kalman Filter (``ContinuousDiscreteEnKF``).

(ControlToolbox §State Estimation for Nonlinear SDE Systems —
*Continuous-Discrete Ensemble Kalman Filter*)

The CD-EnKF approximates the state distribution by a randomly sampled
ensemble of N_p members.  Each member is propagated independently through
the full stochastic dynamics.

Time update — per-member Euler-Maruyama
----------------------------------------
For each i = 1, …, N_p:

    dx̂_k^{(i)}(t) = f(x̂_k^{(i)}, u, d, p, t) dt
                  + sigma(x̂_k^{(i)}, u, d, p, t) dω_k^{(i)}(t)

with independent Wiener increments per member (``n_steps`` sub-steps).

Predicted statistics (Bessel-corrected):

    x̂_{k+1|k} = (1/N_p) Σ_i x̂_{k+1|k}^{(i)}
    P_{k+1|k} = (1/(N_p−1)) Σ_i (x̂^{(i)} − x̂)(…)ᵀ

Measurement update — perturbed observations
--------------------------------------------
    R_xy = (1/(N_p−1)) Σ_i (x̂^{(i)} − x̂)(z^{m,(i)} − ẑ^m)ᵀ
    R_zz = (1/(N_p−1)) Σ_i (z^{m,(i)} − ẑ^m)(…)ᵀ
    R_e = R_zz + Rm,   K = R_xy R_e⁻¹

    y^{m,(i)} = ym_k + v_k^{(i)},  v_k^{(i)} ~ N(0, Rm)
    x̂_{k|k}^{(i)} = x̂_{k|k-1}^{(i)} + K (y^{m,(i)} − z^{m,(i)})

Per-member noise perturbation prevents ensemble covariance collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import ContinuousDiscreteSDE
from .._utils import _cholesky_psd
from ._base import ContinuousDiscreteEstimator, EstimatorParams
from ._ensemble import _ensemble_measurements, _propagate_em_ensemble


# ── Parameter structure ───────────────────────────────────────────────────────


@dataclass
class ContinuousDiscreteEnKFParams(EstimatorParams):
    """
    Algorithm parameters for :class:`ContinuousDiscreteEnKF`.

    Parameters
    ----------
    N : int
        Ensemble size N_p.  Default: 100.
    n_steps : int
        Euler-Maruyama sub-steps per measurement interval.  Default: 10.
    seed : int or None
        Random seed for reproducibility.  Default: None.
    """
    N: int = 100
    n_steps: int = 10
    seed: int | None = None


# ── Estimator ─────────────────────────────────────────────────────────────────


class ContinuousDiscreteEnKF(ContinuousDiscreteEstimator):
    """
    Continuous-Discrete Ensemble Kalman Filter for SDE systems
    (ControlToolbox §SDE State Estimation — *CD-EnKF*).

    Parameters
    ----------
    model : ContinuousDiscreteSDE
        Nonlinear continuous-discrete SDE system.
    x0 : (nx,) ndarray
        Initial ensemble mean.
    P0 : (nx, nx) ndarray
        Initial covariance (used to draw the initial ensemble from N(x0, P0)).
    Ts : float
        Measurement sampling interval (seconds).
    params : ContinuousDiscreteEnKFParams, optional
        Algorithm parameter struct.  Pass to control ensemble size, integration
        steps, and random seed.
    """

    def __init__(
        self,
        model: ContinuousDiscreteSDE,
        x0: np.ndarray,
        P0: np.ndarray,
        Ts: float,
        params: ContinuousDiscreteEnKFParams | None = None,
    ) -> None:
        if params is None:
            params = ContinuousDiscreteEnKFParams()

        self._model = model
        self._Ts = float(Ts)
        self._N = int(params.N)
        self._n_steps = int(params.n_steps)
        self._h_sub = self._Ts / self._n_steps
        self._rng = np.random.default_rng(params.seed)

        nx = len(x0)
        self._nx = nx

        L = _cholesky_psd(P0)
        Z = self._rng.standard_normal((nx, self._N))
        self._X = np.array(x0, dtype=float)[:, None] + L @ Z   # (nx, N)

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def x_hat(self) -> np.ndarray:
        """Ensemble mean x̂ ∈ ℝⁿˣ (copy)."""
        return self._X.mean(axis=1).copy()

    @property
    def P(self) -> np.ndarray:
        """Bessel-corrected ensemble sample covariance P ∈ ℝⁿˣˣⁿˣ (copy)."""
        A = self._X - self._X.mean(axis=1, keepdims=True)
        return (A @ A.T) / (self._N - 1)

    @property
    def ensemble(self) -> np.ndarray:
        """Full ensemble matrix X ∈ ℝⁿˣˣᴺ (copy)."""
        return self._X.copy()

    # ── Filter steps ──────────────────────────────────────────────────────

    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update: propagate each ensemble member through the SDE
        via Euler-Maruyama.

        Parameters
        ----------
        u : (nu,) ndarray  — input applied (ZOH) over [t, t+Ts].
        d : (nd,) ndarray  — disturbance over [t, t+Ts].
        p : (np,) ndarray  — parameter vector.
        t : float          — current time.

        Returns
        -------
        x_pred : (nx,) ensemble mean after propagation.
        P_pred : (nx, nx) Bessel-corrected ensemble covariance.
        """
        self._X = _propagate_em_ensemble(
            self._model, self._X, u, d, p, t,
            h=self._h_sub, n_steps=self._n_steps, rng=self._rng,
        )
        return self.x_hat, self.P

    def update(
        self,
        ym: np.ndarray,
        u: np.ndarray | None,
        d: np.ndarray | None,
        p: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update via perturbed-observations Kalman correction.

        Parameters
        ----------
        ym   : (nym,) ndarray  — measurement vector.
        u    : (nu,) ndarray   — input at measurement time.
        d    : (nd,) ndarray   — disturbance at measurement time.
        p    : (np,) ndarray   — parameter vector.
        mask : (nym,) bool ndarray, optional — active-channel mask.

        Returns
        -------
        x_hat : (nx,) updated ensemble mean.
        P     : (nx, nx) Bessel-corrected ensemble covariance.
        """
        model = self._model
        N = self._N
        R = model.Rm

        Z = _ensemble_measurements(model, self._X, u, d, p)

        if mask is not None:
            active = np.where(mask)[0]
            if len(active) == 0:
                return self.x_hat, self.P
            y_sub = ym[active]
            Z = Z[active, :]
            R_sub = R[np.ix_(active, active)]
        else:
            y_sub = ym
            R_sub = R

        ny_act = y_sub.shape[0]

        x_mean = self._X.mean(axis=1, keepdims=True)
        z_mean = Z.mean(axis=1, keepdims=True)
        A_x = (self._X - x_mean) / np.sqrt(N - 1)
        A_z = (Z - z_mean) / np.sqrt(N - 1)

        R_xy = A_x @ A_z.T
        R_zz = A_z @ A_z.T
        R_e = R_zz + R_sub

        K = np.linalg.solve(R_e.T, R_xy.T).T

        V = self._rng.multivariate_normal(np.zeros(ny_act), R_sub, size=N).T
        Y_pert = y_sub[:, None] + V

        for i in range(N):
            innov_i = Y_pert[:, i] - Z[:, i]
            self._X[:, i] = self._X[:, i] + K @ innov_i

        return self.x_hat, self.P

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
