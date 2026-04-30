"""
Continuous-Discrete Extended Kalman Filter (Ph.D. Ch. 7.1).

The CD-EKF propagates the state estimate and covariance through the
nonlinear continuous SDE dynamics between measurement times, then
applies a standard linearised measurement update.

Prediction (continuous propagation over [t_k, t_{k+1}]):
    dx_hat/dt = f(x_hat, u, d, t)
    dP/dt     = F(t) P + P F(t)ᵀ + G(t) Q_c G(t)ᵀ

where F = ∂f/∂x and G = g evaluated at the current estimate.
Both ODEs are integrated numerically with step size dt / n_steps.

Measurement update (standard EKF correction):
    H   = ∂h/∂x |_{x_hat}
    S   = H P⁻ Hᵀ + R
    K   = P⁻ Hᵀ S⁻¹
    x̂   = x̂⁻ + K (y − h(x̂⁻, d))
    P   = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ   (Joseph form)

Reference:  Ph.D. thesis, Ch. 7.1.
"""

from __future__ import annotations

import numpy as np

from ..models import ContinuousDiscreteModel


class ContinuousDiscreteEKF:
    """
    Continuous-Discrete Extended Kalman Filter (Ph.D. Ch. 7.1).

    Parameters
    ----------
    model : ContinuousDiscreteModel
        Nonlinear continuous-discrete system providing ``f``, ``g``, ``h``,
        ``Q_c``, and ``R``.
    x0 : (nx,) ndarray
        Initial state estimate.
    P0 : (nx, nx) ndarray
        Initial state covariance.
    dt : float
        Measurement sampling interval (seconds).
    n_steps : int, optional
        Number of Euler integration sub-steps per measurement interval.
        Default: 10.
    """

    def __init__(
        self,
        model: ContinuousDiscreteModel,
        x0: np.ndarray,
        P0: np.ndarray,
        dt: float,
        n_steps: int = 10,
    ) -> None:
        raise NotImplementedError(
            "ContinuousDiscreteEKF.__init__ is not yet implemented."
        )

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def x_hat(self) -> np.ndarray:
        """Current state estimate x̂ ∈ ℝⁿˣ (copy)."""
        raise NotImplementedError

    @property
    def P(self) -> np.ndarray:
        """Current state covariance P ∈ ℝⁿˣˣⁿˣ (copy)."""
        raise NotImplementedError

    # ── Filter steps ──────────────────────────────────────────────────────

    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate state estimate and covariance from t to t + dt.

        Integrates the nonlinear drift and continuous Riccati ODE using
        Euler steps of size ``dt / n_steps``.

        Parameters
        ----------
        u : (nu,) ndarray  — control input applied over [t, t+dt].
        d : (nd,) ndarray  — disturbance over [t, t+dt].
        t : float          — current time.

        Returns
        -------
        x_pred : (nx,) predicted state estimate.
        P_pred : (nx, nx) predicted covariance.

        Raises
        ------
        NotImplementedError
            Always — to be implemented in a future commit.
        """
        raise NotImplementedError(
            "ContinuousDiscreteEKF.predict is not yet implemented."
        )

    def update(
        self,
        y: np.ndarray,
        d: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply measurement update given observation y_k.

        Parameters
        ----------
        y : (ny,) ndarray   — observation vector.
        d : (nd,) ndarray   — disturbance at measurement time.
        mask : (ny,) bool ndarray, optional
            When provided, only outputs where ``mask[i]`` is ``True`` are
            used in the update.  ``None`` (default) uses all outputs.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.

        Raises
        ------
        NotImplementedError
            Always — to be implemented in a future commit.
        """
        raise NotImplementedError(
            "ContinuousDiscreteEKF.update is not yet implemented."
        )

    def step(
        self,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        t: float,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combined predict + update step.

        Propagates the estimate from the previous time to ``t``, then
        fuses the measurement ``y``.

        Parameters
        ----------
        y    : (ny,) ndarray  — observation at time t.
        u    : (nu,) ndarray  — input applied over the previous interval.
        d    : (nd,) ndarray  — disturbance at time t.
        t    : float          — current measurement time.
        mask : (ny,) bool ndarray, optional — see :meth:`update`.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.

        Raises
        ------
        NotImplementedError
            Always — to be implemented in a future commit.
        """
        raise NotImplementedError(
            "ContinuousDiscreteEKF.step is not yet implemented."
        )
