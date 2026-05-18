"""
Continuous-Discrete Extended Kalman Filter (CD-EKF) for SDE systems
(ControlToolbox §State Estimation for Nonlinear SDE Systems —
*Continuous-Discrete Extended Kalman Filter*).

Model
-----
    dx(t)  = f(x, u, d, p, t) dt + sigma(x, u, d, p, t) dw(t),  dw ~ N(0, I dt)
    ym(tk) = hm(x(tk), p) + v(tk),                              v ~ N(0, R)

Time update over [t_k, t_{k+1}] (mean trajectory + Lyapunov ODE)
----------------------------------------------------------------
The predicted mean evolves as the expectation of the SDE (the SDE is a
martingale, so the diffusion contributes zero mean):

    dx̂_k/dt(t) = f(x̂_k(t), u(t), d(t), p, t),    x̂_k(t_k) = x̂_{k|k}.

The predicted covariance evolves according to the Lyapunov-type ODE:

    dP_k/dt(t) = A_k(t) P_k(t) + P_k(t) A_kᵀ(t) + sigma_k(t) sigma_kᵀ(t),
                                                  P_k(t_k) = P_{k|k},

where the Jacobian and diffusion are evaluated along the mean trajectory:

    A_k(t)     = ∂f/∂x (x̂_k(t), u(t), d(t), p, t),
    sigma_k(t) = sigma (x̂_k(t), u(t), d(t), p, t).

The predictions at t_{k+1} are x̂_{k+1|k} = x̂_k(t_{k+1}) and
P_{k+1|k} = P_k(t_{k+1}).

Both ODEs are integrated by explicit Euler with ``n_steps`` equidistant
sub-steps of size ``h = dt / n_steps``.  The covariance is symmetrised at
every sub-step to guard against numerical drift.

No implicit propagation scheme is available for the CD-EKF.  The filter
always uses explicit Euler for the mean ODE and the Lyapunov-type covariance
ODE.  This mirrors the design of the SDAE counterpart
(:class:`ContinuousDiscreteDAEEKF`) which fixes implicit-Euler propagation
without offering an explicit alternative.  For simulation of stiff SDE
systems an implicit integrator can be selected via
:class:`~mbc.simulation.SDESimulator`; that choice is independent of the
filter propagation scheme.

Measurement update at t_k (Joseph form)
---------------------------------------
    e_k   = y^m_k − ŷ^m_{k|k-1},         ŷ^m_{k|k-1} = hm(x̂_{k|k-1}, p)
    R_e,k = C_k P_{k|k-1} C_kᵀ + R,       C_k = ∂hm/∂x(x̂_{k|k-1}, p)
    K_k   = P_{k|k-1} C_kᵀ R_e,k⁻¹

    x̂_{k|k} = x̂_{k|k-1} + K_k e_k
    P_{k|k} = (I − K_k C_k) P_{k|k-1} (I − K_k C_k)ᵀ + K_k R K_kᵀ

The Joseph stabilising form is used as it preserves symmetry and positive
definiteness in finite-precision arithmetic.
"""

from __future__ import annotations

import numpy as np

from ..models import ContinuousDiscreteModel


class ContinuousDiscreteEKF:
    """
    Continuous-Discrete Extended Kalman Filter for SDE systems
    (ControlToolbox §SDE State Estimation — *CD-EKF*).

    The filter always uses explicit Euler for the time update (mean ODE
    and Lyapunov covariance ODE).  No implicit propagation scheme is
    available; see the module docstring for the design rationale.

    Parameters
    ----------
    model : ContinuousDiscreteModel
        Nonlinear continuous-discrete SDE system providing ``f``, ``sigma``,
        ``hm``, ``Rm``, and Jacobians ``dfdx`` and ``dhmdx``.
    x0 : (nx,) ndarray
        Initial state estimate x̂_{0|0}.
    P0 : (nx, nx) ndarray
        Initial state covariance P_{0|0}.
    dt : float
        Measurement sampling interval (seconds).
    n_steps : int, optional
        Number of explicit-Euler integration sub-steps per measurement
        interval.  Must be at least 1.  Default: 10.
    """

    def __init__(
        self,
        model: ContinuousDiscreteModel,
        x0: np.ndarray,
        P0: np.ndarray,
        dt: float,
        n_steps: int = 10,
    ) -> None:
        if n_steps < 1:
            raise ValueError(
                f"n_steps must be a positive integer, got {n_steps!r}.  "
                "ContinuousDiscreteEKF always uses explicit Euler propagation "
                "and does not support implicit integration schemes."
            )
        self._model = model
        self._x_np: np.ndarray = np.array(x0, dtype=float)
        self._P_np: np.ndarray = np.array(P0, dtype=float)
        self._dt = dt
        self._n_steps = n_steps
        self._h = dt / n_steps

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def x_hat(self) -> np.ndarray:
        """Current state estimate x̂ ∈ ℝⁿˣ (copy)."""
        return self._x_np.copy()

    @property
    def P(self) -> np.ndarray:
        """Current state covariance P ∈ ℝⁿˣˣⁿˣ (copy)."""
        return self._P_np.copy()

    # ── Filter steps ──────────────────────────────────────────────────────

    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update: integrate the mean ODE and the Lyapunov ODE for the
        covariance from ``t`` to ``t + dt``.

        Parameters
        ----------
        u : (nu,) ndarray  — control input applied over [t, t+dt].
        d : (nd,) ndarray  — disturbance over [t, t+dt].
        p : (nparams,) ndarray  — parameter vector.
        t : float          — current time.

        Returns
        -------
        x_pred : (nx,) predicted state estimate x̂_{k+1|k}.
        P_pred : (nx, nx) predicted covariance P_{k+1|k}.
        """
        x = self._x_np.copy()
        P = self._P_np.copy()
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
            P = (P + P.T) * 0.5  # symmetrise to prevent numerical drift
            t_j += h

        self._x_np = x
        self._P_np = P
        return x.copy(), P.copy()

    def update(
        self,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update: fuse observation ``y_k`` with the current
        prior using the Joseph stabilising form.

        Parameters
        ----------
        y : (nym,) ndarray  — observation vector at the measurement time.
        u : (nu,) ndarray   — input at measurement time.
        d : (nd,) ndarray   — disturbance at measurement time.
        p : (nparams,) ndarray  — parameter vector.
        mask : (nym,) bool ndarray, optional
            When provided, only outputs where ``mask[i]`` is ``True`` are
            used in the update.  ``None`` (default) uses all outputs.

        Returns
        -------
        x_hat : (nx,) corrected state estimate x̂_{k|k}.
        P     : (nx, nx) corrected covariance P_{k|k}.
        """
        x = self._x_np
        P = self._P_np
        nx = x.shape[0]
        R = self._model.Rm

        C = self._model.dhmdx(x, u, d, p, 0.0)        # (nym, nx)
        y_hat = self._model.hm(x, u, d, p, 0.0)       # (nym,)

        if mask is not None:
            active = np.where(mask)[0]
            if len(active) == 0:
                return x.copy(), P.copy()
            C = C[active, :]
            y_hat = y_hat[active]
            y_sub = y[active]
            R_sub = R[np.ix_(active, active)]
        else:
            y_sub = y
            R_sub = R

        # Innovation covariance R_e = C P Cᵀ + R
        R_e = C @ P @ C.T + R_sub

        # Kalman gain  K = P Cᵀ R_e⁻¹  via solve (R_e Kᵀ = C P)
        Kt = np.linalg.solve(R_e, C @ P)
        K = Kt.T

        # State correction
        e = y_sub - y_hat
        x_new = x + K @ e

        # Joseph form:  P = (I − K C) P (I − K C)ᵀ + K R Kᵀ
        IKC = np.eye(nx) - K @ C
        P_new = IKC @ P @ IKC.T + K @ R_sub @ K.T
        P_new = (P_new + P_new.T) * 0.5

        self._x_np = x_new
        self._P_np = P_new
        return x_new.copy(), P_new.copy()

    def step(
        self,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combined time + measurement update.

        Propagates the estimate from the previous time to ``t``, then
        fuses the measurement ``y``.

        Parameters
        ----------
        y    : (nym,) ndarray  — observation at time t.
        u    : (nu,) ndarray   — input applied over the previous interval.
        d    : (nd,) ndarray   — disturbance at time t.
        p    : (nparams,) ndarray  — parameter vector.
        t    : float           — current measurement time.
        mask : (nym,) bool ndarray, optional — see :meth:`update`.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.
        """
        self.predict(u, d, p, t)
        return self.update(y, u, d, p, mask=mask)
