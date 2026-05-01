"""
Continuous-Discrete Kalman Filter.

Implements the Kalman filter for a linear continuous-discrete stochastic
system following the continuous-time formulation of the CD-EKF (Ph.D. Ch. 7.3),
specialised to a linear system where no Jacobian linearisation is required.

Model
-----
    dx = (A_c x + B_c u + E_c d) dt + G dw,   w ~ N(0, Q_c)
    y[k] = C x[k] + v[k],                      v[k] ~ N(0, R)

The filter operates in two steps at each measurement time t_k:

**Prediction** — continuous-time ODE integration over [t_{k-1}, t_k]:

    dx̂/dt  = A_c x̂ + B_c u + E_c d                           (§7.3a)
    dP/dt  = A_c P + P A_cᵀ + G Q_c Gᵀ                       (§7.3b)

Both ODEs are integrated forward from t_{k-1} to t_k with ``n_steps``
forward-Euler sub-steps of size h = dt / n_steps.  Inputs u and
disturbances d are held constant over the interval (zero-order hold).

**Filtering** (measurement update, Joseph stabilised form, §7.8–7.11):

    e_k   = y_k − C x̂[k|k−1]
    R_e   = C P[k|k−1] Cᵀ + R
    K     = P[k|k−1] Cᵀ R_e⁻¹
    x̂[k]  = x̂[k|k−1] + K e_k
    P[k]  = (I − K C) P[k|k−1] (I − K C)ᵀ + K R Kᵀ

The Joseph form guarantees that P remains symmetric positive semi-definite
in finite-precision arithmetic.

Missing observations (M.Sc. thesis, Ch. 5.5) are handled by the ``mask``
argument of :meth:`update`: outputs with ``mask[i] = False`` are excluded
from the measurement update.

Notation (consistent with Ph.D. thesis, Ch. 7.3)
-------------------------------------------------
    n   – state dimension                x ∈ ℝⁿ
    m   – input dimension                u ∈ ℝᵐ
    p   – disturbance dimension          d ∈ ℝᵖ
    l   – output dimension               y ∈ ℝˡ
    Q_c – continuous process-noise cov.  Q_c ∈ ℝᵍˣᵍ
    R   – measurement noise cov.         R ∈ ℝˡˣˡ
    P   – state error covariance         P ∈ ℝⁿˣⁿ
    K   – Kalman gain                    K ∈ ℝⁿˣˡ
    e   – innovation  y − C x̂⁻          e ∈ ℝˡ
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

import numpy as np
from cvxopt import matrix, lapack

from .._utils import _eye, _symmetrise, _np_to_cvx

if TYPE_CHECKING:
    from ..models import LinearContinuousDiscreteModel


class CDKalmanFilter:
    """
    Kalman filter for a linear continuous-discrete stochastic system.

    The prediction step integrates the state ODE (§7.3a) and the matrix
    Riccati ODE (§7.3b) continuously over each sampling interval using
    ``n_steps`` forward-Euler sub-steps.  The system is never discretised;
    inputs are applied as zero-order hold over each sampling interval.

    Parameters
    ----------
    model : LinearContinuousDiscreteModel
        Plant model providing ``A_c``, ``B_c``, ``E_c``, ``G``, ``Q_c``,
        ``C``, ``R``, ``dt``, ``x`` (initial state), ``n_x``, ``n_u``,
        ``n_d``.
    P0 : cvxopt.matrix (n, n), optional
        Initial state error covariance.  Default: Iₙ.
    n_steps : int, optional
        Number of forward-Euler sub-steps per sampling interval.  Larger
        values give a more accurate ODE integration at the cost of runtime.
        Default: 10.
    """

    def __init__(
        self,
        model: "LinearContinuousDiscreteModel",
        P0: matrix | None = None,
        n_steps: int = 10,
    ) -> None:
        self._model = model
        n = model.nx

        # Continuous-time matrices (numpy, cached for speed)
        self._A_c: np.ndarray = model.A_c
        self._B_c: np.ndarray = model.B_c
        self._E_c: np.ndarray = model.E_c
        self._GQcGT: np.ndarray = model.G @ model.Q_c @ model.G.T

        # ODE integration parameters
        self._dt: float = model.dt
        self._n_steps: int = n_steps
        self._h: float = model.dt / n_steps

        # Measurement noise covariance (cvxopt, for Joseph update)
        self._R: matrix = model.R_cvx

        # State error covariance (numpy internally)
        if P0 is not None:
            rows, cols = P0.size
            self._P_np: np.ndarray = np.array(list(P0), dtype=float).reshape(
                (rows, cols), order="F"
            )
        else:
            self._P_np = np.eye(n)

        # State estimate x̂ (numpy internally)
        self._x_np: np.ndarray = np.array(model.x, dtype=float)

        # Previous input and disturbance (numpy, ZOH over the interval)
        self._u_prev_np: np.ndarray = np.zeros(model.nu)
        self._d_prev_np: np.ndarray = np.zeros(model.nd)

        self._first: bool = True

        # Last Kalman innovation e = y − C x̂⁻
        self._last_innovation: Optional[matrix] = None

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def x_hat(self) -> matrix:
        """Current state estimate x̂[k] ∈ ℝⁿ (column vector, copy)."""
        return _np_to_cvx(self._x_np.reshape(-1, 1))

    @property
    def P(self) -> matrix:
        """Current state error covariance P[k] ∈ ℝⁿˣⁿ (copy)."""
        return _np_to_cvx(self._P_np)

    @property
    def last_innovation(self) -> Optional[List[float]]:
        """
        Most recent Kalman innovation e = y − C x̂⁻ as a plain Python list.

        Returns ``None`` until the first filtering step has been completed
        (i.e. until the second call to :meth:`update`).
        """
        if self._last_innovation is None:
            return None
        n = self._last_innovation.size[0]
        return [float(self._last_innovation[i]) for i in range(n)]

    # ── Prediction step (continuous ODE integration) ──────────────────────

    def predict(
        self,
        u_np: np.ndarray,
        d_np: np.ndarray,
    ) -> tuple[matrix, matrix]:
        """
        Time update — integrate state and Riccati ODEs over one sampling
        interval using ``n_steps`` forward-Euler sub-steps.

        Integrates (Ph.D. thesis, §7.3a–b):

            dx̂/dt = A_c x̂ + B_c u + E_c d
            dP/dt = A_c P + P A_cᵀ + G Q_c Gᵀ

        with u and d held constant (ZOH) over [t_{k-1}, t_k].

        Parameters
        ----------
        u_np : (m,) ndarray  — control input applied over the interval.
        d_np : (p,) ndarray  — disturbance over the interval.

        Returns
        -------
        x_pred : (n, 1) cvxopt column — predicted state estimate x̂[k|k−1].
        P_pred : (n, n) cvxopt matrix — predicted error covariance P[k|k−1].
        """
        x = self._x_np.copy()
        P = self._P_np.copy()

        h = self._h
        A = self._A_c
        Bu = self._B_c @ u_np
        Ed = self._E_c @ d_np
        GQG = self._GQcGT

        for _ in range(self._n_steps):
            x_dot = A @ x + Bu + Ed
            P_dot = A @ P + P @ A.T + GQG
            x = x + h * x_dot
            P = P + h * P_dot

        P = (P + P.T) * 0.5

        return _np_to_cvx(x.reshape(-1, 1)), _np_to_cvx(P)

    # ── Filtering step (Joseph stabilised form) ────────────────────────────

    def filter(
        self,
        y: matrix,
        x_pred: matrix,
        P_pred: matrix,
        C: matrix,
    ) -> tuple[matrix, matrix]:
        """
        Measurement update (filtering) using the Joseph stabilised form.

        Given the predicted state and covariance, fuses the measurement y[k]
        (Ph.D. thesis, §7.8–7.11):

            R_e = C P[k|k−1] Cᵀ + R
            K   = P[k|k−1] Cᵀ R_e⁻¹
            e   = y − C x̂[k|k−1]
            x̂   = x̂[k|k−1] + K e
            IKC = I − K C
            P   = IKC P[k|k−1] IKCᵀ + K R Kᵀ    (Joseph form, §7.11c)

        Parameters
        ----------
        y      : (l, 1) measurement vector.
        x_pred : (n, 1) predicted state estimate.
        P_pred : (n, n) predicted covariance.
        C      : (l, n) output matrix.

        Returns
        -------
        x_hat : (n, 1) corrected state estimate.
        P     : (n, n) corrected covariance (symmetric, PSD).
        """
        n = P_pred.size[0]

        # Innovation covariance  R_e = C P⁻ Cᵀ + R
        R_e = C * P_pred * C.T + self._R

        # Solve R_e Kᵀ = C P⁻  for Kᵀ via Cholesky (R_e is SPD)
        Kt = matrix(P_pred * C.T)
        R_e_copy = matrix(R_e)
        lapack.posv(R_e_copy, Kt)
        K = Kt                          # n × l

        # Innovation e = y − C x̂⁻
        e = y - C * x_pred

        # Corrected state estimate
        x_hat = x_pred + K * e

        # Joseph form:  P = (I−KC) P⁻ (I−KC)ᵀ + K R Kᵀ
        I_n = _eye(n)
        IKC = I_n - K * C
        P = IKC * P_pred * IKC.T + K * self._R * K.T
        P = _symmetrise(P)

        self._last_innovation = matrix(e)
        return x_hat, P

    # ── Combined update ───────────────────────────────────────────────────

    def update(
        self,
        y: matrix,
        d: matrix,
        mask: list[bool] | None = None,
    ) -> matrix:
        """
        Assimilate measurement y[k] and return the corrected estimate x̂[k].

        On the first call the state is bootstrapped via the left
        pseudo-inverse of C (reduces to x̂ = y when C = I).  Subsequent
        calls execute the full predict → filter cycle.

        Parameters
        ----------
        y    : (l, 1) measurement vector (cvxopt column).
        d    : (p, 1) current disturbance vector (cvxopt column).
        mask : list of bool, length l, optional
            Output availability mask (M.Sc. thesis, Ch. 5.5).  When
            ``mask[i] = False`` output i is excluded from the measurement
            update.  If all entries are ``False`` the update step is skipped
            entirely (prediction-only).  ``None`` uses all outputs.

        Returns
        -------
        x_hat : (n, 1) corrected state estimate (copy).
        """
        C = self._model.C_cvx
        l = self._model.ny

        active = list(range(l)) if mask is None else [i for i, m in enumerate(mask) if m]

        if self._first:
            # Bootstrap via minimum-norm solution: x̂ = Cᵀ (C Cᵀ)⁻¹ y
            # Works for any full-row-rank C (l ≤ n); solves the l×l system.
            CCt = C * C.T
            alpha = matrix(y)
            lapack.posv(CCt, alpha)
            x_boot = C.T * alpha
            n = self._model.nx
            self._x_np = np.array([float(x_boot[i]) for i in range(n)])
            self._first = False
        else:
            # Continuous ODE prediction using previous u and d (ZOH)
            x_pred, P_pred = self.predict(self._u_prev_np, self._d_prev_np)

            if not active:
                # All outputs masked — skip measurement update, store numpy
                rows, cols = x_pred.size
                self._x_np = np.array(list(x_pred), dtype=float).reshape(rows, order="F")
                rows2 = P_pred.size[0]
                self._P_np = np.array(list(P_pred), dtype=float).reshape(
                    (rows2, rows2), order="F"
                )
            elif len(active) == l:
                # All outputs available — standard update
                x_hat_cvx, P_cvx = self.filter(y, x_pred, P_pred, C)
                n = self._model.nx
                self._x_np = np.array([float(x_hat_cvx[i]) for i in range(n)])
                self._P_np = np.array(list(P_cvx), dtype=float).reshape(
                    (n, n), order="F"
                )
            else:
                # Partial update: restrict C and R to active rows
                n = P_pred.size[0]
                na = len(active)
                C_sub = matrix(
                    [C[i, j] for j in range(n) for i in active], (na, n)
                )
                R_sub = matrix(0.0, (na, na))
                for ii, i in enumerate(active):
                    for jj, j in enumerate(active):
                        R_sub[ii, jj] = self._R[i, j]
                y_sub = matrix([y[i] for i in active], (na, 1))
                R_orig = self._R
                self._R = R_sub
                x_hat_cvx, P_cvx = self.filter(y_sub, x_pred, P_pred, C_sub)
                self._R = R_orig
                self._x_np = np.array([float(x_hat_cvx[i]) for i in range(n)])
                self._P_np = np.array(list(P_cvx), dtype=float).reshape(
                    (n, n), order="F"
                )

        # Update stored disturbance for next prediction
        n_d = self._model.nd
        self._d_prev_np = np.array([float(d[i]) for i in range(n_d)])
        return self.x_hat

    # ── Action recording ──────────────────────────────────────────────────

    def record_action(self, u: matrix) -> None:
        """Record the applied control action u[k] for use in the next prediction."""
        n_u = self._model.nu
        self._u_prev_np = np.array([float(u[i]) for i in range(n_u)])
