"""
Discrete-time Kalman Filter (state estimator).

Implements the standard discrete-time Kalman filter for a linear
discrete-time system of the form:

    x[k+1] = A x[k] + B u[k] + E d[k]       (state transition)
    y[k]   = C x[k]                          (observation)

Algorithm
---------
The filter alternates between two steps each time a new measurement
y[k] arrives:

  **Prediction** (time update):
    x̂⁻  = A x̂[k−1] + B u[k−1] + E d[k−1]
    P⁻   = A P[k−1] Aᵀ + Q                     (standard form, G = I)
    P⁻   = A P[k−1] Aᵀ + G Q Gᵀ               (noise-separated form, M.Sc. Ch. 5.4)

  **Filtering** (measurement update, Joseph stabilised form):
    S    = C P⁻ Cᵀ + R                       (innovation covariance)
    K    = P⁻ Cᵀ S⁻¹                         (Kalman gain)
    x̂    = x̂⁻ + K (y − C x̂⁻)               (corrected state)
    P    = (I − K C) P⁻ (I − K C)ᵀ + K R Kᵀ (Joseph form)

    If a boolean mask is supplied, outputs with mask[i]=False are treated
    as missing and the measurement update is skipped for those channels
    (M.Sc. Ch. 5.5).  When all outputs are masked, the full update is
    skipped (prediction-only step).

The Joseph form  P = (I−KC) P⁻ (I−KC)ᵀ + K R Kᵀ  guarantees that
the posterior covariance remains symmetric positive-semidefinite even
in the presence of finite-precision arithmetic, unlike the conventional
form  P = (I−KC) P⁻.

Notation
--------
    n   – state dimension                     x ∈ ℝⁿ
    m   – input dimension                     u ∈ ℝᵐ
    p   – disturbance dimension               d ∈ ℝᵖ
    l   – output dimension                    y ∈ ℝˡ
    Q   – process noise covariance            Q ∈ ℝⁿˣⁿ
    G   – noise input matrix                  G ∈ ℝⁿˣⁿ  (default I)
    R   – measurement noise covariance        R ∈ ℝˡˣˡ
    P   – state estimation error covariance   P ∈ ℝⁿˣⁿ
    K   – Kalman gain                         K ∈ ℝⁿˣˡ
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
from cvxopt import matrix

from .._utils import _np_to_cvx, _cvx_to_np, _cvx_col_to_np

if TYPE_CHECKING:
    from ..models import LinearDiscreteModel


# ── Kalman Filter ────────────────────────────────────────────────────────


class KalmanFilter:
    """
    Discrete-time Kalman filter with Joseph stabilised covariance update.

    Parameters
    ----------
    model : LinearDiscreteModel
        Linear discrete-time plant providing dimensions, output matrix C,
        initial state x, and the ``discretize(d)`` method.
    Q : cvxopt.matrix (n, n), optional
        Process noise covariance.  Default: 0.01 · Iₙ.
    R : cvxopt.matrix (l, l), optional
        Measurement noise covariance.  Default: 0.1 · Iˡ.
    P0 : cvxopt.matrix (n, n), optional
        Initial state covariance.  Default: Iₙ.
    noise_matrix : cvxopt.matrix (n, n), optional
        Noise input matrix G so that the prediction covariance step becomes
        ``P⁻ = A P Aᵀ + G Q Gᵀ`` (M.Sc. Ch. 5.4).  When ``None`` (default),
        the standard form ``P⁻ = A P Aᵀ + Q`` is used (equivalent to G = I).
    """

    def __init__(
        self,
        model: "LinearDiscreteModel",
        Q: matrix | None = None,
        R: matrix | None = None,
        P0: matrix | None = None,
        noise_matrix: matrix | None = None,
    ) -> None:
        self._model = model
        n = model.n_x
        l = model.C.size[0]

        # Noise covariances (cvxopt — used directly in filter() via self._R)
        self._Q: matrix = Q if Q is not None else matrix(0.0, (n, n))
        if Q is None:
            for i in range(n):
                self._Q[i, i] = 0.01

        self._R: matrix = R if R is not None else matrix(0.0, (l, l))
        if R is None:
            for i in range(l):
                self._R[i, i] = 0.1

        # Noise input matrix G (M.Sc. Ch. 5.4); None means use standard form
        self._G: matrix | None = noise_matrix

        # State error covariance (numpy internally)
        self._P_np: np.ndarray = _cvx_to_np(P0) if P0 is not None else np.eye(n)

        # State estimate x̂ (numpy internally; initialised from the model)
        self._x_np: np.ndarray = np.array(list(model.x), dtype=float)

        # Memory for previous input and disturbance (numpy internally)
        self._u_prev_np: np.ndarray = np.zeros(model.n_u)
        self._d_prev_np: np.ndarray = np.zeros(model.n_d)

        self._first: bool = True

        # Last Kalman innovation ν = y − C x̂⁻  (set after each filtering step)
        self._last_innovation_np: Optional[np.ndarray] = None

    # ── Public properties ────────────────────────────────────────────────

    @property
    def x_hat(self) -> matrix:
        """Current state estimate x̂[k] ∈ ℝⁿ (column vector, copy)."""
        return _np_to_cvx(self._x_np.reshape(-1, 1))

    @property
    def P(self) -> matrix:
        """Current covariance P[k] ∈ ℝⁿˣⁿ (copy)."""
        return _np_to_cvx(self._P_np)

    @property
    def last_innovation(self) -> Optional[List[float]]:
        """
        Most recent Kalman innovation ν = y − C x̂⁻ as a plain Python list.

        Returns ``None`` until the first filtering step has been completed
        (i.e. until the second call to :meth:`update`).
        """
        if self._last_innovation_np is None:
            return None
        return [float(v) for v in self._last_innovation_np]

    # ── Prediction step ──────────────────────────────────────────────────

    def predict(
        self,
        A: matrix,
        B: matrix,
        E: matrix,
    ) -> tuple[matrix, matrix]:
        """
        Time update (prediction).

        Propagates the state estimate and covariance one step forward
        using the previous applied input u[k−1] and disturbance d[k−1]:

            x̂⁻  = A x̂ + B u + E d
            P⁻   = A P Aᵀ + Q

        Returns
        -------
        x_pred : (n, 1) predicted state estimate.
        P_pred : (n, n) predicted covariance.
        """
        A_np = _cvx_to_np(A)
        B_np = _cvx_to_np(B)
        E_np = _cvx_to_np(E)
        Q_np = _cvx_to_np(self._Q)

        x_pred_np = (
            A_np @ self._x_np
            + B_np @ self._u_prev_np
            + E_np @ self._d_prev_np
        )
        if self._G is None:
            P_pred_np = A_np @ self._P_np @ A_np.T + Q_np
        else:
            G_np = _cvx_to_np(self._G)
            P_pred_np = A_np @ self._P_np @ A_np.T + G_np @ Q_np @ G_np.T

        return _np_to_cvx(x_pred_np.reshape(-1, 1)), _np_to_cvx(P_pred_np)

    # ── Filtering step (Joseph form) ─────────────────────────────────────

    def filter(
        self,
        y: matrix,
        x_pred: matrix,
        P_pred: matrix,
        C: matrix,
    ) -> tuple[matrix, matrix]:
        """
        Measurement update (filtering) using the Joseph stabilised form.

        Given the predicted state and covariance from the prediction step,
        fuses the measurement y[k] to produce the corrected estimate:

            S   = C P⁻ Cᵀ + R               (innovation covariance)
            K   = P⁻ Cᵀ S⁻¹                 (Kalman gain)
            x̂   = x̂⁻ + K (y − C x̂⁻)        (corrected state)

            I_KC = I − K C
            P    = I_KC P⁻ I_KCᵀ + K R Kᵀ   (Joseph form)

        Returns
        -------
        x_hat : (n, 1) corrected state estimate.
        P     : (n, n) corrected covariance (symmetric, PSD).
        """
        n = P_pred.size[0]

        # Convert inputs to numpy
        x_pred_np = _cvx_col_to_np(x_pred)
        P_pred_np = _cvx_to_np(P_pred)
        C_np = _cvx_to_np(C)
        y_np = _cvx_col_to_np(y)
        R_np = _cvx_to_np(self._R)

        # Innovation covariance  S = C P⁻ Cᵀ + R
        S_np = C_np @ P_pred_np @ C_np.T + R_np

        # Kalman gain  K = P⁻ Cᵀ S⁻¹
        # Solve  S Kᵀ = C P⁻  for Kᵀ,  then transpose.
        K_np = np.linalg.solve(S_np, C_np @ P_pred_np).T  # (n, l)

        # Innovation  ν = y − C x̂⁻
        nu_np = y_np - C_np @ x_pred_np

        # Corrected state  x̂ = x̂⁻ + K ν
        x_hat_np = x_pred_np + K_np @ nu_np

        # Joseph form  P = (I−KC) P⁻ (I−KC)ᵀ + K R Kᵀ
        I_KC = np.eye(n) - K_np @ C_np
        P_np = I_KC @ P_pred_np @ I_KC.T + K_np @ R_np @ K_np.T
        P_np = (P_np + P_np.T) * 0.5

        # Store innovation for external inspection
        self._last_innovation_np = nu_np

        return _np_to_cvx(x_hat_np.reshape(-1, 1)), _np_to_cvx(P_np)

    # ── Combined update ──────────────────────────────────────────────────

    def update(
        self,
        y: matrix,
        d: matrix,
        mask: list[bool] | None = None,
    ) -> matrix:
        """
        Assimilate measurement y[k] and return corrected estimate x̂[k].

        On the first call the state is bootstrapped directly from the
        measurement (left pseudo-inverse of C).  Subsequent calls run
        the full predict → filter cycle.

        Parameters
        ----------
        y : (l, 1) measurement vector.
        d : (p, 1) current disturbance vector.
        mask : list of bool, length l, optional
            When provided, only outputs where ``mask[i] is True`` are used
            in the measurement update.  If all entries are ``False`` the
            measurement update is skipped entirely (prediction-only step,
            M.Sc. Ch. 5.5).  ``None`` (default) uses all outputs.

        Returns
        -------
        x_hat : (n, 1) corrected state estimate (copy).
        """
        C = self._model.C
        l = C.size[0]
        n = self._model.n_x

        # Build active-output submatrices when a mask is provided
        if mask is not None:
            active = [i for i, m in enumerate(mask) if m]
        else:
            active = list(range(l))

        if self._first:
            # Bootstrap:  x̂ = C⁺ y  (Moore–Penrose pseudoinverse for C
            # with full column rank; for C = I this reduces to x̂ = y).
            C_np = _cvx_to_np(C)
            y_np = _cvx_col_to_np(y)
            CtC = C_np.T @ C_np
            Cty = C_np.T @ y_np
            self._x_np = np.linalg.solve(CtC, Cty)
            self._first = False
        else:
            # Discretise at previous disturbance (convert numpy → cvxopt for model)
            d_prev_cvx = _np_to_cvx(self._d_prev_np.reshape(-1, 1))
            A, B, E = self._model.discretize(d_prev_cvx)

            # Prediction step
            x_pred, P_pred = self.predict(A, B, E)

            if not active:
                # All outputs masked — skip measurement update (M.Sc. Ch. 5.5)
                self._x_np = _cvx_col_to_np(x_pred)
                self._P_np = _cvx_to_np(P_pred)
            elif len(active) == l:
                # All outputs available — full update
                x_hat_cvx, P_cvx = self.filter(y, x_pred, P_pred, C)
                self._x_np = _cvx_col_to_np(x_hat_cvx)
                self._P_np = _cvx_to_np(P_cvx)
            else:
                # Partial update: restrict C, R, y to active rows
                C_sub = matrix([C[i, j] for j in range(n) for i in active],
                               (len(active), n))
                R_sub = matrix(
                    [self._R[i, j] for j in range(l) for i in active
                     if j in active],
                    (len(active), len(active)),
                )
                y_sub = matrix([y[i] for i in active], (len(active), 1))
                # Temporarily swap noise covariance for the filter call
                R_orig = self._R
                self._R = R_sub
                x_hat_cvx, P_cvx = self.filter(y_sub, x_pred, P_pred, C_sub)
                self._R = R_orig
                self._x_np = _cvx_col_to_np(x_hat_cvx)
                self._P_np = _cvx_to_np(P_cvx)

        self._d_prev_np = _cvx_col_to_np(d)
        return self.x_hat

    # ── Action recording ─────────────────────────────────────────────────

    def record_action(self, u: matrix) -> None:
        """Record the applied control action u[k] for the next prediction."""
        self._u_prev_np = _cvx_col_to_np(u)
