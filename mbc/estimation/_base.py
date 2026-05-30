"""
Abstract base classes for all state estimators and their parameter structures.

Hierarchy
---------
EstimatorParams (ABC)
    DiscreteLinearKFParams
    ContinuousDiscreteLinearKFParams
    ContinuousDiscreteEKFParams
    ContinuousDiscreteUKFParams
    ContinuousDiscreteEnKFParams
    ContinuousDiscretePFParams
    ContinuousDiscreteDAEEKFParams

DiscreteEstimator (ABC)
    DiscreteLinearKF

ContinuousDiscreteEstimator (ABC)
    ContinuousDiscreteLinearKF
    ContinuousDiscreteEKF
    ContinuousDiscreteUKF
    ContinuousDiscreteEnKF
    ContinuousDiscretePF

ContinuousDiscreteDAEEstimator (ContinuousDiscreteEstimator, ABC)
    ContinuousDiscreteDAEEKF
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


# ── Integration scheme enum ───────────────────────────────────────────────────


class IntegrationScheme(Enum):
    """
    Numerical integration scheme for continuous-discrete state estimators.

    All schemes use a fixed sub-step size ``h = Ts / n_steps``.

    ``EXPLICIT_EULER``
        Explicit Euler / Euler-Maruyama.  For particle-based estimators
        (UKF, EnKF, PF) the drift and diffusion are both evaluated at the
        *current* sub-step (explicit-explicit, EE).  For the EKF the moment
        ODE ``(dx̂/dt, dP/dt)`` is integrated with a forward Euler step.
        First-order accurate.  Recommended for non-stiff dynamics.

    ``IMPLICIT_EULER``
        Implicit-Explicit Euler.  The drift is evaluated at the *next*
        sub-step and solved via Newton iteration; the diffusion remains
        explicit.  For the EKF the state mean is advanced implicitly and
        the covariance update uses the resulting sensitivity matrix
        ``Φ = (I − h A)⁻¹``, which guarantees positive-definiteness.
        Recommended for stiff drift dynamics.
    """
    EXPLICIT_EULER = 1
    IMPLICIT_EULER = 2


# ── Abstract parameter structure ──────────────────────────────────────────────


class EstimatorParams(ABC):
    """
    Abstract base class for all estimator parameter structures.

    Concrete subclasses are plain :func:`dataclasses.dataclass` objects that
    group the algorithm-specific hyper-parameters for a particular estimator
    (integration step count, ensemble size, tuning scalars, etc.).  Initial
    conditions ``x0`` / ``P0`` and the sampling interval ``Ts`` are passed
    directly to the estimator constructor and are *not* part of the params
    structure.
    """


# ── Discrete-time estimator ───────────────────────────────────────────────────


class DiscreteEstimator(ABC):
    """
    Abstract base class for discrete-time state estimators.

    Every implementation must provide:

    Properties
    ----------
    x_hat : (nx,) ndarray
        Current state estimate (copy).
    P : (nx, nx) ndarray
        Current state error covariance (copy).

    Methods
    -------
    predict(u, d) → (x_pred, P_pred)
        Time update over one sampling interval.
    update(ym, mask=None) → (x_hat, P)
        Measurement update.
    step(ym, u, d, p=None, t=None, mask=None) → (x_hat, P)
        Combined predict + update.
    """

    @property
    @abstractmethod
    def x_hat(self) -> np.ndarray:
        """Current state estimate x̂ ∈ ℝⁿˣ (copy)."""

    @property
    @abstractmethod
    def P(self) -> np.ndarray:
        """Current state error covariance P ∈ ℝⁿˣˣⁿˣ (copy)."""

    @abstractmethod
    def predict(
        self, u: np.ndarray, d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update over one sampling interval.

        Parameters
        ----------
        u : (nu,) ndarray  — input applied (ZOH) over the just-completed interval.
        d : (nd,) ndarray  — disturbance over the same interval.

        Returns
        -------
        x_pred : (nx,) predicted state estimate.
        P_pred : (nx, nx) predicted covariance.
        """

    @abstractmethod
    def update(
        self,
        ym: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update.

        Parameters
        ----------
        ym   : (nym,) ndarray          — measurement vector.
        mask : (nym,) bool ndarray, optional — active-channel mask.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.
        """

    @abstractmethod
    def step(
        self,
        ym: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p=None,
        t: float | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combined time + measurement update.

        Parameters
        ----------
        ym   : (nym,) ndarray            — measurement at ``t_k``.
        u    : (nu,) ndarray             — input over the previous interval.
        d    : (nd,) ndarray             — disturbance over the previous interval.
        p    : ignored                    — accepted for interface uniformity.
        t    : float, ignored             — accepted for interface uniformity.
        mask : (nym,) bool ndarray, optional — active-channel mask.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.
        """


# ── Continuous-discrete estimator ─────────────────────────────────────────────


class ContinuousDiscreteEstimator(ABC):
    """
    Abstract base class for continuous-discrete state estimators.

    Every implementation must provide:

    Properties
    ----------
    x_hat : (nx,) ndarray
        Current state estimate (copy).
    P : (nx, nx) ndarray
        Current state error covariance (copy).

    Methods
    -------
    predict(u, d, p, t) → (x_pred, P_pred)
        Time update by integrating the SDE from ``t`` to ``t + Ts``.
    update(ym, u, d, p, mask=None) → (x_hat, P)
        Measurement update.
    step(ym, u, d, p, t, mask=None) → (x_hat, P)
        Combined predict + update.
    """

    @property
    @abstractmethod
    def x_hat(self) -> np.ndarray:
        """Current state estimate x̂ ∈ ℝⁿˣ (copy)."""

    @property
    @abstractmethod
    def P(self) -> np.ndarray:
        """Current state error covariance P ∈ ℝⁿˣˣⁿˣ (copy)."""

    @abstractmethod
    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update: propagate the state distribution from ``t`` to ``t + Ts``.

        Parameters
        ----------
        u : (nu,) ndarray       — input applied (ZOH) over [t, t+Ts].
        d : (nd,) ndarray       — disturbance over [t, t+Ts].
        p : (np,) ndarray       — parameter vector (``None`` for linear models).
        t : float               — current time.

        Returns
        -------
        x_pred : (nx,) predicted state estimate.
        P_pred : (nx, nx) predicted covariance.
        """

    @abstractmethod
    def update(
        self,
        ym: np.ndarray,
        u: np.ndarray | None,
        d: np.ndarray | None,
        p: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update.

        Parameters
        ----------
        ym   : (nym,) ndarray          — measurement vector.
        u    : (nu,) ndarray           — input at measurement time.
        d    : (nd,) ndarray           — disturbance at measurement time.
        p    : (np,) ndarray           — parameter vector.
        mask : (nym,) bool ndarray, optional — active-channel mask.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.
        """

    @abstractmethod
    def step(
        self,
        ym: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combined time + measurement update.

        Parameters
        ----------
        ym   : (nym,) ndarray            — measurement at ``t``.
        u    : (nu,) ndarray             — input over the previous interval.
        d    : (nd,) ndarray             — disturbance over the previous interval.
        p    : (np,) ndarray             — parameter vector.
        t    : float                     — current measurement time.
        mask : (nym,) bool ndarray, optional — active-channel mask.

        Returns
        -------
        x_hat : (nx,) corrected state estimate.
        P     : (nx, nx) corrected covariance.
        """


# ── Continuous-discrete DAE estimator ─────────────────────────────────────────


class ContinuousDiscreteDAEEstimator(ContinuousDiscreteEstimator):
    """
    Abstract base class for continuous-discrete estimators on SDAE systems.

    Extends :class:`ContinuousDiscreteEstimator` with algebraic-variable
    state and its covariance.  The state covariance ``P`` covers only the
    differential variables ``x``; a separate ``Py`` covers the algebraic
    variables ``y``.

    Overridden return types
    -----------------------
    * ``predict`` → ``(x_pred, y_pred, P_pred)``
    * ``update``  → ``(x_hat, y_hat, P)``
    * ``step``    → ``(x_hat, y_hat, P)``
    """

    @property
    @abstractmethod
    def y_hat(self) -> np.ndarray:
        """Current algebraic-state estimate ŷ ∈ ℝⁿʸ (copy)."""

    @property
    @abstractmethod
    def Py(self) -> np.ndarray:
        """Algebraic-variable covariance P_y ∈ ℝⁿʸˣⁿʸ (copy)."""

    @abstractmethod
    def predict(
        self,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Time update; returns ``(x_pred, y_pred, P_pred)``.
        """

    @abstractmethod
    def update(
        self,
        ym: np.ndarray,
        u: np.ndarray | None,
        d: np.ndarray | None,
        p: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Measurement update; returns ``(x_hat, y_hat, P)``.
        """

    @abstractmethod
    def step(
        self,
        ym: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Combined step; returns ``(x_hat, y_hat, P)``.
        """
