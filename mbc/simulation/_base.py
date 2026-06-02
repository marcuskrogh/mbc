"""
Abstract base classes for all simulators and their parameter structures.

Hierarchy
---------
SimulatorParams (ABC)
    ContinuousDiscreteSDESimulatorParams
    ContinuousDiscreteSDAESimulatorParams

ContinuousDiscreteSimulator (ABC)
    ContinuousDiscreteSDESimulator

ContinuousDiscreteDAESimulator (ContinuousDiscreteSimulator, ABC)
    ContinuousDiscreteSDAESimulator
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ── Abstract parameter structure ──────────────────────────────────────────────


class SimulatorParams(ABC):
    """
    Abstract base class for all simulator parameter structures.

    Concrete subclasses are plain :func:`dataclasses.dataclass` objects that
    group the algorithm-specific hyper-parameters for a particular simulator
    (integration step count, seed, Newton solver settings, etc.).  The sampling
    interval ``Ts`` is read from the model's ``Ts`` property and is *not* part
    of the params structure.
    """


# ── Continuous-discrete simulator ─────────────────────────────────────────────


class ContinuousDiscreteSimulator(ABC):
    """
    Abstract base class for continuous-discrete SDE simulators.

    Every implementation must provide:

    Methods
    -------
    step(x, u, d, p, t) → x_next
        Simulate one measurement interval from t to t + Ts.
    simulate(x0, U, D, P, t0=0.0) → X
        Simulate over a full horizon of T measurement intervals.
    """

    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Simulate one measurement interval from t to t + Ts.

        Parameters
        ----------
        x : (nx,) ndarray  — state at time t.
        u : (nu,) ndarray  — control input over [t, t+Ts].
        d : (nd,) ndarray  — disturbance over [t, t+Ts].
        p : (nparams,) ndarray  — parameter vector.
        t : float          — current time.

        Returns
        -------
        x_next : (nx,) state at t + Ts (one realisation).
        """

    @abstractmethod
    def simulate(
        self,
        x0: np.ndarray,
        U: np.ndarray,
        D: np.ndarray,
        P: np.ndarray,
        t0: float = 0.0,
    ) -> np.ndarray:
        """
        Simulate over a full horizon of T measurement intervals.

        Parameters
        ----------
        x0 : (nx,) ndarray
            Initial state at time t0.
        U : (T, nu) ndarray
            Input trajectory; U[k] is applied over [t_k, t_{k+1}].
        D : (T, nd) ndarray
            Disturbance trajectory; D[k] applies over [t_k, t_{k+1}].
        P : (T, nparams) ndarray
            Parameter trajectory; P[k] applies over [t_k, t_{k+1}].
        t0 : float, optional
            Start time.  Default: 0.

        Returns
        -------
        X : (T+1, nx) ndarray
            State trajectory where X[0] = x0 and X[k+1] = step(X[k], ...).
        """


# ── Continuous-discrete DAE simulator ─────────────────────────────────────────


class ContinuousDiscreteDAESimulator(ABC):
    """
    Abstract base class for continuous-discrete SDAE simulators.

    Extends the simulation interface with algebraic variables ``y``.

    Methods
    -------
    step(x, y, u, d, p, t) → (x_next, y_next)
        Simulate one measurement interval from t to t + Ts.
    simulate(x0, y0, U, D, P, t0=0.0) → (X, Y)
        Simulate over a full horizon of T measurement intervals.
    """

    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate one measurement interval from t to t + Ts.

        Parameters
        ----------
        x : (nx,) ndarray  — differential state at time t.
        y : (ny,) ndarray  — algebraic state at time t (consistent).
        u : (nu,) ndarray  — control input over [t, t+Ts].
        d : (nd,) ndarray  — disturbance over [t, t+Ts].
        p : (nparams,) ndarray  — parameter vector.
        t : float          — current time.

        Returns
        -------
        x_next : (nx,) differential state at t + Ts.
        y_next : (ny,) algebraic state at t + Ts (consistent).
        """

    @abstractmethod
    def simulate(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        U: np.ndarray,
        D: np.ndarray,
        P: np.ndarray,
        t0: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate over a full horizon of T measurement intervals.

        Parameters
        ----------
        x0 : (nx,) ndarray
            Initial differential state at t0.
        y0 : (ny,) ndarray
            Initial algebraic state (consistent with g(x0, y0, ...) = 0).
        U : (T, nu) ndarray
            Input trajectory.
        D : (T, nd) ndarray
            Disturbance trajectory.
        P : (T, nparams) ndarray
            Parameter trajectory.
        t0 : float, optional
            Start time.  Default: 0.

        Returns
        -------
        X : (T+1, nx) ndarray  — differential state trajectory.
        Y : (T+1, ny) ndarray  — algebraic state trajectory.
        """
