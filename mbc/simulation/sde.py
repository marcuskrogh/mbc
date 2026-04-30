"""
Euler-Maruyama simulator for continuous-discrete SDE models (Ph.D. Ch. 5).

Implements two numerical integration schemes for the Itô SDE:

    dx = f(x, u, d, t) dt + g(x, u, d, t) dw,   w ~ N(0, Q_c)

**Explicit-Explicit (EE) scheme** — both drift and diffusion evaluated at
the beginning of each sub-step:

    x_{k+1} = x_k + f(x_k, u, d, t_k) h + g(x_k, u, d, t_k) √h w_k

**Implicit-Explicit (IE) scheme** — drift evaluated implicitly at t_{k+1},
diffusion evaluated explicitly:

    x_{k+1} = x_k + f(x_{k+1}, u, d, t_{k+1}) h + g(x_k, u, d, t_k) √h w_k

where ``h = dt / n_steps`` is the sub-step size and w_k ~ N(0, Q_c).

The implicit step for IE is solved by fixed-point / Newton iteration.

Reference:  Ph.D. thesis, Ch. 5.
"""

from __future__ import annotations

import numpy as np

from ..models import ContinuousDiscreteModel


class SDESimulator:
    """
    Euler-Maruyama simulator for continuous-discrete SDE models (Ph.D. Ch. 5).

    Parameters
    ----------
    model : ContinuousDiscreteModel
        The nonlinear SDE system to simulate.
    dt : float
        Measurement sampling interval (seconds).  This defines the
        coarse time step between observations.
    n_steps : int, optional
        Number of Euler-Maruyama sub-steps per measurement interval.
        Default: 10.
    scheme : {"EE", "IE"}, optional
        Integration scheme.  ``"EE"`` = Explicit-Explicit (default),
        ``"IE"`` = Implicit-Explicit.
    seed : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: ContinuousDiscreteModel,
        dt: float,
        n_steps: int = 10,
        scheme: str = "EE",
        seed: int | None = None,
    ) -> None:
        raise NotImplementedError(
            "SDESimulator.__init__ is not yet implemented."
        )

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Simulate one measurement interval from t to t + dt.

        Applies ``n_steps`` Euler-Maruyama sub-steps with step size
        ``h = dt / n_steps``.  The input u and disturbance d are held
        constant over the interval (zero-order hold).

        Parameters
        ----------
        x : (nx,) ndarray  — state at time t.
        u : (nu,) ndarray  — control input over [t, t+dt].
        d : (nd,) ndarray  — disturbance over [t, t+dt].
        t : float          — current time.

        Returns
        -------
        x_next : (nx,) state at t + dt (one realisation).

        Raises
        ------
        NotImplementedError
            Always — to be implemented in a future commit.
        """
        raise NotImplementedError(
            "SDESimulator.step is not yet implemented."
        )

    def simulate(
        self,
        x0: np.ndarray,
        U: np.ndarray,
        D: np.ndarray,
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
        t0 : float, optional
            Start time.  Default: 0.

        Returns
        -------
        X : (T+1, nx) ndarray
            State trajectory where X[0] = x0 and X[k+1] = step(X[k], ...).

        Raises
        ------
        NotImplementedError
            Always — to be implemented in a future commit.
        """
        raise NotImplementedError(
            "SDESimulator.simulate is not yet implemented."
        )
