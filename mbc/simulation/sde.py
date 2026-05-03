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
        if scheme not in ("EE", "IE"):
            raise ValueError(f"Unknown scheme '{scheme}'; choose 'EE' or 'IE'.")
        self._model = model
        self._dt = dt
        self._n_steps = n_steps
        self._scheme = scheme
        self._rng = np.random.default_rng(seed)
        # Pre-compute Cholesky factor of Q_c for noise generation: L L^T = Q_c
        Q_c = np.asarray(model.Q_c, dtype=float)
        try:
            self._L = np.linalg.cholesky(Q_c)
        except np.linalg.LinAlgError:
            self._L = np.linalg.cholesky(Q_c + 1e-14 * np.eye(Q_c.shape[0]))

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
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
        p : (nparams,) ndarray  — parameter vector.
        t : float          — current time.

        Returns
        -------
        x_next : (nx,) state at t + dt (one realisation).
        """
        h = self._dt / self._n_steps
        sqrt_h = np.sqrt(h)
        L = self._L
        nw = L.shape[0]
        nx = x.shape[0]

        x_cur = x.copy()
        t_cur = t

        for _ in range(self._n_steps):
            # Stochastic increment: dW ~ N(0, Q_c * h)
            # Generated as  L @ z * sqrt(h)  where z ~ N(0, I_nw)
            z = self._rng.standard_normal(nw)
            dW = L @ z * sqrt_h

            g_val = self._model.g(x_cur, u, d, p, t_cur)  # (nx, nw)
            noise = g_val @ dW                              # (nx,)

            if self._scheme == "EE":
                f_val = self._model.f(x_cur, u, d, p, t_cur)
                x_cur = x_cur + f_val * h + noise
            else:  # IE — implicit drift, explicit diffusion
                # Solve: x_next = x_cur + f(x_next, ..., t+h) * h + noise
                # via Newton:  F(x_next) = x_next - x_cur - f(x_next)*h - noise = 0
                x_next = x_cur.copy()
                for _ in range(50):
                    f_val = self._model.f(x_next, u, d, p, t_cur + h)
                    F = x_next - x_cur - f_val * h - noise
                    if np.linalg.norm(F) < 1e-12:
                        break
                    J = np.eye(nx) - h * self._model.dfdx(x_next, u, d, p, t_cur + h)
                    x_next = x_next - np.linalg.solve(J, F)
                x_cur = x_next

            t_cur += h

        return x_cur

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
        T = U.shape[0]
        nx = x0.shape[0]
        X = np.empty((T + 1, nx))
        X[0] = x0.copy()
        t_cur = t0
        for k in range(T):
            X[k + 1] = self.step(X[k], U[k], D[k], P[k], t_cur)
            t_cur += self._dt
        return X
