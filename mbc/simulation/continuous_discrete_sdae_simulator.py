"""
Numerical integrator for continuous-discrete SDAE models
(ControlToolbox §SDAE — *Numerical Integration: Implicit-Explicit Method*).

The SDAE model is

    dx(t) = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),
    dw ~ N(0, I dt),
    0 = g(x, y, u, d, p, t).

The interval ``[t_k, t_{k+1}]`` is divided into ``N`` equidistant sub-steps
of size ``Δt = Ts / N`` with conventions

    x_{k,n} ≈ x(t_k + n Δt),   y_{k,n} ≈ y(t_k + n Δt),
    u_{k,n} = u_k,             d_{k,n} = d_k,            (zero-order hold)
    Δω_{k,n} = z_{k,n} √Δt,    z_{k,n} ~ N(0, I).

Implicit-Explicit Update
------------------------
Drift and algebraic constraint are evaluated at the *next* sub-step
(implicit); diffusion is explicit.  Combined variable
``z_{k,i} = (x_{k,i}, y_{k,i})``.  At each sub-step solve

    R(z_{k,n+1}) = [
        x_{k,n+1} − x_{k,n} − f(x_{k,n+1}, y_{k,n+1}, u_k, d_k, p, t_{k,n+1}) Δt
                            − sigma(x_{k,n}, y_{k,n}, u_k, d_k, p, t_{k,n}) Δω_{k,n};
        g(x_{k,n+1}, y_{k,n+1}, p)
    ] = 0

with residual Jacobian

    ∂R/∂z = [
        I − (∂f/∂x) Δt,    −(∂f/∂y) Δt;
        ∂g/∂x,              ∂g/∂y
    ]

evaluated at (x_{k,n+1}, y_{k,n+1}).  Roots are computed by Newton's method.

For consistent initial conditions the user-provided ``y0`` must satisfy
``g(x0, y0, ...) = 0``.  No explicit-explicit variant exists — the SDAE
always requires the Newton solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import ContinuousDiscreteSDAE
from .._utils import _newton_solve
from ._base import ContinuousDiscreteDAESimulator, SimulatorParams


# ── Parameter structure ───────────────────────────────────────────────────────


@dataclass
class ContinuousDiscreteSDAESimulatorParams(SimulatorParams):
    """
    Algorithm parameters for :class:`ContinuousDiscreteSDAESimulator`.

    Parameters
    ----------
    n_steps : int
        Number of integration sub-steps per interval.  Default: 10.
    newton_tol : float
        Newton convergence tolerance.  Default: 1e-10.
    newton_max_iter : int
        Maximum Newton iterations per sub-step.  Default: 50.
    seed : int or None
        Random seed for reproducibility.  Default: None.
    """
    n_steps: int = 10
    newton_tol: float = 1e-10
    newton_max_iter: int = 50
    seed: int | None = None


# ── Simulator ─────────────────────────────────────────────────────────────────


class ContinuousDiscreteSDAESimulator(ContinuousDiscreteDAESimulator):
    """
    Implicit-explicit Euler-Maruyama simulator for SDAE models
    (ControlToolbox §SDAE).

    Parameters
    ----------
    model : ContinuousDiscreteSDAE
        The nonlinear SDAE system to simulate.  Must expose a ``Ts``
        property giving the measurement sampling interval (seconds).
    params : ContinuousDiscreteSDAESimulatorParams, optional
        Algorithm parameter struct.  Pass to control ``n_steps``, Newton
        solver settings, and random seed.
    """

    def __init__(
        self,
        model: ContinuousDiscreteSDAE,
        params: ContinuousDiscreteSDAESimulatorParams | None = None,
    ) -> None:
        if params is None:
            params = ContinuousDiscreteSDAESimulatorParams()

        self._model = model
        self._Ts = float(model.Ts)
        self._n_steps = int(params.n_steps)
        self._newton_tol = params.newton_tol
        self._newton_max_iter = params.newton_max_iter
        self._rng = np.random.default_rng(params.seed)

    def _consistent_y(
        self,
        x: np.ndarray,
        y_init: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Solve g(x, y, ...) = 0 for y by Newton iteration."""
        return _newton_solve(
            residual=lambda y_: self._model.g(x, y_, u, d, p, t),
            jacobian=lambda y_: self._model.dgdy(x, y_, u, d, p, t),
            x0=y_init,
            tol=self._newton_tol,
            max_iter=self._newton_max_iter,
        )

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
        Simulate one measurement interval from t to t + Ts using the
        implicit-explicit Newton scheme.

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
        h = self._Ts / self._n_steps
        sqrt_h = np.sqrt(h)
        model = self._model
        nx = x.shape[0]
        ny = y.shape[0]

        x_cur = x.copy()
        y_cur = y.copy()
        t_cur = t

        for _ in range(self._n_steps):
            # Diffusion evaluated explicitly at the start of the sub-step
            sigma_val = model.sigma(x_cur, y_cur, u, d, p, t_cur)  # (nx, nw)
            nw = sigma_val.shape[1]
            xi = self._rng.standard_normal(nw)
            noise = sigma_val @ xi * sqrt_h

            t_next = t_cur + h
            rhs_x = x_cur + noise

            def residual(z: np.ndarray) -> np.ndarray:
                xn = z[:nx]
                yn = z[nx:]
                f_val = model.f(xn, yn, u, d, p, t_next)
                g_val = model.g(xn, yn, u, d, p, t_next)
                return np.concatenate([
                    xn - rhs_x - f_val * h,
                    g_val,
                ])

            def jacobian(z: np.ndarray) -> np.ndarray:
                xn = z[:nx]
                yn = z[nx:]
                dfdx = model.dfdx(xn, yn, u, d, p, t_next)
                dfdy = model.dfdy(xn, yn, u, d, p, t_next)
                dgdx = model.dgdx(xn, yn, u, d, p, t_next)
                dgdy = model.dgdy(xn, yn, u, d, p, t_next)
                top = np.concatenate([np.eye(nx) - dfdx * h, -dfdy * h], axis=1)
                bot = np.concatenate([dgdx, dgdy], axis=1)
                return np.concatenate([top, bot], axis=0)

            z0 = np.concatenate([x_cur, y_cur])
            z_new = _newton_solve(
                residual, jacobian, z0,
                tol=self._newton_tol, max_iter=self._newton_max_iter,
            )
            x_cur = z_new[:nx]
            y_cur = z_new[nx:]
            t_cur = t_next

        return x_cur, y_cur

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
        T = U.shape[0]
        nx = x0.shape[0]
        ny = y0.shape[0]
        X = np.empty((T + 1, nx))
        Y = np.empty((T + 1, ny))
        X[0] = x0.copy()
        Y[0] = y0.copy()
        t_cur = t0
        for k in range(T):
            X[k + 1], Y[k + 1] = self.step(X[k], Y[k], U[k], D[k], P[k], t_cur)
            t_cur += self._Ts
        return X, Y
