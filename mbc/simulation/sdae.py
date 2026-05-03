"""
Euler-Maruyama simulator for continuous-discrete SDAE models (Ph.D. Ch. 6).

Extends ``SDESimulator`` to handle differential-algebraic systems:

    dx = f(x, z, u, d, t) dt + g(x, z, u, d, t) dw
    0  = l(x, z, u, d, t)

At each Euler-Maruyama sub-step:
  1. The Euler drift update is applied to x.
  2. The algebraic constraint ``l(x, z, u, d, t) = 0`` is solved for z
     via :func:`_newton_solve`.
  3. Diffusion noise is added.

Reference:  Ph.D. thesis, Ch. 6.
"""

from __future__ import annotations

import numpy as np

from ..models import ContinuousDiscreteDAEModel
from .._utils import _newton_solve


class SDAESimulator:
    """
    Euler-Maruyama simulator for continuous-discrete SDAE models (Ph.D. Ch. 6).

    Parameters
    ----------
    model : ContinuousDiscreteDAEModel
        The nonlinear SDAE system to simulate.
    dt : float
        Measurement sampling interval (seconds).
    n_steps : int, optional
        Number of Euler-Maruyama sub-steps per interval.  Default: 10.
    scheme : {"EE", "IE"}, optional
        Integration scheme.  Default: ``"EE"``.
    newton_tol : float, optional
        Convergence tolerance for the Newton solver on ``l = 0``.
        Default: 1e-10.
    newton_max_iter : int, optional
        Maximum Newton iterations per sub-step.  Default: 50.
    seed : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: ContinuousDiscreteDAEModel,
        dt: float,
        n_steps: int = 10,
        scheme: str = "EE",
        newton_tol: float = 1e-10,
        newton_max_iter: int = 50,
        seed: int | None = None,
    ) -> None:
        if scheme not in ("EE", "IE"):
            raise ValueError(f"Unknown scheme '{scheme}'; choose 'EE' or 'IE'.")
        self._model = model
        self._dt = dt
        self._n_steps = n_steps
        self._scheme = scheme
        self._newton_tol = newton_tol
        self._newton_max_iter = newton_max_iter
        self._rng = np.random.default_rng(seed)

    def _solve_constraint(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Solve l(x, z, u, d, p, t) = 0 for z via :func:`_newton_solve`.

        Uses ``z`` as the initial guess and iterates until
        ``‖l‖ < newton_tol`` or ``newton_max_iter`` steps have been taken.
        """
        return _newton_solve(
            residual=lambda z_: self._model.l(x, z_, u, d, p, t),
            jacobian=lambda z_: self._model.dldz(x, z_, u, d, p, t),
            x0=z,
            tol=self._newton_tol,
            max_iter=self._newton_max_iter,
        )

    def step(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate one measurement interval from t to t + dt.

        Parameters
        ----------
        x : (nx,) ndarray  — differential state at time t.
        z : (nz,) ndarray  — algebraic state at time t (consistent).
        u : (nu,) ndarray  — control input over [t, t+dt].
        d : (nd,) ndarray  — disturbance over [t, t+dt].
        p : (nparams,) ndarray  — parameter vector.
        t : float          — current time.

        Returns
        -------
        x_next : (nx,) differential state at t + dt.
        z_next : (nz,) algebraic state at t + dt (consistent).
        """
        h = self._dt / self._n_steps
        sqrt_h = np.sqrt(h)
        nx = x.shape[0]

        x_cur = x.copy()
        z_cur = z.copy()
        t_cur = t

        for _ in range(self._n_steps):
            # Diffusion: sigma encodes continuous-time noise magnitude; dw ~ N(0, I dt)
            sigma_val = self._model.sigma(x_cur, z_cur, u, d, p, t_cur)  # (nx, nw)
            nw = sigma_val.shape[1]
            xi = self._rng.standard_normal(nw)
            dW = sigma_val @ xi * sqrt_h

            if self._scheme == "EE":
                # 1. Euler drift step on x
                f_val = self._model.f(x_cur, z_cur, u, d, p, t_cur)
                x_next = x_cur + f_val * h
                # 2. Solve algebraic constraint at the new x
                z_next = self._solve_constraint(x_next, z_cur, u, d, p, t_cur + h)
                # 3. Add diffusion noise
                x_next = x_next + dW
                # Re-solve constraint after noise perturbation
                z_next = self._solve_constraint(x_next, z_next, u, d, p, t_cur + h)
            else:  # IE — implicit drift, explicit diffusion
                # Noise term first (explicit)
                noise = dW
                # Solve coupled system:
                #   F(x_next) = x_next − (x_cur + noise) − f(x_next, z_next,…) h = 0
                #   l(x_next, z_next, …) = 0
                # via alternating Newton: z given x (inner), then x (outer).
                rhs = x_cur + noise
                t_next = t_cur + h
                z_ref = [z_cur.copy()]

                def residual(xn: np.ndarray) -> np.ndarray:
                    z_ref[0] = self._solve_constraint(xn, z_ref[0], u, d, p, t_next)
                    return xn - rhs - self._model.f(xn, z_ref[0], u, d, p, t_next) * h

                def jacobian(xn: np.ndarray) -> np.ndarray:
                    return (np.eye(nx)
                            - h * self._model.dfdx(xn, z_ref[0], u, d, p, t_next))

                x_next = _newton_solve(
                    residual, jacobian, x_cur.copy(),
                    tol=self._newton_tol, max_iter=self._newton_max_iter,
                )
                z_next = self._solve_constraint(x_next, z_ref[0], u, d, p, t_next)

            x_cur = x_next
            z_cur = z_next
            t_cur += h

        return x_cur, z_cur

    def simulate(
        self,
        x0: np.ndarray,
        z0: np.ndarray,
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
        z0 : (nz,) ndarray
            Initial algebraic state (consistent with l(x0, z0, ...) = 0).
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
        Z : (T+1, nz) ndarray  — algebraic state trajectory.
        """
        T = U.shape[0]
        nx = x0.shape[0]
        nz = z0.shape[0]
        X = np.empty((T + 1, nx))
        Z = np.empty((T + 1, nz))
        X[0] = x0.copy()
        Z[0] = z0.copy()
        t_cur = t0
        for k in range(T):
            X[k + 1], Z[k + 1] = self.step(X[k], Z[k], U[k], D[k], P[k], t_cur)
            t_cur += self._dt
        return X, Z
