"""
SDE sub-step kernels and ensemble propagation shared by the continuous-discrete
particle-based state estimators (UKF, EnKF, PF).

Each particle/sigma-point is an independent trajectory through the SDE.
The integration scheme is controlled by :class:`IntegrationScheme`:

``EULER``   — Explicit Euler-Maruyama (EE).  Drift and diffusion evaluated at
              the current sub-step.
``IMPLICIT_EULER`` — Implicit-Explicit (IE).  Drift implicit (Newton), diffusion
              explicit at the current sub-step.

The Wiener increment ``dw`` supplied to each sub-step callable is the
*pre-scaled* increment ``z √h`` where ``z ~ N(0, I_nw)`` and ``h`` is the
sub-step size.  For deterministic sigma-points (UKF) a structured ``dw`` is
passed directly.
"""

from __future__ import annotations

import numpy as np

from .._utils import _newton_solve


# ── Single-particle SDE sub-step kernels ──────────────────────────────────────


class _EESubstep:
    """Explicit Euler-Maruyama SDE sub-step (drift and diffusion explicit)."""

    def __init__(self, model) -> None:
        self._m = model

    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
        h: float,
        dw: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        x   : (nx,) current state.
        u, d, p, t : ZOH inputs, disturbance, parameters, current time.
        h   : sub-step size.
        dw  : (nw,) pre-scaled Wiener increment ``z √h``.

        Returns
        -------
        x_next : (nx,)
        """
        return x + h * self._m.f(x, u, d, p, t) + self._m.sigma(x, u, d, p, t) @ dw


class _IESubstep:
    """
    Implicit-Explicit Euler-Maruyama SDE sub-step.

    Drift evaluated at the next sub-step (implicit, Newton); diffusion
    evaluated at the current sub-step (explicit).
    """

    def __init__(self, model, newton_tol: float, newton_max_iter: int) -> None:
        self._m = model
        self._tol = newton_tol
        self._mi = newton_max_iter

    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None,
        t: float,
        h: float,
        dw: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        x   : (nx,) current state.
        u, d, p, t : ZOH inputs, disturbance, parameters, current time.
        h   : sub-step size.
        dw  : (nw,) pre-scaled Wiener increment ``z √h``.

        Returns
        -------
        x_next : (nx,)
        """
        rhs = x + self._m.sigma(x, u, d, p, t) @ dw   # diffusion explicit at t
        t1 = t + h

        def residual(xk: np.ndarray) -> np.ndarray:
            return xk - rhs - h * self._m.f(xk, u, d, p, t1)

        def jacobian(xk: np.ndarray) -> np.ndarray:
            return np.eye(len(x)) - h * self._m.dfdx(xk, u, d, p, t1)

        return _newton_solve(residual, jacobian, x.copy(), self._tol, self._mi)


# ── Generic ensemble propagator ───────────────────────────────────────────────


def _propagate_ensemble(
    substep: _EESubstep | _IESubstep,
    X: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    p: np.ndarray | None,
    t: float,
    h: float,
    n_steps: int,
    rng: np.random.Generator,
    nw: int,
) -> np.ndarray:
    """
    Propagate an ensemble of N particles through ``n_steps`` SDE sub-steps.

    Each particle receives an independent Wiener increment
    ``dw^{(i)} = z^{(i)} √h``, ``z^{(i)} ~ N(0, I_nw)``, drawn fresh at
    every sub-step.

    Parameters
    ----------
    substep  : callable — single-particle SDE sub-step (``_EESubstep`` or
               ``_IESubstep``).
    X        : (nx, N) ensemble at time ``t``.
    u, d, p  : ZOH input, disturbance, parameter vector.
    t        : current time.
    h        : sub-step size.
    n_steps  : number of sub-steps.
    rng      : NumPy ``Generator`` for the Wiener increments.
    nw       : noise dimension (columns of ``sigma``).

    Returns
    -------
    X_next : (nx, N) ensemble after ``n_steps`` sub-steps.
    """
    sqrt_h = np.sqrt(h)
    for _ in range(n_steps):
        X_new = np.empty_like(X)
        for i in range(X.shape[1]):
            dw = rng.standard_normal(nw) * sqrt_h
            X_new[:, i] = substep(X[:, i], u, d, p, t, h, dw)
        X = X_new
        t += h
    return X


# ── Measurement evaluator ─────────────────────────────────────────────────────


def _ensemble_measurements(
    model,
    X: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    p: np.ndarray | None,
    t: float = 0.0,
) -> np.ndarray:
    """
    Evaluate the measurement function at every ensemble member.

    Returns ``Z`` with ``Z[:, i] = hm(X[:, i], u, d, p, t)``.

    Parameters
    ----------
    model   : ContinuousDiscreteSDE
    X       : (nx, N) ensemble.
    u, d, p : input, disturbance, parameter vectors.
    t       : evaluation time (default 0.0).

    Returns
    -------
    Z : (nym, N) ndarray of predicted measurements per particle.
    """
    return np.column_stack([
        model.hm(X[:, i], u, d, p, t) for i in range(X.shape[1])
    ])
