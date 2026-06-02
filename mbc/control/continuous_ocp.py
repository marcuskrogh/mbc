"""
Continuous-discrete nonlinear OCP (``ContinuousOCP``).

Implements the ControlToolbox §Economic Model Predictive Control direct-
simultaneous formulation for continuous-discrete SDE / SDAE plant models.

The OCP solves a finite-horizon NLP supporting any convex combination of:

* Setpoint tracking          phi_z       — ‖z(t) − z̄(t)‖²_{Q_z}
* Stage input cost           phi_R       — ‖u_k‖²_R          (built-in)
* Terminal tracking cost     phi_P       — ‖z_M − z_ref‖²_P  (built-in)
* Input ROM penalty          phi_{Δu}    — ‖Δu_k‖²_{Q_du} T_s
* Input economy              phi_{u,eco} — p_{u,eco}^T u_k T_s
* General Lagrange / Mayer    phi_{P,eco}
* Soft-constraint exact penalty  phi_pq

Hard constraints:
* Input box              u_min  ≤ u_k ≤ u_max
* Input rate-of-movement du_min ≤ Δu_k ≤ du_max

Soft (slacked) constraints with combined L1 + L2 exact penalty:
    x_min − p ≤ x_n ≤ x_max + q,   p, q ≥ 0
    z_min − s ≤ z_n ≤ z_max + s,   s ≥ 0

Discretisation
--------------
Each control interval ``[t_k, t_{k+1}]`` is split into ``n_steps``
equidistant sub-steps of size ``Δt = T_s / n_steps``.  Two integration
schemes are supported (selectable via ``scheme``):

``IntegrationScheme.IMPLICIT_EULER`` (default)
    Drift evaluated at the *next* sub-step — the dynamics residual is

        x_{n+1} − x_n − f(x_{n+1}, y_{n+1}, u_k, d_k, θ) Δt = 0.

    Recommended for stiff drift dynamics; guarantees positive-definiteness
    of the resulting sensitivity.

``IntegrationScheme.EXPLICIT_EULER``
    Drift evaluated at the *current* sub-step — the dynamics residual is

        x_{n+1} − x_n − f(x_n, y_n, u_k, d_k, θ) Δt = 0.

    Recommended for non-stiff dynamics; not supported for SDAE models.

The Lagrange integral uses the right-rectangular rule (evaluated at the
*next* sub-step) regardless of integration scheme.
"""

from __future__ import annotations

import numpy as np

from ..models import ContinuousDiscreteSDAE, ContinuousDiscreteSDE
from ..estimation._base import IntegrationScheme
from ._base import OCP
from .nlp_solver import (
    NLPConstraint,
    NLPProblem,
    NLPScalingPolicy,
    NLPSolverBackend,
    make_nlp_backend,
)


# ── Internal helpers ──────────────────────────────────────────────────────────


class _DecisionLayout:
    """
    Layout of the flat NLP decision vector z.

    Order:
        [u_0, …, u_{N-1},                               (control inputs)
         x_0, …, x_M,                                   (differential state at sub-steps)
         y_0, …, y_M,                                   (algebraic state, SDAE only)
         px_lo_0, …, px_lo_M, px_hi_0, …, px_hi_M,     (state slacks, if soft x)
         pz_0, …, pz_M]                                 (output slacks, if soft z)

    M = N * n_steps is the total number of sub-step intervals; states are
    stored at M + 1 grid points.
    """

    def __init__(
        self,
        N: int,
        n_steps: int,
        nx: int,
        nu: int,
        ny: int,
        nz: int,
        soft_x: bool,
        soft_z: bool,
    ) -> None:
        self.N = N
        self.n_steps = n_steps
        self.M = N * n_steps
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nz = nz
        self.soft_x = soft_x
        self.soft_z = soft_z

        offset = 0
        self.u_off = offset
        self.u_size = N * nu
        offset += self.u_size

        self.x_off = offset
        self.x_size = (self.M + 1) * nx
        offset += self.x_size

        self.y_off = offset
        self.y_size = (self.M + 1) * ny
        offset += self.y_size

        self.px_lo_off = offset
        self.px_lo_size = (self.M + 1) * nx if soft_x else 0
        offset += self.px_lo_size
        self.px_hi_off = offset
        self.px_hi_size = (self.M + 1) * nx if soft_x else 0
        offset += self.px_hi_size

        self.pz_off = offset
        self.pz_size = (self.M + 1) * nz if soft_z else 0
        offset += self.pz_size

        self.total = offset

    def get_U(self, z: np.ndarray) -> np.ndarray:
        return z[self.u_off:self.u_off + self.u_size].reshape(self.N, self.nu)

    def get_X(self, z: np.ndarray) -> np.ndarray:
        return z[self.x_off:self.x_off + self.x_size].reshape(self.M + 1, self.nx)

    def get_Y(self, z: np.ndarray) -> np.ndarray:
        return z[self.y_off:self.y_off + self.y_size].reshape(self.M + 1, self.ny)

    def get_PX_lo(self, z: np.ndarray) -> np.ndarray:
        return z[self.px_lo_off:self.px_lo_off + self.px_lo_size].reshape(self.M + 1, self.nx)

    def get_PX_hi(self, z: np.ndarray) -> np.ndarray:
        return z[self.px_hi_off:self.px_hi_off + self.px_hi_size].reshape(self.M + 1, self.nx)

    def get_PZ(self, z: np.ndarray) -> np.ndarray:
        return z[self.pz_off:self.pz_off + self.pz_size].reshape(self.M + 1, self.nz)


# ── Continuous-discrete nonlinear OCP ─────────────────────────────────────────


class ContinuousOCP(OCP):
    """
    Continuous-discrete nonlinear OCP for SDE / SDAE plants
    (ControlToolbox §EMPC direct-simultaneous formulation).

    Decision variables are the inputs ``{u_k}_{k=0..N-1}`` together with the
    differential and algebraic states ``{x_n, y_n}_{n=0..M}`` at every
    sub-step (M = N · n_steps).

    Parameters
    ----------
    model : ContinuousDiscreteSDE or ContinuousDiscreteSDAE
        Plant model providing ``f``, ``g`` (SDAE only), ``gm``, ``hm``, and
        their Jacobians.
    N : int
        Prediction horizon (number of control intervals).
    scheme : IntegrationScheme, optional
        Integration scheme for the sub-step dynamics.
        :attr:`~IntegrationScheme.IMPLICIT_EULER` (default) is recommended
        for stiff dynamics; :attr:`~IntegrationScheme.EXPLICIT_EULER` is
        available for non-stiff SDE models (not supported for SDAE).
    R_stage : (nu, nu) ndarray, optional
        Built-in quadratic stage input cost ``‖u_k‖²_R · Δt`` per sub-step.
        Added to any user-supplied ``lagrange``.
    P_terminal : (nz, nz) ndarray, optional
        Built-in terminal tracking cost ``‖gm(x_M) − z_ref_M‖²_P``.
        Requires ``Q_z`` and ``z_ref`` to be set as well.
        Added to any user-supplied ``mayer``.
    lagrange : callable (t, x, y, u, theta) → float, optional
        General Lagrange (stage) cost.
    lagrange_jac : callable (t, x, y, u, theta) → (grad_x, grad_y, grad_u), optional
        Gradient of ``lagrange`` w.r.t. ``(x, y, u)``.
    mayer : callable (x, y, theta) → float, optional
        General Mayer (terminal) cost.
    mayer_jac : callable (x, y, theta) → (grad_x, grad_y), optional
        Gradient of ``mayer`` w.r.t. ``(x, y)``.
    Q_z : (nz, nz) ndarray, optional
        Tracking weight; activates ``phi_z = Σ_n ‖z_n − z̄_n‖²_{Q_z} Δt``.
    z_ref : (nz,) or (M+1, nz) ndarray, optional
        Constant or time-varying tracking reference.  Required with ``Q_z``
        or ``P_terminal``.
    Q_du : (nu, nu) ndarray, optional
        Quadratic ROM penalty on Δu_k = u_k − u_{k−1}.
    p_u_eco : (nu,) ndarray, optional
        Linear input cost ``p_u_eco^T u_k T_s``.
    u_min, u_max : (nu,) ndarray, optional
        Hard input box.
    du_min, du_max : (nu,) ndarray, optional
        Hard input rate-of-movement box.
    x_min, x_max : (nx,) ndarray, optional
        Soft state box.
    rho_x_1 : float, optional
        L1 penalty weight on state slacks (default 0).
    rho_x_2 : float, optional
        L2 penalty weight on state slacks (default 1e4).
    z_min, z_max : (nz,) ndarray, optional
        Soft output box.
    rho_z_1 : float, optional
        L1 penalty weight on output slacks (default 0).
    rho_z_2 : float, optional
        L2 penalty weight on output slacks (default 1e4).
    n_steps : int, optional
        Integration sub-steps per control interval.  Default: 10.
    solver : str or NLPSolverBackend, optional
        NLP backend.  Default: ``"SLSQP"``.
    solver_options : dict or None, optional
        Forwarded to the NLP solver.
    solver_scaling : NLPScalingPolicy or dict or None, optional
        Backend-agnostic scaling controls.
    dt : float or None, optional
        Sampling interval ``T_s``.  ``None`` → ``model.Ts`` if available,
        else ``1.0``.
    """

    def __init__(
        self,
        model: ContinuousDiscreteSDE,
        N: int,
        *,
        scheme: IntegrationScheme = IntegrationScheme.IMPLICIT_EULER,
        R_stage: np.ndarray | None = None,
        P_terminal: np.ndarray | None = None,
        lagrange: Callable[..., float] | None = None,
        lagrange_jac: Callable[..., tuple] | None = None,
        mayer: Callable[..., float] | None = None,
        mayer_jac: Callable[..., tuple] | None = None,
        Q_z: np.ndarray | None = None,
        z_ref: np.ndarray | None = None,
        Q_du: np.ndarray | None = None,
        p_u_eco: np.ndarray | None = None,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        du_min: np.ndarray | None = None,
        du_max: np.ndarray | None = None,
        x_min: np.ndarray | None = None,
        x_max: np.ndarray | None = None,
        rho_x_1: float = 0.0,
        rho_x_2: float = 1e4,
        z_min: np.ndarray | None = None,
        z_max: np.ndarray | None = None,
        rho_z_1: float = 0.0,
        rho_z_2: float = 1e4,
        n_steps: int = 10,
        solver: str | NLPSolverBackend = "SLSQP",
        solver_options: dict | None = None,
        solver_scaling: NLPScalingPolicy | dict | None = None,
        dt: float | None = None,
    ) -> None:
        if not isinstance(scheme, IntegrationScheme):
            raise TypeError(
                f"scheme must be an IntegrationScheme member, got {scheme!r}."
            )
        self._is_dae = isinstance(model, ContinuousDiscreteSDAE)
        if scheme is IntegrationScheme.EXPLICIT_EULER and self._is_dae:
            raise ValueError(
                "IntegrationScheme.EXPLICIT_EULER is not supported for SDAE models; "
                "use IntegrationScheme.IMPLICIT_EULER instead."
            )

        self._model = model
        self._scheme = scheme
        self._N = int(N)
        self._n_steps = int(n_steps)
        self._nx = int(model.nx)
        self._nu = int(model.nu)
        self._nd = int(model.nd)
        self._nz = int(model.nz)
        self._ny = int(model.ny) if self._is_dae else 0

        self._dt: float = (
            float(dt) if dt is not None else float(getattr(model, "Ts", 1.0))
        )
        self._h = self._dt / self._n_steps

        # Built-in R and P terms
        self._R_stage = (
            np.asarray(R_stage, dtype=float) if R_stage is not None else None
        )
        self._P_terminal = (
            np.asarray(P_terminal, dtype=float) if P_terminal is not None else None
        )

        self._lagrange = lagrange
        self._lagrange_jac = lagrange_jac
        self._mayer = mayer
        self._mayer_jac = mayer_jac
        self._Q_z = (
            np.asarray(Q_z, dtype=float) if Q_z is not None else None
        )
        if self._Q_z is not None and z_ref is None:
            raise ValueError("Q_z requires z_ref to be supplied as well.")
        if self._P_terminal is not None and z_ref is None:
            raise ValueError("P_terminal requires z_ref to be supplied as well.")
        self._z_ref = self._broadcast_zref(z_ref) if z_ref is not None else None
        self._Q_du = (
            np.asarray(Q_du, dtype=float) if Q_du is not None else None
        )
        self._p_u_eco = (
            np.asarray(p_u_eco, dtype=float) if p_u_eco is not None else None
        )

        self._u_min = self._asfloat(u_min)
        self._u_max = self._asfloat(u_max)
        self._du_min = self._asfloat(du_min)
        self._du_max = self._asfloat(du_max)

        self._x_min = self._asfloat(x_min)
        self._x_max = self._asfloat(x_max)
        self._z_min = self._asfloat(z_min)
        self._z_max = self._asfloat(z_max)
        self._rho_x_1 = float(rho_x_1)
        self._rho_x_2 = float(rho_x_2)
        self._rho_z_1 = float(rho_z_1)
        self._rho_z_2 = float(rho_z_2)
        self._has_soft_x = self._x_min is not None or self._x_max is not None
        self._has_soft_z = self._z_min is not None or self._z_max is not None

        self._solver_backend = make_nlp_backend(
            solver,
            solver_options=solver_options,
            scaling=solver_scaling,
        )

        self._layout = _DecisionLayout(
            N=self._N,
            n_steps=self._n_steps,
            nx=self._nx,
            nu=self._nu,
            ny=self._ny,
            nz=self._nz,
            soft_x=self._has_soft_x,
            soft_z=self._has_soft_z,
        )

    # ── OCP abstract properties ────────────────────────────────────────────

    @property
    def N(self) -> int:
        """Prediction horizon (number of control intervals)."""
        return self._N

    @property
    def nu(self) -> int:
        """Input dimension nᵘ."""
        return self._nu

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _asfloat(arr) -> np.ndarray | None:
        return np.asarray(arr, dtype=float) if arr is not None else None

    def _broadcast_zref(self, z_ref) -> np.ndarray:
        z_ref = np.asarray(z_ref, dtype=float)
        M = self._N * self._n_steps
        if z_ref.ndim == 1:
            return np.tile(z_ref, (M + 1, 1))
        if z_ref.shape == (M + 1, self._nz):
            return z_ref
        raise ValueError(
            f"z_ref must have shape ({self._nz},) or "
            f"({M + 1}, {self._nz}); got {z_ref.shape}."
        )

    def _f(self, x, y, u, d, p, t):
        return (
            self._model.f(x, y, u, d, p, t)
            if self._is_dae
            else self._model.f(x, u, d, p, t)
        )

    def _g(self, x, y, u, d, p, t):
        if self._is_dae:
            return self._model.g(x, y, u, d, p, t)
        return np.empty(0)

    def _gm(self, x, y, u, d, p, t):
        return (
            self._model.gm(x, y, u, d, p, t)
            if self._is_dae
            else self._model.gm(x, u, d, p, t)
        )

    # ── NLP construction ──────────────────────────────────────────────────────

    def _build_initial_guess(
        self,
        x0: np.ndarray,
        u_prev: np.ndarray | None,
        x_prev: np.ndarray | None,
        y_prev: np.ndarray | None,
    ) -> np.ndarray:
        L = self._layout
        z0 = np.zeros(L.total)

        if u_prev is not None and u_prev.shape == (self._N, self._nu):
            U_init = np.empty_like(u_prev)
            U_init[:-1] = u_prev[1:]
            U_init[-1] = u_prev[-1]
        else:
            U_init = np.zeros((self._N, self._nu))
        z0[L.u_off:L.u_off + L.u_size] = U_init.ravel()

        if x_prev is not None and x_prev.shape == (L.M + 1, self._nx):
            X_init = np.empty_like(x_prev)
            X_init[:L.M + 1 - self._n_steps] = x_prev[self._n_steps:]
            X_init[L.M + 1 - self._n_steps:] = x_prev[-1]
        else:
            X_init = np.tile(x0, (L.M + 1, 1))
        z0[L.x_off:L.x_off + L.x_size] = X_init.ravel()

        if self._is_dae:
            if y_prev is not None and y_prev.shape == (L.M + 1, self._ny):
                Y_init = np.empty_like(y_prev)
                Y_init[:L.M + 1 - self._n_steps] = y_prev[self._n_steps:]
                Y_init[L.M + 1 - self._n_steps:] = y_prev[-1]
            else:
                Y_init = np.zeros((L.M + 1, self._ny))
            z0[L.y_off:L.y_off + L.y_size] = Y_init.ravel()

        return z0

    def _objective(
        self,
        z: np.ndarray,
        x_hat: np.ndarray,
        d_traj: np.ndarray,
        u_prev_0: np.ndarray,
        p_theta: np.ndarray,
        t0: float,
    ) -> float:
        L = self._layout
        U = L.get_U(z)
        X = L.get_X(z)
        Y = L.get_Y(z)
        h = self._h
        Ts = self._dt

        total = 0.0

        # Lagrange (right-rectangular over all sub-steps)
        for n in range(L.M):
            k = n // self._n_steps
            u_k = U[k]
            d_k = d_traj[k]
            x_np1 = X[n + 1]
            y_np1 = Y[n + 1] if self._is_dae else np.empty(0)
            t_np1 = t0 + (n + 1) * h

            if self._lagrange is not None:
                total += float(self._lagrange(t_np1, x_np1, y_np1, u_k, p_theta)) * h

            if self._Q_z is not None:
                z_np1 = self._gm(x_np1, y_np1, u_k, d_k, p_theta, t_np1)
                e = z_np1 - self._z_ref[n + 1]
                total += float(e @ self._Q_z @ e) * h

            if self._R_stage is not None:
                total += float(u_k @ self._R_stage @ u_k) * h

        # Per-control-interval terms (ROM, input economy)
        for k in range(self._N):
            u_k = U[k]
            u_km1 = U[k - 1] if k > 0 else u_prev_0

            if self._Q_du is not None:
                du = u_k - u_km1
                total += float(du @ self._Q_du @ du) * Ts

            if self._p_u_eco is not None:
                total += float(self._p_u_eco @ u_k) * Ts

        # Mayer (terminal cost)
        x_M = X[L.M]
        y_M = Y[L.M] if self._is_dae else np.empty(0)
        u_last = U[self._N - 1]
        d_last = d_traj[self._N - 1]
        t_M = t0 + L.M * h

        if self._mayer is not None:
            total += float(self._mayer(x_M, y_M, p_theta))

        if self._P_terminal is not None and self._z_ref is not None:
            z_M = self._gm(x_M, y_M, u_last, d_last, p_theta, t_M)
            e = z_M - self._z_ref[L.M]
            total += float(e @ self._P_terminal @ e)

        # Soft-constraint exact penalty
        if self._has_soft_x:
            PX_lo = L.get_PX_lo(z)
            PX_hi = L.get_PX_hi(z)
            for n in range(L.M + 1):
                if self._x_min is not None:
                    p_n = PX_lo[n]
                    total += (self._rho_x_1 * p_n.sum() + self._rho_x_2 * (p_n @ p_n)) * h
                if self._x_max is not None:
                    q_n = PX_hi[n]
                    total += (self._rho_x_1 * q_n.sum() + self._rho_x_2 * (q_n @ q_n)) * h

        if self._has_soft_z:
            PZ = L.get_PZ(z)
            for n in range(L.M + 1):
                s_n = PZ[n]
                total += (self._rho_z_1 * s_n.sum() + self._rho_z_2 * (s_n @ s_n)) * h

        return total

    def _equality_constraints(
        self,
        z: np.ndarray,
        x_hat: np.ndarray,
        d_traj: np.ndarray,
        p_theta: np.ndarray,
        t0: float,
    ) -> np.ndarray:
        """
        Equality constraints (must equal zero):
        * x_0 − x_hat = 0
        * Dynamics residual per sub-step (scheme-dependent)
        * g(x_{n+1}, y_{n+1}, …) = 0  (SDAE only)
        * g(x_0, y_0, …) = 0          (SDAE algebraic IC consistency)
        """
        L = self._layout
        U = L.get_U(z)
        X = L.get_X(z)
        Y = L.get_Y(z)
        h = self._h
        implicit = self._scheme is IntegrationScheme.IMPLICIT_EULER

        residuals: list[np.ndarray] = [X[0] - x_hat]

        for n in range(L.M):
            k = n // self._n_steps
            u_k = U[k]
            d_k = d_traj[k]
            x_n = X[n]
            x_np1 = X[n + 1]

            if implicit:
                y_np1 = Y[n + 1] if self._is_dae else np.empty(0)
                t_np1 = t0 + (n + 1) * h
                f_val = self._f(x_np1, y_np1, u_k, d_k, p_theta, t_np1)
                residuals.append(x_np1 - x_n - f_val * h)
                if self._is_dae:
                    residuals.append(self._g(x_np1, y_np1, u_k, d_k, p_theta, t_np1))
            else:
                # EXPLICIT_EULER: drift at current sub-step
                t_n = t0 + n * h
                f_val = self._f(x_n, np.empty(0), u_k, d_k, p_theta, t_n)
                residuals.append(x_np1 - x_n - f_val * h)

        if self._is_dae:
            residuals.append(self._g(X[0], Y[0], U[0], d_traj[0], p_theta, t0))

        return np.concatenate(residuals)

    def _inequality_constraints(
        self,
        z: np.ndarray,
        u_prev_0: np.ndarray,
        d_traj: np.ndarray,
        p_theta: np.ndarray,
        t0: float,
    ) -> np.ndarray:
        """Inequality constraints (returned values must be ≥ 0)."""
        L = self._layout
        U = L.get_U(z)
        X = L.get_X(z)
        Y = L.get_Y(z)
        h = self._h

        out: list[float] = []

        if self._du_min is not None or self._du_max is not None:
            for k in range(self._N):
                u_k = U[k]
                u_km1 = U[k - 1] if k > 0 else u_prev_0
                du = u_k - u_km1
                if self._du_min is not None:
                    out.extend(du - self._du_min)
                if self._du_max is not None:
                    out.extend(self._du_max - du)

        if self._has_soft_x:
            PX_lo = L.get_PX_lo(z)
            PX_hi = L.get_PX_hi(z)
            for n in range(L.M + 1):
                if self._x_min is not None:
                    out.extend(X[n] + PX_lo[n] - self._x_min)
                if self._x_max is not None:
                    out.extend(self._x_max + PX_hi[n] - X[n])

        if self._has_soft_z:
            PZ = L.get_PZ(z)
            for n in range(L.M + 1):
                k = min(n // self._n_steps, self._N - 1)
                u_k = U[k]
                d_k = d_traj[k]
                t_n = t0 + n * h
                y_n = Y[n] if self._is_dae else np.empty(0)
                z_n = self._gm(X[n], y_n, u_k, d_k, p_theta, t_n)
                if self._z_min is not None:
                    out.extend(z_n + PZ[n] - self._z_min)
                if self._z_max is not None:
                    out.extend(self._z_max + PZ[n] - z_n)

        return np.asarray(out, dtype=float)

    # ── Analytical Jacobians ──────────────────────────────────────────────────

    def _equality_constraint_jac(
        self,
        z: np.ndarray,
        x_hat: np.ndarray,
        d_traj: np.ndarray,
        p_theta: np.ndarray,
        t0: float,
    ) -> np.ndarray:
        """
        Dense Jacobian of the equality constraints w.r.t. the NLP decision
        vector ``z``.  Dispatches to the implicit- or explicit-Euler form
        depending on ``self._scheme``.
        """
        L = self._layout
        U = L.get_U(z)
        X = L.get_X(z)
        Y = L.get_Y(z)
        h = self._h
        nx, nu, ny = self._nx, self._nu, self._ny
        M = L.M
        implicit = self._scheme is IntegrationScheme.IMPLICIT_EULER

        if self._is_dae:
            n_eq = nx + M * nx + M * ny + ny
        else:
            n_eq = nx + M * nx

        J = np.zeros((n_eq, L.total))

        # Block 0: x_0 − x_hat = 0
        J[0:nx, L.x_off:L.x_off + nx] = np.eye(nx)
        row = nx

        for n in range(M):
            k = n // self._n_steps
            u_k = U[k]
            d_k = d_traj[k]

            x_np1_col = L.x_off + (n + 1) * nx
            x_n_col = L.x_off + n * nx
            u_k_col = L.u_off + k * nu

            if implicit:
                x_np1 = X[n + 1]
                y_np1 = Y[n + 1] if self._is_dae else np.empty(0)
                t_np1 = t0 + (n + 1) * h

                if self._is_dae:
                    dfdx_val = self._model.dfdx(x_np1, y_np1, u_k, d_k, p_theta, t_np1)
                    dfdu_val = self._model.dfdu(x_np1, y_np1, u_k, d_k, p_theta, t_np1)
                    dfdy_val = self._model.dfdy(x_np1, y_np1, u_k, d_k, p_theta, t_np1)
                else:
                    dfdx_val = self._model.dfdx(x_np1, u_k, d_k, p_theta, t_np1)
                    dfdu_val = self._model.dfdu(x_np1, u_k, d_k, p_theta, t_np1)

                # x_{n+1} − x_n − f(x_{n+1}, …) h = 0
                J[row:row + nx, x_np1_col:x_np1_col + nx] = np.eye(nx) - dfdx_val * h
                J[row:row + nx, x_n_col:x_n_col + nx] = -np.eye(nx)
                J[row:row + nx, u_k_col:u_k_col + nu] = -dfdu_val * h
                if self._is_dae:
                    y_np1_col = L.y_off + (n + 1) * ny
                    J[row:row + nx, y_np1_col:y_np1_col + ny] = -dfdy_val * h
                row += nx

                if self._is_dae:
                    dgdx_val = self._model.dgdx(x_np1, y_np1, u_k, d_k, p_theta, t_np1)
                    dgdy_val = self._model.dgdy(x_np1, y_np1, u_k, d_k, p_theta, t_np1)
                    y_np1_col = L.y_off + (n + 1) * ny
                    J[row:row + ny, x_np1_col:x_np1_col + nx] = dgdx_val
                    J[row:row + ny, y_np1_col:y_np1_col + ny] = dgdy_val
                    row += ny
            else:
                # EXPLICIT_EULER: x_{n+1} − x_n − f(x_n, …) h = 0
                x_n = X[n]
                t_n = t0 + n * h
                dfdx_val = self._model.dfdx(x_n, u_k, d_k, p_theta, t_n)
                dfdu_val = self._model.dfdu(x_n, u_k, d_k, p_theta, t_n)

                J[row:row + nx, x_np1_col:x_np1_col + nx] = np.eye(nx)
                J[row:row + nx, x_n_col:x_n_col + nx] = -(np.eye(nx) + dfdx_val * h)
                J[row:row + nx, u_k_col:u_k_col + nu] = -dfdu_val * h
                row += nx

        if self._is_dae:
            x_0, y_0, u_0, d_0 = X[0], Y[0], U[0], d_traj[0]
            dgdx_0 = self._model.dgdx(x_0, y_0, u_0, d_0, p_theta, t0)
            dgdy_0 = self._model.dgdy(x_0, y_0, u_0, d_0, p_theta, t0)
            J[row:row + ny, L.x_off:L.x_off + nx] = dgdx_0
            J[row:row + ny, L.y_off:L.y_off + ny] = dgdy_0

        return J

    def _inequality_constraint_jac(
        self,
        z: np.ndarray,
        u_prev_0: np.ndarray,
        d_traj: np.ndarray,
        p_theta: np.ndarray,
        t0: float,
    ) -> np.ndarray:
        """Dense Jacobian of the inequality constraints w.r.t. the NLP decision vector."""
        L = self._layout
        U = L.get_U(z)
        X = L.get_X(z)
        Y = L.get_Y(z)
        h = self._h
        nx, nu, ny, nz = self._nx, self._nu, self._ny, self._nz

        rows: list[np.ndarray] = []

        if self._du_min is not None or self._du_max is not None:
            for k in range(self._N):
                u_k_col = L.u_off + k * nu
                u_km1_col = L.u_off + (k - 1) * nu if k > 0 else None

                if self._du_min is not None:
                    jrow = np.zeros((nu, L.total))
                    jrow[:, u_k_col:u_k_col + nu] = np.eye(nu)
                    if u_km1_col is not None:
                        jrow[:, u_km1_col:u_km1_col + nu] = -np.eye(nu)
                    rows.append(jrow)

                if self._du_max is not None:
                    jrow = np.zeros((nu, L.total))
                    jrow[:, u_k_col:u_k_col + nu] = -np.eye(nu)
                    if u_km1_col is not None:
                        jrow[:, u_km1_col:u_km1_col + nu] = np.eye(nu)
                    rows.append(jrow)

        if self._has_soft_x:
            for n in range(L.M + 1):
                x_n_col = L.x_off + n * nx
                px_lo_n_col = L.px_lo_off + n * nx
                px_hi_n_col = L.px_hi_off + n * nx

                if self._x_min is not None:
                    jrow = np.zeros((nx, L.total))
                    jrow[:, x_n_col:x_n_col + nx] = np.eye(nx)
                    jrow[:, px_lo_n_col:px_lo_n_col + nx] = np.eye(nx)
                    rows.append(jrow)

                if self._x_max is not None:
                    jrow = np.zeros((nx, L.total))
                    jrow[:, x_n_col:x_n_col + nx] = -np.eye(nx)
                    jrow[:, px_hi_n_col:px_hi_n_col + nx] = np.eye(nx)
                    rows.append(jrow)

        if self._has_soft_z:
            for n in range(L.M + 1):
                k = min(n // self._n_steps, self._N - 1)
                u_k = U[k]
                d_k = d_traj[k]
                t_n = t0 + n * h
                x_n = X[n]
                y_n = Y[n] if self._is_dae else np.empty(0)

                x_n_col = L.x_off + n * nx
                u_k_col = L.u_off + k * nu
                pz_n_col = L.pz_off + n * nz

                if self._is_dae:
                    dgmdx_val = self._model.dgmdx(x_n, y_n, u_k, d_k, p_theta, t_n)
                    dgmdu_val = self._model.dgmdu(x_n, y_n, u_k, d_k, p_theta, t_n)
                    dgmdy_val = self._model.dgmdy(x_n, y_n, u_k, d_k, p_theta, t_n)
                    y_n_col = L.y_off + n * ny
                else:
                    dgmdx_val = self._model.dgmdx(x_n, u_k, d_k, p_theta, t_n)
                    dgmdu_val = self._model.dgmdu(x_n, u_k, d_k, p_theta, t_n)

                if self._z_min is not None:
                    jrow = np.zeros((nz, L.total))
                    jrow[:, x_n_col:x_n_col + nx] = dgmdx_val
                    jrow[:, u_k_col:u_k_col + nu] += dgmdu_val
                    jrow[:, pz_n_col:pz_n_col + nz] = np.eye(nz)
                    if self._is_dae:
                        jrow[:, y_n_col:y_n_col + ny] = dgmdy_val
                    rows.append(jrow)

                if self._z_max is not None:
                    jrow = np.zeros((nz, L.total))
                    jrow[:, x_n_col:x_n_col + nx] = -dgmdx_val
                    jrow[:, u_k_col:u_k_col + nu] -= dgmdu_val
                    jrow[:, pz_n_col:pz_n_col + nz] = np.eye(nz)
                    if self._is_dae:
                        jrow[:, y_n_col:y_n_col + ny] = -dgmdy_val
                    rows.append(jrow)

        if not rows:
            return np.zeros((0, L.total))
        return np.vstack(rows)

    def _objective_jac(
        self,
        z: np.ndarray,
        d_traj: np.ndarray,
        u_prev_0: np.ndarray,
        p_theta: np.ndarray,
        t0: float,
    ) -> np.ndarray:
        """Analytical gradient of the objective w.r.t. the NLP decision vector."""
        L = self._layout
        U = L.get_U(z)
        X = L.get_X(z)
        Y = L.get_Y(z)
        h = self._h
        Ts = self._dt
        nx, nu, ny, nz = self._nx, self._nu, self._ny, self._nz

        grad = np.zeros(L.total)

        for n in range(L.M):
            k = n // self._n_steps
            u_k = U[k]
            d_k = d_traj[k]
            x_np1 = X[n + 1]
            y_np1 = Y[n + 1] if self._is_dae else np.empty(0)
            t_np1 = t0 + (n + 1) * h

            x_np1_col = L.x_off + (n + 1) * nx
            u_k_col = L.u_off + k * nu

            if self._lagrange_jac is not None:
                lag_gx, lag_gy, lag_gu = self._lagrange_jac(
                    t_np1, x_np1, y_np1, u_k, p_theta
                )
                grad[x_np1_col:x_np1_col + nx] += np.asarray(lag_gx, dtype=float) * h
                grad[u_k_col:u_k_col + nu] += np.asarray(lag_gu, dtype=float) * h
                if self._is_dae and ny > 0:
                    y_np1_col = L.y_off + (n + 1) * ny
                    grad[y_np1_col:y_np1_col + ny] += (
                        np.asarray(lag_gy, dtype=float) * h
                    )

            if self._Q_z is not None:
                z_np1 = self._gm(x_np1, y_np1, u_k, d_k, p_theta, t_np1)
                e = z_np1 - self._z_ref[n + 1]
                Qze = self._Q_z @ e

                if self._is_dae:
                    dgmdx_val = self._model.dgmdx(
                        x_np1, y_np1, u_k, d_k, p_theta, t_np1
                    )
                    dgmdu_val = self._model.dgmdu(
                        x_np1, y_np1, u_k, d_k, p_theta, t_np1
                    )
                    dgmdy_val = self._model.dgmdy(
                        x_np1, y_np1, u_k, d_k, p_theta, t_np1
                    )
                    y_np1_col = L.y_off + (n + 1) * ny
                    grad[y_np1_col:y_np1_col + ny] += (
                        dgmdy_val.T @ (2.0 * Qze) * h
                    )
                else:
                    dgmdx_val = self._model.dgmdx(x_np1, u_k, d_k, p_theta, t_np1)
                    dgmdu_val = self._model.dgmdu(x_np1, u_k, d_k, p_theta, t_np1)

                grad[x_np1_col:x_np1_col + nx] += dgmdx_val.T @ (2.0 * Qze) * h
                grad[u_k_col:u_k_col + nu] += dgmdu_val.T @ (2.0 * Qze) * h

            if self._R_stage is not None:
                grad[u_k_col:u_k_col + nu] += 2.0 * self._R_stage @ u_k * h

        for k in range(self._N):
            u_k = U[k]
            u_km1 = U[k - 1] if k > 0 else u_prev_0
            u_k_col = L.u_off + k * nu

            if self._Q_du is not None:
                du = u_k - u_km1
                Qdu_du = self._Q_du @ du
                grad[u_k_col:u_k_col + nu] += 2.0 * Qdu_du * Ts
                if k > 0:
                    u_km1_col = L.u_off + (k - 1) * nu
                    grad[u_km1_col:u_km1_col + nu] -= 2.0 * Qdu_du * Ts

            if self._p_u_eco is not None:
                grad[u_k_col:u_k_col + nu] += self._p_u_eco * Ts

        x_M = X[L.M]
        y_M = Y[L.M] if self._is_dae else np.empty(0)
        x_M_col = L.x_off + L.M * nx

        if self._mayer_jac is not None:
            mayer_gx, mayer_gy = self._mayer_jac(x_M, y_M, p_theta)
            grad[x_M_col:x_M_col + nx] += np.asarray(mayer_gx, dtype=float)
            if self._is_dae and ny > 0:
                y_M_col = L.y_off + L.M * ny
                grad[y_M_col:y_M_col + ny] += np.asarray(mayer_gy, dtype=float)

        if self._P_terminal is not None and self._z_ref is not None:
            u_last = U[self._N - 1]
            d_last = d_traj[self._N - 1]
            t_M = t0 + L.M * h
            z_M = self._gm(x_M, y_M, u_last, d_last, p_theta, t_M)
            e = z_M - self._z_ref[L.M]
            Pe = self._P_terminal @ e

            if self._is_dae:
                dgmdx_val = self._model.dgmdx(x_M, y_M, u_last, d_last, p_theta, t_M)
                dgmdy_val = self._model.dgmdy(x_M, y_M, u_last, d_last, p_theta, t_M)
                y_M_col = L.y_off + L.M * ny
                grad[y_M_col:y_M_col + ny] += dgmdy_val.T @ (2.0 * Pe)
            else:
                dgmdx_val = self._model.dgmdx(x_M, u_last, d_last, p_theta, t_M)

            grad[x_M_col:x_M_col + nx] += dgmdx_val.T @ (2.0 * Pe)

        if self._has_soft_x:
            PX_lo = L.get_PX_lo(z)
            PX_hi = L.get_PX_hi(z)
            for n in range(L.M + 1):
                px_lo_n_col = L.px_lo_off + n * nx
                px_hi_n_col = L.px_hi_off + n * nx
                if self._x_min is not None:
                    p_n = PX_lo[n]
                    grad[px_lo_n_col:px_lo_n_col + nx] += (
                        self._rho_x_1 + 2.0 * self._rho_x_2 * p_n
                    ) * h
                if self._x_max is not None:
                    q_n = PX_hi[n]
                    grad[px_hi_n_col:px_hi_n_col + nx] += (
                        self._rho_x_1 + 2.0 * self._rho_x_2 * q_n
                    ) * h

        if self._has_soft_z:
            PZ = L.get_PZ(z)
            for n in range(L.M + 1):
                pz_n_col = L.pz_off + n * nz
                s_n = PZ[n]
                grad[pz_n_col:pz_n_col + nz] += (
                    self._rho_z_1 + 2.0 * self._rho_z_2 * s_n
                ) * h

        return grad

    def _can_use_analytical_objective_jac(self) -> bool:
        """True when all objective terms admit analytical gradients."""
        if self._lagrange is not None and self._lagrange_jac is None:
            return False
        if self._mayer is not None and self._mayer_jac is None:
            return False
        return True

    # ── Public solve / step ───────────────────────────────────────────────────

    def solve(
        self,
        x0: np.ndarray,
        d_trajectory: np.ndarray,
        u_prev: np.ndarray | None = None,
        x_prev: np.ndarray | None = None,
        y_prev: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t0: float = 0.0,
    ) -> tuple[np.ndarray, float, dict]:
        """
        Solve the OCP from initial state ``x0``.

        Parameters
        ----------
        x0 : (nx,) ndarray
            Filtered initial state ``x̂_{0|0}``.
        d_trajectory : (N, nd) ndarray
            ZOH disturbance per control interval.
        u_prev : (N, nu) ndarray, optional
            Previous optimal input sequence for warm-starting.
        x_prev : (M+1, nx) ndarray, optional
            Previous optimal state trajectory for warm-starting.
        y_prev : (M+1, ny) ndarray, optional
            Previous optimal algebraic state trajectory (SDAE only).
        p : (nparams,) ndarray, optional
            Parameter vector θ.  ``None`` → ``model.params``.
        t0 : float, optional
            Start time.

        Returns
        -------
        u_opt : (N, nu) ndarray
            Optimal input sequence.
        cost : float
            Optimal NLP objective value.
        info : dict
            ``{"X": (M+1, nx), "Y": (M+1, ny), "result": OptimizeResult}``
        """
        L = self._layout
        x_hat = np.asarray(x0, dtype=float)
        d_traj = np.asarray(d_trajectory, dtype=float)
        if d_traj.shape != (self._N, self._nd):
            raise ValueError(
                f"d_trajectory must have shape ({self._N}, {self._nd}); got {d_traj.shape}."
            )
        p_theta = (
            np.asarray(p, dtype=float)
            if p is not None
            else np.asarray(self._model.params, dtype=float)
        )
        u_prev_0 = (
            u_prev[-1] if u_prev is not None else np.zeros(self._nu)
        )

        z0 = self._build_initial_guess(x_hat, u_prev, x_prev, y_prev)

        lb = np.full(L.total, -np.inf)
        ub = np.full(L.total, np.inf)
        if self._u_min is not None:
            lb[L.u_off:L.u_off + L.u_size] = np.tile(self._u_min, self._N)
        if self._u_max is not None:
            ub[L.u_off:L.u_off + L.u_size] = np.tile(self._u_max, self._N)
        if self._has_soft_x:
            lb[L.px_lo_off:L.px_lo_off + L.px_lo_size] = 0.0
            lb[L.px_hi_off:L.px_hi_off + L.px_hi_size] = 0.0
        if self._has_soft_z:
            lb[L.pz_off:L.pz_off + L.pz_size] = 0.0

        constraints: list[NLPConstraint] = [
            NLPConstraint(
                kind="eq",
                fun=lambda z: self._equality_constraints(z, x_hat, d_traj, p_theta, t0),
                jac=lambda z: self._equality_constraint_jac(z, x_hat, d_traj, p_theta, t0),
            )
        ]
        has_ineq = (
            self._du_min is not None
            or self._du_max is not None
            or self._has_soft_x
            or self._has_soft_z
        )
        if has_ineq:
            constraints.append(
                NLPConstraint(
                    kind="ineq",
                    fun=lambda z: self._inequality_constraints(z, u_prev_0, d_traj, p_theta, t0),
                    jac=lambda z: self._inequality_constraint_jac(z, u_prev_0, d_traj, p_theta, t0),
                )
            )

        obj_jac = (
            (lambda z: self._objective_jac(z, d_traj, u_prev_0, p_theta, t0))
            if self._can_use_analytical_objective_jac()
            else None
        )

        result = self._solver_backend.solve(
            NLPProblem(
                objective=lambda z: self._objective(z, x_hat, d_traj, u_prev_0, p_theta, t0),
                objective_jac=obj_jac,
                x0=z0,
                lb=lb,
                ub=ub,
                constraints=tuple(constraints),
            )
        )

        z_opt = result.x
        U = L.get_U(z_opt).copy()
        X = L.get_X(z_opt).copy()
        Y = L.get_Y(z_opt).copy() if self._is_dae else np.zeros((L.M + 1, 0))
        info = {"X": X, "Y": Y, "result": result}
        return U, float(result.fun), info

    def step(
        self,
        x0: np.ndarray,
        d_trajectory: np.ndarray,
        u_prev: np.ndarray | None = None,
        x_prev: np.ndarray | None = None,
        y_prev: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t0: float = 0.0,
    ) -> np.ndarray:
        """Solve and return only the first optimal control action (receding horizon).

        Returns
        -------
        u0 : (nu,) ndarray
        """
        u_opt, _, _ = self.solve(
            x0, d_trajectory,
            u_prev=u_prev, x_prev=x_prev, y_prev=y_prev,
            p=p, t0=t0,
        )
        return u_opt[0]
