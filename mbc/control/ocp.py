"""
Optimal Control Problem (OCP) for linear discrete-time MPC.

Linear specialisation of the ControlToolbox §EMPC formulation: when the
plant dynamics are linear and the OCP is restricted to quadratic stage
costs and box / soft-box constraints, the entire NLP reduces to a single
finite-horizon **quadratic program** that the lifted (batch) form solves
directly with a convex-QP backend (HiGHS by default) — strictly more
efficient than the implicit-Euler direct-simultaneous formulation used by
:class:`~mbc.control.EconomicOptimalControlProblem` for nonlinear plants.

Plant model (ControlToolbox notation, discrete-time specialisation)
-------------------------------------------------------------------
    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
    z[k]   = Cz x[k] + Dz u[k] + Fz d[k]
    ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)

The OCP optimises the *output* ``z[k] = Cz x[k] + …`` — the same continuous
output ``g^m`` used in the nonlinear EOCP.  When the plant has
``Cz = Cm`` (the default of :class:`~mbc.models.LinearDiscreteModel`) the
output and measurement coincide and the OCP tracks the measured channel
directly.

Cost function over horizon N
----------------------------
    Φ(U) = Σ_{k=0}^{N-1} [ ‖z[k+1] − z_ref‖²_Q + ‖u[k]‖²_R + ‖Δu[k]‖²_S ]
         + ‖z[N] − z_ref‖²_P
         + ρ Σ_{k=0}^{N-1} ‖ε[k+1]‖²

with Δu[k] = u[k] − u[k−1] (rate of movement) and ε[k] the soft-output
slack variable.

Constraints
-----------
    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k]                (deterministic dynamics)
    u_min ≤ u[k] ≤ u_max                                  (hard input box)
    z_ref − δ − ε[k+1] ≤ Cz x[k+1] ≤ z_ref + δ + ε[k+1]  (soft output box)
    ε[k+1] ≥ 0                                            (slack non-negativity)

The decision variable is ``Z = [U; ε]`` and the QP is cast as

    min_Z   ½ Zᵀ H Z + fᵀ Z
    s.t.    G_qp Z ≤ h_qp
            lb ≤ Z ≤ ub

The hard input box and the slack non-negativity are passed as variable
bounds; the soft-output band is passed as the inequality rows ``G_qp``.
The QP is solved through a :class:`~mbc.control.qp_solver.QPSolverBackend`
(default: HiGHS).

Batch (lifted) form
-------------------
Stacking the state predictions ``X = [x[1]; …; x[N]]`` gives

    X = Ψ x₀ + Γ U + Λ D

where Ψ, Γ, Λ are the standard prediction matrices built from Ad, Bd, Ed.
Stacking the outputs ``Z = C̄_z X`` (block-diagonal C̄_z) gives

    Z = C̄_z Ψ x₀ + C̄_z Γ U + C̄_z Λ D

Notation
--------
    nx   – state dimension          x ∈ ℝⁿˣ
    nu   – input dimension          u ∈ ℝⁿᵘ
    nd   – disturbance dimension    d ∈ ℝⁿᵈ
    nz   – output dimension         z ∈ ℝⁿᶻ
    N    – prediction horizon
    Ψ    – free-response matrix     Ψ ∈ ℝᴺⁿˣˣⁿˣ
    Γ    – forced-response matrix   Γ ∈ ℝᴺⁿˣˣᴺⁿᵘ
    Λ    – disturbance matrix       Λ ∈ ℝᴺⁿˣˣᴺⁿᵈ
"""

from __future__ import annotations

import warnings
from typing import Any, TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag

from .._utils import _any_to_np1d, _any_to_np2d
from .qp_solver import QPProblem, QPSolverBackend, make_qp_backend

if TYPE_CHECKING:
    from ..models import LinearDiscreteModel


# ── First-difference operator ────────────────────────────────────────────


def _build_D_diff(nu: int, N: int) -> np.ndarray:
    """
    Block first-difference matrix ``D_diff`` for the rate-of-movement
    penalty ``ΔU = D_diff U + d₀``:

        [ I               ]       Δu[0] = u[0] − u_prev
        [−I   I           ]       Δu[1] = u[1] − u[0]
        [    −I   I       ]       …
        [         ⋱    I  ]       Δu[N−1] = u[N−1] − u[N−2]
    """
    dim = N * nu
    D = np.zeros((dim, dim))
    for k in range(N):
        for i in range(nu):
            D[k * nu + i, k * nu + i] = 1.0
            if k > 0:
                D[k * nu + i, (k - 1) * nu + i] = -1.0
    return D


# ── Optimal Control Problem ─────────────────────────────────────────────


class OptimalControlProblem:
    """
    Receding-horizon QP with hard input and soft output box constraints.

    The OCP tracks the **output** ``z[k] = Cz x[k]`` against a constant
    reference ``z_ref``.  When ``Cz = Cm`` (the default of
    :class:`~mbc.models.LinearDiscreteModel`) the output and the
    measurement coincide and the OCP tracks the measured channel
    directly.

    Parameters
    ----------
    model : LinearDiscreteModel
        Plant model providing ``Ad``, ``Bd``, ``Ed``, ``Cz``, ``u_bounds``.
    N : int
        Prediction horizon (number of control intervals).
    Q : (nz, nz) array-like
        Stage tracking cost ``‖z − z_ref‖²_Q``.
    R : (nu, nu) array-like
        Stage input cost ``‖u‖²_R``.
    P : (nz, nz) array-like, optional
        Terminal tracking cost ``‖z[N] − z_ref‖²_P``.  Default: ``Q``.
    S : (nu, nu) array-like, optional
        Input rate-of-movement cost ``‖Δu‖²_S``.  ``None`` disables.
    rho : float, optional
        Quadratic penalty on the soft-output slack variable ``ε``.
        Default: 1e4.
    y_offset : float, optional
        Symmetric half-width δ of the soft-output band ``[z_ref − δ,
        z_ref + δ]``.  Default: 2.0.
    solver : str or QPSolverBackend, optional
        Convex-QP backend selector.  ``"highs"`` (default) uses the
        MIT-licensed HiGHS solver via ``highspy``.  A
        :class:`~mbc.control.qp_solver.QPSolverBackend` instance may also be
        supplied directly.
    solver_options : dict, optional
        Backend-specific options forwarded to the QP solver.
    """

    def __init__(
        self,
        model: "LinearDiscreteModel",
        N: int,
        Q: Any,
        R: Any,
        P: Any | None = None,
        S: Any | None = None,
        rho: float = 1e4,
        y_offset: float = 2.0,
        solver: str | QPSolverBackend = "highs",
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._N = N
        self._Q = _any_to_np2d(Q)
        self._R = _any_to_np2d(R)
        self._P = _any_to_np2d(P) if P is not None else self._Q.copy()
        self._S = _any_to_np2d(S) if S is not None else None
        self._rho = rho
        self._y_offset = y_offset
        self._backend = make_qp_backend(solver, solver_options=solver_options)

        nu = model.nu
        # Pre-compute constant structures
        self._D_diff: np.ndarray | None = None
        self._S_bar: np.ndarray | None = None
        if self._S is not None:
            self._D_diff = _build_D_diff(nu, N)
            self._S_bar = block_diag(*([self._S] * N))

    def solve(
        self,
        x0: Any,
        D: Any,
        x_ref: Any,
        u_prev: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the QP starting from state estimate ``x0``.

        Parameters
        ----------
        x0    : (nx,) array-like — current state estimate ``x̂_{k|k}``.
        D     : (N · nd,) array-like — stacked disturbance forecast
                ``[d[0]; d[1]; …; d[N − 1]]``.
        x_ref : (nx,) array-like — state reference; the output reference is
                ``z_ref = Cz x_ref``.
        u_prev : (nu,) array-like, optional
            Previously-applied input — used only when an input rate-of-
            movement penalty ``S`` is active.

        Returns
        -------
        U : (N · nu,) ndarray — optimal input sequence.
        X : (N · nx,) ndarray — predicted state trajectory ``[x[1]; …; x[N]]``.
        """
        N = self._N
        nx = self._model.nx
        nu = self._model.nu
        nd = self._model.nd
        Cz = _any_to_np2d(self._model.Cz)
        nz = Cz.shape[0]

        # ── Coerce inputs to numpy 1-D ──────────────────────────────────
        x0 = _any_to_np1d(x0).reshape(-1)
        x_ref = _any_to_np1d(x_ref).reshape(-1)
        D = _any_to_np1d(D).reshape(-1) if D is not None else np.zeros(N * nd)

        # ── Discrete-time matrices (LTI; LPV scheduling not handled here) ──
        Ad = _any_to_np2d(self._model.Ad)
        Bd = _any_to_np2d(self._model.Bd)
        Ed = _any_to_np2d(self._model.Ed)

        # ── Powers of Ad ─────────────────────────────────────────────────
        Ad_pow = [np.eye(nx)]
        for _ in range(N):
            Ad_pow.append(Ad @ Ad_pow[-1])

        # ── State prediction matrices  X = Ψ x₀ + Γ U + Λ D ──────────────
        Psi = np.zeros((N * nx, nx))
        Gamma = np.zeros((N * nx, N * nu))
        Lambda = np.zeros((N * nx, N * nd))

        for k in range(N):
            Psi[k * nx:(k + 1) * nx, :] = Ad_pow[k + 1]
            for j_step in range(k + 1):
                Ak_j = Ad_pow[k - j_step]
                Gamma[k * nx:(k + 1) * nx, j_step * nu:(j_step + 1) * nu] = Ak_j @ Bd
                Lambda[k * nx:(k + 1) * nx, j_step * nd:(j_step + 1) * nd] = Ak_j @ Ed

        # ── Output prediction matrices  Z = C̄_z X ──────────────────────
        Cz_bar = np.kron(np.eye(N), Cz)
        CG = Cz_bar @ Gamma        # (N · nz) × (N · nu)
        CP = Cz_bar @ Psi          # (N · nz) × nx
        CL = Cz_bar @ Lambda       # (N · nz) × (N · nd)

        # ── Cost matrices ────────────────────────────────────────────────
        Q_bar = block_diag(*([self._Q] * (N - 1) + [self._P])) if N > 1 else self._P
        R_bar = block_diag(*([self._R] * N))

        # Reference and free response in OUTPUT space
        z_ref = Cz @ x_ref                                   # (nz,)
        z_ref_bar = np.tile(z_ref, N)                        # (N · nz,)
        Z_free = CP @ x0 + CL @ D                            # (N · nz,)
        e_free = Z_free - z_ref_bar                          # (N · nz,)

        # Hessian and gradient for U-part
        H_uu = CG.T @ Q_bar @ CG + R_bar                     # (N · nu) × (N · nu)
        f_u = CG.T @ Q_bar @ e_free                          # (N · nu,)

        # Rate-of-movement penalty
        if self._S is not None:
            if u_prev is None:
                u_prev = np.zeros(nu)
            else:
                u_prev = _any_to_np1d(u_prev).reshape(-1)
            d0_shift = np.zeros(N * nu)
            d0_shift[:nu] = -u_prev
            H_uu = H_uu + self._D_diff.T @ self._S_bar @ self._D_diff
            f_u = f_u + self._D_diff.T @ self._S_bar @ d0_shift

        # ── Decision variable Z = [U; ε],  ε ∈ ℝ^{N · nz} ───────────────
        n_U = N * nu
        n_eps = N * nz
        n_Z = n_U + n_eps

        # Full Hessian  H = blkdiag(H_uu, ρ I)  and gradient  f = [f_u; 0]
        H = np.zeros((n_Z, n_Z))
        H[:n_U, :n_U] = H_uu
        H[n_U:, n_U:] = self._rho * np.eye(n_eps)
        # Symmetrise to guard against floating-point skew (HiGHS wants a
        # symmetric Hessian for the convex QP).
        H = 0.5 * (H + H.T)

        f = np.zeros(n_Z)
        f[:n_U] = f_u

        # ── Variable bounds: hard input box, slack non-negativity ────────
        u_min_np, u_max_np = self._model.u_bounds
        u_min_tiled = np.tile(_any_to_np1d(u_min_np).reshape(-1), N)
        u_max_tiled = np.tile(_any_to_np1d(u_max_np).reshape(-1), N)
        lb = np.concatenate([u_min_tiled, np.zeros(n_eps)])
        ub = np.concatenate([u_max_tiled, np.full(n_eps, np.inf)])

        # ── Soft output box as inequality rows  G_qp Z ≤ h_qp ────────────
        #   (2a) −C̄_z Γ U − ε ≤ −z_min + Z_free
        #   (2b)  C̄_z Γ U − ε ≤  z_max − Z_free
        z_min = z_ref - self._y_offset
        z_max = z_ref + self._y_offset
        z_min_tiled = np.tile(z_min, N)
        z_max_tiled = np.tile(z_max, N)

        neg_I = -np.eye(n_eps)
        G_lo = np.hstack([-CG, neg_I])
        G_hi = np.hstack([CG, neg_I])
        G_qp = np.vstack([G_lo, G_hi])
        h_qp = np.concatenate(
            [-z_min_tiled + Z_free, z_max_tiled - Z_free]
        )

        # ── Solve QP ─────────────────────────────────────────────────────
        result = self._backend.solve(
            QPProblem(P=H, q=f, lb=lb, ub=ub, G=G_qp, h=h_qp)
        )

        if not result.success:
            warnings.warn(
                f"OptimalControlProblem.solve: QP solver returned status "
                f"'{result.status}'; returning zero inputs as fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            U_flat = np.zeros(n_U)
        else:
            U_flat = np.asarray(result.x[:n_U], dtype=float)

        # ── Predicted state trajectory ───────────────────────────────────
        X_flat = Psi @ x0 + Gamma @ U_flat + Lambda @ D

        return U_flat, X_flat
