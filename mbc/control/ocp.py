"""
Optimal Control Problem (OCP) for linear discrete-time MPC.

Linear specialisation of the ControlToolbox §EMPC formulation: when the
plant dynamics are linear and the OCP is restricted to quadratic stage
costs and box / soft-box constraints, the entire NLP reduces to a single
finite-horizon **quadratic program** that the lifted (batch) form solves
directly with ``cvxopt.solvers.qp`` — strictly more efficient than the
implicit-Euler direct-simultaneous formulation used by
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

solved using ``cvxopt.solvers.qp``.

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
from typing import TYPE_CHECKING

from cvxopt import matrix, spmatrix, solvers

from .._utils import _eye, _zeros, _np_to_cvx

if TYPE_CHECKING:
    from ..models import LinearDiscreteModel

# Silence cvxopt solver output
solvers.options["show_progress"] = False


# ── Helpers ──────────────────────────────────────────────────────────────


def _block_diag(M: matrix, N_blocks: int) -> matrix:
    """Build a block-diagonal matrix by repeating M along the diagonal."""
    r, c = M.size
    out = _zeros(N_blocks * r, N_blocks * c)
    for k in range(N_blocks):
        for i in range(r):
            for j in range(c):
                out[k * r + i, k * c + j] = M[i, j]
    return out


def _block_diag_terminal(M: matrix, M_terminal: matrix, N_blocks: int) -> matrix:
    """
    Block-diagonal matrix with ``M`` for the first ``N − 1`` blocks and
    ``M_terminal`` for the last block (terminal-cost slot).
    """
    r, c = M.size
    out = _zeros(N_blocks * r, N_blocks * c)
    for k in range(N_blocks):
        blk = M_terminal if k == N_blocks - 1 else M
        for i in range(r):
            for j in range(c):
                out[k * r + i, k * c + j] = blk[i, j]
    return out


def _tile_column(v: matrix, N_blocks: int) -> matrix:
    """Vertically stack ``N_blocks`` copies of column vector ``v``."""
    n = v.size[0]
    out = matrix(0.0, (N_blocks * n, 1))
    for k in range(N_blocks):
        out[k * n:(k + 1) * n] = v
    return out


# ── First-difference operator ────────────────────────────────────────────


def _build_D_diff(nu: int, N: int) -> matrix:
    """
    Block first-difference matrix ``D_diff`` for the rate-of-movement
    penalty ``ΔU = D_diff U + d₀``:

        [ I               ]       Δu[0] = u[0] − u_prev
        [−I   I           ]       Δu[1] = u[1] − u[0]
        [    −I   I       ]       …
        [         ⋱    I  ]       Δu[N−1] = u[N−1] − u[N−2]
    """
    dim = N * nu
    D = _zeros(dim, dim)
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
    Q : cvxopt.matrix (nz, nz)
        Stage tracking cost ``‖z − z_ref‖²_Q``.
    R : cvxopt.matrix (nu, nu)
        Stage input cost ``‖u‖²_R``.
    P : cvxopt.matrix (nz, nz), optional
        Terminal tracking cost ``‖z[N] − z_ref‖²_P``.  Default: ``Q``.
    S : cvxopt.matrix (nu, nu), optional
        Input rate-of-movement cost ``‖Δu‖²_S``.  ``None`` disables.
    rho : float, optional
        Quadratic penalty on the soft-output slack variable ``ε``.
        Default: 1e4.
    y_offset : float, optional
        Symmetric half-width δ of the soft-output band ``[z_ref − δ,
        z_ref + δ]``.  Default: 2.0.
    """

    def __init__(
        self,
        model: "LinearDiscreteModel",
        N: int,
        Q: matrix,
        R: matrix,
        P: matrix | None = None,
        S: matrix | None = None,
        rho: float = 1e4,
        y_offset: float = 2.0,
    ) -> None:
        self._model = model
        self._N = N
        self._Q = Q
        self._R = R
        self._P = P if P is not None else matrix(Q)
        self._S = S
        self._rho = rho
        self._y_offset = y_offset

        nu = model.nu
        # Pre-compute constant structures
        self._D_diff: matrix | None = None
        self._S_bar: matrix | None = None
        if S is not None:
            self._D_diff = _build_D_diff(nu, N)
            self._S_bar = _block_diag(S, N)

    def solve(
        self,
        x0: matrix,
        D: matrix,
        x_ref: matrix,
        u_prev: matrix | None = None,
    ) -> tuple[matrix, matrix]:
        """
        Solve the QP starting from state estimate ``x0``.

        Parameters
        ----------
        x0    : (nx, 1) current state estimate ``x̂_{k|k}``.
        D     : (N · nd, 1) stacked disturbance forecast
                ``[d[0]; d[1]; …; d[N − 1]]``.
        x_ref : (nx, 1) state reference; the output reference is
                ``z_ref = Cz x_ref + Dz · 0 + Fz · 0 = Cz x_ref``.
        u_prev : (nu, 1), optional
            Previously-applied input — used only when an input rate-of-
            movement penalty ``S`` is active.

        Returns
        -------
        U : (N · nu, 1) optimal input sequence.
        X : (N · nx, 1) predicted state trajectory ``[x[1]; …; x[N]]``.
        """
        N = self._N
        nx = self._model.nx
        nu = self._model.nu
        nd = self._model.nd
        Cz = _np_to_cvx(self._model.Cz)
        nz = Cz.size[0]

        # ── Convert numpy inputs to cvxopt if needed ────────────────────
        import numpy as _np
        if isinstance(x0, _np.ndarray):
            x0 = _np_to_cvx(x0.reshape(-1, 1))
        if isinstance(x_ref, _np.ndarray):
            x_ref = _np_to_cvx(x_ref.reshape(-1, 1))

        # ── Discrete-time matrices (LTI; LPV scheduling not handled here) ──
        Ad = _np_to_cvx(self._model.Ad)
        Bd = _np_to_cvx(self._model.Bd)
        Ed = _np_to_cvx(self._model.Ed)

        # ── Powers of Ad ─────────────────────────────────────────────────
        Ad_pow = [_eye(nx)]
        for _ in range(N):
            Ad_pow.append(Ad * Ad_pow[-1])

        # ── State prediction matrices  X = Ψ x₀ + Γ U + Λ D ──────────────
        Psi = _zeros(N * nx, nx)
        Gamma = _zeros(N * nx, N * nu)
        Lambda = _zeros(N * nx, N * nd)

        for k in range(N):
            # Ψ block-row k:  Ad^{k+1}
            for i in range(nx):
                for j in range(nx):
                    Psi[k * nx + i, j] = Ad_pow[k + 1][i, j]

            # Γ and Λ block-row k
            for j_step in range(k + 1):
                Ak_j = Ad_pow[k - j_step]
                AB = Ak_j * Bd
                AE = Ak_j * Ed
                for i in range(nx):
                    for jj in range(nu):
                        Gamma[k * nx + i, j_step * nu + jj] = AB[i, jj]
                    for jj in range(nd):
                        Lambda[k * nx + i, j_step * nd + jj] = AE[i, jj]

        # ── Output prediction matrices  Z = C̄_z X ──────────────────────
        Cz_bar = _block_diag(Cz, N)
        CG = Cz_bar * Gamma        # (N · nz) × (N · nu)
        CP = Cz_bar * Psi          # (N · nz) × nx
        CL = Cz_bar * Lambda       # (N · nz) × (N · nd)

        # ── Cost matrices ────────────────────────────────────────────────
        Q_bar = _block_diag_terminal(self._Q, self._P, N)   # (N · nz) × (N · nz)
        R_bar = _block_diag(self._R, N)                      # (N · nu) × (N · nu)

        # Reference and free response in OUTPUT space
        z_ref = Cz * x_ref                                   # (nz, 1)
        z_ref_bar = _tile_column(z_ref, N)                   # (N · nz, 1)
        Z_free = CP * x0 + CL * D                            # (N · nz, 1)
        e_free = Z_free - z_ref_bar                          # (N · nz, 1)

        # Hessian and gradient for U-part
        H_uu = CG.T * Q_bar * CG + R_bar                    # (N · nu) × (N · nu)
        f_u = CG.T * Q_bar * e_free                          # (N · nu, 1)

        # Rate-of-movement penalty
        if self._S is not None:
            if u_prev is None:
                u_prev = _zeros(nu, 1)
            d0_shift = _zeros(N * nu, 1)
            d0_shift[:nu] = -u_prev
            H_uu += self._D_diff.T * self._S_bar * self._D_diff
            f_u += self._D_diff.T * self._S_bar * d0_shift

        # ── Soft output slack variables ε ────────────────────────────────
        #   Decision variable Z_qp = [U; ε],  ε ∈ ℝ^{N · nz}
        n_U = N * nu
        n_eps = N * nz
        n_Z = n_U + n_eps

        # Full Hessian  H = [ H_uu   0  ]
        #                    [  0    ρI  ]
        H = _zeros(n_Z, n_Z)
        for i in range(n_U):
            for j in range(n_U):
                H[i, j] = H_uu[i, j]
        for i in range(n_eps):
            H[n_U + i, n_U + i] = self._rho

        # Full gradient  f = [ f_u; 0 ]
        f = _zeros(n_Z, 1)
        for i in range(n_U):
            f[i] = f_u[i]

        # ── Inequality constraints  G_qp Z_qp ≤ h_qp ─────────────────────
        #
        # 1) Hard input box:      u_min ≤ u[k] ≤ u_max
        #    →  -u[k] ≤ -u_min    and    u[k] ≤ u_max
        #
        # 2) Soft output box:    z[k+1] ≥ z_min − ε[k+1]
        #                         →  -C̄_z Γ U + ε ≤ -(z_min − Z_free)
        #                         z[k+1] ≤ z_max + ε[k+1]
        #                         →   C̄_z Γ U − ε ≤   z_max − Z_free
        #
        # 3) Slack non-negativity:  ε ≥ 0   →  -ε ≤ 0

        u_min_np, u_max_np = self._model.u_bounds
        u_min = _np_to_cvx(u_min_np.reshape(-1, 1))
        u_max = _np_to_cvx(u_max_np.reshape(-1, 1))
        u_min_tiled = _tile_column(u_min, N)
        u_max_tiled = _tile_column(u_max, N)

        # Output bounds in z-space
        z_min = z_ref - matrix(self._y_offset, (nz, 1))
        z_max = z_ref + matrix(self._y_offset, (nz, 1))
        z_min_tiled = _tile_column(z_min, N)
        z_max_tiled = _tile_column(z_max, N)

        n_ineq = 2 * n_U + 2 * n_eps + n_eps
        G_qp = _zeros(n_ineq, n_Z)
        h_qp = _zeros(n_ineq, 1)

        row = 0

        # (1a)  -U ≤ -u_min
        for i in range(n_U):
            G_qp[row + i, i] = -1.0
            h_qp[row + i] = -u_min_tiled[i]
        row += n_U

        # (1b)   U ≤ u_max
        for i in range(n_U):
            G_qp[row + i, i] = 1.0
            h_qp[row + i] = u_max_tiled[i]
        row += n_U

        # (2a)  −C̄_z Γ U − ε ≤ −z_min + Z_free
        for i in range(n_eps):
            for j in range(n_U):
                G_qp[row + i, j] = -CG[i, j]
            G_qp[row + i, n_U + i] = -1.0
            h_qp[row + i] = -z_min_tiled[i] + Z_free[i]
        row += n_eps

        # (2b)   C̄_z Γ U − ε ≤  z_max − Z_free
        for i in range(n_eps):
            for j in range(n_U):
                G_qp[row + i, j] = CG[i, j]
            G_qp[row + i, n_U + i] = -1.0
            h_qp[row + i] = z_max_tiled[i] - Z_free[i]
        row += n_eps

        # (3) -ε ≤ 0  (slack non-negativity)
        for i in range(n_eps):
            G_qp[row + i, n_U + i] = -1.0
            h_qp[row + i] = 0.0
        row += n_eps

        # ── Solve QP with cvxopt ─────────────────────────────────────────
        sol = solvers.qp(H, f, G_qp, h_qp)

        if sol["status"] != "optimal":
            warnings.warn(
                f"OptimalControlProblem.solve: QP solver returned status "
                f"'{sol['status']}'; returning zero inputs as fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            U_flat = _zeros(n_U, 1)
        else:
            z_opt = sol["x"]
            U_flat = z_opt[:n_U]

        # ── Predicted state trajectory ───────────────────────────────────
        X_flat = Psi * x0 + Gamma * U_flat + Lambda * D

        return U_flat, X_flat
