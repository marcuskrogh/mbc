"""
Discrete-time linear receding-horizon QP (``DiscreteLinearOCP``).

Linear specialisation of the ControlToolbox §EMPC formulation: when the
plant dynamics are linear and the OCP is restricted to quadratic stage
costs and box / soft-box constraints, the entire NLP reduces to a single
finite-horizon **quadratic program** that is solved directly with a convex-QP
backend (OSQP by default; HiGHS also available) — strictly more efficient than
the direct-simultaneous formulation used by
:class:`~mbc.control.ContinuousOCP` for nonlinear plants.

Plant model (ControlToolbox notation, discrete-time specialisation)
-------------------------------------------------------------------
    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
    z[k]   = Cz x[k] + Dz u[k] + Fz d[k]
    ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)

The OCP optimises the *output* ``z[k] = Cz x[k] + …``.

Cost function over horizon N
----------------------------
    Φ(U) = Σ_{k=0}^{N-1} [ ‖z[k+1] − z_ref‖²_Q + ‖u[k]‖²_R + ‖Δu[k]‖²_S ]
         + ‖z[N] − z_ref‖²_P
         + ρ Σ_{k=0}^{N-1} ‖ε[k+1]‖²
         + ρ_lin Σ_{k=0}^{N-1} 1ᵀε[k+1]

with Δu[k] = u[k] − u[k−1] (rate of movement) and ε[k] the soft-output
slack variable.

Two equivalent formulations
---------------------------
The same QP is built in one of two ways, selected by ``formulation``:

* **condensed** — eliminate the states via the lifted prediction
  ``X = Ψ x₀ + Γ U + Λ D`` and optimise over ``Z = [U; ε]`` only.

* **sparse** (a.k.a. simultaneous / non-condensed) — keep the states as
  decision variables ``Z = [X; U; ε]`` and impose the dynamics as
  block-banded linear *equality* constraints.

``formulation="auto"`` (default) is **backend-aware**: OSQP → ``sparse``,
HiGHS → ``condensed``.

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
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag

from .._utils import _any_to_np1d, _any_to_np2d
from ._base import DiscreteOptimalControlProblem
from .qp_solver import QPProblem, QPSolverBackend, make_qp_backend
from .input_linear_cost import (
    InputLinearCostMode,
    absolute_quadratic_input_regularisation_linear_term,
    augment_condensed_qp,
    augment_sparse_qp,
    infer_signed_magnitude_input_indices,
    resolve_input_linear_cost,
)

if TYPE_CHECKING:
    from ..models import DiscreteLinearSDE


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


def _shift_warm_start(
    U_seq: np.ndarray,
    X_seq: np.ndarray,
    nu: int,
    nx: int,
) -> dict[str, np.ndarray]:
    """
    Shift a previous horizon solution forward by one step for warm-starting.

    Given the previous optimal input sequence ``U_seq = [u₀ … u_{N−1}]`` and
    predicted states ``X_seq = [x₁ … x_N]``, the next receding-horizon QP
    starts one step later, so a good initial guess re-uses the tail and
    repeats the final element:

        U_warm = [u₁, …, u_{N−1}, u_{N−1}]
        X_warm = [x₂, …, x_N,     x_N]

    Returns a ``{"U": …, "X": …}`` dict suitable for ``solve(warm_start=…)``.
    """
    U = np.asarray(U_seq, dtype=float).reshape(-1, nu)
    X = np.asarray(X_seq, dtype=float).reshape(-1, nx)
    U_warm = np.vstack([U[1:], U[-1:]]) if U.shape[0] > 1 else U
    X_warm = np.vstack([X[1:], X[-1:]]) if X.shape[0] > 1 else X
    return {"U": U_warm.reshape(-1), "X": X_warm.reshape(-1)}


# ── Discrete-time linear OCP ─────────────────────────────────────────────────


class StandardLinearDiscreteOCP(DiscreteOptimalControlProblem):
    """
    Receding-horizon QP with hard input and soft output box constraints for
    discrete-time linear systems.

    The OCP tracks the **output** ``z[k] = Cz x[k]`` against a constant
    reference ``z_ref``.  When ``Cz = Cm`` (the default of
    :class:`~mbc.models.DiscreteLinearSDE`) the output and the
    measurement coincide and the OCP tracks the measured channel directly.

    Parameters
    ----------
    model : DiscreteLinearSDE
        Plant model providing ``Ad``, ``Bd``, ``Ed``, ``Cz``, ``u_bounds``.
    N : int
        Prediction horizon (number of control intervals).
    Q : array-like
        Stage tracking cost ``‖z − z_ref‖²_Q``.  Three forms are accepted:

        * **scalar** — constant weight ``Q_k = scalar · I_{nz}`` at every step.
        * **(N,) array** — per-step scalar ``Q_k = arr[k] · I_{nz}``.
        * **(N, nz) array** — per-step diagonal ``Q_k = diag(arr[k, :])``.
        * **(nz, nz) array** — constant matrix (existing behaviour).

        When ``N == nz`` a 1-D array of length ``N`` is always interpreted as
        per-step scalars, not as a constant diagonal.
    R : array-like
        Stage input cost ``‖u‖²_R``.  Same four forms as ``Q`` with ``nu``
        replacing ``nz``.
    P : array-like, optional
        Terminal tracking cost ``‖z[N] − z_ref‖²_P``.  Accepts scalar,
        ``(nz,)`` diagonal vector, or ``(nz, nz)`` matrix.  Default: last
        step's ``Q`` matrix.
    S : array-like, optional
        Input rate-of-movement cost ``‖Δu‖²_S``.  Same four forms as ``R``
        with ``nu``.  ``None`` disables.
    du_min, du_max : (nu,) array-like, optional
        Hard input rate-of-movement box ``du_min ≤ Δu ≤ du_max``.
        ``None`` disables the corresponding bound.
    rho : float, (N,) or (N, nz) array-like, optional
        Quadratic penalty on the soft-output slack variable ``ε``.

        * scalar → same weight for every step and output channel.
        * (N,) → per-step scalar, broadcast across output channels.
        * (N, nz) → per-step, per-channel weight.

        Default: 1e4.
    rho_lin : float, (N,) or (N, nz) array-like, optional
        Linear (L1-style) penalty on ``ε``.  Same three forms as ``rho``.
        Default: 0.0 (disabled).
    y_offset : float, (N,) or (N, nz) array-like, optional
        Symmetric half-width δ of the soft-output band ``[z_ref − δ,
        z_ref + δ]``.  Same three forms as ``rho``.  Default: 2.0.
    solver : str or QPSolverBackend, optional
        Convex-QP backend selector.  ``"highs"`` (default) or ``"osqp"`` (optional).
    solver_options : dict, optional
        Backend-specific options forwarded to the QP solver.
    formulation : {"auto", "condensed", "sparse"}, optional
        QP construction strategy.  Default ``"auto"`` is backend-aware.
    """

    def __init__(
        self,
        model: "DiscreteLinearSDE",
        N: int,
        Q: Any,
        R: Any,
        P: Any | None = None,
        S: Any | None = None,
        du_min: Any | None = None,
        du_max: Any | None = None,
        rho: float = 1e4,
        rho_lin: float = 0.0,
        y_offset: float = 2.0,
        solver: str | QPSolverBackend = "highs",
        solver_options: dict[str, Any] | None = None,
        formulation: str = "auto",
    ) -> None:
        super().__init__()
        if formulation not in ("auto", "condensed", "sparse"):
            raise ValueError(
                f"formulation must be 'auto', 'condensed', or 'sparse'; "
                f"got {formulation!r}."
            )
        self._model = model
        self._N = N
        self._rho = rho
        self._rho_lin = rho_lin
        self._y_offset = y_offset
        self._backend = make_qp_backend(solver, solver_options=solver_options)
        self._formulation = formulation

        nu = model.nu
        nz = np.asarray(model.Cz, dtype=float).shape[0]

        self._Q_mats: list[np.ndarray] = self._per_step_weight_matrices(Q, N, nz)
        self._R_mats: list[np.ndarray] = self._per_step_weight_matrices(R, N, nu)
        self._P_mat: np.ndarray = (
            self._terminal_weight_matrix(P, nz)
            if P is not None
            else self._Q_mats[-1].copy()
        )

        self._du_min = (
            _any_to_np1d(du_min).reshape(-1) if du_min is not None else None
        )
        self._du_max = (
            _any_to_np1d(du_max).reshape(-1) if du_max is not None else None
        )

        self._D_diff: np.ndarray | None = None
        self._S_bar: np.ndarray | None = None
        if S is not None:
            S_mats = self._per_step_weight_matrices(S, N, nu)
            self._D_diff = _build_D_diff(nu, N)
            self._S_bar = block_diag(*S_mats)
        elif self._du_min is not None or self._du_max is not None:
            self._D_diff = _build_D_diff(nu, N)
        else:
            self._D_diff = None

    # ── OCP abstract properties ────────────────────────────────────────────

    @property
    def N(self) -> int:
        """Prediction horizon (number of control intervals)."""
        return self._N

    @property
    def nu(self) -> int:
        """Input dimension nᵘ."""
        return self._model.nu

    # ── Internal helpers ───────────────────────────────────────────────────

    def _resolve_formulation(self) -> str:
        """Return the concrete formulation ('condensed' or 'sparse').

        ``"auto"`` is backend-aware: OSQP → ``"sparse"``, HiGHS → ``"condensed"``.
        """
        if self._formulation != "auto":
            return self._formulation
        from .qp_solver import OSQPBackend
        return "sparse" if isinstance(self._backend, OSQPBackend) else "condensed"

    def _rate_offset(self, u_prev_np: np.ndarray | None, nu: int, N: int) -> np.ndarray:
        """Affine offset ``d0`` in ``Δu = D_diff U + d0``."""
        d0 = np.zeros(N * nu)
        if u_prev_np is not None:
            d0[:nu] = -u_prev_np
        return d0

    @staticmethod
    def _per_step_scales(values: Any | None, N: int, default: float = 1.0) -> np.ndarray:
        if values is None:
            return np.full(N, default)
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0 or arr.size == 1:
            return np.full(N, float(arr.reshape(-1)[0]))
        if arr.shape[0] != N:
            raise ValueError(f"Horizon profile length {arr.shape[0]} != N={N}.")
        return arr.reshape(N)

    @staticmethod
    def _per_step_weight_matrices(param: Any, N: int, dim: int) -> list[np.ndarray]:
        """Convert a weight param to a list of N (dim×dim) matrices.

        Accepts:
        - scalar → ``scalar · I_dim`` replicated N times.
        - (dim, dim) ndarray → constant matrix replicated N times.
        - (N,) ndarray → per-step ``arr[k] · I_dim``.
        - (N, dim) ndarray → per-step ``diag(arr[k, :])``.

        When N == dim a 1-D input of length N is treated as per-step scalars.
        """
        arr = np.asarray(param, dtype=float)
        if arr.ndim == 0 or arr.size == 1:
            mat = float(arr.flat[0]) * np.eye(dim)
            return [mat] * N
        if arr.ndim == 2 and arr.shape == (dim, dim):
            return [arr] * N
        if arr.ndim == 1 and arr.shape[0] == N:
            return [float(arr[k]) * np.eye(dim) for k in range(N)]
        if arr.ndim == 2 and arr.shape == (N, dim):
            return [np.diag(arr[k]) for k in range(N)]
        raise ValueError(
            f"Cannot interpret weight of shape {arr.shape} for N={N}, dim={dim}. "
            f"Expected: scalar, ({dim},{dim}) matrix, ({N},) per-step scalars, "
            f"or ({N},{dim}) per-step diagonal vectors."
        )

    @staticmethod
    def _terminal_weight_matrix(param: Any, dim: int) -> np.ndarray:
        """Convert a terminal weight param to a (dim×dim) matrix.

        Accepts scalar, (dim,) diagonal vector, or (dim, dim) matrix.
        """
        arr = np.asarray(param, dtype=float)
        if arr.ndim == 0 or arr.size == 1:
            return float(arr.flat[0]) * np.eye(dim)
        if arr.ndim == 1 and arr.size == dim:
            return np.diag(arr)
        if arr.ndim == 2 and arr.shape == (dim, dim):
            return arr.copy()
        raise ValueError(
            f"Cannot interpret terminal weight of shape {arr.shape} for dim={dim}. "
            f"Expected: scalar, ({dim},) diagonal vector, or ({dim},{dim}) matrix."
        )

    @staticmethod
    def _per_step_weight_vectors(param: Any, N: int, dim: int) -> np.ndarray:
        """Convert a weight param to an (N, dim) array.

        Accepts:
        - scalar → ``np.full((N, dim), scalar)``.
        - (N,) ndarray → per-step scalar broadcast to ``(N, dim)``.
        - (N, dim) ndarray → returned as-is.
        """
        arr = np.asarray(param, dtype=float)
        if arr.ndim == 0 or arr.size == 1:
            return np.full((N, dim), float(arr.flat[0]))
        if arr.ndim == 1 and arr.shape[0] == N:
            return np.tile(arr.reshape(N, 1), (1, dim))
        if arr.ndim == 2 and arr.shape == (N, dim):
            return arr.copy()
        raise ValueError(
            f"Cannot interpret weight of shape {arr.shape} for N={N}, dim={dim}. "
            f"Expected: scalar, ({N},) per-step scalars, or ({N},{dim}) per-step vectors."
        )

    def _resolve_disturbance(self, D: Any | None, nd: int) -> np.ndarray:
        if D is not None:
            return _any_to_np1d(D).reshape(-1)
        prof = self._horizon_profile.disturbance_profile
        if prof is None:
            raise ValueError(
                "Disturbance forecast required: pass D to solve() or "
                "call set_disturbance_profile()."
            )
        return _any_to_np1d(prof).reshape(-1)

    def _resolve_x_ref(self, x_ref: Any | None, nx: int) -> np.ndarray:
        if x_ref is not None:
            x_ref_np = _any_to_np1d(x_ref).reshape(-1)
        else:
            x_ref_np = np.asarray(self._model.x_ref, dtype=float).reshape(-1)
        prof = self._horizon_profile.output_reference_deviation_profile
        if prof is None:
            return x_ref_np
        dev = np.asarray(prof, dtype=float)
        if dev.ndim == 1 and dev.size == nx:
            return x_ref_np + dev
        if dev.ndim == 1 and dev.size == self._N * nx:
            return x_ref_np + dev[:nx]
        return x_ref_np

    def _per_step_output_references(
        self, Cz: np.ndarray, x_ref: np.ndarray, nz: int, N: int,
    ) -> np.ndarray:
        z_ref = Cz @ x_ref
        prof = self._horizon_profile.output_reference_deviation_profile
        if prof is None:
            return np.tile(z_ref, N)
        dev = np.asarray(prof, dtype=float)
        if dev.ndim == 1 and dev.size == nz:
            return np.tile(z_ref + dev, N)
        if dev.ndim == 2 and dev.shape == (N, nz):
            return (np.tile(z_ref, (N, 1)) + dev).reshape(-1)
        if dev.ndim == 1 and dev.size == N * nz:
            return (np.tile(z_ref, N) + dev).reshape(-1)
        return np.tile(z_ref, N)

    def _resolve_input_equilibrium(self, nu: int) -> np.ndarray | None:
        prof = self._horizon_profile
        if prof.input_equilibrium is None:
            return None
        return np.asarray(prof.input_equilibrium, dtype=float).reshape(nu)

    def _resolve_linear_cost_layout(self):
        prof = self._horizon_profile
        u_min, u_max = self._model.u_bounds
        return resolve_input_linear_cost(
            coefficient_profile=prof.input_linear_cost_coefficient_profile,
            N=self._N,
            nu=self._model.nu,
            u_min=_any_to_np1d(u_min),
            u_max=_any_to_np1d(u_max),
            slack_input_indices=prof.slack_input_indices,
            positive_slack_coefficient_profile=prof.positive_slack_coefficient_profile,
            negative_slack_coefficient_profile=prof.negative_slack_coefficient_profile,
        )

    def _per_step_band_half_widths(self, N: int, nz: int) -> np.ndarray:
        """Return (N, nz) band half-width array (per-step, per-output)."""
        prof = self._horizon_profile.soft_output_band_half_width_profile
        source = prof if prof is not None else self._y_offset
        return self._per_step_weight_vectors(source, N, nz)

    # ── Public solve ────────────────────────────────────────────────────────

    def solve(
        self,
        x0: Any,
        D: Any | None = None,
        x_ref: Any | None = None,
        u_prev: Any | None = None,
        warm_start: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the QP starting from state estimate ``x0``.

        Disturbances, references, and other horizon quantities are taken from
        call-time arguments when provided; otherwise from :attr:`horizon_profile`
        (configured via the profile setters).

        Parameters
        ----------
        x0    : (nx,) array-like — current state estimate ``x̂_{k|k}``.
        D     : (N · nd,) array-like, optional — disturbance forecast.  Falls
                back to :attr:`horizon_profile.disturbance_profile`.
        x_ref : (nx,) array-like, optional — state reference.  Falls back to
                ``model.x_ref`` plus any output-reference deviation profile.
        u_prev : (nu,) array-like, optional
            Previously-applied input (used for ROM penalty / hard ROM limits).
        warm_start : dict, optional
            ``{"U": (N·nu,), "X": (N·nx,)}`` primal warm-start trajectory.

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

        x0 = _any_to_np1d(x0).reshape(-1)
        x_ref = self._resolve_x_ref(x_ref, nx)
        D = self._resolve_disturbance(D, nd)

        Ad = _any_to_np2d(self._model.Ad)
        Bd = _any_to_np2d(self._model.Bd)
        Ed = _any_to_np2d(self._model.Ed)

        if self._S_bar is not None or self._du_min is not None or self._du_max is not None:
            u_prev_np = (
                np.zeros(nu) if u_prev is None
                else _any_to_np1d(u_prev).reshape(-1)
            )
        else:
            u_prev_np = None

        formulation = self._resolve_formulation()
        if formulation == "sparse":
            qp_kwargs, extract = self._build_sparse(
                x0, D, x_ref, Ad, Bd, Ed, Cz, nx, nu, nd, nz, u_prev_np
            )
        else:
            qp_kwargs, extract = self._build_condensed(
                x0, D, x_ref, Ad, Bd, Ed, Cz, nx, nu, nd, nz, u_prev_np
            )

        z_warm = self._assemble_warm(formulation, warm_start, N, nx, nu, nz)

        result = self._backend.solve(QPProblem(warm_start=z_warm, **qp_kwargs))

        if not result.success:
            warnings.warn(
                f"StandardLinearDiscreteOCP.solve: QP solver returned status "
                f"'{result.status}'; returning zero inputs as fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            U_flat = np.zeros(N * nu)
            X_flat = self._simulate(x0, U_flat, D, Ad, Bd, Ed, N, nx, nu, nd)
            return U_flat, X_flat

        return extract(np.asarray(result.x, dtype=float))

    # ── Warm-start assembly ────────────────────────────────────────────────

    def _assemble_warm(
        self,
        formulation: str,
        warm_start: dict[str, np.ndarray] | None,
        N: int,
        nx: int,
        nu: int,
        nz: int,
    ) -> np.ndarray | None:
        """Map a physical ``{"U", "X"}`` warm start to the decision vector."""
        if warm_start is None:
            return None
        layout = self._resolve_linear_cost_layout()
        n_slack_st = 2 * layout.n_st if layout is not None and layout.has_slack else 0
        n_U = N * nu
        n_eps = N * nz
        n_X = N * nx
        U_w = warm_start.get("U")
        if U_w is None:
            return None
        U_w = np.asarray(U_w, dtype=float).reshape(-1)
        if U_w.shape[0] != n_U:
            return None
        slack_pad = np.zeros(n_slack_st)
        if formulation == "condensed":
            return np.concatenate([U_w, slack_pad, np.zeros(n_eps)])
        X_w = warm_start.get("X")
        if X_w is None:
            return None
        X_w = np.asarray(X_w, dtype=float).reshape(-1)
        if X_w.shape[0] != n_X:
            return None
        return np.concatenate([X_w, U_w, slack_pad, np.zeros(n_eps)])

    # ── Forward simulation (fallback X reconstruction) ─────────────────────

    @staticmethod
    def _simulate(
        x0: np.ndarray,
        U_flat: np.ndarray,
        D: np.ndarray,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Ed: np.ndarray,
        N: int,
        nx: int,
        nu: int,
        nd: int,
    ) -> np.ndarray:
        """Roll the deterministic dynamics forward to get ``X = [x₁ … x_N]``."""
        X = np.zeros(N * nx)
        xk = x0
        for k in range(N):
            uk = U_flat[k * nu:(k + 1) * nu]
            dk = D[k * nd:(k + 1) * nd]
            xk = Ad @ xk + Bd @ uk + Ed @ dk
            X[k * nx:(k + 1) * nx] = xk
        return X

    # ── Condensed (dense, state-eliminated) builder ────────────────────────

    def _build_condensed(
        self, x0, D, x_ref, Ad, Bd, Ed, Cz, nx, nu, nd, nz, u_prev_np,
    ) -> tuple[dict[str, Any], Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        N = self._N
        prof = self._horizon_profile
        q_scales = self._per_step_scales(prof.output_tracking_weight_scale_profile, N)
        r_scales = self._per_step_scales(
            prof.input_regularisation_weight_scale_profile, N
        )

        Ad_pow = [np.eye(nx)]
        for _ in range(N):
            Ad_pow.append(Ad @ Ad_pow[-1])

        Psi = np.zeros((N * nx, nx))
        Gamma = np.zeros((N * nx, N * nu))
        Lambda = np.zeros((N * nx, N * nd))
        for k in range(N):
            Psi[k * nx:(k + 1) * nx, :] = Ad_pow[k + 1]
            for j in range(k + 1):
                Ak = Ad_pow[k - j]
                Gamma[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = Ak @ Bd
                Lambda[k * nx:(k + 1) * nx, j * nd:(j + 1) * nd] = Ak @ Ed

        Cz_bar = np.kron(np.eye(N), Cz)
        CG = Cz_bar @ Gamma
        CP = Cz_bar @ Psi
        CL = Cz_bar @ Lambda

        Q_mats = [self._Q_mats[k] * q_scales[k] for k in range(N)]
        P_scaled = self._P_mat * q_scales[N - 1]
        R_mats = [self._R_mats[k] * r_scales[k] for k in range(N)]

        Q_bar = (
            block_diag(*(Q_mats[:N - 1] + [P_scaled])) if N > 1 else P_scaled
        )
        R_bar = block_diag(*R_mats)

        z_ref_bar = self._per_step_output_references(Cz, x_ref, nz, N)
        Z_free = CP @ x0 + CL @ D
        e_free = Z_free - z_ref_bar

        H_uu = CG.T @ Q_bar @ CG + R_bar
        f_u = CG.T @ Q_bar @ e_free

        if self._S_bar is not None:
            d0 = np.zeros(N * nu)
            d0[:nu] = -u_prev_np
            H_uu = H_uu + self._D_diff.T @ self._S_bar @ self._D_diff
            f_u = f_u + self._D_diff.T @ self._S_bar @ d0

        u_eq = self._resolve_input_equilibrium(nu)
        if u_eq is not None:
            for k in range(N):
                f_u[k * nu:(k + 1) * nu] += 2.0 * (R_mats[k] @ u_eq)

        n_U = N * nu
        n_eps = N * nz
        n_Z = n_U + n_eps

        rho_arr = self._per_step_weight_vectors(self._rho, N, nz)
        rho_diag = rho_arr.reshape(-1)

        H = np.zeros((n_Z, n_Z))
        H[:n_U, :n_U] = H_uu
        H[n_U:, n_U:] = np.diag(rho_diag)
        H = 0.5 * (H + H.T)

        f = np.zeros(n_Z)
        f[:n_U] = f_u
        rho_lin_arr = self._per_step_weight_vectors(self._rho_lin, N, nz)
        rho_lin_vec = rho_lin_arr.reshape(-1)
        if np.any(rho_lin_vec != 0.0):
            f[n_U:] = rho_lin_vec

        u_min, u_max = self._model.u_bounds
        prof = self._horizon_profile
        if prof.input_min_profile is not None and prof.input_max_profile is not None:
            u_min_t = np.asarray(prof.input_min_profile, dtype=float).reshape(-1)
            u_max_t = np.asarray(prof.input_max_profile, dtype=float).reshape(-1)
        else:
            u_min_t = np.tile(_any_to_np1d(u_min).reshape(-1), N)
            u_max_t = np.tile(_any_to_np1d(u_max).reshape(-1), N)
        lb = np.concatenate([u_min_t, np.zeros(n_eps)])
        ub = np.concatenate([u_max_t, np.full(n_eps, np.inf)])

        band = self._per_step_band_half_widths(N, nz)  # (N, nz)
        z_min_t = np.zeros(N * nz)
        z_max_t = np.zeros(N * nz)
        for k in range(N):
            z_ref_k = z_ref_bar[k * nz:(k + 1) * nz]
            z_min_t[k * nz:(k + 1) * nz] = z_ref_k - band[k]
            z_max_t[k * nz:(k + 1) * nz] = z_ref_k + band[k]
        neg_I = -np.eye(n_eps)
        G = np.vstack([np.hstack([-CG, neg_I]), np.hstack([CG, neg_I])])
        h = np.concatenate([-z_min_t + Z_free, z_max_t - Z_free])

        if self._du_min is not None or self._du_max is not None:
            d0 = self._rate_offset(u_prev_np, nu, N)
            G_rate: list[np.ndarray] = []
            h_rate: list[np.ndarray] = []
            if self._du_max is not None:
                G_rate.append(np.hstack([self._D_diff, np.zeros((N * nu, n_eps))]))
                h_rate.append(np.tile(self._du_max, N) - d0)
            if self._du_min is not None:
                G_rate.append(np.hstack([-self._D_diff, np.zeros((N * nu, n_eps))]))
                h_rate.append(-np.tile(self._du_min, N) + d0)
            G = np.vstack([G] + G_rate)
            h = np.concatenate([h] + h_rate)

        def extract(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            U_flat = z[:n_U]
            X_flat = Psi @ x0 + Gamma @ U_flat + Lambda @ D
            return U_flat, X_flat

        qp = {"P": H, "q": f, "lb": lb, "ub": ub, "G": G, "h": h}
        layout = self._resolve_linear_cost_layout()
        if layout is not None and layout.has_any:
            qp = augment_condensed_qp(
                qp, layout=layout, N=N, nu=nu, n_eps=n_eps,
                input_equilibrium=u_eq,
            )
        return qp, extract

    # ── Sparse (simultaneous, banded) builder ──────────────────────────────

    def _build_sparse(
        self, x0, D, x_ref, Ad, Bd, Ed, Cz, nx, nu, nd, nz, u_prev_np,
    ) -> tuple[dict[str, Any], Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        import scipy.sparse as sp

        N = self._N
        n_X = N * nx
        n_U = N * nu
        n_eps = N * nz
        n_Z = n_X + n_U + n_eps
        oU = n_X
        oE = n_X + n_U

        prof = self._horizon_profile
        q_scales = self._per_step_scales(prof.output_tracking_weight_scale_profile, N)
        r_scales = self._per_step_scales(
            prof.input_regularisation_weight_scale_profile, N
        )
        z_ref_bar = self._per_step_output_references(Cz, x_ref, nz, N)
        band = self._per_step_band_half_widths(N, nz)  # (N, nz)

        Q_mats = [self._Q_mats[k] * q_scales[k] for k in range(N)]
        P_scaled = self._P_mat * q_scales[N - 1]
        R_mats = [self._R_mats[k] * r_scales[k] for k in range(N)]

        f = np.zeros(n_Z)

        Czt = Cz.T
        x_blocks = []
        for k in range(N):
            Qk = P_scaled if k == N - 1 else Q_mats[k]
            x_blocks.append(Czt @ Qk @ Cz)
            z_ref_k = z_ref_bar[k * nz:(k + 1) * nz]
            f[k * nx:(k + 1) * nx] = -Czt @ (Qk @ z_ref_k)

        R_bar = block_diag(*R_mats)
        if self._S_bar is not None:
            d0 = np.zeros(n_U)
            d0[:nu] = -u_prev_np
            H_uu = R_bar + self._D_diff.T @ self._S_bar @ self._D_diff
            f[oU:oU + n_U] = self._D_diff.T @ self._S_bar @ d0
        else:
            H_uu = R_bar

        u_eq = self._resolve_input_equilibrium(nu)
        if u_eq is not None:
            for k in range(N):
                f[oU + k * nu:oU + (k + 1) * nu] += 2.0 * (R_mats[k] @ u_eq)

        rho_arr = self._per_step_weight_vectors(self._rho, N, nz)
        rho_diag = rho_arr.reshape(-1)
        rho_lin_arr = self._per_step_weight_vectors(self._rho_lin, N, nz)
        rho_lin_vec = rho_lin_arr.reshape(-1)
        if np.any(rho_lin_vec != 0.0):
            f[oE:oE + n_eps] = rho_lin_vec

        H = sp.block_diag(
            x_blocks + [H_uu, sp.diags(rho_diag, format="csc")],
            format="csc",
        )

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        def _add_block(r0, c0, M):
            Mr, Mc = M.shape
            for i in range(Mr):
                for j in range(Mc):
                    v = M[i, j]
                    if v != 0.0:
                        rows.append(r0 + i)
                        cols.append(c0 + j)
                        vals.append(v)

        b_eq = np.zeros(n_X)
        eye_nx = np.eye(nx)
        for k in range(N):
            rs = k * nx
            _add_block(rs, k * nx, eye_nx)
            _add_block(rs, oU + k * nu, -Bd)
            dk = D[k * nd:(k + 1) * nd]
            if k == 0:
                b_eq[rs:rs + nx] = Ad @ x0 + Ed @ dk
            else:
                _add_block(rs, (k - 1) * nx, -Ad)
                b_eq[rs:rs + nx] = Ed @ dk
        A_eq = sp.csc_matrix((vals, (rows, cols)), shape=(n_X, n_Z))

        u_min, u_max = self._model.u_bounds
        if prof.input_min_profile is not None and prof.input_max_profile is not None:
            u_min_t = np.asarray(prof.input_min_profile, dtype=float).reshape(-1)
            u_max_t = np.asarray(prof.input_max_profile, dtype=float).reshape(-1)
        else:
            u_min_t = np.tile(_any_to_np1d(u_min).reshape(-1), N)
            u_max_t = np.tile(_any_to_np1d(u_max).reshape(-1), N)
        lb = np.concatenate([np.full(n_X, -np.inf), u_min_t, np.zeros(n_eps)])
        ub = np.concatenate([np.full(n_X, np.inf), u_max_t, np.full(n_eps, np.inf)])

        g_rows: list[int] = []
        g_cols: list[int] = []
        g_vals: list[float] = []

        def _add_g(r0, c0, M):
            Mr, Mc = M.shape
            for i in range(Mr):
                for j in range(Mc):
                    v = M[i, j]
                    if v != 0.0:
                        g_rows.append(r0 + i)
                        g_cols.append(c0 + j)
                        g_vals.append(v)

        neg_eye_nz = -np.eye(nz)
        h = np.zeros(2 * n_eps)
        for k in range(N):
            xs = k * nx
            es = oE + k * nz
            r_hi = k * nz
            r_lo = n_eps + k * nz
            z_ref_k = z_ref_bar[k * nz:(k + 1) * nz]
            z_min = z_ref_k - band[k]   # band[k] is (nz,) vector
            z_max = z_ref_k + band[k]
            _add_g(r_hi, xs, Cz)
            _add_g(r_hi, es, neg_eye_nz)
            h[r_hi:r_hi + nz] = z_max
            _add_g(r_lo, xs, -Cz)
            _add_g(r_lo, es, neg_eye_nz)
            h[r_lo:r_lo + nz] = -z_min
        G = sp.csc_matrix((g_vals, (g_rows, g_cols)), shape=(2 * n_eps, n_Z))

        if self._du_min is not None or self._du_max is not None:
            d0 = self._rate_offset(u_prev_np, nu, N)
            G_rate_list = []
            h_rate_list: list[np.ndarray] = []
            if self._du_max is not None:
                G_rate_list.append(
                    sp.hstack([
                        sp.csc_matrix((N * nu, n_X)),
                        self._D_diff,
                        sp.csc_matrix((N * nu, n_eps)),
                    ])
                )
                h_rate_list.append(np.tile(self._du_max, N) - d0)
            if self._du_min is not None:
                G_rate_list.append(
                    sp.hstack([
                        sp.csc_matrix((N * nu, n_X)),
                        -self._D_diff,
                        sp.csc_matrix((N * nu, n_eps)),
                    ])
                )
                h_rate_list.append(-np.tile(self._du_min, N) + d0)
            G = sp.vstack([G] + G_rate_list, format="csc")
            h = np.concatenate([h] + h_rate_list)

        def extract(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            X_flat = z[:n_X]
            U_flat = z[oU:oU + n_U]
            return U_flat, X_flat

        qp = {
            "P": H, "q": f, "lb": lb, "ub": ub, "G": G, "h": h, "A": A_eq, "b": b_eq,
        }
        layout = self._resolve_linear_cost_layout()
        if layout is not None and layout.has_any:
            qp = augment_sparse_qp(
                qp, layout=layout, N=N, nu=nu, nx=nx, nz=nz,
                input_equilibrium=u_eq,
            )
        return qp, extract
