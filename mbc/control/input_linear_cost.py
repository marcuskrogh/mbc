"""
Linear Mayer penalties on control inputs for discrete-time QPs.

Two regimes are supported:

**Direct** (one-sided inputs)
    Add ``c[k, j] · u[k, j]`` to the objective.  Appropriate when the input
    is economically one-sided, e.g. ``u_min[j] ≥ 0`` (heat-only valve) or
    ``u_max[j] ≤ 0``.

**Signed magnitude** (bidirectional inputs)
    Introduce nonnegative slacks ``s, t ≥ 0`` with ``u = s − t`` and penalise
    ``c⁺[k, j]·s[k, j] + c⁻[k, j]·t[k, j]``.  When ``c⁺ = c⁻`` this is a
    differentiable surrogate for ``c·|u|``.  Use for inputs that can be
    positive or negative (heat pumps, reversible actuators) while keeping a
    linear operating cost such as electricity price × power draw.

Selection rule
--------------
If ``slack_input_indices`` is omitted, inputs whose box bounds span zero
(``u_min[j] < 0 < u_max[j]``) automatically use signed-magnitude slacks;
all other inputs with a nonzero coefficient use the direct penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class InputLinearCostMode(str, Enum):
    """How the linear Mayer penalty is applied to one input channel."""

    DIRECT = "direct"
    """``c·u`` on the control variable directly."""

    SIGNED_MAGNITUDE = "signed_magnitude"
    """``u = s − t``, ``s, t ≥ 0``, penalise ``c⁺·s + c⁻·t`` (defaults ``c⁺ = c⁻ = c``)."""


@dataclass(frozen=True)
class InputLinearCostLayout:
    """Resolved per-horizon linear input-cost structure for QP assembly."""

    coeff: np.ndarray
    """``(N, nu)`` Mayer coefficients (same naming as horizon profile)."""
    direct_indices: np.ndarray
    """Input indices penalised as ``c·u``."""
    slack_indices: np.ndarray
    """Input indices using ``u = s − t`` decomposition."""
    pos_slack_coeff: np.ndarray
    """``(N, n_slack)`` coefficients on positive slacks ``s``."""
    neg_slack_coeff: np.ndarray
    """``(N, n_slack)`` coefficients on negative slacks ``t``."""

    @property
    def n_slack(self) -> int:
        return int(self.slack_indices.size)

    @property
    def n_st(self) -> int:
        """Total scalar slack variables ``s`` (equals ``t`` count)."""
        return self.n_slack * self.coeff.shape[0]

    @property
    def has_slack(self) -> bool:
        return self.n_slack > 0

    @property
    def has_direct(self) -> bool:
        return self.direct_indices.size > 0

    @property
    def has_any(self) -> bool:
        return self.has_slack or np.any(self.coeff != 0.0)


def infer_signed_magnitude_input_indices(
    u_min: np.ndarray,
    u_max: np.ndarray,
) -> np.ndarray:
    """
    Return input indices whose bounds span zero.

    These are the default candidates for signed-magnitude slack decomposition.
  """
    u_min = np.asarray(u_min, dtype=float).reshape(-1)
    u_max = np.asarray(u_max, dtype=float).reshape(-1)
    return np.flatnonzero((u_min < 0.0) & (u_max > 0.0))


def resolve_input_linear_cost(
    *,
    coefficient_profile: np.ndarray | None,
    N: int,
    nu: int,
    u_min: np.ndarray,
    u_max: np.ndarray,
    slack_input_indices: np.ndarray | None = None,
    positive_slack_coefficient_profile: np.ndarray | None = None,
    negative_slack_coefficient_profile: np.ndarray | None = None,
) -> InputLinearCostLayout | None:
    """
    Build a resolved layout from horizon-profile linear-cost fields.

    Parameters
    ----------
    coefficient_profile
        ``(N, nu)`` or ``(nu,)`` Mayer coefficients.  For slack inputs the
        entry is the default magnitude coefficient when asymmetric profiles
        are omitted.
    slack_input_indices
        Explicit list of input indices using ``u = s − t``.  When ``None``,
        defaults to :func:`infer_signed_magnitude_input_indices`.
    positive_slack_coefficient_profile, negative_slack_coefficient_profile
        Optional ``(N, n_slack)`` or ``(n_slack,)`` asymmetric coefficients.
    """
    if coefficient_profile is None:
        return None

    coeff = np.asarray(coefficient_profile, dtype=float)
    if coeff.ndim == 1:
        coeff = np.tile(coeff.reshape(1, -1), (N, 1))
    if coeff.shape != (N, nu):
        raise ValueError(
            f"coefficient_profile must have shape ({N}, {nu}); got {coeff.shape}."
        )

    if slack_input_indices is None:
        slack_idx = infer_signed_magnitude_input_indices(u_min, u_max)
    else:
        slack_idx = np.asarray(slack_input_indices, dtype=int).reshape(-1)

    slack_set = set(slack_idx.tolist())
    direct_idx = np.array([j for j in range(nu) if j not in slack_set], dtype=int)

    n_slack = slack_idx.size
    if n_slack == 0:
        pos = np.zeros((N, 0))
        neg = np.zeros((N, 0))
    else:
        default_pos = coeff[:, slack_idx]
        default_neg = coeff[:, slack_idx]
        pos = _expand_slack_coeff(positive_slack_coefficient_profile, N, n_slack, default_pos)
        neg = _expand_slack_coeff(negative_slack_coefficient_profile, N, n_slack, default_neg)

    if not np.any(coeff != 0.0) and n_slack == 0:
        return None

    return InputLinearCostLayout(
        coeff=coeff,
        direct_indices=direct_idx,
        slack_indices=slack_idx,
        pos_slack_coeff=pos,
        neg_slack_coeff=neg,
    )


def _expand_slack_coeff(
    profile: np.ndarray | None,
    N: int,
    n_slack: int,
    default: np.ndarray,
) -> np.ndarray:
    if profile is None:
        return default.copy()
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 1:
        arr = np.tile(arr.reshape(1, -1), (N, 1))
    if arr.shape != (N, n_slack):
        raise ValueError(
            f"slack coefficient profile must have shape ({N}, {n_slack}); got {arr.shape}."
        )
    return arr


def _slack_flat_index(k: int, m: int, n_slack: int) -> int:
    return k * n_slack + m


def absolute_quadratic_input_regularisation_linear_term(
    R: np.ndarray,
    u_equilibrium: np.ndarray,
    N: int,
    nu: int,
    r_scales: np.ndarray,
) -> np.ndarray:
    """
    Linear QP term so ``½δuᵀRδu + fᵀδu`` matches ``½(u_eq+δu)ᵀR(u_eq+δu)`` up to constant.

    When the QP optimises deviation inputs ``δu`` but ``R`` should penalise
  absolute inputs ``u = u_eq + δu``, the cross term ``2 u_eqᵀ R δu`` belongs in
    ``f``.
    """
    u_eq = np.asarray(u_equilibrium, dtype=float).reshape(nu)
    R = np.asarray(R, dtype=float)
    scales = np.asarray(r_scales, dtype=float).reshape(N)
    f = np.zeros(N * nu)
    for k in range(N):
        f[k * nu:(k + 1) * nu] = 2.0 * float(scales[k]) * (R @ u_eq)
    return f


def augment_condensed_qp(
    qp: dict[str, Any],
    *,
    layout: InputLinearCostLayout,
    N: int,
    nu: int,
    n_eps: int,
    input_equilibrium: np.ndarray | None = None,
) -> dict[str, Any]:
    """Append signed slack variables and linking equalities to a condensed QP."""
    H = np.asarray(qp["P"], dtype=float)
    f = np.asarray(qp["q"], dtype=float).reshape(-1)
    lb = np.asarray(qp["lb"], dtype=float).reshape(-1)
    ub = np.asarray(qp["ub"], dtype=float).reshape(-1)
    G = np.asarray(qp["G"], dtype=float)
    h = np.asarray(qp["h"], dtype=float).reshape(-1)

    n_U = N * nu
    n_base = n_U + n_eps
    u_eq = (
        None if input_equilibrium is None
        else np.asarray(input_equilibrium, dtype=float).reshape(nu)
    )

    for k in range(N):
        for j in layout.direct_indices:
            f[k * nu + int(j)] += layout.coeff[k, int(j)]

    if not layout.has_slack:
        qp["q"] = f
        return qp

    n_st = layout.n_st
    n_Z = n_U + 2 * n_st + n_eps
    oE = n_U + 2 * n_st

    H_new = np.zeros((n_Z, n_Z))
    H_new[:n_U, :n_U] = H[:n_U, :n_U]
    if n_eps > 0:
        H_new[oE:oE + n_eps, oE:oE + n_eps] = H[n_U:n_base, n_U:n_base]

    f_new = np.zeros(n_Z)
    f_new[:n_U] = f[:n_U]
    if n_eps > 0:
        f_new[oE:oE + n_eps] = f[n_U:n_base]

    lb_new = np.full(n_Z, -np.inf)
    ub_new = np.full(n_Z, np.inf)
    lb_new[:n_U] = lb[:n_U]
    ub_new[:n_U] = ub[:n_U]
    lb_new[n_U:n_U + 2 * n_st] = 0.0
    if n_eps > 0:
        lb_new[oE:oE + n_eps] = lb[n_U:n_base]
        ub_new[oE:oE + n_eps] = ub[n_U:n_base]

    oS = n_U
    oT = n_U + n_st
    for k in range(N):
        for m, j in enumerate(layout.slack_indices):
            st = _slack_flat_index(k, m, layout.n_slack)
            f_new[oS + st] += layout.pos_slack_coeff[k, m]
            f_new[oT + st] += layout.neg_slack_coeff[k, m]

    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    for k in range(N):
        for m, j in enumerate(layout.slack_indices):
            st = _slack_flat_index(k, m, layout.n_slack)
            row = np.zeros(n_Z)
            row[k * nu + int(j)] = 1.0
            row[oS + st] = -1.0
            row[oT + st] = 1.0
            A_rows.append(row)
            rhs = 0.0 if u_eq is None else -float(u_eq[int(j)])
            b_rows.append(rhs)

    A_link = np.vstack(A_rows)
    b_link = np.asarray(b_rows, dtype=float)

    if G.shape[1] == n_base:
        G_pad = np.hstack([
            G[:, :n_U],
            np.zeros((G.shape[0], 2 * n_st)),
            G[:, n_U:],
        ])
    else:
        G_pad = np.hstack([G, np.zeros((G.shape[0], 2 * n_st))])

    A_existing = qp.get("A")
    b_existing = qp.get("b")
    if A_existing is not None:
        A_existing = np.asarray(A_existing, dtype=float)
        if A_existing.shape[1] == n_base:
            A_pad = np.hstack([
                A_existing[:, :n_U],
                np.zeros((A_existing.shape[0], 2 * n_st)),
                A_existing[:, n_U:],
            ])
        else:
            A_pad = np.hstack([
                A_existing,
                np.zeros((A_existing.shape[0], 2 * n_st)),
            ])
        A_all = np.vstack([A_pad, A_link])
        b_all = np.concatenate([np.asarray(b_existing, dtype=float).reshape(-1), b_link])
    else:
        A_all = A_link
        b_all = b_link

    out = dict(qp)
    out["P"] = H_new
    out["q"] = f_new
    out["lb"] = lb_new
    out["ub"] = ub_new
    out["G"] = G_pad
    out["h"] = h
    out["A"] = A_all
    out["b"] = b_all
    return out


def augment_sparse_qp(
    qp: dict[str, Any],
    *,
    layout: InputLinearCostLayout,
    N: int,
    nu: int,
    nx: int,
    nz: int,
    input_equilibrium: np.ndarray | None = None,
) -> dict[str, Any]:
    """Append signed slack variables and linking equalities to a sparse QP."""
    import scipy.sparse as sp

    n_X = N * nx
    n_U = N * nu
    n_eps = N * nz
    n_base = n_X + n_U + n_eps
    oU = n_X
    oE = n_X + n_U

    f = np.asarray(qp["q"], dtype=float).reshape(-1)
    lb = np.asarray(qp["lb"], dtype=float).reshape(-1)
    ub = np.asarray(qp["ub"], dtype=float).reshape(-1)
    u_eq = (
        None if input_equilibrium is None
        else np.asarray(input_equilibrium, dtype=float).reshape(nu)
    )

    for k in range(N):
        for j in layout.direct_indices:
            f[oU + k * nu + int(j)] += layout.coeff[k, int(j)]

    if not layout.has_slack:
        qp["q"] = f
        return qp

    n_st = layout.n_st
    n_Z = n_base + 2 * n_st
    oS = n_base
    oT = n_base + n_st

    H = qp["P"]
    if not sp.issparse(H):
        H = sp.csc_matrix(H)
    H_new = sp.block_diag([H, sp.csc_matrix((2 * n_st, 2 * n_st))], format="csc")

    f_new = np.zeros(n_Z)
    f_new[:n_base] = f
    for k in range(N):
        for m, j in enumerate(layout.slack_indices):
            st = _slack_flat_index(k, m, layout.n_slack)
            f_new[oS + st] += layout.pos_slack_coeff[k, m]
            f_new[oT + st] += layout.neg_slack_coeff[k, m]

    lb_new = np.full(n_Z, -np.inf)
    ub_new = np.full(n_Z, np.inf)
    lb_new[:n_base] = lb
    ub_new[:n_base] = ub
    lb_new[oS:oT + n_st] = 0.0

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_link: list[float] = []
    for k in range(N):
        for m, j in enumerate(layout.slack_indices):
            st = _slack_flat_index(k, m, layout.n_slack)
            r = len(b_link)
            cols.extend([oU + k * nu + int(j), oS + st, oT + st])
            rows.extend([r, r, r])
            vals.extend([1.0, -1.0, 1.0])
            rhs = 0.0 if u_eq is None else -float(u_eq[int(j)])
            b_link.append(rhs)

    A_link = sp.csc_matrix((vals, (rows, cols)), shape=(len(b_link), n_Z))

  # Pad existing constraints
    def _pad_mat(M, old_cols: int) -> Any:
        if M is None:
            return None
        M = M.tocsr() if sp.issparse(M) else sp.csr_matrix(M)
        if M.shape[1] == n_Z:
            return M
        pad = sp.csr_matrix((M.shape[0], n_Z - old_cols))
        return sp.hstack([M, pad], format="csc")

    G_pad = _pad_mat(qp.get("G"), n_base)
    A_dyn = _pad_mat(qp.get("A"), n_base)

    if A_dyn is not None:
        A_all = sp.vstack([A_dyn, A_link], format="csc")
        b_all = np.concatenate([
            np.asarray(qp["b"], dtype=float).reshape(-1),
            np.asarray(b_link, dtype=float),
        ])
    else:
        A_all = A_link
        b_all = np.asarray(b_link, dtype=float)

    out = dict(qp)
    out["P"] = H_new
    out["q"] = f_new
    out["lb"] = lb_new
    out["ub"] = ub_new
    out["G"] = G_pad
    out["A"] = A_all
    out["b"] = b_all
    return out
