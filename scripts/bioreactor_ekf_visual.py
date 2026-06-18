"""
Visual inspection: open-loop simulation + CD-EKF state estimation
for a Monod fed-batch bioreactor.

Usage
-----
    python scripts/bioreactor_ekf_visual.py

Scenario
--------
A 20-hour fed-batch run with two step disturbances to stress the filter:

    t =  0–8 h   F/V = 0.05 h⁻¹,  S_in = 20 g/L
    t =  8–15 h  F/V = 0.10 h⁻¹   (washout risk)
    t = 15–20 h  F/V = 0.05 h⁻¹   (recovery)
    t = 10 h     S_in drops to 15 g/L (substrate step-down)

True kinetics: μ_max = 0.5 h⁻¹,  K_s = 0.2 g/L.
The EKF uses the correct parameters and observes biomass X only (S is hidden).

Figure layout (4 rows)
----------------------
1. Substrate S  — true (with stochastic noise), EKF mean ± 2σ (hidden state)
2. Biomass  X  — true, EKF mean ± 2σ, noisy measurements (×)
3. State variances  Var(S), Var(X)  on a log scale
4. Inputs F/V and disturbance S_in
"""

from __future__ import annotations

import sys
import os

# Make the package importable when run from the project root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from mbc.models import ContinuousDiscreteSDE
from mbc.estimation import ContinuousDiscreteEKF


# ── Model ─────────────────────────────────────────────────────────────────────

class MonodBioreactor(ContinuousDiscreteSDE):
    """
    Fed-batch bioreactor with Monod growth kinetics.

    State      : x = [S (g/L), X (g/L)]   substrate, biomass
    Input      : u = [F/V (1/h)]
    Disturbance: d = [S_in (g/L)]
    Output     : y = [X (g/L)]             biomass only
    Params     : p = [mu_max, K_s]

        dS/dt = −μ(S)·X/Y  + (S_in − S)·F/V
        dX/dt =  μ(S)·X    −  X·F/V
        μ(S)  = μ_max · S / (K_s + S)
    """

    _Y    = 0.5                        # yield  [g-biomass / g-substrate]
    _Qc   = np.diag([1e-4, 1e-4])     # continuous process-noise covariance
    _R    = np.array([[0.01]])         # measurement noise variance  (g/L)²

    # ── ContinuousDiscreteSDE abstract interface ────────────────────────

    @property
    def nx(self) -> int: return 2
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 1
    @property
    def ny(self) -> int: return 1
    @property
    def nw(self) -> int: return 2
    @property
    def Q_c(self) -> np.ndarray: return self._Qc.copy()
    @property
    def R(self) -> np.ndarray: return self._R.copy()

    def f(self, x, u, d, p, t):
        S, X = float(x[0]), float(x[1])
        S = max(S, 0.0)
        FV    = float(u[0])
        S_in  = float(d[0])
        mu_max, K_s = float(p[0]), float(p[1])
        mu = mu_max * S / (K_s + S)
        return np.array([
            -mu * X / self._Y + (S_in - S) * FV,
             mu * X            - X          * FV,
        ])

    def g(self, x, u, d, p, t):
        return np.eye(2)

    def h(self, x, u, d, p):
        return np.array([x[1]])          # observe biomass X


# ── Simulation helper (Euler-Maruyama with process noise) ─────────────────────

def simulate_em(
    model: MonodBioreactor,
    x0: np.ndarray,
    U: np.ndarray,        # (T, nu)
    D: np.ndarray,        # (T, nd)
    P_traj: np.ndarray,   # (T, nparams)
    dt: float,
    n_sub: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Euler-Maruyama simulation of a ContinuousDiscreteSDE.

    Returns
    -------
    X : (T+1, nx) state trajectory  (X[0] = x0)
    """
    h      = dt / n_sub
    Q_c    = model.Q_c
    L_Q    = np.linalg.cholesky(Q_c)        # lower-triangular factor
    x      = np.array(x0, dtype=float)
    X_hist = [x.copy()]
    T      = U.shape[0]

    for k in range(T):
        u_k = U[k]; d_k = D[k]; p_k = P_traj[k]
        for j in range(n_sub):
            drift = model.f(x, u_k, d_k, p_k, k * dt + j * h)
            dw    = rng.standard_normal(model.nw)
            x     = x + h * drift + np.sqrt(h) * L_Q @ dw
            x     = np.maximum(x, 0.0)         # enforce non-negative states
        X_hist.append(x.copy())

    return np.array(X_hist)


# ── Scenario ──────────────────────────────────────────────────────────────────

# Kinetic parameters
P_TRUE = np.array([0.5, 0.2])          # μ_max = 0.5 h⁻¹,  K_s = 0.2 g/L

# Timing
DT     = 0.1    # measurement sampling interval (h)
T_END  = 20.0   # simulation horizon (h)
N_SIM  = int(T_END / DT)              # number of measurement intervals = 200
N_SUB  = 20                            # EM sub-steps per interval
t_grid = np.arange(N_SIM + 1) * DT    # (N_SIM+1,) time points

# Initial conditions
X0_TRUE = np.array([5.0, 0.5])        # true IC: S=5 g/L, X=0.5 g/L
X0_EST  = np.array([5.8, 0.65])       # EKF IC: offset from truth
P0      = np.diag([1.0, 0.5])         # initial covariance


def _build_scenario(n: int, dt: float):
    """Build piecewise-constant U and D arrays for the 20 h scenario."""
    t = np.arange(n) * dt
    FV  = np.where(t < 8.0,  0.05, np.where(t < 15.0, 0.10, 0.05))
    Sin = np.where(t < 10.0, 20.0, 15.0)
    return FV[:, None], Sin[:, None]   # (N, 1) each


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    rng   = np.random.default_rng(42)
    model = MonodBioreactor()

    U, D    = _build_scenario(N_SIM, DT)
    P_traj  = np.tile(P_TRUE, (N_SIM, 1))

    # ── True stochastic trajectory ────────────────────────────────────────
    X_true = simulate_em(model, X0_TRUE, U, D, P_traj, DT, N_SUB, rng)

    # ── Noisy measurements (biomass only) ─────────────────────────────────
    R_std  = float(np.sqrt(model.R[0, 0]))
    Y_meas = X_true[1:, 1] + R_std * rng.standard_normal(N_SIM)

    # ── Open-loop trajectory from offset IC (no filter, for reference) ────
    X_ol = simulate_em(model, X0_EST, U, D, P_traj, DT, N_SUB,
                       np.random.default_rng(0))   # different noise seed

    # ── CD-EKF ────────────────────────────────────────────────────────────
    ekf = ContinuousDiscreteEKF(model, X0_EST.copy(), P0.copy(),
                                 dt=DT, n_steps=N_SUB)

    x_hist = np.zeros((N_SIM + 1, 2))
    p_diag = np.zeros((N_SIM + 1, 2))   # diagonal of P
    x_hist[0] = ekf.x_hat
    p_diag[0] = np.diag(ekf.P)

    for k in range(N_SIM):
        y_k = np.array([Y_meas[k]])
        ekf.step(y_k, U[k], D[k], P_TRUE, k * DT)
        x_hist[k + 1] = ekf.x_hat
        p_diag[k + 1] = np.diag(ekf.P)

    # ── Figure ────────────────────────────────────────────────────────────
    _plot(t_grid, X_true, X_ol, x_hist, p_diag, Y_meas, U, D)


def _plot(t, X_true, X_ol, x_hist, p_diag, Y_meas, U, D):
    C = {
        "true":    "#e07b39",   # warm orange — true trajectory
        "ol":      "#888888",   # grey        — open-loop
        "ekf":     "#2166ac",   # blue        — EKF mean
        "band":    "#2166ac",   # (same, transparent for band)
        "meas":    "#c0392b",   # red         — measurements
        "FV":      "#27ae60",   # green       — dilution rate
        "Sin":     "#8e44ad",   # purple      — S_in
    }

    fig = plt.figure(figsize=(11, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Monod Fed-Batch Bioreactor  —  Open-Loop Simulation & CD-EKF",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        hspace=0.50,
        top=0.93, bottom=0.06,
        left=0.09, right=0.91,
    )
    ax_S  = fig.add_subplot(gs[0])
    ax_X  = fig.add_subplot(gs[1], sharex=ax_S)
    ax_V  = fig.add_subplot(gs[2], sharex=ax_S)
    ax_IO = fig.add_subplot(gs[3], sharex=ax_S)

    sig_S = 2.0 * np.sqrt(np.abs(p_diag[:, 0]))
    sig_X = 2.0 * np.sqrt(np.abs(p_diag[:, 1]))
    t_meas = t[1:]

    # ── Row 1 : Substrate S (unobserved) ─────────────────────────────────
    ax_S.fill_between(
        t, x_hist[:, 0] - sig_S, x_hist[:, 0] + sig_S,
        color=C["band"], alpha=0.18, label="EKF ±2σ",
    )
    ax_S.plot(t, X_true[:, 0], color=C["true"],  lw=1.8,  label="True (noisy sim)")
    ax_S.plot(t, X_ol[:, 0],   color=C["ol"],    lw=1.0,  ls="--", label="Open-loop")
    ax_S.plot(t, x_hist[:, 0], color=C["ekf"],   lw=1.8,  label="EKF mean")
    ax_S.set_ylabel("S  (g/L)", fontsize=10)
    ax_S.set_title("Substrate  S  —  hidden state  (not measured)", fontsize=10)
    ax_S.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_S.grid(True, alpha=0.25)
    ax_S.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # ── Row 2 : Biomass X (observed) ─────────────────────────────────────
    ax_X.fill_between(
        t, x_hist[:, 1] - sig_X, x_hist[:, 1] + sig_X,
        color=C["band"], alpha=0.18, label="EKF ±2σ",
    )
    ax_X.plot(t, X_true[:, 1], color=C["true"],  lw=1.8,  label="True (noisy sim)")
    ax_X.plot(t, X_ol[:, 1],   color=C["ol"],    lw=1.0,  ls="--", label="Open-loop")
    ax_X.plot(t, x_hist[:, 1], color=C["ekf"],   lw=1.8,  label="EKF mean")
    ax_X.scatter(
        t_meas, Y_meas, s=7, color=C["meas"], alpha=0.55,
        zorder=5, label="Measurement  y_k",
    )
    ax_X.set_ylabel("X  (g/L)", fontsize=10)
    ax_X.set_title("Biomass  X  —  measured output", fontsize=10)
    ax_X.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_X.grid(True, alpha=0.25)
    ax_X.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # ── Row 3 : Covariance (log scale) ───────────────────────────────────
    ax_V.semilogy(t, p_diag[:, 0], color=C["true"],  lw=1.5, label="Var(S)")
    ax_V.semilogy(t, p_diag[:, 1], color=C["ekf"],   lw=1.5, label="Var(X)")
    ax_V.set_ylabel("Variance  (g/L)²", fontsize=10)
    ax_V.set_title("EKF state variance  —  P diagonal  (log scale)", fontsize=10)
    ax_V.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_V.grid(True, alpha=0.25, which="both")
    ax_V.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    # ── Row 4 : Inputs and disturbances ──────────────────────────────────
    ax_IO.step(t[:-1], U[:, 0] * 100, where="post",
               color=C["FV"], lw=1.8, label="F/V  ×100  (h⁻¹)")
    ax_IO.set_ylabel("F/V × 100  (h⁻¹)", fontsize=10, color=C["FV"])
    ax_IO.tick_params(axis="y", labelcolor=C["FV"])
    ax_IO.set_ylim(0, 15)

    ax_Sin = ax_IO.twinx()
    ax_Sin.step(t[:-1], D[:, 0], where="post",
                color=C["Sin"], lw=1.8, ls="--", label="$S_{in}$  (g/L)")
    ax_Sin.set_ylabel("$S_{in}$  (g/L)", fontsize=10, color=C["Sin"])
    ax_Sin.tick_params(axis="y", labelcolor=C["Sin"])
    ax_Sin.set_ylim(0, 30)

    ax_IO.set_xlabel("Time  (h)", fontsize=10)
    ax_IO.set_title("Inputs and disturbances", fontsize=10)
    ax_IO.grid(True, alpha=0.25)

    # Combined legend across both y-axes
    h1, l1 = ax_IO.get_legend_handles_labels()
    h2, l2 = ax_Sin.get_legend_handles_labels()
    ax_IO.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", framealpha=0.85)

    # Hide shared x-tick labels on upper rows
    plt.setp(ax_S.get_xticklabels(),  visible=False)
    plt.setp(ax_X.get_xticklabels(),  visible=False)
    plt.setp(ax_V.get_xticklabels(),  visible=False)

    # Step-change annotations
    for ax in (ax_S, ax_X, ax_V, ax_IO):
        for t_ev, label, va in [(8.0, "↑F/V", "top"), (10.0, "↓S_in", "top"), (15.0, "↓F/V", "top")]:
            ax.axvline(t_ev, color="dimgray", lw=0.8, ls=":", alpha=0.6)
    for t_ev, label in [(8.0, "↑F/V"), (10.0, "↓S_in"), (15.0, "↓F/V")]:
        ax_S.text(t_ev + 0.1, ax_S.get_ylim()[1] * 0.97, label,
                  fontsize=7, color="dimgray", va="top")

    plt.show()


if __name__ == "__main__":
    run()
