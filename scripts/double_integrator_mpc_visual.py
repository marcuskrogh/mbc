"""
Visual example: Discrete Linear KF + Linear Discrete MPC on a double integrator.

System
------
A position-velocity double integrator with noisy position measurements and an
external wind-force disturbance:

    x[k+1] = [[1, Ts], [0, 1]] x[k] + [[Ts²/2], [Ts]] u[k] + [[Ts²/2], [Ts]] d[k]
    y[k]   = [1, 0] x[k] + v[k],    v[k] ~ N(0, Rm)

    x = [p (m), v (m/s)],  u = [a (m/s²)] acceleration,  d = [w (m/s²)] wind force
    Ts = 0.5 s,  |u| ≤ 2 m/s²,  position noise σ = 0.2 m

Estimation
----------
A discrete-time Linear Kalman Filter (DiscreteLinearKF) infers both position
and velocity from position-only measurements.  Velocity is entirely unobserved
— the filter derives it by tracking measurement rate-of-change.

Control
-------
A Linear Discrete MPC (StandardLinearDiscreteMPC) solves a receding-horizon
QP at each step to track a piecewise-constant position reference, respecting
the ±2 m/s² input bound.  The wind force is unknown to the controller; the
MPC rejects it via feedback alone.

Scenario (50 s, 100 steps)
--------------------------
    t =  0–10 s  p_ref = 0 m    (at rest)
    t = 10–25 s  p_ref = 4 m    (step to 4 m)
    t = 20–23 s  wind = +0.2 m/s² (disturbance mid-manoeuvre)
    t = 25–40 s  p_ref = 1 m    (step back)
    t = 40–50 s  p_ref = 5 m    (step up)

Figure layout (4 rows)
-----------------------
1. Position — true (noisy sim), KF mean ± 2σ, measurements, reference
2. Velocity — true, KF mean ± 2σ  (inferred from position measurements)
3. Acceleration input with ±2 m/s² hard bounds
4. Wind disturbance + KF state-error variances (log scale)

Usage
-----
    python scripts/double_integrator_mpc_visual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from mbc.models import DiscreteLinearSDE
from mbc.estimation import DiscreteLinearKF
from mbc.control import StandardLinearDiscreteOCP, StandardLinearDiscreteMPC


# ── Timing ────────────────────────────────────────────────────────────────────

TS    = 0.5        # sampling period (s)
T_END = 50.0       # simulation horizon (s)
N_SIM = int(T_END / TS)   # 100 steps
N_MPC = 20         # MPC prediction horizon (steps = 10 s)

# ── Model ─────────────────────────────────────────────────────────────────────


class DoubleIntegrator(DiscreteLinearSDE):
    """
    Discrete-time double integrator.

    State      : x = [p (m), v (m/s)]
    Input      : u = [a (m/s²)]   — applied acceleration (control)
    Disturbance: d = [w (m/s²)]   — wind force (uncontrolled)
    Output     : y = [p (m)]      — position only

    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + w[k],   w[k] ~ N(0, Qd)
    y[k]   = Cm x[k] + v[k],                         v[k] ~ N(0, Rm)
    """

    _TS = TS
    _Ad = np.array([[1.0, _TS], [0.0, 1.0]])
    _Bd = np.array([[_TS ** 2 / 2], [_TS]])
    _Ed = np.array([[_TS ** 2 / 2], [_TS]])   # wind has same dynamics as input
    _Cm = np.array([[1.0, 0.0]])              # measure position only
    _Qd = np.diag([5e-3, 2e-2])              # process noise (position, velocity)
    _Rm = np.array([[4e-2]])                  # measurement noise (σ_p = 0.2 m)
    _U_MIN = np.array([-2.0])
    _U_MAX = np.array([+2.0])

    def __init__(self) -> None:
        self._x_ref = np.zeros(2)

    @property
    def nx(self) -> int: return 2
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 1
    @property
    def Ts(self) -> float: return self._TS
    @property
    def Ad(self) -> np.ndarray: return self._Ad.copy()
    @property
    def Bd(self) -> np.ndarray: return self._Bd.copy()
    @property
    def Ed(self) -> np.ndarray: return self._Ed.copy()
    @property
    def Cm(self) -> np.ndarray: return self._Cm.copy()
    @property
    def Qd(self) -> np.ndarray: return self._Qd.copy()
    @property
    def Rm(self) -> np.ndarray: return self._Rm.copy()
    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._U_MIN.copy(), self._U_MAX.copy()
    @property
    def x_ref(self) -> np.ndarray: return self._x_ref.copy()
    @x_ref.setter
    def x_ref(self, val: np.ndarray) -> None:
        self._x_ref = np.asarray(val, dtype=float).reshape(2)


# ── Simulation ────────────────────────────────────────────────────────────────


def simulate(
    model: DoubleIntegrator,
    x0: np.ndarray,
    U: np.ndarray,   # (N_SIM, nu)
    D: np.ndarray,   # (N_SIM, nd)
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the discrete-time plant one step at a time.

    Returns
    -------
    X_true : (N_SIM+1, nx) — true state trajectory (starts at x0)
    Y_meas : (N_SIM, nym)  — noisy position measurements
    """
    nx, nym = model.nx, model.nym
    L_Q = np.linalg.cholesky(model.Qd)
    R_std = float(np.sqrt(model.Rm[0, 0]))

    X_true = np.empty((N_SIM + 1, nx))
    Y_meas = np.empty((N_SIM, nym))
    X_true[0] = x0

    x = x0.copy()
    for k in range(N_SIM):
        noise = L_Q @ rng.standard_normal(nx)
        x = model.Ad @ x + model.Bd @ U[k] + model.Ed @ D[k] + noise
        X_true[k + 1] = x
        Y_meas[k] = model.Cm @ x + R_std * rng.standard_normal(nym)

    return X_true, Y_meas


# ── Scenario ──────────────────────────────────────────────────────────────────

X0_TRUE = np.array([0.0, 0.0])
X0_EST  = np.array([0.2, 0.0])   # small initial position error
P0_EST  = np.diag([0.5, 0.1])


def _build_scenario() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reference, wind disturbance, and measurement input arrays."""
    t = np.arange(N_SIM) * TS

    # Piecewise-constant position reference
    p_ref = np.where(t < 10.0, 0.0,
            np.where(t < 25.0, 4.0,
            np.where(t < 40.0, 1.0, 5.0)))  # (N_SIM,)

    # Wind disturbance (unknown to controller)
    wind = np.where((t >= 20.0) & (t < 23.0), 0.2, 0.0)   # (N_SIM,)

    return p_ref, wind


# ── Main ──────────────────────────────────────────────────────────────────────


def run() -> None:
    rng   = np.random.default_rng(7)
    model = DoubleIntegrator()

    p_ref, wind = _build_scenario()
    U_true = np.zeros((N_SIM, 1))   # filled during closed loop
    D_true = wind[:, None]          # (N_SIM, 1)

    # ── Estimator ──────────────────────────────────────────────────────────
    kf = DiscreteLinearKF(model, x0=X0_EST.copy(), P0=P0_EST.copy())

    # ── OCP ────────────────────────────────────────────────────────────────
    ocp = StandardLinearDiscreteOCP(
        model, N=N_MPC,
        Q=10.0,    # position tracking weight
        R=0.5,     # acceleration cost
        P=50.0,    # terminal tracking weight
        rho=1e3,   # soft-output penalty (not binding for this scenario)
        z_offset=0.5,
        solver="highs",
    )

    # ── MPC ────────────────────────────────────────────────────────────────
    mpc = StandardLinearDiscreteMPC(model, kf, ocp)

    # Initialise disturbance profile to zero (wind unknown to MPC)
    mpc.set_disturbance_profile(np.zeros((N_MPC, 1)))

    # ── Storage ────────────────────────────────────────────────────────────
    x_true = X0_TRUE.copy()
    X_true_hist = np.zeros((N_SIM + 1, 2))
    X_true_hist[0] = x_true

    x_hist     = np.zeros((N_SIM + 1, 2))   # KF estimates
    p_diag     = np.zeros((N_SIM + 1, 2))   # KF covariance diagonals
    x_hist[0]  = kf.x_hat
    p_diag[0]  = np.diag(kf.P)

    Y_meas_arr = np.zeros((N_SIM, 1))
    U_arr      = np.zeros((N_SIM, 1))

    # Horizon previews for overlay plot (save at 3 time indices)
    preview_steps = [20, 50, 80]
    previews: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ── Closed-loop simulation ──────────────────────────────────────────────
    # We drive the true plant directly (discrete map) + measurement noise
    L_Q    = np.linalg.cholesky(model.Qd)
    R_std  = float(np.sqrt(model.Rm[0, 0]))

    u_k = np.zeros(1)  # initial input

    for k in range(N_SIM):
        # ─ Measurement from true plant ─────────────────────────────────
        ym_k = model.Cm @ x_true + R_std * rng.standard_normal(1)
        Y_meas_arr[k] = ym_k

        # ─ Reference for this time step ────────────────────────────────
        mpc.set_output_reference_profile(np.full((N_MPC, 1), p_ref[k]))

        # ─ MPC compute ─────────────────────────────────────────────────
        u_k, U_seq, X_seq = mpc.compute(ym_k)
        U_arr[k] = u_k

        # Save horizon preview at key steps
        if k in preview_steps:
            t_prev = np.arange(k + 1, k + 1 + N_MPC) * TS
            X_dev_seq = X_seq.reshape(N_MPC, 2)
            previews[k] = (t_prev, X_dev_seq.copy())

        # ─ Advance true plant ──────────────────────────────────────────
        noise = L_Q @ rng.standard_normal(2)
        x_true = (
            model.Ad @ x_true
            + model.Bd @ u_k
            + model.Ed @ D_true[k]
            + noise
        )
        X_true_hist[k + 1] = x_true

        # ─ Store KF state ──────────────────────────────────────────────
        x_hist[k + 1]  = kf.x_hat
        p_diag[k + 1]  = np.diag(kf.P)

    _plot(
        X_true_hist, Y_meas_arr, x_hist, p_diag, U_arr, D_true, p_ref, previews,
        u_min=model._U_MIN[0], u_max=model._U_MAX[0],
    )


# ── Figure ────────────────────────────────────────────────────────────────────

COLORS = {
    "true":   "#e07b39",   # orange  — true trajectory
    "kf":     "#2166ac",   # blue    — KF mean
    "band":   "#2166ac",
    "meas":   "#c0392b",   # red     — measurements
    "ref":    "#555555",   # grey    — reference
    "input":  "#27ae60",   # green   — control input
    "wind":   "#8e44ad",   # purple  — wind disturbance
    "var_p":  "#e07b39",
    "var_v":  "#2166ac",
    "prev":   "#aaaaaa",   # light grey — MPC preview
}

PREVIEW_COLS = ["#6baed6", "#2171b5", "#084594"]  # blue shades per preview
PREVIEW_LS   = ["--", "-.", ":"]


def _plot(
    X_true: np.ndarray,
    Y_meas: np.ndarray,
    x_hist: np.ndarray,
    p_diag: np.ndarray,
    U_arr:  np.ndarray,
    D_arr:  np.ndarray,
    p_ref:  np.ndarray,
    previews: dict,
    u_min: float,
    u_max: float,
) -> None:
    t      = np.arange(N_SIM + 1) * TS
    t_meas = t[1:]

    sig_p = 2.0 * np.sqrt(np.abs(p_diag[:, 0]))
    sig_v = 2.0 * np.sqrt(np.abs(p_diag[:, 1]))

    fig = plt.figure(figsize=(11, 13))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Double Integrator  —  Discrete Linear KF  +  Linear Discrete MPC\n"
        "Position tracking from noisy measurements  (velocity inferred by filter)",
        fontsize=12, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        hspace=0.45,
        top=0.93, bottom=0.06,
        left=0.10, right=0.93,
    )
    ax_p = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_p)
    ax_u = fig.add_subplot(gs[2], sharex=ax_p)
    ax_c = fig.add_subplot(gs[3], sharex=ax_p)

    # ── Row 1 : Position ─────────────────────────────────────────────────

    # MPC horizon previews (plotted first so they appear under the main traces)
    for i, (k, (t_prev, X_prev)) in enumerate(previews.items()):
        t_prev_clip = t_prev[t_prev <= T_END]
        n = len(t_prev_clip)
        ax_p.plot(t_prev_clip, X_prev[:n, 0], color=PREVIEW_COLS[i],
                  lw=1.0, ls=PREVIEW_LS[i], alpha=0.7,
                  label=f"MPC horizon (t={k*TS:.0f} s)" if i == 0 else None)

    ax_p.fill_between(t, x_hist[:, 0] - sig_p, x_hist[:, 0] + sig_p,
                      color=COLORS["band"], alpha=0.18, label="KF ±2σ")
    ax_p.step(t_meas, p_ref, where="post", color=COLORS["ref"],
              lw=1.5, ls="--", label="Reference p_ref")
    ax_p.plot(t, X_true[:, 0], color=COLORS["true"], lw=1.8, label="True position")
    ax_p.plot(t, x_hist[:, 0], color=COLORS["kf"],   lw=1.8, label="KF mean")
    ax_p.scatter(t_meas, Y_meas[:, 0], s=6, color=COLORS["meas"],
                 alpha=0.5, zorder=5, label="Measurements")

    ax_p.set_ylabel("Position  p  (m)", fontsize=10)
    ax_p.set_title("Position  (measured output)", fontsize=10)
    ax_p.legend(fontsize=7, loc="upper left", framealpha=0.85, ncol=2)
    ax_p.grid(True, alpha=0.25)

    # ── Row 2 : Velocity (inferred) ──────────────────────────────────────

    ax_v.fill_between(t, x_hist[:, 1] - sig_v, x_hist[:, 1] + sig_v,
                      color=COLORS["band"], alpha=0.18, label="KF ±2σ")
    ax_v.plot(t, X_true[:, 1], color=COLORS["true"], lw=1.8, label="True velocity")
    ax_v.plot(t, x_hist[:, 1], color=COLORS["kf"],   lw=1.8, label="KF mean")
    ax_v.axhline(0, color="black", lw=0.6, ls="--", alpha=0.3)

    ax_v.set_ylabel("Velocity  v  (m/s)", fontsize=10)
    ax_v.set_title("Velocity  (hidden state — inferred by KF from position only)", fontsize=10)
    ax_v.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_v.grid(True, alpha=0.25)

    # ── Row 3 : Acceleration input ────────────────────────────────────────

    ax_u.step(t_meas, U_arr[:, 0], where="post",
              color=COLORS["input"], lw=1.8, label="Applied acceleration")
    ax_u.axhline(u_min, color="crimson", lw=1.0, ls="--", alpha=0.7, label="Bounds ±2 m/s²")
    ax_u.axhline(u_max, color="crimson", lw=1.0, ls="--", alpha=0.7)
    ax_u.set_ylabel("Acceleration  a  (m/s²)", fontsize=10)
    ax_u.set_title("Control input  (MPC-computed acceleration, |a| ≤ 2 m/s²)", fontsize=10)
    ax_u.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_u.grid(True, alpha=0.25)

    # ── Row 4 : Wind + covariance ─────────────────────────────────────────

    ax_c.semilogy(t, p_diag[:, 0], color=COLORS["var_p"], lw=1.5, label="Var(p)")
    ax_c.semilogy(t, p_diag[:, 1], color=COLORS["var_v"], lw=1.5, label="Var(v)")
    ax_c.set_ylabel("KF variance (m², m²/s²)", fontsize=10, color="black")
    ax_c.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_c.grid(True, alpha=0.25, which="both")

    ax_wind = ax_c.twinx()
    ax_wind.step(t_meas, D_arr[:, 0], where="post",
                 color=COLORS["wind"], lw=1.8, ls="--", label="Wind  d  (m/s²)")
    ax_wind.set_ylabel("Wind force  d  (m/s²)", color=COLORS["wind"], fontsize=10)
    ax_wind.tick_params(axis="y", labelcolor=COLORS["wind"])
    ax_wind.set_ylim(-0.05, 0.5)

    ax_c.set_xlabel("Time  (s)", fontsize=10)
    ax_c.set_title("KF state variances (log) + wind disturbance", fontsize=10)

    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_wind.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", framealpha=0.85)

    # ── Shared annotations ────────────────────────────────────────────────

    for ax in (ax_p, ax_v, ax_u, ax_c):
        for t_ev, label in [(10.0, "p→4m"), (20.0, "wind"), (25.0, "p→1m"), (40.0, "p→5m")]:
            ax.axvline(t_ev, color="dimgray", lw=0.7, ls=":", alpha=0.5)

    for t_ev, label in [(10.0, "p→4"), (20.0, "↑wind"), (25.0, "p→1"), (40.0, "p→5")]:
        ax_p.text(t_ev + 0.3, ax_p.get_ylim()[1] * 0.97, label,
                  fontsize=7, color="dimgray", va="top")

    plt.setp(ax_p.get_xticklabels(), visible=False)
    plt.setp(ax_v.get_xticklabels(), visible=False)
    plt.setp(ax_u.get_xticklabels(), visible=False)

    plt.show()


if __name__ == "__main__":
    run()
