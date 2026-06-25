"""
Visual example: CD-EKF + Nonlinear Continuous MPC on a van de Vusse CSTR.

System
------
Van de Vusse kinetics (A → B → C, 2A → D) in a CSTR:

    dc_A/dt = (c_Af − c_A)·D − k1·c_A − k3·c_A²
    dc_B/dt = −c_B·D + k1·c_A − k2·c_B

    x = [c_A, c_B] (mol/L),  u = [D] (h⁻¹),  y = c_B (measured)

Estimation
----------
ContinuousDiscreteEKF infers both concentrations from c_B measurements only;
c_A is entirely hidden.

Control
-------
StandardNonlinearContinuousMPC with GeneralContinuousOCP tracks a piecewise
c_B setpoint by manipulating dilution rate D ∈ [0.1, 2.0] h⁻¹.

Scenario (5 h, dt = 0.1 h)
--------------------------
    t = 0–1.5 h   c_B,ref = 0.05 mol/L
    t = 1.5–3.5 h c_B,ref = 0.12 mol/L
    t = 3.5–5 h   c_B,ref = 0.08 mol/L

Usage
-----
    python scripts/van_de_vusse_enmpc_visual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from mbc.models import ContinuousDiscreteSDE
from mbc.estimation import ContinuousDiscreteEKF, ContinuousDiscreteEKFParams, IntegrationScheme
from mbc.control import GeneralContinuousOCP, StandardNonlinearContinuousMPC


DT      = 0.1
T_END   = 5.0
N_SIM   = int(T_END / DT)
N_MPC   = 20
N_SUB   = 10


class VanDeVusseCSTR(ContinuousDiscreteSDE):
    _k1, _k2, _k3, _c_Af = 50.0, 100.0, 10.0, 10.0
    _Qc = np.diag([0.01, 0.005])
    _Rm = np.array([[0.05]])

    @property
    def Ts(self) -> float: return DT
    @property
    def nx(self) -> int: return 2
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 0
    @property
    def nym(self) -> int: return 1
    @property
    def nw(self) -> int: return 2
    @property
    def nz(self) -> int: return 1
    @property
    def Rm(self) -> np.ndarray: return self._Rm.copy()

    def f(self, x, u, d, p, t):
        c_A, c_B = x
        D = u[0]
        dc_A = (self._c_Af - c_A) * D - self._k1 * c_A - self._k3 * c_A ** 2
        dc_B = -c_B * D + self._k1 * c_A - self._k2 * c_B
        return np.array([dc_A, dc_B])

    def sigma(self, x, u, d, p, t):
        return np.diag([0.1, np.sqrt(0.005)])

    def hm(self, x, u, d, p, t=0.0):
        return np.array([x[1]])

    def gm(self, x, u, d, p, t):
        return np.array([x[1]])


def _cb_ref(k: int) -> float:
    t = k * DT
    if t < 1.5:
        return 0.15
    if t < 3.5:
        return 0.25
    return 0.35


def _cb_ref_horizon(k: int) -> np.ndarray:
    """Piecewise c_B reference over every OCP sub-step in the horizon."""
    M = N_MPC * N_SUB
    refs = np.empty((M + 1, 1))
    for n in range(M + 1):
        refs[n, 0] = _cb_ref(k + n // N_SUB)
    return refs


def _em_step(x: np.ndarray, u: np.ndarray, model: VanDeVusseCSTR, rng, n_sub: int = N_SUB) -> np.ndarray:
    h = DT / n_sub
    for _ in range(n_sub):
        dw = rng.standard_normal(model.nw) * np.sqrt(h)
        sig = model.sigma(x, u, np.zeros(0), np.zeros(0), 0.0)
        x = x + h * model.f(x, u, np.zeros(0), np.zeros(0), 0.0) + sig @ dw
        x[0] = max(x[0], 0.0)
        x[1] = max(x[1], 0.0)
    return x


def run() -> None:
    rng   = np.random.default_rng(3)
    model = VanDeVusseCSTR()
    x0    = np.array([9.5, 0.05])
    P0    = np.diag([0.5, 0.02])

    ekf = ContinuousDiscreteEKF(
        model, x0=x0.copy(), P0=P0,
        params=ContinuousDiscreteEKFParams(n_steps=N_SUB),
    )
    # Explicit Euler matches the plant simulator; implicit Euler mis-predicts the
    # short-horizon inverse response in c_B and causes persistent undershoot.
    ocp = GeneralContinuousOCP(
        model, N=N_MPC,
        Q_z=np.diag([200.0]),
        R_stage=np.diag([0.02]),
        P_terminal=np.diag([50.0]),
        z_ref=np.array([_cb_ref(0)]),
        u_min=np.array([0.1]),
        u_max=np.array([5.0]),
        dt=DT,
        n_steps=N_SUB,
        scheme=IntegrationScheme.EXPLICIT_EULER,
        solver_options={"maxiter": 250},
    )
    mpc = StandardNonlinearContinuousMPC(ekf, ocp)
    d_horizon = np.zeros((N_MPC, 0))
    mpc.set_disturbance_profile(d_horizon)

    x_true = x0.copy()
    X_true = np.zeros((N_SIM + 1, 2))
    X_true[0] = x_true
    x_hist = np.zeros((N_SIM + 1, 2))
    p_diag = np.zeros((N_SIM + 1, 2))
    x_hist[0] = ekf.x_hat
    p_diag[0] = np.diag(ekf.P)
    Y_meas = np.zeros(N_SIM)
    U_arr  = np.zeros(N_SIM)
    R_std  = float(np.sqrt(model.Rm[0, 0]))

    u_k = np.array([0.5])
    for k in range(N_SIM):
        ym = model.hm(x_true, u_k, np.zeros(0), np.zeros(0), k * DT)
        ym = ym + R_std * rng.standard_normal(1)
        Y_meas[k] = ym[0]

        ocp._z_ref = _cb_ref_horizon(k)
        u_k = mpc.compute(ym, d_horizon, p=None, t=k * DT)
        U_arr[k] = u_k[0]

        x_true = _em_step(x_true, u_k, model, rng)
        X_true[k + 1] = x_true
        x_hist[k + 1] = ekf.x_hat
        p_diag[k + 1] = np.diag(ekf.P)

    z_ref_arr = np.array([_cb_ref(k) for k in range(N_SIM)])
    _plot(X_true, Y_meas, x_hist, p_diag, U_arr, z_ref_arr)


def _plot(X_true, Y_meas, x_hist, p_diag, U_arr, z_ref) -> None:
    t = np.arange(N_SIM + 1) * DT
    t_meas = t[1:]
    sig_A = 2.0 * np.sqrt(np.abs(p_diag[:, 0]))
    sig_B = 2.0 * np.sqrt(np.abs(p_diag[:, 1]))

    fig = plt.figure(figsize=(11, 11))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Van de Vusse CSTR  —  CD-EKF  +  Nonlinear Continuous MPC",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45, top=0.92, bottom=0.07,
                           left=0.10, right=0.93)
    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1], sharex=ax_A)
    ax_u = fig.add_subplot(gs[2], sharex=ax_A)
    ax_v = fig.add_subplot(gs[3], sharex=ax_A)

    ax_A.fill_between(t, x_hist[:, 0] - sig_A, x_hist[:, 0] + sig_A,
                      color="#2166ac", alpha=0.18, label="EKF ±2σ")
    ax_A.plot(t, X_true[:, 0], color="#e07b39", lw=1.8, label="True c_A")
    ax_A.plot(t, x_hist[:, 0], color="#2166ac", lw=1.8, label="EKF mean")
    ax_A.set_ylabel("c_A  (mol/L)")
    ax_A.set_title("Substrate A  —  hidden state")
    ax_A.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_A.grid(True, alpha=0.25)

    ax_B.fill_between(t, x_hist[:, 1] - sig_B, x_hist[:, 1] + sig_B,
                      color="#2166ac", alpha=0.18, label="EKF ±2σ")
    ax_B.step(t_meas, z_ref, where="post", color="#555", ls="--", lw=1.5, label="c_B ref")
    ax_B.plot(t, X_true[:, 1], color="#e07b39", lw=1.8, label="True c_B")
    ax_B.plot(t, x_hist[:, 1], color="#2166ac", lw=1.8, label="EKF mean")
    ax_B.scatter(t_meas, Y_meas, s=6, color="#c0392b", alpha=0.45, label="Measurements")
    ax_B.set_ylabel("c_B  (mol/L)")
    ax_B.set_title("Product B  —  measured & controlled output")
    ax_B.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_B.grid(True, alpha=0.25)

    ax_u.step(t_meas, U_arr, where="post", color="#27ae60", lw=1.8, label="D = F/V")
    ax_u.axhline(0.1, color="crimson", ls="--", lw=1.0, alpha=0.7)
    ax_u.axhline(2.0, color="crimson", ls="--", lw=1.0, alpha=0.7, label="Bounds")
    ax_u.set_ylabel("D  (h⁻¹)")
    ax_u.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_u.grid(True, alpha=0.25)

    ax_v.semilogy(t, p_diag[:, 0], color="#e07b39", lw=1.5, label="Var(c_A)")
    ax_v.semilogy(t, p_diag[:, 1], color="#2166ac", lw=1.5, label="Var(c_B)")
    ax_v.set_ylabel("EKF variance")
    ax_v.set_xlabel("Time  (h)")
    ax_v.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_v.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_v.grid(True, alpha=0.25, which="both")

    for ax in (ax_A, ax_B, ax_u, ax_v):
        for t_ev in (1.5, 3.5):
            ax.axvline(t_ev, color="dimgray", lw=0.7, ls=":", alpha=0.5)
    plt.setp(ax_A.get_xticklabels(), visible=False)
    plt.setp(ax_B.get_xticklabels(), visible=False)
    plt.setp(ax_u.get_xticklabels(), visible=False)
    plt.show()


if __name__ == "__main__":
    run()
