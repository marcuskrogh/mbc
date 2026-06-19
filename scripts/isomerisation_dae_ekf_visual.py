"""
Visual example: CD-DAE-EKF on an isomerisation reactor with fast equilibrium.

System
------
Differential-algebraic model (total concentration + fast equilibrium):

    dC_tot/dt = u · (C_feed − C_tot)
    0         = (K_eq + 1)·C_A − C_tot        (algebraic)
    y           = C_A                          (measured species A)

Estimation
----------
ContinuousDiscreteDAEEKF propagates the differential state through the
implicit constraint while estimating both C_tot (differential) and C_A
(algebraic, via the constraint).

Scenario (30 steps, dt = 1 s)
-----------------------------
    u = 0.3 for t < 10, then u = 0.8 (step change in dilution/feed rate)

Usage
-----
    python scripts/isomerisation_dae_ekf_visual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from mbc.models import ContinuousDiscreteSDAE
from mbc.estimation import ContinuousDiscreteDAEEKF, ContinuousDiscreteDAEEKFParams
from mbc.simulation import SDAESimulator


DT    = 1.0
N_SIM = 30
N_SUB = 10


class IsomerisationReactor(ContinuousDiscreteSDAE):
    _K_eq = 3.0
    _C_feed = 5.0

    @property
    def nx(self) -> int: return 1
    @property
    def ny(self) -> int: return 1
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 0
    @property
    def nw(self) -> int: return 1
    @property
    def nym(self) -> int: return 1
    @property
    def nz(self) -> int: return 1
    @property
    def Ts(self) -> float: return DT
    @property
    def Rm(self) -> np.ndarray: return np.array([[0.02]])

    def f(self, x, y, u, d, p, t):
        return np.array([u[0] * (self._C_feed - x[0])])

    def sigma(self, x, y, u, d, p, t):
        return np.array([[0.03]])

    def g(self, x, y, u, d, p, t):
        return np.array([(self._K_eq + 1.0) * y[0] - x[0]])

    def gm(self, x, y, u, d, p, t):
        return np.array([y[0]])

    def hm(self, x, y, u, d, p, t):
        return np.array([y[0]])


def _u_profile(k: int) -> float:
    return 0.3 if k * DT < 10.0 else 0.8


def run() -> None:
    rng   = np.random.default_rng(5)
    model = IsomerisationReactor()
    sim   = SDAESimulator(model, dt=DT, n_steps=N_SUB, seed=5)

    x0 = np.array([4.0])
    y0 = np.array([x0[0] / (model._K_eq + 1.0)])
    P0 = np.array([[0.1]])

    ekf = ContinuousDiscreteDAEEKF(
        model, x0=x0.copy(), y0=y0.copy(), P0=P0,
        params=ContinuousDiscreteDAEEKFParams(n_steps=N_SUB),
    )

    x_true, y_true = x0.copy(), y0.copy()
    X_true = np.zeros((N_SIM + 1, 1))
    Y_true = np.zeros((N_SIM + 1, 1))
    X_true[0], Y_true[0] = x_true, y_true
    x_hist = np.zeros((N_SIM + 1, 1))
    y_hist = np.zeros((N_SIM + 1, 1))
    p_diag = np.zeros(N_SIM + 1)
    x_hist[0] = ekf.x_hat
    y_hist[0] = ekf.y_hat
    p_diag[0] = float(ekf.P[0, 0])
    Y_meas = np.zeros(N_SIM)
    U_arr  = np.zeros(N_SIM)
    R_std  = float(np.sqrt(model.Rm[0, 0]))
    p_empty = np.array([], dtype=float)

    u_k = np.array([_u_profile(0)])
    for k in range(N_SIM):
        ym = model.hm(x_true, y_true, u_k, np.zeros(0), p_empty, k * DT)
        ym = ym + R_std * rng.standard_normal(1)
        Y_meas[k] = ym[0]

        ekf.step(ym, u_k, np.zeros(0), p_empty, k * DT)
        x_hist[k + 1] = ekf.x_hat
        y_hist[k + 1] = ekf.y_hat
        p_diag[k + 1] = float(ekf.P[0, 0])

        u_k = np.array([_u_profile(k)])
        U_arr[k] = u_k[0]
        x_true, y_true = sim.step(x_true, y_true, u_k, np.zeros(0), p_empty, k * DT)
        X_true[k + 1] = x_true
        Y_true[k + 1] = y_true

    _plot(X_true, Y_true, x_hist, y_hist, p_diag, Y_meas, U_arr)


def _plot(X_true, Y_true, x_hist, y_hist, p_diag, Y_meas, U_arr) -> None:
    t = np.arange(N_SIM + 1) * DT
    t_meas = t[1:]
    sig = 2.0 * np.sqrt(np.abs(p_diag))

    fig = plt.figure(figsize=(11, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Isomerisation Reactor (SDAE)  —  CD-DAE-EKF",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45, top=0.92, bottom=0.07,
                           left=0.10, right=0.93)
    ax_tot = fig.add_subplot(gs[0])
    ax_ca  = fig.add_subplot(gs[1], sharex=ax_tot)
    ax_u   = fig.add_subplot(gs[2], sharex=ax_tot)
    ax_v   = fig.add_subplot(gs[3], sharex=ax_tot)

    ax_tot.plot(t, X_true[:, 0], color="#e07b39", lw=1.8, label="True C_tot")
    ax_tot.plot(t, x_hist[:, 0], color="#2166ac", lw=1.8, label="EKF C_tot")
    ax_tot.set_ylabel("C_tot  (mol/L)")
    ax_tot.set_title("Differential state  C_tot")
    ax_tot.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_tot.grid(True, alpha=0.25)

    ax_ca.fill_between(t, y_hist[:, 0] - sig, y_hist[:, 0] + sig,
                       color="#2166ac", alpha=0.18, label="±2σ (via P_y)")
    ax_ca.plot(t, Y_true[:, 0], color="#e07b39", lw=1.8, label="True C_A")
    ax_ca.plot(t, y_hist[:, 0], color="#2166ac", lw=1.8, label="EKF C_A")
    ax_ca.scatter(t_meas, Y_meas, s=8, color="#c0392b", alpha=0.5, label="Measurements")
    ax_ca.set_ylabel("C_A  (mol/L)")
    ax_ca.set_title("Algebraic state  C_A  (fast equilibrium)")
    ax_ca.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_ca.grid(True, alpha=0.25)

    ax_u.step(t_meas, U_arr, where="post", color="#27ae60", lw=1.8, label="Feed rate u")
    ax_u.set_ylabel("u")
    ax_u.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_u.grid(True, alpha=0.25)

    ax_v.semilogy(t, p_diag, color="#2166ac", lw=1.5)
    ax_v.set_ylabel("Var(C_tot)")
    ax_v.set_xlabel("Time  (s)")
    ax_v.set_title("Differential-state variance (log scale)")
    ax_v.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_v.grid(True, alpha=0.25, which="both")

    for ax in (ax_tot, ax_ca, ax_u, ax_v):
        ax.axvline(10.0, color="dimgray", lw=0.7, ls=":", alpha=0.5)
    plt.setp(ax_tot.get_xticklabels(), visible=False)
    plt.setp(ax_ca.get_xticklabels(), visible=False)
    plt.setp(ax_u.get_xticklabels(), visible=False)
    plt.show()


if __name__ == "__main__":
    run()
