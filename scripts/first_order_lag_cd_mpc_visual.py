"""
Visual example: CD Linear KF + Linear CD MPC on a first-order lag.

System
------
A continuous-discrete first-order lag with noisy output measurements:

    dx/dt = −x + u,    y = x + v,    v ~ N(0, Rm)

    x ∈ ℝ (state),  u ∈ ℝ (input),  Ts = 0.5 s,  |u| ≤ 3

Estimation
----------
ContinuousDiscreteLinearKF propagates the linear SDE between samples and
updates from noisy measurements.

Control
-------
StandardLinearContinuousMPC solves a ZOH-discretised QP each step to track a
piecewise-constant output reference.

Scenario (30 s, 60 steps)
-------------------------
    t =  0–10 s  y_ref = 0
    t = 10–20 s  y_ref = 2
    t = 20–30 s  y_ref = 1

Usage
-----
    python scripts/first_order_lag_cd_mpc_visual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from mbc.models import ContinuousDiscreteLinearSDE
from mbc.estimation import ContinuousDiscreteLinearKF, ContinuousDiscreteLinearKFParams
from mbc.control import StandardLinearContinuousDiscreteOCP, StandardLinearContinuousMPC
from mbc._utils import _zoh_full


TS    = 0.5
T_END = 30.0
N_SIM = int(T_END / TS)
N_MPC = 15


class FirstOrderLag(ContinuousDiscreteLinearSDE):
    """dx/dt = −x + u,  y = x."""

    _A  = np.array([[-1.0]])
    _B  = np.array([[1.0]])
    _E  = np.array([[0.0]])
    _G  = np.array([[0.08]])
    _Cm = np.array([[1.0]])
    _Qc = np.array([[2e-3]])
    _Rm = np.array([[4e-2]])
    _U_MIN = np.array([-3.0])
    _U_MAX = np.array([+3.0])

    def __init__(self, x0: float = 0.0) -> None:
        self._x = [x0]
        self._x_ref = np.array([0.0])

    @property
    def nx(self) -> int: return 1
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 1
    @property
    def Ts(self) -> float: return TS
    @property
    def A(self) -> np.ndarray: return self._A.copy()
    @property
    def B(self) -> np.ndarray: return self._B.copy()
    @property
    def E(self) -> np.ndarray: return self._E.copy()
    @property
    def G(self) -> np.ndarray: return self._G.copy()
    @property
    def Cm(self) -> np.ndarray: return self._Cm.copy()
    @property
    def Qc(self) -> np.ndarray: return self._Qc.copy()
    @property
    def Rm(self) -> np.ndarray: return self._Rm.copy()
    @property
    def x(self) -> list[float]: return self._x
    @x.setter
    def x(self, val) -> None: self._x = list(val)
    @property
    def x_ref(self) -> np.ndarray: return self._x_ref.copy()
    @x_ref.setter
    def x_ref(self, val: np.ndarray) -> None:
        self._x_ref = np.asarray(val, dtype=float).reshape(1)
    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._U_MIN.copy(), self._U_MAX.copy()


def _y_ref_profile() -> np.ndarray:
    t = np.arange(N_SIM) * TS
    return np.where(t < 10.0, 0.0, np.where(t < 20.0, 2.0, 1.0))


def run() -> None:
    rng   = np.random.default_rng(11)
    model = FirstOrderLag(x0=0.0)

    y_ref = _y_ref_profile()
    Ad, Bd, _ = _zoh_full(model.A, model.B, model.E, model.Ts)
    Gd = model.G
    L_G = np.linalg.cholesky(Gd @ Gd.T * model.Ts)
    R_std = float(np.sqrt(model.Rm[0, 0]))

    kf = ContinuousDiscreteLinearKF(
        model, x0=np.array([0.15]), P0=np.array([[0.3]]),
        params=ContinuousDiscreteLinearKFParams(n_steps=10),
    )
    ocp = StandardLinearContinuousDiscreteOCP(
        model, N=N_MPC, Q=8.0, R=0.2, P=20.0, z_offset=0.5, solver="highs",
    )
    mpc = StandardLinearContinuousMPC(model, kf, ocp)
    mpc.set_disturbance_profile(np.zeros((N_MPC, 1)))

    x_true = np.array([0.0])
    X_true = np.zeros((N_SIM + 1, 1))
    X_true[0] = x_true
    x_hist = np.zeros((N_SIM + 1, 1))
    p_diag = np.zeros(N_SIM + 1)
    x_hist[0] = kf.x_hat
    p_diag[0] = float(kf.P[0, 0])
    Y_meas = np.zeros((N_SIM, 1))
    U_arr  = np.zeros((N_SIM, 1))

    for k in range(N_SIM):
        ym = model.Cm @ x_true + R_std * rng.standard_normal(1)
        Y_meas[k] = ym
        mpc.set_output_reference_profile(np.full((N_MPC, 1), y_ref[k]))
        u_k, _, _ = mpc.compute(ym)
        U_arr[k] = u_k
        proc = L_G @ rng.standard_normal(1)
        x_true = Ad @ x_true + Bd @ u_k + proc
        X_true[k + 1] = x_true
        x_hist[k + 1] = kf.x_hat
        p_diag[k + 1] = float(kf.P[0, 0])

    _plot(X_true, Y_meas, x_hist, p_diag, U_arr, y_ref)


def _plot(X_true, Y_meas, x_hist, p_diag, U_arr, y_ref) -> None:
    t = np.arange(N_SIM + 1) * TS
    t_meas = t[1:]
    sig = 2.0 * np.sqrt(np.abs(p_diag))

    fig = plt.figure(figsize=(11, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "First-Order Lag  —  CD Linear KF  +  Linear CD MPC",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.42, top=0.91, bottom=0.08,
                           left=0.10, right=0.93)
    ax_y = fig.add_subplot(gs[0])
    ax_u = fig.add_subplot(gs[1], sharex=ax_y)
    ax_v = fig.add_subplot(gs[2], sharex=ax_y)

    ax_y.fill_between(t, x_hist[:, 0] - sig, x_hist[:, 0] + sig,
                      color="#2166ac", alpha=0.18, label="KF ±2σ")
    ax_y.step(t_meas, y_ref, where="post", color="#555", ls="--", lw=1.5, label="Reference")
    ax_y.plot(t, X_true[:, 0], color="#e07b39", lw=1.8, label="True state")
    ax_y.plot(t, x_hist[:, 0], color="#2166ac", lw=1.8, label="KF mean")
    ax_y.scatter(t_meas, Y_meas[:, 0], s=8, color="#c0392b", alpha=0.5, label="Measurements")
    ax_y.set_ylabel("State  x")
    ax_y.set_title("Output tracking (y = x)")
    ax_y.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_y.grid(True, alpha=0.25)

    ax_u.step(t_meas, U_arr[:, 0], where="post", color="#27ae60", lw=1.8, label="Input u")
    ax_u.axhline(-3.0, color="crimson", ls="--", lw=1.0, alpha=0.7)
    ax_u.axhline(+3.0, color="crimson", ls="--", lw=1.0, alpha=0.7, label="Bounds ±3")
    ax_u.set_ylabel("Input  u")
    ax_u.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_u.grid(True, alpha=0.25)

    ax_v.semilogy(t, p_diag, color="#2166ac", lw=1.5)
    ax_v.set_ylabel("Var(x)")
    ax_v.set_xlabel("Time  (s)")
    ax_v.set_title("KF variance (log scale)")
    ax_v.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_v.grid(True, alpha=0.25, which="both")

    for ax in (ax_y, ax_u, ax_v):
        for t_ev in (10.0, 20.0):
            ax.axvline(t_ev, color="dimgray", lw=0.7, ls=":", alpha=0.5)
    plt.setp(ax_y.get_xticklabels(), visible=False)
    plt.setp(ax_u.get_xticklabels(), visible=False)
    plt.show()


if __name__ == "__main__":
    run()
