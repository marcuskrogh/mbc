"""
Visual example: Discrete Linear KF + Linearised Discrete MPC on a nonlinear
double integrator with velocity-dependent drag.

System
------
Nonlinear discrete dynamics (position–velocity with quadratic drag):

    p[k+1] = p + Ts·v
    v[k+1] = v + Ts·(u − 0.15·v·|v|)
    y[k]   = p + v[k]

The KF uses a *linear* double-integrator model (model mismatch).  The MPC
re-linearises the nonlinear map at each step.

Scenario (40 s, Ts = 0.5 s)
---------------------------
    p_ref steps: 0 → 3 m at t = 8 s, 3 → 1 m at t = 24 s

Usage
-----
    python scripts/nonlinear_double_integrator_ldmpc_visual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mbc.models import DiscreteLinearSDE
from mbc.estimation import DiscreteLinearKF
from mbc.control import StandardLinearisedDiscreteMPC


TS    = 0.5
T_END = 40.0
N_SIM = int(T_END / TS)
N_MPC = 15
DRAG  = 0.15


class NonlinearDoubleIntegrator(DiscreteLinearSDE):
    """Nonlinear plant with linear matrices for the KF."""

    _Ad = np.array([[1.0, TS], [0.0, 1.0]])
    _Bd = np.array([[TS ** 2 / 2], [TS]])
    _Ed = np.zeros((2, 1))
    _Cm = np.array([[1.0, 0.0]])
    _Qd = np.diag([5e-3, 2e-2])
    _Rm = np.array([[4e-2]])
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
    def Ts(self) -> float: return TS
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
    def x_ref(self) -> np.ndarray: return self._x_ref.copy()
    @x_ref.setter
    def x_ref(self, val) -> None:
        self._x_ref = np.asarray(val, dtype=float).reshape(2)
    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._U_MIN.copy(), self._U_MAX.copy()

    def f(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        p, v = x
        drag = DRAG * v * abs(v)
        p_next = p + TS * v
        v_next = v + TS * (u[0] + d[0] - drag)
        return np.array([p_next, v_next])

    def hm(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        return self._Cm @ x

    def gm(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        return self._Cm @ x


def _pref(k: int) -> float:
    t = k * TS
    if t < 8.0: return 0.0
    if t < 24.0: return 3.0
    return 1.0


def run() -> None:
    rng   = np.random.default_rng(23)
    model = NonlinearDoubleIntegrator()
    L_Q   = np.linalg.cholesky(model.Qd)
    R_std = float(np.sqrt(model.Rm[0, 0]))

    kf = DiscreteLinearKF(model, x0=np.array([0.1, 0.0]), P0=np.diag([0.4, 0.1]))
    mpc = StandardLinearisedDiscreteMPC(
        model=model, estimator=kf, N=N_MPC,
        Q=10.0, R=0.4, P=40.0,
        u_min=model._U_MIN, u_max=model._U_MAX,
        x_ref=np.array([0.0, 0.0]), z_offset=0.5, solver="highs",
    )

    x_true = np.array([0.0, 0.0])
    X_true = np.zeros((N_SIM + 1, 2))
    X_true[0] = x_true
    x_hist = np.zeros((N_SIM + 1, 2))
    x_hist[0] = kf.x_hat
    Y_meas = np.zeros(N_SIM)
    U_arr  = np.zeros(N_SIM)
    p_ref  = np.array([_pref(k) for k in range(N_SIM)])

    u_k = np.zeros(1)
    d_k = np.zeros(1)
    for k in range(N_SIM):
        ym = model.hm(x_true, u_k, d_k) + R_std * rng.standard_normal(1)
        Y_meas[k] = ym[0]
        mpc.set_output_reference_profile(np.full((N_MPC, 1), p_ref[k]))
        u_k, _, _ = mpc.compute(ym, d_k)
        U_arr[k] = u_k[0]
        noise = L_Q @ rng.standard_normal(2)
        x_true = model.f(x_true, u_k, d_k) + noise
        X_true[k + 1] = x_true
        x_hist[k + 1] = kf.x_hat

    _plot(X_true, Y_meas, x_hist, U_arr, p_ref)


def _plot(X_true, Y_meas, x_hist, U_arr, p_ref) -> None:
    t = np.arange(N_SIM + 1) * TS
    t_meas = t[1:]

    fig = plt.figure(figsize=(11, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Nonlinear Double Integrator  —  Linear KF  +  Linearised Discrete MPC",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.40, top=0.91, bottom=0.08,
                           left=0.10, right=0.93)
    ax_p = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_p)
    ax_u = fig.add_subplot(gs[2], sharex=ax_p)

    ax_p.step(t_meas, p_ref, where="post", color="#555", ls="--", lw=1.5, label="p_ref")
    ax_p.plot(t, X_true[:, 0], color="#e07b39", lw=1.8, label="True position")
    ax_p.plot(t, x_hist[:, 0], color="#2166ac", lw=1.8, label="KF mean")
    ax_p.scatter(t_meas, Y_meas, s=6, color="#c0392b", alpha=0.45, label="Measurements")
    ax_p.set_ylabel("Position  p  (m)")
    ax_p.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_p.grid(True, alpha=0.25)

    ax_v.plot(t, X_true[:, 1], color="#e07b39", lw=1.8, label="True velocity")
    ax_v.plot(t, x_hist[:, 1], color="#2166ac", lw=1.8, label="KF mean")
    ax_v.set_ylabel("Velocity  v  (m/s)")
    ax_v.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_v.grid(True, alpha=0.25)

    ax_u.step(t_meas, U_arr, where="post", color="#27ae60", lw=1.8, label="Acceleration")
    ax_u.axhline(-2.0, color="crimson", ls="--", lw=1.0, alpha=0.7)
    ax_u.axhline(+2.0, color="crimson", ls="--", lw=1.0, alpha=0.7, label="Bounds ±2 m/s²")
    ax_u.set_ylabel("u  (m/s²)")
    ax_u.set_xlabel("Time  (s)")
    ax_u.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_u.grid(True, alpha=0.25)

    for ax in (ax_p, ax_v, ax_u):
        for t_ev in (8.0, 24.0):
            ax.axvline(t_ev, color="dimgray", lw=0.7, ls=":", alpha=0.5)
    plt.setp(ax_p.get_xticklabels(), visible=False)
    plt.setp(ax_v.get_xticklabels(), visible=False)
    plt.show()


if __name__ == "__main__":
    run()
