"""
Visual example: CD-EKF + Linearised Continuous MPC on a bounded nonlinear plant.

System
------
Scalar continuous-discrete plant with quadratic nonlinearity:

    dx/dt = −x + 0.2·x² + u + 0.5·d,    y = x

    |u| ≤ 1,  Ts = 1 s

Estimation
----------
ContinuousDiscreteEKF tracks the state from noisy measurements.

Control
-------
StandardLinearisedContinuousMPC re-linearises the plant at each sample and
solves a ZOH-discretised QP in deviation coordinates.  Compare the linearised
MPC trajectory against a naive linear MPC that uses a fixed linearisation.

Scenario (25 steps)
-------------------
    y_ref = 2.0 from t = 5 onward (step from 0)

Usage
-----
    python scripts/scalar_bounded_lmmpc_visual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mbc.models import ContinuousDiscreteSDE
from mbc.estimation import ContinuousDiscreteEKF, ContinuousDiscreteEKFParams
from mbc.control import StandardLinearisedContinuousMPC


DT    = 1.0
N_SIM = 25
N_MPC = 10
X_REF = np.array([2.0])


class BoundedNonlinearScalar(ContinuousDiscreteSDE):
    @property
    def nx(self) -> int: return 1
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 1
    @property
    def nw(self) -> int: return 1
    @property
    def nym(self) -> int: return 1
    @property
    def nz(self) -> int: return 1
    @property
    def Ts(self) -> float: return DT
    @property
    def Rm(self) -> np.ndarray: return np.array([[0.04]])

    def f(self, x, u, d, p, t):
        return np.array([-x[0] + 0.2 * x[0] ** 2 + u[0] + 0.5 * d[0]])

    def sigma(self, x, u, d, p, t):
        return np.array([[0.08]])

    def hm(self, x, u, d, p, t=0.0):
        return np.array([x[0]])

    def gm(self, x, u, d, p, t):
        return np.array([x[0]])


def _euler_maruyama_step(x, u, d, model, rng, dt=DT) -> np.ndarray:
    dw = rng.standard_normal(model.nw) * np.sqrt(dt)
    sig = model.sigma(x, u, d, np.zeros(0), 0.0)
    return x + dt * model.f(x, u, d, np.zeros(0), 0.0) + sig @ dw


def run() -> None:
    rng   = np.random.default_rng(17)
    model = BoundedNonlinearScalar()
    x0    = np.array([0.0])
    P0    = np.array([[0.5]])
    R_std = float(np.sqrt(model.Rm[0, 0]))

    ekf = ContinuousDiscreteEKF(
        model, x0=x0.copy(), P0=P0,
        params=ContinuousDiscreteEKFParams(n_steps=8),
    )
    mpc = StandardLinearisedContinuousMPC(
        model=model, estimator=ekf, N=N_MPC,
        Q=np.eye(1) * 8.0, R=np.eye(1) * 0.05, dt=DT,
        u_min=np.array([-1.0]), u_max=np.array([1.0]),
        x_ref=X_REF, z_offset=10.0,
    )

    x_true = x0.copy()
    X_true = np.zeros((N_SIM + 1, 1))
    X_true[0] = x_true
    x_hist = np.zeros((N_SIM + 1, 1))
    x_hist[0] = ekf.x_hat
    Y_meas = np.zeros(N_SIM)
    U_arr  = np.zeros(N_SIM)
    z_ref  = np.where(np.arange(N_SIM) < 5, 0.0, X_REF[0])

    u_k = np.zeros(1)
    d_k = np.zeros(1)
    for k in range(N_SIM):
        ym = model.hm(x_true, u_k, d_k, np.zeros(0), float(k)) + R_std * rng.standard_normal(1)
        Y_meas[k] = ym[0]
        mpc.set_output_reference_profile(np.full((N_MPC, 1), z_ref[k]))
        u_k, _, _ = mpc.compute(y=ym, d=d_k, p=np.array([]), t=float(k))
        U_arr[k] = u_k[0]
        x_true = _euler_maruyama_step(x_true, u_k, d_k, model, rng)
        X_true[k + 1] = x_true
        x_hist[k + 1] = ekf.x_hat

    _plot(X_true, Y_meas, x_hist, U_arr, z_ref)


def _plot(X_true, Y_meas, x_hist, U_arr, z_ref) -> None:
    t = np.arange(N_SIM + 1) * DT
    t_meas = t[1:]

    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Bounded Nonlinear Scalar  —  CD-EKF  +  Linearised CD MPC",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.38, top=0.90, bottom=0.10,
                           left=0.10, right=0.93)
    ax_y = fig.add_subplot(gs[0])
    ax_u = fig.add_subplot(gs[1], sharex=ax_y)

    ax_y.step(t_meas, z_ref, where="post", color="#555", ls="--", lw=1.5, label="Reference")
    ax_y.plot(t, X_true[:, 0], color="#e07b39", lw=1.8, label="True state")
    ax_y.plot(t, x_hist[:, 0], color="#2166ac", lw=1.8, label="EKF mean")
    ax_y.scatter(t_meas, Y_meas, s=10, color="#c0392b", alpha=0.5, label="Measurements")
    ax_y.set_ylabel("State  x")
    ax_y.set_title("Output tracking with successive linearisation")
    ax_y.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_y.grid(True, alpha=0.25)

    ax_u.step(t_meas, U_arr, where="post", color="#27ae60", lw=1.8, label="Input u")
    ax_u.axhline(-1.0, color="crimson", ls="--", lw=1.0, alpha=0.7)
    ax_u.axhline(+1.0, color="crimson", ls="--", lw=1.0, alpha=0.7, label="Bounds ±1")
    ax_u.set_ylabel("Input  u")
    ax_u.set_xlabel("Time  (s)")
    ax_u.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_u.grid(True, alpha=0.25)

    ax_y.axvline(5.0, color="dimgray", lw=0.7, ls=":", alpha=0.5)
    ax_u.axvline(5.0, color="dimgray", lw=0.7, ls=":", alpha=0.5)
    plt.setp(ax_y.get_xticklabels(), visible=False)
    plt.show()


if __name__ == "__main__":
    run()
