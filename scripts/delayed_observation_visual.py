"""
Visual example: DelayedObservationFilter vs ignoring delayed measurements.

System
------
Two-state linear discrete plant with two outputs (both measured):

    x₁[k+1] = 0.90 x₁ + 0.10 u
    x₂[k+1] = 0.85 x₂ + 0.10 u
    ym      = [x₁, x₂]

Channel x₂ arrives with a fixed lag of τ = 4 samples.  Three estimators
are compared:

    1. Immediate KF using only x₁ (channel 2 masked out)
    2. Naive KF using both channels as if they were current
    3. DelayedObservationFilter with delay = [0, τ]

Usage
-----
    python scripts/delayed_observation_visual.py
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
from mbc.estimation.delayed_observation_filter import DelayedObservationFilter


N_SIM = 50
TAU   = 4
TS    = 1.0


class TwoStatePlant(DiscreteLinearSDE):
    def __init__(self) -> None:
        self._x = [0.0, 0.0]

    @property
    def nx(self) -> int: return 2
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 1
    @property
    def Ad(self) -> np.ndarray: return np.array([[0.9, 0.0], [0.0, 0.85]])
    @property
    def Bd(self) -> np.ndarray: return np.array([[0.1], [0.1]])
    @property
    def Ed(self) -> np.ndarray: return np.zeros((2, 1))
    @property
    def Cm(self) -> np.ndarray: return np.eye(2)
    @property
    def Qd(self) -> np.ndarray: return 0.01 * np.eye(2)
    @property
    def Rm(self) -> np.ndarray: return 0.08 * np.eye(2)
    @property
    def Ts(self) -> float: return TS
    @property
    def x(self) -> list[float]: return self._x
    @x.setter
    def x(self, val) -> None: self._x = list(val)
    @property
    def x_ref(self) -> np.ndarray: return np.zeros(2)
    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0]), np.array([1.0])


def _input_profile(k: int) -> float:
    if k < 15: return 0.0
    if k < 35: return 0.8
    return -0.3


def _simulate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = TwoStatePlant()
    L_Q = np.linalg.cholesky(model.Qd)
    L_R = np.linalg.cholesky(model.Rm)
    x = np.zeros(2)
    X = [x.copy()]
    Y_immediate = []
    U = []
    for k in range(N_SIM):
        u = np.array([_input_profile(k)])
        U.append(u[0])
        x = model.Ad @ x + model.Bd @ u + L_Q @ rng.standard_normal(2)
        ym = model.Cm @ x + L_R @ rng.standard_normal(2)
        Y_immediate.append(ym.copy())
        X.append(x.copy())
    return np.array(X), np.array(Y_immediate), np.array(U)


def run() -> None:
    rng = np.random.default_rng(42)
    X_true, Y_imm, U = _simulate(rng)
    X_true = X_true[1:]   # align with measurements

    model = TwoStatePlant()
    kf_ch1 = DiscreteLinearKF(model, x0=np.zeros(2), P0=np.eye(2) * 0.5)
    kf_naive = DiscreteLinearKF(model, x0=np.zeros(2), P0=np.eye(2) * 0.5)
    kf_delay = DelayedObservationFilter(
        DiscreteLinearKF(model, x0=np.zeros(2), P0=np.eye(2) * 0.5),
        lag_max=2 * TAU,
    )
    d = np.array([0.0])
    delay = np.array([0, TAU])

    est_ch1 = np.zeros((N_SIM, 2))
    est_naive = np.zeros((N_SIM, 2))
    est_delay = np.zeros((N_SIM, 2))

    for k in range(N_SIM):
        u = np.array([U[k]])
        ym_now = Y_imm[k].copy()
        if k >= TAU:
            ym_now[1] = Y_imm[k - TAU, 1]   # x₂ arrives τ steps late

        kf_ch1.step(ym_now, u, d, mask=[True, False])
        est_ch1[k] = kf_ch1.x_hat

        kf_naive.step(Y_imm[k], u, d)
        est_naive[k] = kf_naive.x_hat

        kf_delay.step(ym_now, u, d, delay=delay)
        est_delay[k] = kf_delay.x_hat

    rmse_ch1 = np.sqrt(np.mean((est_ch1 - X_true) ** 2, axis=0))
    rmse_naive = np.sqrt(np.mean((est_naive - X_true) ** 2, axis=0))
    rmse_delay = np.sqrt(np.mean((est_delay - X_true) ** 2, axis=0))
    print(f"RMSE x1 only:     {rmse_ch1}")
    print(f"RMSE naive:       {rmse_naive}")
    print(f"RMSE delayed KF:  {rmse_delay}")

    _plot(X_true, Y_imm, est_ch1, est_naive, est_delay, U)


def _plot(X_true, Y_imm, est_ch1, est_naive, est_delay, U) -> None:
    t = np.arange(N_SIM) * TS

    fig = plt.figure(figsize=(11, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Delayed Observations  —  τ = {TAU} samples on channel x₂",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.42, top=0.91, bottom=0.08,
                           left=0.10, right=0.93)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax_u = fig.add_subplot(gs[2], sharex=ax1)

    ax1.plot(t, X_true[:, 0], color="#e07b39", lw=1.8, label="True x₁")
    ax1.plot(t, est_ch1[:, 0], color="#888", ls="--", lw=1.2, label="KF ch1 only")
    ax1.plot(t, est_naive[:, 0], color="#c0392b", ls=":", lw=1.2, label="Naive (no delay)")
    ax1.plot(t, est_delay[:, 0], color="#2166ac", lw=1.8, label="Delayed KF")
    ax1.scatter(t, Y_imm[:, 0], s=6, color="#e07b39", alpha=0.35)
    ax1.set_ylabel("x₁")
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.85, ncol=2)
    ax1.grid(True, alpha=0.25)

    ax2.plot(t, X_true[:, 1], color="#e07b39", lw=1.8, label="True x₂")
    ax2.plot(t, est_ch1[:, 1], color="#888", ls="--", lw=1.2, label="KF ch1 only")
    ax2.plot(t, est_naive[:, 1], color="#c0392b", ls=":", lw=1.2, label="Naive")
    ax2.plot(t, est_delay[:, 1], color="#2166ac", lw=1.8, label="Delayed KF")
    ax2.scatter(t[TAU:], Y_imm[TAU:, 1], s=6, color="#8e44ad", alpha=0.35,
                label=f"x₂ arrival (τ={TAU} late)")
    ax2.set_ylabel("x₂")
    ax2.legend(fontsize=7, loc="upper left", framealpha=0.85, ncol=2)
    ax2.grid(True, alpha=0.25)

    ax_u.step(t, U, where="post", color="#27ae60", lw=1.8)
    ax_u.set_ylabel("Input  u")
    ax_u.set_xlabel("Time  (s)")
    ax_u.grid(True, alpha=0.25)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.show()


if __name__ == "__main__":
    run()
