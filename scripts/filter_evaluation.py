"""
Systematic evaluation and tuning of CD-EKF and CD-UKF on the
product-inhibition bioreactor.

Two independent sweeps:

  EKF sweep   — Q_c scale factor ∈ {0.5, 1, 2, 4, 8}  (alpha fixed)
  UKF sweep   — alpha             ∈ {1e-3, 0.1, 0.5, 1.0}  (Q_c scale = 1)

For each configuration the filter is run on a single realisation (fixed
seed) and we report:

  • Per-state RMSE trajectory  e(t) = |x̂_i(t) − x_true_i(t)|
  • Normalised Innovation Squared (NIS)  ε_k = νᵀ Sₖ⁻¹ ν
    Under a consistent Gaussian filter ε_k ~ χ²(ny), so mean(ε_k) ≈ ny=1.

Layout (2 columns × 4 rows):
  Left  — EKF sweep   (3 state rows + 1 NIS row)
  Right — UKF sweep   (3 state rows + 1 NIS row)

Usage::

    python scripts/filter_evaluation.py

"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from mbc.estimation.ekf import ContinuousDiscreteEKF
from mbc.estimation.ukf import ContinuousDiscreteUKF
from mbc.models import ContinuousDiscreteModel

# ── Scenario (identical to bioreactor_comparative.py) ─────────────────────────

RNG_SEED  = 0
T_END     = 60.0
DT        = 0.5
N_SUB     = 20

P_TRUE = np.array([0.40, 0.10, 8.00, 0.40, 0.20])

Q_C_TRUE = np.diag([0.05, 0.02, 0.10])
R_TRUE   = np.array([[0.09]])

X0_TRUE = np.array([0.50, 0.10, 0.00])
X0_EST  = np.array([0.50, 0.50, 1.50])
P0_EST  = np.diag([2.00, 1.00, 8.00])

D_CONST = 0.06

def _sin_schedule(t: float) -> float:
    if t < 20.0:  return 8.0
    if t < 45.0:  return 4.0
    return 6.0


# ── Model ─────────────────────────────────────────────────────────────────────

class ProductInhibitionBioreactor(ContinuousDiscreteModel):
    """3-state product-inhibition CSTR (identical to comparative script)."""

    def __init__(self, Q_c: np.ndarray, R: np.ndarray) -> None:
        self._Q_c = Q_c
        self._R   = R

    @property
    def nx(self) -> int: return 3

    @property
    def nu(self) -> int: return 1

    @property
    def nd(self) -> int: return 1

    @property
    def ny(self) -> int: return 1

    @property
    def nw(self) -> int: return 3

    @property
    def Q_c(self) -> np.ndarray: return self._Q_c.copy()

    @property
    def R(self) -> np.ndarray: return self._R.copy()

    @staticmethod
    def _mu(S, P, mu_max, K_s, K_I):
        return mu_max * S / (K_s + S) / (1.0 + P / K_I)

    def f(self, x, u, d, p, t):
        S, X, P = float(x[0]), float(x[1]), float(x[2])
        D = float(u[0]); S_in = float(d[0])
        mu_max, K_s, K_I, Y_XS, alpha = p
        S = max(S, 0.0); X = max(X, 0.0); P = max(P, 0.0)
        mu = self._mu(S, P, mu_max, K_s, K_I)
        return np.array([
            (S_in - S) * D - mu * X / Y_XS,
            (mu - D) * X,
            alpha * mu * X - D * P,
        ])

    def g(self, x, u, d, p, t):
        return np.eye(3)   # identity — Q_c is the direct noise covariance

    def h(self, x, u, d, p):
        return np.array([x[1]])

    def dfdx(self, x, u, d, p, t):
        S, X, P = float(x[0]), float(x[1]), float(x[2])
        D = float(u[0])
        mu_max, K_s, K_I, Y_XS, alpha = p
        S = max(S, 1e-12); X = max(X, 0.0); P = max(P, 0.0)
        inh = 1.0 / (1.0 + P / K_I)
        mu   = mu_max * S / (K_s + S) * inh
        mu_S = mu_max * K_s / (K_s + S) ** 2 * inh
        mu_P = -mu * inh / K_I
        return np.array([
            [-D - mu_S * X / Y_XS,  -mu / Y_XS,       -mu_P * X / Y_XS],
            [mu_S * X,               mu - D,            mu_P * X        ],
            [alpha * mu_S * X,       alpha * mu,        alpha * mu_P * X - D],
        ])

    def dhdx(self, x, u, d, p):
        return np.array([[0.0, 1.0, 0.0]])

    def dhdu(self, x, u, d, p):
        return np.zeros((1, 1))

    def dhdd(self, x, u, d, p):
        return np.zeros((1, 1))


# ── Simulation ────────────────────────────────────────────────────────────────

def _simulate(rng: np.random.Generator):
    """Return (X_true, Y_meas, U_arr, D_arr, t_meas) for the fixed scenario."""
    t_meas = np.arange(0.0, T_END, DT)
    N = len(t_meas)
    h = DT / N_SUB; sh = np.sqrt(h)
    G_std = np.sqrt(np.diag(Q_C_TRUE))
    R_std = np.sqrt(float(R_TRUE[0, 0]))

    X_true = np.empty((N + 1, 3))
    Y_meas = np.empty((N, 1))
    U_arr  = np.empty((N, 1))
    D_arr  = np.empty((N, 1))

    x = X0_TRUE.copy()
    X_true[0] = x
    model_sim = ProductInhibitionBioreactor(Q_C_TRUE, R_TRUE)

    for k, t_k in enumerate(t_meas):
        u_k = np.array([D_CONST]); d_k = np.array([_sin_schedule(t_k)])
        U_arr[k] = u_k; D_arr[k] = d_k
        for _ in range(N_SUB):
            x = np.maximum(x + h * model_sim.f(x, u_k, d_k, P_TRUE, t_k)
                           + G_std * rng.standard_normal(3) * sh, 0.0)
        Y_meas[k] = model_sim.h(x, u_k, d_k, P_TRUE) + R_std * rng.standard_normal(1)
        X_true[k + 1] = x

    return X_true, Y_meas, U_arr, D_arr, t_meas


# ── Filter runner ─────────────────────────────────────────────────────────────

def _run_ekf(q_scale: float, X_true, Y_meas, U_arr, D_arr, t_meas):
    """Run CD-EKF with the given Q_c scale.  Return (errors (N+1,3), nis (N,))."""
    model = ProductInhibitionBioreactor(q_scale * Q_C_TRUE, R_TRUE)
    filt  = ContinuousDiscreteEKF(model, X0_EST.copy(), P0_EST.copy(), DT, n_steps=N_SUB)

    N   = len(t_meas)
    err = np.empty((N + 1, 3)); err[0] = filt.x_hat - X_true[0]
    nis = np.empty(N)

    for k, t_k in enumerate(t_meas):
        y_k = Y_meas[k]; u_k = U_arr[k]; d_k = D_arr[k]
        x_pred, P_pred = filt.predict(u_k, d_k, P_TRUE, t_k)

        # NIS before update
        H    = model.dhdx(x_pred, u_k, d_k, P_TRUE)
        y_hat = model.h(x_pred, u_k, d_k, P_TRUE)
        S    = H @ P_pred @ H.T + model.R
        innov = y_k - y_hat
        nis[k] = float(innov @ np.linalg.solve(S, innov))

        x_hat, _ = filt.update(y_k, u_k, d_k, P_TRUE)
        err[k + 1] = x_hat - X_true[k + 1]

    return err, nis


def _run_ukf(alpha: float, q_scale: float, X_true, Y_meas, U_arr, D_arr, t_meas):
    """Run CD-UKF with the given alpha and Q_c scale.  Return (errors, nis)."""
    model = ProductInhibitionBioreactor(q_scale * Q_C_TRUE, R_TRUE)
    filt  = ContinuousDiscreteUKF(model, X0_EST.copy(), P0_EST.copy(), DT,
                                   n_steps=N_SUB, alpha=alpha)

    N   = len(t_meas)
    err = np.empty((N + 1, 3)); err[0] = filt.x_hat - X_true[0]
    nis = np.empty(N)

    for k, t_k in enumerate(t_meas):
        y_k = Y_meas[k]; u_k = U_arr[k]; d_k = D_arr[k]
        x_pred, P_pred = filt.predict(u_k, d_k, P_TRUE, t_k)

        # NIS before update (using EKF linearisation as approximation)
        H     = model.dhdx(x_pred, u_k, d_k, P_TRUE)
        y_hat = model.h(x_pred, u_k, d_k, P_TRUE)
        S     = H @ P_pred @ H.T + model.R
        innov = y_k - y_hat
        nis[k] = float(innov @ np.linalg.solve(S, innov))

        x_hat, _ = filt.update(y_k, u_k, d_k, P_TRUE)
        err[k + 1] = x_hat - X_true[k + 1]

    return err, nis


# ── Plotting ──────────────────────────────────────────────────────────────────

STATE_LABELS = ["$S$ error (g/L)", "$X$ error (g/L)", "$P$ error (g/L)"]
STATE_NAMES  = ["S (substrate)", "X (biomass, observed)", "P (product)"]


def _palette(n: int) -> list[str]:
    """Return n distinguishable colours."""
    cm = plt.get_cmap("plasma")
    return [cm(v) for v in np.linspace(0.1, 0.85, n)]


def _plot_sweep(axes_err, ax_nis, t_plot, t_meas, sweep_results, labels, title):
    """
    Fill axes_err (list of 3 Axes, one per state) and ax_nis with the sweep.

    sweep_results : list of (err (N+1,3), nis (N,)) tuples
    labels        : list of str, one per configuration
    """
    colors = _palette(len(sweep_results))

    for (err, nis), lab, c in zip(sweep_results, labels, colors):
        for j, ax in enumerate(axes_err):
            ax.plot(t_plot, err[:, j], color=c, lw=1.4, label=lab)
            ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)

        # Smoothed NIS (rolling mean over 5 steps)
        nis_smooth = np.convolve(nis, np.ones(5) / 5, mode="same")
        ax_nis.plot(t_meas, nis_smooth, color=c, lw=1.4, label=lab)

    # Reference: chi²(1) mean = 1 (consistent filter)
    ax_nis.axhline(1.0, color="k", lw=1.2, ls="--", label="NIS = 1 (consistent)")
    ax_nis.set_ylim(0, None)

    # Labels
    for j, ax in enumerate(axes_err):
        ax.set_ylabel(STATE_LABELS[j], fontsize=8)
        ax.set_title(STATE_NAMES[j], fontsize=8)
        ax.set_xlim(t_plot[0], t_plot[-1])
        ax.tick_params(labelsize=7)
        for t_ev in (20.0, 45.0):
            ax.axvline(t_ev, color="gray", lw=0.6, ls="--", alpha=0.5)

    ax_nis.set_ylabel("NIS (smoothed)", fontsize=8)
    ax_nis.set_title("Normalised Innovation Squared", fontsize=8)
    ax_nis.set_xlim(t_plot[0], t_plot[-1])
    ax_nis.tick_params(labelsize=7)
    for t_ev in (20.0, 45.0):
        ax_nis.axvline(t_ev, color="gray", lw=0.6, ls="--", alpha=0.5)

    axes_err[0].set_title(f"{title}\n{STATE_NAMES[0]}", fontsize=8)
    axes_err[-1].set_xlabel("Time (h)", fontsize=8)
    ax_nis.set_xlabel("Time (h)", fontsize=8)
    ax_nis.legend(fontsize=7, loc="upper right", framealpha=0.85)


def plot(t_plot, t_meas,
         ekf_sweep, ekf_labels,
         ukf_sweep, ukf_labels) -> plt.Figure:
    """
    4 rows × 2 cols:
      rows 0-2 : error per state
      row  3   : NIS
      col  0   : EKF sweep
      col  1   : UKF sweep
    """
    fig = plt.figure(figsize=(13, 11))
    fig.suptitle(
        "CD filter evaluation — product-inhibition bioreactor\n"
        "Left: EKF Q_c scale sweep  |  Right: UKF α sweep  (Q_c scale = 1)",
        fontsize=10, fontweight="bold",
    )

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        hspace=0.55, wspace=0.38,
        top=0.91, bottom=0.06, left=0.08, right=0.97,
        height_ratios=[1.6, 1.6, 1.6, 1.8],
    )

    ekf_err_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ukf_err_axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    ax_nis_ekf   = fig.add_subplot(gs[3, 0])
    ax_nis_ukf   = fig.add_subplot(gs[3, 1])

    _plot_sweep(ekf_err_axes, ax_nis_ekf, t_plot, t_meas,
                ekf_sweep, ekf_labels, "EKF")
    _plot_sweep(ukf_err_axes, ax_nis_ukf, t_plot, t_meas,
                ukf_sweep, ukf_labels, "UKF")

    # Shared y-limits for comparability across both columns
    for j in range(3):
        ys = [ax.get_ylim() for ax in (ekf_err_axes[j], ukf_err_axes[j])]
        lo = min(y[0] for y in ys); hi = max(y[1] for y in ys)
        ekf_err_axes[j].set_ylim(lo, hi)
        ukf_err_axes[j].set_ylim(lo, hi)
    nis_hi = max(ax_nis_ekf.get_ylim()[1], ax_nis_ukf.get_ylim()[1])
    ax_nis_ekf.set_ylim(0, min(nis_hi, 20.0))
    ax_nis_ukf.set_ylim(0, min(nis_hi, 20.0))

    # Row y-labels on right column
    for j, ax in enumerate(ukf_err_axes):
        ax.set_ylabel("")
    ax_nis_ukf.set_ylabel("")

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

EKF_Q_SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]
UKF_ALPHAS   = [1e-3, 0.1, 0.5, 1.0]


def main():
    rng = np.random.default_rng(RNG_SEED)
    X_true, Y_meas, U_arr, D_arr, t_meas = _simulate(rng)
    t_plot = np.concatenate([t_meas, [t_meas[-1] + DT]])

    print("Running EKF Q_c scale sweep…")
    ekf_sweep  = []
    ekf_labels = []
    for qs in EKF_Q_SCALES:
        print(f"  Q_c_scale = {qs}")
        err, nis = _run_ekf(qs, X_true, Y_meas, U_arr, D_arr, t_meas)
        ekf_sweep.append((err, nis))
        ekf_labels.append(f"scale = {qs}")

    print("Running UKF α sweep…")
    ukf_sweep  = []
    ukf_labels = []
    for alpha in UKF_ALPHAS:
        print(f"  alpha = {alpha}")
        err, nis = _run_ukf(alpha, 1.0, X_true, Y_meas, U_arr, D_arr, t_meas)
        ukf_sweep.append((err, nis))
        ukf_labels.append(f"α = {alpha}")

    # Print summary RMSE (time-averaged over last 40 h)
    tail = int(40 / DT)
    print("\n── EKF summary RMSE (last 40 h) ────────────────────")
    print(f"{'Scale':>8}  {'S':>8}  {'X':>8}  {'P':>8}  {'NIS mean':>9}")
    for (err, nis), qs in zip(ekf_sweep, EKF_Q_SCALES):
        rmse = np.sqrt(np.mean(err[-tail:] ** 2, axis=0))
        print(f"{qs:>8.1f}  {rmse[0]:>8.4f}  {rmse[1]:>8.4f}  {rmse[2]:>8.4f}  {np.mean(nis[-tail:]):>9.3f}")

    print("\n── UKF summary RMSE (last 40 h) ────────────────────")
    print(f"{'alpha':>8}  {'S':>8}  {'X':>8}  {'P':>8}  {'NIS mean':>9}")
    for (err, nis), alpha in zip(ukf_sweep, UKF_ALPHAS):
        rmse = np.sqrt(np.mean(err[-tail:] ** 2, axis=0))
        print(f"{alpha:>8.4f}  {rmse[0]:>8.4f}  {rmse[1]:>8.4f}  {rmse[2]:>8.4f}  {np.mean(nis[-tail:]):>9.3f}")

    fig = plot(t_plot, t_meas, ekf_sweep, ekf_labels, ukf_sweep, ukf_labels)
    plt.show()


if __name__ == "__main__":
    main()
