"""
Comparative benchmark: CD nonlinear state estimators on a product-inhibition
bioreactor (3 states, 1 measurement, 2 step-change disturbances).

Model
-----
Three-state CSTR with Monod kinetics and product inhibition:

    dS/dt = (S_in - S)*D  -  mu(S,P)*X / Y_XS
    dX/dt = (mu(S,P) - D)*X
    dP/dt =  alpha*mu(S,P)*X  -  D*P

    mu(S,P) = mu_max * S/(K_s + S) * 1/(1 + P/K_I)

States  x = [S, X, P]  (g/L)   — only X is measured
Input   u = [D]         (h⁻¹)   — dilution rate (constant)
Dist.   d = [S_in]      (g/L)   — feed substrate (step changes)
Params  p = [mu_max, K_s, K_I, Y_XS, alpha]

Design goals
------------
* Large process and measurement noise → covariance stays visible throughout
* Two step changes in S_in create visible transients
* S and P are never directly measured → EKF variances remain non-trivial
* General filter registry: EKF runs; UKF/EnKF/PF slots show "not yet
  implemented" but their API is already called — they will run automatically
  once those classes are implemented.

Usage::

    python scripts/bioreactor_comparative.py

"""

from __future__ import annotations

import inspect
import os
import sys

# Make the package importable when run from the project root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from mbc.estimation import (
    ContinuousDiscreteEKF,
    ContinuousDiscreteEKFParams,
    ContinuousDiscreteEnKF,
    ContinuousDiscreteEnKFParams,
    ContinuousDiscretePF,
    ContinuousDiscretePFParams,
    ContinuousDiscreteUKF,
    ContinuousDiscreteUKFParams,
)
from mbc.models import ContinuousDiscreteSDE

# ── Reproducibility ───────────────────────────────────────────────────────────

RNG_SEED = 0

# ── Simulation horizon and sampling ──────────────────────────────────────────

T_END = 60.0   # h
DT = 0.5       # h  — measurement sampling period
N_SUB = 20     # Euler-Maruyama substeps per DT
N_STEPS = len(np.arange(0.0, T_END, DT))   # number of discrete-time steps

# ── True parameter vector ─────────────────────────────────────────────────────

P_TRUE = np.array([
    0.40,   # mu_max  (h⁻¹)
    0.10,   # K_s     (g/L)
    8.00,   # K_I     (g/L)
    0.40,   # Y_XS    (g-X / g-S)
    0.20,   # alpha   (g-P / g-X)
])

# ── Noise covariances (simulation truth) ────────────────────────────────────
# These are the TRUE process and measurement noise levels used to drive the
# stochastic simulation.  Filters are given separate tuned versions below.
# Diffusion standard deviations: σ_S = 0.22, σ_X = 0.14, σ_P = 0.32  g/L/√h
# Measurement standard deviation: σ_y = 0.30 g/L

Q_C_TRUE = np.diag([0.05, 0.02, 0.10])   # true continuous-time process noise
R_TRUE   = np.array([[0.09]])             # true discrete-time measurement noise

# ── Filter noise tuning ───────────────────────────────────────────────────────
# Slight Q_c inflation keeps P from collapsing and preserves responsiveness to
# unmodelled disturbances.  R is set to the true value (known sensor spec).
#
# Rule of thumb: inflate each diagonal of Q_c by ~3–5× to account for model
# mismatch and keep the filter in a "cautious" regime.

Q_C_FILTER = np.diag([0.20, 0.08, 0.40])  # inflated for EKF / UKF
R          = R_TRUE.copy()                 # measurement noise (true value)

# Convenience alias used by the simulation helper
Q_C = Q_C_TRUE

# ── Initial conditions ────────────────────────────────────────────────────────

X0_TRUE = np.array([0.50, 0.10, 0.00])   # true startup (S, X, P)
X0_EST = np.array([0.50, 0.50, 1.50])    # biased initial estimate
P0_EST = np.diag([2.00, 1.00, 8.00])     # large initial covariance

# ── Input schedule (constant D, step changes in S_in) ────────────────────────

D_CONST = 0.06   # h⁻¹  dilution rate (held constant)

def _sin_schedule(t: float) -> float:
    """Piecewise S_in: 8 g/L → 4 g/L at t=20 h → 6 g/L at t=45 h."""
    if t < 20.0:
        return 8.0
    if t < 45.0:
        return 4.0
    return 6.0


# ── Filter display configuration ─────────────────────────────────────────────

FILTER_STYLES: dict[str, dict] = {
    "EKF":  {"color": "#2166ac", "lw": 1.8, "zorder": 4},
    "UKF":  {"color": "#1a9641", "lw": 1.8, "zorder": 3},
    "EnKF": {"color": "#d7191c", "lw": 1.8, "zorder": 3},
    "PF":   {"color": "#7b2d8b", "lw": 1.8, "zorder": 3},
}
BAND_ALPHA = 0.15   # opacity of ±2σ shaded bands


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class ProductInhibitionBioreactor(ContinuousDiscreteSDE):
    """
    Product-inhibition CSTR with 3 states and analytic Jacobians.

    States  x = [S, X, P]   (g/L)
    Input   u = [D]           (h⁻¹)
    Dist.   d = [S_in]        (g/L)
    Params  p = [mu_max, K_s, K_I, Y_XS, alpha]
    Output  y = [X]           (g/L)  — biomass only
    """

    def __init__(self, Q_c: np.ndarray, Rm: np.ndarray, Ts: float = DT) -> None:
        self._Q_c = Q_c
        self._Rm = Rm
        self._Ts = Ts
        self._sigma = np.diag(np.sqrt(np.diag(Q_c)))

    # ── Dimensions ──────────────────────────────────────────────────────────

    @property
    def Ts(self) -> float:
        return self._Ts

    @property
    def nx(self) -> int:
        return 3

    @property
    def nu(self) -> int:
        return 1

    @property
    def nd(self) -> int:
        return 1

    @property
    def nym(self) -> int:
        return 1

    @property
    def nz(self) -> int:
        return 1

    @property
    def nw(self) -> int:
        return 3   # one noise channel per state

    # ── Noise covariances ────────────────────────────────────────────────────

    @property
    def Rm(self) -> np.ndarray:
        return self._Rm.copy()

    # ── Kinetics helper ──────────────────────────────────────────────────────

    @staticmethod
    def _mu(S: float, P: float, mu_max: float, K_s: float, K_I: float
            ) -> float:
        """Monod rate with product inhibition."""
        return mu_max * S / (K_s + S) * 1.0 / (1.0 + P / K_I)

    # ── Model functions ──────────────────────────────────────────────────────

    def f(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        S, X, P = x
        D = float(u[0])
        S_in = float(d[0])
        mu_max, K_s, K_I, Y_XS, alpha = p

        # Clamp inputs to non-negative only when evaluating kinetics, so that
        # sigma points / finite-difference perturbations at negative coordinates
        # get smoothly extrapolated (not hard-clamped) dynamics.
        S_c = max(S, 0.0)
        X_c = max(X, 0.0)
        P_c = max(P, 0.0)

        mu = self._mu(S_c, P_c, mu_max, K_s, K_I)

        dS = (S_in - S) * D - mu * X_c / Y_XS
        dX = (mu - D) * X
        dP = alpha * mu * X_c - D * P
        return np.array([dS, dX, dP])

    def sigma(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Diagonal diffusion: dx = f dt + sigma dw, sigma sigmaᵀ = Q_c."""
        return self._sigma.copy()

    def hm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        return np.array([x[1]])   # ym = X

    def gm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        return np.array([x[1]])   # z = X

    # ── Analytic Jacobians ───────────────────────────────────────────────────

    def dfdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        S, X, P = x
        D = float(u[0])
        mu_max, K_s, K_I, Y_XS, alpha = p

        S = max(S, 1e-12)
        X = max(X, 0.0)
        P = max(P, 0.0)

        inv_inh = 1.0 / (1.0 + P / K_I)   # 1/(1 + P/K_I)
        mu = mu_max * S / (K_s + S) * inv_inh

        # Partial derivatives of mu
        mu_S = mu_max * K_s / (K_s + S) ** 2 * inv_inh
        mu_P = -mu * inv_inh / K_I

        # df_S = (S_in-S)*D - mu*X/Y_XS
        row_S = np.array([
            -D - mu_S * X / Y_XS,   # ∂f_S/∂S
            -mu / Y_XS,              # ∂f_S/∂X
            -mu_P * X / Y_XS,       # ∂f_S/∂P
        ])
        # df_X = (mu-D)*X
        row_X = np.array([
            mu_S * X,   # ∂f_X/∂S
            mu - D,     # ∂f_X/∂X
            mu_P * X,   # ∂f_X/∂P
        ])
        # df_P = alpha*mu*X - D*P
        row_P = np.array([
            alpha * mu_S * X,         # ∂f_P/∂S
            alpha * mu,               # ∂f_P/∂X
            alpha * mu_P * X - D,     # ∂f_P/∂P
        ])
        return np.vstack([row_S, row_X, row_P])

    def dfdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        S, X, P = x
        S_in = float(d[0])
        # d = [D], ∂f/∂D
        return np.array([[S_in - S], [-X], [-P]])

    def dfdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        D = float(u[0])
        # d = [S_in], ∂f/∂S_in
        return np.array([[D], [0.0], [0.0]])

    def dhmdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0]])   # ∂(X)/∂[S,X,P]

    def dhmdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        return np.zeros((1, 1))

    def dhmdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        return np.zeros((1, 1))


# ══════════════════════════════════════════════════════════════════════════════
# Simulation helper
# ══════════════════════════════════════════════════════════════════════════════

def simulate(
    model: ProductInhibitionBioreactor,
    x0: np.ndarray,
    times: np.ndarray,
    D_val: float,
    sin_fn,
    p: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler-Maruyama simulation.

    Returns
    -------
    X_true  : (N+1, nx) true state trajectory (includes x0)
    Y_meas  : (N, nym)  noisy measurements at each sample time
    U_arr   : (N, nu)   applied inputs
    D_arr   : (N, nd)   disturbances
    """
    N = len(times)
    nx = model.nx
    nym = model.nym
    dt = float(times[1] - times[0])
    h_sub = dt / N_SUB
    sqrt_h = np.sqrt(h_sub)

    X_true = np.empty((N + 1, nx))
    Y_meas = np.empty((N, nym))
    U_arr = np.empty((N, 1))
    D_arr = np.empty((N, 1))

    x = x0.copy()
    X_true[0] = x

    R_chol = np.sqrt(float(model.Rm[0, 0]))

    for k, t_k in enumerate(times):
        S_in_k = sin_fn(t_k)
        u_k = np.array([D_val])
        d_k = np.array([S_in_k])
        U_arr[k] = u_k
        D_arr[k] = d_k

        # Euler-Maruyama substeps
        for _ in range(N_SUB):
            f_val = model.f(x, u_k, d_k, p, t_k)
            sig = model.sigma(x, u_k, d_k, p, t_k)
            noise = sig @ rng.standard_normal(model.nw) * sqrt_h
            x = x + h_sub * f_val + noise
            # Clamp to non-negative
            x = np.maximum(x, 0.0)

        # Measurement with noise
        y_clean = model.hm(x, u_k, d_k, p, t_k)
        y_noisy = y_clean + R_chol * rng.standard_normal(nym)

        X_true[k + 1] = x
        Y_meas[k] = y_noisy

    return X_true, Y_meas, U_arr, D_arr


# ══════════════════════════════════════════════════════════════════════════════
# Filter adapter
# ══════════════════════════════════════════════════════════════════════════════

def _has_p_in_step(filt) -> bool:
    """True if the filter's step() signature includes a ``p`` parameter."""
    try:
        sig = inspect.signature(filt.step)
        return "p" in sig.parameters
    except (TypeError, ValueError):
        return False


def step_filter(
    filt,
    y: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    p: np.ndarray,
    t: float,
    mask=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unified step call that handles both the EKF signature (with ``p``) and
    any filter that omits ``p``.  Returns (x_hat, P).
    """
    if _has_p_in_step(filt):
        return filt.step(y, u, d, p, t, mask=mask)
    else:
        return filt.step(y, u, d, t, mask=mask)


def run_filter(
    name: str,
    filt,
    times: np.ndarray,
    Y_meas: np.ndarray,
    U_arr: np.ndarray,
    D_arr: np.ndarray,
    p: np.ndarray,
) -> dict | None:
    """
    Run *filt* over the whole trajectory. Returns a result dict with keys
    ``x_hist`` (N+1, nx) and ``std_hist`` (N+1, nx), or ``None`` if the
    filter raises ``NotImplementedError``.
    """
    N = len(times)
    nx = filt.x_hat.shape[0] if hasattr(filt, "_x_np") else filt.x_hat.shape[0]

    x_hist = np.empty((N + 1, nx))
    std_hist = np.empty((N + 1, nx))

    x_hist[0] = filt.x_hat
    std_hist[0] = np.sqrt(np.diag(filt.P))

    for k, t_k in enumerate(times):
        y_k = Y_meas[k]
        u_k = U_arr[k]
        d_k = D_arr[k]

        try:
            x_hat_k, P_k = step_filter(filt, y_k, u_k, d_k, p, t_k)
        except NotImplementedError:
            print(f"  [{name}]  NotImplementedError — skipping (not yet implemented).")
            return None

        x_hist[k + 1] = x_hat_k
        std_hist[k + 1] = np.sqrt(np.maximum(np.diag(P_k), 0.0))

    return {"x_hist": x_hist, "std_hist": std_hist}


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

STATE_LABELS = ["$S$ (g/L)", "$X$ (g/L)", "$P$ (g/L)"]
STATE_NAMES  = ["S (substrate)", "X (biomass, measured)", "P (product)"]
SIGMA_LABELS = [r"$\sigma_S$", r"$\sigma_X$", r"$\sigma_P$"]


def _plot_input_panel(ax: plt.Axes, times: np.ndarray,
                      U_arr: np.ndarray, D_arr: np.ndarray,
                      t_plot: np.ndarray) -> None:
    """Plot D and S_in step functions on a twin-axis panel."""
    # D is constant — draw as flat line
    ax.step(t_plot[:-1], U_arr[:, 0], where="post",
            color="#555555", lw=1.5, label="$D$ (h⁻¹)")
    ax.set_ylabel("$D$ (h⁻¹)", color="#555555")
    ax.tick_params(axis="y", labelcolor="#555555")
    ax.set_ylim(0.0, 0.12)

    ax2 = ax.twinx()
    ax2.step(t_plot[:-1], D_arr[:, 0], where="post",
             color="#b15928", lw=1.5, label="$S_{in}$ (g/L)", ls="--")
    ax2.set_ylabel("$S_{in}$ (g/L)", color="#b15928")
    ax2.tick_params(axis="y", labelcolor="#b15928")
    ax2.set_ylim(0.0, 12.0)

    # Combined legend
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
              fontsize=7, framealpha=0.8)
    ax.set_xlabel("Time (h)")
    ax.set_title("Inputs & disturbances", fontsize=9)


def plot_results(
    t_plot: np.ndarray,
    t_meas: np.ndarray,
    X_true: np.ndarray,
    Y_meas: np.ndarray,
    U_arr: np.ndarray,
    D_arr: np.ndarray,
    filter_results: dict[str, dict | None],
) -> plt.Figure:
    """
    Layout (3 rows × 3 cols + wide bottom row):

        [0,0] S estimate     [0,1] X estimate + measurements     [0,2] P estimate
        [1,0] σ_S            [1,1] σ_X                           [1,2] σ_P
        [2, 0:3] Inputs / disturbances (wide)
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Nonlinear CD state estimators — product-inhibition bioreactor\n"
        "(S, P unobserved; X measured with σ = 0.30 g/L)",
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[2.8, 1.8, 1.4],
        hspace=0.42, wspace=0.38,
        top=0.90, bottom=0.07, left=0.07, right=0.97,
    )

    axes_state = [fig.add_subplot(gs[0, j]) for j in range(3)]
    axes_sigma = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ax_input = fig.add_subplot(gs[2, :])

    # ── True trajectory ──────────────────────────────────────────────────

    for j, ax in enumerate(axes_state):
        ax.plot(t_plot, X_true[:, j], "k-", lw=1.2, label="Truth",
                zorder=5, alpha=0.75)

    # ── Measurements (X only, panel j=1) ────────────────────────────────

    axes_state[1].scatter(
        t_meas, Y_meas[:, 0],
        s=10, marker=".", color="gray", alpha=0.6, label="Measurement",
        zorder=6,
    )

    # ── Filter overlays ──────────────────────────────────────────────────

    legend_handles: list = []
    legend_labels:  list = []

    for name, res in filter_results.items():
        style = FILTER_STYLES[name]
        c = style["color"]
        lw = style["lw"]
        zo = style["zorder"]

        if res is None:
            # Filter not implemented — add a greyed-out legend entry
            import matplotlib.lines as mlines
            h_stub = mlines.Line2D(
                [], [], color=c, lw=lw, ls=":",
                label=f"{name} (not implemented)",
            )
            legend_handles.append(h_stub)
            legend_labels.append(f"{name} (not yet implemented)")
            continue

        x_hist = res["x_hist"]
        std_hist = res["std_hist"]

        for j, ax in enumerate(axes_state):
            line, = ax.plot(t_plot, x_hist[:, j], color=c, lw=lw,
                            zorder=zo, label=name)
            ax.fill_between(
                t_plot,
                x_hist[:, j] - 2 * std_hist[:, j],
                x_hist[:, j] + 2 * std_hist[:, j],
                color=c, alpha=BAND_ALPHA, zorder=zo - 1,
            )

        for j, ax in enumerate(axes_sigma):
            ax.plot(t_plot, std_hist[:, j], color=c, lw=lw, zorder=zo)

        import matplotlib.patches as mpatches
        h_patch = mpatches.Patch(color=c, label=name)
        legend_handles.append(h_patch)
        legend_labels.append(name)

    # ── Axes labels and titles ───────────────────────────────────────────

    for j, ax in enumerate(axes_state):
        ax.set_ylabel(STATE_LABELS[j], fontsize=9)
        ax.set_title(STATE_NAMES[j], fontsize=9)
        ax.set_xlim(t_plot[0], t_plot[-1])
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.tick_params(labelsize=7)

    for j, ax in enumerate(axes_sigma):
        ax.set_ylabel(SIGMA_LABELS[j], fontsize=9)
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.set_xlim(t_plot[0], t_plot[-1])
        ax.set_ylim(bottom=0.0)
        ax.tick_params(labelsize=7)

    # ── Mark step-change times ───────────────────────────────────────────

    for ax in axes_state + axes_sigma:
        for t_ev in (20.0, 45.0):
            ax.axvline(t_ev, color="black", lw=0.7, ls="--", alpha=0.4)

    # ── Input panel ──────────────────────────────────────────────────────

    _plot_input_panel(ax_input, t_meas, U_arr, D_arr, t_plot)
    for t_ev in (20.0, 45.0):
        ax_input.axvline(t_ev, color="black", lw=0.7, ls="--", alpha=0.4)

    # ── Shared legend ────────────────────────────────────────────────────

    # Truth and measurement handles from first state panel
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    truth_h = mlines.Line2D([], [], color="k", lw=1.2, alpha=0.75,
                             label="Truth")
    meas_h  = mlines.Line2D([], [], color="gray", marker=".", ls="",
                             markersize=5, label="Measurement")

    all_handles = [truth_h, meas_h] + legend_handles
    all_labels  = ["Truth", "Measurement (X)"] + legend_labels

    fig.legend(
        all_handles, all_labels,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.89),
        fontsize=8,
        ncol=1,
        framealpha=0.9,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # ── Build models ─────────────────────────────────────────────────────────
    # Simulation uses the true noise covariances.
    # Gradient-based filters (EKF, UKF) use inflated Q_c for robustness.
    # Stochastic filters (EnKF, PF) draw noise from the model's Q_c directly,
    # so they also get the inflated version to stay appropriately uncertain.

    model_sim    = ProductInhibitionBioreactor(Q_c=Q_C_TRUE, Rm=R_TRUE, Ts=DT)
    model_filter = ProductInhibitionBioreactor(Q_c=Q_C_FILTER, Rm=R, Ts=DT)

    # ── Simulate true trajectory ─────────────────────────────────────────────

    t_meas = np.arange(0.0, T_END, DT)   # measurement times: t_0, …, t_{N-1}
    X_true, Y_meas, U_arr, D_arr = simulate(
        model_sim, X0_TRUE, t_meas, D_CONST, _sin_schedule, P_TRUE, rng,
    )
    t_plot = np.concatenate([t_meas, [t_meas[-1] + DT]])   # includes t_N

    print(f"Simulation: {len(t_meas)} steps, T={T_END} h, dt={DT} h")
    print(f"True final state: S={X_true[-1,0]:.3f}  X={X_true[-1,1]:.3f}"
          f"  P={X_true[-1,2]:.3f}  g/L")

    # ── Build filter registry ────────────────────────────────────────────────

    # Each filter gets its own model instance with the tuned (inflated) Q_c.
    # UKF uses alpha=0.3 so sigma points span a physically meaningful region of
    # state-space from the start (alpha=1e-3 collapses points too tightly when P0
    # has large eigenvalues, causing clamped-state artefacts in the bioreactor).
    filter_specs: list[tuple[str, type, dict]] = [
        ("EKF", ContinuousDiscreteEKF, dict(
            model=model_filter, x0=X0_EST.copy(), P0=P0_EST.copy(),
            params=ContinuousDiscreteEKFParams(n_steps=N_SUB),
        )),
        ("UKF", ContinuousDiscreteUKF, dict(
            model=model_filter, x0=X0_EST.copy(), P0=P0_EST.copy(),
            params=ContinuousDiscreteUKFParams(
                n_steps=N_SUB, alpha=1.0, kappa=0.0, beta=2.0,
            ),
        )),
        ("EnKF", ContinuousDiscreteEnKF, dict(
            model=model_filter, x0=X0_EST.copy(), P0=P0_EST.copy(),
            params=ContinuousDiscreteEnKFParams(
                N=200, n_steps=N_SUB, seed=RNG_SEED + 1,
            ),
        )),
        ("PF", ContinuousDiscretePF, dict(
            model=model_filter, x0=X0_EST.copy(), P0=P0_EST.copy(),
            params=ContinuousDiscretePFParams(
                N=500, n_steps=N_SUB, seed=RNG_SEED + 2,
            ),
        )),
    ]

    # ── Run all filters ──────────────────────────────────────────────────────

    filter_results: dict[str, dict | None] = {}
    for name, cls, kwargs in filter_specs:
        print(f"Running {name}…")
        try:
            filt = cls(**kwargs)
        except NotImplementedError:
            print(f"  [{name}]  constructor not yet implemented — skipping.")
            filter_results[name] = None
            continue

        res = run_filter(name, filt, t_meas, Y_meas, U_arr, D_arr, P_TRUE)
        filter_results[name] = res

    # ── RMSE summary ─────────────────────────────────────────────────────────

    state_names = ["S", "X", "P"]
    print()
    print(f"{'Filter':<8}  {'RMSE_S':>8}  {'RMSE_X':>8}  {'RMSE_P':>8}  {'RMSE_all':>10}")
    print("-" * 52)
    for name, res in filter_results.items():
        if res is None:
            print(f"{name:<8}  {'(skip)':>8}")
            continue
        # x_hist is (N+1, nx); X_true is (N+1, nx) — compare at same time points
        err = res["x_hist"] - X_true
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        rmse_all = np.sqrt(np.mean(err ** 2))
        print(f"{name:<8}  {rmse[0]:>8.4f}  {rmse[1]:>8.4f}  {rmse[2]:>8.4f}  {rmse_all:>10.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────

    fig = plot_results(t_plot, t_meas, X_true, Y_meas, U_arr, D_arr,
                       filter_results)
    plt.show()


if __name__ == "__main__":
    main()
