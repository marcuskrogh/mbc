"""
Visual example: Heated room with reversible heat pump — linearised CD OCP.

System
------
Single-zone thermal model (RC network) for a ~25 m² room with a reversible
heat pump:

    C · dT_in/dt = P_hp + Q_solar − (T_in − T_out) / R

    x = T_in  (°C)          indoor temperature
    u = P_hp  (kW)          heat-pump thermal power (+ heat, − cool)
    d = [T_out, Q_solar]    outdoor temperature (°C), solar gain (kW)

Physical parameters (order-of-magnitude realistic):
    UA ≈ 75 W/K  →  R ≈ 13 °C/kW
    C  ≈ 1.0 kWh/°C  (air + light structure)
    |P_hp| ≤ 3.0 kW thermal

The plant is formulated as :class:`~mbc.models.ContinuousDiscreteLinearisedSDE`
around a winter operating point ``(u_s, d_s)``.  Control uses
:class:`~mbc.control.StandardLinearizedContinuousDiscreteOCP`, which solves the
QP in deviation coordinates and returns absolute inputs.

Comfort band
------------
Soft output constraints keep ``T_in`` within **22–24 °C** (``z_offset = 1 °C``
around a 23 °C setpoint).

Disturbance forecast
--------------------
Horizon forecast of outdoor temperature and solar gain (sinusoidal winter day).

Electricity price (slack formulation)
-------------------------------------
Bidirectional heat pump — signed-magnitude slack decomposition::

    P_hp = s − t,    s, t ≥ 0
    stage cost  +=  price[k] · (s + t)   ≈  price[k] · |P_hp|

``price(t)`` is a sinusoidal EUR/kWh curve (cheap overnight, expensive evening).

Scenario (48 h, Ts = 0.5 h)
---------------------------
Cold winter week; MPC pre-heats during cheap hours and exploits solar gain.

Usage
-----
    python scripts/heated_room_linearised_ocp_visual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mbc.models import ContinuousDiscreteLinearisedSDE
from mbc.estimation import ContinuousDiscreteLinearKF, ContinuousDiscreteLinearKFParams
from mbc.control import StandardLinearizedContinuousDiscreteOCP


# ── Timing ────────────────────────────────────────────────────────────────────

TS    = 0.5          # h
T_END = 48.0         # h
N_SIM = int(T_END / TS)
N_MPC = 24           # 12 h horizon

T_COMFORT = 23.0     # °C — centre of comfort band
T_COMFORT_LO = 22.0  # °C
T_COMFORT_HI = 24.0  # °C
Z_OFFSET = 1.0       # °C — soft band half-width → [22, 24] °C


# ── Thermal parameters (realistic single-zone room) ───────────────────────────

UA   = 0.075         # kW/K  — envelope conductance (~75 W/K)
R_TH = 1.0 / UA      # °C/kW
C_TH = 1.0           # kWh/°C — thermal capacitance (air + light structure)
_A   = -1.0 / (R_TH * C_TH)
_B   = 1.0 / C_TH
_E   = np.array([[-_A, _B]])   # d = [T_out, Q_solar]

# Winter operating point: ~23 °C indoors at mild outdoor conditions
U_SS = np.array([1.3])         # kW — steady heating at d_s
D_SS = np.array([2.0, 0.3])    # °C, kW  →  x_s ≈ 23 °C

P_HP_MAX = 3.0       # kW thermal — room-sized air-to-air heat pump


# ── Model ─────────────────────────────────────────────────────────────────────


class HeatedRoom(ContinuousDiscreteLinearisedSDE):
    """
    Linearised single-zone heated room with heat pump and two disturbances.

    State      : x = [T_in (°C)]
    Input      : u = [P_hp (kW)]   bidirectional heat pump
    Disturbance: d = [T_out (°C), Q_solar (kW)]
    Measurement: y = T_in
    """

    _G   = np.array([[0.03]])
    _Rm  = np.array([[0.04]])
    _U_MIN = np.array([-P_HP_MAX])
    _U_MAX = np.array([+P_HP_MAX])

    def __init__(self) -> None:
        self._x = [float(T_COMFORT)]

    @property
    def Ts(self) -> float:
        return TS

    @property
    def nx(self) -> int:
        return 1

    @property
    def nu(self) -> int:
        return 1

    @property
    def nd(self) -> int:
        return 2

    @property
    def A(self) -> np.ndarray:
        return np.array([[_A]])

    @property
    def B(self) -> np.ndarray:
        return np.array([[_B]])

    @property
    def E(self) -> np.ndarray:
        return _E.copy()

    @property
    def G(self) -> np.ndarray:
        return self._G.copy()

    @property
    def Cm(self) -> np.ndarray:
        return np.eye(1)

    @property
    def Cz(self) -> np.ndarray:
        return np.eye(1)

    @property
    def Rm(self) -> np.ndarray:
        return self._Rm.copy()

    @property
    def u_s(self) -> np.ndarray:
        return U_SS.copy()

    @property
    def d_s(self) -> np.ndarray:
        return D_SS.copy()

    @property
    def x(self) -> list[float]:
        return self._x

    @x.setter
    def x(self, val) -> None:
        self._x = list(val)

    @property
    def x_ref(self) -> np.ndarray:
        return np.array([T_COMFORT])

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._U_MIN.copy(), self._U_MAX.copy()


# ── Forecast helpers ──────────────────────────────────────────────────────────


def outdoor_temperature(t_h: float) -> float:
    """Outdoor temperature (°C): ~2 °C night, ~5 °C afternoon (mild winter)."""
    return 15.0 + 1.5 * np.sin(2.0 * np.pi * (t_h - 9.0) / 24.0)


def solar_gain(t_h: float) -> float:
    """Solar heating (kW): south-facing glazing, peak ~1.5 kW."""
    phase = (t_h % 24.0) - 7.0
    if phase < 0.0 or phase > 11.0:
        return 0.0
    return 1.5 * np.sin(np.pi * phase / 11.0) ** 2


def electricity_price(t_h: float) -> float:
    """Day-ahead electricity price (EUR/kWh): trough ~02 h, peak ~18 h."""
    return 0.11 + 0.03 * np.sin(2.0 * np.pi * (t_h - 14.0) / 24.0)


def disturbance_at(t_h: float) -> np.ndarray:
    return np.array([outdoor_temperature(t_h), solar_gain(t_h)])


def build_disturbance_forecast(k: int) -> np.ndarray:
    return np.array([disturbance_at((k + j) * TS) for j in range(N_MPC)])


def build_price_forecast(k: int) -> np.ndarray:
    return np.array([[electricity_price((k + j) * TS)] for j in range(N_MPC)])


# ── Simulation ────────────────────────────────────────────────────────────────


def run() -> None:
    rng   = np.random.default_rng(31)
    model = HeatedRoom()
    disc  = model.discretize()
    Ad, Bd, Ed = disc.Ad, disc.Bd, disc.Ed
    L_Q = np.linalg.cholesky(disc.Qd)
    R_std = float(np.sqrt(model.Rm[0, 0]))

    kf = ContinuousDiscreteLinearKF(
        model,
        x0=np.array([T_COMFORT]),
        P0=np.array([[0.5]]),
        params=ContinuousDiscreteLinearKFParams(n_steps=8),
    )

    ocp = StandardLinearizedContinuousDiscreteOCP(
        model,
        N=N_MPC,
        Q=400.0,
        R=0.05,
        P=800.0,
        S=np.eye(1) * 0.5,
        z_offset=Z_OFFSET,
        rho=5e4,
        rho_lin=200.0,
        solver="highs",
    )

    x_true = np.array([T_COMFORT])
    X_true = np.zeros(N_SIM + 1)
    X_true[0] = x_true[0]
    x_hist = np.zeros(N_SIM + 1)
    x_hist[0] = kf.x_hat[0]
    Y_meas = np.zeros(N_SIM)
    U_arr  = np.zeros(N_SIM)
    T_out_arr = np.zeros(N_SIM)
    Q_sol_arr = np.zeros(N_SIM)
    price_arr = np.zeros(N_SIM)

    u_prev = U_SS.copy()
    d_now  = disturbance_at(0.0)

    for k in range(N_SIM):
        t_h = k * TS
        d_now = disturbance_at(t_h)
        T_out_arr[k] = d_now[0]
        Q_sol_arr[k] = d_now[1]
        price_arr[k] = electricity_price(t_h)

        ym = np.array([x_true[0] + R_std * rng.standard_normal()])
        Y_meas[k] = ym[0]
        kf.step(ym, u_prev, d_now)

        D_fc = build_disturbance_forecast(k)
        price_fc = build_price_forecast(k)

        ocp.set_disturbance_profile(D_fc)
        ocp.set_input_linear_cost_profile(
            price_fc,
            signed_magnitude_input_indices=np.array([0]),
        )

        U_seq, _ = ocp.solve(
            kf.x_hat,
            D_fc.reshape(-1),
            np.array([T_COMFORT]),
            u_prev=u_prev,
        )
        u_k = U_seq[: model.nu].copy()
        U_arr[k] = u_k[0]
        u_prev = u_k

        proc = L_Q @ rng.standard_normal(1)
        x_true = Ad @ x_true + Bd @ u_k + Ed @ d_now + proc
        X_true[k + 1] = x_true[0]
        x_hist[k + 1] = kf.x_hat[0]

    max_u = float(np.max(np.abs(U_arr)))
    assert max_u <= P_HP_MAX + 1e-4, f"Heat pump exceeded bounds: {max_u:.3f} kW"
    in_band = np.mean((X_true >= T_COMFORT_LO) & (X_true <= T_COMFORT_HI)) * 100.0
    print(
        f"Peak |P_hp| = {max_u:.2f} kW (limit +/-{P_HP_MAX} kW)  |  "
        f"T_in range [{X_true.min():.1f}, {X_true.max():.1f}] C  |  "
        f"{in_band:.0f}% inside comfort band"
    )

    _plot(X_true, x_hist, Y_meas, U_arr, T_out_arr, Q_sol_arr, price_arr)


# ── Figure ────────────────────────────────────────────────────────────────────


def _plot(X_true, x_hist, Y_meas, U_arr, T_out, Q_sol, price) -> None:
    t = np.arange(N_SIM + 1) * TS
    t_meas = t[1:]

    fig = plt.figure(figsize=(12, 11))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Heated Room  —  Linearised CD OCP  +  Forecasted Disturbances\n"
        f"Comfort band {T_COMFORT_LO:.0f}–{T_COMFORT_HI:.0f} °C  |  "
        f"Heat pump ±{P_HP_MAX:.1f} kW  |  price·|P| slack penalty",
        fontsize=11, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.48,
                           top=0.91, bottom=0.07, left=0.09, right=0.92)
    ax_T = fig.add_subplot(gs[0])
    ax_u = fig.add_subplot(gs[1], sharex=ax_T)
    ax_d = fig.add_subplot(gs[2], sharex=ax_T)
    ax_p = fig.add_subplot(gs[3], sharex=ax_T)

    ax_T.axhspan(T_COMFORT_LO, T_COMFORT_HI, color="#d5e8d4", alpha=0.5,
                 label=f"Comfort {T_COMFORT_LO:.0f}–{T_COMFORT_HI:.0f} °C")
    ax_T.axhline(T_COMFORT, color="#555", ls="--", lw=1.0, label=f"Setpoint {T_COMFORT:.0f} °C")
    ax_T.plot(t, X_true, color="#e07b39", lw=1.8, label="True T_in")
    ax_T.plot(t, x_hist, color="#2166ac", lw=1.8, label="KF estimate")
    ax_T.scatter(t_meas, Y_meas, s=5, color="#c0392b", alpha=0.4, label="Measurements")
    ax_T.set_ylabel("T_in  (°C)")
    ax_T.set_title("Indoor temperature")
    ax_T.legend(fontsize=8, loc="lower right", framealpha=0.85)
    ax_T.grid(True, alpha=0.25)

    ax_u.step(t_meas, U_arr, where="post", color="#27ae60", lw=1.8, label="P_hp (kW)")
    ax_u.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax_u.axhline(P_HP_MAX, color="crimson", ls="--", lw=0.9, alpha=0.7)
    ax_u.axhline(-P_HP_MAX, color="crimson", ls="--", lw=0.9, alpha=0.7,
                 label=f"Bounds ±{P_HP_MAX:.1f} kW")
    ax_u.set_ylabel("P_hp  (kW)")
    ax_u.set_title("Heat pump  (+ heat / − cool)")
    ax_u.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_u.grid(True, alpha=0.25)

    ax_d.plot(t_meas, T_out, color="#5dade2", lw=1.5, label="T_out")
    ax_d.set_ylabel("T_out  (°C)", color="#5dade2")
    ax_d.tick_params(axis="y", labelcolor="#5dade2")
    ax_d2 = ax_d.twinx()
    ax_d2.fill_between(t_meas, 0, Q_sol, color="#f4d03f", alpha=0.45, label="Q_solar")
    ax_d2.set_ylabel("Q_solar  (kW)", color="#b7950b")
    ax_d2.tick_params(axis="y", labelcolor="#b7950b")
    ax_d.set_title("Disturbance forecasts")
    h1, l1 = ax_d.get_legend_handles_labels()
    h2, l2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", framealpha=0.85)
    ax_d.grid(True, alpha=0.25)

    ax_p.step(t_meas, price, where="post", color="#8e44ad", lw=1.8, label="Electricity price")
    ax_p.set_ylabel("Price  (EUR/kWh)")
    ax_p.set_xlabel("Time  (h)")
    ax_p.set_title("Sinusoidal day-ahead price")
    ax_p.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_p.grid(True, alpha=0.25)

    for ax in (ax_T, ax_u, ax_d, ax_p):
        for day in (24.0, 48.0):
            if day <= T_END:
                ax.axvline(day, color="dimgray", lw=0.6, ls=":", alpha=0.45)

    plt.setp(ax_T.get_xticklabels(), visible=False)
    plt.setp(ax_u.get_xticklabels(), visible=False)
    plt.setp(ax_d.get_xticklabels(), visible=False)
    plt.show()


if __name__ == "__main__":
    run()
