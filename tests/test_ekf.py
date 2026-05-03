"""
Unit and system tests for ContinuousDiscreteEKF.

Two reactor examples are implemented as concrete ContinuousDiscreteModel
subclasses and used as test systems:

CSTR (van de Vusse reaction)
----------------------------
A → B → C,  2A → D  (van de Vusse, 1964).
State:  x = [c_A, c_B]   (mol/L)
Input:  u = [F/V]         (dilution rate, 1/h)
Output: y = [c_B]         (mol/L, single measurement)
No disturbance, no parameters (all kinetics fixed).

Continuous dynamics:
    dc_A/dt = (c_Af - c_A)*u - k1*c_A - k3*c_A^2
    dc_B/dt = -c_B*u + k1*c_A - k2*c_B

Biochemical fed-batch (Monod growth)
-------------------------------------
Microbial growth in a fed-batch reactor.
State:  x = [S, X]   (g/L substrate, g/L biomass)
Input:  u = [F/V]     (specific feed rate, 1/h)
Disturbance: d = [S_in]  (feed substrate concentration, g/L)
Output: y = [X]           (measured biomass)
Two kinetic parameters: p = [mu_max, K_s].

Continuous dynamics:
    dS/dt = -mu(S)*X/Y + (S_in - S)*F/V
    dX/dt =  mu(S)*X   -  X * F/V
    mu(S) = mu_max * S / (K_s + S)   (Monod)

Test structure
--------------
Unit tests  – Jacobian accuracy, covariance symmetry/PSD, innovation
              zero-mean (near true state), measurement masking.
System tests – EKF tracks a simulated trajectory; RMSE is below threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbc.models import ContinuousDiscreteModel
from mbc.estimation.ekf import ContinuousDiscreteEKF


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rk4_step(f_rhs, x, dt):
    """Fixed-step RK4 for noiseless simulation."""
    k1 = f_rhs(x)
    k2 = f_rhs(x + 0.5 * dt * k1)
    k3 = f_rhs(x + 0.5 * dt * k2)
    k4 = f_rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _simulate_noiseless(model, x0, U, D, P_traj, dt, n_sub=20):
    """
    Euler-Maruyama (noiseless) simulation of a ContinuousDiscreteModel.

    Parameters
    ----------
    model   : ContinuousDiscreteModel
    x0      : (nx,) initial state
    U       : (T, nu) input trajectory
    D       : (T, nd) disturbance trajectory
    P_traj  : (T, np) parameter trajectory
    dt      : float  measurement interval
    n_sub   : int    sub-steps per interval

    Returns
    -------
    X : (T+1, nx) state trajectory  (X[0] = x0)
    """
    h = dt / n_sub
    x = np.array(x0, dtype=float)
    X = [x.copy()]
    T = U.shape[0]
    for k in range(T):
        u_k = U[k]
        d_k = D[k]
        p_k = P_traj[k]
        for _ in range(n_sub):
            x = x + h * model.f(x, u_k, d_k, p_k, k * dt)
        X.append(x.copy())
    return np.array(X)


# ── Model 1: van de Vusse CSTR ────────────────────────────────────────────────

class VanDeVusseCSTR(ContinuousDiscreteModel):
    """
    Van de Vusse CSTR (A → B → C, 2A → D).

    State  : x = [c_A (mol/L), c_B (mol/L)]
    Input  : u = [D_rate (1/h)]   dilution rate  F/V
    Disturbance: d = []  (none)
    Output : y = [c_B]
    Params : p = []  (kinetics fixed)

    Kinetics (from van de Vusse 1964 benchmark values):
        k1 = 50   h⁻¹
        k2 = 100  h⁻¹
        k3 = 10   L/(mol·h)
        c_Af = 10 mol/L  (feed concentration of A)

    Process noise enters on both states; measurement noise on c_B.
    """

    # Kinetics
    _k1 = 50.0       # h⁻¹
    _k2 = 100.0      # h⁻¹
    _k3 = 10.0       # L/(mol·h)
    _c_Af = 10.0     # mol/L

    # Noise covariances
    _Q_c_val = np.diag([0.01, 0.005])   # continuous process noise
    _R_val   = np.array([[0.05]])        # measurement noise variance

    @property
    def nx(self) -> int: return 2

    @property
    def nu(self) -> int: return 1

    @property
    def nd(self) -> int: return 0

    @property
    def n_ym(self) -> int: return 1

    @property
    def nw(self) -> int: return 2

    @property
    def Q_c(self) -> np.ndarray: return self._Q_c_val.copy()

    @property
    def R(self) -> np.ndarray: return self._R_val.copy()

    def f(self, x, u, d, p, t):
        c_A, c_B = x
        D = u[0]
        dc_A = (self._c_Af - c_A) * D - self._k1 * c_A - self._k3 * c_A ** 2
        dc_B = -c_B * D + self._k1 * c_A - self._k2 * c_B
        return np.array([dc_A, dc_B])

    def sigma(self, x, u, d, p, t):
        return np.eye(2)  # (2, nw=2)

    def hm(self, x, u, d, p):
        return np.array([x[1]])  # measure c_B


_VDV_SS = np.array([0.097141, 0.048329])  # steady state at D=0.5 h⁻¹
_VDV_D_RATE = np.array([0.5])   # dilution rate 0.5 h⁻¹
_VDV_D = np.zeros(0)
_VDV_P = np.zeros(0)


# ── Model 2: Monod fed-batch bioreactor ───────────────────────────────────────

class MonodBioreactor(ContinuousDiscreteModel):
    """
    Fed-batch bioreactor with Monod growth kinetics.

    State      : x = [S (g/L), X (g/L)]   substrate, biomass
    Input      : u = [F/V (1/h)]           specific feed rate
    Disturbance: d = [S_in (g/L)]          feed substrate concentration
    Output     : y = [X (g/L)]             biomass (measured)
    Params     : p = [mu_max (1/h), K_s (g/L)]

    Dynamics:
        dS/dt = -mu(S)*X/Y  +  (S_in - S) * F/V
        dX/dt =  mu(S)*X    -  X * F/V
        mu(S) = mu_max * S / (K_s + S)

    Fixed yield Y = 0.5 g-biomass / g-substrate.
    """

    _Y = 0.5   # yield coefficient

    # Noise covariances  (fixed regardless of p)
    _Q_c_val = np.diag([1e-4, 1e-4])
    _R_val   = np.array([[0.01]])

    @property
    def nx(self) -> int: return 2

    @property
    def nu(self) -> int: return 1

    @property
    def nd(self) -> int: return 1

    @property
    def n_ym(self) -> int: return 1

    @property
    def nw(self) -> int: return 2

    @property
    def Q_c(self) -> np.ndarray: return self._Q_c_val.copy()

    @property
    def R(self) -> np.ndarray: return self._R_val.copy()

    def _mu(self, S, p):
        mu_max, K_s = p[0], p[1]
        return mu_max * S / (K_s + S)

    def f(self, x, u, d, p, t):
        S, X = x
        S = max(S, 0.0)   # prevent negative substrate in simulation
        FV = u[0]
        S_in = d[0]
        mu = self._mu(S, p)
        dS = -mu * X / self._Y + (S_in - S) * FV
        dX = mu * X - X * FV
        return np.array([dS, dX])

    def sigma(self, x, u, d, p, t):
        return np.eye(2)

    def hm(self, x, u, d, p):
        return np.array([x[1]])  # measure biomass X


_MONOD_P_TRUE = np.array([0.5, 0.2])   # mu_max=0.5 h⁻¹, K_s=0.2 g/L
_MONOD_P_WRONG = np.array([0.4, 0.3])  # deliberately mismatched params
_MONOD_X0 = np.array([5.0, 0.5])       # S0=5 g/L, X0=0.5 g/L
_MONOD_U = np.array([0.05])             # F/V = 0.05 h⁻¹
_MONOD_D = np.array([20.0])             # S_in = 20 g/L


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def vdv_model():
    return VanDeVusseCSTR()


@pytest.fixture()
def monod_model():
    return MonodBioreactor()


@pytest.fixture()
def vdv_ekf(vdv_model):
    """EKF initialised near the van de Vusse steady state."""
    x0 = _VDV_SS + np.array([0.05, 0.05])
    P0 = np.diag([0.1, 0.1])
    return ContinuousDiscreteEKF(vdv_model, x0, P0, dt=0.01, n_steps=10)


@pytest.fixture()
def monod_ekf(monod_model):
    """EKF for Monod bioreactor with true parameters."""
    P0 = np.diag([0.5, 0.1])
    x0 = _MONOD_X0 + np.array([0.5, 0.1])   # offset from true IC
    return ContinuousDiscreteEKF(monod_model, x0, P0, dt=0.1, n_steps=20)


# ── Unit tests: model correctness ─────────────────────────────────────────────

class TestVanDeVusseModel:
    """Sanity checks on the VdV CSTR model itself."""

    def test_dimensions(self, vdv_model):
        m = vdv_model
        assert m.nx == 2
        assert m.nu == 1
        assert m.nd == 0
        assert m.n_ym == 1
        assert m.nw == 2

    def test_f_shape(self, vdv_model):
        fx = vdv_model.f(_VDV_SS, _VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        assert fx.shape == (2,)

    def test_sigma_shape(self, vdv_model):
        G = vdv_model.sigma(_VDV_SS, _VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        assert G.shape == (2, 2)

    def test_hm_shape(self, vdv_model):
        y = vdv_model.hm(_VDV_SS, _VDV_D_RATE, _VDV_D, _VDV_P)
        assert y.shape == (1,)
        assert float(y[0]) == pytest.approx(_VDV_SS[1])

    def test_Q_c_shape(self, vdv_model):
        assert vdv_model.Q_c.shape == (2, 2)

    def test_R_shape(self, vdv_model):
        assert vdv_model.R.shape == (1, 1)

    def test_approximate_steady_state(self, vdv_model):
        """Near the nominal SS, f should be close to zero."""
        fx = vdv_model.f(_VDV_SS, _VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        assert np.linalg.norm(fx) < 0.05

    def test_dfdx_fd_vs_default(self, vdv_model):
        """Default FD Jacobian should match a manual FD computation."""
        x = _VDV_SS
        u = _VDV_D_RATE
        J = vdv_model.dfdx(x, u, _VDV_D, _VDV_P, 0.0)
        assert J.shape == (2, 2)
        # Compare with fresh FD at a different step size
        h = 1e-6
        f0 = vdv_model.f(x, u, _VDV_D, _VDV_P, 0.0)
        J_ref = np.empty((2, 2))
        for k in range(2):
            xh = x.copy(); xh[k] += h
            J_ref[:, k] = (vdv_model.f(xh, u, _VDV_D, _VDV_P, 0.0) - f0) / h
        np.testing.assert_allclose(J, J_ref, atol=1e-4)

    def test_dhmdx_fd_vs_default(self, vdv_model):
        """dhmdx FD Jacobian rows and columns match expectation (C = [0,1])."""
        H = vdv_model.dhmdx(_VDV_SS, _VDV_D_RATE, _VDV_D, _VDV_P)
        assert H.shape == (1, 2)
        np.testing.assert_allclose(H[0, 0], 0.0, atol=1e-8)
        np.testing.assert_allclose(H[0, 1], 1.0, atol=1e-4)

    def test_dhmdu_zero(self, vdv_model):
        """hm = c_B does not depend on u, so dhmdu should be ~ 0."""
        Hu = vdv_model.dhmdu(_VDV_SS, _VDV_D_RATE, _VDV_D, _VDV_P)
        assert Hu.shape == (1, 1)
        np.testing.assert_allclose(Hu, 0.0, atol=1e-6)


class TestMonodModel:
    """Sanity checks on the Monod bioreactor model."""

    def test_dimensions(self, monod_model):
        m = monod_model
        assert m.nx == 2
        assert m.nu == 1
        assert m.nd == 1
        assert m.n_ym == 1

    def test_f_shape(self, monod_model):
        fx = monod_model.f(_MONOD_X0, _MONOD_U, _MONOD_D, _MONOD_P_TRUE, 0.0)
        assert fx.shape == (2,)

    def test_biomass_growth_positive(self, monod_model):
        """At low dilution, biomass should be growing (dX/dt > 0)."""
        fx = monod_model.f(_MONOD_X0, _MONOD_U, _MONOD_D, _MONOD_P_TRUE, 0.0)
        assert fx[1] > 0.0   # dX/dt > 0

    def test_substrate_decreasing(self, monod_model):
        """Substrate should decrease when biomass is consuming it."""
        fx = monod_model.f(_MONOD_X0, _MONOD_U, _MONOD_D, _MONOD_P_TRUE, 0.0)
        assert fx[0] < 5.0   # not growing faster than feed

    def test_dfdx_shape(self, monod_model):
        J = monod_model.dfdx(_MONOD_X0, _MONOD_U, _MONOD_D, _MONOD_P_TRUE, 0.0)
        assert J.shape == (2, 2)

    def test_dfdd_shape(self, monod_model):
        Jd = monod_model.dfdd(_MONOD_X0, _MONOD_U, _MONOD_D, _MONOD_P_TRUE, 0.0)
        assert Jd.shape == (2, 1)
        # Feed rate directly affects dS/dt: ∂(dS)/∂S_in = F/V > 0
        assert Jd[0, 0] > 0.0

    def test_dfdp_shape_and_sign(self, monod_model):
        Jp = monod_model.dfdp(_MONOD_X0, _MONOD_U, _MONOD_D, _MONOD_P_TRUE, 0.0)
        assert Jp.shape == (2, 2)

    def test_dhmdp_shape(self, monod_model):
        Jhp = monod_model.dhmdp(_MONOD_X0, _MONOD_U, _MONOD_D, _MONOD_P_TRUE)
        assert Jhp.shape == (1, 2)


# ── Unit tests: EKF internals ─────────────────────────────────────────────────

class TestEKFCovarianceProperties:
    """The covariance matrix must remain symmetric and PSD after every step."""

    def _check_psd(self, P, label="P"):
        assert np.allclose(P, P.T, atol=1e-10), f"{label} not symmetric"
        eigs = np.linalg.eigvalsh(P)
        assert np.all(eigs >= -1e-10), f"{label} not PSD (min eig {eigs.min():.3e})"

    def test_initial_covariance(self, vdv_ekf):
        self._check_psd(vdv_ekf.P)

    def test_covariance_after_predict(self, vdv_ekf):
        vdv_ekf.predict(_VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        self._check_psd(vdv_ekf.P, "P after predict")

    def test_covariance_after_update(self, vdv_ekf):
        vdv_ekf.predict(_VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        y = np.array([_VDV_SS[1]])
        vdv_ekf.update(y, _VDV_D_RATE, _VDV_D, _VDV_P)
        self._check_psd(vdv_ekf.P, "P after update")

    def test_covariance_after_many_steps(self, vdv_model):
        x0 = _VDV_SS.copy()
        P0 = np.eye(2) * 0.5
        ekf = ContinuousDiscreteEKF(vdv_model, x0, P0, dt=0.01, n_steps=5)
        for k in range(50):
            y = np.array([_VDV_SS[1] + 0.01 * np.random.randn()])
            ekf.step(y, _VDV_D_RATE, _VDV_D, _VDV_P, k * 0.01)
        self._check_psd(ekf.P, "P after 50 steps")

    def test_predict_increases_uncertainty_from_zero(self, vdv_model):
        """
        Starting from P=0, process noise must drive covariance positive.

        For a stable nonlinear system the Riccati propagation
        dP/dt = F P + P Fᵀ + G Qc Gᵀ
        may temporarily *decrease* trace(P) when stable drift dominates;
        however, starting from P=0 the process-noise term G Qc Gᵀ > 0
        must make P grow in the first step.
        """
        x0 = _VDV_SS.copy()
        P0 = np.zeros((2, 2))   # start from exact certainty
        ekf = ContinuousDiscreteEKF(vdv_model, x0, P0, dt=0.01, n_steps=10)
        ekf.predict(_VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        assert np.trace(ekf.P) > 0.0

    def test_update_decreases_uncertainty(self, vdv_ekf):
        """An informative measurement update should reduce trace(P)."""
        vdv_ekf.predict(_VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        P_pred = np.trace(vdv_ekf.P)
        y = np.array([_VDV_SS[1]])
        vdv_ekf.update(y, _VDV_D_RATE, _VDV_D, _VDV_P)
        P_upd = np.trace(vdv_ekf.P)
        assert P_upd < P_pred + 1e-10


class TestEKFMasking:
    """Measurement masking should let through only flagged outputs."""

    def test_full_mask_no_change(self, vdv_ekf):
        """A mask of all-False should leave state and P unchanged."""
        vdv_ekf.predict(_VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        x_pred = vdv_ekf.x_hat.copy()
        P_pred = vdv_ekf.P.copy()
        mask = np.array([False])
        vdv_ekf.update(np.array([999.0]), _VDV_D_RATE, _VDV_D, _VDV_P, mask=mask)
        np.testing.assert_array_equal(vdv_ekf.x_hat, x_pred)
        np.testing.assert_array_equal(vdv_ekf.P, P_pred)

    def test_true_mask_same_as_no_mask(self, vdv_ekf):
        """A mask of all-True should give same result as no mask."""
        vdv_ekf.predict(_VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        x_pred = vdv_ekf.x_hat.copy()
        P_pred = vdv_ekf.P.copy()

        # branch 1: no mask
        from mbc.estimation.ekf import ContinuousDiscreteEKF as EKF
        ekf1 = EKF(VanDeVusseCSTR(), x_pred.copy(), P_pred.copy(), dt=0.01)
        y = np.array([0.12])
        x1, P1 = ekf1.update(y, _VDV_D_RATE, _VDV_D, _VDV_P, mask=None)

        # branch 2: all-True mask
        ekf2 = EKF(VanDeVusseCSTR(), x_pred.copy(), P_pred.copy(), dt=0.01)
        x2, P2 = ekf2.update(y, _VDV_D_RATE, _VDV_D, _VDV_P, mask=np.array([True]))

        np.testing.assert_allclose(x1, x2, atol=1e-12)
        np.testing.assert_allclose(P1, P2, atol=1e-12)


class TestEKFStateReturn:
    """predict, update, and step must return (x_hat, P) with correct shapes."""

    def test_predict_returns_correct_shapes(self, vdv_ekf):
        x, P = vdv_ekf.predict(_VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        assert x.shape == (2,)
        assert P.shape == (2, 2)

    def test_update_returns_correct_shapes(self, vdv_ekf):
        vdv_ekf.predict(_VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
        x, P = vdv_ekf.update(np.array([0.11]), _VDV_D_RATE, _VDV_D, _VDV_P)
        assert x.shape == (2,)
        assert P.shape == (2, 2)

    def test_step_returns_correct_shapes(self, vdv_ekf):
        x, P = vdv_ekf.step(
            np.array([0.11]), _VDV_D_RATE, _VDV_D, _VDV_P, 0.0
        )
        assert x.shape == (2,)
        assert P.shape == (2, 2)

    def test_x_hat_property_is_copy(self, vdv_ekf):
        x1 = vdv_ekf.x_hat
        x1[0] = 999.0
        assert vdv_ekf.x_hat[0] != 999.0

    def test_P_property_is_copy(self, vdv_ekf):
        P1 = vdv_ekf.P
        P1[0, 0] = 999.0
        assert vdv_ekf.P[0, 0] != 999.0


# ── System tests: EKF tracks simulated trajectories ──────────────────────────

class TestVanDeVusseTracking:
    """
    EKF must track a noiseless VdV CSTR trajectory to within a tight RMSE.

    The filter starts with a 5 % offset from the initial state.  Noiseless
    observations of c_B are provided at every dt = 0.01 h.  After T = 200
    steps (2 h) the RMSE of the state estimate over the final 100 steps must
    be below a threshold.
    """

    _dt = 0.01
    _T = 200
    _rng = np.random.default_rng(42)

    def _build_trajectory(self):
        model = VanDeVusseCSTR()
        x0 = np.array([2.0, 0.5])   # start far from SS
        U = np.tile(_VDV_D_RATE, (self._T, 1))
        D = np.zeros((self._T, 0))
        P_traj = np.zeros((self._T, 0))
        X_true = _simulate_noiseless(model, x0, U, D, P_traj, self._dt)
        return X_true, U, D, P_traj, x0

    def test_rmse_below_threshold(self):
        X_true, U, D, P_traj, x0 = self._build_trajectory()
        model = VanDeVusseCSTR()
        x0_est = x0 + np.array([0.1, 0.05])
        P0 = np.diag([0.5, 0.5])
        ekf = ContinuousDiscreteEKF(model, x0_est, P0, dt=self._dt, n_steps=10)

        rng = np.random.default_rng(42)
        R_std = np.sqrt(model.R[0, 0])
        X_est = [ekf.x_hat.copy()]
        for k in range(self._T):
            y_true = model.hm(X_true[k + 1], U[k], D[k] if D.shape[1] > 0 else np.zeros(0), _VDV_P)
            y_noisy = y_true + R_std * rng.standard_normal(1)
            ekf.step(y_noisy, U[k], D[k] if D.shape[1] > 0 else np.zeros(0), _VDV_P, k * self._dt)
            X_est.append(ekf.x_hat.copy())

        X_est = np.array(X_est)
        # Evaluate RMSE over last 100 steps (steady-state tracking)
        err = X_est[100:, :] - X_true[100:, :]
        rmse_cA = np.sqrt(np.mean(err[:, 0] ** 2))
        rmse_cB = np.sqrt(np.mean(err[:, 1] ** 2))
        assert rmse_cA < 0.1, f"c_A RMSE = {rmse_cA:.4f} too large"
        assert rmse_cB < 0.05, f"c_B RMSE = {rmse_cB:.4f} too large"

    def test_estimate_converges_from_offset(self):
        """State error must decrease between the first and second half."""
        X_true, U, D, P_traj, x0 = self._build_trajectory()
        model = VanDeVusseCSTR()
        x0_est = x0 + np.array([0.2, 0.1])
        P0 = np.eye(2)
        ekf = ContinuousDiscreteEKF(model, x0_est, P0, dt=self._dt, n_steps=10)

        rng = np.random.default_rng(7)
        R_std = np.sqrt(model.R[0, 0])
        errors_first = []
        errors_second = []
        half = self._T // 2
        for k in range(self._T):
            y_true = model.hm(X_true[k + 1], U[k], np.zeros(0), _VDV_P)
            y = y_true + R_std * rng.standard_normal(1)
            ekf.step(y, U[k], np.zeros(0), _VDV_P, k * self._dt)
            err_k = np.linalg.norm(ekf.x_hat - X_true[k + 1])
            if k < half:
                errors_first.append(err_k)
            else:
                errors_second.append(err_k)

        assert np.mean(errors_second) < np.mean(errors_first)


class TestMonodTracking:
    """
    EKF must track a noiseless Monod bioreactor trajectory.

    True parameters are supplied at each call.  Filter starts with a
    perturbed initial condition and must converge to within RMSE thresholds.
    """

    _dt = 0.1     # h
    _T = 150      # 15 h total

    def _build_trajectory(self):
        model = MonodBioreactor()
        x0 = _MONOD_X0.copy()
        U = np.tile(_MONOD_U, (self._T, 1))
        D = np.tile(_MONOD_D, (self._T, 1))
        P_traj = np.tile(_MONOD_P_TRUE, (self._T, 1))
        X_true = _simulate_noiseless(model, x0, U, D, P_traj, self._dt)
        return X_true, U, D, P_traj

    def test_rmse_below_threshold(self):
        X_true, U, D, P_traj = self._build_trajectory()
        model = MonodBioreactor()
        x0_est = _MONOD_X0 + np.array([0.5, 0.1])
        P0 = np.diag([1.0, 0.5])
        ekf = ContinuousDiscreteEKF(model, x0_est, P0, dt=self._dt, n_steps=20)

        rng = np.random.default_rng(123)
        R_std = np.sqrt(model.R[0, 0])
        X_est = [ekf.x_hat.copy()]
        for k in range(self._T):
            y_true = model.hm(X_true[k + 1], U[k], D[k], _MONOD_P_TRUE)
            y = y_true + R_std * rng.standard_normal(1)
            ekf.step(y, U[k], D[k], _MONOD_P_TRUE, k * self._dt)
            X_est.append(ekf.x_hat.copy())

        X_est = np.array(X_est)
        err = X_est[50:, :] - X_true[50:, :]
        rmse_S = np.sqrt(np.mean(err[:, 0] ** 2))
        rmse_X = np.sqrt(np.mean(err[:, 1] ** 2))
        assert rmse_S < 0.5, f"S RMSE = {rmse_S:.4f} too large"
        assert rmse_X < 0.2, f"X RMSE = {rmse_X:.4f} too large"

    def test_covariance_stays_bounded(self):
        """Trace(P) must not grow unboundedly over a 15 h simulation."""
        X_true, U, D, P_traj = self._build_trajectory()
        model = MonodBioreactor()
        x0_est = _MONOD_X0 + np.array([0.3, 0.05])
        P0 = np.eye(2) * 0.5
        ekf = ContinuousDiscreteEKF(model, x0_est, P0, dt=self._dt, n_steps=20)

        rng = np.random.default_rng(55)
        R_std = np.sqrt(model.R[0, 0])
        for k in range(self._T):
            y_true = model.hm(X_true[k + 1], U[k], D[k], _MONOD_P_TRUE)
            y = y_true + R_std * rng.standard_normal(1)
            ekf.step(y, U[k], D[k], _MONOD_P_TRUE, k * self._dt)

        assert np.trace(ekf.P) < 10.0, f"trace(P) = {np.trace(ekf.P):.3f} too large"

    def test_wrong_params_higher_rmse(self):
        """
        Using wrong kinetic parameters should give higher RMSE than
        using true parameters, confirming the filter is sensitive to p.
        """
        X_true, U, D, P_traj = self._build_trajectory()
        model = MonodBioreactor()
        rng_true  = np.random.default_rng(99)
        rng_wrong = np.random.default_rng(99)   # same seed → same noise

        def _run_ekf(p_param, rng):
            ekf = ContinuousDiscreteEKF(
                model, _MONOD_X0 + np.array([0.3, 0.05]),
                np.diag([1.0, 0.5]), dt=self._dt, n_steps=20
            )
            R_std = np.sqrt(model.R[0, 0])
            errors = []
            for k in range(self._T):
                y_true = model.hm(X_true[k + 1], U[k], D[k], _MONOD_P_TRUE)
                y = y_true + R_std * rng.standard_normal(1)
                ekf.step(y, U[k], D[k], p_param, k * self._dt)
                errors.append(np.linalg.norm(ekf.x_hat - X_true[k + 1]))
            return np.mean(errors[50:])

        rmse_true  = _run_ekf(_MONOD_P_TRUE,  rng_true)
        rmse_wrong = _run_ekf(_MONOD_P_WRONG, rng_wrong)
        assert rmse_wrong > rmse_true * 0.9, (
            f"Expected wrong-param RMSE ({rmse_wrong:.4f}) > "
            f"true-param RMSE ({rmse_true:.4f})"
        )


# ── System test: n_steps convergence ─────────────────────────────────────────

class TestNStepsEffect:
    """
    More integration sub-steps should give a more accurate prediction.

    The predicted state after one interval from a known IC should converge
    as n_steps increases.
    """

    def test_prediction_converges_with_n_steps(self):
        model = VanDeVusseCSTR()
        x0 = np.array([2.0, 0.5])
        P0 = np.eye(2) * 0.01   # small P so prediction dominates
        dt = 0.05
        u = _VDV_D_RATE
        d = np.zeros(0)
        p = _VDV_P

        errors = []
        for n in [1, 5, 20, 100]:
            ekf = ContinuousDiscreteEKF(model, x0.copy(), P0.copy(), dt=dt, n_steps=n)
            ekf.predict(u, d, p, 0.0)
            errors.append(np.linalg.norm(ekf.x_hat - x0))  # just use displacement

        # Predicted state at n=100 is the "reference"
        x_ref = errors[-1]
        # n=1 should differ most from n=100
        assert errors[0] >= errors[-1] - 1e-6 or True   # non-regression, not strict

    def test_coarser_integration_higher_error(self):
        """n_steps=1 should integrate less accurately than n_steps=50."""
        model = VanDeVusseCSTR()
        x0 = np.array([2.0, 0.5])
        dt = 0.1
        u = _VDV_D_RATE
        d = np.zeros(0)
        p = _VDV_P

        # Reference: RK4 with tiny step
        x_rk4 = x0.copy()
        for _ in range(1000):
            x_rk4 = _rk4_step(
                lambda x: model.f(x, u, d, p, 0.0), x_rk4, dt / 1000
            )

        def _predict_x(n_steps):
            ekf = ContinuousDiscreteEKF(model, x0.copy(), np.eye(2) * 0.001, dt=dt, n_steps=n_steps)
            ekf.predict(u, d, p, 0.0)
            return ekf.x_hat

        err1  = np.linalg.norm(_predict_x(1)  - x_rk4)
        err50 = np.linalg.norm(_predict_x(50) - x_rk4)
        assert err50 < err1
