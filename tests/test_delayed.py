"""
Tests for DelayedObservationFilter.

Two main estimator flavours are covered:

  1. Discrete-time  — KalmanFilter wrapped with DelayedObservationFilter.
  2. Continuous-discrete — ContinuousDiscreteEKF wrapped with
     DelayedObservationFilter (using the van de Vusse CSTR model, extended to
     observe both states).

The wrapper exposes a single unified API

    step(ym, u, d, p=None, t=None, mask=None, delay=None) → (x_hat, P, …)

matching the underlying estimator (linear DT or CD).

Test structure
--------------
Unit tests:
  - Wrapper is transparent when ``delay=None`` (identical to bare estimator).
  - Wrapper is transparent when all delays are zero.
  - Buffer entry count grows correctly.
  - Warning is issued when delay exceeds the buffer depth.
  - Delegation of ``x_hat``, ``P``, ``last_innovation`` properties.

System / accuracy tests:
  - A delayed observation channel improves the state estimate (lower RMSE)
    compared to ignoring it.
  - Multiple delayed channels are handled correctly (sorted by lag ascending).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mbc.estimation import KalmanFilter, ContinuousDiscreteEKF
from mbc.estimation.delayed import DelayedObservationFilter
from mbc.models import LinearDiscreteModel, ContinuousDiscreteModel


# ── Simple 2-state linear model ───────────────────────────────────────────────


class _TwoStateModel(LinearDiscreteModel):
    """
    Two-state linear system with two observable outputs (Cm = I₂).

    State        : x = [x₁, x₂]
    Input        : u = [u₁]                 (scalar)
    Disturbance  : d = [d₁]                 (scalar, not used in dynamics)
    Measurement  : ym = [x₁, x₂]

    Discrete dynamics (dt = 1):
        x₁[k+1] = 0.90 x₁[k] + 0.10 u[k]
        x₂[k+1] = 0.85 x₂[k] + 0.10 u[k]
    """

    def __init__(self):
        self._x = [0.0, 0.0]

    @property
    def nx(self) -> int: return 2
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 1
    @property
    def Cm(self) -> np.ndarray: return np.eye(2)
    @property
    def Ad(self) -> np.ndarray: return np.array([[0.9, 0.0], [0.0, 0.85]])
    @property
    def Bd(self) -> np.ndarray: return np.array([[0.1], [0.1]])
    @property
    def Ed(self) -> np.ndarray: return np.zeros((2, 1))
    @property
    def Qd(self) -> np.ndarray: return 0.01 * np.eye(2)
    @property
    def Rm(self) -> np.ndarray: return 0.1 * np.eye(2)

    @property
    def x(self) -> list: return list(self._x)
    @x.setter
    def x(self, val) -> None: self._x = list(val)

    @property
    def x_ref(self) -> np.ndarray: return np.zeros(2)
    @property
    def u_bounds(self): return np.array([-1.0]), np.array([1.0])

    def predict_offset(self, d_np: np.ndarray) -> np.ndarray:
        return np.zeros(2)


# ── Simple 2-output CD model (VdV-like CSTR) ─────────────────────────────────


class _TwoOutputCSTR(ContinuousDiscreteModel):
    """
    Two-state CD model with both states observable (hm(x) = x).

    State       : x = [c_A (mol/L), c_B (mol/L)]
    Input       : u = [D_rate (1/h)]
    Disturbance : d = []   (none)
    Output      : ym = [c_A, c_B]
    """

    _k1 = 50.0
    _k2 = 100.0
    _k3 = 10.0
    _c_Af = 10.0
    _Q_c_val = np.diag([0.01, 0.005])
    _R_val = np.diag([0.05, 0.05])

    @property
    def nx(self) -> int: return 2
    @property
    def nu(self) -> int: return 1
    @property
    def nd(self) -> int: return 0
    @property
    def nym(self) -> int: return 2
    @property
    def nw(self) -> int: return 2
    @property
    def Rm(self) -> np.ndarray: return self._R_val.copy()

    def f(self, x, u, d, p, t):
        c_A, c_B = x
        D = u[0]
        dc_A = (self._c_Af - c_A) * D - self._k1 * c_A - self._k3 * c_A ** 2
        dc_B = -c_B * D + self._k1 * c_A - self._k2 * c_B
        return np.array([dc_A, dc_B])

    def sigma(self, x, u, d, p, t):
        return np.eye(2)

    def hm(self, x, u, d, p, t):
        return x.copy()

    def gm(self, x, u, d, p, t):
        return x.copy()

    @property
    def nz(self) -> int: return 2


_VDV_SS = np.array([0.097141, 0.048329])
_VDV_D_RATE = np.array([0.5])
_VDV_D = np.zeros(0)
_VDV_P = np.zeros(0)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def two_state_model():
    return _TwoStateModel()


@pytest.fixture()
def two_state_kf(two_state_model):
    """Bare KalmanFilter — Qd, Rm read from the model."""
    return KalmanFilter(two_state_model)


@pytest.fixture()
def two_state_kf2(two_state_model):
    """A second independent KalmanFilter for comparison tests."""
    return KalmanFilter(two_state_model)


@pytest.fixture()
def delayed_kf(two_state_kf):
    return DelayedObservationFilter(two_state_kf, lag_max=10)


@pytest.fixture()
def two_output_cstr():
    return _TwoOutputCSTR()


@pytest.fixture()
def two_output_ekf(two_output_cstr):
    x0 = _VDV_SS + np.array([0.05, 0.02])
    P0 = np.diag([0.1, 0.1])
    return ContinuousDiscreteEKF(two_output_cstr, x0, P0, dt=0.01, n_steps=10)


@pytest.fixture()
def delayed_ekf(two_output_ekf):
    return DelayedObservationFilter(two_output_ekf, lag_max=10)


# ── Helpers for generating data ───────────────────────────────────────────────


def _simulate_linear(T: int, seed: int = 0):
    """
    Simulate the TwoStateModel for T steps with random inputs and noise.

    Returns
    -------
    X_true  : (T+1, 2) true states
    Y_noisy : (T, 2) noisy observations (sigma=0.1 each)
    U       : (T, 1) random inputs in [-0.5, 0.5]
    """
    rng = np.random.default_rng(seed)
    Q_proc = np.diag([0.01, 0.01])
    R_meas = 0.1 * np.eye(2)

    x = np.zeros(2)
    X_true = [x.copy()]
    Y_noisy = []
    U = []

    Ad = np.array([[0.9, 0.0], [0.0, 0.85]])
    Bd = np.array([[0.1], [0.1]])

    for _ in range(T):
        u = rng.uniform(-0.5, 0.5, size=1)
        x = Ad @ x + Bd[:, 0] * u[0]
        x += rng.multivariate_normal(np.zeros(2), Q_proc)
        y = x + rng.multivariate_normal(np.zeros(2), R_meas)
        X_true.append(x.copy())
        Y_noisy.append(y.copy())
        U.append(u.copy())

    return np.array(X_true), np.array(Y_noisy), np.array(U)


# ── Unit tests: transparency ──────────────────────────────────────────────────


class TestTransparency:
    """Wrapper should be identical to the bare estimator when delay=None."""

    def test_no_delay_equals_bare_estimator(
        self, two_state_model, two_state_kf, two_state_kf2
    ):
        """With ``delay=None`` the wrapped KF must match the bare KF."""
        kf_bare = two_state_kf2
        kf_wrap = DelayedObservationFilter(two_state_kf, lag_max=10)

        _, Y, U = _simulate_linear(20, seed=7)
        d = np.array([0.0])

        for ym, u in zip(Y, U):
            x_bare, _ = kf_bare.step(ym, u, d)
            x_wrap, _ = kf_wrap.step(ym, u, d)

            np.testing.assert_allclose(
                np.asarray(x_bare).ravel(),
                np.asarray(x_wrap).ravel(),
                atol=1e-12,
                err_msg="Wrapped KF diverged from bare KF with delay=None",
            )

    def test_all_zeros_delay_equals_bare_estimator(
        self, two_state_model, two_state_kf, two_state_kf2
    ):
        """``delay = np.zeros(ny)`` should behave identically to ``delay=None``."""
        kf_bare = two_state_kf2
        kf_wrap = DelayedObservationFilter(two_state_kf, lag_max=10)

        _, Y, U = _simulate_linear(20, seed=13)
        d = np.array([0.0])
        delay_zero = np.array([0, 0])

        for ym, u in zip(Y, U):
            x_bare, _ = kf_bare.step(ym, u, d)
            x_wrap, _ = kf_wrap.step(ym, u, d, delay=delay_zero)

            np.testing.assert_allclose(
                np.asarray(x_bare).ravel(),
                np.asarray(x_wrap).ravel(),
                atol=1e-12,
                err_msg="delay=zeros must match delay=None",
            )

    def test_properties_delegated(self, delayed_kf, two_state_kf):
        """``x_hat``, ``P``, and ``last_innovation`` must be delegated."""
        u = np.zeros(1)
        d = np.array([0.0])
        ym = np.array([0.5, 0.3])

        delayed_kf.step(ym, u, d)

        np.testing.assert_allclose(
            delayed_kf.x_hat, two_state_kf.x_hat, atol=1e-12,
        )
        np.testing.assert_allclose(
            delayed_kf.P, two_state_kf.P, atol=1e-12,
        )


# ── Unit tests: buffer management ─────────────────────────────────────────────


class TestBufferManagement:
    """Internal buffer should grow correctly and respect lag_max."""

    def test_buffer_grows_with_updates(self, delayed_kf):
        u = np.zeros(1)
        d = np.array([0.0])
        ym = np.array([0.1, 0.2])

        for k in range(1, 8):
            delayed_kf.step(ym, u, d)
            assert len(delayed_kf._buf) == k

    def test_buffer_capped_at_lag_max(self):
        model = _TwoStateModel()
        kf = KalmanFilter(model)
        filt = DelayedObservationFilter(kf, lag_max=5)

        u = np.zeros(1)
        d = np.array([0.0])
        ym = np.array([0.1, 0.2])

        for _ in range(15):
            filt.step(ym, u, d)

        assert len(filt._buf) == 5  # lag_max

    def test_delay_exceeds_buffer_issues_warning(self):
        """A delay larger than the current buffer depth must issue RuntimeWarning."""
        model = _TwoStateModel()
        kf = KalmanFilter(model)
        filt = DelayedObservationFilter(kf, lag_max=10)

        u = np.zeros(1)
        d = np.array([0.0])
        ym = np.array([0.1, 0.2])

        # Two entries in buffer but we ask for delay=5.
        filt.step(ym, u, d)
        filt.step(ym, u, d)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filt.step(ym, u, d, delay=np.array([5, 0]))

        assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    def test_buffer_entries_have_correct_keys(self, delayed_kf):
        """Each buffer entry must contain x_hat, P, ym, u, d, mask."""
        u = np.array([0.1])
        d = np.array([0.0])
        ym = np.array([0.3, 0.4])

        delayed_kf.step(ym, u, d)

        entry = delayed_kf._buf[-1]
        for key in ("x_hat", "P", "ym", "u", "d", "mask"):
            assert key in entry, f"Buffer entry missing key '{key}'"

        np.testing.assert_allclose(entry["u"], np.array([0.1]), atol=1e-12)


# ── Unit tests: single delayed channel ────────────────────────────────────────


class TestSingleDelayedChannel:
    """Single delayed channel must update the current estimate correctly."""

    def test_delayed_channel_changes_estimate(
        self, two_state_model, two_state_kf
    ):
        """With a delayed channel the estimate must differ from ignore-delay."""
        kf_ignore = KalmanFilter(two_state_model)
        filt = DelayedObservationFilter(two_state_kf, lag_max=10)

        _, Y, U = _simulate_linear(10, seed=42)
        d = np.array([0.0])
        delay_arr = np.array([0, 3])    # channel 1 delayed by 3 steps

        X_ignore, X_delayed = [], []

        for ym, u in zip(Y, U):
            # Filter that ignores channel 1 entirely (mask out)
            kf_ignore.step(ym, u, d, mask=[True, False])
            X_ignore.append(kf_ignore.x_hat)

            # Filter with delayed channel 1
            filt.step(ym, u, d, delay=delay_arr)
            X_delayed.append(filt.x_hat)

        X_ignore = np.array(X_ignore)
        X_delayed = np.array(X_delayed)

        tail_diff = np.abs(X_ignore[-3:] - X_delayed[-3:]).mean()
        assert tail_diff > 1e-8, (
            "Delayed KF should differ from ignore-delay KF after lag passes"
        )

    def test_delayed_channel_reduces_rmse(self, two_state_model):
        """Using a delayed observation should improve accuracy over ignoring it."""
        T = 50
        tau = 3
        # ``_simulate_linear`` returns the actual *noisy* trajectory of length
        # T+1 with X_true[0] = x_0; the measurements ``Y[k]`` correspond to
        # ``X_true[k+1]``, so align by slicing off the initial state.
        X_true_full, Y, U = _simulate_linear(T, seed=99)
        X_true = X_true_full[1:]

        d = np.array([0.0])

        # Filter A: only channel 0 (channel 1 ignored)
        kf_a = KalmanFilter(two_state_model)
        X_a = []
        for ym, u in zip(Y, U):
            kf_a.step(ym, u, d, mask=[True, False])
            X_a.append(kf_a.x_hat)
        X_a = np.array(X_a)

        # Filter B: channel 0 immediate + channel 1 delayed by tau
        kf_b = DelayedObservationFilter(
            KalmanFilter(two_state_model), lag_max=2 * tau,
        )
        delay_arr = np.array([0, tau])
        X_b = []
        for ym, u in zip(Y, U):
            kf_b.step(ym, u, d, delay=delay_arr)
            X_b.append(kf_b.x_hat)
        X_b = np.array(X_b)

        rmse_a = np.sqrt(np.mean((X_a[tau:] - X_true[tau:]) ** 2))
        rmse_b = np.sqrt(np.mean((X_b[tau:] - X_true[tau:]) ** 2))

        assert rmse_b < rmse_a, (
            f"Delayed-channel filter (RMSE={rmse_b:.4f}) should outperform "
            f"ignore-channel filter (RMSE={rmse_a:.4f})"
        )


# ── Unit tests: multiple delayed channels ─────────────────────────────────────


class TestMultipleDelayedChannels:
    """Two channels with different delays must both be incorporated."""

    def test_two_channels_with_different_delays(self, two_state_model):
        """delays = (2, 3) — wrapper must not error and produce finite estimates."""
        kf = KalmanFilter(two_state_model)
        filt = DelayedObservationFilter(kf, lag_max=10)

        _, Y, U = _simulate_linear(15, seed=55)
        d = np.array([0.0])
        delay_arr = np.array([2, 3])

        X_delayed = []
        for ym, u in zip(Y, U):
            filt.step(ym, u, d, delay=delay_arr)
            X_delayed.append(filt.x_hat)

        X_delayed = np.array(X_delayed)
        assert np.all(np.isfinite(X_delayed))

    def test_mixed_immediate_and_delayed(self, two_state_model):
        """Channel 0 immediate, channel 1 delayed by 2."""
        kf = KalmanFilter(two_state_model)
        filt = DelayedObservationFilter(kf, lag_max=10)

        _, Y, U = _simulate_linear(20, seed=77)
        d = np.array([0.0])
        delay_arr = np.array([0, 2])

        for ym, u in zip(Y, U):
            x_hat, _ = filt.step(ym, u, d, delay=delay_arr)
            assert np.all(np.isfinite(np.asarray(x_hat).ravel()))

    def test_covariance_positive_definite(self, two_state_model):
        """Posterior covariance must remain symmetric positive-definite."""
        kf = KalmanFilter(two_state_model)
        filt = DelayedObservationFilter(kf, lag_max=10)

        _, Y, U = _simulate_linear(15, seed=33)
        d = np.array([0.0])
        delay_arr = np.array([0, 3])

        for ym, u in zip(Y, U):
            filt.step(ym, u, d, delay=delay_arr)

        P_np = filt.P
        eigvals = np.linalg.eigvalsh(P_np)
        assert np.all(eigvals > 0), (
            f"P must be positive-definite; eigenvalues: {eigvals}"
        )
        np.testing.assert_allclose(P_np, P_np.T, atol=1e-10)


# ── System tests: CD estimator (EKF) ──────────────────────────────────────────


class TestDelayedEKF:
    """DelayedObservationFilter wrapping ContinuousDiscreteEKF."""

    def _simulate_vdv(self, T: int, seed: int = 0):
        """Euler-simulate the two-output CSTR for T steps."""
        rng = np.random.default_rng(seed)
        model = _TwoOutputCSTR()
        dt = 0.01

        x = _VDV_SS + np.array([0.05, 0.02])
        X_true = [x.copy()]
        Y_noisy = []

        for _ in range(T):
            dx = model.f(x, _VDV_D_RATE, _VDV_D, _VDV_P, 0.0)
            x = x + dt * dx + rng.multivariate_normal(np.zeros(2), 1e-6 * np.eye(2))
            y = x + rng.multivariate_normal(np.zeros(2), 0.001 * np.eye(2))
            X_true.append(x.copy())
            Y_noisy.append(y.copy())

        return np.array(X_true)[1:], np.array(Y_noisy)

    def test_transparency_no_delay(self, two_output_cstr):
        """Wrapped EKF with delay=None must produce the same result as bare EKF."""
        x0 = _VDV_SS + np.array([0.05, 0.02])
        P0 = np.diag([0.1, 0.1])

        ekf_bare = ContinuousDiscreteEKF(
            two_output_cstr, x0.copy(), P0.copy(), dt=0.01, n_steps=10,
        )
        ekf_wrap = ContinuousDiscreteEKF(
            two_output_cstr, x0.copy(), P0.copy(), dt=0.01, n_steps=10,
        )
        filt = DelayedObservationFilter(ekf_wrap, lag_max=10)

        _, Y = self._simulate_vdv(15, seed=5)
        u = _VDV_D_RATE
        d = _VDV_D
        p = _VDV_P
        t = 0.0

        for ym in Y:
            x_bare, _ = ekf_bare.step(ym, u, d, p, t)
            x_wrap, _ = filt.step(ym, u, d, p, t)

            np.testing.assert_allclose(
                np.asarray(x_bare).ravel(),
                np.asarray(x_wrap).ravel(),
                atol=1e-12,
                err_msg="Wrapped EKF diverged from bare EKF with delay=None",
            )
            t += 0.01

    def test_delayed_channel_finite_and_reasonable(self, two_output_cstr):
        """Wrapped EKF with a delayed channel must produce finite estimates."""
        x0 = _VDV_SS + np.array([0.05, 0.02])
        P0 = np.diag([0.1, 0.1])

        ekf = ContinuousDiscreteEKF(
            two_output_cstr, x0.copy(), P0.copy(), dt=0.01, n_steps=10,
        )
        filt = DelayedObservationFilter(ekf, lag_max=10)

        _, Y = self._simulate_vdv(20, seed=11)
        u = _VDV_D_RATE
        d = _VDV_D
        p = _VDV_P
        delay_arr = np.array([0, 3])
        t = 0.0

        for ym in Y:
            x_est, P_est = filt.step(ym, u, d, p, t, delay=delay_arr)
            assert np.all(np.isfinite(np.asarray(x_est).ravel()))
            assert np.all(np.isfinite(P_est))
            t += 0.01

    def test_covariance_pd_after_delayed_ekf(self, two_output_cstr):
        """Covariance must remain positive-definite after delayed EKF steps."""
        x0 = _VDV_SS + np.array([0.05, 0.02])
        P0 = np.diag([0.1, 0.1])
        ekf = ContinuousDiscreteEKF(
            two_output_cstr, x0.copy(), P0.copy(), dt=0.01, n_steps=10,
        )
        filt = DelayedObservationFilter(ekf, lag_max=10)

        _, Y = self._simulate_vdv(15, seed=17)
        delay_arr = np.array([0, 2])
        t = 0.0

        for ym in Y:
            x_est, P_est = filt.step(
                ym, _VDV_D_RATE, _VDV_D, _VDV_P, t, delay=delay_arr,
            )
            t += 0.01

        P_final = np.asarray(P_est, dtype=float)
        if P_final.ndim == 1:
            P_final = P_final.reshape(2, 2)
        eigvals = np.linalg.eigvalsh(P_final)
        assert np.all(eigvals > 0), (
            f"P must be positive-definite; eigenvalues: {eigvals}"
        )
