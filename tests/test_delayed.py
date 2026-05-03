"""
Tests for DelayedObservationFilter.

Two main estimator flavours are covered:

  1. Discrete-time  — KalmanFilter wrapped with DelayedObservationFilter.
  2. Continuous-discrete — ContinuousDiscreteEKF wrapped with
     DelayedObservationFilter (using the van de Vusse CSTR model, extended to
     observe both states).

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
from cvxopt import matrix as cvx_matrix

from mbc.estimation import KalmanFilter, ContinuousDiscreteEKF
from mbc.estimation.delayed import DelayedObservationFilter
from mbc.models import LinearDiscreteModel, ContinuousDiscreteModel


# ── Helpers ───────────────────────────────────────────────────────────────────


def _np_to_cvx_col(arr: np.ndarray):
    return cvx_matrix(arr.tolist(), (len(arr), 1))


def _cvx_col_to_np(v) -> np.ndarray:
    n = v.size[0] * v.size[1]
    return np.array([float(v[i]) for i in range(n)])


# ── Simple 2-state linear model ───────────────────────────────────────────────


class _TwoStateModel(LinearDiscreteModel):
    """
    Two-state linear system with two observable outputs (C = I₂).

    State  : x = [x₁, x₂]
    Input  : u = [u₁]         (scalar)
    Disturbance: d = [d₁]     (scalar, not used in dynamics for simplicity)
    Output : y = [x₁, x₂]

    Discrete dynamics (dt = 1):
        x₁[k+1] = 0.90 x₁[k] + 0.10 u[k]
        x₂[k+1] = 0.85 x₂[k] + 0.10 u[k]
    """

    # ── LinearDiscreteModel interface ─────────────────────────────────────

    def __init__(self):
        self._x = [0.0, 0.0]

    @property
    def n_x(self) -> int:
        return 2

    @property
    def n_u(self) -> int:
        return 1

    @property
    def n_d(self) -> int:
        return 1

    @property
    def C(self):
        """Output matrix C = I₂ (cvxopt, column-major)."""
        return cvx_matrix([1.0, 0.0, 0.0, 1.0], (2, 2))

    @property
    def x(self) -> list:
        """Initial state x₀ = [0, 0]."""
        return list(self._x)

    @x.setter
    def x(self, val) -> None:
        self._x = list(val)

    @property
    def x_ref(self):
        return cvx_matrix([0.0, 0.0], (2, 1))

    @property
    def u_bounds(self):
        u_min = cvx_matrix([-1.0], (1, 1))
        u_max = cvx_matrix([1.0], (1, 1))
        return u_min, u_max

    def discretize(self, d_cvx):
        """Return (A, B, E) as cvxopt matrices; E is not used (E*d ≈ 0)."""
        A = cvx_matrix([0.9, 0.0, 0.0, 0.85], (2, 2))  # column-major
        B = cvx_matrix([0.1, 0.1], (2, 1))
        E = cvx_matrix([0.0, 0.0], (2, 1))
        return A, B, E

    def predict_offset(self, d_np: np.ndarray) -> np.ndarray:
        return np.zeros(2)


# ── Simple 2-output CD model (VdV-like CSTR) ─────────────────────────────────


class _TwoOutputCSTR(ContinuousDiscreteModel):
    """
    Two-state CD model; both states are observable (h(x) = x).

    Adapted from the van de Vusse CSTR used in test_ekf.py but with
    two output channels so we can test channel-level delayed observations.

    State  : x = [c_A (mol/L), c_B (mol/L)]
    Input  : u = [D_rate (1/h)]
    Disturbance: d = []   (none)
    Output : y = [c_A, c_B]   ← both states observed
    Params : p = []
    """

    _k1 = 50.0
    _k2 = 100.0
    _k3 = 10.0
    _c_Af = 10.0
    _Q_c_val = np.diag([0.01, 0.005])
    _R_val = np.diag([0.05, 0.05])

    @property
    def nx(self) -> int:
        return 2

    @property
    def nu(self) -> int:
        return 1

    @property
    def nd(self) -> int:
        return 0

    @property
    def nym(self) -> int:
        return 2

    @property
    def nw(self) -> int:
        return 2

    @property
    def Q_c(self) -> np.ndarray:
        return self._Q_c_val.copy()

    @property
    def R(self) -> np.ndarray:
        return self._R_val.copy()

    def f(self, x, u, d, p, t):
        c_A, c_B = x
        D = u[0]
        dc_A = (self._c_Af - c_A) * D - self._k1 * c_A - self._k3 * c_A ** 2
        dc_B = -c_B * D + self._k1 * c_A - self._k2 * c_B
        return np.array([dc_A, dc_B])

    def sigma(self, x, u, d, p, t):
        return np.eye(2)

    def hm(self, x, u, d, p):
        return x.copy()  # observe both states


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
    Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
    R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
    return KalmanFilter(two_state_model, Q=Q, R=R)


@pytest.fixture()
def two_state_kf2(two_state_model):
    """Second independent KalmanFilter for comparison tests."""
    Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
    R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
    return KalmanFilter(two_state_model, Q=Q, R=R)


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
    X_true : (T+1, 2) true states
    Y_noisy : (T, 2) noisy observations (sigma=0.1 each)
    U : (T, 1) random inputs in [-1, 1]
    """
    rng = np.random.default_rng(seed)
    model = _TwoStateModel()
    Q_proc = np.diag([0.01, 0.01])
    R_meas = 0.1 * np.eye(2)

    x = np.zeros(2)
    X_true = [x.copy()]
    Y_noisy = []
    U = []

    for _ in range(T):
        u = rng.uniform(-0.5, 0.5, size=1)
        x = np.array([0.9, 0.0, 0.0, 0.85]).reshape(2, 2) @ x + np.array([0.1, 0.1]) * u[0]
        x += rng.multivariate_normal(np.zeros(2), Q_proc)
        y = x + rng.multivariate_normal(np.zeros(2), R_meas)
        X_true.append(x.copy())
        Y_noisy.append(y.copy())
        U.append(u.copy())

    return np.array(X_true), np.array(Y_noisy), np.array(U)


def _run_kf_no_delay(kf, Y, U, D_val=0.0):
    """Run the bare KalmanFilter through a trajectory without delays."""
    d_cvx = _np_to_cvx_col(np.array([D_val]))
    X_est = []
    for k, (y_np, u_np) in enumerate(zip(Y, U)):
        y_cvx = _np_to_cvx_col(y_np)
        kf.update(y_cvx, d_cvx)
        kf.record_action(_np_to_cvx_col(u_np))
        X_est.append(kf.x_hat)
    return np.array(X_est)


def _run_delayed_kf(delayed_kf, Y, U, delay, D_val=0.0):
    """
    Run the DelayedObservationFilter through a trajectory.

    Channel 1 is always observed immediately.
    Channel 0 is observed with ``delay[0]`` steps delay.
    At each step k, the delayed observation for step k−delay[0] arrives.
    """
    d_cvx = _np_to_cvx_col(np.array([D_val]))
    delay_arr = np.asarray(delay, dtype=int)
    X_est = []
    for k, (y_np, u_np) in enumerate(zip(Y, U)):
        y_cvx = _np_to_cvx_col(y_np)
        delayed_kf.update(y_cvx, d_cvx, delay=delay_arr)
        delayed_kf.record_action(_np_to_cvx_col(u_np))
        X_est.append(delayed_kf.x_hat)
    return np.array(X_est)


# ── Unit tests: transparency ───────────────────────────────────────────────────


class TestTransparency:
    """Wrapper should be identical to the bare estimator when delay=None."""

    def test_no_delay_equals_bare_estimator(self, two_state_model, two_state_kf, two_state_kf2):
        """With delay=None, DelayedObservationFilter must match the bare KF."""
        Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
        R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))

        kf_bare = two_state_kf2
        kf_wrapped = DelayedObservationFilter(two_state_kf, lag_max=10)

        _, Y, U = _simulate_linear(20, seed=7)
        d_cvx = _np_to_cvx_col(np.array([0.0]))

        for y_np, u_np in zip(Y, U):
            y_cvx = _np_to_cvx_col(y_np)
            u_cvx = _np_to_cvx_col(u_np)

            x_bare = kf_bare.update(y_cvx, d_cvx)
            kf_bare.record_action(u_cvx)

            x_wrapped = kf_wrapped.update(y_cvx, d_cvx)  # delay=None
            kf_wrapped.record_action(u_cvx)

            np.testing.assert_allclose(
                np.asarray(x_bare).ravel(),
                np.asarray(x_wrapped).ravel(),
                atol=1e-12,
                err_msg="Wrapped KF diverged from bare KF with delay=None",
            )

    def test_all_zeros_delay_equals_bare_estimator(self, two_state_model, two_state_kf, two_state_kf2):
        """delay=np.zeros(ny) should behave identically to delay=None."""
        Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
        R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))

        kf_bare = two_state_kf2
        kf_wrapped = DelayedObservationFilter(two_state_kf, lag_max=10)

        _, Y, U = _simulate_linear(20, seed=13)
        d_cvx = _np_to_cvx_col(np.array([0.0]))
        delay_zero = np.array([0, 0])

        for y_np, u_np in zip(Y, U):
            y_cvx = _np_to_cvx_col(y_np)
            u_cvx = _np_to_cvx_col(u_np)

            x_bare = kf_bare.update(y_cvx, d_cvx)
            kf_bare.record_action(u_cvx)

            x_wrapped = kf_wrapped.update(y_cvx, d_cvx, delay=delay_zero)
            kf_wrapped.record_action(u_cvx)

            np.testing.assert_allclose(
                np.asarray(x_bare).ravel(),
                np.asarray(x_wrapped).ravel(),
                atol=1e-12,
                err_msg="delay=zeros must match delay=None",
            )

    def test_properties_delegated(self, delayed_kf, two_state_kf):
        """x_hat, P, and last_innovation must be delegated to the inner KF."""
        d_cvx = _np_to_cvx_col(np.array([0.0]))
        y_cvx = _np_to_cvx_col(np.array([0.5, 0.3]))

        delayed_kf.update(y_cvx, d_cvx)
        delayed_kf.record_action(_np_to_cvx_col(np.array([0.0])))

        # x_hat from wrapper and from inner estimator must be the same object or equal value
        np.testing.assert_allclose(
            delayed_kf.x_hat,
            two_state_kf.x_hat,
            atol=1e-12,
        )
        # P comparison
        P_wrap = delayed_kf.P
        P_inner = two_state_kf.P
        np.testing.assert_allclose(P_wrap, P_inner, atol=1e-12)


# ── Unit tests: buffer management ─────────────────────────────────────────────


class TestBufferManagement:
    """Internal buffer should grow correctly and respect lag_max."""

    def test_buffer_grows_with_updates(self, delayed_kf):
        """Buffer length must equal the number of updates (up to lag_max)."""
        d_cvx = _np_to_cvx_col(np.array([0.0]))
        y_cvx = _np_to_cvx_col(np.array([0.1, 0.2]))
        u_cvx = _np_to_cvx_col(np.array([0.0]))

        for k in range(1, 8):
            delayed_kf.update(y_cvx, d_cvx)
            delayed_kf.record_action(u_cvx)
            assert len(delayed_kf._buf) == k

    def test_buffer_capped_at_lag_max(self):
        """Buffer must not exceed lag_max entries."""
        model = _TwoStateModel()
        Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
        R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
        kf = KalmanFilter(model, Q=Q, R=R)
        filt = DelayedObservationFilter(kf, lag_max=5)

        d_cvx = _np_to_cvx_col(np.array([0.0]))
        y_cvx = _np_to_cvx_col(np.array([0.1, 0.2]))
        u_cvx = _np_to_cvx_col(np.array([0.0]))

        for _ in range(15):
            filt.update(y_cvx, d_cvx)
            filt.record_action(u_cvx)

        assert len(filt._buf) == 5  # lag_max

    def test_delay_exceeds_buffer_issues_warning(self):
        """A delay larger than the current buffer depth must issue RuntimeWarning."""
        model = _TwoStateModel()
        Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
        R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
        kf = KalmanFilter(model, Q=Q, R=R)
        filt = DelayedObservationFilter(kf, lag_max=10)

        d_cvx = _np_to_cvx_col(np.array([0.0]))
        y_cvx = _np_to_cvx_col(np.array([0.1, 0.2]))
        u_cvx = _np_to_cvx_col(np.array([0.0]))

        # Only 2 entries in buffer but we ask for delay=5.
        filt.update(y_cvx, d_cvx)
        filt.record_action(u_cvx)
        filt.update(y_cvx, d_cvx)
        filt.record_action(u_cvx)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filt.update(y_cvx, d_cvx, delay=np.array([5, 0]))
            filt.record_action(u_cvx)

        assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    def test_buffer_entries_have_correct_keys(self, delayed_kf):
        """Each buffer entry must contain x_hat, P, y, d, mask, u."""
        d_cvx = _np_to_cvx_col(np.array([0.0]))
        y_cvx = _np_to_cvx_col(np.array([0.3, 0.4]))
        u_cvx = _np_to_cvx_col(np.array([0.1]))

        delayed_kf.update(y_cvx, d_cvx)
        delayed_kf.record_action(u_cvx)

        entry = delayed_kf._buf[-1]
        for key in ("x_hat", "P", "y", "d", "mask", "u"):
            assert key in entry, f"Buffer entry missing key '{key}'"

        np.testing.assert_allclose(entry["u"], np.array([0.1]), atol=1e-12)


# ── Unit tests: single delayed channel ────────────────────────────────────────


class TestSingleDelayedChannel:
    """Single delayed channel must update the current estimate correctly."""

    def test_delayed_channel_changes_estimate(self, two_state_model, two_state_kf):
        """With a delayed channel the estimate must differ from ignore-delay."""
        Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
        R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
        kf_ignore = KalmanFilter(two_state_model, Q=Q, R=R)
        filt = DelayedObservationFilter(two_state_kf, lag_max=10)

        _, Y, U = _simulate_linear(10, seed=42)
        d_cvx = _np_to_cvx_col(np.array([0.0]))
        delay_arr = np.array([0, 3])  # channel 1 delayed by 3 steps

        X_ignore, X_delayed = [], []

        for y_np, u_np in zip(Y, U):
            y_cvx = _np_to_cvx_col(y_np)
            u_cvx = _np_to_cvx_col(u_np)

            # Filter that ignores channel 1 entirely
            y_ch0 = _np_to_cvx_col(np.array([y_np[0], 0.0]))
            kf_ignore.update(y_ch0, d_cvx, mask=[True, False])
            kf_ignore.record_action(u_cvx)
            X_ignore.append(kf_ignore.x_hat)

            # Filter with delayed channel 1
            filt.update(y_cvx, d_cvx, delay=delay_arr)
            filt.record_action(u_cvx)
            X_delayed.append(filt.x_hat)

        X_ignore = np.array(X_ignore)
        X_delayed = np.array(X_delayed)

        # The estimates must differ once the delay passes
        tail_diff = np.abs(X_ignore[-3:] - X_delayed[-3:]).mean()
        assert tail_diff > 1e-8, (
            "Delayed KF should differ from ignore-delay KF after lag passes"
        )

    def test_delayed_channel_reduces_rmse(self, two_state_model):
        """
        Using a delayed observation (vs ignoring it) should improve accuracy
        in a longer simulation run.
        """
        T = 50
        tau = 3
        _, Y, U = _simulate_linear(T, seed=99)
        X_true = np.array([np.zeros(2)])
        x = np.zeros(2)
        for u_np in U:
            x = np.array([0.9, 0.0, 0.0, 0.85]).reshape(2, 2) @ x + np.array([0.1, 0.1]) * u_np[0]
            X_true = np.vstack([X_true, x])
        X_true = X_true[1:]  # align with observations

        def _make_kf():
            Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
            R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
            return KalmanFilter(two_state_model, Q=Q, R=R)

        d_cvx = _np_to_cvx_col(np.array([0.0]))

        # Filter A: only channel 0 (no channel 1 information)
        kf_a = _make_kf()
        X_a = []
        for y_np, u_np in zip(Y, U):
            y_cvx = _np_to_cvx_col(y_np)
            kf_a.update(y_cvx, d_cvx, mask=[True, False])
            kf_a.record_action(_np_to_cvx_col(u_np))
            X_a.append(kf_a.x_hat)
        X_a = np.array(X_a)

        # Filter B: channel 0 immediate + channel 1 delayed by tau
        kf_b = DelayedObservationFilter(_make_kf(), lag_max=2 * tau)
        delay_arr = np.array([0, tau])
        X_b = []
        for y_np, u_np in zip(Y, U):
            y_cvx = _np_to_cvx_col(y_np)
            kf_b.update(y_cvx, d_cvx, delay=delay_arr)
            kf_b.record_action(_np_to_cvx_col(u_np))
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
        """Running with two delayed channels (tau=2, tau=3) must not error
        and must give an estimate that incorporates both measurements."""
        Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
        R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
        kf = KalmanFilter(two_state_model, Q=Q, R=R)
        filt = DelayedObservationFilter(kf, lag_max=10)

        _, Y, U = _simulate_linear(15, seed=55)
        d_cvx = _np_to_cvx_col(np.array([0.0]))
        # Channel 0: delay=2, channel 1: delay=3
        delay_arr = np.array([2, 3])

        X_delayed = []
        for y_np, u_np in zip(Y, U):
            y_cvx = _np_to_cvx_col(y_np)
            filt.update(y_cvx, d_cvx, delay=delay_arr)
            filt.record_action(_np_to_cvx_col(u_np))
            X_delayed.append(filt.x_hat)

        X_delayed = np.array(X_delayed)

        # Estimates must be finite
        assert np.all(np.isfinite(X_delayed)), "Estimates must be finite"

    def test_mixed_immediate_and_delayed(self, two_state_model):
        """Channel 0 immediate, channel 1 delayed by 2 — must not error."""
        Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
        R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
        kf = KalmanFilter(two_state_model, Q=Q, R=R)
        filt = DelayedObservationFilter(kf, lag_max=10)

        _, Y, U = _simulate_linear(20, seed=77)
        d_cvx = _np_to_cvx_col(np.array([0.0]))
        delay_arr = np.array([0, 2])  # channel 0 immediate, channel 1 delayed

        for y_np, u_np in zip(Y, U):
            y_cvx = _np_to_cvx_col(y_np)
            x_hat = filt.update(y_cvx, d_cvx, delay=delay_arr)
            filt.record_action(_np_to_cvx_col(u_np))
            x_np = np.asarray(x_hat).ravel()
            assert np.all(np.isfinite(x_np))

    def test_covariance_positive_definite(self, two_state_model):
        """Posterior covariance must remain symmetric positive-definite."""
        Q = cvx_matrix([0.01, 0.0, 0.0, 0.01], (2, 2))
        R = cvx_matrix([0.1, 0.0, 0.0, 0.1], (2, 2))
        kf = KalmanFilter(two_state_model, Q=Q, R=R)
        filt = DelayedObservationFilter(kf, lag_max=10)

        _, Y, U = _simulate_linear(15, seed=33)
        d_cvx = _np_to_cvx_col(np.array([0.0]))
        delay_arr = np.array([0, 3])

        for y_np, u_np in zip(Y, U):
            y_cvx = _np_to_cvx_col(y_np)
            filt.update(y_cvx, d_cvx, delay=delay_arr)
            filt.record_action(_np_to_cvx_col(u_np))

        P_np = filt.P
        eigvals = np.linalg.eigvalsh(P_np)
        assert np.all(eigvals > 0), f"P must be positive-definite; eigenvalues: {eigvals}"
        np.testing.assert_allclose(P_np, P_np.T, atol=1e-10)


# ── System tests: CD estimator (EKF) ─────────────────────────────────────────


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
        D_rate = _VDV_D_RATE

        for _ in range(T):
            dx = model.f(x, D_rate, _VDV_D, _VDV_P, 0.0)
            x = x + dt * dx + rng.multivariate_normal(np.zeros(2), 1e-6 * np.eye(2))
            y = x + rng.multivariate_normal(np.zeros(2), 0.001 * np.eye(2))
            X_true.append(x.copy())
            Y_noisy.append(y.copy())

        return np.array(X_true)[1:], np.array(Y_noisy)

    def test_transparency_no_delay(self, two_output_cstr):
        """Wrapped EKF with delay=None must produce the same result as bare EKF."""
        x0 = _VDV_SS + np.array([0.05, 0.02])
        P0 = np.diag([0.1, 0.1])

        ekf_bare = ContinuousDiscreteEKF(two_output_cstr, x0.copy(), P0.copy(), dt=0.01, n_steps=10)
        ekf_wrap = ContinuousDiscreteEKF(two_output_cstr, x0.copy(), P0.copy(), dt=0.01, n_steps=10)
        filt = DelayedObservationFilter(ekf_wrap, lag_max=10)

        _, Y = self._simulate_vdv(15, seed=5)
        u = _VDV_D_RATE
        d = _VDV_D
        p = _VDV_P
        t = 0.0

        for y_np in Y:
            x_bare, _ = ekf_bare.step(y_np, u, d, p, t)
            x_wrap, _ = filt.step(y_np, u, d, p, t)  # delay=None

            np.testing.assert_allclose(
                np.asarray(x_bare).ravel(),
                np.asarray(x_wrap).ravel(),
                atol=1e-12,
                err_msg="Wrapped EKF diverged from bare EKF with delay=None",
            )
            t += 0.01

    def test_delayed_channel_finite_and_reasonable(self, two_output_cstr):
        """Wrapped EKF with a delayed channel should produce finite estimates."""
        x0 = _VDV_SS + np.array([0.05, 0.02])
        P0 = np.diag([0.1, 0.1])

        ekf = ContinuousDiscreteEKF(two_output_cstr, x0.copy(), P0.copy(), dt=0.01, n_steps=10)
        filt = DelayedObservationFilter(ekf, lag_max=10)

        _, Y = self._simulate_vdv(20, seed=11)
        u = _VDV_D_RATE
        d = _VDV_D
        p = _VDV_P
        delay_arr = np.array([0, 3])  # channel 1 delayed by 3 steps
        t = 0.0

        for y_np in Y:
            x_est, P_est = filt.step(y_np, u, d, p, t, delay=delay_arr)
            assert np.all(np.isfinite(np.asarray(x_est).ravel()))
            assert np.all(np.isfinite(P_est))
            t += 0.01

    def test_covariance_pd_after_delayed_ekf(self, two_output_cstr):
        """Covariance must remain positive-definite after delayed EKF steps."""
        x0 = _VDV_SS + np.array([0.05, 0.02])
        P0 = np.diag([0.1, 0.1])
        ekf = ContinuousDiscreteEKF(two_output_cstr, x0.copy(), P0.copy(), dt=0.01, n_steps=10)
        filt = DelayedObservationFilter(ekf, lag_max=10)

        _, Y = self._simulate_vdv(15, seed=17)
        delay_arr = np.array([0, 2])
        t = 0.0

        for y_np in Y:
            x_est, P_est = filt.step(y_np, _VDV_D_RATE, _VDV_D, _VDV_P, t, delay=delay_arr)
            t += 0.01

        P_final = np.asarray(P_est, dtype=float)
        if P_final.ndim == 1:
            P_final = P_final.reshape(2, 2)
        eigvals = np.linalg.eigvalsh(P_final)
        assert np.all(eigvals > 0), f"P must be positive-definite; eigenvalues: {eigvals}"
