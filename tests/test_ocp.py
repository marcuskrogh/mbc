"""
Dedicated tests for optimal control problem formulations.

Uses lightweight scalar LTI fixtures from :mod:`tests.ocp_fixtures` and
existing nonlinear CD models from :mod:`tests.test_mpc`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbc.control import (
    StandardLinearDiscreteOCP,
    StandardLinearContinuousDiscreteOCP,
    StandardLinearizedContinuousDiscreteOCP,
    GeneralContinuousOCP,
    StandardContinuousOCP,
)
from tests.ocp_fixtures import ScalarDiscretePlant, ScalarCDPlant, TwoOutputDiscretePlant


@pytest.fixture
def scalar_disc():
    return ScalarDiscretePlant()


@pytest.fixture
def scalar_cd():
    return ScalarCDPlant()


class TestStandardLinearDiscreteOCP:
    def test_setpoint_tracking(self, scalar_disc):
        model = scalar_disc
        ocp = StandardLinearDiscreteOCP(model, N=5, Q=np.eye(1), R=np.eye(1) * 0.1)
        D = np.zeros(5)
        U, X = ocp.solve(x0=[0.0], D=D, x_ref=[2.0])
        assert U.shape == (5,)
        assert X.shape == (5,)
        assert U[0] > 0.0

    def test_hard_input_bounds(self, scalar_disc):
        model = scalar_disc
        ocp = StandardLinearDiscreteOCP(model, N=3, Q=np.eye(1), R=np.eye(1) * 0.01)
        D = np.zeros(3)
        U, _ = ocp.solve(x0=[0.0], D=D, x_ref=[10.0])
        u_min, u_max = model.u_bounds
        assert np.all(U.reshape(3, 1) >= u_min - 1e-6)
        assert np.all(U.reshape(3, 1) <= u_max + 1e-6)

    def test_soft_output_constraint(self, scalar_disc):
        ocp = StandardLinearDiscreteOCP(
            model=scalar_disc, N=4, Q=np.eye(1), R=np.eye(1) * 0.1,
            z_offset=0.5, rho=1e4,
        )
        U, X = ocp.solve(x0=[0.0], D=np.zeros(4), x_ref=[1.0])
        z = np.array(X).reshape(-1, 1)
        assert z.max() <= 1.0 + 0.5 + 0.1

    def test_rom_penalty(self, scalar_disc):
        ocp_smooth = StandardLinearDiscreteOCP(
            model=scalar_disc, N=5, Q=np.eye(1), R=np.eye(1) * 0.01, S=np.eye(1) * 1.0,
        )
        ocp_plain = StandardLinearDiscreteOCP(
            model=scalar_disc, N=5, Q=np.eye(1), R=np.eye(1) * 0.01,
        )
        D = np.zeros(5)
        U_s, _ = ocp_smooth.solve([0.0], D, [2.0], u_prev=[0.5])
        U_p, _ = ocp_plain.solve([0.0], D, [2.0], u_prev=[0.5])
        assert np.std(U_s) <= np.std(U_p) + 1e-6

    def test_hard_rom_limits(self, scalar_disc):
        ocp = StandardLinearDiscreteOCP(
            model=scalar_disc, N=4, Q=np.eye(1), R=np.eye(1) * 0.01,
            du_min=np.array([-0.2]), du_max=np.array([0.2]),
        )
        U, _ = ocp.solve(x0=[0.0], D=np.zeros(4), x_ref=[5.0], u_prev=[0.0])
        U_mat = U.reshape(4, 1)
        du0 = U_mat[0, 0] - 0.0
        assert -0.2 - 1e-6 <= du0 <= 0.2 + 1e-6
        for k in range(1, 4):
            du = U_mat[k, 0] - U_mat[k - 1, 0]
            assert -0.2 - 1e-6 <= du <= 0.2 + 1e-6


class TestStandardLinearContinuousDiscreteOCP:
    def test_zoh_tracking(self, scalar_cd):
        ocp = StandardLinearContinuousDiscreteOCP(
            scalar_cd, N=4, Q=np.eye(1), R=np.eye(1) * 0.1,
        )
        U, X = ocp.solve(x0=[0.0], D=np.zeros(4), x_ref=[1.5])
        assert U[0] > 0.0
        assert X.shape == (4,)


class TestHorizonProfiles:
    def test_ocp_reads_disturbance_from_profile(self, scalar_disc):
        ocp = StandardLinearDiscreteOCP(scalar_disc, N=3, Q=np.eye(1), R=np.eye(1) * 0.1)
        ocp.set_disturbance_profile(np.zeros(3))
        U, X = ocp.solve(x0=[0.0], x_ref=[1.0])
        assert U.shape == (3,)

    def test_ocp_time_varying_input_bounds(self, scalar_disc):
        ocp = StandardLinearDiscreteOCP(
            scalar_disc, N=3, Q=np.eye(1), R=np.eye(1) * 0.01,
        )
        ocp.set_disturbance_profile(np.zeros(3))
        ocp.set_input_bound_profiles(
            np.array([[-0.1], [-0.2], [-0.3]]),
            np.array([[0.1], [0.2], [0.3]]),
        )
        U, _ = ocp.solve(x0=[0.0], x_ref=[5.0])
        U_mat = U.reshape(3, 1)
        assert U_mat[0, 0] <= 0.1 + 1e-6
        assert U_mat[2, 0] <= 0.3 + 1e-6

    def test_mpc_shares_profile_with_ocp(self, scalar_disc):
        from mbc.estimation import DiscreteLinearKF
        from mbc.control import StandardLinearDiscreteMPC

        kf = DiscreteLinearKF(scalar_disc)
        ocp = StandardLinearDiscreteOCP(scalar_disc, N=3, Q=np.eye(1), R=np.eye(1) * 0.1)
        ctrl = StandardLinearDiscreteMPC(scalar_disc, kf, ocp)
        ctrl.set_output_tracking_weight_scale_profile(np.array([2.0, 2.0, 2.0]))
        ctrl.set_disturbance_profile(np.zeros(3))
        u, U, X = ctrl.compute(ym=[0.0])
        assert u.shape == (1,)
        assert ocp.horizon_profile is ctrl.horizon_profile

    def test_per_step_weight_scale_profile(self, scalar_disc):
        """output_tracking_weight_scale_profile still works as (N,) scalar multiplier."""
        N = 4
        D = np.zeros(N * scalar_disc.nd)  # nd=0 → empty

        ocp_scaled = StandardLinearDiscreteOCP(scalar_disc, N=N, Q=np.eye(1) * 2, R=np.eye(1))
        ocp_profile = StandardLinearDiscreteOCP(scalar_disc, N=N, Q=np.eye(1), R=np.eye(1))
        ocp_profile.set_output_tracking_weight_scale_profile(np.full(N, 2.0))

        U_scaled, _ = ocp_scaled.solve(x0=[0.0], D=D, x_ref=[1.0])
        U_profile, _ = ocp_profile.solve(x0=[0.0], D=D, x_ref=[1.0])
        np.testing.assert_allclose(U_scaled, U_profile, atol=1e-8)

    def test_tracking_and_bounds(self, scalar_cd):
        model = scalar_cd.nonlinear_model
        ocp = StandardContinuousOCP(
            model, N=3, dt=1.0, Q_z=np.eye(1), z_ref=np.array([1.0]),
            u_min=np.array([-2.0]), u_max=np.array([2.0]),
            n_steps=2,
        )
        u_opt, cost, info = ocp.solve(
            x0=np.array([0.0]),
            d_trajectory=np.zeros((3, model.nd)),
        )
        assert u_opt.shape == (3, model.nu)
        assert np.all(u_opt >= -2.0 - 1e-5)
        assert np.all(u_opt <= 2.0 + 1e-5)
        assert "X" in info


@pytest.fixture
def two_out():
    return TwoOutputDiscretePlant()


D4 = np.zeros(4 * 0)  # nd=0, so D has 0 elements per step


def _ocp(model, N=4, **kw):
    """Helper: OCP with zero disturbances pre-loaded."""
    ocp = StandardLinearDiscreteOCP(model, N=N, **kw)
    ocp.set_disturbance_profile(np.zeros(N * model.nd))
    return ocp


class TestPerStepWeightForms:
    """Test that Q, R, P, S, rho, rho_lin, z_offset accept all three width forms."""

    # ── Q (stage tracking weight) ─────────────────────────────────────────

    def test_Q_scalar_equals_matrix(self, two_out):
        ocp_s = _ocp(two_out, Q=2.0, R=np.eye(2))
        ocp_m = _ocp(two_out, Q=np.eye(2) * 2.0, R=np.eye(2))
        U_s, _ = ocp_s.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        U_m, _ = ocp_m.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        np.testing.assert_allclose(U_s, U_m, atol=1e-8)

    def test_Q_per_step_scalars(self, two_out):
        """(N,) Q form sets different per-step weights; earlier steps track faster."""
        N = 4
        # heavy weight early, light late
        Q_early = _ocp(two_out, Q=np.array([10.0, 10.0, 1.0, 1.0]), R=np.eye(2) * 0.01)
        Q_late  = _ocp(two_out, Q=np.array([1.0,  1.0, 10.0, 10.0]), R=np.eye(2) * 0.01)
        U_early, X_early = Q_early.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        U_late,  X_late  = Q_late.solve( [0.0, 0.0], x_ref=[1.0, 1.0])
        # Early-heavy forces the first input to be larger
        assert abs(U_early[0]) > abs(U_late[0]) - 1e-6

    def test_Q_per_step_vectors(self, two_out):
        """(N, nz) Q form penalises outputs asymmetrically per step; solutions differ."""
        N = 4
        nz = 2
        # heavy on output 0 only → first input channel works harder
        Q_ch0 = np.zeros((N, nz))
        Q_ch0[:, 0] = 10.0
        Q_ch0[:, 1] = 0.1
        Q_ch1 = np.zeros((N, nz))
        Q_ch1[:, 0] = 0.1
        Q_ch1[:, 1] = 10.0
        ocp0 = _ocp(two_out, Q=Q_ch0, R=np.eye(2) * 0.01)
        ocp1 = _ocp(two_out, Q=Q_ch1, R=np.eye(2) * 0.01)
        U0, _ = ocp0.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        U1, _ = ocp1.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        # Q_ch0 penalises output-0 → drives input-0 harder
        assert U0[0] > U1[0] - 1e-6

    def test_Q_matrix_form_still_works(self, two_out):
        ocp = _ocp(two_out, Q=np.diag([3.0, 1.0]), R=np.eye(2))
        U, X = ocp.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert U.shape == (8,)

    # ── R (input regularisation weight) ──────────────────────────────────

    def test_R_scalar_equals_matrix(self, two_out):
        ocp_s = _ocp(two_out, Q=np.eye(2), R=0.5)
        ocp_m = _ocp(two_out, Q=np.eye(2), R=np.eye(2) * 0.5)
        U_s, _ = ocp_s.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        U_m, _ = ocp_m.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        np.testing.assert_allclose(U_s, U_m, atol=1e-8)

    def test_R_per_step_scalars(self, two_out):
        """(N,) R form: low early penalisation → larger first input."""
        N = 4
        R_low_early  = _ocp(two_out, Q=np.eye(2), R=np.array([0.001, 0.001, 10.0, 10.0]))
        R_high_early = _ocp(two_out, Q=np.eye(2), R=np.array([10.0,  10.0,  0.001, 0.001]))
        U_le, _ = R_low_early.solve( [0.0, 0.0], x_ref=[1.0, 1.0])
        U_he, _ = R_high_early.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert np.linalg.norm(U_le[:2]) > np.linalg.norm(U_he[:2]) - 1e-6

    def test_R_per_step_vectors(self, two_out):
        """(N, nu) R form: per-input penalty asymmetry shifts solution."""
        N = 4
        nu = 2
        # penalise input-0 heavily at every step
        R_heavy0 = np.ones((N, nu))
        R_heavy0[:, 0] = 10.0
        R_light0 = np.ones((N, nu))
        R_light0[:, 0] = 0.01
        ocp_h = _ocp(two_out, Q=np.eye(2), R=R_heavy0)
        ocp_l = _ocp(two_out, Q=np.eye(2), R=R_light0)
        U_h, _ = ocp_h.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        U_l, _ = ocp_l.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert abs(U_h[0]) < abs(U_l[0]) + 1e-6

    # ── P (terminal weight) ───────────────────────────────────────────────

    def test_P_scalar(self, two_out):
        ocp = _ocp(two_out, Q=np.eye(2), R=np.eye(2), P=5.0)
        U, X = ocp.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert U.shape == (8,)

    def test_P_diagonal_vector(self, two_out):
        """(nz,) P vector is treated as a diagonal matrix."""
        ocp_v = _ocp(two_out, Q=np.eye(2), R=np.eye(2), P=np.array([3.0, 1.0]))
        ocp_m = _ocp(two_out, Q=np.eye(2), R=np.eye(2), P=np.diag([3.0, 1.0]))
        U_v, _ = ocp_v.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        U_m, _ = ocp_m.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        np.testing.assert_allclose(U_v, U_m, atol=1e-8)

    def test_P_defaults_to_last_Q_step(self, two_out):
        """When P is None, it defaults to the last step's Q matrix."""
        N = 4
        q_arr = np.zeros((N, 2))
        q_arr[:-1, :] = 1.0
        q_arr[-1, :] = 5.0   # last step has weight 5
        ocp_default = _ocp(two_out, N=N, Q=q_arr, R=np.eye(2))
        # Explicit P equal to last Q step should give same result
        ocp_explicit = _ocp(two_out, N=N, Q=q_arr, R=np.eye(2), P=np.diag(q_arr[-1]))
        U_d, _ = ocp_default.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        U_e, _ = ocp_explicit.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        np.testing.assert_allclose(U_d, U_e, atol=1e-8)

    # ── S (rate-of-movement weight) ───────────────────────────────────────

    def test_S_per_step_scalars(self, two_out):
        """Per-step (N,) S weight: heavy early ROM makes first Δu small."""
        N = 4
        ocp_h = _ocp(two_out, Q=np.eye(2), R=np.eye(2) * 0.01,
                     S=np.array([10.0, 10.0, 0.01, 0.01]))
        ocp_l = _ocp(two_out, Q=np.eye(2), R=np.eye(2) * 0.01,
                     S=np.array([0.01, 0.01, 10.0, 10.0]))
        U_h, _ = ocp_h.solve([0.0, 0.0], x_ref=[1.0, 1.0], u_prev=[0.0, 0.0])
        U_l, _ = ocp_l.solve([0.0, 0.0], x_ref=[1.0, 1.0], u_prev=[0.0, 0.0])
        assert np.linalg.norm(U_h[:2]) <= np.linalg.norm(U_l[:2]) + 1e-6

    def test_S_per_step_vectors(self, two_out):
        """(N, nu) S form: asymmetric ROM penalisation per input channel."""
        N = 4
        nu = 2
        S_heavy0 = np.ones((N, nu)) * 0.01
        S_heavy0[:, 0] = 10.0
        ocp = _ocp(two_out, Q=np.eye(2), R=np.eye(2) * 0.01, S=S_heavy0)
        U, _ = ocp.solve([0.0, 0.0], x_ref=[1.0, 1.0], u_prev=[0.0, 0.0])
        assert U.shape == (8,)

    # ── rho (quadratic slack penalty) ─────────────────────────────────────

    def test_rho_per_step_vectors(self, two_out):
        """(N, nz) rho: different penalty per output channel."""
        N = 4
        nz = 2
        # very tight band, high rho on output-0 → strong push to track output-0
        rho_arr = np.ones((N, nz))
        rho_arr[:, 0] = 1e6
        rho_arr[:, 1] = 1.0
        ocp = _ocp(two_out, Q=np.eye(2), R=np.eye(2), rho=rho_arr, z_offset=0.01)
        U, X = ocp.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert U.shape == (8,)

    def test_rho_per_step_scalars(self, two_out):
        """(N,) rho differs from scalar rho in practice."""
        N = 4
        rho_vary = np.array([1e6, 1e6, 1.0, 1.0])
        ocp_v = _ocp(two_out, Q=np.eye(2), R=np.eye(2), rho=rho_vary, z_offset=0.5)
        ocp_c = _ocp(two_out, Q=np.eye(2), R=np.eye(2), rho=1.0,      z_offset=0.5)
        U_v, _ = ocp_v.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        U_c, _ = ocp_c.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert not np.allclose(U_v, U_c)

    # ── rho_lin (linear slack penalty) ────────────────────────────────────

    def test_rho_lin_per_step_vectors(self, two_out):
        """(N, nz) rho_lin: per-output linear slack penalty."""
        N = 4
        nz = 2
        rho_lin_arr = np.ones((N, nz)) * 100.0
        ocp = _ocp(two_out, Q=np.eye(2), R=np.eye(2),
                   rho=1e4, rho_lin=rho_lin_arr, z_offset=0.5)
        U, X = ocp.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert U.shape == (8,)

    # ── z_offset (soft-output band half-width) ────────────────────────────

    def test_z_offset_per_step_scalars(self, two_out):
        """(N,) z_offset: wider band later → slack needed only in early steps."""
        N = 4
        y_tight_early = _ocp(two_out, Q=np.eye(2), R=np.eye(2),
                              rho=1e4, z_offset=np.array([0.1, 0.1, 5.0, 5.0]))
        U, X = y_tight_early.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert U.shape == (8,)

    def test_z_offset_per_step_vectors(self, two_out):
        """(N, nz) z_offset: different band per output channel."""
        N = 4
        nz = 2
        band_arr = np.ones((N, nz)) * 2.0
        band_arr[:, 0] = 0.1   # tight for output-0, wide for output-1
        ocp = _ocp(two_out, Q=np.eye(2), R=np.eye(2), rho=1e4, z_offset=band_arr)
        U, X = ocp.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert U.shape == (8,)

    def test_z_offset_scalar_profile_override(self, two_out):
        """Profile (N, nz) band overrides init-time z_offset."""
        N = 4
        nz = 2
        ocp = _ocp(two_out, Q=np.eye(2), R=np.eye(2), rho=1e4, z_offset=0.5)
        band_arr = np.ones((N, nz)) * 2.0
        ocp.set_soft_output_band_half_width_profile(band_arr)
        U, X = ocp.solve([0.0, 0.0], x_ref=[1.0, 1.0])
        assert U.shape == (8,)

    # ── Both formulations ─────────────────────────────────────────────────

    @pytest.mark.parametrize("formulation", ["condensed", "sparse"])
    def test_both_formulations_agree_on_per_step_vector_Q(self, two_out, formulation):
        """condensed and sparse give same answer with per-step vector Q."""
        N = 4
        nz = 2
        Q_arr = np.column_stack([np.linspace(1, 4, N), np.linspace(2, 5, N)])
        ocp_c = _ocp(two_out, N=N, Q=Q_arr, R=np.eye(2), formulation="condensed")
        ocp_s = _ocp(two_out, N=N, Q=Q_arr, R=np.eye(2), formulation="sparse")
        U_c, X_c = ocp_c.solve([0.5, -0.3], x_ref=[1.0, 0.5])
        U_s, X_s = ocp_s.solve([0.5, -0.3], x_ref=[1.0, 0.5])
        np.testing.assert_allclose(U_c, U_s, atol=1e-6)
        np.testing.assert_allclose(X_c, X_s, atol=1e-6)

    # ── Shape error handling ───────────────────────────────────────────────

    def test_bad_Q_shape_raises(self, two_out):
        with pytest.raises(ValueError, match="Cannot interpret weight"):
            StandardLinearDiscreteOCP(two_out, N=4, Q=np.ones((3, 2)), R=np.eye(2))

    def test_bad_rho_shape_raises(self, two_out):
        with pytest.raises(ValueError, match="Cannot interpret weight"):
            _ocp(two_out, Q=np.eye(2), R=np.eye(2), rho=np.ones((3, 2))).solve(
                [0.0, 0.0], x_ref=[1.0, 1.0]
            )


class TestPerChannelWeightScaleProfiles:
    """Req 1: output/input weight-scale profiles accept (N, nz)/(N, nu) arrays."""

    @pytest.mark.parametrize("formulation", ["condensed", "sparse"])
    def test_per_channel_Q_scale_equivalence(self, two_out, formulation):
        """(N, nz) output_tracking_weight_scale_profile equals baking scales into Q."""
        N = 4
        nz = 2
        # base Q = diag(1, 2), scales vary per step and per channel
        Q_base = np.diag([1.0, 2.0])
        scales = np.array([[1.0, 3.0], [2.0, 1.5], [0.5, 2.0], [1.0, 1.0]])

        # Manually bake scales in: Q_k_scaled = diag(sqrt(s)) @ Q @ diag(sqrt(s))
        Q_baked = [
            np.diag(np.sqrt(scales[k])) @ Q_base @ np.diag(np.sqrt(scales[k]))
            for k in range(N)
        ]
        ocp_baked = _ocp(two_out, N=N, Q=np.stack(Q_baked, axis=0).reshape(N, nz, nz)[0],
                         R=np.eye(2), formulation=formulation)
        # Build per-step (N, nz, nz) manually — use per-step Q_baked as (N, nz) diag
        Q_baked_diag = np.array([[scales[k, 0], scales[k, 1] * 2] for k in range(N)])
        ocp_baked2 = _ocp(two_out, N=N, Q=Q_baked_diag, R=np.eye(2), formulation=formulation)

        ocp_scaled = _ocp(two_out, N=N, Q=Q_base, R=np.eye(2), formulation=formulation)
        ocp_scaled.set_output_tracking_weight_scale_profile(scales)

        x0 = [0.5, -0.3]
        x_ref = [1.0, 0.5]
        U_baked, _ = ocp_baked2.solve(x0, x_ref=x_ref)
        U_scaled, _ = ocp_scaled.solve(x0, x_ref=x_ref)
        np.testing.assert_allclose(U_scaled, U_baked, atol=1e-6)

    @pytest.mark.parametrize("formulation", ["condensed", "sparse"])
    def test_scalar_scale_profile_backward_compat(self, two_out, formulation):
        """(N,) scale profile (backward-compat) and (N,1)-broadcast agree."""
        N = 4
        scales_1d = np.array([1.0, 2.0, 0.5, 3.0])
        scales_2d = np.column_stack([scales_1d, scales_1d])  # (N, nz=2) uniform

        ocp1 = _ocp(two_out, N=N, Q=np.eye(2), R=np.eye(2), formulation=formulation)
        ocp1.set_output_tracking_weight_scale_profile(scales_1d)

        ocp2 = _ocp(two_out, N=N, Q=np.eye(2), R=np.eye(2), formulation=formulation)
        ocp2.set_output_tracking_weight_scale_profile(scales_2d)

        U1, _ = ocp1.solve([0.5, -0.3], x_ref=[1.0, 0.5])
        U2, _ = ocp2.solve([0.5, -0.3], x_ref=[1.0, 0.5])
        np.testing.assert_allclose(U1, U2, atol=1e-6)

    @pytest.mark.parametrize("formulation", ["condensed", "sparse"])
    def test_per_channel_R_scale(self, two_out, formulation):
        """(N, nu) input_regularisation_weight_scale_profile: higher R scale → smaller u."""
        N = 4
        ocp_lo = _ocp(two_out, N=N, Q=np.eye(2), R=np.eye(2), formulation=formulation)
        ocp_hi = _ocp(two_out, N=N, Q=np.eye(2), R=np.eye(2), formulation=formulation)
        # Scale up input cost on channel 0 only
        r_scales = np.ones((N, 2))
        r_scales[:, 0] = 10.0
        ocp_hi.set_input_regularisation_weight_scale_profile(r_scales)

        x0 = [0.5, -0.3]
        x_ref = [1.0, 0.5]
        U_lo, _ = ocp_lo.solve(x0, x_ref=x_ref)
        U_hi, _ = ocp_hi.solve(x0, x_ref=x_ref)
        # u[0] should be smaller (in magnitude) when its cost is higher
        assert abs(U_hi[0]) < abs(U_lo[0]) + 1e-6


class TestLinearisedContinuousMPCReq2:
    """Req 2: rho_lin exposed on StandardLinearisedContinuousMPC constructor."""

    def _make_mpc(self, rho_lin=0.0):
        from mbc.control import StandardLinearisedContinuousMPC
        from tests.test_mpc import ScalarNonlinear, _DummyEstimator2

        model = ScalarNonlinear()
        estimator = _DummyEstimator2([0.0])
        return StandardLinearisedContinuousMPC(
            model=model,
            estimator=estimator,
            N=4,
            Q=1.0,
            R=0.1,
            dt=1.0,
            u_min=np.array([-2.0]),
            u_max=np.array([2.0]),
            rho_lin=rho_lin,
        )

    def test_rho_lin_param_accepted(self):
        """Constructor accepts rho_lin without error."""
        mpc = self._make_mpc(rho_lin=50.0)
        assert mpc._ocp._rho_lin == 50.0

    def test_rho_lin_zero_default(self):
        """Default rho_lin=0 leaves linear slack cost disabled."""
        mpc = self._make_mpc()
        assert mpc._ocp._rho_lin == 0.0
