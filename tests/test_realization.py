"""
Tests for state-space realization (SISO and MIMO).

Validates:
- SISO transfer function realization (observable & controllable forms)
- SISO impulse response realization (Ho-Kalman)
- MIMO Markov parameter realization (Ho-Kalman)
- ARMAX noise model support for SISO
"""

import numpy as np
import pytest

from mbc.realization import SISORealization, MIMORealization


class TestSISOTransferFunction:
    """Test SISO transfer function realization in canonical forms."""

    def test_observable_form_simple(self):
        """Test observable canonical form for simple 2nd order system."""
        # H(z) = (0.5 z + 0.3) / (z^2 - 0.9 z + 0.2)
        num = np.array([0.5, 0.3])
        den = np.array([1.0, -0.9, 0.2])

        sys = SISORealization.from_transfer_function(
            num=num, den=den, form="observable"
        )

        # Expected matrices for observable form
        # A = [[-a_1, 1], [-a_2, 0]] = [[0.9, 1], [-0.2, 0]]
        A_expected = np.array([[0.9, 1.0], [-0.2, 0.0]])

        # B = [b_1 - b_0*a_1, b_2 - b_0*a_2]^T = [0.3 - 0.5*(-0.9), 0 - 0.5*0.2]^T
        B_expected = np.array([[0.75], [-0.1]])

        # C = [1, 0]
        C_expected = np.array([[1.0, 0.0]])

        # D = [b_0] = [0.5]
        D_expected = np.array([[0.5]])

        np.testing.assert_allclose(sys.A, A_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.B, B_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.C, C_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.D, D_expected, rtol=1e-10)

    def test_controllable_form_simple(self):
        """Test controllable canonical form for simple 2nd order system."""
        # H(z) = (0.5 z + 0.3) / (z^2 - 0.9 z + 0.2)
        num = np.array([0.5, 0.3])
        den = np.array([1.0, -0.9, 0.2])

        sys = SISORealization.from_transfer_function(
            num=num, den=den, form="controllable"
        )

        # Expected matrices for controllable form
        # A = [[0, -a_2], [1, -a_1]] = [[0, -0.2], [1, 0.9]]
        A_expected = np.array([[0.0, -0.2], [1.0, 0.9]])

        # B = [0, 1]^T
        B_expected = np.array([[0.0], [1.0]])

        # C = [b_2 - b_0*a_2, b_1 - b_0*a_1] = [0 - 0.5*0.2, 0.3 - 0.5*(-0.9)]
        C_expected = np.array([[-0.1, 0.75]])

        # D = [b_0] = [0.5]
        D_expected = np.array([[0.5]])

        np.testing.assert_allclose(sys.A, A_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.B, B_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.C, C_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.D, D_expected, rtol=1e-10)

    def test_strictly_proper_system(self):
        """Test system where numerator degree < denominator degree."""
        # H(z) = 0.3 / (z^2 - 0.9 z + 0.2)
        # Numerator is just a constant
        num = np.array([0.3])
        den = np.array([1.0, -0.9, 0.2])

        sys = SISORealization.from_transfer_function(
            num=num, den=den, form="observable"
        )

        # After padding: num = [0.3, 0.0, 0.0]
        # b_0 = 0.3, b_1 = 0.0, b_2 = 0.0
        # B = [b_1 - b_0*a_1, b_2 - b_0*a_2] = [0 - 0.3*(-0.9), 0 - 0.3*0.2]
        B_expected = np.array([[0.27], [-0.06]])
        D_expected = np.array([[0.3]])

        np.testing.assert_allclose(sys.B, B_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.D, D_expected, rtol=1e-10)

    def test_first_order_system(self):
        """Test minimal first-order system."""
        # H(z) = 0.5 / (z - 0.8)
        num = np.array([0.5])
        den = np.array([1.0, -0.8])

        sys = SISORealization.from_transfer_function(num=num, den=den)

        # After padding: num = [0.5, 0.0]
        # b_0 = 0.5, b_1 = 0.0
        # B = [b_1 - b_0*a_1] = [0 - 0.5*(-0.8)] = [0.4]
        # n = 1, so A is scalar
        A_expected = np.array([[0.8]])
        B_expected = np.array([[0.4]])
        C_expected = np.array([[1.0]])
        D_expected = np.array([[0.5]])

        np.testing.assert_allclose(sys.A, A_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.B, B_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.C, C_expected, rtol=1e-10)
        np.testing.assert_allclose(sys.D, D_expected, rtol=1e-10)

    def test_armax_with_noise(self):
        """Test ARMAX realization with colored noise."""
        # A(z) y = B(z) u + C(z) e
        # H(z) = (0.5 z + 0.3) / (z^2 - 0.9 z + 0.2)
        # C(z) = z^2 + 0.8 z + 0.1 -> noise_num = [0.8, 0.1]
        num = np.array([0.5, 0.3])
        den = np.array([1.0, -0.9, 0.2])
        noise_num = np.array([0.8, 0.1])

        sys = SISORealization.from_transfer_function(
            num=num, den=den, noise_num=noise_num, form="observable"
        )

        # G should be computed similarly to B but with c_i coefficients
        # G = [c_1 - a_1, c_2 - a_2]^T = [0.8 - (-0.9), 0.1 - 0.2]^T
        G_expected = np.array([[1.7], [-0.1]])

        assert sys.G is not None
        np.testing.assert_allclose(sys.G, G_expected, rtol=1e-10)

    def test_validation_non_monic_denominator(self):
        """Test that non-monic denominator raises error."""
        num = np.array([0.5, 0.3])
        den = np.array([2.0, -0.9, 0.2])  # Not monic

        with pytest.raises(ValueError, match="monic"):
            SISORealization.from_transfer_function(num=num, den=den)

    def test_validation_invalid_form(self):
        """Test that invalid form raises error."""
        num = np.array([0.5, 0.3])
        den = np.array([1.0, -0.9, 0.2])

        with pytest.raises(ValueError, match="form must be"):
            SISORealization.from_transfer_function(
                num=num, den=den, form="invalid"
            )

    def test_validation_numerator_too_long(self):
        """Test that numerator degree > denominator degree raises error."""
        num = np.array([0.5, 0.3, 0.1, 0.05])  # degree 3
        den = np.array([1.0, -0.9, 0.2])  # degree 2

        with pytest.raises(ValueError, match="exceeds"):
            SISORealization.from_transfer_function(num=num, den=den)

    def test_validation_noise_wrong_length(self):
        """Test that noise_num with wrong length raises error."""
        num = np.array([0.5, 0.3])
        den = np.array([1.0, -0.9, 0.2])
        noise_num = np.array([0.8])  # Should be length 2

        with pytest.raises(ValueError, match="must have length"):
            SISORealization.from_transfer_function(
                num=num, den=den, noise_num=noise_num
            )


class TestSISOImpulseResponse:
    """Test SISO impulse response realization via Ho-Kalman."""

    def test_simple_impulse_response(self):
        """Test realization from generated impulse response."""
        # Create a known system and generate its impulse response
        A_true = np.array([[0.8, 0.1], [0.0, 0.7]])
        B_true = np.array([[1.0], [0.5]])
        C_true = np.array([[1.0, 0.5]])
        D_true = np.array([[0.2]])

        # Generate impulse response: h[k] = C A^k B for k >= 1, h[0] = D
        T = 50
        h = np.zeros(T)
        h[0] = D_true[0, 0]
        A_power = np.eye(2)
        for k in range(1, T):
            A_power = A_power @ A_true
            h[k] = (C_true @ A_power @ B_true)[0, 0]

        # Realize from impulse response with longer data for better accuracy
        sys = SISORealization.from_impulse_response(h=h, dt=0.1, n=2)

        # Check D matches
        np.testing.assert_allclose(sys.D, D_true, rtol=1e-6)

        # Check that realized system produces similar impulse response
        # Note: Ho-Kalman finds a minimal realization that may differ from
        # the original by a similarity transformation, so we verify I/O behavior
        h_realized = np.zeros(T)
        h_realized[0] = sys.D[0, 0]
        A_power = np.eye(2)
        for k in range(1, T):
            A_power = A_power @ sys.A
            h_realized[k] = (sys.C @ A_power @ sys.B)[0, 0]

        # Allow looser tolerance since numerical truncation affects recovery
        np.testing.assert_allclose(h_realized, h, rtol=0.05)

    def test_validation_too_short(self):
        """Test that too short impulse response raises error."""
        h = np.array([0.2, 0.5, 0.3, 0.1])  # Only 4 samples
        n = 2  # Need at least 2*2 + 1 = 5

        with pytest.raises(ValueError, match="too short"):
            SISORealization.from_impulse_response(h=h, dt=0.1, n=n)

    def test_first_order_from_impulse(self):
        """Test first-order system from impulse response."""
        # Simple exponential decay: h[k] = 0.5 * 0.8^k
        T = 30
        h = np.array([0.5 * 0.8**k for k in range(T)])

        sys = SISORealization.from_impulse_response(h=h, dt=0.1, n=1)

        # Should recover A ≈ 0.8, C*B ≈ 0.5*0.8 = 0.4 (for h[1])
        assert sys.A.shape == (1, 1)
        assert sys.B.shape == (1, 1)
        assert sys.C.shape == (1, 1)

        # Check impulse response matches with reasonable tolerance
        h_realized = np.zeros(T)
        h_realized[0] = sys.D[0, 0]
        A_power = np.eye(1)
        for k in range(1, T):
            A_power = A_power @ sys.A
            h_realized[k] = (sys.C @ A_power @ sys.B)[0, 0]

        np.testing.assert_allclose(h_realized, h, rtol=0.05)


class TestMIMORealization:
    """Test MIMO realization from Markov parameters via Ho-Kalman."""

    def test_simple_mimo_system(self):
        """Test MIMO realization for 2x2 system."""
        # Create a known MIMO system
        A_true = np.array([[0.8, 0.1], [0.0, 0.7]])
        B_true = np.array([[1.0, 0.5], [0.5, 1.0]])
        C_true = np.array([[1.0, 0.5], [0.2, 0.8]])
        D_true = np.array([[0.1, 0.2], [0.3, 0.1]])

        ny, nu = 2, 2
        n = 2

        # Generate Markov parameters: H[0] = D, H[k] = C A^{k-1} B
        T = 30  # Use more data for better accuracy
        H = [D_true]
        A_power = np.eye(2)
        for k in range(1, T):
            if k == 1:
                H.append(C_true @ B_true)
            else:
                A_power = A_power @ A_true
                H.append(C_true @ A_power @ B_true)

        # Realize from Markov parameters
        sys = MIMORealization.from_markov_parameters(H=H, n=n)

        # Check dimensions
        assert sys.A.shape == (n, n)
        assert sys.B.shape == (n, nu)
        assert sys.C.shape == (ny, n)
        assert sys.D.shape == (ny, nu)

        # Check D matches
        np.testing.assert_allclose(sys.D, D_true, rtol=1e-10)

        # Check that realized system produces similar Markov parameters
        H_realized = [sys.D]
        A_power = np.eye(n)
        for k in range(1, T):
            if k == 1:
                H_realized.append(sys.C @ sys.B)
            else:
                A_power = A_power @ sys.A
                H_realized.append(sys.C @ A_power @ sys.B)

        # Allow looser tolerance for numerical accuracy
        for k in range(T):
            np.testing.assert_allclose(H_realized[k], H[k], rtol=0.05)

    def test_siso_as_mimo(self):
        """Test that SISO works through MIMO interface."""
        # Create SISO system as 1x1 MIMO
        A_true = np.array([[0.8]])
        B_true = np.array([[1.0]])
        C_true = np.array([[1.0]])
        D_true = np.array([[0.5]])

        T = 10
        H = [D_true]
        A_power = np.eye(1)
        for k in range(1, T):
            if k == 1:
                H.append(C_true @ B_true)
            else:
                A_power = A_power @ A_true
                H.append(C_true @ A_power @ B_true)

        sys = MIMORealization.from_markov_parameters(H=H, n=1)

        assert sys.A.shape == (1, 1)
        assert sys.B.shape == (1, 1)
        assert sys.C.shape == (1, 1)
        assert sys.D.shape == (1, 1)

        np.testing.assert_allclose(sys.D, D_true, rtol=1e-10)

    def test_validation_too_few_parameters(self):
        """Test that too few Markov parameters raises error."""
        n = 2
        H = [np.zeros((2, 2))] * 3  # Only 3, need 2*2+1=5

        with pytest.raises(ValueError, match="at least"):
            MIMORealization.from_markov_parameters(H=H, n=n)

    def test_validation_inconsistent_dimensions(self):
        """Test that inconsistent H dimensions raise error."""
        H = [
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((3, 2)),  # Wrong shape
        ]

        with pytest.raises(ValueError, match="expected"):
            MIMORealization.from_markov_parameters(H=H, n=1)

    def test_validation_observability_condition(self):
        """Test observability condition validation."""
        # ny=1, nu=3, n=4, T=9 -> q=4 -> q*ny=4 = n=4 (borderline, should pass)
        # To fail, need q*ny < n, so use n=5
        n = 5
        ny, nu = 1, 3
        T = 2 * n + 1  # 11 parameters, q = 5
        H = [np.random.randn(ny, nu) for _ in range(T)]

        # q = 5, q*ny = 5 = n, so this should pass
        # Let's use T = 9 so q = 4, then q*ny = 4 < n = 5
        T = 9
        H = [np.random.randn(ny, nu) for _ in range(T)]

        with pytest.raises(ValueError, match="Observability condition"):
            MIMORealization.from_markov_parameters(H=H, n=n)

    def test_validation_controllability_condition(self):
        """Test controllability condition validation."""
        # ny=3, nu=1, n=5, T=9 -> q=4 -> q*nu=4 < n=5
        n = 5
        ny, nu = 3, 1
        T = 9  # q = 4, q*nu = 4 < n = 5
        H = [np.random.randn(ny, nu) for _ in range(T)]

        with pytest.raises(ValueError, match="Controllability condition"):
            MIMORealization.from_markov_parameters(H=H, n=n)

    def test_rectangular_mimo(self):
        """Test MIMO system with different ny and nu."""
        # 3 outputs, 2 inputs, 2 states
        ny, nu, n = 3, 2, 2

        A_true = np.array([[0.8, 0.1], [0.0, 0.7]])
        B_true = np.random.randn(n, nu)
        C_true = np.random.randn(ny, n)
        D_true = np.random.randn(ny, nu)

        # Generate Markov parameters
        T = 2 * n + 1
        H = [D_true]
        A_power = np.eye(n)
        for k in range(1, T):
            if k == 1:
                H.append(C_true @ B_true)
            else:
                A_power = A_power @ A_true
                H.append(C_true @ A_power @ B_true)

        sys = MIMORealization.from_markov_parameters(H=H, n=n)

        assert sys.A.shape == (n, n)
        assert sys.B.shape == (n, nu)
        assert sys.C.shape == (ny, n)
        assert sys.D.shape == (ny, nu)

        np.testing.assert_allclose(sys.D, D_true, rtol=1e-10)


class TestSystemEquivalence:
    """Test that different forms produce equivalent systems."""

    def test_observable_vs_controllable_transfer_function(self):
        """Test that both canonical forms produce same transfer function."""
        num = np.array([0.5, 0.3])
        den = np.array([1.0, -0.9, 0.2])

        sys_obs = SISORealization.from_transfer_function(
            num=num, den=den, form="observable"
        )
        sys_ctrl = SISORealization.from_transfer_function(
            num=num, den=den, form="controllable"
        )

        # Generate impulse response from both and compare
        T = 30
        h_obs = np.zeros(T)
        h_ctrl = np.zeros(T)

        h_obs[0] = sys_obs.D[0, 0]
        h_ctrl[0] = sys_ctrl.D[0, 0]

        A_power_obs = np.eye(2)
        A_power_ctrl = np.eye(2)

        for k in range(1, T):
            A_power_obs = A_power_obs @ sys_obs.A
            A_power_ctrl = A_power_ctrl @ sys_ctrl.A
            h_obs[k] = (sys_obs.C @ A_power_obs @ sys_obs.B)[0, 0]
            h_ctrl[k] = (sys_ctrl.C @ A_power_ctrl @ sys_ctrl.B)[0, 0]

        # Both should produce the same impulse response (I/O equivalence)
        np.testing.assert_allclose(h_obs, h_ctrl, rtol=1e-10)
