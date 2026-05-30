"""
Linearised discrete-time SDE model interface.

``DiscreteLinearisedSDE`` — extends ``DiscreteLinearSDE`` with an explicit
steady-state operating point and helpers to convert between absolute and
deviation variables:

    δx[k+1] = Ad δx[k] + Bd δu[k] + Ed δd[k] + Gd w[k],   w[k] ~ N(0, Qd)
    δz[k]   = Cz δx[k] + Dz δu[k] + Fz δd[k]
    δym[k]  = Cm δx[k] + Dm δu[k] + Fm δd[k] + v[k],      v[k] ~ N(0, Rm)

where δx = x - x_s, δu = u - u_s, δd = d - d_s.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .discrete_linear_sde import DiscreteLinearSDE


class DiscreteLinearisedSDE(DiscreteLinearSDE):
    """
    Abstract interface for a linearised discrete-time stochastic system.

    Extends :class:`DiscreteLinearSDE` with an explicit steady-state
    operating point ``(x_s, u_s, d_s, z_s, ym_s)``.  The system matrices
    act on deviation variables:

        δx[k+1] = Ad δx[k] + Bd δu[k] + Ed δd[k] + Gd w[k],   w[k] ~ N(0, Qd)
        δz[k]   = Cz δx[k] + Dz δu[k] + Fz δd[k]
        δym[k]  = Cm δx[k] + Dm δu[k] + Fm δd[k] + v[k],      v[k] ~ N(0, Rm)

    where ``δx = x − x_s``, ``δu = u − u_s``, ``δd = d − d_s``.

    Subclasses must implement ``u_s`` and ``d_s``.  The steady-state state
    ``x_s`` is derived as the fixed point of the linear dynamics:

        x_s = (I − Ad)⁻¹ (Bd u_s + Ed d_s)

    and ``z_s``, ``ym_s`` follow from the output equations.  Subclasses may
    override ``x_s``, ``z_s``, or ``ym_s`` to supply precomputed values (e.g.
    from a nonlinear solver).

    Coordinate transforms
    ---------------------
    The ``*_dev`` and ``*_abs`` helpers support the standard closed-loop
    workflow for linearised systems:

        δym = model.ym_dev(ym)   # absolute measurement → deviation
        # ... Kalman filter / MPC in deviation space ...
        u   = model.u_abs(δu)    # optimal deviation input → absolute

    These methods are specific to linearised models and are not present on
    the base :class:`DiscreteLinearSDE`, which works only in absolute variables.
    """

    # ── Abstract steady-state operating point ─────────────────────────────

    @property
    @abstractmethod
    def u_s(self) -> np.ndarray:
        """Steady-state input u_s ∈ ℝⁿᵘ."""

    @property
    @abstractmethod
    def d_s(self) -> np.ndarray:
        """Steady-state disturbance d_s ∈ ℝⁿᵈ."""

    # ── Derived steady-state quantities ───────────────────────────────────

    @property
    def x_s(self) -> np.ndarray:
        """Steady-state state x_s = (I − Ad)⁻¹ (Bd u_s + Ed d_s)."""
        return np.linalg.solve(
            np.eye(self.nx) - self.Ad,
            self.Bd @ self.u_s + self.Ed @ self.d_s,
        )

    @property
    def z_s(self) -> np.ndarray:
        """Steady-state output z_s = Cz x_s + Dz u_s + Fz d_s."""
        return self.Cz @ self.x_s + self.Dz @ self.u_s + self.Fz @ self.d_s

    @property
    def ym_s(self) -> np.ndarray:
        """Steady-state measurement ym_s = Cm x_s + Dm u_s + Fm d_s."""
        return self.Cm @ self.x_s + self.Dm @ self.u_s + self.Fm @ self.d_s

    # ── Coordinate transforms: absolute → deviation ───────────────────────

    def x_dev(self, x: np.ndarray) -> np.ndarray:
        """Deviation state δx = x − x_s."""
        return x - self.x_s

    def u_dev(self, u: np.ndarray) -> np.ndarray:
        """Deviation input δu = u − u_s."""
        return u - self.u_s

    def d_dev(self, d: np.ndarray) -> np.ndarray:
        """Deviation disturbance δd = d − d_s."""
        return d - self.d_s

    def z_dev(self, z: np.ndarray) -> np.ndarray:
        """Deviation output δz = z − z_s."""
        return z - self.z_s

    def ym_dev(self, ym: np.ndarray) -> np.ndarray:
        """Deviation measurement δym = ym − ym_s."""
        return ym - self.ym_s

    # ── Coordinate transforms: deviation → absolute ───────────────────────

    def x_abs(self, dx: np.ndarray) -> np.ndarray:
        """Absolute state x = δx + x_s."""
        return dx + self.x_s

    def u_abs(self, du: np.ndarray) -> np.ndarray:
        """Absolute input u = δu + u_s."""
        return du + self.u_s

    def d_abs(self, dd: np.ndarray) -> np.ndarray:
        """Absolute disturbance d = δd + d_s."""
        return dd + self.d_s

    def z_abs(self, dz: np.ndarray) -> np.ndarray:
        """Absolute output z = δz + z_s."""
        return dz + self.z_s

    def ym_abs(self, dym: np.ndarray) -> np.ndarray:
        """Absolute measurement ym = δym + ym_s."""
        return dym + self.ym_s
