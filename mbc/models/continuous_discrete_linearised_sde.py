"""
Linearised continuous-discrete SDE model interface.

``ContinuousDiscreteLinearisedSDE`` — extends ``ContinuousDiscreteLinearSDE``
with an explicit steady-state operating point and helpers to convert
deviation-variable results back to absolute values:

    dδx(t) = (A δx(t) + B δu(t) + E δd(t)) dt + G dw(t),  dw(t) ~ N(0, I dt)
    δz(t)  = Cz δx(t) + Dz δu(t) + Fz δd(t)
    δym(tk) = Cm δx(tk) + Dm δu(tk) + Fm δd(tk) + v(tk),  v(tk) ~ N(0, Rm)

where δx = x − x_s, δu = u − u_s, δd = d − d_s.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .continuous_discrete_linear_sde import ContinuousDiscreteLinearSDE


class ContinuousDiscreteLinearisedSDE(ContinuousDiscreteLinearSDE):
    """
    Abstract interface for a linearised continuous-discrete stochastic system.

    Extends :class:`ContinuousDiscreteLinearSDE` with an explicit steady-state
    operating point ``(x_s, u_s, d_s, z_s, ym_s)``.  The system matrices act
    on deviation variables:

        dδx(t) = (A δx(t) + B δu(t) + E δd(t)) dt + G dw(t),  dw(t) ~ N(0, I dt)
        δz(t)  = Cz δx(t) + Dz δu(t) + Fz δd(t)
        δym(tk) = Cm δx(tk) + Dm δu(tk) + Fm δd(tk) + v(tk),  v(tk) ~ N(0, Rm)

    where ``δx = x − x_s``, ``δu = u − u_s``, ``δd = d − d_s``.

    Subclasses must implement the steady-state properties ``x_s``, ``u_s``,
    ``d_s``, ``z_s``, and ``ym_s``.  The helpers ``x_abs``, ``u_abs``,
    ``d_abs``, ``z_abs``, and ``ym_abs`` convert deviation-variable results
    back to absolute values.
    """

    # ── Abstract steady-state operating point ─────────────────────────────

    @property
    @abstractmethod
    def x_s(self) -> np.ndarray:
        """Steady-state state x_s ∈ ℝⁿˣ."""

    @property
    @abstractmethod
    def u_s(self) -> np.ndarray:
        """Steady-state input u_s ∈ ℝⁿᵘ."""

    @property
    @abstractmethod
    def d_s(self) -> np.ndarray:
        """Steady-state disturbance d_s ∈ ℝⁿᵈ."""

    @property
    @abstractmethod
    def z_s(self) -> np.ndarray:
        """Steady-state output z_s ∈ ℝⁿᶻ."""

    @property
    @abstractmethod
    def ym_s(self) -> np.ndarray:
        """Steady-state measurement ym_s ∈ ℝⁿʸᵐ."""

    # ── Discretisation ────────────────────────────────────────────────────

    def discretized_model(self, d: np.ndarray | None = None) -> "DiscreteLinearisedSDE":
        """
        Return a :class:`DiscreteLinearisedSDE` obtained by ZOH-discretising
        this linearised continuous-discrete system.

        The steady-state operating point ``(x_s, u_s, d_s, z_s, ym_s)`` is
        carried over to the returned model so the caller can recover absolute
        values from deviation-variable Kalman-filter estimates.

        Parameters
        ----------
        d : (nd,) ndarray, optional  — passed to :meth:`discretize` for
            LPV scheduling (ignored for LTI models).

        Returns
        -------
        DiscreteLinearisedSDE
        """
        from ._concrete import _ConcreteDiscreteLinearisedSDE

        Ad, Bd, Ed = self.discretize(d)
        Qd = self.discretize_noise()
        return _ConcreteDiscreteLinearisedSDE(
            Ad=Ad, Bd=Bd, Ed=Ed,
            Cm=self.Cm, Qd=Qd, Rm=self.Rm,
            Cz=self.Cz, Dz=self.Dz, Fz=self.Fz,
            Dm=self.Dm, Fm=self.Fm,
            x_s=self.x_s, u_s=self.u_s, d_s=self.d_s,
            z_s=self.z_s, ym_s=self.ym_s,
        )

    # ── Absolute-value getters ────────────────────────────────────────────

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
