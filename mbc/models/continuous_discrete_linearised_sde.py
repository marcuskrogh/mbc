"""
Linearised continuous-discrete SDE model interface.

``ContinuousDiscreteLinearisedSDE`` — extends ``ContinuousDiscreteLinearSDE``
with an explicit steady-state operating point and deviation-variable formulation:

    dδx(t) = (A δx(t) + B δu(t) + E δd(t)) dt + G dw(t),  dw(t) ~ N(0, I dt)
    δz(t)  = Cz δx(t) + Dz δu(t) + Fz δd(t)
    δym(tk) = Cm δx(tk) + Dm δu(tk) + Fm δd(tk) + v(tk),  v(tk) ~ N(0, Rm)

where the deviation variables are defined by the steady-state operating point:

    δx = x − x_s,   δu = u − u_s,   δd = d − d_s
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .continuous_discrete_linear_sde import ContinuousDiscreteLinearSDE


class ContinuousDiscreteLinearisedSDE(ContinuousDiscreteLinearSDE):
    """
    Abstract interface for a linearised continuous-discrete stochastic system.

    Extends :class:`ContinuousDiscreteLinearSDE` with an explicit steady-state
    operating point ``(x_s, u_s, d_s)``.  The matrices ``A``, ``B``, ``E``,
    ``G``, ``Cm``, ``Cz``, etc. are the Jacobians evaluated at this point
    and act on **deviation variables**:

        dδx(t) = (A δx(t) + B δu(t) + E δd(t)) dt + G dw(t),  dw(t) ~ N(0, I dt)
        δz(t)  = Cz δx(t) + Dz δu(t) + Fz δd(t)
        δym(tk) = Cm δx(tk) + Dm δu(tk) + Fm δd(tk) + v(tk),  v(tk) ~ N(0, Rm)

    where ``δx = x − x_s``, ``δu = u − u_s``, ``δd = d − d_s``.

    The inherited methods ``f``, ``sigma``, ``gm``, ``hm`` from
    :class:`ContinuousDiscreteLinearSDE` are linear and therefore already
    correct for deviation variables directly (pass ``δx, δu, δd`` to obtain
    the corresponding deviation-variable results).  When the steady-state
    satisfies the equilibrium condition ``A x_s + B u_s + E d_s = 0`` the
    inherited ``f`` is also correct for absolute variables; in general, use
    ``f_absolute`` to evaluate the drift at an absolute operating point.

    The class provides:

    * Derived steady-state outputs ``z_s`` and ``ym_s``.
    * Scalar conversion helpers (``x_deviation``, ``x_absolute``, etc.).
    * Deviation-variable dynamics (``f_deviation``, ``output_deviation``,
      ``measurement_deviation``).
    * Absolute-variable dynamics (``f_absolute``, ``output_absolute``,
      ``measurement_absolute``).
    """

    # ── Abstract steady-state operating point ─────────────────────────────

    @property
    @abstractmethod
    def x_s(self) -> np.ndarray:
        """Steady-state / operating-point state x_s ∈ ℝⁿˣ."""

    @property
    @abstractmethod
    def u_s(self) -> np.ndarray:
        """Steady-state / operating-point input u_s ∈ ℝⁿᵘ."""

    @property
    @abstractmethod
    def d_s(self) -> np.ndarray:
        """Steady-state / operating-point disturbance d_s ∈ ℝⁿᵈ."""

    # ── Derived steady-state outputs ──────────────────────────────────────

    @property
    def z_s(self) -> np.ndarray:
        """Steady-state output z_s = Cz x_s + Dz u_s + Fz d_s ∈ ℝⁿᶻ."""
        return self.Cz @ self.x_s + self.Dz @ self.u_s + self.Fz @ self.d_s

    @property
    def ym_s(self) -> np.ndarray:
        """Steady-state measurement ym_s = Cm x_s + Dm u_s + Fm d_s ∈ ℝⁿʸᵐ."""
        return self.Cm @ self.x_s + self.Dm @ self.u_s + self.Fm @ self.d_s

    # ── To-deviation conversions ──────────────────────────────────────────

    def x_deviation(self, x: np.ndarray) -> np.ndarray:
        """State deviation δx = x − x_s."""
        return x - self.x_s

    def u_deviation(self, u: np.ndarray) -> np.ndarray:
        """Input deviation δu = u − u_s."""
        return u - self.u_s

    def d_deviation(self, d: np.ndarray) -> np.ndarray:
        """Disturbance deviation δd = d − d_s."""
        return d - self.d_s

    # ── To-absolute conversions ───────────────────────────────────────────

    def x_absolute(self, dx: np.ndarray) -> np.ndarray:
        """Absolute state x = δx + x_s."""
        return dx + self.x_s

    def u_absolute(self, du: np.ndarray) -> np.ndarray:
        """Absolute input u = δu + u_s."""
        return du + self.u_s

    def d_absolute(self, dd: np.ndarray) -> np.ndarray:
        """Absolute disturbance d = δd + d_s."""
        return dd + self.d_s

    def z_absolute(self, dz: np.ndarray) -> np.ndarray:
        """Absolute output z = δz + z_s."""
        return dz + self.z_s

    def ym_absolute(self, dym: np.ndarray) -> np.ndarray:
        """Absolute measurement ym = δym + ym_s."""
        return dym + self.ym_s

    # ── Deviation-variable dynamics ───────────────────────────────────────

    def f_deviation(
        self,
        dx: np.ndarray,
        du: np.ndarray,
        dd: np.ndarray,
    ) -> np.ndarray:
        """
        Drift in deviation variables.

            f_deviation(δx, δu, δd) = A δx + B δu + E δd

        This is identical to the inherited ``f(δx, δu, δd, p, t)`` and is
        provided for clarity when working explicitly in deviation form.

        Parameters
        ----------
        dx : (nx,) deviation state δx.
        du : (nu,) deviation input δu.
        dd : (nd,) deviation disturbance δd.

        Returns
        -------
        (nx,) drift value dδx/dt.
        """
        return self.A @ dx + self.B @ du + self.E @ dd

    def output_deviation(
        self,
        dx: np.ndarray,
        du: np.ndarray,
        dd: np.ndarray,
    ) -> np.ndarray:
        """
        Noiseless output in deviation variables.

            δz = Cz δx + Dz δu + Fz δd

        Returns
        -------
        (nz,) deviation output δz.
        """
        return self.Cz @ dx + self.Dz @ du + self.Fz @ dd

    def measurement_deviation(
        self,
        dx: np.ndarray,
        du: np.ndarray,
        dd: np.ndarray,
    ) -> np.ndarray:
        """
        Noiseless measurement in deviation variables.

            δym = Cm δx + Dm δu + Fm δd

        Returns
        -------
        (nym,) deviation measurement δym.
        """
        return self.Cm @ dx + self.Dm @ du + self.Fm @ dd

    # ── Absolute-variable dynamics ────────────────────────────────────────

    def f_absolute(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        """
        Drift evaluated at absolute variables.

            f_absolute(x, u, d) = A (x − x_s) + B (u − u_s) + E (d − d_s)

        This is the correct linearised drift for absolute state trajectories
        regardless of whether the operating point is a true equilibrium.
        When ``A x_s + B u_s + E d_s = 0`` this equals ``A x + B u + E d``.

        Parameters
        ----------
        x : (nx,) absolute state.
        u : (nu,) absolute input.
        d : (nd,) absolute disturbance.

        Returns
        -------
        (nx,) drift dx/dt.
        """
        return self.f_deviation(
            self.x_deviation(x), self.u_deviation(u), self.d_deviation(d)
        )

    def output_absolute(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        """
        Noiseless output in absolute variables.

            z = z_s + Cz (x − x_s) + Dz (u − u_s) + Fz (d − d_s)

        Returns
        -------
        (nz,) absolute output z.
        """
        return self.z_absolute(
            self.output_deviation(
                self.x_deviation(x), self.u_deviation(u), self.d_deviation(d)
            )
        )

    def measurement_absolute(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        """
        Noiseless measurement in absolute variables.

            ym = ym_s + Cm (x − x_s) + Dm (u − u_s) + Fm (d − d_s)

        Returns
        -------
        (nym,) absolute measurement ym.
        """
        return self.ym_absolute(
            self.measurement_deviation(
                self.x_deviation(x), self.u_deviation(u), self.d_deviation(d)
            )
        )
