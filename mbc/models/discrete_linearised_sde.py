"""
Linearised discrete-time SDE model interface.

``DiscreteLinearisedSDE`` — extends ``DiscreteLinearSDE`` with an explicit
steady-state operating point and deviation-variable formulation:

    δx[k+1] = Ad δx[k] + Bd δu[k] + Ed δd[k] + Gd w[k],   w[k] ~ N(0, Qd)
    δz[k]   = Cz δx[k] + Dz δu[k] + Fz δd[k]
    δym[k]  = Cm δx[k] + Dm δu[k] + Fm δd[k] + v[k],      v[k] ~ N(0, Rm)

where the deviation variables are defined by the steady-state operating point:

    δx = x - x_s,   δu = u - u_s,   δd = d - d_s
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .discrete_linear_sde import DiscreteLinearSDE


class DiscreteLinearisedSDE(DiscreteLinearSDE):
    """
    Abstract interface for a linearised discrete-time stochastic system.

    Extends :class:`DiscreteLinearSDE` with an explicit steady-state
    operating point ``(x_s, u_s, d_s)``.  The matrices ``Ad``, ``Bd``,
    ``Ed``, ``Cm``, ``Cz``, etc. are the Jacobians evaluated at this point
    and act on **deviation variables**:

        δx[k+1] = Ad δx[k] + Bd δu[k] + Ed δd[k] + Gd w[k],   w[k] ~ N(0, Qd)
        δz[k]   = Cz δx[k] + Dz δu[k] + Fz δd[k]
        δym[k]  = Cm δx[k] + Dm δu[k] + Fm δd[k] + v[k],      v[k] ~ N(0, Rm)

    where ``δx = x − x_s``, ``δu = u − u_s``, ``δd = d − d_s``.

    The class provides:

    * Derived steady-state outputs ``z_s`` and ``ym_s``.
    * Scalar conversion helpers (``x_deviation``, ``x_absolute``, etc.).
    * Deviation-variable dynamics (``predict_deviation``, ``output_deviation``,
      ``measurement_deviation``).
    * Absolute-variable dynamics (``predict_absolute``, ``output_absolute``,
      ``measurement_absolute``).
    * An override of ``predict_offset`` so that the existing
      :class:`~mbc.estimation.KalmanFilter` can be driven with absolute
      states and inputs without modification.
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

    def predict_deviation(
        self,
        dx: np.ndarray,
        du: np.ndarray,
        dd: np.ndarray,
    ) -> np.ndarray:
        """
        One-step prediction in deviation variables.

            δx[k+1] = Ad δx[k] + Bd δu[k] + Ed δd[k]

        Parameters
        ----------
        dx : (nx,) deviation state δx[k].
        du : (nu,) deviation input δu[k].
        dd : (nd,) deviation disturbance δd[k].

        Returns
        -------
        (nx,) deviation state δx[k+1] (noiseless).
        """
        return self.Ad @ dx + self.Bd @ du + self.Ed @ dd

    def output_deviation(
        self,
        dx: np.ndarray,
        du: np.ndarray,
        dd: np.ndarray,
    ) -> np.ndarray:
        """
        Noiseless output in deviation variables.

            δz[k] = Cz δx[k] + Dz δu[k] + Fz δd[k]

        Returns
        -------
        (nz,) deviation output δz[k].
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

            δym[k] = Cm δx[k] + Dm δu[k] + Fm δd[k]

        Returns
        -------
        (nym,) deviation measurement δym[k].
        """
        return self.Cm @ dx + self.Dm @ du + self.Fm @ dd

    # ── Absolute-variable dynamics ────────────────────────────────────────

    def predict_absolute(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        """
        One-step prediction returning the absolute next state.

            x[k+1] = x_s + Ad (x[k] − x_s) + Bd (u[k] − u_s) + Ed (d[k] − d_s)

        Parameters
        ----------
        x : (nx,) absolute state x[k].
        u : (nu,) absolute input u[k].
        d : (nd,) absolute disturbance d[k].

        Returns
        -------
        (nx,) absolute next state x[k+1] (noiseless).
        """
        return self.x_absolute(
            self.predict_deviation(
                self.x_deviation(x), self.u_deviation(u), self.d_deviation(d)
            )
        )

    def output_absolute(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        """
        Noiseless output in absolute variables.

            z[k] = z_s + Cz (x[k] − x_s) + Dz (u[k] − u_s) + Fz (d[k] − d_s)

        Returns
        -------
        (nz,) absolute output z[k].
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

            ym[k] = ym_s + Cm (x[k] − x_s) + Dm (u[k] − u_s) + Fm (d[k] − d_s)

        Returns
        -------
        (nym,) absolute measurement ym[k].
        """
        return self.ym_absolute(
            self.measurement_deviation(
                self.x_deviation(x), self.u_deviation(u), self.d_deviation(d)
            )
        )

    # ── Kalman-filter compatibility override ──────────────────────────────

    def predict_offset(self, d_np: np.ndarray) -> np.ndarray:
        """
        Constant additive offset that makes the standard linear predictor

            Ad x + Bd u + Ed d + predict_offset(d)

        equivalent to the absolute-variable prediction

            x_s + Ad (x − x_s) + Bd (u − u_s) + Ed (d − d_s)

        when called with absolute states and inputs.  This allows
        :class:`~mbc.estimation.KalmanFilter` to operate directly on
        absolute states without modification.

        Returns
        -------
        (nx,) offset = (I − Ad) x_s − Bd u_s − Ed d_s.
        """
        return (np.eye(self.nx) - self.Ad) @ self.x_s - self.Bd @ self.u_s - self.Ed @ self.d_s
