"""
Discrete linear SDE model interface.

``DiscreteLinearSDE`` — abstract base for linear discrete-time stochastic systems:

    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
    z[k]   = Cz x[k] + Dz u[k] + Fz d[k]
    ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)

    x ∈ ℝⁿˣ state, u ∈ ℝⁿᵘ input, d ∈ ℝⁿᵈ disturbance,
    z ∈ ℝⁿᶻ output, ym ∈ ℝⁿʸᵐ measurement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DiscreteLinearSDE(ABC):
    """
    Abstract interface for a linear discrete-time stochastic system
    (ControlToolbox notation, discrete-time specialisation):

        x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
        z[k]   = Cz x[k] + Dz u[k] + Fz d[k]                      (output ``g^m``)
        ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)
                                                                  (measurement ``h^m``)

    The system matrices ``Ad, Bd, Ed, Gd, Cm, Cz, Dm, Dz, Fm, Fz`` are
    constant (LTI).  This interface is the discrete-time analogue of
    :class:`ContinuousDiscreteLinearSDE` and uses the same naming
    conventions (``Cz``/``Cm``/``Qd``/``Rm``/``Gd``) that the
    continuous-discrete state-estimation documents
    (ControlToolbox §SDE / §SDAE) prescribe for the linearised filter.

    Dimensions
    ----------
        nx   – state dimension              x ∈ ℝⁿˣ
        nu   – input dimension              u ∈ ℝⁿᵘ
        nd   – disturbance dimension        d ∈ ℝⁿᵈ
        nw   – process-noise dimension      w ∈ ℝⁿʷ   (derived: Gd.shape[1])
        nym  – measurement output dimension ym ∈ ℝⁿʸᵐ (derived: Cm.shape[0])
        nz   – output dimension             z ∈ ℝⁿᶻ   (derived: Cz.shape[0])

    Parameter-identification interface
    -----------------------------------
    Subclasses that support system identification may additionally implement
    ``params`` and ``with_params``.
    """

    # ── Abstract dimensions ────────────────────────────────────────────────

    @property
    @abstractmethod
    def nx(self) -> int:
        """State dimension nx."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Input dimension nu."""

    @property
    @abstractmethod
    def nd(self) -> int:
        """Disturbance dimension nd."""

    # ── Abstract discrete-time matrices (numpy) ───────────────────────────

    @property
    @abstractmethod
    def Ad(self) -> np.ndarray:
        """Discrete state-transition matrix Ad ∈ ℝⁿˣˣⁿˣ (numpy ndarray)."""

    @property
    @abstractmethod
    def Bd(self) -> np.ndarray:
        """Discrete input matrix Bd ∈ ℝⁿˣˣⁿᵘ (numpy ndarray)."""

    @property
    @abstractmethod
    def Ed(self) -> np.ndarray:
        """Discrete disturbance matrix Ed ∈ ℝⁿˣˣⁿᵈ (numpy ndarray)."""

    @property
    @abstractmethod
    def Cm(self) -> np.ndarray:
        """Measurement output matrix Cm ∈ ℝⁿʸᵐˣⁿˣ (numpy ndarray)."""

    @property
    @abstractmethod
    def Qd(self) -> np.ndarray:
        """Discrete process-noise covariance Qd ∈ ℝⁿˣˣⁿˣ (numpy ndarray)."""

    @property
    @abstractmethod
    def Rm(self) -> np.ndarray:
        """Measurement noise covariance Rm ∈ ℝⁿʸᵐˣⁿʸᵐ (numpy ndarray)."""

    # ── Derived (non-abstract, overridable) ──────────────────────────────

    @property
    def nym(self) -> int:
        """Measurement output dimension nym = Cm.shape[0]."""
        return self.Cm.shape[0]

    @property
    def Gd(self) -> np.ndarray:
        """
        Discrete noise input matrix Gd ∈ ℝⁿˣˣⁿˣ.

        Default: identity (direct additive noise on all states).
        Subclasses may override to use a different noise structure.
        """
        return np.eye(self.nx)

    @property
    def Cz(self) -> np.ndarray:
        """
        Output matrix Cz ∈ ℝⁿᶻˣⁿˣ — discrete analogue of the continuous
        output function ``g^m`` (ControlToolbox §EMPC).

        Default: Cm (same output set as measurements).
        Subclasses may override for a different output selection.
        """
        return self.Cm

    @property
    def Dz(self) -> np.ndarray:
        """
        Output feedthrough Dz ∈ ℝⁿᶻˣⁿᵘ.

        Default: zeros (no direct feedthrough from inputs to outputs).
        """
        return np.zeros((self.nz, self.nu))

    @property
    def Fz(self) -> np.ndarray:
        """
        Output disturbance feedthrough Fz ∈ ℝⁿᶻˣⁿᵈ.

        Default: zeros (no direct feedthrough from disturbances to outputs).
        """
        return np.zeros((self.nz, self.nd))

    @property
    def Dm(self) -> np.ndarray:
        """
        Measurement input feedthrough Dm ∈ ℝⁿʸᵐˣⁿᵘ.

        Default: zeros (no direct feedthrough from inputs to measurements).
        """
        return np.zeros((self.nym, self.nu))

    @property
    def Fm(self) -> np.ndarray:
        """
        Measurement disturbance feedthrough Fm ∈ ℝⁿʸᵐˣⁿᵈ.

        Default: zeros (no direct feedthrough from disturbances to measurements).
        """
        return np.zeros((self.nym, self.nd))

    @property
    def nz(self) -> int:
        """Output dimension nz = Cz.shape[0]."""
        return self.Cz.shape[0]

    @property
    def nw(self) -> int:
        """Process-noise dimension nw = Gd.shape[1]."""
        return self.Gd.shape[1]

    # ── Coordinate-transform helpers ─────────────────────────────────────
    #
    # For a non-linearised model absolute values and deviation variables are
    # the same thing, so all transforms are the identity.  DiscreteLinearisedSDE
    # overrides these to shift by the operating point.  Code that uses these
    # methods therefore works transparently with both model types:
    #
    #   δym = model.ym_dev(ym)       # absolute → deviation (pre-filter)
    #   δu  = controller.solve(δx̂)
    #   u   = model.u_abs(δu)        # deviation → absolute (post-controller)

    def x_dev(self, x: np.ndarray) -> np.ndarray:
        """Deviation state δx = x − x_s  (identity: returns x unchanged)."""
        return x

    def u_dev(self, u: np.ndarray) -> np.ndarray:
        """Deviation input δu = u − u_s  (identity: returns u unchanged)."""
        return u

    def d_dev(self, d: np.ndarray) -> np.ndarray:
        """Deviation disturbance δd = d − d_s  (identity: returns d unchanged)."""
        return d

    def z_dev(self, z: np.ndarray) -> np.ndarray:
        """Deviation output δz = z − z_s  (identity: returns z unchanged)."""
        return z

    def ym_dev(self, ym: np.ndarray) -> np.ndarray:
        """Deviation measurement δym = ym − ym_s  (identity: returns ym unchanged)."""
        return ym

    def x_abs(self, dx: np.ndarray) -> np.ndarray:
        """Absolute state x = δx + x_s  (identity: returns dx unchanged)."""
        return dx

    def u_abs(self, du: np.ndarray) -> np.ndarray:
        """Absolute input u = δu + u_s  (identity: returns du unchanged)."""
        return du

    def d_abs(self, dd: np.ndarray) -> np.ndarray:
        """Absolute disturbance d = δd + d_s  (identity: returns dd unchanged)."""
        return dd

    def z_abs(self, dz: np.ndarray) -> np.ndarray:
        """Absolute output z = δz + z_s  (identity: returns dz unchanged)."""
        return dz

    def ym_abs(self, dym: np.ndarray) -> np.ndarray:
        """Absolute measurement ym = δym + ym_s  (identity: returns dym unchanged)."""
        return dym

    # ── Overridable hooks ─────────────────────────────────────────────────

    def predict_offset(self, d_np: np.ndarray) -> np.ndarray:
        """
        Additive constant term for the one-step prediction:

            x_pred = Ad x + Bd u + Ed d + predict_offset(d)

        Default: zero vector.  Subclasses may override to add a known
        bias or affine term without modifying the system matrices.

        Parameters
        ----------
        d_np : (nd,) ndarray  — current disturbance vector.

        Returns
        -------
        offset : (nx,) ndarray
        """
        return np.zeros(self.nx)

    # ── Sampling interval (non-abstract, overridable) ─────────────────────

    @property
    def Ts(self) -> float:
        """
        Sampling interval (seconds).

        Default: raises :class:`AttributeError`.  Subclasses and factory-
        returned instances should override this property.
        """
        raise AttributeError(
            f"{type(self).__name__} does not define Ts. "
            "Override this property to specify the sampling interval."
        )

    # ── Parameter-identification interface (non-abstract, overridable) ────

    @property
    def params(self) -> np.ndarray:
        """
        Current parameter vector *θ* as a flat numpy array.

        Default: empty.  Subclasses should override to return the natural
        parameter vector for system identification.
        """
        return np.array([], dtype=float)

    def with_params(self, theta: np.ndarray) -> "DiscreteLinearSDE":
        """
        Return a **new** model instance constructed from parameter vector *θ*.

        The default implementation raises :class:`NotImplementedError`.
        Subclasses that expose ``params`` should override this method.

        Parameters
        ----------
        theta : (p,) ndarray — parameter vector (same layout as ``params``).

        Returns
        -------
        DiscreteLinearSDE
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_params."
        )
