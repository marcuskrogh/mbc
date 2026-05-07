"""
Abstract model interfaces for the mbc toolbox.

Notation follows the ControlToolbox conventions:

* Drift            ``f``
* Diffusion        ``sigma``
* Algebraic        ``g`` (SDAE constraint, ``g(x, y, ...) = 0``)
* Measurement      ``hm`` (discrete noisy measurement function ``h^m``)
* Output           ``gm`` (continuous noiseless output ``g^m``, used in EMPC)

Linear discrete-time interface
-------------------------------
``LinearDiscreteModel`` — abstract base for linear discrete-time systems:

    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
    z[k]   = Cz x[k] + Dz u[k] + Fz d[k]
    ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)

    x ∈ ℝⁿˣ state, u ∈ ℝⁿᵘ input, d ∈ ℝⁿᵈ disturbance,
    z ∈ ℝⁿᶻ output, ym ∈ ℝⁿʸᵐ measurement.

Continuous-discrete SDE interface
---------------------------------
``ContinuousDiscreteModel`` — abstract base for continuous-discrete
stochastic systems (ControlToolbox §SDE):

    dx(t)   = f(x, u, d, p, t) dt + sigma(x, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
    z(t)    = gm(x, u, d, p, t)
    ym(tk)  = hm(x, u, d, p, tk) + v(tk),                         v(tk) ~ N(0, Rm)

``LinearContinuousDiscreteModel`` — extends ``ContinuousDiscreteModel`` for
linear systems where the drift, diffusion, output, and observation functions
take the specific forms:

    f(x, u, d, p, t)     = A x + B u + E d
    sigma(x, u, d, p, t) = G                         (constant diffusion)
    gm(x, u, d, p, t)    = Cz x + Dz u + Fz d
    hm(x, u, d, p, t)    = Cm x + Dm u + Fm d

``ContinuousDiscreteDAEModel`` — extends ``ContinuousDiscreteModel`` with an
algebraic constraint and algebraic states y (ControlToolbox §SDAE):

    dx(t)   = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
    0       = g(x, y, u, d, p, t)
    z(t)    = gm(x, y, u, d, p, t)
    ym(tk)  = hm(x, y, u, d, p, tk) + v(tk),                            v(tk) ~ N(0, Rm)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

import numpy as np

from ._utils import _fd_jacobian

if TYPE_CHECKING:
    from cvxopt import matrix


class LinearDiscreteModel(ABC):
    """
    Abstract interface for a linear discrete-time stochastic system
    (ControlToolbox notation, discrete-time specialisation):

        x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
        z[k]   = Cz x[k] + Dz u[k] + Fz d[k]                      (output ``g^m``)
        ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)
                                                                  (measurement ``h^m``)

    The system matrices ``Ad, Bd, Ed, Gd, Cm, Cz, Dm, Dz, Fm, Fz`` are
    constant (LTI).  This interface is the discrete-time analogue of
    :class:`LinearContinuousDiscreteModel` and uses the same naming
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

    # ── Abstract control-interface properties ────────────────────────────

    @property
    @abstractmethod
    def x(self) -> list[float]:
        """Current state x as a plain list of floats."""

    @x.setter
    @abstractmethod
    def x(self, val: list[float]) -> None:
        ...

    @property
    @abstractmethod
    def x_ref(self) -> np.ndarray:
        """Reference / setpoint x_ref ∈ ℝⁿˣ (numpy 1-D array, length nx)."""

    @property
    @abstractmethod
    def u_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Box constraint on inputs (u_min, u_max), each a (nu,) ndarray."""

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

    # ── Parameter-identification interface (non-abstract, overridable) ────

    def predict_offset(self, d_np: np.ndarray) -> np.ndarray:
        """
        Additive constant term for the one-step prediction:

            x_pred = Ad x + Bd u + Ed d + predict_offset(d)

        The default implementation returns a zero vector.  Subclasses that
        model a known constant disturbance or an estimated bias term should
        override this method.

        Parameters
        ----------
        d_np : (nd,) ndarray  — current disturbance vector.

        Returns
        -------
        offset : (nx,) ndarray
        """
        return np.zeros(self.nx)

    @property
    def params(self) -> np.ndarray:
        """
        Current parameter vector *θ* as a flat numpy array.

        Default: empty.  Subclasses should override to return the natural
        parameter vector for system identification.
        """
        return np.array([], dtype=float)

    def with_params(self, theta: np.ndarray) -> "LinearDiscreteModel":
        """
        Return a **new** model instance constructed from parameter vector *θ*.

        The default implementation raises :class:`NotImplementedError`.
        Subclasses that expose ``params`` should override this method.

        Parameters
        ----------
        theta : (p,) ndarray — parameter vector (same layout as ``params``).

        Returns
        -------
        LinearDiscreteModel
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_params."
        )


# ── Continuous-Discrete SDE Model ────────────────────────────────────────────


class ContinuousDiscreteModel(ABC):
    """
    Abstract interface for a continuous-discrete stochastic system
    (ControlToolbox §SDE).

    The system is governed by the Itô SDE

        dx(t)  = f(x, u, d, p, t) dt + sigma(x, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
        z(t)   = gm(x, u, d, p, t)                                  (continuous output)
        ym(tk) = hm(x, u, d, p, tk) + v(tk),                         v(tk) ~ N(0, Rm)

    Subclasses must implement the drift ``f``, diffusion ``sigma``, output
    function ``gm``, measurement function ``hm``, and the noise covariance
    property ``Rm``.

    Dimensions
    ----------
        nx   – state dimension              x ∈ ℝⁿˣ
        nu   – input dimension              u ∈ ℝⁿᵘ
        nd   – disturbance dimension        d ∈ ℝⁿᵈ
        nw   – process-noise dimension      w ∈ ℝⁿʷ  (columns of sigma's output)
        nz   – output dimension             z ∈ ℝⁿᶻ
        nym  – measurement output dimension ym ∈ ℝⁿʸᵐ
    """

    @abstractmethod
    def f(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Drift function f(x, u, d, p, t).

        Parameters
        ----------
        x : (nx,) state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.
        t : current time.

        Returns
        -------
        (nx,) drift value.
        """

    @abstractmethod
    def sigma(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Diffusion function sigma(x, u, d, p, t).

        Returns
        -------
        (nx, nw) diffusion matrix such that dx = f dt + sigma dw, w ~ N(0, I dt).
        """

    @abstractmethod
    def gm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Continuous output function gm(x, u, d, p, t)  (ControlToolbox ``g^m``).

        Parameters
        ----------
        x : (nx,) state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.
        t : current time.

        Returns
        -------
        (nz,) continuous output vector.
        """

    @abstractmethod
    def hm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Measurement function hm(x, u, d, p, t)  (ControlToolbox ``h^m``)
        → (nym,) predicted observation.
        """

    @property
    @abstractmethod
    def Rm(self) -> np.ndarray:
        """Measurement noise covariance Rm ∈ ℝⁿʸᵐˣⁿʸᵐ."""

    @property
    @abstractmethod
    def nym(self) -> int:
        """Measurement output dimension."""

    @property
    @abstractmethod
    def nz(self) -> int:
        """Output dimension."""

    @property
    @abstractmethod
    def nx(self) -> int:
        """State dimension."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Input dimension."""

    @property
    @abstractmethod
    def nd(self) -> int:
        """Disturbance dimension."""

    @property
    @abstractmethod
    def nw(self) -> int:
        """Process-noise / diffusion dimension nw (columns of sigma's output)."""

    # ── Jacobian methods ──────────────────────────────────────────────────
    #
    # All defaults delegate to :func:`mbc._utils._fd_jacobian`, which
    # implements the single forward-FD kernel used throughout the toolbox.
    # Subclasses may override any method with an analytic Jacobian.

    def dfdx(self, x, u, d, p, t) -> np.ndarray:
        """Jacobian ∂f/∂x at (x, u, d, p, t)  →  (nx, nx) ndarray."""
        return _fd_jacobian(lambda v: self.f(v, u, d, p, t), x)

    def dfdu(self, x, u, d, p, t) -> np.ndarray:
        """Jacobian ∂f/∂u at (x, u, d, p, t)  →  (nx, nu) ndarray."""
        return _fd_jacobian(lambda v: self.f(x, v, d, p, t), u)

    def dfdd(self, x, u, d, p, t) -> np.ndarray:
        """Jacobian ∂f/∂d at (x, u, d, p, t)  →  (nx, nd) ndarray."""
        return _fd_jacobian(lambda v: self.f(x, u, v, p, t), d)

    def dfdp(self, x, u, d, p, t) -> np.ndarray:
        """
        Jacobian ∂f/∂p at (x, u, d, p, t)  →  (nx, nparams) ndarray.

        Returns an empty ``(nx, 0)`` array when ``p`` is empty.
        """
        return _fd_jacobian(lambda v: self.f(x, u, d, v, t), p, m_out=self.nx)

    def dhmdx(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Jacobian ∂hm/∂x at (x, u, d, p, t)  →  (nym, nx) ndarray."""
        return _fd_jacobian(lambda v: self.hm(v, u, d, p, t), x)

    def dhmdu(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Jacobian ∂hm/∂u at (x, u, d, p, t)  →  (nym, nu) ndarray."""
        return _fd_jacobian(lambda v: self.hm(x, v, d, p, t), u)

    def dhmdd(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Jacobian ∂hm/∂d at (x, u, d, p, t)  →  (nym, nd) ndarray."""
        return _fd_jacobian(lambda v: self.hm(x, u, v, p, t), d)

    def dhmdp(self, x, u, d, p, t=0.0) -> np.ndarray:
        """
        Jacobian ∂hm/∂p at (x, u, d, p, t)  →  (nym, nparams) ndarray.

        Returns an empty ``(nym, 0)`` array when ``p`` is empty.
        """
        return _fd_jacobian(lambda v: self.hm(x, u, d, v, t), p, m_out=self.nym)

    # ── Parameters ────────────────────────────────────────────────────────

    @property
    def params(self) -> np.ndarray:
        """
        Default parameter vector θ as a flat numpy array.

        Default: empty.  Subclasses should override to return the current
        parameter vector, which callers may use as the default ``p``.
        """
        return np.array([], dtype=float)


# ── Linear Continuous-Discrete Model ─────────────────────────────────────────


class LinearContinuousDiscreteModel(ContinuousDiscreteModel):
    """
    Abstract interface for a linear continuous-discrete stochastic system.

    Extends :class:`ContinuousDiscreteModel` with the specific linear forms:

        dx(t)  = (A x(t) + B u(t) + E d(t)) dt + G dw(t),  dw(t) ~ N(0, I dt)
        z(t)   = Cz x(t) + Dz u(t) + Fz d(t)
        ym(tk) = Cm x(tk) + Dm u(tk) + Fm d(tk) + v(tk),   v(tk) ~ N(0, Rm)

    with zero-order-hold (ZOH) inputs and disturbances over each sampling
    interval [t_k, t_{k+1}].

    The model exposes the ZOH-discretised matrices via ``discretize`` and the
    exact discrete process-noise covariance via ``discretize_noise``.

    Dimensions
    ----------
        nx   – state dimension              x ∈ ℝⁿˣ
        nu   – input dimension              u ∈ ℝⁿᵘ
        nd   – disturbance dimension        d ∈ ℝⁿᵈ
        nw   – process-noise dimension      w ∈ ℝⁿʷ  (derived: G.shape[1])
        nz   – output dimension             z ∈ ℝⁿᶻ  (derived: Cz.shape[0])
        nym  – measurement output dimension ym ∈ ℝⁿʸᵐ (derived: Cm.shape[0])
        A    – continuous state matrix      A ∈ ℝⁿˣˣⁿˣ
        B    – continuous input matrix      B ∈ ℝⁿˣˣⁿᵘ
        E    – continuous disturbance mat.  E ∈ ℝⁿˣˣⁿᵈ
        G    – noise input matrix           G ∈ ℝⁿˣˣⁿʷ
        Cm   – measurement output matrix    Cm ∈ ℝⁿʸᵐˣⁿˣ
        Dm   – measurement input D-term     Dm ∈ ℝⁿʸᵐˣⁿᵘ  (default: 0)
        Fm   – measurement disturbance D    Fm ∈ ℝⁿʸᵐˣⁿᵈ  (default: 0)
        Cz   – output matrix                Cz ∈ ℝⁿᶻˣⁿˣ   (default: Cm)
        Dz   – output input D               Dz ∈ ℝⁿᶻˣⁿᵘ   (default: 0)
        Fz   – output disturbance D         Fz ∈ ℝⁿᶻˣⁿᵈ   (default: 0)
        Rm   – measurement noise cov.       Rm ∈ ℝⁿʸᵐˣⁿʸᵐ
        dt   – sampling interval

    ZOH discretisation (``discretize``)
    -------------------------------------
    Computed via the augmented-matrix method (no matrix inverse required):

        [Ad | Bd | Ed] = expm([[A, B, E], [0, 0, 0], [0, 0, 0]] · dt)[:nx, :]

    Discrete process noise (``discretize_noise``)
    -----------------------------------------------
    Computed by the Van Loan (1978) method:

        Qd = ∫₀^{dt} expm(A τ) G Gᵀ expm(A τ)ᵀ dτ
    """

    # ── Abstract dimensions (inherited from ContinuousDiscreteModel) ──────
    #   nx, nu, nd are abstract in the parent and must be implemented by
    #   concrete subclasses.  nym and nw are provided as concrete derivations
    #   from Cm and G below.

    # ── Abstract continuous-time matrices (numpy) ─────────────────────────

    @property
    @abstractmethod
    def A(self) -> np.ndarray:
        """Continuous state matrix A ∈ ℝⁿˣˣⁿˣ."""

    @property
    @abstractmethod
    def B(self) -> np.ndarray:
        """Continuous input matrix B ∈ ℝⁿˣˣⁿᵘ."""

    @property
    @abstractmethod
    def E(self) -> np.ndarray:
        """Continuous disturbance matrix E ∈ ℝⁿˣˣⁿᵈ."""

    @property
    @abstractmethod
    def G(self) -> np.ndarray:
        """Noise input matrix G ∈ ℝⁿˣˣⁿʷ."""

    # ── Abstract observation matrices (numpy) ────────────────────────────

    @property
    @abstractmethod
    def Cm(self) -> np.ndarray:
        """Measurement output matrix Cm ∈ ℝⁿʸᵐˣⁿˣ (numpy ndarray)."""

    @property
    @abstractmethod
    def Rm(self) -> np.ndarray:
        """Measurement noise covariance Rm ∈ ℝⁿʸᵐˣⁿʸᵐ (numpy ndarray)."""

    # ── Abstract sampling interval ────────────────────────────────────────

    @property
    @abstractmethod
    def dt(self) -> float:
        """Sampling interval (seconds)."""

    # ── Abstract control-interface properties ────────────────────────────

    @property
    @abstractmethod
    def x(self) -> list[float]:
        """Current state x as a plain list of floats."""

    @x.setter
    @abstractmethod
    def x(self, val: list[float]) -> None: ...

    @property
    @abstractmethod
    def x_ref(self) -> np.ndarray:
        """Reference / setpoint x_ref ∈ ℝⁿˣ (numpy 1-D array, length nx)."""

    @property
    @abstractmethod
    def u_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Box constraint on inputs (u_min, u_max), each a (nu,) ndarray."""

    # ── Concrete implementations of ContinuousDiscreteModel abstracts ─────

    @property
    def nym(self) -> int:
        """Measurement output dimension nym = Cm.shape[0]."""
        return self.Cm.shape[0]

    @property
    def nw(self) -> int:
        """Process-noise dimension nw = G.shape[1]."""
        return self.G.shape[1]

    @property
    def Cz(self) -> np.ndarray:
        """
        Output matrix Cz ∈ ℝⁿᶻˣⁿˣ.

        Default: Cm (same output set as measurements).
        Subclasses may override for a different output selection.
        """
        return self.Cm

    @property
    def Dz(self) -> np.ndarray:
        """
        Output input feedthrough Dz ∈ ℝⁿᶻˣⁿᵘ.

        Default: zeros.  Subclasses may override.
        """
        return np.zeros((self.nz, self.nu))

    @property
    def Fz(self) -> np.ndarray:
        """
        Output disturbance feedthrough Fz ∈ ℝⁿᶻˣⁿᵈ.

        Default: zeros.  Subclasses may override.
        """
        return np.zeros((self.nz, self.nd))

    @property
    def Dm(self) -> np.ndarray:
        """
        Measurement input feedthrough Dm ∈ ℝⁿʸᵐˣⁿᵘ.

        Default: zeros.  Subclasses may override.
        """
        return np.zeros((self.nym, self.nu))

    @property
    def Fm(self) -> np.ndarray:
        """
        Measurement disturbance feedthrough Fm ∈ ℝⁿʸᵐˣⁿᵈ.

        Default: zeros.  Subclasses may override.
        """
        return np.zeros((self.nym, self.nd))

    @property
    def nz(self) -> int:
        """Output dimension nz = Cz.shape[0]."""
        return self.Cz.shape[0]

    def f(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Drift f(x, u, d, p, t) = A x + B u + E d  (p, t ignored for LTI)."""
        return self.A @ x + self.B @ u + self.E @ d

    def sigma(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Diffusion sigma(x, u, d, p, t) = G  (constant; arguments ignored)."""
        return self.G

    def gm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Output gm(x, u, d, p, t) = Cz x + Dz u + Fz d."""
        return self.Cz @ x + self.Dz @ u + self.Fz @ d

    def hm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Measurement hm(x, u, d, p, t) = Cm x + Dm u + Fm d."""
        return self.Cm @ x + self.Dm @ u + self.Fm @ d

    # ── Analytic Jacobian overrides ───────────────────────────────────────

    def dfdx(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂f/∂x = A."""
        return self.A.copy()

    def dfdu(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂f/∂u = B."""
        return self.B.copy()

    def dfdd(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂f/∂d = E."""
        return self.E.copy()

    def dfdp(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂f/∂p = 0  (f = A x + B u + E d does not depend on p)."""
        return np.zeros((self.nx, p.shape[0]))

    def dhmdx(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Analytic ∂hm/∂x = Cm."""
        return self.Cm.copy()

    def dhmdu(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Analytic ∂hm/∂u = Dm."""
        return self.Dm.copy()

    def dhmdd(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Analytic ∂hm/∂d = Fm."""
        return self.Fm.copy()

    def dhmdp(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Analytic ∂hm/∂p = 0  (LTI hm does not depend on p)."""
        return np.zeros((self.nym, p.shape[0]))

    # ── Concrete discretisation methods ───────────────────────────────────

    def discretize(self, d: "matrix") -> Tuple["matrix", "matrix", "matrix"]:
        """
        ZOH-discretised matrices (Ad, Bd, Ed) as cvxopt dense matrices.

        Uses the augmented-matrix method so that no matrix inverse is
        required:

            expm([[A, B, E], [0, 0, 0], [0, 0, 0]] · dt)[:nx, :]
            = [Ad | Bd | Ed]

        The ``d`` argument is accepted for interface compatibility with
        ``OptimalControlProblem`` (LPV sub-classes may override this method
        to schedule matrices on the current disturbance); the default LTI
        implementation ignores it.

        Parameters
        ----------
        d : (nd, 1) cvxopt column  — current disturbance (ignored for LTI).

        Returns
        -------
        Ad : (nx, nx) cvxopt dense — discrete state-transition matrix.
        Bd : (nx, nu) cvxopt dense — discrete input matrix.
        Ed : (nx, nd) cvxopt dense — discrete disturbance matrix.

        Note
        ----
        Returns cvxopt.matrix for compatibility with ``OptimalControlProblem``.
        """
        from ._utils import _zoh_full, _np_to_cvx

        A_d_np, B_d_np, E_d_np = _zoh_full(self.A, self.B, self.E, self.dt)
        return _np_to_cvx(A_d_np), _np_to_cvx(B_d_np), _np_to_cvx(E_d_np)

    def discretize_noise(self) -> "matrix":
        """
        Exact discrete process-noise covariance Qd via Van Loan (1978).

        Computes

            Qd = ∫₀^{dt} expm(A τ) G Gᵀ expm(A τ)ᵀ dτ

        using the augmented 2nx×2nx matrix method.  The result is symmetric
        positive semi-definite by construction.

        Returns
        -------
        Qd : (nx, nx) cvxopt dense — discrete process-noise covariance.

        Note
        ----
        Returns cvxopt.matrix for compatibility with ``OptimalControlProblem``.
        """
        from ._utils import _van_loan, _np_to_cvx

        # dw ~ N(0, I dt), so the noise intensity is G G^T.
        # Computed via the Van Loan (1978) augmented matrix method.
        Q_d_np = _van_loan(self.A, self.G, np.eye(self.nw), self.dt)
        return _np_to_cvx(Q_d_np)

    # ── Parameter-identification interface (non-abstract, overridable) ────

    @property
    def params(self) -> np.ndarray:
        """
        Current parameter vector θ as a flat numpy array.

        Default: empty (no identifiable parameters exposed).  Subclasses
        should override to support system identification.
        """
        return np.array([], dtype=float)

# ── Continuous-Discrete SDAE Model ───────────────────────────────────────────


class ContinuousDiscreteDAEModel(ContinuousDiscreteModel):
    """
    Abstract interface for a continuous-discrete stochastic DAE
    (ControlToolbox §SDAE).

    Extends ``ContinuousDiscreteModel`` with algebraic states y and constraint:

        dx(t)  = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
        0      = g(x, y, u, d, p, t)
        z(t)   = gm(x, y, u, d, p, t)
        ym(tk) = hm(x, y, u, d, p, tk) + v(tk),                            v(tk) ~ N(0, Rm)

    At each integration step the algebraic constraint is enforced by solving
    ``g = 0`` for y (via Newton iteration in the simulator/estimator).

    Subclasses must additionally implement ``g``, ``ny``, and override
    ``gm`` and ``hm`` with the DAE-specific signature (which now depends on
    the algebraic state ``y``).  The ``nz`` abstract property (output dimension)
    is inherited from ``ContinuousDiscreteModel`` and must also be implemented
    by concrete subclasses.

    The ``nw`` abstract property is inherited from ``ContinuousDiscreteModel``
    and must be implemented by concrete subclasses.
    """

    @abstractmethod
    def g(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Algebraic constraint residual g(x, y, u, d, p, t)
        (ControlToolbox §SDAE).

        The constraint is satisfied when ``g(x, y, u, d, p, t) = 0``.

        Parameters
        ----------
        x : (nx,) differential state vector.
        y : (ny,) algebraic state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.
        t : current time.

        Returns
        -------
        (ny,) residual vector.
        """

    @property
    @abstractmethod
    def ny(self) -> int:
        """Algebraic state dimension."""

    @abstractmethod
    def gm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Continuous output function gm(x, y, u, d, p, t).

        Parameters
        ----------
        x : (nx,) differential state vector.
        y : (ny,) algebraic state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.
        t : current time.

        Returns
        -------
        (nz,) output vector.
        """

    # ── Jacobian methods for DAE systems ─────────────────────────────────
    #
    # All defaults delegate to :func:`mbc._utils._fd_jacobian`.  These
    # override the base-class methods to include the algebraic state ``y``
    # in their signatures.  Subclasses may override any method with an
    # analytic Jacobian.

    def dfdx(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂f/∂x at (x, y, u, d, p, t)  →  (nx, nx) ndarray."""
        return _fd_jacobian(lambda v: self.f(v, y, u, d, p, t), x)

    def dfdy(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂f/∂y at (x, y, u, d, p, t)  →  (nx, ny) ndarray."""
        return _fd_jacobian(lambda v: self.f(x, v, u, d, p, t), y)

    def dfdu(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂f/∂u at (x, y, u, d, p, t)  →  (nx, nu) ndarray."""
        return _fd_jacobian(lambda v: self.f(x, y, v, d, p, t), u)

    def dfdd(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂f/∂d at (x, y, u, d, p, t)  →  (nx, nd) ndarray."""
        return _fd_jacobian(lambda v: self.f(x, y, u, v, p, t), d)

    def dfdp(self, x, y, u, d, p, t) -> np.ndarray:
        """
        Jacobian ∂f/∂p at (x, y, u, d, p, t)  →  (nx, nparams) ndarray.

        Returns an empty ``(nx, 0)`` array when ``p`` is empty.
        """
        return _fd_jacobian(lambda v: self.f(x, y, u, d, v, t), p, m_out=self.nx)

    def dhmdx(self, x, y, u, d, p, t=0.0) -> np.ndarray:
        """Jacobian ∂hm/∂x at (x, y, u, d, p, t)  →  (nym, nx) ndarray."""
        return _fd_jacobian(lambda v: self.hm(v, y, u, d, p, t), x)

    def dhmdy(self, x, y, u, d, p, t=0.0) -> np.ndarray:
        """Jacobian ∂hm/∂y at (x, y, u, d, p, t)  →  (nym, ny) ndarray."""
        return _fd_jacobian(lambda v: self.hm(x, v, u, d, p, t), y)

    def dhmdu(self, x, y, u, d, p, t=0.0) -> np.ndarray:
        """Jacobian ∂hm/∂u at (x, y, u, d, p, t)  →  (nym, nu) ndarray."""
        return _fd_jacobian(lambda v: self.hm(x, y, v, d, p, t), u)

    def dhmdd(self, x, y, u, d, p, t=0.0) -> np.ndarray:
        """Jacobian ∂hm/∂d at (x, y, u, d, p, t)  →  (nym, nd) ndarray."""
        return _fd_jacobian(lambda v: self.hm(x, y, u, v, p, t), d)

    def dhmdp(self, x, y, u, d, p, t=0.0) -> np.ndarray:
        """
        Jacobian ∂hm/∂p at (x, y, u, d, p, t)  →  (nym, nparams) ndarray.

        Returns an empty ``(nym, 0)`` array when ``p`` is empty.
        """
        return _fd_jacobian(lambda v: self.hm(x, y, u, d, v, t), p, m_out=self.nym)

    def dgdx(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂g/∂x at (x, y, u, d, p, t)  →  (ny, nx) ndarray."""
        return _fd_jacobian(lambda v: self.g(v, y, u, d, p, t), x)

    def dgdy(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂g/∂y at (x, y, u, d, p, t)  →  (ny, ny) ndarray."""
        return _fd_jacobian(lambda v: self.g(x, v, u, d, p, t), y)

    def dgdu(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂g/∂u at (x, y, u, d, p, t)  →  (ny, nu) ndarray."""
        return _fd_jacobian(lambda v: self.g(x, y, v, d, p, t), u)

    def dgdd(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂g/∂d at (x, y, u, d, p, t)  →  (ny, nd) ndarray."""
        return _fd_jacobian(lambda v: self.g(x, y, u, v, p, t), d)

    def dgdp(self, x, y, u, d, p, t) -> np.ndarray:
        """
        Jacobian ∂g/∂p at (x, y, u, d, p, t)  →  (ny, nparams) ndarray.

        Returns an empty ``(ny, 0)`` array when ``p`` is empty.
        """
        return _fd_jacobian(lambda v: self.g(x, y, u, d, v, t), p, m_out=self.ny)
