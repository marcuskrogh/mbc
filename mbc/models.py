"""
Abstract model interfaces for the mbc toolbox.

Linear discrete-time interface
-------------------------------
``LinearDiscreteModel`` — abstract base for linear discrete-time systems:

    x[k+1] = Ad x[k] + Bd u[k] + Ed d[k],   y[k] = Cm x[k]

    x ∈ ℝⁿˣ  state,  u ∈ ℝⁿᵘ  input,
    d ∈ ℝⁿᵈ  disturbance,    y ∈ ℝⁿʸᵐ  output.

    Matrices Ad, Bd, Ed and Cm are constant (LTI).

Continuous-discrete SDE interface (Ph.D. Ch. 5–6)
--------------------------------------------------
``ContinuousDiscreteModel`` — abstract base for continuous-discrete
stochastic systems:

    dx = f(x, u, d, t) dt + sigma(x, u, d, t) dw,   w ~ N(0, I dt)
    y_k = hm(x_k, u_k, d_k) + v_k,                  v_k ~ N(0, Rm)

``LinearContinuousDiscreteModel`` — extends ``ContinuousDiscreteModel`` for
linear systems where the drift, diffusion, and observation functions take
the specific forms:

    f(x, u, d, t)     = A x + B u + E d
    sigma(x, u, d, t) = G            (constant diffusion)
    hm(x, u, d)       = Cm x         (linear output; u ignored)

``ContinuousDiscreteDAEModel`` — extends ``ContinuousDiscreteModel`` with an
algebraic constraint:

    0 = l(x, z, u, d, t)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cvxopt import matrix


_H_FD: float = 1e-5  # default finite-difference step for Jacobians


class LinearDiscreteModel(ABC):
    """
    Abstract interface for a linear discrete-time system:

        x[k+1] = Ad x[k] + Bd u[k] + Ed d[k],   y[k] = Cm x[k]

    The system matrices Ad, Bd, Ed and the output matrix Cm are constant
    (LTI).  This interface is analogous to
    :class:`LinearContinuousDiscreteModel` but for discrete-time systems.

    Dimensions
    ----------
        nx   – state dimension              x ∈ ℝⁿˣ
        nu   – input dimension              u ∈ ℝⁿᵘ
        nd   – disturbance dimension        d ∈ ℝⁿᵈ
        nym  – output dimension             y ∈ ℝⁿʸᵐ  (derived: Cm.shape[0])

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
        """Output matrix Cm ∈ ℝⁿʸᵐˣⁿˣ (numpy ndarray)."""

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

    # ── Derived ───────────────────────────────────────────────────────────

    @property
    def nym(self) -> int:
        """Measurement output dimension nym = Cm.shape[0]."""
        return self.Cm.shape[0]

    # ── Deprecated aliases ────────────────────────────────────────────────

    @property
    def A_d(self) -> np.ndarray:
        """Deprecated: use ``Ad``."""
        return self.Ad

    @property
    def B_d(self) -> np.ndarray:
        """Deprecated: use ``Bd``."""
        return self.Bd

    @property
    def E_d(self) -> np.ndarray:
        """Deprecated: use ``Ed``."""
        return self.Ed

    @property
    def C(self) -> np.ndarray:
        """Deprecated: use ``Cm``."""
        return self.Cm

    @property
    def Q_d(self) -> np.ndarray:
        """Deprecated: use ``Qd``."""
        return self.Qd

    @property
    def R(self) -> np.ndarray:
        """Deprecated: use ``Rm``."""
        return self.Rm

    @property
    def ny(self) -> int:
        """Deprecated: use ``nym``."""
        return self.nym

    @property
    def n_x(self) -> int:
        """Deprecated alias for ``nx``.  Use ``nx`` instead."""
        return self.nx

    @property
    def n_u(self) -> int:
        """Deprecated alias for ``nu``.  Use ``nu`` instead."""
        return self.nu

    @property
    def n_d(self) -> int:
        """Deprecated alias for ``nd``.  Use ``nd`` instead."""
        return self.nd

    # ── Parameter-identification interface (non-abstract, overridable) ────

    def predict_offset(self, d_np: np.ndarray) -> np.ndarray:
        """
        Additive constant term for the one-step prediction:

            x_pred = A_d x + B_d u + E_d d + predict_offset(d)

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
    Abstract interface for a continuous-discrete stochastic system (Ph.D. Ch. 5).

    The system is governed by the Itô SDE

        dx = f(x, u, d, t) dt + sigma(x, u, d, t) dw,   w ~ N(0, I dt)

    with discrete-time observations

        y_k = hm(x_k, u_k, d_k) + v_k,   v_k ~ N(0, Rm)

    Subclasses must implement the drift ``f``, diffusion ``sigma``,
    measurement function ``hm``, and the noise covariance property ``Rm``.

    Dimensions
    ----------
        nx   – state dimension              x ∈ ℝⁿˣ
        nu   – input dimension              u ∈ ℝⁿᵘ
        nd   – disturbance dimension        d ∈ ℝⁿᵈ
        nym  – measurement output dimension y ∈ ℝⁿʸᵐ
        nw   – process-noise dimension      w ∈ ℝⁿʷ  (columns of sigma's output)
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
    def hm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Measurement function hm(x, u, d, p, t) → (nym,) predicted observation.
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

    # ── Deprecated non-abstract members ──────────────────────────────────

    def g(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Controlled output g(x, u, d, p, t).  Default implementation returns hm."""
        return self.hm(x, u, d, p, t)

    def h(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``hm``. Calls hm(x, u, d, p, 0.0)."""
        return self.hm(x, u, d, p, 0.0)

    @property
    def Q_c(self) -> np.ndarray:
        """Deprecated: returns np.eye(nw)."""
        return np.eye(self.nw)

    @property
    def R(self) -> np.ndarray:
        """Deprecated: use ``Rm``."""
        return self.Rm

    @property
    def ny(self) -> int:
        """Deprecated: use ``nym``."""
        return self.nym

    # ── Jacobian methods (default: forward finite differences) ────────────

    def dfdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian F = ∂f/∂x evaluated at (x, u, d, p, t)  →  (nx, nx) ndarray.

        Default implementation uses forward finite differences with step
        ``_H_FD``.  Subclasses may override with an analytic Jacobian.
        """
        f0 = self.f(x, u, d, p, t)
        nx = x.shape[0]
        J = np.empty((nx, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.f(x_fwd, u, d, p, t) - f0) / _H_FD
        return J

    def dhmdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian H = ∂hm/∂x evaluated at (x, u, d, p, t)  →  (nym, nx) ndarray.

        Default implementation uses forward finite differences with step
        ``_H_FD``.  Subclasses may override with an analytic Jacobian.
        """
        h0 = self.hm(x, u, d, p, t)
        nx = x.shape[0]
        ny = h0.shape[0]
        J = np.empty((ny, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.hm(x_fwd, u, d, p, t) - h0) / _H_FD
        return J

    def dhdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdx``. Calls dhmdx(x, u, d, p, 0.0)."""
        return self.dhmdx(x, u, d, p, 0.0)

    def dhmdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂u evaluated at (x, u, d, p, t)  →  (nym, nu) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        h0 = self.hm(x, u, d, p, t)
        nu = u.shape[0]
        ny = h0.shape[0]
        J = np.empty((ny, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, u_fwd, d, p, t) - h0) / _H_FD
        return J

    def dhdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdu``. Calls dhmdu(x, u, d, p, 0.0)."""
        return self.dhmdu(x, u, d, p, 0.0)

    def dfdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂u evaluated at (x, u, d, p, t)  →  (nx, nu) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        f0 = self.f(x, u, d, p, t)
        nu = u.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.f(x, u_fwd, d, p, t) - f0) / _H_FD
        return J

    def dfdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂d evaluated at (x, u, d, p, t)  →  (nx, nd) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        f0 = self.f(x, u, d, p, t)
        nd = d.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.f(x, u, d_fwd, p, t) - f0) / _H_FD
        return J

    def dhmdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂d evaluated at (x, u, d, p, t)  →  (nym, nd) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        h0 = self.hm(x, u, d, p, t)
        nd = d.shape[0]
        ny = h0.shape[0]
        J = np.empty((ny, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, u, d_fwd, p, t) - h0) / _H_FD
        return J

    def dhdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdd``. Calls dhmdd(x, u, d, p, 0.0)."""
        return self.dhmdd(x, u, d, p, 0.0)

    def dfdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂p evaluated at (x, u, d, p, t)  →  (nx, nparams) ndarray.

        Default: forward finite differences.  Subclasses may override.
        Returns an empty (nx, 0) array when p is empty.
        """
        nparams = p.shape[0]
        nx = self.nx
        if nparams == 0:
            return np.empty((nx, 0))
        f0 = self.f(x, u, d, p, t)
        J = np.empty((nx, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.f(x, u, d, p_fwd, t) - f0) / _H_FD
        return J

    def dhmdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂p evaluated at (x, u, d, p, t)  →  (nym, nparams) ndarray.

        Default: forward finite differences.  Subclasses may override.
        Returns an empty (nym, 0) array when p is empty.
        """
        nparams = p.shape[0]
        ny = self.nym
        if nparams == 0:
            return np.empty((ny, 0))
        h0 = self.hm(x, u, d, p, t)
        J = np.empty((ny, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, u, d, p_fwd, t) - h0) / _H_FD
        return J

    def dhdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdp``. Calls dhmdp(x, u, d, p, 0.0)."""
        return self.dhmdp(x, u, d, p, 0.0)

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

        dx = (A x[t] + B u[t] + E d[t]) dt + G dw[t],
        w[t] ~ N(0, I dt)

    with zero-order-hold (ZOH) inputs and disturbances over each sampling
    interval [t_k, t_{k+1}].  Observations are collected at the discrete
    measurement times t_k:

        y[k] = Cm x[k] + v[k],   v[k] ~ N(0, Rm)

    The model exposes the ZOH-discretised matrices via ``discretize`` so that
    the existing ``OptimalControlProblem`` (and ``CDOptimalControlProblem``) can
    be used without modification.

    Notation (M.Sc. thesis, Ch. 5)
    --------------------------------
        nx   – state dimension              x ∈ ℝⁿˣ
        nu   – input dimension              u ∈ ℝⁿᵘ
        nd   – disturbance dimension        d ∈ ℝⁿᵈ
        nym  – output dimension             y ∈ ℝⁿʸᵐ  (derived: Cm.shape[0])
        nw   – process-noise dimension      w ∈ ℝⁿʷ  (derived: G.shape[1])
        A    – continuous state matrix      A ∈ ℝⁿˣˣⁿˣ
        B    – continuous input matrix      B ∈ ℝⁿˣˣⁿᵘ
        E    – continuous disturbance mat.  E ∈ ℝⁿˣˣⁿᵈ
        G    – noise input matrix           G ∈ ℝⁿˣˣⁿʷ
        Rm   – measurement noise cov.       Rm ∈ ℝⁿʸᵐˣⁿʸᵐ
        Cm   – output matrix                Cm ∈ ℝⁿʸᵐˣⁿˣ (time-invariant)
        dt   – sampling interval

    Concrete implementations
    ------------------------
    The following abstract methods from :class:`ContinuousDiscreteModel` are
    implemented concretely:

        f(x, u, d, t)     = A x + B u + E d
        sigma(x, u, d, t) = G            (constant diffusion; arguments ignored)
        hm(x, u, d, t)    = Cm x         (linear output; u, d, t ignored for LTI)
        nym               = Cm.shape[0]
        nw                = G.shape[1]

    ZOH discretisation (``discretize``)
    -------------------------------------
    Computed via the augmented-matrix method (no matrix inverse required):

        Ad = expm(A · dt)
        [Ad | Bd | Ed] = expm([[A, B, E], [0, 0, 0], [0, 0, 0]] · dt)[:nx, :]

    Discrete process noise (``discretize_noise``)
    -----------------------------------------------
    Computed by the Van Loan (1978) method:

        Qd = ∫₀^{dt} expm(A τ) G Gᵀ expm(A τ)ᵀ dτ

    Backward-compatible aliases
    ----------------------------
    The deprecated properties ``n_x``, ``n_u``, ``n_d`` are provided as
    concrete aliases mapping to ``nx``, ``nu``, ``nd`` respectively.
    The deprecated properties ``A_c``, ``B_c``, ``E_c``, ``C``, ``R``,
    ``Q_c``, ``ny`` are provided as aliases for ``A``, ``B``, ``E``,
    ``Cm``, ``Rm``, ``np.eye(nw)``, ``nym`` respectively.

    Consumers requiring cvxopt-format matrices should use the alias
    properties ``C_cvx``, ``R_cvx``, ``x_ref_cvx``, and ``u_bounds_cvx``.
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
        """Output matrix Cm ∈ ℝⁿʸᵐˣⁿˣ (numpy ndarray)."""

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

    def f(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Drift f(x, u, d, p, t) = A x + B u + E d  (p ignored for linear)."""
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

    def g(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Controlled output g(x, u, d, p, t) = Cm x."""
        return self.Cm @ x

    def hm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Measurement hm(x, u, d, p, t) = Cm x  (u, d, p, t ignored for LTI)."""
        return self.Cm @ x

    # ── Analytic Jacobian overrides ───────────────────────────────────────

    def dfdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Analytic Jacobian ∂f/∂x = A  (arguments ignored)."""
        return self.A.copy()

    def dhmdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Analytic Jacobian ∂hm/∂x = Cm  (arguments ignored)."""
        return self.Cm.copy()

    def dhdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdx``."""
        return self.dhmdx(x, u, d, p, 0.0)

    def dhmdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Analytic Jacobian ∂hm/∂u = 0  (hm = Cm x does not depend on u)."""
        return np.zeros((self.nym, self.nu))

    def dhdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdu``."""
        return self.dhmdu(x, u, d, p, 0.0)

    def dfdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Analytic Jacobian ∂f/∂u = B  (arguments ignored)."""
        return self.B.copy()

    def dfdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Analytic Jacobian ∂f/∂d = E  (arguments ignored)."""
        return self.E.copy()

    def dhmdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Analytic Jacobian ∂hm/∂d = 0  (hm = Cm x does not depend on d)."""
        return np.zeros((self.nym, self.nd))

    def dhdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdd``."""
        return self.dhmdd(x, u, d, p, 0.0)

    def dfdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Analytic Jacobian ∂f/∂p = 0  (f = A x + B u + E d does not depend on p)."""
        return np.zeros((self.nx, p.shape[0]))

    def dhmdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Analytic Jacobian ∂hm/∂p = 0  (hm = Cm x does not depend on p)."""
        return np.zeros((self.nym, p.shape[0]))

    def dhdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdp``."""
        return self.dhmdp(x, u, d, p, 0.0)

    # ── Backward-compatible dimension aliases ─────────────────────────────

    @property
    def n_x(self) -> int:
        """Deprecated alias for ``nx``.  Use ``nx`` instead."""
        return self.nx

    @property
    def n_u(self) -> int:
        """Deprecated alias for ``nu``.  Use ``nu`` instead."""
        return self.nu

    @property
    def n_d(self) -> int:
        """Deprecated alias for ``nd``.  Use ``nd`` instead."""
        return self.nd

    @property
    def A_c(self) -> np.ndarray:
        """Deprecated: use ``A``."""
        return self.A

    @property
    def B_c(self) -> np.ndarray:
        """Deprecated: use ``B``."""
        return self.B

    @property
    def E_c(self) -> np.ndarray:
        """Deprecated: use ``E``."""
        return self.E

    @property
    def C(self) -> np.ndarray:
        """Deprecated: use ``Cm``."""
        return self.Cm

    @property
    def R(self) -> np.ndarray:
        """Deprecated: use ``Rm``."""
        return self.Rm

    @property
    def Q_c(self) -> np.ndarray:
        """Deprecated: returns np.eye(nw)."""
        return np.eye(self.nw)

    @property
    def ny(self) -> int:
        """Deprecated: use ``nym``."""
        return self.nym

    # ── cvxopt alias properties (for legacy consumers) ────────────────────

    @property
    def C_cvx(self) -> "matrix":
        """Output matrix Cm as a cvxopt dense matrix (for legacy consumers)."""
        from ._utils import _np_to_cvx
        return _np_to_cvx(self.Cm)

    @property
    def R_cvx(self) -> "matrix":
        """Measurement noise covariance Rm as a cvxopt dense matrix."""
        from ._utils import _np_to_cvx
        return _np_to_cvx(self.Rm)

    @property
    def x_ref_cvx(self) -> "matrix":
        """Reference setpoint x_ref as a cvxopt (nx, 1) column vector."""
        from ._utils import _np_to_cvx
        return _np_to_cvx(np.asarray(self.x_ref, dtype=float).reshape(-1, 1))

    @property
    def u_bounds_cvx(self) -> Tuple["matrix", "matrix"]:
        """Input bounds (u_min, u_max) as cvxopt (nu, 1) column vectors."""
        from ._utils import _np_to_cvx
        lo, hi = self.u_bounds
        return (
            _np_to_cvx(np.asarray(lo, dtype=float).reshape(-1, 1)),
            _np_to_cvx(np.asarray(hi, dtype=float).reshape(-1, 1)),
        )

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
    Abstract interface for a continuous-discrete stochastic DAE (Ph.D. Ch. 6).

    Extends ``ContinuousDiscreteModel`` with algebraic state z and constraint:

        dx = f(x, z, u, d, t) dt + g(x, z, u, d, t) dw
        0  = l(x, z, u, d, t)
        y_k = h(x_k, z_k, u_k, d_k) + v_k

    At each integration step the algebraic constraint is enforced by solving
    ``l = 0`` for z (typically via Newton iteration in the simulator).

    Subclasses must additionally implement ``l`` and ``nz``.  The observation
    function ``h`` should be overridden to accept ``(x, z, d)`` if the
    outputs depend on z; the base signature ``h(x, d)`` is retained for
    compatibility with ``ContinuousDiscreteModel``-typed interfaces.

    The ``nw`` abstract property is inherited from ``ContinuousDiscreteModel``
    and must be implemented by concrete subclasses.
    """

    @abstractmethod
    def l(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Algebraic constraint residual.

        The constraint is satisfied when ``l(x, z, u, d, p, t) = 0``.

        Parameters
        ----------
        x : (nx,) differential state vector.
        z : (nz,) algebraic state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.
        t : current time.

        Returns
        -------
        (nz,) residual vector.
        """

    @property
    @abstractmethod
    def nz(self) -> int:
        """Algebraic state dimension."""

    # ── Jacobian methods for DAE systems (default: forward FD) ───────────
    #
    # These override the base-class methods to include the algebraic state z
    # and parameter vector p in their signatures.

    def dfdx(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂x evaluated at (x, z, u, d, p, t)  →  (nx, nx) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        f0 = self.f(x, z, u, d, p, t)
        nx = x.shape[0]
        J = np.empty((nx, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.f(x_fwd, z, u, d, p, t) - f0) / _H_FD
        return J

    def dfdz(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂z evaluated at (x, z, u, d, p, t)  →  (nx, nz) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        f0 = self.f(x, z, u, d, p, t)
        nz = z.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nz))
        for k in range(nz):
            z_fwd = z.copy()
            z_fwd[k] += _H_FD
            J[:, k] = (self.f(x, z_fwd, u, d, p, t) - f0) / _H_FD
        return J

    def dhmdx(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂x at (x, z, u, d, p, t)  →  (ny, nx) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.hm(x, z, u, d, p, t)
        nx = x.shape[0]
        ny = h0.shape[0]
        J = np.empty((ny, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.hm(x_fwd, z, u, d, p, t) - h0) / _H_FD
        return J

    def dhdx(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdx``."""
        return self.dhmdx(x, z, u, d, p, 0.0)

    def dhmdz(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂z at (x, z, u, d, p, t)  →  (ny, nz) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.hm(x, z, u, d, p, t)
        nz = z.shape[0]
        ny = h0.shape[0]
        J = np.empty((ny, nz))
        for k in range(nz):
            z_fwd = z.copy()
            z_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, z_fwd, u, d, p, t) - h0) / _H_FD
        return J

    def dhdz(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdz``."""
        return self.dhmdz(x, z, u, d, p, 0.0)

    def dhmdu(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂u at (x, z, u, d, p, t)  →  (ny, nu) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.hm(x, z, u, d, p, t)
        nu = u.shape[0]
        ny = h0.shape[0]
        J = np.empty((ny, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, z, u_fwd, d, p, t) - h0) / _H_FD
        return J

    def dhdu(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdu``."""
        return self.dhmdu(x, z, u, d, p, 0.0)

    def dldx(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂l/∂x evaluated at (x, z, u, d, p, t)  →  (nz, nx) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        l0 = self.l(x, z, u, d, p, t)
        nx = x.shape[0]
        nz_out = l0.shape[0]
        J = np.empty((nz_out, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.l(x_fwd, z, u, d, p, t) - l0) / _H_FD
        return J

    def dldz(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂l/∂z evaluated at (x, z, u, d, p, t)  →  (nz, nz) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        l0 = self.l(x, z, u, d, p, t)
        nz = z.shape[0]
        nz_out = l0.shape[0]
        J = np.empty((nz_out, nz))
        for k in range(nz):
            z_fwd = z.copy()
            z_fwd[k] += _H_FD
            J[:, k] = (self.l(x, z_fwd, u, d, p, t) - l0) / _H_FD
        return J

    def dfdu(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂u evaluated at (x, z, u, d, p, t)  →  (nx, nu) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        f0 = self.f(x, z, u, d, p, t)
        nu = u.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.f(x, z, u_fwd, d, p, t) - f0) / _H_FD
        return J

    def dfdd(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂d evaluated at (x, z, u, d, p, t)  →  (nx, nd) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        f0 = self.f(x, z, u, d, p, t)
        nd = d.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.f(x, z, u, d_fwd, p, t) - f0) / _H_FD
        return J

    def dhmdd(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂d at (x, z, u, d, p, t)  →  (ny, nd) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.hm(x, z, u, d, p, t)
        nd = d.shape[0]
        ny = h0.shape[0]
        J = np.empty((ny, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, z, u, d_fwd, p, t) - h0) / _H_FD
        return J

    def dhdd(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdd``."""
        return self.dhmdd(x, z, u, d, p, 0.0)

    def dldu(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂l/∂u evaluated at (x, z, u, d, p, t)  →  (nz, nu) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        l0 = self.l(x, z, u, d, p, t)
        nu = u.shape[0]
        nz_out = l0.shape[0]
        J = np.empty((nz_out, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.l(x, z, u_fwd, d, p, t) - l0) / _H_FD
        return J

    def dldd(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂l/∂d evaluated at (x, z, u, d, p, t)  →  (nz, nd) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        l0 = self.l(x, z, u, d, p, t)
        nd = d.shape[0]
        nz_out = l0.shape[0]
        J = np.empty((nz_out, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.l(x, z, u, d_fwd, p, t) - l0) / _H_FD
        return J

    def dfdp(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂p evaluated at (x, z, u, d, p, t)  →  (nx, nparams) ndarray.

        Default: forward finite differences.  Override with analytic form.
        Returns an empty (nx, 0) array when p is empty.
        """
        nparams = p.shape[0]
        nx = self.nx
        if nparams == 0:
            return np.empty((nx, 0))
        f0 = self.f(x, z, u, d, p, t)
        J = np.empty((nx, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.f(x, z, u, d, p_fwd, t) - f0) / _H_FD
        return J

    def dhmdp(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂p at (x, z, u, d, p, t)  →  (ny, nparams) ndarray.

        Default: forward finite differences.  Override with analytic form.
        Returns an empty (ny, 0) array when p is empty.
        """
        nparams = p.shape[0]
        ny = self.nym
        if nparams == 0:
            return np.empty((ny, 0))
        h0 = self.hm(x, z, u, d, p, t)
        J = np.empty((ny, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, z, u, d, p_fwd, t) - h0) / _H_FD
        return J

    def dhdp(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Deprecated: use ``dhmdp``."""
        return self.dhmdp(x, z, u, d, p, 0.0)

    def dldp(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂l/∂p evaluated at (x, z, u, d, p, t)  →  (nz, nparams) ndarray.

        Default: forward finite differences.  Override with analytic form.
        Returns an empty (nz, 0) array when p is empty.
        """
        nparams = p.shape[0]
        nz_out = self.l(x, z, u, d, p, t).shape[0]
        if nparams == 0:
            return np.empty((nz_out, 0))
        l0 = self.l(x, z, u, d, p, t)
        J = np.empty((nz_out, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.l(x, z, u, d, p_fwd, t) - l0) / _H_FD
        return J
