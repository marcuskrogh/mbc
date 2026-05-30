"""
Continuous-discrete SDE model interface.

``ContinuousDiscreteSDE`` — abstract base for continuous-discrete
stochastic systems (ControlToolbox §SDE):

    dx(t)   = f(x, u, d, p, t) dt + sigma(x, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
    z(t)    = gm(x, u, d, p, t)
    ym(tk)  = hm(x, u, d, p, tk) + v(tk),                         v(tk) ~ N(0, Rm)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from .._utils import _fd_jacobian


class Linearisation(NamedTuple):
    """
    System matrices of a linearisation evaluated at a steady-state operating
    point ``(x_s, u_s, d_s)``.

    The linearised dynamics are:

        dδx(t) = (A δx + B δu + E δd) dt + G dw(t),  dw(t) ~ N(0, I dt)
        δz(t)  = Cz δx + Dz δu + Fz δd
        δym(tk)= Cm δx + Dm δu + Fm δd + v(tk),       v(tk) ~ N(0, Rm)

    where δx = x − x_s, δu = u − u_s, δd = d − d_s.

    The operating-point outputs ``z_s`` and ``ym_s`` are included so the
    caller can reconstruct absolute outputs without re-evaluating the model.

    Call :meth:`discretize` to obtain a :class:`DiscreteLinearisedSDE` at a
    given sampling interval.
    """
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    G: np.ndarray
    Cm: np.ndarray
    Dm: np.ndarray
    Fm: np.ndarray
    Cz: np.ndarray
    Dz: np.ndarray
    Fz: np.ndarray
    Rm: np.ndarray
    x_s: np.ndarray
    u_s: np.ndarray
    d_s: np.ndarray
    z_s: np.ndarray
    ym_s: np.ndarray

    def discretize(self, Ts: float) -> "DiscreteLinearisedSDE":
        """
        ZOH-discretise the linearisation at sampling interval ``Ts``.

        Uses the augmented matrix exponential (ZOH) for ``(Ad, Bd, Ed)``
        and the Van Loan (1978) method for the process-noise covariance ``Qd``.

        Parameters
        ----------
        Ts : sampling interval (seconds).

        Returns
        -------
        DiscreteLinearisedSDE
        """
        from .._utils import _zoh_full, _van_loan
        from ._concrete import _ConcreteDiscreteLinearisedSDE

        Ad, Bd, Ed = _zoh_full(self.A, self.B, self.E, Ts)
        Qd = _van_loan(self.A, self.G, np.eye(self.G.shape[1]), Ts)
        return _ConcreteDiscreteLinearisedSDE(
            Ad=Ad, Bd=Bd, Ed=Ed,
            Cm=self.Cm, Qd=Qd, Rm=self.Rm,
            Cz=self.Cz, Dz=self.Dz, Fz=self.Fz,
            Dm=self.Dm, Fm=self.Fm,
            x_s=self.x_s, u_s=self.u_s, d_s=self.d_s,
            z_s=self.z_s, ym_s=self.ym_s,
        )


class ContinuousDiscreteSDE(ABC):
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

    def dgmdx(self, x, u, d, p, t) -> np.ndarray:
        """Jacobian ∂gm/∂x at (x, u, d, p, t)  →  (nz, nx) ndarray."""
        return _fd_jacobian(lambda v: self.gm(v, u, d, p, t), x)

    def dgmdu(self, x, u, d, p, t) -> np.ndarray:
        """Jacobian ∂gm/∂u at (x, u, d, p, t)  →  (nz, nu) ndarray."""
        return _fd_jacobian(lambda v: self.gm(x, v, d, p, t), u)

    def dgmdd(self, x, u, d, p, t) -> np.ndarray:
        """Jacobian ∂gm/∂d at (x, u, d, p, t)  →  (nz, nd) ndarray."""
        return _fd_jacobian(lambda v: self.gm(x, u, v, p, t), d)

    # ── Linearisation factory methods ─────────────────────────────────────

    def linearisation(
        self,
        x_s: np.ndarray,
        u_s: np.ndarray,
        d_s: np.ndarray,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> "Linearisation":
        """
        Return a :class:`Linearisation` with the system matrices evaluated at
        the operating point ``(x_s, u_s, d_s)``.

        All Jacobians are evaluated via the registered analytic or
        finite-difference methods.  The diffusion matrix ``G`` is the
        value of ``sigma`` at the operating point.

        Parameters
        ----------
        x_s : (nx,) operating-point state.
        u_s : (nu,) operating-point input.
        d_s : (nd,) operating-point disturbance.
        p   : parameter vector; defaults to ``self.params``.
        t   : evaluation time (default 0.0).

        Returns
        -------
        Linearisation
        """
        if p is None:
            p = self.params
        return Linearisation(
            A=self.dfdx(x_s, u_s, d_s, p, t),
            B=self.dfdu(x_s, u_s, d_s, p, t),
            E=self.dfdd(x_s, u_s, d_s, p, t),
            G=self.sigma(x_s, u_s, d_s, p, t),
            Cm=self.dhmdx(x_s, u_s, d_s, p, t),
            Dm=self.dhmdu(x_s, u_s, d_s, p, t),
            Fm=self.dhmdd(x_s, u_s, d_s, p, t),
            Cz=self.dgmdx(x_s, u_s, d_s, p, t),
            Dz=self.dgmdu(x_s, u_s, d_s, p, t),
            Fz=self.dgmdd(x_s, u_s, d_s, p, t),
            Rm=self.Rm,
            x_s=np.asarray(x_s),
            u_s=np.asarray(u_s),
            d_s=np.asarray(d_s),
            z_s=self.gm(x_s, u_s, d_s, p, t),
            ym_s=self.hm(x_s, u_s, d_s, p, t),
        )

    def linearise(
        self,
        x_s: np.ndarray,
        u_s: np.ndarray,
        d_s: np.ndarray,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> "ContinuousDiscreteLinearisedSDE":
        """
        Return a :class:`ContinuousDiscreteLinearisedSDE` linearised at the
        operating point ``(x_s, u_s, d_s)``.

        The sampling interval ``Ts`` is not required here; pass it to
        :meth:`ContinuousDiscreteLinearisedSDE.discretize` when you need a
        discrete-time model.

        Parameters
        ----------
        x_s : (nx,) operating-point state.
        u_s : (nu,) operating-point input.
        d_s : (nd,) operating-point disturbance.
        p   : parameter vector; defaults to ``self.params``.
        t   : evaluation time (default 0.0).

        Returns
        -------
        ContinuousDiscreteLinearisedSDE
        """
        from ._concrete import _ConcreteContinuousDiscreteLinearisedSDE

        lin = self.linearisation(x_s, u_s, d_s, p=p, t=t)
        return _ConcreteContinuousDiscreteLinearisedSDE(
            A=lin.A, B=lin.B, E=lin.E, G=lin.G,
            Cm=lin.Cm, Dm=lin.Dm, Fm=lin.Fm,
            Cz=lin.Cz, Dz=lin.Dz, Fz=lin.Fz,
            Rm=lin.Rm,
            x_s=lin.x_s, u_s=lin.u_s, d_s=lin.d_s,
            z_s=lin.z_s, ym_s=lin.ym_s,
        )

    # ── Parameters ────────────────────────────────────────────────────────

    @property
    def params(self) -> np.ndarray:
        """
        Default parameter vector θ as a flat numpy array.

        Default: empty.  Subclasses should override to return the current
        parameter vector, which callers may use as the default ``p``.
        """
        return np.array([], dtype=float)
