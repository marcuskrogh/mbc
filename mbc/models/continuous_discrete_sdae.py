"""
Continuous-discrete SDAE model interface.

``ContinuousDiscreteSDAE`` — standalone abstract base for continuous-discrete
stochastic differential-algebraic systems (ControlToolbox §SDAE):

    dx(t)   = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
    0       = g(x, y, u, d, p, t)
    z(t)    = gm(x, y, u, d, p, t)
    ym(tk)  = hm(x, y, u, d, p, tk) + v(tk),                            v(tk) ~ N(0, Rm)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .._utils import _fd_jacobian


class ContinuousDiscreteSDAE(ABC):
    """
    Abstract interface for a continuous-discrete stochastic DAE
    (ControlToolbox §SDAE).

    The system couples an Itô SDE for the differential states x with an
    algebraic constraint that implicitly determines the algebraic states y:

        dx(t)  = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
        0      = g(x, y, u, d, p, t)
        z(t)   = gm(x, y, u, d, p, t)                               (continuous output)
        ym(tk) = hm(x, y, u, d, p, tk) + v(tk),                      v(tk) ~ N(0, Rm)

    At each integration step the algebraic constraint is enforced by solving
    ``g = 0`` for y (via Newton iteration in the simulator/estimator).

    Subclasses must implement all abstract methods and dimension properties.
    Default Jacobian implementations use forward finite differences; subclasses
    may override any Jacobian with an analytic form.

    Dimensions
    ----------
        nx   – differential state dimension   x ∈ ℝⁿˣ
        ny   – algebraic state dimension      y ∈ ℝⁿʸ
        nu   – input dimension                u ∈ ℝⁿᵘ
        nd   – disturbance dimension          d ∈ ℝⁿᵈ
        nw   – process-noise dimension        w ∈ ℝⁿʷ  (columns of sigma's output)
        nz   – output dimension               z ∈ ℝⁿᶻ
        nym  – measurement output dimension   ym ∈ ℝⁿʸᵐ
    """

    # ── Abstract model functions ──────────────────────────────────────────

    @abstractmethod
    def f(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Drift function f(x, y, u, d, p, t).

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
        (nx,) drift value.
        """

    @abstractmethod
    def sigma(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Diffusion function sigma(x, y, u, d, p, t).

        Returns
        -------
        (nx, nw) diffusion matrix such that dx = f dt + sigma dw, w ~ N(0, I dt).
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
        Algebraic constraint residual g(x, y, u, d, p, t).

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
        Continuous output function gm(x, y, u, d, p, t)  (ControlToolbox ``g^m``).

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
        (nz,) continuous output vector.
        """

    @abstractmethod
    def hm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Measurement function hm(x, y, u, d, p, t)  (ControlToolbox ``h^m``).

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
        (nym,) predicted observation vector.
        """

    # ── Abstract properties ───────────────────────────────────────────────

    @property
    @abstractmethod
    def Rm(self) -> np.ndarray:
        """Measurement noise covariance Rm ∈ ℝⁿʸᵐˣⁿʸᵐ."""

    @property
    @abstractmethod
    def nx(self) -> int:
        """Differential state dimension."""

    @property
    @abstractmethod
    def ny(self) -> int:
        """Algebraic state dimension."""

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

    @property
    @abstractmethod
    def nz(self) -> int:
        """Output dimension."""

    @property
    @abstractmethod
    def nym(self) -> int:
        """Measurement output dimension."""

    # ── Jacobian methods ──────────────────────────────────────────────────
    #
    # All defaults delegate to :func:`mbc._utils._fd_jacobian`.
    # Subclasses may override any method with an analytic Jacobian.

    # ── Drift Jacobians ───────────────────────────────────────────────────

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

    # ── Algebraic constraint Jacobians ────────────────────────────────────

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

    # ── Measurement Jacobians ─────────────────────────────────────────────

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

    # ── Continuous output Jacobians ───────────────────────────────────────

    def dgmdx(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂gm/∂x at (x, y, u, d, p, t)  →  (nz, nx) ndarray."""
        return _fd_jacobian(lambda v: self.gm(v, y, u, d, p, t), x)

    def dgmdy(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂gm/∂y at (x, y, u, d, p, t)  →  (nz, ny) ndarray."""
        return _fd_jacobian(lambda v: self.gm(x, v, u, d, p, t), y)

    def dgmdu(self, x, y, u, d, p, t) -> np.ndarray:
        """Jacobian ∂gm/∂u at (x, y, u, d, p, t)  →  (nz, nu) ndarray."""
        return _fd_jacobian(lambda v: self.gm(x, y, v, d, p, t), u)

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

    # ── Parameters ────────────────────────────────────────────────────────

    @property
    def params(self) -> np.ndarray:
        """
        Default parameter vector θ as a flat numpy array.

        Default: empty.  Subclasses should override to return the current
        parameter vector, which callers may use as the default ``p``.
        """
        return np.array([], dtype=float)
