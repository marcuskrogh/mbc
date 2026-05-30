"""
Linearised continuous-discrete SDE model interface.

``ContinuousDiscreteLinearisedSDE`` — extends ``ContinuousDiscreteLinearSDE``
with an explicit steady-state operating point and helpers to convert between
absolute and deviation variables:

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

    Subclasses must implement ``u_s`` and ``d_s``.  The steady-state state
    ``x_s`` is derived as the solution to ``A x_s + B u_s + E d_s = 0``:

        x_s = −A⁻¹ (B u_s + E d_s)

    and ``z_s``, ``ym_s`` follow from the output equations.  Subclasses may
    override ``x_s``, ``z_s``, or ``ym_s`` to supply precomputed values (e.g.
    from a nonlinear solver).

    Discretisation
    --------------
    ``discretize(Ts=None, d=None)`` matches the signature of the parent's
    ``discretize(d=None)``: both use ``self.Ts`` when no explicit sampling
    interval is given.  Pass ``Ts`` explicitly when the model does not have
    a preset sampling interval (e.g. factory-returned linearisations):

        dm = model.linearise(u_s, d_s, Ts=0.1).discretize()

    Coordinate transforms
    ---------------------
    Mirrors :class:`DiscreteLinearisedSDE`: the ``*_dev`` and ``*_abs``
    helpers shift by the operating point and allow the same closed-loop
    code to work with both linearised and non-linearised models.
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
        """Steady-state state x_s = −A⁻¹ (B u_s + E d_s)."""
        return np.linalg.solve(self.A, -(self.B @ self.u_s + self.E @ self.d_s))

    @property
    def z_s(self) -> np.ndarray:
        """Steady-state output z_s = Cz x_s + Dz u_s + Fz d_s."""
        return self.Cz @ self.x_s + self.Dz @ self.u_s + self.Fz @ self.d_s

    @property
    def ym_s(self) -> np.ndarray:
        """Steady-state measurement ym_s = Cm x_s + Dm u_s + Fm d_s."""
        return self.Cm @ self.x_s + self.Dm @ self.u_s + self.Fm @ self.d_s

    # ── Discretisation ────────────────────────────────────────────────────

    def discretize(self, d: np.ndarray | None = None) -> "DiscreteLinearisedSDE":
        """
        Return a :class:`DiscreteLinearisedSDE` obtained by ZOH-discretising
        this linearised system at ``self.Ts``.

        Identical signature to :meth:`ContinuousDiscreteLinearSDE.discretize`.
        The steady-state operating point ``(x_s, u_s, d_s, z_s, ym_s)`` is
        carried over so the returned discrete model supports the full
        coordinate-transform interface (``*_dev`` / ``*_abs``).

        Parameters
        ----------
        d : (nd,) ndarray, optional — disturbance for LPV scheduling
            (ignored for LTI models).

        Returns
        -------
        DiscreteLinearisedSDE
        """
        from .._utils import _zoh_full, _van_loan
        from ._concrete import _ConcreteDiscreteLinearisedSDE

        Ts = self.Ts  # raises AttributeError with a clear message if not set
        Ad, Bd, Ed = _zoh_full(self.A, self.B, self.E, Ts)
        Qd = _van_loan(self.A, self.G, np.eye(self.nw), Ts)
        return _ConcreteDiscreteLinearisedSDE(
            Ad=Ad, Bd=Bd, Ed=Ed,
            Cm=self.Cm, Qd=Qd, Rm=self.Rm,
            Cz=self.Cz, Dz=self.Dz, Fz=self.Fz,
            Dm=self.Dm, Fm=self.Fm,
            Ts=Ts,
            u_s=self.u_s, d_s=self.d_s,
            x_s=self.x_s, z_s=self.z_s, ym_s=self.ym_s,
        )

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
