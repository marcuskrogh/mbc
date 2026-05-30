"""
Private concrete data classes returned by model factory methods.

These classes are not part of the public API.  They are thin matrix-stores
that implement every abstract property required by their parent abstractions,
so that ``linearise_model`` and ``discretized_model`` can return fully
usable model instances without requiring the user to define a subclass.
"""

from __future__ import annotations

import numpy as np

from .discrete_linear_sde import DiscreteLinearSDE
from .discrete_linearised_sde import DiscreteLinearisedSDE
from .continuous_discrete_linearised_sde import ContinuousDiscreteLinearisedSDE


# ── Discrete ──────────────────────────────────────────────────────────────────

class _ConcreteDiscreteLinearSDE(DiscreteLinearSDE):

    def __init__(
        self, *,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Ed: np.ndarray,
        Cm: np.ndarray,
        Qd: np.ndarray,
        Rm: np.ndarray,
        Cz: np.ndarray | None = None,
        Dz: np.ndarray | None = None,
        Fz: np.ndarray | None = None,
        Dm: np.ndarray | None = None,
        Fm: np.ndarray | None = None,
        Gd: np.ndarray | None = None,
    ) -> None:
        self._Ad = np.asarray(Ad, dtype=float)
        self._Bd = np.asarray(Bd, dtype=float)
        self._Ed = np.asarray(Ed, dtype=float)
        self._Cm = np.asarray(Cm, dtype=float)
        self._Qd = np.asarray(Qd, dtype=float)
        self._Rm = np.asarray(Rm, dtype=float)
        self._Cz = np.asarray(Cz, dtype=float) if Cz is not None else None
        self._Dz = np.asarray(Dz, dtype=float) if Dz is not None else None
        self._Fz = np.asarray(Fz, dtype=float) if Fz is not None else None
        self._Dm = np.asarray(Dm, dtype=float) if Dm is not None else None
        self._Fm = np.asarray(Fm, dtype=float) if Fm is not None else None
        self._Gd = np.asarray(Gd, dtype=float) if Gd is not None else None

    @property
    def nx(self) -> int: return self._Ad.shape[0]

    @property
    def nu(self) -> int: return self._Bd.shape[1]

    @property
    def nd(self) -> int: return self._Ed.shape[1]

    @property
    def Ad(self) -> np.ndarray: return self._Ad

    @property
    def Bd(self) -> np.ndarray: return self._Bd

    @property
    def Ed(self) -> np.ndarray: return self._Ed

    @property
    def Cm(self) -> np.ndarray: return self._Cm

    @property
    def Qd(self) -> np.ndarray: return self._Qd

    @property
    def Rm(self) -> np.ndarray: return self._Rm

    @property
    def Gd(self) -> np.ndarray:
        return self._Gd if self._Gd is not None else super().Gd

    @property
    def Cz(self) -> np.ndarray:
        return self._Cz if self._Cz is not None else super().Cz

    @property
    def Dz(self) -> np.ndarray:
        return self._Dz if self._Dz is not None else super().Dz

    @property
    def Fz(self) -> np.ndarray:
        return self._Fz if self._Fz is not None else super().Fz

    @property
    def Dm(self) -> np.ndarray:
        return self._Dm if self._Dm is not None else super().Dm

    @property
    def Fm(self) -> np.ndarray:
        return self._Fm if self._Fm is not None else super().Fm


class _ConcreteDiscreteLinearisedSDE(DiscreteLinearisedSDE):

    def __init__(
        self, *,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Ed: np.ndarray,
        Cm: np.ndarray,
        Qd: np.ndarray,
        Rm: np.ndarray,
        x_s: np.ndarray,
        u_s: np.ndarray,
        d_s: np.ndarray,
        z_s: np.ndarray,
        ym_s: np.ndarray,
        Cz: np.ndarray | None = None,
        Dz: np.ndarray | None = None,
        Fz: np.ndarray | None = None,
        Dm: np.ndarray | None = None,
        Fm: np.ndarray | None = None,
        Gd: np.ndarray | None = None,
    ) -> None:
        self._Ad = np.asarray(Ad, dtype=float)
        self._Bd = np.asarray(Bd, dtype=float)
        self._Ed = np.asarray(Ed, dtype=float)
        self._Cm = np.asarray(Cm, dtype=float)
        self._Qd = np.asarray(Qd, dtype=float)
        self._Rm = np.asarray(Rm, dtype=float)
        self._x_s = np.asarray(x_s, dtype=float)
        self._u_s = np.asarray(u_s, dtype=float)
        self._d_s = np.asarray(d_s, dtype=float)
        self._z_s = np.asarray(z_s, dtype=float)
        self._ym_s = np.asarray(ym_s, dtype=float)
        self._Cz = np.asarray(Cz, dtype=float) if Cz is not None else None
        self._Dz = np.asarray(Dz, dtype=float) if Dz is not None else None
        self._Fz = np.asarray(Fz, dtype=float) if Fz is not None else None
        self._Dm = np.asarray(Dm, dtype=float) if Dm is not None else None
        self._Fm = np.asarray(Fm, dtype=float) if Fm is not None else None
        self._Gd = np.asarray(Gd, dtype=float) if Gd is not None else None

    @property
    def nx(self) -> int: return self._Ad.shape[0]

    @property
    def nu(self) -> int: return self._Bd.shape[1]

    @property
    def nd(self) -> int: return self._Ed.shape[1]

    @property
    def Ad(self) -> np.ndarray: return self._Ad

    @property
    def Bd(self) -> np.ndarray: return self._Bd

    @property
    def Ed(self) -> np.ndarray: return self._Ed

    @property
    def Cm(self) -> np.ndarray: return self._Cm

    @property
    def Qd(self) -> np.ndarray: return self._Qd

    @property
    def Rm(self) -> np.ndarray: return self._Rm

    @property
    def x_s(self) -> np.ndarray: return self._x_s

    @property
    def u_s(self) -> np.ndarray: return self._u_s

    @property
    def d_s(self) -> np.ndarray: return self._d_s

    @property
    def z_s(self) -> np.ndarray: return self._z_s

    @property
    def ym_s(self) -> np.ndarray: return self._ym_s

    @property
    def Gd(self) -> np.ndarray:
        return self._Gd if self._Gd is not None else super().Gd

    @property
    def Cz(self) -> np.ndarray:
        return self._Cz if self._Cz is not None else super().Cz

    @property
    def Dz(self) -> np.ndarray:
        return self._Dz if self._Dz is not None else super().Dz

    @property
    def Fz(self) -> np.ndarray:
        return self._Fz if self._Fz is not None else super().Fz

    @property
    def Dm(self) -> np.ndarray:
        return self._Dm if self._Dm is not None else super().Dm

    @property
    def Fm(self) -> np.ndarray:
        return self._Fm if self._Fm is not None else super().Fm


# ── Continuous-discrete ───────────────────────────────────────────────────────

class _ConcreteContinuousDiscreteLinearisedSDE(ContinuousDiscreteLinearisedSDE):

    def __init__(
        self, *,
        A: np.ndarray,
        B: np.ndarray,
        E: np.ndarray,
        G: np.ndarray,
        Cm: np.ndarray,
        Rm: np.ndarray,
        dt: float,
        x_s: np.ndarray,
        u_s: np.ndarray,
        d_s: np.ndarray,
        z_s: np.ndarray,
        ym_s: np.ndarray,
        Cz: np.ndarray | None = None,
        Dz: np.ndarray | None = None,
        Fz: np.ndarray | None = None,
        Dm: np.ndarray | None = None,
        Fm: np.ndarray | None = None,
    ) -> None:
        self._A = np.asarray(A, dtype=float)
        self._B = np.asarray(B, dtype=float)
        self._E = np.asarray(E, dtype=float)
        self._G = np.asarray(G, dtype=float)
        self._Cm = np.asarray(Cm, dtype=float)
        self._Rm = np.asarray(Rm, dtype=float)
        self._dt = float(dt)
        self._x_s = np.asarray(x_s, dtype=float)
        self._u_s = np.asarray(u_s, dtype=float)
        self._d_s = np.asarray(d_s, dtype=float)
        self._z_s = np.asarray(z_s, dtype=float)
        self._ym_s = np.asarray(ym_s, dtype=float)
        self._Cz = np.asarray(Cz, dtype=float) if Cz is not None else None
        self._Dz = np.asarray(Dz, dtype=float) if Dz is not None else None
        self._Fz = np.asarray(Fz, dtype=float) if Fz is not None else None
        self._Dm = np.asarray(Dm, dtype=float) if Dm is not None else None
        self._Fm = np.asarray(Fm, dtype=float) if Fm is not None else None

    @property
    def nx(self) -> int: return self._A.shape[0]

    @property
    def nu(self) -> int: return self._B.shape[1]

    @property
    def nd(self) -> int: return self._E.shape[1]

    @property
    def A(self) -> np.ndarray: return self._A

    @property
    def B(self) -> np.ndarray: return self._B

    @property
    def E(self) -> np.ndarray: return self._E

    @property
    def G(self) -> np.ndarray: return self._G

    @property
    def Cm(self) -> np.ndarray: return self._Cm

    @property
    def Rm(self) -> np.ndarray: return self._Rm

    @property
    def dt(self) -> float: return self._dt

    @property
    def x_s(self) -> np.ndarray: return self._x_s

    @property
    def u_s(self) -> np.ndarray: return self._u_s

    @property
    def d_s(self) -> np.ndarray: return self._d_s

    @property
    def z_s(self) -> np.ndarray: return self._z_s

    @property
    def ym_s(self) -> np.ndarray: return self._ym_s

    @property
    def Cz(self) -> np.ndarray:
        return self._Cz if self._Cz is not None else super().Cz

    @property
    def Dz(self) -> np.ndarray:
        return self._Dz if self._Dz is not None else super().Dz

    @property
    def Fz(self) -> np.ndarray:
        return self._Fz if self._Fz is not None else super().Fz

    @property
    def Dm(self) -> np.ndarray:
        return self._Dm if self._Dm is not None else super().Dm

    @property
    def Fm(self) -> np.ndarray:
        return self._Fm if self._Fm is not None else super().Fm
