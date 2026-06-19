"""Lightweight LTI plants for OCP unit tests."""

from __future__ import annotations

import numpy as np

from mbc.models import DiscreteLinearSDE, ContinuousDiscreteLinearSDE


class ScalarDiscretePlant(DiscreteLinearSDE):
    """Scalar stable discrete integrator."""

    def __init__(self) -> None:
        self._x = [0.0]

    @property
    def nx(self) -> int:
        return 1

    @property
    def nu(self) -> int:
        return 1

    @property
    def nd(self) -> int:
        return 0

    @property
    def Ad(self) -> np.ndarray:
        return np.array([[0.9]])

    @property
    def Bd(self) -> np.ndarray:
        return np.array([[0.5]])

    @property
    def Ed(self) -> np.ndarray:
        return np.zeros((1, 0))

    @property
    def Cm(self) -> np.ndarray:
        return np.eye(1)

    @property
    def Cz(self) -> np.ndarray:
        return np.eye(1)

    @property
    def Qd(self) -> np.ndarray:
        return np.eye(1) * 1e-6

    @property
    def Rm(self) -> np.ndarray:
        return np.eye(1)

    @property
    def x(self) -> list[float]:
        return self._x

    @x.setter
    def x(self, val: list[float]) -> None:
        self._x = list(val)

    @property
    def x_ref(self) -> np.ndarray:
        return np.array([0.0])

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0]), np.array([1.0])


class TwoOutputDiscretePlant(DiscreteLinearSDE):
    """2-state, 2-input, 2-output discrete plant for per-variable weight tests."""

    @property
    def nx(self) -> int:
        return 2

    @property
    def nu(self) -> int:
        return 2

    @property
    def nd(self) -> int:
        return 0

    @property
    def Ad(self) -> np.ndarray:
        return np.array([[0.9, 0.1], [0.0, 0.8]])

    @property
    def Bd(self) -> np.ndarray:
        return np.array([[0.5, 0.0], [0.0, 0.4]])

    @property
    def Ed(self) -> np.ndarray:
        return np.zeros((2, 0))

    @property
    def Cm(self) -> np.ndarray:
        return np.eye(2)

    @property
    def Cz(self) -> np.ndarray:
        return np.eye(2)

    @property
    def Qd(self) -> np.ndarray:
        return np.eye(2) * 1e-6

    @property
    def Rm(self) -> np.ndarray:
        return np.eye(2)

    @property
    def x_ref(self) -> np.ndarray:
        return np.zeros(2)

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-2.0, -2.0]), np.array([2.0, 2.0])


class ScalarCDPlant:
    """Wrapper exposing a linear CD model for ZOH OCP tests."""

    def __init__(self) -> None:
        self._inner = _ScalarCDLinear()

    @property
    def nonlinear_model(self):
        from tests.test_mpc import ScalarNonlinear
        return ScalarNonlinear()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class _ScalarCDLinear(ContinuousDiscreteLinearSDE):
    @property
    def nx(self) -> int:
        return 1

    @property
    def nu(self) -> int:
        return 1

    @property
    def nd(self) -> int:
        return 0

    @property
    def nym(self) -> int:
        return 1

    @property
    def nz(self) -> int:
        return 1

    @property
    def Ts(self) -> float:
        return 1.0

    @property
    def A(self) -> np.ndarray:
        return np.array([[-0.5]])

    @property
    def B(self) -> np.ndarray:
        return np.array([[1.0]])

    @property
    def E(self) -> np.ndarray:
        return np.zeros((1, 0))

    @property
    def Cm(self) -> np.ndarray:
        return np.eye(1)

    @property
    def Cz(self) -> np.ndarray:
        return np.eye(1)

    @property
    def G(self) -> np.ndarray:
        return np.zeros((1, 0))

    @property
    def Qc(self) -> np.ndarray:
        return np.eye(1) * 1e-6

    @property
    def Rm(self) -> np.ndarray:
        return np.eye(1)

    @property
    def x(self) -> list[float]:
        return [0.0]

    @property
    def x_ref(self) -> np.ndarray:
        return np.array([0.0])

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0]), np.array([1.0])
