"""
Model Predictive Controller for linear discrete-time systems.

Abstract :class:`LinearDiscreteMPC` defines the closed-loop interface;
:class:`StandardLinearDiscreteMPC` composes
:class:`~mbc.estimation.DiscreteLinearKF` with
:class:`~mbc.control.StandardLinearDiscreteOCP`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from .._utils import _any_to_np1d
from ..estimation import DiscreteLinearKF
from .discrete_linear_ocp import StandardLinearDiscreteOCP, _shift_warm_start
from .forecast_ocp import solve_forecast_qp
from .mpc_horizon import HorizonProfileMPC

if TYPE_CHECKING:
    from ..models import DiscreteLinearSDE


class LinearDiscreteMPC(ABC):
    """Abstract MPC for linear discrete-time plant + estimator + discrete OCP."""

    @abstractmethod
    def step(
        self,
        ym: Any,
        D: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute one closed-loop MPC step."""


class StandardLinearDiscreteMPC(HorizonProfileMPC, LinearDiscreteMPC):
    """
    Standard MPC for a linear discrete-time plant.

    Combines :class:`~mbc.estimation.DiscreteLinearKF` and
    :class:`StandardLinearDiscreteOCP`.  Horizon-varying quantities are
    configured via the profile setters before :meth:`step`.
    """

    def __init__(
        self,
        model: "DiscreteLinearSDE",
        estimator: DiscreteLinearKF,
        ocp: StandardLinearDiscreteOCP,
        warm_start: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self._estimator = estimator
        self._ocp = ocp
        self._warm_start = bool(warm_start)
        self._u_prev_np: np.ndarray = np.zeros(model.nu)
        self._d_prev_np: np.ndarray = np.zeros(model.nd)
        self._prev_U: np.ndarray | None = None
        self._prev_X: np.ndarray | None = None

    def step(
        self,
        ym: Any,
        D: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nu = self._model.nu
        nd = self._model.nd

        ym_np = _any_to_np1d(ym)
        x_hat_np, _ = self._estimator.step(
            ym_np, self._u_prev_np, self._d_prev_np,
        )

        prof = self._horizon_profile
        if D is not None:
            D_np = _any_to_np1d(D)
        elif prof.disturbance_profile is not None:
            D_np = _any_to_np1d(prof.disturbance_profile)
        else:
            raise ValueError(
                "Provide disturbance forecast via step(D=…) or set_disturbance_profile()."
            )

        x_ref_np = np.asarray(self._model.x_ref, dtype=float).reshape(-1)
        if prof.output_reference_deviation_profile is not None:
            ref_dev = np.asarray(prof.output_reference_deviation_profile, dtype=float)
            if ref_dev.ndim == 1 and ref_dev.size == x_ref_np.size:
                x_ref_np = x_ref_np + ref_dev

        warm = None
        if self._warm_start and self._prev_U is not None:
            warm = _shift_warm_start(
                self._prev_U, self._prev_X, nu, self._model.nx
            )

        if prof.disturbance_profile is not None or any(
            getattr(prof, n) is not None
            for n in (
                "output_tracking_weight_scale_profile",
                "input_regularisation_weight_scale_profile",
                "soft_output_band_half_width_profile",
                "input_min_profile",
                "input_max_profile",
                "input_linear_cost_coefficient_profile",
            )
        ):
            if prof.disturbance_profile is None:
                prof.disturbance_profile = D_np
            U_seq, X_seq = solve_forecast_qp(
                self._ocp, x_hat_np, prof,
                x_ref=x_ref_np, u_prev=self._u_prev_np, warm_start=warm,
            )
        else:
            U_seq, X_seq = self._ocp.solve(
                x_hat_np, D_np, x_ref_np, u_prev=self._u_prev_np, warm_start=warm,
            )

        u = U_seq[:nu]
        self._u_prev_np = np.asarray(u, dtype=float).copy()
        self._d_prev_np = D_np[:nd].copy()
        self._prev_U, self._prev_X = U_seq, X_seq

        return u, U_seq, X_seq
