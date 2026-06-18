"""
Model-agnostic horizon profiles for MPC controllers.

Controllers configure time-varying references, bounds, weights, and
disturbances via these profiles before each :meth:`step` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MPCLinearisationPoint:
    """Operating point for successive linearisation MPC."""

    x: np.ndarray
    u: np.ndarray
    d: np.ndarray


@dataclass
class MPCHorizonProfile:
    """
    Bundle of optional horizon-varying control quantities.

    All array fields use shape ``(N, ·)`` unless noted.  ``None`` means
    "use the static OCP / model default".
    """

    disturbance_profile: np.ndarray | None = None
    output_reference_deviation_profile: np.ndarray | None = None
    soft_output_band_half_width_profile: np.ndarray | None = None
    output_tracking_weight_scale_profile: np.ndarray | None = None
    input_regularisation_weight_scale_profile: np.ndarray | None = None
    input_min_profile: np.ndarray | None = None
    input_max_profile: np.ndarray | None = None
    input_linear_cost_coefficient_profile: np.ndarray | None = None
    slack_input_indices: np.ndarray | None = None
    positive_slack_coefficient_profile: np.ndarray | None = None
    negative_slack_coefficient_profile: np.ndarray | None = None
    input_equilibrium: np.ndarray | None = None
    linearisation_point: MPCLinearisationPoint | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class HorizonProfileMPC:
    """
    Mixin providing model-agnostic horizon profile setters for MPC controllers.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._horizon_profile = MPCHorizonProfile()

    @property
    def horizon_profile(self) -> MPCHorizonProfile:
        return self._horizon_profile

    def clear_horizon_profile(self) -> None:
        self._horizon_profile = MPCHorizonProfile()

    def set_horizon_profile(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self._horizon_profile, key):
                setattr(self._horizon_profile, key, value)
            else:
                self._horizon_profile.extra[key] = value

    def set_disturbance_profile(self, profile: np.ndarray) -> None:
        self._horizon_profile.disturbance_profile = np.asarray(profile, dtype=float)

    def set_output_reference_profile(self, profile: np.ndarray) -> None:
        self._horizon_profile.output_reference_deviation_profile = np.asarray(
            profile, dtype=float
        )

    def set_soft_output_band_half_width_profile(self, profile: np.ndarray) -> None:
        self._horizon_profile.soft_output_band_half_width_profile = np.asarray(
            profile, dtype=float
        )

    def set_output_tracking_weight_scale_profile(self, profile: np.ndarray) -> None:
        self._horizon_profile.output_tracking_weight_scale_profile = np.asarray(
            profile, dtype=float
        )

    def set_input_regularisation_weight_scale_profile(self, profile: np.ndarray) -> None:
        self._horizon_profile.input_regularisation_weight_scale_profile = np.asarray(
            profile, dtype=float
        )

    def set_input_bound_profiles(
        self,
        input_min_profile: np.ndarray,
        input_max_profile: np.ndarray,
    ) -> None:
        self._horizon_profile.input_min_profile = np.asarray(input_min_profile, dtype=float)
        self._horizon_profile.input_max_profile = np.asarray(input_max_profile, dtype=float)

    def set_input_linear_cost_profile(
        self,
        coefficient_profile: np.ndarray,
        *,
        slack_input_indices: np.ndarray | None = None,
        positive_slack_coefficient_profile: np.ndarray | None = None,
        negative_slack_coefficient_profile: np.ndarray | None = None,
        input_equilibrium: np.ndarray | None = None,
    ) -> None:
        self._horizon_profile.input_linear_cost_coefficient_profile = np.asarray(
            coefficient_profile, dtype=float
        )
        self._horizon_profile.slack_input_indices = (
            None if slack_input_indices is None
            else np.asarray(slack_input_indices, dtype=int)
        )
        self._horizon_profile.positive_slack_coefficient_profile = (
            None if positive_slack_coefficient_profile is None
            else np.asarray(positive_slack_coefficient_profile, dtype=float)
        )
        self._horizon_profile.negative_slack_coefficient_profile = (
            None if negative_slack_coefficient_profile is None
            else np.asarray(negative_slack_coefficient_profile, dtype=float)
        )
        self._horizon_profile.input_equilibrium = (
            None if input_equilibrium is None
            else np.asarray(input_equilibrium, dtype=float)
        )

    def set_linearisation_point(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
    ) -> None:
        self._horizon_profile.linearisation_point = MPCLinearisationPoint(
            x=np.asarray(x, dtype=float),
            u=np.asarray(u, dtype=float),
            d=np.asarray(d, dtype=float),
        )

    def _resolved_disturbance_profile(self, nd: int) -> np.ndarray | None:
        prof = self._horizon_profile.disturbance_profile
        if prof is None:
            return None
        arr = np.asarray(prof, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1)
        return arr.reshape(arr.shape[0], nd)
