"""
Abstract base classes for optimal control problems and MPC controllers.

Horizon-varying references, bounds, weights, and disturbances are configured
via :class:`HorizonProfile` setters on every OCP and MPC base class — there
are no separate forecast-aware implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LinearisationPoint:
    """Operating point for successive linearisation MPC."""

    x: np.ndarray
    u: np.ndarray
    d: np.ndarray


@dataclass
class HorizonProfile:
    """
    Optional horizon-varying control quantities.

    Array fields use shape ``(N, ·)`` unless noted.  ``None`` means use the
    static OCP / model default for that quantity.
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
    linearisation_point: LinearisationPoint | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class _HorizonProfileSupport:
    """Shared horizon-profile state and setters for OCPs and MPC controllers."""

    def __init__(self) -> None:
        self._horizon_profile = HorizonProfile()

    @property
    def horizon_profile(self) -> HorizonProfile:
        return self._horizon_profile

    def clear_horizon_profile(self) -> None:
        """Reset all horizon-profile fields to their defaults in place.

        Mutates the existing :class:`HorizonProfile` object rather than
        replacing it, so the shared reference established by :meth:`_bind_ocp`
        (MPC ↔ OCP) is preserved.
        """
        prof = self._horizon_profile
        fresh = HorizonProfile()
        for f in fresh.__dataclass_fields__:
            setattr(prof, f, getattr(fresh, f))

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
        self._horizon_profile.input_min_profile = np.asarray(
            input_min_profile, dtype=float
        )
        self._horizon_profile.input_max_profile = np.asarray(
            input_max_profile, dtype=float
        )

    def set_input_linear_cost_profile(
        self,
        coefficient_profile: np.ndarray,
        *,
        slack_input_indices: np.ndarray | None = None,
        signed_magnitude_input_indices: np.ndarray | None = None,
        positive_slack_coefficient_profile: np.ndarray | None = None,
        negative_slack_coefficient_profile: np.ndarray | None = None,
        input_equilibrium: np.ndarray | None = None,
    ) -> None:
        """
        Configure a linear Mayer penalty ``cᵀu`` on inputs over the horizon.

        Parameters
        ----------
        coefficient_profile : (N, nu) or (nu,) array
            Mayer coefficient per input.  For **direct** inputs this is the
            coefficient on ``u``.  For **signed-magnitude** inputs it is the
            default coefficient on both slacks when asymmetric profiles are
            omitted (penalising ``|u|`` when ``c⁺ = c⁻ = c``).
        slack_input_indices, signed_magnitude_input_indices
            Input indices that use ``u = s − t``, ``s, t ≥ 0``.  When both are
            omitted, defaults to inputs whose bounds span zero
            (:func:`~mbc.control.infer_signed_magnitude_input_indices`).
        positive_slack_coefficient_profile, negative_slack_coefficient_profile
            Optional ``(N, n_slack)`` asymmetric coefficients on ``s`` and
            ``t``.  Omit for symmetric magnitude penalisation ``c·(s + t)``.
        input_equilibrium
            Absolute input offset ``u_eq`` when the QP optimises deviation inputs
            ``δu``.  Quadratic regularisation ``‖u‖²_R`` and signed-magnitude
            slacks ``u = s − t`` are then expressed relative to ``u = u_eq + δu``.
            Linearised OCPs set this automatically from the operating-point
            input ``u_s``.
        """
        idx = signed_magnitude_input_indices if signed_magnitude_input_indices is not None else slack_input_indices
        self._horizon_profile.input_linear_cost_coefficient_profile = np.asarray(
            coefficient_profile, dtype=float
        )
        self._horizon_profile.slack_input_indices = (
            None if idx is None else np.asarray(idx, dtype=int)
        )
        self._horizon_profile.positive_slack_coefficient_profile = (
            None if positive_slack_coefficient_profile is None
            else np.asarray(positive_slack_coefficient_profile, dtype=float)
        )
        self._horizon_profile.negative_slack_coefficient_profile = (
            None if negative_slack_coefficient_profile is None
            else np.asarray(negative_slack_coefficient_profile, dtype=float)
        )
        if input_equilibrium is not None:
            self._horizon_profile.input_equilibrium = np.asarray(
                input_equilibrium, dtype=float
            )

    def set_input_equilibrium(self, u_equilibrium: np.ndarray) -> None:
        """Set absolute input offset for deviation-coordinate regularisation."""
        self._horizon_profile.input_equilibrium = np.asarray(
            u_equilibrium, dtype=float
        )

    def set_linearisation_point(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
    ) -> None:
        self._horizon_profile.linearisation_point = LinearisationPoint(
            x=np.asarray(x, dtype=float),
            u=np.asarray(u, dtype=float),
            d=np.asarray(d, dtype=float),
        )

    def _share_horizon_profile_with(self, other: _HorizonProfileSupport) -> None:
        """Bind another host to this object's horizon profile."""
        other._horizon_profile = self._horizon_profile


class OptimalControlProblem(_HorizonProfileSupport, ABC):
    """
    Common abstract base for all optimal control problems.

    Every concrete OCP exposes the prediction horizon ``N`` and input dimension
    ``nu``, and supports horizon profiles for time-varying parameters.
    """

    @property
    @abstractmethod
    def N(self) -> int:
        """Prediction horizon (number of control intervals)."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Input dimension nᵘ."""


class DiscreteOptimalControlProblem(OptimalControlProblem):
    """
    Abstract base for discrete-time optimal control problems.

    Concrete subclasses solve a finite-horizon QP over a discrete prediction
    model.  :meth:`solve` reads disturbances and other horizon quantities from
    the configured :attr:`horizon_profile` when call-time arguments are omitted.
    """


class ContinuousOptimalControlProblem(OptimalControlProblem):
    """
    Abstract base for continuous-time optimal control problems.

    Concrete subclasses formulate a finite-horizon NLP whose dynamics are
    discretised internally.  :meth:`solve` reads horizon quantities from
    :attr:`horizon_profile` when call-time arguments are omitted.
    """


class ModelPredictiveController(_HorizonProfileSupport, ABC):
    """
    Common abstract base for all MPC controllers.

    Horizon profiles configured on the controller are shared with the wrapped
    OCP via :meth:`_bind_ocp` so that setters and :meth:`compute` use one profile.
    """

    def _bind_ocp(self, ocp: OptimalControlProblem) -> None:
        """Share this controller's horizon profile with ``ocp``."""
        self._share_horizon_profile_with(ocp)
