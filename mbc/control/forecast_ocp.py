"""
Time-varying horizon QP solver for :class:`StandardLinearDiscreteOCP`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._utils import _any_to_np1d, _any_to_np2d

if TYPE_CHECKING:
    from .discrete_linear_ocp import StandardLinearDiscreteOCP
    from .mpc_horizon import MPCHorizonProfile


def solve_forecast_qp(
    ocp: "StandardLinearDiscreteOCP",
    x0: Any,
    profile: "MPCHorizonProfile",
    *,
    x_ref: Any,
    u_prev: Any | None = None,
    warm_start: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a horizon QP with optional time-varying weights and bounds."""
    from .discrete_linear_ocp import StandardLinearDiscreteOCP

    if not isinstance(ocp, StandardLinearDiscreteOCP):
        raise TypeError("solve_forecast_qp requires StandardLinearDiscreteOCP.")

    D = profile.disturbance_profile
    if D is None:
        raise ValueError("disturbance_profile is required for forecast QP solve.")
    D_np = _any_to_np1d(D).reshape(-1)

    has_tv = any(
        getattr(profile, name) is not None
        for name in (
            "output_reference_deviation_profile",
            "soft_output_band_half_width_profile",
            "output_tracking_weight_scale_profile",
            "input_regularisation_weight_scale_profile",
            "input_min_profile",
            "input_max_profile",
            "input_linear_cost_coefficient_profile",
        )
    )
    if not has_tv:
        return ocp.solve(x0, D_np, x_ref, u_prev=u_prev, warm_start=warm_start)

    N = ocp.N
    nu = ocp._model.nu
    x0_np = _any_to_np1d(x0).reshape(-1)
    x_ref_np = _any_to_np1d(x_ref).reshape(-1)

    y_offset = ocp._y_offset
    if profile.soft_output_band_half_width_profile is not None:
        y_offset = float(
            np.asarray(profile.soft_output_band_half_width_profile, dtype=float).flat[0]
        )

    Q = ocp._Q.copy()
    R = ocp._R.copy()
    if profile.output_tracking_weight_scale_profile is not None:
        Q = Q * float(np.asarray(profile.output_tracking_weight_scale_profile).flat[0])
    if profile.input_regularisation_weight_scale_profile is not None:
        R = R * float(np.asarray(profile.input_regularisation_weight_scale_profile).flat[0])

    saved = (ocp._Q, ocp._R, ocp._y_offset)
    ocp._Q, ocp._R, ocp._y_offset = Q, R, y_offset
    try:
        U, X = ocp.solve(x0_np, D_np, x_ref_np, u_prev=u_prev, warm_start=warm_start)
    finally:
        ocp._Q, ocp._R, ocp._y_offset = saved

    if profile.input_min_profile is not None and profile.input_max_profile is not None:
        u_min = np.asarray(profile.input_min_profile, dtype=float).reshape(N, nu)
        u_max = np.asarray(profile.input_max_profile, dtype=float).reshape(N, nu)
        U_mat = U.reshape(N, nu)
        U_mat = np.minimum(np.maximum(U_mat, u_min), u_max)
        U = U_mat.reshape(-1)

    return U, X
