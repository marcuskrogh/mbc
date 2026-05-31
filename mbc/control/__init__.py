"""Model predictive control sub-package."""

from ..ocp import _shift_warm_start  # re-export helper used by tests
from .mpc import MPCController
from .cd_mpc import CDMPCController
from .cd_linearized_mpc import CDLinearizedMPCController, linearize_cd_model, discretize_cd_linearization
from .enmpc import EconomicOptimalControlProblem, CDNMPCController
# Legacy convenience re-exports from the old control package.
from .ocp import OptimalControlProblem
from .cd_ocp import CDOptimalControlProblem, CDTrackingOptimalControlProblem
from ..ocp import (
    NLPConstraint,
    NLPProblem,
    NLPScalingPolicy,
    NLPSolverBackend,
    ScipyNLPBackend,
    IpoptNLPBackend,
    QPProblem,
    QPResult,
    QPSolverBackend,
    HighsQPBackend,
    OSQPBackend,
    make_qp_backend,
)

__all__ = [
    # MPC controllers
    "MPCController",
    "CDMPCController",
    "CDLinearizedMPCController",
    "linearize_cd_model",
    "discretize_cd_linearization",
    "EconomicOptimalControlProblem",
    "CDNMPCController",
    # Legacy OCP re-exports
    "OptimalControlProblem",
    "CDOptimalControlProblem",
    "CDTrackingOptimalControlProblem",
    # Helper
    "_shift_warm_start",
    # NLP solver (re-exported for backward compat)
    "NLPConstraint",
    "NLPProblem",
    "NLPScalingPolicy",
    "NLPSolverBackend",
    "ScipyNLPBackend",
    "IpoptNLPBackend",
    # QP solver (re-exported for backward compat)
    "QPProblem",
    "QPResult",
    "QPSolverBackend",
    "HighsQPBackend",
    "OSQPBackend",
    "make_qp_backend",
]
