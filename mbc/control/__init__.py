"""Model predictive control sub-package."""

from ..ocp import _shift_warm_start
from .mpc import MPCController
from .cd_mpc import CDMPCController
from .cd_linearized_mpc import CDLinearizedMPCController, linearize_cd_model, discretize_cd_linearization
from .enmpc import CDNMPCController
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
    "CDNMPCController",
    # Helper
    "_shift_warm_start",
    # NLP solver (re-exported for convenience)
    "NLPConstraint",
    "NLPProblem",
    "NLPScalingPolicy",
    "NLPSolverBackend",
    "ScipyNLPBackend",
    "IpoptNLPBackend",
    # QP solver (re-exported for convenience)
    "QPProblem",
    "QPResult",
    "QPSolverBackend",
    "HighsQPBackend",
    "OSQPBackend",
    "make_qp_backend",
]
