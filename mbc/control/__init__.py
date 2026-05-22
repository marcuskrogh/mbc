"""Optimal control sub-package for mbc."""

from .ocp import OptimalControlProblem
from .mpc import MPCController
from .cd_ocp import CDOptimalControlProblem, CDTrackingOptimalControlProblem
from .cd_mpc import CDMPCController
from .cd_linearized_mpc import (
    CDLinearizedMPCController,
    linearize_cd_model,
    discretize_cd_linearization,
)
from .enmpc import (
    EconomicOptimalControlProblem,
    CDNMPCController,
)
from .nlp_solver import (
    NLPConstraint,
    NLPProblem,
    NLPScalingPolicy,
    NLPSolverBackend,
    ScipyNLPBackend,
    IpoptNLPBackend,
)

__all__ = [
    "OptimalControlProblem",
    "MPCController",
    "CDOptimalControlProblem",
    "CDTrackingOptimalControlProblem",
    "CDMPCController",
    "CDLinearizedMPCController",
    "linearize_cd_model",
    "discretize_cd_linearization",
    "EconomicOptimalControlProblem",
    "CDNMPCController",
    "NLPConstraint",
    "NLPProblem",
    "NLPScalingPolicy",
    "NLPSolverBackend",
    "ScipyNLPBackend",
    "IpoptNLPBackend",
]
