"""Optimal control sub-package for mbc."""

from ._base import OCP
from .discrete_linear_ocp import DiscreteLinearOCP, _shift_warm_start
from .discrete_linearised_ocp import DiscreteLinearisedOCP
from .continuous_linear_ocp import ContinuousLinearOCP
from .continuous_ocp import ContinuousOCP
from .continuous_linearised_ocp import ContinuousLinearisedOCP
from .mpc import MPCController
from .cd_mpc import CDMPCController
from .cd_linearized_mpc import (
    CDLinearizedMPCController,
    linearize_cd_model,
    discretize_cd_linearization,
)
from .enmpc import CDNMPCController
from .nlp_solver import (
    NLPConstraint,
    NLPProblem,
    NLPScalingPolicy,
    NLPSolverBackend,
    ScipyNLPBackend,
    IpoptNLPBackend,
)
from .qp_solver import (
    QPProblem,
    QPResult,
    QPSolverBackend,
    HighsQPBackend,
    OSQPBackend,
    make_qp_backend,
)

__all__ = [
    # Abstract base
    "OCP",
    # OCP canonical names
    "DiscreteLinearOCP",
    "DiscreteLinearisedOCP",
    "ContinuousLinearOCP",
    "ContinuousOCP",
    "ContinuousLinearisedOCP",
    # MPC controllers
    "MPCController",
    "CDMPCController",
    "CDLinearizedMPCController",
    "linearize_cd_model",
    "discretize_cd_linearization",
    "CDNMPCController",
    # NLP solver
    "NLPConstraint",
    "NLPProblem",
    "NLPScalingPolicy",
    "NLPSolverBackend",
    "ScipyNLPBackend",
    "IpoptNLPBackend",
    # QP solver
    "QPProblem",
    "QPResult",
    "QPSolverBackend",
    "HighsQPBackend",
    "OSQPBackend",
    "make_qp_backend",
    # Helpers
    "_shift_warm_start",
]
