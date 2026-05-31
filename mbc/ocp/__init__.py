"""Optimal control sub-package."""
from ._base import (
    OCP,
    DiscreteLinearOCPBase,
    DiscreteLinearisedOCPBase,
    ContinuousLinearOCPBase,
    ContinuousLinearisedOCPBase,
    ContinuousNonlinearOCPBase,
)
from .discrete_linear import DiscreteLinearOCP, _shift_warm_start
from .discrete_linearised import DiscreteLinearisedOCP
from .continuous_linear import ContinuousLinearOCP
from .continuous_linearised import ContinuousLinearisedOCP
from .continuous_nonlinear import ContinuousNonlinearOCP
from .qp_solver import QPProblem, QPResult, QPSolverBackend, HighsQPBackend, OSQPBackend, make_qp_backend
from .nlp_solver import NLPConstraint, NLPProblem, NLPScalingPolicy, NLPSolverBackend, ScipyNLPBackend, IpoptNLPBackend

__all__ = [
    # Abstract bases
    "OCP",
    "DiscreteLinearOCPBase",
    "DiscreteLinearisedOCPBase",
    "ContinuousLinearOCPBase",
    "ContinuousLinearisedOCPBase",
    "ContinuousNonlinearOCPBase",
    # Concrete OCPs
    "DiscreteLinearOCP",
    "DiscreteLinearisedOCP",
    "ContinuousLinearOCP",
    "ContinuousLinearisedOCP",
    "ContinuousNonlinearOCP",
    # Helper
    "_shift_warm_start",
    # QP solver
    "QPProblem",
    "QPResult",
    "QPSolverBackend",
    "HighsQPBackend",
    "OSQPBackend",
    "make_qp_backend",
    # NLP solver
    "NLPConstraint",
    "NLPProblem",
    "NLPScalingPolicy",
    "NLPSolverBackend",
    "ScipyNLPBackend",
    "IpoptNLPBackend",
]
