"""Optimal control sub-package for mbc."""

from ._base import (
    OptimalControlProblem,
    DiscreteOptimalControlProblem,
    ContinuousOptimalControlProblem,
    ModelPredictiveController,
    HorizonProfile,
    LinearisationPoint,
)
from .discrete_linear_ocp import StandardLinearDiscreteOCP, _shift_warm_start
from .discrete_linearised_ocp import StandardLinearisedDiscreteOCP
from .continuous_linear_ocp import StandardLinearContinuousDiscreteOCP
from .continuous_ocp import GeneralContinuousOCP, StandardContinuousOCP
from .continuous_linearised_ocp import StandardLinearizedContinuousDiscreteOCP
from .mpc import LinearDiscreteMPC, StandardLinearDiscreteMPC
from .linearised_discrete_mpc import (
    LinearisedDiscreteMPC,
    StandardLinearisedDiscreteMPC,
    linearize_discrete_model,
)
from .cd_mpc import LinearContinuousMPC, StandardLinearContinuousMPC
from .cd_linearized_mpc import (
    LinearisedContinuousMPC,
    StandardLinearisedContinuousMPC,
    linearize_cd_model,
    discretize_cd_linearization,
)
from .enmpc import NonlinearContinuousMPC, StandardNonlinearContinuousMPC
from .input_linear_cost import (
    InputLinearCostMode,
    absolute_quadratic_input_regularisation_linear_term,
    infer_signed_magnitude_input_indices,
    resolve_input_linear_cost,
)
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
    # Abstract bases
    "OptimalControlProblem",
    "DiscreteOptimalControlProblem",
    "ContinuousOptimalControlProblem",
    "ModelPredictiveController",
    # Horizon profile types
    "HorizonProfile",
    "LinearisationPoint",
    # Standard OCP implementations
    "StandardLinearDiscreteOCP",
    "StandardLinearisedDiscreteOCP",
    "StandardLinearContinuousDiscreteOCP",
    "StandardLinearizedContinuousDiscreteOCP",
    "GeneralContinuousOCP",
    "StandardContinuousOCP",
    # MPC abstract bases
    "LinearDiscreteMPC",
    "LinearisedDiscreteMPC",
    "LinearContinuousMPC",
    "LinearisedContinuousMPC",
    "NonlinearContinuousMPC",
    # Standard MPC implementations
    "StandardLinearDiscreteMPC",
    "StandardLinearisedDiscreteMPC",
    "StandardLinearContinuousMPC",
    "StandardLinearisedContinuousMPC",
    "StandardNonlinearContinuousMPC",
    # CD linearisation helpers
    "linearize_cd_model",
    "discretize_cd_linearization",
    "linearize_discrete_model",
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
    # Input linear cost
    "InputLinearCostMode",
    "absolute_quadratic_input_regularisation_linear_term",
    "infer_signed_magnitude_input_indices",
    "resolve_input_linear_cost",
    # Helpers
    "_shift_warm_start",
]
