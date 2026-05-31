"""NLP solver backends — re-exported from mbc.ocp.nlp_solver for backward compatibility."""
from ..ocp.nlp_solver import (
    NLPScalingPolicy,
    NLPConstraint,
    NLPProblem,
    NLPResult,
    NLPSolverBackend,
    ScipyNLPBackend,
    IpoptNLPBackend,
    make_nlp_backend,
    _apply_scaling,
    _normalize_scale_vector,
    _identity_vector,
    _SCIPY_KNOWN_METHODS,
    _SCIPY_METHODS_WITH_HESS,
)

__all__ = [
    "NLPScalingPolicy",
    "NLPConstraint",
    "NLPProblem",
    "NLPResult",
    "NLPSolverBackend",
    "ScipyNLPBackend",
    "IpoptNLPBackend",
    "make_nlp_backend",
]
