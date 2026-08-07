"""
unreliable_oracle -- proposal 'oracles' for the RIFT portfolio integrator.

An oracle only PROPOSES candidate points; the portfolio evaluates the true
likelihood there and folds them into the training data used to adapt the other
integrators.  Because oracles never contribute to the integral estimate
directly, an inaccurate oracle cannot bias the result -- it can only waste a few
likelihood evaluations.  That makes them a safe channel for injecting cheap,
approximate posterior knowledge (a Fisher matrix, a hill-climbed hotspot, or a
previous run's posterior samples).
"""
from .resampling import ResamplingOracle
from .puffball import PuffballOracle
from .hill_climber import ClimbingOracle
from .fisher_gaussian import FisherGaussianOracle

__all__ = [
    "ResamplingOracle",
    "PuffballOracle",
    "ClimbingOracle",
    "FisherGaussianOracle",
]
