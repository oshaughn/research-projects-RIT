"""Benchmark harness for jax_gp interpolators (synthetic truths + metrics)."""
from .truth_functions import (
    TruthFunction,
    CorrelatedGaussian,
    BananaRidge,
    MultimodalMixture,
    SharpPeak,
    all_truths,
)

__all__ = [
    "TruthFunction",
    "CorrelatedGaussian",
    "BananaRidge",
    "MultimodalMixture",
    "SharpPeak",
    "all_truths",
]
