"""Tracer-placement sampler kernels.

Each kernel has the signature:
    kernel(particles, surrogate=None, surrogate_prev=None, prior_box, rng, **kw)
        -> (X_new, info_dict)

For backwards compatibility with the prototype harness (proto/run_tier1.py)
the kernels also accept the older positional form
    kernel(particles, lnL_noisy, prior_box, rng, _state_key=...)
which builds a fresh local QuadFit; the production tools call the surrogate-
provided form.
"""
from .smc_mala import iterate as smc_mala
from .birth_death import iterate as birth_death
from .smc_mala_bd import iterate as smc_mala_bd

__all__ = ["smc_mala", "birth_death", "smc_mala_bd"]
