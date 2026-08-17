"""RIFT.misc.tracer_placement — engine for tracer-particle iterative placement.

Drop-in alternative to the puffball grid update used by both the event-level
RIFT iterator and the hyperpipeline. The math is shared; the file I/O wrapper
lives in the two thin command-line tools:

  util_ParameterTracerUpdate.py        (event-level; XML I/O via lalsimutils)
  util_HyperparameterTracerUpdate.py   (hyperpipeline; .dat I/O)

Public API:
    samplers.smc_mala_bd(particles, surrogate, surrogate_prev, prior_box, rng, **kw)
                                     -> (X_new, info)
    samplers.smc_mala(...)            -> (X_new, info)
    samplers.birth_death(...)         -> (X_new, info)
    fits.build(method, X, Y, sigma=None, lnl_floor_delta=None)
                                     -> Fit (callable + .grad helper)
"""
from . import samplers, fits

__all__ = ["samplers", "fits"]
