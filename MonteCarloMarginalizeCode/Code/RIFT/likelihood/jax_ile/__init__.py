"""
RIFT.likelihood.jax_ile
=======================

A JAX, automatic-differentiation-compatible reimplementation of the RIFT ILE
extrinsic likelihood.  It mirrors the structure of the production
``factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop``
(the "...NoLoop", array-vector, fused code branch) but expresses the
extrinsic-parameter -> lnL map in pure ``jax.numpy`` so that the result is
differentiable, ``jit``-able and ``vmap``-able.

Scope and division of labour
----------------------------
The *expensive, data-touching* precompute is **unchanged** and is reused
verbatim from the production code: frame reading, PSD handling, waveform
generation and the per-detector inner products
``<h_lm(t)|d>`` (the rholm timeseries) and ``<h_lm|h_l'm'>`` (the U,V cross
terms) are all produced by
``factored_likelihood.PrecomputeLikelihoodTerms`` and packed with
``factored_likelihood.PackLikelihoodDataStructuresAsArrays``.  This module
only re-expresses the cheap combination

    lnL(theta) = log integral_t exp( Re kappa(theta,t) - 1/2 rho^2(theta,t) ) dt

as a JAX function of the extrinsic parameters
``theta = (RA, DEC, psi, incl, phiref, distance)`` (time is marginalized).

Sub-modules
-----------
detector    : ComputeDetAMResponse / TimeDelayFromEarthCenter in JAX
spherical   : spin (-2) weighted spherical harmonics in JAX
core        : the fused factored log-likelihood and its data container
"""

from .core import (
    JAXLikelihoodData,
    build_likelihood_data,
    fused_log_likelihood,
    fused_log_likelihood_distmarg,
    make_distance_grid,
    make_log_likelihood,
)
from .wrapper import (
    JAXExtrinsicLikelihood,
    JAXDistPhiMargLikelihood,
    JAXDistPsiMargLikelihood,
    build_data_from_precompute,
    build_rotation_data_from_precompute,
    build_freqresponse_data_from_precompute,
    EXTRINSIC_PARAM_ORDER,
)
from .banded import build_rotation_data, build_freqresponse_data
from .coordinates import (
    build_network_frame,
    equatorial_to_network,
    network_to_equatorial,
    polarization_phase_fold,
)

__all__ = [
    "JAXLikelihoodData",
    "build_likelihood_data",
    "fused_log_likelihood",
    "fused_log_likelihood_distmarg",
    "make_distance_grid",
    "make_log_likelihood",
    "JAXExtrinsicLikelihood",
    "JAXDistPhiMargLikelihood",
    "JAXDistPsiMargLikelihood",
    "build_data_from_precompute",
    "build_rotation_data_from_precompute",
    "build_freqresponse_data_from_precompute",
    "build_rotation_data",
    "build_freqresponse_data",
    "EXTRINSIC_PARAM_ORDER",
    "build_network_frame",
    "equatorial_to_network",
    "network_to_equatorial",
    "polarization_phase_fold",
]
