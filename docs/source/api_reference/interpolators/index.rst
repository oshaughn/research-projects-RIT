Interpolators (Surrogates)
===========================

Gaussian process and surrogate modeling for efficient parameter inference.

Overview
--------

These modules provide surrogate models and interpolation methods that accelerate
likelihood evaluation. Instead of computing expensive waveform models for each point,
the integrators can query these surrogate models for fast approximations.

.. seealso::

   - :doc:`../integrators/index` - MC integration modules that use these surrogates
   - :doc:`../likelihood/index` - Full likelihood evaluation (for comparison)

Bayesian Least Squares
----------------------

.. automodule:: RIFT.interpolators.BayesianLeastSquares
   :members:
   :undoc-members:
   :show-inheritance:

Constrained Quadratic Likelihood
--------------------------------

.. automodule:: RIFT.interpolators.ConstrainedQuadraticLikelihood
   :members:
   :undoc-members:
   :show-inheritance:

Efficient SKLearn GP Save
-------------------------

.. automodule:: RIFT.interpolators.efficient_save_sklearn_gp
   :members:
   :undoc-members:
   :show-inheritance:

Basic GP
--------

.. automodule:: RIFT.interpolators.gp
   :members:
   :undoc-members:
   :show-inheritance:

GPyTorch Wrapper
----------------

.. automodule:: RIFT.interpolators.gpytorch_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Internal GP
-----------

.. automodule:: RIFT.interpolators.internal_GP
   :members:
   :undoc-members:
   :show-inheritance:

GPU Interpolation
-----------------

.. automodule:: RIFT.interpolators.interp_gpu
   :members:
   :undoc-members:
   :show-inheritance:

SENNI (Surrogate Enabled Nested Nested Importance Sampling)
-----------------------------------------------------------

.. automodule:: RIFT.interpolators.senni
   :members:
   :undoc-members:
   :show-inheritance:


Differentiable (JAX) surrogates: ``jax_gp``
===========================================

``RIFT.interpolators.jax_gp`` is a pure-JAX framework for fitting a continuous,
``jax.grad``-able surrogate of the marginalized log-likelihood ``lnL(theta)`` over
the ILE intrinsic grid, and exporting it as a small, portable artifact that reloads
with **no RIFT/lalsimutils dependency**.  It is the differentiable counterpart to the
GP/RF fits the legacy CIP uses, intended for AD use cases (gradient-based sampling,
population inference) and for shipping a self-contained likelihood surrogate.

Interpolators
-------------

Each interpolator fits in a per-dimension *whitened* space and exposes a pure-JAX
``lnL_physical(theta)`` (differentiable in the fit coordinates).

.. automodule:: RIFT.interpolators.jax_gp.interface
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: RIFT.interpolators.jax_gp.quad_gp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: RIFT.interpolators.jax_gp.svgp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: RIFT.interpolators.jax_gp.exact
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: RIFT.interpolators.jax_gp.rff
   :members:
   :undoc-members:
   :show-inheritance:

Artifact export / import
------------------------

.. automodule:: RIFT.interpolators.jax_gp.export
   :members:
   :undoc-members:
   :show-inheritance:

Applications
------------

``export_artifact`` packages a single ILE ``.net`` into a differentiable artifact;
``jax_cip`` is a standalone pure-JAX intrinsic-posterior path; ``export_at_scale``
points at a *real run directory*, exports the artifact, and validates it against the
run's own CIP posterior (apples-to-apples, with RIFT's priors), locally or over
HTCondor.  See ``applications/EXPORT_AT_SCALE.md`` and ``applications/ARTIFACT.md``.

.. automodule:: RIFT.interpolators.jax_gp.applications.export_at_scale
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: RIFT.interpolators.jax_gp.applications.export_artifact
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: RIFT.interpolators.jax_gp.applications.jax_cip
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: RIFT.interpolators.jax_gp.applications.compare
   :members:
   :undoc-members:
   :show-inheritance: