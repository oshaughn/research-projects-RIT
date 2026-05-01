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