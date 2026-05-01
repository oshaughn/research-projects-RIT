Integrators (MC Sampling)
=========================

Monte Carlo integration modules for likelihood evaluation and parameter inference.

Overview
--------

Integration in RIFT is primarily focused on the computation of the evidence (marginal likelihood) and the characterization of the posterior distribution. Given the high dimensionality and potential complexity of the gravitational-wave parameter space, RIFT employs a variety of Monte Carlo (MC) integration strategies.

The core goal of these integrators is to evaluate integrals of the form:

.. math::

   Z = \int \mathcal{L}(\theta) \pi(\theta) d\theta

where :math:`\mathcal{L}` is the likelihood and :math:`\pi` is the prior.

Integration Strategies
----------------------

RIFT provides several specialized integrators, each optimized for different regimes of the likelihood surface or computational budgets.

**Standard MC Sampler (``mcsampler``)**
The default integrator. It performs basic Monte Carlo integration by sampling from the prior and averaging the likelihood. While robust, it can be inefficient for highly peaked likelihoods.

**GPU-Accelerated Sampler (``mcsamplerGPU``)**
A high-performance implementation designed to leverage CUDA-enabled GPUs. By parallelizing the likelihood evaluations across thousands of GPU cores, it can achieve orders-of-magnitude speedups over the CPU implementation, especially when using vectorized likelihoods.
Reference: See `Wysocki et al. <https://arxiv.org/abs/1902.04934>`_ for details on GPU acceleration.

**Ensemble / GMM Sampler (``mcsamplerEnsemble``)**
Based on Gaussian Mixture Model (GMM) approximations of the likelihood. This approach identifies the primary modes of the posterior and uses an ensemble of GMMs to efficiently sample the most important regions of the parameter space.
Reference: See `Wofford et al. <https://arxiv.org/pdf/2210.07912>`_ for details on ensemble-based marginalization.

**Adaptive Volume Sampler (``mcsamplerAdaptiveVolume``)**
An integrator that dynamically adjusts the sampling volume based on the observed likelihood distribution. By focusing samples in regions of high likelihood, it reduces the variance of the evidence estimate.
Reference: See `Wagner et al. <https://arxiv.org/abs/2505.11655>`_ for adaptive sampling techniques.

**Normalizing Flows Sampler (``mcsamplerNFlow``)**
Leverages deep learning-based Normalizing Flows to learn a bijective mapping between a simple base distribution (e.g., a Gaussian) and the complex posterior. This allows for nearly exact sampling from the posterior and highly efficient integration.
Reference: See `Wagner et al. <https://arxiv.org/abs/2505.11655>`_.

**Portfolio Integrator (``mcsamplerPortfolio``)**
A meta-integrator that manages a "portfolio" of different sampling strategies. It can dynamically allocate computational resources among various integrators to optimize the trade-off between accuracy and wall-clock time.
Reference: See `Wagner et al. <https://arxiv.org/abs/2505.11655>`_.

API Reference
--------------

The following sections provide the detailed API for the integration modules.

Base Integrators
----------------

.. automodule:: RIFT.integrators.mcsampler_generic
   :members:
   :undoc-members:
   :show-inheritance:

Standard MC Sampler
-------------------

.. automodule:: RIFT.integrators.mcsampler
   :members:
   :undoc-members:
   :show-inheritance:

MC Sampler with Adaptive Volume
-------------------------------

.. automodule:: RIFT.integrators.mcsamplerAdaptiveVolume
   :members:
   :undoc-members:
   :show-inheritance:

Ensemble MC Sampler
-------------------

.. automodule:: RIFT.integrators.mcsamplerEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

GPU MC Sampler
--------------

.. automodule:: RIFT.integrators.mcsamplerGPU
   :members:
   :undoc-members:
   :show-inheritance:

Portfolio MC Sampler
--------------------

.. automodule:: RIFT.integrators.mcsamplerPortfolio
   :members:
   :undoc-members:
   :show-inheritance:

Normalizing Flows MC Sampler
----------------------------

.. automodule:: RIFT.integrators.mcsamplerNFlow
   :members:
   :undoc-members:
   :show-inheritance:

Monte Carlo Ensemble
--------------------

.. automodule:: RIFT.integrators.MonteCarloEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

Weighted GMM
------------

.. automodule:: RIFT.integrators.weighted_gmm
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Utilities
---------------------

.. automodule:: RIFT.integrators.statutils
   :members:
   :undoc-members:
   :show-inheritance:

Gaussian Mixture Model
----------------------

.. automodule:: RIFT.integrators.gaussian_mixture_model
   :members:
   :undoc-members:
   :show-inheritance:

Direct Quadrature
-----------------

.. automodule:: RIFT.integrators.direct_quadrature
   :members:
   :undoc-members:
   :show-inheritance:

Multivariate Truncated Normal
-----------------------------

.. automodule:: RIFT.integrators.multivariate_truncnorm
   :members:
   :undoc-members:
   :show-inheritance:

Unreliable Oracle
-----------------

.. automodule:: RIFT.integrators.unreliable_oracle
   :members:
   :undoc-members:
   :show-inheritance:
