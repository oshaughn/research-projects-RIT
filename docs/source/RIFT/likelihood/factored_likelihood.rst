Factored Likelihood
===================

.. automodule:: RIFT.likelihood.factored_likelihood
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``factored_likelihood`` module provides tools to compute the log-likelihood of gravitational waveform parameters. It employs a "factored" approach where terms depending only on intrinsic parameters are precomputed, allowing for efficient evaluation over extrinsic parameters.

Main Driver Functions
---------------------

.. autofunction:: PrecomputeLikelihoodTerms

   Computes the inner products between waveform modes and data ($\langle h_{lm}(t) | d \rangle$) and the cross terms between modes ($\langle h_{lm} | h_{l'm'} \rangle$).

.. autofunction:: FactoredLogLikelihood

   Computes the log-likelihood for a given set of extrinsic parameters using precomputed terms.

.. autofunction:: FactoredLogLikelihoodTimeMarginalized

   Computes the log-likelihood marginalized over the arrival time.

Helper Functions
----------------

.. autofunction:: ComputeModeIPTimeSeries

   Computes the complex-valued overlap between each waveform mode and the interferometer data, weighted by the power spectral density (PSD).

.. autofunction:: InterpolateRholms

   Returns a dictionary of interpolating functions for the overlap $\langle h_{lm}(t) | d \rangle$.

.. autofunction:: ComputeModeCrossTermIP

   Computes the inner products (cross terms) between waveform modes $\langle h_{lm} | h_{l'm'} \rangle$.

.. autofunction:: ComplexAntennaFactor

   Computes the complex-valued antenna pattern function $F_+ + i F_\times$.

.. autofunction:: ComputeYlms

   Computes the spin-weighted spherical harmonics $-2Y_{lm}(\theta, \phi)$.

.. autofunction:: ComputeArrivalTimeAtDetector

   Computes the time of arrival at a detector from the geocenter reference time.
