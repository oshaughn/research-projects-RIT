tools
=====

The ``RIFT.misc.tools`` module provides a suite of utilities for transforming gravitational wave source parameters between different coordinate systems. These transformations are critical for linearizing the likelihood surface and optimizing the sampling process.

Mass Parameter Conversions
--------------------------

The module provides standard conversions between different mass representations:
- **Component Masses ($\mathbf{m_1, m_2}$)**: The individual masses of the binary components.
- **Chirp Mass ($\mathbf{\mathcal{M}_c}$)**: The leading-order parameter determining the frequency evolution of the signal.
- **Symmetric Mass Ratio ($\mathbf{\eta}$)**: A measure of the mass asymmetry of the binary.
- **Mass Ratio ($\mathbf{q}$)**: Defined as $m_2 / m_1$.

Key functions include ``m1m2ToMc``, ``qToeta``, and ``McqTom1m2``.

PN Phase and Mu-Parameters
--------------------------

A central feature of RIFT is the transformation of physical parameters into a more "linear" space based on Post-Newtonian (PN) phase coefficients.

1. **PN Phase Coefficients**: Functions like ``psi0``, ``psi2``, and ``psi3`` calculate the phase coefficients at 0PN, 1PN, and 1.5PN orders, respectively.
2. **The $\mathbf{U}$ Matrix**: RIFT uses a linear transformation matrix $\mathbf{U}$ to map these phase coefficients to a set of parameters denoted as $\mu_1, \mu_2, \mu_3$. This mapping is designed to decouple the parameters and make the likelihood surface more Gaussian, which significantly improves the efficiency of interpolators and samplers.
3. **Inversion**: Since the mapping from physical parameters to $\mu$ parameters is non-linear, the module implements numerical inversion (e.g., using bisection search in ``mu1mu2etaToMc``) to recover physical parameters from $\mu$ values.

High-Level Converters
--------------------

For ease of use, the module provides dictionary-based converters that can be integrated directly into the sampling pipeline:
- ``convert_m1m2chi1chi2_to_Mcqmu1mu2``: Transforms a set of component masses and spins into chirp mass, mass ratio, and $\mu$ parameters.
- ``convert_Mcqchi1chi2_to_mu1mu2``: Specifically targets the $\mu$-parameter mapping.

Sampling and Jacobians
----------------------

When changing coordinate systems during sampling, the prior volume must be preserved. The function ``m1m2chi1chi2Tomu1mu2qchi2Jacobian`` computes the determinant of the transformation Jacobian, allowing the sampler to correctly weight the prior in the $\mu$-coordinate space.

API Reference
-------------

.. automodule:: RIFT.misc.tools
    :members:
    :undoc-members:
    :show-inheritance:
