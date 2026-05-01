Precessing Fisher Matrix
========================

This module provides tools for computing the Fisher information matrix in the context of precessing gravitational-wave signals. The Fisher matrix is used to estimate the covariance of parameter estimates and their uncertainties.

Overview
--------

For precessing binaries, the Fisher matrix calculation is complicated by the fact that the waveform depends on multiple angles (inclination, spin orientations) in a coupled way. This module provides:

1. **Numerical Fisher Matrix**: Full finite-difference calculation using waveform derivatives
2. **Approximate Precessing Fisher**: Fast approximation using phase derivatives
3. **Effective Fisher**: Fitting-based approach for diagonal elements
4. **Overlap Utilities**: Functions for computing match/MATCH as a function of parameter variation

Matrix Operations
-----------------

FisherProject
^^^^^^^^^^^^^^

``FisherProject(mtx, indx_list_preserve)`` projects a Fisher matrix onto a subset of parameters using matrix algebra. It computes:

$$\Gamma_{AA} - \Gamma_{AB} \cdot \Gamma_{BB}^{-1} \cdot \Gamma_{BA}$$

This is useful for marginalizing over nuisance parameters.

Phase Derivative Framework
---------------------------

The precessing Fisher calculation relies on decomposing the phase into components:

- $\Psi_{2F}$: Orbital phase contribution
- $\alpha_F$: Precession angle
- $\gamma_F$: Secondary precession angle

PhaseDerivativeSeries
^^^^^^^^^^^^^^^^^^^^^

``PhaseDerivativeSeries(P, p1)`` computes derivatives of these phase components with respect to waveform parameters. This is cached for efficiency when computing multiple Fisher matrix elements.

Numerical Fisher Matrix
------------------------

``NumericalFisherMatrixElement(P, p1, p2, psd, ...)`` performs a full finite-difference calculation of the Fisher matrix element:

$$\Gamma_{ij} = \langle \frac{\partial h}{\partial \theta_i} | \frac{\partial h}{\partial \theta_j} \rangle$$

This uses the inner product defined by the noise PSD.

Approximate Precessing Fisher
-----------------------------

``ApproximatePrecessingFisherMatrixElement(P, p1, p2, psd)`` provides a faster approximation that:

1. Computes phase derivatives analytically/semi-analytically
2. Weights by the one-sided PSD integral
3. Accounts for the geometric factors from spin-weighted spherical harmonics

The approximation is valid when the waveform is dominated by the $(2, \pm 2)$ modes.

Effective Fisher (Diagonal Elements)
------------------------------------

``EffectiveFisherMatrixElement(P, p1, p1, psd)`` fits a quadratic to the overlap function:

$$|\langle h(\theta_0) | h(\theta_0 + \delta\theta) \rangle|^2 \approx 1 - \frac{1}{2} \Gamma_{ii} (\delta\theta)^2$$

This is used for diagonal elements where $p_1 = p_2$.

Overlap Utilities
-----------------

These functions compute the inner product as a function of parameter variation:

- ``OverlapVersusChangingParameter``: 1D scan over a single parameter
- ``OverlapGridVersusChangingParameter``: 2D grid scan over two parameters

Supported Parameters
--------------------

The module uses tolerance scales for numerical derivatives of:

+----------------------+-------------+
| Parameter            | Tolerance   |
+======================+=============+
| chirp mass (mc)      | 1e-3 rel   |
| total mass (mtot)     | 1e-3 rel   |
| time (tref)          | 1e-3 abs   |
| phase (phiref)       | 0.2 abs    |
| spin (chi1)          | 0.01 abs   |
| eta                  | 0.001 abs  |
| thetaJN              | 0.1 abs    |
| phiJL                | 0.1 abs    |
+----------------------+-------------+

Use Cases
---------

1. **Rapid Uncertainty Estimation**: Use the approximate Fisher for quick covariance estimates during pipeline development
2. **Injection Recovery**: Validate parameter estimation accuracy by comparing Fisher predictions to actual recovery results
3. **Parameter Space Exploration**: Identify which parameters are most/least well-constrained
4. **Pipeline Comparison**: Compare different waveform approximants by their Fisher matrix structure
