Effective Fisher Matrix
=======================

This module provides tools for computing an effective Fisher matrix by fitting the overlap (ambiguity) function on a discrete parameter grid. It was originally developed by Evan Ochsner.

Overview
--------

The effective Fisher matrix approach provides a computationally tractable alternative to direct numerical differentiation of the waveform. Instead, it:

1. Evaluates the overlap $\langle h(\theta_0) | h(\theta_0 + \delta\theta) \rangle$ on a grid of parameter variations
2. Fits a quadratic form to the resulting ambiguity function
3. Extracts the Fisher matrix elements from the quadratic coefficients

Core Functions
--------------

effectiveFisher
^^^^^^^^^^^^^^

``effectiveFisher(residual_func, *flat_grids)`` fits a quadratic to the overlap function on an N-dimensional grid.

**Inputs:**
- ``residual_func``: Function to compute residuals (e.g., ``residuals2d``)
- ``flat_grids``: N+1 flat arrays (N parameter arrays + 1 overlap array)

**Returns:**
- Flat array of upper-triangular Fisher matrix elements

**Example:**

.. code-block:: python

    x1s = np.linspace(mc - 0.1, mc + 0.1, 5)
    x2s = np.linspace(eta - 0.02, eta + 0.02, 5)
    overlaps = evaluate_ip_on_grid(hfSIG, P, IP, ['mc', 'eta'], grid)
    gamma = effectiveFisher(residuals2d, x1s, x2s, overlaps)

find_effective_Fisher_region
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``find_effective_Fisher_region(P, IP, target_match, param_names, param_bounds)`` finds parameter bounds where the match exceeds a target value.

**Inputs:**
- ``P``: ChooseWaveformParams object describing the target signal
- ``IP``: Inner product object
- ``target_match``: Threshold match value (e.g., 0.9)
- ``param_names``: List of parameter names to vary
- ``param_bounds``: Nx2 array of bounds for each parameter

**Returns:**
- Array of [min, max] bounds for each parameter

This function uses Brent's method to find the boundary where overlap drops below the target.

Grid Generation Utilities
------------------------

The module provides several functions for creating multi-dimensional grids:

make_regular_1d_grids
^^^^^^^^^^^^^^^^^^^^^

Creates evenly-spaced 1D grids for each parameter.

multi_dim_meshgrid
^^^^^^^^^^^^^^^^^^

Generalized version of ``np.meshgrid`` for arbitrary dimensions.

multi_dim_grid
^^^^^^^^^^^^^^

Creates an array of all points on a multi-dimensional grid.

Least-Squares Fitting
--------------------

The module includes residual and evaluation functions for quadratic fitting in 2D through 5D:

- ``residuals2d`` / ``evalfit2d``: 2-parameter case
- ``residuals3d`` / ``evalfit3d``: 3-parameter case
- ``residuals4d`` / ``evalfit4d``: 4-parameter case
- ``residuals5d`` / ``evalfit5d``: 5-parameter case

Each fits the form:

$$1 - \frac{1}{2} \sum_{i,j} \Gamma_{ij} \delta\theta_i \delta\theta_j$$

Matrix Utilities
---------------

array_to_symmetric_matrix
^^^^^^^^^^^^^^^^^^^^^^^^

Converts a flat array of upper-triangular elements to a symmetric matrix.

eigensystem
^^^^^^^^^^^

Computes eigenvalues, eigenvectors, and the rotation matrix for basis transformation.

Ellipsoid Sampling
-------------------

For generating samples inside the Fisher-constrained ellipsoid:

uniform_spoked_ellipsoid
^^^^^^^^^^^^^^^^^^^^^^^^

Distributes points uniformly along radial spokes inside a D-dimensional ellipsoid.

uniform_random_ellipsoid
^^^^^^^^^^^^^^^^^^^^^^^

Generates random points uniformly distributed inside an ellipsoid.

These functions are useful for:

- **Initial sample placement**: Generate starting points for MCMC or iterative schemes
- **Importance sampling**: Draw samples from the Gaussian approximation
- **Visualization**: Show the Fisher-constrained parameter space

Parameter Handling
-----------------

The module supports parameter aliases through ``update_params_ip`` and ``update_params_norm_hoff``:

- Direct parameters: ``m1``, ``m2``, ``incl``, etc.
- Derived parameters: ``Mc`` (chirp mass), ``eta`` (symmetric mass ratio)

This allows flexible grid definitions using whichever parameterization is most convenient.

Use Cases
---------

1. **Rapid Fisher Estimation**: Avoid expensive numerical derivatives by fitting to a pre-computed grid
2. **Grid-based Likelihood**: Use the fitted Fisher as a proposal or importance sampling weights
3. **Convergence Diagnostics**: Compare effective Fisher to numerical Fisher to verify accuracy
4. **Initial Sample Placement**: Use ellipsoid bounds to initialize parameter estimation runs
