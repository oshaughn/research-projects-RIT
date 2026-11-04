util_ConstructIntrinsicPosterior_GenericCoordinates
=====================================================

The ``util_ConstructIntrinsicPosterior_GenericCoordinates`` executable loads marginalized likelihood data (typically from ``integrate_likelihood_extrinsic``), fits the peak using various interpolation methods, and generates posterior samples for the intrinsic parameters.

Overview
--------

This tool represents the "Construct Intrinsic Posterior" (CIP) stage of the RIFT pipeline. After ``integrate_likelihood_extrinsic`` has computed the marginalized likelihood for a grid of intrinsic parameters, ``util_ConstructIntrinsicPosterior_GenericCoordinates`` takes that grid and:
1. Fits a continuous surface to the discrete log-likelihood ($\ln \mathcal{L}$) data.
2. Uses a Monte Carlo sampler to draw posterior samples based on the fit and a specified prior.
3. Performs coordinate transformations and adds derived quantities to the resulting samples.

Usage
-----

Basic Command Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   util_ConstructIntrinsicPosterior_GenericCoordinates [options]

Primary Options
~~~~~~~~~~~~~~

Input/Output
------------
- ``--fname``: Filename of the ``*.dat`` file containing the marginalized likelihood data (standard ILE output).
- ``--fname-output-samples``: Filename for the resulting posterior samples (default: ``output-ILE-samples``).
- ``--fname-output-integral``: Filename for the resulting integral result (default: ``integral_result``).
- ``--n-output-samples``: Number of posterior samples to generate (default: 3000).

Fitting Configuration
---------------------
- ``--fit-method``: Method used to interpolate the likelihood surface. Options include:
    - ``rf``: Random Forest (default)
    - ``gp``: Gaussian Process
    - ``quadratic``: Quadratic fit to the peak
    - ``polynomial``: General polynomial fit
    - ``kde``: Kernel Density Estimation
- ``--fit-order``: Order of the polynomial fit (default: 2 for quadratic).
- ``--lnL-offset``: Threshold relative to the peak ($\ln \mathcal{L}_{max} - \text{offset}$) used to filter data points for the fit.

Parameter and Prior Configuration
---------------------------------
- ``--parameter``: List of parameters to be used as fitting parameters and varied for the posterior.
- ``--parameter-implied``: Parameters used in the fit but not independently varied for MC.
- ``--mc-range``, ``--eta-range``, ``--mtot-range``: Manual ranges for the Monte Carlo sampling to avoid wasting time in low-probability regions.
- ``--aligned-prior``: Prior for aligned spins (``uniform``, ``volumetric``, ``alignedspin-zprior``).
- ``--transverse-prior``: Prior for transverse spins (``uniform``, ``uniform-mag``, ``taper-down``, ``sqrt-prior``, etc.).
- ``--prior-gaussian-mass-ratio``: Applies a Gaussian mass ratio prior.

Advanced Analysis
-----------------
- ``--downselect-parameter`` & ``--downselect-parameter-range``: Used to eliminate grid points that fall outside specified ranges.
- ``--fname-lalinference``: Filename of LALInference posterior samples to overlay on corner plots for comparison.

Functional Logic
----------------

1. **Data Loading**: Reads the marginalized likelihood grid from the input file.
2. **Downselection**: Filters out points based on the ``--downselect-parameter`` criteria or EOS-based constraints.
3. **Fitting**:
    - Applies the chosen ``--fit-method`` (e.g., Gaussian Process or Random Forest) to create a continuous representation of the $\ln \mathcal{L}$ surface.
    - Optionally shifts the $\ln \mathcal{L}$ values to prevent numerical overflow.
4. **Sample Generation**:
    - Uses an integrator (like ``adaptive_cartesian`` or ``GMM``) to draw samples proportional to $\mathcal{L}_{fit}(\theta) \times \text{Prior}(\theta)$.
    - This converts the fitted surface into a set of posterior samples.
5. **Post-Processing**:
    - Transforms samples into requested coordinate systems.
    - Calculates derived parameters (e.g., $\chi_{eff}$, $\chi_{minus}$).
    - Writes the samples to the output file and computes the total evidence (integral).

Output Details
---------------

The primary output is a structured numpy array (or XML/HDF5 file) containing the posterior samples of the intrinsic parameters. It also produces an integral result representing the marginalized likelihood over the intrinsic space.
