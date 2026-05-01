convergence_test_samples
==========================

The ``convergence_test_samples`` executable compares two sets of posterior samples to determine if the results have converged. It is primarily used in iterative pipelines to signal when further iterations are no longer significantly changing the results.

Overview
--------

In iterative Bayesian analysis, it is crucial to know when the posterior distributions have stabilized. ``convergence_test_samples`` implements several statistical tests to quantify the difference between two sample sets (e.g., from iteration :math:`N` and iteration :math:`N+1`). If the difference is below a specified threshold, the tool can trigger a "success" signal to terminate the pipeline.

Usage
------

Basic Command Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   convergence_test_samples.py [options]

Primary Options
~~~~~~~~~~~~~~~

**Input Samples**

* ``--samples``: Filenames of the two sample sets to compare. Must be provided twice.
* ``--parameter``: List of parameters to be used in the convergence test.

**Test Configuration**

* ``--method``: The statistical method used to compare the samples. Options include:
    * ``lame``: Computes a multivariate Gaussian estimate (mean and covariance) for both sets and calculates the Kullback-Leibler (KL) divergence.
    * ``KS_1d``: Performs a one-dimensional Kolmogorov-Smirnov (KS) test on the first specified parameter.
    * ``JS``: Calculates the additive Jensen-Shannon (JS) divergence across all specified parameters using KDE estimates.
* ``--threshold``: The value below which the samples are considered "converged."

**Pipeline Integration**

* ``--iteration``: The current iteration number.
* ``--iteration-threshold``: The iteration number before which the convergence test is skipped.
* ``--write-file-on-success``: The path to a file (e.g., ``INTRINSIC_CONVERGED``) that is created if the test passes.
* ``--always-succeed``: Forces the tool to return a success exit code regardless of the test result.

Functional Logic
----------------

1. **Sample Loading**: Reads two sets of samples from the provided files.
2. **Pre-processing**:
    * Expands samples to include standard derived fields (e.g., :math:`\chi_{eff}`, :math:`\chi_{minus}`).
    * Ensures consistent parameter names across both sets.
3. **Divergence Calculation**:
    * Based on the ``--method``, it calculates a scalar value representing the "distance" between the two distributions.
    * For ``lame``, it uses the closed-form KL divergence for multivariate Gaussians.
    * For ``JS``, it uses random resampling and Kernel Density Estimation (KDE) to compute the JS divergence for each parameter.
4. **Thresholding**:
    * Compares the calculated distance to the ``--threshold``.
    * If ``distance < threshold``, the samples are considered converged.
5. **Exit Codes**:
    * The tool is designed to work with Condor DAGs. In some configurations, it returns a failure exit code (1) when the test succeeds to terminate the DAG.

Output Details
--------------

The tool prints the calculated divergence value to the standard output. If the convergence criteria are met, it creates the file specified by ``--write-file-on-success``.
