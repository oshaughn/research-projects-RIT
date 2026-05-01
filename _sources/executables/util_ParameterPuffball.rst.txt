util_ParameterPuffball
=======================

The ``util_ParameterPuffball.py`` executable is used to "puff up" a distribution of parameters. It reads an existing set of parameters, assesses their covariance, and generates a new, expanded set of parameters by adding random noise based on that covariance. This is typically used to create a broader sampling distribution for subsequent integration or testing steps.

Overview
--------

In many RIFT workflows, it is necessary to move from a highly constrained set of parameters (e.g., a maximum likelihood point or a narrow grid) to a wider distribution that still respects the underlying correlations of the parameters. ``util_ParameterPuffball.py`` achieves this by calculating the covariance matrix of the input parameters and sampling from a multivariate normal distribution to perturb the points.

Usage
-----

Basic Command Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python util_ParameterPuffball.py [options]

Primary Options
~~~~~~~~~~~~~~

**Input and Output**

- ``--inj-file``: Path to the input XML file containing the parameter set.
- ``--inj-file-out``: Path to the output XML file (default: ``output-puffball``).
- ``--puff-factor``: Scaling factor for the covariance matrix. A value of 1.0 maintains the original covariance; values > 1.0 "puff up" the distribution (default: 1.0).
- ``--approx-output``: Waveform approximant to use when writing the output XML (default: ``SEOBNRv2``).

**Parameter Selection**

- ``--parameter``: Parameters to be used for covariance assessment and perturbation. These are the "fitting" parameters.
- ``--no-correlation``: A list of parameter pairs (e.g., ``"['mc','eta']"``) whose mutual correlation should be eliminated by zeroing out the corresponding covariance matrix terms.
- ``--random-parameter``: Parameters to be sampled uniformly at random over a specified range, uncorrelated with others.
- ``--random-parameter-range``: The ranges for the ``--random-parameter`` variables (passed as string evaluations of Python lists).

**Constraints and Filtering**

- ``--downselect-parameter``: Parameters used to filter out points that fall outside a specific range.
- ``--downselect-parameter-range``: The allowed ranges for the downselection parameters.
- ``--mc-range``, ``--eta-range``, ``--mtot-range``: Convenient shortcuts for downselecting based on chirp mass, eta, or total mass.
- ``--reflect-parameter``: Parameters that should be "reflected" back into their allowed range if they exceed the boundaries.
- ``--force-away``: If > 0, uses the inverse covariance matrix to compute a Mahalanobis distance and discards points that are too close to existing points.
- ``--enforce-duration-bound``: Maximum allowed waveform duration. Points producing signals longer than this are discarded.
- ``--fail-if-empty``: Causes the tool to exit with an error if no points survive the filtering process.

Functional Logic
----------------

1. **Coordinate Extraction**: Reads the input XML and extracts the specified ``--parameter`` values. Physical masses are scaled to solar masses.
2. **Transformation**: Performs a natural log transformation on parameters that are strictly positive (e.g., tidal parameters like ``lambda1``, ``lambda2``, ``LambdaTilde``) to ensure perturbed values remain positive.
3. **Covariance Calculation**:
    - Computes the covariance matrix of the extracted parameters.
    - Applies the ``--puff-factor`` scaling.
    - Handles singular matrices by adding a small regularization term to the diagonal of the inverse covariance (pseudo-inverse).
    - Zeroes out correlations for pairs specified via ``--no-correlation``.
4. **Perturbation**:
    - Samples random offsets from a multivariate normal distribution based on the modified covariance matrix.
    - Adds these offsets to the original parameters.
5. **Boundary Handling**:
    - **Reflection**: Reflects parameters specified by ``--reflect-parameter`` back into their defined ranges.
    - **Eta Constraint**: Specifically ensures ``eta`` remains within the physical range [0, 0.25].
6. **Filtering (Downselection)**:
    - Converts parameters to a common coordinate system to check constraints.
    - Discards points that violate ``--downselect-parameter-range`` or the predefined physical bounds for masses and eta.
    - Discards points that violate the ``--enforce-duration-bound`` or the unit spin magnitude constraint ($\chi \le 1$).
    - Optionally discards points too close to the original set using the ``--force-away`` Mahalanobis distance check.
7. **Finalization**: 
    - Undoes the log transformations.
    - Applies random values to parameters specified by ``--random-parameter``.
    - Writes the resulting parameter set to the output XML.

Output Details
---------------

The tool prints the updated covariance matrix and the final count of exported points. The output is a standard `ligolw` XML file containing the "puffed" parameter set.
