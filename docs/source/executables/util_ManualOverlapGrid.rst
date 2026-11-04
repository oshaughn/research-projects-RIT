util_ManualOverlapGrid
=======================

The ``util_ManualOverlapGrid`` executable generates a grid of waveform overlaps (matches) between a reference signal (the "base point") and a set of templates varying over a specified parameter space.

Overview
--------

This tool is used to generate a grid of points covering a targeted region, in the data format needed by our codes. It supports variable dimensionality and the same coordinate charts as our main code.

Historically, this tool was used for exploring the likelihood surface and estimating the Fisher information matrix. By calculating the overlap (inner product) between a reference signal and a grid of templates, the tool can determine how the match varies with changes in intrinsic parameters (like mass or spin). Because it computes overlaps between a reference signal h0 and h(\lambda) over a target grid \lambda, these match calculations can be used to estimate measurement accuracy via an empirical Fisher matrix; or to qualitatively characterize expected waveform systematics.

Usage
-----

Basic Command Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   util_ManualOverlapGrid.py [options]

Primary Options
~~~~~~~~~~~~~~

Base Point Configuration
------------------------
- ``--inj``: Path to an inspiral XML file containing the reference signal parameters.
- ``--event``: Event ID of the injection to use from the XML file.
- ``--mass1``, ``--mass2``: Manual specification of component masses (if ``--inj`` is not used).
- ``--s1z``: Manual specification of the first component's z-spin.

Grid Definition
---------------
- ``--parameter``: List of parameters to vary across the grid (e.g., ``LambdaTilde``, ``eta``).
- ``--parameter-range``: Range for each specified parameter (e.g., ``'[0, 1000]'``).
- ``--random-parameter``: Parameters to be sampled randomly rather than on a cartesian grid.
- ``--random-parameter-range``: Ranges for the random parameters.
- ``--grid-cartesian-npts``: Number of points per dimension for the cartesian grid.
- ``--latin-hypercube-sampling``: Use LHS instead of a regular cartesian grid.

Waveform and Overlap Settings
-----------------------------
- ``--approx``: Waveform approximant to use (e.g., ``SpinTaylorT4``).
- ``--fmin``, ``--fmax``: Lower and upper frequency limits for the overlap integral.
- ``--psd-file``: Path to a PSD file to use for the inner product.
- ``--seglen``: Window size for processing.
- ``--use-external-EOB`` / ``--use-external-NR``: Use external EOB or NR waveform generators for templates.

Analysis and Tuning
------------------
- ``--use-fisher``: Perform a quadratic fit to the overlaps to estimate the Fisher matrix.
- ``--reset-grid-via-match``: Automatically tune the parameter ranges so that the grid boundaries correspond to a specific match value.
- ``--match-value``: The match threshold used for range tuning or filtering (default: 0.01).
- ``--skip-overlap``: Generate the grid without actually performing the expensive overlap calculations.

Output
------
- ``--fname``: Base filename for the output (produces both ``.dat`` and ``.xml.gz`` files).
- ``--verbose``: Enable detailed logging of the grid generation process.

Functional Logic
----------------

1. **Base Point Setup**: Initializes the reference signal using either an injection XML file or manual parameter inputs.
2. **Grid Generation**:
    - Creates a coordinate grid based on the ``--parameter`` and ``--parameter-range`` specifications.
    - Optionally incorporates random parameters or uses Latin Hypercube Sampling.
    - Applies "downselection" to remove unphysical points (e.g., violating the Kerr bound $a \le 1$ or requiring $m_1 \ge m_2$).
3. **Overlap Calculation**:
    - For each point in the grid, generates a template waveform.
    - Computes the inner product (overlap) between the template and the reference signal, normalized by their norms.
4. **Fisher Analysis (Optional)**:
    - If ``--use-fisher`` is enabled, fits a quadratic surface to the matches.
    - Extracts the Fisher information matrix from the quadratic coefficients.
    - Optionally resamples the grid using the Fisher matrix to better capture the peak.
5. **Output**: Saves the grid coordinates and their corresponding overlaps to a text file and the template parameters to an XML file.

Output Details
---------------

- ``.dat`` file: A text file where each row contains the parameter values and the resulting match (inner product).
- ``.xml.gz`` file: An inspiral XML file containing the parameters of the templates that were generated.
