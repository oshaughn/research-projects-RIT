plot_posterior_corner
=======================

The ``plot_posterior_corner`` executable generates "corner plots" (also known as triangle plots) to visualize the posterior probability distributions of intrinsic parameters.

Overview
--------

This tool takes posterior samples (typically produced by ``util_ConstructIntrinsicPosterior_GenericCoordinates``) or intermediate grid data (composite files) and creates multi-panel figures showing:
1. **1D Marginal Distributions**: Histograms/KDEs on the diagonal.
2. **2D Joint Distributions**: Contour plots or scatter plots on the off-diagonal.

It is used to assess parameter correlations and the constraints placed on the physical properties of the gravitational wave source.

Usage
-----

Basic Command Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   plot_posterior_corner.py [options]

Primary Options
~~~~~~~~~~~~~~

Input Data
-----------
- ``--posterior-file``: Filename of the posterior samples ``*.dat`` file. Can be specified multiple times to compare different runs.
- ``--posterior-label``: Label for each corresponding posterior file.
- ``--composite-file``: Filename of an intermediate grid file (composite).
- ``--truth-file``: File containing the true parameters to be plotted as reference lines.

Parameter Selection
-------------------
- ``--parameter``: List of parameters to include in the plot (e.g., ``mc``, ``eta``, ``lambda1``).
- ``--parameter-log-scale``: Parameters to be plotted on a $\log_{10}$ scale.
- ``--change-parameter-label``: Custom label for a parameter (format ``name=string``).

Plotting Controls
-----------------
- ``--use-legend``: Display a legend for multiple posterior files.
- ``--use-title``: Add a title to the figure.
- ``--pdf``: Export the plot to PDF instead of PNG.
- ``--publication``: Use specific figure sizes optimized for journals (e.g., PRD).
- ``--use-smooth-1d``: Enable smoothed 1D distributions.
- ``--plot-1d-extra``: Generate separate 1D PDF and CDF plots for each parameter.

Bounds and Cuts
---------------
- ``--bind-param`` & ``--param-bound``: Impose specific bounds on parameters for the plot.
- ``--lnL-cut``: Cut points with log-likelihood below a certain threshold.
- ``--sigma-cut``: Eliminate points with high fit error from the plot.

Functional Logic
----------------

1. **Data Loading**: Loads the specified posterior or composite files. It handles various formats, including tidal and eccentricity-based files.
2. **Coordinate Transformation**: 
    - Expands samples to include derived quantities (e.g., $\chi_{eff}$ from component spins).
    - Applies periodicity (e.g., $\psi \pmod \pi$).
    - Handles log-scaling if requested.
3. **Range Determination**: Automatically calculates the plotting range for each parameter based on the sample distribution or predefined ``special_param_ranges``.
4. **Corner Plot Generation**:
    - Uses the ``corner`` library to generate the multi-panel figure.
    - For multiple files, it iterates through the lists, assigning colors and linestyles.
    - If composite files are used, it can color-code the points by their log-likelihood.
5. **Reference Lines**: If a truth file is provided, it draws vertical/horizontal lines at the true parameter values.
6. **Export**: Saves the resulting figure to a file named ``corner_<params>.png`` (or ``.pdf``).

Output Details
---------------

The output is a high-resolution image file. The diagonal panels show the 1D marginalized posterior for each parameter, and the off-diagonal panels show the 2D joint posterior for pairs of parameters.
