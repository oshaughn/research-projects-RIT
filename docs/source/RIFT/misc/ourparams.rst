ourparams
===========

The ``RIFT.misc.ourparams`` module serves as the central configuration hub for the RIFT pipeline. It manages the transition from command-line arguments to internal software settings, ensuring consistency across different modules.

Main Functionalities
--------------------

The module provides two primary entry points:
1. ``ParseStandardArguments()``: Processes command-line inputs using ``argparse`` and returns a namespace of options along with a debugging message dictionary.
2. ``PopulatePrototypeSignal()``: Uses the parsed options to construct a fiducial signal (``Psig``), which is used as a reference for sampling and likelihood calculations.

Configuration Categories
------------------------

The parameters managed by this module can be grouped into several logical categories:

Pipeline Control
~~~~~~~~~~~~~~~~
These settings determine the effort and precision of the inference process.
- **Iterations (``--Niter``)**: Total number of MC iterations.
- **Effective Samples (``--Neff``)**: The target number of independent samples to be achieved.
- **Convergence Tests (``--convergence-tests-on``)**: Enables automated checks for sampler convergence.

Likelihood & Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~
RIFT allows users to switch between different likelihood computation strategies.
- **Likelihood Type**: Options include ``MargTdisc_array`` (default), ``vectorized``, and ``vectorized_noloops`` (GPU optimized).
- **Interpolation Control (``--skip-interpolation``)**: Allows bypassing the interpolation layer for diagnostic purposes.

Astrophysical & Waveform Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Settings that define the signal model.
- **Approximants (``--approx``)**: Specifies the waveform model (e.g., ``TaylorT4``).
- **Physical Bounds**: Defines the range for mass, spin, and distance.
- **Fixed Parameters**: Using flags like ``--fix-distance``, users can pin specific parameters to injected values to reduce the dimensionality of the search.

Data & Infrastructure
~~~~~~~~~~~~~~~~~~~~~
Handles the loading of external data and environment setup.
- **LIGO Cache (``--cache-file``)**: Path to the GWOSC or internal cache file.
- **PSDs (``--psd-name``, ``--psd-file``)**: Configuration for the Power Spectral Density of the detectors.
- **NR Catalogs**: Integration with Numerical Relativity (NR) catalogs via ``--NR-signal-group``.

Sampling & Priors
~~~~~~~~~~~~~~~~~
Controls how the parameter space is explored.
- **Sky Location**: Support for uniform sampling, injected locations, or external skymaps (via ``--sampling-prior-use-skymap``).
- **Distance Priors**: Implements specific distance sampling distributions (e.g., quadratic priors).

API Reference
-------------

.. automodule:: RIFT.misc.ourparams
    :members:
    :undoc-members:
    :show-inheritance:
