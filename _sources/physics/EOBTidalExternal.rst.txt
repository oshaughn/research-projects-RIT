EOB Tidal Waveforms (External)
================================

This module provides an interface to the Bernuzzi EOB (Effective-One-Body) model for generating tidal gravitational-wave waveforms. It acts as a wrapper around an external MATLAB-based implementation, allowing RIFT to incorporate high-fidelity tidal effects into its inference pipelines.

Requirements
-------------

Because this module interfaces with external code, the following dependencies must be configured on the system:

1. **MATLAB**: The core EOB tidal model is implemented in MATLAB.
2. **Execution Script**: The ``run_goeob_standalone.sh`` script must be present in the EOB base directory.
3. **Environment Variables**:
    - ``EOB_BASE``: Path to the directory containing the EOB source code and execution scripts.
    - ``EOB_ARCHIVE``: Path to the directory where generated waveforms are archived to avoid redundant computations.
    - ``MATLAB_BASE``: Path to the MATLAB executable or base installation.

Core Functionality: WaveformModeCatalog
--------------------------------------

The primary interface is the ``WaveformModeCatalog`` class, which handles the generation, caching, and manipulation of tidal harmonic modes.

Initialization
^^^^^^^^^^^^^^^

When initialized, the catalog:
1. Checks for existing archived waveforms in ``EOB_ARCHIVE`` based on the waveform parameters (masses, tidal lambdas, and minimum frequency).
2. If no archive exists, it invokes the external MATLAB script via ``run_goeob_standalone.sh``.
3. Loads the resulting harmonic modes from disk (e.g., ``test_h22.dat``).
4. Applies necessary scaling factors (e.g., symmetric mass ratio $\nu$) and performs phase unwinding.

Key Methods
^^^^^^^^^^^^^

- ``hlmoft(force_T=False, deltaT=1./16384, ...)``:
    Generates a dictionary of complex-valued time-domain modes $h_{lm}(t)$. It interpolates the raw EOB data onto a uniform time grid and applies physical scaling based on the source distance.
- ``hlmoff(...)``:
    Performs a Fourier transform on the time-domain modes to provide frequency-domain representations $h_{lm}(f)$.
- ``real_hoft(...)``:
    Projects the harmonic modes onto a specific detector's strain, producing a real-valued time series $h(t)$ based on the detector's orientation ($\theta, \phi, \psi$).
- ``complex_hoft(...)``:
    Combines the $h_{lm}$ modes into a single complex time series $\Psi_4$ using spin-weighted spherical harmonics.

Tidal Mode Support
------------------

The interface supports a range of harmonic modes, typically including:
- $(2, \pm 2), (2, \pm 1), (2, 0)$
- $(3, \pm 3), (3, \pm 2), (3, \pm 1)$

Workflow Summary
----------------

1. **Parameter Input**: Pass a parameter object containing $m_1, m_2, \lambda_1, \lambda_2$.
2. **External Generation**: The module converts $\lambda$ to $\kappa$ and calls the Bernuzzi MATLAB code.
3. **Data Retrieval**: Raw data files are read from the archive or the current run directory.
4. **Physical Scaling**: Dimensionless outputs are scaled by $M_{total}$ and $1/d_L$.
5. **Projection**: Modes are combined into a detector-specific strain for likelihood evaluation.
