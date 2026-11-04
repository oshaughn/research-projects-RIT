ROM Waveform Manager
====================

The ``ROMWaveformManager.py`` module provides an interface to Reduced Order Models (ROMs), which are highly efficient surrogate models of gravitational-wave waveforms. These are used in RIFT to significantly accelerate the generation of waveforms in iterative sampling pipelines.

Overview
--------

The ROM manager handles the loading and evaluation of surrogate models (e.g., NRSur7dq4, NRHybSur). It transforms internal RIFT parameters into the specific coordinate systems required by the surrogate models and manages the reconstruction of waveforms using a reduced basis of functions.

Core Classes and Functions
--------------------------

**WaveformModeCatalog**

The primary class for managing ROM models. It handles:
- **Model Loading**: Dynamically loads surrogates based on the specified group and parameter set.
- **Coordinate Conversion**: Maps RIFT's ``ChooseWaveformParams`` to surrogate-specific parameters (e.g., mass ratio $q$).
- **Waveform Reconstruction**: 
    - ``hlmoft``: Generates harmonic modes $h_{lm}(t)$ in physical units.
    - ``coefficients``: Computes the ROM coefficients for a given parameter set.
    - ``basis_oft``: Generates the dimensionless basis functions.
- **Hybridization**: Supports hybridization of ROM waveforms with other models using the ``LALHybrid`` library.

**Utility Functions**

- ``ConvertWPtoSurrogateParams``: A family of functions (Aligned, Precessing, Full) that map RIFT parameters to the inputs expected by various surrogate models.
- ``CreateCompatibleComplexOverlap``: Ensures that overlap integrals are computed with consistent frequency grids and Nyquist limits.

Implementation Details
------------------------

- **Basis Functions**: The manager can either use the surrogate's internal evaluation or reconstruct the signal using a precomputed basis of functions, which is often faster for repeated evaluations.
- **Tapering**: Implements manual tapering at both the start and end of the ROM waveforms to ensure spectral purity.
- **Symmetry**: Leverages reflection symmetry $(l, m) \to (l, -m)$ to reduce the number of modes that need to be explicitly loaded from the surrogate.
