LALSimulation Utilities
=======================

The ``lalsimutils.py`` module serves as the low-level utility layer for waveform generation in RIFT. It provides thin wrappers around the LAL and LALSimulation libraries, facilitating the creation and manipulation of gravitational-wave time series.

Overview
--------

LALSimUtils is designed to handle the boilerplate associated with LAL library calls, ensuring that RIFT can easily generate basic waveforms and perform essential time-series operations without repeatedly implementing the LAL-specific API.

Data Structures
----------------

**ChooseWaveformParams**
The primary data container used to characterize gravitational-wave compact binary merger events. It encapsulates all parameters required by LALSimulation's waveform generators, including:

- **Masses**: Individual masses (``m1``, ``m2``).
- **Spins**: Cartesian spin components for both bodies (``s1x, s1y, s1z``, ``s2x, s2y, s2z``).
- **Tidal Parameters**: Love numbers (``lambda1``, ``lambda2``) for neutron stars.
- **Orientation**: Inclination (``incl``), polarization (``psi``), and sky location (``theta, phi``).
- **Orbital Parameters**: Eccentricity and mean per ano.
- **LAL Integration**: Includes a ``to_lal_dict()`` method that converts these parameters into a ``lal.Dict``, which is the required input format for low-level LALSimulation functions.

Coordinate Management
----------------------

**convert_waveform_coordinates**
A high-performance utility for transforming arrays of binary parameters between different coordinate charts. This is critical for efficient sampling and likelihood evaluations where parameters may be sampled in one chart (e.g., chirp mass and symmetric mass ratio) but required by generators in another (e.g., individual masses).

- **Vectorized Transformations**: Operates on NumPy arrays, avoiding the overhead of creating individual parameter objects for every sample.
- **Supported Transformations**:
    - Chirp mass (``mc``) and symmetric mass ratio (``eta``) $\leftrightarrow$ Individual masses (``m1, m2``).
    - Individual spins (``s1z, s2z``) $\leftrightarrow$ Effective spin (``xi``) and ``chiMinus``.
    - Eccentricity and mean per ano transformations.
    - Source-frame to detector-frame mass conversions via redshift.
- **Flexibility**: Allows specifying the input and output coordinate names, automatically copying identity transformations and calculating only the required changes.

Waveform Generation
-------------------

LALSimUtils provides the original implementation of the waveform generation protocols used throughout RIFT. These functions directly interface with the LALSimulation library to produce gravitational-wave signals.

Core Functions
^^^^^^^^^^^^^^

- ``hlmoft(P, ...)``: Generates the spin-weighted spherical harmonic decompositions $h_{lm}(t)$ in the time domain. It returns a dictionary mapping $(l, m)$ to ``lal.COMPLEX16TimeSeries`` objects.
- ``hlmoff(P, ...)``: Produces the Fourier-transformed harmonic modes $h_{lm}(f)$.
- ``hoft(P, ...)``: Generates a real-valued time-domain waveform projected onto a specific detector strain.
- ``hoft_from_hlm(hlms, P, ...)``: A utility to project an existing set of harmonic modes into a detector strain, allowing for flexible post-processing of the signal.

As the prototype for the ``GWSignal`` interface, these functions define the la-standard API for waveform generation within the RIFT framework.

General Utility Functions
-------------------------

LALSimUtils also provides essential helper functions for signal processing:

- **Time-Series Creation**: Wrappers for creating ``lal.COMPLEX16TimeSeries`` and ``lal.Real16TimeSeries`` objects.
- **Unit Management**: Management of physical units (e.g., ``lsu_DimensionlessUnit``) to ensure consistency across different waveform generators.
- **LAL Integration**: Thin wrappers for LALSimulation's core waveform models and coordinate transformations.
- **Signal Processing**: Implementation of tapering and Fourier transforms used by higher-level interfaces like ``GWSignal`` and ``ROMWaveformManager``.

Implementation Details
-----------------------

The module focuses on efficiency and minimal overhead, acting as a bridge between the high-level RIFT physics parameters and the low-level C-based LAL libraries.
