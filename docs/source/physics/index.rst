Waveform Generation
====================

This section describes the tools RIFT uses to generate gravitational-wave signals. The primary objective is to provide a conceptual and workflow-oriented guide to how waveforms are produced, specifically focusing on the generation of spin-weighted spherical harmonic decompositions $h_{lm}Y_{lm}$.

Waveform Interfaces
-------------------

The RIFT physics module provides a set of interfaces to external waveform generators. RIFT utilizes several interfaces to produce waveforms, depending on the required speed and accuracy:

1. **LALSimulation Utilities**: The base layer for many waveform calls, providing low-level access to the LAL library.
2. **GWSignal Interface**: A wrapper around the ``gwsignal`` library, providing a clean interface for high-fidelity waveforms.
3. **ROM Waveform Manager**: A specialized interface for Reduced Order Models (ROMs) used in high-performance iterative pipelines.
4. **EOB Tidal External**: Interface to the Bernuzzi EOB model for tidal waveforms, providing high-fidelity tidal effects via external MATLAB code.
5. **EOB Tidal External C**: C-based implementation of EOB tidal waveforms, offering faster generation without MATLAB dependencies.
6. **Monotonic Spline**: Utility for creating and evaluating monotonic interpolating splines, used throughout RIFT for waveform and EOS interpolation.
7. **Precessing Fisher Matrix**: Tools for computing the Fisher information matrix in the presence of precessing spin configurations.
8. **Effective Fisher**: Efficient Fisher matrix calculations for rapid parameter estimation and covariance estimation.

Conceptual Overview
-------------------

Waveform Characterization
^^^^^^^^^^^^^^^^^^^^^^^^^

Waveforms in RIFT are primarily characterized by their spin-weighted spherical harmonic decompositions $h_{lm}(t)$. Most operations are performed in the time domain, characterizing the signal versus time ($t$).

Data Format
^^^^^^^^^^^

The standard representation for these waveforms is a Python dictionary (often referred to as an ``hlmoft`` dictionary):

- **Key**: A tuple $(l, m)$ representing the harmonic mode.
- **Value**: A ``lal.COMPLEX16TimeSeries`` object.

The ``COMPLEX16TimeSeries`` stores the complex amplitude of the mode at discrete time steps, typically handled in dimensionless units using ``lalsimutils.lsu_DimensionlessUnit``.

Key Functional Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

- **Time-Domain Generation (``hlmoft``)**: The primary operation to produce the $h_{lm}(t)$ dictionary.
- **Fourier Transformation (``hlmoff``)**: Converting time-domain modes to frequency-domain modes.
- **Projection (``hoft``)**: Projecting harmonic modes onto a specific detector strain (producing a real-valued time series).

**Post-processing**

- **Tapering**: Application of cosine tapers to avoid spectral leakage.
- **Epoch Alignment**: Shifting the time axis (e.g., aligning to the peak of total power) to ensure consistent signal windowing.

For detailed guides on each generator interface, see the following:

.. toctree::
   :maxdepth: 1

   LALSimUtils
   GWSignal
   ROMWaveformManager
   EOBTidalExternal
   EOBTidalExternalC
   MonotonicSpline
   PrecessingFisherMatrix
   effectiveFisher
