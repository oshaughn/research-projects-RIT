GWSignal Interface
===================

The ``GWSignal.py`` module acts as a wrapper around the ``gwsignal`` library. It is designed to avoid some of the stdout noise and slow import times associated with the base library while providing a clean interface for generating both time-domain waveforms and their harmonic decompositions.

Overview
--------

This interface allows RIFT to leverage the wide array of waveform models available in the ``gwsignal`` ecosystem. It handles the conversion between RIFT's internal parameter structures (``ChooseWaveformParams``) and the Python dictionaries required by the ``gwsignal`` generators.

Core Functions
---------------

**Harmonic Decomposition**

- ``hlmoft(P, Lmax=2, ...)``: The core function for generating harmonic modes. It takes a parameter set ``P`` and returns a dictionary of LAL ``COMPLEX16TimeSeries`` objects for each $(l, m)$ mode up to ``Lmax``.
    - It handles the necessary coordinate conversions (e.g., masses to solar masses).
    - It applies required tapering and phase shifts.
    - It can optionally adjust the waveform epoch based on the peak of the signal.
- ``hlmoff(P, ...)``: A convenience wrapper around ``hlmoft`` that returns Fourier-transformed harmonic modes.
- ``std_and_conj_hlmoff(P, ...)``: Returns both the Fourier-transformed modes and their complex conjugates.

**Time-Domain Waveforms**

- ``hoft(P, ...)``: Generates a real-valued time-domain waveform. It handles the projection of the $h_+$ and $h_\times$ polarizations onto a specific detector strain based on the sky location and orientation.
- ``complex_hoft(P, ...)``: Generates a complex-valued time-domain waveform ($h_+ + i h_\times$), which is useful for certain internal RIFT calculations.

Implementation Details
------------------------

- **Coordinate Handling**: The module carefully converts RIFT's internal units to the units expected by ``gwsignal`` (e.g., using ``astropy.units``).
- **Tapering**: Since ``gwsignal`` output is often not tapered, the module manually applies a cosine taper to the start of the segments to avoid spectral leakage.
- **Epoch Alignment**: The module supports an ``align_method='peak'`` option, which shifts the waveform epoch to coincide with the peak of the total power $\sum |h_{lm}|^2$.
