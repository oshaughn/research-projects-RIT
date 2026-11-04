.. py:module:: GWSignal

.. _gwsignal_api:

GWSignal API Reference
======================

Overview
--------

This module provides a wrapper around the ``gwsignal`` library (part of LALSimulation), providing a clean interface for generating high-fidelity gravitational wave waveforms. The gwsignal interface is particularly useful when working with surrogate models and advanced waveform approximants.

For a high-level discussion of waveform generators and how GWSignal fits into the broader RIFT architecture, see :doc:`../../physics/GWSignal`.

Functions
---------

.. py:function:: hlmoft(P, Lmax=2, approx_string=None, no_trust_align_method=None, internal_phase_shift=π/2, **kwargs)

   Generates time-domain harmonic modes using the gwsignal library.

   :param P: Waveform parameters object.
   :type P: ChooseWaveformParams
   :param Lmax: Maximum l-mode to generate. Default is 2.
   :type Lmax: int
   :param approx_string: Approximant string. If None, P.approx is used.
   :type approx_string: str, optional
   :param no_trust_align_method: If 'peak', shifts epoch to the peak of the total signal power.
   :type no_trust_align_method: str, optional
   :param internal_phase_shift: Phase shift applied to the modes. Default is π/2.
   :type internal_phase_shift: float
   :param kwargs: Additional arguments (e.g., 'lmax_nyquist').
   :returns: A dictionary mapping (l, m) to LAL COMPLEX16TimeSeries objects.
   :rtype: dict

.. py:function:: hlmoff(P, Lmax=2, approx_string=None, **kwargs)

   Generates Fourier-transformed harmonic modes.

   :param P: Waveform parameters object.
   :type P: ChooseWaveformParams
   :param Lmax: Maximum l-mode to generate. Default is 2.
   :type Lmax: int
   :param approx_string: Approximant string. If None, P.approx is used.
   :type approx_string: str, optional
   :param kwargs: Additional arguments passed to hlmoft.
   :returns: A dictionary mapping (l, m) to Fourier-transformed series.
   :rtype: dict

.. py:function:: std_and_conj_hlmoff(P, Lmax=2, approx_string=None, **kwargs)

   Generates both the Fourier-transformed harmonic modes and their complex conjugates.

   :param P: Waveform parameters object.
   :type P: ChooseWaveformParams
   :param Lmax: Maximum l-mode to generate. Default is 2.
   :type Lmax: int
   :param approx_string: Approximant string. If None, P.approx is used.
   :type approx_string: str, optional
   :param kwargs: Additional arguments passed to hlmoft.
   :returns: A tuple (hlmsF, hlms_conj_F) where both are dictionaries mapping (l, m) to Fourier-transformed series.
   :rtype: tuple

See Also
--------

- :doc:`../../physics/GWSignal` - High-level discussion of the GWSignal interface
- :doc:`../../physics/index` - Overview of waveform generation in RIFT