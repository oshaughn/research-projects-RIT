integrate_likelihood_extrinsic_batchmode
==========================================

The ``integrate_likelihood_extrinsic_batchmode`` executable integrates the extrinsic parameters of the prefactored likelihood function. It is used to compute the likelihood over a grid of extrinsic parameters (such as sky position and orientation) for a given set of intrinsic parameters.

Overview
--------

This tool is a core component of the RIFT likelihood integration process. It evaluates the likelihood function efficiently by leveraging pre-computed data and, when available, GPU acceleration (via CuPy). It supports various waveform approximants, NR lookup tables, and ROM (Reduced Order Model) bases.

Usage
------

Basic Command Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   integrate_likelihood_extrinsic_batchmode [options]

Primary Options
~~~~~~~~~~~~~~~

**Data and Event Configuration**

* ``-c, --cache-file``: LIGO cache file containing all data needed.
* ``-C, --channel-name``: Instrument and channel name (e.g., ``H1=FAKE-STRAIN``). Can be specified multiple times.
* ``-p, --psd-file``: Instrument and PSD file (e.g., ``H1=H1_PSD.xml.gz``). Can be specified multiple times.
* ``-k, --skymap-file``: Use a skymap stored in a given FITS file.
* ``-x, --coinc-xml``: gstlal_inspiral XML file containing coincidence information.
* ``-E, --event``: Event number used for this run (default: 0).
* ``--n-events-to-analyze``: Number of events to analyze from the XML (default: 1).
* ``--random-event``: Pick a random event from the file.

**Simulation and Grid**

* ``-I, --sim-xml``: XML file containing the parameter grid to be evaluated.
* ``--sim-grid``: ASCII file with labels containing the parameter grid to be evaluated.
* ``-a, --approximant``: Waveform family to use for templates (e.g., ``TaylorT4``).
* ``--l-max``: Include all (l,m) modes with ``l`` less than or equal to this value (default: 2).
* ``-A, --amp-order``: Include amplitude corrections up to this order (default: 0).

**Frequency and Timing**

* ``-f, --reference-freq``: Waveform reference frequency (default: 100 Hz).
* ``--fmin-template``: Waveform starting frequency (default: 40 Hz).
* ``-s, --data-start-time``: GPS start time of the data segment.
* ``-e, --data-end-time``: GPS end time of the data segment.
* ``-F, --fmax``: Upper frequency of signal integration.

**Advanced/Internal Options**

* ``--nr-lookup``: Look up parameters from an NR catalog instead of using the specified approximant.
* ``--rom-use-basis``: Use the ROM basis for inner products.
* ``--rom-integrate-intrinsic``: Integrate over intrinsic variables (requires ``--rom-use-basis``).
* ``--use-gwsignal``: Use the ``gwsignal`` interface for waveforms.
* ``--maximize-only``: Find the single best fitting point after integrating.
* ``--check-good-enough``: Terminate with success if a file ``ile_good_enough`` exists and is non-empty.

Functional Logic
----------------

.. include:: _common_extrinsic_logic.rst

Output Details
--------------

The tool typically outputs the integrated likelihood values. If specific flags like ``--dump-lnL-time-series`` are used, it can provide detailed time-series information for the injected parameters.
