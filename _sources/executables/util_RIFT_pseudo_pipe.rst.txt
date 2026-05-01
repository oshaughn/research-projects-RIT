#############################
util_RIFT_pseudo_pipe.py
#############################

Overview
========

``util_RIFT_pseudo_pipe.py`` is the primary user-facing interface for RIFT analyses.
It provides a command-line interface similar to ``lalinference_pipe`` for running
parameter estimation on gravitational wave events.

This executable sets up RIFT analyses, manages prior configurations, handles
GraceDB event retrieval, and coordinates the overall analysis workflow.

Location
=======

``MonteCarloMarginalizeCode/Code/bin/util_RIFT_pseudo_pipe.py``

Usage
=====

Run on a GraceDB event::

    python util_RIFT_pseudo_pipe.py --gracedb-id G329483 --approx NRHybSur3dq8 --l-max 4

Reproduce a previously run analysis using an INI file::

    python util_RIFT_pseudo_pipe.py --use-ini test.ini --use-coinc coinc.xml --use-rundir test --use-online-psd-file psd.xml.gz

With online PSD file::

    python util_RIFT_pseudo_pipe.py --gracedb-id G123456 --approx IMRPhenomXHM --use-online-psd-file psd.xml.gz

Command-Line Arguments
====================

Event Selection
---------------

.. program:: util_RIFT_pseudo_pipe.py

.. option:: --gracedb-id <id>

   GraceDB event ID to analyze. This is the primary way to specify an event.

.. option:: --event-time <time>

   Event time (GPS). Must be used with ``--manual-initial-grid``.

.. option:: --use-ini <file>

   INI file for parsing. Intended to reproduce lalinference_pipe functionality.
   Full path recommended.

.. option:: --use-rundir <path>

   Working directory. Must be an absolute path.

.. option:: --use-coinc <file>

   Coincidence XML file.

Waveform Configuration
--------------------

.. option:: --approx <name>

   Waveform approximant (REQUIRED). Examples: ``NRHybSur3dq8``, ``IMRPhenomXHM``.

.. option:: --l-max <int>

   Maximum l for spherical harmonics (default: 2).

.. option:: --use-gwsurrogate

   Use gwsurrogate instead of lalsuite.

.. option:: --use-gwsignal

   Use gwsignal interface.

Spin Configuration
------------------

.. option:: --assume-nospin

   Force analysis with zero spin.

.. option:: --assume-precessing

   Force analysis *with* transverse spins.

.. option:: --assume-nonprecessing

   Force analysis *without* transverse spins.

.. option:: --assume-matter

   Force analysis *with* matter effects (BNS).

.. option:: --assume-matter-eos <eos>

   Force analysis *with* matter and specify EOS.

.. option:: --assume-highq

   Force high-q strategy, neglecting spin2.

Distance and Marginalization
--------------------------

.. option:: --internal-marginalize-distance

   Marginalize over distance variable.

.. option:: --internal-marginalize-distance-file <file>

   Filename for marginalization file.

.. option:: --internal-distance-max <value>

   Upper limit on distance (required for marginalization).

.. option:: --distance-reweighting

   Reweight posterior samples due to different distance prior (LVK production prior).

Calibration
-----------

.. option:: --calibration-reweighting

   Add job to DAG to reweight posterior samples due to calibration uncertainty.

.. option:: --calibration-reweighting-count <int>

   Number of calibration curves to request when marginalizing (default: 100).

.. option:: --bilby-ini-file <file>

   Pass INI file for calibration reweighting.

.. option:: --bilby-pickle-file <file>

   Bilby pickle file with event settings.

Advanced Options
-----------------

.. option:: --add-extrinsic

   Add extrinsic parameters to analysis.

.. option:: --add-extrinsic-time-resampling

   Add time resampling option (for vectorized calculations).

.. option:: --internal-use-amr

   Use adaptive mesh refinement strategy.

.. option:: --internal-force-iterations <n>

   Override internal guidance on number of iterations.

.. option:: --internal-test-convergence-threshold <value>

   Convergence test threshold (default: 0.02).

Examples
========

Basic BNS Analysis
------------------

Run a basic BNS analysis on a GraceDB event::

    python util_RIFT_pseudo_pipe.py --gracedb-id G329483 --approx NRHybSur3dq8 --l-max 4

Analysis with Calibration Reweighting
-----------------------------------

Run with calibration uncertainty marginalization::

    python util_RIFT_pseudo_pipe.py --gracedb-id G329483 --approx NRHybSur3dq8 --calibration-reweighting --calibration-reweighting-count 200

Precessing Binary Analysis
-----------------------

Run with precessing spins::

    python util_RIFT_pseudo_pipe.py --gracedb-id G329483 --approx IMRPhenomXHM --assume-precessing

Distance Marginalization
---------------------

Run with distance marginalization::

    python util_RIFT_pseudo_pipe.py --gracedb-id G329483 --approx NRHybSur3dq8 --internal-marginalize-distance --internal-distance-max 200

See Also
========

- :doc:`create_event_parameter_pipeline_BasicIteration` - Pipeline generator
- :doc:`../api_reference/index` - API Reference