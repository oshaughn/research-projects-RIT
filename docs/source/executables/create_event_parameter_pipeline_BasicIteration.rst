###################################################
create_event_parameter_pipeline_BasicIteration
###################################################

Overview
========

``create_event_parameter_pipeline_BasicIteration`` is the core pipeline writer for RIFT.
It creates condor DAG (Directed Acyclic Graph) workflows for parameter estimation,
generating ILE (Inference Library Engine) jobs combined with fitting/iteration jobs.

This executable manages the overall workflow structure, creating workspaces for each
task and iteration, handling job dependencies, and coordinating the analysis pipeline.

Location
========

``MonteCarloMarginalizeCode/Code/bin/create_event_parameter_pipeline_BasicIteration``

Usage
=====

Basic pipeline creation::

    python create_event_parameter_pipeline_BasicIteration.py --ile-args args_ile.txt --cip-args args_cip.txt --working-directory output/

With calibration reweighting::

    python create_event_parameter_pipeline_BasicIteration.py --ile-args args_ile.txt --cip-args args_cip.txt --working-directory output/ --calibration-reweighting

With puffball refinement::

    python create_event_parameter_pipeline_BasicIteration.py --ile-args args_ile.txt --cip-args args_cip.txt --working-directory output/ --puff-exe util_ParameterPuffball.py --puff-cadence 2

Command-Line Arguments
===================

I/O Configuration
------------------

.. program:: create_event_parameter_pipeline_BasicIteration

.. option:: --working-directory <path>

   Output directory for DAG. All pipeline outputs will be written here.

.. option:: --input-grid <file>

   Initial overlap grid file (default: ``overlap-grid.xml.gz``).

ILE Configuration
-----------------

.. option:: --ile-args <file>

   Filename of args_ile.txt file which holds ILE arguments.

.. option:: --ile-exe <path>

   ILE executable. Will default to ``integrate_likelihood_extrinsic``.

.. option:: --ile-retries <n>

   Number of retry attempts for ILE jobs.

.. option:: --ile-group-subdag

   Group ILE jobs in a subdag.

.. option:: --ile-n-events-to-analyze <n>

   Number of events to analyze (for ILE batchmode).

.. option:: --ile-runtime-max-minutes <n>

   Maximum runtime for ILE jobs in minutes.

CIP (Conditional Inference Pipeline) Configuration
---------------------------------------------------

.. option:: --cip-args <file>

   Filename of args_cip.txt file which holds CIP arguments.

.. option:: --cip-args-list <file>

  _filename of args_cip_list.file with CIP arguments. One CIP_n.sub file per line.

.. option:: --cip-exe <path>

   CIP executable. Defaults to ``util_ConstructIntrinsicPosterior_GenericCoordinates``.

.. option:: --cip-exe-G <path>

   Alternate CIP executable for 'G' iterations.

.. option:: --cip-explode-jobs <n>

   Number of CIP jobs to run in parallel to produce posterior samples.

Puffball Configuration
-----------------------

.. option:: --puff-exe <path>

   Puffball executable (default: ``util_ParameterPuffball.py``).

.. option:: --puff-args <args>

   Puffball arguments. If not specified, puffball is not performed.

.. option:: --puff-cadence <n>

   Puffball frequency: every n iterations (not including 0).

.. option:: --puff-max-it <n>

   Maximum iteration number for puffball.

Calibration Reweighting
---------------------

.. option:: --calibration-reweighting

   Run calibration reweighting on final posterior samples.

.. option:: --calibration-reweighting-count <n>

   Number of calibration marginalization realizations (default: 100).

.. option:: --calibration-reweighting-batchsize <n>

   Group calibration reweighting jobs by fixed size.

Workflow Control
----------------

.. option:: --first-iteration-jumpstart

   No ILE jobs on first iteration. Assumes composite files already exist.

.. option:: --last-iteration-extrinsic

   Extract one set of extrinsic parameters from each intrinsic point (last iteration).

.. option:: --last-iteration-extrinsic-nsamples <n>

   Number of extrinsic samples to construct (default: 3000).

.. option:: --frame-rotation

   Rotate from J frame to L frame (for extrinsic samples).

Plotting and Testing
--------------------

.. option:: --plot-exe <path>

   Plot code executable (default: ``plot_posterior_corner.py``).

.. option:: --test-exe <path>

   Test code for convergence testing.

.. option:: --gridinit-exe <path>

   Initial grid creation executable (default: ``util_ManualOverlapGrid.py``).

Disk and Memory
---------------

.. option:: --general-request-disk <size>

   Request disk for general jobs (default: 10M).

.. option:: --ile-request-disk <size>

   Request disk for ILE jobs.

.. option:: --cip-request-disk <size>

   Request disk for CIP jobs.

.. option:: --ile-request-memory <size>

   Request memory for ILE jobs in Mb.

Eccentricity
------------

.. option:: --use-eccentricity

   Enable eccentricity in the analysis.

.. option:: --use-meanPerAno

   Use mean anomaly parameterization.

.. option:: --use-eccentricity-squared-sampling

   Sample in eccentricity squared.

Examples
========

Basic Pipeline
--------------

Create a basic pipeline::

    python create_event_parameter_pipeline_BasicIteration.py --ile-args args_ile.txt --cip-args args_cip.txt --working-directory output/

With Puffball Refinement
-----------------------

Add puffball refinement every 2 iterations::

    python create_event_parameter_pipeline_BasicIteration.py --ile-args args_ile.txt --cip-args args_cip.txt --working-directory output/ --puff-exe util_ParameterPuffball.py --puff-cadence 2

With Calibration Marginalization
------------------------------

Add calibration uncertainty marginalization with 200 realizations::

    python create_event_parameter_pipeline_BasicIteration.py --ile-args args_ile.txt --cip-args args_cip.txt --working-directory output/ --calibration-reweighting --calibration-reweighting-count 200

Jumpstart First Iteration
-------------------------

Skip ILE jobs on first iteration (use existing composite files)::

    python create_event_parameter_pipeline_BasicIteration.py --ile-args args_ile.txt --cip-args args_cip.txt --working-directory output/ --first-iteration-jumpstart

Workflow Structure
=================

The pipeline creates the following directory structure::

    working_directory/
    ├── iteration_0/
    │   ├── output-ILE-0.xml.gz
    │   ├── *.composite files
    │   └── logs/
    ├── iteration_1/
    │   ├── output-ILE-1.xml.gz
    │   ├── *.composite files
    │   └── logs/
    ...
    ├── all.net (union of all composite files)
    ├── args_ile.txt
    ├── args_cip.txt
    └── *.dag files

Each iteration produces:
- ``output-ILE-<n>.xml.gz`` - ILE output for that iteration
- ``*.composite`` files - Consolidated samples
- ``all.net`` - Union of all composite files

See Also
========

- :doc:`util_RIFT_pseudo_pipe` - Main user interface
- :doc:`../api_reference/index` - API Reference