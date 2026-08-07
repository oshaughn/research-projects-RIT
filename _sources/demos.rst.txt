=====
Demos
=====

RIFT ships several demonstration directories under
``MonteCarloMarginalizeCode/Code/demo``.  Use this page as a map from the
problem you want to understand to the smallest existing example that exercises
that workflow.

The demos are not all the same kind of artifact.  Some are fast local smoke
tests, some build Condor DAGs without submitting them, and some are advanced
operator tutorials that assume LIGO cluster credentials or external software.
Each entry below states what it is useful for before pointing you at the source
files.

.. contents:: Demo catalog
   :local:
   :depth: 2

Pipeline-builder smoke tests
============================

Path
   ``MonteCarloMarginalizeCode/Code/demo/pipeline``

Use this when
   You want a fast, submission-free check that ``util_RIFT_pseudo_pipe.py`` can
   build a complete RIFT run directory and thread command-line options through
   CEPP into the generated Condor submit files.

What it demonstrates
   The demo uses fake data and reference ``.ini`` / ``coinc.xml`` inputs to
   create run directories without submitting jobs or requiring real frames,
   PSDs, GPUs, or a Condor pool.  It is especially useful for regression tests
   of argument plumbing: a flag supplied to ``util_RIFT_pseudo_pipe.py`` should
   survive into the correct ``args_*.txt`` file and ``*.sub`` file.

Primary files
   * ``README.md`` — target descriptions and expected assertions.
   * ``Makefile`` — ``baseline``, ``grid``, ``slices``, ``all``, and ``clean``
     targets.

Typical command
   From the demo directory, in a configured RIFT environment::

      make all

   or, using the repository Pixi environment from the README::

      pixi run --manifest-path ../../../../../pixi.toml make all

Notes
   The ``grid`` and ``slices`` targets exercise last-iteration extrinsic export
   behavior.  They check that distance-grid or distance-slice flags land on the
   final extrinsic ILE stage without leaking into the intrinsic ILE jobs.

Zero-spin IMRPhenomD distance-grid validation
============================================

Path
   ``MonteCarloMarginalizeCode/Code/demo/pipeline/zero_spin_phenomD``

Use this when
   You need a compact end-to-end validation of per-distance likelihood export,
   consolidation, and posterior reconstruction on a laptop-scale example.

What it demonstrates
   This demo uses zero-spin IMRPhenomD, the AV sampler, fake zero-noise BBH
   inputs, and a small mass grid.  It bypasses Condor for the local execution
   step but uses the same production code paths for building the pipeline,
   running ``integrate_likelihood_extrinsic_batchmode``, consolidating
   ``.dgrid`` files, and reconstructing a joint intrinsic-plus-distance
   posterior.

Primary files
   * ``README.md`` — full four-stage validation walkthrough.
   * ``Makefile`` — ``build``, ``run-extr``, ``consolidate``, ``posterior``,
     ``all``, and ``clean`` targets.
   * ``zero_spin_phenomD.ini`` — minimal pseudo-pipe configuration.

Typical command
   From the demo directory::

      make all

Notes
   The default settings intentionally evaluate only a few events so the test is
   fast.  This is a code-path validation, not a scientific accuracy benchmark.
   Increase ``N_EVENTS`` if you want a more meaningful posterior check.

HyperPipe demos
===============

Path
   ``MonteCarloMarginalizeCode/Code/demo/hyperpipe``

Use this when
   You want to learn HyperPipe, test generalized likelihood drivers, compare
   baseline posterior resampling to tracer placement, or adapt a toy
   coordinate-free workflow into a real one.

What it demonstrates
   The directory contains runnable YAML configurations for the same 3-D
   Gaussian toy likelihood.  The variants exercise the iterative
   ``MARG -> CON -> UNIFY -> EOS_POST -> PUFF/placement -> TEST`` loop, OSG
   submit-host settings, coordinate transformation, and parsimonious/tracer
   placement.

Primary files
   * ``README.md`` — detailed description of every configuration.
   * ``technical_doc.txt`` — pedagogical implementation notes.
   * ``hyperpipe_conf.yaml`` — baseline posterior-resampling workflow.
   * ``hyperpipe_conf_tracer.yaml`` — tracer/parsimonious-placement workflow.
   * ``hyperpipe_conf_osg.yaml`` — OSG/IGWN-oriented submit configuration.
   * ``hyperpipe_conf_linear_uvw.yaml`` — fit in transformed coordinates while
     sampling in the original coordinates.
   * ``example_gaussian*.py`` — toy likelihood drivers.
   * ``Makefile`` — convenience targets such as ``rundir`` and
     ``rundir_tracer``.

Typical commands
   Baseline demo::

      util_RIFT_hyperpipe.py --config ./hyperpipe_conf.yaml

   Tracer-placement demo::

      util_RIFT_hyperpipe.py --config ./hyperpipe_conf_tracer.yaml

Notes
   Start here before writing a new HyperPipe configuration from scratch.  The
   YAML files show the expected schema and the generated run directories expose
   the exact executable arguments in ``args_*.txt`` and Condor ``*.sub`` files.

Population-study demo
=====================

Path
   ``MonteCarloMarginalizeCode/Code/demo/populations``

Use this when
   You want a worked outline for generating mock compact-binary populations
   with GWKokab and producing RIFT parameter estimates for those injections.

What it demonstrates
   The README describes a multi-environment workflow: generate injections with
   GWKokab, validate the population inference setup, switch to a separate RIFT
   environment, prepare injections, generate MDC files, create RIFT run
   directories, submit PE jobs, and produce diagnostics.

Primary files
   * ``README.md`` — full tutorial and environment notes.
   * ``Makefile`` — workflow automation points.
   * ``pop-example.ini`` — example RIFT configuration for the population run.
   * ``injections.dat`` — example injection table.
   * ``write_mdc.py`` and ``gwk_pop_conversion.py`` — conversion/setup helpers.
   * ``plot_all.sh`` and ``collect_all.sh`` — post-processing helpers.

Typical command
   This is an advanced, environment-dependent workflow.  Read and edit the
   Makefile variables and ``pop-example.ini`` before running targets.  The
   README starts with GWKokab setup and then moves into RIFT setup.

Notes
   Keep GWKokab and RIFT in separate environments.  The prior ranges in
   ``pop-example.ini`` must match the population used to generate
   ``injections.dat``; otherwise the resulting PE runs are not meaningful.

Distance-grid export demo
=========================

Path
   ``MonteCarloMarginalizeCode/Code/demo/rift/add_distance_grids``

Use this when
   You need to understand or validate the ``--export-marginal-distance-grid``
   path and the generated ``.dgrid`` likelihood-density files.

What it demonstrates
   The demo builds a small zero-spin RIFT workflow with distance-grid export
   enabled for ILE jobs.  It reuses fake zero-noise CI assets and verifies that
   the generated ILE arguments include ``--export-marginal-distance-grid`` and
   ``--internal-use-lnL``.

Primary files
   * ``README.md`` — build/submit instructions and environment warning notes.
   * ``PLAN_B_DESIGN.md`` — design notes for fixed-distance slice export and
     re-marginalization.
   * ``Makefile`` — ``dag`` and ``submit`` targets.
   * ``add_distance_grids.ini`` — zero-spin/fake-data configuration.
   * ``validate_distance_grid.py`` and ``validate_distance_slices.py`` — helper
     validation scripts.

Typical commands
   Build the DAG without submitting::

      make dag

   Submit the generated workflow after inspection::

      make submit

Notes
   LALSuite/SWIG compatibility warnings may appear in some environments.  The
   README explains how to distinguish those warnings from distance-grid
   failures.

Numerical relativity with RIFT
==============================

Path
   ``MonteCarloMarginalizeCode/Code/demo/nr_w_rift``

Use this when
   You want an advanced tutorial for comparing gravitational-wave data to
   numerical-relativity simulations rather than analytic waveform models.

What it demonstrates
   The workflow obtains event data, constructs an NR simulation grid, builds a
   RIFT run directory through NR-specific pipeline tools, and runs a refine
   stage before the final CIP posterior construction.

Primary files
   * ``README.md`` — cluster-oriented tutorial and required manual settings.
   * ``Makefile`` — data, grid, and run-directory construction targets.

Typical commands
   This tutorial assumes LIGO computing access and event-specific manual edits.
   Read the README first, then configure the event identifiers, channels, NR
   group, mass range, and event time in the Makefile before running targets such
   as ``make data``, ``make grid``, and ``make rundir``.

Notes
   This is not a quickstart.  It assumes a working RIFT environment, LIGO data
   access, NR catalog access, and familiarity with production RIFT runs.

Internal and test-oriented demos
================================

Some demo-like directories are primarily regression or development harnesses.
They are useful for developers, but should not be presented as first-stop user
quickstarts until their assumptions are documented.

Known examples include:

* ``MonteCarloMarginalizeCode/Code/demo/rift/test_frameworks/zero_likelihood``
  — zero-likelihood HyperPipe/Condor smoke-test material.

When promoting one of these into a user-facing tutorial, first document:

* whether it requires Condor, GPUs, GraceDB, LIGO credentials, or external data;
* whether it submits jobs or only builds run directories;
* expected runtime and expected outputs;
* cleanup commands; and
* which scientific result, if any, should be trusted.

Choosing a starting point
=========================

.. list-table:: Demo selection guide
   :header-rows: 1
   :widths: 24 34 42

   * - If you want to...
     - Start with...
     - Why
   * - Check pseudo-pipe argument plumbing quickly
     - ``demo/pipeline``
     - Fast fake-data DAG construction without submission.
   * - Validate distance-grid export end to end
     - ``demo/pipeline/zero_spin_phenomD``
     - Exercises build, extrinsic likelihood, consolidation, and posterior
       reconstruction.
   * - Learn HyperPipe
     - ``demo/hyperpipe``
     - Small Gaussian likelihood with baseline, tracer, OSG, and coordinate
       transform variants.
   * - Try tracer placement
     - ``demo/hyperpipe/hyperpipe_conf_tracer.yaml``
     - Existing parsimonious-placement example with generated DAG output.
   * - Explore population-study workflows
     - ``demo/populations``
     - End-to-end GWKokab-to-RIFT outline, with environment caveats.
   * - Inspect distance-grid export internals
     - ``demo/rift/add_distance_grids``
     - Focused DAG build and validation helpers for ``.dgrid`` output.
   * - Work with numerical-relativity simulations
     - ``demo/nr_w_rift``
     - Advanced cluster-oriented NR workflow.

Related pages
=============

* :doc:`hyperpipe`
* :doc:`using-pipeline`
* :doc:`examples-ini`
* :doc:`osg`
* :doc:`plotting`
* :doc:`troubleshooting`
