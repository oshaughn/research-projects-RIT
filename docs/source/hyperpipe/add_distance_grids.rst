===========================
``add_distance_grids`` demo
===========================

The ``add_distance_grids`` demo builds a small zero-spin RIFT workflow that
exports a luminosity-distance likelihood grid for each completed ILE
evaluation. Use it to verify the distance-grid export path before adapting the
same options to another RIFT workflow.

The runnable sources are in
``MonteCarloMarginalizeCode/Code/demo/rift/add_distance_grids``. The demo uses
the fake-data inputs in ``.travis/ILE-GPU-Paper/demos`` and its
``add_distance_grids.ini`` records the corresponding configuration.

Build the DAG
=============

From the demo directory, first check that the CI-style inputs are present:

.. code-block:: console

   $ make inputs

Then create the DAG:

.. code-block:: console

   $ make dag

.. warning::

   ``make dag`` removes and recreates ``rundir/`` before generating the DAG.
   Copy any results you need from that directory before rerunning it.

The target writes the generated run directory and checks that
``rundir/args_ile.txt`` contains both ``--export-marginal-distance-grid`` and
``--internal-use-lnL``. Inspect the DAG and arguments before submitting. The
demo never submits automatically; submission is an explicit separate action:

.. code-block:: console

   $ make submit

Validate an output grid
=======================

After ILE jobs complete, use the loader before reconstruction: the
``reconstruct_marginal_lnL`` API accepts the parsed grid table, not a filename.
For one generated ``*.dgrid`` file:

.. code-block:: python

   from RIFT.misc.distance_grid import (
       load_distance_grid,
       reconstruct_marginal_lnL,
   )

   grid = load_distance_grid("path/to/point.dgrid")
   reconstructed = reconstruct_marginal_lnL(grid)

With the default argument, reconstruction uses the stored sampling-distance
prior when it is present. Compare the result to the ordinary marginalized
likelihood for the same intrinsic point, allowing for that run's Monte Carlo
uncertainty. For controlled synthetic checks, run
``validate_distance_grid.py`` and ``validate_distance_slices.py`` from the
demo directory; they exercise the table-level reconstruction paths directly.

Environment note
================

LALSuite SWIG/Python memory-leak messages can indicate an incompatible local
binding build rather than a distance-grid failure. The demo README describes a
known-good environment constraint: changing the local SWIG executable does not
alter already-built LAL Python bindings.

Review checklist
================

Before requesting human review, verify that this page is reachable from the
HyperPipe landing-page toctree, retain the Sphinx error-delta result, and have
an independent reviewer check the rendered guide against the demo sources.
