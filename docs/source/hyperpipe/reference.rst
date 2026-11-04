Reference
===========

This page provides technical specifications for inputs, outputs, and legacy support.

Directory and Output Reference
------------------------------

When a run is initiated, HyperPipe creates a structured directory. The following map describes the resulting layout:

.. code-block:: console

   rundir/
      -> grid-0.dat              # The initial seed grid
      -> local.cache             # Internal caching
      -> iteration_N_marg/       # MARG output for iteration N
      -> iteration_N_post/       # EOS_POST output for iteration N
      -> iteration_N_con/        # CON output for iteration N
      -> iteration_N_puff/       # PUFF output for iteration N
      -> MARG_0.sub              # Submission script for MARG
      -> CON.sub                 # Submission script for CON
      -> CON_PROD.sub            # Submission script for CON_PROD
      -> UNIFY.sub               # Submission script for UNIFY
      -> EOS_POST.sub            # Submission script for EOS_POST
      -> JOIN_POST.sub           # Submission script for JOIN_POST
      -> PUFF.sub                # Submission script for PUFF
      -> marginalize_hyperparameters.dag # The top-level Condor DAG

Detailed File Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - File/Directory
     - Description
   * - ``grid-0.dat``
     - The initial seed grid.
   * - ``grid-*.dat``
     - Posterior samples from iteration N.
   * - ``iteration_N_marg/``
     - MARG output for iteration N.
   * - ``iteration_N_post/``
     - EOS_POST output for iteration N.
   * - ``iteration_N_con/``
     - CON output for iteration N.
   * - ``iteration_N_puff/``
     - PUFF output for iteration N.
   * - ``all.marg_net``
     - Cumulative likelihood table (The primary result).
   * - ``consolidated_*.net_marg``
     - Per-iteration MARG results.
   * - ``posterior-*.dat``
     - Posterior samples drawn by EOS_POST.
   * - ``marginalize_hyperparameters.dag``
     - The Condor DAG submission file.
   * - ``*.sub``
     - Condor submission scripts.

Legacy Pipeline: Args-File Interface
------------------------------------

The original approach uses hand-crafted ``args_*.txt`` files and
``create_eos_posterior_pipeline.py`` directly. This is still
supported and is what the demo Makefile does.

To create a run directory and submit via the legacy interface::

    mkdir my_run && cd my_run
    create_eos_posterior_pipeline \
        --marg-event-exe-list-file args_marg_eos_exe.txt \
        --marg-event-args-list-file args_marg_eos.txt \
        --eos-post-exe $(which util_ConstructEOSPosterior.py) \
        --eos-post-args args_eos_post.txt \
        --puff-exe $(which util_HyperparameterPuffball.py) \
        --puff-args args_puff.txt \
        --input-grid initial_grid.dat \
        --n-samples-per-job 1000 \
        --n-iterations 5 \
        --event-file my_event.txt \
        --working-dir . \
        --use-full-submit-paths \
        --eos-post-explode-jobs 5

    condor_submit_dag marginalize_hyperparameters.dag

Required Input Files
^^^^^^^^^^^^^^^^^^^^

**Initial grid** — a text file with header ``# lnL sigma_lnL <params...>``. 
The first two columns are set to zero; the pipeline overwrites them.

*Note: If your external executable requires a different input format, you should write a wrapper script that translates RIFT format parameters to match your code's requirements to avoid modifying the core executable.*

Generate an initial grid with ``util_HyperparameterGrid.py``::

    util_HyperparameterGrid.py \
        --random-parameter x --random-parameter-range [-5,-2] \
        --random-parameter y --random-parameter-range [2,5] \
        --random-parameter z --random-parameter-range [2,5] \
        --npts 1000 --fname-out initial_grid.dat

**Executable** — a script that computes the likelihood for a
parameter point. See the driver contract in :doc:`driver_dev` and the example
``example_gaussian.py`` in the demo directory.
