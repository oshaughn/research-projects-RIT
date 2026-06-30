Getting Started
================

HyperPipe is an adaptive parameter estimation pipeline designed for generic, simulation-based inference. 

Basics: The Iterative Loop
--------------------------

The pipeline performs inference by iteratively improving the posterior distribution through a three-step cycle:

1. **Evaluation**: The marginalized likelihood is evaluated on a grid of parameters using a user-provided executable.
2. **Posterior Computation**: The posterior is computed via Monte Carlo integration, which generates two new grids for the next iteration:
    - A **posterior grid** based directly on the MC integral.
    - A **"puffed" grid** that expands the search area to ensure the posterior isn't skewed by the initial guess.
3. **Iteration**: Steps 1 and 2 are repeated for the desired number of iterations.

To run HyperPipe, you need:
- An **executable** that calculates the likelihood for a set of parameters.
- An **initial parameter grid** (standard RIFT format).
- **Exploration ranges** (min/max) for each parameter.

Demo & Prototypes
-----------------

Before building your own pipeline, explore the demo environment. This contains example executables and a `Makefile` that serves as the primary template for creating run directories.

**Demo Path**: ``MonteCarloMarginalizeCode/Code/demo/hyperpipe``

We strongly recommend using the demo `Makefile` as a starting point for your own executable paths and parameter settings.


Running Your First Pipeline
---------------------------

The standard workflow involves creating a run directory and submitting a Condor DAG.

1. **Create the Run Directory**:
   Use the ``create_eos_posterior_pipeline.py`` script. This tool prepares the SUB files (with script references and arguments) and constructs the DAG.

   Example command:
   .. code-block:: console

      $ create_eos_posterior_pipeline \
          --marg-event-exe-list-file `pwd`/args_marg_eos_exe.txt \
          --marg-event-args-list-file `pwd`/args_marg_eos.txt \
          --eos-post-args `pwd`/args_eos_post.txt \
          --eos-post-exe `which util_ConstructEOSPosterior.py` \
          --puff-exe `which util_HyperparameterPuffball.py` \
          --puff-args `pwd`/args_puff.txt \
          --input-grid initial_grid.dat \
          --n-samples-per-job 1000 \
          --use-full-submit-paths \
          --working-dir `pwd` \
          --event-file `pwd`/my_event_A.txt \
          --n-iterations 5 \
          --eos-post-explode-jobs 5

2. **Submit the Workflow**:
   HyperPipe requires an HTCondor environment. Submit the generated DAG:

   .. code-block:: console

      $ condor_submit_dag marginalize_hyperparameters.dag

3. **Monitor Progress**:
   The workflow consists of **worker MARG jobs** (physics evaluation) and **fitting/posterior jobs** (distribution estimation). Monitor the queue with:

   .. code-block:: console

      $ watch condor_q

Pipeline Stages
----------------

Whether launched via the modern YAML config or the legacy args-file interface, the DAG executes the following stages:

MARG / MARG_PUFF
    Invokes the marg driver(s) over batches of the hyperparameter grid.

CON / CON_PROD
    Consolidates per-chunk output files. ``CON_PROD`` joins multiple
    events into a single combined result (default: multiply likelihoods).

UNIFY
    Merges results from all previous iterations into the growing cumulative
    grid stored in ``all.marg_net``.

EOS_POST
    Runs ``util_ConstructEOSPosterior.py`` to perform MC integration of
    ``likelihood × prior`` and draw a new posterior sample.

PUFF
    Runs ``util_HyperparameterPuffball.py`` (or a tracer updater) to
    widen coverage beyond the posterior modes.

TEST
    Runs ``convergence_test_samples`` on the last two iterations' grids.
    If the convergence metric falls below threshold, the DAG halts.
