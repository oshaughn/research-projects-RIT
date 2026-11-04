The adaptive HyperPipe code, written by Richard O'Shaughnessy and Atul Kedia,
aims to conduct parameter estimation adaptively on **any** observable or
simulated data. The code generalizes RIFT for applications other than
gravitational-wave data analysis.

HyperPipe adaptively explores regions of parameter space for fully generic,
simulation-based inference. Given an initial grid and a user-supplied
*likelihood evaluator* (a.k.a. a "marg driver"), the pipeline iteratively:

1. Evaluates the (marginalized) likelihood on the current hyperparameter grid.
2. Constructs a posterior via Monte Carlo integration.
3. Draws new exploration points from the posterior (and optionally a
   "puff" stage that widens coverage).
4. Repeats for the configured number of iterations.

Two interfaces are available:

:``util_RIFT_hyperpipe.py`` (recommended)
    Hydra/OmegaConf-based configuration. Write a single YAML file,
    run one command. Supports multi-event inputs, coordinate transforms,
    and the parsimonious-placement tracer workflow.

:``create_eos_posterior_pipeline.py`` + hand-crafted args files (legacy)
    The original approach. Assembles DAG submission files from
    manually-written ``args_*.txt`` files. Still fully supported.

This document covers both interfaces, using the Gaussian toy demo as a
running example.

---------------------------------------------
Quick-start: the Gaussian toy demo
---------------------------------------------

The fastest way to understand HyperPipe is to run the included 3-D
Gaussian demo::

    cd RIFT/MonteCarloMarginalizeCode/Code/demo/hyperpipe
    make rundir

This invokes ``util_RIFT_hyperpipe.py`` with the demo configuration
``hyperpipe_conf.yaml``. When it finishes, submit the DAG::

    cd rundir
    condor_submit_dag marginalize_hyperparameters.dag

Monitor with ``watch condor_q``. When done, plot the result::

    plot_posterior_corner.py \
        --posterior-file posterior-3.dat \
        --composite-file all.marg_net \
        --composite-file-has-labels \
        --parameter x --parameter y --parameter z \
        --lnL-cut 15 --use-all-composite-but-grayscale
