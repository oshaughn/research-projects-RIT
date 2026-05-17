=========
HyperPipe
=========

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

.. contents::
   :local:


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


---------------------------------------------
``util_RIFT_hyperpipe.py``: Configuration-driven pipeline
---------------------------------------------

``util_RIFT_hyperpipe.py`` is the top-level driver for the hyperpipeline.
It consumes a Hydra/OmegaConf configuration file and drives the
same ``create_eos_posterior_pipeline.py`` machinery that the legacy
Makefile path uses — but without the manual args-file authoring.

Usage::

    util_RIFT_hyperpipe.py --config hyperpipe_conf.yaml

Hydra overrides work as expected::

    util_RIFT_hyperpipe.py --config hyperpipe_conf.yaml \
        arch.n-iterations=10 general.use-osg=true

A bare run (no ``--config``) loads the installed default
``hyperpipe_conf.yaml`` next to the script. The demo configuration in
``MonteCarloMarginalizeCode/Code/demo/hyperpipe/hyperpipe_conf.yaml`` is
the reference template.


Configuration schema
^^^^^^^^^^^^^^^^^^^^

The configuration is organized into six top-level sections.

``arch``
    Controls iteration count, chunk size, and parallelization.

``post``
    Configures the posterior-construction stage
    (``util_ConstructEOSPosterior.py``): which parameters to fit,
    integration ranges, coordinate module, and MC sampler settings.

``marg-list``
    One entry per (likelihood driver, event) pair. Each entry names an
    executable, its argument string, the event file, and batch size.
    Multiple entries enable heterogeneous multi-event inference.

``puff``
    Configures the puffball / parsimonious-placement stage
    (``util_HyperparameterPuffball.py`` or a tracer updater):
    puff factor, force-away distance, and (for tracers) sampler
    hyperparameters.

``test``
    Convergence-test settings (which stage runs
    ``convergence_test_samples``).

``init``
    Sourcing for the initial hyperparameter grid: either a path to an
    existing grid file, or a ``generation`` block that calls
    ``util_HyperparameterGrid.py`` automatically.

``general``
    Working directory, Condor resource requests, OSG/singularity flags,
    and retry behaviour.


.. include:: hyperpipe_examples.rst


Initial grid generation
^^^^^^^^^^^^^^^^^^^^^^^

If you do not supply an existing grid, ``util_RIFT_hyperpipe.py`` can
generate one automatically via ``util_HyperparameterGrid.py``::

    init:
      generation:
        placement-method: uniform
        params-and-ranges: "x:[-8,8] y:[-8,8] z:[-8,8]"
        npts: 1000

Alternatively, invoke an arbitrary external grid generator::

    init:
      generation:
        external-code: my_grid_maker.sh
        external-args: "--output initial_grid.dat"


Multi-event inference
^^^^^^^^^^^^^^^^^^^^^

Add multiple entries to ``marg-list``::

    marg-list:
      - name: gw_event
        exe: example_gaussian.py
        args: "--outdir gw_out --conforming-output-name"
        n-chunk: 100
      - name: nicer_event
        exe: example_gaussian2.py
        args: "--outdir nicer_out --conforming-output-name"
        n-chunk: 200

Each entry results in a separate set of MARG jobs. Results are combined
by the ``CON_PROD`` stage (default: multiply likelihoods; change
``--combination product`` to ``sum`` in ``con_prod.sh`` for additive
combination).


Coordinate transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``post`` section accepts a ``coord-module`` (an importable Python
module name) and three parameter lists:

``coords-fit``
    Space-separated names of parameters to include in the GP fit.
    Example: ``"x y z"``

``coords-sample``
    Space-separated ``name:[lo,hi]`` ranges for MC integration.
    Example: ``"x:[-7,7] y:[-7,7] z:[-7,7]"``

``coords-implied``
    Parameters used in the fit but not independently sampled.
    Example: ``"R1.4 Mmax"``

``coords-nofit``
    Parameters sampled but excluded from the fit.
    Example: ``"delta_mc s1z s2z"``

These are emitted as
``--parameter``, ``--integration-parameter-range``,
``--parameter-implied``, and ``--parameter-nofit`` flags for the
post-stage executable — the same convention that
``util_ConstructEOSPosterior.py`` already uses. Existing CIP coordinate
modules (e.g. for neutron-star EOS inference) are reusable without change.


Parsimonious-placement tracer workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *tracer* workflow replaces the puffball's "draw from posterior +
puff" combination with a single placement step that reads the evaluated
likelihood table (``all.marg_net``) directly, skipping the MARG_PUFF lane
entirely and saving ~1.7–1.8× in wall time.

Enable it by setting ``puff.exe`` to a tracer-aware updater and
``puff.input-source: marg_net``::

    puff:
      exe: util_HyperparameterTracerUpdate.py
      input-source: marg_net
      puff-factor: 0.5
      force-away: 0.03
      settings:
        update-method: smc-mala-bd
        tracer-fit-method: rf
        n-mala-steps: 8
        target-ess-frac: 0.5
        birth-death-rate: 1.0

Run the tracer demo::

    make rundir_tracer

Or directly::

    util_RIFT_hyperpipe.py --config hyperpipe_conf_tracer.yaml


Convergence testing
^^^^^^^^^^^^^^^^^^^

The ``test`` section controls ``convergence_test_samples``::

    test:
      exe: convergence_test_samples
      method: JS
      threshold: 0.05
      settings:
        always-succeed: true   # diagnostic mode: never stop the DAG

The DAG halts when the convergence metric (default: Jensen–Shannon
divergence, JS) falls below ``threshold``. Set ``always-succeed: true``
to run all iterations regardless for diagnostic purposes.


---------------------------------------------
Writing a marg driver
---------------------------------------------

A *marg driver* is the executable that evaluates the (marginalized)
likelihood for each hyperparameter point. The hyperpipeline expects
drivers to accept a specific CLI contract; a base class and a Gaussian
toy driver are provided in ``RIFT.hyperpipe.drivers``.


Driver contract
^^^^^^^^^^^^^^^

Every marg driver must accept these flags:

``--using-eos PATH``
    Input hyperparameter grid. The ``file:`` prefix is tolerated.
    Grid format::

        # lnL sigma_lnL <param1> <param2> ...

``--eos_start_index INT``
    First row to evaluate (inclusive).

``--eos_end_index INT``
    Last row to evaluate (exclusive).

``--fname-output-integral PATH``
    Output file path. The driver writes the same rows back with
    columns 0 and 1 replaced by the evaluated ``lnL`` and ``sigma_lnL``.

``--outdir PATH``
    Output directory; created if absent.

``--conforming-output-name``
    If set, append ``+annotation.dat`` to the output file name.

``--fname``
    Legacy passthrough; may be None or ignored.


Built-in driver: ``util_HyperMargGaussian.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to run the Gaussian demo without authoring an executable
is to use the structured driver installed as a command-line tool::

    util_HyperMargGaussian.py \
        --using-eos file:my_grid.dat \
        --eos_start_index 0 --eos_end_index 1000 \
        --outdir out \
        --fname-output-integral lnL.txt \
        --conforming-output-name

The driver is a thin shim around
``RIFT.hyperpipe.drivers.gaussian.GaussianMargDriver`` and supports:

``--x-offset FLOAT``
    Position of the two modes along x (default: 4.0).

``--sigma2 FLOAT``
    Diagonal of the covariance matrix (default: 2.0).

``--unimodal``
    Drop the second mode at ``-x_offset`` for a single-mode Gaussian.

``--params X,Y,Z``
    Comma-separated names of the three grid columns (default: ``x,y,z``).


DIY driver with ``MargDriverBase``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For custom physics, subclass ``MargDriverBase``::

    from RIFT.hyperpipe.drivers.base import MargDriverBase

    class MyDriver(MargDriverBase):
        description = "My custom marg driver."

        def log_likelihood(self, row_values, column_names, opts):
            # row_values: strings from columns 2.. of the grid
            # column_names: list of parameter names from the header
            params = dict(zip(column_names, map(float, row_values)))
            # your physics here
            lnL = -0.5 * (params["x"]**2 + params["y"]**2)
            return lnL, 1e-3

    if __name__ == "__main__":
        MyDriver().run()

``MargDriverBase`` handles all CLI parsing, grid I/O, and output
formatting. Implement ``log_likelihood`` and (optionally)
``add_arguments`` for driver-specific flags.


---------------------------------------------
Legacy pipeline: args-file interface
---------------------------------------------

The original approach uses hand-crafted ``args_*.txt`` files and
``create_eos_posterior_pipeline.py`` directly. This is still
supported and is what the demo Makefile does.

Create a run directory::

    mkdir my_run
    cd my_run
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

Then submit::

    condor_submit_dag marginalize_hyperparameters.dag


Required input files
^^^^^^^^^^^^^^^^^^^^

**Initial grid** — a text file with header::

    # lnL sigma_lnL <param1> <param2> ...

The first two columns are set to zero; the pipeline overwrites them.
Generate one with ``util_HyperparameterGrid.py``::

    util_HyperparameterGrid.py \
        --random-parameter x --random-parameter-range [-5,-2] \
        --random-parameter y --random-parameter-range [2,5] \
        --random-parameter z --random-parameter-range [2,5] \
        --npts 1000 --fname-out initial_grid.dat

**Executable** — a script that computes the likelihood for a
parameter point. See the driver contract above and the example
``example_gaussian.py`` in the demo directory.


---------------------------------------------
Pipeline stages
---------------------------------------------

Whether launched via ``util_RIFT_hyperpipe.py`` or the legacy args-file
interface, the DAG runs the same stages:

MARG / MARG_PUFF
    Invokes the marg driver(s) over batches of the hyperparameter grid.
    Each job writes a per-chunk output file.

CON / CON_PROD
    Consolidates per-chunk output files. ``CON_PROD`` joins multiple
    events into a single combined result (default: multiply likelihoods).

UNIFY
    Merges results from all previous iterations into the growing cumulative
    grid stored in ``all.marg_net``.

EOS_POST
    Runs ``util_ConstructEOSPosterior.py`` to perform MC integration of
    ``likelihood × prior`` and draw a new posterior sample. This becomes
    the next iteration's exploration grid.

PUFF
    Runs ``util_HyperparameterPuffball.py`` (or a tracer updater) to
    widen coverage beyond the posterior modes, producing
    ``grid_puff-*.dat``. The puffed grid is combined with the EOS_POST
    grid for the next MARG stage.

TEST
    Runs ``convergence_test_samples`` on the last two iterations' grids.
    If the convergence metric falls below threshold, the DAG halts.


Directory structure
^^^^^^^^^^^^^^^^^^^

The pipeline creates::

    rundir/
        grid-0.dat              # initial grid
        initial_grid.dat        # copy of the seed grid
        iteration_N_marg/       # MARG output for iteration N
        iteration_N_post/       # EOS_POST output for iteration N
        iteration_N_con/        # CON output for iteration N
        iteration_N_puff/       # PUFF output (if used)
        all.marg_net            # cumulative likelihood table
        marginalize_hyperparameters.dag
        *.sub                   # Condor submission scripts


Output files
^^^^^^^^^^^^

``grid-*.dat``
    Posterior samples from iteration N. Compatible with
    ``plot_posterior_corner.py`` and other RIFT postprocessing tools.

``consolidated_*.net_marg``
    Per-iteration MARG results (one per event).

``posterior-*.dat``
    Posterior samples drawn by EOS_POST (the same content as the
    corresponding ``grid-*.dat`` written by the tracer path).


---------------------------------------------
For RIFT users
---------------------------------------------

HyperPipe is the fully generalized sibling of the RIFT gravitational-wave
PE pipeline. The mapping between the two:

+------------------+------------------------------------------+
| RIFT pipeline    | HyperPipe equivalent                     |
+==================+==========================================+
| ``ILE.sub``      | ``MARG.sub`` (one per event/driver)      |
+------------------+------------------------------------------+
| ``PUFF.sub``     | ``MARG_PUFF.sub`` (legacy) or tracer     |
+------------------+------------------------------------------+
| ``CIP.sub``      | ``EOS_POST.sub``                         |
+------------------+------------------------------------------+
| ``ILE.ini``      | ``hyperpipe_conf.yaml`` (Hydra YAML)     |
+------------------+------------------------------------------+
| ``util_RIFT_pseudo_pipe.py`` | ``util_RIFT_hyperpipe.py`` |
+------------------+------------------------------------------+

The key difference is that HyperPipe is *coordinate-free*: there is no
requirement that parameters be GW-related. Any set of parameters,
likelihood evaluator, and prior can be used. The coordinate system is
declared in the YAML and passed to the post stage via the same
``--parameter``, ``--integration-parameter-range``, and
``--supplementary-coordinate-code`` flags that CIP uses.


---------------------------------------------
Multiple constraints
---------------------------------------------

To incorporate multiple observational constraints, add a second
(executable, args, event-file) entry to ``marg-list``::

    marg-list:
      - name: gw_constraint
        exe: my_gw_driver.py
        args: "--outdir gw_out"
        n-chunk: 100
      - name: nicer_constraint
        exe: my_nicer_driver.py
        args: "--outdir nicer_out"
        n-chunk: 200

The pipeline multiplies the likelihoods from each constraint. To switch
to additive combination, edit ``con_prod.sh`` and change
``--combination product`` to ``--combination sum``.


---------------------------------------------
Monitoring and debugging
---------------------------------------------

Dry run::

    general:
      dry-run: true

    util_RIFT_hyperpipe.py --config my_conf.yaml

    # prints the create_eos_posterior_pipeline command without running it

Check DAG status::

    condor_q

Tail a worker's log::

    condor_tail -f <job-id> <log-file>

Re-submit a stalled DAG::

    condor_submit_dag marginalize_hyperparameters.dag

Inspect the generated args files::

    cat rundir/args_marg_eos.txt
    cat rundir/args_puff.txt
    cat rundir/args_eos_post.txt