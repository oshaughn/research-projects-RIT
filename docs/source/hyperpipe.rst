=========
HyperPipe
=========

.. include:: hyperpipe_intro.rst

.. contents::
   :local:


.. include:: hyperpipe_config.rst


.. include:: hyperpipe_examples.rst


Initial grid generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not supply an existing grid, ``util_RIFT_hyperpipe.py`` can
generate one automatically via ``util_HyperparameterGrid.py``::

    init:
      generation:
        placement-method: uniform # Options: uniform (grid), random (MC), or custom
        params-and-ranges: "x:[-8,8] y:[-8,8] z:[-8,8]" # "name:[min,max]" format
        npts: 1000 # Initial sample size; scale based on parameter dimensionality

Alternatively, invoke an arbitrary external grid generator::

    init:
      generation:
        external-code: my_grid_maker.sh
        external-args: "--output initial_grid.dat"


Multi-Constraint Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperPipe can evaluate multiple observational constraints simultaneously (e.g., combining GW data with nuclear physics experiments). 

To incorporate multiple constraints, add multiple entries to ``marg-list``::

    marg-list:
      - name: gw_event # Used for output directory and log tagging
        exe: example_gaussian.py
        args: "--outdir gw_out --conforming-output-name"
        n-chunk: 100 # Parallelization lever: higher = more jobs, smaller chunks
      - name: nicer_event
        exe: example_gaussian2.py
        args: "--outdir nicer_out --conforming-output-name"
        n-chunk: 200

Each entry results in a separate set of MARG jobs. Results are combined
by the ``CON_PROD`` stage. By default, the pipeline multiplies the likelihoods; 
to switch to additive combination, edit ``con_prod.sh`` and change 
``--combination product`` to ``--combination sum``.


Coordinate transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *tracer* workflow replaces the puffball's "draw from posterior +
puff" combination with a single placement step that reads the evaluated
likelihood table (``all.marg_net``) directly, skipping the MARG_PUFF lane
entirely and saving ~1.7–1.8× in wall time.

Enable it by setting ``puff.exe`` to a tracer-aware updater and
``puff.input-source: marg_net``::

    puff:
      exe: util_HyperparameterTracerUpdate.py
      input-source: marg_net # Bypasses PUFF jobs; reads cumulative grid directly
      puff-factor: 0.5       # Search width around maxima; lower = tighter focus
      force-away: 0.03       # Min distance from existing points to prevent clustering
      settings:
        update-method: smc-mala-bd # Sequential Monte Carlo / Metropolis-Adjusted Langevin
        tracer-fit-method: rf      # Random Forest fit for the likelihood surface
        n-mala-steps: 8            # Number of Langevin steps for proposal refinement
        target-ess-frac: 0.5       # Target Effective Sample Size fraction for resampling
        birth-death-rate: 1.0      # Rate of particle birth/death in the SMC process

Run the tracer demo::

    make rundir_tracer

Or directly::

    util_RIFT_hyperpipe.py --config hyperpipe_conf_tracer.yaml


Convergence testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` section controls ``convergence_test_samples``::

    test:
      exe: convergence_test_samples
      method: JS # Options: JS (Jensen-Shannon), KL, etc.
      threshold: 0.05 # Halts DAG when divergence < threshold; typically 0.01-0.1
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
^^^^^^^^^^^^^^^^^^^

Every marg driver must accept the following CLI contract to integrate with the pipeline:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Flag
     - Description
   * - ``--using-eos PATH``
     - Input hyperparameter grid. The ``file:`` prefix is tolerated. Format: ``# lnL sigma_lnL <params...>``
   * - ``--eos_start_index INT``
     - First row to evaluate (inclusive).
   * - ``--eos_end_index INT``
     - Last row to evaluate (exclusive).
   * - ``--fname-output-integral PATH``
     - Output file path. Driver must write back same rows with evaluated ``lnL`` and ``sigma_lnL``.
   * - ``--outdir PATH``
     - Output directory; created if absent.
   * - ``--conforming-output-name``
     - If set, append ``+annotation.dat`` to the output file name.
   * - ``--fname``
     - Legacy passthrough; may be None or ignored.


Built-in driver: ``util_HyperMargGaussian.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Required input files
^^^^^^^^^^^^^^^^^^^^

**Initial grid** — a text file with header ``# lnL sigma_lnL <params...>``. 
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


Directory and Output Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
Troubleshooting
---------------------------------------------

For common monitoring, debugging, and recovery commands, see :doc:`hyperpipe_troubleshooting`.

Dry run validation::

    general:
      dry-run: true # Validates config -> DAG translation without submitting to Condor

    util_RIFT_hyperpipe.py --config my_conf.yaml

    # prints the create_eos_posterior_pipeline command without running it
