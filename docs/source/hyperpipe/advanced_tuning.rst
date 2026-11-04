Advanced Tuning
================

This guide is for power users looking to optimize their inference or implement complex constraints.

Multi-Constraint Inference
---------------------------

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

Coordinate Transformation
--------------------------

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

Coordinate-module options and the puff stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``coord-function``, ``coord-ini`` and ``coord-chart`` set
``--supplementary-coordinate-{function,ini,chart}``.  Older configs that put
these flags in ``post.extra-args`` keep working; give them one way or the
other, not both.

``coord-input-parameters`` names the data-file columns the module maps from,
e.g. ``"x y z"``.  Grid and posterior files are always written in these
columns.  Set it when ``coords-fit`` names the module's *outputs*, say
``"u v w"``.  The puff stage then gets the module and its options.  It puffs
in ``(u, v, w)`` and maps the result back to ``(x, y, z)``.  The convergence
test reads ``(x, y, z)``.  Without the key, both stages are handed ``u v w``
as file columns.

``puff.coord-basis`` takes three values.  ``auto`` does the above.
``file`` always uses the data-file columns and no module.  ``plugin`` puffs in
the module's output basis even when the MC samples file columns, as in a
``coords-implied``/``coords-nofit`` config.  Every puff-basis name with a ``coords-sample`` range is passed to
the puff as a downselect range.

A coordinate module may also define
``get_bounds(coord_names, ranges, **kwargs)`` returning
``{name: [lo, hi]}`` in the sampling basis.  ``--get-range-from-external``
(post and puffball) uses it; ``--external-range-args key=value`` supplies the
keyword arguments.  Command-line ranges win, and an error in the hook stops
the run.

Parsimonious-placement Tracer Workflow
--------------------------------------

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

Convergence Testing
--------------------

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
