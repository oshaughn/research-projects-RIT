Initial grid generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
