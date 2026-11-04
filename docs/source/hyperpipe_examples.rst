Minimal working example
^^^^^^^^^^^^^^^^^^^^^^^

This YAML reproduces the baseline Gaussian demo:

.. code-block:: yaml

    arch:
      n-iterations: 3
      n-samples-per-job: 1000
      explode-marg-jobs: 5

    post:
      coords-fit: "x y z"
      coords-sample: "x:[-7,7] y:[-7,7] z:[-7,7]"

    marg-list:
      - name: Gaussian
        exe: example_gaussian.py
        args: "--outdir Gaussian_example --conforming-output-name"
        n-chunk: 100

    puff:
      puff-factor: 0.5
      force-away: 0.03

    init:
      file: blind_gaussian_3d_xy_plus.dat

    general:
      rundir: rundir
      request-memory: 200

Save as ``my_gauss.yaml`` and run::

    util_RIFT_hyperpipe.py --config my_gauss.yaml

The script prints the ``create_eos_posterior_pipeline`` command it
generates (useful for debugging), then executes it unless
``general.dry-run: true`` is set.

Tracer example
^^^^^^^^^^^^^^

To use the parsimonious-placement tracer workflow, use the
``hyperpipe_conf_tracer.yaml`` configuration. Example:

.. code-block:: yaml

    arch:
      method: default
      n-iterations: 5
      n-samples-per-job: 1000
      explode-marg-jobs: 5
      start-iteration: 0

    post:
      coords-fit: "x y z"
      coords-sample: "x:[-7,7] y:[-7,7] z:[-7,7]"
      settings:
        fit-method: rf

    marg-list:
      - name: Gaussian
        exe: example_gaussian.py
        args: "--outdir Gaussian_example --conforming-output-name"
        n-chunk: 100

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
        rng-seed: null

    init:
      file: blind_gaussian_3d_xy_plus.dat

    general:
      rundir: rundir_tracer
      request-memory: 200

Run with::

    util_RIFT_hyperpipe.py --config hyperpipe_conf_tracer.yaml

This config enables the tracer to consume ``all.marg_net`` and write the
next iteration grid directly, bypassing the MARG_PUFF lane and saving
~1.7–1.8× wall time.
