Troubleshooting
================

This guide provides recovery steps and environment-specific caveats for HyperPipe.

Operational Context
--------------------

The HyperPipe workflow is split into two primary functional roles:

1. **Worker MARG Jobs**: These are the high-volume physics evaluations. If these are failing, check your marginal driver's CLI contract or the input grid format.
2. **Fitting & Posterior Jobs**: These are the single-node tasks that compute the posterior and generate the next grid. Failures here typically indicate issues with MC integration or `util_ConstructEOSPosterior.py`.

For RIFT Users
---------------

HyperPipe is a generalized version of the RIFT gravitational-wave PE pipeline. The mapping between the two:

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

The key difference is that HyperPipe is *coordinate-free*. The coordinate system is declared in the YAML and passed to the post stage via the same flags that CIP uses.

Diagnostic Commands
-------------------

Use these commands to monitor and recover your pipeline:

Check DAG status::

    condor_q

Tail a worker's log (Essential for debugging driver crashes)::

    condor_tail -f <job-id> <log-file>

Re-submit a stalled DAG (Use after fixing a driver bug)::

    condor_submit_dag marginalize_hyperparameters.dag

Inspect the generated args files (Verify that the DAG created the correct paths)::

    cat rundir/args_marg_eos.txt
    cat rundir/args_puff.txt
    cat rundir/args_eos_post.txt

Dry run validation::

    general:
      dry-run: true # Validates config -> DAG translation without submitting to Condor

    util_RIFT_hyperpipe.py --config my_conf.yaml

    # prints the create_eos_posterior_pipeline command without running it
