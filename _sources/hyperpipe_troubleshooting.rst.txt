=====================
HyperPipe Troubleshooting
=====================

This guide provides common commands for monitoring, debugging, and recovering HyperPipe runs.

Monitoring DAG Status
^^^^^^^^^^^^^^^^^^^^^

To check the status of your submitted DAG and see which jobs are running, held, or completed::

    condor_q

Tail a Worker's Log
^^^^^^^^^^^^^^^^^^^^

If a specific job is failing or behaving unexpectedly, tail its log file in real-time::

    condor_tail -f <job-id> <log-file>

Re-submitting a Stalled DAG
^^^^^^^^^^^^^^^^^^^^^^^^^^

If the DAG has stopped due to a transient error or a manual hold, you can re-submit the existing DAG file::

    condor_submit_dag marginalize_hyperparameters.dag

Inspecting Generated Args Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline generates intermediate ``args_*.txt`` files that dictate exactly what is passed to the executables. Inspect these to verify configuration mapping::

    cat rundir/args_marg_eos.txt
    cat rundir/args_puff.txt
    cat rundir/args_eos_post.txt
