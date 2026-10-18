# Pipeline deployment regression checks

Run in an installed LALSuite/GWPy/HTCondor environment:

```sh
python -m pytest MonteCarloMarginalizeCode/Code/test/pipeline_deployment/test_deployment_cli.py -q
```

These tests execute the real BasicIteration and AlternateIteration builders,
pseudo_pipe, runtime subdag builder, frame truncation entry point, and native
`condor_submit -dry-run`. They submit no jobs and evaluate no likelihoods. The
synthetic frame checks write GWF files, join multiple input files, crop a padded
fractional interval, read the output back, and compare every strain sample and
the complete generated cache against `lalapps_path2cache`.

Fixtures cover Condor quoted waveform arguments, flag-like payload strings,
missing/default and conflicting batch settings, exact caps, shortened CIP
schedules, container/disk/OAuth propagation, extrinsic time exports, actual puff
producer dependencies, missing channels/detectors, gaps/endpoints, paths with
spaces, and rollback on a failed cache publication. Package installation supplies
both the retained shell entry point and the Python implementation; GWPy is already
in the package requirements.

Ordinary default-builder smoke passes on the unmodified branch. Regressions
address shared optional truncation/runtime-subdag paths and AlternateIteration
worker deployment parity; they do not demonstrate a defect in every standard
inference. These build and frame IO tests do not claim posterior recovery.

`test_mock_workflow.py` adds bounded local execution of the actual emitted
Basic/Alternate two-stage DAGs, including real runtime subdag builders,
native Condor cluster/process argument and environment parsing, and real POST
existence checks. Drop-in executables replace ILE, CIP, PUFF, join, unify,
evidence, and final extrinsic conversion. Their outputs carry input provenance;
fits require current ordinary and puff likelihood records, and conversion requires
all three final batch workers (six points). Seven-point intrinsic coverage with
batch size two checks the trailing partial batch. Removing the puff producer edge
must fail at the real puff subdag builder; a worker returning success without its
output must also fail before the final artifact.

The executor covers an explicit local/shared-filesystem DAG subset and rejects
unsupported control directives, legacy environments, input redirection and output
remapping. It validates transfer-input existence but does not emulate remote
Condor sandboxes, container execution, retries or scientific calculations.
Native submit parsing uses dry-run only; no scheduler jobs are submitted.

Normal data staging is also tested without `--fake-data-cache`: a drop-in
`gw_data_find` returns synthetic local GWF URLs, while the real helper converts
and assembles `H_local.cache` and `local.cache`. A counted wrapper executes the
real frame helper, proving it is called exactly once before DAG construction in
both builders, with `RIFT_TRUNCATE_CHECK` enabled and disabled. Explicit-cache
staging with that check enabled is covered separately. No data service is used.
