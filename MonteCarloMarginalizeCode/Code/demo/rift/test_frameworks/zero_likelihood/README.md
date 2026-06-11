# Zero-likelihood workflow smoke tests

This directory contains local, heavier-than-CI workflow smoke tests for RIFT
pipeline development.  The goal is to exercise the DAG framework end to end in
HTCondor without requiring detector frames, PSDs, or a full PE-sized run.

The current smoke test uses tiny stand-ins for ILE and CIP.  They write and
read hyperpipeline-style `grid-N.dat` files and assert that Condor propagated
`RIFT_HYPERPIPELINE_FORMAT=1`.  The real workflow pieces still run:

- `create_event_parameter_pipeline_BasicIteration`
- `util_ILEdagPostprocess.sh`
- `util_CleanILE_hyperpipeline.py`
- `unify.sh`
- alias creation for `posterior_samples-*.dat`
- `convergence_test_samples.py`

This is not a replacement for astrophysical validation.  It is a fast local
framework test intended to catch broken DAG wiring, environment propagation,
ASCII-grid joins, and postprocessing assumptions.

## Run

From a checkout with the usual RIFT runtime available:

```bash
./MonteCarloMarginalizeCode/Code/demo/rift/test_frameworks/zero_likelihood/run_hyperpipeline_condor_smoke.sh
```

Useful environment overrides:

- `RIFT_ZERO_LIKE_PYTHON`: Python executable used by Condor worker stubs.
- `RIFT_ZERO_LIKE_WORKDIR`: working directory; defaults under `/tmp`.
- `RIFT_ZERO_LIKE_SUBMIT=0`: generate the DAG but do not submit it.
- `RIFT_ZERO_LIKE_WAIT=0`: submit but do not wait for DAG completion.

On Richard's local setup, for example:

```bash
RIFT_ZERO_LIKE_PYTHON=/Users/rossma/miniconda3/envs/junior_tools/bin/python \
  ./MonteCarloMarginalizeCode/Code/demo/rift/test_frameworks/zero_likelihood/run_hyperpipeline_condor_smoke.sh
```

Expected output includes `All jobs Completed!` in the DAGMan log and non-empty
`all.net`, `consolidated_*.composite`, `overlap-grid-*.dat`, plus symlinks
`posterior_samples-*.dat -> overlap-grid-*.dat`.

## Future extension

The natural next step is a second mode that runs the real
`integrate_likelihood_extrinsic_batchmode --zero-likelihood`.  At the time this
test was added, that path still attempted to read frame/PSD inputs before the
zero-likelihood shortcut could make the workflow data-free.
