# RIFT Developer Guide

This guide is for humans and agents changing core RIFT components.  The main
habit to preserve is simple: identify the component being changed, run the
cheapest test that can catch obvious breakage, then choose at least one
component-appropriate integration test before trusting the change.

RIFT is scientific software.  A green sanity test means "not obviously broken,"
not "scientifically validated."  The tests below are a ladder, not a single
gate.

## Testing Ladder

### 0. CI and local sanity checks

CI is the zeroth-level sanity check.  It catches import failures, script drift,
and short deterministic examples.  It should be run locally before pushing when
possible, especially after changing public command-line tools.

Useful entry points include:

- `.travis/test-all-bin.sh`
- `.travis/test-build.sh`
- `.travis/test-docs.sh`
- individual tests under `MonteCarloMarginalizeCode/Code/test/`

Local runs need the normal RIFT environment, including `lalsuite`; on many
systems set `GW_SURROGATE=''`.

### 1. Pipeline framework tests

Pipeline changes should not be validated only by importing Python modules.
They need generated DAGs and, when possible, actual local Condor execution.

The lightweight end-to-end framework test lives at:

```bash
MonteCarloMarginalizeCode/Code/demo/rift/test_frameworks/zero_likelihood/run_hyperpipeline_condor_smoke.sh
```

This test runs a small HTCondor DAG using tiny ILE/CIP stand-ins.  It exercises
the real pipeline generator, join/postprocess steps, hyperpipeline ASCII
`grid-N.dat` flow, `all.net` unification, `posterior_samples-*.dat` aliases,
and the convergence-test node.  It is heavier than CI but far lighter than a
full PE run.

Scope: this validates workflow plumbing, Condor environment propagation,
intermediate file naming, and postprocessing assumptions.

Non-scope: this does not validate waveform physics, real ILE data ingestion, or
posterior quality.  A future mode should run real
`integrate_likelihood_extrinsic_batchmode --zero-likelihood` once that path is
fully data-free.

### 2. Full scientific workflows

When a change can alter likelihood values, posterior weights, event setup,
coordinate semantics, calibration, or waveform generation, at least one full
workflow comparison is needed.  Prefer an existing, archived run with known
`overlap-grid-N.xml.gz` and `consolidated*.composite` outputs so the new code
can be compared against a fixed reference.

Full workflow tests should record:

- branch and commit
- command lines and environment
- input grid(s)
- PSD/data setup
- likelihood point comparisons
- posterior/postprocessing diagnostics

## Component Notes

### Pipeline

Pipeline edits affect DAG shape, submit files, environment propagation, file
transfer, naming conventions, and postprocessing dependencies.  Use the
zero-likelihood-style Condor smoke framework first, then a real run if the
change can affect science outputs.

Common checks:

- inspect generated `.dag` and `.sub` files
- confirm expected `PARENT/CHILD` ordering
- confirm submit-file `environment` and `getenv` behavior
- confirm expected files are non-empty
- confirm aliases such as `posterior_samples-*.dat` exist when downstream
  tooling expects them

### Waveforms

Waveform tests are necessarily fuzzy.  GR waveform modeling is not perfectly
known, and different approximants/backends can legitimately disagree within
model-dependent tolerances.  The goal is to catch discontinuities, broken data
conditioning, obvious convention errors, and regressions against expected
behavior.

Relevant tests live under:

```bash
MonteCarloMarginalizeCode/Code/test/waveform/
```

Important scripts include:

- `check_waveform_random.py`
- `check_waveform_taper.py`
- `compare_in_ifo.py`
- `plot-waveform-ci.py`

Use these when adding new waveforms, changing waveform argument handling, or
changing data-conditioning/tapering logic.  Do not treat a single numerical
tolerance as universal; document the approximant, parameter range, and expected
comparison behavior.

### Coordinates

Coordinate changes are easy to make locally consistent while breaking a
pipeline boundary.  Run the vector-coordinate test after any change to
coordinate naming, conversion, implied parameters, or sampler coordinates:

```bash
.travis/test-coord.sh
```

This currently drives:

```bash
python MonteCarloMarginalizeCode/Code/test/test_vector_coordinates.py --as-test
```

### Integrator

Integrator changes should run the Travis integrator smoke and then a targeted
stress test for the changed sampling mode:

```bash
.travis/test-integrate.sh
```

This currently exercises `test_mcsamplerEnsemble_extended.py` with and without
`--use-lnL`.  For changes to adaptation, GPU/vectorized paths, portfolio
samplers, or pinned distributions, also inspect:

```bash
MonteCarloMarginalizeCode/Code/test/integrators/
MonteCarloMarginalizeCode/Code/test/test_mcsamplerEnsemble_extended.py
```

Use reproducible seeds where possible, and report effective sample sizes and
failure modes rather than only pass/fail.

### Likelihood

Likelihood changes need special care.  Unit tests and smoke tests catch many
interface errors, but the important validation is agreement at identical points.

A lightweight first test should run quickly on fake data.  The Travis run
scripts are the current starting point:

```bash
.travis/test-run.sh
.travis/test-build.sh
```

The fake-data setup can be run with high Monte Carlo accuracy and tunable SNR
through the PSD/data configuration.  That makes it useful for stress-testing a
new likelihood implementation or a branch such as `calmarg_in_loop`: raise the
accuracy, tune SNR, and compare the old and new code on exactly the same
intrinsic/extrinsic points.

Minimum useful likelihood comparison:

- same data
- same PSD
- same waveform approximant and conditioning
- same intrinsic point
- same extrinsic point or controlled marginalization settings
- high enough Monte Carlo accuracy that implementation differences dominate
  sampling noise

Preferred full test:

1. Point a comparison script at an existing run directory containing
   `overlap-grid-N.xml.gz` files and `consolidated*.composite` files.
2. Recompute likelihood values with the candidate implementation on the same
   points.
3. Compare against the existing run, separating deterministic differences from
   Monte Carlo uncertainty.
4. Report both pointwise differences and downstream postprocessing impact.

This full comparison script does not yet exist in a polished form; building it
would make future likelihood work much safer.

### Asimov Tests

An Asimov testing layer is needed but does not exist yet.  It should provide
deterministic, noise-free validation cases where the expected likelihood shape,
posterior concentration, and convergence behavior are known well enough to
catch regressions.

There are exploratory files under `MonteCarloMarginalizeCode/Code/test/` with
`asimov` in their names, but the missing framework is a maintained, documented
workflow that can be run routinely.  Desired properties:

- no stochastic data realization
- tunable SNR
- known injected parameters
- predictable convergence diagnostics
- compatibility with both lightweight and full pipeline modes

## When Changing Code

Use this checklist before claiming a core change is ready:

1. Name the affected component: pipeline, waveform, coordinates, integrator,
   likelihood, postprocessing, or backend.
2. Run CI-level sanity checks locally when practical.
3. Run the component-specific smoke test.
4. For pipeline or likelihood changes, run a DAG or workflow-level test.
5. Save the run directory or enough command-line detail for another developer
   to reproduce the result.
6. In the commit or PR notes, say what was tested and what was not tested.

If a test has to be skipped because of local environment limitations, say so
explicitly.  Silent missing validation is worse than an honest known gap.
