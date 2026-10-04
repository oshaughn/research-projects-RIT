# AGENTS.md - RIFT

## What is RIFT?
RIFT = Iterative parameter estimation pipeline for gravitational wave astronomy (scientific research code). Not a web app or typical software project.

## REDLINE: every ILE command line must reach the NoLoop likelihood

The maintained likelihood is `DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`
(`RIFT/likelihood/factored_likelihood.py`). All current development (sub-sample
stencils, band-limited and peak-local time quadrature, Q pregrid, continuous time
export) lives there. `integrate_likelihood_extrinsic_batchmode` reaches it only with
ALL of:

    --time-marginalization --vectorized --gpu --force-xpy

- `--time-marginalization`: without it ILE calls `FactoredLogLikelihood`.
- `--vectorized`: without it ILE calls the scalar `FactoredLogLikelihoodTimeMarginalized`.
- `--gpu`: without it ILE calls `DiscreteFactoredLogLikelihoodViaArrayVector`.
- `--force-xpy`: on a host without cupy, ILE prints `Override --gpu (not available)`,
  sets `opts.gpu=False`, and drops to `ViaArrayVector`. `--force-xpy` keeps the NoLoop
  path on numpy. It is inert without `--gpu`.

`--gpu` and `--force-xpy` select a code path, not hardware. For a CPU run remove only
`--force-gpu-only`; never remove `--gpu` or `--force-xpy`. `--rotation-slow` and
`--freqresponse` (each with `--vectorized`) reach their own NoLoop variants.

Who adds them: `helper_LDG_Events.py` emits `--vectorized --gpu`;
`util_RIFT_pseudo_pipe.py` adds `--force-xpy` only with `--ile-no-gpu` or
`--ile-sampler-method AV`. Hand-written commands (tests, demos, smoke runs, export-only
reruns) must spell out all four.

Confirm the path in the log. Any of these lines means the run did not use the
maintained likelihood:
- `Override --gpu (not available)` on a run without `--force-xpy`;
- `this is the old vectorized code path not the xpy path` (printed with distance
  marginalization);
- `Q_lm stencil DEFAULT 'sinc' NOT APPLIED`.

An explicit `--interpolate-time` or `--time-marginalization-quadrature` is refused off
NoLoop. The default stencil instead falls back to `nearest` without an error.

## Key directories
- `MonteCarloMarginalizeCode/Code/` - Main source
- `MonteCarloMarginalizeCode/Code/bin/` - Executable scripts (100+ CLI tools)
- `MonteCarloMarginalizeCode/Code/RIFT/` - Python package
- `MonteCarloMarginalizeCode/Code/test/` - Tests

## Developer commands
```bash
# Install in editable mode
pip install -e .

# Run a specific test
python MonteCarloMarginalizeCode/Code/test/test_likelihood.py
```

## Required environment
- **lalsuite** must be installed (LIGO analysis library)
- Set: `export GW_SURROGATE=''`
- On LDG clusters: `export LIGO_USER_NAME=... LIGO_ACCOUNTING=...`

## Testing
Tests use pytest but have no standard runner. Run individual test files directly.

### Merge gate for integrator changes (IMPORTANT)
Any change under `MonteCarloMarginalizeCode/Code/RIFT/integrators/` must pass the
**posterior shape-recovery gate** in
`MonteCarloMarginalizeCode/Code/test/expensive_before_merging/integrators/`
before merging into a production line (run base + candidate with identical seeds,
then `compare_shape_results.py base.json pr.json`; exit 1 = merge-blocking).
The fast CI integral test is NOT sufficient: integrators have shipped confident,
integral-invisible shape failures and silent n_eff~1 degradations that only this
gate catches. See `RIFT/integrators/TESTING.md` for the recipe and caveats.

## Important CLI tools
- `integrate_likelihood_extrinsic_batchmode` - Main PE engine
- `create_event_parameter_pipeline_BasicIteration` - Full pipeline
- `plot_posterior_corner.py` - Visualization

## Further agent lore (outside this repo)
- `oshaughnessy-junior/rift-integrator-lore`: samplers, option combinations, recommended configs.
- `oshaughnessy-junior/rift-review-lore`: merge gates for critical-path changes.
- `oshaughnessy-junior/rift-profiling-lore`: cost model and benchmarks.
