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
- `--force-xpy`: on a host without cupy, ILE prints ` Override --gpu  (not available)`
  (two spaces; grep `Override --gpu`), sets `opts.gpu=False`, and drops to
  `ViaArrayVector`. With `--force-xpy` the same line still prints, but `--gpu` is
  restored and NoLoop runs on numpy. `--force-xpy` does nothing without `--gpu`, and
  nothing on a host where cupy loads.

The diagnostics `--zero-likelihood` and `--calibration-dump-responsibilities` bypass
NoLoop by design, even with all four flags.

`--gpu` and `--force-xpy` select a code path, not hardware. For a CPU run remove only
`--force-gpu-only`; never remove `--gpu` or `--force-xpy`. `--rotation-slow` and
`--freqresponse` reach their own NoLoop variants with `--time-marginalization
--vectorized` (`--gpu`/`--force-xpy` optional). Without `--time-marginalization`
they run the scalar `FactoredLogLikelihood`, and the startup checks do not catch it.

Who adds the flags:
- `helper_LDG_Events.py` emits `--vectorized --gpu`, and `--time-marginalization`
  under `--propose-ile-convergence-options` (which `util_RIFT_pseudo_pipe.py` always
  passes).
- `create_event_parameter_pipeline_BasicIteration` and its siblings append
  `--vectorized --gpu` under `--request-gpu-ILE`.
- `util_RIFT_pseudo_pipe.py` adds `--force-xpy` only with `--ile-no-gpu` or
  `--ile-sampler-method AV`. A default pseudo_pipe GPU run with another sampler
  therefore reaches NoLoop only if cupy loads on the execute node; otherwise it drops
  to `ViaArrayVector`. `ILE_extr` inherits the same flags.
- Hand-written commands (tests, demos, smoke runs, export-only reruns) must spell
  out all four.

Confirm the path in the log. Each of these lines means the run is not on NoLoop:
- `Override --gpu` on a run without `--force-xpy`;
- `this is the old vectorized code path not the xpy path` (printed with distance
  marginalization);
- `Q_lm stencil DEFAULT ... NOT APPLIED -- this configuration cannot honour a
  sub-sample stencil` (grep `NOT APPLIED`). The same line with a
  `--calibration-fused-kernel` reason is still on NoLoop, with the `nearest` stencil.

An explicit `--interpolate-time` is refused off NoLoop; the default stencil instead
falls back to `nearest`. A non-default `--time-marginalization-quadrature`
(`bandlimited`/`peak-local`) is refused unless NoLoop runs without calibration
marginalization, `--rotation-slow` or `--freqresponse`; `peak-local` also refuses
phase marginalization.

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

## Important CLI tools (`MonteCarloMarginalizeCode/Code/bin/`)
- `util_RIFT_pseudo_pipe.py`: builds a run from an ini; calls `helper_LDG_Events.py`
  for ILE/CIP arguments, then `create_event_parameter_pipeline_BasicIteration` for the DAG.
- `integrate_likelihood_extrinsic_batchmode` (ILE): extrinsic marginal likelihood. See
  the REDLINE above.
- `integrate_likelihood_extrinsic_jax`: JAX ILE driver; selected in pseudo_pipe with
  `--use-jax-ile`. pseudo_pipe refuses it with calibration marginalization, with
  `--lisa-known-sky`, and under `--use-osg` unless `--jax-ile-container-ok` is given.
- `util_ConstructIntrinsicPosterior_GenericCoordinates.py` (CIP): fits lnL over
  intrinsic parameters and draws the posterior.
- `util_RIFT_hyperpipe.py`: hyperparameter pipeline (EOS, population, and similar):
  the iterative marginalize/fit/puff loop over hyperparameters.
- `plot_posterior_corner.py`: visualization.

## Design docs: read before changing a module
Paths are under `MonteCarloMarginalizeCode/Code/RIFT/`. A bare filename is in the same
directory as the first entry in its row. List the current set with
`git ls-files MonteCarloMarginalizeCode/Code/RIFT | grep -E 'DESIGN|HANDOFF|TESTING|REVIEW|BACKENDS'`.

| Area | Start with |
|---|---|
| Likelihood, time marginalization, stencils | `likelihood/DESIGN_q_window_stencil.md`, `DESIGN_time_marginalization_quadrature.md`, `DESIGN_noloop_per_detector_glue.md` |
| Peak-local time marginalization | `likelihood/DESIGN_peak_local_framework.md`, `DESIGN_time_marginalization_peak_local.md` |
| JAX ILE | `likelihood/jax_ile/README.md` and its `DESIGN_*.md` |
| Integrators (AV, GMM, portfolio) | `integrators/TESTING.md`, `REVIEW_CHECKLIST.md`, `DESIGN_portfolio_freeze_policy.md` |
| Calibration marginalization | `calmarg/DESIGN_calmarg_in_loop.md`, `DESIGN_extrinsic_handoff.md` |
| Slow rotation, finite-size response | `likelihood/SLOWROT_HANDOFF.md`, `DESIGN_rotating_freqresponse.md` |
| GP interpolators | `interpolators/jax_gp/DESIGN.md`, `HANDOFF.md` |
| Simulation manager | `simulation_manager/DESIGN.md`, `BACKENDS.md` |

CI tiers, and the GPU checks to run by hand before merging likelihood or JAX
changes: `.travis/PRECOMMIT.md`.

## Further agent lore (outside this repo)
- `oshaughnessy-junior/rift-integrator-lore`: samplers, option combinations, recommended configs.
- `oshaughnessy-junior/rift-review-lore`: merge gates for critical-path changes.
- `oshaughnessy-junior/rift-profiling-lore`: cost model and benchmarks.
- `oshaughnessy-junior/gw-coordinate-lore`: coordinates, frames, priors, boundaries.

Commit-pinned code-structure indexes (sidecars) for `rift_O4d` and `rift_O4c` are
maintained outside this repo. Use them to navigate only: check the commit each was
built from against your checkout, and read the diffs since then and the exact source
before drawing a conclusion.
