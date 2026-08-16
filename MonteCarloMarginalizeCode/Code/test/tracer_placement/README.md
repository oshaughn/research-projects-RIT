# RIFT.misc.tracer_placement test environment

Self-contained [pixi](https://pixi.sh) environment for the tracer-placement
engine (`Code/RIFT/misc/tracer_placement/`) and the fit-side behaviour of the
two tracer drop-in tools, so the engine can be tested from any RIFT clone
without a lalsuite install and without polluting your global Python.

The suite itself lives one level up, with the rest of the RIFT tests:
`Code/test/test_tracer_placement_gp.py`.

## Quick run

```sh
# one-time, if you don't have pixi:
curl -fsSL https://pixi.sh/install.sh | bash

cd MonteCarloMarginalizeCode/Code/test/tracer_placement
pixi run test           # full pytest suite
```

Auxiliary entry points:

```sh
pixi run test-minimal   # pytest-free run (the suite has its own __main__)
pixi run demo           # same, and prints the extrapolation table below
pixi run which-suite    # confirm the paths resolved correctly
```

## Why this is separate from `test/hyperpipe/`

`test/hyperpipe/` installs the full lalsuite stack because
`import RIFT.hyperpipe.*` pays for `RIFT/__init__.py`, which imports
`lalsimutils` unconditionally. The tracer engine has no such dependency: the
core is numpy-only, `rf` adds scikit-learn and `rbf` adds scipy, and the suite
loads `tracer_placement` directly off `RIFT/misc` rather than importing the
`RIFT` package. So this environment is python + numpy + scipy + scikit-learn +
pytest and installs in about a minute.

`PYTHONPATH` is deliberately *not* pointed at `Code/`, so a stray
`import RIFT.<anything>` fails loudly here instead of half-working.

The one thing this buys asymmetric coverage on: `util_HyperparameterTracerUpdate.py`
(.dat I/O, numpy-only) is run end-to-end, while `util_ParameterTracerUpdate.py`
(XML I/O via `lalsimutils`) is checked by static parser inspection. Use
`test/hyperpipe/` or the root pixi project if you need to run the event-level
tool for real.

## What the suite proves

| Group | What it proves |
|---|---|
| `test_gp_extrapolates_where_rf_goes_flat` | The headline argument for `gp_linmean`. On a synthetic lnL surface whose peak lies outside the training hull, `rf` is exactly flat with zero gradient and gains nothing outside, while the GP rises monotonically toward the peak and its argmax over the wider box lands outside the sampled region. |
| `test_linear_mean_extrapolates_where_const_mean_reverts` | Isolates the *mean function* as the cause: same kernel, same data, `mean="const"` reverts toward a flat prior away from data. Runs without sklearn. |
| `test_uncertainty_grows_outside_the_hull` | `predict_with_std` is the calibrated sigma `samplers/ucb.py` needs: small on training points, saturating at `sqrt(sf2)` in the unsampled frontier. |
| `test_analytic_grad_matches_finite_difference` | The analytic gradient (used by UCB's `_polish_gradient`) matches finite differences. |
| GP mechanics | Training-data interpolation, output shapes, 1-D parameter spaces, seamless internal chunking of `predict_with_std`, loud rejection of bad input, duplicate training rows absorbed by the Cholesky jitter. |
| Dispatch | `gp_linmean` is registered (including the hyphenated spelling), unknown methods still raise, constructor kwargs pass through `build()`. |
| lnL floor | Default `None` is a pass-through (legacy bit-for-bit); the floor clamps without dropping points; it rescues a GP wrecked by a single -1e9 outlier; it applies across every fit method. |
| Integration | UCB end-to-end on a GP surrogate places outside the old hull; both tracer CLI tools expose `gp_linmean` and `--tracer-lnl-floor-delta`; the hyperpipe yaml-key → CLI-flag table passes the new flag through; a live run of `util_HyperparameterTracerUpdate.py` with `--tracer-fit-method gp_linmean --tracer-lnl-floor-delta 50`. |

`pixi run demo` prints the numbers behind the headline test — peak at x = 3.0,
training data confined to x <= 1.0:

```
      x:      0.900     1.500     2.000     2.500     3.000
 true lnL:   -3.445    -1.758    -0.781    -0.195    -0.000
 gp_linmean:  -3.446   -1.375    0.664    2.619    4.554
 rf:          -3.494   -3.144   -3.144   -3.144   -3.144
```

The `rf` row wobbles in the last digits between runs (see the `random_state`
note below); what does not wobble, and is what the test asserts, is that it is
constant across the four out-of-hull columns.

## When something fails

* **`ModuleNotFoundError: tracer_placement`**: the suite resolves
  `RIFT/misc/tracer_placement` from its own `__file__`, so this means the test
  file has been moved away from `Code/test/`. `pixi run which-suite` should
  print both paths.
* **A `predict_with_std` / Cholesky failure on a real grid** is usually
  duplicate or near-duplicate training rows. The fit escalates jitter six times
  before giving up; if it does give up, the error names the likely cause.
* **`rf` results are not reproducible** between runs even with `--rng-seed`:
  that is a known pre-existing gap — `fits/_rf.py` does not set
  `random_state` on the `RandomForestRegressor`. Not something this suite
  asserts against.
