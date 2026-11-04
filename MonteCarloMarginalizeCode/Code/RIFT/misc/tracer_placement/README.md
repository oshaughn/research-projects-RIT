# RIFT.misc.tracer_placement (phase H0 — engine layout)

This is the shared engine package for tracer-based iterative grid placement.
It is consumed by:

- `util_ParameterTracerUpdate.py` (event-level RIFT)
- `util_HyperparameterTracerUpdate.py` (RIFT hyperpipeline)

Both tools are thin I/O wrappers; the math lives here.

## Layout

```
samplers/         — kernels with the production signature
  smc_mala.py     — tempered SMC + MALA moves; accepts surrogate, surrogate_prev
  birth_death.py  — Langevin + kNN birth-death corrector
  smc_mala_bd.py  — composite: smc_mala then birth_death rejuvenation
  surrogate.py    — legacy in-engine quadratic helper (used by toys, not by tools)
  _knn.py         — numpy-only kNN helpers (no scipy)
fits/             — surrogate builders the tools call
  __init__.py     — exposes build(method, X, Y, sigma=None, lnl_floor_delta=None)
  _rf.py          — RandomForest (default, production)
  _rbf.py         — scipy RBFInterpolator
  _quadratic.py   — Tikhonov-regularized quadratic (smoke tests only)
  _polynomial.py  — degree-N polynomial (default 3)
  _gp_linmean.py  — linear-mean RBF GP, numpy-only; extrapolates + real sigma
  _base.py        — FitBase with FD gradient
  _dispatch.py    — build(), plus the optional lnL floor
```

## Extrapolating fits, and why the mean function matters

`rf` (the production default) is piecewise-constant: outside the convex hull of
the training points it is exactly FLAT (`smooth_gradient = False`). When the lnL
peak is clipped at a box edge — the grid was drawn too narrow and lnL is still
rising as it leaves the sampled region — a flat surrogate gives placement
nothing to chase and the next iteration re-piles points on the wall.

`gp_linmean` fits an RBF GP with a LINEAR MEAN, so extrapolation follows the
fitted global trend outward instead of relaxing to a flat prior. (A zero-mean GP
has the same failure as `rf` here, and worse: it relaxes to 0 — cf. CIP's
`--lnL-shift-prevent-overflow` help text.) It also exposes a calibrated
`predict_with_std`, which is what `samplers/ucb.py` wants for
`mu + kappa*sigma`. Pass `mean="const"` for the conservative behaviour.

Ported from the R3 kilonova placement study, where the same construction
recovered a lnL peak clipped at the `v_outer` box edge.

## lnL floor vs lnL cut

`build(..., lnl_floor_delta=D)` — CLI `--tracer-lnl-floor-delta` on both tools,
**default off** — clamps training lnL at `max(lnL) - D` rather than cutting
those points as RIFT does elsewhere
(`indx_ok = Y > np.max(Y) - opts.lnL_offset`). With catastrophic-fit outliers
(a failed model can land lnL at -1e9) cutting discards the geometry of the
known-bad region entirely; clamping keeps those points as anchors that still
pin the surrogate's length scale and signal variance. With the default `None`
the training data is passed through untouched.

## Sampler signature

All three samplers expose:

    iterate(particles, *, surrogate, surrogate_prev=None,
            prior_box, rng, state=None, **kw) -> (X_new, info)

`info` always contains `"state"` (the updated state dict the caller must persist
between iterations for adaptive step-size etc.).

## Status

- [x] H0: layout, sampler signature unification, no module-level state.
- [ ] H1: drop into util_HyperparameterTracerUpdate.py end-to-end on the demo
      hyperpipe Gaussian. Validate the bimodal demo path.
- [ ] H2: full toy test suite per `../test_suite_hyperpipe.md`.

## Promotion to RIFT

When this package is promoted, the target path is
`MonteCarloMarginalizeCode/Code/RIFT/misc/tracer_placement/`. The two tools
should import as:

    from RIFT.misc.tracer_placement import samplers, fits

No other RIFT changes are required (see ../decision_no_cip_changes.md).
