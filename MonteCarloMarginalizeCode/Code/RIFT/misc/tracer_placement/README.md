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
  __init__.py     — exposes build(method, X, Y, sigma=None)
  _rf.py          — RandomForest (default, production)
  _rbf.py         — scipy RBFInterpolator
  _quadratic.py   — Tikhonov-regularized quadratic (smoke tests only)
  _polynomial.py  — degree-N polynomial (default 3)
  _base.py        — FitBase with FD gradient
```

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
