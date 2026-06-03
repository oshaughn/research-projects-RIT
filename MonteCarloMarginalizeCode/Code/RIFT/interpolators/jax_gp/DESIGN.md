# jax_gp — design & rationale

Status: living design doc for the AD-compatible likelihood-interpolation effort.
See `README.md` for usage; this file is the *why* and the long-term plan.

## Problem

CIP fits `lnL` over intrinsic parameters, then samples it. Two long-standing
limitations motivate this work:

1. **Scaling of the exact GP.** The legacy sklearn `GaussianProcessRegressor`
   path is O(N³); at our scale (N ~ 2·10⁴–5·10⁴, d ~ 8–12) it is intractable, so
   the pipeline leans on `--cap-points`, `--lnL-offset`, pooling, etc.
2. **Non-differentiable export.** We export lnL grid evaluations (or a NumPy
   sklearn pickle / a black-box NN) and hope. Downstream users who need a
   *differentiable* `lnL(θ)` cannot get one.

## The honest competitive picture: RF is the bar

Random forests (RIFT's default `fit_rf`, ExtraTrees) are **excellent** and very
hard to beat on raw timing+accuracy:

- robust, no hyperparameter tuning,
- cover the full lnL dynamic range,
- extremely fast to fit.

Measured, GW170817 (good coords, lnLmax−20 band, ~7.6k train pts):

| method | peak-wtd rmse | fit time | differentiable? |
|---|---|---|---|
| RF (ExtraTrees) | 1.84 | **5 s** | **no** |
| SVGP (this work) | 1.64 | 432 s | **yes** |

So the GP is *not* going to win a straight timing+accuracy race against RF, and
we should stop pretending it will. **The GP's reason to exist is different:**

1. **Fewer function evaluations.** RF needs large training volumes to be accurate;
   a GP reaches comparable accuracy from far fewer points. Our training points are
   *expensive to make* (each is an ILE evaluation), so "accurate with less data"
   directly reduces the dominant cost — even if the fit itself is slower. Reducing
   function evaluations is an explicit program goal, not a side benefit.
2. **Automatic differentiation.** RF is piecewise-constant — no usable gradient.
   The GP gives a smooth, exact `∇lnL`. This is the capability several downstream
   applications *cannot do without* (below).

The GP's extra fit cost is acceptable **if** it is part of a cohesive AD tooling
set we need for other reasons — which it is.

## Why AD is critical (the applications driving this)

These are the concrete reasons a differentiable `lnL(θ)` is a hard requirement,
not a nicety:

1. **Population inference (AD / numpyro).** Hierarchical/population analyses built
   on AD frameworks (numpyro, etc.) consume *individual-event* likelihoods and need
   their **derivatives**. A differentiable per-event lnL surrogate (this package's
   export) is the missing piece that lets per-event results flow into gradient-based
   population inference.
2. **Differentiable sampling (replace brute-force MC).** Our integrals are
   currently done with brute-force Monte Carlo. A **derivative-aware sampler**
   (HMC/NUTS, normalizing-flow / SVI, Langevin, …) is *enormously* more efficient,
   especially at **high SNR** where the posterior is sharp and MC is wasteful. To
   get there:
   - **CIP:** needs (a) **AD fits** — delivered here by the GP — and (b) a
     **derivative-aware sampler** wired into the CIP integrator (pending).
   - **ILE:** needs (a) porting one of the likelihoods from **cupy → JAX** so the
     extrinsic integral itself is differentiable, and (b) a derivative-aware
     sampler. Larger effort; longer-term.

## Long-term roadmap

Ordered roughly by dependency, not committed dates:

1. **(done)** AD-compatible GP fits + a self-contained differentiable export
   (`export.py`). RFF / SVGP / exact behind one interface; heteroscedastic noise;
   ARD; ILE `.net` loader; good fit coordinates.
2. **Tune the scalable GP.** Make SVGP competitive with RFF (more/better inducing
   points, more steps, possibly input warping for sharp peaks); decide the
   per-regime default from `benchmark/scaling_study.py`.
3. **Population-inference hookup.** Provide a clean numpyro-friendly loader so an
   exported per-event lnL drops into an AD population model. (Adjacent: the
   `gwkokab` work.)
4. **Derivative-aware sampler for CIP.** Swap/augment the CIP integrator with an
   HMC/flow sampler that consumes `lnL_and_grad` from the exported GP. This is
   where the GP's AD pays for its fit cost — large-SNR efficiency.
5. **ILE cupy→JAX likelihood port + derivative-aware extrinsic sampler.** The big
   one: a differentiable ILE likelihood end-to-end. Enables differentiable
   extrinsic marginalization, not just a differentiable surrogate of its output.

## Design choices (and why)

- **JAX**, not PyTorch: composability with numpyro / optax / the population-inference
  stack, and a clean pure-function export users can `jax.grad`.
- **Hand-rolled SGPR** (Titsias collapsed bound), not gpjax: no version coupling to
  a fast-moving lib against jax 0.10; the predictive mean stays a transparent,
  exportable closed form.
- **Whitening + ARD + good coordinates** carry most of the fit-quality water.
  Coordinate choice (e.g. `mu1,mu2,delta_mc,LambdaTilde,DeltaLambdaTilde` for BNS)
  matters more than the interpolator; see README "Coordinates matter".
- **Heteroscedastic noise**: ILE lnL has a per-point MC error (the `sigma/L`
  column); using it cuts held-out error by 10–70× on noisy data. Always on when
  errors are available.
- **Export is differentiable in fit coordinates.** Pushing derivatives back to raw
  physical parameters needs a JAX reimplementation of the coordinate transforms —
  deferred (would also be needed for a fully-AD CIP sampler).

## Current benchmark findings (evidence, not final)

- RFF currently **beats** SVGP on the synthetic sweep and is faster
  (d=12, N=2000: RFF rmse 0.26 vs SVGP 1.14; both grad-cosine ≈ 1.0). RFF is the
  strong AD candidate today; SVGP needs the tuning in roadmap step 2.
- Both scalable methods improve with N (good scaling) and have near-exact gradients.
- On GW170817 the GP modestly beats RF on accuracy but not on speed — exactly the
  tradeoff above: we buy AD (and the few-evaluations regime), not raw throughput.
