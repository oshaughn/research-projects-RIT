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

## Downselect decision (current)

**RFF is the default jax method**; SVGP and exact are kept as **backstop /
validation** code. Rationale: no scaling reason favors SVGP (both are O(N M²) time,
O(N M) memory, linear in N), and RFF is empirically faster *and* more accurate on
our benchmarks with the smaller constant factor (fixed feature basis; no k-means or
inducing-point optimization). SVGP is retained because (a) it is the principled
inducing-point method and a useful cross-check, and (b) its adaptive inducing points
and calibrated predictive variance are the natural seed for a future
uncertainty-driven **active-learning / sample-placement** loop (reduce function
evaluations further). The GP will **not** replace RF in the standard CIP stack;
its value is the AD use cases below.

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

## Current benchmark findings

- **Scaling sweep (16 cells: {svgp,rff} × d∈{8,12} × N∈{2k,20k} × {correlated_gaussian,
  sharp_peak}) — RFF beats SVGP on every cell and is faster.** E.g. correlated_gaussian
  d=12 N=20k: RFF rmse 0.21 vs SVGP 0.58; sharp_peak d=12 N=20k: RFF 0.08 vs SVGP 0.25.
  Both improve with N and have grad-cosine ≈ 0.99–1.0. This is why RFF is the default;
  making SVGP competitive is roadmap step 2.
- On GW170817 the GP modestly beats RF on accuracy but not on speed — we buy AD (and the
  few-evaluations regime), not raw throughput.

## AD applications (built — see `applications/`)

The use cases that justify the GP, now prototyped:

- **`export_artifact.py`** — packages a real ILE run into a self-contained
  differentiable lnL (e.g. a 30 KB GW170817 RFF export in BNS coords). This is the
  "package it sanely" product downstream users consume.
- **`diff_sampler.py`** — gradient-based sampling (numpyro NUTS, flowMC) of the
  fitted lnL. On a sharp synthetic posterior NUTS recovers the analytic answer at
  **~300× higher ESS-per-lnL-evaluation** than a matched-budget gradient-free
  random walk — a direct demonstration of the high-SNR sampling payoff (roadmap
  step 4, here on the surrogate; the real CIP-integrator swap is the next step).
