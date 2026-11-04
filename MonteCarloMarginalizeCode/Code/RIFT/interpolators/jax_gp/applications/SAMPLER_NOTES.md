# Differentiable sampling of the jax_gp surrogate — design note

`diff_sampler.py` demonstrates the concrete payoff of a **differentiable**
`lnL(theta)`: we can sample the (unnormalized) posterior with a **gradient-based**
sampler instead of brute-force Monte Carlo. This is roadmap item 4 in
`../DESIGN.md` ("derivative-aware sampler for CIP") exercised end-to-end on a
fitted surrogate.

## Why gradient-based sampling (vs brute-force MC)

CIP currently turns a fitted `lnL` into a posterior by **brute-force Monte
Carlo** (draw from a proposal, reweight by `exp(lnL)`). That works, but its
efficiency collapses as the posterior gets **sharp**: a proposal tuned to the
prior volume places almost all its samples where `lnL` is negligible, so the
effective sample size (ESS) per likelihood evaluation falls off a cliff. This is
exactly the **high-SNR** regime — the posterior occupies a tiny, often strongly
correlated sliver of parameter space.

A gradient-aware sampler (HMC/NUTS, MALA, normalizing-flow MCMC) uses `∇lnL` to
walk *along* the posterior ridge rather than guessing. Where a random walk has
to shrink its step size to the width of the sharp peak (and then accepts almost
nothing), NUTS follows the gradient and adapts its trajectory length, keeping a
high acceptance rate and decorrelating fast. The GP is what makes this possible:
RF is piecewise-constant and has no usable gradient; the RFF GP exports a smooth,
exact `∇lnL`.

## What the demo measures

`demo_synthetic()` (d=5):

1. builds a **known** sharp, correlated-Gaussian `lnL` (per-direction widths
   ~0.05–0.2, random rotation — a sharp high-SNR-like peak),
2. fits an RFF surrogate to 3000 points (heteroscedastic noise on),
3. samples the surrogate with **NUTS** (`sample_nuts`) and with a **gradient-free
   random-walk Metropolis** baseline (`sample_rwm`) given the **same lnL-evaluation
   budget**, and
4. compares both recovered posteriors against the **analytic** Gaussian posterior
   (the exact product of the known lnL Gaussian and the broad Normal prior).

The "evaluation budget" is matched honestly: NUTS's cost is its total leapfrog
step count (each leapfrog step is one `lnL`+gradient evaluation, reported via
numpyro's `num_steps` extra field), and RWM is run for that same number of
proposals (one `lnL` evaluation each).

### Measured numbers (CPU, gwkokab env, seed 0)

| sampler | wall-clock | lnL evals | ESS (min/dim) | ESS / eval | posterior mean max\|z\| | cov rel-err |
|---|---|---|---|---|---|---|
| **NUTS** (gradient)      | 7.6 s | 12 804 | 1868 | **0.146** | 0.05 | 0.08 |
| RWM (gradient-free)      | 2.7 s | 12 804 |    3 | 0.0003    | 0.83 | 0.80 |
| flowMC (gradient, bonus) | ~30 s | n/a    |  —   | —         | 0.05 | 0.05 |

Surrogate held-out RMSE: **0.008** lnL units (the RFF fit is essentially exact on
this smooth target).

**Headline:** NUTS achieves **~570× higher ESS per lnL-evaluation** than the
gradient-free baseline on this sharp posterior, and recovers the analytic
posterior mean to ≤0.05σ in every dimension with an 8% covariance error. The
random-walk baseline, given the identical budget, has a **0.1% acceptance rate**,
≈3 effective samples, and a badly biased posterior (max 0.83σ mean error, 80%
covariance error) — the textbook failure mode of brute-force MC on a sharp peak,
and precisely the regime where the differentiable GP earns its (slower) fit cost.

flowMC (normalizing-flow + MALA, also gradient-based) is wired up as a
best-effort bonus and recovers the posterior just as well (max\|z\|=0.05, 5% cov
error); it is heavier to set up and is **not** the required path. If its
constructor API drifts in a future release, `sample_flowMC` catches the exception
and skips with a logged note so the NUTS demo is never blocked.

## Limitations (read before over-claiming)

- **We sample the FIT, not the true likelihood.** The posterior recovered here is
  the posterior of the *surrogate* `lnL`, including any GP fit error. On this
  synthetic the surrogate is near-exact (RMSE 0.008), so surrogate error is not
  the bottleneck; on real, noisier ILE data it will be, and the ESS gain must be
  weighed against fit fidelity.
- **Fit coordinates, not raw physical parameters.** `lnL_physical` is
  differentiable in the GP's *fit* coordinates (`model.coord_names`). Pushing the
  gradient back to raw physical parameters needs a JAX reimplementation of CIP's
  coordinate transforms — deliberately out of scope here (same caveat as
  `export.py`).
- **Prior is a convenience, not the science prior.** The demo uses a broad Normal
  around `x_mean` (scale `3·x_std`) to (a) localize the relevant region and (b)
  keep NUTS in a well-behaved unconstrained space. A real run substitutes the
  actual astrophysical prior; the analytic-posterior comparison accounts for this
  exact broad-Normal prior so the recovery check is apples-to-apples.
- **Synthetic is Gaussian.** A Gaussian posterior is the friendly case for both
  the analytic comparison and NUTS. Multimodal / heavy-tailed real posteriors are
  where flowMC (global proposals) earns its keep over plain NUTS.

## Path to a real CIP integrator swap

1. Fit/export a real ILE `lnL` (the existing `export.save` / `export.load` path;
   e.g. the GW170817 artifact). `diff_sampler.py --artifact <base>` already loads
   such a bundle and runs NUTS on it (guarded by `export.exists`).
2. Replace the demo's broad-Normal prior with CIP's actual intrinsic-parameter
   prior, expressed in fit coordinates (or add the JAX coordinate transform so the
   prior can be stated physically).
3. Swap CIP's brute-force integrator for `sample_nuts` to draw posterior samples
   and estimate the evidence (NUTS does not give the normalization directly —
   pair with thermodynamic integration / bridge sampling, or use the flow's
   density for an importance-sampling evidence estimate).
4. Validate against the existing brute-force CIP posterior on a few events before
   making it a selectable integrator, then quantify the function-evaluation
   savings — the GP's whole reason to exist (fewer expensive ILE evaluations) plus
   this sampler's per-evaluation efficiency compound here.

## API

```python
from RIFT.interpolators.jax_gp import get_interpolator
from RIFT.interpolators.jax_gp.applications.diff_sampler import (
    sample_nuts, sample_rwm, sample_flowMC, demo_synthetic,
)

model = get_interpolator("rff")(n_features=512, n_opt_steps=300).fit(X, y, y_errors=yerr)
res = sample_nuts(model, num_warmup=500, num_samples=2000)
# res["samples"], res["ess_min"], res["n_grad_evals"], res["wall_clock"], res["mean"], res["cov"]
```

Run the demo:

```bash
cd MonteCarloMarginalizeCode/Code
PYTHONPATH="$PWD:$PYTHONPATH" python \
  RIFT/interpolators/jax_gp/applications/diff_sampler.py --demo synthetic
# optionally also sample a real exported artifact:
#   ... --artifact /tmp/gw170817_rff
```
