# Export & validate a differentiable GP likelihood

A breadcrumb demo for the `jax_gp` likelihood-export workflow: how to **(a) create**
a self-contained, differentiable `lnL(theta)` surrogate from an ILE `.net` file, and
**(b) validate** that it is unbiased against the production CIP+RF posterior.

This exercises the optional `RIFT.interpolators.jax_gp` subpackage (JAX stack:
jax, numpyro, tinygp, flowMC). It is an *offline / handoff* workflow — GP training is
slow on purpose; the goal is a portable, differentiable, **validated** likelihood for
downstream use (population inference, differentiable samplers, cross-pipeline reuse),
not to beat the production random forest on speed.

## Files

| file | role |
| --- | --- |
| `config.sh` | shared, override-able config (interpreter, input `.net`, RF benchmark glob, prior box, surrogate/sampler sizes). **Source this first.** |
| `01_create_likelihood.sh` | **(a)** package the `.net` into a differentiable `<base>.npz` + `<base>.meta.json` bundle, then cold-reload it and confirm `jax.grad` works. |
| `02_validate.sh` | **(b)** draw a posterior from the same surrogate with mu-frame-preconditioned NUTS, then print the Jensen-Shannon divergence of every 1D marginal vs the RF benchmark. |

## Running

In an environment that has both RIFT (this checkout) and the JAX stack — on the dev
box that is the `gwkokab` conda env:

```bash
cd demo/rift/export_likelihoods
source config.sh            # edit PY / NET / BENCHMARK_GLOB / MC_RANGE for your event
./01_create_likelihood.sh   # (a) create + export the differentiable likelihood
./02_validate.sh            # (b) validate it against the RF benchmark
```

Defaults reproduce the GW170817-like development case (`NET=/home/oshaughn/all.net`,
benchmark fleet under `/home/oshaughn/jaxcip_benchmark/out/`). To run a different
event, override the variables in `config.sh` (they all honor pre-set values) and
point `BENCHMARK_GLOB` at your own RF+AV reference (build one with
`../../../RIFT/interpolators/jax_gp/applications/benchmark_condor/`).

## (a) What "create" produces

`export_artifact.py` loads the ILE points (with per-point MC errors, sigma-cut,
dedupe), transforms to **decorrelated BNS fit coordinates** (`mu1, mu2, delta_mc,
LambdaTilde, DeltaLambdaTilde`), fits the **quadgp** surrogate (quadratic Fisher core
+ GP residual — the PE-grade choice for the razor-sharp chirp-mass peak), and writes a
two-file bundle:

```
<base>.npz        whitening vectors + quadratic core + (nested) residual-GP params
<base>.meta.json  schema, method, target scaling, coord_names, residual meta
```

`RIFT.interpolators.jax_gp.export.load(base)` reconstructs a **pure-JAX, differentiable**
predictor with no RIFT/lalsimutils dependency at load time — `jax.grad(model.lnL_physical)`
works out of the box. The script's step [2] proves this on a cold reload.

## (b) What "validate" checks, and why this sampler

Export plumbing only proves the bundle round-trips — **not** that the surrogate is any
good. We validate against the production answer: draw a posterior from the surrogate
and measure the **Jensen-Shannon divergence** (bits) of each 1D marginal vs a pooled
CIP+RF+AV reference posterior. The PE target is JS ~ few×10⁻³ bits. (ESS is *not* the
success metric — an efficient sampler can still be badly biased; only JS vs the
benchmark certifies it.)

The sampler is **mu-frame-preconditioned NUTS** (`--sampler nuts-mu`). The posterior is
a razor-thin ridge in chirp mass; importance sampling under-explores the weakly
constrained directions (mass ratio, tides) and they come out too narrow, while plain
NUTS cannot even find a step size on the ill-conditioned ridge. We precondition NUTS's
mass matrix with the well-conditioned covariance built in the decorrelated mu frame
and pulled back to the sampling coordinates, so NUTS explores the wings *and* resolves
the sharp peak — the JS test then reflects the **surrogate**, not the sampler.

## Expected output (demo defaults, ~minutes on CPU)

The differentiability check in step (a) prints:

```
  method        : quadgp (residual: svgp )
  grad finite + jax.grad matches lnL_and_grad: True
```

and step (b) prints a JS table. With the committed demo defaults (`CAP_POINTS=6000`,
`N_OPT_STEPS=150`, 2×3000 NUTS samples) a representative run gives (bits):

| mc | delta_mc | s1z | s2z | lambda1 | lambda2 | LambdaTilde |
|----|----------|-----|-----|---------|---------|-------------|
| 0.020 | 0.054 | 0.011 | 0.008 | 0.026 | 0.015 | 0.079 |

— a uniformly small JS across every marginal (contrast the earlier importance-sampling
baseline, which reached 0.36 on `delta_mc` alone). Exact numbers move a little with
seed and config; the spins sit essentially at the PE bar, and the residual gap on the
chirp-mass width and the broadest directions is **surrogate/data-limited** (it shrinks
with more points and opt steps), not sampler-limited.

**Paper-grade run.** For tighter numbers, bump the sizes — the development reference
used `CAP_POINTS=12000 N_FEATURES=800 N_OPT_STEPS=250 NUM_WARMUP=800 NUM_SAMPLES=4000`
(≈8 min on the dev CPU), which pushes `LambdaTilde` to ~0.028 and the spins to ~0.008.

## See also

- `../../../RIFT/interpolators/jax_gp/README.md` — methods, the interpolator zoo, the export API.
- `../../../RIFT/interpolators/jax_gp/HANDOFF.md` — current status and next steps.
- `../../../RIFT/interpolators/jax_gp/applications/compare.py` — the JS metric.
- `../../../RIFT/interpolators/jax_gp/applications/benchmark_condor/` — build your own RF reference.
