# jax_gp — scalable, AD-compatible likelihood interpolation for RIFT/CIP

JAX-based likelihood interpolators with a shared interface, built for two goals
the legacy CIP fit path does not meet:

1. **Scale.** Fit lnL over N ~ 2·10⁴–5·10⁴ points in d ~ 8–12 without the O(N³)
   blow-up of the exact sklearn GP (and without the `--cap-points` workarounds
   that throw information away).
2. **Differentiable export.** Produce a self-contained `lnL(θ)` that downstream
   users can `jax.grad` through — replacing "dump the lnL grid and hope."

This is an **optional** subpackage. It is never imported by the production CIP
path unless a `gp-jax-*` fit method is selected, so the JAX stack is not required
for normal RIFT operation. Install the extra with:

```
pip install RIFT[jax-interp]      # jax, optax, equinox, tinygp
```

## Methods

| method (`--fit-method`) | class | approach | cost | notes |
|---|---|---|---|---|
| `gp-jax-rff`   | `RFFInterpolator`   | random Fourier features GP (Bayesian linear regression in feature space) | O(N M²) | cheapest export; weaker on sharp/non-stationary peaks |
| `gp-jax-svgp`  | `SVGPInterpolator`  | Titsias collapsed sparse GP (SGPR), M inducing points (k-means init) | O(N M²) | scalable production-regime default; hand-rolled pure JAX |
| `gp-jax-exact` | `ExactGPInterpolator` | exact GP (tinygp) | O(N³) | accuracy reference baseline only — not for production N |

All three use **ARD** (per-dimension lengthscales), which matters for the strongly
anisotropic / curved degeneracies in real lnL surfaces (mc–eta, lambda1–lambda2):
on the curved `banana_ridge` benchmark it lifts gradient-cosine-vs-truth from
~0.78 to ~0.95–1.0.

## Shared interface (`interface.BaseInterpolator`)

```python
from RIFT.interpolators.jax_gp import get_interpolator
model = get_interpolator("svgp")().fit(X, y, y_errors=yerr)

fn   = model.predict_callable()      # callable(np.ndarray[n,d]) -> np.ndarray[n]   (CIP contract)
v, g = model.lnL_and_grad(theta)     # differentiable lnL + gradient at one point
gfn  = model.grad_fn()               # jitted pure-JAX theta -> (lnL, grad)
```

Every method fits on per-dimension *whitened* coordinates and centered targets;
because whitening is affine, JAX threads the chain rule through it, so gradients
come back in physical (fit-coordinate) units automatically. 64-bit JAX is enabled
on import — lnL gradients need it.

## Differentiable export (`export.py`)

```python
from RIFT.interpolators.jax_gp import export
export.save(model, "myfit", coord_names=["mc","eta","chi_eff", ...])
#   -> myfit.npz + myfit.meta.json

loaded = export.load("myfit")        # pure-JAX, differentiable
import jax; grad = jax.grad(loaded.lnL_physical)(theta)
```

The exported lnL is differentiable in the *fit* coordinates the GP was trained on
(recorded in `meta.json` as `coord_names`). Pushing the derivative back to raw
physical parameters would require a JAX reimplementation of CIP's coordinate
transforms — out of scope here, noted as future work.

## Coordinates matter (a lot)

The single biggest lever on fit quality is **which coordinates you fit in** — not
the interpolator. A stationary GP fits a far simpler surface in RIFT's decorrelated
coordinates than in raw `(m1, m2, s1z, s2z, lambda1, lambda2)`. For low-mass / BNS
(e.g. GW170817), fit in `mu1, mu2, delta_mc, LambdaTilde, DeltaLambdaTilde`, where
`mu1, mu2` are Morisaki's orthogonalized PN-phase combinations (`RIFT/misc/tools.py`)
that decorrelate chirp-mass/mass-ratio/spin at low mass.

In CIP this is done with the parameter flags, e.g.:

```
--parameter-implied mu1 --parameter-implied mu2 --parameter-nofit mc \
--parameter delta_mc --parameter-nofit s1z --parameter-nofit s2z \
--parameter-implied LambdaTilde --parameter-implied DeltaLambdaTilde \
--parameter-nofit lambda1 --parameter-nofit lambda2
```

`--parameter` / `--parameter-implied` form the fit coordinates the GP sees;
`--parameter-nofit` are sampled but only used to derive the implied ones. The
conversion is `lalsimutils.convert_waveform_coordinates`; CIP applies it before the
fit. For offline experiments, `benchmark/datasets.to_fit_coordinates` /
`BNS_FIT_COORDS` wrap the same transform. On GW170817, naive→good coordinates cut
SVGP peak-weighted rmse from ~3.0 to ~2.2 nats (and the dimension from 6 to 5).

## Use from CIP

```
util_ConstructIntrinsicPosterior_GenericCoordinates.py \
    --fit-method gp-jax-svgp \
    --fit-save-jax myfit \
    ... (usual CIP args) ...
```

`--fit-save-jax <base>` writes the differentiable export alongside the run.
`--fit-load-gp <base>` reloads such an export instead of refitting.

## Benchmarking

`benchmark/harness.py` sweeps `{method} × {N} × {truth}` against synthetic
ground-truth lnL functions with analytic gradients (in `truth_functions.py`),
scoring value RMSE, peak-weighted RMSE, gradient accuracy, and fit/predict time
(`metrics.py`). The exact GP also serves as the yardstick when no analytic truth
is available.

```
python -m RIFT.interpolators.jax_gp.benchmark.harness --d 8 --N 2000 8000 \
    --methods rff svgp exact
```

`benchmark/scaling_study.py` sweeps the scalable methods across dimension, N, and
surface shape, writing one JSON line per cell as it completes (crash-resilient):

```
python -m RIFT.interpolators.jax_gp.benchmark.scaling_study --out study.jsonl \
    --dims 8 12 --N 2000 20000 --methods svgp rff
```

`benchmark/datasets.py` (`load_ile_net`) loads real RIFT ILE `.net`/`.composite`
output so methods can be tested on production lnL surfaces, not only synthetics.

## Tests

```
python -m RIFT.interpolators.jax_gp.test_interpolators
```

Checks recovery of a known target, AD-vs-finite-difference gradient agreement,
and export round-trip (including `jax.grad` on the reloaded model) for all three
methods.
