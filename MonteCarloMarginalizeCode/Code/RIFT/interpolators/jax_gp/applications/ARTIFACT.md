# Differentiable lnL artifact

`export_artifact.py` turns a RIFT ILE `.net` file (the per-point Monte-Carlo
likelihood evaluations CIP normally consumes) into a small, self-contained,
**differentiable** surrogate for the marginalized log-likelihood `lnL(theta)`.

Build one with:

```bash
python export_artifact.py --net /path/to/all.net --out /tmp/gw170817_rff --coords bns
```

## What the artifact is

A random-Fourier-feature (RFF) regression of `lnL` over the ILE samples, fit in
*decorrelated fit coordinates* using the per-point ILE Monte-Carlo errors
(`sigma_lnL`) as observation noise. Only the informative high-likelihood region is
kept (`lnL > max - lnL_offset`), de-duplicated and sigma-cut exactly as CIP does.
The result is a pure-JAX `lnL(theta)` that `jax.grad` / `jax.value_and_grad`
differentiate out of the box — no RIFT or lalsimutils import is needed to *load* it.

## File format

The export is two files sharing a base path:

- `<base>.npz` — whitening vectors (`x_mean`, `x_std`) plus the RFF parameters
  (frequencies, weights), as NumPy arrays.
- `<base>.meta.json` — schema/method/dimension, target centering/scaling
  (`y_mean`, `y_std`), and `coord_names`: the names of the axes of `theta`.

## Coordinate caveat (important)

The artifact is differentiable in its **fit coordinates** — the list recorded as
`coord_names` in the meta — *not* in the raw physical parameters. For `--coords bns`
these are `('mu1','mu2','delta_mc','LambdaTilde','DeltaLambdaTilde')`; for
`--coords raw` they are the six raw params `(m1,m2,s1z,s2z,lambda1,lambda2)`.
A gradient in fit coordinates is what CIP works in; pushing it back to raw physical
parameters would require a JAX reimplementation of CIP's coordinate transforms,
which is deliberately out of scope. Always read `coord_names` before differentiating.

## How a downstream user loads and differentiates it

```python
import jax, jax.numpy as jnp
from RIFT.interpolators.jax_gp import export
model = export.load("/tmp/gw170817_rff")          # reconstructs pure-JAX lnL
print(model.coord_names)                            # axis order of theta
theta = jnp.array([0., 0., 0., 300., 0.])           # a point in fit coordinates
lnL, grad = jax.value_and_grad(model.lnL_physical)(theta)
print(float(lnL), grad)                             # scalar lnL + finite gradient
```
