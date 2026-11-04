# GP ↔ RF head-to-head test (BNS GW170817)

The concrete, runnable test behind the methods write-up: does the differentiable GP
surrogate reproduce RIFT's production random-forest (RF) fit **well enough for
inference**? It runs against a fixed, committed example (`data/all.net`, see
`PROVENANCE.md`) so we can re-run it many times while tuning settings.

> Science is never "validated", only tested-so-far. These numbers are a snapshot on
> one BNS example; revisit as the code and settings change.

## Idea (one paragraph)

RF is RIFT's robust, evaluate-anywhere fit — treat it as an **accuracy oracle on its
support** (the ILE points). We fit the GP on the **on-support** high-lnL backbone (the
points that inform lnL, peak-outward), with clean RF-smoothed / noise-modelled targets,
and a far-field residual→0 anchor for tail safety. We then ask two questions: (A) does
the GP reproduce the lnL **surface** over the dynamic range, and (B) does any residual
**move the posterior**? (B) is the one that matters; surface error turns out to be an
over-strict proxy because the quadratic Fisher core carries the razor-sharp `mc` width
through to the posterior. See the paper / `docs/` for the full story.

## Run it

```bash
make help          # all targets + the PY / CODE / BENCH settings
make all           # surface + posterior + figures  (~15 min on CPU)
# or piecewise:
make surface       # RF+GP on a train split, evaluate held-out  -> results/surface.npz
make posterior     # GP on the backbone, mu-frame NUTS           -> results/gp_posterior.npz
make figures       # -> paper/figures/{relerr_vs_lnL,corner_test}.png
```

Needs an env with RIFT + lal + the JAX stack (numpyro, tinygp, corner, matplotlib) —
the `gwkokab` conda env on the dev box (`make PY=/path/to/python ...` to override).
`BENCH` points at your production RF+AV posterior fleet for the corner overlay (optional).

## Figures

- **`relerr_vs_lnL.png`** — relative error `lnL_a − lnL_b` vs `lnL`: GP−RF
  (surrogate-vs-surrogate), and GP−data / RF−data (both vs leave-some-out held-out
  points). Shows the surface agreement and how much of the residual is shared
  (correlated) smoother error vs MC noise.
- **`corner_test.png`** — the GP posterior (mu-frame NUTS) overlaid on the production
  RF+AV benchmark. The inference-level head-to-head.

## Files

| file | role |
| --- | --- |
| `lib.py` | oracle (RF) + on-support design + boundary framework + GP fit |
| `run_test.py` | `--stage surface` / `--stage posterior`; saves arrays to `results/` |
| `make_figures.py` | builds the two figures into `paper/figures/` |
| `Makefile` | idiot-proof targets (above) |
| `data/all.net` | the fixed test data (`PROVENANCE.md`) |

Related: `../` (create + validate export demo), `../validation/` (full-pipeline
GP-vs-standard ladder — the larger-scale plan we test against over time).
