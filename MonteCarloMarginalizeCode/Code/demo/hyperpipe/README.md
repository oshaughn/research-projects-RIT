# `demo/hyperpipe` — HyperPipe example pipelines

This directory holds runnable demos for the RIFT HyperPipe pipeline.  Each
configuration drives the iterative `MARG → CON → UNIFY → EOS_POST → PUFF →
TEST → next iteration` loop on the same 3-D Gaussian toy, but each one
exercises a different slice of the pipeline so you can pick the one closest
to what you are testing.

The auto-generated documentation at
<https://rift-documentation.readthedocs.io/en/latest/hyperpipe.html>
covers the pipeline design.  `technical_doc.txt` in this directory has a
pedagogical writeup of the procedure and implementation.


## The driver

All hydra-style demos in this directory are launched with the same tool:

```bash
util_RIFT_hyperpipe.py --config ./<one of the .yaml files below>
```

`util_RIFT_hyperpipe.py` reads the yaml, emits `args_*.txt` files into the
rundir specified by `general.rundir`, calls `create_eos_posterior_pipeline`
to assemble the condor DAG (`marginalize_hyperparameters.dag`), and submits
it.  Inspect any rundir's `.sub` files to see exactly what every executable
was invoked with.

Configs in this directory:

| Config                              | Rundir                | What it exercises                                                                   |
|-------------------------------------|-----------------------|-------------------------------------------------------------------------------------|
| `hyperpipe_conf.yaml`               | `rundir/`             | Baseline pipeline: posterior-only resampling (no puff lane).                        |
| `hyperpipe_conf_tracer.yaml`        | `rundir_tracer/`      | Parsimonious-placement (tracer) variant — MARG_PUFF lane suppressed.                |
| `hyperpipe_conf_osg.yaml`           | `rundir_osg/`         | OSG / IGWN submit-host setup (containers, OSG attributes).                          |
| `hyperpipe_conf_linear_uvw.yaml`    | `rundir_linear_uvw/`  | Coordinate transformation: fit in `(u,v,w)` while sampling in `(x,y,z)`.            |


## `hyperpipe_conf.yaml` — baseline

**What it exercises.**  The simplest end-to-end loop: MARG evaluates
likelihoods on the initial grid, the EOS posterior fits a GP/RF, the next
iteration draws candidate grid points by resampling the posterior.  There
is no PUFF lane (no `puff.exe` set) and no tracer placement — the
posterior alone seeds the next iteration's grid.

**Build.**

```bash
util_RIFT_hyperpipe.py --config ./hyperpipe_conf.yaml
# equivalent to:  make rundir
```

**Inspect.**  After the DAG finishes:

```bash
(cd rundir; plot_posterior_corner.py \
     --posterior-file posterior-2.dat \
     --composite-file all.marg_net --composite-file-has-labels \
     --parameter x --parameter y --parameter z \
     --lnL-cut 15 --use-all-composite-but-grayscale)
```

You should see the recovered isotropic Gaussian centred where
`example_gaussian.py` placed it (default `[-5, 0, 0]`).


## `hyperpipe_conf_tracer.yaml` — parsimonious placement

**What it exercises.**  The tracer pathway (`util_HyperparameterTracerUpdate.py`
as `puff.exe`).  This consumes `all.marg_net` directly and writes
`grid-{k+1}.dat`, suppressing the MARG_PUFF lane entirely.  Per the comment
in the config file, the saving is about 1.7–1.8× for `N=5–6` iterations on
this toy — MARG still runs every iteration, but no separate MARG_PUFF
lane is built.

The yaml exposes the SMC-MALA / birth-death sampler hyperparameters in
`puff.settings` so you can twiddle them without escaping into
`extra-args`.

**Build.**

```bash
util_RIFT_hyperpipe.py --config ./hyperpipe_conf_tracer.yaml
# equivalent to:  make rundir_tracer
```

**Inspect.**  Same plot command as above with `--posterior-file
rundir_tracer/posterior-3.dat`, or use the shell helper:

```bash
make rundir_tracer_plots
```


## `hyperpipe_conf_osg.yaml` — OSG / IGWN setup

**What it exercises.**  Submit-host configuration for a condor pool with
OSG attributes and an IGWN-prefixed condor-local nonworker setup.  Same
3-D Gaussian toy as the baseline, but `general.use-osg`,
`general.condor-local-nonworker`, and
`general.condor-local-nonworker-igwn-prefix` are all enabled.  Use this
as a starting point when adapting one of the other demos for OSG or LIGO
clusters.

**Build.**

```bash
util_RIFT_hyperpipe.py --config ./hyperpipe_conf_osg.yaml
```

> **Note.**  As of writing the OSG yaml has a couple of stale keys
> (`explode marg jobs`, `puff factor`) that hydra would reject because
> they use spaces where the schema expects hyphens.  Fix those locally
> before running (`explode-marg-jobs`, `puff-factor`) or copy from the
> baseline yaml.


## `hyperpipe_conf_linear_uvw.yaml` — coordinate transformation

**What it exercises.**  The decoupled-bases path in
`util_ConstructEOSPosterior.py`: the iteration (puff, marg evaluator,
convergence test) operates in the data-file basis `(x, y, z)`, but the
EOS posterior step *fits its GP/RF in a transformed basis `(u, v, w)`*
routed through the `linear_coordinate_convert.py` plugin.  The
likelihood evaluator (`example_gaussian_uvw.py`) uses the same plugin
library to define a Gaussian whose principal axes lie along `(u, v, w)`
with deliberately unequal sigmas — so the rotation is observable in the
recovered posterior.

Two pieces of the new machinery are exercised together:

1. **Coordinate plugin contract.** `post.coord-module:
   linear_coordinate_convert.py` (plus `--supplementary-coordinate-ini`
   and `--supplementary-coordinate-chart uvw_rotated` via
   `post.extra-args`) walks the loader at
   `RIFT.misc.coordinate_plugin.load_coordinate_converter`.

2. **`--parameter-implied` / `--parameter-nofit` semantics in
   EOSPosterior.**  `post.coords-implied: u v w` is the fit basis,
   `post.coords-nofit: x y z` is the MC sampling basis, and
   `post.coords-fit` is empty.  This is the IntrinsicPosterior-style
   coord-arg mechanism, now wired through to EOSPosterior.

**Build.**

```bash
util_RIFT_hyperpipe.py --config ./hyperpipe_conf_linear_uvw.yaml
```

**Inspect.**  The recovered `(x, y, z)` posterior should be an
elongated ellipsoid pointing along the rotated `v`-axis
(`(-1, +1, 0) / sqrt(2)`), with the narrowest extent along the
`u`-axis (`(+1, +1, 0) / sqrt(2)`):

```bash
(cd rundir_linear_uvw; plot_posterior_corner.py \
     --posterior-file posterior-4.dat \
     --composite-file all.marg_net --composite-file-has-labels \
     --parameter x --parameter y --parameter z \
     --lnL-cut 15 --use-all-composite-but-grayscale)
```


## Pre-hydra Makefile-style demos

The Makefile also carries two targets that bypass `util_RIFT_hyperpipe.py`
and drive `create_eos_posterior_pipeline` directly:

- `make Gaussian_adaptive_unimodal` — single-event adaptive run with the
  baseline puffball.  Useful when you want to inspect the
  `args_*.txt` files by hand or when hydra is in the way.
- `make Gaussian_adaptive_bimodal` — two events (`example_gaussian.py`
  + `example_gaussian2.py`) into the same posterior, illustrating
  multi-event heterogeneous-driver mode.

These predate the hydra wrapper; the hydra yamls above are the
recommended entry point for new work.


## Initial grid

Every demo above seeds from `blind_gaussian_3d_xy_plus.dat` (1000 uniform
points covering a corner of the `[-7, 7]^3` cube).  Regenerate it with
`make blind_gaussian_3d_xy_plus.dat` if you delete it.  The Gaussian's
centre is configurable in `example_gaussian.py` — `make
change_center_location` rewrites it via `sed`.
