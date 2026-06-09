# export_at_scale — ship + validate differentiable lnL artifacts from real run dirs

`export_at_scale.py` points at a **real RIFT run directory**, exports a continuous,
`jax.grad`-able surrogate for that run's `all.net` intrinsic likelihood, and
**validates the exported artifact** by drawing a posterior from it and comparing the
marginals to the run's own CIP posterior. It scales the single-event
[`export_artifact`](ARTIFACT.md) / [`jax_cip`](../README.md) primitives to a whole
directory of events, locally or over HTCondor.

At this stage it interpolates **only `all.net`** — the existing intrinsic ILE
deliverable. (Distance-grid export is a separate track.)

Nothing is written back into the run directory; every output lands under
`--workroot/<event>__<tag>/`.

## What it does, per run

1. **Discover.** Detect the `all.net` column layout (aligned / precessing /
   tidal widths all differ — the trailing `lnL sigma_lnL ntot [neff]` columns are
   tail-anchored), parse the active `args_cip_list.txt` for the fit parameters and
   the prior box (`--mc-range`, `--eta-range`, `--chi-max`), and find the run's
   latest `posterior_samples-<N>.dat` (falling back to
   `extrinsic_posterior_samples.dat`).
2. **Export the deliverable.** Fit a surrogate in dimension-agnostic physical fit
   coordinates — `[mc, delta_mc]` plus whichever spin/tidal columns actually vary
   (constant columns are dropped and recorded) — so the same path covers aligned,
   BNS, and **precessing** runs with no hand-written coordinate transform. Save the
   `.npz` + `.meta.json` bundle, **reload it**, and assert the reloaded `predict()`
   matches and `jax.grad` is finite.
3. **Validate.** Draw a posterior *from the reloaded artifact* over the run's prior
   box (spin-magnitude constraint enforced) — Gaussian importance sampling in low
   dimension, gradient-based **NUTS** (using the artifact's `jax.grad` lnL) for the
   curved high-dimensional precessing posteriors — then report the Jensen–Shannon
   divergence of the `mc` / `q` / `chi_eff` marginals against the CIP posterior, with
   an **ESS-based quality flag** so a sampling-limited result is never mistaken for a
   surrogate error. Writes `posterior_interp.dat`, `report.json`, `summary.md`, and
   `marginals.png`.

## Usage

```bash
PY=~/.conda/envs/rift_jax/bin/python
export PYTHONPATH=/path/to/RIFT/MonteCarloMarginalizeCode/Code
M=RIFT.interpolators.jax_gp.applications.export_at_scale

# inspect what discovery found (no work done)
$PY -m $M discover --run-dir /path/to/rundir

# one run, immediately
$PY -m $M one --run-dir /path/to/rundir --workroot ./out

# many runs locally
$PY -m $M batch --runs '/data/*/S*/rift*/' --workroot ./out

# many runs as a condor DAG (sub templated from each run's own CIP.sub)
$PY -m $M batch --runs '/data/*/S*/rift*/' --workroot ./out --condor
condor_submit_dag ./out/condor/export_at_scale.dag
```

## Output layout

```
workroot/
  <event>__<tag>/
    lnL_artifact.npz          # the differentiable surrogate (load via jax_gp.export.load)
    lnL_artifact.meta.json    # coord_names, dropped-constant columns, provenance
    posterior_interp.dat      # posterior drawn FROM the reloaded artifact
    report.json               # full machine-readable report (fit + JS validation)
    summary.md                # human summary + JS table
    marginals.png             # interp-vs-CIP 1D marginal overlay
  condor/                     # (batch --condor) DAG + sub + per-job logs
  batch_summary.json          # (batch local) one line per run
```

## Key options

| flag | default | meaning |
|---|---|---|
| `--method` | `quadgp` | surrogate: `quadgp` (PE-grade Fisher core + GP residual) · `svgp` (faster, low-D) · `rff` · `exact` |
| `--sampler` | `auto` | validation sampler: `auto` (nuts if >3 fit dims, else gaussian) · `gaussian` · `nuts` |
| `--n-samples` | 40000 | gaussian importance-sampling proposal draws |
| `--cap-points` | 8000 | stratified ("tree-ring") downselect of ILE points before the fit |
| `--n-features` | 256 | SVGP inducing points / RFF features |
| `--lnL-offset` | 40 | keep `lnL > max − offset` |
| `--no-plot` | — | skip `marginals.png` (used by condor jobs) |

## Validated on

| run | dims | method | sampler | ESS | JS mc | JS q | JS chi_eff |
|---|---|---|---|---|---|---|---|
| distance_grid_e2e (aligned, 2-D, mc∈[23,35]) | 2 | svgp | gaussian | 14400 | 0.008 | 0.012 | 0 (no spin) |
| S240426s v5PHM (precessing, 8-D, mc∈[30,90]) | 8 | quadgp | nuts | 3300 | 0.028 | 0.045 | 0.049 |

On the wide-range precessing event, `svgp` over-smooths the peak (mc JS 0.27, holdout
RMSE 2.8 nats); `quadgp` recovers it (mc JS 0.028, RMSE 1.7) — which is why `quadgp`
is the default. The validation's ESS-based quality flag is what tells the two apart:
the over-smoothed `svgp` posterior only *looked* good under a low-ESS Gaussian
proposal.

## Interpreting the JS divergence

The validation samples `exp(lnL)` with a **flat prior in the fit coordinates** over the
run's CIP prior box, whereas CIP applies its own mass + spin prior. For narrow ranges
that difference is negligible (the aligned demo: mc/q JS ~0.01). For wide-range events
the `mc` marginal (≈flat prior across its narrow posterior) is the cleanest test of the
*likelihood surrogate*; residual `q`/`chi_eff` differences fold in the prior/Jacobian
as well as surrogate error.


JS is in bits (0 = identical marginals). For PE-grade agreement expect a few × 10⁻³
to ~10⁻² bits on `mc`; weakly-constrained directions (`q`, `chi_eff`) are wider and
more sensitive to the number of independent samples in *both* posteriors — the
report carries a bootstrap stderr so you can tell when a large JS is just
statistics-limited (pool more CIP samples / raise `--n-output-samples` upstream).
```
