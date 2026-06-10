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
3. **Validate (apples-to-apples).** Draw a posterior *from the reloaded artifact*
   whose target is the run's **actual** `lnL + ln prior` — using RIFT's own priors,
   sampled in RIFT's own coordinates (spins in `(χ, cosθ, φ)` where the isotropic
   prior is flat and there's no Cartesian 1/χ² singularity; the non-uniform mass
   prior `mc_prior ∝ mc`, `eta_prior ∝ η^(−6/5)`; the `alignedspin-zprior` for
   aligned runs) — by Gaussian importance sampling in low dimension or gradient-based
   **NUTS** (using the artifact's `jax.grad` lnL) in high dimension. Then report the
   Jensen–Shannon divergence of the `mc` / `q` / `chi_eff` marginals against the CIP
   posterior, with an **ESS-based quality flag** so a sampling-limited result is never
   mistaken for surrogate error. Writes `posterior_interp.dat`, `report.json`,
   `summary.md`, and `marginals.png`.

## Environment

Runs in the `rift_ad_export` conda env (a clone of `rift_jax` so the shared env is
never modified): jax 0.9.2, numpyro, flowMC 0.6.0, tinygp, RIFT. `gaussian`/`nuts`
need only jax+numpyro; `--sampler flow` needs flowMC (this env has a 0.6.0-correct
flow sampler built into the tool — the legacy `jax_cip.sample_flow_is` breaks under
flowMC 0.6.0's keyword-only `Sampler` API). Always export `PYTHONPATH` to the RIFT
source tree.

## Usage

```bash
PY=~/.conda/envs/rift_ad_export/bin/python
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

## Surrogate tuning (mc/q fidelity) — a fully-diagnosed case study

On the 8-D precessing **S240426s** (mc∈[30,90]), `mc` (JS ~0.015) and `chi_eff`
(JS ~0.009) are apples-to-apples good, but `q` sits at JS ~0.056 with interp mean
0.638 vs CIP 0.718 (a low-q tail; see `marginals.png`). We chased this exhaustively;
the result is a clean negative on every cheap lever and a precise root cause.

**Ruled out** (each tested directly):
- *Prior Jacobian factor.* Keeping the `(1−4η)^(−1/2) = d(delta_mc)/dη` factor
  (`eta_full`) sends q to 0.91 (JS 0.35) and wrecks mc → the `η^(−6/5)` prior
  (RIFT's `delta_mc_prior`) is correct; the factor genuinely cancels when sampling
  in `delta_mc`.
- *Smoothing length.* `--ls-hi-frac` 1.0→0.25 leaves q at 0.059 (RMSE only worsens).
- *Core curvature.* `--keep-curv-frac` 0.05→0.6 hurts (global quadratic over a wide
  mc range); 0.05→0.002 does **nothing** (the local Fisher is genuinely flat in
  `delta_mc`, so there is no curvature to retain).
- *GP capacity.* `--n-features` 256→512, `--cap-points` 8k→16k: no change. An
  **exact** GP residual (no inducing-point sparsity) gives the *same* q (0.641).
- *Data support.* The ILE data covers the full q-range; the lnL-weighted raw data
  gives q≈0.757 (≈CIP) — so the data's likelihood does favour high q.

**Root cause (measured).** Binning surrogate-lnL vs data-lnL by q shows the smooth GP
**over-predicts lnL in the q∈[0.3,0.5] shoulder by ~1 nat** (and under-predicts the
high-q peak by ~0.1), i.e. exp(+1)≈2.6× too much shoulder weight → q pulled down.
This is intrinsic to a **smooth** surrogate: CIP fit this run with `--fit-method rf`
(Random Forest), whose piecewise-constant partitions impose a sharper shoulder; a GP
rounds it. Our artifact **must** be a smooth GP to be `jax.grad`-able, so this is the
AD-vs-RF trade-off in the least-constrained direction — *not* a tunable defect.

**Bottom line:** defaults (`--method quadgp --keep-curv-frac 0.05`, lengthscale
`[0.2,1.0]`) are the tuned optimum. `mc`/`chi_eff` are PE-grade and apples-to-apples;
the residual `q` (~0.05 bit) is a smooth-GP-vs-RF limit. Closing it needs a different
*differentiable* representation of the shoulder (e.g. a shape/monotonicity-constrained
GP), not knob-jittering.

## Interpreting the JS divergence

Because the validation now samples `exp(lnL + ln prior)` with **RIFT's own priors**
(mass-ratio prior ∝ η^(−6/5); isotropic uniform-magnitude spin prior, sampled in
`(χ,cosθ,φ)`; `alignedspin-zprior` when used), the comparison is apples-to-apples:
a non-zero JS reflects *surrogate* error, not a prior-convention mismatch. The report
carries a bootstrap stderr and an ESS quality flag; pool more CIP samples (raise
`--n-output-samples` upstream) if a small JS is statistics-limited.


JS is in bits (0 = identical marginals). For PE-grade agreement expect a few × 10⁻³
to ~10⁻² bits on `mc`; weakly-constrained directions (`q`, `chi_eff`) are wider and
more sensitive to the number of independent samples in *both* posteriors — the
report carries a bootstrap stderr so you can tell when a large JS is just
statistics-limited (pool more CIP samples / raise `--n-output-samples` upstream).
```
