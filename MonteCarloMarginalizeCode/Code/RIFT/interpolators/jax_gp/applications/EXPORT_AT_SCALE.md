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
| `--mass-coord` | `eta` | second mass coordinate to fit in: `eta` (Fisher-quadratic; correct `q`) · `delta_mc` |
| `--keep-curv-frac` | `0.01` | keep core eigen-curvature above this fraction of max (small ⇒ retains the gentle eta curvature) |
| `--sampler` | `auto` | validation sampler: `auto` (nuts if >3 fit dims, else gaussian) · `gaussian` · `nuts` · `flow` |
| `--n-samples` | 40000 | gaussian importance-sampling proposal draws |
| `--cap-points` | 8000 | stratified ("tree-ring") downselect of ILE points before the fit |
| `--n-features` | 256 | SVGP inducing points / RFF features |
| `--lnL-offset` | 40 | keep `lnL > max − offset` |
| `--no-plot` | — | skip `marginals.png` (used by condor jobs) |

## Use cases / coverage

| case | status | notes |
|---|---|---|
| **(a) precessing** (8-D: mc, eta, s1x..s2z) | ✅ supported | spins sampled in `(χ,cosθ,φ)`; all 11 marginals reported |
| **(b) aligned** (2–4-D: mc, eta, [s1z,s2z]) | ✅ supported | zero-spin / aligned-spin runs; constant spin columns auto-dropped |
| **(c) + distance export** (`*.dgrid` / `all_dgrid.dat`) | 🚧 detected, **not yet exported** | `discover_run` sets `has_dgrid`; the run's *intrinsic* `all.net` export still runs and validates. The (intrinsic + luminosity-distance) surrogate is the **next active track** — the dgrid data is still being produced. |

The all-parameter JS (masses, `chi_eff`, `chiMinus`, cylindrical-polar spins) lets you
see immediately which physical direction a given run gets wrong — e.g. low-mass events
tend to stress the *aligned-spin* (`chi_eff`/`s1z`/`s2z`) direction.

## Validated on

| run | dims | method | sampler | ESS | JS mc | JS q | JS chi_eff |
|---|---|---|---|---|---|---|---|
| distance_grid_e2e (aligned, 2-D, mc∈[23,35]) | 2 | quadgp | gaussian | ~30000 | 0.008 | 0.011 | 0 (no spin) |
| S240426s v5PHM (precessing, 8-D, mc∈[30,90]) | 8 | quadgp | nuts | 2100 | 0.007 | 0.011 | 0.008 |

All three marginals are PE-grade and apples-to-apples on both runs, using the defaults
(`--method quadgp --mass-coord eta --keep-curv-frac 0.01`). See the tuning note below
for why the fit coordinate (`eta`) is what makes `q` work.

## Surrogate tuning (mc/q fidelity) — the fit-coordinate matters

The mass-ratio (`q`) marginal is recovered correctly only when the surrogate is fit
in the variable the lnL **Fisher is actually quadratic in: `eta`, not `delta_mc`.**
Since `eta = ¼(1−delta_mc²)`, the curvature in `delta_mc` at the peak is suppressed by
`delta_mc*²` (it vanishes toward equal mass) — so in `delta_mc` the quadratic core
sees a *flat* direction it cannot capture, the GP residual must carry the whole `q`
falloff, over-smooths it, and the posterior grows a spurious low-q tail.

Fix (now the **default**): fit in `eta` (`--mass-coord eta`) with
`--keep-curv-frac 0.01` so the core *retains* the now-real eta curvature, while still
**sampling in `delta_mc`** (smooth prior `∝ eta^(−6/5)`, no equal-mass singularity;
better NUTS geometry). On the 8-D precessing **S240426s** (mc∈[30,90]):

| fit coord | keep_curv_frac | JS mc | JS q | JS chi_eff | q (interp/CIP) |
|---|---|---|---|---|---|
| delta_mc | 0.05 | 0.015 | 0.056 | 0.009 | 0.638 / 0.718 |
| **eta** | **0.01** | **0.007** | **0.011** | **0.008** | **0.694 / 0.718** |

i.e. `q` JS **5×** better and `mc` **2×** better — all three now PE-grade and
apples-to-apples. The reported `holdout_rmse` is over the peak region (within 15 nats);
the eta quadratic core extrapolates steeply in the deep low-lnL tail (`holdout_rmse_all`
is large but that region has ~zero posterior weight).

Other knobs (`--ls-lo-frac/--ls-hi-frac` smoothing length, `--n-features`,
`--cap-points`) are second-order once the fit coordinate is right; raising
`keep_curv_frac` (fewer core directions) *re-hides* the eta curvature.

### Spin: the same lesson — fit in `(chi_eff, chiMinus)`, not `(s1z, s2z)`

The 139-event O4b sweep showed the median event PE-grade on all 11 params, but
**low-mass events fail on aligned spin** (`chi_eff` JS median: mc 0–15 → 0.40, 15–30 →
0.07, 30–60 → 0.008 — a 50× mass gradient). Cause: low-mass systems measure aligned
spin sharply, and the well-measured `chi_eff` is a **diagonal ridge** in `(s1z, s2z)`
that an axis-aligned ARD GP + per-dimension-whitened quadratic core over-smooth.

Fix (default `--spin-coord aligned_eff`): rotate the aligned-spin fit coordinates to
the Fisher principal axes `(chi_eff, chiMinus)` — short ARD lengthscale on the sharp
`chi_eff`, long on the broad `chiMinus` — while still **sampling** in the smooth
spherical spin coords (the per-body spin-sampling structure is taken from the *raw*
physics, not the fit-coordinate names — a subtlety: inferring it from `fit_names`
breaks once `s1z`/`s2z` are replaced by `chi_eff`/`chiMinus`).

| event | mc | `cartesian` chi_eff JS | `aligned_eff` chi_eff JS |
|---|---|---|---|
| S250119cv | ~10 | 0.505 | **0.016** |
| S240413p | ~6.4 | 0.192 | **0.101** |
| S240426s | ~60 | 0.011 | **0.005** |

No regression at high mass; the very-low-mass extreme (mc~6) is halved but remains the
hardest case. Across the 53 failing events `aligned_eff` improved 40 (15 now pass) and
the low-mass `chi_eff` median dropped 0.40→0.054 — **but it regressed 9 events** below
cartesian (real, not sampling noise).

So the default is **`--spin-coord auto`**: fit *both* `aligned_eff` and `cartesian` and
keep the one with the lower peak-region holdout RMSE. Holdout RMSE reliably picks the
better coordinate (it selects `cartesian` exactly where `aligned_eff` would regress),
so `auto` captures the wins and is **never worse than `cartesian`** — e.g. S250119cv →
aligned_eff (chi_eff 0.016), S240513ei → cartesian (avoids the 0.61 regression). Cost is
~2× the fit (two fits, one sample). Defaults: `--method quadgp --mass-coord eta
--spin-coord auto --keep-curv-frac 0.01`.

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
