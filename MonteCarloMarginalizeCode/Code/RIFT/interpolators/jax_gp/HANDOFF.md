# jax_gp — handoff / resume here

Snapshot for whoever picks this up next (branch `rift_O4d_junior_interp_jax` → `junior`).
Read alongside `DESIGN.md` (rationale/roadmap) and `README.md` (usage). Dev env: the
`gwkokab` conda env; run with `PYTHONPATH=<.../MonteCarloMarginalizeCode/Code>`.

## The goal (reframed)
Produce a **differentiable** lnL surrogate for CIP that is **unbiased** vs the
production CIP+RF posterior, to **PE standards: JS divergence ~ few×10⁻³ bits** on
every 1D marginal (mc, LambdaTilde/tides, q/delta_mc, spins). GP training is **slow →
offline / handoff use**; we are NOT trying to beat RF on speed, only to validate the
exported surrogate is good enough and not biased.

## What is SOLID (don't relitigate)
- **Architecture is right: `quadgp` = quadratic/Fisher core + GP residual** (`quad_gp.py`).
  A pure GP cannot match a razor-sharp quasi-quadratic peak to few-% width; the
  quadratic captures the exact Fisher curvature (sharp eigen-dirs only, via
  `keep_curv_frac`), the GP fits the smooth residual. **This nails mc: width 6.5e-5 vs
  truth 6.9e-5 (JS ~0.035) in every config.** That was the wall; it's cleared.
- **Constrain GP lengthscales** (`svgp.py`/`exact.py`): free hyperopt over-smooths
  (lengthscale runs long); we clip the ARD lengthscale to ~the near-peak width. Keep this.
- **Sampler for a SHARP surrogate = importance sampling, NOT the flow.** A flow can't
  learn a 5e-5 peak in a 3e-3 box (ESS→5). `sample_gaussian_is` with a peak-matched
  Gaussian proposal works (ESS ~hundreds).
- **Morisaki (mu) frame proposal** (`_muframe_proposal`): build the proposal covariance
  in fit coords (well-conditioned; the physical low-level cov is near-singular in mc),
  pull back via the JAX Jacobian `P_low = J^T C_fit^-1 J + diag(1/prior_var)`. Sample
  stays physical — no inverse transform.
- **Benchmark + JS harness done.** 10× CIP RF+AV in `applications/benchmark_condor/`
  (matches the paper's `posterior_samples-6.dat`, LambdaTilde 343±183 ✓). 50k pooled
  samples cached at `/home/oshaughn/jaxcip_benchmark/out/cip_rf_*.xml.gz`.
  `applications/compare.py` computes JS (bits) with a bootstrap stderr.
  **CAVEAT (RO): the RF benchmark is a REFERENCE, not assumed-converged ground truth.**
  Those runs were short — harvested to accumulate likelihoods, not tuned for perfect
  sampling convergence. So part of the residual JS may be the *benchmark*, not our
  surrogate. SAFEST validation (TODO): re-benchmark BOTH the surrogate and a fresh RF
  run from the SAME initial lnL grid (use the 'large' grid from the tabular runs or
  Atul's runs), so the comparison isolates surrogate-vs-RF with no grid/convergence
  confound. Current numbers are good enough for downstream teams to code against.

## Current JS (quadgp + svgp-residual 10.8k, vs benchmark) — sampler comparison
| sampler | mc | delta_mc | s1z | s2z | lambda1 | lambda2 | LambdaTilde |
|---|---|---|---|---|---|---|---|
| mu-frame gaussian-IS | 0.035 | **0.356** | 0.053 | 0.067 | 0.152 | — | 0.116 |
| **mu-frame NUTS (`nuts-mu`)** | **0.023** | **0.056** | **0.008** | **0.008** | **0.016** | **0.015** | **0.028** |

**NUTS-in-mu DONE and it confirmed the diagnosis: it was the sampler.** Every marginal
improved; delta_mc (the worst IS regression) 6.4×; spins ~8e-3 ≈ at the bar. From
catastrophic IS (0.04–0.36) → uniformly small JS. Still NOT uniformly few×10⁻³ (mc 0.023,
LambdaTilde 0.028, delta_mc 0.056) — but the residual now behaves **surrogate/data-limited,
not sampler-limited** (spins, where the surrogate is best, are at the bar; gap is in the mc
width ~16% too broad + the broadest dirs). Single-seed JS is noisy on razor-sharp mc.

## What `nuts-mu` is (DONE; `sample_nuts_muframe` in applications/jax_cip.py)
NUTS in **low-level** coords (output + box are natural; the 5→6 fit→low map isn't
invertible so we can't sample in fit coords), **preconditioned** with the mu-frame
covariance: `_muframe_proposal` builds a well-conditioned cov in the fit frame and pulls
it back to low-level (`P_low = Jᵀ C_fit⁻¹ J + diag(1/prior_var)`). numpyro reparam's the
Uniform box as `theta = lo+(hi-lo)·sigmoid(u)`, so we seed the dense mass matrix with that
cov **mapped into u-space** by the local sigmoid Jacobian (`imm = S⁻¹ gcov S⁻¹`,
`S = (hi-lo)·s·(1-s)` at the peak); init at the peak; adapt_mass_matrix=True re-adapts.
Unit-tested on a 4-orders-of-mag-anisotropic correlated Gaussian (ESS ~4–5k/6k, 0 div,
σ recovered 0.5%). Run via `--sampler nuts-mu --num-chains N`. Demo:
`demo/rift/export_likelihoods/`.

### Next step (highest value): close the last factor (now surrogate/data, not sampler)
1. **More/uncapped data + `--quadgp-residual exact` cross-check** at the largest tractable
   N — does the inducing-point approx cost accuracy at scale? Push mc width + LambdaTilde down.
2. **Reduce the 14 divergences** (raise target_accept; check if they cluster at box edges in
   the weakly-constrained dirs).
3. **Multi-seed JS + bootstrap** for publication-grade error bars (single-seed mc JS is noisy).
4. **Tighten the quadratic-core mc localization** (residual ~16% mc width is the dominant
   remaining bias on the sharpest direction).

## How to run
```bash
cd .../MonteCarloMarginalizeCode/Code
P=/home/oshaughn/.conda/envs/gwkokab/bin/python
# surrogate + sampler -> posterior XML
PYTHONPATH="$PWD" $P -m RIFT.interpolators.jax_gp.applications.jax_cip \
  --fname /home/oshaughn/all.net \
  --parameter delta_mc --parameter-implied mu1 --parameter-implied mu2 \
  --parameter-implied LambdaTilde --parameter-implied DeltaLambdaTilde \
  --parameter-nofit mc --parameter-nofit s1z --parameter-nofit s2z \
  --parameter-nofit lambda1 --parameter-nofit lambda2 \
  --mc-range '[1.196,1.199]' --chi-max 0.05 \
  --cap-points 12000 --jax-fit-method quadgp --quadgp-residual svgp \
  --n-features 800 --n-opt-steps 250 --sampler gaussian \
  --fname-output-samples /tmp/jaxcip_out
# JS vs the cached benchmark (per param)
PYTHONPATH="$PWD" $P -m RIFT.interpolators.jax_gp.applications.compare \
  --a /tmp/jaxcip_out.xml.gz \
  --b '/home/oshaughn/jaxcip_benchmark/out/cip_rf_*.xml.gz' --param mc
```
Fast surrogate-only diagnostic (fit + importance-sample widths, ~3 min): see the
inline scripts in the session, or fit `get_interpolator("quadgp")(...)` and IS in
low-level coords (order: mc, delta_mc, s1z, s2z, lambda1, lambda2 — don't swap mc/delta_mc).

## Key files
- `quad_gp.py` — quadratic core + GP residual (mc-exact). Export DONE: overrides
  `export_state`/`from_state` to embed the nested residual model (its arrays under a
  `_resid_` prefix, its meta under `meta["resid_meta"]`); round-trips cross-process and
  stays `jax.grad`-able after load (see `test_export_roundtrip_quadgp_*`).
- `svgp.py` / `exact.py` — constrained-lengthscale GPs (peak-matched bounds).
- `coordinates.py` — pure-JAX Morisaki/tidal transforms (validated vs lalsimutils).
- `applications/jax_cip.py` — pipeline: tree-ring downselect, fit, samplers
  (`flow|nuts|gaussian|mixture`), `_muframe_proposal`, legacy-CIP-compatible CLI + output.
- `applications/compare.py` — JS metric. `applications/benchmark_condor/` — RF+AV fleet.

## Don'ts (learned the hard way)
- Don't let GP hyperparameters fit freely (over-smooths). Constrain the lengthscale.
- Don't use RFF for IS targets — it rings/overshoots → IS ESS collapse.
- Don't expect a flow to sample a razor-sharp surrogate (use peak-matched IS / NUTS-in-mu).
- Don't trust ESS as the success metric — use JS vs the benchmark. ESS-good can be 12× biased.
- Don't re-derive prior bounds from the data — trust the CLI ranges (the grid extends past
  the prior on purpose). The naive Gauss+flow mixture is a dead end without box-normalization.
