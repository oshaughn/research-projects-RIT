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

## Current JS (quadgp + svgp-residual 10.8k + mu-frame gaussian-IS, vs benchmark)
| mc | s1z | s2z | LambdaTilde | lambda1 | delta_mc |
|---|---|---|---|---|---|
| 0.035 | 0.053 | 0.067 | 0.116 | 0.152 | **0.356** |

From catastrophic (~0.5–0.7) → ~0.04–0.15, with mc essentially exact. **NOT at the
few×10⁻³ bar.** delta_mc regressed (too narrow).

## THE diagnosed bottleneck (start here)
**It's the sampler, not the surrogate.** ESS ~170 → the importance sampler is
*proposal-limited*: it under-explores the weakly-constrained directions (delta_mc, λ),
which then come out too narrow. More data fixed the surrogate's mc bias but cannot fix
an under-exploring sampler — that's why refinements now trade one marginal against
another.

### Next step (highest value): NUTS in the (now well-conditioned) mu frame
Early NUTS failed because the raw posterior was a razor-thin, ill-conditioned ridge.
But the mu-frame construction makes the geometry **well-conditioned and axis-aligned**,
so NUTS should finally mix — and unlike IS it explores the wings by construction. Plan:
1. Run NUTS on `lnL_low` but with a **mass matrix = the mu-frame `gcov`** from
   `_muframe_proposal` (or sample in whitened-by-`gcov` coordinates). numpyro NUTS
   supports `dense_mass=True`; seed/precondition it with `gcov`.
2. Keep the prior box (sigmoid reparam or hard prior) and the quadgp surrogate.
3. Re-measure JS on **all four** marginals vs the benchmark. Target few×10⁻³.
Fallback if NUTS still struggles: **iterate the IS proposal** — fit a Gaussian (or flow)
to the reweighted draws, repeat until the proposal == posterior (ESS → 1).

### Then: exact-residual comparison
Run the same with `--quadgp-residual exact` at ~5–8k (slow, offline) and compare JS to
the svgp-residual run — does the inducing-point approximation cost accuracy at scale?

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
