# Audit: `_rvs` consumers that prefer a cached column over the canonical components

Motivated by PR #51, where `_pool_replica_rvs` rewrote the canonical weight components while two
exporters read a cached `log_weights` column instead. This sweep asked the same question of every
writer and reader in the tree.

## The central finding: `log_weights` does not mean one thing

| sampler | what it stores in `_rvs['log_weights']` |
|---|---|
| `mcsamplerPortfolio` (`:1531/1536`) | `lnL + ln p - ln p_s` -- the TRUE importance weight |
| `mcsamplerGPU` (`:766/771`) | `tempering_exp*lnL + ln p - ln p_s` -- the ADAPTATION weight |

`tempering_exp` is the adapt-weight-exponent. It is **not 1 in production**: `helper_LDG_Events`
sets it from the SNR (`helper_LDG_Events.py:1472/1477`), and `--no-adapt` drives it to **0**, which
removes the likelihood from the column entirely. So a consumer preferring the cache reweights its
output by `L^(e-1)` whenever the GPU/AC sampler is in use -- and by `1/L` under `--no-adapt`.

## Verdicts

| site | reads | verdict |
|---|---|---|
| `integrate_likelihood_extrinsic_batchmode` extrinsic-proposal fit (~:3022) | components | **already correct** -- derives deliberately, with a comment naming the tempering hazard |
| same file, `.dgrid` exporter | cached `log_weights` first | **FIXED** -- now derives |
| same file, calibration-posterior exporter | cached `log_weights` first | **FIXED** -- now derives |
| same file, `_lnZ_of_rvs`, `_kish_neff_of_rvs` | own inline copies of the derivation | **FIXED** -- collapsed onto the shared function |
| `mcsamplerGPU:847` (`weights_alt`) | cached `log_weights` | **correct by design** -- this is the adaptation path, and the column IS the adaptation weight |
| `mcsamplerGPU:1536`, `mcsampler.py:1115`, `mcsamplerEnsemble.py:923` | `_rvs['weights']` | read the linear cache; see the separate defect below |
| `mcsamplerGPU:1559`, `mcsampler.py:1138` | recompute from components | already worked around a divergence: *"rvs['weights'] is **sorted** (side effect?), breaking test. Recalculated weights are not."* -- a third instance of the same theme, worked around rather than fixed. Not chased here. |

## Second defect found: a one-of-two-paths bug

`mcsamplerGPU.py:1194` appended the new weights onto **`joint_s_prior`** instead of `weights`,
corrupting that column from the second chunk onward:

```python
self._rvs["weights"] = xpy_here.hstack( (self._rvs["joint_s_prior"], fval*joint_p_prior/joint_p_s) )
```

`mcsampler.py:571` carries the identical block **with the fix and an explicit `BUGFIX` comment**.
The GPU copy never received it. Reachable: `mcsamplerGPU:1536` reads `_rvs['weights']`. **Fixed**,
with a comment pointing at its twin so the pair stays visible.

## What changed

`ln_weights_from_rvs()` is now the single definition of "the importance weight of an `_rvs`
record": log components first, then the linear (`mcsamplerEnsemble`) form with out-of-support rows
sent to `-inf`, and an explicit exception when neither is present -- because a loud failure beats a
plausible wrong number in a science output. Five call sites share it. The ambiguous cache is not
read on any weight path.

## Not done

The `weights`-is-sorted side effect noted at `mcsampler.py:1138` is a real divergence between a
cached column and its components, currently worked around by two callers. Worth its own pass.
