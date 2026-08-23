# Choosing `--adapt-weight-exponent` on the JAX extrinsic path

**Status: a record, not a specification.** It is expected to be superseded. Where
it disagrees with the code, the code wins; where it disagrees with a later
measurement, the measurement wins. Dated 2026-08-23, RIFT branch
`rift_O4d_jax_adapt_weight_chooser` (stacked on `claude/jax-fairdraw-extrinsic`,
PR #180).

The live decision is `samplers.export_ess_fraction` /
`samplers.beta_for_export_ess` plus the driver's `resolve_tempering_exponent`,
pinned by `test/jax/test_jax_tempering_chooser.py`.

## The question

RO'S, 2026-08-23: *"of course it's the default, we use intelligent choices to set
that weight-exponent with the helper historically. We just don't have that
intelligence yet for the JAX protocol (only some of the non-JAX samplers use it)
.. but we should have something."*

## 1. The historical rule, and what it actually controls

`bin/helper_LDG_Events.py:1472-1481`:

```python
prefactor = 0.1
snr_fac = max(1.0, event_dict["SNR"]/15.)          # line 708
if snr_fac > 1.5:                                   # i.e. SNR > 22.5
    beta = prefactor/np.power(snr_fac/1.5, 2)
else:
    beta = prefactor
```

Equivalently, with `lnLmax = SNR^2/2`:

```
beta = min(0.1, 25.31 / lnLmax)
```

— the rule holds **`beta * lnLmax` constant at ~25.3 nats** above SNR 22.5 and
caps `beta` at 0.1 below. It is a *fixed tempered dynamic range* rule.
`util_RIFT_pseudo_pipe.py` does not set the exponent itself; the helper is the
only place the choice is made.

### Who actually consumes it

`bin/integrate_likelihood_extrinsic_batchmode:1806` passes
`tempering_exp = opts.adapt_weight_exponent` (0.0 under `--no-adapt`).

| sampler | consumes it? | how |
|---|---|---|
| `mcsampler` | yes | 1-D adapted histogram weights |
| `mcsamplerGPU` | yes | `log_weights = tempering_exp*lnL + ln p - ln p_s` |
| `mcsamplerEnsemble` | yes | `update_sampling_prior`, `ln_weights *= tempering_exp` |
| `mcsamplerNFlow` | yes | `update_sampling_prior` |
| `mcsamplerPortfolio` | yes | via the member samplers' `update_sampling_prior` |
| `MonteCarloEnsemble` | yes | `_solve_tempering_exp` (adaptive beta) |
| **`mcsamplerAdaptiveVolume`** | **NO** | reads it (line 1575) and never acts on it |

**AV is not a consumer.** It has no `update_sampling_prior` (only
`update_sampling_prior_selfish`, which ignores `tempering_exp`), and the single
line that would apply it is commented out at line 1735:
`# log_weights = tempering_exp*lnL + log_joint_p_prior`. Its *only* effect in AV
is the side effect at line 1595, forcing `save_intg = True`. Anything that says
AV honours the exponent is wrong.

### The structural point

In every real consumer the exponent shapes **only the adaptive sampling prior**.
The estimator stays `log_integrand = lnL + ln p - ln p_s`
(`mcsamplerGPU.py:753`), so the integral is **unbiased at any beta** and the
exported sample count is untouched. `beta` there is a pure proposal-breadth knob:
free to use, and smaller = more defensive.

## 2. Why the rule does not transfer

On the JAX path (`samplers.flowmc_sample_phimarg`, driver line ~1178)
`beta = inv_T` is the exponent of the target the MCMC **samples**:

```python
inv_T = 1.0/temper
def logpdf(theta, data):  return inv_T*like._scalar(theta) + log_prior(theta)
```

The draws **are** the deliverable, so the export must be reweighted by
`post_weight ∝ L^(1-beta)` (PR #180) — and that reweight costs effective samples,
which the non-JAX path never pays.

For a locally Gaussian peak `ln Z_g = g*lnLmax - (d/2) ln g + const`, so

> **ESS/N = Z_1² / (Z_beta · Z_{2-beta}) = [beta (2 - beta)]^(d/2)**

with `d` the **sampled dimension**. **No `lnLmax`. No SNR.** The two pictures make
opposite predictions, and the JAX one is the one that holds here.

*(Recorded because it was the first guess and it is a plausible one: the
lognormal-weight estimate `exp(-(1-beta)² Var_beta[lnL])` is **wrong**, by ~70
orders of magnitude at beta=0.1. The reweight weights are heavy-tailed, not
lognormal. `Var_beta[lnL] = d/(2 beta²)` is right — it is the ESS step that fails.)*

## 3. Measurements

### 3a. Exact offline sweep — real BNS likelihood, SNR 23.8, 4-D

`~/rift_bns_jax_run_scratch/beta_ess_offline.py`, event 0 of
`~/rift_bns_jax_run/rundir/final_post.xml.gz`. Defensive importance sample,
300k draws, prior-pilot centres over a 5-scale bandwidth ladder + 12% prior
floor. **Reference ESS(g=1) = 5235**, and every `g` in [0,2] the estimator needs
is resolved (worst-case IS ESS 508), so the whole sweep is converged.

| beta | measured ESS/N | law `[b(2-b)]²` | measured/law | export rows (of 4800) |
|---|---|---|---|---|
| 0.05 | 7.55e-3 | 9.51e-3 | 0.79 | 55 |
| **0.0951** (historical) | **2.76e-2** | 3.28e-2 | 0.84 | **199** |
| 0.20 | 1.04e-1 | 1.30e-1 | 0.80 | 751 |
| 0.50 | 5.00e-1 | 5.63e-1 | 0.89 | 3600 |
| 0.70 | 7.92e-1 | 8.28e-1 | 0.96 | 4800 |
| 0.90 | 9.75e-1 | 9.80e-1 | 0.995 | 4800 |
| 1.00 | 1.000 | 1.000 | 1.00 | 4800 |

The law holds to 0.79–1.00 across the range; it is therefore slightly
**optimistic** and is safe as a design rule, not as a promise.

`Var_beta[lnL]` independently tracks `d/(2 beta²)`: 213 vs 200 (beta=0.1),
8.08 vs 8.00 (0.5), 2.50 vs 2.00 (1.0).

### 3b. End-to-end on the real sampler — no reference involved

Full `integrate_likelihood_extrinsic_jax` runs, same event, 156 s each. The
driver's own `fairdraw: N weighted samples (ESS=...)` line is a direct
measurement, so this shares no machinery with 3a.

| arm | lnZ | sigma_lnL | export ESS | rows written | law predicted ESS |
|---|---|---|---|---|---|
| beta = 1.0 (default) | 253.686 | 0.0118 | n/a (uniform) | 4800 | 4800 |
| **beta = 0.0951 (historical)** | 253.908 | **0.0779** | **184.9** | **278** | 157 |
| beta = 0.7735 (auto, 90% target) | see §4 | | 4202.8 | 4800 | 4320 |

Predicted vs measured: 157 vs 185 (ratio 1.17) and 4320 vs 4203 (0.97). The law
is validated within ~20% on the real sampler by a route that never touches the
offline reference.

**Porting the historical exponent literally costs 17x in exported rows
(4800 -> 278) and 6.6x in evidence precision**, and the driver itself prints
*"NOT a usable posterior sample however it is drawn."*

### 3c. Is the cost SNR-set or dimension-set?

Structurally the law has no `lnLmax` term, and `Var_beta[lnL]` matches
`d/(2 beta²)` at the measured SNR. An offline sweep over synthetic injections at
SNR 20/40/80/160 (`beta_ess_vs_snr.py`) reproduced the same ESS(beta) curve, but
its reference collapsed above SNR 20 (ESS(g=1) = 4.1 at SNR 40 — the Hessian at
the truth is near-flat in inclination and the eigenvalue floor mis-scales the
proposal). **Those rows are not evidence and are not quoted here.** The SNR axis
is carried instead by the driver-reported ESS at fixed beta across injected
distances (§3d).

## 4. What was built, and what was deliberately NOT

**Built.**

- `samplers.export_ess_fraction(beta, n_dim)` / `beta_for_export_ess(target, n_dim)`
  — the law and its inverse. The signature carries no SNR argument, on purpose;
  a test pins that.
- `--auto-adapt-weight-exponent` + `--target-export-ess-frac` (default 0.9):
  picks the smallest beta meeting the export budget, keyed on the **sampled
  dimension** (d=3 -> 0.740, d=4 -> 0.773, d=5 -> 0.797).
- A guard: any beta whose predicted export ESS falls below the driver's own
  usability floor of 200 is **refused** (exit 1, no file written), with a message
  naming the historical rule as the trap. `--allow-degenerate-tempering` overrides.

**Not built: a port of the SNR rule to the static exponent.** It keys on a
variable the cost does not depend on, and at this event it selects a value the
driver already declares unusable.

**Not built: `--adapt-adapt` on by default.** It is a different mechanism — an
annealing *schedule* that ladders `inv_T` up and always terminates at
`inv_T = 1` (the loop breaks on `inv_T >= 1`, and `post_weight` is then uniform).
So it delivers the historical rule's *benefit* — broad exploration, no collapse
onto a sub-resolution MAP — at **zero** reweight cost. That makes it the right
answer to the high-SNR problem and the wrong thing to call an "exponent chooser".
Whether it should be the default is a separate question that needs the high-SNR
bake-off in PR #183, not this change.

### The closer structural analogue, for whoever picks this up

Old-RIFT's beta broadens a *proposal* while the estimator stays exact. The JAX
constructs that do that are **not** `inv_T` — they are the hardcoded defensive
inflations: `cov = cov * 2.0` (`samplers.py:816`, `:1462`, the moment-matched IS
evidence proposal) and `fisher_is_inflate=1.3` (`:1134`, the high-SNR Fisher-IS
fallback). Those are where "intelligence" could go without paying any export-ESS
cost. Not touched here — out of scope, and unmeasured.

## 5. Limitations — axes swept, and axes presumed load-bearing

**Swept:** beta over [0.05, 1]; sampled dimension via the closed form (3/4/5,
verified analytically, only d=4 measured); two independent estimators (offline
defensive IS, and the driver's own reported ESS).

**NOT swept, presumed load-bearing:**

- **SNR above ~24 end-to-end.** §3a/3b are one event at SNR 23.8. The law's
  SNR-independence is structural + supported by `Var_beta`, not yet demonstrated
  end-to-end at 3G SNRs, which is exactly where tempering is claimed to matter.
- **Posterior accuracy.** Everything above measures export *ESS*, not whether the
  beta<1 posterior is *right*. The arm-vs-reference scoring is in
  `RESULTS_jax_tempering_2026-08-23.md` in the paper repo.
- **Non-Gaussian / strongly multimodal targets.** The law is a Gaussian-peak
  result; the measured 0.79 shortfall at small beta is that approximation
  failing. A target with well-separated equal-mass modes may do worse.
- **Modes other than `flowmc-phimarg`.** `flowmc` (5-D), `flowmc-phipsimarg`
  (3-D) and `flowmc-dpsimarg` (4-D) take the same code path and the same `dim`,
  but were not run.
- **Seeds.** §3b arms are seed 0; the two-seed matrix is in the results note.

## 6. Reproduce

```
# offline exact sweep (needs the BNS run products, read-only)
python beta_ess_offline.py --code <worktree>/MonteCarloMarginalizeCode/Code \
    --event 0 --n-pilot 100000 --n-ref 300000 --tag REF0
# one end-to-end arm
./run_arm.sh ARM_b095_s0 0 --adapt-weight-exponent 0.09508
# score an arm against the independent reference
python score_arm.py --ref REF0_ev0_ref.npz --files ARM_b095_s0_0_samples.dat
```

All three live in `~/rift_bns_jax_run_scratch/` on CIT (NFS home — **not** on
condor execute nodes). Run on `ldas-pcdev11`/`13` with `OMP_NUM_THREADS=1` and
`taskset`; JAX sizes its XLA pool from the visible CPU count.
