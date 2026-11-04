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
| beta = 0.7735 | see §4 | | 4202.8 | 4800 | 4320 |

Predicted vs measured: 157 vs 185 (ratio 1.17) and 4320 vs 4203 (0.97). The law
is validated within ~20% on the real sampler by a route that never touches the
offline reference.

**Porting the historical exponent literally costs 17x in exported rows
(4800 -> 278) and 6.6x in evidence precision**, and the driver itself prints
*"NOT a usable posterior sample however it is drawn."*

### 3c. Is the cost SNR-set or dimension-set?  (reference-free)

Same source, SNR set by injected distance (network SNR is exactly propto 1/d),
beta held fixed, reading the driver's own reported export ESS. No reference cloud
is involved, so this cannot inherit a reference's convergence problems.

| injected d | lnZ | implied SNR | beta=0.5 ESS/N | beta=0.1 ESS/N |
|---|---|---|---|---|
| 1200 Mpc | 113.1 | ~15.0 | 0.620 | 0.0472 |
| 600 Mpc | 533.1 | ~32.7 | 0.517 | 0.0371 |
| 300 Mpc | 2230.7 | ~66.8 | 0.560 | 0.00823 |
| 150 Mpc | — | ~134 | 0.126 | 0.00763 |
| **law** | | | **0.5625** | **0.0361** |

**At beta = 0.5 the cost is flat to +-10% from SNR 15 to 67** — a factor 20 in
lnLmax, over which the historical rule would have demanded beta fall by the same
factor 20. That is the discriminator, and it settles it: the reweight cost is not
SNR-set.

**Two honest caveats, both visible in that table.**

1. **The law is optimistic at small beta, and increasingly so at high SNR.** At
   beta=0.1 the measured cost falls from 0.047 (SNR 15) to 0.0082 (SNR 67) while
   the law says 0.036 throughout. The Gaussian-peak approximation degrades where
   the target is both very sharp and very heavily tempered. This matters for the
   guard: near its threshold the guard can *under*-refuse, because it trusts a
   law that is too kind in exactly that corner. It never over-refuses.
2. **beta = 0.5 breaks down by SNR ~134** (0.126 against 0.5625). At the highest
   rung the flow itself is struggling, not just the reweight. Static tempering is
   therefore NOT the tool for the 3G regime, which is the regime it is usually
   proposed for.

An earlier offline sweep over the same injections (`beta_ess_vs_snr.py`) is
**not** quoted: its reference collapsed above SNR 20 (ESS(g=1) = 4.1 at SNR 40 —
the Hessian at the truth is near-flat in inclination and the eigenvalue floor
mis-scales the proposal). Those rows are not evidence.

### 3d. Does beta < 1 buy accuracy?  (two seeds)

Scored against the independent defensive-IS reference, floor at the arm's own
row count.

| arm | rows | JS psi s0/s1 | JS incl s0/s1 | sd psi s0/s1 |
|---|---|---|---|---|
| beta = 1.0 | 4800 | 0.0735 / 0.0358 | 0.0617 / 0.0325 | 0.940 / 0.990 |
| beta = 0.7735 | 4800 | 0.0407 / 0.0289 | 0.0445 / 0.0296 | 0.983 / 1.037 |
| beta = 0.0951 | 278 / 203 | 0.1354 / 0.1438 | 0.1310 / 0.0954 | 1.081 / 0.872 |
| `--adapt-adapt` | 4800 | 0.0462 / **0.5690** | 0.0774 / **0.3825** | 1.043 / **0.031** |

**Not resolved: whether beta=0.7735 beats beta=1.** Lower psi and incl JS in both
seeds, but beta=1's own psi JS varies 2x across seeds — a spread comparable to
the gap. Two seeds cannot settle it, and **no part of this change rests on it**.
What is solid is that the auto exponent costs nothing measurable: same 4800 rows,
export ESS 4203 / 4266.

**Resolved: `--adapt-adapt` collapsed on one seed of two.** Seed 1 returned psi
spanning only [1.599, 1.769] rad of the full [0, pi] — 3% of the reference width
— against [0.026, 3.131] for seed 0 and both beta=1 arms. Verified in the raw
export, not only in the score. Seed 0 was also 21% narrow in inclination. At
SNR 23.8, not an extreme case.

## 4. What was built, and what was deliberately NOT

**Built.**

- `samplers.export_ess_fraction(beta, n_dim)` / `beta_for_export_ess(target, n_dim)`
  — the law and its inverse. The signature carries no SNR argument, on purpose;
  a test pins that.
- `--auto-adapt-weight-exponent` + `--target-export-ess-frac` (default 0.9):
  picks the smallest beta meeting the export budget, keyed on the **sampled
  dimension** (at a 0.9 target: d=3 -> 0.786, d=4 -> 0.807, d=5 -> 0.824).
  These solve the CALIBRATED lower bound, not the bare law -- see §4a.
- A guard: any beta whose predicted export ESS falls below the driver's own
  usability floor of 200 is **refused** (exit 1, no file written), with a message
  naming the historical rule as the trap. `--allow-degenerate-tempering` overrides.

**Not built: a port of the SNR rule to the static exponent.** It keys on a
variable the cost does not depend on, and at this event it selects a value the
driver already declares unusable.

**Not built: `--adapt-adapt` on by default.** It is a different mechanism — an
annealing *schedule* that ladders `inv_T` up and always terminates at
`inv_T = 1` (the loop breaks on `inv_T >= 1`, and `post_weight` is then uniform),
so it costs nothing at export. But it is **not** a free robustness win: it
collapsed on one seed of two at SNR 23.8 (§3d), and cost **>24 min against 156 s**
for a static run. On this evidence it must not be a default. PR #183 needs this:
the anneal cannot be assumed safe.

### The closer structural analogue, for whoever picks this up

Old-RIFT's beta broadens a *proposal* while the estimator stays exact. The JAX
constructs that do that are **not** `inv_T` — they are the hardcoded defensive
inflations: `cov = cov * 2.0` (`samplers.py:816`, `:1462`, the moment-matched IS
evidence proposal) and `fisher_is_inflate=1.3` (`:1134`, the high-SNR Fisher-IS
fallback). Those are where "intelligence" could go without paying any export-ESS
cost. Not touched here — out of scope, and unmeasured.

### 4a. The law is optimistic — and the calibration is NOT a bound either

Raised in review of this change, and correct.  `export_ess_fraction` is the
Gaussian-peak law, which the §3a sweep shows is optimistic by up to 21%
(measured/law 0.79 at beta=0.05, rising to 1.00 at beta=1).  Inverting it to pick
beta, and guarding on it, both hand back something already known to fall short of
what was asked for: at d=4 a 0.9 target returned **beta=0.77347, whose measured
lower bound is 0.866**.

`export_ess_estimate(beta, n_dim) = cal(beta) * law(beta, n_dim)` is what the
chooser solves and what the export check reports.  `cal` is a piecewise-linear
envelope over the measured ratios, every knot at or below every measured point:

| beta | 0.05 | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|
| cal | 0.79 | 0.79 | 0.83 | 0.91 | 0.97 | 1.00 |

A single flat factor would have been the wrong shape twice over: the shortfall is
strongly beta-dependent, and a flat 0.79 would make any target above 0.79
unreachable, since it never rises to 1.  The inverse has no closed form once the
envelope is included and is solved by bisection (both factors are monotone in
beta, so the product is).

**IT IS NOT A LOWER BOUND, AND A SECOND REVIEW ROUND CAUGHT IT BEING USED AS
ONE.**  `cal` is fitted at SNR ~= 23.8, and the shortfall grows with SNR.  §3c's
own ladder already said so, and it supplies the counterexample:

| | beta=0.1, d=4 |
|---|---|
| estimate (`cal x law`) | 0.0285 |
| **measured, SNR ~= 67** | **0.00823** |

a factor 3.5 the wrong way.  On a 10 000-row cloud that is an estimated ESS of
285 against a measured ~82: comfortably over a 200-row floor while being well
under it.  The first version of this section inverted the estimate to pick beta
AND refused runs on it, i.e. built a hard guarantee out of a single-SNR fit while
the caveat disproving it sat two sections above.

A real bound needs a calibration in **(beta, SNR)**, and the driver has no
trustworthy SNR where the choice is made — `guess_snr` is an explicit
guesstimate (10.32 against a true network 23.78 on the study event).  So:

* the export check **warns, it does not refuse** — a refusal is a guarantee;
* `--auto-adapt-weight-exponent` is documented **EXPERIMENTAL and not
  sufficiently validated**, and no paper result uses it;
* **measured at d=4 only** — the ratio at other dimensions is assumed, not
  measured, and is the first thing to check if a d=3 or d=5 budget comes up short.

## 5. Limitations — axes swept, and axes presumed load-bearing

**Swept:** beta over [0.05, 1]; SNR ~15 to ~134 at fixed beta; two seeds on the
accuracy arms; sampled dimension via the closed form (3/4/5, verified
analytically, only d=4 measured); two independent estimators (offline defensive
IS, and the driver's own reported ESS).

**NOT swept, presumed load-bearing:**

- **Accuracy above SNR 24.** The accuracy arms (§3d) are one event at SNR 23.8.
  The ESS ladder reaches SNR ~134, but only measures export ESS there, not
  whether the resulting posterior is right.
- **The guard's threshold in the corner where the law is optimistic** (§3c
  caveat 1): near ESS ~200 at small beta and high SNR the guard trusts a law that
  over-predicts. It errs toward passing, not refusing. Not characterised.
- **The calibration is single-SNR** (§4a).  Fitted at SNR ~= 23.8; the ladder
  shows it optimistic by 3.5x at SNR ~= 67 and beta=0.1.  Nothing may refuse a
  run on it, and the chooser's target is met only on that calibration.
- **The calibration at dimensions other than 4** (§4a): assumed from the d=4
  sweep, not measured.
- **Only two seeds.** Enough to show the `--adapt-adapt` collapse (it is a 30x
  effect) and to leave the beta=0.7735-vs-1 question open. Not enough for either
  to be a width claim.
- **Two paths where the exponent is inert**, both found by reading rather than
  running. `--smc-puffball` routes to `smc_puffball_sample`, which swallows
  `temper` in `**_ignore` and exports uniform weights: the guard is skipped there
  and the no-op reported instead. `--fisher-is-samples` is conditional — a
  *successful* Fisher-IS pass replaces the cloud with an already-fair-drawn
  uniform-weight set so the reweight cost never materialises, but it falls back
  to the tempered draws when it fails. The guard still refuses there
  (fail-closed), with `--allow-degenerate-tempering` as the documented escape.
  Neither path was measured.
- **Non-Gaussian / strongly multimodal targets.** The law is a Gaussian-peak
  result; the measured 0.79 shortfall at small beta is that approximation
  failing. A target with well-separated equal-mass modes may do worse.
- **Modes other than `flowmc-phimarg`.** `flowmc` (5-D), `flowmc-phipsimarg`
  (3-D) and `flowmc-dpsimarg` (4-D) take the same code path and the same `dim`,
  but were not run.

## 5b. A defect class found by READING, not by running

The chooser originally wrote its answer back to `opts`. `opts` is per-RUN;
`analyze_one` is per-EVENT. On event 1 of a batch the chooser read its own
event-0 output as a user-supplied exponent and aborted with `SystemExit` — and
`ILE_extr.sub` runs batches, so every real multi-event `--auto` run would have
died. **No single-event test can see this**, and every measurement in §3 is
single-event. The repo's own `RIFT/integrators/REVIEW_CHECKLIST.md` names the
shape ("Is per-point state cleared on ENTRY"). The chooser is now pure and
returns the exponent; `analyze_one` keeps it in a local.

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
