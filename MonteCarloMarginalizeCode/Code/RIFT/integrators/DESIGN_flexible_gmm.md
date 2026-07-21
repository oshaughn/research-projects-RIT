# Flexible (data-driven) GMM component allocation for the extrinsic integrator

Status: prototype, benchmarked on S250114ax (SNR~82, real H1L1). See the
measured results and the honest tradeoffs at the bottom -- the headline is that
**this fixed the GMM's hard-coding and two warm-start bugs, but pure GMM
importance sampling is still not the right tool for this particular event**; the
flexible GMM belongs in the portfolio (with AV) or on milder problems.

## The problem being fixed

`bin/integrate_likelihood_extrinsic_batchmode` built the GMM (`mcsamplerEnsemble`)
proposal with a **hard-coded** per-group component layout:

```
gmm_dict  = {(ra,dec):None, (distance,inclination):None, (psi,phi_orb):wide-frozen}
comp_dict = {(ra,dec):4,     (distance,inclination):2,     (psi,phi_orb):4}
```

That pairing (large sky ring = 4 components; a single distance-inclination lobe = 2;
a wide frozen phase-polarization component) targets **quadrupole-dominated,
poorly-localized** binaries. It does not adapt to the actual posterior, and a
product of per-group GMMs cannot represent cross-group correlations. `--internal-
gmm-correlate-all` switches to a single full-dim group but still with a **fixed**
component count. Choosing that count by hand is exactly the "horrible hacky
hard-coding" this work removes.

## What the extrinsic posterior actually looks like (S250114ax)

Not a 10^-11 needle -- a **broad, correlated/degenerate 6-D blob** (see
`BREADCRUMB_av_neff_reproduction.md`). Sky is tight; **distance and inclination
are broad and coupled** (the classic distance-inclination degeneracy -- a curved
arc); phase and polarization are similarly degenerate. The prior is huge (all-
sky/all-distance/all-inclination), so the peak is a tiny *fraction* of the prior
even though it is not narrow in absolute terms.

## Design

Three orthogonal knobs, prototyped in order of leverage:

### 1. Scalable component matching (enabling; committed separately)
`gmm._match_components` enumerated all **k! permutations** to align old->new
mixture components in `update()`. Fine for k<=6, impossible for k>=10 (12!~5e8,
16!~2e13). The objective is additive over matched pairs, so it is a linear
assignment problem: `scipy.optimize.linear_sum_assignment` gives the **same
optimum in O(k^3)**. Verified identical to the permutation optimum for k=2..6;
k=16 now matches in ~6 ms. Without this, "wrap the arc with many small Gaussians"
is not even runnable.

### 2. Warm-start survival (bug fix; high leverage)
`bootstrap_from_samples` fits proposal models and stores them on
`self.integrator`, but `MCSampler.integrate()` **rebuilt a fresh integrator** from
the passed `gmm_dict` (values `None`) and never looked at `self.integrator` -- so
the warm fit was **silently discarded** and a "warm" run started cold (measured:
first chunk n_eff=1.0 despite a `[GMM warm-start] fitted ...` log line).
`integrate()` now transfers any fitted model whose dim-group key matches into the
new integrator (key mismatch -> cold, never biases).

### 3. Data-driven component count (the flexible allocation)
`fit_gmm_adaptive(samples, bounds, log_weights, k_max, ...)`:
  * fit k over a ladder `[1,2,3,4,6,8,...] <= k_max`,
  * score each by a **weighted BIC**  `-2*wLL + p*ln(N_eff)`  (p = free params of
    a k-component d-dim mixture, N_eff = Kish effective sample size of the fit
    weights),
  * keep the best k, then **prune** components whose weight < floor.

BIC allocates more components only where the importance-weighted cloud is
genuinely non-Gaussian, and stays at k=1 for a single blob; the ln(N_eff) penalty
makes it self-limiting, so it avoids the instability of a fixed over-allocated k.

**Init-only, then stable merge.** A group with `gmm_adaptive[group]=k_max` picks
its k by BIC at *initialization*, then hands off to the existing, proven-stable
merge adaptation (`model.update()`). A per-chunk BIC refit-fresh was tried and
**rejected**: it makes the proposal wander (n_eff peaks then collapses) because
each fit sees a different elite cloud; the incremental merge smooths that out.

**Defensive tail coverage (optional, `add_defensive_component`).** A broad box-
covering component with weight `defensive_frac` bounds the importance weights so a
tight fit to the elite cloud cannot blow up n_eff on the broad/degenerate
directions (the AV sampler gets the same guarantee from its cover-fraction floor).
Kept as an option; see the caveat below.

### Safety: opt-in, and a floor at the stress-tested layout
The flexible allocation is a **refinement layer, never a replacement**:
  * It is **opt-in** (`--internal-gmm-adaptive-components`, default OFF).  With
    the flag off the driver builds the exact hard-coded `gmm_dict`/`comp_dict`/
    `gmm_adapt` as before -- byte-identical behavior for the primary ILE use case.
  * When on, BIC chooses k in **[k_min, k_max]** with `k_min` = the group's
    stress-tested hard-coded count, and pruning never drops below `k_min`.  So
    adaptive can only ADD components where the data earns them; it can never
    allocate fewer than the validated layout.  This protects broad multi-modal
    posteriors (e.g. a multi-modal sky keeps its default components even if the
    *initial* elite cloud -- fit before the proposal has explored every mode --
    looks single-peaked; the spare capacity lets the merge adaptation grow into
    the other modes as they appear).

### Portfolio
`--internal-gmm-adaptive-components` also works with `--sampler-method portfolio`:
the driver injects `gmm_adaptive` as a **scalar cap** into the shared setup
kwargs, which the portfolio forwards to every member; the GMM member honors it in
`update_sampling_prior` (the path the portfolio drives), floored at its own
`n_comp`.  Non-GMM members ignore it.  Default OFF -> the portfolio's
stress-tested GMM member config is unchanged.  (Note: a working portfolio needs
`draw_simplified`-capable members; the `AV,GMM` combo currently fails in the
portfolio draw path for a PRE-EXISTING, unrelated reason -- `AV` has no
`draw_simplified`.)  This is the recommended way to use the flexible GMM: AV
carries convergence, and the GMM member contributes a hands-free correlated
proposal instead of a hand-tuned component layout.

### Driver flags
```
--internal-gmm-adaptive-components         # enable BIC allocation (opt-in; OFF by default)
--internal-gmm-max-components N   (def 8)  # per-group cap (floor = the hard-coded count)
--internal-gmm-defensive-frac F   (def 0)  # opt-in defensive tail component
--internal-gmm-inflate X          (def 1)  # covariance (std) inflation
```

## Measured results

### Synthetic (data-free, moderate SNR -- where importance sampling is viable)
6-D target: a curved **banana ridge** in 2 dims (distance-inclination analogue),
a strongly-correlated Gaussian pair (phase-pol analogue), a tight blob (sky).
`test/integrators/synth`-style harness, n_eff vs N (cumulative samples):

| proposal                     | n_eff>=100 at N | final n_eff | lnI    |
|------------------------------|-----------------|-------------|--------|
| correlate-all, fixed k=1     | 76 k            | ~50         | 3.06   |
| correlate-all, fixed k=2     | **28 k**        | **312**     | 3.05   |
| correlate-all, fixed k=4     | 752 k           | ~96         | 3.06   |
| **flexible (BIC, k<=8)**     | 220 k           | 135         | 3.03   |

Flexible is **robust and unbiased**: it beats k=1, avoids the k=4 over-allocation
collapse, and lands the same integral -- without any hand-tuning. It does not beat
the *oracle-best* fixed k=2, which is the expected price of a hands-free allocator.
Note fixed k=4 being far worse than k=2 is exactly the "a wrong hard-coded count
hurts" failure the flexible allocation exists to avoid.

### S250114ax (real, SNR~82) -- n_eff vs N, `--n-max 4e6 --n-eff 100`
Same worker, byte-identical data, only the proposal/config varies:

| sampler / config                                   | warm | peak n_eff (<=4 M) |
|----------------------------------------------------|------|--------------------|
| **AV (VARAHA), warm (reference)**                  | yes  | **~89 (->100 @~3.4 M)** |
| hard-coded pairing, cold                           | no   | 1.29 |
| correlate-all k=2, warm (warm-start now survives)  | yes  | 5.7 |
| correlate-all k=8 / k=16, warm + --adapt-adapt     | yes  | ~1.0 |
| **flexible (BIC k<=8), warm + --adapt-adapt**      | yes  | 1.03 |
| flexible (BIC k<=8), cold + --adapt-adapt          | no   | 1.00 |
| (independent) prior pure-GMM logs gmm*/cold*       | --   | 1.3 - 3.2 |

**Pure GMM importance sampling stalls at n_eff ~ 1-7 on this event, regardless of
component allocation (fixed k=2..16, BIC-adaptive), warm start, defensive coverage
(0.05), or covariance inflation (2x-5x).** This is corroborated by five pre-
existing pure-GMM logs in the pipeline (peak 1.3-3.2). The runs that reached ~100
in that directory are **AV / portfolio** runs, not pure GMM.

## Why GMM stalls here (and AV does not)

At SNR~82 the likelihood spans exp(~1210). The honest per-chunk effective sample
size `ESS(lnL + ln p - ln q)` is **exactly 1** every chunk: within any chunk of
proposal draws, the single sample nearest the sharp peak carries ~100+ nats more
log-weight than the rest, so it dominates. For n_eff to exceed 1, the proposal
would have to match `exp(lnL)*prior` to within O(1) *across* the peak -- i.e. be
almost the posterior already. A moving Gaussian-mixture importance proposal does
not get there from a broad start:
  * the beta-tempered refit drives the exponent to ~0.005 (nearly flat) to keep
    its own ESS up, so it barely uses the likelihood and never concentrates;
  * the rank-elite (cross-entropy) refit fits the top-k by lnL, but those elites
    are spread along the curved degeneracy ridge, so the fit is a broad Gaussian
    over the ridge -- more components do not help because the ESS-1 domination is
    set by the *narrow* constrained directions, not the ridge.
AV (VARAHA) is not importance sampling: it contracts an axis-aligned live volume
against a likelihood threshold with a coverage floor, which is the right structure
for a sharply-peaked target (at the cost of not wrapping the diagonal arc, hence
its own ~3e-5 efficiency ceiling).

## Recommendation / tradeoffs

* **Keep** the three fixes -- they are correct and independently valuable:
  O(k^3) matching, warm-start survival, and BIC allocation remove the hard-coding
  and make a warm GMM actually warm. The synthetic shows the allocation is robust
  and unbiased.
* **Do not** expect pure GMM to beat AV on high-SNR, strongly-degenerate events.
  Use the flexible GMM **inside the portfolio** (`--sampler-method portfolio`),
  where AV carries convergence and the balance-heuristic mixture density keeps a
  weak member from biasing -- now with a **hands-free** GMM member instead of a
  hand-tuned component layout. `--internal-gmm-adaptive-components` is wired
  through the portfolio (see Portfolio above); the remaining pre-existing blocker
  is the portfolio draw path's `draw_simplified` requirement for the `AV` member.
* **Future levers** for making GMM itself competitive here would be *coordinate*
  changes that de-curve the arc (a distance-inclination reparametrization,
  rotate-phase for phase<->pol) so a low-k axis-aligned mixture fits -- i.e.
  attack the correlation in coordinates, then let BIC pick k. See the breadcrumb's
  "real levers" section.

## Code map
* `RIFT/integrators/gaussian_mixture_model.py`: `_match_components` (Hungarian),
  `fit_gmm_adaptive`, `gmm.prune_components`, `gmm.num_free_params`,
  `add_defensive_component`, `_mixture_log_density_normalized`.
* `RIFT/integrators/MonteCarloEnsemble.py`: `integrator.gmm_adaptive/
  gmm_defensive_frac/gmm_inflate`; BIC-at-init in `_train`.
* `RIFT/integrators/mcsamplerEnsemble.py`: warm-start transfer in `integrate()`;
  `gmm_adaptive` threading in `integrate()`/`setup()`.
* `bin/integrate_likelihood_extrinsic_batchmode`: the four flags + `gmm_adaptive`
  dict construction in the GMM section.
* `test/integrators/test_gmm_adaptive.py`: unit tests (5/5).
