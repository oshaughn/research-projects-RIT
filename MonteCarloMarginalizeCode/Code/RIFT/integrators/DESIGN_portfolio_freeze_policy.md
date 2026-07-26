# Portfolio freeze policy — tuning + benchmarks

## The problem (starved VARAHA workhorse)

The portfolio integrator (`mcsamplerPortfolio.py`) mixes several member samplers and reweights
them each chunk by their per-chunk effective sample size `n_ess` (`portfolio_default_weights`).
A member whose weight falls below `portfolio_freeze_wt` (0.05) stops updating its proposal.

A **VARAHA / AV** member is special: it contracts its live volume **only on the chunk it is
updated** (`update_sampling_prior_selfish`). On its first chunks — before it has contracted — its
`n_ess` is ~1, so the heuristic gives it weight ~0.01 < 0.05 and it is **frozen from chunk 1** and
never contracts. The portfolio then rides a stalling GMM member. On the high-SNR S250114ax event
this produced portfolio cold-peak `n_eff ≈ 1.9` versus standalone AV `≈ 89–100`.

## Why grace/revive alone cannot fix it

The earlier fix (commit f2d51de0) added GRACE (never freeze for the first `grace_iters` chunks)
and REVIVE (update a frozen member every `revive_period` chunks). These are **not sufficient** for
a VARAHA member, because of a weight **feedback loop**: AV only earns allocation weight once its
`n_ess` climbs, and its `n_ess` only climbs once it contracts, and it only contracts when updated.
Updating it merely 1/`revive_period` of the time contracts it far too slowly for its `n_ess` to
ever win weight, so it stays starved. The revive sweep below confirms this: revive-every-{8,4,2}
all leave AV at `n_eff` ≈ 1–3, essentially no better than a hard freeze.

## The fix: VARAHA members are freeze-EXEMPT by default

Because the portfolio combines members with the **balance-heuristic mixture density** `q_mix`
(`integrate_log`), the estimate is unbiased for **any** member weights — a continuously-updated
VARAHA member can only ever cost a few extra selfish-draw evaluations, never bias the integral. So
the new default (`portfolio_varaha_never_freeze=True`) makes VARAHA/AV members update **every
chunk** past their activation breakpoint, exactly like a standalone AV. Set it False to fall back
to the grace/revive schedule (e.g. to save eval cycles on a VARAHA member known to be a bad fit).

Also added: a **plateau-aware revive** (`_climbing`) that keeps updating *any* low-weight member
while its own per-chunk `n_ess` is still rising, and a per-member `n_ess` history for diagnostics.

### Knobs (sampler defaults, all overridable)
| knob | default | meaning |
|------|---------|---------|
| `portfolio_varaha_never_freeze` | **True** | VARAHA members update every chunk past breakpoint |
| `portfolio_grace_iters` | 25 | never freeze ANY member during the first N chunks |
| `portfolio_revive_period` | 8 | update even a frozen member every N chunks (0 disables) |
| `portfolio_freeze_wt` | 0.05 | weight below which a (non-exempt) member stops updating |

### CLI flags (driver `integrate_likelihood_extrinsic_batchmode`, thread into `sampler.setup`)
`--portfolio-grace-iters N`, `--portfolio-revive-period N`, `--portfolio-freeze-wt X`,
`--portfolio-varaha-never-freeze` (explicit; already the default), `--portfolio-varaha-can-freeze`
(disable the exemption). Also fixed: `--sampler-portfolio AV,GMM` is now comma-split as documented
(it was silently becoming a single bogus member).

## Benchmark 1 — S250114ax (deliberately HARD: ρ≈82, broad/degenerate extrinsic posterior)

iteration-0 worker, IMRPhenomD, real H1L1, GPU A100. `N` = sample count at which `n_eff` first
crosses each threshold; `--n-max 4e6 --n-eff 100`. Warm = PE-oracle seed (cover 0.05/inflate 1.3).

| run | Neff≥5 | ≥10 | ≥20 | ≥50 | ≥100 | final (N, Neff) |
|-----|-------:|----:|----:|----:|-----:|:---------------:|
| **av_warm** (standalone, reference) | 0.70M | 1.37M | 1.69M | 2.27M | **3.64M** | 3.64M, 100 |
| **pf_nf_warm** (portfolio, never-freeze = NEW default) | **0.52M** | **1.10M** | 1.86M | 3.82M | — | 4.0M, **53** |
| pf_cf_warm (portfolio, can-freeze, grace25/revive8) | — | — | — | — | — | 4.0M, 3.4 |
| pf_cf_warm, revive=4 | — | — | — | — | — | 4.0M, 1.3 |
| pf_cf_warm, revive=2 | — | — | — | — | — | 4.0M, 1.6 |
| av_cold (standalone) | 1.75M | — | — | — | — | 4.0M, 3.7 |
| pf_nf_cold (portfolio, never-freeze) | — | — | — | — | — | 0.21M, 1.1 (cold-degenerate, clean stop) |
| GMM (standalone, cold AND warm) | — | — | — | — | — | **NaN on chunk 1 → bails** |

**Reading it.**
- **Never-freeze rescues the workhorse: `n_eff` 3.4 → 53** (~15×) vs the frozen policy, and it
  *tracks or beats* standalone AV through the useful range — it reaches `n_eff`=5 and 10 EARLIER
  than standalone AV (0.52M vs 0.70M; 1.10M vs 1.37M), because early on the AV+GMM mixture covers
  better than warm-AV alone.
- **grace/revive tuning does NOT rescue AV** (all can-freeze variants stay at `n_eff` 1–3),
  confirming the feedback-loop argument above.
- On this *atypical* event the portfolio's deep tail (`n_eff` 50→100) is ~1.5× slower than pure AV
  (53 @ 4M vs 100 @ 3.64M): AV alone is optimal here, and the portfolio pays a modest cost for
  carrying a weak GMM member. This is the honest limit — on a genuinely AV-optimal, GMM-hostile
  event the portfolio cannot beat standalone AV, but with never-freeze it is now in the same
  regime instead of starved.
- **Standalone GMM is unusable on this event** (NaN on chunk 1, cold and warm) — it only works
  *inside* the portfolio, where AV's coverage via `q_mix` + the NaN-weight guard stabilize it.

## Benchmark 2 — multi-event robustness (typical events)

Goal (per reviewer): confirm the AV+GMM portfolio (never-freeze) **replicates the standalone-AV
integral** — same ln Z within MC error — and converges comparably across a spread of *typical*
real events, not just the hard S250114ax target. Method: each event's real iteration-0 ILE worker
(real strain/PSD/intrinsic grid), run in the event's own production container (SEOBNRv5PHM +
gwsignal + cuda118 cupy) with this branch's integrator on `PYTHONPATH`; only `--sampler-method`
differs between the two configs. Single intrinsic point, `--n-eff 30 --n-max 8e5`.

| event | AV lnZ (n_eff @ N) | portfolio lnZ (n_eff @ N) | ΔlnZ | replicated? |
|-------|:------------------:|:-------------------------:|-----:|:-----------:|
| S231026ab | 17.64 (33 @ 81k)  | 17.62 (19 @ 800k) | 0.02 | **yes** |
| S240426s  | 29.75 (31 @ 60k)  | 29.68 (31 @ 310k) | 0.08 | **yes** |
| S240513ei | 85.73 (9.5 @ 809k)| 85.33 (1.6 @ 800k)| 0.40 | yes, within MC err (both under-converged) |
| S240703ad | 42.67 (11 @ 803k) | 41.70 (2.8 @ 800k)| 0.97 | ~ (pf under-converged, n_eff 2.8) |
| S240601aj | — missing BayesWave glitch-subtracted frame (event-data issue, not integrator) — dropped |

**Verdict.** The portfolio (never-freeze) **replicates the standalone-AV integral** — every ΔlnZ is
within the (often large) MC error of the lower-n_eff run, and there is no bias. AV is **never
frozen** on any event (0 freeze notices) and, on typical events, becomes the in-portfolio workhorse
(its balance weight climbs from 0.5 to ~0.65–0.72 within a few chunks). So the freeze mechanism is
**NOT fundamentally at odds** with VARAHA's need for continuous contraction — never-freeze gives it
exactly that, unbiased.

The remaining limit is **efficiency, and it is a DIFFERENT lever than freezing** (addressed next):
with the plain n_ess reweighting the portfolio spent a fixed ~half its budget on the GMM member, so
on AV-favorable events it reached a given n_eff in more evals than standalone AV (S240426s: same
n_eff at 310k vs 60k), and on the hardest events it stayed under-converged.

## Adaptive-probe draw allocation (the efficiency lever)

The reason the plain reweighting couldn't concentrate is a **draw catch-22**, structurally identical
to the freeze bug but on draws instead of updates: a member's per-chunk n_ess is *suppressed while
it has few draws* (a VARAHA member contracts slower with fewer samples; any Kish n_ess is noisy on a
small slice), so the member that *should* win is stuck under-observed and never earns more draws.

Mechanism (`portfolio_adaptive_alloc`, **OPT-IN, default OFF** — see the regression below): keep a
per-member **quality** estimate updated ONLY from chunks where the member had a *fair* allocation;
allocate draws by `quality^exponent` above a small floor; and **round-robin probe** one member per
`probe_period` chunks at a raised share so a suppressed member gets a fair look. `q_mix` keeps every
allocation unbiased. Knobs (`setup` + CLI `--portfolio-adaptive-alloc` to enable):
`portfolio_alloc_exponent` (2.0), `portfolio_alloc_floor` (0.05), `portfolio_quality_decay` (0.5),
`portfolio_probe_period` (4), `portfolio_probe_frac` (0.6).

### Choosing the quality signal (`portfolio_quality_signal`)

Three candidates were implemented and measured. Only the third is defensible, and even it cannot
rescue S250114ax — for a reason that turns out **not** to be about allocation at all.

1. **`ness`** — per-member Kish n_ess. **Fails.** Kish is *scale-invariant* (`(Σw)²/Σw²` is
   unchanged if all `w` are scaled), so it cannot see whether a member's samples carry any integral
   mass: a self-consistent member sitting off-peak scores as well as one covering the peak. A warm
   GMM is instantly self-consistent (n_ess ~120) while a warm AV's per-chunk n_ess is genuinely ~1
   during its slow *cumulative* contraction (value emerges over ~70 chunks). On S250114ax this drove
   the true AV workhorse to the floor: **n_eff 8 vs 53** for the legacy allocation — a regression.
2. **mean weight** (per-sample contribution). **Also fails, backwards.** A *well-matched* proposal
   correctly has small uniform weights, while a broad proposal's rare huge-weight outlier sets the
   maximum. Measured on S250114ax: AV **1e-40** vs GMM **2e-4** — it penalizes the good member.
3. **`global` (default when adaptive is on)** — marginal gain in **pooled** n_eff per sample,
   `g_m = 2·mean_w_m/S − mean_w2_m/Q` (`S=Σw`, `Q=Σw²` over all samples). This is the right
   objective: it credits weight *mass* and debits weight *variance*. It works on the synthetic
   (below), but on S250114ax it still ranks GMM first — and the numbers say exactly why.

**The S250114ax diagnosis (allocation is not the bottleneck).** With the `global` signal the
measured values are AV `~1e-73` and GMM `1.053e-4`. That GMM value is precisely `1/9500 = 1/n_GMM`,
which is the analytic signature of **one sample owning the entire estimator**: for a member holding
the single dominant outlier, `g = 2/n − 1/n = 1/n`. So the chunk's maximum weight is a catastrophic
GMM outlier ~**10⁷³×** larger than any AV weight — a draw landing where `q_mix ≈ 0` but the target is
nonzero. No allocation signal computed from the current weights can rank AV above that, because the
pooled estimator genuinely *is* dominated by that one sample.

The consequence: on this event the ceiling is set by the GMM member's **unbounded importance
weights**, not by how draws are split. Even floored at 5% the GMM member still injects outliers, which
is why the legacy allocation reached only 53 (not AV's 100). **The next lever is therefore weight
bounding / member exclusion, not allocation**: a defensive covering component to bound `w`, clipping
or winsorizing member weights, or dropping a member whose weight distribution is unbounded. (PR #27's
`--internal-gmm-defensive-frac` is the related Hesterberg-defensive knob; its help notes it did not
help n_eff on this SNR~82 benchmark, consistent with this being a hard pathology.)

Because of this, adaptive allocation remains **opt-in**; the default keeps never-freeze + legacy
reweighting.

## Benchmark 3 — adaptive allocation on synthetic correlated targets

`test/integrators/test_portfolio_adaptive_alloc.py` (fast, GPU, no ILE). Standalone AV vs standalone
GMM vs AV+GMM portfolio, fixed budget (`nmax 4e5`, ndim 5), reporting the integrator's own eff_samp
and bias vs the analytic ln Z. The GMM member is broad-seeded (a wide peak-covering proposal) so the
test exercises the *allocation* given a functional GMM rather than gambling on cold GMM finding a
thin ridge; AV starts cold. Representative (seed 1234):

| target | AV n_eff (bias) | GMM n_eff (bias) | **portfolio** n_eff (bias) | portfolio wts (AV,GMM) |
|--------|:---------------:|:----------------:|:--------------------------:|:----------------------:|
| uncorrelated (axis-aligned) | 107 (−0.54) | 398 (−0.01) | **386** (−0.04) | 0.15, 0.85 |
| **correlated (compound-symmetric)** | 23 (−0.67) | 400 (−0.00) | **387** (−0.03) | 0.13, 0.87 |

- **On the correlated target GMM's full covariance crushes AV's axis-aligned bins (400 vs 23), and
  adaptive allocation concentrates on GMM (weight 0.87) so the portfolio BEATS standalone AV** — the
  whole point of a portfolio on a correlated problem, and the case the reviewer flagged as the only
  regime where beating AV is expected.
- A cold VARAHA/AV under-covers the Gaussian tails and is **biased low** (−0.6 to −0.9); the
  portfolio stays **unbiased** because the covering GMM enters `q_mix` — a second reason to prefer
  the portfolio. (This is the opposite regime from a warm, cover-frac'd AV on a real ILE likelihood,
  where AV is the unbiased workhorse; the point demonstrated is that adaptive allocation follows
  whichever member is actually winning — GMM here, AV on a real AV-favorable event.)

The synthetic result shows the *mechanism* is sound **when the quality signal is right** (there GMM
is genuinely better and adaptive follows it). The S250114ax regression shows the *signal* is wrong on
real high-SNR AV-favorable events. Hence adaptive is shipped **opt-in**, and the default portfolio is
never-freeze + legacy allocation.

**Overall verdict.** Never-freeze (default) makes the portfolio unbiased and never-starved — it
**replicates standalone AV** and, on typical events, lets AV be the workhorse. Adaptive-probe
allocation (opt-in, with the `global` marginal-pooled-n_eff signal) makes a portfolio **beat AV on a
strongly-correlated target** (synthetic: ~375 vs ~61 n_eff) — the only regime where beating AV is
expected.

It is still opt-in because of what the global signal *revealed* rather than any deficiency in it:
on S250114ax the pooled estimator is dominated by a single GMM outlier ~10⁷³× the next weight, so
**allocation is not the bottleneck there — unbounded member weights are**. Turning adaptive on
everywhere requires first bounding those weights (defensive component / weight clipping / dropping a
member with an unbounded weight distribution). That is the concrete next lever, and it is a
*member-quality* fix, not an allocation-policy one.

**Robustness bug fixed along the way (important):** portfolio plugin discovery hard-loaded every
registered plugin at import, and the `NF` plugin does `import torch`, absent in the production GPU
container — so `import mcsamplerPortfolio` raised there, the driver silently set
`mcsampler_Portfolio_ok=False`, and every portfolio run died with a `NameError`. The portfolio
integrator was effectively **unusable in the production container**. Plugin loading is now wrapped
in try/except so a plugin with missing optional deps is skipped, not fatal.

## Weight clipping (truncated IS) — ADAPTATION-STREAM ONLY; **do not promote to AV yet**

`portfolio_weight_clip` / `--portfolio-weight-clip C` (OPT-IN, default off) caps weights at
`tau = C*sqrt(n)*mean(w)` (Ionides 2008 truncated IS). Clipping is a **biased** operation, so the
key design decision is *what it is allowed to touch*.

**⚠ SUPERSEDED — the outliers were an ARTIFACT after all (see PR #33).** This section originally
concluded the 10⁷³× weights were genuine heavy tails, because an instrumented run counted **zero**
`q_mix = max(acc, 1e-300)` underflows. That test was correct but incomplete: it rules out a density
*underflow*, not a density *lie*. PR #33 subsequently found exactly such a lie — AV's
`draw_simplified` head-sliced a **bin-ordered** cloud (returning only ~50–60% of the live-volume
bins) while `sampling_density` claimed uniform coverage of **all** occupied bins, so `q_mix` was
simply wrong, and a member drawing in the region AV never populates gets an arbitrarily inflated
weight. PR #33 also found that default-wired portfolio GMM members **never trained** (`n_comp=None`
silently no-op'd), so the runs below carried an untrained corpse member — which both produced junk
draws and partially masked the density bug.

**Consequences for everything below:** the S250114ax numbers in this document were taken with a dead
GMM member and a lying AV draw density, so they characterize *those bugs*, not the integrator's real
behavior. The clipping study remains valid as a *methodological* result (which quantities may be
clipped, and why — those arguments are analytic, not event-specific), but every S250114ax
efficiency/ln Z figure needs re-measuring on top of PR #33. The `q_mix UNDERFLOW` counter stays as a
permanent guard, and the lesson generalizes: **when a weight looks impossible, test the density for a
LIE (does the member actually draw where it claims density?), not just for underflow.**

**First attempt — clip the estimator: a disguised disaster (kept as the cautionary result).**

| run | Neff≥5 | Neff=100 | final n_eff | **ln Z** |
|-----|-------:|---------:|------------:|---------:|
| standalone AV (reference) | 0.695M | 3.638M | 100.2 | **1191.79** |
| portfolio, no clip | 0.520M | — | 52.6 @4M | 1183.12 |
| portfolio, **estimator**-clip C=1 | **0.030M** | **1.870M** | 100.1 | **1180.25** |

Clipping the estimator reached n_eff=100 in 1.87M evals — 2× faster than standalone AV — and reported
a perfectly converged run, while biasing ln Z **11.5 nats low** (independent reparam/aniso runs give
~1191.9). **n_eff stops being a validity check the moment the estimator is clipped.** Do not do this.

**The unbiased redesign — clip the PROPOSAL-FIT INPUT only.** Clipping is both biased *and*
n_ess-distorting, so its scope must be narrow. Final split:
- `log_integrand` → the ESTIMATE (`init_log`/`update_log` → ln Z; `maxval` → n_eff) — UNCLIPPED.
- per-member **n_ess report** and the **allocation signal** — UNCLIPPED (true weights).
- ONLY `member.update_sampling_prior` (the GMM covariance fit) gets the clipped copy
  `log_weights_adapt`, so one enormous weight can't make that fit degenerate.

Two failure modes ruled this scoping. (a) Clipping the **estimator** biases ln Z (the trap above).
(b) Clipping the **n_ess report / allocation** is *also* wrong, and subtly: clipping flattens
weights, which INFLATES a member's Kish n_ess, so the allocation perversely rewards the very member
whose weights had to be clipped. Measured on S250114ax, a first attempt that clipped the report
starved the AV workhorse to the 1% floor and stuck n_eff at ~1 (worse than no-clip's 53). Restricting
clipping to the proposal fit removed that: a short run climbs n_eff normally again. Proposal fitting
only *shapes* the proposal (like warm-starts / oracles / `q_mix`), so it cannot bias ln Z — and this
is strictly better than *dropping* a clipped chunk, which (being conditional on "a big weight
appeared") is data-dependent selection that would bias ln Z low.

**Synthetic ground truth** (`test/integrators/bench_weight_clip.py`, analytic ln Z, 2 seeds): with the
adaptation-only design the estimator bias is **unchanged at every C** (the falsifiable proof the
estimate is untouched), and where weights are well-behaved clipping is a complete no-op:

| target | C | n_eff | bias |
|--------|---|------:|-----:|
| uncorrelated | 0 / 1 / 5 | 386 / 385 / 385 | −0.035 / −0.039 / −0.039 |
| correlated   | 0 / 1 / 5 | 389 / 387 / 388 | −0.029 / −0.032 / −0.031 |

**Real S250114ax — unbiased, and (with the correct scope) not harmful.** Proposal-fit clip C=1
keeps ln Z at the unclipped ~1183 (NOT the biased estimator-clip 1180.3 — the estimator is provably
untouched) and n_eff climbs normally again (a short run tracks the no-clip curve). It does not *speed
up* this event either: the GMM member's proposal is fundamentally heavy-tailed here, and clipping
only stops its covariance fit from going degenerate — it cannot make a bad proposal good. The right
move on this event remains to not carry the GMM member. (An earlier, wrongly-scoped attempt that also
clipped the n_ess report collapsed n_eff to ~1 by starving AV — see the redesign note above; that was
the bug, not a property of clipping.)

**Side finding (refines the Benchmark-2 claim).** Even *unclipped* the AV+GMM portfolio reads
ln Z = 1183.1 here, 8.7 nats below AV. Heavy-tailed IS is unbiased in expectation but realizes LOW in
almost every run, so in production it behaves like a bias. "Portfolio replicates AV's ln Z" holds on
the four *typical* events (Benchmark 2); it does **not** on S250114ax.

**Can clipping rescue adaptive allocation on S250114ax?** No. Idea: the global allocation signal was
fooled because one 10⁷³ outlier owned the pooled estimator; since the signal now reads the *clipped*
adaptation weights, a gentle clip (C=20) that removes only that single outlier does flip the
first-chunk signal to correctly favor AV (contrib AV 2e-4 vs GMM 9e-21). But it does not *persist*:
the per-chunk marginal-n_eff signal is too noisy on this event — it oscillates back to GMM within a
few chunks, and both C=1 and C=20 stall the estimator at n_eff ~1 (worse than legacy's 53). So on an
AV-favorable event adaptive allocation fails clipped or not; **legacy allocation stays the right
default there, and adaptive remains a correlated-problem tool.**

**Typical-event safety validation** (proposal-fit clip C=1, 4 real events in-container, vs no-clip).
Clipping is unbiased everywhere (estimator untouched) and a no-op-to-mild-help on n_eff — and it
recovered the two hard events the broad-scope bug had degraded:

| event | no-clip n_eff (lnZ) | proposal-fit clip n_eff (lnZ) |
|-------|:-------------------:|:-----------------------------:|
| S231026ab | 19 (17.62) | 25.5 (17.47) |
| S240426s  | 31 (29.68) | 30 (29.74) |
| S240513ei | 1.6 (85.33) | 1.3 (83.20) |
| S240703ad | 2.8 (41.70) | **6.9** (42.43) |

On S240703ad clipping the GMM covariance fit *helped* (n_eff 2.8 → 6.9): protecting the fit from
outliers yielded a better proposal. lnZ differences are within the large MC error at these low n_eff
(the unbiased estimator realizes low on heavy-tailed events; see the side finding below).

**Verdict.** Proposal-fit clipping is the *correct, unbiased* form of the tool: on well-behaved
weights it is a no-op, and it is a safety valve against a single pathological weight wrecking a
member's covariance fit or the allocation signal. It does **not** manufacture n_eff — where the tail
carries the integral, clipping the adaptation only hurts. The durably valuable artifact is the tracked
withheld-mass fraction, a sharp cheap statement of whether an estimate/fit hangs on a handful of
samples.

**Guidance before promoting to the individual integrators (AV in particular):**
1. Clip only quantities that feed *adaptation*, never the estimator. If a future design must clip an
   estimator, it needs the mass tracking **and** a refuse-to-clip gate (abort once the withheld
   fraction exceeds ~1e-3) — otherwise it trades evidence accuracy for a flattering n_eff.
2. Surface the withheld-mass fraction as a first-class run diagnostic regardless — it detects the "a
   few samples carry the integral" regime that also invalidates n_eff.
3. Deep weight changes inside AV need the **full LVK PP campaign**, not one-off runs: a −0.1 nat ln Z
   bias passes a single-event n_eff check but shows up as PP miscalibration. The −11.5 nat bias hiding
   behind a perfect n_eff above is exactly why.

## POST-#33 RE-MEASUREMENT (S250114ax) — the live-GMM result

Re-run on top of PR #33 (GMM members actually train; AV's `draw_simplified` no longer lies about its
density), warm, same budget:

| run | Neff≥5 | Neff≥10 | final n_eff @4M | note |
|-----|-------:|--------:|----------------:|------|
| standalone AV (reference) | 0.695M | 1.374M | **100.2** | unchanged by #33 (standalone AV paths untouched) |
| portfolio, default | — | — | **2.1** | was 52.6 pre-#33 (when the GMM was a corpse) |
| portfolio, adaptive alloc | — | — | 1.1 | |
| portfolio, VARAHA draw floor 0.5 / 0.7 | — | — | ~1–2 | AV got 0.97 / 0.85 of draws |
| portfolio + defensive GMM 0.05 | — | — | 2.5 | defensive mixture alone does not fix it |
| **portfolio + proposal-fit clip** | **0.420M** | **0.590M** | 14.4 | **beats AV to the production target** |

**The portfolio got *worse* when the GMM member came alive.** Pre-#33 the GMM member was a corpse, so
the portfolio was effectively AV-only and scored 52.6; now that it genuinely trains and draws, the
allocation hands it ~0.84 of the budget and the pooled n_eff collapses to ~2 — against standalone
AV's 100. This is not a freeze problem (never-freeze is working: AV updates every chunk) and not a
clipping problem. It is the **draw-allocation** pathology in its clean form: both allocation rules
score members by per-chunk n_ess, and a VARAHA member's per-chunk n_ess sits at ~1 throughout its
slow *cumulative* contraction, so a member that looks instantly good takes the budget.

**New opt-in lever: `portfolio_varaha_min_frac` / `--portfolio-varaha-min-frac`** reserves a combined
draw fraction for VARAHA members, applied after either allocation rule (legacy or adaptive). It does
what it says — AV's share went to 0.97 (floor 0.5) and 0.85 (floor 0.7) — but it **does not rescue
this event**: even a 3–15% GMM share still poisons the pooled n_eff, which stayed ~1–2.

**What actually fixes it: clipping the PROPOSAL-FIT input (the "don't poison the sampling model"
lever).** With the GMM member alive, the thing that goes wrong is its *fit* being corrupted by a few
enormous weights; capping the weights that train it (`--portfolio-weight-clip 1.0`, estimator
untouched) turns the collapse around:

* **n_eff=5 at 0.420M vs AV's 0.695M (1.7× faster); n_eff=10 at 0.590M vs AV's 1.374M (2.3× faster).**
* **This is the production regime**: the real O4 event configs run `--n-eff 10`, so at the target that
  actually ships, the clipped AV+GMM portfolio *beats* standalone AV by ~2.3× on this event.
* It then **plateaus at ~14** rather than climbing to AV's 100, so AV alone still wins the *stress*
  target (n_eff 100). The ceiling, not the approach, is what the live GMM member still costs.

Note this only became visible after #33: pre-#33 the GMM member was a corpse, so there was nothing to
poison and clipping did nothing. Ordering of levers on this event: clip (2.1 → 14.4) ≫ defensive
mixture (2.1 → 2.5) ≈ VARAHA draw floor (no rescue) > adaptive allocation (actively worse, 1.1).

**Remaining gap.** For the stress target the portfolio is still ceiling-limited by the GMM member, and
down-weighting does not fix that (the draw-floor runs show even a 3–15% share caps the pool). Closing
it needs member *exclusion* (drop a member whose credit is persistently negligible) rather than more
re-weighting. Practical guidance today: **use the portfolio with proposal-fit clipping for production
n_eff targets; use standalone AV if you need n_eff ≫ 10 on an AV-favorable event.**

## Shape-recovery merge gate (PR #31 requirement)

Run on a quiet node (pcdev11; pcdev12 at load ~440 kills the suite via RLIMIT_NPROC), base
`rift_O4d @4bac7444` vs this branch, both incl. PR #33. Harness: `~/rift_gate_out/run_gate.sh`.

**Result: `COMPARE_EXIT=0`, 0 blocking regressions** — base and PR identical in aggregate (strict 8/8,
warn-only 5/5, starved 45/45), and 22 of 23 portfolio rows bitwise identical to base.

Getting there required fixing a regression the gate caught, which is worth recording because the
cause was the opposite of the obvious one:

* The first gate run showed ~13 of 20 portfolio rows losing n_eff vs base, several by 2–4×. It did
  **not block** (portfolio is warn-only, strict = AV+GMM), but this PR changes the portfolio default
  path, so it needed attribution rather than a pass-by-classification.
* **never-freeze — the headline default — was NOT the cause**: toggling it gives ratio 1.00 on those
  targets (it only engages where a member would actually be frozen).
* **The plateau-aware `_climbing` revive WAS the sole cause.** Toggling it reproduces base exactly:
  `d4_n1_s101 25.9→53.5`, `d4_n3_s202 29.1→83.8`, `d6_n1_s202 64.0→102.1`, `d6_n3_s202 7.2→31.4`,
  `d8_n1_s101 37.3→61.9` (base 53.5 / 83.8 / 102.1 / 31.4 / 61.9). Forcing updates of members the
  freeze schedule would have parked makes their proposals **worse** — the inverse of the intuition
  that motivated it. It now defaults **off** (opt-in only).

**The one remaining difference** is `mix_d2_n3_s303` (n_eff 736→517, bias −0.0060→−0.0064): the row
where never-freeze genuinely engages. Both PASS comfortably (n_eff ≫ 100, bias unchanged), but it
quantifies never-freeze's cost — **it buys starvation-immunity and pays ~30% n_eff where freezing
would have been harmless.**

**Method note (a trap worth avoiding).** An isolation that drives `shape_recovery` as a library must
export `PYTHONPATH` (the checkout under test), `CUDA_VISIBLE_DEVICES=""` and `OMP_NUM_THREADS` — only
the wrapper `run_shape_recovery.sh` sets these. My first isolation didn't, silently imported the
**installed** RIFT (where the knob under test does not exist), and confidently reported "no effect"
with n_eff nowhere near the gate's. **A valid isolation reproduces the gate's absolute numbers
row-for-row**; that check is what exposed it. `probe_portfolio_optin_flags.py` now sets this env itself.

### Flag-ON probe (TESTING.md requirement for opt-in changes)

`test/expensive_before_merging/integrators/probe_portfolio_optin_flags.py` scores the opt-in features
with the gate's own targets, metrics and `evaluate()`, so a PASS here is a PASS by gate criteria.
**Result: 0 opt-in regressions**, and both features materially help:

| target | flags OFF | adaptive_alloc ON | weight_clip ON |
|--------|----------:|------------------:|---------------:|
| d2_n1_s303 | 1502 | **3021** | **3037** |
| d2_n3_s303 |  517 | **1058** |  **816** |
| d4_n1_s303 |  163 |  **415** |  **263** |
| d4_n3_s303 |    7 (starved) | 15 | **54** |

Bias stays small on every PASS row (|lnI−lnZ| ≤ 0.024). This independently corroborates the
S250114ax finding: clipping lifts the worst (starved) row 7 → 54. Caveat: on that starved row the
bias grows (−0.147 → −0.279) — at n_eff ≲ 50 the shape is untestable, so treat clipping's gains in
the starved regime as unvalidated for *shape*, even though n_eff improves.

Note `adaptive+clip` is identical to `clip` alone on these targets — with clipping active the
allocation rule made no further difference here.

## Multi-event clip validation, post-#33 (does the S250114ax clip win generalise? — NO)

4 typical O4 events, warm, `--n-eff 30`, in-container real SEOBNRv5PHM, no-clip vs proposal-fit
`--portfolio-weight-clip 1.0`. Run interactively in the `cuda128` container on idle Blackwell nodes
(pcdev11/13) — which also confirmed SEOBNRv5PHM+cupy run on CC 12.0, matching the A100 result.

| event | no-clip lnZ (n_eff) | clip lnZ (n_eff) | ΔlnZ | n_eff ratio |
|-------|:-------------------:|:----------------:|-----:|:-----------:|
| S231026ab | 17.54 (28.9) | 17.49 (29.8) | −0.05 | ×1.03 |
| S240426s  | 29.74 (31.2) | 29.56 (30.3) | −0.18 | ×0.97 |
| S240513ei | 83.76 (3.1)  | 83.83 (1.3)  | +0.07 | ×0.42 |
| S240703ad | 41.89 (3.3)  | 42.27 (5.0)  | +0.38 | ×1.53 |

**Conclusion.** Clipping's dramatic S250114ax result (n_eff=10 2.3× faster than standalone AV) is
**specific to that event's extreme heavy-tailed pathology and does NOT generalise.** On typical events
it is a near-noop (×0.97–1.03); on the two under-converged hard events it is a wash (one up, one down,
both inside the n_eff≈1–5 scatter). ln Z agrees everywhere (|ΔlnZ| ≤ 0.38, within MC error at these
n_eff) — the portfolio replicates the AV integral with or without clipping. This is exactly why
clipping ships **opt-in, default off**: a targeted tool for a specific failure mode, not a general
speedup to impose on typical runs. It closes the study's last open question.

## The portfolio's actual purpose: rescuing a high-SNR BEST-FIT evaluation

Everything above tunes the integrator on a *trial* grid point (`overlap-grid-0.xml.gz --event 0`,
m1/m2 28.29/26.69, lnLmax≈1212) — fine for A/B-ing policy, but it is not the science target. The
target that has to work for a loud event is the **best-fit on-source point**
(`target_params.xml.gz`, m1/m2 37.71/34.03, lnLmax≈3040, ρ≈78): sharply peaked AND
distance-inclination/sky correlated. Measured on GPU, warm, bias-safe cover 0.5, `bench_onsource.sh`:

| sampler | final n_eff @4M | note |
|---------|----------------:|------|
| AV alone | **1.0** | stalls — axis-aligned bins cannot wrap the correlated peak (also confirms the cardassia CPU result on GPU) |
| GMM alone (adaptive) | **NaN chunk-1** | no coverage floor → weights blow up |
| **AV+GMM portfolio (adaptive)** | **14.7** (lnZ 3016) | ~15× AV, and works where NEITHER member works alone |

**This is the clearest demonstration of why the portfolio exists.** It is a *GMM-peak + AV-coverage*
event: the GMM member wraps the correlated peak (which AV cannot), and the "dead" AV member — it
reports `nan` per-chunk n_ess in 381/400 chunks and sits at the 1% floor — is NOT wasted, because its
broad warm density still enters `q_mix = frac_AV q_AV + frac_GMM q_GMM`, providing the coverage floor
that keeps GMM's importance weights bounded. Remove AV (run GMM alone) and GMM NaNs; remove GMM (run
AV alone) and it stalls at 1.0. The portfolio is exactly the vehicle that combines a peak-finder with
a coverage member, and the never-freeze/allocation machinery above is what lets it hand the budget to
whichever one is actually working — here, GMM.

**Adaptive GMM coverage is the lever — but it is NON-MONOTONIC, with a sweet spot:**

| portfolio config (AV+GMM, warm 0.5) | GMM BIC cap | inflate | n_eff @4M | lnZ |
|-------------------------------------|-----------:|--------:|----------:|-------:|
| baseline | 8 | 1.0 | 14.7 | 3016.13 |
| **sweet spot** | 16 | 1.3 | **56.1** | 3016.08 |
| over-cranked | 24 | 1.5 | **2.3** | 3009.5 ⚠ |

At the sweet spot: **56× standalone AV** (which stalls at 1.0), ln Z unchanged (3016.08 vs 3016.13) —
real efficiency, not a coverage-shortcut bias. **But more is not better**: cap 24 / inflate 1.5
collapses to 2.3 AND ln Z drops 6.6 nats (3009.5) — the lnZ shift means over-inflation is biasing,
not just adding variance (an over-wide GMM proposal + a few enormous weights, the same heavy-tail
mode weight clipping was aimed at). So the reviewer's "GMM event with adaptive coverage" framing is
confirmed quantitatively, with the caveat that the coverage knobs need *tuning to a sweet spot*, not
maximizing. Which of the two knobs (BIC cap vs inflation) drives the collapse is under isolation.
Harness: `test/integrators/bench_onsource.sh` (pins the best-fit point; documents it is NOT the trial
point).

## Files
- `RIFT/integrators/mcsamplerPortfolio.py` — freeze-policy + adaptive-probe allocation, knobs,
  n_ess history, plugin-load guard, NaN guard.
- `bin/integrate_likelihood_extrinsic_batchmode` — CLI flags (freeze + allocation) + comma-split fix.
- `test/integrators/bench_portfolio_freeze.sh` — S250114ax single-config runner.
- `test/integrators/bench_multi_event.py` + `run_multi_event.sh` — multi-event robustness suite.
- `test/integrators/parse_neff_traj.py` — trajectory → n_eff-vs-N table parser.
- `test/integrators/test_portfolio_adaptive_alloc.py` — synthetic correlated/uncorrelated test that
  the portfolio tracks the winning member and beats AV on a correlated target (Benchmark 3).
- `test/integrators/bench_weight_clip.py` — clipping bias-vs-n_eff sweep against analytic ln Z.

## The high-SNR rescue is FLAKY — a single run is not a posterior (seed ensemble)

The 56.1 sweet-spot number above is a **single lucky draw**, not the typical outcome. Repeating
cap 16 / inflate 1.3 across seeds (same best-fit on-source point, warm 0.5, n_max 4M):

| copy | n_eff @4M | lnZ |
|------|----------:|-------:|
| unseeded | **56.1** | 3016.08 |
| seed 1 | 1.3 | 3011.20 |
| seed 2 | 1.5 | 3017.07 |
| seed 3 | 9.2 | 3015.96 |
| seed 4 | 1.2 | 3010.65 |

Median n_eff ≈ **1.5**; the distribution is bimodal (mostly collapsed, occasionally lands). lnZ
swings 3010.6 → 3017.1 (6.4 nats) with **no clean sign** — a single dominating outlier can push
evidence high (seed 2: n_eff 1.5 but lnZ 3017.1) or low (seed 4). **One low-n_eff portfolio run on a
high-SNR event is not a usable posterior at any budget.** The operational recipe for such events is
to run MANY independent copies and pool (below), or find a proposal that reliably lands high n_eff.

**Pooling recovers the answer — but only if pooled by reliability, not naively.** n_eff-weighted mean
of the five copies' lnZ = **3016.0** (the two high-n_eff copies, 3016.08 & 3015.96, agree and
dominate); the *unweighted* mean is 3014.2, biased ~1.8 nats low by the collapsed copies. Naive
concatenation of raw importance samples is WORSE than either: it is dominated by whichever copy owns
the single largest weight — which may be a *collapsed* copy. So "pool many copies" means pool enough
that the **pooled cloud's own n_eff** is high; a handful of copies can still be outlier-dominated.

**cap-too-high is the failure mode (confirmed).** Bigger BIC cap / inflation → wider GMM proposal →
more prone to the single-enormous-weight collapse: cap 24 median n_eff 2.3–2.8 vs cap 16's occasional
56. The knob buys peak coverage at the cost of tail control; past the sweet spot the tail wins.

### `--save-samples` is unusable for portfolio shape-checks in this regime (four independent layers)

Trying to export the extrinsic cloud for a weighted-shape check surfaced that the export path fails
for a peaked (low-to-moderate n_eff) portfolio run at *four* layers — every seed above exported
**0 rows**, even seed 3 at n_eff 9.2:

1. **Fairdraw** (`--fairdraw-extrinsic-output`) resamples ∝ weight → at n_eff≈1 it returns copies of
   the one dominant point or nothing. Useless at low n_eff (per reviewer guidance).
2. **`--save-P` defaults to 0.1** — the export prunes the bottom 10% of *probability*; on a peaked
   cloud that discards nearly everything. Raw weighted export needs `--save-P 0`.
3. **`mcsamplerPortfolio` `_rvs` cleanup (draft-inherited, ~line 1135) cumulative-sums the LOG-weights**,
   not the weights, and is poisoned by any `-inf` ln_wt entry → 0 rows survive even at n_eff 9.2.
   (Pre-existing pattern copied from the ensemble sampler; flagged as a separate fix, not touched here.)
4. **The XML only carries `loglikelihood = log_integrand` (lnL), not the IS weight**
   (`log_integrand + log_joint_prior − log_joint_s_prior`). A weighted-posterior check off the XML is
   therefore wrong-by-construction (weights by likelihood, not posterior). `shape_extrinsic.py` had
   this bug.

**Correct path for a weight-aware shape/posterior check: `--extrinsic-proposal-output`** — it builds
the TRUE importance log-weights from the raw `_rvs` cloud (driver ~line 2856) and fits a per-group
GMM, bypassing all four failure layers. Pool/compare those GMM fits across copies for the posterior.
(NB: it still needs `--save-P 0`, or the same buggy `_rvs` cleanup prunes the cloud to 0 rows and the
fit dies with "zero-size array to reduction cupy_max". Same root bug as layer 3 above.)

## The real answer: n_eff is a LOTTERY for every config — pooling is mandatory

Goal (per reviewer): not max n_eff — *reliable modest* n_eff with a stable, unbiased extrinsic
posterior and no failure mode tied to extrinsic multimodality/degeneracy. Seed ensemble on the
best-fit high-SNR point (warm 0.5, n_max 4M), AV+GMM portfolio, GMM-coverage configs.

**FIRST, A CORRECTION / METHOD LESSON.** A 3-seed run of cap8 gave {7.0, 10.1, 13.7} and I wrote
"cap8 is reliably modest." That was survivorship bias on 3 draws — the exact trap this document warns
about. Extending cap8 to **10 draws** (GPU runs are non-deterministic even at fixed `--seed`: float
reduction order) gives:

    cap8 n_eff (10 draws):  1.00, 1.00, 1.06, 1.57, 7.0, 10.1, 13.7, 22.0, 55.3, 70.1
                            median ~8.5, range 1 -> 70, ~40% collapsed to ~1

cap8 is **just as bimodal as cap16** — it is not a reliability fix. n_eff on this high-SNR best-fit
point is a **lottery** for the portfolio regardless of the BIC cap: most runs collapse to ~1, a
minority land 10–70. lnZ tracks the mode (collapsed runs bias lnZ 5–11 nats low). Across configs:

| config | GMM coverage | n_eff draws | reliability |
|--------|-------------|-------------|-------------|
| cap8 (factored) | cap 8, inflate 1.0 | 1,1,1.06,1.6,7,10,14,22,55,70 (n=10) | bimodal lottery |
| cap16 (factored) | cap 16, inflate 1.3 | 1.2,1.3,1.5,3.5,9.2,13.6,56,59 (n=8) | bimodal lottery (~same) |
| corr (correlate-all) | single 6-D GMM, cap 8 | 1.8,1.9,20.6 (n=3) | strictly WORSE (see below) |

**Consequence (this is the reviewer's original point, now proven on real data): a single run — any
config — is NOT a posterior on this event. The only robust recipe is to run MANY independent copies
and pool.** Pool by reliability, not naively (see the pooling note above): the pooled *cloud's own*
n_eff must be high. The cap knob changes the odds of a good draw only marginally; it does not remove
the need to pool.

**The "strongly-correlated problem → correlate-all" hypothesis is REFUTED.** A single full-dimension
(6-D) GMM that *can* represent cross-group (sky–phase, dL–ι) correlation is the WORST here: it
collapses on 2 of 3 seeds and biases lnZ up to 11 nats low (3004.5). Reason: a 6-D mixture needs
~(d+2) effective samples per component; at the modest n_eff these runs produce, its covariances go
near-singular and a few enormous weights dominate. The **factored per-group (2-D) proposal is more
robust** precisely because each low-dimensional fit is cheap and well-conditioned — the correlation
it cannot represent costs less than the fitting variance a full-dim GMM incurs. So *more* proposal
expressiveness is the wrong lever; the lever is **more copies**.

Harness: `test/integrators/bench_onsource_ensemble.sh` + `compare_extrinsic_breadcrumbs.py`. The
weight-correct extrinsic export needed for the pooled shape check is unblocked by PR #35 (the
`--save-samples`/`_rvs` cleanup fix: linear-weight cumsum + `-inf` guard), cherry-picked here.

### The failure mode IS an extrinsic-degeneracy collapse — and pooling landed copies is robust

9-copy cap8 pool (seeds 10–18, `--extrinsic-proposal-output`): 4 landed (n_eff 15,36,39,41), 5
collapsed (n_eff 1–3.2). The weight-correct per-group GMM fits give a clean picture:

- **Mode count is a perfect collapse diagnostic, and the collapse is exactly the reviewer's worry.**
  Every LANDED copy fits **3–4 modes** in each degeneracy group — (ra,dec) sky **ring**, (distance,ι)
  arc, (φ,ψ). Every COLLAPSED copy fits **1 mode** in every group: a single degenerate blob that has
  **lost the sky ring / dL–ι arc / phase-pol structure**. So low n_eff ⟺ extrinsic *mode collapse*;
  the settings' instability is tied directly to multimodality/degeneracy, and n_eff (or the fitted
  mode count) detects it.
- **Landed copies AGREE — when it lands, the posterior is stable and reproducible.** Across the 4
  landers the (ra,dec) mixture mean agrees to ~0.01 (frame units) and (distance,ι) to ~0.02 in ι —
  i.e. the recovered extrinsic posterior is *consistent copy-to-copy*, no hidden instability among
  good runs. The exception is (φ_orb,ψ): scatter ~1.5 even among landers, because the 2-IFO phase–
  polarization degeneracy is genuinely the least-constrained extrinsic direction (expected, not a bug).
- **Reliability-weighted pooling ≈ good-only (correct); naive pooling is biased by the collapsed
  copies.** For the well-constrained sky group all three pooling recipes coincide, but for the looser
  distance and phase groups naive-unweighted pooling is pulled off the good-only answer (distance:
  naive vs good differ ~80 units; phase: −0.74 vs −1.88) while the n_eff-weighted pool tracks good-only.
  Reliability-weighted **effective #copies (Kish over n_eff) = 4.1** — the 9-copy pool really rests on
  its ~4 landers. **Operational recipe: run ~2–3× as many copies as landers you need, pool weighted by
  n_eff (or simply drop n_eff<5 copies).**

Caveat (honest): the comparator's *physical* un-normalization of the GMM means is in the wrong frame
(the RIFT GMM's internal normalization is not the naive [0,1]-on-bounds I assumed — all means flag
out-of-bounds, so that flag is unreliable). The conclusions above rest only on the frame-INDEPENDENT
signals — mode counts and copy-to-copy agreement (`good-scatter`) — not on absolute mean values. A
correct physical read needs the GMM model's normalization; the mode-collapse / pooling story does not.

### Coordinates/adaptation help the LANDERS, not the collapse rate — the collapse is an AV peak-lock lottery

Testing the reviewer's high-SNR recipe (`--force-adapt-all` + rotations). CONFOUND first: the
coordinate-transform flags (`--internal-rotate-phase`, `--internal-sky-network-coordinates`) change
what the sampler's parameter slots MEAN, but `--sampler-warmstart-samples` maps the seed by column
NAME without transforming values -> a PHYSICAL seed poisons the rotated/network proposal. Naive
"add the flags" run: 0/9 landed (every copy collapsed). Fix = a frame-matched seed
(`seed_phi_orb=mod(phi+psi,4pi)`, `seed_psi=mod(phi-psi,4pi)` for rotate-phase; `--force-adapt-all`
is frame-preserving and needs no transform). Now in the lore repo's gotchas.

With a frame-matched seed (`--force-adapt-all --internal-rotate-phase`, 9 copies):

| metric | baseline (physical) | +force-adapt-all+rotate-phase |
|--------|--------------------:|------------------------------:|
| landed fraction (n_eff>=5) | 4/9 | **4/9 (unchanged)** |
| landed n_eff | 15,36,39,41 | **41,52,52,41** (higher, tighter) |
| landed sky modes | 3-4 | 3-4 (ring preserved) |

So phase-decorrelation + full adaptation is a real efficiency win FOR THE LANDERS (n_eff ~50 vs ~30)
but does NOT move the ~55% collapse rate. The lottery is now robust across EVERY config tried (cap8,
cap16, correlate-all, +rotate-phase): same ~50% collapse, same signature (n_eff~1, single mode). The
root cause is therefore not the proposal/coordinates but **AV's contracting box locking onto the
sharp high-SNR peak or contracting around the wrong spot ~50/50** — and the portfolio cannot backstop
better than AV's own contraction reliability, because AV itself is the coin-flip.

**Targeted fix under test: L0 auto-rescue** (`--sampler-warmstart-retry-neff`). If a pass finishes
n_eff < threshold, re-seed AV from the run's OWN highest-L samples (the peak it did find) and re-run
— same-problem reuse, cannot bias, frame-safe by construction. Was gated to standalone AV; relaxed to
fire for the portfolio too (peak-seed bootstraps into the AV member). This is the in-loop version of
"pool copies": convert each collapsed draw into a land instead of discarding it. Result pending (prr_).

### THE HIGH-SNR FIX: L0 auto-rescue roughly DOUBLES the landed fraction (4/9 -> 8/9)

Since the collapse is AV losing the sharp peak ~50/50 (not a proposal/coordinate defect), the fix is
to re-seed a collapsed run from the peak IT DID FIND and re-run: `--sampler-warmstart-retry-neff 5`
(L0 auto-rescue). Bug found + fixed first: the rescue is gated on the sampler being AV or a
portfolio, but `opts.sampler_method` is CLOBBERED to 'GMM' during portfolio member setup (line ~1231,
`opts.sampler_method='GMM'` forces GMM arg-parsing), so an AV+GMM portfolio reports method 'GMM'
everywhere downstream. The gate now detects the portfolio via `opts.sampler_portfolio` (the member
list, which survives the clobber). [Same clobber makes the portfolio-only block at ~1641 dead code --
harmless, the GMM branch picks up the gmm_adaptive forwarding as a per-group dict -- but a latent
footgun; flagged for cleanup.]

9-copy pool, cap8 + `--force-adapt-all --internal-rotate-phase` (frame-matched seed) +
`--sampler-warmstart-retry-neff 5`:

| seed | prior behavior | prr_ result | rescue |
|------|---------------|------------:|:------:|
| s10 | collapse | 4.5 | fired (just under) |
| s11 | chronic ~1 collapse | **33.4** | fired -> LAND |
| s12 | collapse | 33.4 | (landed pass 1) |
| s13 | 35-52 | **47.0** | fired -> LAND |
| s14 | chronic ~1 collapse | **25.2** | fired -> LAND |
| s15 | mixed | **19.6** | fired -> LAND |
| s16 | collapse | **38.1** | fired -> LAND |
| s17 | 15 | 10.0 | (landed pass 1) |
| s18 | 39-41 | 21.4 | (landed pass 1) |

**LANDED 8/9** (baseline 4/9, pr_ 4/9); 6 rescues fired, 5 converted to clean lands and the 6th to
4.5. Chronic collapsers (s11, s14, both stuck at n_eff~1 across every prior config) now land at 25-33.
Cost: a rescued run does 2 integration passes (~2x). This is the in-loop equivalent of "pool copies",
and it is the real high-SNR lever -- coordinates/adaptation improve the LANDERS, the rescue fixes the
COLLAPSE RATE.

**Validated high-SNR recipe:** portfolio AV+GMM (cap8, adaptive components) + `--force-adapt-all`
+ `--internal-rotate-phase` (with a phase-frame-matched warm seed) + `--sampler-warmstart-retry-neff 5`.
Even so, for a publication-grade posterior at n_eff this modest, still pool a few landed copies.

## EVIDENCE AUDIT: which numbers in this document are single draws

The n_eff lottery (documented above) was discovered LATE, after much of this document was written.
Because a single run on a lottery-prone point is noise-dominated, several earlier claims here rest on
n=1 and must be read as suggestive, not established. Explicit audit:

**Downgraded to UNPROVEN (single draw on a bimodal quantity):**
- The GMM coverage ladder cap8=14.7 / cap16=56.1 / cap24=2.3 -- all n=1. The cap16 "sweet spot" is
  already retracted above; **the companion claim that cap24 over-cranking BIASES lnZ (3009.5, -6.6
  nats) is likewise a single draw and is NOT established.** A collapsed copy shifts lnZ in either
  direction (seed 2 of the cap16 ensemble: n_eff 1.5 but lnZ 3017.1, i.e. HIGH). Distinguishing
  genuine over-inflation bias from collapse noise needs a seed ensemble per cap, which has not run.
- `--internal-gmm-correlate-all` is worse: n=3 (2/3 collapsed, lnZ up to 11 nats low). Directionally
  supported and mechanistically plausible (a 6-D mixture needs ~(d+2) eff-samples/component), but not
  firm at n=3.
- Benchmark 1's cold rows (av_cold 3.7, pf_nf_cold 1.1): single draws on the lottery-prone point.

**Robust (large effect, understood mechanism, and/or well sampled):**
- Never-freeze rescues the workhorse (3.4 -> 53): large, mechanism understood (frozen at chunk 1),
  and independently corroborated by zero freeze notices across the multi-event suite.
- Multi-event ln Z replication (Benchmark 2, NON-warm-started): an UNBIASEDNESS claim, structurally
  guaranteed by the balance-heuristic q_mix (the estimate is unbiased for any member weights). The
  ΔlnZ agreement stands. (The n_eff-efficiency comparisons in that same table are single draws.)
- The lottery itself (cap8 n=10, cap16 n=8), the mode-collapse diagnosis (9 copies, clean 1-mode vs
  3-4-mode split), and the L0 auto-rescue 4/9 -> 8/9 (9 copies + a post-cleanup regression).

**OPEN: is the lottery high-SNR-only?** Every ensemble here is on the ultra-sharp best-fit point of a
loud event. If typical events are unimodal in n_eff, single-draw comparisons on them (Benchmark 2) are
fine as-is; if not, that table's efficiency numbers need ensembles too. Cheap to settle: one seed
ensemble on a typical event.

**Not a factor: the sampler_method clobber.** For an AV+GMM portfolio the clobber changed only whether
`return_lnI` was passed, and `mcsamplerPortfolio` never reads it (`use_lnL` was set either way, because
the portfolio branch force-sets `internal_use_lnL=True` before the clobber). Verified by regression:
identical per-group `gmm_adaptive` forwarding and identical rescue behaviour. No result in this
document is invalidated by removing it.

## COLD-START ensemble: n_eff does NOT certify correctness (the confidently-wrong failure)

Rerun of the cold (non-warm-started) case after two fixes landed: the pre-existing
`mcsamplerEnsemble` loop-invariant clobber (which had made EVERY cold portfolio start crash at
chunk ~8 with no output at all -- 0/9), and the L0 rescue now firing on degenerate early
termination. Config: portfolio AV+GMM cap8 adaptive, `--force-adapt-all --internal-rotate-phase
--interpolate-time True --sampler-warmstart-retry-neff 5`, 9 seeds, cold.

| seed | n_eff | lnZ | modes/group | rescue |
|------|------:|--------:|:-----------:|:------:|
| s10 | 1.5 | 3006.16 | 1 | fired |
| s11 | 13.9 | **3012.47** | 1 | fired |
| s12 | 31.0 | **3013.87** | 1 | fired |
| s13 | 1.0 | 3001.19 | 1 | fired |
| s14 | 38.0 | **3013.61** | 4 | fired |
| s15 | 1.1 | 3002.36 | 1 | fired |
| s16 | 18.0 | **3012.54** | 1 | fired |
| s17 | **58.0** | **3001.68** ⚠ | 1 | fired |
| s18 | 3.8 | 3015.53 | 2 | fired |

(lnZ is only comparable WITHIN this table: `--internal-rotate-phase` doubles the prior, so these
values are offset from the non-rotated benchmarks earlier in this document.)

**9/9 now produce output (was 0/9 -- the crash), 5/9 land (n_eff>=5).** Cold is materially worse than
warm+rescue (8/9), so a warm seed still earns its keep; but cold now WORKS, which it did not before.

**The headline result is the lnZ column, not the landed count.** Among the five landed copies lnZ
spans **3001.7 - 3013.9 (12 nats)**, and the single most wrong copy is the one with the **HIGHEST
n_eff**: s17, n_eff 58, lnZ 11 nats below the consensus. Four of five landers agree to within 1.4
nats (3012.5-3013.9); s17 dissents while looking, by n_eff, like the best run in the ensemble.

**Consequences (this changes the recommended practice):**
1. **n_eff is NECESSARY BUT NOT SUFFICIENT.** It measures weight concentration, not coverage. A pass
   that locks onto one narrow region has low weight variance (high n_eff) while missing posterior
   mass (lnZ too low) -- confidently wrong. You CANNOT pick the trustworthy copy by max n_eff, and a
   single high-n_eff run is not self-certifying.
2. **Use CONSENSUS across copies, not the best-n_eff copy.** The outlier here is detectable only by
   disagreeing with the pool. Prefer the median lnZ over landed copies (median 3012.54 correctly
   rejects s17) to an n_eff-argmax or even an n_eff-weighted mean (which s17's weight would drag
   down). This is a direct strengthening of the "run MANY copies" recipe: copies are needed not just
   to find a good draw, but to DETECT a bad one that looks good.
3. Mode count is a useful but imperfect cross-check here: s14 (4 modes) sits in the consensus, but
   s11/s12/s16 are 1-mode and also in the consensus, so a low mode count alone does not condemn a
   run at this sample size. Cross-copy agreement remains the strongest signal.

### AUTO-COLLECTED raw results: AV-backstop / mode-budget sweep (cold, high-SNR best-fit point)

Config base: portfolio AV+GMM, adaptive components, `--force-adapt-all --internal-rotate-phase`,
`--interpolate-time True`, `--sampler-warmstart-retry-neff 5`, cold (no warm seed).
`bk` = `--portfolio-varaha-min-frac 0.25` (cap 8); `md` = `--internal-gmm-max-components 3`
(no floor); `bkmd` = both.  Judged by lnZ CONSENSUS across seeds, not n_eff.

| config | seed | n_eff | lnZ | AV final frac |
|--------|------|------:|----:|--------------:|
| bk | s10 | 1.0 | 3013.17 | 0.25 |
| bk | s12 | 1.0 | 3009.18 | 0.9900964290627214 |
| bk | s14 | 6.5 | 3013.41 | 0.25 |
| bk | s17 | 11.2 | 3014.23 | 0.25 |
| md | s10 | 12.4 | 3012.42 | 0.009900990099393974 |
| md | s12 | 5.6 | 3006.52 | 0.009900990099649775 |
| md | s14 | 123.6 | 3003.09 | 0.00990112295232892 |
| md | s17 | 22.7 | 3015.00 | 0.009900990099929107 |
| bkmd | s10 | 9.5 | 3010.88 | 0.25 |
| bkmd | s12 | 11.8 | 3011.52 | 0.25 |
| bkmd | s14 | 1.9 | 3011.12 | 0.25 |
| bkmd | s17 | 26.6 | 3013.15 | 0.25 |

Baseline for the SAME four seeds (no floor, cap 8): s10 3006.16 / s12 3013.87 / s14 3013.61 /
s17 3001.68  -> 12.2 nat spread, with the highest-n_eff copy (s17, n_eff 58) the most wrong.

lnZ spread per config (max-min over the four seeds):
- `bk`: lnZ = 3013.17 3009.18 3013.41 3014.23  -> spread 5.05 nats
- `md`: lnZ = 3012.42 3006.52 3003.09 3015.00  -> spread 11.91 nats
- `bkmd`: lnZ = 3010.88 3011.52 3011.12 3013.15  -> spread 2.27 nats

Shape-recovery merge gate: 
