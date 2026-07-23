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
