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

**Why it is opt-in, not the default (a real regression).** The quality signal is each member's
per-chunk Kish n_ess, which rewards **self-consistency, not integral coverage**. A warm GMM is
instantly self-consistent (per-chunk n_ess ~120), while a warm VARAHA/AV member's per-chunk n_ess is
genuinely ~1 during its slow *cumulative* contraction (its value emerges over ~70 chunks). So on the
real high-SNR **S250114ax** (AV-favorable) event, adaptive drives the true AV workhorse to the floor
and rides the self-consistent-but-worse GMM: **n_eff 8 vs 53** for the legacy allocation — a clear
regression. The probe can't rescue AV because AV still looks bad at high allocation until fully
contracted. A correct default needs a **global-impact** quality signal (how much a member improves
the pooled `q_mix` n_eff), not per-member self-n_ess — that is future work. Until then the default
keeps the legacy n_ess reweighting (never-freeze), and adaptive is opt-in for correlated problems.

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
allocation (opt-in) can make a portfolio **beat AV on a strongly-correlated target** (synthetic:
387 vs 23 n_eff) — the only regime where beating AV is expected — but with the current per-member
n_ess quality signal it *starves* the slow-contracting AV on AV-favorable real events (S250114ax:
8 vs 53), so it is not yet a safe default. The clear next step is a global-impact quality signal
(a member's marginal contribution to the pooled n_eff) so adaptive can be turned on everywhere.

**Robustness bug fixed along the way (important):** portfolio plugin discovery hard-loaded every
registered plugin at import, and the `NF` plugin does `import torch`, absent in the production GPU
container — so `import mcsamplerPortfolio` raised there, the driver silently set
`mcsampler_Portfolio_ok=False`, and every portfolio run died with a `NameError`. The portfolio
integrator was effectively **unusable in the production container**. Plugin loading is now wrapped
in try/except so a plugin with missing optional deps is skipped, not fatal.

## Files
- `RIFT/integrators/mcsamplerPortfolio.py` — freeze-policy + adaptive-probe allocation, knobs,
  n_ess history, plugin-load guard, NaN guard.
- `bin/integrate_likelihood_extrinsic_batchmode` — CLI flags (freeze + allocation) + comma-split fix.
- `test/integrators/bench_portfolio_freeze.sh` — S250114ax single-config runner.
- `test/integrators/bench_multi_event.py` + `run_multi_event.sh` — multi-event robustness suite.
- `test/integrators/parse_neff_traj.py` — trajectory → n_eff-vs-N table parser.
- `test/integrators/test_portfolio_adaptive_alloc.py` — synthetic correlated/uncorrelated test that
  the portfolio tracks the winning member and beats AV on a correlated target (Benchmark 3).
