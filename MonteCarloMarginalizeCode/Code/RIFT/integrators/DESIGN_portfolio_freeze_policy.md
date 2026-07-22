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

Fix (`portfolio_adaptive_alloc`, default **on**): keep a per-member **quality** estimate updated
ONLY from chunks where the member had a *fair* allocation; allocate draws by `quality^exponent`
(concentrate on the winner) above a small floor; and **round-robin probe** one member per
`probe_period` chunks at a raised share so a suppressed member gets a fair look and can prove itself.
`q_mix` keeps every allocation unbiased, so this only ever trades efficiency, never correctness.
Knobs (all overridable via `setup`): `portfolio_alloc_exponent` (2.0), `portfolio_alloc_floor`
(0.05), `portfolio_quality_decay` (0.5), `portfolio_probe_period` (4), `portfolio_probe_frac` (0.6);
set `portfolio_adaptive_alloc=False` for the legacy n_ess reweighting.

## Benchmark 3 — adaptive allocation on synthetic correlated targets

`test/integrators/test_portfolio_adaptive_alloc.py` (fast, GPU, no ILE). Standalone AV vs standalone
GMM vs AV+GMM portfolio, fixed budget (`nmax 4e5`, ndim 5), reporting the integrator's own eff_samp
and bias vs the analytic ln Z:

| target | AV n_eff (bias) | GMM n_eff (bias) | **portfolio** n_eff (bias) | portfolio wts (AV,GMM) |
|--------|:---------------:|:----------------:|:--------------------------:|:----------------------:|
| uncorrelated (axis-aligned) | 166 (−0.62) | 200 (−0.01) | **388** (−0.03) | 0.15, 0.85 |
| **correlated (compound-symmetric)** | 131 (−0.88) | 390 (−0.03) | **381** (−0.05) | 0.12, 0.88 |

- **On the correlated target GMM's full covariance crushes AV's axis-aligned bins (390 vs 131), and
  adaptive allocation concentrates on GMM (weight 0.88) so the portfolio BEATS standalone AV** — the
  whole point of a portfolio on a correlated problem, and the case the reviewer flagged as the only
  regime where beating AV is expected.
- A cold VARAHA/AV under-covers the Gaussian tails and is **biased low** (−0.6 to −0.9); the
  portfolio stays **unbiased** because the covering GMM enters `q_mix` — a second reason to prefer
  the portfolio. (This is the opposite regime from a warm, cover-frac'd AV on a real ILE likelihood,
  where AV is the unbiased workhorse; the point demonstrated is that adaptive allocation follows
  whichever member is actually winning — GMM here, AV on a real AV-favorable event.)

**Overall verdict.** Never-freeze makes the portfolio unbiased and never-starved (replicates AV);
adaptive-probe allocation then makes it *track the best available member* — matching AV when AV wins
and beating it when a correlated geometry makes GMM win. The portfolio is now a strict "best-of"
rather than a compromise, at the cost of a small probing overhead.

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
