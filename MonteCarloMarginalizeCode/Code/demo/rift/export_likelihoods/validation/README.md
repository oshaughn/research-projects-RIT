# GP-vs-standard-RIFT validation ladder

Is the exported GP likelihood accurate enough **for inference**, vs a standard RIFT
posterior? We answer this one natural parameter-combination at a time, easiest
first, so a failure tells you *which ingredient* broke it.

## Division of labor (important)

- **You run the full RIFT pipeline** for a case, to convergence, **elsewhere**
  (cluster / real full run). A full run *adaptively samples* the likelihood. A
  hand-picked fixed grid does **not** — and then you have no idea whether the
  posterior is reliable. So this harness deliberately does **not** run ILE or
  fabricate a grid.
- **This harness only analyzes a completed run.** Given the run's converged
  likelihood grid and its standard posterior, it fits the GP to that grid, samples
  with mu-frame-preconditioned NUTS, and compares marginals (JS, bits).

See `CASES.md` for the per-case full-run recipe (what you launch).

## Workflow

```bash
# 1. (elsewhere) full RIFT run for the case -> converged grid + standard posterior
#    (see CASES.md)

# 2. (here) analyze that run
cd demo/rift/export_likelihoods/validation
GRID=/path/to/run/all_dgrid.dat \
STD=/path/to/run/standard_posterior.dat \
  source config.sh && ./analyze_case.sh mcq_dL
```

`analyze_case.sh` fits the GP to `GRID`, NUTS-samples it, and prints the JS
divergence of each 1D marginal vs `STD`. Prior ranges default to the grid extent
(`--auto-range`); pass `RANGES="m1:[..] m2:[..] ..."` to match the run's prior
exactly.

## The ladder

| case (`analyze_case.sh` arg) | parameters | spin | tides |
| --- | --- | --- | --- |
| `mcq_dL` | mc, q, dL | none | none |
| `mcq_aligned_dL` | mc, q, s1z, s2z, dL | aligned | none |
| *(precessing)* | + precessing spin, dL | precessing | none |
| `mcq_aligned_tides` | mc, q, aligned spin, tides | aligned | yes |
| *(full)* | mass + spin + tides | aligned | yes |

Build *up* the ladder: the full mass+spin+tides case is the hard one (the GW170817
run analyzed in the parent `demo/rift/export_likelihoods/`); the earlier rungs
isolate where the GP first struggles. Precessing-spin needs precessing
fit-coordinate transforms that do not exist yet (`coordinates.py` is aligned/BNS
only) — noted in `CASES.md`.

## Files

| file | role |
| --- | --- |
| `config.sh` | env + the `GRID`/`STD` inputs you set from your completed run (source first) |
| `CASES.md` | the ladder + the full-run recipe to launch for each case |
| `analyze_case.sh` | consume a completed run: GP-from-grid → NUTS → JS vs standard |
| `gp_from_grid.py` | generic: fit quadgp to a grid + nuts-mu → posterior `.dat` |
| `compare_marginals.py` | JS (bits) per marginal; derives mc/q/eta/LambdaTilde from primaries |

## What does NOT belong here

Toy/synthetic likelihoods and raw GP↔RF "L2 over the posterior box" comparisons
stay **out of the RIFT tree** (a separate, uncommitted workspace). No one needs a
toy model in-tree, and a likelihood-space L2 always shows *some* difference without
telling you whether it matters for inference. This directory is only the real,
full-run-grounded head-to-head.
