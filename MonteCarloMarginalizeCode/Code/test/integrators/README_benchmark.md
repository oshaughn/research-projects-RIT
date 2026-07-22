# Integrator benchmark harness (`benchmark_integrators.py`)

A quantitative, API-matching benchmark for RIFT Monte-Carlo integrators, giving a
*testable* definition of "a better integrator".

## Targets (known truth)
- `corrgauss{3,5,8}` — scaled correlated Gaussian, one narrow dim (generalizes the
  CI target `test_mcsamplerEnsemble_extended.py` to arbitrary D).
- `rosenbrock`       — 2-D Rosenbrock, true log-evidence -5.804.
- `gaussmix{4,8}`    — superposition of Gaussians (the high-D stress test where AV
  degrades; cf. FinerNet `multigauss_direct`).

## Metrics
`bias_ln = lnI - lnZ_true`, fractional MC error `sqrt(var)/I`, RIFT `n_eff = Σp/max p`,
Kish `n_ESS = (Σw)²/Σw²`, **efficiency `n_eff/N_eval`** (headline), `N_eval`/wallclock to
reach the target `neff`, and Jensen–Shannon divergence (nats) of recovered vs analytic
1-D marginals.

## Backend
Selected by the caller's environment, exactly as production ILE: `CUDA_VISIBLE_DEVICES=""`
→ CPU/numpy; set to an idle GPU index → cupy.  Each row reports the backend actually used.

## Run
```
source ~/RIFT_develUWM/bin/activate
export PYTHONPATH=<this-worktree>/MonteCarloMarginalizeCode/Code:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
python benchmark_integrators.py --target gaussmix4 --samplers default,AC,GMM,AV --nmax 200000 --neff 1000 --json out.json
```

## Cold-vs-warm
`run(..., warm_start=callable(sampler,target))` seeds prior information before
`integrate()`, for measuring bootstrap gains (see the bootstrappable-AV work).

NOTE: the benchmarks here measure efficiency/accuracy interactively; the pre-merge REQUIREMENT is the shape-recovery gate in ../expensive_before_merging/integrators/ (see RIFT/integrators/TESTING.md).
