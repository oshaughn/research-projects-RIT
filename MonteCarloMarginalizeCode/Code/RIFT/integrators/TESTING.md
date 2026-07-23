# Before merging changes to this directory

**Any PR that touches the integrators must pass the posterior SHAPE-recovery
merge gate**, not just the fast CI integral test:

    MonteCarloMarginalizeCode/Code/test/expensive_before_merging/integrators/

Integrals are easy — importance-sampling estimates of Z are unbiased under
weak conditions — while the recovered posterior *shape* (the weighted sample
cloud that CIP/fairdraws consume) can be confidently, silently wrong.  Real
examples caught by this gate: a GMM run with evidence correct to +0.009 nats
whose marginals had JS=0.29 and widths 2.8-3.8x too broad; and a GPU-port
change whose swallowed per-refit exceptions returned n_eff~1 with no error
flag (rift_O4c -> rift_O4d GMM regression, bisected 2026-07).

Quick recipe (see the suite README for details; ~10 min per branch on a
quiet head node):

    source ~/RIFT_develUWM/bin/activate           # or equivalent env
    cd .../test/expensive_before_merging/integrators
    SHAPE_JOBS=12 OMP_NUM_THREADS=1 ./run_shape_recovery.sh <base-checkout> base.json
    SHAPE_JOBS=12 OMP_NUM_THREADS=1 ./run_shape_recovery.sh <pr-checkout>   pr.json
    python compare_shape_results.py base.json pr.json   # exit 1 = merge-blocking

Notes for agents:
- The suite is self-contained: it runs against ANY checkout via PYTHONPATH
  (the two runs above use the SAME suite files against different checkouts).
- Run one suite at a time: LDG head nodes have RLIMIT_NPROC=500.
- CPU-only by design (CUDA_VISIBLE_DEVICES="") — this also exercises the
  cupy-installed-but-no-GPU worker configuration that has repeatedly bitten
  production (module-level cupy selection without a device probe).
- "STARVED" rows (n_eff < 100) are not absolute failures — high-D mixtures
  legitimately exhaust production budgets — but base-healthy -> starved IS a
  blocking regression.
- If your change is behind an opt-in flag, the default-path gate will show
  bitwise-identical results; you must ALSO probe the flag ON (use
  shape_recovery.py as a library; see its docstring).
