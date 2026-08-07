#!/bin/bash
# Standard merge-gate invocation of the shape-recovery suite.
#
# A FAIL FROM THIS SCRIPT IS NOT A VERDICT.  Every gate threshold (n_eff >= 100, JS, pull, width)
# is a hard cut on a stochastic quantity, so any cell sitting near a threshold flips on
# realization alone.  Before treating a blocking regression as real, re-test it at fresh seeds:
#
#     compare_shape_results.py base.json cand.json \
#         --confirm-base-checkout DIR --confirm-cand-checkout DIR --confirm-repeats 5
#
# With those flags the comparison ENFORCES confirmation: blocking rows are re-tested and the exit
# code is the confirmed verdict.  Without them it still exits 1 on a blocking row, but says so
# explicitly rather than implying the row was confirmed.  (confirm_regressions.py also runs
# standalone against an existing pair of JSONs.)
#
# It re-runs only the disputed cells, in BOTH arms, at several new run seeds, and blocks only if
# the candidate is worse in a majority.  A candidate that produces NO record where the base did
# counts against the candidate, and too few usable pairs is INCONCLUSIVE (exit 1), never a clear.  Worked example: `GMM mix_d6_n3_s303` was reported as a
# blocking REGRESSION in two consecutive full runs (base 119, candidate 66) and looked
# reproducible -- but at 5 fresh seeds the two arms were BIT-IDENTICAL (93/93, 80/80, 119/119,
# 95/95, 96/96) and 4 of the 5 starved.  The cell simply sits on the n_eff=100 floor; its PASS at
# the default seed was the lucky draw, and the apparent regression was an artifact of where the
# job landed in the worker pool.
#
# Do NOT "fix" this by seeding the samplers deterministically.  Independent copies that localize
# differently are the working detector for support/mode-collapse failures; pinning every fit to
# one seed silences it, and makes N production copies no better than one.
#
#   ./run_shape_recovery.sh /path/to/checkout results.json [extra args...]
#
# Runs CPU-only (deterministic; also exercises the cupy-installed-but-no-GPU
# worker configuration that has repeatedly bitten production).  Use --jobs to
# parallelize across cores.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CHECKOUT=${1:?usage: run_shape_recovery.sh /path/to/checkout results.json [extra args]}
OUT=${2:?need output json path}
shift 2

export PYTHONPATH="${CHECKOUT}/MonteCarloMarginalizeCode/Code:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# NOT `python`: several IGWN/conda environments (and this submit host) provide only python3,
# where a bare `python` makes the whole gate exit 127 before it starts.
exec "${PYTHON:-python3}" "${HERE}/shape_recovery.py" --preset standard \
    --jobs "${SHAPE_JOBS:-8}" --warm-cases auto --json "${OUT}" "$@"
