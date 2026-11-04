#!/usr/bin/env bash
# Chunk-size study driver. Runs BOTH designs, because they answer different questions:
#   A) FIXED BUDGET  -- same cost per run, so steps = nmax/n_chunk falls as the chunk grows.
#      This is the PRODUCTION question: "at the budget I already pay, should the chunk be bigger?"
#   B) FIXED STEPS   -- nmax = n_chunk*steps, equal adaptation opportunities, cost grows with chunk.
#      This is the MECHANISM question: "do richer per-step statistics help, independent of steps?"
# CPU-only and truth-known (MixtureTarget.true_lnZ), so copies are cheap and we measure real BIAS.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/../results"
export PATH=/home/richard.oshaughnessy/RIFT_develUWM/bin:$PATH
export PYTHONPATH="$WT/MonteCarloMarginalizeCode/Code"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS=${JOBS:-32}
COPIES=${COPIES:-24}
mkdir -p "$OUT"
echo "############ A) FIXED BUDGET (same cost) ############"
python -u "$HERE/chunk_study.py" --snrs 20,40,80,160 --chunks 10000,40000,160000 \
    --nmax 2000000 --copies $COPIES --jobs $JOBS --kinds AV,portfolio \
    --json "$OUT/chunk_fixed_budget.json" 2>&1 | grep -vE "Adding parameter|Adapting|^ *[0-9]+ "
echo
echo "############ B) FIXED STEPS (isolate per-step statistics) ############"
python -u "$HERE/chunk_study.py" --snrs 20,40,80,160 --chunks 10000,40000,160000 \
    --steps 50 --copies $COPIES --jobs $JOBS --kinds AV,portfolio \
    --json "$OUT/chunk_fixed_steps.json" 2>&1 | grep -vE "Adding parameter|Adapting|^ *[0-9]+ "
echo "CHUNK STUDY DONE"
