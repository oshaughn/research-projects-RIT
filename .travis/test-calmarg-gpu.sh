#!/usr/bin/env bash
# Calibration-marginalization CUDA gate: exercises the fused GPU kernels
# (cuda_Q_fused_calmarg[_distmarg].cu) via the fused reduction against the
# brute-force reference, including the per-realization self-term.  Requires a GPU
# + cupy; run on the GitLab gpu runner.  Any nonzero exit fails the job.
set -euo pipefail
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3)"
CODE="MonteCarloMarginalizeCode/Code"
export OMP_NUM_THREADS=1
( cd "$CODE" \
  && "$PY" -m RIFT.calmarg.test_selfterm_reduction --backend gpu \
  && "$PY" -m RIFT.calmarg.backtest_calmarg --backend gpu --n-cal 12 --methods reference,in_loop_B,in_loop_C \
  && "$PY" -m RIFT.calmarg.backtest_calmarg --backend gpu --n-cal 12 --loglikelihood distmarg --methods reference,in_loop_B,in_loop_C )
echo "calmarg GPU (CUDA) regression gate: PASS"
