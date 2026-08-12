#!/usr/bin/env bash
# Calibration-marginalization regression gate (CPU).  Covers the in-loop calmarg
# reduction and the per-realization self-term fix: precompute time-alignment +
# identity-cal cross terms, the loop/fused reduction vs a brute-force reference
# (incl. n_cal==1), the low-rank SVD self-term basis vs a direct band integral, and
# the backtest of the cal reduction (default + distance-marginalization helpers).
# Any nonzero exit fails the job (set -e).  GPU/CUDA paths are exercised separately
# on hardware; here every check runs on the numpy backend.
set -euo pipefail

PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3)"
CODE="MonteCarloMarginalizeCode/Code"
export OMP_NUM_THREADS=1

# precompute alignment + identity-cal self-term cross terms == baseline
"$PY" "$CODE/RIFT/calmarg/test_precompute_alignment.py"

# reduction + self-term basis + backtest run as modules from the code root
( cd "$CODE" \
  && "$PY" -m RIFT.calmarg.test_selfterm_basis \
  && "$PY" -m RIFT.calmarg.test_selfterm_reduction --backend cpu \
  && "$PY" -m RIFT.calmarg.test_calmarg_reduction \
  && "$PY" -m RIFT.calmarg.backtest_calmarg --backend cpu --n-cal 8 --methods reference,in_loop_B \
  && "$PY" -m RIFT.calmarg.backtest_calmarg --backend cpu --n-cal 8 --loglikelihood distmarg --methods reference,in_loop_B )

echo "calmarg CPU regression gate: PASS"
