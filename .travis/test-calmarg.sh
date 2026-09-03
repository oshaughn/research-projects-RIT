#!/usr/bin/env bash
# Calibration-marginalization regression gate (CPU).  Covers the in-loop calmarg
# reduction and the per-realization self-term fix: precompute time-alignment +
# identity-cal cross terms, the loop/fused reduction vs a brute-force reference
# (incl. n_cal==1), the low-rank SVD self-term basis vs a direct band integral, and
# the backtest of the cal reduction (default + distance-marginalization helpers),
# and the driver's calibration OPTION-COMPATIBILITY gate (which flag combinations
# are refused at startup instead of silently degrading to the zero-calibration
# likelihood, and which must still be accepted).
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

# Option-compatibility gate: which calibration option combinations the ILE driver
# REFUSES at startup, and -- the load-bearing half -- which it must still ACCEPT.
# pytest here, not a bare module run: these are parametrized accept/refuse pairs plus
# real subprocess invocations of the driver, and `set -e` turns pytest's exit 5 ("no
# tests ran") into a job failure rather than a silent green.  A collection floor is
# asserted for the same reason: a file that stops collecting is a lost gate.
_n=$( cd "$CODE" && "$PY" -m pytest -q --collect-only "RIFT/calmarg/test_option_compat.py" 2>/dev/null | grep -c '::' || true )
if [ "${_n:-0}" -lt 20 ]; then
  echo "test-calmarg.sh: option-compat suite collected ${_n:-0} tests, expected >= 20." >&2
  echo "  Tests were renamed, removed, or the file stopped importing.  Fix it; as is," >&2
  echo "  the gate covers less than it claims." >&2
  exit 1
fi
( cd "$CODE" && PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pytest -q "RIFT/calmarg/test_option_compat.py" )

echo "calmarg CPU regression gate: PASS"
