#!/bin/bash
# Standard merge-gate invocation of the shape-recovery suite.
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

exec python "${HERE}/shape_recovery.py" --preset standard --jobs "${SHAPE_JOBS:-8}" \
    --json "${OUT}" "$@"
