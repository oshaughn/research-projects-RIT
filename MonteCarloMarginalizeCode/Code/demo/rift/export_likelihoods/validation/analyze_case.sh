#!/usr/bin/env bash
#
# Analyze a COMPLETED full RIFT run: fit the GP to the run's converged likelihood
# grid, sample it with NUTS, and compare its 1D marginals to the run's standard
# posterior (JS, bits).  This does NOT run any pipeline -- you produce GRID and STD
# with a full run elsewhere (see CASES.md), then point this at them.
#
# Usage:
#   GRID=/path/all_dgrid.dat STD=/path/joint_posterior.dat \
#     source config.sh && ./analyze_case.sh <case>
#
# <case> selects the parameter set + comparison marginals (see CASES.md):
#   mcq_dL            : m1 m2 dist            (compare mc q dist)
#   mcq_aligned_dL    : m1 m2 s1z s2z dist    (compare mc q s1z s2z dist)
#   mcq_aligned_tides : m1 m2 s1z s2z lambda1 lambda2   (compare mc q s1z s2z LambdaTilde)
# Prior ranges default to the grid extent (--auto-range); override per parameter
# with RANGES="m1:[24,64] m2:[15,42] ..." to match the run's prior exactly.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "${CODE_DIR:-}" ] || source "${HERE}/config.sh"

CASE="${1:-}"
case "${CASE}" in
  mcq_dL)            FIT_PARAMS=(m1 m2 dist);                       CMP_PARAMS=(mc q dist) ;;
  mcq_aligned_dL)    FIT_PARAMS=(m1 m2 s1z s2z dist);               CMP_PARAMS=(mc q s1z s2z dist) ;;
  mcq_aligned_tides) FIT_PARAMS=(m1 m2 s1z s2z lambda1 lambda2);    CMP_PARAMS=(mc q s1z s2z LambdaTilde) ;;
  *) echo "usage: ./analyze_case.sh {mcq_dL|mcq_aligned_dL|mcq_aligned_tides}"; exit 2 ;;
esac

[ -s "${GRID:-/nonexistent}" ] || { echo "set GRID to your full run's consolidated lnL grid (.dat)"; exit 2; }
[ -s "${STD:-/nonexistent}"  ] || { echo "set STD to your full run's standard posterior (.dat)"; exit 2; }

WORK="${OUTDIR}/${CASE}"; mkdir -p "${WORK}"
GP_OUT="${WORK}/gp_posterior.dat"

# Build --param / --range args.
param_args=(); range_args=()
for p in "${FIT_PARAMS[@]}"; do param_args+=(--param "$p"); done
for kv in ${RANGES:-}; do range_args+=(--range "$kv"); done   # optional explicit ranges

echo "=== fit GP to the run's grid + NUTS (case ${CASE}; params ${FIT_PARAMS[*]}) ==="
"${PY}" "${HERE}/gp_from_grid.py" \
  --grid "${GRID}" "${param_args[@]}" "${range_args[@]}" --auto-range \
  --lnL-offset "${LNL_OFFSET:-30}" \
  --fit-method "${FIT_METHOD:-quadgp}" --quadgp-residual "${QUADGP_RESIDUAL:-svgp}" \
  --n-features "${N_FEATURES:-600}" --n-opt-steps "${N_OPT_STEPS:-200}" \
  --num-warmup "${NUM_WARMUP:-800}" --num-samples "${NUM_SAMPLES:-4000}" \
  --num-chains "${NUM_CHAINS:-2}" \
  --out "${GP_OUT}"

echo "=== head-to-head marginals (JS bits): GP vs standard ==="
cmp_args=(); for p in "${CMP_PARAMS[@]}"; do cmp_args+=(--param "$p"); done
"${PY}" "${HERE}/compare_marginals.py" --standard "${STD}" --gp "${GP_OUT}" "${cmp_args[@]}"

echo
echo "GP posterior: ${GP_OUT}"
