#!/bin/bash
# One CIP + RF + AV intrinsic-posterior job for the GW170817 benchmark.
# Arg $1 = job index (Condor $(Process)); writes cip_rf_<idx>.xml.gz in $OUTDIR.
#
# No --seed in CIP -> each launch uses an independent random stream, so running
# this N times accumulates independent statistics for the benchmark posterior.
#
# Env (override via the submit file's `environment`):
#   RIFT_CODE  path to a working MonteCarloMarginalizeCode/Code (for bin/ + RIFT)
#   PYTHON     python with RIFT + sklearn + lal (e.g. the gwkokab env)
#   NET        the ILE .net input
#   OUTDIR     where to write outputs
set -euo pipefail
IDX="${1:-0}"
RIFT_CODE="${RIFT_CODE:-/home/oshaughn/research-projects-RIT/MonteCarloMarginalizeCode/Code}"
PYTHON="${PYTHON:-/home/oshaughn/.conda/envs/gwkokab/bin/python}"
NET="${NET:-/home/oshaughn/all.net}"
OUTDIR="${OUTDIR:-$(pwd)}"

export PYTHONPATH="${RIFT_CODE}:${PYTHONPATH:-}"
# Keep each job's CPU/memory footprint small and predictable across 10 parallel jobs.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

exec "${PYTHON}" "${RIFT_CODE}/bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py" \
  --fname "${NET}" --fit-method rf --sampler-method AV \
  --parameter delta_mc --parameter-implied mu1 --parameter-implied mu2 \
  --parameter-implied LambdaTilde --parameter-implied DeltaLambdaTilde \
  --parameter-nofit mc --parameter-nofit s1z --parameter-nofit s2z \
  --parameter-nofit lambda1 --parameter-nofit lambda2 \
  --mc-range '[1.196,1.199]' --chi-max 0.05 --input-tides --cap-points 30000 \
  --n-eff 3000 --n-output-samples 5000 --no-plots \
  --fname-output-samples "cip_rf_${IDX}" \
  --fname-output-integral "cip_rf_int_${IDX}"
