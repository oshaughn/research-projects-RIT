#!/usr/bin/env bash
# measure_chunk_memory.sh -- the COUNTERWEIGHT to the chunk-size study.
#
# Enlarging the chunk is only useful if the resulting job still fits the resources production
# actually has. GPU memory scales with chunk size, and a job that needs more memory matches fewer
# slots -- which in practice means held jobs, hand-tuned RequestMemory, and idle capacity. This
# records PEAK GPU MEMORY and WALL TIME per chunk size on the REAL ILE likelihood (not the synthetic
# target), so the statistics study can be read against a real resource cost.
#
# Method: run the pinned on-source bench at a small fixed budget for each chunk size, sampling
# nvidia-smi for this PID's GPU memory throughout, and report the peak.
#
# Usage: [GPU=2] [NMAX=400000] measure_chunk_memory.sh [chunk ...]      (default 10000 40000 160000)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
BENCH="$WT/MonteCarloMarginalizeCode/Code/test/integrators/bench_onsource.sh"
RUNPE=~/RIFT_roboto_paper/analyses/integrator_demos/S250114ax_pipeline/run_PE/iteration_0_ile
GPU=${GPU:-2}
NMAX=${NMAX:-400000}
CHUNKS=${@:-"10000 40000 160000"}
OUT="$HERE/../results/chunk_memory.txt"
mkdir -p "$(dirname "$OUT")"

{
  echo "# peak GPU memory + wall time vs n-chunk (real ILE likelihood, on-source point)"
  echo "# GPU=$GPU  nmax=$NMAX  $(date)"
  printf "%-10s %14s %12s %10s\n" chunk peakMiB wall_s n_eff
} | tee "$OUT"

for nc in $CHUNKS; do
  name="mem_c${nc}"
  # sample this job's GPU memory while it runs
  ( while true; do
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, -v u="$(id -u)" '{print $1, $2}'
      sleep 2
    done ) > /tmp/memsample_$$.txt 2>/dev/null &
  SAMPLER=$!
  t0=$(date +%s)
  env GPU=$GPU WARM=0 NAME=$name NMAX=$NMAX NCHUNK=$nc bash "$BENCH" \
      --sampler-method portfolio --sampler-portfolio AV --sampler-portfolio GMM \
      --internal-gmm-adaptive-components --internal-gmm-max-components 8 \
      --force-adapt-all --internal-rotate-phase --seed 10 >/dev/null 2>&1
  t1=$(date +%s)
  kill $SAMPLER 2>/dev/null
  # peak over samples belonging to any of our python children (coarse but sufficient for scaling)
  peak=$(sort -k2 -n /tmp/memsample_$$.txt 2>/dev/null | tail -1 | awk '{print $2}')
  rm -f /tmp/memsample_$$.txt
  d=$RUNPE/os_${name}.xml_0_.dat
  ne=$([ -e "$d" ] && awk 'END{printf "%.1f", $NF}' "$d" || echo "-")
  printf "%-10s %14s %12s %10s\n" "$nc" "${peak:-?}" "$((t1-t0))" "$ne" | tee -a "$OUT"
done
echo "# wrote $OUT"
