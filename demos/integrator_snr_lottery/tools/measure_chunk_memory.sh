#!/usr/bin/env bash
# measure_chunk_memory.sh -- price the SNR-scaled chunk size, on BOTH memory surfaces.
#
# WHY TWO NUMBERS.  GPU memory and host memory are DIFFERENT control surfaces, and condor's
# RequestMemory governs the HOST, not the GPU.  Enlarging --n-chunk grows the device-side sample
# arrays; whether it moves host RSS at all is a separate question.  Reporting only one (as an earlier
# version of this script did) cannot answer "must I raise RequestMemory?".
#
# Relevant prior: RIFT extrinsic jobs have been measured requesting 35-105x the host RAM they
# actually use.  So the likely correct conclusion is "raise the chunk, leave RequestMemory alone" --
# but that must be MEASURED, not assumed, and changing a production resource setting without evidence
# is exactly the kind of unmotivated churn to avoid.
#
# METHOD.  Run the pinned on-source ILE bench per chunk size, sample every 2 s:
#   host : max over the job's process TREE of VmHWM (peak RSS) from /proc/<pid>/status  [KiB -> MiB]
#   gpu  : nvidia-smi --query-compute-apps, FILTERED TO THIS JOB'S PIDs (the earlier version took the
#          max over all compute apps and so reported other users' jobs -- identical 37946 MiB for
#          every chunk, which is what exposed the bug)
#
# Usage: [GPU=2] [NMAX=400000] [WARM=1] measure_chunk_memory.sh [chunk ...]   (default 10000 40000 160000)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
BENCH="$WT/MonteCarloMarginalizeCode/Code/test/integrators/bench_onsource.sh"
RUNPE=~/RIFT_roboto_paper/analyses/integrator_demos/S250114ax_pipeline/run_PE/iteration_0_ile
GPU=${GPU:-2}
NMAX=${NMAX:-400000}
WARM=${WARM:-1}          # warm so the run does real work rather than bailing early
CHUNKS=${@:-"10000 40000 160000"}
OUT="$HERE/../results/chunk_memory.txt"
mkdir -p "$(dirname "$OUT")"

{
  echo "# chunk-size resource cost, REAL ILE likelihood, on-source point"
  echo "# GPU=$GPU nmax=$NMAX warm=$WARM  $(date)"
  echo "# host_MiB = peak VmHWM over the job's process tree (what RequestMemory governs)"
  echo "# gpu_MiB  = peak nvidia-smi used_memory for THIS job's pids only"
  printf "%-9s %10s %10s %9s %9s\n" chunk host_MiB gpu_MiB wall_s n_eff
} | tee "$OUT"

descendants() {  # pid -> pid and all descendants
  local p=$1; echo "$p"
  local kids; kids=$(pgrep -P "$p" 2>/dev/null)
  local k; for k in $kids; do descendants "$k"; done
}

for nc in $CHUNKS; do
  name="mem_c${nc}"
  t0=$(date +%s)
  env GPU=$GPU WARM=$WARM NAME=$name NMAX=$NMAX NCHUNK=$nc bash "$BENCH" \
      --sampler-method portfolio --sampler-portfolio AV --sampler-portfolio GMM \
      --internal-gmm-adaptive-components --internal-gmm-max-components 8 \
      --force-adapt-all --internal-rotate-phase --seed 10 >/dev/null 2>&1 &
  JOB=$!
  host_peak=0; gpu_peak=0
  while kill -0 $JOB 2>/dev/null; do
    pids=$(descendants $JOB 2>/dev/null | sort -u)
    # host: peak RSS over the tree
    for p in $pids; do
      v=$(awk '/VmHWM/{print $2}' /proc/$p/status 2>/dev/null)
      [ -n "${v:-}" ] && [ "$v" -gt "$host_peak" ] 2>/dev/null && host_peak=$v
    done
    # gpu: only rows whose pid is in our tree
    while read -r gp gm; do
      case " $pids " in *" $gp "*)
        [ -n "$gm" ] && [ "$gm" -gt "$gpu_peak" ] 2>/dev/null && gpu_peak=$gm ;;
      esac
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr -d ',')
    sleep 2
  done
  wait $JOB 2>/dev/null
  t1=$(date +%s)
  d=$RUNPE/os_${name}.xml_0_.dat
  ne=$([ -e "$d" ] && awk 'END{printf "%.1f", $NF}' "$d" || echo "-")
  printf "%-9s %10s %10s %9s %9s\n" "$nc" "$((host_peak/1024))" "${gpu_peak:-0}" "$((t1-t0))" "$ne" | tee -a "$OUT"
done
echo "# wrote $OUT"
