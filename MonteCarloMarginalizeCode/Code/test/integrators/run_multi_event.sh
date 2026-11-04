#!/bin/bash
# Orchestrate the multi-event robustness suite: for each event run standalone AV and the
# never-freeze AV+GMM portfolio (in the event's container), then dump an lnZ/n_eff comparison.
set -u
CODE=/home/richard.oshaughnessy/RIFT_develUWM/src/research-projects-RIT/.claude/worktrees/rift-adaptive-integrator/.claude/worktrees/gifted-herschel-caf99c/MonteCarloMarginalizeCode/Code
PY=/home/richard.oshaughnessy/RIFT_develUWM/bin/python
MEV=$CODE/test/integrators/bench_multi_event.py
EVENTS="S231026ab S240426s S240513ei S240601aj S240703ad"
export CONTAINER=1 NEFF=${NEFF:-30} NMAX=${NMAX:-800000}
GPUS=(1 3)
MAXJOBS=4

launch() {  # event tag gpu sampler...
  local ev=$1 tag=$2 gpu=$3; shift 3
  GPU=$gpu $PY $MEV run $ev $tag "$@" >/dev/null 2>&1 &
}

i=0
for ev in $EVENTS; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  launch $ev av_$ev $g AV
  launch $ev pf_$ev $g portfolio --sampler-portfolio AV,GMM
  i=$((i+1))
  # throttle: wait if too many background jobs
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 15; done
done
wait
echo "=== ALL MULTI-EVENT RUNS DONE ==="
for ev in $EVENTS; do
  $PY $MEV read $ev av_$ev
  $PY $MEV read $ev pf_$ev
done
