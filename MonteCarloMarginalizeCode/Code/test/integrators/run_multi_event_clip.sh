#!/bin/bash
# Safety check: does ADAPTATION-only weight clipping (C=1) hurt TYPICAL events?  Runs the portfolio
# with --portfolio-weight-clip 1.0 on each event (in its container) and prints lnZ/n_eff, to compare
# against the earlier no-clip portfolio results (clipping must leave lnZ unbiased and not hurt n_eff).
set -u
CODE=/home/richard.oshaughnessy/RIFT_develUWM/src/research-projects-RIT/.claude/worktrees/rift-adaptive-integrator/.claude/worktrees/gifted-herschel-caf99c/MonteCarloMarginalizeCode/Code
PY=/home/richard.oshaughnessy/RIFT_develUWM/bin/python
MEV=$CODE/test/integrators/bench_multi_event.py
EVENTS="S231026ab S240426s S240513ei S240703ad"
export CONTAINER=1 NEFF=${NEFF:-30} NMAX=${NMAX:-800000}
GPUS=(0 1)
MAXJOBS=2
i=0
for ev in $EVENTS; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  GPU=$g $PY $MEV run $ev pfclip_$ev portfolio --sampler-portfolio AV,GMM --portfolio-weight-clip 1.0 >/dev/null 2>&1 &
  i=$((i+1))
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 15; done
done
wait
echo "=== CLIP MULTI-EVENT DONE ==="
for ev in $EVENTS; do $PY $MEV read $ev pfclip_$ev; done
