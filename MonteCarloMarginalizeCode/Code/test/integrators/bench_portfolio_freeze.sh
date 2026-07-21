#!/bin/bash
#
# bench_portfolio_freeze.sh -- S250114ax iteration-0 ILE benchmark for the portfolio
# freeze-policy tuning (grace / revive / VARAHA-never-freeze).  Runs ONE integrator
# configuration to the n_eff target or the n-max cap and tees the trajectory to a log.
#
# The core args are the iteration-0 S250114ax worker args (from run_PE/ILE.sub, macros
# substituted: macroiteration->0 macroevent->0 macrongroup->1).  The sampler / warm-start /
# freeze-policy flags are supplied by the caller so one script drives every variant.
#
# Usage:
#   NAME=<tag> [GPU=2] [WARM=1] bench_portfolio_freeze.sh <sampler + freeze-policy flags...>
#     WARM=1  -> append the PE-oracle warm-start flags (cover 0.05 / inflate 1.3 / retry 5)
#     WARM=0  -> cold (no warm start)                                         [default]
#   Writes $OUTDIR/frz_<NAME>.log  and  $OUTDIR/frz_<NAME>.xml_0_.dat
#
# Examples:
#   NAME=av_warm            WARM=1 bench_portfolio_freeze.sh --sampler-method AV
#   NAME=pf_neverfreeze_warm WARM=1 bench_portfolio_freeze.sh --sampler-method portfolio --sampler-portfolio AV,GMM
#   NAME=pf_canfreeze_warm   WARM=1 bench_portfolio_freeze.sh --sampler-method portfolio --sampler-portfolio AV,GMM --portfolio-varaha-can-freeze
#
set -u

WT=/home/richard.oshaughnessy/RIFT_develUWM/src/research-projects-RIT/.claude/worktrees/rift-adaptive-integrator/.claude/worktrees/gifted-herschel-caf99c
CODE=$WT/MonteCarloMarginalizeCode/Code
PIPE=/home/richard.oshaughnessy/RIFT_roboto_paper/analyses/integrator_demos/S250114ax_pipeline
RUNPE=$PIPE/run_PE
SEED=$PIPE/pe_warm_seed.dat
OUTDIR=$RUNPE/iteration_0_ile
GPU=${GPU:-2}
WARM=${WARM:-0}
NAME=${NAME:?set NAME=<tag>}

# use MY worktree's source tree (has the freeze-policy code + CLI flags); the installed venv bin is stale
export PYTHONPATH=$CODE:${PYTHONPATH:-}
export PATH=$CODE/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=2

BIN=$CODE/bin/integrate_likelihood_extrinsic_batchmode
LOG=$OUTDIR/frz_${NAME}.log
OUT=$OUTDIR/frz_${NAME}.xml

CORE=( --save-P 0.1 --fmax 1792.0 --cache $PIPE/local.cache --event-time 1420878141.22266 \
  --channel-name H1=DCS-CALIB_STRAIN_CLEAN_AR01 --psd-file H1=$RUNPE/H1-psd.xml.gz --fmin-ifo H1=20 \
  --channel-name L1=DCS-CALIB_STRAIN_CLEAN_AR01 --psd-file L1=$RUNPE/L1-psd.xml.gz --fmin-ifo L1=20 \
  --fmin-template 20.0 --reference-freq 20 --d-max 10000 \
  --data-start-time 1420878135.222656 --data-end-time 1420878143.222656 --inv-spec-trunc-time 0 \
  --window-shape 0.1 --time-marginalization --inclination-cosine-sampler --declination-cosine-sampler \
  --n-max 4000000 --n-eff 100 --vectorized --gpu --srate 4096 --adapt-weight-exponent 0.1 --l-max 2 \
  --approx IMRPhenomD --force-xpy --internal-waveform-fd-L-frame --n-events-to-analyze 1 \
  --sim-xml $RUNPE/overlap-grid-0.xml.gz --event 0 )

WARMFLAGS=()
if [ "$WARM" = "1" ]; then
  WARMFLAGS=( --sampler-warmstart-samples $SEED --sampler-warmstart-cover-frac 0.05 \
              --sampler-warmstart-inflate 1.3 --sampler-warmstart-retry-neff 5 )
fi

cd $OUTDIR
echo "# BENCH $NAME  GPU=$GPU WARM=$WARM  $(date)" > $LOG
echo "# extra flags: $*" >> $LOG
echo "# BIN=$BIN" >> $LOG
/home/richard.oshaughnessy/RIFT_develUWM/bin/python -u "$BIN" \
  "${CORE[@]}" "${WARMFLAGS[@]}" "$@" --output-file "$OUT" >> $LOG 2>&1
echo "# EXIT $? $(date)" >> $LOG
