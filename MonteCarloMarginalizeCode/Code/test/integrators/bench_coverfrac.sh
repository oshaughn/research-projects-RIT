#!/bin/bash
#
# bench_coverfrac.sh -- answer the cardassia cover_frac ask on the REFERENCE point/backend.
#
# Runs the S250114ax **iteration-0 worker** point (overlap-grid-0.xml.gz, --event 0) -- the point the
# reference matrix in BREADCRUMB_av_neff_reproduction.md actually used (lnLmax~1212,
# sqrt(2 lnLmax)~49.2) -- on GPU, varying ONLY the warm-start coverage floor, with the auto-rescue
# DISABLED so nothing masks the seed's own behaviour.
#
# NB this is deliberately NOT the "loud on-source coinc point" (lnLmax~3020): that is a different,
# much louder target, and mixing the two is what made the reference look unreproducible.
#
# Usage: NAME=<tag> COVER=<frac> [GPU=n] bench_coverfrac.sh [extra ILE flags...]
set -u
WT=/home/richard.oshaughnessy/RIFT_develUWM/src/research-projects-RIT/.claude/worktrees/rift-adaptive-integrator/.claude/worktrees/gifted-herschel-caf99c
CODE=$WT/MonteCarloMarginalizeCode/Code
PIPE=/home/richard.oshaughnessy/RIFT_roboto_paper/analyses/integrator_demos/S250114ax_pipeline
RUNPE=$PIPE/run_PE
SEED=$PIPE/pe_warm_seed.dat
OUTDIR=$RUNPE/iteration_0_ile
NAME=${NAME:?set NAME}; COVER=${COVER:?set COVER}; GPU=${GPU:-0}
NCHUNK=${NCHUNK:-10000}; NMAX=${NMAX:-4000000}; NEFF=${NEFF:-999}

export PYTHONPATH=$CODE:${PYTHONPATH:-}
export PATH=$CODE/bin:$PATH
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2

LOG=$OUTDIR/cf_${NAME}.log
cd $OUTDIR
echo "# COVERFRAC bench NAME=$NAME cover=$COVER gpu=$GPU nchunk=$NCHUNK nmax=$NMAX (rescue OFF) $(date)" > $LOG

/home/richard.oshaughnessy/RIFT_develUWM/bin/python -u $CODE/bin/integrate_likelihood_extrinsic_batchmode \
  --save-P 0.1 --fmax 1792.0 --cache $PIPE/local.cache --event-time 1420878141.22266 \
  --channel-name H1=DCS-CALIB_STRAIN_CLEAN_AR01 --psd-file H1=$RUNPE/H1-psd.xml.gz --fmin-ifo H1=20 \
  --channel-name L1=DCS-CALIB_STRAIN_CLEAN_AR01 --psd-file L1=$RUNPE/L1-psd.xml.gz --fmin-ifo L1=20 \
  --fmin-template 20.0 --reference-freq 20 --d-max 10000 \
  --data-start-time 1420878135.222656 --data-end-time 1420878143.222656 --inv-spec-trunc-time 0 \
  --window-shape 0.1 --time-marginalization --inclination-cosine-sampler --declination-cosine-sampler \
  --n-max $NMAX --n-eff $NEFF --n-chunk $NCHUNK --vectorized --gpu --srate 4096 \
  --adapt-weight-exponent 0.1 --l-max 2 --approx IMRPhenomD --force-xpy \
  --internal-waveform-fd-L-frame --n-events-to-analyze 1 \
  --sim-xml $RUNPE/overlap-grid-0.xml.gz --event 0 \
  --sampler-method AV \
  --sampler-warmstart-samples $SEED --sampler-warmstart-cover-frac $COVER --sampler-warmstart-inflate 1.3 \
  "$@" --output-file $OUTDIR/cf_${NAME}.xml >> $LOG 2>&1
echo "# EXIT $? $(date)" >> $LOG
