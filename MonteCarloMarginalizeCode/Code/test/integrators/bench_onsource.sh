#!/bin/bash
#
# bench_onsource.sh -- the HIGH-SNR BEST-FIT (on-source) point, not a trial grid point.
#
# Two DIFFERENT problems have been conflated in this study; this script exists to keep them apart:
#   * integrator tuning on a TRIAL point   : overlap-grid-0.xml.gz --event 0  (m1/m2 28.29/26.69,
#     lnLmax~1212).  Fine for A/B-ing integrator policy; says nothing about the science target.
#   * rescuing the BEST-FIT evaluation     : target_params.xml.gz (m1/m2 37.71/34.03, lnLmax~3020,
#     rho~78).  THIS is the one that has to work for a high-SNR event, and where AV stalls at
#     n_eff~1 (measured independently on cardassia CPU and reproduced here).
#
# The PE warm seed is the production extrinsic posterior FOR THIS EVENT, so warm-starting is
# physically appropriate here in a way it is not for an off-source trial point.  Coverage floor
# defaults to the bias-safe 0.5 (do NOT use 0.05/0: under-covered, biases ln Z low).
#
# Usage: NAME=<tag> [COVER=0.5] [GPU=n] bench_onsource.sh <sampler flags...>
set -u
WT=/home/richard.oshaughnessy/RIFT_develUWM/src/research-projects-RIT/.claude/worktrees/rift-adaptive-integrator/.claude/worktrees/gifted-herschel-caf99c
CODE=$WT/MonteCarloMarginalizeCode/Code
PIPE=/home/richard.oshaughnessy/RIFT_roboto_paper/analyses/integrator_demos/S250114ax_pipeline
RUNPE=$PIPE/run_PE
SEED=$PIPE/pe_warm_seed.dat
OUTDIR=$RUNPE/iteration_0_ile
NAME=${NAME:?set NAME}; COVER=${COVER:-0.5}; GPU=${GPU:-0}
NCHUNK=${NCHUNK:-10000}; NMAX=${NMAX:-4000000}; NEFF=${NEFF:-999}
WARM=${WARM:-1}

export PYTHONPATH=$CODE:${PYTHONPATH:-}
export PATH=$CODE/bin:$PATH
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2

WARMFLAGS=()
[ "$WARM" = "1" ] && WARMFLAGS=( --sampler-warmstart-samples $SEED \
    --sampler-warmstart-cover-frac $COVER --sampler-warmstart-inflate 1.3 )

LOG=$OUTDIR/os_${NAME}.log
cd $OUTDIR
echo "# ONSOURCE bench NAME=$NAME cover=$COVER warm=$WARM gpu=$GPU nmax=$NMAX $(date)" > $LOG
echo "# point: target_params.xml.gz (best-fit, m1/m2 37.71/34.03) -- NOT the trial grid point" >> $LOG

/home/richard.oshaughnessy/RIFT_develUWM/bin/python -u $CODE/bin/integrate_likelihood_extrinsic_batchmode \
  --save-P 0.1 --fmax 1792.0 --cache $PIPE/local.cache --event-time 1420878141.22266 \
  --channel-name H1=DCS-CALIB_STRAIN_CLEAN_AR01 --psd-file H1=$RUNPE/H1-psd.xml.gz --fmin-ifo H1=20 \
  --channel-name L1=DCS-CALIB_STRAIN_CLEAN_AR01 --psd-file L1=$RUNPE/L1-psd.xml.gz --fmin-ifo L1=20 \
  --fmin-template 20.0 --reference-freq 20 --d-max 10000 \
  --data-start-time 1420878135.222656 --data-end-time 1420878143.222656 --inv-spec-trunc-time 0 \
  --window-shape 0.1 --time-marginalization --inclination-cosine-sampler --declination-cosine-sampler \
  --n-max $NMAX --n-eff $NEFF --n-chunk $NCHUNK --vectorized --gpu --srate 4096 \
  --adapt-weight-exponent 0.1 --l-max 2 --approx IMRPhenomD --force-xpy \
  `# CUBIC Q_lm time interpolation instead of nearest-sample-bin. Requires the maintained NoLoop` \
  `# likelihood, i.e. the --vectorized --gpu --force-xpy combo set above. Removes a superfluous` \
  `# extrinsic non-smoothness (time quantization), which makes convergence more robust.` \
  --interpolate-time True \
  --internal-waveform-fd-L-frame --n-events-to-analyze 1 \
  --sim-xml $RUNPE/target_params.xml.gz --event 0 \
  "${WARMFLAGS[@]}" "$@" --output-file $OUTDIR/os_${NAME}.xml >> $LOG 2>&1
echo "# EXIT $? $(date)" >> $LOG
