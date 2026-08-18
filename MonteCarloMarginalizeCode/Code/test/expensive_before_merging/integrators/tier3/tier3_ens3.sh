#!/bin/bash
# Tier 3 ensemble v2. GPU ILE is NOT deterministic at fixed --seed, so this compares
# DISTRIBUTIONS. Arms interleaved within each replicate so machine drift hits both equally.
# All configs carry --vectorized --gpu => DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop
# with xpy=cupy (proved by noloop_probe.py, 20 calls, scalar path 0).
T=/local/richard.oshaughnessy/tier3; D=$T/ILE-GPU-Paper/demos
RP=/cvmfs/software.igwn.org/conda/envs/igwn/bin
W=/home/richard.oshaughnessy/rift_O4d_junior_ralph/.claude/worktrees
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1

COMMON="--n-chunk 10000 --time-marginalization --sim-xml $D/overlap-grid.xml.gz --reference-freq 100.0 --adapt-weight-exponent 0.1 --event-time 1000000014.236547946 --save-P 0.1 --cache-file $D/zero_noise.cache --fmin-template 10 --n-max 200000 --fmax 1700.0 --save-deltalnL inf --l-max 2 --n-eff 30 --approximant SEOBNRv4 --adapt-floor-level 0.1 --d-max 1000 --psd-file H1=$D/HLV-ILIGO_PSD.xml.gz --psd-file L1=$D/HLV-ILIGO_PSD.xml.gz --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --inclination-cosine-sampler --declination-cosine-sampler --data-start-time 1000000008 --data-end-time 1000000016 --inv-spec-trunc-time 0 --no-adapt-after-first --no-adapt-distance --srate 4096 --vectorized --gpu --n-events-to-analyze 1 --fairdraw-extrinsic-output"
REP="--mc-error-replicas 3 --mc-error-sigma-trigger 0.0"
DG="--export-marginal-distance-grid --internal-use-lnL"

cfg_opts () {
  case $1 in
    A)   echo "" ;;                                    # GPU linear backend, plain
    B)   echo "$REP" ;;                                # linear backend + replica POOLING
    D)   echo "--interpolate-time True" ;;             # cubic NoLoop time interpolation
    AV)  echo "--sampler-method AV  $DG $REP" ;;       # lnL family + pooling + .dgrid export
    GMM) echo "--sampler-method GMM $DG $REP" ;;
  esac
}

CSV=$T/ensemble3.csv
echo "cfg,arm,rep,rc,failed,lnL,sigma_lnL,neff,dgrid_lnL_mean,dgrid_lnL_max,secs" > $CSV
: > $T/ens3_progress.txt

N=${1:-30}
for i in $(seq 1 $N); do
 for cfg in A B D AV GMM; do
  for arm in base cand; do
    [ "$arm" = base ] && code=$W/base-v2/MonteCarloMarginalizeCode/Code || code=$W/rvs-naming/MonteCarloMarginalizeCode/Code
    o=$T/ens3/${cfg}_${arm}_$i; rm -rf $o; mkdir -p $o; cd $o
    t0=$SECONDS
    PATH=$code/bin:$RP:$PATH PYTHONPATH=$code timeout 900 $RP/python \
      $code/bin/integrate_likelihood_extrinsic_batchmode $COMMON $(cfg_opts $cfg) \
      --seed $((7000+i)) --output-file o > $o/ile.log 2>&1
    rc=$?; dt=$((SECONDS-t0))
    fa=$(grep -c 'FAILED ANALYSIS' $o/ile.log)
    vals=$($RP/python -c "
import json,os
import numpy as np
try:
    d=json.load(open('$o/o_0_integrator_status.json'))
    a=[d.get('lnL'),d.get('sigma_lnL'),d.get('neff')]
except Exception: a=[float('nan')]*3
g=[float('nan')]*2
if os.path.exists('$o/o_0_.dgrid'):
    try:
        x=np.loadtxt('$o/o_0_.dgrid')
        g=[float(np.mean(x[:,0])), float(np.max(x[:,0]))]
    except Exception: pass
print(','.join(repr(v) for v in a+g))")
    echo "$cfg,$arm,$i,$rc,$fa,$vals,$dt" >> $CSV
    [ "$rc" = 0 -a "$fa" = 0 ] && rm -f $o/*.dat
  done
 done
 echo "replicate $i done ($(date +%H:%M:%S))" >> $T/ens3_progress.txt
done
echo "ENSEMBLE3 COMPLETE" >> $T/ens3_progress.txt
