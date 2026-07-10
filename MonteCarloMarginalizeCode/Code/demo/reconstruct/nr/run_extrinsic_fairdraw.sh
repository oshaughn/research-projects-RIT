#!/usr/bin/env bash
#
# run_extrinsic_fairdraw.sh  CONFIG.sh  SUFFIX
#
# Run ONE RIFT ILE extrinsic job for a fixed NR simulation, configured to
# emit fair-draw posterior samples each carrying its OWN coalescence time,
# then extract them to a compact .npz.
#
# The flag choices below are the whole point -- see README.md "What went wrong".
# Correct, load-bearing flags:
#   --time-marginalization            : correct absolute lnL scale (peak lnL = SNR^2/2)
#   --fairdraw-extrinsic-output       : output rows are fair posterior draws
#   --resample-time-marginalization   : draw a geocent time per output row from that
#                                       row's own lnL(t), coherent with its phase
#                                       (=> reconstruction coheres with NO alignment)
#   --no-adapt-after-first            : keep the sampler stable; without it a mid-run
#                                       adaptation reset crushes the fair-draw yield
#   --d-min/--d-max                   : bracket the distance posterior; excludes the
#                                       <~2 Mpc region where the likelihood overflows to NaN
#   (do NOT use --maximize-only, and do NOT use --no-adapt-distance)
#
# CONFIG.sh must export: PYBIN ILEBIN EXTRACT_PY WORKDIR EVENT_TIME SRATE FMAX FLOW
#   NR_GROUP NR_PARAM SIM_XML D_MIN D_MAX N_MAX N_EFF DATA_ARGS
# where DATA_ARGS is the detector/data/psd block, e.g.
#   DATA_ARGS="--cache local.cache \
#     --channel-name H1=... --psd-file H1=H1-psd.xml.gz --fmin-ifo H1=20 \
#     --channel-name L1=... --psd-file L1=L1-psd.xml.gz --fmin-ifo L1=20"
set -e
CONFIG=$1; SUF=$2
[ -z "$SUF" ] && { echo "usage: $0 CONFIG.sh SUFFIX"; exit 1; }
source "$CONFIG"
cd "$WORKDIR"
OUT=ile_fd_${SUF}

# thread caps so several of these can (if needed) coexist without pthread EAGAIN
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}

"$ILEBIN" \
  --save-P 0.1 --save-deltalnL 20 --fmax "$FMAX" --srate "$SRATE" \
  --event-time "$EVENT_TIME" $DATA_ARGS \
  --fmin-template "$FLOW" --reference-freq "$FLOW" --inv-spec-trunc-time 0 --window-shape 0.2 \
  --time-marginalization --fairdraw-extrinsic-output --resample-time-marginalization \
  --inclination-cosine-sampler --declination-cosine-sampler \
  --n-max "$N_MAX" --n-eff "$N_EFF" --vectorized --gpu \
  --no-adapt-after-first --adapt-weight-exponent 0.1 --l-max 4 \
  --force-reset-all --sampler-method adaptive_cartesian_gpu --force-xpy \
  --d-min "$D_MIN" --d-max "$D_MAX" --n-events-to-analyze 1 \
  --nr-lookup --nr-lookup-group "$NR_GROUP" --nr-group "$NR_GROUP" \
  --nr-param "$NR_PARAM" --nr-use-provided-strain --save-eccentricity \
  --sim-xml "$SIM_XML" --save-samples --output-file "$OUT"

"$PYBIN" "$EXTRACT_PY" ${OUT}_0_.xml.gz ${OUT}_compact.npz
echo "run_extrinsic_fairdraw: done $SUF"
