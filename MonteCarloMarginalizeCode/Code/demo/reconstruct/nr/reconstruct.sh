#!/usr/bin/env bash
#
# reconstruct.sh  CONFIG.sh
#
# End-to-end driver:
#   1. run ILE fair-draw jobs SEQUENTIALLY until >= TARGET_SAMPLES fair draws
#      accumulate (per-run yield is stochastic, often 5-40, so we loop);
#   2. pool all compact .npz and reconstruct the whitened 90% strain band.
#
# Concurrent ILE jobs on a shared node tend to hit pthread_create EAGAIN
# (per-user thread cap), so we run them one at a time.
set -e
CONFIG=$1
[ -z "$CONFIG" ] && { echo "usage: $0 CONFIG.sh"; exit 1; }
source "$CONFIG"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

TARGET_SAMPLES=${TARGET_SAMPLES:-150}
MAX_RUNS=${MAX_RUNS:-12}

count_samples() {  # total fair-draw rows across all compact npz so far
  "$PYBIN" - "$@" <<'PY'
import sys, glob, numpy as np
n=0
for f in glob.glob(sys.argv[1]):
    try: n+=len(np.load(f)['time'])
    except Exception: pass
print(n)
PY
}

i=0; total=0
while [ "$total" -lt "$TARGET_SAMPLES" ] && [ "$i" -lt "$MAX_RUNS" ]; do
  s=$(printf "%03d" "$i")
  echo "=== ILE fair-draw run $s  (have $total / $TARGET_SAMPLES) ==="
  bash "$HERE/run_extrinsic_fairdraw.sh" "$CONFIG" "$s" || echo "run $s failed, continuing"
  total=$(count_samples "$WORKDIR/ile_fd_*_compact.npz")
  i=$((i+1))
done
echo "=== accumulated $total fair-draw samples in $i runs ==="

# pool everything and reconstruct
POOL=""; for f in "$WORKDIR"/ile_fd_*_compact.npz; do POOL="$POOL --samples $f"; done
"$PYBIN" "$HERE/../reconstruct_strain.py" $POOL --fair-draw \
  --group "$NR_GROUP" --nr-param "$NR_PARAM" $PLOT_PSD_ARGS \
  --event-time "$EVENT_TIME" --event-name "$EVENT_NAME" --sim-id "$SIM_ID" \
  --srate "$SRATE" --flow "$FLOW" ${INTRINSIC:+--intrinsic $INTRINSIC} \
  --nproc "${NPROC:-8}" --tlo "${TLO:--0.10}" --thi "${THI:-0.06}" \
  --out "$OUT_PNG"
echo "=== reconstruction written: $OUT_PNG ==="
