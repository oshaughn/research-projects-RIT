#!/usr/bin/env bash
# make_coinc.sh -- build coinc.xml for GW150914 (time + IFOs; masses are a seed).
#
# There is no injection: we synthesize a minimal sim_inspiral row at seed
# intrinsics + the trigger time so util_SimInspiralToCoinc.py can emit a coinc
# carrying the right event time / IFOs / (nominal) SNR.  RIFT only needs the
# coinc for time + IFOs; the intrinsic grid is proposed fresh from --force-mc-range.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
rift_env
cd "$MODEL_DIR"

echo "[coinc] seed sim_inspiral at m1=$SEED_M1 m2=$SEED_M2 s1z=$SEED_S1Z s2z=$SEED_S2Z t=$EVENT_TIME"
util_WriteInjectionFile.py \
  --parameter m1   --parameter-value "$SEED_M1" \
  --parameter m2   --parameter-value "$SEED_M2" \
  --parameter s1z  --parameter-value "$SEED_S1Z" \
  --parameter s2z  --parameter-value "$SEED_S2Z" \
  --parameter dist --parameter-value "$SEED_DIST" \
  --parameter tref --parameter-value "$EVENT_TIME" \
  --parameter fmin --parameter-value "$FMIN" \
  --approximant "$APPROX" --fname seed_inj

# one --ifo per detector (H1 L1; no Virgo in O1)
IFO_ARGS=()
for ifo in $IFOS; do IFO_ARGS+=(--ifo "$ifo"); done

util_SimInspiralToCoinc.py --sim-xml seed_inj.xml.gz --event 0 \
  "${IFO_ARGS[@]}" --output "$COINC" --injected-snr "$SEED_SNR"

test -s "$COINC" || { echo "FAIL: coinc.xml not produced"; exit 1; }
echo "[coinc] wrote $COINC"
