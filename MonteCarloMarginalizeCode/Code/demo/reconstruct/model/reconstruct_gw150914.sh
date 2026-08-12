#!/usr/bin/env bash
# reconstruct_gw150914.sh -- the payoff: whitened strain band from the posterior.
#
# Runs once the DAG has produced rundir_gw150914_D/extrinsic_posterior_samples.dat.
# That file is a fair-draw posterior whose 'time' column varies per row (because
# ILE_extr.sub carried --fairdraw-extrinsic-output --resample-time-marginalization),
# so reconstruct_strain.py can place each IMRPhenomD realization at its OWN (time,
# phase) and get a phase-coherent 90% band with NO post-hoc alignment.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
rift_env
cd "$MODEL_DIR"

test -s "$POSTERIOR" || {
  echo "posterior not ready: $POSTERIOR"
  echo "  (the condor DAG must finish first -- check: condor_q ; make status)"
  exit 1
}
test -s "$H1_PSD" && test -s "$L1_PSD" || { echo "missing PSDs (run: make psd)"; exit 1; }

echo "[reconstruct] $POSTERIOR -> $SAMPLES_NPZ"
"$PYBIN" "$DAT2COMPACT_PY" "$POSTERIOR" "$SAMPLES_NPZ"

echo "[reconstruct] building whitened strain band -> $OUT_PNG"
"$PYBIN" "$RECON_PY" --samples "$SAMPLES_NPZ" --fair-draw --approx "$APPROX" \
  --psd-file H1="$H1_PSD" --psd-file L1="$L1_PSD" \
  --event-time "$EVENT_TIME" --event-name "$EVENT_NAME" --sim-id "$APPROX" \
  --srate "$SRATE" --flow "$FMIN" --fref "$FREF" --lmax "$LMAX" \
  --out "$OUT_PNG"
echo "[reconstruct] wrote $OUT_PNG"
