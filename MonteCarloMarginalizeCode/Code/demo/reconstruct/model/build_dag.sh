#!/usr/bin/env bash
# build_dag.sh -- generate the RIFT parameter-estimation DAG for GW150914.
#
# IMRPhenomD forces aligned spin; --assume-nospin pins the spins to zero; --l-max 2.
# The extrinsic flags are the whole point of this demo:
#
#   --add-extrinsic --batch-extrinsic --add-extrinsic-time-resampling
#   --internal-ile-srate-time-resampling 4096
#
# make the pipeline's FINAL ILE_extr stage emit, into ILE_extr.sub,
#   --fairdraw-extrinsic-output --resample-time-marginalization
# so every posterior sample in extrinsic_posterior_samples.dat carries its OWN
# coalescence time (coherent with its phase) -- exactly what reconstruct_strain.py
# needs for a phase-coherent band with NO post-hoc alignment.
#
# PSDs: pseudo_pipe has no --psd-file flag in this version; the helper points each
# analysis IFO at <rundir>/<IFO>-psd.xml.gz.  So we build the rundir, then COPY the
# estimated per-IFO PSDs in (the file is only read at DAG-RUN time).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
rift_env
cd "$MODEL_DIR"

test -s "$COINC"  || { echo "missing coinc.xml (run: make coinc)"; exit 1; }
test -s "$CACHE"  || { echo "missing event.cache (run: make data)"; exit 1; }
test -s "$H1_PSD" && test -s "$L1_PSD" || { echo "missing PSDs (run: make psd)"; exit 1; }
test -s "$INI"    || { echo "missing $INI"; exit 1; }

echo "[dag] util_RIFT_pseudo_pipe.py -> $RUNDIR"
rm -rf "$RUNDIR" 2>/dev/null || true

util_RIFT_pseudo_pipe.py \
  --use-ini "$INI" --use-coinc "$COINC" --fake-data-cache "$CACHE" \
  --approx "$APPROX" --l-max "$LMAX" --assume-nospin \
  --fmin-template "$FMIN" --force-mc-range "$MC_RANGE" \
  --ile-sampler-method AV --ile-n-eff "$ILE_NEFF" \
  --ile-jobs-per-worker "$JOBS_PER_WORKER" --internal-force-iterations "$NIT" \
  --add-extrinsic --batch-extrinsic --add-extrinsic-time-resampling \
  --internal-ile-srate-time-resampling "$SRATE_TIME" \
  --n-output-samples-last "$NSAMP_LAST" \
  --use-osg --use-osg-cip --use-osg-file-transfer \
  --internal-truncate-files-for-osg-file-transfer \
  --internal-ile-request-disk "$DISK_ILE" \
  --internal-cip-request-disk "$DISK_CIP" \
  --internal-general-request-disk "$DISK_GEN" \
  --use-rundir "$RUNDIR"

# stage the per-IFO PSDs into the rundir (read at DAG-run time)
for ifo in $IFOS; do
  cp -v "$MODEL_DIR/${ifo}-psd.xml.gz" "$RUNDIR/${ifo}-psd.xml.gz"
done

# ---- verify the build ----------------------------------------------------
test -s "$RUNDIR/$PP_DAG" || { echo "FAIL: no DAG produced"; exit 1; }
for ifo in $IFOS; do
  grep -q -- "${ifo}-psd.xml.gz" "$RUNDIR/ILE.sub" \
    || { echo "FAIL: ILE.sub missing per-IFO PSD ${ifo}-psd.xml.gz"; exit 1; }
done
grep -q -- "--resample-time-marginalization" "$RUNDIR/ILE_extr.sub" \
  || { echo "FAIL: ILE_extr.sub missing --resample-time-marginalization"; exit 1; }
grep -q -- "--fairdraw-extrinsic-output" "$RUNDIR/ILE_extr.sub" \
  || { echo "FAIL: ILE_extr.sub missing --fairdraw-extrinsic-output"; exit 1; }
echo "[dag] OK: DAG built in $RUNDIR"
echo "      ILE_extr.sub has --fairdraw-extrinsic-output --resample-time-marginalization"
echo "      Next: make submit"
