#!/usr/bin/env bash
# submit.sh -- apply CIT-local fixes to the built rundir, then condor_submit_dag.
#
# The DAG util_RIFT_pseudo_pipe.py emits uses the OSG/OSDF idioms (vanilla universe
# + MY.SingularityImage, osdf .sif delivery, getenv=True, no GPU capability floor)
# that the CIT local pool rejects or that fail to open the container there.  These
# shared, tested fixes (read-only under $FIX_TOOLS) rewrite the subs in place:
#   fix_getenv               getenv=True -> getenv=*   (CIT schedd bans getenv=True)
#   fix_container_universe   vanilla+SingularityImage -> container universe
#   fix_container_filexfer   deliver the LOCAL .sif by file transfer (OSDF .sif
#                            can't be opened on CIT execute nodes)
#   fix_cip_single_container CIP runs on CPU slots (no GPU cap) -> single image
#   pin_local                pin worker jobs to the CIT-local GPU pool
#   require_gpu_floor        require GPU capability in [floor, ceil] the image supports
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
rift_env
cd "$MODEL_DIR"

test -s "$RUNDIR/$PP_DAG" || { echo "no DAG in $RUNDIR (run: make dag)"; exit 1; }
test -f "$CONTAINER_SIF"  || { echo "container .sif not found: $CONTAINER_SIF"; exit 1; }
T="$FIX_TOOLS"

# Convert 'vanilla + MY.SingularityImage=<local abs .sif>' subs to HTCondor
# container universe.  This build bakes the LOCAL .sif directly into
# MY.SingularityImage (no OSDF / $$([...]) token), so the shared
# fix_container_universe.sh (which harvests the image from an osdf entry in
# transfer_input_files) is a no-op here.  We take the image straight from
# MY.SingularityImage and emit container_image=<that path>.  Idempotent.
convert_local_container_universe() {
  local rd="$1" n=0 s img tmp
  while IFS= read -r s; do
    grep -qE '^universe[[:space:]]*=[[:space:]]*container' "$s" && continue
    grep -qE '^[[:space:]]*MY\.SingularityImage[[:space:]]*=' "$s" || continue
    img=$(sed -nE 's/^[[:space:]]*MY\.SingularityImage[[:space:]]*=[[:space:]]*"?([^"]+)"?[[:space:]]*$/\1/p' "$s" | head -1)
    [ -n "$img" ] || { echo "WARN $(basename "$s"): no image path in MY.SingularityImage"; continue; }
    tmp=$(mktemp)
    awk -v img="$img" '
      /^universe[[:space:]]*=[[:space:]]*vanilla/     { print "universe = container"; print "container_image = " img; next }
      /^[[:space:]]*MY\.SingularityImage[[:space:]]*=/    { next }
      /^[[:space:]]*MY\.SingularityBindCVMFS[[:space:]]*=/ { next }
      /^[[:space:]]*container_image[[:space:]]*=/         { next }
      { print }
    ' "$s" > "$tmp"
    if grep -qE '^universe[[:space:]]*=[[:space:]]*container' "$tmp" \
       && grep -qE '^container_image[[:space:]]*=' "$tmp" \
       && ! grep -qE 'MY\.SingularityImage' "$tmp"; then
      mv "$tmp" "$s"; echo "local-container-universe $(basename "$s")"; n=$((n+1))
    else
      rm -f "$tmp"; echo "WARN $(basename "$s"): local container-universe conversion failed"
    fi
  done < <(find "$rd" -name '*.sub' 2>/dev/null)
  echo "OK: converted $n sub(s) to container universe (local .sif) under $rd"
}

echo "[submit] applying CIT-local fixes to $RUNDIR"
bash "$T/fix_getenv.sh"            "$RUNDIR"
convert_local_container_universe  "$RUNDIR"
bash "$T/fix_container_universe.sh" "$RUNDIR"    # secondary pass (osdf builds); no-op here
bash "$T/fix_container_filexfer.sh" "$RUNDIR" "$CONTAINER_SIF"
bash "$T/fix_cip_single_container.sh" "$RUNDIR" "$CONTAINER_SIF"
bash "$T/pin_local.sh"            "$RUNDIR"
bash "$T/require_gpu_floor.sh"    "$RUNDIR" "$GPU_CAP_FLOOR" "$GPU_CAP_CEIL"

echo "[submit] condor_submit_dag -f $PP_DAG"
cd "$RUNDIR"
condor_submit_dag -f "$PP_DAG"
echo "[submit] submitted. Watch: condor_q ; tail -f $RUNDIR/$PP_DAG.dagman.out"
echo "         held? condor_q -held"
