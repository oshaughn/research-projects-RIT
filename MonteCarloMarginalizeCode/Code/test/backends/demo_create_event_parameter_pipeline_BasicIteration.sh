#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo: drive create_event_parameter_pipeline_BasicIteration under each
# workflow backend (htcondor, glue, slurm).
#
# Each invocation emits a complete DAG (or sbatch driver script) plus all
# accompanying submit / sbatch files into a per-backend output directory.
#
# This script does NOT execute the resulting workflow; it only verifies that
# the same pipeline driver script produces the appropriate set of artefacts
# under each backend.
#
# Run:
#     bash demo_create_event_parameter_pipeline_BasicIteration.sh
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INPUTS="${HERE}/inputs"
OUTPUTS_BASE="${HERE}/outputs"

# ----------------------------------------------------------------------------
# Make in-tree RIFT discoverable without `pip install -e .`.
# ----------------------------------------------------------------------------
RIFT_CODE_ROOT="$(cd "${HERE}/../.." && pwd)"        # MonteCarloMarginalizeCode/Code
export PYTHONPATH="${RIFT_CODE_ROOT}:${PYTHONPATH:-}"
export PATH="${RIFT_CODE_ROOT}/bin:${PATH}"

# Resolve the pipeline driver: prefer an absolute path (caller-provided or
# PATH-discovered), fall back to the in-tree copy.
CEPP_BIN="${CEPP_BIN:-}"
if [[ -z "${CEPP_BIN}" ]]; then
    CEPP_BIN="$(command -v create_event_parameter_pipeline_BasicIteration 2>/dev/null || true)"
fi
if [[ -z "${CEPP_BIN}" ]]; then
    CEPP_BIN="${RIFT_CODE_ROOT}/bin/create_event_parameter_pipeline_BasicIteration"
fi
if [[ ! -e "${CEPP_BIN}" ]]; then
    echo "ERROR: cannot locate create_event_parameter_pipeline_BasicIteration." >&2
    echo "Searched PATH and ${RIFT_CODE_ROOT}/bin/.  Set CEPP_BIN=/path/to/the/script." >&2
    exit 2
fi
echo "Using CEPP_BIN=${CEPP_BIN}"

run_for_backend() {
    local backend="$1"
    local outdir="${OUTPUTS_BASE}/${backend}/cepp_basic_iteration"
    # Tolerant cleanup: some filesystems (FUSE mounts, restricted shares)
    # refuse rm.  If we can't remove old artefacts we just overwrite them
    # in place.
    rm -rf "${outdir}" 2>/dev/null || true
    mkdir -p "${outdir}"

    # Stage the args files into the per-backend working dir.
    cp "${INPUTS}/args_ile.txt"         "${outdir}/args_ile.txt"
    cp "${INPUTS}/args_cip.txt"         "${outdir}/args_cip.txt"
    cp "${INPUTS}/args_test.txt"        "${outdir}/args_test.txt"
    # Sanitise the grid placeholder for modern (igwn_ligolw / ligo.lw)
    # parsers: strip deprecated process_id columns and ilwd:char types.
    python3 "${INPUTS}/sanitize_grid.py" \
        "${INPUTS}/overlap-grid.xml.gz" \
        "${outdir}/overlap-grid.xml.gz"

    echo
    echo "============================================================"
    echo "Backend: ${backend}"
    echo "Output:  ${outdir}"
    echo "============================================================"

    local rc=0
    (
        cd "${outdir}"
        RIFT_DAG_BACKEND="${backend}" \
        python3 "${CEPP_BIN}" \
            --working-directory      "${outdir}" \
            --input-grid             "${outdir}/overlap-grid.xml.gz" \
            --ile-args               "${outdir}/args_ile.txt" \
            --cip-args               "${outdir}/args_cip.txt" \
            --test-args              "${outdir}/args_test.txt" \
            --n-iterations           2 \
            --ile-n-events-to-analyze 1 \
            --request-memory-ILE     4096 \
            --request-memory-CIP     4096 \
            --general-request-disk   "10M" \
            --general-retries        1 \
            --condor-nogrid-nonworker \
            2>&1 | tee "${outdir}/driver.log"
        # `tee` swallows the python exit code; recover it from PIPESTATUS.
        exit "${PIPESTATUS[0]}"
    ) || rc=$?
    # Fail fast on any per-backend driver failure, *and* if the driver
    # exited 0 but its log contains a python traceback (which can happen
    # when the script logs the exception but still hands control back).
    if [[ "${rc}" -ne 0 ]] || grep -q "^Traceback " "${outdir}/driver.log"; then
        echo "[${backend}] driver FAILED (exit ${rc}; see ${outdir}/driver.log)" >&2
        BACKEND_FAILURES+=("${backend}")
    fi

    echo
    echo "Artefacts produced for ${backend}:"
    find "${outdir}" -maxdepth 3 -type f \
        \( -name '*.sub' -o -name '*.sbatch' -o -name '*.dag' -o -name '*_dag.sh' \) \
        | sort | sed "s|^${outdir}/|  |"
}

mkdir -p "${OUTPUTS_BASE}"
BACKEND_FAILURES=()
for backend in htcondor glue slurm; do
    run_for_backend "${backend}"
done

echo
echo "Done.  Output trees:"
for backend in htcondor glue slurm; do
    echo "  ${OUTPUTS_BASE}/${backend}/cepp_basic_iteration"
done

if (( ${#BACKEND_FAILURES[@]} > 0 )); then
    echo >&2
    echo "FAILURE: backend(s) failed: ${BACKEND_FAILURES[*]}" >&2
    exit 1
fi
