#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo: drive create_eos_posterior_pipeline under each workflow backend.
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
CEPOS_BIN="${CEPOS_BIN:-}"
if [[ -z "${CEPOS_BIN}" ]]; then
    CEPOS_BIN="$(command -v create_eos_posterior_pipeline 2>/dev/null || true)"
fi
if [[ -z "${CEPOS_BIN}" ]]; then
    CEPOS_BIN="${RIFT_CODE_ROOT}/bin/create_eos_posterior_pipeline"
fi
if [[ ! -e "${CEPOS_BIN}" ]]; then
    echo "ERROR: cannot locate create_eos_posterior_pipeline." >&2
    echo "Searched PATH and ${RIFT_CODE_ROOT}/bin/.  Set CEPOS_BIN=/path/to/the/script." >&2
    exit 2
fi
echo "Using CEPOS_BIN=${CEPOS_BIN}"

run_for_backend() {
    local backend="$1"
    local outdir="${OUTPUTS_BASE}/${backend}/eos_posterior"
    # Tolerant cleanup: some filesystems (FUSE mounts, restricted shares)
    # refuse rm.  If we can't remove old artefacts we just overwrite them
    # in place.
    rm -rf "${outdir}" 2>/dev/null || true
    mkdir -p "${outdir}"

    # eos-grid is plain-text -- create_eos_posterior_pipeline does
    # np.loadtxt("grid-0.dat") on it, so we use a parseable text grid
    # rather than the .xml.gz used by the BasicIteration / pseudo_pipe
    # demos.
    cp "${INPUTS}/eos-grid.dat"               "${outdir}/eos-grid.dat"
    # Use the list-file pathway: one args / exe / nchunk entry per event.
    # The single-args pathway (--marg-event-args) leaves
    # marg_event_nchunk_list=None which crashes the multi-event loop in
    # the upstream script.
    cp "${INPUTS}/args_marg_event_list.txt"   "${outdir}/args_marg_event_list.txt"
    cp "${INPUTS}/marg_event_exe_list.txt"    "${outdir}/marg_event_exe_list.txt"
    cp "${INPUTS}/marg_event_nchunk_list.txt" "${outdir}/marg_event_nchunk_list.txt"
    cp "${INPUTS}/args_eos_post.txt"          "${outdir}/args_eos_post.txt"
    cp "${INPUTS}/args_test.txt"              "${outdir}/args_test.txt"
    cp "${INPUTS}/event-1.net"                "${outdir}/event-1.net"
    cp "${INPUTS}/event-2.net"                "${outdir}/event-2.net"

    echo
    echo "============================================================"
    echo "Backend: ${backend}"
    echo "Output:  ${outdir}"
    echo "============================================================"

    local rc=0
    (
        cd "${outdir}"
        RIFT_DAG_BACKEND="${backend}" \
        python3 "${CEPOS_BIN}" \
            --working-directory             "${outdir}" \
            --input-grid                    "${outdir}/eos-grid.dat" \
            --event-file                    "${outdir}/event-1.net" \
            --event-file                    "${outdir}/event-2.net" \
            --marg-event-args-list-file     "${outdir}/args_marg_event_list.txt" \
            --marg-event-exe-list-file      "${outdir}/marg_event_exe_list.txt" \
            --marg-event-nchunk-list-file   "${outdir}/marg_event_nchunk_list.txt" \
            --eos-post-args                 "${outdir}/args_eos_post.txt" \
            --test-args                     "${outdir}/args_test.txt" \
            --n-iterations                  2 \
            --request-memory-marg           8192 \
            --general-request-disk          10M \
            --condor-nogrid-nonworker \
            2>&1 | tee "${outdir}/driver.log"
        exit "${PIPESTATUS[0]}"
    ) || rc=$?
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
    echo "  ${OUTPUTS_BASE}/${backend}/eos_posterior"
done

if (( ${#BACKEND_FAILURES[@]} > 0 )); then
    echo >&2
    echo "FAILURE: backend(s) failed: ${BACKEND_FAILURES[*]}" >&2
    exit 1
fi
