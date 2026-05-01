#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo: drive util_RIFT_pseudo_pipe.py under each workflow backend.
#
# pseudo_pipe is the high-level driver: it does data discovery (PSDs, frame
# files, channel selection, ...), then invokes a `create_event_parameter_*`
# pipeline.  The full happy-path requires either a GraceDb event ID or a
# fully-populated INI file.
#
# This demo uses the minimal "manual event-time + manual initial grid" entry
# path so that the pipeline can be exercised without contacting GraceDb.
# Replace the --gracedb-id / --use-ini variants with your preferred entry
# point on a real LIGO submission node.
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
PIPE_BIN="${PIPE_BIN:-}"
if [[ -z "${PIPE_BIN}" ]]; then
    PIPE_BIN="$(command -v util_RIFT_pseudo_pipe.py 2>/dev/null || true)"
fi
if [[ -z "${PIPE_BIN}" ]]; then
    PIPE_BIN="${RIFT_CODE_ROOT}/bin/util_RIFT_pseudo_pipe.py"
fi
if [[ ! -e "${PIPE_BIN}" ]]; then
    echo "ERROR: cannot locate util_RIFT_pseudo_pipe.py." >&2
    echo "Searched PATH and ${RIFT_CODE_ROOT}/bin/.  Set PIPE_BIN=/path/to/util_RIFT_pseudo_pipe.py." >&2
    exit 2
fi
echo "Using PIPE_BIN=${PIPE_BIN}"

# pseudo_pipe needs a real INI file + matching coinc.xml; without --use-ini
# it tries (and fails) to look up the event in GraceDb.  Use the in-tree
# travis fixtures (GW150914.ini + coinc.xml) which the project already
# guarantees parses cleanly.  Override these via env vars to point at your
# own event.
PSEUDO_INI="${PSEUDO_INI:-${INPUTS}/GW150914.ini}"
PSEUDO_COINC="${PSEUDO_COINC:-${INPUTS}/coinc.xml}"
PSEUDO_CACHE="${OUTPUTS_BASE}/empty.cache"
mkdir -p "${OUTPUTS_BASE}"
: > "${PSEUDO_CACHE}"

# The travis driver sets these; pseudo_pipe expects them to be present.
export RIFT_LOWLATENCY="${RIFT_LOWLATENCY:-True}"
export SINGULARITY_RIFT_IMAGE="${SINGULARITY_RIFT_IMAGE:-foo}"
export SINGULARITY_BASE_EXE_DIR="${SINGULARITY_BASE_EXE_DIR:-/usr/bin/}"

run_for_backend() {
    local backend="$1"
    local outdir="${OUTPUTS_BASE}/${backend}/pseudo_pipe"
    # Wipe any previous run.  pseudo_pipe.py's --use-rundir does its own
    # os.mkdir on the rundir and aborts if it already exists, so we wipe
    # the path entirely and DO NOT pre-create outdir.  We do create the
    # parent directory so cd / output of the driver.log works.
    rm -rf "${outdir}" 2>/dev/null || true
    mkdir -p "${OUTPUTS_BASE}/${backend}"

    echo
    echo "============================================================"
    echo "Backend: ${backend}"
    echo "Output:  ${outdir}"
    echo "============================================================"

    # NOTE: most pseudo_pipe options below are required by argparse; they do
    # not all influence the DAG-emission stage.  Adjust as needed for your
    # production runs.  The driver log goes to a sibling of the rundir
    # because pseudo_pipe creates the rundir itself.
    local driver_log="${OUTPUTS_BASE}/${backend}/pseudo_pipe.driver.log"
    # Mirror the in-tree .travis/test-build.sh invocation pattern: use
    # --use-ini + --use-coinc, which is the path through pseudo_pipe that
    # actually populates event_dict["IFOs"] etc.  Without --use-ini the
    # script tries to look the event up in GraceDb.
    local rc=0
    (
        cd "${OUTPUTS_BASE}/${backend}"
        RIFT_DAG_BACKEND="${backend}" \
        python3 "${PIPE_BIN}" \
            --use-ini            "${PSEUDO_INI}" \
            --use-coinc          "${PSEUDO_COINC}" \
            --use-rundir         "${outdir}" \
            --fake-data-cache    "${PSEUDO_CACHE}" \
            --skip-reproducibility \
            --condor-nogrid-nonworker \
            2>&1 | tee "${driver_log}"
        exit "${PIPESTATUS[0]}"
    ) || rc=$?
    if [[ "${rc}" -ne 0 ]] || grep -q "^Traceback " "${driver_log}"; then
        echo "[${backend}] driver FAILED (exit ${rc}; see ${driver_log})" >&2
        BACKEND_FAILURES+=("${backend}")
    fi
    # Move the driver log into the rundir if pseudo_pipe successfully
    # created it.
    if [[ -d "${outdir}" && -f "${driver_log}" ]]; then
        mv "${driver_log}" "${outdir}/driver.log" 2>/dev/null || true
    fi

    echo
    echo "Artefacts produced for ${backend}:"
    if [[ -d "${outdir}" ]]; then
        find "${outdir}" -maxdepth 4 -type f \
            \( -name '*.sub' -o -name '*.sbatch' -o -name '*.dag' -o -name '*_dag.sh' \) \
            | sort | sed "s|^${outdir}/|  |"
    else
        echo "  (none -- pseudo_pipe did not produce a rundir; see ${driver_log})"
    fi
}

BACKEND_FAILURES=()
for backend in htcondor glue slurm; do
    run_for_backend "${backend}"
done

if (( ${#BACKEND_FAILURES[@]} > 0 )); then
    echo >&2
    echo "FAILURE: backend(s) failed: ${BACKEND_FAILURES[*]}" >&2
    exit 1
fi

echo
echo "Done.  Output trees:"
for backend in htcondor glue slurm; do
    echo "  ${OUTPUTS_BASE}/${backend}/pseudo_pipe"
done
