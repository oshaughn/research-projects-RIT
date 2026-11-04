#!/bin/bash
#
# CIT-deployable test suite for the v2 simulation archive.
#
# Submits multiple per-(sim, level) jobs to the local condor schedd
# under OSG-style pure-transfer mode (no submit-FS access on workers)
# and asserts that:
#   * every level file lands at its canonical archive path on the
#     submit node via transfer_output_remaps
#   * chained DAG dependencies + transfer_input_files correctly ferry
#     prior level outputs into level-N workers (proven via prev_count)
#   * refinement, dedup, multi-sim batches all work
#
# Gated on RIFT_TEST_CONDOR=1 so it does NOT run in CI by default.
# Run from a CIT submit host:
#
#     RIFT_TEST_CONDOR=1 ./.travis/test-simulation-manager-v2-condor.sh
#
# Optional environment / passthrough flags:
#     LIGO_ACCOUNTING        -> --accounting-group
#     LIGO_USER_NAME         -> --accounting-group-user
#     RIFT_TEST_RUN_POOL     -> --run-pool       (cross-pool: condor_submit_dag -name)
#     RIFT_TEST_RUN_COLLECTOR -> --run-collector (for poll() Schedd lookups)
#     RIFT_TEST_SINGULARITY  -> --singularity-image  (e.g. /cvmfs/.../rift:latest)
#     RIFT_TEST_DESIRED_SITES -> --desired-sites
#     RIFT_TEST_BASE         -> workspace dir (default: $TMPDIR/simmgr_v2_$$)
#     RIFT_TEST_TIMEOUT      -> per-case timeout, seconds (default 600)
#     RIFT_TEST_ONLY         -> comma-separated case names to restrict
#                               (e.g. case_1_single_sim_single_level)

set -e

if [[ "${RIFT_TEST_CONDOR}" != "1" ]]; then
    echo "Skipping v2 condor end-to-end test (set RIFT_TEST_CONDOR=1 to enable)."
    exit 0
fi

for c in condor_submit_dag condor_q; do
    if ! command -v "$c" >/dev/null 2>&1; then
        echo "ERROR: $c not on PATH; is HTCondor installed?" >&2
        exit 1
    fi
done
python3 -c "
import sys
for m in ('htcondor2', 'htcondor'):
    try:
        __import__(m); print('Using', m, 'bindings'); sys.exit(0)
    except ImportError: continue
sys.exit(1)" || {
    echo "ERROR: neither htcondor2 nor htcondor python bindings available" >&2
    exit 1
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DRIVER="${SCRIPT_DIR}/_simmgr_v2_e2e.py"
[[ -f "${DRIVER}" ]] || { echo "ERROR: missing ${DRIVER}" >&2; exit 1; }

BASE="${RIFT_TEST_BASE:-${TMPDIR:-/tmp}/simmgr_v2_$$}"
TIMEOUT="${RIFT_TEST_TIMEOUT:-600}"

ARGS=(--base "${BASE}" --timeout "${TIMEOUT}")

# Accounting (env -> CLI passthrough; the driver also picks env directly)
[[ -n "${LIGO_ACCOUNTING}" ]] && ARGS+=(--accounting-group "${LIGO_ACCOUNTING}")
[[ -n "${LIGO_USER_NAME}"  ]] && ARGS+=(--accounting-group-user "${LIGO_USER_NAME}")

# Cross-pool / OSG-specific knobs
[[ -n "${RIFT_TEST_RUN_POOL}"      ]] && ARGS+=(--run-pool "${RIFT_TEST_RUN_POOL}")
[[ -n "${RIFT_TEST_RUN_COLLECTOR}" ]] && ARGS+=(--run-collector "${RIFT_TEST_RUN_COLLECTOR}")
[[ -n "${RIFT_TEST_SINGULARITY}"   ]] && ARGS+=(--singularity-image "${RIFT_TEST_SINGULARITY}")
[[ -n "${RIFT_TEST_DESIRED_SITES}" ]] && ARGS+=(--desired-sites "${RIFT_TEST_DESIRED_SITES}")

# Restrict to specific cases
if [[ -n "${RIFT_TEST_ONLY}" ]]; then
    IFS=',' read -ra CASES <<< "${RIFT_TEST_ONLY}"
    for c in "${CASES[@]}"; do ARGS+=(--only "$c"); done
fi

echo "Running: python3 ${DRIVER} ${ARGS[*]}"
echo
exec python3 "${DRIVER}" "${ARGS[@]}"
