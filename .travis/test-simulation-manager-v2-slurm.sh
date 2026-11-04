#!/bin/bash
#
# SLURM end-to-end test suite for the v2 simulation archive.
#
# Submits per-(sim, level) sbatch scripts (chained via --dependency=afterok)
# against the partition specified in RIFT_TEST_SLURM_PARTITION. Assumes
# compute nodes share the archive's filesystem (the standard HPC SLURM
# model). Sites without a shared FS should configure RIFT_TEST_SLURM_PRELUDE
# to bind-mount the archive into a container, or use sbcast.
#
# Gated on RIFT_TEST_SLURM=1; fails gracefully with a clear error
# message on a non-slurm cluster (sbatch/squeue not on PATH).
#
# Run:
#     RIFT_TEST_SLURM=1 RIFT_TEST_SLURM_PARTITION=compute \\
#         ./.travis/test-simulation-manager-v2-slurm.sh
#
# Optional environment:
#     RIFT_TEST_SLURM_PARTITION   -- required; SLURM partition / queue
#     RIFT_TEST_SLURM_TIME        -- per-job --time= (default 00:10:00)
#     RIFT_TEST_SLURM_ACCOUNT     -- pass to --account
#     RIFT_TEST_SLURM_QOS         -- pass to --qos
#     RIFT_TEST_SLURM_PRELUDE     -- shell snippet before the bootstrap
#                                    (e.g. 'module load python/3.11')
#     RIFT_TEST_SLURM_PYTHON      -- python executable used inside the
#                                    sbatch script (default: python3)
#     RIFT_TEST_BASE              -- workspace dir
#     RIFT_TEST_TIMEOUT           -- per-case timeout (default 600)
#     RIFT_TEST_ONLY              -- comma-separated case names

set -e

if [[ "${RIFT_TEST_SLURM}" != "1" ]]; then
    echo "Skipping v2 slurm end-to-end test (set RIFT_TEST_SLURM=1 to enable)."
    exit 0
fi

if [[ -z "${RIFT_TEST_SLURM_PARTITION}" ]]; then
    echo "ERROR: RIFT_TEST_SLURM_PARTITION must be set (e.g. =compute)" >&2
    exit 1
fi

# Verify slurm tools are present. Each is needed; fail with a per-tool
# message so the operator knows exactly what's missing.
missing=0
for tool in sbatch squeue; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: $tool not on PATH; this host doesn't appear to have a"  >&2
        echo "       SLURM client installed. The v2 slurm test suite requires" >&2
        echo "       sbatch and squeue (sacct is recommended for richer polling)." >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    exit 1
fi
# sacct is optional — only needed if you extend poll() to use it.
if ! command -v sacct >/dev/null 2>&1; then
    echo "Note: sacct not on PATH; poll() will use squeue alone."
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DRIVER="${SCRIPT_DIR}/_simmgr_v2_slurm_e2e.py"
[[ -f "${DRIVER}" ]] || { echo "ERROR: missing ${DRIVER}" >&2; exit 1; }

BASE="${RIFT_TEST_BASE:-${TMPDIR:-/tmp}/simmgr_v2_slurm_$$}"
TIMEOUT="${RIFT_TEST_TIMEOUT:-600}"
TIME_LIMIT="${RIFT_TEST_SLURM_TIME:-00:10:00}"
PYTHON_EXEC="${RIFT_TEST_SLURM_PYTHON:-python3}"

ARGS=(--base "${BASE}"
      --partition "${RIFT_TEST_SLURM_PARTITION}"
      --time-limit "${TIME_LIMIT}"
      --python-executable "${PYTHON_EXEC}"
      --timeout "${TIMEOUT}")

[[ -n "${RIFT_TEST_SLURM_ACCOUNT}" ]] && ARGS+=(--account "${RIFT_TEST_SLURM_ACCOUNT}")
[[ -n "${RIFT_TEST_SLURM_QOS}"     ]] && ARGS+=(--qos "${RIFT_TEST_SLURM_QOS}")
[[ -n "${RIFT_TEST_SLURM_PRELUDE}" ]] && ARGS+=(--prelude "${RIFT_TEST_SLURM_PRELUDE}")

if [[ -n "${RIFT_TEST_ONLY}" ]]; then
    IFS=',' read -ra CASES <<< "${RIFT_TEST_ONLY}"
    for c in "${CASES[@]}"; do ARGS+=(--only "$c"); done
fi

echo "Running: python3 ${DRIVER} ${ARGS[*]}"
echo
exec python3 "${DRIVER}" "${ARGS[@]}"
