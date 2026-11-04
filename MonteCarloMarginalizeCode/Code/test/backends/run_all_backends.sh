#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run the full backend test suite:
#   1. low-level unit tests for dag_utils_generic
#   2. self-contained pipeline synthesizer (no science deps)
#   3. real pipeline drivers (require lalsimulation + scipy + RIFT install)
# -----------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}"

# ----------------------------------------------------------------------------
# Make in-tree RIFT discoverable without `pip install -e .`.
# ----------------------------------------------------------------------------
RIFT_CODE_ROOT="$(cd "${HERE}/../.." && pwd)"        # MonteCarloMarginalizeCode/Code
RIFT_BIN_DIR="${RIFT_CODE_ROOT}/bin"
export PYTHONPATH="${RIFT_CODE_ROOT}:${PYTHONPATH:-}"
export PATH="${RIFT_BIN_DIR}:${PATH}"

PASS=0
FAIL=0
SKIP=0
SECTIONS=()

run_section() {
    local label="$1"; shift
    echo
    echo "============================================================"
    echo "  ${label}"
    echo "============================================================"
    if "$@"; then
        echo "[PASS] ${label}"
        PASS=$((PASS + 1))
        SECTIONS+=("PASS  ${label}")
    else
        echo "[FAIL] ${label} (exit $?)"
        FAIL=$((FAIL + 1))
        SECTIONS+=("FAIL  ${label}")
    fi
}

skip_section() {
    local label="$1"; shift
    local reason="$1"; shift
    echo
    echo "============================================================"
    echo "  ${label}  [SKIPPED: ${reason}]"
    echo "============================================================"
    SKIP=$((SKIP + 1))
    SECTIONS+=("SKIP  ${label}  (${reason})")
}

# 1. Low-level tests -- always runnable.
run_section "Low-level backend tests" \
    python3 test_backends_lowlevel.py

# 2. Self-contained pipeline synthesizer -- always runnable, no science deps.
run_section "Pipeline-output synthesizer (htcondor / glue / slurm)" \
    python3 synthesize_pipeline_outputs.py

# 3. Real pipeline drivers.  These require the RIFT science stack (lalsim,
#    scipy, ...) AND ideally the htcondor / glue python bindings for the
#    htcondor and glue backends respectively.
run_real() {
    local script="$1"
    local label="$2"
    if ! python3 -c "import RIFT.lalsimutils" 2>/dev/null; then
        skip_section "${label}" "RIFT/lalsimutils not importable in this environment"
        return
    fi
    run_section "${label}" bash "${script}"
}

run_real demo_create_event_parameter_pipeline_BasicIteration.sh \
    "create_event_parameter_pipeline_BasicIteration (3 backends)"
run_real demo_create_eos_posterior_pipeline.sh \
    "create_eos_posterior_pipeline (3 backends)"
run_real demo_util_RIFT_pseudo_pipe.sh \
    "util_RIFT_pseudo_pipe (3 backends)"

echo
echo "============================================================"
echo "Summary"
echo "============================================================"
for s in "${SECTIONS[@]}"; do
    echo "  ${s}"
done
echo
echo "  ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"

if [[ "${FAIL}" -gt 0 ]]; then
    exit 1
fi
exit 0
