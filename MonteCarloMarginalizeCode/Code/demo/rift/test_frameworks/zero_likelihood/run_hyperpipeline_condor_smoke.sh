#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BIN_DIR="${CODE_DIR}/bin"

PYTHON_BIN="${RIFT_ZERO_LIKE_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x /Users/rossma/miniconda3/envs/junior_tools/bin/python ]]; then
    PYTHON_BIN=/Users/rossma/miniconda3/envs/junior_tools/bin/python
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

WORKDIR="${RIFT_ZERO_LIKE_WORKDIR:-/tmp/rift-zero-likelihood-smoke-$(date +%s)}"
SUBMIT="${RIFT_ZERO_LIKE_SUBMIT:-1}"
WAIT="${RIFT_ZERO_LIKE_WAIT:-1}"

mkdir -p "${WORKDIR}"
cp "${SCRIPT_DIR}/seed-grid.dat" "${WORKDIR}/seed-grid.dat"

cat > "${WORKDIR}/args_ile.txt" <<'EOF'
X --n-max 2 --n-eff 2 --save-samples
EOF

cat > "${WORKDIR}/args_cip.txt" <<'EOF'
X --parameter m1 --parameter m2 --parameter-range [10,80] --parameter-range [10,80] --n-output-samples 4
EOF

cat > "${WORKDIR}/args_test.txt" <<'EOF'
X --parameter m1 --parameter m2 --method KS_1d --threshold 999 --always-succeed
EOF

cat > "${WORKDIR}/stub_ile.py" <<EOF
#!${PYTHON_BIN}
import runpy
runpy.run_path("${SCRIPT_DIR}/stub_ile.py", run_name="__main__")
EOF

cat > "${WORKDIR}/stub_cip.py" <<EOF
#!${PYTHON_BIN}
import runpy
runpy.run_path("${SCRIPT_DIR}/stub_cip.py", run_name="__main__")
EOF

chmod +x "${WORKDIR}/stub_ile.py" "${WORKDIR}/stub_cip.py"

export RIFT_HYPERPIPELINE_FORMAT=1
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${BIN_DIR}:${PATH}"

"${PYTHON_BIN}" "${BIN_DIR}/create_event_parameter_pipeline_BasicIteration" \
  --working-directory "${WORKDIR}" \
  --ile-exe "${WORKDIR}/stub_ile.py" \
  --cip-exe "${WORKDIR}/stub_cip.py" \
  --test-exe "${BIN_DIR}/convergence_test_samples.py" \
  --input-grid "${WORKDIR}/seed-grid.dat" \
  --ile-args "${WORKDIR}/args_ile.txt" \
  --cip-args "${WORKDIR}/args_cip.txt" \
  --test-args "${WORKDIR}/args_test.txt" \
  --n-iterations 2 \
  --n-samples-per-job 2 \
  --n-copies 1 \
  --ile-n-events-to-analyze 1 \
  --ile-request-disk 100M \
  --general-request-disk 100M \
  --cip-request-disk 100M \
  --request-memory-ILE 256 \
  --request-memory-CIP 256

echo "Generated smoke-test DAG in ${WORKDIR}"

if [[ "${SUBMIT}" != 1 ]]; then
  exit 0
fi

(
  cd "${WORKDIR}"
  condor_submit_dag -force marginalize_intrinsic_parameters_BasicIterationWorkflow.dag
)

if [[ "${WAIT}" != 1 ]]; then
  exit 0
fi

DAG_LOG="${WORKDIR}/marginalize_intrinsic_parameters_BasicIterationWorkflow.dag.dagman.out"
echo "Waiting for DAGMan completion in ${DAG_LOG}"
while true; do
  if grep -q "All jobs Completed!" "${DAG_LOG}" 2>/dev/null; then
    break
  fi
  if grep -q "DAG_STATUS_NODE_FAILED" "${DAG_LOG}" 2>/dev/null; then
    tail -120 "${DAG_LOG}" >&2
    exit 1
  fi
  sleep 5
done

test -s "${WORKDIR}/all.net"
test -s "${WORKDIR}/consolidated_0.composite"
test -s "${WORKDIR}/consolidated_1.composite"
test -s "${WORKDIR}/overlap-grid-1.dat"
test -s "${WORKDIR}/overlap-grid-2.dat"
test -L "${WORKDIR}/posterior_samples-1.dat"
test -L "${WORKDIR}/posterior_samples-2.dat"

echo "Zero-likelihood workflow smoke test passed in ${WORKDIR}"
