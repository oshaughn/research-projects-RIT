#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN_DIR="${CODE_DIR}/bin"

PYTHON_BIN="${RIFT_LISA_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

WORKDIR="${RIFT_LISA_WORKDIR:-/tmp/rift-lisa-zero-likelihood-$(date +%s)}"
RENDER_CEPP="${RIFT_LISA_RENDER_CEPP:-1}"

mkdir -p "${WORKDIR}"

export RIFT_HYPERPIPELINE_FORMAT=1
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${BIN_DIR}:${PATH}"

"${PYTHON_BIN}" "${BIN_DIR}/helper_LISA_Events.py" \
  --working-directory "${WORKDIR}" \
  --zero-likelihood \
  --grid-size 1 \
  --n-iterations 1 \
  --n-samples-per-job 1 \
  --request-memory-ILE 1024 \
  --request-memory-CIP 1024 \
  "$@"

echo "Generated LISA helper bundle in ${WORKDIR}"

if [[ "${RENDER_CEPP}" == "0" ]]; then
  exit 0
fi

"${PYTHON_BIN}" "${BIN_DIR}/create_event_parameter_pipeline_BasicIteration" \
  --ile-n-events-to-analyze 1 \
  --input-grid "${WORKDIR}/proposed-grid.dat" \
  --ile-exe "${BIN_DIR}/integrate_likelihood_extrinsic_batchmode_lisa" \
  --ile-args "${WORKDIR}/args_ile.txt" \
  --cip-args-list "${WORKDIR}/args_cip_list.txt" \
  --test-args "${WORKDIR}/args_test.txt" \
  --working-directory "${WORKDIR}" \
  --n-iterations 1 \
  --n-samples-per-job 1 \
  --n-copies 1 \
  --request-memory-ILE 1024 \
  --request-memory-CIP 1024 \
  --transfer-file-list "${WORKDIR}/helper_transfer_files.txt"

echo "Rendered LISA zero-likelihood CEPP DAG in ${WORKDIR}"
