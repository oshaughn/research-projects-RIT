#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEMO_DIR="${CODE_DIR}/demo/rift/lisa"
BIN_DIR="${CODE_DIR}/bin"

PYTHON_BIN="${RIFT_LISA_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

WORKDIR="${RIFT_PP_LISA_WORKDIR:-/tmp/rift-pp-lisa-known-sky-$(date +%s)}"
BUNDLE_DIR="${WORKDIR}/event_0"
RUNDIR="${WORKDIR}/analysis_event_0"
RUN_ILE="${RIFT_PP_LISA_RUN_ILE:-0}"
VARY_SKY="${RIFT_PP_LISA_VARY_SKY:-0}"
SKY_ARGS=()
if [[ "${VARY_SKY}" == "1" ]]; then
  SKY_ARGS=(--lisa-vary-sky --lisa-grid-size 3 --lisa-sky-grid-width 0.01)
else
  SKY_ARGS=(--lisa-grid-size 1)
fi

mkdir -p "${BUNDLE_DIR}"

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${BIN_DIR}:${PATH}"
export MPLCONFIGDIR="${WORKDIR}/.matplotlib"

"${PYTHON_BIN}" "${DEMO_DIR}/make_synthetic_lisa_inputs.py" \
  --output-directory "${BUNDLE_DIR}" \
  --duration 1024 \
  --deltaT 4 \
  "$@"

"${PYTHON_BIN}" "${DEMO_DIR}/make_lisa_psds.py" \
  --output-directory "${BUNDLE_DIR}" \
  --fmax 0.125 \
  --npts 513 \
  --write-ascii

set -a
source "${BUNDLE_DIR}/synthetic-params.env"
set +a

"${PYTHON_BIN}" "${BIN_DIR}/util_RIFT_pseudo_pipe.py" \
  --lisa-known-sky \
  --use-rundir "${RUNDIR}" \
  --approx IMRPhenomD \
  --event-time 0 \
  --ecliptic-longitude "${ECLIPTIC_LONGITUDE}" \
  --ecliptic-latitude "${ECLIPTIC_LATITUDE}" \
  --lisa-cache-file "${CACHE_FILE}" \
  --lisa-channel-name A=fake_strain \
  --lisa-channel-name E=fake_strain \
  --lisa-channel-name T=fake_strain \
  --lisa-psd-file "A=${BUNDLE_DIR}/A_psd.xml.gz" \
  --lisa-psd-file "E=${BUNDLE_DIR}/E_psd.xml.gz" \
  --lisa-psd-file "T=${BUNDLE_DIR}/T_psd.xml.gz" \
  --lisa-srate "${SRATE}" \
  --lisa-fmin-template "${FMIN}" \
  --lisa-fmax "${FMAX}" \
  --lisa-reference-freq "${FREF}" \
  "${SKY_ARGS[@]}" \
  --lisa-n-iterations 1 \
  --lisa-n-samples-per-job 1 \
  --internal-ile-request-memory 1024 \
  --internal-cip-request-memory 1024

if [[ "${RUN_ILE}" == "1" ]]; then
  RIFT_LISA_WORKDIR="${BUNDLE_DIR}" "${DEMO_DIR}/run_lisa_synthetic_ile.sh" \
    --duration 1024 \
    --deltaT 4
fi

echo "LISA PP known-sky bundle: ${BUNDLE_DIR}"
echo "LISA PP known-sky run: ${RUNDIR}"
