#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN_DIR="${CODE_DIR}/bin"
CHECK_SCRIPT="${CODE_DIR}/RIFT/LISA/run_checks/plot_RIFT.py"

PYTHON_BIN="${RIFT_LISA_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

WORKDIR="${RIFT_LISA_WORKDIR:-/tmp/rift-lisa-end-to-end-$(date +%s)}"
BUNDLE_DIR="${WORKDIR}/event_0"
RUNDIR="${WORKDIR}/analysis_event_0"
RUN_ILE="${RIFT_LISA_RUN_ILE:-1}"
VARY_SKY="${RIFT_LISA_VARY_SKY:-0}"

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
export RIFT_HYPERPIPELINE_FORMAT=1

"${PYTHON_BIN}" "${SCRIPT_DIR}/make_synthetic_lisa_inputs.py" \
  --output-directory "${BUNDLE_DIR}" \
  --duration 1024 \
  --deltaT 4 \
  "$@"

"${PYTHON_BIN}" "${SCRIPT_DIR}/make_lisa_psds.py" \
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

if [[ "${RUN_ILE}" == "0" ]]; then
  echo "Rendered LISA end-to-end demo products in ${WORKDIR}"
  exit 0
fi

"${PYTHON_BIN}" "${BIN_DIR}/integrate_likelihood_extrinsic_batchmode_lisa" \
  --LISA \
  --h5-frame-FD \
  --time-marginalization \
  --lisa-fixed-sky 1 \
  --ecliptic-longitude "${ECLIPTIC_LONGITUDE}" \
  --ecliptic-latitude "${ECLIPTIC_LATITUDE}" \
  --lisa-reference-time 0 \
  --lisa-reference-frequency "${FREF}" \
  --data-integration-window-half 8 \
  --modes "[(2,2)]" \
  --cache-file "${CACHE_FILE}" \
  --channel-name A=fake_strain \
  --channel-name E=fake_strain \
  --channel-name T=fake_strain \
  --psd-file "A=${BUNDLE_DIR}/A_psd.xml.gz" \
  --psd-file "E=${BUNDLE_DIR}/E_psd.xml.gz" \
  --psd-file "T=${BUNDLE_DIR}/T_psd.xml.gz" \
  --fmin-template "${FMIN}" \
  --fmin-ifo "A=${FMIN}" \
  --fmin-ifo "E=${FMIN}" \
  --fmin-ifo "T=${FMIN}" \
  --fmax "${FMAX}" \
  --reference-freq "${FREF}" \
  --srate "${SRATE}" \
  --l-max 2 \
  --approx IMRPhenomD \
  --mass1 "${MASS1}" \
  --mass2 "${MASS2}" \
  --spin1z "${SPIN1Z}" \
  --spin2z "${SPIN2Z}" \
  --d-max 5000 \
  --d-min 1 \
  --n-eff 2 \
  --n-max 40 \
  --n-chunk 20 \
  --save-P 1 \
  --no-adapt \
  --internal-use-lnL \
  --sampler-method AV \
  --inclination "${INCLINATION}" \
  --phi-orb "${PHIREF}" \
  --distance "${DISTANCE_MPC:-1000}" \
  --right-ascension 0 \
  --declination 0 \
  --internal-hard-fail-on-error \
  --output-file "${WORKDIR}/lisa_end_to_end"

test -s "${WORKDIR}/lisa_end_to_end_0_.dat"
"${PYTHON_BIN}" "${CHECK_SCRIPT}" "${WORKDIR}/lisa_end_to_end_0_.dat" --json \
  > "${WORKDIR}/lisa_end_to_end_summary.json"

echo "LISA end-to-end demo products: ${WORKDIR}"
echo "LISA end-to-end ILE output: ${WORKDIR}/lisa_end_to_end_0_.dat"
echo "LISA end-to-end summary: ${WORKDIR}/lisa_end_to_end_summary.json"
