#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN_DIR="${CODE_DIR}/bin"

PYTHON_BIN="${RIFT_LISA_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

WORKDIR="${RIFT_LISA_WORKDIR:-/tmp/rift-lisa-synthetic-ile-$(date +%s)}"
RUN_ILE="${RIFT_LISA_RUN_ILE:-1}"

mkdir -p "${WORKDIR}"

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${BIN_DIR}:${PATH}"
export MPLCONFIGDIR="${WORKDIR}/.matplotlib"

"${PYTHON_BIN}" "${SCRIPT_DIR}/make_synthetic_lisa_inputs.py" \
  --output-directory "${WORKDIR}" \
  "$@"

set -a
source "${WORKDIR}/synthetic-params.env"
set +a

"${PYTHON_BIN}" "${BIN_DIR}/helper_LISA_Events.py" \
  --working-directory "${WORKDIR}" \
  --cache-file "${CACHE_FILE}" \
  --psd-file "A=${A_PSD}" \
  --psd-file "E=${E_PSD}" \
  --psd-file "T=${T_PSD}" \
  --mass1 "${MASS1}" \
  --mass2 "${MASS2}" \
  --spin1z "${SPIN1Z}" \
  --spin2z "${SPIN2Z}" \
  --ecliptic-latitude "${ECLIPTIC_LATITUDE}" \
  --ecliptic-longitude "${ECLIPTIC_LONGITUDE}" \
  --fmin-template "${FMIN}" \
  --fmax "${FMAX}" \
  --reference-freq "${FREF}" \
  --lisa-reference-frequency "${FREF}" \
  --data-integration-window-half 8 \
  --srate "${SRATE}" \
  --d-min 1 \
  --d-max 5000 \
  --n-eff 2 \
  --n-max 40 \
  --n-chunk 20 \
  --save-P 1 \
  --grid-size 1 \
  --n-iterations 1 \
  --n-samples-per-job 1 \
  --request-memory-ILE 1024 \
  --request-memory-CIP 1024

if [[ "${RUN_ILE}" == "0" ]]; then
  echo "Generated synthetic LISA analysis inputs in ${WORKDIR}"
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
  --psd-file "A=${A_PSD}" \
  --psd-file "E=${E_PSD}" \
  --psd-file "T=${T_PSD}" \
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
  --output-file "${WORKDIR}/lisa_ile"

test -s "${WORKDIR}/lisa_ile_0_.dat"
echo "Synthetic LISA ILE run wrote ${WORKDIR}/lisa_ile_0_.dat"
