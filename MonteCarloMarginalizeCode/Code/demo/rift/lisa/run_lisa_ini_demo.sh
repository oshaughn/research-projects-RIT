#!/usr/bin/env bash
# Render the LISA toy demo through the PRODUCTION-INI entry point
# (util_RIFT_pseudo_pipe.py --use-ini), to show the .ini path drives the same
# workflow as the --lisa-* CLI flags.  Builds synthetic A/E/T inputs, instantiates
# BBH_lisa_demo.ini against that bundle, and renders the CEPP DAG.  Does NOT submit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN_DIR="${CODE_DIR}/bin"

PYTHON_BIN="${RIFT_LISA_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then PYTHON_BIN="$(command -v python3)"; fi

WORKDIR="${RIFT_LISA_WORKDIR:-/tmp/rift-lisa-ini-$(date +%s)}"
BUNDLE_DIR="${WORKDIR}/event_0"
RUNDIR="${WORKDIR}/analysis_event_0"
mkdir -p "${BUNDLE_DIR}"

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${BIN_DIR}:${PATH}"
export RIFT_HYPERPIPELINE_FORMAT=1

# Tiny synthetic A/E/T inputs (real grid-matched PSD + lisa.cache + frames).
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_synthetic_lisa_inputs.py" \
  --output-directory "${BUNDLE_DIR}" --duration 16384 --deltaT 4 \
  --fmin 1e-4 --distance-mpc 20000 "$@"

# Instantiate the template ini against this bundle (ConfigParser has no path
# interpolation, so substitute the placeholder).
INI="${WORKDIR}/BBH_lisa_demo.ini"
sed "s#__BUNDLE_DIR__#${BUNDLE_DIR}#g" "${SCRIPT_DIR}/BBH_lisa_demo.ini" > "${INI}"

# Render the workflow from the ini.
"${PYTHON_BIN}" "${BIN_DIR}/util_RIFT_pseudo_pipe.py" \
  --use-ini "${INI}" --use-rundir "${RUNDIR}"

echo "=== rendered LISA ini workflow in ${RUNDIR} ==="
ls "${RUNDIR}"/*.dag "${RUNDIR}"/args_ile.txt "${RUNDIR}"/args_cip_list.txt
