#!/bin/bash
#
# Live HTCondor end-to-end test for RIFT.simulation_manager.
#
# Submits a small DAG (3 trivial sims) to the local schedd, polls until the
# jobs complete, then re-instantiates the archive from disk and asserts every
# sim is 'complete' and every output file is retrievable.
#
# Gated on RIFT_TEST_CONDOR=1 so it does NOT run in CI by default. Run on a
# real submit host:
#
#     RIFT_TEST_CONDOR=1 ./.travis/test-simulation-manager-condor.sh
#
# Optional environment:
#     RIFT_TEST_CONDOR_TIMEOUT  -- seconds to wait for jobs (default 300)
#     RIFT_TEST_CONDOR_BASE     -- archive base dir (default: $TMPDIR/rift_simmgr)

set -e

if [[ "${RIFT_TEST_CONDOR}" != "1" ]]; then
    echo "Skipping live-condor end-to-end test (set RIFT_TEST_CONDOR=1 to enable)."
    exit 0
fi

# Sanity-check the condor environment.
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
        __import__(m)
        print('Using', m, 'bindings')
        sys.exit(0)
    except ImportError:
        continue
sys.exit(1)
" || {
    echo "ERROR: neither 'htcondor2' nor 'htcondor' python bindings available" >&2
    exit 1
}
python3 -c "from glue import pipeline" 2>/dev/null || {
    echo "ERROR: 'glue.pipeline' not available (pip install lscsoft-glue)" >&2
    exit 1
}

# Locate the test driver, which lives next to this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DRIVER="${SCRIPT_DIR}/_simmgr_condor_e2e.py"
if [[ ! -f "${DRIVER}" ]]; then
    echo "ERROR: missing driver ${DRIVER}" >&2
    exit 1
fi

TIMEOUT="${RIFT_TEST_CONDOR_TIMEOUT:-300}"
BASE="${RIFT_TEST_CONDOR_BASE:-${TMPDIR:-/tmp}/rift_simmgr_$$}"
echo "Running live-condor e2e in ${BASE} (timeout ${TIMEOUT}s)"
python3 "${DRIVER}" --base "${BASE}" --n-sims 3 --timeout "${TIMEOUT}"
