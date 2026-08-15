#! /bin/bash
set -euo pipefail

# The RIFT Asimov plugin is currently developed and validated against the
# Asimov 0.5 series.  This test skips cleanly for unsupported/future series
# from inside pytest, so developers can preflight 0.6/0.7 environments without
# editing the test.
python -m pytest -q MonteCarloMarginalizeCode/Code/test/asimov_integration

# Bootstrap-source selection ("scheduler: bootstrap file:") is driven against a stub
# production rather than a project on disk, so it lives outside asimov_integration/.
# It still needs asimov importable, and this is the only lane that installs it, so run
# it here: otherwise the suite is never invoked and the behaviour can regress while the
# required checks stay green.
if python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_asimov_bootstrap_source.py; then
    :
else
    status=$?
    if [[ ${status} -ne 5 ]]; then
        exit "${status}"
    fi
    echo "Optional Asimov bootstrap suite collected no runnable tests; continuing."
fi
