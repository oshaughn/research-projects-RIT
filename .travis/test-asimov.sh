#! /bin/bash
set -euo pipefail

# The RIFT Asimov plugin is validated against the legacy 0.5 series and the
# plugin-based 0.7 series. Unsupported API series skip cleanly in pytest.
# Bootstrap-source selection ("scheduler: bootstrap file:") is driven against a stub
# production rather than a project on disk, so it lives outside asimov_integration/.
# It still needs asimov importable, and this is the only lane that installs it, so run
# it here: otherwise the suite is never invoked and the behaviour can regress while the
# required checks stay green.
python -m pytest -q \
    MonteCarloMarginalizeCode/Code/test/asimov_integration \
    MonteCarloMarginalizeCode/Code/test/test_asimov_compatibility.py \
    MonteCarloMarginalizeCode/Code/test/test_asimov_bootstrap_source.py
