#! /bin/bash
set -euo pipefail

# The RIFT Asimov plugin is currently developed and validated against the
# Asimov 0.5 series.  This test skips cleanly for unsupported/future series
# from inside pytest, so developers can preflight 0.6/0.7 environments without
# editing the test.
python -m pytest -q MonteCarloMarginalizeCode/Code/test/asimov_integration
