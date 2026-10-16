#! /bin/bash
set -euo pipefail

# Exercise plugin discovery, real ASIMOV project/ledger application, the
# frozen RIFT build command, and the cross-version adapter contracts.
python -m pytest -q \
    MonteCarloMarginalizeCode/Code/test/asimov_integration \
    MonteCarloMarginalizeCode/Code/test/test_asimov_compatibility.py
