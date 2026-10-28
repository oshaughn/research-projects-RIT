#! /bin/bash
set -euo pipefail

# Exercise plugin discovery, real ASIMOV project/ledger application, the
# frozen RIFT build command, and the cross-version adapter contracts.
python -m pytest -q \
    MonteCarloMarginalizeCode/Code/test/asimov_integration \
    MonteCarloMarginalizeCode/Code/test/test_asimov_compatibility.py \
    MonteCarloMarginalizeCode/Code/test/test_asimov_bootstrap_source.py \
    MonteCarloMarginalizeCode/Code/test/test_eccentric_bootstrap.py \
    MonteCarloMarginalizeCode/Code/test/test_q_time_pregrid_rc11.py \
    MonteCarloMarginalizeCode/Code/test/test_q_time_pregrid_transport_rc11.py \
    MonteCarloMarginalizeCode/Code/test/test_native_integer_mass_ranges.py \
    MonteCarloMarginalizeCode/Code/test/test_native_frame_cache.py
