#!/bin/bash
#
# Lightweight smoke test for RIFT.simulation_manager.
# Runs only the no-schedd portions: in-memory archive, on-disk archive,
# queue FSM, and (when glue.pipeline is available) a build_master_job +
# generate_dag_for_all_ready_simulations round trip. Does NOT submit
# anything to a real schedd.
#
# The live-condor end-to-end test lives in test-simulation-manager-condor.sh
# and is gated on RIFT_TEST_CONDOR=1.

set -e

# Legacy v1 simulation_manager smoke tests.
python3 -m pytest -v MonteCarloMarginalizeCode/Code/test/test_simulation_manager.py

# v2 archive unit tests (database.py + queues + admin operations).
python3 -m pytest -v MonteCarloMarginalizeCode/Code/test/test_database.py

# In-package tests under RIFT/simulation_manager/tests/. These were not
# collected by anything before, so nearby_reuse and the condor transfer
# hooks shipped without CI coverage.
python3 -m pytest -v MonteCarloMarginalizeCode/Code/RIFT/simulation_manager/tests/
