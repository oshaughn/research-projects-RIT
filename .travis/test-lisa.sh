#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${RIFT_LISA_PYTHON:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

"${PYTHON_BIN}" -m pytest -q \
  MonteCarloMarginalizeCode/Code/test/test_lisa_auxiliary_imports.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_response_import.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_lalsimutils_compat.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_run_checks.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_demo_contract.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_helper_contract.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_pseudo_pipe_contract.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_pp_surface.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_synthetic_demo.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_fairdraw_weights.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_l0_rescue.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_sampler_plumbing.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_av_state.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_driver_drift.py
