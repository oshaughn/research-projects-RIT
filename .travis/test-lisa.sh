#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q \
  MonteCarloMarginalizeCode/Code/test/test_lisa_auxiliary_imports.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_response_import.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_lalsimutils_compat.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_demo_contract.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_helper_contract.py \
  MonteCarloMarginalizeCode/Code/test/test_lisa_synthetic_demo.py
