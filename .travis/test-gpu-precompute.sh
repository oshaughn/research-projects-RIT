#!/usr/bin/env bash
# CPU coverage of the optional GPU path: run each file in a fresh process so
# synthetic import stubs and directory-local conftest modules cannot leak.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${RIFT_JAX_PYTHON:-${PYTHON:-python}}"
export PYTHONPATH="$PWD/MonteCarloMarginalizeCode/Code${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
C="MonteCarloMarginalizeCode/Code/test"
FILES=(
  "$C/gpu_precompute/test_array_contract.py"
  "$C/gpu_precompute/test_basis_reuse.py"
  "$C/gpu_precompute/test_classic_compound_waveform_forwarding.py"
  "$C/gpu_precompute/test_cross_driver_parity.py"
  "$C/gpu_precompute/test_device_handoff_adversarial.py"
  "$C/gpu_precompute/test_dispatch_contract.py"
  "$C/gpu_precompute/test_highlevel_integration.py"
  "$C/gpu_precompute/test_lal_fft_oracle.py"
  "$C/gpu_precompute/test_smoke_config_contract.py"
  "$C/gpu_precompute/test_streamed_v_weighting.py"
  "$C/test_gpu_jax_handoff.py"
  "$C/waveforms/test_gpu_legacy_compat.py"
  "$C/waveforms/test_gpu_waveform.py"
)
report_dir="$(mktemp -d)"
for file in "${FILES[@]}"; do
  report="$report_dir/$(basename "$file").xml"
  "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider --junit-xml="$report" "$file"
  # CUDA and optional waveform backends may skip; every file must still run
  # at least one CPU assertion. Pytest success alone also permits all-skipped.
  "${PYTHON_BIN}" - "$report" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = list(ET.parse(sys.argv[1]).iter("testcase"))
passed = sum(not any(case.find(tag) is not None
                     for tag in ("skipped", "failure", "error")) for case in cases)
if passed < 1:
    raise SystemExit("GPU precompute CPU gate ran no passing tests: " + sys.argv[1])
PY
done
