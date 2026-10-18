#!/usr/bin/env bash
# Critical O4d -> O4c ports: scientific references and adversarial regressions.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PYTHONPATH="$root/MonteCarloMarginalizeCode/Code${PYTHONPATH:+:$PYTHONPATH}"
export GW_SURROGATE="${GW_SURROGATE:-}"
junit="$(mktemp "${TMPDIR:-/tmp}/rift-o4c-ports.XXXXXX.xml")"
trap 'rm -f "$junit"' EXIT
python -m pytest -q --junitxml="$junit" \
    MonteCarloMarginalizeCode/Code/test/test_o4c_critical_ports.py \
    MonteCarloMarginalizeCode/Code/test/test_vectorized_lal_tools_split.py \
    MonteCarloMarginalizeCode/Code/test/test_noloop_accumulator_shapes.py \
    MonteCarloMarginalizeCode/Code/test/test_fd_end_taper_cap.py \
    MonteCarloMarginalizeCode/Code/test/test_std_and_conj_hlmoff_sphharm.py \
    MonteCarloMarginalizeCode/Code/RIFT/likelihood/test_precompute_crossterm_batching.py
python - "$junit" <<'PY'
import os
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).findall('.//testcase')
skipped = sum(c.find('skipped') is not None for c in cases)
passed = sum(all(c.find(n) is None for n in ('skipped', 'failure', 'error')) for c in cases)
max_skipped = 0 if os.environ.get('RIFT_CI_REQUIRE_GPU') == '1' else 1
if len(cases) < 84 or passed < 83 or skipped > max_skipped:
    raise SystemExit('Critical ports gate lost coverage: {} cases, {} passed, {} skipped'.format(
        len(cases), passed, skipped))
print('Critical ports gate: {} cases, {} passed, {} skipped'.format(len(cases), passed, skipped))
PY
