#!/usr/bin/env bash
# CPU gate for unit suites that were reachable from NO CI job until this script landed.
#
# WHY.  On 2026-09-03 an audit of every test_*.py under Code/ found 86 of 200 named by no
# workflow and no .travis script (see .travis/ci_roster.txt and .travis/test-ci-roster.py for
# the census that now keeps that number honest).  Most of those 86 should stay out -- they are
# hand-run studies, plotting demos, or scripts importing pre-package flat modules that have not
# existed since RIFT was packaged.  The files below are the ones that should NOT: they are
# ordinary pytest suites, numpy/scipy/lal/sklearn only, that collect and PASS in seconds, and
# they guard things that regress SILENTLY -- an evidence accounting, a seeding path, a
# distance grid, a container manifest, a parameter port.  A wrong number there is still a
# plausible number.
#
# Every file listed here was run individually on CIT (IGWN conda python 3.11, numpy 1.26.4,
# lal 7.7.0) before it was added; the measured collection counts are the floors below.
#
# SHAPE.  Modelled on .travis/test-slowrot.sh, and it keeps that script's defences, because
# the trap it documents is live in this very set: several files elsewhere in these directories
# collect ZERO items and pytest exits 5, "no tests ran", which reads as a pass.  Membership
# here is an explicit FILES manifest rather than the marker line used by
# .travis/test-q-window-stencil.sh: this set spans six unrelated subject areas with no shared
# filename pattern, so a marker's scope check would have nothing to scope over.  The two
# mechanisms answer different questions -- see the PR body.
set -uo pipefail
# NOTE: deliberately no -e, matching test-slowrot.sh.  Every command's rc is handled below.

cd "$(dirname "$0")/.." || { echo "test-core-units.sh: cannot cd to repo root" >&2; exit 1; }

# INVARIANT: test THIS CHECKOUT, never an installed build.  Must PREPEND -- appending lets a
# caller's PYTHONPATH win.
export PYTHONPATH="$PWD/MonteCarloMarginalizeCode/Code${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${RIFT_COREUNIT_PYTHON:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

# Guard every probe whose pass condition is empty output: a missing interpreter plus a
# redirected stderr is indistinguishable from a clean result.
"${PYTHON_BIN}" -c 'import pytest' || { echo "test-core-units.sh: pytest unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import numpy, scipy; print("numpy", numpy.__version__)' \
  || { echo "test-core-units.sh: numpy/scipy unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import lal; print("lal", lal.__version__)' \
  || { echo "test-core-units.sh: lal unavailable" >&2; exit 1; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

C="MonteCarloMarginalizeCode/Code"

FILES=(
  # -- calibration marginalization (module-level suites the calmarg gate never picked up)
  "$C/RIFT/calmarg/test_cal_mc_error.py"
  "$C/RIFT/calmarg/test_seed_fallback.py"
  "$C/test/test_calmarg_calibration.py"
  # -- likelihood dispatch
  "$C/RIFT/likelihood/test_td_dispatch_epoch.py"
  "$C/test/test_ile_scalar_edge_cases.py"
  "$C/test/test_srate_resample_time_marginalization.py"
  # -- integrators: seeding, allocation, weight derivation
  "$C/test/integrators/test_convergence_sample_order.py"
  "$C/test/integrators/test_gmm_adaptive.py"
  "$C/test/integrators/test_portfolio_gmm_member_trains.py"
  "$C/test/integrators/test_portfolio_restrict_and_warm.py"
  "$C/test/integrators/test_replica_pooling.py"
  "$C/test/integrators/test_rvs_weight_derivation.py"
  "$C/test/integrators/test_seeding_public_paths.py"
  "$C/test/integrators/test_seeding_reproducibility.py"
  "$C/test/test_mc_error.py"
  # -- CIP / evidence / distance export
  "$C/test/test_cip_evidence_consolidation.py"
  "$C/test/test_cip_pipeline.py"
  "$C/test/test_distance_grid.py"
  "$C/test/test_distance_tail.py"
  "$C/test/test_dslice_device_native.py"
  # -- hyperpipe (paper4 area; the hydra leg is rostered OPTDEP, not here)
  "$C/test/hyperpipe/tests/test_config.py"
  "$C/test/hyperpipe/tests/test_coords.py"
  "$C/test/hyperpipe/tests/test_drivers.py"
  "$C/test/hyperpipe/tests/test_marg_list.py"
  "$C/test/test_hyperpipeline_io.py"
  # -- packaging / config contracts / waveform conventions
  "$C/test/test_advanced_parameter_ports.py"
  "$C/test/test_container_manifest.py"
  "$C/test/test_lisa_ini_contract.py"
  "$C/test/test_tracer_placement_gp.py"
  "$C/test/waveforms/test_uv_symmetry.py"
)

# A manifest entry that stops existing is a SILENT no-op: the gate keeps passing while
# covering less.  Same defence as test-slowrot.sh's DESELECT-still-resolves check.
missing=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "test-core-units.sh: manifest names $f, which does not exist." >&2; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "  Fix the manifest or restore the file; left as is it covers nothing." >&2; exit 1; }

# PER-FILE collection floor of 1.  A file that collects nothing is the exit-5 trap arriving
# through the front door: inside a multi-file run pytest's exit 5 never appears at all, so it
# has to be checked per file.
echo "== per-file collection floor =="
floor_rc=0
for f in "${FILES[@]}"; do
  n=$("${PYTHON_BIN}" -m pytest --collect-only -q -p no:cacheprovider "$f" 2>/dev/null | grep -c '::')
  printf '  %-72s %3d\n' "$f" "$n"
  if [ "$n" -lt 1 ]; then
    echo "test-core-units.sh: $f collects 0 tests." >&2
    echo "  pytest exits 5 on that ('no tests ran'), which reads as a pass.  Either it is not" >&2
    echo "  a pytest target and belongs in .travis/ci_roster.txt, or its entry points broke." >&2
    floor_rc=1
  fi
done
[ "$floor_rc" -eq 0 ] || exit 1

# Pinned TOTAL floor, so a renamed file or a dropped test_* entry point goes red rather than
# green-on-fewer-tests.  MEASURED 2026-09-03 on CIT with the IGWN conda python (3.11, numpy
# 1.26.4, scipy 1.14.1, lal 7.7.0), whole manifest in one run: 296 collected, 284 passed,
# 12 skipped (11 pytest.skip + 1 xfail), ~55 s.  (Was 278/266 before test_replica_pooling.py
# and test_marg_list.py joined the manifest -- both were rostered BROKEN until their defects
# were fixed.  RAISE these when files are added: a floor left at the old value passes while
# covering less, which is the failure this gate exists to catch.)
EXPECTED_TESTS=296
# Outcomes, not just exit status: a collection floor cannot see a test that collects, runs and
# asserts nothing, and a pytest.skip can quietly absorb a lost gate.  The 12 skips are
# environment legs -- cupy in test_seeding_reproducibility, device legs in
# test_dslice_device_native, and the xfail in test_uv_symmetry -- and a GitHub runner has no
# GPU either, so they skip there too.
#
# CONFIRMED ON A RUNNER.  The first CI run of this job (PR #243, ubuntu-latest, python 3.10 +
# editable install) reported the same 278 / 266 / 12, in 24.7 s.  So these floors are exact on
# both stacks, not merely the CIT numbers copied across, and a future divergence is a real
# change rather than an environment difference to be explained away.
EXPECTED_PASSED=284
MAX_SKIPPED=12

junit="$(mktemp -t core-units-junit-XXXXXX.xml)"
echo "== running =="
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider --durations=10 --junit-xml="${junit}" "${FILES[@]}"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "test-core-units.sh: pytest exited ${rc}" >&2
  [ "$rc" -eq 5 ] && echo "  Exit 5 is 'no tests ran'.  That is a FAILURE here, not a pass." >&2
  rm -f "${junit}"
  exit 1
fi

read -r TOT FAIL ERR SKIP < <("${PYTHON_BIN}" - "${junit}" <<'PY'
import sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
s = r if r.tag == 'testsuite' else r.find('testsuite')
g = lambda k: int(s.get(k, 0))
print(g('tests'), g('failures'), g('errors'), g('skipped'))
PY
)
rm -f "${junit}"
PASSED=$(( TOT - FAIL - ERR - SKIP ))
echo "== outcomes: ${TOT} collected, ${PASSED} passed, ${SKIP} skipped, ${FAIL} failed, ${ERR} errored =="

out_rc=0
if [ "${TOT}" -lt "${EXPECTED_TESTS}" ]; then
  echo "test-core-units.sh: collected ${TOT} tests, expected at least ${EXPECTED_TESTS}." >&2
  echo "  A file was renamed, or a test_* entry point was dropped.  Restore it, or lower this" >&2
  echo "  floor DELIBERATELY in the same commit that removes the tests." >&2
  out_rc=1
fi
if [ "${PASSED}" -lt "${EXPECTED_PASSED}" ]; then
  echo "test-core-units.sh: only ${PASSED} PASSED, expected at least ${EXPECTED_PASSED}." >&2
  out_rc=1
fi
if [ "${SKIP}" -gt "${MAX_SKIPPED}" ]; then
  echo "test-core-units.sh: ${SKIP} SKIPPED, at most ${MAX_SKIPPED} expected." >&2
  echo "  A pytest.skip can absorb a lost gate without failing anything.  If the new skip is" >&2
  echo "  legitimate, raise MAX_SKIPPED here and say which test and why." >&2
  out_rc=1
fi
[ "$out_rc" -eq 0 ] || exit 1

echo "core unit gate: PASS"
