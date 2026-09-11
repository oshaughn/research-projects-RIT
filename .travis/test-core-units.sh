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
# The original manifest was run file by file on CIT (IGWN conda python 3.11, numpy 1.26.4,
# lal 7.7.0) before it was added; the measured collection counts are the floors below.  Later
# entries are verified by this gate itself, which collects every file individually before the
# combined run, so an addition that collects nothing or fails is caught here rather than
# trusted on a quoted number.
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
  "$C/RIFT/likelihood/test_precompute_crossterm_batching.py"
  "$C/test/test_ile_scalar_edge_cases.py"
  "$C/test/test_mcsamplerGPU_cdf_inverse_scalar_probe.py"
  "$C/test/test_srate_resample_time_marginalization.py"
  "$C/test/test_vectorized_lal_tools_split.py"
  "$C/test/test_noloop_accumulator_shapes.py"
  # -- integrators: seeding, allocation, weight derivation
  "$C/test/integrators/test_convergence_sample_order.py"
  "$C/test/integrators/test_gmm_adaptive.py"
  "$C/test/integrators/test_portfolio_gmm_member_trains.py"
  "$C/test/integrators/test_portfolio_restrict_and_warm.py"
  # Wraps the five integrator studies as subprocesses (29 s).  They collect nothing
  # themselves -- pytest exits 5 on each -- so this is how their gates reach CI at all.
  "$C/test/integrators/test_integrator_studies.py"
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
  # -- promoted out of the roster after roster-verify-check caught its reason being false ON
  # THE RUNNER: it was OPTDEP needs:glue,htcondor, and with htcondor absent there it still
  # collected 15 and passed 15.  Confirmed locally with BOTH blocked via a sys.meta_path
  # finder: 15/15.  Its `import htcondor` / `from glue import pipeline` are capability
  # probes inside the tests, not requirements.  CIT has both, which is exactly why CIT could
  # not see this and a runner could.
  "$C/test/backends/test_backends_lowlevel.py"
  # -- promoted out of the roster: it was OPTDEP on prose ("unverified on a runner") and
  # .travis/test-roster-verify.py caught it collecting and passing COMPLETELY with nothing
  # missing.  (test_rimsky_integration.py was promoted alongside it and REVERTED: it
  # importorskips asimov, which CIT has and this job does not, so it collected 15 here and
  # 0 on the runner.  The per-file collection floor below caught that -- see the roster.)
  "$C/test/test_teobresums_compat.py"
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

# Pinned TOTAL floor, so a renamed file or a dropped test_* entry point goes red rather than
# green-on-fewer-tests.  MEASURED 2026-09-03 on CIT with the IGWN conda python (3.11, numpy
# 1.26.4, scipy 1.14.1, lal 7.7.0), whole manifest in one run: 319 collected,
# 307 passed, 12 skipped (11 pytest.skip + 1 xfail).
#
# COST: 363 s total on CIT for the 347-test manifest, of which the pytest run is ~145 s.  The
# rest is the per-file collection loop below -- one interpreter per manifest entry, each
# importing RIFT (lal, numpy, numba), so it grows linearly with the manifest and now dominates.
# That is the price of the exit-5 defence and it is worth paying, but it is why this job's
# timeout-minutes is 20 rather than something tight: measure before trimming the budget.
#
# History.  Both branches of a merge have now moved these numbers twice, and the arithmetic is
# NOT the way to combine them -- this gate collects every file individually before the combined
# run, so the merged floor is RE-MEASURED, never added up:
#   278/266  original manifest
#   296/284  + test_replica_pooling.py, test_marg_list.py (rostered BROKEN until fixed)
#   299/287  + test_vectorized_lal_tools_split.py (from rift_O4d, then 3 -> 7 tests: its
#            original comparison pitted the combined wrapper against the composition of its own
#            two halves, which after the split IS the wrapper, so it could not fail.  Rewritten
#            against a FROZEN copy of the pre-split bodies, parametrized per detector -- #256)
#   307/295  + test_noloop_accumulator_shapes.py (from rift_O4d, then 8 -> 9 tests: the added
#            one asserts the synthetic inputs actually exercise the data term, after the first
#            version put the Q window entirely outside the buffer and left kappa_sq identically
#            zero -- an assertion that held for the wrong reason)
#   312/300  rift_O4d #258, a pure FLOOR correction: the growth above had already landed in the
#            files while the floors still said 299/287, so 312 collected was clearing a 307
#            floor -- the silent under-coverage this gate exists to prevent, appearing in the
#            gate's own bookkeeping.  Worth noting how the two branches differed here: #258
#            reconstructed 312 by arithmetic (299 + 4 + 9) and this branch measured 347 with
#            those same files already grown, because it never adds.  Merging changed this
#            manifest's numbers by ZERO.
#   +        + test_teobresums_compat.py (OPTDEP on prose until test-roster-verify.py caught it
#            passing completely with nothing missing; its companion test_rimsky_integration.py
#            was promoted alongside and REVERTED -- see the note by the manifest entry)
#   +        + test_integrator_studies.py, wrapping the five integrator studies that collect
#            nothing themselves (+43 s, with --as-test)
#   +        + test_backends_lowlevel.py (OPTDEP needs:glue,htcondor until roster-verify-check
#            found it passing 15/15 on a runner with htcondor absent)
#   358/346  + test_mcsamplerGPU_cdf_inverse_scalar_probe.py (11 tests: mcsamplerGPU.cdf_inverse
#            fed odeint's float probe to len(x) pdfs; the t_ref wiring in all three ILE drivers)
#
# RAISE these when files are added: a floor left at the old value passes while covering less,
# which is the failure this gate exists to catch.
# DO NOT RAISE THESE TO THE RUNNER'S NUMBERS.  The GitHub runner reports 350 collected / 338
# passed for this same manifest, CIT reports 347 / 335, and the underlying results are IDENTICAL:
# both say "335 passed, 11 skipped, 1 xfailed".  The gap is pytest-subtests, which is in the
# runner's dependency closure and not in CIT's IGWN environment; with it, the three `subTest`
# blocks in test/backends/test_backends_lowlevel.py are counted as separate cases in the junit
# XML this gate parses.  Per-FILE collection is 347 on both, file by file.
#
# So the floors are pinned to the PLUGIN-FREE count.  That is the robust choice in the only
# direction that matters: 350 >= 347 passes today, and if pytest-subtests ever leaves the
# runner's closure the count falls back to 347 and still passes.  Pinning 350 would turn an
# unrelated dependency change into a red gate.
EXPECTED_TESTS=370
# Outcomes, not just exit status: a collection floor cannot see a test that collects, runs and
# asserts nothing, and a pytest.skip can quietly absorb a lost gate.  The 12 skips are
# environment legs -- cupy in test_seeding_reproducibility, device legs in
# test_dslice_device_native, and the xfail in test_uv_symmetry.
EXPECTED_PASSED=358
MAX_SKIPPED=12

# The floors must be INTEGERS, and this is checked rather than assumed.  `[ 347 -lt FOO ]` does
# not fail the build: bash prints "integer expression expected", returns 2, and the `if` is
# simply FALSE -- so a malformed floor silently disables the check it looks like it performs,
# and the gate goes green having compared nothing.  Observed: a placeholder left in during a
# merge resolution produced exactly that, and only the stderr line gave it away.  `set -u`
# catches a MISSING floor; nothing caught a malformed one until this did.
for _f in EXPECTED_TESTS EXPECTED_PASSED MAX_SKIPPED; do
  eval "_v=\${${_f}}"
  case "${_v}" in
    ''|*[!0-9]*)
      echo "test-core-units.sh: ${_f}='${_v}' is not a non-negative integer." >&2
      echo "  A non-numeric floor makes its comparison a silent no-op -- the gate would pass" >&2
      echo "  while checking nothing.  Set it to the MEASURED value." >&2
      exit 1
      ;;
  esac
done

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
