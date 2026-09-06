#!/usr/bin/env bash
# CPU gate for the Q_lm sub-sample stencil / time-interpolation / log-sum-exp-offset
# tests, driven from ci.yml's q-window-stencil-check job.
#
# WHY THIS SCRIPT EXISTS.  Until it landed, this job's membership was a hand-maintained
# backslash-continued file list inside ci.yml.  Three PRs appended to that one list in
# two days and it conflicted twice; resolving such a conflict by taking one side rather
# than both silently UNREGISTERS the other side's test, and in this CI an unlisted test
# never runs and the job reports green forever.  That is coverage that looks like a guard
# and is not.
#
# It is modelled on .travis/test-slowrot.sh and keeps that script's defences, with ONE
# deliberate difference.  slowrot's membership is an explicit FILES array; here membership
# is declared by a MARKER LINE INSIDE EACH TEST FILE, so a PR adding a test to this area
# edits only its own new file and no shared line exists to conflict on.  Everything else
# -- the fail-closed manifest, the pinned floors, the junit OUTCOME assertion, the
# EXCLUDED list with a reason per entry -- follows slowrot deliberately.
#
# Five defences.  Do not simplify any of them into a bare `pytest <dir>`:
#
#   1. MARKER-BASED membership, not a directory glob.  A glob would sweep up files that
#      collect ZERO items, and pytest exits 5 on those -- "no tests ran" reads as a pass
#      in a log skim.  (test_slowrot_*.py in one of these very directories is five such
#      files; they belong to slowrot-check, not here.)
#   2. A fail-closed MANIFEST over the filename patterns this job owns: a file matching
#      one of them that carries no marker and is not explicitly EXCLUDED fails the job.
#      This is what makes "added a test and forgot to register it" RED instead of silent.
#   3. A PER-FILE collection floor of 1.  A registered file that collects nothing is the
#      exit-5 trap arriving through the front door; it must be visible, not green.
#   4. A pinned TOTAL collection floor, so a renamed file or a dropped test_* entry point
#      goes red instead of green-on-fewer-tests.
#   5. A hard fail on ANY nonzero pytest exit (5 included) plus a junit OUTCOME assertion
#      on tests that actually PASSED.  The collection floors count COLLECTION, which
#      cannot see a test that collects, runs, and asserts nothing.
#
# Needs numpy + lal only: no GPU, no jax, no numpyro.
set -uo pipefail
# NOTE: deliberately no -e, matching test-slowrot.sh.  Every command below has its rc
# handled explicitly so the failure messages stay specific; if you add one, guard it.

# The paths below are repo-relative, so anchor cwd rather than trusting the caller.
cd "$(dirname "$0")/.." || { echo "test-q-window-stencil.sh: cannot cd to repo root" >&2; exit 1; }

# INVARIANT: this gate always tests THIS CHECKOUT, never an installed build.  Must
# PREPEND -- appending lets a caller's PYTHONPATH win.  (test_interpolate_time_cli.py
# launches the RIFT scripts as real subprocesses, and they inherit this.)
export PYTHONPATH="$PWD/MonteCarloMarginalizeCode/Code${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------------
# DISCOVERY RUNS FIRST, before the dependency probes below.
#
# Nothing here needs pytest, numpy or lal: finding the registered files is a grep.  Doing
# it first is what lets this script answer RIFT_CI_GATE_LIST=1 (see below) for the
# repo-wide census in .travis/test-ci-roster.py, which runs as its own job with NO
# `needs: install` and so has none of those libraries.  Keep this above the probes.
CODEDIR="MonteCarloMarginalizeCode/Code"

# ---------------------------------------------------------------------------------
# MEMBERSHIP.  A test file joins this gate by carrying this line, on its own, verbatim:
#
#     # RIFT-CI-GATE: q-window-stencil
#
# The match is whole-line and fixed-string (grep -x -F), so prose mentioning the tag --
# including the explanatory line the registered files put directly underneath it -- does
# NOT register a file.  Search is limited to test_*.py under Code/, so a doc or a script
# quoting the tag cannot enrol itself either.
MARKER="# RIFT-CI-GATE: q-window-stencil"

mapfile -t FILES < <(grep -rlxF --include='test_*.py' -- "${MARKER}" "${CODEDIR}" 2>/dev/null | LC_ALL=C sort)

# LIST MODE.  Print the files this gate would run, one per line, and stop.
#
# ci-roster-check honours marker-based membership ONLY for a gate whose own discovery it
# can execute.  That is not fussiness: no text pattern distinguishes "uses the marker to
# find files" from "uses the marker", and this script contains both -- the mapfile above,
# and the `grep -qxF -- "${MARKER}"` further down that asserts an EXCLUDED file does NOT
# carry it.  A census that pattern-matched would keep passing with the mapfile deleted.
# So it asks, and this is the answer.
if [ -n "${RIFT_CI_GATE_LIST:-}" ]; then printf '%s\n' "${FILES[@]}"; exit 0; fi

PYTHON_BIN="${RIFT_QWINDOW_PYTHON:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

# Guard the tool checks: a missing interpreter plus a redirected stderr is
# indistinguishable from a clean result.
"${PYTHON_BIN}" -c 'import pytest' || { echo "test-q-window-stencil.sh: pytest unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import numpy; print("numpy", numpy.__version__)' \
  || { echo "test-q-window-stencil.sh: numpy unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import lal, lalsimulation; print("lal", lal.__version__)' \
  || { echo "test-q-window-stencil.sh: lal/lalsimulation unavailable" >&2; exit 1; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"



if [ "${#FILES[@]}" -eq 0 ]; then
  echo "test-q-window-stencil.sh: no test file carries the marker line" >&2
  echo "  Expected at least one file under ${CODEDIR} containing exactly:" >&2
  echo "    ${MARKER}" >&2
  echo "  Either the marker text was edited here without updating the files, or the" >&2
  echo "  registrations were lost.  This is a hard failure, not an empty run." >&2
  exit 1
fi

# ---------------------------------------------------------------------------------
# MANIFEST SCOPE.  The filename patterns this job owns.  Every file matching one of them
# must carry the marker or appear in EXCLUDED below, so a new test in this area forces a
# decision instead of being silently unrun -- this gate's own failure mode, one level up.
#
# The patterns are deliberately NOT "every test_*.py in these directories": Code/test/
# holds ~90 files belonging to other jobs, and Code/RIFT/likelihood/ holds the
# test_slowrot_*.py suite, which is slowrot-check's business (issue #169).  The cost of
# that narrowness is real and stated: a new stencil test filed under some OTHER prefix
# escapes the manifest.  Name it to match one of these, or add a pattern here.
#
# Registration itself needs no edit to this file -- only the marker in the new test.
SCOPE_GLOBS=(
  "${CODEDIR}/RIFT/likelihood/test_q_window_*.py"
  "${CODEDIR}/RIFT/likelihood/test_time_interp_*.py"
  "${CODEDIR}/RIFT/likelihood/test_interpolate_time_*.py"
  "${CODEDIR}/RIFT/likelihood/test_calmarg_stencil_*.py"
  "${CODEDIR}/RIFT/likelihood/test_batchmode_stencil_*.py"
  "${CODEDIR}/RIFT/likelihood/test_noloop_*.py"
  "${CODEDIR}/RIFT/misc/test_psd_bandwidth*.py"
  "${CODEDIR}/test/test_noloop_*.py"
  "${CODEDIR}/test/test_calmarg_running_max_*.py"
)

# EXCLUDED, with the reason each is out.
#
#   test_q_window_interp_gpu.py   Need a GPU.  On a CPU runner they report as SKIPPED
#   test_noloop_gpu_stencils.py   with exit 0, and the junit check below treats extra
#                                 skips as a failure.  Run by hand on a GPU node; the
#                                 numbers are in PR #97.  Same treatment as the GPU files
#                                 in slowrot-check.
#
#   test_noloop_accumulator_    Belongs to another job, not to a GPU.  It matches the
#   shapes.py                   test_noloop_* pattern by NAME but not by subject: it pins
#                               NoLoop's rho_sq and kappa_sq ACCUMULATOR shapes against a
#                               reference, and its time-integral test is about the
#                               quadrature rule, not about sub-sample interpolation of
#                               Q_lm.  It is registered with core-unit-check, whose FILES
#                               manifest carries it and whose floors count it.  Listed
#                               here rather than renamed so the decision is recorded where
#                               the next such file will hit it: renaming to dodge a
#                               manifest is how these gates quietly stop covering things.
EXCLUDED=(
  "${CODEDIR}/RIFT/likelihood/test_q_window_interp_gpu.py"
  "${CODEDIR}/RIFT/likelihood/test_noloop_gpu_stencils.py"
  "${CODEDIR}/test/test_noloop_accumulator_shapes.py"
)

echo "== registered files (marker: ${MARKER}) =="
printf '  %s\n' "${FILES[@]}"

# A scope pattern that stops matching anything is a SILENT no-op: the manifest keeps
# passing while covering less.  Assert each still matches at least one file, exactly as
# test-slowrot.sh asserts each DESELECT nodeid still resolves.
echo "== scope pattern check (every pattern still matches something) =="
scope_rc=0
for g in "${SCOPE_GLOBS[@]}"; do
  # shellcheck disable=SC2206
  matches=( ${g} )
  if [ ! -e "${matches[0]}" ]; then
    echo "test-q-window-stencil.sh: scope pattern ${g} matches no file." >&2
    echo "  Those tests were renamed or removed.  Fix the pattern or drop it; left as" >&2
    echo "  is, it silently covers nothing." >&2
    scope_rc=1
  fi
done
[ "${scope_rc}" -eq 0 ] || exit 1

# An EXCLUDED entry for a file that no longer exists is the same silent no-op, and an
# EXCLUDED file that ALSO carries the marker is a contradiction that would run it anyway.
echo "== exclusion check =="
excl_rc=0
for e in "${EXCLUDED[@]}"; do
  if [ ! -f "${e}" ]; then
    echo "test-q-window-stencil.sh: EXCLUDED entry ${e} does not exist." >&2
    echo "  It was renamed or removed; drop it from EXCLUDED or fix the path." >&2
    excl_rc=1
    continue
  fi
  if grep -qxF -- "${MARKER}" "${e}"; then
    echo "test-q-window-stencil.sh: ${e} is EXCLUDED but carries the marker." >&2
    echo "  Remove the marker, or remove the file from EXCLUDED.  As it stands the" >&2
    echo "  exclusion's stated reason does not describe what this gate does." >&2
    excl_rc=1
  fi
done
[ "${excl_rc}" -eq 0 ] || exit 1

echo "== manifest check (every in-scope test file is registered or explicitly excluded) =="
manifest_rc=0
for g in "${SCOPE_GLOBS[@]}"; do
  for f in ${g}; do
    [ -f "${f}" ] || continue
    known=0
    for k in "${FILES[@]}" "${EXCLUDED[@]}"; do
      [ "${f}" = "${k}" ] && { known=1; break; }
    done
    if [ "${known}" -eq 0 ]; then
      echo "test-q-window-stencil.sh: ${f} is neither registered nor explicitly excluded." >&2
      manifest_rc=1
    fi
  done
done
if [ "${manifest_rc}" -ne 0 ]; then
  echo "  Register it by adding this line, on its own, near the top of that file:" >&2
  echo "    ${MARKER}" >&2
  echo "  and raise EXPECTED_TESTS / EXPECTED_PASSED below.  If it must NOT run here" >&2
  echo "  (needs a GPU, is a print-only study, belongs to another job), add it to" >&2
  echo "  EXCLUDED in this script WITH A REASON." >&2
  exit 1
fi

# ---------------------------------------------------------------------------------
# The pinned floors, as of this commit.  Re-derive both after adding or removing a test:
#   EXPECTED_TESTS   `pytest --collect-only -q` over the registered files.
#   EXPECTED_PASSED  the "N passed" from a full run (tests minus skips).
# Never lower either without saying why in the commit message.
EXPECTED_TESTS=73
EXPECTED_PASSED=71

# The only legitimate skips here are the two cupy legs -- one in
# test_noloop_time_marg_row_offset.py, one in test_calmarg_running_max_row_offset.py --
# which pytest.importorskip's away on these GPU-less runners.  A THIRD skip means a gate
# was disabled, which is the exact shape this script exists to prevent, so cap it rather
# than letting skips absorb losses silently.
MAX_SKIPS=2

# PER-FILE collection floor.  A registered file that collects nothing contributes zero
# gates while looking like membership; on its own pytest would exit 5 on it, and inside a
# multi-file run that exit code never appears at all.
echo "== per-file collection check (each registered file must collect >= 1 test) =="
perfile_rc=0
n_collected=0
for f in "${FILES[@]}"; do
  out="$("${PYTHON_BIN}" -m pytest --collect-only -q -p no:cacheprovider "${f}" 2>&1)"
  rc=$?
  if [ "${rc}" -ne 0 ] && [ "${rc}" -ne 5 ]; then
    printf '%s\n' "${out}"
    echo "test-q-window-stencil.sh: collection of ${f} failed (exit ${rc})" >&2
    exit 1
  fi
  # Anchor to '<path>.py::' at line start.  An unanchored grep -c '::' also counts merged
  # stderr and warning text, and because these are >= tests, OVER-counting is the
  # dangerous direction: one stray line masks exactly one lost test.
  n="$(printf '%s\n' "${out}" | grep -cE '^[^[:space:]]+\.py::')"
  printf '  %3d  %s\n' "${n}" "${f}"
  if [ "${n}" -eq 0 ]; then
    echo "test-q-window-stencil.sh: ${f} carries the marker but collects 0 tests." >&2
    perfile_rc=1
  fi
  n_collected=$(( n_collected + n ))
done
if [ "${perfile_rc}" -ne 0 ]; then
  echo "  pytest exits 5 (\"no tests ran\") on such a file when run alone, and inside a" >&2
  echo "  multi-file run it contributes nothing while looking registered.  Give it" >&2
  echo "  test_* functions, or remove the marker and add it to EXCLUDED with a reason." >&2
  exit 1
fi

echo "== collection floor check (expect >= ${EXPECTED_TESTS} tests) =="
echo "collected ${n_collected} tests from ${#FILES[@]} files"
if [ "${n_collected}" -lt "${EXPECTED_TESTS}" ]; then
  echo "test-q-window-stencil.sh: collected ${n_collected} tests, expected at least ${EXPECTED_TESTS}." >&2
  echo "  A file was renamed/moved, or a test_* entry point was dropped and pytest is" >&2
  echo "  now passing on fewer tests than this gate promises.  Fix the file, or update" >&2
  echo "  EXPECTED_TESTS in this script and say why." >&2
  exit 1
fi

junit="$(mktemp -t qwindowci-junit-XXXXXX.xml)" || { echo "test-q-window-stencil.sh: mktemp failed" >&2; exit 1; }
trap 'rm -f "${junit}"' EXIT

echo "== pytest =="
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider --durations=10 --junit-xml="${junit}" "${FILES[@]}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  # rc 5 == "no tests ran"; it is a FAILURE here, not a pass.
  echo "test-q-window-stencil.sh: pytest exited ${rc}" >&2
  exit "${rc}"
fi

# OUTCOME check.  The floors above count COLLECTION, which cannot see a test that
# collects, runs, and asserts nothing: one pytest.skip() or importorskip() disables a gate
# while both the collected count and the pytest exit status stay green.  So assert what
# the RUN did, in PASSED tests, not in collected ones.
"${PYTHON_BIN}" - "${junit}" "${EXPECTED_PASSED}" "${MAX_SKIPS}" <<'PYCHECK'
import sys, xml.etree.ElementTree as ET
path, expected, max_skips = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
root = ET.parse(path).getroot()
ts = root if root.tag == "testsuite" else root.find("testsuite")
if ts is None:
    sys.stderr.write("test-q-window-stencil.sh: no <testsuite> in the junit report\n"); sys.exit(1)
g = lambda k: int(ts.get(k, 0) or 0)
tests, skipped, failures, errors = g("tests"), g("skipped"), g("failures"), g("errors")
passed = tests - skipped - failures - errors
print("junit: tests=%d passed=%d skipped=%d failures=%d errors=%d"
      % (tests, passed, skipped, failures, errors))
bad = []
if failures or errors:
    bad.append("%d failures, %d errors" % (failures, errors))
if passed < expected:
    bad.append("only %d tests PASSED, expected at least %d -- tests were lost, not just "
               "reported differently" % (passed, expected))
if skipped > max_skips:
    bad.append("%d SKIPPED, at most %d expected (the cupy legs) -- a skip silently "
               "disables a gate here; if a new skip is legitimate, raise MAX_SKIPS and "
               "say which test and why" % (skipped, max_skips))
if bad:
    sys.stderr.write("test-q-window-stencil.sh: " + "; ".join(bad) + "\n"); sys.exit(1)
PYCHECK
if [ $? -ne 0 ]; then exit 1; fi

echo "q-window stencil gate: PASS (${#FILES[@]} registered files, ${n_collected} tests collected)"
