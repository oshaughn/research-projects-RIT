#!/usr/bin/env bash
# CPU regression gate for the slow-rotation / finite-size likelihood
# (RIFT/likelihood/factored_likelihood_with_rotation.py, slowrot_response.py,
# slowrot_freqresponse.py), driven from RIFT/likelihood/test_slowrot_*.py.
#
# Four defences, each guarding a way this directory can go green while testing nothing.
# Do not simplify any of them into a bare `pytest <dir>`:
#
#   1. An EXPLICIT file list, not a glob.  Several test_slowrot_*.py files collect ZERO
#      items, and pytest exits 5 on those -- "no tests ran" reads as a pass in a log skim.
#   2. A FLOOR on the collected count, so a renamed file or a dropped test_* entry point
#      goes RED instead of green-on-fewer-tests.
#   3. A hard fail on ANY nonzero pytest exit (5 included), plus a junit OUTCOME
#      assertion.  The floor counts COLLECTION, which cannot see a test that collects,
#      runs, and asserts nothing.
#   4. A SCRIPTS tier for the assert-carrying files pytest cannot count.
#
# Needs numpy + lal only: no GPU, no jax, no numpyro.  Rationale and measured cost:
# PR #172 (2026-08); the sibling gate it is modelled on is .travis/test-jax.sh.
set -uo pipefail
# NOTE: deliberately no -e.  Every command below has its rc handled explicitly so the
# failure messages stay specific; if you add a command, guard it yourself.

# SLOWDIR below is repo-relative, so anchor cwd rather than trusting the caller.
cd "$(dirname "$0")/.." || { echo "test-slowrot.sh: cannot cd to repo root" >&2; exit 1; }

# INVARIANT: this gate always tests THIS CHECKOUT, never an installed build.  Without
# this line the two tiers disagree -- pytest prepends .../Code to sys.path and gets the
# checkout, while a directly-run script gets RIFT/likelihood/ as sys.path[0] and falls
# through to whatever RIFT is installed.  Must PREPEND: appending lets a caller's
# PYTHONPATH win and restores the split.  If you need to validate a wheel or a container
# rather than the checkout, run its test files directly -- do not "fix" it here.
export PYTHONPATH="$PWD/MonteCarloMarginalizeCode/Code${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${RIFT_SLOWROT_PYTHON:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

# Guard the tool checks: a missing interpreter plus a redirected stderr is
# indistinguishable from a clean result.
"${PYTHON_BIN}" -c 'import pytest' || { echo "test-slowrot.sh: pytest unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import numpy; print("numpy", numpy.__version__)' \
  || { echo "test-slowrot.sh: numpy unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import lal, lalsimulation; print("lal", lal.__version__)' \
  || { echo "test-slowrot.sh: lal/lalsimulation unavailable" >&2; exit 1; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SLOWDIR="MonteCarloMarginalizeCode/Code/RIFT/likelihood"

# ---------------------------------------------------------------------------------
# TIER 1: files whose tests pytest can collect and count.
#
# Per-file counts are deliberately NOT listed: they need maintenance on every test added,
# and `pytest --collect-only -q <file>` answers the question in seconds.  EXPECTED_TESTS
# below is the pinned total, and the manifest check keeps the list complete.
#
# Two files carry guards whose whole deliverable is the guard, so do not drop them from
# this list to save time:
#   test_slowrot_fd_ops.py        the Nyquist derivative weight, zeroed at ODD p and left
#                                 alone at EVEN p.  The jax gate is structurally blind to
#                                 this -- at p=1 the correct and the over-zeroing weights
#                                 are bit-identical, because 1 is odd either way.  (#163)
#   test_slowrot_freqresponse.py  the unpaired-Nyquist response weight and its Hermitian
#                                 average.  (#165)
FILES=(
  "${SLOWDIR}/test_slowrot_fd_ops.py"
  "${SLOWDIR}/test_slowrot_freqresponse.py"
  "${SLOWDIR}/test_slowrot_harmonic_width.py"
  "${SLOWDIR}/test_slowrot_headtohead.py"
  "${SLOWDIR}/test_slowrot_likelihood_v1.py"
  "${SLOWDIR}/test_slowrot_noloop.py"
  "${SLOWDIR}/test_slowrot_pathB.py"
  "${SLOWDIR}/test_slowrot_precompute_integration.py"
  "${SLOWDIR}/test_slowrot_response.py"
)

# DESELECTED, and EXPECTED_TESTS is one lower because of it.
#
# test_W5_jax_packer_loses_nothing catches ImportError on jax and RETURNS.  That is not a
# pytest skip -- it COLLECTS, RUNS, ASSERTS NOTHING and REPORTS PASSED.  This job installs
# no jax, so leaving it selected would raise the floor and the junit count while gating
# nothing, which is this script's own failure mode one level down.
#
# Gating W5 needs a jax install.  jax-ile-check does NOT cover it either -- that manifest
# scans test/jax/ only.  Stated gap, not a claim of coverage.
#
# CAUTION: --deselect is a PREFIX match, not an exact nodeid match.  A future sibling named
# test_W5_jax_packer_loses_nothing_v2 would be swallowed silently, and a >= floor cannot see
# a test that was never selected.  Name any successor differently.
DESELECT=(
  "${SLOWDIR}/test_slowrot_harmonic_width.py::test_W5_jax_packer_loses_nothing"
)

# ---------------------------------------------------------------------------------
# TIER 2: files that assert but define no test_* function, so pytest collects 0 from each
# and would exit 5 on any of them alone.  Run as `python <file>`, exit 0 required.
#
# The scope of each file's asserts decides how it is gated, and the three differ:
#
#   test_slowrot_cauchy_schwarz.py          asserts at MODULE scope
#   test_slowrot_noloop_bruteforce.py       asserts at MODULE scope
#   test_slowrot_freqresponse_likelihood.py asserts inside a function, called from __main__
#
# All three are gated ONLY by being executed here.  They are not in FILES, so pytest never
# imports them and the collection floor cannot see a failed assert in any of them -- the
# scope differences above change WHEN each file's asserts run, not how many gates it has.
#
# ANTI-INSTRUCTION: do not "tidy" these asserts into functions.  A function this tier never
# calls leaves `python <file>` exiting 0 having asserted nothing, and no count notices.
#
# cauchy_schwarz is the one that pins lnL <= (1/2)<d|d>, i.e. that <d|h> and <h|h> are
# evaluated for the SAME h.  Both it and noloop_bruteforce fail on a dropped arrival-time
# post-phase; this tier stops at the first failing script, so a mutation run will normally
# only show you the first.  Neither is redundant with the other.
SCRIPTS=(
  "${SLOWDIR}/test_slowrot_cauchy_schwarz.py"
  "${SLOWDIR}/test_slowrot_noloop_bruteforce.py"
  "${SLOWDIR}/test_slowrot_freqresponse_likelihood.py"
)

# EXCLUDED, with the reason each is out.  The manifest check below fails if a
# test_slowrot_*.py is in none of FILES, SCRIPTS or EXCLUDED, so a new one forces a
# decision instead of being silently unrun -- this gate's own failure mode, one level up.
#
#   test_slowrot_gpu.py                 Need a GPU.  On a CPU runner they report as
#   test_slowrot_freqresponse_gpu.py    SKIPPED with exit 0, and the junit check below
#                                       treats a skip as a failure.  Run by hand on a GPU
#                                       node.  Same treatment as the GPU parity files in
#                                       q-window-stencil-check.
#
#   test_slowrot_pathB_groundtruth.py   ZERO assert statements at any scope: both are
#   test_slowrot_pathB_bruteforce.py    print-only convergence studies, so running them
#                                       can fail only on an exception, and the import
#                                       surface is already covered by TIER 1 and TIER 2.
#                                       If either grows an assert, move it to SCRIPTS.
EXCLUDED=(
  "${SLOWDIR}/test_slowrot_gpu.py"
  "${SLOWDIR}/test_slowrot_freqresponse_gpu.py"
  "${SLOWDIR}/test_slowrot_pathB_groundtruth.py"
  "${SLOWDIR}/test_slowrot_pathB_bruteforce.py"
)

# The manifest globs test_slowrot_*.py, NOT test_*.py: this directory also holds
# test_q_window_interp.py, test_calmarg_stencil_gating.py and friends, which belong to
# q-window-stencil-check and are not this gate's business.  A new slow-rotation test
# filed under some other prefix would escape the manifest; name it test_slowrot_*.
echo "== manifest check (every test_slowrot_*.py is gated or explicitly excluded) =="
manifest_rc=0
for f in "${SLOWDIR}"/test_slowrot_*.py; do
  known=0
  for g in "${FILES[@]}" "${SCRIPTS[@]}" "${EXCLUDED[@]}"; do
    [ "${f}" = "${g}" ] && { known=1; break; }
  done
  if [ "${known}" -eq 0 ]; then
    echo "test-slowrot.sh: ${f} is neither gated nor explicitly excluded." >&2
    manifest_rc=1
  fi
done
if [ "${manifest_rc}" -ne 0 ]; then
  echo "  Add it to FILES (and raise EXPECTED_TESTS), or to SCRIPTS if it asserts" >&2
  echo "  outside a test_* function, or to EXCLUDED with a reason." >&2
  exit 1
fi

# The pinned floor: the number TIER 1 collects after DESELECT, as of this commit.
# Re-derive with `pytest --collect-only -q` over FILES; never lower it without saying why
# in the commit message.  A bare `pytest ${SLOWDIR}` would sweep up files that collect 0,
# and a partial loss still exits 0, which is what this pins against.
EXPECTED_TESTS=43

DESELECT_ARGS=()
for d in "${DESELECT[@]}"; do DESELECT_ARGS+=(--deselect "${d}"); done

echo "== collection floor check (expect >= ${EXPECTED_TESTS} tests) =="
collect_out="$("${PYTHON_BIN}" -m pytest --collect-only -q -p no:cacheprovider \
  "${DESELECT_ARGS[@]}" "${FILES[@]}" 2>&1)"
collect_rc=$?
if [ "${collect_rc}" -ne 0 ]; then
  printf '%s\n' "${collect_out}"
  echo "test-slowrot.sh: pytest collection failed (exit ${collect_rc})" >&2
  exit 1
fi
# Anchor to '<path>.py::' at line start.  An unanchored grep -c '::' also counts merged
# stderr and warning text, and because the floor is a >= test, OVER-counting is the
# dangerous direction: one stray line masks exactly one lost test.
n_collected="$(printf '%s\n' "${collect_out}" | grep -cE '^[^[:space:]]+\.py::')"
echo "collected ${n_collected} tests from ${#FILES[@]} files"
if [ "${n_collected}" -lt "${EXPECTED_TESTS}" ]; then
  printf '%s\n' "${collect_out}"
  echo "test-slowrot.sh: collected ${n_collected} tests, expected at least ${EXPECTED_TESTS}." >&2
  echo "  A file was renamed/moved, or a test_* entry point was dropped and pytest is" >&2
  echo "  now passing on fewer tests than this gate promises.  Fix the file, or update" >&2
  echo "  EXPECTED_TESTS in this script and say why." >&2
  exit 1
fi

# A deselect that stops matching is silent: pytest warns nothing and the count simply
# goes UP, which a >= floor cannot see.  Assert each one still selects something.
for d in "${DESELECT[@]}"; do
  if ! "${PYTHON_BIN}" -m pytest --collect-only -q -p no:cacheprovider "${d}" >/dev/null 2>&1; then
    echo "test-slowrot.sh: DESELECT entry ${d} no longer resolves to a test." >&2
    echo "  It was renamed or removed; drop it from DESELECT and lower EXPECTED_TESTS," >&2
    echo "  or fix the nodeid.  Left as is, the deselect is a no-op." >&2
    exit 1
  fi
done

junit="$(mktemp -t slowrotci-junit-XXXXXX.xml)" || { echo "test-slowrot.sh: mktemp failed" >&2; exit 1; }
trap 'rm -f "${junit}"' EXIT

echo "== TIER 1: pytest =="
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider --durations=10 --junit-xml="${junit}" \
  "${DESELECT_ARGS[@]}" "${FILES[@]}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  # rc 5 == "no tests ran"; it is a FAILURE here, not a pass.
  echo "test-slowrot.sh: pytest exited ${rc}" >&2
  exit "${rc}"
fi

# OUTCOME check.  The floor above counts COLLECTION, which cannot see a test that
# collects, runs, and asserts nothing: one pytest.skip() or importorskip() disables a
# gate while both the collected count and the pytest exit status stay green.  That is
# the very shape this script exists to prevent, so assert what the RUN did.
"${PYTHON_BIN}" - "${junit}" "${EXPECTED_TESTS}" <<'PYCHECK'
import sys, xml.etree.ElementTree as ET
path, expected = sys.argv[1], int(sys.argv[2])
root = ET.parse(path).getroot()
ts = root if root.tag == "testsuite" else root.find("testsuite")
if ts is None:
    sys.stderr.write("test-slowrot.sh: no <testsuite> in the junit report\n"); sys.exit(1)
g = lambda k: int(ts.get(k, 0) or 0)
tests, skipped, failures, errors = g("tests"), g("skipped"), g("failures"), g("errors")
print("junit: tests=%d skipped=%d failures=%d errors=%d" % (tests, skipped, failures, errors))
bad = []
if tests < expected:
    bad.append("ran %d tests, expected at least %d" % (tests, expected))
if skipped:
    bad.append("%d SKIPPED -- a skip silently disables a gate here; if a skip is "
               "legitimate, exclude the file in FILES and say why" % skipped)
if failures or errors:
    bad.append("%d failures, %d errors" % (failures, errors))
if bad:
    sys.stderr.write("test-slowrot.sh: " + "; ".join(bad) + "\n"); sys.exit(1)
PYCHECK
if [ $? -ne 0 ]; then exit 1; fi

echo "== TIER 2: assert scripts =="
for s in "${SCRIPTS[@]}"; do
  echo "-- ${s}"
  "${PYTHON_BIN}" "${s}"
  src="$?"
  if [ "${src}" -ne 0 ]; then
    echo "test-slowrot.sh: ${s} exited ${src}" >&2
    echo "  This file asserts outside any test_* function, so a nonzero exit here is" >&2
    echo "  a failed assertion, not a harness problem." >&2
    exit 1
  fi
done

echo "slowrot CPU regression gate: PASS (${n_collected} tests + ${#SCRIPTS[@]} assert scripts)"
