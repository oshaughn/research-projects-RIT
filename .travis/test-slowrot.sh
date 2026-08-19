#!/usr/bin/env bash
# CPU regression gate for the slow-rotation / finite-size likelihood
# (RIFT/likelihood/factored_likelihood_with_rotation.py, slowrot_response.py,
# slowrot_freqresponse.py), driven from RIFT/likelihood/test_slowrot_*.py.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# Until this gate landed, NOTHING in .github/workflows/ci.yml ran any
# test_slowrot_* file: `grep -rn slowrot .github/workflows/ci.yml` returned one hit and
# it was a comment (issue #169).  That mattered more than an ordinary coverage gap,
# because the two most recent changes to this code are changes whose DELIVERABLE IS THE
# GUARD -- #163 (the Nyquist derivative weight, both parities) and #165 (the Hermitian
# Nyquist response weight, which provably moves no number).  A guard that never runs
# automatically leaves exactly nothing behind.
#
# It is modelled on .travis/test-jax.sh, which solved the same problem for test/jax/,
# and it keeps that script's three defences, because this directory needs all three:
#
#   1. An EXPLICIT file list, not a glob.  Five test_slowrot_*.py files collect ZERO
#      items and exit 5, "no tests ran", which reads as a pass in a skim of the log.
#   2. A FLOOR on the collected count, so a renamed file or a dropped test_* entry
#      point turns this job RED instead of green-on-fewer-tests.
#   3. A hard fail on ANY nonzero pytest exit (which includes exit 5), plus a junit
#      OUTCOME assertion.  The floor counts COLLECTION, and collection cannot see a
#      test that collects, runs, and asserts nothing.
#
# It adds a fourth, because this directory has a shape test/jax/ does not:
#
#   4. A SCRIPTS tier.  Three of the zero-collecting files are module-scope scripts
#      that carry real asserts -- they validate at import time and never define a
#      test_* function.  pytest gives them no count and no junit row, so they are run
#      directly as `python <file>` and required to exit 0.
#
# Needs numpy + lal only: no GPU, no jax, no numpyro.  That is deliberate -- see the
# ci.yml comment for why this is a separate job from jax-ile-check rather than more
# files in it.
set -uo pipefail
# NOTE: deliberately no -e.  Every command below has its rc handled explicitly so the
# failure messages stay specific; if you add a command, guard it yourself.

# SLOWDIR below is repo-relative, so anchor cwd rather than trusting the caller.
cd "$(dirname "$0")/.." || { echo "test-slowrot.sh: cannot cd to repo root" >&2; exit 1; }

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
# TIER 1: pytest files, with the count each contributes as of this commit.
#
#   test_slowrot_fd_ops.py                  7  the FD operator identities the rotation
#                                              expansion is built from.  Two of the
#                                              seven are #163: the Nyquist derivative
#                                              weight must be zeroed for ODD p and left
#                                              ALONE for even p.  An earlier revision
#                                              zeroed every p >= 1; that is exact at
#                                              p=1 (odd either way) and wrong at p=2,
#                                              worth 0.207 nats on a real p_max=2 bank.
#   test_slowrot_freqresponse.py           12  the frequency-dependent (finite-size)
#                                              antenna response.  Five of the twelve
#                                              are #165: the unpaired-Nyquist predicate,
#                                              Hermitian symmetry on the grid, the
#                                              conjugation commutation, the untouched-
#                                              away-from-the-bin control, and the
#                                              Hermitian-average value at the bin.
#   test_slowrot_harmonic_width.py          7  harmonic bandwidth: a too-narrow
#                                              `harmonics` request silently truncates
#                                              the model.  ONE of the seven is
#                                              DESELECTED here -- see DESELECT below.
#   test_slowrot_headtohead.py              3  Path A vs Path B vs the baseline on one
#                                              bank.
#   test_slowrot_likelihood_v1.py           2  reduction to the maintained baseline at
#                                              zero sidereal rate, and agreement with
#                                              the brute-force rotation reference.
#   test_slowrot_noloop.py                  3  the vectorized NoLoop rotation kernel.
#   test_slowrot_pathB.py                   3  Path B (p=1) scalar and vector kernels,
#                                              plus the Cauchy-Schwarz bound.
#   test_slowrot_precompute_integration.py  2  the U/V modulation arrives at the right
#                                              scale, at the right reference time.
#   test_slowrot_response.py                3  the rotation response coefficients
#                                              against lal.ComputeDetAMResponse.
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

# DESELECTED, and the floor is 41 rather than 42 because of it.
#
# test_W5_jax_packer_loses_nothing opens with `try: import jax / except ImportError:
# print("W5 SKIPPED (no jax)"); return`.  Without jax that is not a pytest skip -- it
# is a test that COLLECTS, RUNS, ASSERTS NOTHING, and REPORTS PASSED.  This job
# installs no jax (see ci.yml), so leaving it in would add 1 to both the floor and the
# junit `tests` count while gating nothing, which is this script's own failure mode one
# level down.  Deselecting it makes the 41 honest.
#
# The other six tests in that file are numpy+lal and are gated here.  Gating W5 itself
# needs a jax install; it is NOT covered by jax-ile-check either, whose manifest scans
# test/jax/ only.  That is a known, stated gap, not a claim of coverage.
DESELECT=(
  "${SLOWDIR}/test_slowrot_harmonic_width.py::test_W5_jax_packer_loses_nothing"
)

# ---------------------------------------------------------------------------------
# TIER 2: module-scope scripts.  These validate at import time and define no test_*
# function, so pytest collects 0 from each and would exit 5 if one were run alone.
# Run through pytest in a multi-file invocation they would still contribute 0 to the
# floor and 0 to the junit report, while being EXECUTED TWICE (once by --collect-only,
# once by the run).  So they get their own tier: `python <file>`, exit 0 required.
#
#   test_slowrot_cauchy_schwarz.py         6 asserts.  lnL = <d|h> - (1/2)<h|h> cannot
#                                          exceed (1/2)<d|d> for ANY h.  This is the
#                                          file that catches rotation_post_phase() being
#                                          dropped, i.e. term1 and term2 evaluated for
#                                          different templates.
#   test_slowrot_noloop_bruteforce.py      1 assert.  The vectorized rotation NoLoop vs
#                                          an INDEPENDENT time-domain brute force that
#                                          shares no convention with it.
#   test_slowrot_freqresponse_likelihood.py 2 asserts.  The finite-size likelihood
#                                          reduces to the baseline as L -> 0, respects
#                                          the bound, and beats the baseline where the
#                                          effect is genuinely in band.
SCRIPTS=(
  "${SLOWDIR}/test_slowrot_cauchy_schwarz.py"
  "${SLOWDIR}/test_slowrot_noloop_bruteforce.py"
  "${SLOWDIR}/test_slowrot_freqresponse_likelihood.py"
)

# EXCLUDED, with the reason each is out.  The manifest check below fails if a
# test_slowrot_*.py is in none of FILES, SCRIPTS or EXCLUDED, so adding a new one forces
# a decision instead of it being silently unrun -- which is this gate's own failure
# mode, one level up.
#
#   test_slowrot_gpu.py                 Need a GPU.  MEASURED on a CPU node: `2 skipped`
#   test_slowrot_freqresponse_gpu.py    with exit 0 (cupy raises ImportError on
#                                       libcuda.so.1).  There is no GPU on these
#                                       runners, so they would report as skipped, and
#                                       the junit check below treats a skip as a
#                                       failure.  Run by hand on a GPU node.  Same
#                                       treatment as the GPU parity files in
#                                       q-window-stencil-check.
#
#   test_slowrot_pathB_groundtruth.py   ZERO assert statements: both are print-only
#   test_slowrot_pathB_bruteforce.py    convergence studies.  Running them can fail only
#                                       on an exception, and the import surface they
#                                       would smoke-test is already exercised by TIER 1
#                                       and TIER 2.  Cost is real (measured together at
#                                       37 s on citlogin6 / AMD EPYC, 3.5 min extrapolated
#                                       from the Intel timings below) for no assertion.
#                                       If either grows an assert, move it into SCRIPTS.
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
  echo "  Add it to FILES (and raise EXPECTED_TESTS), or to SCRIPTS if it asserts at" >&2
  echo "  module scope, or to EXCLUDED with a reason." >&2
  exit 1
fi

# Sum of the TIER 1 per-file counts above, minus the one deselected test: 42 - 1.
# Pinned deliberately: a bare `pytest ${SLOWDIR}` would also sweep up files that
# collect 0, and a partial loss (say 41 -> 3) still exits 0.
EXPECTED_TESTS=41

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

junit="$(mktemp -t slowrotci-junit-XXXXXX.xml)"
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

echo "== TIER 2: module-scope assert scripts =="
for s in "${SCRIPTS[@]}"; do
  echo "-- ${s}"
  "${PYTHON_BIN}" "${s}"
  src="$?"
  if [ "${src}" -ne 0 ]; then
    echo "test-slowrot.sh: ${s} exited ${src}" >&2
    echo "  This file asserts at MODULE SCOPE and defines no test_* function, so a" >&2
    echo "  nonzero exit here is a failed assertion, not a harness problem." >&2
    exit 1
  fi
done

echo "slowrot CPU regression gate: PASS (${n_collected} tests + ${#SCRIPTS[@]} assert scripts)"
