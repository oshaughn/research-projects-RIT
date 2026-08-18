#!/usr/bin/env bash
# CPU regression gate for the JAX extrinsic likelihood (RIFT/likelihood/jax_ile),
# driven from test/jax/.
#
# WHY THIS SCRIPT EXISTS AT ALL, AND WHY IT COUNTS TESTS
# -----------------------------------------------------
# Until this gate landed, NOTHING in .github/workflows/ci.yml ran test/jax/ -- the
# workflow had zero matches for "jax".  Two real defects survived a month each behind
# that gap (see the PR that adds this file).
#
# The obvious repair -- point pytest at test/jax/ -- would have manufactured MORE
# confidence than it earned.  Several files in that directory are scripts with an
# `if __name__ == "__main__":` block and NO `test_*` function.  Pointing pytest at such
# a file collects ZERO items and exits 5, "no tests ran", which reads as a pass in a
# skim of the log.  So this script does two things a bare pytest invocation does not:
#
#   1. It asserts a FLOOR on the number of collected tests before running anything.
#      If a future refactor drops a `test_*` entry point, renames a file, or moves it,
#      collection silently shrinks and this job goes RED instead of green-on-nothing.
#      The floor is pinned to the exact count as of this commit; raise it when you add
#      tests, and never lower it without saying why in the commit message.
#   2. It fails on ANY nonzero pytest exit, which includes exit 5.
#
# JAX_PLATFORMS=cpu is set: no GPU is required, and jax must not go hunting for one.
set -uo pipefail

PYTHON_BIN="${RIFT_JAX_PYTHON:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

# Guard the tool checks: a missing interpreter plus a redirected stderr is
# indistinguishable from a clean result.
"${PYTHON_BIN}" -c 'import pytest' || { echo "test-jax.sh: pytest unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import jax, jaxlib; print("jax", jax.__version__)' \
  || { echo "test-jax.sh: jax unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import numpyro; print("numpyro", numpyro.__version__)' \
  || { echo "test-jax.sh: numpyro unavailable (needed by test_nuts_phimarg)" >&2; exit 1; }

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

JAXDIR="MonteCarloMarginalizeCode/Code/test/jax"

# Included files, with the count each contributes as of this commit:
#   test_jax_likelihood.py             3  synthetic packed data: nearest-vs-NoLoop, AD
#                                         vs finite differences, jit/vmap
#   test_jax_endtoend.py               1  full precompute -> pack -> JAX vs the numpy
#                                         NoLoop on a real injection (fixed by #144)
#   test_jax_slowrot_coeffs.py         2  rotation + freqresponse response coefficients
#                                         against their numpy references
#   test_jax_slowrot_wrapper.py        1  the one-call build_*_data_from_precompute path
#   test_jax_slowrot.py                3  rotation Path A (p_max=0), Path B (p_max=1)
#                                         and freqresponse: NoLoop parity + AD/jit/
#                                         vmap/hessian
#   test_jax_slowrot_cauchy_schwarz.py 2  the rotation lnL VALUE (bound + explicit
#                                         time-domain model), Path A and Path B.
#                                         Agreement with the NoLoop is necessary but
#                                         not sufficient -- see that file's docstring
#   test_network_coords.py             1  network-frame sky fold on a real injection
#   test_nuts_phimarg.py               1  fisher_nuts_sample_phimarg vs an analytic 4-D
#                                         target (needs numpyro; no lal)
#
# DELIBERATELY EXCLUDED (measured on ldas-pcdev11, JAX_PLATFORMS=cpu, OMP_NUM_THREADS=1):
#
#   test_nuts_phimarg_injection.py  Not a pytest file at all: it runs the whole study at
#                                 module scope and calls sys.exit() there, so pytest
#                                 reports a COLLECTION ERROR rather than zero tests.  It
#                                 is also long -- a full NUTS run on a real injection
#                                 that has exceeded a 1800 s cap in hand testing.  Too
#                                 expensive for every PR; run it by hand.
#
#   test_flow_reuse.py            Collects 0 (pytest exit 5); passes as a script.
#                                 Excluded on DEPENDENCY risk, not runtime: three flowMC
#                                 runs, and flowMC is an extra heavy dependency with a
#                                 fast-moving sampler API that this test tracks closely,
#                                 so an unpinned flowMC release would redden the gate
#                                 for reasons unrelated to RIFT.  Reasonable to add
#                                 later behind a PINNED flowMC.  Run it by hand when
#                                 touching samplers.flowmc_sample.
#
#   demo_*.py, debug_*.py,        Demos, debugging scripts and a figure generator, not
#   benchmark_snr_sequence.py,    assertions.  None defines a test_* function and none
#   make_3g_figdata.py            is intended as a gate.
FILES=(
  "${JAXDIR}/test_jax_likelihood.py"
  "${JAXDIR}/test_jax_endtoend.py"
  "${JAXDIR}/test_jax_slowrot_coeffs.py"
  "${JAXDIR}/test_jax_slowrot_wrapper.py"
  "${JAXDIR}/test_jax_slowrot.py"
  "${JAXDIR}/test_jax_slowrot_cauchy_schwarz.py"
  "${JAXDIR}/test_network_coords.py"
  "${JAXDIR}/test_nuts_phimarg.py"
)

# Sum of the per-file counts above.  Pinned deliberately: a bare `pytest test/jax/`
# that collected 0 would exit 5, and a partial loss (say 14 -> 3) would still exit 0.
EXPECTED_TESTS=14

echo "== collection floor check (expect >= ${EXPECTED_TESTS} tests) =="
collect_out="$("${PYTHON_BIN}" -m pytest --collect-only -q -p no:cacheprovider "${FILES[@]}" 2>&1)"
collect_rc=$?
if [ "${collect_rc}" -ne 0 ]; then
  printf '%s\n' "${collect_out}"
  echo "test-jax.sh: pytest collection failed (exit ${collect_rc})" >&2
  exit 1
fi
n_collected="$(printf '%s\n' "${collect_out}" | grep -c '::')"
echo "collected ${n_collected} tests from ${#FILES[@]} files"
if [ "${n_collected}" -lt "${EXPECTED_TESTS}" ]; then
  printf '%s\n' "${collect_out}"
  echo "test-jax.sh: collected ${n_collected} tests, expected at least ${EXPECTED_TESTS}." >&2
  echo "  A file was renamed/moved, or a test_* entry point was dropped and pytest is" >&2
  echo "  now passing on fewer tests than this gate promises.  Fix the file, or update" >&2
  echo "  EXPECTED_TESTS in this script and say why." >&2
  exit 1
fi

echo "== running =="
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider --durations=0 "${FILES[@]}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  # rc 5 == "no tests ran"; it is a FAILURE here, not a pass.
  echo "test-jax.sh: pytest exited ${rc}" >&2
  exit "${rc}"
fi

echo "jax_ile CPU regression gate: PASS (${n_collected} tests)"
