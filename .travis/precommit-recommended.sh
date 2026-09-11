#!/usr/bin/env bash
# Tier-2 gate: "strongly recommended before merging into rift_O4d" (see
# .travis/PRECOMMIT.md).  GPU + JAX suites no GitHub runner can exercise, plus the
# expensive_before_merging/ regression tests.  Run BY HAND on a GPU/big node before
# merging anything touching the ILE likelihood, the NoLoop stencils, calibration
# marginalization, or the JAX driver.
#
# REFUSES, NOT SKIPS.  A tier-2 run that quietly no-ops when cupy or the jax stack is
# missing is worse than no run at all: it produces the same "ran clean" impression as a
# real pass.  Every prerequisite below is checked before anything runs, the same way
# .travis/test-jax.sh guards its own interpreter/jax/numpyro imports and
# .travis/test-calmarg-gpu.sh assumes a real GPU because it only ever runs on one.
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "precommit-recommended.sh: cannot cd to repo root" >&2; exit 1; }

PYTHON_BIN="${PYTHON:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_BIN="$(command -v python3)"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "precommit-recommended.sh: no python interpreter found" >&2; exit 1; }

echo "== environment guard: GPU / cupy =="
# Device(0).compute_capability alone is NOT a guard: it reads a static property and
# returns cleanly even when the device is busy/unavailable for real work (measured:
# passes, then the first actual allocation raises cudaErrorDevicesUnavailable).  Force
# a real allocation + kernel + sync so a device that cannot currently run anything
# fails HERE, not partway through a tier-2 test file with a confusing traceback.
"${PYTHON_BIN}" -c '
import cupy
x = cupy.arange(1024)
y = int((x * 2).sum())
cupy.cuda.Stream.null.synchronize()
assert y == 2 * 1024 * 1023 // 2
' || {
  echo "precommit-recommended.sh: cupy could not run a real op on a CUDA device in ${PYTHON_BIN}." >&2
  echo "  This gate requires a real, currently-available GPU -- it has no CPU fallback." >&2
  echo "  Run it on a GPU/big node (see infra-atlas: pcdev11/13) or GitLab's gpu_integration"\
       "runner, and check nvidia-smi if the device looks present but busy." >&2
  exit 1
}
echo "cupy + CUDA device: OK (real allocation + kernel + sync)"

echo "== environment guard: jax stack =="
"${PYTHON_BIN}" -c 'import pytest' || {
  echo "precommit-recommended.sh: pytest unavailable in ${PYTHON_BIN}" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import jax, jaxlib; print("jax", jax.__version__)' || {
  echo "precommit-recommended.sh: jax unavailable in ${PYTHON_BIN}" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import numpyro; print("numpyro", numpyro.__version__)' || {
  echo "precommit-recommended.sh: numpyro unavailable in ${PYTHON_BIN}" >&2; exit 1; }
echo "jax + jaxlib + numpyro: OK"

CODE="MonteCarloMarginalizeCode/Code"
fail=0

echo "== RIFT/likelihood/test_q_window_interp_gpu.py + test_noloop_gpu_stencils.py (real cupy) =="
"${PYTHON_BIN}" -m pytest -q \
  "${CODE}/RIFT/likelihood/test_q_window_interp_gpu.py" \
  "${CODE}/RIFT/likelihood/test_noloop_gpu_stencils.py" || fail=1

echo "== .travis/test-calmarg-gpu.sh =="
bash .travis/test-calmarg-gpu.sh || fail=1

echo "== test/jax/test_angle_marg_exact.py =="
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  "${PYTHON_BIN}" -m pytest -q "${CODE}/test/jax/test_angle_marg_exact.py" || fail=1

echo "== full .travis/test-jax.sh (measured ~859s+~600s locally as of its own comments;"\
     "grows with the suite) =="
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  bash .travis/test-jax.sh || fail=1

echo "== test/expensive_before_merging/ (RIFT_RUN_EXPENSIVE=1) =="
RIFT_RUN_EXPENSIVE=1 "${PYTHON_BIN}" -m pytest -q "${CODE}/test/expensive_before_merging/" \
  || fail=1

if [ "${fail}" -ne 0 ]; then
  echo "precommit-recommended.sh: one or more tier-2 gates FAILED" >&2
  exit 1
fi
echo "precommit-recommended.sh: all tier-2 gates PASSED"
