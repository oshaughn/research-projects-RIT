#! /usr/bin/env bash

set -euo pipefail

if [[ "${RIFT_CI_REQUIRE_GPU:-0}" == "1" ]]; then
    python - <<'PY'
import sys

try:
    import cupy
except Exception as exc:
    raise SystemExit(f"RIFT_CI_REQUIRE_GPU=1 but cupy could not be imported: {exc}") from exc

try:
    n_devices = cupy.cuda.runtime.getDeviceCount()
except Exception as exc:
    raise SystemExit(f"RIFT_CI_REQUIRE_GPU=1 but CUDA devices could not be queried: {exc}") from exc

if n_devices < 1:
    raise SystemExit("RIFT_CI_REQUIRE_GPU=1 but cupy reported zero CUDA devices")

x = cupy.arange(8, dtype=cupy.float64)
if float(cupy.asnumpy((x * x).sum())) != 140.0:
    raise SystemExit("RIFT_CI_REQUIRE_GPU=1 but a basic cupy device calculation failed")

from RIFT.integrators import mcsamplerGPU

if not getattr(mcsamplerGPU, "cupy_ok", False):
    raise SystemExit("RIFT_CI_REQUIRE_GPU=1 but RIFT.integrators.mcsamplerGPU did not enable cupy")

print(f"GPU preflight OK: cupy={cupy.__version__}, cuda_devices={n_devices}")
PY
fi

# Unit/regression tests for the sampler helpers themselves (fast, no data needed).
# The extrinsic "zoom box" limits under the cosine samplers live here: they are pure
# coordinate-transform + prior-mass identities, so they belong with the integrator gate
# rather than with the end-to-end run tests.
python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_limit_cosine_samplers.py
python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_mcsampler_ensemble_log_contract.py

# Supplementary-likelihood plugin hook: the NAL reader/evaluator (pure numpy, no data) and the
# static guard on the drivers' prepare-hook wiring, which is what makes the plugin receive the
# SAMPLING basis at all. Both are seconds-long and protect a silent-wrong-answer path.
python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_nal_io.py \
                   MonteCarloMarginalizeCode/Code/test/test_supplementary_likelihood_hook.py

# Time-marginalization quadrature.  The historical rule integrates exp(lnL(t)) with Simpson at
# the FIXED spacing deltaT=1/srate, while the integrand's width sigma_t = 1/(2 pi rho sigma_f) is
# set by the SIGNAL and shrinks as 1/rho -- so production under-resolves its own integrand, worse
# at higher SNR (measured: the reported lnL moves 1.649 nats when the grid phase is scanned over
# 2*deltaT at srate 4096, rho=40).  This gate covers the opt-in band-limited quadrature against an
# ANALYTIC continuous reference, plus its fail-closed guards and -- the part that matters most
# here -- that the option actually reaches the shipped likelihood rather than being inert.
_TMARG_TESTS=MonteCarloMarginalizeCode/Code/test/test_time_marginalization_quadrature.py
# Count guard, matching .travis/test-slowrot.sh and test-jax.sh.  `set -e` already
# catches a total collection failure (pytest exits 5), but a silent shrink from 60
# tests to 3 -- a rename, a stale -k, a decorator that stops matching -- reads as
# green.  Raise EXPECTED by RUNNING collection, never by arithmetic.
_TMARG_EXPECTED=73
_TMARG_FOUND=$(python -m pytest -q --collect-only "$_TMARG_TESTS" 2>/dev/null | grep -c '::' || true)
if [ "$_TMARG_FOUND" -ne "$_TMARG_EXPECTED" ]; then
    echo "time-marginalization gate: collected $_TMARG_FOUND tests, expected $_TMARG_EXPECTED" >&2
    exit 1
fi
# SKIP guard.  `pytest -q` exits 0 with skips, so a test that quietly stops
# running reads as green -- and the count guard above catches DESELECTION, not
# SKIPPING.  The GPU-parity test is expected to skip on a CPU runner (exactly 1);
# anything else skipping means an importorskip started firing and a gate is
# reporting green having never executed what it names.  On a GPU runner
# RIFT_CI_REQUIRE_GPU=1 makes that test FAIL rather than skip, so expect 0.
if [[ "${RIFT_CI_REQUIRE_GPU:-0}" == "1" ]]; then _TMARG_EXPECT_SKIP=0; else _TMARG_EXPECT_SKIP=1; fi
_TMARG_OUT=$(python -m pytest -q -rs "$_TMARG_TESTS" 2>&1) || { echo "$_TMARG_OUT"; exit 1; }
echo "$_TMARG_OUT" | tail -20
_TMARG_SKIPPED=$(echo "$_TMARG_OUT" | grep -oE '[0-9]+ skipped' | grep -oE '^[0-9]+' || true)
_TMARG_SKIPPED=${_TMARG_SKIPPED:-0}
if [ "$_TMARG_SKIPPED" -ne "$_TMARG_EXPECT_SKIP" ]; then
    echo "time-marginalization gate: $_TMARG_SKIPPED tests skipped, expected $_TMARG_EXPECT_SKIP" >&2
    exit 1
fi

python MonteCarloMarginalizeCode/Code/test/test_mcsamplerEnsemble_extended.py --as-test --n-max 100000

python MonteCarloMarginalizeCode/Code/test/test_mcsamplerEnsemble_extended.py --as-test --n-max 100000 --use-lnL
