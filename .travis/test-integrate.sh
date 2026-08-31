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
# ANALYTIC continuous reference, plus its finite-window reconstruction and resolution guards and
# -- the part that matters most
# here -- that the option actually reaches the shipped likelihood rather than being inert.
# The PIPELINE file is listed alongside it deliberately: the quadrature is inert unless it
# survives util_RIFT_pseudo_pipe.py -> helper_LDG_Events.py -> args_ile.txt ->
# create_event_parameter_pipeline_BasicIteration -> ILE*.sub, and the last link is an
# INHERITANCE (ile_args_extr = ile_args + ...), not an explicit forward.  An unlisted test never
# runs in this CI, so wiring the file in is part of shipping the wiring.
_TMARG_TESTS=(
    MonteCarloMarginalizeCode/Code/test/test_time_marginalization_quadrature.py
    MonteCarloMarginalizeCode/Code/test/test_time_marginalization_quadrature_pipeline.py
    MonteCarloMarginalizeCode/Code/test/test_continuous_time_posterior_export.py
)
# Count guard, matching .travis/test-slowrot.sh and test-jax.sh.  `set -e` already
# catches a total collection failure (pytest exits 5), but a silent shrink from 60
# tests to 3 -- a rename, a stale -k, a decorator that stops matching -- reads as
# green.  Raise EXPECTED by RUNNING collection, never by arithmetic.
_TMARG_EXPECTED=161
_TMARG_FOUND=$(python -m pytest -q --collect-only "${_TMARG_TESTS[@]}" 2>/dev/null | grep -c '::' || true)
if [ "$_TMARG_FOUND" -ne "$_TMARG_EXPECTED" ]; then
    echo "time-marginalization gate: collected $_TMARG_FOUND tests, expected $_TMARG_EXPECTED" >&2
    exit 1
fi
# SKIP guard.  `pytest -q` exits 0 with skips, so a test that quietly stops
# running reads as green, and the count guard above catches DESELECTION, not
# SKIPPING.
#
# This IDENTIFIES rather than COUNTS, which an earlier version of it did not.
# Counting is wrong twice over: a compensating pair (one importorskip starts
# firing while the GPU test stops) keeps the total unchanged, and the expected
# total is a property of the RUNNER, not of the code -- on a GPU-equipped runner
# that does not set RIFT_CI_REQUIRE_GPU=1 the cupy test legitimately stops
# skipping, and a count guard then fails a perfectly good run.  So: allow skips
# whose REASON names cupy/GPU, and fail on any other skip whatever the total.
_TMARG_OUT=$(python -m pytest -q -rs "${_TMARG_TESTS[@]}" 2>&1) || { echo "$_TMARG_OUT"; exit 1; }
echo "$_TMARG_OUT" | tail -20
_TMARG_BAD=$(echo "$_TMARG_OUT" | grep -E '^SKIPPED' | grep -vciE 'cupy|gpu|cuda' || true)
_TMARG_BAD=${_TMARG_BAD:-0}
if [ "$_TMARG_BAD" -ne 0 ]; then
    echo "time-marginalization gate: $_TMARG_BAD test(s) skipped for a reason other than an absent GPU:" >&2
    echo "$_TMARG_OUT" | grep -E '^SKIPPED' | grep -viE 'cupy|gpu|cuda' >&2
    exit 1
fi

# Peak-local time-marginalization quadrature.  Same defect, same derived-resolution
# discipline; what changes is WHERE the refined grid is placed -- around the enumerated
# peaks of a band-limited kappa rather than over the whole window, so the cost stops
# growing with SNR.  What this gate has to protect, beyond accuracy: that the intervals
# are MERGED (the un-merged variant double-counts the overlap, +1.6 nats at rho~6), that
# the omitted mass is BOUNDED rather than assumed (a deliberately sabotaged enumeration
# must be caught and sent to the dense path), that the local evaluator reconstructs the
# same interpolant the dense FFT does at every production npts including the odd ones,
# and that the option reaches the shipped likelihood instead of being inert.
_TMARG_PL_TESTS=MonteCarloMarginalizeCode/Code/test/test_time_marginalization_peak_local.py
# Raise EXPECTED by RUNNING collection, never by arithmetic.
_TMARG_PL_EXPECTED=109
_TMARG_PL_FOUND=$(python -m pytest -q --collect-only "$_TMARG_PL_TESTS" 2>/dev/null | grep -c '::' || true)
if [ "$_TMARG_PL_FOUND" -ne "$_TMARG_PL_EXPECTED" ]; then
    echo "peak-local gate: collected $_TMARG_PL_FOUND tests, expected $_TMARG_PL_EXPECTED" >&2
    exit 1
fi
# SKIP guard, IDENTIFYING rather than counting, for the reasons the band-limited gate
# above gives: a compensating pair leaves the total unchanged, and the expected total is
# a property of the RUNNER (a GPU-equipped runner that does not set RIFT_CI_REQUIRE_GPU=1
# legitimately stops skipping, and a count guard then fails a good run).  Allow skips
# whose REASON names cupy/GPU, and fail on any other skip whatever the total.
_TMARG_PL_OUT=$(python -m pytest -q -rs "$_TMARG_PL_TESTS" 2>&1) || { echo "$_TMARG_PL_OUT"; exit 1; }
echo "$_TMARG_PL_OUT" | tail -20
_TMARG_PL_BAD=$(echo "$_TMARG_PL_OUT" | grep -E '^SKIPPED' | grep -vciE 'cupy|gpu|cuda' || true)
_TMARG_PL_BAD=${_TMARG_PL_BAD:-0}
if [ "$_TMARG_PL_BAD" -ne 0 ]; then
    echo "peak-local gate: $_TMARG_PL_BAD test(s) skipped for a reason other than an absent GPU:" >&2
    echo "$_TMARG_PL_OUT" | grep -E '^SKIPPED' | grep -viE 'cupy|gpu|cuda' >&2
    exit 1
fi

python MonteCarloMarginalizeCode/Code/test/test_mcsamplerEnsemble_extended.py --as-test --n-max 100000

python MonteCarloMarginalizeCode/Code/test/test_mcsamplerEnsemble_extended.py --as-test --n-max 100000 --use-lnL
