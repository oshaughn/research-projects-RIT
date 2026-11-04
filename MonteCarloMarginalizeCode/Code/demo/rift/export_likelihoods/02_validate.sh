#!/usr/bin/env bash
#
# (b) VALIDATE the GP likelihood.
#
# Export plumbing only proves the artifact round-trips; it does NOT prove the
# surrogate is any GOOD.  We validate against the production answer: draw a
# posterior from the SAME quadgp surrogate with mu-frame-preconditioned NUTS,
# then measure the Jensen-Shannon divergence (bits) of every 1D marginal against
# a CIP+RF+AV reference posterior.  The PE bar is JS ~ few x 1e-3 bits.
#
# Why nuts-mu (not importance sampling): the posterior is a razor-thin ridge in
# mc.  An IS proposal is proposal-LIMITED -- it under-explores the weakly-
# constrained directions (delta_mc, tides) and they come out too narrow.  NUTS
# preconditioned with the well-conditioned mu-frame covariance explores those
# wings by construction, so the JS test reflects the SURROGATE, not the sampler.
#
# Usage:
#     source config.sh && ./02_validate.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "${RIFT_CODE_DIR:-}" ] || source "${HERE}/config.sh"

echo "=== [1] draw a posterior from the quadgp surrogate (mu-frame NUTS) ==="
"${PY}" -m RIFT.interpolators.jax_gp.applications.jax_cip \
  --fname "${NET}" \
  "${COORD_ARGS[@]}" \
  --mc-range "${MC_RANGE}" --chi-max "${CHI_MAX}" \
  --cap-points "${CAP_POINTS}" \
  --jax-fit-method "${METHOD}" --quadgp-residual "${QUADGP_RESIDUAL}" \
  --n-features "${N_FEATURES}" --n-opt-steps "${N_OPT_STEPS}" \
  --sampler nuts-mu \
  --num-warmup "${NUM_WARMUP}" --num-samples "${NUM_SAMPLES}" \
  --num-chains "${NUM_CHAINS}" \
  --fname-output-samples "${POSTERIOR}" \
  --fname-output-integral "${OUTDIR}/integral_result"

echo
echo "=== [2] JS divergence vs the RF benchmark, per marginal (bits; bar ~ few x 1e-3) ==="
echo "    a = ${POSTERIOR}.xml.gz"
echo "    b = ${BENCHMARK_GLOB}"
for prm in "${JS_PARAMS[@]}"; do
  printf '    %-12s ' "${prm}"
  "${PY}" -m RIFT.interpolators.jax_gp.applications.compare \
    --a "${POSTERIOR}.xml.gz" \
    --b ${BENCHMARK_GLOB} --param "${prm}" 2>/dev/null \
    | grep -i 'JS(' || echo "(param not in one of the files)"
done

echo
echo "Interpretation: small JS on the well-measured directions (mc, spins) confirms"
echo "the surrogate nails the sharp peak; small JS on the broad directions (delta_mc,"
echo "tides) confirms the sampler explored the wings.  Persistently large JS on a"
echo "single marginal points at the surrogate (more data / opt steps), not the sampler."
