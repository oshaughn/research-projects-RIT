# Shared configuration for the export_likelihoods demo.  Source this first:
#     source config.sh
#
# Everything downstream (creating the artifact, sampling, JS validation) reads
# these variables, so point them at your own ILE output / benchmark to re-run on
# a different event.  Defaults reproduce the GW170817-like development case the
# method was built and validated on.

# --- python + import path ---------------------------------------------------
# jax_gp is an OPTIONAL subpackage; it needs the JAX stack (jax, numpyro, tinygp,
# flowMC, ...).  On the dev machine that lives in the `gwkokab` conda env.  Set
# PY to whatever interpreter has both RIFT-on-PYTHONPATH and the JAX deps.
: "${PY:=/home/oshaughn/.conda/envs/gwkokab/bin/python}"

# Use the dev RIFT in THIS checkout (the env-installed RIFT may be stale).
# demo/rift/export_likelihoods -> up 3 = MonteCarloMarginalizeCode/Code
RIFT_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${RIFT_CODE_DIR}:${PYTHONPATH}"

# --- inputs -----------------------------------------------------------------
# An ILE `.net` file: the per-point Monte-Carlo lnL evaluations CIP consumes.
: "${NET:=/home/oshaughn/all.net}"

# RF+AV reference posterior(s) to validate against (a glob is fine -> pooled).
# Build your own with applications/benchmark_condor/ ; this is the cached fleet.
: "${BENCHMARK_GLOB:=/home/oshaughn/jaxcip_benchmark/out/cip_rf_*.xml.gz}"

# --- outputs ----------------------------------------------------------------
: "${OUTDIR:=/tmp/export_likelihoods_demo}"
mkdir -p "${OUTDIR}"
: "${ARTIFACT:=${OUTDIR}/gw170817_quadgp}"        # exported bundle base path
: "${POSTERIOR:=${OUTDIR}/jaxcip_nutsmu}"         # validation posterior base

# --- physics / prior box (TRUST these; the grid may extend past them) -------
# Narrow detector-frame chirp-mass box around the true value + small-spin BNS.
: "${MC_RANGE:=[1.196,1.199]}"
: "${CHI_MAX:=0.05}"

# --- surrogate + sampler knobs ----------------------------------------------
# quadgp = quadratic Fisher core + GP residual (PE-grade on the razor-sharp mc
# peak).  svgp residual scales to more data than exact.  These are the "fast
# demo" sizes (~minutes on CPU); the paper-grade run uses cap=12000,
# n-opt-steps=250, num-samples=4000 (see README).
: "${METHOD:=quadgp}"
: "${QUADGP_RESIDUAL:=svgp}"
: "${CAP_POINTS:=6000}"
: "${N_FEATURES:=600}"        # SVGP inducing points (or RFF features)
: "${N_OPT_STEPS:=150}"
: "${NUM_WARMUP:=600}"
: "${NUM_SAMPLES:=3000}"
: "${NUM_CHAINS:=2}"

# Fit/sample coordinates (BNS Morisaki + tidal).  fit = --parameter + implied;
# sampled low-level = --parameter + --parameter-nofit.
COORD_ARGS=(
  --parameter delta_mc
  --parameter-implied mu1 --parameter-implied mu2
  --parameter-implied LambdaTilde --parameter-implied DeltaLambdaTilde
  --parameter-nofit mc --parameter-nofit s1z --parameter-nofit s2z
  --parameter-nofit lambda1 --parameter-nofit lambda2
)

# Marginals to score with the Jensen-Shannon divergence.
JS_PARAMS=(mc delta_mc s1z s2z lambda1 lambda2 LambdaTilde)

echo "[config] PY=${PY}"
echo "[config] RIFT_CODE_DIR=${RIFT_CODE_DIR}"
echo "[config] NET=${NET}"
echo "[config] OUTDIR=${OUTDIR}"
