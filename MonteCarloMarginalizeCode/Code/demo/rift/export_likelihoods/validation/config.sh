# Shared configuration for the GP-vs-standard-RIFT validation ladder.
# Source this first:  source config.sh
#
# DIVISION OF LABOR (read this):
#   * YOU run the full RIFT pipeline for a case, to convergence, ELSEWHERE
#     (cluster / full run -- see CASES.md).  A full run adaptively samples the
#     likelihood; a hand-picked fixed grid does NOT, and you'd have no idea
#     whether the posterior is reliable.
#   * THIS harness only ANALYZES a completed run: fit the GP to the run's
#     converged likelihood grid, sample it with NUTS, and compare marginals to
#     the run's standard posterior.  It never runs ILE itself.
#
# Point GRID and STD at your completed run's outputs, then run analyze_case.sh.

# Interpreter with RIFT + lal + the JAX stack (the gwkokab conda env).
: "${PY:=/home/oshaughn/.conda/envs/gwkokab/bin/python}"

# validation -> up 4 = MonteCarloMarginalizeCode/Code ; the dev RIFT lives here.
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"
export GW_SURROGATE=""
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- inputs from YOUR completed full run (set these) ------------------------
# GRID: the converged, consolidated likelihood grid the run produced -- a
#   named-column .dat with an lnL column plus the physical parameter columns.
#   For the distance-grid (Plan A) pipeline this is the consolidated
#   all_dgrid.dat (columns: lnL sigmaL m1 m2 ... dist ...).
: "${GRID:=}"
# STD: the run's standard posterior samples (.dat, named columns over the same
#   physical parameters) -- e.g. the CIP / util_ConstructEOSPosterior output.
: "${STD:=}"

# --- outputs ----------------------------------------------------------------
: "${OUTDIR:=/tmp/gp_validation}"
mkdir -p "${OUTDIR}"

echo "[config] PY=${PY}"
echo "[config] CODE_DIR=${CODE_DIR}"
echo "[config] GRID=${GRID:-<unset: point at your full-run grid>}"
echo "[config] STD =${STD:-<unset: point at your full-run standard posterior>}"
echo "[config] OUTDIR=${OUTDIR}"
