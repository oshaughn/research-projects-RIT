#!/usr/bin/env bash
# Seed ensemble on the S250114ax best-fit on-source point, comparing GMM-coverage configs for
# RELIABILITY (n_eff distribution) and EXTRINSIC-POSTERIOR STABILITY.  The goal is NOT max n_eff:
# it is MODEST, RELIABLE n_eff with a stable, unbiased extrinsic posterior that preserves the real
# degeneracy structure (sky ring / dL-inclination / psi-phi), i.e. no copy silently collapsing a mode.
#
# Uses --extrinsic-proposal-output (weight-correct GMM fit of the run's TRUE-weighted extrinsic
# posterior) instead of --save-samples, which is unusable here (fairdraw/save-P/log-cumsum/lnL-only,
# see DESIGN_portfolio_freeze_policy.md).  Compare the breadcrumbs with compare_extrinsic_breadcrumbs.py.
#
# Usage: bench_onsource_ensemble.sh [GPU] [SEEDS...]      (default GPU 2, seeds 1 2 3)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$HERE/bench_onsource.sh"
GPU="${1:-2}"; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3)
RUNPE=~/RIFT_roboto_paper/analyses/integrator_demos/S250114ax_pipeline/run_PE/iteration_0_ile
MAXCONC=3

# The bench does NOT select a sampler (driver default is adaptive_cartesian_gpu, the classic GPU
# Cartesian sampler -- NOT VARAHA/AV, NOT the portfolio).  Every config must explicitly request the
# AV(=mcsamplerAdaptiveVolume)+GMM(=mcsamplerEnsemble) portfolio, or the --internal-gmm-* flags are inert.
# canonical form: --sampler-portfolio is action='append', so repeat the flag (one member each).
# (comma-separated AV,GMM also works via a recent driver split, but repeated flags are the robust form.)
PORTFOLIO="--sampler-method portfolio --sampler-portfolio AV --sampler-portfolio GMM"

# config name -> extra GMM-coverage flags (portfolio selection prepended in run_one).
declare -A CFG
CFG[cap8]="--internal-gmm-adaptive-components --internal-gmm-max-components 8  --internal-gmm-inflate 1.0"
CFG[cap16]="--internal-gmm-adaptive-components --internal-gmm-max-components 16 --internal-gmm-inflate 1.3"
CFG[corr]="--internal-gmm-correlate-all --internal-gmm-adaptive-components --internal-gmm-max-components 8"

run_one() {
  local cfg="$1"
  local seed="$2"
  local name="e_${cfg}_s${seed}"
  GPU=$GPU NAME="$name" bash "$BENCH" $PORTFOLIO ${CFG[$cfg]} \
    --seed "$seed" --extrinsic-proposal-output "$RUNPE/ext_${name}.npz"
}

# simple 3-wide job pool
running=0
for cfg in cap8 cap16 corr; do
  for s in "${SEEDS[@]}"; do
    run_one "$cfg" "$s" &
    running=$((running+1))
    if [ $running -ge $MAXCONC ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
  done
done
wait
echo "ENSEMBLE DONE"
