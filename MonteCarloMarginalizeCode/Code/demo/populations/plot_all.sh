#!/usr/bin/env zsh
set -euo pipefail

# Usage:
#   ./plot_all.sh /path/to/main_dir [extra plotting flags...]
# Example:
#   ./plot_all.sh . --save-png

if (( $# < 1 )); then
  echo "Usage: $0 <MAIN_DIR> [extra plotting args...]"
  exit 1
fi

MAIN_DIR=$1
shift  # remaining args go to the plotting script
PLOT=~/plot_iterationsAndSubdag_animation_with.sh

if [[ ! -d "$MAIN_DIR" ]]; then
  echo "Error: MAIN_DIR '$MAIN_DIR' not found"
  exit 1
fi
if [[ ! -x "$PLOT" ]]; then
  echo "Error: plotting script '$PLOT' not found or not executable"
  exit 1
fi

# Collect all rundir paths like:
#   <MAIN_DIR>/analysis_event_*/rundir
typeset -a RUN_DIRS
RUN_DIRS=(${MAIN_DIR}/analysis_event_*/rundir(N/))

if (( ${#RUN_DIRS} == 0 )); then
  echo "No rundir directories found under: ${MAIN_DIR}/analysis_event_*/rundir"
  exit 0
fi

echo "Found ${#RUN_DIRS} rundir(s). Running plots…"

any_failed=0

for RD in "${RUN_DIRS[@]}"; do
  echo "========================================"
  echo "Working in: $RD"
  echo "========================================"
  pushd "$RD" >/dev/null

  # Use `if ! cmd; then` so `set -e` does not kill the script on failure
  if ! "$PLOT" \
    --parameter mc \
    --parameter q \
    --parameter a1z \
    --parameter a2z \
    --parameter eccentricity \
    --eccentricity \
    --use-legend \
    --ci-list '[0.9]' \
    --quantiles None \
    --lnL-cut 15 \
    --use-all-composite-but-grayscale \
    --bind-param eccentricity \
    --param-bound '[0,0.5]' \
    "$@"; then
    echo "Warning: plotting failed in '$RD', skipping to next rundir."
    any_failed=1
  fi

  popd >/dev/null
done

if (( any_failed )); then
  echo "All plots attempted, but some rundirs failed."
  exit 1
fi

echo "All plots completed successfully."
