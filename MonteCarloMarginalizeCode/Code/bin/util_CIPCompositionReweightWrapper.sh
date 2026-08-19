#!/bin/bash
#
# util_CIPCompositionReweightWrapper.sh -- drop-in CIP wrapper (use via
# create_event_parameter_pipeline_BasicIteration --cip-exe, or the pseudo-pipe opt-in
# --internal-cip-composition-reweight).
#
# Scans its argv for the CIP training file (--fname=X or --fname X), runs
# util_CompositionReweightNet.py on it (composition-equalising thinning, fail-safe mode),
# then execs the REAL CIP with the identical argv except --fname pointing at the thinned
# file.  If the reweight cannot run at all, it warns LOUDLY and execs CIP on the ORIGINAL
# file, so the pipeline never breaks.  Keeps CIP fully modular: nothing in CIP or the merge
# step changes; flag-off is byte-identical to today.
#
# Concurrency: exploded CIP workers each invoke this wrapper on the same all.net, so the
# thinned file gets a unique per-process name (deterministic --seed 0 content, but unique
# names avoid read-during-write races) and is removed when CIP exits.
#
# Test hook: CIP_REWEIGHT_REAL_CIP=<exe> substitutes the CIP executable.

set -u

TOOL=$(command -v util_CompositionReweightNet.py || true)
CIP_EXE="${CIP_REWEIGHT_REAL_CIP:-$(command -v util_ConstructIntrinsicPosterior_GenericCoordinates.py || true)}"

if [[ -z "$CIP_EXE" ]]; then
    echo "util_CIPCompositionReweightWrapper: FATAL: real CIP not on PATH" >&2
    exit 1
fi

# --- locate the training file in argv: handle BOTH --fname=X and --fname X ---
args=("$@")
n=${#args[@]}
fname=""
for ((i = 0; i < n; i++)); do
    a="${args[$i]}"
    case "$a" in
        --fname=*) fname="${a#--fname=}" ;;
        --fname)   if ((i + 1 < n)); then fname="${args[$((i + 1))]}"; fi ;;
    esac
done

if [[ -z "$fname" || -z "$TOOL" ]]; then
    echo "util_CIPCompositionReweightWrapper: WARNING: $([[ -z "$fname" ]] && echo 'no --fname in argv' || echo 'util_CompositionReweightNet.py not on PATH'); exec CIP unchanged" >&2
    exec "$CIP_EXE" "$@"
fi

dir=$(dirname "$fname")
tag="$(date +%Y%m%d%H%M%S)_p$$"
comp="$dir/all_comp_${tag}.net"
stats="$dir/comp_stats_${tag}.json"

use="$fname"
# tool stdout -> stderr: CIP's stdout may carry pipeline data and must stay clean.
# NON-strict: on degenerate input the tool copies input->output unchanged and exits 0,
# so CIP still runs on (a copy of) the original data and the warning is on stderr.
if "$TOOL" "$fname" --output "$comp" --seed 0 --stats-json "$stats" 1>&2; then
    use="$comp"
    trap 'rm -f "$comp"' EXIT
    echo "util_CIPCompositionReweightWrapper: training file $fname -> $comp (stats: $stats)" >&2
else
    echo "**************************************************************************" >&2
    echo "util_CIPCompositionReweightWrapper: LOUD WARNING: composition reweight" >&2
    echo "  FAILED (rc!=0); falling back to the ORIGINAL training file: $fname" >&2
    echo "**************************************************************************" >&2
    rm -f "$comp"
fi

# --- rebuild argv verbatim, substituting only the fname (quoted ranges etc. survive) ---
out=()
skip=0
for ((i = 0; i < n; i++)); do
    if ((skip)); then skip=0; continue; fi
    a="${args[$i]}"
    if [[ "$a" == --fname=* ]]; then
        out+=("--fname=$use")
    elif [[ "$a" == "--fname" ]] && ((i + 1 < n)); then
        out+=("--fname" "$use")
        skip=1
    else
        out+=("$a")
    fi
done

# not exec: the EXIT trap must outlive CIP to clean up the thinned file
"$CIP_EXE" "${out[@]}"
exit $?
