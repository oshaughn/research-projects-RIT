#!/bin/bash
#
# util_CIPCompositionReweightWrapper.sh -- GATED drop-in CIP wrapper (opt in from
# util_RIFT_pseudo_pipe.py via --internal-cip-composition-reweight, or pass directly to
# create_event_parameter_pipeline_BasicIteration --cip-exe).  Default pipeline behaviour is
# untouched: nothing uses this wrapper unless explicitly requested.
#
# WHAT IT DOES (detect -> repair, two CIP passes):
#   pass 1  run the REAL CIP on the ORIGINAL training set, with the sample export bumped to
#           >= 20000 and redirected to temp files (the real output paths are never touched
#           by pass 1);
#   gate    util_CIPTailDeficitGate.py on (training set, pass-1 posterior): FIRE only on a
#           SEVERE tail deficit (R < 0.32, calibrated in this exact fresh-CIP channel) AND
#           only when the implied tail mass is resolvable (validity floor >= 50/n_post --
#           MANDATORY, the gate tool has no bypass);
#   final   FIRE    -> util_CompositionReweightNet.py thins the training set and the final
#                      CIP runs on the thinned set (original argv otherwise verbatim);
#           NO-FIRE / ABSTAIN / any failure -> the final CIP runs on the ORIGINAL argv
#                      verbatim, and the decision (with R and which condition decided it)
#                      is logged LOUDLY on stderr -- a no-op is always visible in the log.
#
# COST: one extra CIP per invocation when enabled (pass 1).  CIP is the cheap CPU stage.
#
# SCOPE (measured; details in the gate tool's docstring): repairs SEVERE deficits only
# (~half the affected low-mass events); mild deficit and healthy width are not separable in
# the mid-R band and are left alone, loudly.  Safety is the strongly supported side (zero
# false fires in-sample and on known-truth healthy-narrow benchmarks).
#
# FAIL-SAFE THROUGHOUT: any tool error, unreadable input, or degenerate case -> the final
# CIP runs on the original argv unchanged, with a loud warning.  The pipeline never breaks.
#
# Test hooks: CIP_REWEIGHT_REAL_CIP=<exe> substitutes the CIP executable;
#             CIP_REWEIGHT_CONVERT=<exe> substitutes convert_output_format_ile2inference.

set -u

TOOL=$(command -v util_CompositionReweightNet.py || true)
GATE=$(command -v util_CIPTailDeficitGate.py || true)
CONVERT="${CIP_REWEIGHT_CONVERT:-$(command -v convert_output_format_ile2inference || true)}"
CIP_EXE="${CIP_REWEIGHT_REAL_CIP:-$(command -v util_ConstructIntrinsicPosterior_GenericCoordinates.py || true)}"
GATE_MIN_EXPORT=20000
TAGP="util_CIPCompositionReweightWrapper"

if [[ -z "$CIP_EXE" ]]; then
    echo "$TAGP: FATAL: real CIP not on PATH" >&2
    exit 1
fi

loud() {
    echo "**************************************************************************" >&2
    echo "$TAGP: $1" >&2
    echo "**************************************************************************" >&2
}

# --- parse argv: training file, output-sample / integral paths, export count ---
args=("$@")
n=${#args[@]}
fname=""; out_samples=""; n_output=""
for ((i = 0; i < n; i++)); do
    a="${args[$i]}"
    case "$a" in
        --fname=*)                  fname="${a#--fname=}" ;;
        --fname)                    ((i + 1 < n)) && fname="${args[$((i + 1))]}" ;;
        --fname-output-samples=*)   out_samples="${a#--fname-output-samples=}" ;;
        --fname-output-samples)     ((i + 1 < n)) && out_samples="${args[$((i + 1))]}" ;;
        --n-output-samples=*)       n_output="${a#--n-output-samples=}" ;;
        --n-output-samples)         ((i + 1 < n)) && n_output="${args[$((i + 1))]}" ;;
    esac
done

run_original() {
    # single exit path for every no-op / fail-safe branch: original argv, verbatim
    exec "$CIP_EXE" "${args[@]}"
}

if [[ -z "$fname" || -z "$out_samples" || -z "$TOOL" || -z "$GATE" || -z "$CONVERT" ]]; then
    missing=""
    [[ -z "$fname" ]]       && missing="$missing --fname-not-in-argv"
    [[ -z "$out_samples" ]] && missing="$missing --fname-output-samples-not-in-argv"
    [[ -z "$TOOL" ]]        && missing="$missing util_CompositionReweightNet.py-not-on-PATH"
    [[ -z "$GATE" ]]        && missing="$missing util_CIPTailDeficitGate.py-not-on-PATH"
    [[ -z "$CONVERT" ]]     && missing="$missing converter-not-on-PATH"
    loud "cannot gate ($missing); running CIP UNCHANGED on the original training set"
    run_original
fi

dir=$(dirname "$fname")
tag="$(date +%Y%m%d%H%M%S)_p$$"
p1base="$dir/gatepass1_${tag}"
p1dat="$dir/gatepass1_${tag}.dat"
gatejson="$dir/gate_decision_${tag}.json"
comp="$dir/all_comp_${tag}.net"
cleanup() { rm -f "$p1base".xml.gz "$p1base"_intg* "$p1dat" "$comp"; }
trap cleanup EXIT

# --- pass 1: original training set, export bumped, outputs redirected to temp names ---
p1=()
skip=0
for ((i = 0; i < n; i++)); do
    if ((skip)); then skip=0; continue; fi
    a="${args[$i]}"
    case "$a" in
        --fname-output-samples=*)  p1+=("--fname-output-samples=$p1base") ;;
        --fname-output-samples)    p1+=("--fname-output-samples" "$p1base"); skip=1 ;;
        --fname-output-integral=*) p1+=("--fname-output-integral=${p1base}_intg") ;;
        --fname-output-integral)   p1+=("--fname-output-integral" "${p1base}_intg"); skip=1 ;;
        --n-output-samples=*)      nv="${a#--n-output-samples=}"
                                   ((nv < GATE_MIN_EXPORT)) && nv=$GATE_MIN_EXPORT
                                   p1+=("--n-output-samples=$nv") ;;
        --n-output-samples)        nv="${args[$((i + 1))]}"
                                   ((nv < GATE_MIN_EXPORT)) && nv=$GATE_MIN_EXPORT
                                   p1+=("--n-output-samples" "$nv"); skip=1 ;;
        *)                         p1+=("$a") ;;
    esac
done

echo "$TAGP: pass 1 (gate-quality CIP on the ORIGINAL training set, temp outputs)" >&2
if ! "$CIP_EXE" "${p1[@]}" 1>&2 || [[ ! -s "$p1base.xml.gz" ]]; then
    loud "pass-1 CIP FAILED; falling back: final CIP on the ORIGINAL training set"
    run_original
fi
if ! "$CONVERT" "$p1base.xml.gz" > "$p1dat" 2>/dev/null || [[ ! -s "$p1dat" ]]; then
    loud "pass-1 sample conversion FAILED; falling back: final CIP on the ORIGINAL training set"
    run_original
fi

# --- gate ---
decision_line=$("$GATE" "$fname" "$p1dat" --json "$gatejson" | tail -1)
if [[ "$decision_line" != GATE\ DECISION=* ]]; then
    loud "gate tool FAILED to evaluate; falling back: final CIP on the ORIGINAL training set"
    run_original
fi
echo "$TAGP: $decision_line (record: $gatejson)" >&2

use="$fname"
case "$decision_line" in
    "GATE DECISION=FIRE"*)
        echo "$TAGP: SEVERE tail deficit detected -> composition-reweight thinning" >&2
        if "$TOOL" "$fname" --output "$comp" --seed 0 --stats-json "$dir/comp_stats_${tag}.json" 1>&2 \
           && [[ -s "$comp" ]]; then
            use="$comp"
            echo "$TAGP: final CIP will train on $comp (original preserved at $fname)" >&2
        else
            loud "reweight tool FAILED after a FIRE; falling back to the ORIGINAL training set"
        fi
        ;;
    "GATE DECISION=NO-FIRE"*)
        echo "$TAGP: NO-OP by threshold: no severe deficit ($decision_line)" >&2
        echo "$TAGP: (mild deficits above the threshold are NOT separable from healthy" >&2
        echo "$TAGP:  width and are deliberately left alone -- see util_CIPTailDeficitGate.py)" >&2
        ;;
    "GATE DECISION=ABSTAIN-FLOOR"*)
        echo "$TAGP: NO-OP by validity floor: implied tail mass unresolvable at this sample" >&2
        echo "$TAGP: size ($decision_line); raising --n-output-samples would sharpen the gate" >&2
        ;;
esac

# --- final CIP: original argv verbatim, fname swapped only on a successful FIRE ---
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

"$CIP_EXE" "${out[@]}"
exit $?
