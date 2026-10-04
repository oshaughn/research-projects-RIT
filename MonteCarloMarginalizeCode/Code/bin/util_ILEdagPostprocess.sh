#! /bin/bash
# util_NRdagPostprocess.sh
#
# GOAL
#   For NR-based DAGs, (a) consolidates the output, (b) runs ILE simplification, then (c) creates an NR-indexed version.
#   The second format uses a *portable* name, which is stable to me changing the underlying relationship between spins and label.

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

resolve_helper() {
    local helper=$1
    if [ -x "${SCRIPT_DIR}/${helper}" ]; then
        printf '%s\n' "${SCRIPT_DIR}/${helper}"
    elif command -v "${helper}" >/dev/null 2>&1; then
        command -v "${helper}"
    else
        echo "ERROR: unable to locate required helper ${helper}" >&2
        return 127
    fi
}

DIR_PROCESS=$1
BASE_OUT=$2
# Everything after the first two arguments is the advanced-physics flag list
# handed to util_CleanILE.py (--eccentricity, --meanPerAno, --a6c,
# --hyperbolic, --tabular-eos-file, ...).  BasicIteration can enable several
# groups at once, so forward ALL of them: selecting one flag and dropping the
# rest made the cleaner parse rows with a layout the run never wrote.
CLEAN_FLAGS=()
for arg in "${@:3}"; do
    if [ -n "$arg" ]; then
        CLEAN_FLAGS+=("$arg")
    fi
done

fail() {
    echo "ERROR: $2" >&2
    rm -f "${BASE_OUT}.composite"
    exit $1
}

# No shards at all is not an error here.  With --first-iteration-jumpstart the
# first consolidate node has no ILE parents, and BasicIteration deliberately
# omits the nonempty-composite POST check on that node.  Keep the old behaviour
# for that case: an empty composite and exit 0.
HAVE_SHARDS=`find ${DIR_PROCESS} -name 'CME*.dat' -print -quit 2>/dev/null`

# --------------------------------------------------------------------------
# Hyperpipeline ASCII output path (opt-in via env var).
# When RIFT_HYPERPIPELINE_FORMAT is truthy, ILE shards are written in the
# new self-describing header-bearing hyperpipeline format.  The legacy
# `cat | util_CleanILE.py | sort -rg` chain below cannot handle these shards
# (different column layout, embedded `#`-comment headers).  We therefore
# delegate to util_CleanILE_hyperpipeline.py which does the equivalent
# weighted-average consolidation and emits a single composite file.
# --------------------------------------------------------------------------
if [ -z "${HAVE_SHARDS}" ]; then
    echo " WARNING: no CME*.dat files under ${DIR_PROCESS}; writing an empty composite "
    : > ${BASE_OUT}.composite
else
case "$(echo "${RIFT_HYPERPIPELINE_FORMAT:-}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    echo " Joining data files (hyperpipeline format) .... "
    # Forward the SAME flag list as the legacy branch.  Dropping it here meant
    # --intrinsic-digits never reached the hyperpipeline join, so that path kept
    # coalescing intrinsic points at five decimals while the legacy path honoured
    # the request -- the two formats silently disagreeing about which grid points
    # are distinct.  util_CleanILE_hyperpipeline.py accepts --intrinsic-digits as
    # an alias for --digits and ignores the advanced-physics flags it has no use
    # for (its columns are self-describing), but it now REFUSES a leftover value
    # rather than reading it as a shard filename.
    HYPER_CLEAN="$(resolve_helper util_CleanILE_hyperpipeline.py)" || exit $?
    "${HYPER_CLEAN}" \
        --output "${BASE_OUT}.composite" \
        "${CLEAN_FLAGS[@]}" \
        ${DIR_PROCESS}/CME*.dat
    clean_status=$?
    if [ ${clean_status} -ne 0 ]; then
        fail ${clean_status} "ILE consolidation failed with status ${clean_status}"
    fi
    ;;
  *)
    # join together the .dat files
    echo " Joining data files .... "
    CLEAN_ILE="$(resolve_helper util_CleanILE.py)" || exit $?
    rm -f tmp.dat tmp2.dat
    # CAT can be ineffective
    FNAME=`pwd`/tmp.dat
    #cat ${DIR_PROCESS}/CME*.dat > tmp.dat
    export RND=`echo ${RANDOM}`
    find ${DIR_PROCESS} -name 'CME*.dat' -exec cat {} \; > ${RND}_tmp.dat

    # clean them (=join duplicate lines)
    echo " Consolidating multiple instances of the monte carlo  .... "
    "${CLEAN_ILE}" ${RND}_tmp.dat "${CLEAN_FLAGS[@]}" > ${RND}_clean.dat
    clean_status=$?
    if [ ${clean_status} -ne 0 ]; then
        rm -f ${RND}_clean.dat
        fail ${clean_status} "ILE consolidation failed with status ${clean_status}"
    fi

    # Sort on lnL.  The composite row is
    #   (event_id, intrinsic..., lnL, sigma_lnL, ntotal, neff)
    # so lnL is ALWAYS the 4th field from the end, whichever advanced-physics
    # groups are enabled; derive the key from the row width instead of
    # hard-coding one column index per flag combination (which silently
    # mis-sorted, i.e. discarded the composite ordering, for combined runs).
    NCOL=`awk 'NF>0 && $1 !~ /^#/ {print NF; exit}' ${RND}_clean.dat`
    if [ -z "${NCOL}" ] || [ "${NCOL}" -lt 5 ]; then
        rm -f ${RND}_clean.dat
        fail 1 "ILE consolidation produced no usable rows from ${DIR_PROCESS}"
    fi
    sort -rg -k$((NCOL-3)) ${RND}_clean.dat > $BASE_OUT.composite
    output_status=$?
    rm -f ${RND}_clean.dat
    if [ ${output_status} -ne 0 ]; then
        fail ${output_status} "failed to write consolidated ILE output with status ${output_status}"
    fi
    ;;
esac
if [ ! -s "$BASE_OUT.composite" ]; then
    fail 1 "ILE consolidation produced an empty composite: $BASE_OUT.composite"
fi
fi

# Manifest
rm -f ${BASE_OUT}.manifest
echo '#User:' `whoami` >>  ${BASE_OUT}.manifest
echo '#Date:' `date` >>  ${BASE_OUT}.manifest
echo '#Host:' `hostname -f` >>  ${BASE_OUT}.manifest
echo '#Directory:' `pwd`/${DIR_PROCESS} >>  ${BASE_OUT}.manifest
md5sum ${DIR_PROCESS}/*psd.xml.gz >> ${BASE_OUT}.manifest
cat ${DIR_PROCESS}/command-single.sh >>  ${BASE_OUT}.manifest  
env >> ${BASE_OUT}.environment  

# tar file
if ! tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}.composite  ${BASE_OUT}.manifest ${BASE_OUT}.environment; then
    echo "ERROR: failed to create ${BASE_OUT}.tgz" >&2
    exit 1
fi

exit 0
