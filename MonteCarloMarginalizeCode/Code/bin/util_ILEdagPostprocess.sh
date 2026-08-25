#! /bin/bash
# util_NRdagPostprocess.sh
#
# GOAL
#   For NR-based DAGs, (a) consolidates the output, (b) runs ILE simplification, then (c) creates an NR-indexed version.
#   The second format uses a *portable* name, which is stable to me changing the underlying relationship between spins and label.


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

# --------------------------------------------------------------------------
# Hyperpipeline ASCII output path (opt-in via env var).
# When RIFT_HYPERPIPELINE_FORMAT is truthy, ILE shards are written in the
# new self-describing header-bearing hyperpipeline format.  The legacy
# `cat | util_CleanILE.py | sort -rg` chain below cannot handle these shards
# (different column layout, embedded `#`-comment headers).  We therefore
# delegate to util_CleanILE_hyperpipeline.py which does the equivalent
# weighted-average consolidation and emits a single composite file.
# --------------------------------------------------------------------------
case "$(echo "${RIFT_HYPERPIPELINE_FORMAT:-}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    echo " Joining data files (hyperpipeline format) .... "
    util_CleanILE_hyperpipeline.py \
        --output "${BASE_OUT}.composite" \
        ${DIR_PROCESS}/CME*.dat
    ;;
  *)
    # join together the .dat files
    echo " Joining data files .... "
    rm -f tmp.dat tmp2.dat
    # CAT can be ineffective
    FNAME=`pwd`/tmp.dat
    #cat ${DIR_PROCESS}/CME*.dat > tmp.dat
    export RND=`echo ${RANDOM}`
    find ${DIR_PROCESS} -name 'CME*.dat' -exec cat {} \; > ${RND}_tmp.dat

    # clean them (=join duplicate lines)
    echo " Consolidating multiple instances of the monte carlo  .... "
    util_CleanILE.py ${RND}_tmp.dat "${CLEAN_FLAGS[@]}" > ${RND}_clean.dat

    # Sort on lnL.  The composite row is
    #   (event_id, intrinsic..., lnL, sigma_lnL, ntotal, neff)
    # so lnL is ALWAYS the 4th field from the end, whichever advanced-physics
    # groups are enabled; derive the key from the row width instead of
    # hard-coding one column index per flag combination (which silently
    # mis-sorted, i.e. discarded the composite ordering, for combined runs).
    NCOL=`awk 'NF>0 && $1 !~ /^#/ {print NF; exit}' ${RND}_clean.dat`
    if [ -z "${NCOL}" ] || [ "${NCOL}" -lt 5 ]; then
        echo " WARNING: no usable rows in consolidated ILE output "
        cp ${RND}_clean.dat $BASE_OUT.composite
    else
        sort -rg -k$((NCOL-3)) ${RND}_clean.dat > $BASE_OUT.composite
    fi
    rm -f ${RND}_clean.dat
    ;;
esac

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
tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}.composite  ${BASE_OUT}.manifest ${BASE_OUT}.environment

exit 0 ;  # force end on success, for DAG
