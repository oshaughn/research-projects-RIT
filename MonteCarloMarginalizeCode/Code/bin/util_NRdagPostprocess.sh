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

CLEAN_ILE="$(resolve_helper util_CleanILE.py)" || exit $?
RELABEL_ILE="$(resolve_helper util_NRRelabelILE.py)" || exit $?

DIR_PROCESS=$1
BASE_OUT=$2
GROUP=$3
ECC=$4

# join together the .dat files
echo " Joining data files .... "
rm -f tmp.dat tmp2.dat
#cat ${DIR_PROCESS}/CME*.dat > tmp.dat
find ${DIR_PROCESS} -name 'CME*.dat' -exec cat {} \; > ${DIR_PROCESS}_tmp.dat

# clean them (=join duplicate lines)
echo " Consolidating multiple instances of the monte carlo  .... "
if [ "$4" == '--eccentricity' ]
then
    "${CLEAN_ILE}" ${DIR_PROCESS}_tmp.dat $4 | sort -rg -k11 > $BASE_OUT.composite
else
    "${CLEAN_ILE}" ${DIR_PROCESS}_tmp.dat $4 | sort -rg -k10 > $BASE_OUT.composite
fi
clean_status=$?
if [ ${clean_status} -ne 0 ]; then
    echo "ERROR: NR consolidation failed with status ${clean_status}" >&2
    rm -f "$BASE_OUT.composite"
    exit ${clean_status}
fi
if [ ! -s "$BASE_OUT.composite" ]; then
    echo "ERROR: NR consolidation produced an empty composite: $BASE_OUT.composite" >&2
    rm -f "$BASE_OUT.composite"
    exit 1
fi

# index them
echo " Reindexing the data to   .... "
#util_ILEtoNRIndex.py --group ${GROUP} --fname ${BASE_OUT}.composite | grep '^-1*' > ${BASE_OUT}.indexed
if [ "$4" == '--eccentricity' ]
then
    #    util_NRRelabelILE.py --group ${GROUP} --fname ${BASE_OUT}.composite --eccentricity | grep '^-1*' > ${BASE_OUT}.indexed
    "${RELABEL_ILE}" --group Sequence-RIT-All --fname ${BASE_OUT}.composite --eccentricity | sed -n '/ -----  BEST MATCHES ------ /,$p' > ${BASE_OUT}.indexed
else
    "${RELABEL_ILE}" --group ${GROUP} --fname ${BASE_OUT}.composite | grep '^-1*' > ${BASE_OUT}.indexed
fi
relabel_status=$?
if [ ${relabel_status} -ne 0 ]; then
    echo "ERROR: NR relabeling failed with status ${relabel_status}" >&2
    rm -f "$BASE_OUT.indexed"
    exit ${relabel_status}
fi
if [ ! -s "$BASE_OUT.indexed" ]; then
    echo "ERROR: NR relabeling produced an empty index: $BASE_OUT.indexed" >&2
    rm -f "$BASE_OUT.indexed"
    exit 1
fi

# Manifest
rm -f ${BASE_OUT}.manifest
echo '#User:' `whoami` >>  ${BASE_OUT}.manifest
echo '#Date:' `date` >>  ${BASE_OUT}.manifest
echo '#Host:' `hostname -f` >>  ${BASE_OUT}.manifest
echo '#Directory:' `pwd`/${DIR_PROCESS} >>  ${BASE_OUT}.manifest
md5sum ${DIR_PROCESS}/*psd.xml.gz >> ${BASE_OUT}.manifest
cat ${DIR_PROCESS}/command-single.sh >>  ${BASE_OUT}.manifest  
cat ${DIR_PROCESS}/integrate.sub >>  ${BASE_OUT}.submit

env >> ${BASE_OUT}.environment  

# tar file
if ! tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}.composite ${BASE_OUT}.indexed ${BASE_OUT}.manifest ${BASE_OUT}.environment ${BASE_OUT}.submit; then
    echo "ERROR: failed to create ${BASE_OUT}.tgz" >&2
    exit 1
fi
