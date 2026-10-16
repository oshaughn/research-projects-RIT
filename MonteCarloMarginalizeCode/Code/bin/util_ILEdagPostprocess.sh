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

DIR_PROCESS=$1
BASE_OUT=$2
ECC=$3 # Liz (Capstone): this will only be non-blank in the case where my eccentric PE Makefile has inserted "--eccentricity" into join.sub
MPA=$4

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
if [ "$3" == '--eccentricity' ]; then
    if [ "$4" == '--meanPerAno' ]; then
	"${CLEAN_ILE}" ${RND}_tmp.dat $3 $4 | sort -rg -k12 > $BASE_OUT.composite
    else
	"${CLEAN_ILE}" ${RND}_tmp.dat $3 | sort -rg -k11 > $BASE_OUT.composite
    fi
else
    "${CLEAN_ILE}" ${RND}_tmp.dat $3 | sort -rg -k10 > $BASE_OUT.composite
fi
clean_status=$?
if [ ${clean_status} -ne 0 ]; then
    echo "ERROR: ILE consolidation failed with status ${clean_status}" >&2
    rm -f "$BASE_OUT.composite"
    exit ${clean_status}
fi
if [ ! -s "$BASE_OUT.composite" ]; then
    echo "ERROR: ILE consolidation produced an empty composite: $BASE_OUT.composite" >&2
    rm -f "$BASE_OUT.composite"
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
env >> ${BASE_OUT}.environment  

# tar file
if ! tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}.composite  ${BASE_OUT}.manifest ${BASE_OUT}.environment; then
    echo "ERROR: failed to create ${BASE_OUT}.tgz" >&2
    exit 1
fi

exit 0
