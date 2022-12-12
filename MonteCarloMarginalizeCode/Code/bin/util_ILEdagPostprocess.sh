#! /bin/bash
# util_NRdagPostprocess.sh
#
# GOAL
#   For NR-based DAGs, (a) consolidates the output, (b) runs ILE simplification, then (c) creates an NR-indexed version.
#   The second format uses a *portable* name, which is stable to me changing the underlying relationship between spins and label.


DIR_PROCESS=$1
BASE_OUT=$2
ECC=$3 # Liz (Capstone): this will only be non-blank in the case where my eccentric PE Makefile has inserted "--eccentricity" into join.sub

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
if [ "$3" == '--eccentricity' ]
then
    util_CleanILE.py ${RND}_tmp.dat $3 | sort -rg -k11 > $BASE_OUT.composite
else
    util_CleanILE.py ${RND}_tmp.dat $3 | sort -rg -k10 > $BASE_OUT.composite
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
tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}.composite  ${BASE_OUT}.manifest ${BASE_OUT}.environment

exit 0 ;  # force end on success, for DAG
