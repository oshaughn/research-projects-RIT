#! /bin/bash
# util_ILEROMdagPostprocess.sh
#
# GOAL
#   For ROM-based DAGs, (a) consolidates the output, then (b) wraps it up, with a manifest, so we can track it down later


DIR_PROCESS=$1
BASE_OUT=$2

# join together the .dat files
echo " Joining data files .... "
rm -f tmp.dat tmp2.dat
# CAT can be ineffective
FNAME=`pwd`/tmp.dat
find ${DIR_PROCESS} -name 'CME*.xml.gz' -exec convert_output_format_ile2inference {}  \;   > ${BASE_OUT}.tmp
grep '#' ${BASE_OUT}.tmp | uniq > ${BASE_OUT}.rom_composite
grep -v '#' ${BASE_OUT}.tmp >> ${BASE_OUT}.rom_composite


# Manifest
rm -f ${BASE_OUT}.manifest
echo '#User:' `whoami` >>  ${BASE_OUT}.manifest
echo '#Date:' `date` >>  ${BASE_OUT}.manifest
echo '#Host:' `hostname` >>  ${BASE_OUT}.manifest
echo '#Directory:' `pwd`/${DIR_PROCESS} >>  ${BASE_OUT}.manifest
cat ${DIR_PROCESS}/testme-command.sh >>  ${BASE_OUT}.manifest  
env >> ${BASE_OUT}.environment  

# tar file
tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}.rom_composite  ${BASE_OUT}.manifest ${BASE_OUT}.environment
