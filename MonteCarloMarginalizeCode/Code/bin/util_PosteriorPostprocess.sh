#! /bin/bash
# util_NRdagPostprocess.sh
#
# GOAL
#   For NR-based DAGs, (a) consolidates the output, (b) runs ILE simplification, then (c) creates an NR-indexed version.
#   The second format uses a *portable* name, which is stable to me changing the underlying relationship between spins and label.


DIR_PROCESS=$1
BASE_OUT=$2

# join together the .dat files
# Manifest
rm -f ${BASE_OUT}.manifest
echo '#User:' `whoami` >>  ${BASE_OUT}.manifest
echo '#Date:' `date` >>  ${BASE_OUT}.manifest
echo '#Host:' `hostname -f` >>  ${BASE_OUT}.manifest
echo '#Directory:' `pwd`/${DIR_PROCESS} >>  ${BASE_OUT}.manifest
md5sum ${DIR_PROCESS}/all.composite >> ${BASE_OUT}.manifest
cat ${DIR_PROCESS}/args.txt >>  ${BASE_OUT}.manifest  
env >> ${BASE_OUT}.environment  
cp ${DIR_PROCESS}/posterior-samples.dat ${BASE_OUT}_posterior.dat
cp ${DIR_PROCESS}/run.log ${BASE_OUT}_run.log
cp ${DIR_PROCESS}/fnames_used  ${BASE_OUT}_fnames_used.txt

# tar file
tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}_run.log ${BASE_OUT}_posterior.dat  ${BASE_OUT}.manifest ${BASE_OUT}.environment ${BASE_OUT}_fnames_used.txt
