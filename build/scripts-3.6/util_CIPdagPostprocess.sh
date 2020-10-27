#! /bin/bash
#
# GOAL
#   - wrap up results from
#         - create_postprocessing_event_dag.py  


DIR_PROCESS=$1
BASE_OUT=$2

# join together the .dat files
echo " Joining data files .... "
find ${DIR_PROCESS} -name 'integral*_withpriorchange+annotation.dat' -exec cat {} \;  | sort | uniq | sort -r >  ${BASE_OUT}.integral_withprior
(for i in ` find ${DIR_PROCESS} -name 'integral*+annotation.dat' | grep -v _withpriorchange`; do cat $i; done)  | sort | uniq | sort -r >  ${BASE_OUT}.integral



# Manifest
rm -f ${BASE_OUT}.manifest
echo '#User:' `whoami` >>  ${BASE_OUT}.manifest
echo '#Date:' `date` >>  ${BASE_OUT}.manifest
echo '#Host:' `hostname -f` >>  ${BASE_OUT}.manifest
echo '#Directory:' `pwd`/${DIR_PROCESS} >>  ${BASE_OUT}.manifest
cat ${DIR_PROCESS}/command-single_fit.sh >>  ${BASE_OUT}.manifest  
env >> ${BASE_OUT}.environment  

# tar file
tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}.integral ${BASE_OUT}.integral_withprior  ${BASE_OUT}.manifest ${BASE_OUT}.environment

exit 0 ;  # force end on success, for DAG
