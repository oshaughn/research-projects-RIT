#! /bin/bash
# util_NRdagPostprocess.sh
#
# GOAL
#   For NR-based DAGs, (a) consolidates the output, (b) runs ILE simplification, then (c) creates an NR-indexed version.
#   The second format uses a *portable* name, which is stable to me changing the underlying relationship between spins and label.


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
    util_CleanILE.py ${DIR_PROCESS}_tmp.dat $4 | sort -rg -k11 > $BASE_OUT.composite
else
    util_CleanILE.py ${DIR_PROCESS}_tmp.dat $4 | sort -rg -k10 > $BASE_OUT.composite
fi

# index them
echo " Reindexing the data to   .... "
#util_ILEtoNRIndex.py --group ${GROUP} --fname ${BASE_OUT}.composite | grep '^-1*' > ${BASE_OUT}.indexed
if [ "$4" == '--eccentricity' ]
then
    #    util_NRRelabelILE.py --group ${GROUP} --fname ${BASE_OUT}.composite --eccentricity | grep '^-1*' > ${BASE_OUT}.indexed
    util_NRRelabelILE.py --group Sequence-RIT-All --fname ${BASE_OUT}.composite --eccentricity | sed -n '/ -----  BEST MATCHES ------ /,$p' > ${BASE_OUT}.indexed
else
    util_NRRelabelILE.py --group ${GROUP} --fname ${BASE_OUT}.composite | grep '^-1*' > ${BASE_OUT}.indexed
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
tar cvzf ${BASE_OUT}.tgz ${BASE_OUT}.composite ${BASE_OUT}.indexed ${BASE_OUT}.manifest ${BASE_OUT}.environment ${BASE_OUT}.submit
