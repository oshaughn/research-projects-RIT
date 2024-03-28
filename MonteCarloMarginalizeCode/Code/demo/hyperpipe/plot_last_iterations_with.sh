#! /bin/bash
DIR_NAME=$1; shift;
CMD_LINE=$@
echo ${CMD_LINE}


fnames=`ls ${DIR_NAME}/grid-*.dat`
echo ${fnames}
USECOUNTER=0
(for name in ${fnames}; do echo --posterior-file ${name} --posterior-label ${USECOUNTER} ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile.txt
echo ${CMD_LINE}  `cat myfile.txt` > myargs.txt

plot_posterior_corner.py `cat myargs.txt`
