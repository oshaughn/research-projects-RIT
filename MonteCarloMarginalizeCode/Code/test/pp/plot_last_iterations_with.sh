#! /bin/bash

CMD_LINE=$@

# need to insure correct sorting of these names, to produce correct order
fnames=`ls posterior_samples*.dat | sort --field-separator=- -k3 -n`
echo ${fnames}
USECOUNTER=0
for name in ${fnames}; do echo --posterior-file ${name} --posterior-label ${USECOUNTER} ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done | tr '\n' ' ' > myfile.txt
echo ${CMD_LINE}  `cat myfile.txt` > myargs.txt

plot_posterior_corner.py `cat myargs.txt`
