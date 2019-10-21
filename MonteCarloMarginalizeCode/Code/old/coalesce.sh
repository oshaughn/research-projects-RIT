#!/bin/bash

massset=$1

#touch pre_coalesce_${JOBID}.log

#echo $1 >> pre_coalesce_${JOBID}.log

#ls >> pre_coalesce_${JOBID}.log

if [[ -z "${massset}" ]]; then
    echo "No mass ID found for this job ID."
    #cat *.cache > ILE_all.cache
    find . -name "*.sqlite" | lalapps_path2cache -a > ILE_all.cache
    exit
fi

ocache=ILE_${massset}.cache

find . -name "*-${massset}-*.xml.gz" | lalapps_path2cache -a > ${ocache}
