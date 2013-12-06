#!/bin/bash

JOBID=$1

# Need the MASSID, since it won't be available to this script
massset=""
# FIXME: Verify they're all from the same mass set
while read fname; do
    # Strip path
    fname=`basename ${fname}`
    # Strip extension
    fbasename=${fname%:*}
    # Get mass set ID
    massset=`echo ${fbasename} | awk -F"-" '{print $2}'`
done < <(find . -name "*${JOBID}*.xml.gz")

if [[ -z "${massset}" ]]; then
    echo "No mass ID found for this job ID."
    exit 1
fi

ocache=ILE_${massset}.cache

find . -name "*${JOBID}*.xml.gz" | lalapps_path2cache -a > ${ocache}
