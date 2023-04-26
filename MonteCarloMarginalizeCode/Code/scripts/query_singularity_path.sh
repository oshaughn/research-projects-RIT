#! /bin/bash
#
# uses SINGULARITY_RIFT_IMAGE if present
#
# query_singuarity_path.sh    # returns the path needed to be set as 

export IMAGE=${SINGULARITY_RIFT_IMAGE}
if [ ! -z "$1" ]; then
IMAGE=$1
fi

CMD_NAME=`singularity exec  --bind /home --no-home --containall --pwd ${PWD} ${IMAGE} which integrate_likelihood_extrinsic_batchmode`
DIR_NAME=$(dirname $(CMD_NAME))
echo ${DIR_NAME}
