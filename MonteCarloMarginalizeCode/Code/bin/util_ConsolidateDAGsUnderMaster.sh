#! /bin/bash
#
# ASSUMES
#   - user passes directory names (which are subdirectories of current
#   - we need to create unique names
#   - the subdag names all are marginalize_extrinsic_parameters.dag


# Make sure dags are being automatically retried.  Very frustrating with unstable cluster environments/nodes
if [ -z "${DAG_RETRY}" ]; then
  DAG_RETRY=10
fi
if [ -z "${DAG_NAME}" ]; then
  DAG_NAME=master.dag
fi

for i in `find $@ -name '*.dag' -print`
do
DAG_ID=`echo $i | sed -e 's/\//_/g' | sed -e 's/.dag//g' `
echo SUBDAG EXTERNAL ${DAG_ID} `pwd`/$i  DIR `echo $i | sed 's/marginalize_extrinsic_parameters.dag//g' | sed 's/marginalize_extrinsic_parameters_grid.dag//g' | sed 's/marginalize_intrinsic_parameters_BasicIterationWorkflow.dag//g' `
echo RETRY ${DAG_ID} ${DAG_RETRY}
done > ${DAG_NAME}

echo condor_submit_dag -maxjobs 1 ${DAG_NAME}
