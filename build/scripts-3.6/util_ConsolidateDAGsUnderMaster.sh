#! /bin/bash
#
# ASSUMES
#   - user passes directory names (which are subdirectories of current
#   - we need to create unique names
#   - the subdag names all are marginalize_extrinsic_parameters.dag


for i in `find $@ -name '*.dag' -print`
do
echo SUBDAG EXTERNAL `echo $i | sed -e 's/\//_/g' | sed -e 's/.dag//g' ` `pwd`/$i  DIR `echo $i | sed 's/marginalize_extrinsic_parameters.dag//g' | sed 's/marginalize_extrinsic_parameters_grid.dag//g' | sed 's/marginalize_intrinsic_parameters_BasicIterationWorkflow.dag//g' `
done > master.dag

echo condor_submit_dag -maxjobs 1 master.dag
