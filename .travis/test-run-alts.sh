#! /bin/bash

# GETTING_STARTED.md example

if [ ! -d ILE-GPU-Paper ]; then 
  git  clone https://github.com/oshaughn/ILE-GPU-Paper.git
fi 
cd ILE-GPU-Paper/demos/
if [ -d test_workflow_batch_gpu_lowlatency ]; then
  echo " Deleting test directory !"
  rm -rf test_workflow_batch_gpu_lowlatency
fi 
make test_workflow_batch_gpu_lowlatency
cd test_workflow_batch_gpu_lowlatency
# force standard code path
switcheroo '--maximize-only '  ' --force-xpy ' command-single.sh
# Reduce number of analyses for this worker to 1, to reduce runtime
switcheroo '\$\(macrongroup\)' 1  command-single.sh
alias macrongroup='echo 1'
echo 'echo 1' > macrongroup; chmod a+x macrongroup; PATH=${PATH}:`pwd`#
# Reduce the number of points investigated by x100
#   ... and save-samples
switcheroo 'n-max 2000000' 'n-max 50000 --save-samples --output-file my_stuff ' command-single.sh
switcheroo '--save-samples ' '--save-samples --resample-time-marginalization --fairdraw-extrinsic-output ' command-single.sh
./command-single.sh

