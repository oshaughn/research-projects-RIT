#! /bin/bash

# GETTING_STARTED.md example

#if  command -v apt ; then
  #apt install lalsuite=7.22   # problem with 7.23 and newer lalapps_path2cache, workaround
#fi
#pip install lalsuite==7.22 --break-system-packges
#python -m pip install --upgrade lalsuite==7.22 --break-system-packages

git  clone https://github.com/oshaughn/ILE-GPU-Paper.git
cd ILE-GPU-Paper/demos/
make test_workflow_batch_gpu_lowlatency
cd test_workflow_batch_gpu_lowlatency
# force standard code path
switcheroo '--maximize-only '  ' --force-xpy ' command-single.sh
# Reduce number of analyses for this worker to 1, to reduce runtime
switcheroo '\$\(macrongroup\)' 1  command-single.sh
# new format for n-events-to-analyze.  Backstop
alias macrongroup='echo 1'
echo 'echo 1' > macrongroup; chmod a+x macrongroup; PATH=${PATH}:`pwd`
# Reduce the number of points investigated by x100
switcheroo 'n-max 2000000' 'n-max 50000' command-single.sh
./command-single.sh
