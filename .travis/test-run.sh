#! /bin/bash

# GETTING_STARTED.md example

git  clone https://github.com/oshaughn/ILE-GPU-Paper.git
cd ILE-GPU-Paper/demos/
make test_workflow_batch_gpu_lowlatency
cd test_workflow_batch_gpu_lowlatency
# force standard code path
switcheroo '--maximize-only '  ' --force-xpy ' command-single.sh
# Reduce number of analyses for this worker to 1, to reduce runtime
switcheroo '\$\(macrongroup\)' 1  command-single.sh
# new format for n-events-to-analyze.  Backstop
export macrongroup='echo 1'
# Reduce the number of points investigated by x100
switcheroo 'n-max 2000000' 'n-max 50000' command-single.sh
./command-single.sh
