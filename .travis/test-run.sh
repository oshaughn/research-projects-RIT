#! /bin/bash

# GETTING_STARTED.md example

git  clone https://github.com/oshaughn/ILE-GPU-Paper.git
cd ILE-GPU-Paper/demos/
make test_workflow_batch_gpu_lowlatency
cd test_workflow_batch_gpu_lowlatency
# Reduce number of analyses for this worker to 1, to reduce runtime
switcheroo 'n-events-to-analyze 20' 'n-events-to-analyze 1' command-single.sh
# Reduce the number of points investigated by x100
switcheroo 'n-max 2000000' 'n-max 50000' command-single.sh
./command-single.sh
