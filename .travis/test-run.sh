#! /bin/bash

# GETTING_STARTED.md example

git  clone https://github.com/oshaughn/ILE-GPU-Paper.git
cd ILE-GPU-Paper/demos/
make test_workflow_batch_gpu_lowlatency
cd test_workflow_batch_gpu_lowlatency
./command-single.sh
