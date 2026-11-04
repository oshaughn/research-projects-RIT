#! /bin/bash

# GETTING_STARTED.md example.  Same NoLoop lane as test-run.sh
# (--time-marginalization --vectorized --gpu, --force-xpy for the CPU
# fallback), plus --resample-time-marginalization --fairdraw-extrinsic-output
# on top -- see test-run.sh for why the flags are explicit and what the
# banner assertion below is checking.  No separate legacy-scalar lane here;
# test-run.sh already carries that one.

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
# Exercise the maintained NoLoop path explicitly (see test-run.sh).
switcheroo '--maximize-only '  ' --vectorized --gpu --force-xpy ' command-single.sh
# Reduce number of analyses for this worker to 1, to reduce runtime
switcheroo '\$\(macrongroup\)' 1  command-single.sh
alias macrongroup='echo 1'
echo 'echo 1' > macrongroup; chmod a+x macrongroup; PATH=${PATH}:`pwd`#
# Reduce the number of points investigated by x100
#   ... and save-samples
switcheroo 'n-max 2000000' 'n-max 50000 --save-samples --output-file my_stuff ' command-single.sh
switcheroo '--save-samples ' '--save-samples --resample-time-marginalization --fairdraw-extrinsic-output ' command-single.sh

./command-single.sh 2>&1 | tee run_command_single.log
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "command-single.sh FAILED (exit $status)"
    exit "$status"
fi
if ! grep -q 'Q_lm sub-sample time stencil.*vectorized=True gpu=True' run_command_single.log; then
    echo "the run did NOT take the NoLoop path -- vectorized=True gpu=True not in the startup banner:"
    grep 'Q_lm sub-sample time stencil' run_command_single.log
    exit 1
fi
echo "NoLoop path confirmed (vectorized=True gpu=True in run_command_single.log)"
