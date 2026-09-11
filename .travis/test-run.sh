#! /bin/bash

# GETTING_STARTED.md example.  Two lanes, from one build:
#
#   1. NoLoop (production path): --time-marginalization --vectorized --gpu,
#      the exact option set helper_LDG_Events.py emits for every real run
#      (--propose-ile-convergence-options).  --force-xpy takes the identical
#      NoLoop code (DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop) on
#      plain numpy when no cupy device is present, instead of silently
#      downgrading --gpu to a no-op.  This is asserted below by grepping the
#      run's own startup banner ("Q_lm sub-sample time stencil ... [... ]")
#      for vectorized=True gpu=True, rather than trusting the flags alone:
#      command-single.sh already carries an incidental --vectorized --gpu
#      from create_event_parameter_pipeline_BasicIteration's --request-gpu-ILE
#      handling, so passing the flags here is about making this script
#      correct on its own (and future-proof if that incidental append ever
#      goes away), and the banner check is what actually proves the path ran.
#   2. legacy-scalar (sanity only, NOT what production runs): the same build,
#      with --vectorized/--gpu/--force-xpy stripped back out and n-max/n-chunk
#      cut so it costs seconds.  FactoredLogLikelihoodTimeMarginalized already
#      has unit coverage (test_ile_scalar_edge_cases.py,
#      factored_likelihood_test.py); this lane exists because unit tests miss
#      the driver/CLI seam, and only checks that the legacy branch still runs
#      at all, not that its answer is right.

#if  command -v apt ; then
  #apt install lalsuite=7.22   # problem with 7.23 and newer lalapps_path2cache, workaround
#fi
#pip install lalsuite==7.22 --break-system-packges
#python -m pip install --upgrade lalsuite==7.22 --break-system-packages

git  clone https://github.com/oshaughn/ILE-GPU-Paper.git
cd ILE-GPU-Paper/demos/
make test_workflow_batch_gpu_lowlatency
cd test_workflow_batch_gpu_lowlatency
# Exercise the maintained NoLoop path explicitly (see banner above).
switcheroo '--maximize-only '  ' --vectorized --gpu --force-xpy ' command-single.sh
# Reduce number of analyses for this worker to 1, to reduce runtime.
# create_event_parameter_pipeline_BasicIteration replaces $(macrongroup) with the literal
# '5' at command-single.sh GENERATION time (bin/create_event_parameter_pipeline_
# BasicIteration, arg_list.replace('$(macrongroup)','5')), before this script ever runs --
# so the line below targeted a token this file no longer carries, a silent no-op (PR #283
# review, MINOR).  Target the literal value actually written instead.
switcheroo '--n-events-to-analyze  5 ' '--n-events-to-analyze  1 '  command-single.sh
# Backstop for an older create_event_parameter_pipeline_BasicIteration that still leaves
# $(macrongroup) as a literal token: bash would then treat it as command substitution
# (running a command named macrongroup) when command-single.sh executes, so make that
# resolve to 1 too rather than failing with "macrongroup: command not found".
switcheroo '\$\(macrongroup\)' 1  command-single.sh
alias macrongroup='echo 1'
echo 'echo 1' > macrongroup; chmod a+x macrongroup; PATH=${PATH}:`pwd`
# Reduce the number of points investigated by x100
switcheroo 'n-max 2000000' 'n-max 50000' command-single.sh

./command-single.sh 2>&1 | tee run_command_single_noloop.log
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "NoLoop lane: command-single.sh FAILED (exit $status)"
    exit "$status"
fi
if ! grep -q 'Q_lm sub-sample time stencil.*vectorized=True gpu=True' run_command_single_noloop.log; then
    echo "NoLoop lane: the run did NOT take the NoLoop path -- vectorized=True gpu=True not in the startup banner:"
    grep 'Q_lm sub-sample time stencil' run_command_single_noloop.log
    exit 1
fi
echo "NoLoop lane: confirmed (vectorized=True gpu=True in run_command_single_noloop.log)"

# --- legacy-scalar sanity lane (cheap; NOT the production path) ----------
cp command-single.sh command-single-legacy-scalar.sh
switcheroo ' --vectorized' '' command-single-legacy-scalar.sh
switcheroo ' --gpu' '' command-single-legacy-scalar.sh
switcheroo ' --force-xpy' '' command-single-legacy-scalar.sh
switcheroo 'n-max 50000' 'n-max 500' command-single-legacy-scalar.sh
switcheroo 'n-chunk 10000' 'n-chunk 500' command-single-legacy-scalar.sh

./command-single-legacy-scalar.sh 2>&1 | tee run_command_single_legacy_scalar.log
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "legacy-scalar lane: command-single-legacy-scalar.sh FAILED (exit $status)"
    exit "$status"
fi
if ! grep -q 'Q_lm sub-sample time stencil.*vectorized=False gpu=False' run_command_single_legacy_scalar.log; then
    echo "legacy-scalar lane: the run did NOT take the legacy scalar path -- vectorized=False gpu=False not in the startup banner:"
    grep 'Q_lm sub-sample time stencil' run_command_single_legacy_scalar.log
    exit 1
fi
echo "legacy-scalar lane: confirmed (vectorized=False gpu=False in run_command_single_legacy_scalar.log)"
